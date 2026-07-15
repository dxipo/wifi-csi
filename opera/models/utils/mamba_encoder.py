# Copyright (c) Hikvision Research Institute. All rights reserved.
import torch
import torch.nn as nn

from .builder import TRANSFORMER_LAYER_SEQUENCE

try:
    from mamba_ssm import Mamba
except ImportError as exc:  # pragma: no cover - depends on the runtime env
    Mamba = None
    _MAMBA_IMPORT_ERROR = exc
else:
    _MAMBA_IMPORT_ERROR = None


class MambaEncoderLayer(nn.Module):
    """Pre-norm residual Mamba block for PETR encoder tokens."""

    def __init__(self,
                 embed_dims=256,
                 d_state=16,
                 d_conv=4,
                 expand=2,
                 dropout=0.1,
                 bidirectional=False,
                 use_local_conv=False,
                 local_kernel_size=3,
                 local_init_scale=0.1,
                 num_antennas=3,
                 num_time_steps=10):
        super().__init__()
        if Mamba is None:
            raise ImportError(
                'MambaEncoder requires mamba-ssm. Activate the '
                'wifi3d_mamba environment before building the model.') from \
                _MAMBA_IMPORT_ERROR
        self.norm = nn.LayerNorm(embed_dims)
        self.mixer = Mamba(
            d_model=embed_dims, d_state=d_state, d_conv=d_conv, expand=expand)
        self.dropout = nn.Dropout(dropout)
        self.bidirectional = bidirectional
        self.use_local_conv = use_local_conv
        self.num_antennas = num_antennas
        self.num_time_steps = num_time_steps
        if use_local_conv:
            if local_kernel_size <= 0 or local_kernel_size % 2 != 1:
                raise ValueError('local_kernel_size must be a positive odd integer')
            self.local_conv = nn.Conv2d(
                embed_dims,
                embed_dims,
                kernel_size=local_kernel_size,
                padding=local_kernel_size // 2,
                groups=embed_dims)
            self.local_activation = nn.GELU()
            self.local_scale = nn.Parameter(
                torch.full((embed_dims, ), float(local_init_scale)))
        else:
            self.local_conv = None
            self.local_activation = None
            self.local_scale = None

    def local_context(self, x):
        batch, length, channels = x.shape
        expected = self.num_antennas * self.num_time_steps
        if length != expected:
            raise ValueError(
                f'Local Conv expects {self.num_antennas} antennas x '
                f'{self.num_time_steps} time steps = {expected} tokens, got '
                f'{length}')
        grid = x.reshape(batch, self.num_antennas, self.num_time_steps,
                         channels).permute(0, 3, 1, 2).contiguous()
        local = self.local_activation(self.local_conv(grid))
        return local.permute(0, 2, 3, 1).contiguous().reshape(
            batch, length, channels)

    def forward(self, x):
        normed = self.norm(x)
        mixed = self.mixer(normed)
        if self.bidirectional:
            reversed_x = normed.flip(dims=(1, )).contiguous()
            reversed_y = self.mixer(reversed_x)
            backward = reversed_y.flip(dims=(1, )).contiguous()
            mixed = 0.5 * (mixed + backward)
        if self.use_local_conv:
            local = self.local_context(normed)
            mixed = mixed + local * self.local_scale.view(1, 1, -1)
        return x + self.dropout(mixed)


@TRANSFORMER_LAYER_SEQUENCE.register_module()
class MambaEncoder(nn.Module):
    """Mamba replacement for the PETR Transformer encoder.

    PETR/MMCV passes encoder features as ``[length, batch, channels]`` while
    Mamba expects ``[batch, length, channels]``. M0 scans the existing 30
    flattened CSI tokens in one direction. M1 optionally averages forward and
    reverse scans from the same mixer, preserving the M0 parameter count. M3
    optionally adds a depthwise Local Conv branch over the antenna-time grid.
    """

    skip_global_xavier_init = True

    def __init__(self,
                 embed_dims=256,
                 num_layers=6,
                 d_state=16,
                 d_conv=4,
                 expand=2,
                 dropout=0.1,
                 bidirectional=False,
                 use_local_conv=False,
                 local_kernel_size=3,
                 local_init_scale=0.1,
                 num_antennas=3,
                 num_time_steps=10,
                 final_norm=True):
        super().__init__()
        self.embed_dims = embed_dims
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.use_local_conv = use_local_conv
        self.num_antennas = num_antennas
        self.num_time_steps = num_time_steps
        self.layers = nn.ModuleList([
            MambaEncoderLayer(
                embed_dims=embed_dims,
                d_state=d_state,
                d_conv=d_conv,
                expand=expand,
                dropout=dropout,
                bidirectional=bidirectional,
                use_local_conv=use_local_conv,
                local_kernel_size=local_kernel_size,
                local_init_scale=local_init_scale,
                num_antennas=num_antennas,
                num_time_steps=num_time_steps) for _ in range(num_layers)
        ])
        self.final_norm = nn.LayerNorm(
            embed_dims) if final_norm else nn.Identity()

    def forward(self,
                query,
                key=None,
                value=None,
                query_pos=None,
                key_pos=None,
                attn_masks=None,
                query_key_padding_mask=None,
                key_padding_mask=None,
                **kwargs):
        if query.dim() != 3 or query.size(-1) != self.embed_dims:
            raise ValueError(
                f'MambaEncoder expects [L, B, {self.embed_dims}], got '
                f'{tuple(query.shape)}')
        if not query.is_cuda:
            raise RuntimeError(
                'M0 MambaEncoder requires CUDA selective-scan execution.')
        if query_key_padding_mask is not None and query_key_padding_mask.any():
            raise ValueError(
                'M0 MambaEncoder does not support padded CSI tokens.')
        if key_padding_mask is not None and key_padding_mask.any():
            raise ValueError(
                'M0 MambaEncoder does not support padded CSI tokens.')
        if attn_masks is not None:
            raise ValueError(
                'M0 MambaEncoder does not support attention masks.')

        x = query if query_pos is None else query + query_pos
        x = x.permute(1, 0, 2).contiguous()
        for layer in self.layers:
            x = layer(x)
        x = self.final_norm(x)
        return x.permute(1, 0, 2).contiguous()
