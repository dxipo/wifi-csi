# Copyright (c) Hikvision Research Institute. All rights reserved.
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
                 dropout=0.1):
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

    def forward(self, x):
        return x + self.dropout(self.mixer(self.norm(x)))


@TRANSFORMER_LAYER_SEQUENCE.register_module()
class MambaEncoder(nn.Module):
    """Mamba replacement for the PETR Transformer encoder.

    PETR/MMCV passes encoder features as ``[length, batch, channels]`` while
    Mamba expects ``[batch, length, channels]``. M0 intentionally scans the
    existing 30 flattened CSI tokens in their current order so that the
    encoder family is the only experimental variable.
    """

    skip_global_xavier_init = True

    def __init__(self,
                 embed_dims=256,
                 num_layers=6,
                 d_state=16,
                 d_conv=4,
                 expand=2,
                 dropout=0.1,
                 final_norm=True):
        super().__init__()
        self.embed_dims = embed_dims
        self.num_layers = num_layers
        self.layers = nn.ModuleList([
            MambaEncoderLayer(
                embed_dims=embed_dims,
                d_state=d_state,
                d_conv=d_conv,
                expand=expand,
                dropout=dropout) for _ in range(num_layers)
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
