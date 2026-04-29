"""Lightweight SDP-to-image-like translation network.

The model follows the domain-translation idea used by DensePose From WiFi:
CSI-domain features are encoded into a compact latent map and decoded into an
RGB-sized image-domain representation. This implementation is adapted for the
offline SDP tensors produced by the current preprocessing pipeline.

Default input:
    (B, 30, 270, 3) for HWC SDP, or (B, 3, 30, 270) for CHW SDP.

Default output:
    (B, 3, 360, 640), matching the RGB image size used in this project.
"""

from __future__ import annotations

from typing import Literal, Tuple

import torch
from torch import Tensor, nn
import torch.nn.functional as F


Layout = Literal["auto", "bhwc", "bchw"]
OutputLayout = Literal["bchw", "bhwc"]
Activation = Literal["none", "sigmoid", "tanh"]


class ConvNormAct(nn.Sequential):
    """Conv2d + BatchNorm + ReLU block."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int | Tuple[int, int] = 1,
        padding: int | None = None,
    ) -> None:
        if padding is None:
            padding = kernel_size // 2
        super().__init__(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )


class ResidualBlock(nn.Module):
    """Small residual block used after each resolution change."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.block = nn.Sequential(
            ConvNormAct(channels, channels, kernel_size=3),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
        )
        self.act = nn.ReLU(inplace=True)

    def forward(self, x: Tensor) -> Tensor:
        return self.act(x + self.block(x))


class UpsampleBlock(nn.Module):
    """Bilinear upsample followed by local convolutional refinement."""

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.refine = nn.Sequential(
            ConvNormAct(in_channels, out_channels, kernel_size=3),
            ResidualBlock(out_channels),
        )

    def forward(self, x: Tensor, size: Tuple[int, int]) -> Tensor:
        x = F.interpolate(x, size=size, mode="bilinear", align_corners=False)
        return self.refine(x)


class SDPToImageLikeTranslator(nn.Module):
    """Translate preprocessed SDP tensors into RGB-sized image-like features.

    The network is intentionally compact:
      1. A convolutional SDP encoder preserves local structure in subcarrier
         and lag-window axes.
      2. A compact latent map is resized to 1/8 of the target image size.
      3. Three decoder stages upsample to the final RGB resolution.

    Args:
        in_channels: Number of SDP channels. The current preprocessing uses 3.
        target_size: Output size as (height, width). RGB image size is
            represented as (360, 640) in PyTorch channel-first format.
        stem_channels: Channels in the first SDP encoder stage.
        latent_channels: Channels in the compact latent representation.
        decoder_channels: Channels used by the three upsampling stages.
        output_channels: Number of output channels. Use 3 for RGB-like output.
        input_layout: "bhwc", "bchw", or "auto".
        output_layout: "bchw" for PyTorch models, or "bhwc" for direct
            HWC image-like arrays.
        output_activation: Optional final activation. Use "none" when the
            output is a learned feature map, "sigmoid" for [0, 1] image targets,
            or "tanh" for [-1, 1] normalized image targets.
    """

    def __init__(
        self,
        in_channels: int = 3,
        target_size: Tuple[int, int] = (360, 640),
        stem_channels: int = 32,
        latent_channels: int = 128,
        decoder_channels: Tuple[int, int, int] = (96, 48, 24),
        output_channels: int = 3,
        input_layout: Layout = "auto",
        output_layout: OutputLayout = "bchw",
        output_activation: Activation = "none",
    ) -> None:
        super().__init__()
        if len(decoder_channels) != 3:
            raise ValueError("decoder_channels must contain exactly three stages")
        if output_activation not in ("none", "sigmoid", "tanh"):
            raise ValueError(f"Unsupported output_activation={output_activation}")
        if input_layout not in ("auto", "bhwc", "bchw"):
            raise ValueError(f"Unsupported input_layout={input_layout}")
        if output_layout not in ("bchw", "bhwc"):
            raise ValueError(f"Unsupported output_layout={output_layout}")

        self.in_channels = in_channels
        self.target_size = target_size
        self.input_layout = input_layout
        self.output_layout = output_layout
        self.output_activation = output_activation

        self.stem = nn.Sequential(
            ConvNormAct(in_channels, stem_channels, kernel_size=3),
            ResidualBlock(stem_channels),
        )
        self.encoder = nn.Sequential(
            ConvNormAct(stem_channels, stem_channels * 2, kernel_size=3, stride=2),
            ResidualBlock(stem_channels * 2),
            ConvNormAct(stem_channels * 2, latent_channels, kernel_size=3, stride=2),
            ResidualBlock(latent_channels),
            ConvNormAct(latent_channels, latent_channels, kernel_size=3),
        )
        self.bridge = ResidualBlock(latent_channels)

        d0, d1, d2 = decoder_channels
        self.decoder1 = UpsampleBlock(latent_channels, d0)
        self.decoder2 = UpsampleBlock(d0, d1)
        self.decoder3 = UpsampleBlock(d1, d2)
        self.output = nn.Conv2d(d2, output_channels, kernel_size=1)

        self._init_weights()

    def _init_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.BatchNorm2d):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

    def _to_bchw(self, x: Tensor) -> Tensor:
        if x.ndim != 4:
            raise ValueError(f"Expected 4D input, got shape {tuple(x.shape)}")

        if self.input_layout == "bchw":
            if x.shape[1] != self.in_channels:
                raise ValueError(f"Expected channel dimension {self.in_channels}, got shape {tuple(x.shape)}")
            return x.contiguous()

        if self.input_layout == "bhwc":
            if x.shape[-1] != self.in_channels:
                raise ValueError(f"Expected last dimension {self.in_channels}, got shape {tuple(x.shape)}")
            return x.permute(0, 3, 1, 2).contiguous()

        if x.shape[1] == self.in_channels:
            return x.contiguous()
        if x.shape[-1] == self.in_channels:
            return x.permute(0, 3, 1, 2).contiguous()
        raise ValueError(
            f"Could not infer input layout from shape {tuple(x.shape)}; "
            f"expected channel dimension {self.in_channels} at dim 1 or dim -1"
        )

    def _decoder_sizes(self) -> Tuple[Tuple[int, int], Tuple[int, int], Tuple[int, int], Tuple[int, int]]:
        h, w = self.target_size
        base = (max(1, h // 8), max(1, w // 8))
        half = (max(1, h // 4), max(1, w // 4))
        quarter = (max(1, h // 2), max(1, w // 2))
        full = (h, w)
        return base, half, quarter, full

    def forward(self, x: Tensor) -> Tensor:
        x = self._to_bchw(x).float()
        x = self.stem(x)
        x = self.encoder(x)
        x = self.bridge(x)

        base_size, size1, size2, size3 = self._decoder_sizes()
        x = F.interpolate(x, size=base_size, mode="bilinear", align_corners=False)
        x = self.decoder1(x, size=size1)
        x = self.decoder2(x, size=size2)
        x = self.decoder3(x, size=size3)
        x = self.output(x)

        if self.output_activation == "sigmoid":
            x = torch.sigmoid(x)
        elif self.output_activation == "tanh":
            x = torch.tanh(x)

        if self.output_layout == "bhwc":
            return x.permute(0, 2, 3, 1).contiguous()
        return x.contiguous()


def image_translation_loss(
    prediction: Tensor,
    target: Tensor,
    loss_type: Literal["mse", "l1", "smooth_l1"] = "l1",
) -> Tensor:
    """Compute a simple supervised translation loss.

    Both channel-first (B, 3, H, W) and channel-last (B, H, W, 3) tensors are
    accepted. If target spatial size differs from prediction, target is resized
    to prediction size.
    """

    if prediction.ndim != 4:
        raise ValueError(f"Expected 4D prediction, got shape {tuple(prediction.shape)}")
    if prediction.shape[1] not in (1, 3) and prediction.shape[-1] in (1, 3):
        prediction = prediction.permute(0, 3, 1, 2).contiguous()
    if prediction.shape[1] not in (1, 3):
        raise ValueError(f"Could not infer prediction channel layout from shape {tuple(prediction.shape)}")

    if target.ndim != 4:
        raise ValueError(f"Expected 4D target, got shape {tuple(target.shape)}")
    if target.shape[1] != prediction.shape[1] and target.shape[-1] == prediction.shape[1]:
        target = target.permute(0, 3, 1, 2).contiguous()
    if target.shape[1] != prediction.shape[1]:
        raise ValueError(f"Target channel count does not match prediction: {tuple(target.shape)} vs {tuple(prediction.shape)}")
    if target.shape[-2:] != prediction.shape[-2:]:
        target = F.interpolate(target.float(), size=prediction.shape[-2:], mode="bilinear", align_corners=False)

    if loss_type == "mse":
        return F.mse_loss(prediction, target.float())
    if loss_type == "smooth_l1":
        return F.smooth_l1_loss(prediction, target.float())
    if loss_type == "l1":
        return F.l1_loss(prediction, target.float())
    raise ValueError(f"Unsupported loss_type={loss_type}")


if __name__ == "__main__":
    model = SDPToImageLikeTranslator()
    dummy = torch.randn(1, 30, 270, 3)
    with torch.no_grad():
        out = model(dummy)
    print("input :", tuple(dummy.shape))
    print("output:", tuple(out.shape))
