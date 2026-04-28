#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Preview-only SDP300 image-like preprocessing for CSI samples.

Target output layout:
    (Subcarrier, SDP-width, Antenna-channel) = (30, X, 3)

This keeps the core SDP computation path from
preprocess_csi_sdp_offline_power_xfall.py:
    amplitude -> Hampel/moving-average -> power -> windowed ACF
    -> positive clipping -> column normalization

The main change is structural: subcarriers are preserved as the image height,
one antenna axis is preserved as the 3-channel image dimension, and the other
antenna axis is averaged. The SDP window/lag plane is flattened into the image
width X.
"""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Sequence, Tuple
import sys

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from tqdm import tqdm

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from preprocess_csi_sdp_offline_power_xfall import (
    EPS,
    compute_acf_for_subcarrier,
    compute_power_response,
    ensure_dir,
    load_csi_from_mat,
    normalize_sdp_columns,
    preprocess_amplitude_matrix,
    save_array,
)


CONFIG = {
    "src_root": "/home/xl/CSI/Person-in-WiFi-3D-repo/data/wifipose/all_single_test_data_hold_out",
    "dst_root": "/home/xl/CSI/Person-in-WiFi-3D-repo/data/wifipose/all_single_test_data_hold_out_offline_sdp300_imagelike_power_xfall_preview",
    "src_csi_dir": "csi",
    "dst_csi_dir": "csi_sdp300_imagelike_offline",
    "preview_dir": "preview_sdp300_imagelike",
    "group_list_name": "sdp300_imagelike_group_list.txt",
    "input_ext": ".mat",
    "mat_key": "csi_out",
    "mode": "test",

    "sample_rate_hz": 300,
    "group_size": 15,
    "group_stride": 1,
    "max_groups": 10,

    "expected_rx": 3,
    "expected_tx": 3,
    "expected_subcarrier": 30,
    "expected_clip_time": 20,
    "expected_time": 300,

    # Preserve this antenna axis as the final image-like 3-channel dimension.
    # The other antenna axis is averaged before saving.
    # Options: "rx" -> output channels are Rx antennas, average Tx links.
    #          "tx" -> output channels are Tx antennas, average Rx links.
    "channel_axis": "rx",

    # SDP300 defaults for a 1000 ms sequence at 300 Hz.
    "use_hampel": True,
    "hampel_window": 5,
    "hampel_sigma": 3.0,
    "use_moving_average": True,
    "ma_window": 7,
    "window_size": 150,
    "stride": 15,
    "n_delta": 75,
    "acf_unbiased": False,
    "positive_clip": True,
    "zero_column_fill": "uniform",

    # Flatten order for X:
    # "window_lag" -> [w0 lag0..lagN, w1 lag0..lagN, ...]
    # "lag_window" -> [lag0 w0..wN, lag1 w0..wN, ...]
    "flatten_order": "window_lag",

    "dtype": "float32",
    "overwrite": True,
    "save_preview_grid": True,
}


def parse_sample_name(name: str) -> Tuple[str, int]:
    prefix, index = name.rsplit("_", 1)
    return prefix, int(index)


def load_name_list(path: Path) -> List[str]:
    names: List[str] = []
    with path.open("r") as f:
        for line in f:
            line = line.strip()
            if line:
                names.append(line.split()[0])
    return names


def contiguous_runs(names: Sequence[str]) -> List[List[str]]:
    by_prefix: Dict[str, List[Tuple[int, str]]] = defaultdict(list)
    for name in names:
        prefix, idx = parse_sample_name(name)
        by_prefix[prefix].append((idx, name))

    runs: List[List[str]] = []
    for prefix in sorted(by_prefix):
        items = sorted(by_prefix[prefix])
        current: List[str] = []
        prev_idx = None
        for idx, name in items:
            if prev_idx is None or idx == prev_idx + 1:
                current.append(name)
            else:
                if current:
                    runs.append(current)
                current = [name]
            prev_idx = idx
        if current:
            runs.append(current)
    return runs


def build_groups(names: Sequence[str], group_size: int, group_stride: int, max_groups: int) -> List[List[str]]:
    groups: List[List[str]] = []
    for run in contiguous_runs(names):
        for start in range(0, len(run) - group_size + 1, group_stride):
            groups.append(run[start:start + group_size])
            if len(groups) >= max_groups:
                return groups
    return groups


def group_output_name(group: Sequence[str]) -> str:
    return f"{group[0]}_to_{group[-1]}"


def load_group_csi(group: Sequence[str], src_csi_root: Path, cfg: dict) -> np.ndarray:
    clips: List[np.ndarray] = []
    expected_clip = (
        cfg["expected_rx"],
        cfg["expected_tx"],
        cfg["expected_subcarrier"],
        cfg["expected_clip_time"],
    )
    for name in group:
        path = src_csi_root / f"{name}{cfg['input_ext']}"
        csi = load_csi_from_mat(str(path), mat_key=cfg["mat_key"])
        if csi.shape != expected_clip:
            raise ValueError(f"Unexpected CSI shape for {path}: got {csi.shape}, expected {expected_clip}")
        clips.append(csi)

    grouped = np.concatenate(clips, axis=3)
    expected_group = (
        cfg["expected_rx"],
        cfg["expected_tx"],
        cfg["expected_subcarrier"],
        cfg["expected_time"],
    )
    if grouped.shape != expected_group:
        raise ValueError(f"Unexpected grouped CSI shape: got {grouped.shape}, expected {expected_group}")
    return grouped


def extract_subcarrier_sdp_from_power(power_response: np.ndarray, cfg: dict) -> np.ndarray:
    if power_response.ndim != 2:
        raise ValueError(f"power_response must be 2D (Time, Subcarrier), got {power_response.shape}")
    nt, ns = power_response.shape
    window_size = cfg["window_size"]
    stride = cfg["stride"]
    n_delta = cfg["n_delta"]
    if window_size > nt:
        raise ValueError(f"window_size={window_size} > time length={nt}")
    if n_delta >= window_size:
        raise ValueError(f"n_delta={n_delta} must be < window_size={window_size}")
    if stride <= 0:
        raise ValueError(f"stride must be positive, got {stride}")

    window_starts = list(range(0, nt - window_size + 1, stride))
    n_windows = len(window_starts)
    sdp = np.zeros((ns, n_windows, n_delta), dtype=np.float32)

    for w_idx, start in enumerate(window_starts):
        end = start + window_size
        window_power = power_response[start:end, :]
        for sc in range(ns):
            sdp[sc, w_idx, :] = compute_acf_for_subcarrier(
                power_series=window_power[:, sc],
                n_delta=n_delta,
                unbiased=cfg["acf_unbiased"],
            )

    for sc in range(ns):
        normalized = normalize_sdp_columns(
            sdp[sc].T,
            positive_clip=cfg["positive_clip"],
            zero_column_fill=cfg["zero_column_fill"],
        )
        sdp[sc] = normalized.T
    return sdp.astype(np.float32)


def flatten_sdp_width(sdp_by_sc_window_lag: np.ndarray, flatten_order: str) -> np.ndarray:
    if sdp_by_sc_window_lag.ndim != 3:
        raise ValueError(f"Expected (SC, Window, Lag), got {sdp_by_sc_window_lag.shape}")
    if flatten_order == "window_lag":
        return sdp_by_sc_window_lag.reshape(sdp_by_sc_window_lag.shape[0], -1)
    if flatten_order == "lag_window":
        return np.transpose(sdp_by_sc_window_lag, (0, 2, 1)).reshape(sdp_by_sc_window_lag.shape[0], -1)
    raise ValueError(f"Unsupported flatten_order: {flatten_order}")


def unflatten_sdp_width(image_like_sdp: np.ndarray, cfg: dict) -> np.ndarray:
    if image_like_sdp.ndim != 3:
        raise ValueError(f"Expected image-like SDP shape (SC, Width, Channel), got {image_like_sdp.shape}")
    n_sc, width, n_ch = image_like_sdp.shape
    n_windows = (cfg["expected_time"] - cfg["window_size"]) // cfg["stride"] + 1
    n_delta = cfg["n_delta"]
    expected_width = n_windows * n_delta
    if width != expected_width:
        raise ValueError(f"Unexpected SDP width: got {width}, expected {expected_width}")

    if cfg["flatten_order"] == "window_lag":
        return image_like_sdp.reshape(n_sc, n_windows, n_delta, n_ch)
    if cfg["flatten_order"] == "lag_window":
        by_lag_window = image_like_sdp.reshape(n_sc, n_delta, n_windows, n_ch)
        return np.transpose(by_lag_window, (0, 2, 1, 3))
    raise ValueError(f"Unsupported flatten_order: {cfg['flatten_order']}")


def extract_imagelike_sdp_from_csi(csi: np.ndarray, cfg: dict):
    if csi.ndim != 4:
        raise ValueError(f"Expected canonical CSI shape (Rx, Tx, SC, T), got {csi.shape}")
    rx_n, tx_n, sc_n, _ = csi.shape

    link_sdp: List[List[np.ndarray]] = []
    link_amp: List[List[np.ndarray]] = []
    link_power: List[List[np.ndarray]] = []
    for rx in range(rx_n):
        row_sdp, row_amp, row_power = [], [], []
        for tx in range(tx_n):
            amp_raw = np.abs(csi[rx, tx]).T.astype(np.float32)
            amp_processed = preprocess_amplitude_matrix(amp_raw, cfg)
            power = compute_power_response(amp_processed)
            sdp_sc_window_lag = extract_subcarrier_sdp_from_power(power, cfg)
            row_sdp.append(sdp_sc_window_lag)
            row_amp.append(amp_processed.T)
            row_power.append(power.T)
        link_sdp.append(row_sdp)
        link_amp.append(row_amp)
        link_power.append(row_power)

    sdp_links = np.asarray(link_sdp, dtype=np.float32)
    amp_processed = np.asarray(link_amp, dtype=np.float32)
    power_response = np.asarray(link_power, dtype=np.float32)

    channel_axis = cfg["channel_axis"]
    if channel_axis == "rx":
        channel_sdp = sdp_links.mean(axis=1)
    elif channel_axis == "tx":
        channel_sdp = sdp_links.mean(axis=0)
    else:
        raise ValueError(f"Unsupported channel_axis: {channel_axis}")

    channels = []
    for ch in range(channel_sdp.shape[0]):
        channels.append(flatten_sdp_width(channel_sdp[ch], cfg["flatten_order"]))
    image_like = np.stack(channels, axis=-1)
    expected = (sc_n, channel_sdp.shape[2] * channel_sdp.shape[3], 3)
    if image_like.shape != expected:
        raise ValueError(f"Unexpected image-like SDP shape: got {image_like.shape}, expected {expected}")

    return image_like.astype(np.float32), amp_processed, power_response, sdp_links


def normalize_for_display(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    lo = np.percentile(x, 1.0)
    hi = np.percentile(x, 99.0)
    if hi - lo < EPS:
        return np.zeros_like(x, dtype=np.float32)
    return np.clip((x - lo) / (hi - lo), 0.0, 1.0)


def plot_imagelike_preview(
    sample_name: str,
    raw_csi: np.ndarray,
    amp_processed: np.ndarray,
    power_response: np.ndarray,
    image_like_sdp: np.ndarray,
    cfg: dict,
    out_dir: Path,
) -> Path:
    ensure_dir(out_dir)
    raw_amp_mean = np.abs(raw_csi).mean(axis=(0, 1))
    power_mean = power_response.mean(axis=(0, 1))
    rgb_view = normalize_for_display(image_like_sdp)
    sdp_sc_window_lag_channel = unflatten_sdp_width(image_like_sdp, cfg)
    channel_sums = sdp_sc_window_lag_channel.sum(axis=2)

    fig, axes = plt.subplots(2, 3, figsize=(15, 7))
    im0 = axes[0, 0].imshow(raw_amp_mean, aspect="auto", cmap="viridis")
    axes[0, 0].set_title("Raw amplitude")
    axes[0, 0].set_xlabel("Time")
    axes[0, 0].set_ylabel("Subcarrier")
    plt.colorbar(im0, ax=axes[0, 0], fraction=0.046, pad=0.04)

    im1 = axes[0, 1].imshow(power_mean, aspect="auto", cmap="magma")
    axes[0, 1].set_title("Processed power")
    axes[0, 1].set_xlabel("Time")
    axes[0, 1].set_ylabel("Subcarrier")
    plt.colorbar(im1, ax=axes[0, 1], fraction=0.046, pad=0.04)

    axes[0, 2].imshow(rgb_view, aspect="auto")
    axes[0, 2].set_title("Image-like SDP RGB view")
    axes[0, 2].set_xlabel("Flattened SDP width")
    axes[0, 2].set_ylabel("Subcarrier")

    for ch in range(3):
        im = axes[1, ch].imshow(image_like_sdp[:, :, ch], aspect="auto", cmap="plasma")
        axes[1, ch].set_title(
            f"Channel {ch} ({cfg['channel_axis']}) "
            f"sum={channel_sums[:, :, ch].min():.3f}/{channel_sums[:, :, ch].max():.3f}"
        )
        axes[1, ch].set_xlabel("Flattened SDP width")
        axes[1, ch].set_ylabel("Subcarrier")
        plt.colorbar(im, ax=axes[1, ch], fraction=0.046, pad=0.04)

    fig.suptitle(
        f"{sample_name} | shape={image_like_sdp.shape} | "
        f"window={cfg['window_size']} stride={cfg['stride']} lag={cfg['n_delta']} | "
        f"order={cfg['flatten_order']}"
    )
    plt.tight_layout(rect=[0, 0, 1, 0.95])

    save_path = out_dir / f"{Path(sample_name).stem}_imagelike_preview.png"
    fig.savefig(save_path, dpi=180)
    plt.close(fig)
    return save_path


def save_preview_grid(preview_images: Sequence[Path], out_path: Path) -> None:
    if not preview_images:
        return
    imgs = [Image.open(p).convert("RGB") for p in preview_images]
    thumb_w, thumb_h = 720, 360
    imgs = [img.resize((thumb_w, thumb_h)) for img in imgs]
    cols = 1
    rows = len(imgs)
    canvas = Image.new("RGB", (cols * thumb_w, rows * thumb_h), color=(255, 255, 255))
    for idx, img in enumerate(imgs):
        canvas.paste(img, (0, idx * thumb_h))
    canvas.save(out_path)


def write_group_list(groups: Sequence[Sequence[str]], out_path: Path) -> None:
    with out_path.open("w") as f:
        for group in groups:
            f.write(f"{group_output_name(group)} " + " ".join(group) + "\n")


def print_sample_stats(name: str, image_like_sdp: np.ndarray, cfg: dict) -> None:
    n_sc, width, n_ch = image_like_sdp.shape
    n_windows = (cfg["expected_time"] - cfg["window_size"]) // cfg["stride"] + 1
    expected_width = n_windows * cfg["n_delta"]
    sdp_sc_window_lag_channel = unflatten_sdp_width(image_like_sdp, cfg)
    lag_sum = sdp_sc_window_lag_channel.sum(axis=2)

    print(f"\n=== Preview: {name} ===")
    print("shape:", image_like_sdp.shape)
    print("expected shape:", (cfg["expected_subcarrier"], expected_width, 3))
    print("dtype:", image_like_sdp.dtype)
    print("min/max:", float(image_like_sdp.min()), float(image_like_sdp.max()))
    print("mean/std:", float(image_like_sdp.mean()), float(image_like_sdp.std()))
    print("nan:", bool(np.isnan(image_like_sdp).any()), "inf:", bool(np.isinf(image_like_sdp).any()))
    print("lag-sum min/max:", float(lag_sum.min()), float(lag_sum.max()))
    print("flatten_order:", cfg["flatten_order"])
    print("width/windows/lags/channels:", width, n_windows, cfg["n_delta"], n_ch)


def process_preview_groups(cfg: dict) -> None:
    src_root = Path(cfg["src_root"])
    dst_root = Path(cfg["dst_root"])
    src_csi_root = src_root / cfg["src_csi_dir"]
    dst_csi_root = dst_root / cfg["dst_csi_dir"]
    preview_root = dst_root / cfg["preview_dir"]
    list_path = src_root / f"{cfg['mode']}_data_list.txt"

    ensure_dir(dst_csi_root)
    ensure_dir(preview_root)

    names = load_name_list(list_path)
    groups = build_groups(
        names,
        group_size=cfg["group_size"],
        group_stride=cfg["group_stride"],
        max_groups=cfg["max_groups"],
    )
    if len(groups) < cfg["max_groups"]:
        raise RuntimeError(f"Only found {len(groups)} valid groups, expected {cfg['max_groups']}")

    write_group_list(groups, dst_root / cfg["group_list_name"])

    n_windows = (cfg["expected_time"] - cfg["window_size"]) // cfg["stride"] + 1
    width = n_windows * cfg["n_delta"]
    print("=" * 78)
    print("Offline SDP300 image-like power-ACF preview preprocessing")
    print("=" * 78)
    print("src_root       :", src_root)
    print("dst_root       :", dst_root)
    print("dst_csi        :", dst_csi_root)
    print("preview_root   :", preview_root)
    print("output layout  :", f"(Subcarrier, SDP-width, Antenna) = (30, {width}, 3)")
    print("channel_axis   :", cfg["channel_axis"])
    print("group_size     :", cfg["group_size"])
    print("group_time     :", cfg["expected_time"])
    print("window/stride  :", cfg["window_size"], cfg["stride"])
    print("n_windows      :", n_windows)
    print("n_delta        :", cfg["n_delta"])
    print("flatten_order  :", cfg["flatten_order"])
    print("num_groups     :", len(groups))
    print("=" * 78)

    preview_paths: List[Path] = []
    for group in tqdm(groups, desc="SDP300 image-like preview"):
        out_name = group_output_name(group)
        out_path = dst_csi_root / f"{out_name}.npy"
        if out_path.exists() and not cfg["overwrite"]:
            continue

        grouped_csi = load_group_csi(group, src_csi_root, cfg)
        image_like_sdp, amp_processed, power_response, _ = extract_imagelike_sdp_from_csi(grouped_csi, cfg)
        save_array(out_path, image_like_sdp, dtype=cfg["dtype"])
        print_sample_stats(out_name, image_like_sdp, cfg)

        preview_path = plot_imagelike_preview(
            sample_name=out_name,
            raw_csi=grouped_csi,
            amp_processed=amp_processed,
            power_response=power_response,
            image_like_sdp=image_like_sdp,
            cfg=cfg,
            out_dir=preview_root,
        )
        preview_paths.append(preview_path)

    if cfg["save_preview_grid"]:
        save_preview_grid(preview_paths, preview_root / "preview_first10_sdp300_imagelike_grid.png")

    plt.close("all")
    print("\nDone.")
    print(f"Saved image-like SDP300 files to: {dst_csi_root}")
    print(f"Saved preview figures to: {preview_root}")
    print(f"Saved group list to: {dst_root / cfg['group_list_name']}")


if __name__ == "__main__":
    process_preview_groups(CONFIG)
