#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Preview-only offline SDP300 preprocessing.

This script groups 15 consecutive CSI .mat clips. Each clip has 20 time samples,
so one grouped sample has 300 time samples, matching a 1000 ms window at 300 Hz.

The core SDP algorithm is intentionally reused from
preprocess_csi_sdp_offline_power_xfall.py:
    amplitude -> Hampel/moving-average -> power -> windowed ACF
    -> subcarrier average -> positive clipping -> column normalization

Only grouping, SDP300 defaults, and visualization/output paths are new.
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
    ensure_dir,
    extract_sdp_from_csi,
    load_csi_from_mat,
    plot_preview_sample,
    print_sample_stats,
    save_array,
)


CONFIG = {
    "src_root": "/home/xl/CSI/Person-in-WiFi-3D-repo/data/wifipose/all_single_test_data_hold_out",
    "dst_root": "/home/xl/CSI/Person-in-WiFi-3D-repo/data/wifipose/all_single_test_data_hold_out_offline_sdp300_power_xfall_preview",
    "src_csi_dir": "csi",
    "dst_csi_dir": "csi_sdp300_offline",
    "preview_dir": "preview_sdp300",
    "group_list_name": "sdp300_group_list.txt",
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

    # SDP300 defaults:
    # - group_size 15 gives a 1000 ms CSI window.
    # - window_size 150 estimates local ACF on 500 ms subwindows.
    # - stride 15 advances by 50 ms.
    # - n_delta 75 covers lags up to 250 ms, while staying below window_size.
    "use_hampel": True,
    "hampel_window": 5,
    "hampel_sigma": 3.0,
    "use_moving_average": True,
    "ma_window": 7,
    "window_size": 150,
    "stride": 15,
    "n_delta": 75,
    "layout": "wtn",
    "acf_unbiased": False,
    "positive_clip": True,
    "zero_column_fill": "uniform",

    "dtype": "float32",
    "overwrite": True,
    "save_preview_stats": True,
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
    for name in group:
        path = src_csi_root / f"{name}{cfg['input_ext']}"
        csi = load_csi_from_mat(str(path), mat_key=cfg["mat_key"])
        expected = (
            cfg["expected_rx"],
            cfg["expected_tx"],
            cfg["expected_subcarrier"],
            cfg["expected_clip_time"],
        )
        if csi.shape != expected:
            raise ValueError(f"Unexpected CSI shape for {path}: got {csi.shape}, expected {expected}")
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


def save_preview_grid(preview_images: Sequence[Path], out_path: Path) -> None:
    if not preview_images:
        return
    imgs = [Image.open(p).convert("RGB") for p in preview_images]
    thumb_w, thumb_h = 560, 360
    imgs = [img.resize((thumb_w, thumb_h)) for img in imgs]
    cols = 2
    rows = int(np.ceil(len(imgs) / cols))
    canvas = Image.new("RGB", (cols * thumb_w, rows * thumb_h), color=(255, 255, 255))
    for idx, img in enumerate(imgs):
        row = idx // cols
        col = idx % cols
        canvas.paste(img, (col * thumb_w, row * thumb_h))
    canvas.save(out_path)


def write_group_list(groups: Sequence[Sequence[str]], out_path: Path) -> None:
    with out_path.open("w") as f:
        for group in groups:
            f.write(f"{group_output_name(group)} " + " ".join(group) + "\n")


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

    print("=" * 78)
    print("Offline SDP300 power-ACF preview preprocessing")
    print("=" * 78)
    print("src_root     :", src_root)
    print("dst_root     :", dst_root)
    print("dst_csi      :", dst_csi_root)
    print("preview_root :", preview_root)
    print("group_size   :", cfg["group_size"])
    print("group_stride :", cfg["group_stride"])
    print("sample_rate  :", cfg["sample_rate_hz"])
    print("group_time   :", cfg["expected_time"])
    print("window_size  :", cfg["window_size"])
    print("stride       :", cfg["stride"])
    print("n_delta      :", cfg["n_delta"])
    print("num_groups   :", len(groups))
    print("=" * 78)

    preview_paths: List[Path] = []
    for group in tqdm(groups, desc="SDP300 preview"):
        out_name = group_output_name(group)
        out_path = dst_csi_root / f"{out_name}.npy"
        if out_path.exists() and not cfg["overwrite"]:
            continue

        grouped_csi = load_group_csi(group, src_csi_root, cfg)
        sdp, amp_processed, power_response = extract_sdp_from_csi(grouped_csi, cfg)
        save_array(out_path, sdp, dtype=cfg["dtype"])

        if cfg["save_preview_stats"]:
            print_sample_stats(out_name, sdp, cfg)

        preview_path = plot_preview_sample(
            sample_name=out_name,
            raw_csi=grouped_csi,
            amp_processed=amp_processed,
            power_response=power_response,
            sdp=sdp,
            cfg=cfg,
            out_dir=preview_root,
        )
        preview_paths.append(preview_path)

    if cfg["save_preview_grid"]:
        save_preview_grid(preview_paths, preview_root / "preview_first10_sdp300_grid.png")

    plt.close("all")
    print("\nDone.")
    print(f"Saved SDP300 files to: {dst_csi_root}")
    print(f"Saved preview figures to: {preview_root}")
    print(f"Saved group list to: {dst_root / cfg['group_list_name']}")


if __name__ == "__main__":
    process_preview_groups(CONFIG)
