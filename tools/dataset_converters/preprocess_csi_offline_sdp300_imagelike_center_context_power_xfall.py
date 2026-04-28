#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Full offline SDP300 image-like preprocessing with centered temporal context.

For each original sample name in train/test data lists, this script builds a
15-clip context:

    target i -> [i-7, ..., i, ..., i+7]

Each clip has 20 CSI time samples, so the grouped context has 300 samples. At
sequence boundaries, the nearest valid clip is repeated so the output sample
count stays exactly aligned with the original labels/data lists.

Two image-like layouts are saved from the same SDP computation:

    window_lag: [window0 lag0..lagN, window1 lag0..lagN, ...]
    lag_window: [lag0 window0..windowN, lag1 window0..windowN, ...]

Output shape for both layouts:

    (Subcarrier, SDP-width, Antenna-channel) = (30, 825, 3)

The core SDP computation path is the same as the previous power-ACF SDP script:
amplitude -> Hampel/moving-average -> power -> windowed ACF
-> positive clipping -> column normalization.
"""

from __future__ import annotations

from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from functools import lru_cache
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple
import random
import shutil
import sys

import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from preprocess_csi_sdp_offline_power_xfall import (
    ensure_dir,
    load_csi_from_mat,
    maybe_copy_metadata,
    save_array,
)
from preprocess_csi_offline_sdp300_imagelike_power_xfall_preview import (
    extract_imagelike_sdp_from_csi,
    flatten_sdp_width,
    plot_imagelike_preview,
    save_preview_grid,
)


CONFIG = {
    "splits": [
        {
            "name": "train",
            "src_root": "/home/xl/CSI/Person-in-WiFi-3D-repo/data/wifipose/all_single_train_data_hold_out",
            "dst_root": "/home/xl/CSI/Person-in-WiFi-3D-repo/data/wifipose/all_single_train_data_hold_out_sdp300_imagelike_centerctx_power_xfall",
            "list_name": "train_data_list.txt",
        },
        {
            "name": "test",
            "src_root": "/home/xl/CSI/Person-in-WiFi-3D-repo/data/wifipose/all_single_test_data_hold_out",
            "dst_root": "/home/xl/CSI/Person-in-WiFi-3D-repo/data/wifipose/all_single_test_data_hold_out_sdp300_imagelike_centerctx_power_xfall",
            "list_name": "test_data_list.txt",
        },
    ],
    "src_csi_dir": "csi",
    "dst_window_lag_dir": "csi_sdp300_imagelike_windowlag_offline",
    "dst_lag_window_dir": "csi_sdp300_imagelike_lagwindow_offline",
    "preview_window_lag_dir": "preview_sdp300_imagelike_windowlag",
    "preview_lag_window_dir": "preview_sdp300_imagelike_lagwindow",
    "context_list_name": "sdp300_center_context_list.txt",
    "preview_list_name": "sdp300_preview_samples.txt",
    "input_ext": ".mat",
    "mat_key": "csi_out",

    "sample_rate_hz": 300,
    "group_size": 15,
    "context_radius": 7,

    "expected_rx": 3,
    "expected_tx": 3,
    "expected_subcarrier": 30,
    "expected_clip_time": 20,
    "expected_time": 300,

    # Preserve Rx as the 3 image-like channels and average over Tx.
    "channel_axis": "rx",

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

    "dtype": "float32",
    "overwrite": False,
    "copy_metadata": True,
    "preview_count": 10,
    "preview_seed": 20260428,
    "save_preview_grid": True,
    "num_workers": 8,
    "parallel_chunksize": 16,
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


def build_center_contexts(names: Sequence[str], cfg: dict) -> Dict[str, List[str]]:
    radius = int(cfg["context_radius"])
    group_size = int(cfg["group_size"])
    if group_size != radius * 2 + 1:
        raise ValueError(f"group_size must equal 2*context_radius+1, got {group_size} and {radius}")

    contexts: Dict[str, List[str]] = {}
    for run in contiguous_runs(names):
        if not run:
            continue
        for pos, target in enumerate(run):
            context: List[str] = []
            for offset in range(-radius, radius + 1):
                src_pos = min(max(pos + offset, 0), len(run) - 1)
                context.append(run[src_pos])
            contexts[target] = context

    missing = [name for name in names if name not in contexts]
    if missing:
        raise RuntimeError(f"Failed to build contexts for {len(missing)} names, first: {missing[0]}")
    return contexts


def choose_preview_names(names: Sequence[str], contexts: Dict[str, List[str]], cfg: dict) -> List[str]:
    preview_count = int(cfg["preview_count"])
    rng = random.Random(int(cfg["preview_seed"]))
    runs = contiguous_runs(names)

    boundary: List[str] = []
    non_boundary: List[str] = []
    for run in runs:
        if not run:
            continue
        boundary.append(run[0])
        if len(run) > 1:
            boundary.append(run[-1])
        if len(run) > cfg["group_size"]:
            non_boundary.extend(run[cfg["context_radius"]:-cfg["context_radius"]])

    selected: List[str] = []
    rng.shuffle(boundary)
    rng.shuffle(non_boundary)
    for name in boundary[: max(2, preview_count // 3)]:
        if name not in selected:
            selected.append(name)
    for name in non_boundary:
        if len(selected) >= preview_count:
            break
        if name not in selected:
            selected.append(name)
    for name in names:
        if len(selected) >= preview_count:
            break
        if name not in selected:
            selected.append(name)

    return selected[:preview_count]


def write_context_list(names: Sequence[str], contexts: Dict[str, List[str]], out_path: Path) -> None:
    with out_path.open("w") as f:
        for name in names:
            f.write(f"{name} " + " ".join(contexts[name]) + "\n")


def write_preview_list(preview_names: Sequence[str], contexts: Dict[str, List[str]], out_path: Path) -> None:
    with out_path.open("w") as f:
        for name in preview_names:
            context = contexts[name]
            is_boundary = len(set(context)) < len(context)
            kind = "boundary" if is_boundary else "non_boundary"
            f.write(f"{name} {kind} " + " ".join(context) + "\n")


def copy_metadata_for_split(src_root: Path, dst_root: Path, cfg: dict) -> None:
    if not cfg["copy_metadata"]:
        ensure_dir(dst_root)
        return

    maybe_copy_metadata(src_root, dst_root, cfg["src_csi_dir"], cfg["dst_window_lag_dir"])
    ensure_dir(dst_root / cfg["dst_lag_window_dir"])


@lru_cache(maxsize=512)
def cached_load_csi(path_str: str, mat_key: str) -> np.ndarray:
    return load_csi_from_mat(path_str, mat_key=mat_key)


def load_context_csi(context: Sequence[str], src_csi_root: Path, cfg: dict) -> np.ndarray:
    clips: List[np.ndarray] = []
    expected_clip = (
        cfg["expected_rx"],
        cfg["expected_tx"],
        cfg["expected_subcarrier"],
        cfg["expected_clip_time"],
    )
    for name in context:
        path = src_csi_root / f"{name}{cfg['input_ext']}"
        csi = cached_load_csi(str(path), cfg["mat_key"])
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


def image_like_from_sdp_links(sdp_links: np.ndarray, cfg: dict, flatten_order: str) -> np.ndarray:
    if cfg["channel_axis"] == "rx":
        channel_sdp = sdp_links.mean(axis=1)
    elif cfg["channel_axis"] == "tx":
        channel_sdp = sdp_links.mean(axis=0)
    else:
        raise ValueError(f"Unsupported channel_axis: {cfg['channel_axis']}")

    channels = [
        flatten_sdp_width(channel_sdp[ch], flatten_order)
        for ch in range(channel_sdp.shape[0])
    ]
    return np.stack(channels, axis=-1).astype(np.float32)


def compute_both_layouts(grouped_csi: np.ndarray, cfg: dict):
    compute_cfg = dict(cfg)
    compute_cfg["flatten_order"] = "window_lag"
    window_lag, amp_processed, power_response, sdp_links = extract_imagelike_sdp_from_csi(
        grouped_csi,
        compute_cfg,
    )
    lag_window = image_like_from_sdp_links(sdp_links, cfg, "lag_window")
    return window_lag, lag_window, amp_processed, power_response


def process_one_target(args: Tuple[str, List[str], str, str, str, dict]) -> str:
    target, context, src_csi_root_str, dst_window_root_str, dst_lag_root_str, cfg = args
    dst_window_root = Path(dst_window_root_str)
    dst_lag_root = Path(dst_lag_root_str)
    out_window = dst_window_root / f"{target}.npy"
    out_lag = dst_lag_root / f"{target}.npy"
    if out_window.exists() and out_lag.exists() and not cfg["overwrite"]:
        return "skipped"

    grouped_csi = load_context_csi(context, Path(src_csi_root_str), cfg)
    window_lag, lag_window, _, _ = compute_both_layouts(grouped_csi, cfg)
    save_array(out_window, window_lag, dtype=cfg["dtype"])
    save_array(out_lag, lag_window, dtype=cfg["dtype"])
    return "processed"


def process_preview_target(
    target: str,
    context: Sequence[str],
    split_name: str,
    src_csi_root: Path,
    dst_window_root: Path,
    dst_lag_root: Path,
    preview_window_root: Path,
    preview_lag_root: Path,
    cfg: dict,
) -> Tuple[Path, Path]:
    grouped_csi = load_context_csi(context, src_csi_root, cfg)
    window_lag, lag_window, amp_processed, power_response = compute_both_layouts(grouped_csi, cfg)
    save_array(dst_window_root / f"{target}.npy", window_lag, dtype=cfg["dtype"])
    save_array(dst_lag_root / f"{target}.npy", lag_window, dtype=cfg["dtype"])

    window_cfg = dict(cfg)
    window_cfg["flatten_order"] = "window_lag"
    lag_cfg = dict(cfg)
    lag_cfg["flatten_order"] = "lag_window"

    is_boundary = len(set(context)) < len(context)
    sample_name = f"{split_name}_{target}_{'boundary' if is_boundary else 'nonboundary'}"
    window_preview = plot_imagelike_preview(
        sample_name=sample_name,
        raw_csi=grouped_csi,
        amp_processed=amp_processed,
        power_response=power_response,
        image_like_sdp=window_lag,
        cfg=window_cfg,
        out_dir=preview_window_root,
    )
    lag_preview = plot_imagelike_preview(
        sample_name=sample_name,
        raw_csi=grouped_csi,
        amp_processed=amp_processed,
        power_response=power_response,
        image_like_sdp=lag_window,
        cfg=lag_cfg,
        out_dir=preview_lag_root,
    )
    return window_preview, lag_preview


def count_existing(names: Sequence[str], dst_root: Path) -> int:
    return sum(1 for name in names if (dst_root / f"{name}.npy").exists())


def process_split(split_cfg: dict, cfg: dict) -> None:
    split_name = split_cfg["name"]
    src_root = Path(split_cfg["src_root"])
    dst_root = Path(split_cfg["dst_root"])
    src_csi_root = src_root / cfg["src_csi_dir"]
    dst_window_root = dst_root / cfg["dst_window_lag_dir"]
    dst_lag_root = dst_root / cfg["dst_lag_window_dir"]
    preview_window_root = dst_root / cfg["preview_window_lag_dir"]
    preview_lag_root = dst_root / cfg["preview_lag_window_dir"]
    list_path = src_root / split_cfg["list_name"]

    if not src_csi_root.exists():
        raise FileNotFoundError(f"Source CSI dir not found: {src_csi_root}")
    if not list_path.exists():
        raise FileNotFoundError(f"Data list not found: {list_path}")

    copy_metadata_for_split(src_root, dst_root, cfg)
    for path in (dst_window_root, dst_lag_root, preview_window_root, preview_lag_root):
        ensure_dir(path)

    names = load_name_list(list_path)
    contexts = build_center_contexts(names, cfg)
    preview_names = choose_preview_names(names, contexts, cfg)
    write_context_list(names, contexts, dst_root / cfg["context_list_name"])
    write_preview_list(preview_names, contexts, dst_root / cfg["preview_list_name"])

    n_windows = (cfg["expected_time"] - cfg["window_size"]) // cfg["stride"] + 1
    width = n_windows * cfg["n_delta"]
    print("=" * 78)
    print(f"SDP300 image-like centered-context preprocessing: {split_name}")
    print("=" * 78)
    print("src_root        :", src_root)
    print("dst_root        :", dst_root)
    print("num samples     :", len(names))
    print("output shape    :", (cfg["expected_subcarrier"], width, 3))
    print("context         :", f"{cfg['group_size']} clips -> {cfg['expected_time']} CSI time samples")
    print("window/stride   :", cfg["window_size"], cfg["stride"])
    print("n_windows       :", n_windows)
    print("n_delta         :", cfg["n_delta"])
    print("window_lag dir  :", dst_window_root)
    print("lag_window dir  :", dst_lag_root)
    print("preview samples :", len(preview_names))
    print("num_workers     :", cfg["num_workers"])
    print("overwrite       :", cfg["overwrite"])
    print("=" * 78)

    preview_window_paths: List[Path] = []
    preview_lag_paths: List[Path] = []
    preview_set = set(preview_names)

    for target in tqdm(preview_names, desc=f"{split_name} preview"):
        out_window = dst_window_root / f"{target}.npy"
        out_lag = dst_lag_root / f"{target}.npy"
        if out_window.exists() and out_lag.exists() and not cfg["overwrite"]:
            continue
        window_preview, lag_preview = process_preview_target(
            target=target,
            context=contexts[target],
            split_name=split_name,
            src_csi_root=src_csi_root,
            dst_window_root=dst_window_root,
            dst_lag_root=dst_lag_root,
            preview_window_root=preview_window_root,
            preview_lag_root=preview_lag_root,
            cfg=cfg,
        )
        preview_window_paths.append(window_preview)
        preview_lag_paths.append(lag_preview)

    tasks = [
        (
            name,
            contexts[name],
            str(src_csi_root),
            str(dst_window_root),
            str(dst_lag_root),
            cfg,
        )
        for name in names
        if name not in preview_set
    ]

    num_workers = min(max(1, int(cfg["num_workers"])), max(1, len(tasks)))
    processed = 0
    skipped = 0
    if tasks:
        if num_workers == 1:
            iterator: Iterable[str] = map(process_one_target, tasks)
            for status in tqdm(iterator, total=len(tasks), desc=f"{split_name} full"):
                processed += int(status == "processed")
                skipped += int(status == "skipped")
        else:
            with ProcessPoolExecutor(max_workers=num_workers) as executor:
                iterator = executor.map(
                    process_one_target,
                    tasks,
                    chunksize=max(1, int(cfg["parallel_chunksize"])),
                )
                for status in tqdm(iterator, total=len(tasks), desc=f"{split_name} full"):
                    processed += int(status == "processed")
                    skipped += int(status == "skipped")

    if cfg["save_preview_grid"]:
        if preview_window_paths:
            save_preview_grid(preview_window_paths, preview_window_root / "preview_first10_sdp300_imagelike_windowlag_grid.png")
        if preview_lag_paths:
            save_preview_grid(preview_lag_paths, preview_lag_root / "preview_first10_sdp300_imagelike_lagwindow_grid.png")

    existing_window = count_existing(names, dst_window_root)
    existing_lag = count_existing(names, dst_lag_root)
    print(f"\nDone split: {split_name}")
    print("processed/skipped full tasks:", processed, skipped)
    print("window_lag files:", existing_window, "/", len(names))
    print("lag_window files:", existing_lag, "/", len(names))
    print("context list:", dst_root / cfg["context_list_name"])
    print("preview list:", dst_root / cfg["preview_list_name"])


def main() -> None:
    cfg = CONFIG
    if cfg["group_size"] != cfg["context_radius"] * 2 + 1:
        raise ValueError("group_size must equal context_radius * 2 + 1")
    for split_cfg in cfg["splits"]:
        process_split(split_cfg, cfg)
    plt.close("all")


if __name__ == "__main__":
    main()
