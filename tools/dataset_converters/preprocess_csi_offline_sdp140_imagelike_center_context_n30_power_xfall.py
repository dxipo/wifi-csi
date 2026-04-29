#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Offline SDP140 image-like preprocessing with centered temporal context.

For each original sample name in train/test data lists, this script builds a
7-clip context:

    target i -> [i-3, ..., i, ..., i+3]

Each clip has 20 CSI time samples, so the grouped context has 140 samples
at 300 Hz, i.e. about 466.7 ms. At sequence boundaries, the nearest valid
clip is repeated so the output sample count stays exactly aligned with the
original labels/data lists.

SDP parameters:

    window_size = 60   # 200.0 ms local ACF window
    stride      = 10   # 33.3 ms step
    n_delta     = 30   # 100.0 ms maximum lag

Two image-like layouts are saved from the same SDP computation:

    window_lag: [window0 lag0..lagN, window1 lag0..lagN, ...]
    lag_window: [lag0 window0..windowN, lag1 window0..windowN, ...]

Output shape for both layouts:

    (Subcarrier, SDP-width, Antenna-channel) = (30, 270, 3)
"""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Iterable, List, Sequence, Tuple
import sys

import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from preprocess_csi_sdp_offline_power_xfall import (
    EPS,
    compute_power_response,
    ensure_dir,
    normalize_sdp_columns,
    preprocess_amplitude_matrix,
    save_array,
)
from preprocess_csi_offline_sdp300_imagelike_center_context_power_xfall import (
    build_center_contexts,
    choose_preview_names,
    copy_metadata_for_split,
    count_existing,
    load_context_csi,
    load_name_list,
    write_context_list,
    write_preview_list,
)
from preprocess_csi_offline_sdp300_imagelike_power_xfall_preview import (
    flatten_sdp_width,
    plot_imagelike_preview,
    save_preview_grid,
)


CONFIG = {
    "splits": [
        {
            "name": "train",
            "src_root": "/home/xl/CSI/Person-in-WiFi-3D-repo/data/wifipose/all_single_train_data_hold_out",
            "dst_root": "/home/xl/CSI/Person-in-WiFi-3D-repo/data/wifipose/all_single_train_data_hold_out_sdp140_imagelike_centerctx_n30_power_xfall",
            "list_name": "train_data_list.txt",
        },
        {
            "name": "test",
            "src_root": "/home/xl/CSI/Person-in-WiFi-3D-repo/data/wifipose/all_single_test_data_hold_out",
            "dst_root": "/home/xl/CSI/Person-in-WiFi-3D-repo/data/wifipose/all_single_test_data_hold_out_sdp140_imagelike_centerctx_n30_power_xfall",
            "list_name": "test_data_list.txt",
        },
    ],
    "src_csi_dir": "csi",
    "dst_window_lag_dir": "csi_sdp140_imagelike_windowlag_offline",
    "dst_lag_window_dir": "csi_sdp140_imagelike_lagwindow_offline",
    "preview_window_lag_dir": "preview_sdp140_imagelike_windowlag",
    "preview_lag_window_dir": "preview_sdp140_imagelike_lagwindow",
    "context_list_name": "sdp140_center_context_list.txt",
    "preview_list_name": "sdp140_preview_samples.txt",
    "input_ext": ".mat",
    "mat_key": "csi_out",

    "sample_rate_hz": 300,
    "group_size": 7,
    "context_radius": 3,

    "expected_rx": 3,
    "expected_tx": 3,
    "expected_subcarrier": 30,
    "expected_clip_time": 20,
    "expected_time": 140,

    # Preserve Rx as the 3 image-like channels and average over Tx.
    "channel_axis": "rx",

    "use_hampel": True,
    "hampel_window": 5,
    "hampel_sigma": 3.0,
    "use_moving_average": True,
    "ma_window": 7,
    "window_size": 60,
    "stride": 10,
    "n_delta": 30,
    "acf_unbiased": False,
    "positive_clip": True,
    "zero_column_fill": "uniform",

    "dtype": "float32",
    "overwrite": False,
    "copy_metadata": True,
    "preview_count": 10,
    "preview_seed": 20260430,
    "save_preview_grid": True,
    "num_workers": 16,
    "parallel_chunksize": 32,
}


def extract_subcarrier_sdp_from_power_fast(power_response: np.ndarray, cfg: dict) -> np.ndarray:
    if power_response.ndim != 2:
        raise ValueError(f"power_response must be 2D (Time, Subcarrier), got {power_response.shape}")
    nt, ns = power_response.shape
    window_size = int(cfg["window_size"])
    stride = int(cfg["stride"])
    n_delta = int(cfg["n_delta"])
    if window_size > nt:
        raise ValueError(f"window_size={window_size} > time length={nt}")
    if n_delta >= window_size:
        raise ValueError(f"n_delta={n_delta} must be < window_size={window_size}")
    if stride <= 0:
        raise ValueError(f"stride must be positive, got {stride}")

    window_starts = list(range(0, nt - window_size + 1, stride))
    sdp = np.zeros((ns, len(window_starts), n_delta), dtype=np.float32)

    for w_idx, start in enumerate(window_starts):
        window_power = power_response[start:start + window_size, :].astype(np.float32, copy=False)
        x = window_power - window_power.mean(axis=0, keepdims=True)
        var = np.var(x, axis=0).astype(np.float32)
        valid = var >= EPS

        if not np.any(valid):
            continue

        acf = np.zeros((n_delta, ns), dtype=np.float32)
        for lag in range(1, n_delta + 1):
            denom_n = (window_size - lag) if cfg["acf_unbiased"] else window_size
            val = np.sum(x[lag:, :] * x[:-lag, :], axis=0) / max(float(denom_n), EPS)
            acf[lag - 1, valid] = val[valid] / np.maximum(var[valid], EPS)
        sdp[:, w_idx, :] = acf.T

    for sc in range(ns):
        normalized = normalize_sdp_columns(
            sdp[sc].T,
            positive_clip=cfg["positive_clip"],
            zero_column_fill=cfg["zero_column_fill"],
        )
        sdp[sc] = normalized.T
    return sdp.astype(np.float32)


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


def extract_imagelike_sdp_from_csi_fast(csi: np.ndarray, cfg: dict):
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
            sdp_sc_window_lag = extract_subcarrier_sdp_from_power_fast(power, cfg)
            row_sdp.append(sdp_sc_window_lag)
            row_amp.append(amp_processed.T)
            row_power.append(power.T)
        link_sdp.append(row_sdp)
        link_amp.append(row_amp)
        link_power.append(row_power)

    sdp_links = np.asarray(link_sdp, dtype=np.float32)
    amp_processed = np.asarray(link_amp, dtype=np.float32)
    power_response = np.asarray(link_power, dtype=np.float32)
    image_like = image_like_from_sdp_links(sdp_links, cfg, cfg["flatten_order"])
    expected = (sc_n, sdp_links.shape[3] * sdp_links.shape[4], 3)
    if image_like.shape != expected:
        raise ValueError(f"Unexpected image-like SDP shape: got {image_like.shape}, expected {expected}")
    return image_like.astype(np.float32), amp_processed, power_response, sdp_links


def compute_both_layouts(grouped_csi: np.ndarray, cfg: dict):
    compute_cfg = dict(cfg)
    compute_cfg["flatten_order"] = "window_lag"
    window_lag, amp_processed, power_response, sdp_links = extract_imagelike_sdp_from_csi_fast(
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
    context_ms = cfg["expected_time"] / cfg["sample_rate_hz"] * 1000.0
    window_ms = cfg["window_size"] / cfg["sample_rate_hz"] * 1000.0
    stride_ms = cfg["stride"] / cfg["sample_rate_hz"] * 1000.0
    lag_ms = cfg["n_delta"] / cfg["sample_rate_hz"] * 1000.0

    print("=" * 78)
    print(f"SDP140 image-like centered-context preprocessing: {split_name}")
    print("=" * 78)
    print("src_root        :", src_root)
    print("dst_root        :", dst_root)
    print("num samples     :", len(names))
    print("output shape    :", (cfg["expected_subcarrier"], width, 3))
    print(
        "context         :",
        f"{cfg['group_size']} clips -> {cfg['expected_time']} CSI time samples ({context_ms:.1f} ms)",
    )
    print(
        "window/stride   :",
        f"{cfg['window_size']} / {cfg['stride']} samples ({window_ms:.1f} / {stride_ms:.1f} ms)",
    )
    print("n_windows       :", n_windows)
    print("n_delta         :", f"{cfg['n_delta']} samples ({lag_ms:.1f} ms max lag)")
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
            save_preview_grid(
                preview_window_paths,
                preview_window_root / "preview_first10_sdp140_imagelike_windowlag_grid.png",
            )
        if preview_lag_paths:
            save_preview_grid(
                preview_lag_paths,
                preview_lag_root / "preview_first10_sdp140_imagelike_lagwindow_grid.png",
            )

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
