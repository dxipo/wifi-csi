
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Formal offline SDP preprocessing script adapted for CSI samples shaped (3, 3, 30, 20).

Main goals
----------
1. Read raw CSI .mat files and canonicalize to (Rx, Tx, Subcarrier, Time)
2. Build an ACF-based SDP feature from amplitude/power information
3. Use XFall-style column-sum normalization (each SDP column sums to 1)
4. Save offline .npy files for later dataloader use
5. Save preview plots for the first N samples

Notes
-----
- This script is designed for short CSI clips such as (3, 3, 30, 20).
- Because the time dimension is only 20, we use windowed power-ACF extraction:
      amplitude -> optional Hampel + moving-average smoothing -> power = amplitude^2
      per window/per subcarrier ACF -> average over subcarriers -> column-sum normalization
- This is closer to XFall normalization than an L2-normalized heatmap, but it is still
  an engineering variant rather than a line-by-line reproduction of XFall's original formula.

Output layout
-------------
- layout='nwt' -> (Rx, Tx, N_delta, N_windows)
- layout='wtn' -> (Rx, Tx, N_windows, N_delta)
"""

from __future__ import annotations

import os
import shutil
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import List, Tuple

import h5py
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

EPS = 1e-8

CONFIG = {
    "src_root": "/home/xl/CSI/Person-in-WiFi-3D-repo/data/wifipose/all_single_train_data_hold_out",
    "dst_root": "/home/xl/CSI/Person-in-WiFi-3D-repo/data/wifipose/all_single_train_data_hold_out_sdp_offline_power_xfall",
    "src_csi_dir": "csi",
    "dst_csi_dir": "csi_sdp_offline",
    "input_ext": ".mat",
    "mat_key": "csi_out",

    "expected_rx": 3,
    "expected_tx": 3,
    "expected_subcarrier": 30,
    "expected_time": 20,

    "use_hampel": True,
    "hampel_window": 2,
    "hampel_sigma": 3.0,
    "use_moving_average": True,
    "ma_window": 3,

    "window_size": 12,
    "stride": 1,
    "n_delta": 6,
    "layout": "wtn",
    "acf_unbiased": False,
    "positive_clip": True,
    "zero_column_fill": "uniform",

    "dtype": "float32",
    "overwrite": False,
    "copy_metadata": True,

    "preview_limit": 10,
    "save_preview_stats": True,
    "preview_dir": "preview_sdp",
    "save_preview_grid": True,

    "num_workers": 8,
    "parallel_chunksize": 32,
}


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def maybe_copy_metadata(src_root: Path, dst_root: Path, src_csi_dir: str, dst_csi_dir: str) -> None:
    ensure_dir(dst_root)
    for item in src_root.iterdir():
        if item.name == src_csi_dir:
            continue
        target = dst_root / item.name
        if item.is_file() and item.suffix.lower() == ".txt":
            shutil.copy2(item, target)
        elif item.is_dir() and not target.exists():
            try:
                os.symlink(item.resolve(), target)
            except OSError:
                shutil.copytree(item, target)
    ensure_dir(dst_root / dst_csi_dir)


def structured_to_complex(arr: np.ndarray) -> np.ndarray:
    if arr.dtype.fields is None:
        return np.asarray(arr)
    field_names = list(arr.dtype.fields.keys())
    if "real" in field_names and "imag" in field_names:
        return arr["real"] + 1j * arr["imag"]
    raise ValueError(f"Unsupported structured dtype fields: {field_names}")


def load_csi_from_mat(path: str, mat_key: str = "csi_out") -> np.ndarray:
    with h5py.File(path, "r") as f:
        if mat_key not in f:
            raise KeyError(f"Key '{mat_key}' not found in {path}. Available keys: {list(f.keys())}")
        raw = f[mat_key][()]
    raw = structured_to_complex(np.array(raw))
    return canonicalize_csi(raw, path=path)


def canonicalize_csi(csi: np.ndarray, path: str = "") -> np.ndarray:
    csi = np.asarray(csi)

    if csi.ndim == 4:
        if csi.shape == (3, 3, 30, 20):
            return csi
        if csi.shape == (20, 30, 3, 3):
            return np.transpose(csi, (2, 3, 1, 0))
        if csi.shape == (3, 3, 20, 30):
            return np.transpose(csi, (0, 1, 3, 2))
        if csi.shape[0] == 3 and csi.shape[1] == 3:
            if csi.shape[2] == 30:
                return csi
            if csi.shape[3] == 30:
                return np.transpose(csi, (0, 1, 3, 2))
        if csi.shape[2] == 3 and csi.shape[3] == 3:
            if csi.shape[1] == 30:
                return np.transpose(csi, (2, 3, 1, 0))
            if csi.shape[0] == 30:
                return np.transpose(csi, (2, 3, 0, 1))

    if csi.ndim == 2:
        if csi.shape == (30, 20):
            return csi[None, None, :, :]
        if csi.shape == (20, 30):
            return csi.T[None, None, :, :]

    raise ValueError(f"Cannot canonicalize CSI shape {csi.shape} from {path}")


def validate_canonical_shape(csi: np.ndarray, cfg: dict) -> None:
    if csi.ndim != 4:
        raise ValueError(f"Expected 4D canonical CSI, got shape {csi.shape}")
    rx, tx, sc, t = csi.shape
    expected = (
        cfg["expected_rx"],
        cfg["expected_tx"],
        cfg["expected_subcarrier"],
        cfg["expected_time"],
    )
    if (rx, tx, sc, t) != expected:
        raise ValueError(f"Unexpected canonical shape: got {(rx, tx, sc, t)}, expected {expected}")


def hampel_filter_1d(x: np.ndarray, window: int = 2, sigma: float = 3.0) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    n = len(x)
    y = x.copy()
    if n == 0:
        return y
    for i in range(n):
        left = max(0, i - window)
        right = min(n, i + window + 1)
        win = x[left:right]
        med = np.median(win)
        mad = np.median(np.abs(win - med))
        scale = max(1.4826 * mad, EPS)
        if abs(x[i] - med) > sigma * scale:
            y[i] = med
    return y


def moving_average_1d(x: np.ndarray, window: int = 3) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    if window <= 1 or len(x) < 2:
        return x.copy()
    w = min(window, len(x))
    kernel = np.ones(w, dtype=np.float32) / float(w)
    return np.convolve(x, kernel, mode="same").astype(np.float32)


def preprocess_amplitude_matrix(csi_amp: np.ndarray, cfg: dict) -> np.ndarray:
    if csi_amp.ndim != 2:
        raise ValueError(f"csi_amp must be 2D (Time, Subcarrier), got {csi_amp.shape}")
    nt, ns = csi_amp.shape
    out = np.zeros((nt, ns), dtype=np.float32)
    for sc in range(ns):
        seq = csi_amp[:, sc].astype(np.float32)
        if cfg["use_hampel"]:
            seq = hampel_filter_1d(seq, window=cfg["hampel_window"], sigma=cfg["hampel_sigma"])
        if cfg["use_moving_average"]:
            seq = moving_average_1d(seq, window=cfg["ma_window"])
        out[:, sc] = seq
    return out


def compute_power_response(csi_amp_processed: np.ndarray) -> np.ndarray:
    return np.square(csi_amp_processed).astype(np.float32)


def compute_acf_for_subcarrier(
    power_series: np.ndarray,
    n_delta: int,
    unbiased: bool = False,
    eps: float = EPS,
) -> np.ndarray:
    x = np.asarray(power_series, dtype=np.float32)
    w = len(x)
    if n_delta >= w:
        raise ValueError(f"n_delta must be < window length, got n_delta={n_delta}, window={w}")
    x = x - x.mean()
    var = np.var(x)
    if var < eps:
        return np.zeros((n_delta,), dtype=np.float32)
    acf = np.zeros((n_delta,), dtype=np.float32)
    for lag in range(1, n_delta + 1):
        denom_n = (w - lag) if unbiased else w
        val = np.sum(x[lag:] * x[:-lag]) / max(float(denom_n), eps)
        acf[lag - 1] = val / max(float(var), eps)
    return acf


def normalize_sdp_columns(
    sdp_aggregated: np.ndarray,
    positive_clip: bool = True,
    zero_column_fill: str = "uniform",
    eps: float = EPS,
) -> np.ndarray:
    x = np.asarray(sdp_aggregated, dtype=np.float32).copy()
    if positive_clip:
        x = np.maximum(x, 0.0)
    col_sum = x.sum(axis=0, keepdims=True)
    zero_mask = col_sum < eps
    if np.any(zero_mask):
        if zero_column_fill == "uniform":
            x[:, zero_mask[0]] = 1.0 / x.shape[0]
            col_sum[:, zero_mask[0]] = 1.0
        elif zero_column_fill == "zeros":
            col_sum[:, zero_mask[0]] = 1.0
        else:
            raise ValueError(f"Unsupported zero_column_fill: {zero_column_fill}")
    return (x / col_sum).astype(np.float32)


def extract_sdp_from_power(
    power_response: np.ndarray,
    window_size: int,
    stride: int,
    n_delta: int,
    positive_clip: bool = True,
    zero_column_fill: str = "uniform",
    acf_unbiased: bool = False,
) -> np.ndarray:
    if power_response.ndim != 2:
        raise ValueError(f"power_response must be 2D (Time, Subcarrier), got {power_response.shape}")
    nt, ns = power_response.shape
    if window_size > nt:
        raise ValueError(f"window_size={window_size} > time length={nt}")
    if n_delta >= window_size:
        raise ValueError(f"n_delta={n_delta} must be < window_size={window_size}")
    if stride <= 0:
        raise ValueError(f"stride must be positive, got {stride}")

    window_starts = list(range(0, nt - window_size + 1, stride))
    n_windows = len(window_starts)
    sdp_tensor = np.zeros((n_delta, n_windows, ns), dtype=np.float32)

    for w_idx, start in enumerate(window_starts):
        end = start + window_size
        window_power = power_response[start:end, :]
        for sc in range(ns):
            acf = compute_acf_for_subcarrier(
                power_series=window_power[:, sc],
                n_delta=n_delta,
                unbiased=acf_unbiased,
            )
            sdp_tensor[:, w_idx, sc] = acf

    sdp_aggregated = np.mean(sdp_tensor, axis=2)
    sdp_final = normalize_sdp_columns(
        sdp_aggregated,
        positive_clip=positive_clip,
        zero_column_fill=zero_column_fill,
    )
    return sdp_final.astype(np.float32)


def extract_sdp_from_csi(csi: np.ndarray, cfg: dict):
    if csi.ndim != 4:
        raise ValueError(f"Expected canonical CSI shape (Rx, Tx, SC, T), got {csi.shape}")
    rx_n, tx_n, sc_n, t_n = csi.shape
    sdp_list, amp_processed_list, power_list = [], [], []

    for rx in range(rx_n):
        row_sdp, row_amp, row_power = [], [], []
        for tx in range(tx_n):
            amp_raw = np.abs(csi[rx, tx]).T.astype(np.float32)
            amp_processed = preprocess_amplitude_matrix(amp_raw, cfg)
            power = compute_power_response(amp_processed)
            sdp = extract_sdp_from_power(
                power_response=power,
                window_size=cfg["window_size"],
                stride=cfg["stride"],
                n_delta=cfg["n_delta"],
                positive_clip=cfg["positive_clip"],
                zero_column_fill=cfg["zero_column_fill"],
                acf_unbiased=cfg["acf_unbiased"],
            )
            row_sdp.append(sdp)
            row_amp.append(amp_processed.T)
            row_power.append(power.T)
        sdp_list.append(np.stack(row_sdp, axis=0))
        amp_processed_list.append(np.stack(row_amp, axis=0))
        power_list.append(np.stack(row_power, axis=0))

    sdp_out = np.stack(sdp_list, axis=0)
    amp_processed_out = np.stack(amp_processed_list, axis=0)
    power_out = np.stack(power_list, axis=0)

    if cfg["layout"] == "wtn":
        sdp_out = np.transpose(sdp_out, (0, 1, 3, 2))
    elif cfg["layout"] != "nwt":
        raise ValueError(f"Unsupported layout: {cfg['layout']}")

    return sdp_out.astype(np.float32), amp_processed_out.astype(np.float32), power_out.astype(np.float32)


def print_sample_stats(name: str, sdp: np.ndarray, cfg: dict) -> None:
    print(f"\n=== Preview: {name} ===")
    print("shape:", sdp.shape)
    print("dtype:", sdp.dtype)
    print("min/max:", float(sdp.min()), float(sdp.max()))
    print("mean/std:", float(sdp.mean()), float(sdp.std()))
    print("nan:", bool(np.isnan(sdp).any()), "inf:", bool(np.isinf(sdp).any()))
    s = sdp[0, 0].T if cfg["layout"] == "wtn" else sdp[0, 0]
    col_sum = s.sum(axis=0)
    col_var = s.var(axis=0)
    print("column sum min/max:", float(col_sum.min()), float(col_sum.max()))
    print("column var min/max/mean:",
          float(col_var.min()), float(col_var.max()), float(col_var.mean()))


def plot_preview_sample(
    sample_name: str,
    raw_csi: np.ndarray,
    amp_processed: np.ndarray,
    power_response: np.ndarray,
    sdp: np.ndarray,
    cfg: dict,
    out_dir: Path,
) -> Path:
    ensure_dir(out_dir)
    raw_amp_mean = np.abs(raw_csi).mean(axis=(0, 1))
    proc_amp_mean = amp_processed.mean(axis=(0, 1))
    power_mean = power_response.mean(axis=(0, 1))
    sdp_mean = sdp.mean(axis=(0, 1)).T if cfg["layout"] == "wtn" else sdp.mean(axis=(0, 1))
    col_sum = sdp_mean.sum(axis=0)

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    im0 = axes[0, 0].imshow(raw_amp_mean, aspect="auto", cmap="viridis")
    axes[0, 0].set_title("Raw amplitude (mean over 3x3 links)")
    axes[0, 0].set_xlabel("Time")
    axes[0, 0].set_ylabel("Subcarrier")
    plt.colorbar(im0, ax=axes[0, 0], fraction=0.046, pad=0.04)

    im1 = axes[0, 1].imshow(power_mean, aspect="auto", cmap="magma")
    axes[0, 1].set_title("Processed power (mean over 3x3 links)")
    axes[0, 1].set_xlabel("Time")
    axes[0, 1].set_ylabel("Subcarrier")
    plt.colorbar(im1, ax=axes[0, 1], fraction=0.046, pad=0.04)

    axes[1, 0].plot(raw_amp_mean[0], label="raw amp, SC0", linewidth=1.4)
    axes[1, 0].plot(proc_amp_mean[0], label="processed amp, SC0", linewidth=1.4)
    axes[1, 0].set_title("SC0 trace (mean over 3x3 links)")
    axes[1, 0].set_xlabel("Time")
    axes[1, 0].set_ylabel("Amplitude")
    axes[1, 0].legend()
    axes[1, 0].grid(alpha=0.25)

    im3 = axes[1, 1].imshow(sdp_mean, aspect="auto", cmap="plasma")
    axes[1, 1].set_title(
        f"SDP (mean over 3x3 links)\ncol-sum min/max={col_sum.min():.3f}/{col_sum.max():.3f}"
    )
    axes[1, 1].set_xlabel("Window index")
    axes[1, 1].set_ylabel("Lag index")
    plt.colorbar(im3, ax=axes[1, 1], fraction=0.046, pad=0.04)

    fig.suptitle(sample_name)
    plt.tight_layout(rect=[0, 0, 1, 0.97])

    save_path = out_dir / f"{Path(sample_name).stem}_preview.png"
    fig.savefig(save_path, dpi=180)
    plt.close(fig)
    return save_path


def save_preview_grid(preview_images: List[Path], out_path: Path) -> None:
    from PIL import Image
    if not preview_images:
        return
    imgs = [Image.open(p).convert("RGB") for p in preview_images]
    thumb_w, thumb_h = 480, 320
    imgs = [img.resize((thumb_w, thumb_h)) for img in imgs]
    cols = 2
    rows = int(np.ceil(len(imgs) / cols))
    canvas = Image.new("RGB", (cols * thumb_w, rows * thumb_h), color=(255, 255, 255))
    for idx, img in enumerate(imgs):
        r = idx // cols
        c = idx % cols
        canvas.paste(img, (c * thumb_w, r * thumb_h))
    canvas.save(out_path)


def save_array(path: Path, arr: np.ndarray, dtype: str) -> None:
    arr = arr.astype(np.float16 if dtype == "float16" else np.float32)
    np.save(path, arr)


def process_one_file_worker(args: Tuple[str, str, dict]) -> str:
    path_str, dst_csi_root_str, cfg = args
    path = Path(path_str)
    dst_csi_root = Path(dst_csi_root_str)
    out_path = dst_csi_root / f"{path.stem}.npy"

    if out_path.exists() and not cfg["overwrite"]:
        return "skipped"

    csi = load_csi_from_mat(str(path), mat_key=cfg["mat_key"])
    validate_canonical_shape(csi, cfg)
    sdp, _, _ = extract_sdp_from_csi(csi, cfg)
    save_array(out_path, sdp, dtype=cfg["dtype"])
    return "processed"


def process_dataset(cfg: dict) -> None:
    src_root = Path(cfg["src_root"])
    dst_root = Path(cfg["dst_root"])
    src_csi_root = src_root / cfg["src_csi_dir"]
    dst_csi_root = dst_root / cfg["dst_csi_dir"]
    preview_root = dst_root / cfg["preview_dir"]

    if not src_csi_root.exists():
        raise FileNotFoundError(f"Source CSI dir not found: {src_csi_root}")

    ensure_dir(dst_csi_root)
    ensure_dir(preview_root)

    if cfg["copy_metadata"]:
        maybe_copy_metadata(src_root, dst_root, cfg["src_csi_dir"], cfg["dst_csi_dir"])

    files = sorted(src_csi_root.glob(f"*{cfg['input_ext']}"))
    if not files:
        raise FileNotFoundError(f"No files matching *{cfg['input_ext']} under {src_csi_root}")

    print("=" * 78)
    print("Formal offline power-ACF SDP preprocessing")
    print("=" * 78)
    print("src_root        :", src_root)
    print("dst_root        :", dst_root)
    print("src_csi         :", src_csi_root)
    print("dst_csi         :", dst_csi_root)
    print("expected shape  :", (
        cfg["expected_rx"], cfg["expected_tx"], cfg["expected_subcarrier"], cfg["expected_time"]
    ))
    print("window_size     :", cfg["window_size"])
    print("stride          :", cfg["stride"])
    print("n_delta         :", cfg["n_delta"])
    print("layout          :", cfg["layout"])
    print("positive_clip   :", cfg["positive_clip"])
    print("zero_column_fill:", cfg["zero_column_fill"])
    print("overwrite       :", cfg["overwrite"])
    print("preview_limit   :", cfg["preview_limit"])
    print("num_workers     :", cfg["num_workers"])
    print("num files       :", len(files))
    print("=" * 78)

    preview_count = 0
    preview_paths: List[Path] = []
    parallel_files: List[Path] = []

    for path in tqdm(files, desc="SDP preprocessing preview/scan"):
        out_path = dst_csi_root / f"{path.stem}.npy"
        if out_path.exists() and not cfg["overwrite"]:
            continue

        if preview_count < cfg["preview_limit"]:
            csi = load_csi_from_mat(str(path), mat_key=cfg["mat_key"])
            validate_canonical_shape(csi, cfg)
            sdp, amp_processed, power_response = extract_sdp_from_csi(csi, cfg)
            save_array(out_path, sdp, dtype=cfg["dtype"])

            if cfg["save_preview_stats"]:
                print_sample_stats(path.name, sdp, cfg)

            preview_path = plot_preview_sample(
                sample_name=path.name,
                raw_csi=csi,
                amp_processed=amp_processed,
                power_response=power_response,
                sdp=sdp,
                cfg=cfg,
                out_dir=preview_root,
            )
            preview_paths.append(preview_path)
            preview_count += 1
        else:
            parallel_files.append(path)

    if parallel_files:
        num_workers = min(max(1, int(cfg["num_workers"])), len(parallel_files))
        tasks = [(str(path), str(dst_csi_root), cfg) for path in parallel_files]

        if num_workers == 1:
            iterator = map(process_one_file_worker, tasks)
        else:
            executor = ProcessPoolExecutor(max_workers=num_workers)
            iterator = executor.map(
                process_one_file_worker,
                tasks,
                chunksize=max(1, int(cfg["parallel_chunksize"])),
            )

        try:
            for _ in tqdm(iterator, total=len(tasks), desc="SDP preprocessing parallel"):
                pass
        finally:
            if num_workers > 1:
                executor.shutdown(wait=True)

    if cfg["save_preview_grid"] and preview_paths:
        try:
            save_preview_grid(preview_paths, preview_root / "preview_first10_grid.png")
        except Exception as exc:
            print(f"[WARN] Failed to save preview grid: {exc}")

    print("\nDone.")
    print(f"Saved offline SDP files to: {dst_csi_root}")
    print(f"Saved preview figures to   : {preview_root}")


if __name__ == "__main__":
    process_dataset(CONFIG)
