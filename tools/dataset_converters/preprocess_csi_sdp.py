#!/usr/bin/env python3
"""
Standalone CSI -> SDP preprocessor for WiFi sensing datasets.

This script is designed for CSI samples shaped like:
    (T, S, Rx, Tx) structured real/imag dataset
or transposed as:
    (Tx, Rx, S, T)
where in the user's case:
    T=20, S=30, Tx=3, Rx=3

It implements a practical SDP extraction that is faithful to the XFall paper's
intent (complex CSI auto-correlation + column normalization), while avoiding a
literal implementation pitfall in Eq. (9): as printed in the paper, the term

    |H(t_i,f_j) H*(t_i-nΔT,f_j)| / (|H(t_i,f_j)| |H(t_i-nΔT,f_j)|)

collapses to 1 whenever both magnitudes are non-zero. Therefore this script uses
windowed normalized complex auto-correlation, which is consistent with the paper's
Eq. (8) ACF interpretation and yields a non-degenerate SDP.

Outputs can be saved as:
    - 3x3 stacked link SDP maps:  (3, 3, H, W)
    - 9 stacked link SDP maps:    (9, H, W)
    - mean SDP over all links:    (H, W)

Example:
python preprocess_csi_sdp.py \
    --input-dir data/wifipose/train/csi \
    --output-dir data/wifipose/train/csi_sdp \
    --lag-bins 8 \
    --acf-window 4 \
    --target-height 14 \
    --target-width 20 \
    --save-mode txrx
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Tuple, Optional

import h5py
import numpy as np

EPS = 1e-8


def _structured_real_imag_to_complex(arr: np.ndarray) -> np.ndarray:
    """Convert structured dtype with fields ('real', 'imag') to complex array."""
    if arr.dtype.fields is None:
        raise ValueError("Input is not a structured real/imag array.")
    field_names = set(arr.dtype.fields.keys())
    if not {"real", "imag"}.issubset(field_names):
        raise ValueError(f"Structured dtype fields {field_names} do not contain real/imag.")
    return arr["real"].astype(np.float64) + 1j * arr["imag"].astype(np.float64)


def _load_h5_dataset(path: Path) -> np.ndarray:
    with h5py.File(path, "r") as f:
        # Prefer the user's reported key.
        if "csi_out" in f:
            data = f["csi_out"][()]
            return data
        # Otherwise, pick the first dataset recursively.
        found = None
        def visitor(_name, obj):
            nonlocal found
            if found is None and isinstance(obj, h5py.Dataset):
                found = obj[()]
        f.visititems(visitor)
        if found is None:
            raise ValueError(f"No dataset found in file: {path}")
        return found


def load_csi_any(path: str | Path) -> np.ndarray:
    """
    Load CSI from .npy / .npz / HDF5-like files and return complex ndarray.

    Supported cases:
    - .npy containing complex ndarray
    - .npy containing structured real/imag ndarray
    - .npz containing key 'csi_out' or first array
    - HDF5/.mat-like file containing dataset 'csi_out'
    """
    path = Path(path)
    suffix = path.suffix.lower()

    if suffix == ".npz":
        obj = np.load(path, allow_pickle=True)
        if "csi_out" in obj:
            arr = obj["csi_out"]
        else:
            first_key = list(obj.keys())[0]
            arr = obj[first_key]
    elif suffix == ".npy":
        arr = np.load(path, allow_pickle=True)
        # Sometimes object arrays are used to wrap dict-like content.
        if isinstance(arr, np.ndarray) and arr.dtype == object and arr.shape == ():
            item = arr.item()
            if isinstance(item, dict):
                if "csi_out" in item:
                    arr = item["csi_out"]
                else:
                    # choose first value
                    arr = next(iter(item.values()))
    else:
        arr = _load_h5_dataset(path)

    # If arr is still a pathologically wrapped object, unwrap once more.
    if isinstance(arr, np.ndarray) and arr.dtype == object and arr.shape == ():
        arr = arr.item()

    if isinstance(arr, np.ndarray) and np.iscomplexobj(arr):
        return arr.astype(np.complex128)

    if isinstance(arr, np.ndarray) and arr.dtype.fields is not None:
        return _structured_real_imag_to_complex(arr)

    if isinstance(arr, np.ndarray):
        # already numeric but not complex; treat as real-only CSI
        return arr.astype(np.float64).astype(np.complex128)

    raise ValueError(f"Unsupported CSI content type in file: {path}")


def canonicalize_csi_shape(csi: np.ndarray) -> np.ndarray:
    """
    Convert CSI to canonical shape: (Tx, Rx, Subcarrier, Time).

    Expected common shapes:
    - (T, S, Rx, Tx)  -> transpose to (Tx, Rx, S, T)
    - (Tx, Rx, S, T)  -> unchanged
    - (Rx, Tx, S, T)  -> transpose to (Tx, Rx, S, T) if dimensions suggest so
    """
    if csi.ndim != 4:
        raise ValueError(f"Expected 4D CSI, got shape {csi.shape}")

    shape = csi.shape

    # User-reported raw layout: (20, 30, 3, 3)
    if shape[0] <= 128 and shape[1] <= 256 and shape[2] <= 8 and shape[3] <= 8:
        # Heuristic: if the last two dims are small MIMO dims and the first dim is time-like.
        if shape[2] in (2, 3, 4) and shape[3] in (2, 3, 4):
            # Assume (T, S, Rx, Tx)
            return np.transpose(csi, (3, 2, 1, 0))

    # Already canonical (Tx, Rx, S, T)
    if shape[0] in (2, 3, 4) and shape[1] in (2, 3, 4):
        return csi

    raise ValueError(
        "Could not infer CSI axis order automatically. "
        f"Please inspect raw shape {shape} and adapt canonicalize_csi_shape()."
    )


def phase_sanitize(csi_txrxst: np.ndarray) -> np.ndarray:
    """
    Simple phase sanitization along the subcarrier axis.

    For each Tx-Rx link and time index:
    1) unwrap phase across subcarriers
    2) remove linear trend across subcarriers
    3) reconstruct complex CSI with original amplitude and sanitized phase

    This is a practical calibration step, not part of XFall's SDP equations.
    """
    tx, rx, s, t = csi_txrxst.shape
    out = np.empty_like(csi_txrxst, dtype=np.complex128)
    k = np.arange(s, dtype=np.float64)
    A = np.vstack([k, np.ones_like(k)]).T

    for ti in range(tx):
        for ri in range(rx):
            link = csi_txrxst[ti, ri]  # (S, T)
            amp = np.abs(link)
            ph = np.unwrap(np.angle(link), axis=0)
            ph_corr = np.empty_like(ph)
            for tt in range(t):
                coef, _, _, _ = np.linalg.lstsq(A, ph[:, tt], rcond=None)
                trend = A @ coef
                ph_corr[:, tt] = ph[:, tt] - trend
            out[ti, ri] = amp * np.exp(1j * ph_corr)
    return out


def temporal_standardize(link_st: np.ndarray) -> np.ndarray:
    """Standardize each subcarrier along time after removing mean."""
    x = link_st.copy().astype(np.complex128)
    mean = np.mean(x, axis=1, keepdims=True)
    x = x - mean
    power = np.sqrt(np.mean(np.abs(x) ** 2, axis=1, keepdims=True) + EPS)
    return x / power


def resize_2d_bilinear(x: np.ndarray, out_h: int, out_w: int) -> np.ndarray:
    """Pure NumPy bilinear resize for a single 2D array."""
    in_h, in_w = x.shape
    if in_h == out_h and in_w == out_w:
        return x.astype(np.float32)

    y_coords = np.linspace(0, in_h - 1, out_h)
    x_coords = np.linspace(0, in_w - 1, out_w)

    y0 = np.floor(y_coords).astype(int)
    x0 = np.floor(x_coords).astype(int)
    y1 = np.clip(y0 + 1, 0, in_h - 1)
    x1 = np.clip(x0 + 1, 0, in_w - 1)

    wy = y_coords - y0
    wx = x_coords - x0

    out = np.empty((out_h, out_w), dtype=np.float64)
    for i in range(out_h):
        top = (1 - wx) * x[y0[i], x0] + wx * x[y0[i], x1]
        bot = (1 - wx) * x[y1[i], x0] + wx * x[y1[i], x1]
        out[i] = (1 - wy[i]) * top + wy[i] * bot
    return out.astype(np.float32)


def compute_sdp_single_link(
    link_st: np.ndarray,
    lag_bins: int = 8,
    lag_step: int = 1,
    acf_window: int = 4,
    normalize_subcarrier_first: bool = True,
    column_normalize: bool = True,
    nonnegative: str = "abs",
) -> np.ndarray:
    """
    Compute SDP for one Tx-Rx link.

    Parameters
    ----------
    link_st : complex ndarray, shape (Subcarrier, Time)
    lag_bins : number of lag samples NΔ
    lag_step : ΔT in integer packet units
    acf_window : local window length used to estimate normalized ACF
    normalize_subcarrier_first : standardize each subcarrier along time
    column_normalize : normalize each time column over lag dimension
    nonnegative : {'abs', 'clip', 'none'}
        - abs : use absolute value of normalized complex ACF, range [0,1]
        - clip: use real part and clip negatives to 0
        - none: keep real part, may include negatives

    Returns
    -------
    S : ndarray, shape (lag_bins, WT)
    """
    if link_st.ndim != 2:
        raise ValueError(f"Expected (Subcarrier, Time), got {link_st.shape}")

    x = link_st.astype(np.complex128)
    if normalize_subcarrier_first:
        x = temporal_standardize(x)

    ns, nt = x.shape
    max_lag = lag_bins * lag_step
    if nt <= max_lag + acf_window - 1:
        raise ValueError(
            f"Time length nt={nt} is too small for lag_bins={lag_bins}, "
            f"lag_step={lag_step}, acf_window={acf_window}."
        )

    # Valid anchor positions for local windowed ACF.
    start_i = max_lag + acf_window - 1
    anchors = np.arange(start_i, nt)
    wt = len(anchors)
    rho = np.zeros((lag_bins, wt, ns), dtype=np.float64)

    for n_idx in range(lag_bins):
        lag = (n_idx + 1) * lag_step
        for a_idx, i in enumerate(anchors):
            cur = x[:, i - acf_window + 1 : i + 1]              # (ns, w)
            pre = x[:, i - lag - acf_window + 1 : i - lag + 1]  # (ns, w)

            num = np.mean(cur * np.conj(pre), axis=1)
            den = np.sqrt(
                np.mean(np.abs(cur) ** 2, axis=1) * np.mean(np.abs(pre) ** 2, axis=1)
            ) + EPS
            coeff = num / den

            if nonnegative == "abs":
                rho[n_idx, a_idx] = np.abs(coeff)
            elif nonnegative == "clip":
                rho[n_idx, a_idx] = np.clip(np.real(coeff), 0.0, None)
            elif nonnegative == "none":
                rho[n_idx, a_idx] = np.real(coeff)
            else:
                raise ValueError(f"Unsupported nonnegative mode: {nonnegative}")

    # Merge subcarriers first: (lag_bins, wt)
    sdp = np.mean(rho, axis=2)

    # Probabilistic normalization over lag dimension for each temporal column.
    if column_normalize:
        denom = np.sum(sdp, axis=0, keepdims=True) + EPS
        sdp = sdp / denom

    return sdp.astype(np.float32)


def compute_sdp_all_links(
    csi_txrxst: np.ndarray,
    lag_bins: int = 8,
    lag_step: int = 1,
    acf_window: int = 4,
    do_phase_sanitize: bool = True,
    target_height: Optional[int] = None,
    target_width: Optional[int] = None,
    save_mode: str = "txrx",
    nonnegative: str = "abs",
) -> np.ndarray:
    """
    Compute SDP for all Tx-Rx links.

    save_mode:
        - 'txrx'      -> (Tx, Rx, H, W)
        - 'flat9'     -> (Tx*Rx, H, W)
        - 'mean'      -> (H, W)
        - 'mean_keep' -> (1, H, W)
    """
    x = csi_txrxst
    if do_phase_sanitize:
        x = phase_sanitize(x)

    tx, rx, _, _ = x.shape
    maps = []
    for ti in range(tx):
        row = []
        for ri in range(rx):
            sdp = compute_sdp_single_link(
                x[ti, ri],
                lag_bins=lag_bins,
                lag_step=lag_step,
                acf_window=acf_window,
                nonnegative=nonnegative,
            )
            if target_height is not None and target_width is not None:
                sdp = resize_2d_bilinear(sdp, target_height, target_width)
            row.append(sdp)
        maps.append(row)

    maps = np.array(maps, dtype=np.float32)  # (Tx, Rx, H, W)

    if save_mode == "txrx":
        return maps
    if save_mode == "flat9":
        return maps.reshape(tx * rx, maps.shape[2], maps.shape[3])
    if save_mode == "mean":
        return maps.mean(axis=(0, 1))
    if save_mode == "mean_keep":
        return maps.mean(axis=(0, 1), keepdims=True)
    raise ValueError(f"Unsupported save_mode: {save_mode}")


def process_one_file(
    in_path: Path,
    out_path: Path,
    args: argparse.Namespace,
) -> Tuple[Tuple[int, ...], Tuple[int, ...]]:
    raw = load_csi_any(in_path)
    canonical = canonicalize_csi_shape(raw)
    sdp = compute_sdp_all_links(
        canonical,
        lag_bins=args.lag_bins,
        lag_step=args.lag_step,
        acf_window=args.acf_window,
        do_phase_sanitize=not args.no_phase_sanitize,
        target_height=args.target_height,
        target_width=args.target_width,
        save_mode=args.save_mode,
        nonnegative=args.nonnegative,
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(out_path, sdp.astype(np.float32))
    return canonical.shape, sdp.shape


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=str, help="Directory of raw CSI files.")
    parser.add_argument("--output-dir", type=str, required=True, help="Directory to save SDP .npy files.")
    parser.add_argument("--suffix", type=str, default=".npy", help="File suffix to scan, e.g. .npy .mat .h5")
    parser.add_argument("--input-file", type=str, default=None, help="Process a single file instead of a directory.")
    parser.add_argument("--lag-bins", type=int, default=8, help="Number of lag bins NΔ.")
    parser.add_argument("--lag-step", type=int, default=1, help="Lag step ΔT in packet units.")
    parser.add_argument("--acf-window", type=int, default=4, help="Local temporal window used for ACF estimation.")
    parser.add_argument("--target-height", type=int, default=14, help="Output SDP height after resize.")
    parser.add_argument("--target-width", type=int, default=20, help="Output SDP width after resize.")
    parser.add_argument("--save-mode", type=str, default="txrx", choices=["txrx", "flat9", "mean", "mean_keep"])
    parser.add_argument("--no-phase-sanitize", action="store_true")
    parser.add_argument("--nonnegative", type=str, default="abs", choices=["abs", "clip", "none"])
    args = parser.parse_args()

    out_root = Path(args.output_dir)
    out_root.mkdir(parents=True, exist_ok=True)

    if args.input_file is not None:
        in_path = Path(args.input_file)
        out_path = out_root / (in_path.stem + ".npy")
        in_shape, out_shape = process_one_file(in_path, out_path, args)
        print(f"Processed: {in_path}")
        print(f"Canonical CSI shape: {in_shape}")
        print(f"Saved SDP shape: {out_shape}")
        print(f"Output: {out_path}")
        return

    if args.input_dir is None:
        raise ValueError("Either --input-file or --input-dir must be provided.")

    in_root = Path(args.input_dir)
    files = sorted([p for p in in_root.rglob(f"*{args.suffix}") if p.is_file()])
    if not files:
        raise FileNotFoundError(f"No files ending with {args.suffix} found under {in_root}")

    count = 0
    for in_path in files:
        rel = in_path.relative_to(in_root)
        out_path = out_root / rel.with_suffix(".npy")
        try:
            in_shape, out_shape = process_one_file(in_path, out_path, args)
            count += 1
            print(f"[{count}/{len(files)}] OK {rel} | CSI {in_shape} -> SDP {out_shape}")
        except Exception as e:
            print(f"[ERROR] {rel}: {e}")

    print(f"Done. Successfully processed {count}/{len(files)} files.")


if __name__ == "__main__":
    main()
