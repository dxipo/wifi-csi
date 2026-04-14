#!/usr/bin/env python3
"""
Offline SDP preprocessing for CSI data.

This script implements the Speed Distribution Profile (SDP) described in
XFall: Domain Adaptive Wi-Fi-Based Fall Detection With Cross-Modal Supervision.

Core implementation follows Eq. (9) and Eq. (10):
1) Build a lagged auto-correlation tensor rho over [lag, time, subcarrier].
2) Merge subcarriers by averaging.
3) Normalize each time column across lag bins to obtain SDP.

Designed for CSI tensors that can be canonicalized to shape (Rx, Tx, Subcarrier, Time),
e.g. (3, 3, 30, 20).
"""

from __future__ import annotations

import argparse
import os
import shutil
from pathlib import Path
from typing import Tuple

import h5py
import numpy as np
from tqdm import tqdm


EPS = 1e-8


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Offline SDP preprocessing for CSI")
    parser.add_argument("--src-root", type=str, required=True,
                        help="Source dataset root, e.g. data/wifipose/all_single_train_data_hold_out")
    parser.add_argument("--dst-root", type=str, required=True,
                        help="Destination dataset root")
    parser.add_argument("--src-csi-dir", type=str, default="csi",
                        help="Directory containing raw CSI files under src-root")
    parser.add_argument("--dst-csi-dir", type=str, default="csi_sdp_offline",
                        help="Directory containing offline SDP files under dst-root")
    parser.add_argument("--input-ext", type=str, default=".mat",
                        help="Raw CSI file extension")
    parser.add_argument("--mat-key", type=str, default="csi_out",
                        help="Dataset key inside .mat/.h5 file")
    parser.add_argument("--lag-step", type=int, default=1,
                        help="Discrete lag resolution ΔT in packet steps")
    parser.add_argument("--n-delta", type=int, default=6,
                        help="Number of lag samples NΔ")
    parser.add_argument("--layout", type=str, default="nwt", choices=["nwt", "wtn"],
                        help="Output layout: nwt -> (Rx,Tx,NΔ,WT), wtn -> (Rx,Tx,WT,NΔ)")
    parser.add_argument("--dtype", type=str, default="float32", choices=["float32", "float16"],
                        help="Output dtype")
    parser.add_argument("--copy-metadata", action="store_true",
                        help="Copy txt files and symlink/copy non-CSI folders to dst-root")
    parser.add_argument("--overwrite", action="store_true",
                        help="Overwrite existing destination files")
    return parser.parse_args()


# -----------------------------
# Loading and canonicalization
# -----------------------------
def _structured_to_complex(arr: np.ndarray) -> np.ndarray:
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

    raw = _structured_to_complex(np.array(raw))
    return canonicalize_csi(raw, path)



def canonicalize_csi(csi: np.ndarray, path: str = "") -> np.ndarray:
    """
    Canonical output: (Rx, Tx, Subcarrier, Time)
    Common inputs seen in your project:
      - (20, 30, 3, 3)  -> transpose to (3, 3, 30, 20)
      - (3, 3, 30, 20)  -> keep
      - (30, 20)        -> treat as single-link (1, 1, 30, 20)
      - (20, 30)        -> treat as single-link (1, 1, 30, 20)
    """
    csi = np.asarray(csi)

    if csi.ndim == 4:
        if csi.shape[0] == 3 and csi.shape[1] == 3:
            # already (3,3,30,20) or similar
            if csi.shape[2] == 30:
                return csi
            # fallback: try infer if last two axes are time/subcarrier
            if csi.shape[3] == 30:
                return np.transpose(csi, (0, 1, 3, 2))
        if csi.shape[2] == 3 and csi.shape[3] == 3:
            # (Time, Subcarrier, Rx, Tx) -> (Rx, Tx, Subcarrier, Time)
            return np.transpose(csi, (2, 3, 1, 0))

    if csi.ndim == 2:
        if csi.shape == (30, 20):
            return csi[None, None, :, :]
        if csi.shape == (20, 30):
            return csi.T[None, None, :, :]

    raise ValueError(f"Cannot canonicalize CSI shape {csi.shape} from {path}")


# -----------------------------
# SDP extraction
# -----------------------------
def build_acf_tensor(H: np.ndarray, lag_step: int, n_delta: int, eps: float = EPS) -> np.ndarray:
    """
    Build rho in Eq. (9).

    Parameters
    ----------
    H : complex ndarray, shape (NT, NS)
        One CSI link with time on axis 0 and subcarrier on axis 1.
    lag_step : int
        ΔT in discrete packet steps.
    n_delta : int
        Number of lag samples NΔ.

    Returns
    -------
    rho : float ndarray, shape (NΔ, WT, NS)
    """
    if H.ndim != 2:
        raise ValueError(f"H must be 2D (NT, NS), got {H.shape}")

    NT, NS = H.shape
    max_lag = lag_step * n_delta
    WT = NT - max_lag
    if WT <= 0:
        raise ValueError(
            f"Invalid parameters: NT={NT}, lag_step={lag_step}, n_delta={n_delta} -> WT={WT}. "
            "Need NT > lag_step * n_delta."
        )

    # Align all lag channels to the same current time span [max_lag, NT).
    curr = H[max_lag: NT, :]  # (WT, NS)

    rho = np.empty((n_delta, WT, NS), dtype=np.float32)
    for n in range(1, n_delta + 1):
        lag = n * lag_step
        prev = H[max_lag - lag: NT - lag, :]  # (WT, NS)

        # Eq. (9): |H(t_i,f_j) H*(t_i-nΔT,f_j)| / (|H(t_i,f_j)| |H(t_i-nΔT,f_j)|)
        numerator = np.abs(curr * np.conj(prev))
        denominator = np.abs(curr) * np.abs(prev)
        rho[n - 1] = numerator / np.maximum(denominator, eps)

    return rho



def acf_to_sdp(rho: np.ndarray, eps: float = EPS) -> np.ndarray:
    """
    Convert rho to SDP following Eq. (10) description.

    The paper text says:
      1) merge all subcarriers;
      2) perform probabilistic normalization to each column.

    Thus we implement:
      merged[n, i] = mean_j rho[n, i, j]
      S[:, i] = merged[:, i] / sum_n merged[n, i]

    Returns
    -------
    S : float ndarray, shape (NΔ, WT)
        Each time column sums to 1.
    """
    if rho.ndim != 3:
        raise ValueError(f"rho must be 3D (NΔ, WT, NS), got {rho.shape}")

    merged = rho.mean(axis=2)  # (NΔ, WT)
    col_sum = merged.sum(axis=0, keepdims=True)
    S = merged / np.maximum(col_sum, eps)
    return S.astype(np.float32)



def extract_sdp_from_csi(csi: np.ndarray, lag_step: int, n_delta: int,
                         layout: str = "nwt") -> np.ndarray:
    """
    Parameters
    ----------
    csi : complex ndarray, shape (Rx, Tx, Subcarrier, Time)

    Returns
    -------
    out : float ndarray
        - layout='nwt' -> (Rx, Tx, NΔ, WT)
        - layout='wtn' -> (Rx, Tx, WT, NΔ)
    """
    if csi.ndim != 4:
        raise ValueError(f"Expected canonical CSI shape (Rx,Tx,Subcarrier,Time), got {csi.shape}")

    RX, TX, NS, NT = csi.shape
    result = np.empty((RX, TX), dtype=object)
    for rx in range(RX):
        for tx in range(TX):
            # paper assumes H ∈ C^{NT x NS}
            H = csi[rx, tx].transpose(1, 0)  # (Time, Subcarrier)
            rho = build_acf_tensor(H, lag_step=lag_step, n_delta=n_delta)
            S = acf_to_sdp(rho)
            result[rx, tx] = S

    out = np.stack([np.stack([result[rx, tx] for tx in range(TX)], axis=0)
                    for rx in range(RX)], axis=0)  # (Rx,Tx,NΔ,WT)

    if layout == "wtn":
        out = np.transpose(out, (0, 1, 3, 2))
    return out.astype(np.float32)


# -----------------------------
# Dataset processing
# -----------------------------
def maybe_copy_metadata(src_root: Path, dst_root: Path, src_csi_dir: str, dst_csi_dir: str) -> None:
    dst_root.mkdir(parents=True, exist_ok=True)

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

    (dst_root / dst_csi_dir).mkdir(parents=True, exist_ok=True)



def save_array(path: Path, arr: np.ndarray, dtype: str) -> None:
    if dtype == "float16":
        arr = arr.astype(np.float16)
    else:
        arr = arr.astype(np.float32)
    np.save(path, arr)



def main() -> None:
    args = parse_args()

    src_root = Path(args.src_root)
    dst_root = Path(args.dst_root)
    src_csi_root = src_root / args.src_csi_dir
    dst_csi_root = dst_root / args.dst_csi_dir

    if not src_csi_root.exists():
        raise FileNotFoundError(f"Source CSI dir not found: {src_csi_root}")

    dst_csi_root.mkdir(parents=True, exist_ok=True)
    if args.copy_metadata:
        maybe_copy_metadata(src_root, dst_root, args.src_csi_dir, args.dst_csi_dir)

    files = sorted(src_csi_root.glob(f"*{args.input_ext}"))
    if not files:
        raise FileNotFoundError(f"No files matching *{args.input_ext} under {src_csi_root}")

    print(f"Found {len(files)} raw CSI files")
    print(f"lag_step={args.lag_step}, n_delta={args.n_delta}, layout={args.layout}")

    for path in tqdm(files, desc="SDP preprocessing"):
        out_path = dst_csi_root / f"{path.stem}.npy"
        if out_path.exists() and not args.overwrite:
            continue

        csi = load_csi_from_mat(str(path), mat_key=args.mat_key)
        sdp = extract_sdp_from_csi(csi, lag_step=args.lag_step, n_delta=args.n_delta, layout=args.layout)
        save_array(out_path, sdp, dtype=args.dtype)

    print("Done.")
    print(f"Saved offline SDP files to: {dst_csi_root}")


if __name__ == "__main__":
    main()
