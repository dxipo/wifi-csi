#!/usr/bin/env python3
import argparse
import os
from pathlib import Path
import numpy as np
import h5py
from scipy.signal import stft
from tqdm import tqdm


def load_csi_mat(path: str, key: str = 'csi_out') -> np.ndarray:
    """
    Load CSI from .mat/.h5 and convert to complex ndarray with shape (3, 3, 30, 20).

    Expected source storage examples:
      - h5 dataset shape: (20, 30, 3, 3), dtype=[('real','<f8'),('imag','<f8')]
      - after transpose -> (3, 3, 30, 20)
    """
    with h5py.File(path, 'r') as f:
        if key not in f:
            raise KeyError(f"Key '{key}' not found in {path}. Available keys: {list(f.keys())}")
        x = f[key][()]

    # Compound dtype with real/imag fields
    if getattr(x.dtype, 'fields', None) is not None and 'real' in x.dtype.fields and 'imag' in x.dtype.fields:
        x = x['real'] + 1j * x['imag']

    x = np.asarray(x)
    if x.shape == (20, 30, 3, 3):
        x = np.transpose(x, (3, 2, 1, 0))
    elif x.shape == (3, 3, 30, 20):
        pass
    else:
        raise ValueError(f"Unexpected CSI shape {x.shape} from {path}; expected (20,30,3,3) or (3,3,30,20)")

    return x.astype(np.complex128)


def stft_amp_online_equivalent(csi: np.ndarray, nperseg: int = 8, noverlap: int = 4, nfft: int = 16) -> np.ndarray:
    """
    Reproduce the online baseline_with_stft logic.

    Input:
        csi: complex ndarray, shape (3, 3, 30, 20)
    Output:
        float32 ndarray, shape (3, 3, 30, Tbin, F)
    """
    amp = np.abs(csi).astype(np.float32)
    out = []
    for rx in range(3):
        rx_list = []
        for tx in range(3):
            sc_list = []
            for sc in range(30):
                seq = amp[rx, tx, sc]  # (20,)
                f, t, Zxx = stft(
                    seq,
                    nperseg=nperseg,
                    noverlap=noverlap,
                    nfft=nfft,
                    boundary=None,
                    padded=False,
                )
                spec = np.abs(Zxx).astype(np.float32)
                spec = np.log1p(spec)
                spec = spec.transpose(1, 0)  # (Tbin, F)
                sc_list.append(spec)
            rx_list.append(np.stack(sc_list, axis=0))   # (30, Tbin, F)
        out.append(np.stack(rx_list, axis=0))           # (3, 30, Tbin, F)
    out = np.stack(out, axis=0)                         # (3, 3, 30, Tbin, F)
    return out.astype(np.float32)


def save_feature(path: str, arr: np.ndarray, fmt: str = 'npy'):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if fmt == 'npy':
        np.save(path, arr)
    elif fmt == 'npz':
        np.savez_compressed(path, csi_feature=arr)
    elif fmt == 'mat':
        # Save as HDF5-based .mat-like file for consistency with current h5py loader style.
        # But for training speed, .npy is generally preferable.
        with h5py.File(path, 'w') as f:
            f.create_dataset('csi_feature', data=arr, compression='gzip')
    else:
        raise ValueError(f'Unsupported fmt: {fmt}')


def process_split(
    input_csi_dir: str,
    output_csi_dir: str,
    key: str,
    nperseg: int,
    noverlap: int,
    nfft: int,
    input_ext: str,
    output_fmt: str,
):
    input_csi_dir = Path(input_csi_dir)
    output_csi_dir = Path(output_csi_dir)
    output_csi_dir.mkdir(parents=True, exist_ok=True)

    files = sorted(input_csi_dir.glob(f'*{input_ext}'))
    if not files:
        raise FileNotFoundError(f'No files matching *{input_ext} under {input_csi_dir}')

    first_shape = None
    for p in tqdm(files, desc=f'Processing {input_csi_dir.name}'):
        csi = load_csi_mat(str(p), key=key)
        feat = stft_amp_online_equivalent(
            csi, nperseg=nperseg, noverlap=noverlap, nfft=nfft
        )
        if first_shape is None:
            first_shape = feat.shape
            print(f'[info] first output shape: {first_shape}, dtype={feat.dtype}')

        out_name = p.stem + ('.npy' if output_fmt == 'npy' else '.npz' if output_fmt == 'npz' else '.mat')
        save_feature(str(output_csi_dir / out_name), feat, fmt=output_fmt)


def maybe_copy_metadata(src_root: str, dst_root: str):
    """Copy list/txt and optionally symlink keypoint/token/all_single_frame folders."""
    src_root = Path(src_root)
    dst_root = Path(dst_root)
    dst_root.mkdir(parents=True, exist_ok=True)

    # Copy txt files
    for txt in src_root.glob('*.txt'):
        (dst_root / txt.name).write_bytes(txt.read_bytes())

    # Symlink other folders if they exist and aren't CSI
    for name in ['keypoint', 'token', 'all_single_frame']:
        s = src_root / name
        d = dst_root / name
        if s.exists() and not d.exists():
            try:
                os.symlink(s.resolve(), d)
                print(f'[info] symlinked {d} -> {s.resolve()}')
            except OSError:
                # fallback if symlink unavailable
                pass


def parse_args():
    ap = argparse.ArgumentParser(description='Offline STFT preprocessing for WiFi CSI')
    ap.add_argument('--src-root', required=True, help='dataset split root, e.g. .../all_single_train_data_hold_out')
    ap.add_argument('--dst-root', required=True, help='new dataset split root, e.g. .../all_single_train_data_hold_out_stft_offline')
    ap.add_argument('--src-csi-dir', default='csi')
    ap.add_argument('--dst-csi-dir', default='csi_stft_offline')
    ap.add_argument('--input-ext', default='.mat', choices=['.mat', '.h5'])
    ap.add_argument('--mat-key', default='csi_out')
    ap.add_argument('--nperseg', type=int, default=8)
    ap.add_argument('--noverlap', type=int, default=4)
    ap.add_argument('--nfft', type=int, default=16)
    ap.add_argument('--output-fmt', default='npy', choices=['npy', 'npz', 'mat'])
    ap.add_argument('--copy-metadata', action='store_true', help='copy txt files and symlink keypoint/token/frame dirs')
    return ap.parse_args()


if __name__ == '__main__':
    args = parse_args()
    src_root = Path(args.src_root)
    dst_root = Path(args.dst_root)
    if args.copy_metadata:
        maybe_copy_metadata(str(src_root), str(dst_root))
    process_split(
        input_csi_dir=str(src_root / args.src_csi_dir),
        output_csi_dir=str(dst_root / args.dst_csi_dir),
        key=args.mat_key,
        nperseg=args.nperseg,
        noverlap=args.noverlap,
        nfft=args.nfft,
        input_ext=args.input_ext,
        output_fmt=args.output_fmt,
    )
