#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Formal batch SDP preprocessing script (real_shift version).

用途：
1. 批量读取原始 CSI .mat 文件
2. 统一成 (Rx, Tx, Subcarrier, Time)
3. 基于 XFall 的 SDP 框架生成 rho -> SDP
4. 使用 real_shift 工程修正版避免 paper_abs 退化成常数特征
5. 保存为离线 .npy，供后续 dataloader 直接读取

说明：
- paper_abs（论文字面式(9)）在你的数据上会退化为 rho=1，从而 S 每列都等于 1/NΔ
- 这里默认使用 real_shift：
    rho_raw = Re( H(t) * conj(H(t-lag)) / (|H(t)||H(t-lag)| + eps) )
    rho = 0.5 * (rho_raw + 1.0)   # 映射到 [0,1]
- 之后仍按 XFall 的 SDP 框架做：
    1) 对子载波维平均
    2) 对每个时间列按 lag 维归一化
"""

from __future__ import annotations

import os
import shutil
from pathlib import Path

import h5py
import numpy as np
from tqdm import tqdm


EPS = 1e-8

# ============================================================
# 默认配置：你只改这里，直接运行即可
# ============================================================

CONFIG = {
    # ---------- 数据路径 ----------
    "src_root": "/home/xl/CSI/Person-in-WiFi-3D-repo/data/wifipose/all_single_test_data_hold_out",
    "dst_root": "/home/xl/CSI/Person-in-WiFi-3D-repo/data/wifipose/all_single_test_data_hold_out_sdp_offline_realshift",
    "src_csi_dir": "csi",
    "dst_csi_dir": "csi_sdp_offline",

    # ---------- 文件格式 ----------
    "input_ext": ".mat",
    "mat_key": "csi_out",

    # ---------- SDP 参数 ----------
    "lag_step": 1,
    "n_delta": 6,

    # 输出 layout:
    #   "nwt" -> (Rx, Tx, NΔ, WT)
    #   "wtn" -> (Rx, Tx, WT, NΔ)
    # 对你当前模型更推荐 wtn，因为最后一维就是 feature dim = NΔ
    "layout": "wtn",

    # ACF 构造方式
    #   "paper_abs"  : 论文字面实现（你的数据上会退化）
    #   "real_shift" : 推荐，工程修正版
    #   "real_raw"   : 只取实部，不映射到[0,1]，后续归一化前可能有负数，不推荐首发
    #   "amp_corr"   : 只用幅度相关，信息更弱，可做对照实验
    "acf_mode": "real_shift",

    # 输出 dtype
    "dtype": "float32",   # "float32" or "float16"

    # ---------- 运行控制 ----------
    "copy_metadata": True,    # 拷贝 txt，链接/复制 keypoint token 等目录
    "overwrite": False,       # 目标文件存在时是否覆盖
    "save_preview_stats": True,
    "preview_limit": 5,       # 前几个样本打印统计信息
}


# ============================================================
# 基础工具
# ============================================================

def ensure_dir(path: Path):
    path.mkdir(parents=True, exist_ok=True)


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
    return canonicalize_csi(raw, path)


def canonicalize_csi(csi: np.ndarray, path: str = "") -> np.ndarray:
    """
    统一输出为: (Rx, Tx, Subcarrier, Time)

    常见输入：
      - (20, 30, 3, 3)  -> transpose to (3, 3, 30, 20)
      - (3, 3, 30, 20)  -> keep
      - (30, 20)        -> single-link -> (1, 1, 30, 20)
      - (20, 30)        -> single-link -> (1, 1, 30, 20)
    """
    csi = np.asarray(csi)

    if csi.ndim == 4:
        if csi.shape == (3, 3, 30, 20):
            return csi

        if csi.shape == (20, 30, 3, 3):
            return np.transpose(csi, (2, 3, 1, 0))

        # 泛化处理
        if csi.shape[0] == 3 and csi.shape[1] == 3:
            # (3,3,T,SC) -> (3,3,SC,T)
            if csi.shape[3] == 30:
                return np.transpose(csi, (0, 1, 3, 2))
            if csi.shape[2] == 30:
                return csi

        if csi.shape[2] == 3 and csi.shape[3] == 3:
            return np.transpose(csi, (2, 3, 1, 0))

    if csi.ndim == 2:
        if csi.shape == (30, 20):
            return csi[None, None, :, :]
        if csi.shape == (20, 30):
            return csi.T[None, None, :, :]

    raise ValueError(f"Cannot canonicalize CSI shape {csi.shape} from {path}")


# ============================================================
# SDP 生成
# ============================================================

def build_acf_tensor(
    H: np.ndarray,
    lag_step: int,
    n_delta: int,
    acf_mode: str = "real_shift",
    eps: float = EPS,
) -> np.ndarray:
    """
    H: (NT, NS) complex
    return rho: (NΔ, WT, NS)
    """
    if H.ndim != 2:
        raise ValueError(f"H must be 2D (NT, NS), got {H.shape}")

    NT, NS = H.shape
    max_lag = lag_step * n_delta
    WT = NT - max_lag
    if WT <= 0:
        raise ValueError(
            f"Invalid params: NT={NT}, lag_step={lag_step}, n_delta={n_delta} -> WT={WT}"
        )

    curr = H[max_lag:NT, :]   # (WT, NS)
    rho = np.empty((n_delta, WT, NS), dtype=np.float32)

    for n in range(1, n_delta + 1):
        lag = n * lag_step
        prev = H[max_lag - lag: NT - lag, :]   # (WT, NS)

        prod = curr * np.conj(prev)
        denom = np.maximum(np.abs(curr) * np.abs(prev), eps)

        if acf_mode == "paper_abs":
            # 论文字面实现：你的数据上会退化为 1
            rho_n = np.abs(prod) / denom

        elif acf_mode == "real_shift":
            # 推荐版本：保留复相关的相位信息（实部），再映射到[0,1]
            rho_raw = np.real(prod / denom)     # usually in [-1, 1]
            rho_n = 0.5 * (rho_raw + 1.0)       # -> [0,1]
            rho_n = np.clip(rho_n, 0.0, 1.0)

        elif acf_mode == "real_raw":
            # 不做shift，保留[-1,1]，后续归一化时可能出现负值
            rho_n = np.real(prod / denom)

        elif acf_mode == "amp_corr":
            # 仅幅度相关；信息更弱，更适合作为对照实验
            a1 = np.abs(curr)
            a2 = np.abs(prev)
            rho_n = (a1 * a2) / (np.maximum(np.sqrt(a1 ** 2) * np.sqrt(a2 ** 2), eps))

        else:
            raise ValueError(f"Unsupported acf_mode: {acf_mode}")

        rho[n - 1] = rho_n.astype(np.float32)

    return rho


def acf_to_sdp(rho: np.ndarray, eps: float = EPS) -> np.ndarray:
    """
    按 XFall SDP 思路：
      1) 对子载波维平均
      2) 对每个时间列做归一化
    输出 S: (NΔ, WT)
    """
    if rho.ndim != 3:
        raise ValueError(f"rho must be 3D (NΔ, WT, NS), got {rho.shape}")

    merged = rho.mean(axis=2)   # (NΔ, WT)

    # 为了兼容 real_raw 可能出现负值，这里做一个最小裁剪
    # real_shift / paper_abs / amp_corr 本身不会有这个问题
    merged = np.maximum(merged, 0.0)

    denom = np.maximum(merged.sum(axis=0, keepdims=True), eps)
    S = merged / denom
    return S.astype(np.float32)


def extract_sdp_from_csi(
    csi: np.ndarray,
    lag_step: int,
    n_delta: int,
    layout: str = "wtn",
    acf_mode: str = "real_shift",
) -> np.ndarray:
    """
    csi: (Rx, Tx, Subcarrier, Time)

    return:
      layout='nwt' -> (Rx, Tx, NΔ, WT)
      layout='wtn' -> (Rx, Tx, WT, NΔ)
    """
    if csi.ndim != 4:
        raise ValueError(f"Expected canonical CSI shape (Rx,Tx,Subcarrier,Time), got {csi.shape}")

    RX, TX, NS, NT = csi.shape
    result = np.empty((RX, TX), dtype=object)

    for rx in range(RX):
        for tx in range(TX):
            H = csi[rx, tx].transpose(1, 0)  # -> (Time, Subcarrier)
            rho = build_acf_tensor(
                H=H,
                lag_step=lag_step,
                n_delta=n_delta,
                acf_mode=acf_mode,
            )
            S = acf_to_sdp(rho)
            result[rx, tx] = S

    out = np.stack(
        [np.stack([result[rx, tx] for tx in range(TX)], axis=0) for rx in range(RX)],
        axis=0
    )   # (Rx, Tx, NΔ, WT)

    if layout == "wtn":
        out = np.transpose(out, (0, 1, 3, 2))   # (Rx, Tx, WT, NΔ)

    return out.astype(np.float32)


# ============================================================
# 数据集处理
# ============================================================

def maybe_copy_metadata(src_root: Path, dst_root: Path, src_csi_dir: str, dst_csi_dir: str):
    """
    拷贝 txt 文件；其余非 CSI 目录尽量软链接，失败再复制
    """
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


def save_array(path: Path, arr: np.ndarray, dtype: str):
    if dtype == "float16":
        arr = arr.astype(np.float16)
    else:
        arr = arr.astype(np.float32)
    np.save(path, arr)


def print_sample_stats(name: str, arr: np.ndarray, layout: str):
    print(f"\n=== Preview: {name} ===")
    print("shape:", arr.shape)
    print("dtype:", arr.dtype)
    print("min/max:", float(arr.min()), float(arr.max()))
    print("mean/std:", float(arr.mean()), float(arr.std()))
    print("nan:", bool(np.isnan(arr).any()), "inf:", bool(np.isinf(arr).any()))

    assert arr.ndim == 4
    assert arr.shape[0] == 3 and arr.shape[1] == 3, arr.shape

    if layout == "wtn":
        s = arr[0, 0].transpose(1, 0)   # -> (NΔ, WT)
    else:
        s = arr[0, 0]

    col_sum = s.sum(axis=0)
    col_var = s.var(axis=0)
    print("column sum min/max:", float(col_sum.min()), float(col_sum.max()))
    print("column var min/max/mean:",
          float(col_var.min()), float(col_var.max()), float(col_var.mean()))


def process_dataset(cfg: dict):
    src_root = Path(cfg["src_root"])
    dst_root = Path(cfg["dst_root"])
    src_csi_root = src_root / cfg["src_csi_dir"]
    dst_csi_root = dst_root / cfg["dst_csi_dir"]

    if not src_csi_root.exists():
        raise FileNotFoundError(f"Source CSI dir not found: {src_csi_root}")

    ensure_dir(dst_csi_root)

    if cfg["copy_metadata"]:
        maybe_copy_metadata(src_root, dst_root, cfg["src_csi_dir"], cfg["dst_csi_dir"])

    files = sorted(src_csi_root.glob(f"*{cfg['input_ext']}"))
    if not files:
        raise FileNotFoundError(f"No files matching *{cfg['input_ext']} under {src_csi_root}")

    print("=" * 70)
    print("Formal batch SDP preprocessing")
    print("=" * 70)
    print("src_root   :", src_root)
    print("dst_root   :", dst_root)
    print("src_csi    :", src_csi_root)
    print("dst_csi    :", dst_csi_root)
    print("lag_step   :", cfg["lag_step"])
    print("n_delta    :", cfg["n_delta"])
    print("layout     :", cfg["layout"])
    print("acf_mode   :", cfg["acf_mode"])
    print("dtype      :", cfg["dtype"])
    print("overwrite  :", cfg["overwrite"])
    print("num files  :", len(files))
    print("=" * 70)

    preview_count = 0

    for path in tqdm(files, desc="SDP preprocessing"):
        out_path = dst_csi_root / f"{path.stem}.npy"

        if out_path.exists() and not cfg["overwrite"]:
            continue

        csi = load_csi_from_mat(str(path), mat_key=cfg["mat_key"])
        sdp = extract_sdp_from_csi(
            csi=csi,
            lag_step=cfg["lag_step"],
            n_delta=cfg["n_delta"],
            layout=cfg["layout"],
            acf_mode=cfg["acf_mode"],
        )
        save_array(out_path, sdp, dtype=cfg["dtype"])

        if cfg["save_preview_stats"] and preview_count < cfg["preview_limit"]:
            print_sample_stats(path.name, sdp, layout=cfg["layout"])
            preview_count += 1

    print("\nDone.")
    print(f"Saved offline SDP files to: {dst_csi_root}")


if __name__ == "__main__":
    process_dataset(CONFIG)
