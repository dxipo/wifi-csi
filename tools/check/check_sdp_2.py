import os
import random
import numpy as np
import matplotlib.pyplot as plt
import h5py


EPS = 1e-8


# ============================================================
# 基础工具
# ============================================================

def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def _structured_to_complex(arr: np.ndarray) -> np.ndarray:
    """
    将 h5py 读出来的 structured dtype:
        [('real','<f8'), ('imag','<f8')]
    转成复数数组
    """
    if arr.dtype.fields is None:
        return np.asarray(arr)

    field_names = list(arr.dtype.fields.keys())
    if "real" in field_names and "imag" in field_names:
        return arr["real"] + 1j * arr["imag"]

    raise ValueError(f"Unsupported structured dtype fields: {field_names}")


def load_csi_from_mat(path: str, mat_key: str = "csi_out") -> np.ndarray:
    """
    读取原始 .mat CSI，并统一成:
        (Rx, Tx, Subcarrier, Time) = (3, 3, 30, 20)

    支持两种常见原始形状:
        (20, 30, 3, 3)
        (3, 3, 30, 20)
    """
    with h5py.File(path, "r") as f:
        if mat_key not in f:
            raise KeyError(f"Key '{mat_key}' not found in {path}. Available keys: {list(f.keys())}")
        raw = f[mat_key][()]

    raw = _structured_to_complex(np.array(raw))

    if raw.ndim != 4:
        raise ValueError(f"Unsupported raw CSI ndim: {raw.ndim}, shape={raw.shape}")

    if raw.shape == (20, 30, 3, 3):
        raw = np.transpose(raw, (2, 3, 1, 0))   # -> (3,3,30,20)
    elif raw.shape == (3, 3, 30, 20):
        pass
    else:
        raise ValueError(f"Unsupported raw CSI shape: {raw.shape}")

    return raw


# ============================================================
# 论文 SDP / rho 检查
# ============================================================

def build_rho(
    H: np.ndarray,
    lag_step: int,
    n_delta: int,
    acf_mode: str = "paper_abs",
    eps: float = EPS
) -> np.ndarray:
    """
    构造 rho: (NΔ, WT, NS)

    Parameters
    ----------
    H : complex ndarray, shape (NT, NS)
        单条链路 CSI，时间在 axis=0，子载波在 axis=1
    lag_step : int
        ΔT 对应的离散步长
    n_delta : int
        NΔ，lag 数量
    acf_mode : str
        "paper_abs"  : 论文字面实现
        "real_shift" : 工程修正版，避免退化成常数特征

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
            # 按论文字面实现:
            # rho = |H(t) H*(t-lag)| / (|H(t)| |H(t-lag)|)
            rho_n = np.abs(prod) / denom

        elif acf_mode == "real_shift":
            # 工程修正版:
            # 取归一化复数乘积的实部，再映射到 [0,1]
            # 这样不会像 paper_abs 一样天然退化到 1
            rho_raw = np.real(prod / denom)     # typically in [-1, 1]
            rho_n = 0.5 * (rho_raw + 1.0)       # -> [0,1]
            rho_n = np.clip(rho_n, 0.0, 1.0)

        else:
            raise ValueError(f"Unsupported acf_mode: {acf_mode}")

        rho[n - 1] = rho_n.astype(np.float32)

    return rho


def rho_to_sdp(rho: np.ndarray, eps: float = EPS) -> np.ndarray:
    """
    按论文思路:
      1) 对子载波维做融合
      2) 对每个时间列做归一化
    得到 S: (NΔ, WT)
    """
    if rho.ndim != 3:
        raise ValueError(f"rho must be 3D (NΔ, WT, NS), got {rho.shape}")

    merged = rho.mean(axis=2)   # (NΔ, WT)
    denom = np.maximum(merged.sum(axis=0, keepdims=True), eps)
    S = merged / denom
    return S.astype(np.float32)


def inspect_rho_from_raw_mat(
    mat_path: str,
    mat_key: str = "csi_out",
    lag_step: int = 1,
    n_delta: int = 6,
    rx: int = 0,
    tx: int = 0,
    acf_mode: str = "paper_abs",
    save_fig: bool = True,
    fig_dir: str = None,
):
    """
    从原始 .mat 中读取 CSI，选一条链路，现算 rho 和 S
    """
    csi = load_csi_from_mat(mat_path, mat_key=mat_key)   # (3,3,30,20)

    # 取某条链路，转成 H: (NT, NS) = (Time, Subcarrier)
    H = csi[rx, tx].transpose(1, 0)   # (20,30)

    rho = build_rho(
        H=H,
        lag_step=lag_step,
        n_delta=n_delta,
        acf_mode=acf_mode
    )   # (NΔ, WT, NS)

    S = rho_to_sdp(rho)   # (NΔ, WT)

    print("\n===== RAW MAT RHO CHECK =====")
    print("mat:", mat_path)
    print("link (rx, tx):", (rx, tx))
    print("acf_mode:", acf_mode)
    print("rho shape:", rho.shape)
    print("rho min/max:", float(rho.min()), float(rho.max()))
    print("rho mean/std:", float(rho.mean()), float(rho.std()))

    lag_var = rho.var(axis=0)   # (WT, NS)
    print("rho lag-var min/max/mean:",
          float(lag_var.min()), float(lag_var.max()), float(lag_var.mean()))

    print("S shape:", S.shape)
    print("S min/max:", float(S.min()), float(S.max()))
    print("S mean/std:", float(S.mean()), float(S.std()))
    print("S column sum min/max:",
          float(S.sum(axis=0).min()), float(S.sum(axis=0).max()))
    print("S column var min/max/mean:",
          float(S.var(axis=0).min()), float(S.var(axis=0).max()), float(S.var(axis=0).mean()))

    if save_fig:
        if fig_dir is None:
            fig_dir = os.path.join(os.path.dirname(mat_path), "_rho_vis")
        ensure_dir(fig_dir)

        # 保存 S 热图
        plt.figure(figsize=(6, 4))
        plt.imshow(S, aspect='auto', origin='lower')
        plt.colorbar()
        plt.title(f"S from raw mat ({acf_mode})")
        plt.xlabel("time window WT")
        plt.ylabel("lag bin NΔ")
        save_path_s = os.path.join(
            fig_dir,
            os.path.basename(mat_path).replace(".mat", f"_{acf_mode}_S.png")
        )
        plt.tight_layout()
        plt.savefig(save_path_s, dpi=200)
        plt.close()
        print("saved fig:", save_path_s)

        # 保存 rho 在子载波均值后的热图（其实和 S 未归一化前更接近）
        rho_mean = rho.mean(axis=2)  # (NΔ, WT)
        plt.figure(figsize=(6, 4))
        plt.imshow(rho_mean, aspect='auto', origin='lower')
        plt.colorbar()
        plt.title(f"rho-mean over subcarriers ({acf_mode})")
        plt.xlabel("time window WT")
        plt.ylabel("lag bin NΔ")
        save_path_rho = os.path.join(
            fig_dir,
            os.path.basename(mat_path).replace(".mat", f"_{acf_mode}_rho_mean.png")
        )
        plt.tight_layout()
        plt.savefig(save_path_rho, dpi=200)
        plt.close()
        print("saved fig:", save_path_rho)

    return rho, S


# ============================================================
# 已保存 SDP 文件检查
# ============================================================

def inspect_one_saved_sdp(path, layout="wtn", save_fig=False, fig_dir=None):
    x = np.load(path)

    print(f"\n=== {os.path.basename(path)} ===")
    print("shape:", x.shape)
    print("dtype:", x.dtype)
    print("min/max:", float(x.min()), float(x.max()))
    print("mean/std:", float(x.mean()), float(x.std()))
    print("nan:", bool(np.isnan(x).any()), "inf:", bool(np.isinf(x).any()))

    assert x.ndim == 4, f"Expect 4D, got {x.shape}"
    assert x.shape[0] == 3 and x.shape[1] == 3, f"Expect first dims=(3,3), got {x.shape}"

    if layout == "nwt":
        # x: (3,3,NΔ,WT)
        s = x[0, 0]
    elif layout == "wtn":
        # x: (3,3,WT,NΔ) -> 转回 (NΔ, WT) 便于统一检查
        s = x[0, 0].transpose(1, 0)
    else:
        raise ValueError(layout)

    print("all >= 0:", bool(np.all(s >= -1e-6)))

    col_sum = s.sum(axis=0)
    print("column sum min/max:", float(col_sum.min()), float(col_sum.max()))
    print("column sum mean:", float(col_sum.mean()))

    col_var = s.var(axis=0)
    print("column var min/max/mean:",
          float(col_var.min()), float(col_var.max()), float(col_var.mean()))

    if save_fig:
        ensure_dir(fig_dir)
        plt.figure(figsize=(6, 4))
        plt.imshow(s, aspect='auto', origin='lower')
        plt.colorbar()
        plt.title(os.path.basename(path))
        plt.xlabel("time window WT")
        plt.ylabel("lag bin NΔ")
        save_path = os.path.join(fig_dir, os.path.basename(path).replace(".npy", ".png"))
        plt.tight_layout()
        plt.savefig(save_path, dpi=200)
        plt.close()
        print("saved fig:", save_path)


def inspect_saved_sdp_dataset(root, num_samples=10, layout="wtn", save_fig=True):
    files = [os.path.join(root, f) for f in os.listdir(root) if f.endswith(".npy")]
    files = sorted(files)
    assert len(files) > 0, f"No .npy found in {root}"

    print("total files:", len(files))

    chosen = random.sample(files, min(num_samples, len(files)))
    fig_dir = os.path.join(root, "_sdp_vis")

    all_shapes = {}
    colsum_means = []
    global_min = []
    global_max = []

    for path in chosen:
        x = np.load(path)
        all_shapes[x.shape] = all_shapes.get(x.shape, 0) + 1
        global_min.append(float(x.min()))
        global_max.append(float(x.max()))

        if layout == "nwt":
            s = x[0, 0]
        else:
            s = x[0, 0].transpose(1, 0)

        colsum_means.append(float(s.sum(axis=0).mean()))

        inspect_one_saved_sdp(path, layout=layout, save_fig=save_fig, fig_dir=fig_dir)

    print("\n=== Summary on sampled files ===")
    print("shape counts:", all_shapes)
    print("global min range:", min(global_min), max(global_min))
    print("global max range:", min(global_max), max(global_max))
    print("column sum mean range:", min(colsum_means), max(colsum_means))


# ============================================================
# 默认配置区：你只改这里，然后直接运行
# ============================================================

MODE = "rho"   # 可选: "rho" 或 "saved_sdp"

# -------- mode = "saved_sdp" 时用 --------
SAVED_SDP_ROOT = "/home/xl/CSI/Person-in-WiFi-3D-repo/data/wifipose/all_single_train_data_hold_out_sdp_offline/csi_sdp_offline"
SAVED_SDP_NUM_SAMPLES = 10
SAVED_SDP_LAYOUT = "wtn"   # "wtn" or "nwt"
SAVED_SDP_SAVE_FIG = True

# -------- mode = "rho" 时用 --------
RAW_MAT_PATH = "/home/xl/CSI/Person-in-WiFi-3D-repo/data/wifipose/all_single_train_data_hold_out/csi/S11_17_246.mat"
RAW_MAT_KEY = "csi_out"
RAW_LAG_STEP = 1
RAW_N_DELTA = 6
RAW_RX = 0
RAW_TX = 0
#RAW_ACF_MODE = "paper_abs"     # "paper_abs" 或 "real_shift"
RAW_ACF_MODE = "real_shift"
RAW_SAVE_FIG = True
RAW_FIG_DIR = "/home/xl/CSI/Person-in-WiFi-3D-repo/tools/check/_rho_vis"


# ============================================================
# 主程序
# ============================================================

if __name__ == "__main__":
    print("Running mode:", MODE)

    if MODE == "saved_sdp":
        inspect_saved_sdp_dataset(
            root=SAVED_SDP_ROOT,
            num_samples=SAVED_SDP_NUM_SAMPLES,
            layout=SAVED_SDP_LAYOUT,
            save_fig=SAVED_SDP_SAVE_FIG
        )

    elif MODE == "rho":
        inspect_rho_from_raw_mat(
            mat_path=RAW_MAT_PATH,
            mat_key=RAW_MAT_KEY,
            lag_step=RAW_LAG_STEP,
            n_delta=RAW_N_DELTA,
            rx=RAW_RX,
            tx=RAW_TX,
            acf_mode=RAW_ACF_MODE,
            save_fig=RAW_SAVE_FIG,
            fig_dir=RAW_FIG_DIR
        )

    else:
        raise ValueError(f"Unsupported MODE: {MODE}")