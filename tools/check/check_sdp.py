import os
import random
import argparse
import numpy as np
import matplotlib.pyplot as plt


def inspect_one(path, layout="nwt", save_fig=False, fig_dir=None):
    x = np.load(path)

    print(f"\n=== {os.path.basename(path)} ===")
    print("shape:", x.shape)
    print("dtype:", x.dtype)
    print("min/max:", x.min(), x.max())
    print("mean/std:", x.mean(), x.std())
    print("nan:", np.isnan(x).any(), "inf:", np.isinf(x).any())

    assert x.ndim == 4, f"Expect 4D, got {x.shape}"
    assert x.shape[0] == 3 and x.shape[1] == 3, f"Expect first dims=(3,3), got {x.shape}"

    if layout == "nwt":
        # x: (3,3,NΔ,WT)
        s = x[0, 0]   # (NΔ, WT)
    elif layout == "wtn":
        # x: (3,3,WT,NΔ)
        s = x[0, 0].transpose(1, 0)  # -> (NΔ, WT)
    else:
        raise ValueError(layout)

    # 非负性检查
    print("all >= 0:", np.all(s >= -1e-6))

    # 每一列是否归一化
    col_sum = s.sum(axis=0)
    print("column sum min/max:", col_sum.min(), col_sum.max())
    print("column sum mean:", col_sum.mean())

    # 是否退化成常数
    col_var = s.var(axis=0)
    print("column var min/max/mean:", col_var.min(), col_var.max(), col_var.mean())

    if save_fig:
        os.makedirs(fig_dir, exist_ok=True)
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


def inspect_dataset(root, num_samples=10, layout="nwt", save_fig=False):
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

        inspect_one(path, layout=layout, save_fig=save_fig, fig_dir=fig_dir)

    print("\n=== Summary on sampled files ===")
    print("shape counts:", all_shapes)
    print("global min range:", min(global_min), max(global_min))
    print("global max range:", min(global_max), max(global_max))
    print("column sum mean range:", min(colsum_means), max(colsum_means))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=str, default='/home/xl/CSI/Person-in-WiFi-3D-repo/data/wifipose/all_single_train_data_hold_out_sdp_offline/csi_sdp_offline', help="directory of offline SDP .npy files")
    parser.add_argument("--num-samples", type=int, default=10)
    parser.add_argument("--layout", type=str, default="wtn", choices=["nwt", "wtn"])
    parser.add_argument("--save-fig", action="store_true")
    args = parser.parse_args()

    inspect_dataset(
        root=args.root,
        num_samples=args.num_samples,
        layout=args.layout,
        save_fig=args.save_fig
    )