# WiFi CSI 实验对比续表 20260526

日期：2026-05-26

本文延续 `docs/wifi_csi_experiment_comparison_summary_20260519.md` 的记录方式，在 R0-R9 之后继续补充近期围绕 `SDP140 image-like` 和 `Translator / conv_patch` 的对照实验。

注意：

1. R0-R9 的完整记录见 `docs/wifi_csi_experiment_comparison_summary_20260519.md`。
2. 本文新增 R10-R11，重点用于回答：`SDP140 image-like` 输入下，Translator 是否比直接 `conv_patch` 更稳定。
3. R10/R11 均已完整训练到 450 epoch；R11 中途因 NVIDIA driver/library mismatch 暂停过一次，后续从 `epoch_360.pth` 恢复并跑完。
4. R10/R11 当前日志只有旧版 `mpjpe/mpjpe_x/mpjpe_y` 指标，没有 R9 那套 `mpjpe_top1/top-k/AP` 新指标。
5. R11 与 R10 的模型配置本质一致，都是 `SDP140 image-like + conv_patch`；R11 是为了把 6.3 translator ablation 单独做成清晰分支和结果目录。

## 新增实验续表

| 编号 | result 目录 | 主要设置 | max epoch | final / latest MPJPE | best MPJPE | 备注 |
| --- | --- | --- | ---: | ---: | ---: | --- |
| R10 | `49_wifipose_2d_baseline_SDP140_no_translator_lagwindow_centerctx` | SDP140 image-like，去掉 Translator，直接 `conv_patch`，`patch=(3,25)`，`num_query=100`，batch 32 | 450 | 146.1580 / x 127.5250 / y 42.8008 | 96.2048 @ 1 | 完整训练。best 出现在 epoch 1，不能作为强结论；final 明显差于 R5 translator final。 |
| R11 | `50_wifipose_2d_baseline_SDP140_conv_patch_lagwindow_centerctx` | SDP140 image-like + `conv_patch` 独立 ablation 分支，配置与 R10 基本一致，`patch=(3,25)`，`num_query=100`，batch 32 | 450 | 297.1058 / x 286.4647 / y 57.7145 | 81.8767 @ 206 | 完整训练。最终 checkpoint 为 `epoch_450.pth`。训练中途因 NVIDIA driver/library mismatch 曾暂停，正式结果以从 `epoch_360.pth` resume 后跑到 450 epoch 的日志为准。 |

## Translator 与 Conv Patch 对照

为方便汇报，下面只列和 `SDP140 image-like` 相关的三组核心对照。

| 对照 | result 目录 | 输入适配方式 | 训练状态 | final / latest MPJPE | best MPJPE | 初步结论 |
| --- | --- | --- | --- | ---: | ---: | --- |
| R5 | `44_wifipose_2d_baseline_SDP140_translator_image_lagwindow_centerctx` | `SDPToImageLikeTranslator`，再 patch embedding | 完整 450 epoch | 112.6839 / x 98.2272 / y 35.1492 | 101.3820 @ 106 | final 最稳定，是当前 Translator 路线的主要基线。 |
| R10 | `49_wifipose_2d_baseline_SDP140_no_translator_lagwindow_centerctx` | 直接 `conv_patch` | 完整 450 epoch | 146.1580 / x 127.5250 / y 42.8008 | 96.2048 @ 1 | final 明显退化，说明直接 `conv_patch` 后期不稳定。 |
| R11 | `50_wifipose_2d_baseline_SDP140_conv_patch_lagwindow_centerctx` | 直接 `conv_patch` | 完整 450 epoch | 297.1058 / x 286.4647 / y 57.7145 | 81.8767 @ 206 | best 很好，但 final 严重退化，说明直接 `conv_patch` 的后期稳定性问题更明显。 |

## 阶段性结论

### 1. Translator 的价值更偏向稳定性

R5 的 final MPJPE 为 `112.6839`，明显优于 R10 的 final `146.1580`，也明显优于 R11 的 final `297.1058`。这说明在同样的 SDP140 image-like 输入下，加入 Translator 后，训练后期更稳定。

R10 的 best 为 `96.2048 @ 1`，数值看起来略优于 R5 的 best `101.3820 @ 106`，但这个 best 出现在第 1 个 epoch，属于非常早的测试集 oracle 指标，不适合作为“no-translator 更好”的证据。更合理的比较应看 final、稳定区间和多轮趋势，因此 R5 目前更可靠。

### 2. 直接 conv_patch 可以学到有效表示，但波动大

R11 的 best 达到 `81.8767 @ 206`，说明 `SDP140 image-like + conv_patch` 并非不可行，候选池中确实能出现很接近 GT 的 query。

但 R11 后期指标波动很大，且 final 退化到 `297.1058`。例如：

| epoch | MPJPE | MPJPE-x | MPJPE-y |
| ---: | ---: | ---: | ---: |
| 361 | 235.5624 | 226.9064 | 44.3634 |
| 365 | 116.6260 | 99.8928 | 38.5463 |
| 367 | 282.5794 | 271.5808 | 50.3713 |
| 371 | 118.9093 | 94.8107 | 49.5941 |
| 374 | 179.9494 | 162.6874 | 50.7734 |
| 420 | 181.1387 | 166.1418 | 52.6239 |
| 450 | 297.1058 | 286.4647 | 57.7145 |

这种波动说明直接 `conv_patch` 的训练过程仍然不稳定，特别是 x 方向误差经常主导 MPJPE 上升。

### 3. R10 与 R11 不是两个不同方法

R10 和 R11 的关键配置相同：

- 数据：`all_single_*_sdp140_imagelike_centerctx_n30_power_xfall`
- 离线 SDP 目录：`csi_sdp140_imagelike_lagwindow_offline`
- 输入：`SDP140 image-like`，shape 约 `(30,270,3)`
- 适配器：`input_adapter='conv_patch'`
- patch：`patch_kernel_size=(3,25)`，`patch_stride=(3,25)`
- query：`num_query=100`
- batch：`samples_per_gpu=32`

因此 R11 更准确地说是对 R10 的独立 ablation 复现实验，而不是新结构。R10 和 R11 均已完整完成；两者共同支持一个结论：直接 `conv_patch` 虽然可能出现较好的中间 best，但 final 稳定性弱于 Translator 路线。

### 4. 和 R3/R5 的关系

目前主线结论仍然可以这样表述：

1. `SDP_offline_power_xfall` baseline，也就是 R3，仍然是旧 MPJPE 下最强主线基线之一，best 为 `77.6551 @ 391`，final 为 `98.5397`。
2. `SDP140 image-like + Translator`，也就是 R5，final 稳定性优于直接 `conv_patch` 的 R10。
3. `SDP140 image-like + conv_patch` 的 R11 出现过很好的 best `81.8767 @ 206`，但 final 退化到 `297.1058`，说明这个路线需要额外的稳定化策略，不能只凭 best 判断优于 Translator。

## 汇报建议

向老师汇报时建议强调三点：

1. 我们补了公平对照：固定 SDP140 image-like 预处理，只改变是否加入 Translator。
2. 完整训练的 R10 表明：去掉 Translator 后 final 变差，因此 Translator 至少提高了训练稳定性。
3. R11 显示直接 `conv_patch` 仍有潜力，但完整训练后 final 退化严重，说明后续优化重点不是简单保留或删除 Translator，而是解决 SDP image-like 表示到 pose query 的稳定映射问题。

后续建议：

1. 后续在 R5/R10/R11 中统一保存 best checkpoint，避免只看到 final 或只看到测试集 best 数字。
2. 在同一 GPU 环境下复跑一次 R5/R10/R11 的短 schedule，观察曲线稳定性是否可复现。
3. 后续正式比较应使用新增的 `mpjpe_top1/top-k/AP` 指标，而不是只看旧版 oracle-all MPJPE。
