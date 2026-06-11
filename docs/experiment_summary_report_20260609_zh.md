# 3D 复现、MMFi 与 SDP 预处理实验阶段总结 - 2026-06-09

## 1. 总体结论

本阶段实验围绕三条主线展开：

1. 复现 Person-in-WiFi 3D 原论文代码，建立可和论文结果对照的 3D 主基线。
2. 分析 MMFi 数据集能否作为新的 3D 姿态估计与蒸馏实验数据源。
3. 将此前 2D 实验中效果较好的 `SDP power_xfall` 预处理迁移到原论文 3D 任务中，验证该预处理在 3D 标签下是否仍有效。

当前最重要的实验结论是：

- `origin_paper` 原始预处理复现实验已经跑通，最优 `MPJPE = 117.20 @ epoch 391`，可以作为后续 3D 可比实验基线。
- `SDP power_xfall` 直接替换原论文 3D 预处理后，当前最好 `MPJPE = 228.50 @ epoch 316`；截至 `epoch 364` 的最新验证为 `MPJPE = 285.93`，明显弱于原始预处理。
- MMFi 可以使用 `ground_truth.npy` 作为 3D 主监督标签，`rgb/frame*.npy` 的 2D 骨架更适合作为训练期蒸馏或先验信息，而不是主标签。
- 原有两阶段蒸馏策略仍有迁移价值，但应迁移为 3D 主任务下的 token-level representation distillation，不能简单复用 2D 像素坐标监督逻辑。

## 2. 分支与实验产物

| 任务 | 分支 / 工作区 | 说明 |
|---|---|---|
| 原论文代码复现 | `origin_paper` | 官方 Person-in-WiFi 3D 代码，删除 demo 后上传远端，用原始数据训练 |
| MMFi 3D baseline | `mmfi_b0_3d_baseline` 相关配置 | 使用 MMFi `ground_truth.npy` 作为 3D 标签 |
| MMFi origin-style stem | `mmfi_b1_origin_preprocess_stem` | 尝试把更接近原论文的预处理/stem 思路迁移到 MMFi |
| 当前 SDP 3D 实验 | `origin_paper_sdp_power_xfall_3d` | 基于 `origin_paper`，把 2D 分支中的 `SDP power_xfall` 预处理迁移到原始 3D 数据 |

当前正在运行的训练是：

- 分支：`origin_paper_sdp_power_xfall_3d`
- 工作区：`/tmp/origin_paper_sdp_power_xfall_3d`
- commit：`d77c161`
- 当前进度：约 `epoch 365/450`
- 训练日志：`result/origin_paper_sdp_power_xfall_3d/20260608_021534.log`
- stdout：`result/origin_paper_sdp_power_xfall_3d/train_seed42_offline_20260608_021532.out`

## 3. 统一实验结果表

下表统一整理本阶段已经完成或正在运行的核心实验。需要注意：Person-in-WiFi 3D 与 MMFi 的坐标系、关键点数量和评价口径不同，因此 MMFi 的 MPJPE 不应与 Person-in-WiFi 3D 结果做直接横向数值比较。

| 编号 | 实验 / 分支 | 数据集与任务 | 输入 / 预处理 | 状态 | best MPJPE | latest / final MPJPE | 主要观察 |
|---|---|---|---|---|---:|---:|---|
| B0-PiW3D | `origin_paper` | Person-in-WiFi 3D，`14 x 3` | 原论文 amp + phase 预处理，`Linear(60,256)` | 已完成，450 epoch | `117.20 @ 391` | `119.77 @ 450` | 当前最可靠的 3D 主基线，后续 PiW3D 改进应与它比较 |
| B0-MMFi | `mmfi_b0_3d_baseline` | MMFi 3D，`17 x 3` | MMFi CSI baseline stem | 已完成，80 epoch | `242.56 @ 10` | `255.48 @ 80` | 后期基本在 250 左右波动，作为 MMFi 初始 baseline |
| B1-MMFi | `mmfi_b1_origin_preprocess_stem` | MMFi 3D，`17 x 3` | 尝试 origin-style preprocess / stem | 已完成，80 epoch | `226.08 @ 8` | `533.06 @ 80` | 早期 best 更低，但后期明显退化，训练稳定性差 |
| A1-PiW3D-SDP | `origin_paper_sdp_power_xfall_3d` | Person-in-WiFi 3D，`14 x 3` | SDP power_xfall，离线 `.npy`，`Linear(6,256)`，`81` query | 运行中，约 `365/450` epoch | `228.50 @ 316` | `285.93 @ 364` | 比早期明显改善，但仍显著弱于 `origin_paper`；说明 SDP power_xfall 不适合作为 3D 主输入直接替换 |

从统一表可以看到，目前最强且最稳定的 3D 可比基线仍是 `origin_paper`。`origin_paper_sdp_power_xfall_3d` 虽然证明 SDP 特征可以被模型学习到一定程度，但作为直接替换主输入时，距离原论文预处理仍有明显差距。

## 4. 原论文 3D 复现实验

### 4.1 实验设置

目标是验证官方 Person-in-WiFi 3D 代码在本地数据和环境下能否复现实验结果，并建立后续 3D 实验的主基线。

关键设置：

- 数据目录：`data/wifipose/train_data`、`data/wifipose/test_data`
- 随机种子：`42`
- checkpoint 目录：`result/origin_paper`
- 配置文件：`configs/wifi/petr_wifi.py`
- 评价指标：代码内置 `mpjpe`
- 训练轮数：`450`

实验结果见第 3 节统一实验结果表。该实验说明本地 `wifi3d` 环境、原始数据和原论文训练流程是可运行的。后续任何 3D 改进都应优先和此结果比较。

### 4.2 指标含义

`origin_paper` 中的 MPJPE 是直接在释放的 3D 坐标上计算预测点和 GT 点之间的欧氏距离。它不是 2D 像素坐标误差，也不是 pelvis 对齐后的 PA-MPJPE。

这意味着：

- 原 2D 实验中的像素坐标 MPJPE 不能和该 3D MPJPE 直接横向比较。
- 如果改用 MMFi 或其他数据集，需要单独说明坐标系、单位、是否 pelvis 对齐。

## 5. 原始数据与标定文件分析

检查过 `/home/xl/BaiduDownloads/rgbd/S11_06/calibration.json` 及已解压的 `/home/xl/BaiduDownloads/S11_06` 后，可以形成以下判断：

1. `calibration.json` 属于 RGBD 数据采集过程中的相机标定信息，通常用于描述相机内参、外参或深度/RGB 对齐关系。
2. 原始 `.mat` 中的 `14 x 3` 骨骼点不是 RGB 图像像素点，而是 3D 坐标形式。
3. 这些小数坐标更接近真实空间坐标或相机坐标系下的坐标，而不是 2D 图像坐标。
4. 若要把 3D 骨骼点投影到 RGB 图像，需要明确：
   - 3D 点所在坐标系；
   - RGB 相机内参；
   - 3D 坐标系到 RGB 相机坐标系的外参；
   - RGB 帧与骨骼标签帧的严格时间同步关系。

当前不建议把 3D 点投影到 RGB 作为主路线前置条件。更稳妥的方案是：3D 主任务继续使用官方 3D GT，视觉信息通过蒸馏或辅助表征在训练阶段引入。

## 6. MMFi 数据集处理与实验

### 6.1 数据理解

MMFi 数据集中计划使用的 3D 标签为：

```text
/home/xl/Downloads/MMFi_Dataset/MMFi_unzipped/E01/S01/A01/ground_truth.npy
```

本阶段明确了两类信息的用途：

- `ground_truth.npy`：作为 3D 主任务标签。
- `rgb/frame*.npy`：形状为 `17 x 2`，是已经提取好的 2D 骨架点，可作为视觉先验或蒸馏输入。

因此，MMFi 并不是不能做 3D supervised learning。关键是不能把 `rgb/frame*.npy` 误认为 3D 标签；真正的 3D 标签应来自 `ground_truth.npy`。

### 6.2 关键点数量

Person-in-WiFi 3D 使用 `14 x 3` 关键点，MMFi 使用 `17 x 3` 关键点。后续实验有两种路线：

- 在 MMFi 上保留 `17` 点，修改输出 head 与评价指标。
- 为了和 Person-in-WiFi 3D 保持结构一致，从 MMFi 17 点中筛出可对应的 14 点。

更推荐前者作为 MMFi 主实验，因为可以避免人为映射造成的误差；后者只适合作为跨数据集结构对照。

### 6.3 已完成实验结果

实验结果见第 3 节统一实验结果表。
这说明 MMFi 任务当前不是简单“换数据集即可稳定工作”。主要问题不是单一的学习率或 loss，而是 CSI 数据形态、标签尺度、输入 stem 和 PETR 查询机制之间还没有完全匹配。

## 7. 两阶段蒸馏方案的整理

此前的 2D 蒸馏方案不是简单加一个 2D auxiliary head，而是基于 Transformer token 的两阶段蒸馏。

原有思路可以概括为：

1. Teacher 使用视觉输入，提取中间层 token。
2. Student 使用 CSI 输入，学习拟合 teacher token。
3. Stage 1 只做 token-level 蒸馏预训练。
4. Stage 2 加载 Stage 1 权重，再进行主任务训练。

迁移到 3D 主任务时，核心逻辑应调整为：

```text
Stage 1:
CSI student token -> 拟合视觉 / 2D skeleton teacher token

Stage 2:
加载 Stage 1 初始化
CSI student -> 预测 3D keypoints
主监督 = 3D GT
```

对于 MMFi，因为公开数据中的 RGB 已经变成 `17 x 2` 骨架点，无法直接复用 VitPose 图像 token。因此更合理的 teacher 设计是：

- 输入：连续帧 `T x 17 x 2` 的 2D 骨架序列；
- 输出：center frame 的 `17 x 3` 或中间 token；
- teacher 使用 `ground_truth.npy` 训练；
- student 在训练期学习 teacher token；
- 推理期仍然只使用 CSI。

该设计保留了“视觉模态提供额外知识”的创新点，同时不破坏 3D MPJPE 的主指标可比性。

## 8. SDP power_xfall 迁移到 3D 的实验

### 8.1 实验目的

该实验的目标是验证此前在 2D 实验中表现较好的 `SDP power_xfall` 预处理，是否也能提升原 Person-in-WiFi 3D 任务。

### 8.2 实验设置

分支：

```text
origin_paper_sdp_power_xfall_3d
```

主要改动：

- 保留原论文 3D 标签、loss、optimizer、学习率和 epoch 设置。
- 将 `42_wifipose_2d_baseline_SDP_offline_power_xfall` 路线中的 SDP power_xfall 预处理迁移到原始 3D 数据。
- 使用离线预处理，避免在线 SDP 提取导致训练速度过慢。

预处理参数：

```text
window_size = 12
stride = 1
n_delta = 6
layout = wtn
use_hampel = True
hampel_window = 2
hampel_sigma = 3.0
use_moving_average = True
ma_window = 3
acf_unbiased = False
positive_clip = True
zero_column_fill = uniform
```

离线特征目录：

```text
data/wifipose/train_data/csi_sdp_power_xfall
data/wifipose/test_data/csi_sdp_power_xfall
```

离线转换结果：

| split | 样本数 | 输出特征 |
|---|---:|---|
| train | `89946` | `.npy`, shape 约 `(3, 3, 9, 6)` |
| test | `7824` | `.npy`, shape 约 `(3, 3, 9, 6)` |

### 8.3 必要结构改动

这轮实验虽然目标是验证预处理，但由于输入形态变化，模型入口必须同步调整：

| 项目 | 原论文预处理 | SDP power_xfall |
|---|---:|---:|
| token 数 | `3 x 3 x 20 = 180` | `3 x 3 x 9 = 81` |
| 每 token 维度 | `60` | `6` |
| stem | `Linear(60, 256)` | `Linear(6, 256)` |
| query / proposal | `100` | `81` |

因此该实验并不是绝对纯粹的“只换预处理”。它更准确地说是：

```text
SDP power_xfall 特征 + 对应输入 stem/query 适配
```

### 8.4 训练状态与结果

截至 2026-06-09 15:10 左右：

- 当前训练仍在运行；
- 进度约 `epoch 365/450`；
- 最新完整验证为 `epoch 364`；
- 预计还需约 `7 小时` 完成。

实验结果见第 3 节统一实验结果表。
当前 SDP 3D 实验相比 early epoch 已经明显改善，但仍明显弱于原始预处理。其误差主要在 h/v 方向仍然偏大，说明直接用 SDP power_xfall 替换原始 amp+phase 预处理，会损失对 3D 空间定位有用的信息。

### 8.5 TensorBoard 状态

当前 `origin_paper_sdp_power_xfall_3d` 没有加入 TensorBoard。

当前配置为：

```python
log_config = dict(interval=50, hooks=[dict(type='TextLoggerHook')])
```

因此当前只记录普通 `.log` 和 `.log.json`，不会生成 TensorBoard event 文件。若后续需要曲线展示，应在新实验或 resume 实验前加入：

```python
log_config = dict(
    interval=50,
    hooks=[
        dict(type='TextLoggerHook'),
        dict(type='TensorboardLoggerHook')
    ])
```

## 9. 阶段性判断

### 9.1 对 SDP power_xfall 的判断

`SDP power_xfall` 在 2D 像素坐标任务中曾经有效，但直接迁移到 3D 主任务后效果不足。可能原因包括：

1. 3D 任务需要深度、相位差、绝对幅度尺度等信息；
2. power_xfall 的 autocorrelation、clip 和归一化过程可能削弱了绝对空间定位信息；
3. 特征从 `180 x 60` 压缩到 `81 x 6`，信息量下降明显；
4. PETR 原结构和 query 机制更适配原始 amp+phase 特征形态。

因此，不建议把 `SDP power_xfall` 作为 3D 主输入的直接替代方案。

更合理的后续方向是：

- 保留原论文 amp+phase 主输入；
- 将 SDP power_xfall 作为辅助分支、额外 token 或正则化特征；
- 或将 SDP 预处理用于蒸馏阶段，而不是替代主监督输入。

### 9.2 对 MMFi 的判断

MMFi 可以作为后续创新实验的数据集，但需要先稳定 B0 baseline。当前 MMFi B0/B1 结果说明：

- 直接迁移 Person-in-WiFi 3D 的 PETR 结构不一定稳定；
- MMFi 的 CSI 形态和标签组织需要专门设计 dataset、stem 和 metric；
- 在 B0 没有稳定前，不应急于加入蒸馏，否则提升或退化都难以解释。

### 9.3 对蒸馏路线的判断

蒸馏仍然是值得保留的创新点，但应围绕 3D 主任务重新组织：

- 主指标必须是 3D MPJPE；
- 主监督必须是真实 3D GT；
- 2D skeleton 或视觉 token 只作为训练期额外知识；
- 推理阶段保持 CSI-only；
- 必须设置 shuffled token / no token / weak teacher 等 ablation，证明提升来自视觉知识，而不是额外正则化。

## 10. 后续建议

短期建议：

1. 让 `origin_paper_sdp_power_xfall_3d` 当前训练完整跑到 `450 epoch`，保留 final 和 best，形成完整对照。
2. 若结果仍显著弱于 `origin_paper`，将结论定为“SDP power_xfall 不适合作为 3D 主输入直接替换”。
3. 新建 hybrid 实验：原始 amp+phase 主输入 + SDP power_xfall 辅助分支。
4. 后续所有新实验默认加入 TensorBoard，便于汇报曲线。

中期建议：

1. 先稳定 MMFi B0 baseline。
2. 实现 2D skeleton teacher：`T x 17 x 2 -> center frame 17 x 3 / teacher token`。
3. 实现 CSI student token distillation。
4. 进行 D1/D2/D3/D4 ablation：
   - D1：token-only 预训练；
   - D2：Stage 1 初始化 + 3D fine-tune；
   - D3：Stage 2 继续 token regularization；
   - D4：shuffled token control。

最终建议主线：

```text
主线 A：Person-in-WiFi 3D
origin_paper baseline -> preprocessing/stem ablation -> hybrid SDP auxiliary

主线 B：MMFi 3D
MMFi CSI-only baseline -> 2D skeleton teacher -> token distillation -> ablation
```

这两条线应分开汇报。Person-in-WiFi 3D 用于和原论文可比，MMFi 用于展示新数据集、新蒸馏设计和 CSI-only 推理约束下的视觉知识迁移。
