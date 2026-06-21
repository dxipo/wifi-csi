# MMFi D5 Root-Relative Distillation 实验分析

日期：2026-06-21

分支：`mmfi_d5_root_relative_distill`

提交：`0f86f9f Add MMFi D5 root-relative distillation`

## 1. 实验目的

D5 是在 D4 主路径耦合蒸馏基础上的进一步尝试。D4 的主要现象是：PA-MPJPE 有一定改善趋势，但 absolute MPJPE 没有稳定超过 B0 baseline。这个现象说明模型可能学到了一些相对人体结构，但全局绝对坐标，尤其深度/根节点位置，没有被有效改善。

因此 D5 的核心目标是：

1. 保留 CSI-only 推理设定，推理阶段不使用 RGB 或 2D pose。
2. 训练阶段继续使用 2D skeleton teacher token 作为辅助知识。
3. 把蒸馏从“独立辅助分支”推进到 PETR 主 decoder/refine 路径。
4. 显式加入 root/global position 约束和按坐标轴的监督，尝试把 PA-MPJPE 的相对姿态改善转化为 absolute MPJPE 改善。

## 2. 结果文件位置

当前 D5 正式实验结果目录：

```text
/home/xl/CSI/Person-in-WiFi-3D-repo/result/mmfi_d5_root_relative_distill
```

主要文件：

| 文件 | 说明 |
|---|---|
| `best_mpjpe_epoch_6.pth` | 当前 D5 最优 checkpoint |
| `epoch_2.pth` / `epoch_4.pth` / `epoch_6.pth` / `epoch_8.pth` / `epoch_10.pth` / `epoch_12.pth` | 按 interval 保存的 checkpoint |
| `latest.pth` | 指向 `epoch_12.pth` |
| `20260617_121741.log` | MMEngine/MMCV 文本日志 |
| `20260617_121741.log.json` | 结构化日志，可用于曲线和指标统计 |
| `train_seed42_20260617_121740.out` | 后台训练 stdout/stderr |
| `tf_logs/events.out.tfevents.1781669863.xl-4090-ubuntu.206522.0` | TensorBoard 标量事件文件 |
| `petr_mmfi_d5_root_relative_distill.py` | 实验运行时拷贝的配置文件 |

TensorBoard 目录：

```text
/home/xl/CSI/Person-in-WiFi-3D-repo/result/mmfi_d5_root_relative_distill/tf_logs
```

## 3. 数据与训练设置

D5 使用 MMFi 数据集：

```text
/home/xl/Downloads/MMFi_Dataset/MMFi_unzipped
```

数据设置：

| 项目 | 配置 |
|---|---|
| dataset | `opera.MMFiPoseDataset` |
| protocol | `protocol2` |
| split | `random_split` |
| random_ratio | `0.8` |
| split random_seed | `0` |
| train seed | `42` |
| CSI 输入 | amplitude + phase |
| normalize_csi | `True` |
| 3D label | `ground_truth.npy` 中的 17 点 3D 坐标 |
| teacher token | `data/mmfi_teacher_tokens/t0/train` |

训练设置：

| 项目 | D5 配置 |
|---|---|
| base config | `petr_mmfi_d4_decoder_coupled_distill.py` |
| work_dir | `result/mmfi_d5_root_relative_distill` |
| load_from | `result/mmfi_b0_3d_baseline/best_mpjpe_epoch_10.pth` |
| optimizer | AdamW |
| learning rate | `1e-5` |
| weight decay | `1e-4` |
| lr schedule | step at epoch 8 |
| max_epochs | `12` |
| batch size | `samples_per_gpu=32` |
| grad clip | `max_norm=0.1` |
| checkpoint interval | `2` |
| TensorBoard | 已开启 |

## 4. D5 模型改动

D5 继承 D4 的主路径耦合蒸馏，并新增 root decoupled 设计。

### 4.1 D4 已有内容

D4 相比 B0 增加：

1. 训练阶段加载 2D skeleton teacher token。
2. 在 PETR 主 decoder/refine token 上加入 token distillation。
3. 加入 skeleton relation distillation。
4. 加入 pelvis-relative 约束。
5. 加入 bone structure 约束。

D4 配置中的主要权重为：

```python
decoder_token_distill_weight = 0.2
decoder_relation_distill_weight = 0.05
rel_pose_loss_weight = 2.0
bone_loss_weight = 1.0
```

### 4.2 D5 新增内容

D5 将上述约束加大，并新增 root/axis 监督：

```python
root_decoupled = True
decoder_token_distill_weight = 1.0
decoder_relation_distill_weight = 0.1
rel_pose_loss_weight = 5.0
bone_loss_weight = 2.0
root_pose_loss_weight = 5.0
axis_pose_loss_weight = 2.0
axis_loss_weights = (2.0, 1.0, 2.0)
```

代码层面在 `opera/models/dense_heads/petr_head.py` 中新增：

1. `refine_root_branches`：从 refine decoder token 的全局均值预测 root offset。
2. `loss_root_gt`：预测 pelvis/root 与 GT pelvis/root 的 SmoothL1 监督。
3. `loss_axis_gt`：按 x/y/z 加权的 full pose SmoothL1 监督。
4. `root_abs_x/y/z` 和 `axis_abs_x/y/z`：训练时记录各轴绝对误差。

D5 的直觉是：如果 PA-MPJPE 低而 absolute MPJPE 高，说明相对姿态可学，但全局坐标弱。因此显式建模 root/global offset，并对 x/z 轴加权，可能帮助模型修正 absolute MPJPE。

## 5. TensorBoard 曲线与日志统计

TensorBoard 事件文件包含以下关键标量：

| 类型 | 标量 |
|---|---|
| train | `train/loss`, `train/loss_kpt`, `train/loss_decoder_token_distill`, `train/distill_main_token_cos` |
| train | `train/loss_root_gt`, `train/loss_axis_gt`, `train/root_abs_x/y/z`, `train/axis_abs_x/y/z` |
| train | `train/grad_norm`, `learning_rate` |
| val | `val/mpjpe`, `val/mpjpe_pelvis`, `val/pa_mpjpe`, `val/mpjpe_x/y/z` |

### 5.1 验证集结果

| epoch | MPJPE | PA-MPJPE | pelvis MPJPE | x | y | z |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 251.6635 | 99.4401 | 136.9300 | 172.1441 | 87.5073 | 98.0194 |
| 2 | 253.7042 | 99.5459 | 136.8144 | 173.5625 | 87.9327 | 99.3054 |
| 3 | 247.5801 | 99.8470 | 137.2964 | 171.0321 | 87.8995 | 92.5069 |
| 4 | 246.5838 | 99.4344 | 136.5015 | 173.3055 | 87.8396 | 90.6065 |
| 5 | 247.1361 | 99.8634 | 136.7043 | 171.6426 | 88.2614 | 91.5960 |
| 6 | **246.1187** | 99.6765 | **136.4558** | 171.2655 | 88.2415 | **89.6984** |
| 7 | 250.4184 | 99.5078 | 136.4908 | 173.7665 | 87.9187 | 97.1293 |
| 8 | 256.4032 | 99.5811 | 136.8722 | 171.4420 | 87.9305 | 108.1207 |
| 9 | 259.9484 | 99.6032 | 137.5129 | 171.3934 | 88.2521 | 113.6222 |
| 10 | 259.3190 | 99.5816 | 137.3325 | 172.4284 | 88.1330 | 111.2394 |
| 11 | 258.9976 | 99.6753 | 137.3032 | 171.9897 | 88.1772 | 111.2898 |
| 12 | 258.0081 | 99.5418 | 137.1917 | 171.6459 | 88.0171 | 110.0977 |

最优 checkpoint：

```text
best_mpjpe_epoch_6.pth
```

最优指标：

| 指标 | 数值 |
|---|---:|
| MPJPE | `246.1187` |
| PA-MPJPE | `99.6765` |
| pelvis MPJPE | `136.4558` |
| MPJPE x | `171.2655` |
| MPJPE y | `88.2415` |
| MPJPE z | `89.6984` |

验证集波动统计：

| 指标 | best | worst | range | std |
|---|---:|---:|---:|---:|
| MPJPE | 246.1187 | 259.9484 | 13.8297 | 5.1806 |
| PA-MPJPE | 99.4344 | 99.8634 | 0.4289 | 0.1322 |
| pelvis MPJPE | 136.4558 | 137.5129 | 1.0572 | 0.3545 |
| MPJPE x | 171.0321 | 173.7665 | 2.7344 | 0.8973 |
| MPJPE y | 87.5073 | 88.2614 | 0.7541 | 0.2108 |
| MPJPE z | 89.6984 | 113.6222 | 23.9238 | 8.7866 |

结论：验证集曲线的不稳定主要不是 PA-MPJPE、pelvis MPJPE 或 x/y 方向造成的，而是 z 方向造成的。epoch 6 之后，z 误差从 `89.70` 快速升到 `108-113`，直接把 MPJPE 拉高。

### 5.2 训练 loss 曲线

按 epoch 的训练均值如下：

| epoch | total loss | loss_kpt | token distill | token cos | root_abs_z | axis_abs_z | grad_norm |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 12.8370 | 2.0913 | 0.1089 | 0.8405 | 0.1206 | 0.1221 | 19.32 |
| 2 | 12.7342 | 2.0893 | 0.0342 | 0.9631 | 0.1204 | 0.1220 | 15.68 |
| 3 | 12.7414 | 2.0931 | 0.0262 | 0.9708 | 0.1207 | 0.1224 | 14.97 |
| 4 | 12.7465 | 2.0953 | 0.0217 | 0.9749 | 0.1209 | 0.1226 | 15.25 |
| 5 | 12.7687 | 2.1005 | 0.0188 | 0.9774 | 0.1215 | 0.1232 | 15.35 |
| 6 | 12.9528 | 2.1371 | 0.0170 | 0.9787 | 0.1252 | 0.1267 | 17.68 |
| 7 | 13.6042 | 2.2626 | 0.0167 | 0.9782 | 0.1353 | 0.1369 | 19.35 |
| 8 | 13.7744 | 2.2960 | 0.0146 | 0.9801 | 0.1399 | 0.1415 | 21.91 |
| 9 | 13.2638 | 2.1940 | 0.0135 | 0.9811 | 0.1314 | 0.1331 | 24.49 |
| 10 | 13.1454 | 2.1700 | 0.0134 | 0.9812 | 0.1291 | 0.1307 | 23.25 |
| 11 | 13.1278 | 2.1659 | 0.0133 | 0.9813 | 0.1284 | 0.1302 | 23.08 |
| 12 | 13.2223 | 2.1842 | 0.0132 | 0.9814 | 0.1304 | 0.1320 | 23.51 |

可以看到：

1. `loss_decoder_token_distill` 从 epoch 1 的 `0.1089` 快速降到 epoch 2 的 `0.0342`，最终到 `0.0132`。
2. `distill_main_token_cos` 从 `0.8405` 升到 `0.9814`，说明 teacher token 拟合很快饱和。
3. 但是主 3D loss 没有跟着下降。`loss_kpt` 从 `2.09` 附近上升到后期 `2.17-2.30`。
4. `root_abs_z` 和 `axis_abs_z` 在 epoch 6 之后明显变大，与验证集 z 方向退化一致。
5. `grad_norm` 后期升高，尤其 epoch 8 之后保持在 `22-24` 附近，说明优化状态并不稳定。

因此，用户观察到的“下降趋势随机、不稳定”是成立的。更精确地说：token distillation 曲线非常稳定地下降，但主任务 3D loss 和验证 MPJPE 没有稳定下降；验证 MPJPE 的随机性主要来自 z 方向绝对坐标。

## 6. 与其他实验对比

| 实验 | 输入/方案 | best epoch | best MPJPE | final MPJPE | PA-MPJPE | pelvis MPJPE | 主要观察 |
|---|---|---:|---:|---:|---:|---:|---|
| T0 teacher | 2D skeleton -> 3D | 72 | 168.7746 | 170.5309 | 32.4988 | 52.2501 | 2D skeleton teacher 对相对姿态很强 |
| B0 | CSI-only 3D baseline | 10 | **242.5610** | 255.4818 | 100.0573 | 137.4452 | 当前 D 系列必须超过的主要基线 |
| D1 Stage2 | 两阶段 token pretrain + 3D finetune | 6 | 240.2957 | 265.0620 | 100.7522 | 137.1818 | best 有偶然小幅改善，但 final 明显退化，不稳定 |
| D4 | 主 decoder/query 耦合蒸馏 | 1 | 248.4440 | 255.6820 | 99.6764 | 137.7437 | PA 有轻微改善，但 MPJPE 未超过 B0 |
| D5 | D4 + root/axis 约束 | 6 | 246.1187 | 258.0081 | 99.6765 | 136.4558 | pelvis/PA 略好，但 absolute MPJPE 未超过 B0 |

D5 相比 B0：

| 指标 | B0 best | D5 best | D5-B0 |
|---|---:|---:|---:|
| MPJPE | **242.5610** | 246.1187 | +3.5577 |
| PA-MPJPE | 100.0573 | **99.6765** | -0.3808 |
| pelvis MPJPE | 137.4452 | **136.4558** | -0.9894 |
| x | **169.0166** | 171.2655 | +2.2489 |
| y | **87.7563** | 88.2415 | +0.4852 |
| z | **87.9405** | 89.6984 | +1.7579 |

结论：D5 的 root/relative 设计确实让 pelvis MPJPE 和 PA-MPJPE 有小幅改善，但没有把这种改善转化为 absolute MPJPE 的提升。absolute MPJPE 仍然比 B0 差，主要差在 x 和 z。

## 7. 是否存在参数相关问题

存在较大可能。

### 7.1 蒸馏和结构损失权重提升过快

D4 的权重是：

```python
decoder_token_distill_weight = 0.2
decoder_relation_distill_weight = 0.05
rel_pose_loss_weight = 2.0
bone_loss_weight = 1.0
```

D5 变为：

```python
decoder_token_distill_weight = 1.0
decoder_relation_distill_weight = 0.1
rel_pose_loss_weight = 5.0
bone_loss_weight = 2.0
root_pose_loss_weight = 5.0
axis_pose_loss_weight = 2.0
axis_loss_weights = (2.0, 1.0, 2.0)
```

这相当于同时提高了 token、relative、bone、root、axis 多个约束。虽然日志中的 `loss_root_gt` 和 `loss_axis_gt` 数值看起来不大，但它们直接作用在主 decoder/refine 输出上，且与原有多层 `loss_kpt`、`loss_kpt_refine` 存在重复约束。多个目标可能在绝对坐标上相互拉扯，导致从 B0 最优点开始反而被破坏。

### 7.2 从 B0 best 继续训练，但没有保护原有解

D5 从：

```text
result/mmfi_b0_3d_baseline/best_mpjpe_epoch_10.pth
```

加载。这个 checkpoint 已经是 B0 的最好点。D5 的 root branch 虽然以接近零影响初始化，但训练时 backbone、stem、head、decoder 都会继续更新。结果是：

1. epoch 1-6 仍接近 B0；
2. epoch 6 达到 D5 best；
3. epoch 7 以后逐渐偏离；
4. epoch 8-12 的 z 方向显著退化。

这说明 D5 更像是在一个已经不错的 B0 解附近做不稳定 finetune，而不是稳定获得新知识。

### 7.3 token distillation 饱和太快，但没有带动 3D 主任务

`loss_decoder_token_distill` 很快下降，`distill_main_token_cos` 很快到 `0.98`。这说明模型能够拟合 teacher token，但这个 token 匹配目标没有持续推动 3D MPJPE 降低。

可能原因：

1. teacher token 主要编码 2D skeleton 的相对结构，而不是 CSI 可恢复的绝对 3D root。
2. token 对齐发生在表征空间，不保证几何输出更准确。
3. token loss 饱和后，后续训练由 3D loss/root/axis loss 主导，仍然可能向错误的 z 方向漂移。

### 7.4 z 方向是当前 D5 的主要不稳定来源

验证集：

```text
z best = 89.6984
z worst = 113.6222
range = 23.9238
```

而 x/y 的波动很小：

```text
x range = 2.7344
y range = 0.7541
```

训练集也显示：

```text
root_abs_z: epoch 1 = 0.1206, epoch 8 = 0.1399
axis_abs_z: epoch 1 = 0.1221, epoch 8 = 0.1415
```

这说明 D5 的不稳定不是整体姿态全面崩坏，而是 absolute depth/root-z 方向漂移。

### 7.5 learning rate schedule 没有解决退化

D5 使用 `lr=1e-5`，epoch 8 step 到 `1e-6`。但是退化从 epoch 7-8 已经开始，step 后也没有恢复 best。说明问题不是单纯学习率过大，而是优化目标或结构设计本身容易把模型推离 B0 的好解。

## 8. 当前结论

1. D5 训练正常完成，TensorBoard、日志、checkpoint 都完整保存。
2. D5 best MPJPE = `246.1187 @ epoch 6`，没有超过 B0 的 `242.5610 @ epoch 10`。
3. D5 的 pelvis MPJPE 和 PA-MPJPE 略好于 B0，说明相对姿态和 root-relative 结构约束有一定作用。
4. D5 没有解决 absolute MPJPE，主要原因是 x/z 绝对坐标仍然弱，其中 z 方向后期明显漂移。
5. token 蒸馏本身已经被模型快速拟合，但没有稳定转化成 3D 主任务收益。
6. 当前 D5 的参数组合偏激进，尤其是在 B0 best checkpoint 上继续全模型 finetune，容易破坏原有解。

## 9. 下一步全局改进建议

### 9.1 先做稳定化版本 D5-lite

目标不是继续加复杂结构，而是验证 D5 方向本身是否有效。

建议配置：

```python
load_from = 'result/mmfi_b0_3d_baseline/best_mpjpe_epoch_10.pth'
lr = 2e-6 或 5e-6
max_epochs = 6 或 8
lr_config = 不在 epoch 8 后继续长训，或者直接不 step
decoder_token_distill_weight = 0.1 或 0.2
decoder_relation_distill_weight = 0.05
rel_pose_loss_weight = 2.0
bone_loss_weight = 1.0
root_pose_loss_weight = 1.0 或 2.0
axis_pose_loss_weight = 0.5 或 1.0
axis_loss_weights = (1.0, 1.0, 1.5) 或只增强 z
```

理由：

1. D5 已经证明高权重版本会漂移。
2. 最优点出现在 epoch 6，因此不需要长训。
3. 先保住 B0 的 absolute MPJPE，再观察 PA/pelvis 是否能小幅改善。

### 9.2 使用分阶段解冻，而不是全模型直接 finetune

建议下一版训练策略：

1. Stage A：冻结 backbone、stem、主 decoder，只训练新增 root branch / 小 adapter，训练 1-2 epoch。
2. Stage B：解冻 head/refine decoder，保持 backbone/stem 冻结，训练 4-6 epoch。
3. Stage C：如有必要，用很小 lr 解冻全模型。

理由：

当前 D5 从 B0 best 出发，但全模型都被新 loss 更新，导致好解被破坏。新增 root branch 应该先学会做 residual correction，而不是让整个网络一起移动。

### 9.3 把 root 和 relative pose 彻底解耦

当前 D5 的 root offset 是从 refine token 均值得到，然后作用回整个人体坐标。这仍然比较粗糙。

更推荐的主体结构：

```text
CSI stem/backbone
    -> relative pose head: 输出 pelvis-centered 17x3
    -> root head: 输出 1x3 global root
    -> final pose = relative pose + root
```

损失：

```text
L = L_abs(final_pose, gt_3d)
  + lambda_rel * L_rel(pred_rel, gt_rel)
  + lambda_root * L_root(pred_root, gt_root)
  + lambda_bone * L_bone(pred_rel, gt_rel)
  + lambda_distill * L_teacher_relation
```

这样可以明确区分：

1. teacher skeleton 适合监督 relative pose / bone / relation；
2. 3D GT 负责监督 root/global position；
3. 不让 teacher 的 2D 深度歧义污染 absolute 3D。

### 9.4 蒸馏只约束相对结构，不直接强推 absolute token

当前结果说明 teacher token 很容易被拟合，但对 MPJPE 没有稳定帮助。后续蒸馏建议改成更几何明确的目标：

1. pairwise joint distance matrix；
2. bone direction / bone length ratio；
3. pelvis-centered joint relation；
4. temporal pose consistency；
5. action-level skeleton dynamics token。

这些目标和 2D skeleton teacher 的优势一致：相对姿态强、绝对深度弱。

不建议直接蒸馏 teacher 的 absolute 3D prediction，因为 teacher 的 absolute MPJPE 仍明显高于 PA-MPJPE，会把 2D 深度歧义传给 student。

### 9.5 针对 z 方向做专门诊断

下一步需要单独保存预测结果，做以下分析：

1. 每个 epoch 的 `pred_root_z - gt_root_z` 均值和方差。
2. 每个 action/subject 的 z 误差分布。
3. z 误差是否存在系统性 bias。
4. z 误差是否集中在某些动作或距离段。
5. root_z 与 CSI 能量/phase 特征是否相关。

如果发现 z 是系统性偏移，可以增加一个校正 head；如果是动作相关，可能需要 temporal modeling；如果是 subject/domain 相关，则需要归一化或 domain adaptation。

### 9.6 回到输入和预处理层面

当前 D 系列主要改蒸馏和 head，但 MMFi 的 CSI 输入和 Person-in-WiFi 3D 的预处理逻辑不同。由于 z 方向不稳定，下一步应重新审视：

1. CSI amplitude/phase 是否都应该使用。
2. phase 是否需要更强的 unwrap/filter。
3. 是否应该使用 temporal window，而不是单帧 CSI。
4. 是否应该借鉴 Person-in-WiFi 3D 的时间聚合方式。
5. stem 是否需要从 `conv2d_3layer` 改成显式时序建模的 TCN/Transformer stem。

如果 CSI 的瞬时输入本身对 depth/root 信息不足，仅靠 head/loss 很难稳定提升 absolute MPJPE。

### 9.7 推荐下一轮实验顺序

推荐按以下顺序推进：

| 编号 | 实验 | 目的 |
|---|---|---|
| D5-lite | 降低 D5 权重 + 小 lr + 6/8 epoch | 验证 D5 是否只是参数过激 |
| D6-freeze | 只训练 root branch/adapter，再逐步解冻 | 防止破坏 B0 best |
| D7-root-relative | 明确拆成 root head + relative pose head | 从结构上解决 PA 到 MPJPE 转换问题 |
| D8-relation-only-distill | 只蒸馏 skeleton relation，不蒸馏 absolute token | 避免 teacher 深度歧义 |
| D9-temporal-stem | 引入 temporal CSI window/stem | 针对 z/root 不稳定 |

短期最值得做的是 D5-lite 和 D6-freeze，因为改动小、风险低、可以快速判断 D5 的失败是否主要来自参数和 finetune 策略。

中期更值得投入的是 D7-root-relative，因为当前所有结果都指向一个核心问题：模型已经能学到一定相对人体结构，但 absolute root/depth 不稳定。主体网络需要把这两个子问题显式拆开，而不是继续把所有监督压到同一个 17x3 输出上。

