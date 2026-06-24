# MMFi D7-root-relative 实验方案

日期：2026-06-24

分支：`mmfi_d7_root_relative_heads`

配置：`configs/mmfi/petr_mmfi_d7_root_relative_heads.py`

目标结果目录：`result/mmfi_d7_root_relative_heads`

## 1. 实验动机

前几轮 MMFi 3D 实验里，一个比较稳定的现象是：PA-MPJPE 明显低于 MPJPE，但 MPJPE 很难下降。这个现象说明模型对人体相对形状、骨架结构有一定学习能力，但绝对空间位置，尤其是 root/pelvis 的估计仍然不稳定。

因此 D7 的核心目标不是继续增强 teacher token 蒸馏，而是从主网络输出结构上把 3D 预测拆成两个部分：

```text
3D pose = root position + root-relative pose
```

这样做的目的，是让网络分别学习：

- `root head`：人体在真实 3D 坐标系中的整体位置，直接影响 MPJPE。
- `relative pose head`：以 pelvis/root 为中心的人体相对骨架，主要对应 PA-MPJPE 中体现出的形状能力。

如果这个拆分有效，理论上应该优先改善 `root_abs_x/y/z`、`axis_abs_x/y/z`，并最终降低 MPJPE。

## 2. 之前结果对照

| 实验 | 设计 | 最好结果 |
| --- | --- | --- |
| B0 baseline | MMFi 3D 基线，直接 CSI -> 17 点 3D | MPJPE 242.5610，PA-MPJPE 100.0573，pelvis 137.4452，x 169.0166，y 87.7563，z 87.9405 |
| D5-lite | 轻量 root/relative loss + 弱 token 蒸馏 | MPJPE 244.8662，PA-MPJPE 99.4802，pelvis 136.4630，x 172.5732，y 87.6556，z 87.6927 |
| D6-freeze | 冻结主网络，只训练 root adapter | MPJPE 247.7970，PA-MPJPE 100.0561，pelvis 137.5631，x 171.8505，y 87.7175，z 96.5621 |

D6 的结果说明：只训练一个 root offset adapter 不足以修正 MPJPE，甚至会破坏 z 方向。因此 D7 不再冻结主网络，而是在 refine 输出处重构 root/relative 的生成方式。

## 3. D7 网络结构

D7 修改位置在 `opera/models/dense_heads/petr_head.py` 的 `PETRHead.forward_refine()`。

原 refine keypoint branch 仍然输出每个关键点的临时 3D 坐标：

```text
tmp_pose = refine_kpt_branch(refine_tokens) + reference_points
```

然后计算 pelvis/root：

```text
root_base = pelvis(tmp_pose)
```

相对姿态分支由 `tmp_pose` 去除 root 得到：

```text
relative_pose = tmp_pose - root_base
```

root 分支使用 refine token 的平均池化结果预测 root residual：

```text
root_offset = root_head(mean(refine_tokens))
root_pred = stopgrad(root_base) + root_offset
```

最终输出为：

```text
pose_pred = relative_pose + root_pred
```

这里使用 `stopgrad(root_base)` 的原因是保持从 B0 checkpoint 初始化时的稳定性：初始 `root_offset = 0`，所以数值上接近 B0；但训练时 root 误差的梯度主要更新 `root_head`，相对姿态误差主要更新 keypoint branch。这样比 D6 的单纯全局 offset 更明确地拆开了 root 与 body shape。

## 4. 和 D6 的关键区别

D6 的形式接近：

```text
pose = tmp_pose + root_offset
```

它只是在整个人体上加一个整体平移量，而且只训练 root adapter。这个设计很难修正相对姿态和 root 之间的耦合错误。

D7 的形式是：

```text
pose = relative_pose + root_pred
```

并且训练所有相关参数。它的重点是让网络在结构和梯度路径上区分：

- 人体形状是否正确。
- pelvis/root 的绝对位置是否正确。
- x/y/z 哪个方向是主要误差来源。

## 5. Loss 设计

D7 暂时关闭 teacher token 蒸馏：

```text
decoder_token_distill_weight = 0.0
decoder_relation_distill_weight = 0.0
```

原因是这一轮实验要先验证结构本身。如果同时加入 token 蒸馏，很难判断改善或退化来自结构，还是来自 teacher token。

保留和加强的监督包括：

```text
L_main = L_kpt_refine
L_rel  = SmoothL1(pred - pelvis(pred), gt - pelvis(gt))
L_root = SmoothL1(pelvis(pred), pelvis(gt))
L_axis = axis-weighted SmoothL1(pred, gt)
L_bone = bone direction/length regularization
```

当前权重：

```text
rel_pose_loss_weight = 3.0
bone_loss_weight = 1.0
root_pose_loss_weight = 3.0
axis_pose_loss_weight = 1.0
axis_loss_weights = (2.0, 1.0, 1.5)
```

x 方向权重设置为 2.0，是因为此前 B0、D5、D6 中 x 方向误差长期最大，root/pelvis 的横向位置估计可能是 MPJPE 的主要瓶颈之一。

## 6. 训练设置

数据集：MMFi，沿用当前 MMFi 3D protocol2/random split 设置。

初始化：`result/mmfi_b0_3d_baseline/best_mpjpe_epoch_10.pth`

训练轮数：8 epoch。

学习率：`5e-6`。

随机种子：`42`。

TensorBoard：开启，沿用已有 `TensorboardLoggerHook`。

checkpoint：每 1 epoch 保存一次，最多保留 8 个。

## 7. 重点观察指标

训练过程中重点看以下指标：

- `mpjpe`：最终目标，必须优先看。
- `pa_mpjpe`：判断相对骨架能力是否被破坏。
- `pelvis_mpjpe`：判断 root 估计是否真正改善。
- `mpjpe_x / mpjpe_y / mpjpe_z`：判断 x 方向是否仍是主要瓶颈。
- `root_abs_x / root_abs_y / root_abs_z`：root head 是否学到绝对位置修正。
- `rel_abs_x / rel_abs_y / rel_abs_z`：relative pose head 是否保持稳定。
- `axis_abs_x / axis_abs_y / axis_abs_z`：整体轴向误差趋势。

## 8. 成功标准

D7 至少需要满足：

1. MPJPE 优于 B0 的 242.5610。
2. pelvis/root 相关指标下降。
3. x 方向误差下降，不能只靠 PA-MPJPE 微小变化。
4. PA-MPJPE 不明显劣化，说明相对骨架没有被 root 结构破坏。

如果 PA-MPJPE 下降但 MPJPE 不降，说明相对结构仍然有效，但 root head 没有学到真实绝对位置，需要继续改 root 表征或加入更强的全局定位先验。

## 9. 主要风险

1. `root_pred` 仍然依赖 `root_base` 作为稳定锚点，严格来说不是完全从 token 独立预测绝对 root。这是为了兼容 B0 checkpoint，避免训练初期直接崩掉。
2. 如果 MMFi 的 CSI 对绝对 x 方向定位信息不足，结构拆分也可能无法显著降低 MPJPE。
3. 当前仍沿用 D5-lite 的数据入口，因此会读取 teacher token，但蒸馏损失权重为 0。本轮结论按结构验证解释，不按蒸馏实验解释。
4. 如果 root loss 权重过大，可能会牺牲相对骨架；如果权重过小，则可能退化成 B0 类似结果。

## 10. 下一步可能方向

如果 D7 有改善：

- 在 D7 基础上重新加入轻量 token/relation distillation。
- 单独调 `root_pose_loss_weight` 和 `axis_loss_weights`。
- 进一步设计 root-aware transformer query，让 root query 和 joint query 显式交互。

如果 D7 没有改善：

- 说明 PA 到 MPJPE 的差距主要不是 head 结构问题，而可能是输入 CSI 表征缺少稳定全局定位信息。
- 下一步应优先考虑预处理和 stem，比如 SDP image-like 表征、相位/幅度组合、时间窗口增强。
- 也可以考虑显式预测 root velocity 或 trajectory，而不是每帧独立预测 root。
