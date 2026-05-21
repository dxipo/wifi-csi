# WiFi CSI 实验对比与 MPJPE 评估复核

日期：2026-05-18

本文基于以下内容整理：

- 原报告：`docs/wifi_csi_experiment_report_20260512.md`
- 已完成训练记录：`result/` 下各实验目录和日志
- 当前评估代码：`opera/datasets/wifi_pose.py`

核心结论先写在前面：

1. 原报告表格只能作为“探索记录”，不能直接作为严格 ablation 对比。
2. 目前最有价值的强基线仍是 `SDP_offline_power_xfall`，但它和 image-like / translator / distill 之间的比较仍有混杂因素。
3. 当前 MPJPE 计算不是严格 top-1 指标，而是 oracle best-of-query 指标，会使用 GT 从多个 query 中挑最接近的预测。
4. 后续要补的关键实验不是继续堆新方法，而是先把预处理、网络结构、蒸馏、训练策略、评估方式逐项拆开。

## 1. 原表格的主要问题

原表格中 E0-E6 的结果来自真实日志，但它们不能直接支持“某个改动带来了提升/下降”的强结论，原因是多个实验同时改变了不止一个因素。

### 1.1 E3 和 E5 不能直接比较预处理方式

E3：

- 预处理：`SDP_offline_power_xfall`
- 输入 layout：`wtn`
- 模型：baseline PETR 输入适配
- `num_query=81`
- 没有 translator

E5：

- 预处理：`SDP140 image-like`
- 输入 layout：`hwc`
- 模型：新增 `SDPToImageLikeTranslator`
- `num_query=100`
- 先将 SDP 特征转成类似 `(3,360,640)` 的图像特征再进入 patch embedding

因此 E3 vs E5 同时改变了：

- 预处理方法
- 输入形状和 layout
- 模型前端结构
- query 数量
- 特征空间是否经过 translator

所以不能说 E5 比 E3 差就是 “SDP140 image-like 预处理差”，也不能说 translator 一定无效。它只能说明“当前这套 SDP140 image-like + translator 总组合没有超过 E3 总组合”。

### 1.2 E4 和 E5 也不能只解释为 SDP300 vs SDP140

E4：

- `SDP300 image-like`
- `conv_patch`
- 样本 shape 约 `(30,825,3)`

E5：

- `SDP140 image-like`
- `SDPToImageLikeTranslator`
- 样本 shape 约 `(30,270,3)`

两者同时改变了上下文长度、输出宽度、网络前端和 patch 化方式。它们不是单纯的 SDP300 vs SDP140 对比。

### 1.3 E5 和 E6 才接近蒸馏对比，但仍有训练策略混杂

E6 从 E5 checkpoint 初始化并加入 `gt_token` 和 `loss_token`。这更接近 stage2 distillation 对照。

但 E6 的问题是：

- 它从已训练的 E5 checkpoint 继续跑 450 epoch。
- `load_from` 只加载模型权重，不恢复优化器状态和 epoch 进度。
- 因此 stage2 不是从 E5 的 late-stage 训练状态平滑继续，而是用新的 optimizer / scheduler 重新训练一大段。

所以 E6 变差不能直接归因于 token 蒸馏本身，也可能是高学习率、长时间 fine-tune 或 checkpoint 选择造成的漂移。

### 1.4 best val 不能等价于严格 test 结果

当前配置中 per-epoch evaluation 使用的是 test 数据目录。如果从 test 上挑选 best epoch，那么 test 实际上被当成 validation 使用，best val 会偏乐观。

更严谨的流程应该是：

1. train 用于训练。
2. val 用于选 best checkpoint。
3. test 只在最终模型确定后评估一次。

当前结果可以作为探索趋势，但不适合直接写成最终严格结论。

## 2. 应补充的对照实验

后续补实验的原则：每次只改变一个主要因素。

### 2.1 预处理方式对比

目标：判断 raw / STFT / SDP / SDP power 到底谁更好。

需要固定：

- 同一 train/test split
- 同一模型主干
- 同一 `num_query`
- 同一 batch size
- 同一 epoch / lr schedule
- 同一 MPJPE 计算方法
- 不使用 translator
- 不使用 distillation

建议补充：

| 实验目的 | 固定条件 | 需要补的实验 |
| --- | --- | --- |
| raw vs STFT vs SDP vs SDP power | baseline PETR，batch 32，统一 `num_query` | 重新跑 raw、STFT offline、SDP offline、SDP power_xfall 的同条件版本 |
| isolate power 作用 | 只变 power 处理 | `SDP_offline` vs `SDP_offline_power_xfall`，其他完全一致 |
| isolate query 数量 | 只变 `num_query` | `SDP_power_xfall num_query=81` vs `num_query=100` |

当前 E3 用 `num_query=81`，其他很多实验用 `num_query=100`。如果要比较预处理，建议统一为 `num_query=100` 或统一为 `81`，不要混用。

### 2.2 image-like 预处理对比

目标：判断 image-like 预处理本身是否有效，而不是 translator 是否有效。

需要补充：

| 实验目的 | 已有实验 | 缺失实验 |
| --- | --- | --- |
| SDP140 image-like 本身是否好 | E5: SDP140 + translator | SDP140 + `conv_patch`，不加 translator |
| translator 是否有贡献 | E5: SDP140 + translator | SDP140 + `conv_patch` 对照 |
| SDP300 vs SDP140 是否有差异 | E4: SDP300 + `conv_patch`，E5: SDP140 + translator | SDP300 + translator，SDP140 + `conv_patch` |
| 上下文长度影响 | SDP300 使用 1000 ms，SDP140 使用约 467 ms | 同一前端下比较 SDP300 vs SDP140 |
| lag 排布影响 | 当前主要是 lagwindow | 同样参数下比较 lagwindow vs windowlag |

在这些实验补齐前，不能把 E3/E4/E5 的差异直接归因于预处理方式。

### 2.3 translator 结构对比

目标：判断 `SDPToImageLikeTranslator` 是否真的提供增益。

建议固定同一个 SDP140 数据集，比较：

| 实验 | 输入 | 前端 | 目的 |
| --- | --- | --- | --- |
| T0 | SDP140 image-like | `conv_patch` | 不使用 translator 的基线 |
| T1 | SDP140 image-like | `SDPToImageLikeTranslator` | 当前 translator 路线 |
| T2 | SDP140 image-like | 更轻量 translator | 检查是否结构过重或过拟合 |
| T3 | SDP140 image-like | translator + 中间约束 | 检查仅靠 pose loss 是否不够 |

当前 translator 没有真实图像重建监督，也没有 teacher image feature 监督；它只是被 pose loss 间接约束。因此它输出的 `(3,360,640)` 不一定真的具备稳定 image-like 语义。

### 2.4 stage2 distillation 对比

目标：判断 token 蒸馏是否有效。

必须先补一个关键 control：

| 实验 | 初始化 | token loss | lr / epoch | 目的 |
| --- | --- | --- | --- | --- |
| D0 | baseline checkpoint | 关闭 | 与 D1 完全相同 | 纯 fine-tune control |
| D1 | baseline checkpoint | 开启，权重 0.03 | 当前 v2 设置 | 判断 token 是否相对 D0 有增益 |
| D2 | baseline checkpoint | 权重 0.01 / 0.1 | 同 D1 | 检查权重敏感性 |
| D3 | baseline checkpoint | token 先 L2Norm/LayerNorm | 同 D1 | 检查 token scale/分布问题 |
| D4 | best 附近 checkpoint | 开启 | 同 D1 | 检查从 final checkpoint 启动是否限制效果 |

当前 `47` 号实验只能说明：从 `result/42/latest.pth` 继续做低学习率 token stage2，best 能到 `87.8896`，但不能证明 token 比普通 fine-tune 有效，因为缺少 D0。

### 2.5 评估方式对比

必须补充严格 top-1 / top-k 评估：

| 指标 | 含义 | 用途 |
| --- | --- | --- |
| top-1 MPJPE | 按分类分数最高的 query 计算 | 最接近部署 |
| top-k oracle MPJPE | 在分数前 k 个 query 中用 GT 挑最接近 | 看候选质量 |
| best-of-all oracle MPJPE | 当前近似做法 | 只看模型是否生成过正确候选 |
| final test MPJPE | best checkpoint 确定后只测一次 | 严格汇报 |

当前日志里的 MPJPE 更接近 best-of-query oracle，不能作为严格 top-1 结果。

## 3. 已完成 result 记录汇总

下面只记录跑到配置 `max_epochs` 的实验。测试阶段只跑了一小段就暂停的目录不作为主结果。

### 3.1 当前主要可读实验

| 编号 | result 目录 | 主要设置 | max epoch | final MPJPE | best MPJPE | 备注 |
| --- | --- | --- | ---: | ---: | ---: | --- |
| R0 | `42_wifipose_2d_baseline_test_recover` | raw hold-out baseline，`num_query=100` | 450 | 175.3183 / x 114.1612 / y 98.6775 | 59.5218 @ 4 | best 过早，oracle/test 选择偏乐观。 |
| R1 | `42_wifipose_2d_baseline_STFT_offline` | STFT offline，`input_dim=9`，`num_query=100` | 450 | 121.5475 / x 103.9709 / y 42.5375 | 85.1846 @ 3 | final 比 raw 好，但 best 极早。 |
| R2 | `42_wifipose_2d_baseline_SDP_offline` | SDP offline，`input_dim=6`，`num_query=100` | 450 | 302.5016 / x 260.4266 / y 126.3197 | 84.3267 @ 235 | 后期严重退化。 |
| R3 | `42_wifipose_2d_baseline_SDP_offline_power_xfall` | SDP + power_xfall，`input_dim=6`，`num_query=81` | 450 | 98.5397 / x 73.0389 / y 45.5856 | 77.6551 @ 391 | 当前最强主线基线。 |
| R4 | `43_wifipose_2d_baseline_SDP300_imagelike_lagwindow_centerctx` | SDP300 image-like，`conv_patch`，shape 约 `(30,825,3)` | 450 | 244.9251 / x 218.6288 / y 66.9030 | 84.9851 @ 264 | best 可用，final 退化明显。 |
| R5 | `44_wifipose_2d_baseline_SDP140_translator_image_lagwindow_centerctx` | SDP140 image-like + translator，shape 约 `(30,270,3)` | 450 | 112.6839 / x 98.2272 / y 35.1492 | 101.3820 @ 106 | 比 R4 final 稳定，但不能证明预处理优劣。 |
| R6 | `45_wifipose_2d_SDP140_image-like_stage2_distill_lagwindow_centerctx` | R5 基础上 stage2 token distill | 450 | 140.4951 / x 101.7064 / y 69.0647 | 102.5173 @ 29 | 蒸馏未带来可见增益。 |
| R7 | `46_wifipose_2d_baseline_SDP_offline_power_xfall_stage2_distill` | R3 基础上 stage2 token distill，450 epoch | 450 | 150.1127 / x 113.3297 / y 78.0895 | 81.7492 @ 57 | 早期接近 R3，但长训练漂移严重。 |
| R8 | `47_wifipose_2d_baseline_SDP_offline_power_xfall_stage2_distill_finetune_v2` | R3 基础上低 LR stage2，100 epoch，`loss_token=0.03` | 100 | 117.3972 / x 101.0378 / y 42.4774 | 87.8896 @ 22 | v2 比 R7 final 稳定，但仍未超过 R3 best。 |

### 3.2 早期历史实验：记录但不纳入主对比

这些实验跑完整了，但由于代码版本、评估尺度、batch size 或数据处理方式明显不同，不建议和 R0-R8 放在同一张严格对比表里。

| result 目录 | 主要特征 | final MPJPE | best MPJPE | 为什么不作为主对比 |
| --- | --- | ---: | ---: | --- |
| `wifipose_2d_baseline` | raw baseline，batch 256 | 0.4794 | 0.1312 @ 51 | 数值明显是 normalized 量级，不是当前 pixel MPJPE。 |
| `wifipose_2d_baseline_test` | raw baseline，batch 32 | 86.2360 | 48.5507 @ 10 | 早期代码/评估流程不同，best 过早。 |
| `wifipose_2d_baseline_STFT_1` | STFT，在线或旧版 STFT 路线 | 227.7167 | 86.1785 @ 166 | 与 offline STFT 版本不同。 |
| `wifipose_4_2d` | 早期 2D baseline，batch 256 | 26.8317 | 24.2132 @ 388 | 结果尺度和当前 pixel MPJPE 不一致，不能混用。 |

### 3.3 未纳入主结果的暂停/不完整记录

以下目录存在日志，但没有跑到配置的 `max_epochs`，或没有可用 MPJPE 记录：

- `wifipose_2d_baseline_STFT`：最后只到 epoch 27。
- `wifipose_2d_baseline_test_recover`：最后只到 epoch 215。
- `wifipose_4_2d_test2`：最后只到 epoch 2。
- `wifipose`、`wifipose_1`、`wifipose_2`、`wifipose_3`、`wifipose_4_2d_test1`：未找到可用于当前表格的完整 MPJPE 记录。

## 4. 结果态势解释

### 4.1 `SDP_offline_power_xfall` 目前最值得保留

R3 是当前最强主线：

- best MPJPE：`77.6551 @ epoch 391`
- final MPJPE：`98.5397 @ epoch 450`

可能原因：

1. power response 更直接反映人体运动对 CSI 能量/幅度结构的扰动。
2. 相比纯 SDP，power_xfall 可能降低了相位噪声或无关多径变化的影响。
3. 该表示仍然保持了相对紧凑的输入结构，没有像 image-like 版本那样把特征展开到很宽的伪图像。

但 R3 使用 `num_query=81`，不是所有实验都一致，因此它是当前强基线，但还不是完全公平对比中的最终结论。

### 4.2 纯 SDP 的后期退化非常严重

R2 的 best 不差：

- best MPJPE：`84.3267 @ epoch 235`

但 final 很差：

- final MPJPE：`302.5016`

从 x/y 看，主要是 x 方向误差变大：

- final x：`260.4266`
- final y：`126.3197`

这说明纯 SDP 表达在训练中可能学到了一些可用候选，但后期表示不稳定，模型可能过拟合到错误的横向位置模式。

### 4.3 SDP300 image-like 不是无效，而是不稳定

R4 的 best 为 `84.9851`，和 R2/R1 接近，说明 SDP300 image-like 可以产生有效候选。

但 final 为 `244.9251`，退化明显。可能原因：

1. `(30,825,3)` 的宽度很长，lagwindow flatten 后的局部邻接关系不一定符合图像卷积假设。
2. `conv_patch` 直接处理伪图像，缺少针对 lag/time/subcarrier 结构的显式建模。
3. 训练没有保存 best checkpoint，final 不能代表该方法的最佳能力。

### 4.4 SDP140 + translator 稳定性较好，但效果未超过强基线

R5 final 为 `112.6839`，比 R4 final 好很多，但 best 只有 `101.3820`，低于 R3。

解释：

1. translator 可能缓解了直接 `conv_patch` 处理伪图像的训练不稳定。
2. 但 translator 增加了一个从 CSI 特征到 image-like 特征的映射瓶颈。
3. 该映射没有真实图像或视觉特征监督，只靠 pose loss 反向约束，输出是否具备可解释图像语义并不确定。

因此现在只能说“translator 路线可训练”，不能说“image-like 预处理优于 SDP power_xfall”。

### 4.5 stage2 distillation 目前没有证明有效

R7：

- best：`81.7492 @ epoch 57`
- final：`150.1127`

R8：

- best：`87.8896 @ epoch 22`
- final：`117.3972`

R8 比 R7 明显稳定，说明低学习率、短 schedule、较小 token loss 权重是正确方向。但它仍未超过 R3 best `77.6551`。

关键问题：

1. R8 是从 `result/42/latest.pth` 启动，也就是 epoch 450 final checkpoint，而不是 R3 的 best epoch 391 checkpoint。
2. 缺少同样 schedule 下的 no-token fine-tune control，所以不能判断 R8 的改善来自 token 蒸馏，还是来自低学习率继续训练本身。
3. `gt_token` 可能是视觉或跨模态 teacher 的全局表示，它与 2D pose 回归目标未必完全对齐。
4. token loss 作用在 shared representation 上，可能与 keypoint loss 产生梯度冲突。

## 5. 当前 MPJPE 计算方法复核

当前实现位于 `opera/datasets/wifi_pose.py`。

### 5.1 它现在实际计算的是什么

`evaluate()` 中每个样本会取：

- GT：`gt_keypoints`
- 预测：`det_keypoints[0]`

然后调用：

```python
mpjpe_2d, mpjpex, mpjpey, _ = self.calc_mpjpe(
    gt_keypoints, kpt_pred, data_name, root=[5, 7]
)
```

`calc_mpjpe()` 的核心逻辑是：

1. 取 GT 和 pred 的 x/y。
2. 对每个 GT person 和每个 predicted query 计算 14 个关节的平均 L2 距离。
3. 用 greedy matching 选择距离最小的 predicted query。
4. 将 normalized 坐标误差乘以 `W=640, H=360`，得到 pixel-space MPJPE。

单人场景下，这等价于：

> 在所有 predicted query 中，用 GT 选择一个最接近 GT 的 query，然后计算这个 query 的 MPJPE。

这不是严格 top-1，而是 oracle best-of-query。

### 5.2 存在的问题

#### 问题 1：使用 GT 选择预测 query

这是最大的问题。

当前 `match_thr = 1e9`，实际没有阈值限制。单人场景下，只要模型 81/100 个 query 中有一个比较接近 GT，MPJPE 就会很好，即使模型自己并不知道应该输出哪一个 query。

这会导致：

- MPJPE 偏乐观。
- query 数量越多，越可能碰到一个接近 GT 的候选。
- `num_query=81` 和 `num_query=100` 之间也会影响指标公平性。
- 该指标不能代表部署时模型按最高置信度输出的误差。

#### 问题 2：没有使用分类分数或 bbox 分数选 top-1

`evaluate()` 里 `det_bboxes` 被取出来了，但没有用于选择预测。严格评估应该按模型分数选择 top-1，或者至少报告 top-k oracle。

#### 问题 3：在 test 集上选 best epoch 会造成偏乐观

如果每个 epoch 都在 test 集上评估，再从 test 里挑 best epoch，就会把 test 用成 validation。

这不是代码计算错误，但会造成实验汇报上的偏差。

#### 问题 4：硬编码图像大小

`calc_mpjpe()` 中写死：

```python
W, H = 640.0, 360.0
```

如果所有 GT 都来自 640x360 图像，这没有问题。但如果后续换数据、换裁剪、换 resize 或使用不同标注尺度，MPJPE 会直接错。

更稳妥的做法是从 `img_shape` 或 dataset metadata 读取。

#### 问题 5：GT 被 clamp 到 `[0,1]`

`get_item_single_frame()` 中会把 GT keypoint 除以 `W/H` 后 clamp 到 `[0,1]`。

如果存在越界点或无效点，clamp 会把它们压回边界，可能降低或扭曲误差。

#### 问题 6：没有使用 visibility mask

GT 的第三维可见性/置信度被保留为 `keypoint_v`，但 `calc_mpjpe()` 只用 `[..., :2]`，没有根据可见性过滤关节。

如果某些关节不可见、置信度为 0 或标注异常，当前 MPJPE 仍会把它们纳入平均。

#### 问题 7：不同历史实验的 MPJPE 尺度不完全一致

例如 `wifipose_2d_baseline` 的 final MPJPE 为 `0.4794`，明显不是当前 pixel-space MPJPE。说明早期实验很可能使用 normalized-space MPJPE 或旧版计算逻辑。

因此旧实验不能直接和当前 R0-R8 混在一起比较。

### 5.3 应该如何改进 MPJPE 评估

建议新增多套指标，不要只保留当前一种：

| 指标 | 选择预测方式 | 作用 |
| --- | --- | --- |
| `mpjpe_top1` | 按分类/检测分数最高 query | 主指标 |
| `mpjpe_top5_oracle` | 分数前 5 个 query 中用 GT 选最近 | 辅助分析候选质量 |
| `mpjpe_top20_oracle` | 分数前 20 个 query 中用 GT 选最近 | 辅助分析候选池 |
| `mpjpe_oracle_all` | 所有 query 中用 GT 选最近 | 保留当前指标，作为上界 |

同时建议：

1. 用 validation 集选 best checkpoint。
2. 固定 best checkpoint 后，在 test 集只评估一次。
3. MPJPE 使用 `img_shape`，不要硬编码 `640/360`。
4. 加入 visibility mask。
5. 同时汇报 final checkpoint 和 val-best checkpoint。
6. 比较不同 `num_query` 时要报告 oracle 指标对 query 数量的敏感性。

## 6. 建议的下一步最小实验集

如果目标是尽快得到可解释的结论，我建议按下面顺序补。

### 6.1 第一组：修正评估

先不改模型，新增 top-1 / top-k / oracle-all 评估。

用已有 checkpoint 先评估：

- R3 `SDP_offline_power_xfall`
- R5 `SDP140 + translator`
- R8 `SDP_power_xfall + stage2 v2`

目的：确认当前 best-of-query 指标和严格 top-1 指标差距有多大。

### 6.2 第二组：蒸馏 control

在 R8 基础上补：

- 同样 `load_from`
- 同样 `lr=2e-6`
- 同样 `max_epochs=100`
- 同样 batch 32
- 关闭 `loss_token`

这能回答：R8 的改善到底来自 fine-tune，还是 token 蒸馏。

### 6.3 第三组：SDP140 translator ablation

补：

- SDP140 image-like + `conv_patch`
- SDP140 image-like + translator

保持其他条件一致。这样才能判断 translator 是否有贡献。

### 6.4 第四组：预处理公平对比

重新跑同条件：

- raw
- STFT offline
- SDP offline
- SDP offline power_xfall

这组要统一 `num_query`，否则 oracle MPJPE 对 query 数量敏感。

## 7. 可用于汇报的谨慎表述

建议这样表述当前阶段结论：

> 目前已有实验主要是探索性组合实验，不能直接作为严格 ablation。当前最强的组合是 `SDP_offline_power_xfall`，best observed MPJPE 为 `77.6551`，但该指标是 oracle best-of-query，并且是在当前 eval 集上选择的 best epoch。image-like 和 translator 路线能够训练，但还没有在公平控制变量下证明优于 SDP power 基线。stage2 token distillation 在低学习率 v2 中比长 schedule v1 稳定，但仍缺少 no-token fine-tune control，因此暂时不能断言蒸馏有效或无效。下一步应优先补 top-1 评估、validation/test 分离和关键 ablation。

