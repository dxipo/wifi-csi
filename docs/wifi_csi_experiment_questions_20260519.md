# WiFi CSI 实验疑问复核与建议

日期：2026-05-19

参考文档：

- `docs/wifi_csi_experiment_comparison_and_mpjpe_review_20260518.md`
- `result/` 下已完成训练日志
- 当前评估实现：`opera/datasets/wifi_pose.py`

## 1. “best 早”是什么意思，为什么不能直接采用

`best 早`指的是某个实验的最低 MPJPE 出现在训练非常早期，例如 epoch 4、epoch 10，而不是训练后期稳定收敛阶段。

需要特别注意：`wifipose_2d_baseline_test` 这个目录混有多次运行。文档表格里写的：

| result | final | best |
| --- | ---: | ---: |
| `wifipose_2d_baseline_test` | `86.2360 @ epoch 450` | `48.5507 @ epoch 10` |

这两个数不是同一次完整训练日志里的结果：

- `48.5507 @ epoch 10` 来自 `20260303_150901.log`，该次运行只到 epoch 23 左右就结束，且配置是 `samples_per_gpu=256`、`lr=3e-5`、随机 seed `2090792071`。
- `86.2360 @ epoch 450` 来自 `20260304_145623.log`，该次运行完整跑到 epoch 450，配置是 `samples_per_gpu=32`、`lr=2e-5`、随机 seed `2102887448`。

所以这里的 `48.5507` 只能作为历史探索记录，不能作为严格 baseline 结果。原因有三个：

1. 它不是完整训练的 final 结果。
2. 它和 `86.2360 @ epoch 450` 不是同一次 run。
3. 它来自 test 集逐 epoch 评估中的最优点，相当于用 test 选 checkpoint，会偏乐观。

相比之下，`42_wifipose_2d_baseline_test_recover` 是固定 seed 42 的完整恢复实验：

- seed：`42`
- deterministic：`True`
- batch：`32`
- lr：`2e-5`
- best：`59.5218 @ epoch 4`
- final：`175.3183 @ epoch 450`

这个实验也有 `best 过早` 的问题，但至少它是单次完整 run 内的结果。严格汇报时应优先报告 final，或用独立 validation 选择 best 后再在 test 上只评估一次。

## 2. top1、top-k、oracle 指标应该怎么选

当前早期 MPJPE 更接近 `oracle_all`，也就是在所有 query 中用 GT 选一个最近的预测。这能衡量“模型有没有生成过一个接近 GT 的候选”，但不能代表实际部署性能。

建议指标分工如下：

| 指标 | 是否适合做主指标 | 含义 |
| --- | --- | --- |
| `mpjpe_top1` | 是 | 按模型分类/检测分数最高的 query 作为预测，最接近真实使用方式。 |
| `mpjpe_top5_oracle` | 否，辅助 | 只在分数前 5 个 query 中用 GT 选最近，衡量高置信候选质量。 |
| `mpjpe_top20_oracle` | 否，辅助 | 衡量候选池中是否包含较好姿态。 |
| `mpjpe_oracle_all` / 旧 `mpjpe` | 否，历史对齐 | 保留旧评估，便于和前期实验对比，但它是偏乐观上界。 |

如果现在还没有 validation 集，建议每个实验同时汇报：

1. `final epoch` 的 `mpjpe_top1` 和旧 `mpjpe/oracle_all`。
2. test 上观察到的 best，但明确标注为 `test-best, exploratory`。
3. top5/top20/oracle_all 只用于分析模型候选质量，不作为最终主结论。

后续一旦划分 validation，主流程应改成：

1. train 训练。
2. val 选 checkpoint。
3. 固定 checkpoint 后，test 只评估一次。

## 3. power response / power_xfall 是否能提升泛化性

`power_xfall` 的思路是合理的，但当前结果还不能证明它已经提升了泛化性。

它可能提升泛化性的原因：

1. power response 更偏向幅度/能量扰动，通常比原始相位更少受随机相位偏移、硬件相位漂移影响。
2. 人体运动会改变多径传播和能量分布，power 特征可以更直接地反映运动造成的 CSI 结构变化。
3. 相比把 CSI 展开成很大的伪图像，`SDP_offline_power_xfall` 表达更紧凑，可能更不容易过拟合局部噪声。

但它也可能损失信息：

1. 去掉相位后，某些空间定位信息会变弱。
2. power 本身仍受距离、朝向、人体位置、房间多径结构影响。
3. 如果 train/test 环境非常接近，结果变好不等于跨场景泛化变好。

因此现在比较准确的说法是：

> `SDP_offline_power_xfall` 是当前最强、最稳定的主线表示之一；它有提升泛化性的动机，但是否真的提升泛化性，需要跨人、跨日期、跨位置或跨场景 split 来验证。

建议补充：

- same-subject / cross-subject 对比。
- same-room / shifted-position 对比。
- 不同日期采集的 test。
- `SDP_offline` vs `SDP_offline_power_xfall`，其余设置完全一致。

## 4. translator 没有图像监督，如何改进

目前 `SDPToImageLikeTranslator` 只是把 SDP 特征映射到类似 `(3,360,640)` 的 image-like 张量，然后靠 pose loss 反向训练。这个输出不一定具有真实图像语义，因为没有真实 RGB、DensePose 或视觉特征监督。

可改进方向：

| 方向 | 做法 | 目的 |
| --- | --- | --- |
| teacher feature distillation | 用配对 RGB 经过 ResNet/FPN/DensePose teacher，监督 CSI translator 输出的中间特征 | 让 image-like 特征靠近视觉特征，而不是只靠 pose loss。 |
| heatmap / skeleton supervision | 给 translator 后面加关键点 heatmap、limb heatmap、mask 或 bbox 辅助头 | 提供空间结构约束。 |
| token-level alignment | 用 stage1 生成的 token 约束 CSI token，配合 pose loss | 比强行生成 RGB 尺寸图更直接。 |
| contrastive alignment | 配对 CSI/RGB 样本做特征对比学习 | 让同一时刻 CSI 与视觉特征对齐。 |
| 更轻量 adapter | 不强制输出 640x360x3，而是输出低分辨率 feature map，例如 80x45xC 或 token 序列 | 降低伪图像映射难度。 |

我的建议是：后续不要把这个模块称为“生成图像”，而称为 `CSI-to-vision-feature adapter` 更准确。优先监督视觉特征或 token，而不是监督 RGB 像素，因为 CSI 本身不包含足够信息恢复真实外观。

## 5. 固定 seed 后反而变差，应该怎么看

你补充的信息很关键：

- `42_...` 表示固定随机种子为 42。
- 后续 `45/46/47/48` 等编号实验也基本是 seed 42。
- 早期没有数字前缀的实验通常是不固定 seed，由程序随机生成 seed。

现在看到的现象是：某些不固定 seed 的历史实验效果非常好，而 seed 42 的 recover 实验 final 很差。例如：

| 实验 | seed | final MPJPE | best MPJPE |
| --- | ---: | ---: | ---: |
| `wifipose_2d_baseline_test` 完整 run | `2102887448` | `86.2360` | 该完整 run 内约 `66.0337 @ epoch 146` |
| `wifipose_2d_baseline_test` 早期中断 run | `2090792071` | 未完整 | `48.5507 @ epoch 10` |
| `42_wifipose_2d_baseline_test_recover` | `42` | `175.3183` | `59.5218 @ epoch 4` |

这说明至少有两类问题：

1. 训练确实对 seed 非常敏感。
2. 早期 result 目录混有不同 run，不能只按目录名汇总。

seed 敏感的可能来源：

- PETR query 初始化会影响 query 分工。
- Hungarian matching 在早期训练阶段对初始化和候选分布敏感。
- 当前旧 MPJPE 是 oracle best-of-query，容易放大“某个 query 碰巧很接近 GT”的幸运情况。
- dataloader shuffle、CUDA 非确定性、初始化、dropout 都会改变轨迹。
- raw CSI 输入本身噪声较大，模型可能落入不同局部最优。

建议做法：

1. 每个关键方法至少跑 3 个固定 seed，最好 5 个。
2. 不要只比较单个 seed 的 best。
3. 汇报 `mean ± std`，同时报告 final 和 val-best。
4. seed 可以用 `42, 2024, 3407, 123, 2090792071`，把历史好 seed 也纳入复现实验。
5. 如果 seed 42 明显偏差很大，不要把它当作唯一结论；它只能说明该方法在某个固定初始化下不稳定。

当前阶段最重要的是把“方法变化”和“seed 变化”拆开。否则无法判断改进来自方法，还是来自初始化运气。

## 6. 为什么 2D baseline 没有明显优于原论文 3D 结果

直觉上 2D 应该比 3D 容易，但当前结果不能直接和原论文 3D MPJPE 对比，原因如下。

第一，指标单位不一致。原论文表格是 3D 空间中的毫米误差，Mean MPJPE 为 `107.2 mm`。你当前很多 2D 结果是像素误差，图像大小是 `640x360`。像素误差和毫米误差不是同一尺度，不能直接说谁更好。

第二，原论文不是只靠 2D keypoint MSE。原始 3D 方法里可能包含更完整的 3D 几何约束、骨架结构约束、多分量 loss 或原始任务适配设计。你为了减少影响去掉了一些 loss，这可能同时去掉了有用的正则项。

第三，2D 不一定天然更容易。WiFi CSI 对人体横向位置、左右肢体、遮挡和相机投影的约束较弱。2D 像素坐标虽然少了深度维度，但它依赖相机坐标系和投影标注，横向误差会被图像宽度放大。

第四，当前评估长期使用 test 作为逐 epoch 评估集，且旧 MPJPE 是 oracle query 指标。它和原论文的评估协议不一致。

因此更合理的比较方式是：

1. 把 2D 误差转为 normalized error，例如 `dx/W`、`dy/H`。
2. 或者基于相机标定/人体尺度，把 2D 误差映射到物理尺度后再比较。
3. 在同一 split、同一评估协议下比较 2D baseline、3D baseline 和改进版。
4. 报告 per-joint 2D MPJPE，和原论文 per-joint 3D MPJPE 做趋势比较，而不是直接数值比较。

## 7. MPJPE 中硬编码图像大小是否能提高效率

当前写死 `W=640, H=360`，如果所有样本确实都是这个尺寸，数值上没有问题。

但硬编码几乎不会带来可感知的效率提升。评估的主要耗时在模型 forward、结果收集和逐样本计算，读取 `img_shape` 或 dataset metadata 的成本可以忽略。

更稳妥的做法是：

- 每个样本从 metadata 读取 `img_shape`；
- 或者 dataset 初始化时缓存统一的 `image_width=640, image_height=360`；
- 如果发现所有样本尺寸一致，就只存一次全局尺寸。

这样既不会明显拖慢速度，也能避免后续换 resize、crop 或数据集时 MPJPE 悄悄算错。

## 8. 为什么 x 方向误差明显大于 y 方向

从当前多组结果看，x 误差大于 y 误差是一个比较明显的趋势。例如：

- R3 final：x `73.0389`，y `45.5856`
- R9 final：x `72.1762`，y `40.1059`
- raw recover final：x `114.1612`，y `98.6775`
- 一些退化 run 中 x 会远大于 y，例如 x 超过 200，而 y 只有几十。

可能原因：

1. 图像宽高不同。x 乘以 `W=640`，y 乘以 `H=360`。如果 normalized 误差相同，x 像素误差天然会是 y 的 `640/360=1.78` 倍。
2. 人体姿态的 y 方向受重力和骨架高度约束更强，头、肩、髋、膝、脚在竖直方向有更稳定的相对结构。
3. x 方向更容易受人体左右移动、朝向、左右肢体混淆和多径对称性影响。
4. WiFi 天线布置如果对横向定位不敏感，x 方向会更难。
5. 旧 oracle matching 可能选到 y 较准但 x 偏移的 query，导致 x/y 差异被放大。

下一步建议同时报告：

- `mpjpe_x_norm = mpjpe_x / 640`
- `mpjpe_y_norm = mpjpe_y / 360`
- per-joint x/y error
- top1 和 oracle_all 下各自的 x/y

这样可以判断 x 大到底是因为像素尺度，还是模型真的在横向定位上更差。

## 9. R8 之后“选最好 epoch 再训练”的验证情况

之前的问题是：

> R8 是从 `result/42/latest.pth` 启动，也就是 epoch 450 final checkpoint，而不是 R3 的 best epoch 391 checkpoint。

后来已经补了一个 R9：

- result：`result/48_wifipose_2d_baseline_SDP_offline_power_xfall_stage2_distill_best_saved_ckpt`
- load_from：`result/42_wifipose_2d_baseline_SDP_offline_power_xfall/epoch_165.pth`
- 选择原因：R3 真正 test-best 是 epoch 391，但 `epoch_391.pth` 没有保存；在保存下来的 checkpoint 中，`epoch_165.pth` 是较好的可用 checkpoint，日志中 `epoch_165` 为 `82.6933`。
- max epoch：`100`
- seed：`42`
- final old MPJPE / oracle_all：`91.7977`
- best old MPJPE / oracle_all：`83.9841 @ epoch 1`
- final top1：`116.9121`
- best top1：`116.8735 @ epoch 73`

这个结果说明：从较好 checkpoint 启动，比从 final checkpoint 启动更稳定。R9 final `91.7977` 优于 R8 final `117.3972`，但仍没有超过 R3 的 test-best `77.6551 @ epoch 391`。

不过 R9 仍然不能证明 token 蒸馏有效，因为还缺一个关键 control：

| 实验 | 初始化 | token loss | 目的 |
| --- | --- | --- | --- |
| R9 | `epoch_165.pth` | 开启 | 当前已完成。 |
| D0-control | `epoch_165.pth` | 关闭 | 判断改善来自 token 蒸馏，还是来自低学习率继续训练。 |

下一步最应该补的是 `D0-control`：同样从 `epoch_165.pth` 启动，同样 lr、epoch、batch、seed，但关闭 token loss。只有 D0 和 R9 对比后，才能判断 stage2 蒸馏是否真的有增益。

## 10. 后续实验优先级

建议按下面顺序做，而不是继续直接堆新结构：

1. 修正实验记录：同一 result 目录内多次运行要拆开记录，不再混用 best/final。
2. 建立 validation split，test 不再用于选 best。
3. 每个关键方法跑 3-5 个固定 seed，汇报 `mean ± std`。
4. 对强基线 `SDP_offline_power_xfall` 补 D0 no-token fine-tune control。
5. 对 `SDP_offline` vs `SDP_offline_power_xfall` 做只改变 power 的 ablation。
6. 对 x/y 误差增加 normalized 版本，判断横向误差是否真的异常。
7. translator 路线优先加入 teacher feature/token 监督，而不是只靠 pose loss。

