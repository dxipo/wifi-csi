# WiFi CSI 人体姿态估计实验阶段汇报

日期：2026-05-12  
当前重点分支：`sdp140_image-like_stage2_distill`

## 1. 实验结果汇总

说明：

- 下表结果来自 `result/` 目录中已有训练日志。
- 指标为日志中的 2D MPJPE，单位可理解为像素坐标误差。
- 当前评估代码会在 100 个 query 预测中，用 GT 选择距离最近的预测作为 MPJPE，因此该 MPJPE 更接近 oracle best-of-100，不是严格 top-1 部署指标。不同实验之间可用于观察趋势，但绝对数值需要谨慎汇报。
- `best val` 表示训练过程中日志记录到的最优验证 MPJPE；`epoch 450` 表示最终第 450 轮的结果。

| 编号 | 实验目录 | 输入 / 预处理 | 主要模型设置 | epoch 450 MPJPE | best val MPJPE | 观察 |
| --- | --- | --- | --- | ---: | ---: | --- |
| E0 | `42_wifipose_2d_baseline_test_recover` | 原始 hold-out 数据 | 原 baseline | 175.32 / x 114.16 / y 98.68 | 59.52 @ epoch 4 | 最优轮次过早且后期退化明显，需要复现实验与严格评估后再作为稳定基线。 |
| E1 | `42_wifipose_2d_baseline_STFT_offline` | STFT offline，`input_dim=9` | baseline 输入适配 STFT | 121.55 / x 103.97 / y 42.54 | 85.18 @ epoch 3 | STFT 后期结果优于 E0 final，但最优也出现在极早期，曲线不够稳定。 |
| E2 | `42_wifipose_2d_baseline_SDP_offline` | SDP offline，`input_dim=6` | baseline 输入适配 SDP | 302.50 / x 260.43 / y 126.32 | 84.33 @ epoch 235 | SDP 原始版本后期严重退化，说明训练稳定性或表示方式存在问题。 |
| E3 | `42_wifipose_2d_baseline_SDP_offline_power_xfall` | SDP + power response / xfall | `num_query=81` | 98.54 / x 73.04 / y 45.59 | 77.66 @ epoch 391 | 当前较有价值的方向，best 与 final 都明显优于多数实验。 |
| E4 | `43_wifipose_2d_baseline_SDP300_imagelike_lagwindow_centerctx` | SDP300 image-like centered-context，样本 shape `(30,825,3)` | `conv_patch`，lagwindow 排布 | 244.93 / x 218.63 / y 66.90 | 84.99 @ epoch 264 | best 不差，但 final 退化严重，说明 SDP300 image-like 表达或训练后期不稳定。 |
| E5 | `44_wifipose_2d_baseline_SDP140_translator_image_lagwindow_centerctx` | SDP140 image-like，样本 shape `(30,270,3)` | `SDPToImageLikeTranslator` 转为 `(3,360,640)`，再 patch embed | 112.68 / x 98.23 / y 35.15 | 101.38 @ epoch 106 | translator 路线可以收敛，final 稳定性好于 E4，但尚未超过 E3。 |
| E6 | `45_wifipose_2d_SDP140_image-like_stage2_distill_lagwindow_centerctx` | 同 E5，并额外读取 `gt_token` | 从 E5 checkpoint 初始化，加入 token 蒸馏 MSE loss | 140.50 / x 101.71 / y 69.06 | 102.52 @ epoch 29 | 当前蒸馏没有带来增益；best 接近 E5，但 final 明显变差。 |

阶段性结论：

1. 从现有日志看，`SDP + power_xfall` 是目前最值得保留的强基线，best val 为 `77.66`，epoch 450 为 `98.54`。
2. `SDP140 image-like + translator` 路线可训练，但目前没有超过 `SDP + power_xfall`。
3. `stage2 token distill` 当前没有提升，反而使 final 结果变差，需要重新检查蒸馏目标、loss 权重和训练策略。
4. 由于评估是 oracle best-of-100，下一阶段应优先补充严格 top-1 / top-k 评估，否则实验结论容易偏乐观。

## 2. 方法与代码逻辑总结

### 2.1 整体代码结构

当前主要代码路径：

- `configs/wifi/petr_wifi.py`：实验配置，包括数据路径、输入 shape、模型适配器、蒸馏开关、训练轮数和日志目录。
- `opera/datasets/wifi_pose.py`：WiFi pose 数据集加载逻辑，读取 CSI / SDP 特征、人体关键点、蒸馏 token，并构造训练需要的 bbox、area、labels 等字段。
- `opera/models/detectors/petr.py`：PETR 检测器主干逻辑，负责把 WiFi 输入转换为 transformer token。
- `dataset_model/sdp_to_image.py`：SDP image-like translator，将 SDP 特征转成类似 RGB 图像的 `(3,360,640)` 特征图。
- `opera/models/dense_heads/petr_head.py`：PETR head，完成 query 解码、关键点回归、分类 loss、关键点 loss 和 token 蒸馏 loss。
- `opera/models/utils/transformer.py`：PETR transformer 结构。

### 2.2 数据流

以当前分支 `sdp140_image-like_stage2_distill` 为例：

1. 数据集从以下目录读取 SDP140 image-like 输入：
   - train：`data/wifipose/all_single_train_data_hold_out_sdp140_imagelike_centerctx_n30_power_xfall`
   - test：`data/wifipose/all_single_test_data_hold_out_sdp140_imagelike_centerctx_n30_power_xfall`
2. 每个样本的 WiFi 输入为 `(30,270,3)`，layout 为 `hwc`。
3. `wifi_pose.py` 将样本整理为模型输入，同时读取人体 2D keypoints 和 `gt_token`。
4. `petr.py` 中的 `input_adapter='sdp_image_translator'` 被启用：
   - 原输入 `(B,30,270,3)` 先变换为 channel-first；
   - 进入 `SDPToImageLikeTranslator`；
   - 输出 `(B,3,360,640)`。
5. translator 输出经过 patch embedding：
   - kernel / stride 为 `(18,32)`；
   - 得到 `20 x 20 = 400` 个 token；
   - token 维度为 `256`。
6. PETR transformer 使用 object query 进行解码，输出 100 个 query 的人体关键点预测。
7. `petr_head.py` 计算分类 loss、2D keypoint loss，以及当前分支新增的 token 蒸馏 loss。

### 2.3 SDP 预处理重点

已尝试的 SDP 方向主要包括：

1. `SDP_offline`：将原始 CSI 转成 SDP 特征后直接输入模型。
2. `SDP_offline_power_xfall`：在 SDP 基础上加入 power response / xfall 相关处理，目前是最强的可保留 baseline。
3. `SDP300 image-like centered-context`：将 SDP 构造成 image-like 结构，单样本 shape 为 `(30,825,3)`，使用 lagwindow 排布和 centered context。
4. `SDP140 image-like centered-context`：将 SDP 构造成更小的 image-like 输入，单样本 shape 为 `(30,270,3)`，再通过 translator 变换到图像大小。

SDP 预处理的核心目的：

- 将 WiFi CSI 中的时序、多径、相位/功率变化转成更适合视觉模型消费的结构化表示。
- 通过 lagwindow 和 centered-context 保留局部时间上下文。
- 通过 image-like layout 尝试复用图像模型或视觉 transformer 的建模方式。

当前问题：

- SDP300 image-like 在 best epoch 有一定效果，但 final 退化严重。
- SDP140 translator 更稳定，但效果尚未超过 `SDP + power_xfall`。
- 说明 image-like 表达还没有充分证明优于直接 SDP 表达，需要更多 ablation。

### 2.4 新增网络结构

当前分支新增或重点使用的网络结构是 `SDPToImageLikeTranslator`：

- 输入：SDP image-like 特征，例如 `(B,30,270,3)`。
- 中间：卷积 stem、残差块、encoder 下采样、bridge、decoder 上采样。
- 输出：类似 RGB 图像的 `(B,3,360,640)`。
- 后续：用 patch embedding 将图像转为 transformer token。

该结构的动机：

- PETR 原本更接近视觉检测/姿态估计框架，直接喂 WiFi SDP 特征与视觉输入分布差异很大。
- translator 试图把 SDP 特征映射到一个更接近图像输入的空间，再交给 transformer 处理。

当前风险：

- translator 只通过最终 pose loss 间接监督，没有真实图像重建或中间特征监督。
- 因此输出不一定真的具备稳定、可解释的 image-like 语义。
- config 中仍保留了一些 ResNet/backbone/neck 配置，但当前 `sdp_image_translator` 路径主要走 translator + patch embed，部分原视觉 backbone 配置可能只是历史遗留。

### 2.5 蒸馏部分

当前分支的蒸馏逻辑：

1. 数据集中额外读取 `gt_token`，token 维度为 `768`。
2. 模型从 E5 的 translator baseline checkpoint 初始化，属于 stage2 fine-tuning。
3. `petr_head.py` 中从 transformer memory 或相关特征中预测 token。
4. 使用 MSE loss 约束预测 token 与 `gt_token` 接近。
5. 当前配置中 `loss_token` 权重为 `0.1`。

当前结果：

- E5 translator baseline：epoch 450 MPJPE `112.68`，best `101.38 @ epoch 106`。
- E6 translator + distill：epoch 450 MPJPE `140.50`，best `102.52 @ epoch 29`。
- 蒸馏并未提升效果，final 反而更差。

可能原因：

- `loss_token` 数值在日志中很小，对总 loss 的实际影响可能不足。
- `gt_token` 与当前 WiFi SDP 输入之间可能存在 domain gap，token 目标未必与 pose 回归最相关。
- stage2 训练时可能破坏了 E5 已学到的 pose 表示。
- token 蒸馏目标、归一化方式、loss 权重、蒸馏位置都需要 ablation。

## 3. 下一步改进策略

### 3.1 先修正实验评估与 checkpoint 选择

优先级最高。建议：

1. 增加严格 top-1 MPJPE：按分类分数或置信度选择最终 query，而不是用 GT 选最近 query。
2. 保留 oracle best-of-100 MPJPE 作为辅助指标，但不要作为唯一主指标。
3. 区分 validation 和 final test：
   - validation 用于选 best checkpoint；
   - test 只用于最终一次汇报；
   - 如果当前训练过程直接用 test 集做 evaluation，则不能把 test 上最优 epoch 当成严格最终结果。
4. 配置 `save_best='mpjpe'`，并保存 best checkpoint。
5. 每个重要实验至少跑 2 到 3 个 seed，报告均值和方差。

### 3.2 以 `SDP + power_xfall` 作为强基线继续推进

目前 E3 是最有竞争力的方向。建议先固定：

- 数据预处理：`SDP_offline_power_xfall`
- 模型结构：当前 baseline
- 评估方式：严格 top-1 + oracle 辅助

在这个强基线上再逐步加 image-like、translator、蒸馏，否则很难判断改动是否真的有效。

### 3.3 重新设计 image-like ablation

建议做以下对比：

1. SDP140 vs SDP300，在相同训练策略和相同评估方式下比较。
2. lagwindow vs windowlag，确认排布方式是否影响模型学习。
3. per-sample / per-channel normalization 对比。
4. `conv_patch` vs `sdp_image_translator` 对比。
5. 可视化 translator 输出，检查是否出现塌缩、过平滑或无结构噪声。

目标是回答一个核心问题：image-like 预处理到底是否比直接 SDP 表达更有效。

### 3.4 重新做蒸馏实验

建议按小步 ablation 做：

1. 检查 `gt_token` 来源、归一化和维度是否稳定。
2. 尝试 `loss_token` 权重：`0.1 / 1.0 / 5.0`。
3. 对 token 做 L2 normalize 或 LayerNorm 后再蒸馏。
4. 比较 stage1 从头训练蒸馏 vs stage2 从 baseline fine-tune。
5. 尝试只蒸馏中间层特征或 query token，而不是单一全局 token。
6. 记录 `loss_token` 与 MPJPE 的相关性，如果 loss 下降但 MPJPE 不下降，说明 token 目标与 pose 任务不一致。

### 3.5 清理模型配置中的不一致点

建议检查：

- `sdp_image_translator` 路径下 ResNet/backbone/neck 是否实际参与 forward。
- `with_kpt_refine=True` 但 `loss_kpt_refine=0` 是否有意义；如果 refine 不监督，建议关闭或补上监督。
- `num_query`、分类 loss 与最终 query 选择是否匹配。
- 训练日志中各 loss 的量级是否合理，避免某个新增 loss 实际不起作用。

## 4. 关于 epoch 450 和最终结果选择

结论：训练到 450 epoch 可以作为统一训练预算，但不建议只用第 450 轮作为最终实验结论。

更合理的做法：

1. 如果有独立 validation 集：用 validation MPJPE 选择 best checkpoint，再在 test 集上评估该 checkpoint。
2. 如果当前 evaluation 用的是 test 集：不能严格地在 test 上挑 best epoch，否则会把 test 用成 validation，最终结果会偏乐观。
3. 如果暂时没有 validation 集：汇报时建议同时给出 `best observed` 和 `epoch 450 final`，并明确说明当前 best 是在现有 eval 集上选择的，后续会补 validation split。

为什么不建议只看 epoch 450：

- 当前曲线明显波动，很多实验 best epoch 与 epoch 450 差距很大。
- 例如 E6 蒸馏实验 best 为 `102.52 @ epoch 29`，但 epoch 450 为 `140.50`。
- 如果只看 final，会低估某些方法的潜力；如果只看 best，又可能高估泛化能力。
- 科研汇报中最好同时报告 best 和 final，并说明 checkpoint 选择规则。

建议给老师的表述：

> 当前我把 450 epoch 作为统一训练预算，但由于验证曲线波动明显，后续正式比较会采用 validation best checkpoint，再在独立 test 上报告最终结果。现阶段表格同时列出 best 和 epoch 450，用于区分方法潜力和训练稳定性。

## 5. 为什么当前结果不理想

可能原因按优先级排序：

1. 评估方式偏乐观且不够贴近部署：当前 MPJPE 使用 GT 在多个 query 中选最近预测，不能反映模型真正选择人的能力。
2. WiFi CSI 到人体 2D 姿态本身是弱观测问题：信号受环境、多径、朝向、遮挡、人体动作差异影响很大。
3. SDP / image-like 预处理可能损失了部分有效信息：尤其是 flatten 顺序、lag window、归一化方式都会显著影响模型学习。
4. translator 缺少中间监督：它被要求把 SDP 转成 image-like 表达，但没有直接约束输出必须有稳定语义。
5. 蒸馏目标可能不匹配：token 与最终 2D pose 回归之间不一定强相关，当前 loss 权重也可能过小。
6. 训练不稳定：多个实验的 best 与 final 差距很大，说明可能存在 overfitting、学习率后期不合适、数据量不足或 eval 抖动问题。
7. 结构配置存在历史遗留：部分 backbone/neck/refine 配置与当前 forward 路径不完全匹配，可能导致实验解释不够干净。

## 6. 建议汇报重点

可以按以下逻辑汇报：

1. 我已经完成了从原始 CSI、STFT、SDP、SDP image-like 到 SDP140 translator 和 token distillation 的一系列探索。
2. 当前最强的可保留 baseline 是 `SDP + power_xfall`，best val MPJPE 为 `77.66`，epoch 450 MPJPE 为 `98.54`。
3. image-like 方向能够训练，但 SDP140 translator 尚未超过强 SDP baseline。
4. stage2 token distillation 当前未带来提升，说明蒸馏目标、loss 权重或训练方式需要重新设计。
5. 下一步优先修正评估方式和 checkpoint 选择，再围绕 SDP 表达、translator 结构和蒸馏 loss 做可控 ablation。
