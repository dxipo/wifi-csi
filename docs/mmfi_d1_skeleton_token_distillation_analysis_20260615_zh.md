# MMFi D1 Skeleton Token 两阶段蒸馏实验记录与分析 - 2026-06-15

## 1. 文档目的

本文档记录当前 `mmfi_skeleton_token_distill` 分支中 MMFi 3D skeleton token 蒸馏实验的完整设计、实现步骤、阶段性结果和问题分析。

本轮实验的核心目标是验证：

```text
训练阶段使用 2D skeleton teacher 提供人体结构表征；
推理阶段 student 仍然只输入 CSI；
最终是否能稳定提升 CSI-only 3D pose estimation 的 MPJPE。
```

这条路线来自此前讨论的组合：

```text
2D skeleton teacher
+ pose-image 时空表示
+ Transformer pose tokens
+ token / relation distillation
+ CSI-only 3D student
```

需要强调：当前 D1 实验是“方法闭环验证”，不是最终论文方案。它的价值主要在于判断 skeleton token 蒸馏是否真的能向 CSI-only 3D 主任务转移有效知识。

截至本文档整理时，D1 Stage2 仍在运行，最新结构化日志已验证到 `epoch 74/80`。当前最佳结果出现在 `epoch 6`，后续训练没有继续改善。

## 2. 当前分支与主要文件

当前分支：

```text
mmfi_skeleton_token_distill
```

主要新增或修改文件：

| 文件 | 作用 |
|---|---|
| `configs/mmfi/skeleton_teacher_t0.py` | 2D skeleton -> 3D teacher 训练配置 |
| `configs/mmfi/petr_mmfi_d1_stage1_token_pretrain.py` | D1 Stage1 token/relation 蒸馏预训练配置 |
| `configs/mmfi/petr_mmfi_d1_stage2_3d_distill.py` | D1 Stage2 3D 主任务 + 蒸馏配置 |
| `configs/mmfi/petr_mmfi_b0_3d_baseline.py` | MMFi CSI-only 3D baseline 配置 |
| `opera/models/detectors/skeleton_pose_teacher.py` | 2D skeleton teacher 模型 |
| `opera/models/detectors/petr_distill.py` | PETRDistill student 模型 |
| `opera/datasets/mmfi_pose.py` | MMFi 数据集、2D skeleton 序列、teacher token 加载逻辑 |
| `tools/dataset_converters/export_mmfi_teacher_tokens.py` | 离线导出 teacher tokens |
| `opera/models/dense_heads/petr_head.py` | 修正 PETRHead 中硬编码 `14` 点 reshape，使其支持 MMFi `17` 点 |

相关运行目录：

| 目录 | 内容 |
|---|---|
| `result/mmfi_t0_skeleton_teacher` | T0 teacher 训练结果 |
| `data/mmfi_teacher_tokens/t0/train` | 离线 teacher token |
| `result/mmfi_d1_stage1_token_pretrain` | D1 Stage1 预训练结果 |
| `result/mmfi_d1_stage2_3d_distill` | D1 Stage2 训练结果 |

## 3. 数据与任务设定

### 3.1 MMFi 主任务标签

MMFi 的 3D 主监督使用：

```text
/home/xl/Downloads/MMFi_Dataset/MMFi_unzipped/<scene>/<subject>/<action>/ground_truth.npy
```

当前任务输出为：

```text
17 x 3
```

这和 Person-in-WiFi 3D 原论文的 `14 x 3` 不同。因此 MMFi 的 MPJPE 只能和 MMFi 自身 baseline 比较，不能直接和 Person-in-WiFi 3D 的 MPJPE 数值横向比较。

### 3.2 MMFi 2D skeleton 信息

MMFi 中的视觉侧信息不是原始 RGB 图片，而是已经提取好的 2D skeleton：

```text
rgb/frame*.npy -> 17 x 2
```

本实验不把它作为主监督标签，而是作为 teacher 输入。这样设计的原因是：

1. 主任务仍然是 3D pose estimation，指标仍然上报 3D MPJPE。
2. 2D skeleton 只在训练期提供视觉模态的结构知识。
3. 推理阶段 student 不读取 `rgb/frame*.npy`，仍然是 CSI-only。

### 3.3 当前数据划分

当前配置沿用 `MMFiPoseDataset` 中的 MMFi protocol 设置：

```text
protocol = protocol2
split_to_use = random_split
random_ratio = 0.8
random_seed = 0
```

对应动作：

```text
A01, A06, A07, A08, A09, A10, A11, A12, A15, A16, A24, A25, A26
```

训练样本量：

```text
123552
```

验证样本量：

```text
30888
```

## 4. 2D Skeleton Teacher: T0

### 4.1 设计动机

MMFi 没有提供原始 RGB 图像，因此不能直接复用此前 2D 实验中的 VitPose 图像 token 蒸馏路径。当前 teacher 改成：

```text
2D skeleton sequence -> Transformer teacher tokens -> 3D pose
```

这仍然保留“视觉模态知识迁移”的核心思想，只是 teacher 的输入从 RGB image 变成了 2D skeleton。

这里使用 2D skeleton 的理由是：

1. 2D skeleton 是 RGB 模态经过人体姿态网络提取后的高层结构表示。
2. 它去掉了背景、光照、衣服纹理等与 CSI 3D pose 任务弱相关的信息。
3. 它保留了人体拓扑、关节相对关系、动作时序等对 3D 姿态有用的信息。

### 4.2 输入表示

Teacher 输入使用以 center frame 为中心的 2D skeleton 时间窗口：

```text
pose2d_seq: T x 17 x C
T = 9
C = 3
```

其中 `C=3` 表示：

```text
x_norm, y_norm, valid_mask
```

归一化逻辑位于 `opera/datasets/mmfi_pose.py`：

1. 对每一帧读取 `rgb/frame*.npy`。
2. 根据非零和有限值生成 valid mask。
3. 如果左右 hip 点有效，则以 hip center 作为 2D 中心；否则使用有效点均值。
4. 使用有效 2D skeleton bbox 的最大边长作为 scale。
5. 对 2D 坐标做中心化和尺度归一化。
6. 缺失点置零，同时通过 mask 通道保留可见性。

随后 teacher 将其视为 pose-image：

```text
B x T x K x C -> B x C x T x K
```

这里的 `T` 相当于图像高度，`K=17` 相当于图像宽度，`C=3` 是通道。这对应 MSS-Former 中 pixel-level spatiotemporal representation 的启发，但本实验中的“像素”不是 RGB 像素，而是 skeleton coordinate image。

### 4.3 Teacher 网络结构

实现文件：

```text
opera/models/detectors/skeleton_pose_teacher.py
```

结构：

```text
pose2d_seq
-> Conv2d pose_stem
-> time embedding + joint embedding
-> Transformer Encoder
-> center-frame joint tokens
-> global token
-> 3D pose head
```

Teacher 输出：

```text
teacher_tokens: 18 x 256
teacher_pred_3d: 17 x 3
```

其中 `18` 个 token 表示：

```text
1 个 global token + 17 个 joint tokens
```

### 4.4 Teacher loss

Teacher 使用 3D GT 训练：

```text
L_teacher = L_abs_3D + L_rel_3D
```

其中：

```text
L_abs_3D = SmoothL1(pred_3d, gt_3d)
L_rel_3D = SmoothL1(pred_3d - pelvis(pred_3d), gt_3d - pelvis(gt_3d))
```

`pelvis` 当前由左右 hip 点计算：

```text
pelvis = (joint_11 + joint_12) / 2
```

设计含义：

1. `L_abs_3D` 约束绝对 3D 坐标。
2. `L_rel_3D` 约束 pelvis 对齐后的相对人体结构。
3. 由于 2D skeleton 对深度和全局位置天然不完整，加入 relative loss 可以让 teacher 更稳定地学习人体形状和姿态。

### 4.5 Teacher 结果

配置：

```text
configs/mmfi/skeleton_teacher_t0.py
```

工作目录：

```text
result/mmfi_t0_skeleton_teacher
```

关键结果：

| 实验 | best MPJPE | final MPJPE | pelvis MPJPE | PA-MPJPE | 观察 |
|---|---:|---:|---:|---:|---|
| T0 Skeleton Teacher | `168.7745 @ epoch 72` | `170.5309 @ epoch 80` | `52.2501 @ epoch 72` | `32.4988 @ epoch 72` | 明显强于 CSI-only baseline，但绝对 x/z 仍有较大误差 |

T0 的意义：

1. 2D skeleton 确实包含强人体姿态信息。
2. Teacher 的 `PA-MPJPE` 和 pelvis-relative 误差很低，说明相对姿态建模较好。
3. Teacher 的绝对 MPJPE 仍明显高于 pelvis-relative 误差，说明 2D skeleton 对全局 3D 位置、深度、尺度仍存在不可观测性。

这对蒸馏非常关键：teacher 更适合提供“人体结构和相对姿态知识”，不一定适合提供绝对 3D 坐标知识。

## 5. Teacher Token 离线导出

### 5.1 导出目的

Stage1 和 Stage2 student 都需要读取 teacher tokens。如果训练时在线跑 teacher，会增加显存和时间开销。因此先离线导出：

```text
sample_id -> teacher_tokens, teacher_pred_3d
```

### 5.2 导出命令

实际使用的 checkpoint 是 T0 best：

```text
result/mmfi_t0_skeleton_teacher/best_mpjpe_epoch_72.pth
```

导出命令：

```bash
python tools/dataset_converters/export_mmfi_teacher_tokens.py \
  configs/mmfi/skeleton_teacher_t0.py \
  --checkpoint result/mmfi_t0_skeleton_teacher/best_mpjpe_epoch_72.pth \
  --split train \
  --out-dir data/mmfi_teacher_tokens/t0 \
  --batch-size 512 \
  --device cuda:0
```

输出目录：

```text
data/mmfi_teacher_tokens/t0/train
```

导出结果：

| 项目 | 数值 |
|---|---:|
| `.npz` token 文件数 | `123552` |
| 目录大小 | `2.4G` |

每个 `.npz` 包含：

```text
tokens: 18 x 256
pred_3d: 17 x 3
sample_id
```

当前 D1 只使用 `tokens` 做蒸馏，未使用 `pred_3d` 作为额外监督。

## 6. D1 Stage1: CSI Student Token Pretrain

### 6.1 Stage1 目的

Stage1 的目标不是直接预测 3D，而是让 CSI student 的表示先学习拟合 teacher 的 skeleton tokens：

```text
CSI -> student_tokens
student_tokens -> match teacher_tokens
```

形式上：

```text
L_stage1 = L_token + beta * L_relation
```

直觉是：如果 CSI 表征能先对齐到 2D skeleton teacher 的人体结构 token，那么 Stage2 做 3D 主任务时，student 应该有更好的人体结构先验。

### 6.2 Student 蒸馏结构

实现文件：

```text
opera/models/detectors/petr_distill.py
```

Student 基于 PETR，新增独立的蒸馏分支：

```text
CSI input
-> PETR input stem / backbone / neck
-> CSI tokens
-> distill queries cross-attention
-> student pose tokens
-> projector
-> match teacher tokens
```

具体来说：

1. `extract_csi_tokens(img)` 得到 CSI token memory。
2. 定义可学习的 `distill_queries`，数量为 `18`。
3. `distill_queries` 对 CSI memory 做 MultiheadAttention。
4. 输出 `student_tokens`。
5. 通过 `student_projector` 映射到 teacher token 维度。
6. 使用 LayerNorm 后和 teacher tokens 对齐。

### 6.3 Token loss

Token 蒸馏损失：

```text
L_token = SmoothL1(student_tokens, teacher_tokens)
```

这里的 teacher tokens 会 `detach`，即 teacher 不参与反向传播。

### 6.4 Relation loss

Relation 蒸馏损失：

```text
R = normalize(tokens) @ normalize(tokens).T
L_relation = MSE(R_student, R_teacher)
```

含义：

1. 不只让每个 token 的数值接近。
2. 还让 token-token 之间的关系矩阵接近。
3. 对人体姿态来说，这相当于让 global token 和各 joint token，以及 joint token 之间的相互关系被迁移。

### 6.5 Stage1 配置

配置文件：

```text
configs/mmfi/petr_mmfi_d1_stage1_token_pretrain.py
```

关键设置：

| 项目 | 设置 |
|---|---|
| model | `opera.PETRDistill` |
| distill_stage | `stage1` |
| distill_num_tokens | `18` |
| distill_loss_weight | `1.0` |
| relation_loss_weight | `0.2` |
| optimizer | `AdamW`, lr `1e-4` |
| max_epochs | `40` |
| validation | 关闭，`evaluation.interval=9999` |
| checkpoint | 每 `5` epoch 保存，最多保留 `10` 个 |
| TensorBoard | 已加入 |
| work_dir | `result/mmfi_d1_stage1_token_pretrain` |

Stage1 运行时只优化 token/relation loss，不计算 3D MPJPE。

### 6.6 Stage1 结果

Stage1 现象非常明确：

| 阶段 | 现象 |
|---|---|
| 训练早期 | `loss_token_distill` 迅速下降，`distill_token_cos` 迅速上升 |
| 训练后期 | `loss_token_distill=0.0000`，`loss_relation_distill=0.0000`，`distill_token_cos=1.0000` |
| epoch 40 | `loss=0.0000`，`grad_norm=0.0000` |

结论：

```text
Stage1 token 拟合任务过于容易，已经完全饱和。
```

这说明 student 的独立蒸馏分支可以拟合 teacher tokens，但不能直接证明这种拟合对 3D 主预测有帮助。

## 7. D1 Stage2: 3D 主任务 + Token Distillation

### 7.1 Stage2 目的

Stage2 加载 Stage1 的权重，然后进行真正的 3D pose 主任务训练：

```text
CSI -> 17 x 3 3D pose
```

训练 loss：

```text
L_stage2 = L_3D + lambda * L_token + beta * L_relation
```

其中：

```text
L_3D = PETRHead 原有 3D keypoint loss
L_token = student token 与 teacher token 对齐
L_relation = token relation matrix 对齐
```

注意：推理阶段仍然只使用 CSI，不需要 teacher token，不需要 2D skeleton。

### 7.2 Stage2 配置

配置文件：

```text
configs/mmfi/petr_mmfi_d1_stage2_3d_distill.py
```

关键设置：

| 项目 | 设置 |
|---|---|
| model | `opera.PETRDistill` |
| distill_stage | `stage2` |
| load_from | `result/mmfi_d1_stage1_token_pretrain/latest.pth` |
| distill_loss_weight | `0.1` |
| relation_loss_weight | `0.03` |
| optimizer | `AdamW`, lr `1e-4` |
| max_epochs | `80` |
| evaluation | 每 epoch 计算 MPJPE，保存 best |
| checkpoint | 每 `5` epoch 保存，最多保留 `10` 个 |
| TensorBoard | 已加入 |
| work_dir | `result/mmfi_d1_stage2_3d_distill` |

### 7.3 运行中遇到的问题

Stage2 第一次启动时遇到 `PETRHead` 关键点 reshape 硬编码问题。

原代码中部分 reshape 写死为：

```text
14 x 3
42
```

这适用于 Person-in-WiFi 3D 的 `14` 点，但不适用于 MMFi 的 `17` 点。已经修正为：

```text
self.num_keypoints
self.num_keypoints * 3
```

修正后 Stage2 正常运行。

### 7.4 Stage2 当前结果

截至本文档整理时，Stage2 仍在运行。结构化日志最新完整验证到：

```text
epoch 74/80
```

当前结果：

| 指标 | 数值 |
|---|---:|
| best MPJPE | `240.2957 @ epoch 6` |
| epoch 73 MPJPE | `259.1217` |
| epoch 74 MPJPE | `263.9853` |
| best checkpoint | `result/mmfi_d1_stage2_3d_distill/best_mpjpe_epoch_6.pth` |
| latest symlink | `epoch_70.pth` |
| 结果目录大小 | `3.1G` |

best epoch 6 的详细指标：

| 指标 | 数值 |
|---|---:|
| MPJPE | `240.2957` |
| MPJPE pelvis | `137.1818` |
| PA-MPJPE | `100.7522` |
| MPJPE x | `169.3197` |
| MPJPE y | `88.4771` |
| MPJPE z | `82.3911` |

epoch 74 的详细指标：

| 指标 | 数值 |
|---|---:|
| MPJPE | `263.9853` |
| MPJPE pelvis | `137.5645` |
| PA-MPJPE | `99.7949` |
| MPJPE x | `171.8816` |
| MPJPE y | `88.4177` |
| MPJPE z | `119.7976` |

训练中出现过明显波动：

| epoch | MPJPE | 主要异常 |
|---:|---:|---|
| 10 | `432.3949` | z 方向误差升至 `350.1840` |
| 41 | `483.1144` | z 方向误差升至 `407.3468` |
| 42 | `564.3085` | z 方向误差升至 `499.4103` |

这说明当前模型对深度或全局坐标的估计不稳定。

## 8. 统一结果对比

下表只在 MMFi 内部比较 D1 和 MMFi baseline。Person-in-WiFi 3D 的结果只作为背景，不做直接数值横比。

| 编号 | 实验 | 数据集 | 输入 | 状态 | best MPJPE | final/latest MPJPE | 观察 |
|---|---|---|---|---|---:|---:|---|
| PiW3D-B0 | `origin_paper` | Person-in-WiFi 3D | 原论文 CSI 预处理 | 已完成 | `117.20 @ 391` | `119.77 @ 450` | 原论文 3D 复现主基线 |
| MMFi-B0 | `petr_mmfi_b0_3d_baseline.py` | MMFi | CSI raw amp+phase | 已完成 | `242.56 @ 10` | `255.48 @ 80` | MMFi CSI-only baseline |
| MMFi-B1 | `petr_mmfi_b1_origin_preprocess_stem.py` | MMFi | origin-style preprocess/stem | 已完成 | `226.08 @ 8` | `533.06 @ 80` | early best 更好，但严重不稳定 |
| T0 | `skeleton_teacher_t0.py` | MMFi | 2D skeleton sequence | 已完成 | `168.77 @ 72` | `170.53 @ 80` | teacher 明显强于 CSI-only，但不是 CSI-only |
| D1-Stage1 | `petr_mmfi_d1_stage1_token_pretrain.py` | MMFi | CSI + offline teacher tokens | 已完成 | 无 MPJPE | token loss 饱和为 `0` | 只证明 token 拟合可行 |
| D1-Stage2 | `petr_mmfi_d1_stage2_3d_distill.py` | MMFi | CSI + training-only teacher tokens | 运行中，已到 `epoch 74/80` | `240.30 @ 6` | `263.99 @ 74` | 相比 B0 best 只提升约 `2.26`，不稳定，不能认为有效 |

当前最重要的结论：

```text
D1-Stage2 的 best MPJPE 仅比 B0 best 低约 2.26。
该幅度很小，且发生在 early epoch，后续训练没有保持。
因此当前 D1 不能作为“蒸馏显著提升”的证据。
```

## 9. 为什么效果不好

### 9.1 Stage1 蒸馏目标过早饱和

Stage1 最明显的问题是：

```text
loss_token_distill = 0.0000
loss_relation_distill = 0.0000
distill_token_cos = 1.0000
grad_norm = 0.0000
```

这说明当前 token matching 任务对 student 的独立蒸馏分支来说太容易。Stage1 最终学到的是“让 distill queries 重构 teacher tokens”，但这不等价于“让主 3D 预测路径获得更好的姿态表示”。

换句话说，Stage1 成功拟合了 teacher token，但这个成功可能停留在辅助分支内部。

### 9.2 蒸馏分支和主 3D head 耦合不足

当前 `PETRDistill` 中的蒸馏路径是：

```text
CSI tokens -> distill_queries cross-attention -> student_tokens -> token loss
```

而主 3D 预测路径是：

```text
CSI tokens -> PETRHead encoder/decoder/refine decoder -> 17 x 3 keypoints
```

也就是说，蒸馏 token 是一组额外的、独立的 `distill_queries`。它们并不是 PETRHead 最终用于预测 3D keypoints 的 decoder queries。

风险在于：

1. 辅助 distill queries 可以很好地拟合 teacher tokens。
2. 但主 3D head 的 query 表示未必真的被 teacher 约束。
3. Stage2 中 `loss_token_distill` 已经接近 0，蒸馏项几乎不再提供梯度。
4. 最终训练退化为“带着 Stage1 初始化的普通 3D 训练”，而不是持续有效的知识迁移。

这是当前 D1 效果不明显的最核心原因。

### 9.3 Teacher 强在相对姿态，不强在绝对 3D 定位

T0 teacher 的结果：

```text
best MPJPE = 168.77
best pelvis MPJPE = 52.25
best PA-MPJPE = 32.50
```

这个差异说明 teacher 对人体相对结构建模很好，但对绝对 3D 坐标仍不完美。原因是 2D skeleton 本身缺少：

1. 真实深度。
2. 全局空间位置。
3. 绝对尺度。
4. 与 CSI 坐标系完全一致的物理观测。

因此，如果把 teacher token 当成“绝对 3D 定位知识”，可能会迁移失败。更合理的迁移对象应该是：

```text
相对骨架结构、关节关系、动作时序、骨长/骨向量关系
```

而不是直接指望 2D skeleton teacher 解决 CSI 的绝对 x/y/z 估计。

### 9.4 MMFi CSI-only 本身存在训练不稳定

B0 和 B1 已经显示 MMFi 3D 训练存在 early best 和后期退化的问题：

```text
B0 best 242.56 @ epoch 10, final 255.48 @ epoch 80
B1 best 226.08 @ epoch 8, final 533.06 @ epoch 80
```

D1 Stage2 也呈现类似现象：

```text
best 240.30 @ epoch 6
epoch 74 = 263.99
```

这说明问题不只是蒸馏，也包括 MMFi CSI 输入、stem、PETR 查询机制和训练 schedule 的稳定性。

尤其是 D1 中出现过 z 方向误差突然升高到 350、407、499 的情况，说明模型对深度或全局坐标的估计会发生阶段性崩坏。

### 9.5 当前提升幅度可能只是随机波动

B0 best：

```text
242.56
```

D1 Stage2 best：

```text
240.30
```

差值：

```text
2.26
```

相对提升约：

```text
0.9%
```

在当前只跑一个 seed 的情况下，这个幅度不能排除随机波动、初始化差异、训练 schedule 差异或 checkpoint early selection 带来的影响。

因此当前不能得出“D1 蒸馏有效”的结论。

### 9.6 Loss 权重不是唯一问题

直觉上可以想到调大 `distill_loss_weight`，但当前问题不是简单“蒸馏 loss 太小”。

因为 Stage2 日志中蒸馏项已经：

```text
loss_token_distill = 0.0000
loss_relation_distill = 0.0000
distill_token_cos = 1.0000
```

在 loss 已经饱和的情况下，单纯调权重可能仍然没有梯度。更根本的问题是蒸馏目标和主预测路径的耦合方式需要改。

## 10. 下一步应该怎么做

### 10.1 先不要直接迁移到 Person-in-WiFi 3D 全量数据

当前 D1 在 MMFi 上还没有证明有效。Person-in-WiFi 3D 还会额外引入：

1. 14 点和 17 点定义差异。
2. 原始 RGB 需要全量提取 2D skeleton。
3. 多人样本中的人员匹配问题。
4. 视觉 skeleton 和 CSI/3D 标签同步噪声。

因此不建议马上把当前 D1 迁移到 PiW3D 全量数据。更合理的是先在 MMFi 上把蒸馏机制验证清楚。

### 10.2 先做最小对照实验，判断当前 D1 的提升来源

建议先补三个低成本对照：

| 编号 | 实验 | 目的 |
|---|---|---|
| C1 | D1-init-only | 加载 Stage1 checkpoint，但 Stage2 关闭 token/relation distill，判断 `240.30` 是否只是初始化带来的 |
| C2 | D3-shuffled-token | 打乱 teacher tokens 后训练，判断是否只是正则化效果 |
| C3 | B0-repeat-same-code | 用当前代码复跑 B0，固定 seed=42，排除代码变动和日志口径差异 |

判断逻辑：

```text
如果 D1 ≈ C1：说明 Stage2 蒸馏项没有贡献，只是初始化作用。
如果 D1 ≈ C2：说明 teacher token 的语义没有被有效利用。
如果 D1 只比 B0 高 1% 以内：不能作为有效提升。
```

### 10.3 核心改法：把蒸馏耦合到主 decoder/query

当前最大问题是 distill queries 是独立辅助分支。下一版应改为：

```text
teacher joint tokens
-> 约束 PETR 主 decoder tokens / keypoint tokens / refine tokens
```

而不是：

```text
teacher joint tokens
-> 约束额外 distill queries
```

建议设计 D4：

```text
D4: decoder-coupled skeleton relation distillation
```

核心思想：

1. 在 `PETRHead` 中暴露 decoder/refine decoder 的中间 token。
2. 从主预测路径中提取和 17 个 keypoints 对应的 token 或 query representation。
3. 将 teacher 的 17 个 joint tokens 对齐到这些主路径 token。
4. 同时蒸馏 joint relation matrix、bone relation、pelvis-relative 结构。

D4 loss 可以设计为：

```text
L = L_3D_abs
  + alpha * L_3D_rel
  + beta  * L_decoder_token
  + gamma * L_joint_relation
  + delta * L_bone
```

其中：

```text
L_decoder_token: 主 decoder token 与 teacher joint token 对齐
L_joint_relation: 17 个关节 token 的关系矩阵对齐
L_bone: 骨向量方向或骨长比例约束
```

这样 teacher 知识会直接进入主 3D 预测路径，而不是停留在独立辅助分支。

### 10.4 蒸馏对象应偏向相对结构，而不是绝对坐标

基于 T0 teacher 的结果，下一步更应该蒸馏：

```text
joint relation
bone direction
bone length ratio
pelvis-relative pose
temporal consistency
```

不建议把 teacher 的 `pred_3d` 直接当强监督。因为 teacher 的绝对 3D 仍有较大误差，直接监督可能把 2D skeleton 的深度歧义传给 student。

可以考虑：

```text
L_rel_pose = SmoothL1(
  normalize(pred_3d - pelvis(pred_3d)),
  normalize(teacher_pred_3d - pelvis(teacher_pred_3d))
)
```

但该项必须小权重，并且最好只作为结构先验，不替代真实 `ground_truth.npy`。

### 10.5 训练策略应缩短并早停

当前 MMFi 的多个实验都在 early epoch 出现 best：

| 实验 | best epoch |
|---|---:|
| B0 | `10` |
| B1 | `8` |
| D1 Stage2 | `6` |

这说明 80 epoch 对当前 MMFi 设置可能偏长，后期训练容易退化。

建议下一轮：

1. 先跑 `20` epoch 快速验证。
2. 每 epoch 评估。
3. 重点观察前 `10` epoch。
4. 加 early stopping 或只比较 best within 20。
5. 降低 lr，例如从 `1e-4` 试到 `2e-5` 或对 backbone/head 使用不同 lr。

### 10.6 预处理和模型主体仍需要同步改进

当前 D1 的蒸馏没有明显提升，不代表“视觉知识迁移无效”。更可能是：

```text
CSI 表征本身不稳定 + 蒸馏耦合位置不对
```

已有 MMFi B1 表明 origin-style stem 的 early best 可以到 `226.08`，虽然后期崩坏。这说明预处理或 stem 改造确实可能带来更大上限，但稳定性没有解决。

因此后续总方案应拆成三条线并逐步组合：

| 方向 | 目标 | 当前判断 |
|---|---|---|
| 预处理 | 让 CSI 输入更稳定、更保留空间定位信息 | 必须继续做，但不能只看 2D 经验 |
| 主体网络 | 让 PETR 更适配 MMFi CSI 和 17 点 3D | 需要暴露 decoder token，并减少不稳定 |
| 蒸馏 | 让 2D skeleton 知识进入主预测路径 | 当前 D1 独立分支不够 |

最终比较应采用：

```text
B0 baseline
-> improved preprocess/stem
-> improved main model
-> + coupled distillation
```

不要一次性叠加所有改动，否则无法判断提升来源。

## 11. 推荐的下一轮实验顺序

### 11.1 立即执行的实验

优先级最高的是验证当前 D1 的真实贡献：

1. `D1-init-only`
   - 复用 Stage1 checkpoint。
   - Stage2 关闭 `distill_loss_weight` 和 `relation_loss_weight`。
   - 如果结果接近 D1，说明 Stage2 蒸馏没有作用。

2. `D3-shuffled-token`
   - 使用 `shuffle_teacher_tokens=True`。
   - 如果结果接近 D1，说明 token 语义没有被有效利用。

3. `B0-repeat-same-code`
   - 在当前分支和当前代码环境下复跑 B0。
   - 避免把代码修正、随机性或环境差异误认为蒸馏提升。

### 11.2 下一版方法实验

如果上述对照证明当前 D1 确实不够，应进入 D4：

```text
D4 decoder-coupled skeleton relation distillation
```

最小实现步骤：

1. 修改 `PETRHead`，在 `forward_train` 中可选返回 decoder/refine decoder 中间 tokens。
2. 在 `PETRDistill` 中使用主预测路径 token 计算 token/relation loss。
3. 删除或弱化独立 `distill_queries` 分支。
4. 新增 joint relation 和 bone-level loss。
5. 先跑 20 epoch 小实验。
6. 与 B0-repeat、D1、D1-init-only、D3-shuffled 做对照。

### 11.3 是否迁移到 Person-in-WiFi 3D

迁移条件建议设为：

```text
MMFi 上 D4 相比 B0-repeat 至少稳定提升 5%；
D3-shuffled 明显低于 D4；
不同 seed 下趋势一致；
best 和 final 不出现严重背离。
```

满足这些条件后，再考虑迁移到 Person-in-WiFi 3D。否则直接迁移会把 MMFi 上尚未解决的问题带入更复杂的数据集。

## 12. 当前结论

当前 D1 实验已经完成了方法闭环：

```text
2D skeleton teacher 训练成功；
teacher tokens 离线导出成功；
CSI student Stage1 token pretrain 成功；
Stage2 3D + distill 训练正在运行并已得到结果。
```

但从结果看，当前方案还不能证明蒸馏有效：

```text
D1 Stage2 best = 240.30
B0 best = 242.56
提升只有约 2.26，且后续 epoch 退化到 260 左右。
```

最关键的问题不是“有没有 teacher”，而是：

```text
teacher token 没有足够强地约束主 3D 预测路径。
```

下一步不应简单调参或直接扩大到 Person-in-WiFi 3D，而应先做对照实验确认 D1 的真实贡献，然后把蒸馏从独立辅助分支改为主 decoder/query 路径上的结构蒸馏。

