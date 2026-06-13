# MMFi 2D Skeleton Token 蒸馏验证方案 - 2026-06-13

## 1. 实验目标

本阶段目标是验证以下路线是否能稳定提升 CSI-only 3D pose estimation：

```text
2D skeleton teacher
+ pose-image 时空表示
+ Transformer pose tokens
+ token / relation distillation
+ CSI-only 3D student
```

核心问题不是马上追求最终最优结果，而是先回答：

```text
2D skeleton teacher 中的人体结构 token / 关系知识，能否稳定提升 CSI-only 3D MPJPE？
```

如果该问题在 MMFi 上成立，再迁移到 Person-in-WiFi3D 单人子集和全量多人数据。

## 2. 为什么先选择 MMFi

先在 MMFi 上做蒸馏验证，原因如下：

1. MMFi 已经提供 `rgb/frame*.npy`，形状为 `17 x 2` 的 2D skeleton，不需要先对 RGB 全量跑 VitPose。
2. MMFi 同时提供 `ground_truth.npy`，可作为 3D 主监督标签。
3. MMFi 当前没有 Person-in-WiFi3D 全量多人中的 2D/3D 人员匹配问题。
4. 当前 Person-in-WiFi3D 的 RGB 2D skeleton 全量提取成本较高，且多人匹配会引入额外噪声。

因此第一阶段使用 MMFi 验证方法闭环，第二阶段再迁移到 Person-in-WiFi3D。

## 3. 已有实验对方案的约束

已有实验给出几个重要约束：

| 实验 | 结论 |
|---|---|
| `origin_paper` 3D 复现 | best MPJPE `117.20`，final `119.77`，是 Person-in-WiFi3D 当前最可靠 3D 主基线 |
| `origin_paper_sdp_power_xfall_3d` | best MPJPE `228.50`，final `252.52`，说明 SDP power_xfall 不适合直接替换 3D 主输入 |
| MMFi B0 | best MPJPE `242.56`，final `255.48`，说明 MMFi 3D baseline 可跑通 |
| MMFi B1 | best MPJPE `226.08`，final `533.06`，说明盲目迁移 origin-style stem 不稳定 |
| 2D R3 | `SDP power_xfall` 是 2D 单人任务强基线，但其优势不能直接外推到 3D |
| 2D R7/R8/R9 蒸馏 | 旧蒸馏没有超过最强 baseline，说明蒸馏对象和权重需要重新设计 |

因此本阶段不同时大改预处理、主干和蒸馏。第一轮只在 MMFi B0 旁边加入 2D skeleton token 蒸馏。

## 4. 数据与预处理

### 4.1 主监督

```text
ground_truth.npy -> 17 x 3 3D pose
```

作为 3D 主任务标签。

### 4.2 Teacher 输入

```text
rgb/frame*.npy -> 17 x 2 2D skeleton
```

以 center frame 为中心构造时间窗口：

```text
pose2d_seq: T x 17 x C
T = 9
C = 3, 即 x_norm, y_norm, valid_mask
```

预处理方法：

1. 缺失或无效点生成 `valid_mask`。
2. 每帧以 pelvis / hip center 或有效点中心做中心化。
3. 每帧以 2D skeleton bbox 尺度归一化。
4. 无效点置零，但通过 mask 通道保留可见性信息。

随后重排为 pose-image：

```text
B x T x K x C -> B x C x T x K
```

这对应 MSS-Former 中“pixel-level spatiotemporal representation”的思想，但这里的像素不是 RGB 像素，而是 skeleton coordinate image。

### 4.3 Student 输入

第一阶段保持 MMFi B0 的 CSI 输入与 stem：

```text
CSIamp + CSIphase -> normalize -> B x 6 x subcarrier x time
```

Student 仍然是 CSI-only。推理阶段不使用 2D skeleton，也不使用 teacher。

## 5. 模型设计

### 5.1 2D Skeleton Teacher

配置文件：

```text
configs/mmfi/skeleton_teacher_t0.py
```

结构：

```text
pose2d_seq
-> pose-image stem, Conv2D
-> Transformer Encoder
-> pose tokens: 1 global token + 17 joint tokens
-> 3D pose head
```

Teacher 输出：

```text
teacher_tokens: 18 x 256
teacher_pred_3d: 17 x 3
```

Teacher loss：

```text
L_teacher = L_abs_3D + L_rel_3D
```

其中：

- `L_abs_3D`：原始 3D 坐标误差；
- `L_rel_3D`：pelvis 对齐后的相对姿态误差。

### 5.2 CSI Student

Student 基于 PETR，新增 `PETRDistill`：

```text
CSI tokens
-> PETR encoder / memory
-> distill queries cross-attention
-> student pose tokens
```

蒸馏目标：

```text
L_token = SmoothL1(student_tokens, teacher_tokens)
```

关系蒸馏：

```text
R = normalize(tokens) @ normalize(tokens).T
L_rel = MSE(R_student, R_teacher)
```

Stage 1：

```text
L = L_token + beta * L_relation
```

Stage 2：

```text
L = L_3D + lambda * L_token + beta * L_relation
```

推荐初始权重：

```text
Stage 1: token=1.0, relation=0.2
Stage 2: token=0.1, relation=0.03
```

## 6. 实验顺序

| 编号 | 配置 | 目的 |
|---|---|---|
| T0 | `configs/mmfi/skeleton_teacher_t0.py` | 训练 `2D skeleton -> 3D` teacher，验证 teacher 是否有足够质量 |
| Export | `tools/dataset_converters/export_mmfi_teacher_tokens.py` | 导出 train/val teacher tokens |
| S0 | `configs/mmfi/petr_mmfi_b0_3d_baseline.py` | 复跑 MMFi CSI-only baseline，固定 seed=42 |
| D1-Stage1 | `configs/mmfi/petr_mmfi_d1_stage1_token_pretrain.py` | CSI student 只做 token/relation 预训练 |
| D1-Stage2 | `configs/mmfi/petr_mmfi_d1_stage2_3d_distill.py` | 加载 Stage1，做 3D 主任务 + 小权重蒸馏 |
| D2 | `configs/mmfi/petr_mmfi_d2_joint_distill_no_stage1.py` | 不做 Stage1，直接 3D + 蒸馏联合训练 |
| D3 | `configs/mmfi/petr_mmfi_d3_shuffled_token_control.py` | 打乱 teacher token，验证提升是否来自有效视觉知识 |

成功标准：

```text
D1 > S0
D1 > D2
D2 > D3
```

如果 D3 也提升，说明可能只是正则化效果，不足以证明视觉知识迁移。

## 7. 运行命令

训练 teacher：

```bash
python tools/train.py configs/mmfi/skeleton_teacher_t0.py --gpu-id 0 --seed 42
```

导出 teacher token：

```bash
python tools/dataset_converters/export_mmfi_teacher_tokens.py \
  configs/mmfi/skeleton_teacher_t0.py \
  --checkpoint result/mmfi_t0_skeleton_teacher/latest.pth \
  --split train \
  --out-dir data/mmfi_teacher_tokens/t0
```

Stage 1：

```bash
python tools/train.py configs/mmfi/petr_mmfi_d1_stage1_token_pretrain.py --gpu-id 0 --seed 42 --no-validate
```

Stage 2：

```bash
python tools/train.py configs/mmfi/petr_mmfi_d1_stage2_3d_distill.py --gpu-id 0 --seed 42
```

## 8. TensorBoard

本分支新增配置均包含：

```python
log_config = dict(
    interval=50,
    hooks=[
        dict(type='TextLoggerHook'),
        dict(type='TensorboardLoggerHook')
    ])
```

因此训练后可在对应 `result/*` 目录查看 TensorBoard event 文件。

## 9. 主要风险

1. 2D skeleton 缺少深度信息，teacher 的绝对 3D 能力可能有限。
2. 如果 teacher 本身 MPJPE 很差，蒸馏会带来负迁移。
3. MMFi B0 baseline 本身仍有波动，需要复跑 S0 作为同分支对照。
4. Token 蒸馏权重过大可能压制 3D 主监督。
5. 当前方案只验证 MMFi，不能直接说明 Person-in-WiFi3D 全量多人可用。

## 10. 后续迁移

如果 MMFi 上 D1 明确优于 S0/D2/D3，则迁移到 Person-in-WiFi3D：

1. 先做单人子集，避免多人匹配问题。
2. 用 VitPose 提取 RGB 2D skeleton。
3. Teacher 输入统一为 2D skeleton sequence。
4. Student 主输入保持 `origin_paper` amp+phase。
5. 再考虑把 SDP / power_xfall 作为辅助 token，而不是替换主输入。
