# MMFi 3D 姿态估计与两阶段蒸馏实验计划 - 2026-06-04

## 1. 实验目标

本实验计划的核心目标是：在保持推理阶段只使用 CSI 信号的前提下，利用 MMFi 数据集中同时存在的 3D 骨骼标签和 2D 视觉骨架信息，构建可解释、可对照的 3D 姿态估计实验。

最终主任务：

```text
CSI -> 3D human pose
```

训练阶段可用信息：

```text
wifi-csi/*.mat       -> CSI 输入
ground_truth.npy     -> 3D 主监督标签
rgb/framexxx.npy     -> 2D 骨架点，用于蒸馏或视觉先验
```

推理阶段可用信息：

```text
wifi-csi/*.mat only
```

因此，2D 信息不作为最终模型输入，不改变 CSI-only 的推理设定，只作为训练期 privileged information。

## 2. 已确认的数据事实

### 2.1 MMFi 数据结构

本地数据目录：

```text
/home/xl/Downloads/MMFi_Dataset/MMFi_unzipped
```

样例目录：

```text
/home/xl/Downloads/MMFi_Dataset/MMFi_unzipped/E01/S01/A01
```

该目录下包含：

```text
ground_truth.npy
rgb/
depth/
infra1/
infra2/
lidar/
mmwave/
wifi-csi/
```

已抽查样例：

```text
ground_truth.npy shape = (297, 17, 3)
rgb/frame001.npy shape = (17, 2)
wifi-csi/frame001.mat:
  CSIamp   shape = (3, 114, 10)
  CSIphase shape = (3, 114, 10)
```

全量检查：

```text
ground_truth.npy 文件数量 = 1080
```

说明每个 action 目录都有对应的 3D 标签。

### 2.2 MMFi 3D 标签含义

`ground_truth.npy` 是 MMFi 中计划用于 3D 姿态估计的主监督标签。

每个 action 的 `ground_truth.npy` 组织方式为：

```text
num_frames x 17 x 3
```

其中：

```text
num_frames: 当前 action 的帧数
17:         COCO 风格 17 个身体关键点
3:          3D 坐标
```

从数值范围看，坐标量级符合米制空间坐标。例如 `E01/S01/A01/ground_truth.npy` 的第一帧中，多数 z 值约为 `3.0`，符合人体与相机/采集平台之间的实际距离量级。

后续计算 MPJPE 时，应按米制坐标处理，并乘以 `1000` 得到毫米：

```text
MPJPE_mm = MPJPE_meter * 1000
```

### 2.3 MMFi 2D 骨架含义

`rgb/framexxx.npy` 不是原始 RGB 图像，而是已经提取好的 2D 骨架点。

单帧格式：

```text
17 x 2
```

这部分不作为 3D 主任务标签，而作为视觉姿态先验或蒸馏信息。

## 3. 关键点评估与 14/17 点处理

### 3.1 MMFi 的 17 点格式

MMFi 的 17 点可以按 COCO-17 关键点顺序理解：

```text
0  nose
1  left_eye
2  right_eye
3  left_ear
4  right_ear
5  left_shoulder
6  right_shoulder
7  left_elbow
8  right_elbow
9  left_wrist
10 right_wrist
11 left_hip
12 right_hip
13 left_knee
14 right_knee
15 left_ankle
16 right_ankle
```

### 3.2 Person-in-WiFi 3D 的 14 点格式

当前 `origin_paper` 代码中，14 点逻辑与 CrowdPose-14 更接近。代码中的定义位于：

```text
opera/datasets/crowd_pose.py
```

14 点顺序为：

```text
0  left_shoulder
1  right_shoulder
2  left_elbow
3  right_elbow
4  left_wrist
5  right_wrist
6  left_hip
7  right_hip
8  left_knee
9  right_knee
10 left_ankle
11 right_ankle
12 top_head
13 neck
```

### 3.3 能否从 17 点直接筛选出 14 点

结论：不能完整直接筛选。

COCO-17 中可以直接对应 CrowdPose-14 的 12 个肢体点：

```text
CrowdPose14 0  left_shoulder <- COCO17 5
CrowdPose14 1  right_shoulder <- COCO17 6
CrowdPose14 2  left_elbow    <- COCO17 7
CrowdPose14 3  right_elbow   <- COCO17 8
CrowdPose14 4  left_wrist    <- COCO17 9
CrowdPose14 5  right_wrist   <- COCO17 10
CrowdPose14 6  left_hip      <- COCO17 11
CrowdPose14 7  right_hip     <- COCO17 12
CrowdPose14 8  left_knee     <- COCO17 13
CrowdPose14 9  right_knee    <- COCO17 14
CrowdPose14 10 left_ankle    <- COCO17 15
CrowdPose14 11 right_ankle   <- COCO17 16
```

但 CrowdPose-14 的：

```text
12 top_head
13 neck
```

在 COCO-17 中没有完全一致的原始点。

可构造近似：

```text
neck = (left_shoulder + right_shoulder) / 2
top_head = 由 nose / eyes / ears 估计
```

但这已经是派生点，不是直接筛选。它会引入额外定义误差。

### 3.4 推荐处理方式

推荐不要强行统一 14 点和 17 点。

建议：

```text
Person-in-WiFi 3D:
  保持官方 14 x 3 输出和官方 MPJPE 评价方式

MMFi:
  保持 17 x 3 输出，使用 ground_truth.npy 训练和评价

跨数据集或联合训练:
  使用共享 CSI encoder + dataset-specific pose head
```

联合模型结构建议：

```text
shared CSI encoder / transformer
  -> Person-in-WiFi head: 14 x 3
  -> MMFi head:           17 x 3
```

这样可以共享 CSI 表征，但不强行把两个数据集的骨架定义混在一起。

## 4. 评价指标设计

### 4.1 Person-in-WiFi 3D 指标

当前 `origin_paper` 代码中的 MPJPE 是直接在释放的 3D 坐标系中计算：

```text
MPJPE = mean_j || pred[j] - gt[j] || * 1000
```

该代码还统计：

```text
mpjpeh: x / horizontal dimension error
mpjpev: y / vertical dimension error
mpjped: z / depth dimension error
```

该指标不做 pelvis 对齐。

### 4.2 MMFi 指标

MMFi 上建议至少报告三类指标。

第一类，绝对 MPJPE：

```text
MPJPE_abs = mean_j || pred[j] - gt[j] || * 1000
```

它同时惩罚人体姿态形状错误和整个人的空间位置偏移。

第二类，pelvis-aligned MPJPE：

对 COCO-17，可以定义：

```text
pelvis = (left_hip + right_hip) / 2
       = (joint[11] + joint[12]) / 2
```

计算前先对齐骨盆：

```text
pred_rel[j] = pred[j] - pred_pelvis
gt_rel[j]   = gt[j] - gt_pelvis

MPJPE_pelvis = mean_j || pred_rel[j] - gt_rel[j] || * 1000
```

该指标主要评价人体相对姿态形状，对全局平移误差不敏感。

第三类，PA-MPJPE：

对预测骨架和 GT 骨架做 Procrustes alignment 后再计算误差。该指标会进一步消除平移、旋转和尺度差异，更关注关节点相对结构质量。

### 4.3 指标对比注意事项

Person-in-WiFi 和 MMFi 的 MPJPE 不能直接横向比较，原因包括：

```text
1. 关键点数量不同：14 vs 17
2. 坐标系不同
3. 评价口径不同：Person-in-WiFi 原代码不做 pelvis 对齐，MMFi 官方指标可能包含 pelvis 对齐
4. 数据采集环境、CSI 设备、动作类型不同
```

因此，建议在论文或报告中分别汇报：

```text
Person-in-WiFi 3D official MPJPE
MMFi MPJPE_abs / MPJPE_pelvis / PA-MPJPE
```

## 5. 主体模型设计

### 5.1 当前 origin_paper 主体结构

当前 `origin_paper` 的 CSI 输入路径并不是常规图像 backbone，而是：

```text
CSI preprocessing
  -> reshape to tokens
  -> Linear(60, 256)
  -> PETR transformer
  -> 3D pose head
```

其中 `Linear(60, 256)` 位于：

```text
opera/models/detectors/petr.py
```

它可以理解为最简单的 CSI stem。

### 5.2 stem 的含义

`stem` 指网络最前端的输入适配模块。

它的作用是把原始输入变成主干网络或 Transformer 能处理的 embedding。

在图像模型中，stem 常见形式是：

```text
image -> conv stem -> feature map
```

在当前 CSI 模型中，stem 是：

```text
CSI token with 60 dims -> Linear(60, 256) -> transformer embedding
```

对 MMFi 来说，CSI 格式变成：

```text
CSIamp   3 x 114 x 10
CSIphase 3 x 114 x 10
```

所以不能直接复用写死的 `Linear(60, 256)`。需要设计新的 MMFi CSI stem。

### 5.3 MMFi CSI stem 推荐设计

建议将 `CSIamp` 和 `CSIphase` 合并为一个输入张量：

```text
CSI input shape candidate:
  6 x 114 x 10

其中:
  6 = 3 antennas/channels from amplitude + 3 antennas/channels from phase
```

推荐 stem：

```text
CSIamp/phase
  -> normalization
  -> Conv2D stem
  -> flatten spatial/time dimensions to tokens
  -> Linear / LayerNorm
  -> PETR transformer
```

示例结构：

```text
Input: B x 6 x 114 x 10

Conv stem:
  Conv2d(6, 64, kernel=3, padding=1)
  BatchNorm / LayerNorm
  ReLU / GELU
  Conv2d(64, 128, kernel=3, stride=(2,1), padding=1)
  ReLU / GELU
  Conv2d(128, 256, kernel=3, stride=(2,1), padding=1)

Tokenization:
  B x 256 x H x W
  -> B x (H*W) x 256

Transformer:
  PETR encoder / decoder

Pose head:
  17 x 3 for MMFi
```

也可以加入显式 embedding：

```text
antenna embedding
subcarrier embedding
time embedding
amp/phase embedding
```

这些结构有助于模型知道不同维度的物理含义，而不是把 CSI 简单当作无结构向量。

## 6. 2D-to-3D Teacher 设计

### 6.1 为什么需要新的 teacher

原来的蒸馏方案是：

```text
RGB image -> VitPose -> intermediate token
CSI model -> fit VitPose token
```

但 MMFi 公开数据中的 `rgb/framexxx.npy` 已经是 2D 骨架点，不是原始 RGB 图像。因此不能继续直接提取 VitPose 图像 token。

新的 teacher 应改为：

```text
2D skeleton -> 2D-to-3D teacher -> teacher token
```

这里的 teacher token 是为了完成 2D 到 3D 姿态恢复任务而学习出来的中间表征。

### 6.2 Teacher 输入输出

推荐使用时序窗口，而不是单帧：

```text
Input:
  T x 17 x 2

Output:
  center frame 的 17 x 3
```

例子：

```text
T = 9

输入:
  frame096.npy
  frame097.npy
  frame098.npy
  frame099.npy
  frame100.npy
  frame101.npy
  frame102.npy
  frame103.npy
  frame104.npy

输出标签:
  ground_truth.npy[99]

原因:
  frame100 对应 numpy index 99
```

输出标签就是 `ground_truth.npy` 中 center frame 对应的 3D 坐标。

### 6.3 Teacher 网络结构

推荐结构：

```text
Input: B x T x 17 x 2

Joint embedding:
  Linear(2, C)

Add embeddings:
  joint positional embedding
  temporal positional embedding

Encoder:
  Transformer Encoder / Graph Transformer / ST-GCN

teacher_token:
  CLS token
  或 center-frame joint token mean pooling

3D head:
  teacher_token / joint tokens -> 17 x 3
```

简化版本：

```text
B x T x 17 x 2
  -> Linear(2, 128)
  -> temporal + joint positional embedding
  -> Transformer Encoder
  -> center frame tokens
  -> MLP head
  -> 17 x 3
```

更强版本：

```text
B x T x 17 x 2
  -> graph-aware joint embedding
  -> spatio-temporal transformer
  -> teacher_token
  -> root-relative 3D head
  -> root position head
  -> combine to absolute 17 x 3
```

### 6.4 Teacher loss

Teacher 训练目标：

```text
2D skeleton window -> 3D pose
```

基础 loss：

```text
L_teacher = L_3D(teacher_pred_3D, ground_truth_3D)
```

建议拆成：

```text
L_teacher =
  L_root_relative_3D
  + beta * L_root_abs
  + gamma * L_bone
  + delta * L_temporal
```

其中：

```text
L_root_relative_3D:
  对齐 pelvis 后的 3D 姿态误差，学习人体形状

L_root_abs:
  预测 pelvis / root 的绝对位置，学习全局位置

L_bone:
  骨长一致性约束

L_temporal:
  相邻帧预测平滑约束
```

第一版可先只做：

```text
L_teacher = L_3D_abs + 0.5 * L_3D_pelvis
```

等 baseline 跑通后再增加骨长和时序约束。

### 6.5 Teacher token 保存

Teacher 训练完成后，冻结 teacher，并为每个 frame 导出 teacher token。

建议保存结构：

```text
data/mmfi_teacher_tokens/
  E01/
    S01/
      A01/
        frame001.npy
        frame002.npy
        ...
```

每个 token 文件：

```text
shape = (token_dim,)
```

例如：

```text
token_dim = 256 / 512 / 768
```

如果沿用之前 VitPose 蒸馏代码风格，可以继续使用：

```text
token_dim = 768
```

但从 MMFi 2D skeleton teacher 重新训练时，`256` 或 `512` 也足够。重要的是 teacher 和 student 的 token 维度一致。

## 7. CSI Student 与蒸馏设计

### 7.1 Student 主任务

Student 是最终部署模型：

```text
wifi-csi -> CSI student -> 17 x 3
```

训练主监督：

```text
ground_truth.npy
```

推理输入：

```text
wifi-csi only
```

### 7.2 Student token

Student 内部可从 Transformer memory 或 encoder output 中池化出一个表征：

```text
CSI -> stem -> transformer -> memory
memory -> pooling -> student_token
```

与之前蒸馏分支类似：

```text
student_token = token_head(pool(memory))
```

token loss：

```text
L_token = MSE(student_token, teacher_token)
```

可选使用标准化后的 MSE 或 cosine loss：

```text
L_token = MSE(norm(student_token), norm(teacher_token))
```

推荐第一版：

```text
LayerNorm + MSELoss
```

避免 teacher token 数值尺度影响训练稳定性。

## 8. 完整实验流程

### 8.1 T0: 训练 2D-to-3D teacher

目的：

```text
学习 2D skeleton -> 3D pose 的视觉骨架表征
```

输入：

```text
T x 17 x 2
```

标签：

```text
center frame 的 ground_truth.npy
```

输出：

```text
teacher_pred_3D: 17 x 3
teacher_token:   token_dim
```

loss：

```text
L_teacher = L_3D
```

产物：

```text
teacher checkpoint
offline teacher tokens
```

### 8.2 B0: MMFi CSI 3D baseline

目的：

```text
建立不使用 2D 蒸馏的 CSI-only 3D baseline
```

输入：

```text
wifi-csi
```

输出：

```text
17 x 3
```

标签：

```text
ground_truth.npy
```

loss：

```text
L = L_3D
```

评价：

```text
MPJPE_abs
MPJPE_pelvis
PA-MPJPE
```

该实验是所有蒸馏实验的对照基线。

### 8.3 D1: Stage 1 token 蒸馏预训练

目的：

```text
让 CSI student 的中间表征靠近 2D-to-3D teacher 的中间表征
```

输入：

```text
wifi-csi
```

训练目标：

```text
student_token -> teacher_token
```

loss：

```text
L = L_token
```

该阶段可以不启用 3D pose loss。

产物：

```text
D1 checkpoint
```

这个 checkpoint 代表已经经过视觉骨架知识预训练的 CSI student 初始化。

### 8.4 D2: Stage 2 纯 3D finetune

依赖：

```text
加载 D1 checkpoint
```

目的：

```text
验证 token 蒸馏预训练作为初始化是否能提升 3D MPJPE
```

输入：

```text
wifi-csi
```

输出：

```text
17 x 3
```

标签：

```text
ground_truth.npy
```

loss：

```text
L = L_3D
```

比较：

```text
B0 vs D2
```

如果 D2 优于 B0，说明 Stage 1 预训练带来的 CSI 表征初始化有效。

### 8.5 D3: Stage 2 3D finetune + token regularization

依赖：

```text
加载 D1 checkpoint
```

目的：

```text
验证在 3D finetune 阶段继续保留 teacher token 约束是否更好
```

loss：

```text
L = L_3D + lambda * L_token
```

推荐初始权重：

```text
lambda = 0.01 / 0.03 / 0.05
```

比较：

```text
D2 vs D3
```

如果 D3 优于 D2，说明 Stage 2 中继续使用视觉结构约束有帮助。

### 8.6 D4: shuffled token 负对照

目的：

```text
验证提升是否来自正确对齐的 2D 视觉骨架知识，而不是额外 loss 或训练扰动
```

做法：

```text
打乱 teacher_token 和 CSI 样本的对应关系
```

可做两个版本：

```text
D4-a:
  Stage 1 中使用 shuffled teacher_token
  L = L_token
  然后做纯 3D finetune

D4-b:
  Stage 2 中使用 shuffled teacher_token
  L = L_3D + lambda * L_token
```

比较：

```text
D3 vs D4
```

如果 D3 有提升而 D4 无提升，才能说明收益来自正确对齐的 2D teacher 知识。

## 9. 实验依赖关系

这些实验编号不是简单顺序编号，而是实验组。

依赖关系如下：

```text
T0: 训练 teacher
 |
 |-- 导出 teacher token
 |
D1: CSI token 蒸馏预训练
 |
 |-- D2: 加载 D1，纯 3D finetune
 |
 |-- D3: 加载 D1，3D finetune + token loss
 |
 |-- D4: shuffled token 对照

B0: 独立 baseline，不依赖 T0/D1
```

推荐执行顺序：

```text
1. B0: 先跑 CSI 3D baseline
2. T0: 训练 2D-to-3D teacher
3. D1: 跑 token 蒸馏预训练
4. D2: 跑纯 3D finetune
5. D3: 跑 3D + token regularization
6. D4: 跑 shuffled token 负对照
```

## 10. 预期结果与判断标准

### 10.1 最关键比较

第一组：

```text
B0 vs D2
```

判断蒸馏预训练是否有效。

第二组：

```text
D2 vs D3
```

判断 Stage 2 中继续 token loss 是否有效。

第三组：

```text
D3 vs D4
```

判断提升是否来自正确对齐的视觉姿态知识。

### 10.2 成功标准

理想结果：

```text
D2 < B0
D3 <= D2
D4 >= D3
```

即：

```text
蒸馏预训练有效
继续 token regularization 有额外收益或至少不退化
打乱 token 后收益消失
```

### 10.3 可能失败情况

情况一：

```text
D2 与 B0 接近
```

说明 teacher token 初始化帮助不明显，需要检查 token 是否有 3D 任务相关性。

情况二：

```text
D3 比 D2 差
```

说明 Stage 2 token loss 权重过大，干扰 3D 主任务，需要减小 `lambda` 或只用 D2。

情况三：

```text
D4 也提升
```

说明提升可能来自额外正则化或训练扰动，而不是视觉知识，需要重新设计对照或 teacher token。

## 11. 实现模块拆分

### 11.1 MMFiDataset

需要新增或扩展 dataset：

```text
MMFiPoseDataset
```

功能：

```text
1. 建立样本索引:
   E/S/A/frame

2. 读取 CSI:
   wifi-csi/framexxx.mat
   CSIamp / CSIphase

3. 读取 3D 标签:
   ground_truth.npy[frame_idx]

4. 读取 2D 骨架:
   rgb/framexxx.npy

5. 可选读取 teacher token:
   data/mmfi_teacher_tokens/E/S/A/framexxx.npy

6. 支持 temporal window:
   对 2D teacher 输入构造 T x 17 x 2
```

### 11.2 Teacher 训练脚本

建议新增：

```text
tools/train_mmfi_2d_to_3d_teacher.py
```

或复用当前训练框架，新增配置：

```text
configs/mmfi/teacher_2d_to_3d.py
```

输出：

```text
result/mmfi_teacher_2d_to_3d/
```

### 11.3 Teacher token 导出脚本

建议新增：

```text
tools/export_mmfi_teacher_tokens.py
```

输出：

```text
data/mmfi_teacher_tokens/
```

### 11.4 MMFi CSI baseline 配置

建议新增：

```text
configs/mmfi/petr_mmfi_3d_baseline.py
```

对应实验：

```text
B0
```

### 11.5 MMFi 蒸馏配置

建议新增：

```text
configs/mmfi/petr_mmfi_stage1_token_distill.py
configs/mmfi/petr_mmfi_stage2_finetune.py
configs/mmfi/petr_mmfi_stage2_token_regularization.py
configs/mmfi/petr_mmfi_shuffled_token_control.py
```

分别对应：

```text
D1
D2
D3
D4
```

## 12. 与原 Person-in-WiFi 3D 实验的关系

Person-in-WiFi 3D 仍然是原论文复现和可比性主线。

保持：

```text
origin_paper branch
14 x 3 标签
官方 MPJPE 计算方式
```

MMFi 是新的数据集实验主线。

MMFi 的作用：

```text
1. 验证 CSI -> 3D pose 在另一套多模态数据上的可行性
2. 利用 MMFi 自带 2D skeleton 和 3D ground truth 设计更完整的蒸馏实验
3. 为后续跨数据集预训练或共享 CSI encoder 提供基础
```

不建议直接把 MMFi 和 Person-in-WiFi 数据混合进同一个 head 训练，除非完成关键点定义、坐标系和评价指标的统一。

更稳妥方式：

```text
先分别完成:
  Person-in-WiFi 14 点 3D baseline / improved model
  MMFi 17 点 3D baseline / distillation model

再考虑:
  shared encoder + dataset-specific head
```

## 13. 当前最优先的下一步

建议按以下顺序推进：

```text
1. 编写 MMFi 数据索引检查脚本
   确认每个 action 中 ground_truth、rgb、wifi-csi 帧数一致

2. 实现 MMFiPoseDataset
   先只返回 CSI 和 ground_truth，不加入蒸馏

3. 实现 B0 baseline
   跑通 CSI -> 17 x 3

4. 实现 MMFi 评价指标
   MPJPE_abs / MPJPE_pelvis / PA-MPJPE

5. 训练 2D-to-3D teacher
   验证 2D skeleton 是否能较好恢复 3D

6. 导出 teacher token

7. 实现 D1/D2/D3/D4 蒸馏实验
```

第一阶段不要同时改太多内容。优先跑通：

```text
MMFiDataset + B0 baseline + MMFi metrics
```

只有 B0 可靠后，蒸馏实验的提升才有解释意义。

## 14. 总结

本实验计划最终形成两条清晰主线：

```text
主线 A:
  Person-in-WiFi 3D 官方复现与改进
  14 x 3
  官方 MPJPE

主线 B:
  MMFi 3D 姿态估计与 2D-to-3D teacher 蒸馏
  17 x 3
  ground_truth.npy
  2D skeleton as teacher information
```

MMFi 中的 `ground_truth.npy` 使 3D 主任务成立；`rgb/framexxx.npy` 的 2D 骨架点则为训练期蒸馏提供了合理信息源。

最终模型在推理阶段仍然只使用 CSI 信号，这一点应作为实验设计和论文表述的核心约束。
