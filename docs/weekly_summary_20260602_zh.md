# 本周工作总结 - 2026-05-27 至 2026-06-02

## 1. 原论文代码分支整理

- 将 Person-in-WiFi 3D 官方代码整理到 `origin_paper` 分支。
- 删除官方代码中的大文件 `demo` 内容，避免 GitHub 推送时出现大文件限制问题。
- 保留当前目录下已有的 `data/` 数据，因为原论文代码训练依赖这些本地数据。
- 当前工作分支为 `origin_paper`。

## 2. 原论文 3D 训练复现

- 使用原论文 3D 流程进行了训练复现。
- 随机种子设置为 `42`。
- checkpoint 输出目录设置为 `result/origin_paper`。
- 训练已完整跑到 `450` epoch。

主要结果：

- 最优验证结果：epoch `391`，MPJPE `117.1981 mm`
- 最优分项指标：`mpjpeh 51.8670`，`mpjpev 65.7973`，`mpjped 52.4855`
- 最终 epoch 结果：epoch `450`，MPJPE `119.7679 mm`
- 最终分项指标：`mpjpeh 54.1742`，`mpjpev 65.7244`，`mpjped 55.2351`

主要日志文件：

- `result/origin_paper/train_seed42_resume_epoch1_20260528_165536.out`

最终 checkpoint：

- `result/origin_paper/epoch_450.pth`

## 3. MPJPE 计算方式确认

查看 `opera/datasets/wifi_pose.py` 后确认，原论文代码中的 MPJPE 计算方式为：

```text
MPJPE = mean(sqrt((x_gt - x_pred)^2 + (y_gt - y_pred)^2 + (z_gt - z_pred)^2)) * 1000
```

结论：

- 官方代码直接在释放的 `14 x 3` 三维关键点坐标系中计算误差。
- `14 x 3` 标签被当作米制 3D 坐标。
- MPJPE 乘以 `1000` 后以毫米为单位上报。
- 该指标不是 2D 像素误差，也不是 RGB 图像坐标误差。
- 因此，`origin_paper` 分支的实验结果可以作为和原论文对比的主流程。

## 4. RGBD 标定与 3D 到 RGB 投影验证

检查了以下文件：

- `/home/xl/BaiduDownloads/S11_06/calibration.json`
- `/home/xl/BaiduDownloads/S11_06/output.mkv`
- `/home/xl/BaiduDownloads/S11_06/time_list.txt`

主要结论：

- `calibration.json` 是 Azure Kinect 的相机标定文件。
- 其中包含 depth 相机内参、RGB/PV 相机内参，以及 depth/RGB 之间的外参。
- 该文件不是骨骼标注文件。
- 仅使用 `calibration.json` 无法将原论文释放的 `14 x 3` 关键点准确投影到 RGB 图像。
- 直接投影时骨架点没有落到画面中的人体上，说明释放的 3D 标签很可能不是原始 Kinect 相机坐标系，而是经过作者转换后的坐标系。

生成的验证图：

- `result/calibration_check/S11_06_303_projection_variants.png`
- `result/calibration_check/S11_06_303_2d_holdout_overlay.png`
- `result/calibration_check/S11_06_303_3d_to_2d_trainfit_overlay.png`

补充结论：

- 仓库中已有的 2D holdout keypoint 可以准确落在 RGB 帧中的人体上。
- 通过拟合一个额外的世界坐标系到 RGB 图像坐标系的变换，可以粗略把 3D 点投影到图像中。
- 但这种拟合只是近似验证，不能替代作者原始的数据转换流程。

## 5. 当前分支与最近 2D 实验状态

当前分支：

- `origin_paper`

最近一次 2D 实验分支：

- `sdp140_image_conv_patch_ablation`

对应结果目录：

- `result/50_wifipose_2d_baseline_SDP140_conv_patch_lagwindow_centerctx`

最近一次 2D 蒸馏实验分支：

- `sdp_powerxfall_stage2_distill_best_saved_ckpt`

对应结果目录：

- `result/48_wifipose_2d_baseline_SDP_offline_power_xfall_stage2_distill_best_saved_ckpt`

## 6. MMFi 数据集处理

本周还检查并整理了 MMFi 数据集的本地处理状态。

数据目录：

- `/home/xl/Downloads/MMFi_Dataset`

已确认内容：

- `Zipfiles/` 中包含 `S01.zip` 至 `S40.zip` 共 40 个 subject 压缩包。
- 未发现缺失的 subject 压缩包，压缩包总大小约 `77.27 GB`。
- 数据已解压到 `/home/xl/Downloads/MMFi_Dataset/MMFi_unzipped`。
- 解压后结构为 `E01` 至 `E04` 四个环境，每个环境包含 10 个 subject，共 `S01` 至 `S40`。
- 共检查到 `40` 个 subject、`1080` 个 action 目录。
- 每个 action 目录包含 7 类模态：`rgb`、`depth`、`infra1`、`infra2`、`lidar`、`mmwave`、`wifi-csi`。
- 各模态目录数量均为 `1080`，说明解压后的目录结构基本完整。
- 抽查 `E01/S01/A01`，每个模态均有 `297` 帧或文件。
- 全量统计中，每个模态均有 `320760` 个文件。
- `MMFi_action_segments.csv` 已检查，字段为 `Environment, Student, Action, Segments`，用于记录每个动作对应的 frame segment 范围。

当前状态：

- MMFi 数据集目前是独立的本地数据目录，尚未集成到当前 `origin_paper` 代码流程。
- 当前仓库中还没有 MMFi 专用的数据转换脚本或训练配置。
- 后续如果将 MMFi 用于 WiFi/CSI 姿态估计，需要额外编写数据索引与格式转换逻辑，将 `wifi-csi`、RGB/视觉信息和标签对齐到当前训练代码可读取的格式。

风险与注意事项：

- MMFi 是多模态动作数据集，原始组织方式和 Person-in-WiFi 3D 的 `data/wifipose/train_data`、`data/wifipose/test_data` 不一致。
- 不能直接把 MMFi 目录接入当前 `WifiPoseDataset`。
- 需要先明确实验目标是动作识别、2D 姿态、3D 姿态，还是跨模态蒸馏，再决定转换格式。

## 7. 二阶段蒸馏方案回顾

查看并梳理了以下蒸馏相关分支：

- `csi_stage1`
- `csi_stage2`
- `stage2_no_losee`
- `sdp_powerxfall_stage2_distill`
- `sdp_powerxfall_stage2_distill_best_saved_ckpt`
- `sdp140_image-like_stage2_distill`

确认之前实现的蒸馏方案不是简单的 2D auxiliary head，而是 Transformer token 蒸馏：

```text
RGB / VitPose teacher -> 保存 768 维 teacher token
CSI student -> PETR transformer memory -> mean pool -> token_head -> pred_token
loss_token = MSE(pred_token, teacher_token)
```

二阶段逻辑：

- Stage 1：只训练 token 蒸馏，`distill_stage=1`
- Stage 2：加载 Stage 1 checkpoint，进行任务训练，并可选择继续加入 token loss，`distill_stage=2`
- `stage2_no_losee` 是一个重要对照：加载 Stage 1 初始化，但 Stage 2 不再加入 token loss。

针对后续 3D 可比实验的结论：

- 原有 token 蒸馏思路可以迁移到 3D 主流程。
- 最终任务应保持为 `CSI -> 3D pose`。
- 最终指标应使用原论文官方 MPJPE，单位为毫米。
- VitPose/RGB token 只作为训练阶段的表征蒸馏辅助，不作为最终输出空间。

## 8. 主要风险

- 目前 teacher token 主要存在于 `all_single_*_hold_out/token`，并不完整覆盖原论文 `train_data/test_data`。
- 原 2D 蒸馏分支中包含将 keypoint 归一化到 640x360 图像坐标的逻辑，这部分不能直接迁移到 3D 实验。
- 3D 实验中必须保留原论文 `14 x 3` 米制标签和官方 MPJPE 评价方式。
- VitPose token 本质上更偏向 2D 视觉姿态表征，可能对平面结构有帮助，但不一定提升深度维度。
- 后续需要重点观察 `mpjpeh`、`mpjpev` 和 `mpjped`，尤其是 `mpjped` 是否恶化。
- MMFi 的数据结构与原论文数据结构不同，若用于后续实验，需要先完成可靠的数据对齐和格式转换，不能直接混入当前 3D 复现实验。

## 9. 后续建议实验

为了保证和原论文具有可比性，建议后续实验主线为：

```text
B0: origin_paper 3D baseline
    L = L_3D

D1: token 蒸馏预训练
    L = L_token

D2: 加载 D1 初始化，然后纯 3D finetune
    L = L_3D

D3: 加载 D1 初始化，然后 3D finetune + 小权重 token regularization
    L = L_3D + lambda * L_token
```

建议增加一个负对照：

```text
D4: 打乱 teacher token 和 CSI 样本的对应关系
```

如果正确对齐的 token 蒸馏有效，而打乱 token 后无效，才能说明提升确实来自 RGB teacher token 中的人体姿态表征，而不是训练扰动或额外正则化。

## 10. 总体结论

本周主要完成了原论文 3D 代码复现、官方 MPJPE 指标确认、RGBD 标定文件分析、MMFi 数据集本地处理状态核对、2D 与 3D 实验关系梳理，以及后续将二阶段 token 蒸馏迁移到 3D 可比实验的可行性分析。

下一步重点应放在：

- 保持 3D 主流程和官方 MPJPE 上报方式；
- 为完整 3D split 生成或匹配 teacher token；
- 将 token 蒸馏机制迁移到 `origin_paper` 3D 标签流程中；
- 如果使用 MMFi，先完成 MMFi 到目标训练格式的数据转换与样本对齐；
- 通过 D2/D3/D4 等 ablation 判断蒸馏是否真正提升 3D MPJPE。
