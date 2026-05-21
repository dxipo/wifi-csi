# WiFi CSI 实验对比续表

日期：2026-05-19

本文延续 `docs/wifi_csi_experiment_comparison_and_mpjpe_review_20260518.md` 中的主表格式，保留原 R0-R8 记录，并把 2026-05-19 后续补充的实验追加在表格末尾。

注意：

1. R0-R8 的结果来自 2026-05-18 文档中已经整理过的完整训练记录。
2. R9 是新补的 “从 best saved baseline checkpoint 启动 stage2 蒸馏” 实验，已于 2026-05-19 17:19 完成 100 epoch 训练。
3. R9 使用的是 `epoch_165.pth`，因为 `result/42_wifipose_2d_baseline_SDP_offline_power_xfall` 中真实 best observed epoch 391 没有保存 checkpoint；在实际保存下来的 checkpoint 中，`epoch_165.pth` 是可用的 best saved baseline checkpoint。
4. R9 已启用新增评估指标，因此日志中除了旧的 `mpjpe/mpjpe_x/mpjpe_y`，还会记录 `mpjpe_top1`、`mpjpe_top5_oracle`、`mpjpe_top20_oracle`、`mpjpe_oracle_all` 和 AP 系列指标。R0-R8 旧日志没有这些新增指标。

## 主实验汇总表

| 编号 | result 目录 | 主要设置 | max epoch | final MPJPE | best MPJPE | 备注 |
| --- | --- | --- | ---: | ---: | ---: | --- |
| R0 | `42_wifipose_2d_baseline_test_recover` | raw hold-out baseline，`num_query=100` | 450 | 175.3183 / x 114.1612 / y 98.6775 | 59.5218 @ 4 | best 过早，oracle/test 选择偏乐观。 |
| R1 | `42_wifipose_2d_baseline_STFT_offline` | STFT offline，`input_dim=9`，`num_query=100` | 450 | 121.5475 / x 103.9709 / y 42.5375 | 85.1846 @ 3 | final 比 raw 好，但 best 极早。 |
| R2 | `42_wifipose_2d_baseline_SDP_offline` | SDP offline，`input_dim=6`，`num_query=100` | 450 | 302.5016 / x 260.4266 / y 126.3197 | 84.3267 @ 235 | 后期严重退化。 |
| R3 | `42_wifipose_2d_baseline_SDP_offline_power_xfall` | SDP + power_xfall，`input_dim=6`，`num_query=81` | 450 | 98.5397 / x 73.0389 / y 45.5856 | 77.6551 @ 391 | 当前最强主线基线；真实 best epoch 391 未保存 checkpoint。 |
| R4 | `43_wifipose_2d_baseline_SDP300_imagelike_lagwindow_centerctx` | SDP300 image-like，`conv_patch`，shape 约 `(30,825,3)` | 450 | 244.9251 / x 218.6288 / y 66.9030 | 84.9851 @ 264 | best 可用，final 退化明显。 |
| R5 | `44_wifipose_2d_baseline_SDP140_translator_image_lagwindow_centerctx` | SDP140 image-like + translator，shape 约 `(30,270,3)` | 450 | 112.6839 / x 98.2272 / y 35.1492 | 101.3820 @ 106 | 比 R4 final 稳定，但不能证明预处理优劣。 |
| R6 | `45_wifipose_2d_SDP140_image-like_stage2_distill_lagwindow_centerctx` | R5 基础上 stage2 token distill | 450 | 140.4951 / x 101.7064 / y 69.0647 | 102.5173 @ 29 | 蒸馏未带来可见增益。 |
| R7 | `46_wifipose_2d_baseline_SDP_offline_power_xfall_stage2_distill` | R3 基础上 stage2 token distill，450 epoch | 450 | 150.1127 / x 113.3297 / y 78.0895 | 81.7492 @ 57 | 早期接近 R3，但长训练漂移严重。 |
| R8 | `47_wifipose_2d_baseline_SDP_offline_power_xfall_stage2_distill_finetune_v2` | R3 `latest.pth` 基础上低 LR stage2，100 epoch，`loss_token=0.03` | 100 | 117.3972 / x 101.0378 / y 42.4774 | 87.8896 @ 22 | v2 比 R7 final 稳定，但仍未超过 R3 best。 |
| R9 | `48_wifipose_2d_baseline_SDP_offline_power_xfall_stage2_distill_best_saved_ckpt` | 从 R3 best saved checkpoint `epoch_165.pth` 启动 stage2，`lr=2e-6`，100 epoch，`loss_token=0.03` | 100 | 91.7977 / x 72.1762 / y 40.1059 | 83.9841 @ 1 | 比 R8 更稳定，final 也优于 R8；但 best 仍未超过 R3 的 `77.6551 @ 391`。 |

## R9 新增指标记录

R9 是第一组带新增评估指标的训练日志，因此可以额外观察 top-1、top-k oracle 和 AP 指标。下面记录 final epoch 和各指标最佳点。

| 阶段 | epoch | mpjpe_oracle_all | mpjpe_top1 | mpjpe_top5_oracle | mpjpe_top20_oracle | AP | AP50 | AP75 | APm | APl |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| best old-MPJPE / best oracle-all | 1 | 83.9841 | 156.8410 | 89.2316 | 87.5031 | 0.0126 | 0.0597 | 0.0004 | 0.0064 | 0.0155 |
| best top-1 MPJPE | 73 | 91.5114 | 116.8735 | 102.7743 | 98.9988 | 0.3255 | 1.2541 | 0.0174 | 0.0013 | 0.4864 |
| best top-5 oracle | 1 | 83.9841 | 156.8410 | 89.2316 | 87.5031 | 0.0126 | 0.0597 | 0.0004 | 0.0064 | 0.0155 |
| best top-20 oracle | 1 | 83.9841 | 156.8410 | 89.2316 | 87.5031 | 0.0126 | 0.0597 | 0.0004 | 0.0064 | 0.0155 |
| best AP | 61 | 91.8900 | 117.2001 | 102.5515 | 98.9306 | 0.4443 | 1.5782 | 0.0371 | 0.0016 | 0.7872 |
| best AP50 | 60 | 92.8261 | 116.9609 | 103.5618 | 100.1031 | 0.3861 | 1.5829 | 0.0419 | 0.0056 | 0.6032 |
| final checkpoint | 100 | 91.7977 | 116.9121 | 103.4379 | 99.6415 | 0.3129 | 1.2654 | 0.0177 | 0.0012 | 0.4746 |

## R9 结果解读

R9 的目的不是重新证明 R3 baseline 强弱，而是验证一个更合理的蒸馏启动点：不用 R3 的 final/latest checkpoint，而是从可用的 best saved baseline checkpoint 启动。由于真实 best observed epoch 391 没有保存，当前只能使用 `epoch_165.pth` 作为替代。

R9 的旧 MPJPE best 为 `83.9841 @ 1`，低于 R8 的 `87.8896 @ 22`，说明 “从更好的保存点启动” 对 stage2 早期效果有帮助。R9 final 为 `91.7977`，也明显好于 R8 final `117.3972` 和 R7 final `150.1127`，说明它比前两版蒸馏训练更稳定。

但 R9 仍未超过 R3 的 best observed `77.6551 @ 391`。因此当前能得出的结论是：best saved checkpoint 启动的 stage2 蒸馏比 R8/R7 更稳定，但还没有证明蒸馏能超过 `SDP_offline_power_xfall` baseline 的最佳状态。

新增指标暴露了一个重要问题：R9 的 best old-MPJPE 在 epoch 1，为 `mpjpe_oracle_all=83.9841`，但当时 `mpjpe_top1=156.8410`；而 best top-1 MPJPE 出现在 epoch 73，为 `116.8735`。这说明旧 MPJPE 仍然主要反映 “候选池里有没有接近 GT 的 query”，而不是模型最高置信度预测是否准确。后续正式比较时，应同时汇报旧 MPJPE、top-1 MPJPE 和 top-k oracle MPJPE。
