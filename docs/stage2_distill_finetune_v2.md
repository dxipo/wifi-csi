# SDP Power X-Fall Stage2 Distillation Finetune V2

## Background

This experiment starts from the recovered baseline:

- Branch: `recover_42_sdp_powerxfall_baseline`
- Baseline result: `result/42_wifipose_2d_baseline_SDP_offline_power_xfall`
- Baseline checkpoint: `result/42_wifipose_2d_baseline_SDP_offline_power_xfall/latest.pth`
- Baseline final metric: `mpjpe=98.5397`, `mpjpe_x=73.0389`, `mpjpe_y=45.5856`
- Baseline best metric observed in log: epoch 391, `mpjpe=77.6551`

The previous stage2 distillation run was:

- Branch: `sdp_powerxfall_stage2_distill`
- Result: `result/46_wifipose_2d_baseline_SDP_offline_power_xfall_stage2_distill`
- Final metric: `mpjpe=150.1127`, `mpjpe_x=113.3297`, `mpjpe_y=78.0895`
- Best observed metric: epoch 57, `mpjpe=81.7492`

The previous run showed that token distillation was not always harmful, because early epochs reached about `81-82` MPJPE. The final checkpoint became worse mainly because the already-trained baseline was fine-tuned for another 450 epochs with the original initial learning rate.

## Main Changes

This v2 experiment changes stage2 into a short, low-learning-rate fine-tuning run.

### Training Schedule

- Old: `max_epochs=450`
- New: `max_epochs=100`

Reason: the previous run reached its best result around epoch 57 and degraded later. A 100 epoch schedule is enough to cover the useful stage2 window while reducing drift.

### Learning Rate

- Old: `lr=2e-5`
- New: `lr=2e-6`

Reason: `load_from` loads only model weights, not optimizer state or epoch state. The previous stage2 therefore restarted from a high learning rate even though the baseline had already converged. This version uses the late-stage baseline learning-rate scale.

### LR Step

- Old: `step=[400]`
- New: `step=[60, 85]`

Reason: with 100 epochs, the learning rate should decay inside the fine-tuning window. The first decay happens near the previous best region.

### Token Loss Weight

- Old: `loss_token.weight=0.1`
- New: `loss_token.weight=0.03`

Reason: token supervision is auxiliary. It should guide representation learning without overpowering the keypoint objective or pulling shared features away from pose estimation.

### Best Checkpoint Saving

- Old: only periodic checkpoints and `latest.pth`
- New: `evaluation = dict(interval=1, metric='mpjpe', save_best='mpjpe', rule='less')`

Reason: the previous stage2 run was much better at early checkpoints than at the final checkpoint. This version records the best validation MPJPE checkpoint directly.

### Checkpoint Retention

- Old: `max_keep_ckpts=100`
- New: `max_keep_ckpts=30`

Reason: the run is shorter, and best-checkpoint saving is enabled. Keeping 30 periodic checkpoints is enough for analysis.

## Unchanged Settings

- Dataset: `all_single_train_data_hold_out_sdp_offline_power_xfall`
- Test set: `all_single_test_data_hold_out_sdp_offline_power_xfall`
- CSI feature directory: `csi_sdp_offline`
- SDP layout: `wtn`
- Batch size: `samples_per_gpu=32`
- Student initialization: `result/42_wifipose_2d_baseline_SDP_offline_power_xfall/latest.pth`
- Distillation target: per-sample `token/*.npy`, shape `(768,)`
- Stage2 loss form: keypoint losses plus `loss_token`

## Result Directory

```text
result/47_wifipose_2d_baseline_SDP_offline_power_xfall_stage2_distill_finetune_v2
```

## Expected Interpretation

This run tests whether stage2 token distillation helps when used as a conservative fine-tuning signal. The primary comparison should use the best validation checkpoint, not just `latest.pth`.

If this v2 run still underperforms the baseline, the next ablation should be:

1. Load the same baseline checkpoint.
2. Use the same 100 epoch and `2e-6` fine-tuning schedule.
3. Disable `loss_token`.

That will separate plain fine-tuning drift from token-distillation conflict.
