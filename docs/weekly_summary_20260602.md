# Weekly Summary - 2026-05-27 to 2026-06-02

## 1. Origin Paper Branch Setup

- Reorganized the official Person-in-WiFi 3D code into the `origin_paper` branch.
- Removed the large `demo` content from the official-code branch to avoid GitHub large-file push problems.
- Kept the existing local `data/` directory available, because the official training code depends on the dataset already present in the workspace.
- Current working branch is `origin_paper`.

## 2. Official 3D Training Reproduction

- Ran the original Person-in-WiFi 3D training flow with seed `42`.
- Updated the checkpoint output directory to `result/origin_paper`.
- Confirmed that the training uses the official 3D `14 x 3` keypoints and reports MPJPE in millimeters.
- Training completed to `450` epochs.

Key result:

- Best validation result: epoch `391`, MPJPE `117.1981 mm`
- Best component metrics: `mpjpeh 51.8670`, `mpjpev 65.7973`, `mpjped 52.4855`
- Final epoch result: epoch `450`, MPJPE `119.7679 mm`
- Final component metrics: `mpjpeh 54.1742`, `mpjpev 65.7244`, `mpjped 55.2351`

Main log:

- `result/origin_paper/train_seed42_resume_epoch1_20260528_165536.out`

Final checkpoint:

- `result/origin_paper/epoch_450.pth`

## 3. MPJPE Evaluation Understanding

Confirmed from `opera/datasets/wifi_pose.py` that the official evaluation computes:

```text
MPJPE = mean(sqrt((x_gt - x_pred)^2 + (y_gt - y_pred)^2 + (z_gt - z_pred)^2)) * 1000
```

Conclusion:

- The metric is computed directly in the released 3D keypoint coordinate system.
- The `14 x 3` values are treated as meter-scale 3D coordinates.
- The reported MPJPE is therefore millimeter-scale 3D error, not 2D pixel error.
- This makes the `origin_paper` branch suitable for comparable reporting against the paper.

## 4. RGBD Calibration and 3D-to-RGB Projection Investigation

Inspected:

- `/home/xl/BaiduDownloads/S11_06/calibration.json`
- `/home/xl/BaiduDownloads/S11_06/output.mkv`
- `/home/xl/BaiduDownloads/S11_06/time_list.txt`

Findings:

- `calibration.json` is Azure Kinect calibration metadata.
- It contains depth camera intrinsics, RGB/PV camera intrinsics, and depth/RGB extrinsics.
- It does not directly contain skeleton labels.
- The official `14 x 3` keypoints cannot be directly projected to RGB using only this file.
- Direct projection places the skeleton away from the visible person, which suggests the released 3D labels are in an author-processed coordinate system, not raw RGB/depth camera coordinates.

Generated validation images:

- `result/calibration_check/S11_06_303_projection_variants.png`
- `result/calibration_check/S11_06_303_2d_holdout_overlay.png`
- `result/calibration_check/S11_06_303_3d_to_2d_trainfit_overlay.png`

Additional finding:

- Existing 2D holdout keypoints align with the RGB frame.
- A fitted world-to-RGB transform can roughly map the released 3D keypoints into the RGB image, but this is an approximation and not a replacement for the missing author-side coordinate conversion.

## 5. 2D Experiments and Branch Status

Current branch:

- `origin_paper`

Most recent 2D experiment branch:

- `sdp140_image_conv_patch_ablation`

Associated recent result directory:

- `result/50_wifipose_2d_baseline_SDP140_conv_patch_lagwindow_centerctx`

Most recent 2D distillation branch:

- `sdp_powerxfall_stage2_distill_best_saved_ckpt`

Associated result directory:

- `result/48_wifipose_2d_baseline_SDP_offline_power_xfall_stage2_distill_best_saved_ckpt`

## 6. Distillation Strategy Review

Reviewed previous distillation branches:

- `csi_stage1`
- `csi_stage2`
- `stage2_no_losee`
- `sdp_powerxfall_stage2_distill`
- `sdp_powerxfall_stage2_distill_best_saved_ckpt`
- `sdp140_image-like_stage2_distill`

Clarified the actual distillation design:

```text
RGB / VitPose teacher -> saved 768-d teacher token
CSI student -> PETR transformer memory -> pooled token -> token_head -> pred_token
loss_token = MSE(pred_token, teacher_token)
```

Two-stage logic:

- Stage 1: token-only pretraining, `distill_stage=1`
- Stage 2: task training initialized from Stage 1, optionally with token regularization, `distill_stage=2`
- `stage2_no_losee` is an important ablation: Stage 1 initialization is used, but Stage 2 removes token loss.

Conclusion for future 3D-comparable work:

- The previous token distillation idea can be adapted to the 3D official workflow.
- The final task should remain `CSI -> 3D pose`.
- The final reported metric should remain official MPJPE in millimeters.
- VitPose/RGB token distillation should be treated as auxiliary representation learning, not as the final output space.

## 7. Main Risks Identified

- Existing teacher tokens are mainly available under `all_single_*_hold_out/token`, not under the full official `train_data/test_data` split.
- The original 2D distillation code changes keypoint handling to normalized 2D coordinates; that part should not be copied into the 3D branch.
- For 3D experiments, the `14 x 3` meter-scale labels and official MPJPE code must be preserved.
- VitPose tokens are 2D visual representations and may help horizontal/vertical pose structure more than depth. Future 3D experiments must track `mpjpeh`, `mpjpev`, and especially `mpjped`.

## 8. Recommended Next Experiments

For comparable 3D reporting:

```text
B0: origin_paper 3D baseline
    L = L_3D

D1: token distillation pretraining
    L = L_token

D2: Stage 1 initialization + pure 3D finetuning
    L = L_3D

D3: Stage 1 initialization + 3D finetuning with small token regularization
    L = L_3D + lambda * L_token
```

Important ablation:

```text
D4: shuffled teacher-token control
```

This checks whether token alignment with the same frame is truly useful, rather than only adding training noise or extra regularization.
