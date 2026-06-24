_base_ = './petr_mmfi_d5_lite_root_relative_distill.py'

# D7 tests an explicit root-relative output decomposition.
# The final refine output is composed as:
#   pose_3d = relative_pose_from_keypoint_branch + root_from_root_head
# This isolates absolute translation/root estimation from body-shape learning.
model = dict(
    bbox_head=dict(
        root_decoupled=False,
        root_relative_decoupled=True,
        root_relative_detach_base=True,
        decoder_token_distill_weight=0.0,
        decoder_relation_distill_weight=0.0,
        rel_pose_loss_weight=3.0,
        bone_loss_weight=1.0,
        root_pose_loss_weight=3.0,
        axis_pose_loss_weight=1.0,
        axis_loss_weights=(2.0, 1.0, 1.5)))

optimizer = dict(type='AdamW', lr=5e-6, weight_decay=0.0001)
optimizer_config = dict(grad_clip=dict(max_norm=0.1, norm_type=2))
lr_config = dict(policy='step', step=[6])
runner = dict(type='EpochBasedRunner', max_epochs=8)

checkpoint_config = dict(interval=1, max_keep_ckpts=8)
work_dir = 'result/mmfi_d7_root_relative_heads'
load_from = 'result/mmfi_b0_3d_baseline/best_mpjpe_epoch_10.pth'
resume_from = None
