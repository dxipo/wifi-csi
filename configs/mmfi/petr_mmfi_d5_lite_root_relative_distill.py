_base_ = './petr_mmfi_d5_root_relative_distill.py'

# D5-lite keeps the root-relative structure, but makes the finetuning less
# aggressive so the B0 checkpoint is not pushed away from its good solution.
model = dict(
    bbox_head=dict(
        root_decoupled=True,
        decoder_token_distill_weight=0.2,
        decoder_relation_distill_weight=0.05,
        rel_pose_loss_weight=2.0,
        bone_loss_weight=1.0,
        root_pose_loss_weight=2.0,
        axis_pose_loss_weight=1.0,
        axis_loss_weights=(1.0, 1.0, 1.5)))

optimizer = dict(type='AdamW', lr=5e-6, weight_decay=0.0001)
lr_config = dict(policy='step', step=[6])
runner = dict(type='EpochBasedRunner', max_epochs=8)

checkpoint_config = dict(interval=1, max_keep_ckpts=8)
work_dir = 'result/mmfi_d5_lite_root_relative_distill'
load_from = 'result/mmfi_b0_3d_baseline/best_mpjpe_epoch_10.pth'
resume_from = None
