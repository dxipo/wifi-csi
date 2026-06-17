_base_ = './petr_mmfi_d4_decoder_coupled_distill.py'

model = dict(
    bbox_head=dict(
        root_decoupled=True,
        decoder_token_distill_weight=1.0,
        decoder_relation_distill_weight=0.1,
        rel_pose_loss_weight=5.0,
        bone_loss_weight=2.0,
        root_pose_loss_weight=5.0,
        axis_pose_loss_weight=2.0,
        axis_loss_weights=(2.0, 1.0, 2.0)))

optimizer = dict(type='AdamW', lr=1e-5, weight_decay=0.0001)
lr_config = dict(policy='step', step=[8])
runner = dict(type='EpochBasedRunner', max_epochs=12)

checkpoint_config = dict(interval=2, max_keep_ckpts=10)
work_dir = 'result/mmfi_d5_root_relative_distill'
load_from = 'result/mmfi_b0_3d_baseline/best_mpjpe_epoch_10.pth'
resume_from = None
