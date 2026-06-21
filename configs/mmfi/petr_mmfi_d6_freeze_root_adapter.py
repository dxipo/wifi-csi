_base_ = './petr_mmfi_d5_lite_root_relative_distill.py'

# D6-freeze isolates the root correction idea:
# keep the B0 feature extractor and existing PETR head fixed, and train only
# the newly added root adapter. This tests whether absolute root/x-z correction
# can improve MPJPE without moving the strong B0 solution.
model = dict(
    bbox_head=dict(
        root_decoupled=True,
        decoder_token_distill_weight=0.0,
        decoder_relation_distill_weight=0.0,
        rel_pose_loss_weight=0.0,
        bone_loss_weight=0.0,
        root_pose_loss_weight=3.0,
        axis_pose_loss_weight=1.0,
        axis_loss_weights=(2.0, 1.0, 1.5)))

optimizer = dict(
    type='AdamW',
    lr=5e-5,
    weight_decay=0.0001,
    paramwise_cfg=dict(
        custom_keys=dict(
            **{
                'bbox_head.refine_root_branches': dict(
                    lr_mult=1.0, decay_mult=1.0),
                'bbox_head': dict(lr_mult=0.0, decay_mult=0.0),
                'backbone': dict(lr_mult=0.0, decay_mult=0.0),
                'neck': dict(lr_mult=0.0, decay_mult=0.0),
                'head': dict(lr_mult=0.0, decay_mult=0.0),
            })))
optimizer_config = dict(grad_clip=dict(max_norm=0.1, norm_type=2))
lr_config = dict(policy='step', step=[4])
runner = dict(type='EpochBasedRunner', max_epochs=6)

checkpoint_config = dict(interval=1, max_keep_ckpts=6)
work_dir = 'result/mmfi_d6_freeze_root_adapter'
load_from = 'result/mmfi_b0_3d_baseline/best_mpjpe_epoch_10.pth'
resume_from = None
