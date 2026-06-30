_base_ = './petr_mmfi_b0_p1_s1_official_like.py'

# Corrected P1-S1 baseline + D5-lite distillation.
# This keeps the corrected time-token CSI input from P1-S1 B0 and only adds
# the conservative D5-lite main-path token/relation distillation and
# root-relative regularization.
teacher_token_root = 'data/mmfi_teacher_tokens/t0_p1_s1'

train_pipeline = [
    dict(
        type='opera.DefaultFormatBundle',
        extra_keys=['gt_keypoints', 'gt_labels', 'teacher_tokens']),
    dict(
        type='mmdet.Collect',
        keys=[
            'img', 'gt_bboxes', 'gt_labels', 'gt_keypoints', 'gt_areas',
            'teacher_tokens'
        ],
        meta_keys=[])
]

data = dict(
    train=dict(
        pipeline=train_pipeline,
        teacher_token_dir=teacher_token_root + '/train',
        strict_teacher_tokens=True))

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
load_from = 'result/mmfi_b0_p1_s1_official_like/best_mpjpe_epoch_77.pth'
resume_from = None
work_dir = 'result/mmfi_p1_s1_d5_lite_distill_corrected'
