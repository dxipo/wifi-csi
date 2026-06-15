_base_ = './petr_mmfi_b0_3d_baseline.py'

teacher_token_root = 'data/mmfi_teacher_tokens/t0'

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
    samples_per_gpu=32,
    workers_per_gpu=2,
    train=dict(
        pipeline=train_pipeline,
        teacher_token_dir=teacher_token_root + '/train',
        strict_teacher_tokens=True),
    val=dict(samples_per_gpu=1),
    test=dict(samples_per_gpu=1))

model = dict(
    type='opera.PETR',
    input_stem='conv2d_3layer',
    stem_mid_channels=128,
    bbox_head=dict(
        decoder_token_distill_weight=0.2,
        decoder_relation_distill_weight=0.05,
        rel_pose_loss_weight=2.0,
        bone_loss_weight=1.0,
        pelvis_indices=(11, 12)))

evaluation = dict(interval=1, metric='mpjpe', save_best='mpjpe', rule='less')
checkpoint_config = dict(interval=5, max_keep_ckpts=10)
log_config = dict(
    interval=50,
    hooks=[
        dict(type='TextLoggerHook'),
        dict(type='TensorboardLoggerHook')
    ])

optimizer = dict(type='AdamW', lr=2e-5, weight_decay=0.0001)
lr_config = dict(policy='step', step=[15])
runner = dict(type='EpochBasedRunner', max_epochs=20)
load_from = 'result/mmfi_b0_3d_baseline/best_mpjpe_epoch_10.pth'
resume_from = None
work_dir = 'result/mmfi_d4_decoder_coupled_distill'
