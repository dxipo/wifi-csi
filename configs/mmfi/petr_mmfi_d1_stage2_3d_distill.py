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
    train=dict(
        pipeline=train_pipeline,
        teacher_token_dir=teacher_token_root + '/train',
        strict_teacher_tokens=True),
    val=dict(samples_per_gpu=1),
    test=dict(samples_per_gpu=1))

model = dict(
    type='opera.PETRDistill',
    distill_stage='stage2',
    distill_num_tokens=18,
    teacher_embed_dims=256,
    distill_loss_weight=0.1,
    relation_loss_weight=0.03,
    distill_num_heads=8)

evaluation = dict(interval=1, metric='mpjpe', save_best='mpjpe', rule='less')
checkpoint_config = dict(interval=5, max_keep_ckpts=10)
log_config = dict(
    interval=50,
    hooks=[
        dict(type='TextLoggerHook'),
        dict(type='TensorboardLoggerHook')
    ])
optimizer = dict(type='AdamW', lr=1e-4, weight_decay=0.0001)
lr_config = dict(policy='step', step=[60])
runner = dict(type='EpochBasedRunner', max_epochs=80)
load_from = 'result/mmfi_d1_stage1_token_pretrain/latest.pth'
work_dir = 'result/mmfi_d1_stage2_3d_distill'
