dataset_type = 'opera.MMFiPoseDataset'
data_root = '/home/xl/Downloads/MMFi_Dataset/MMFi_unzipped'

mmfi_common = dict(
    protocol='protocol2',
    split_to_use='random_split',
    random_ratio=0.8,
    random_seed=0,
    use_phase=True,
    normalize_csi=True,
    return_pose2d=True,
    pose2d_window=9,
    pose2d_normalize=True,
    pose2d_confidence=True)

train_pipeline = [
    dict(
        type='opera.DefaultFormatBundle',
        extra_keys=['gt_keypoints', 'gt_labels', 'pose2d_seq', 'pose2d_mask']),
    dict(
        type='mmdet.Collect',
        keys=[
            'img', 'gt_bboxes', 'gt_labels', 'gt_keypoints', 'gt_areas',
            'pose2d_seq', 'pose2d_mask'
        ],
        meta_keys=[])
]

test_pipeline = [
    dict(
        type='mmdet.MultiScaleFlipAug',
        scale_factor=1.0,
        flip=False,
        transforms=[
            dict(
                type='opera.DefaultFormatBundle',
                extra_keys=['pose2d_seq', 'pose2d_mask']),
            dict(
                type='mmdet.Collect',
                keys=['img', 'pose2d_seq', 'pose2d_mask'],
                meta_keys=[])
        ])
]

data = dict(
    samples_per_gpu=64,
    workers_per_gpu=2,
    train=dict(
        type=dataset_type,
        dataset_root=data_root,
        pipeline=train_pipeline,
        mode='train',
        **mmfi_common),
    val=dict(
        type=dataset_type,
        dataset_root=data_root,
        pipeline=test_pipeline,
        mode='val',
        samples_per_gpu=1,
        **mmfi_common),
    test=dict(
        type=dataset_type,
        dataset_root=data_root,
        pipeline=test_pipeline,
        mode='val',
        samples_per_gpu=1,
        **mmfi_common))

evaluation = dict(interval=1, metric='mpjpe', save_best='mpjpe', rule='less')
checkpoint_config = dict(interval=5, max_keep_ckpts=10)
log_config = dict(
    interval=50,
    hooks=[
        dict(type='TextLoggerHook'),
        dict(type='TensorboardLoggerHook')
    ])
custom_hooks = [dict(type='NumClassCheckHook')]
dist_params = dict(backend='nccl')
log_level = 'INFO'
load_from = None
resume_from = None
workflow = [('train', 1)]
opencv_num_threads = 0
mp_start_method = 'fork'
auto_scale_lr = dict(enable=False, base_batch_size=64)

model = dict(
    type='opera.SkeletonPoseTeacher',
    num_keypoints=17,
    in_channels=3,
    embed_dims=256,
    num_layers=4,
    num_heads=8,
    feedforward_channels=512,
    dropout=0.1,
    max_seq_len=31,
    loss_abs_weight=1.0,
    loss_rel_weight=1.0,
    pelvis_indices=(11, 12))

optimizer = dict(type='AdamW', lr=1e-4, weight_decay=0.0001)
optimizer_config = dict(grad_clip=dict(max_norm=0.1, norm_type=2))
lr_config = dict(policy='step', step=[60])
runner = dict(type='EpochBasedRunner', max_epochs=80)
find_unused_parameters = True
work_dir = 'result/mmfi_t0_skeleton_teacher'
auto_resume = False
gpu_ids = range(0, 1)
