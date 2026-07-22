_base_ = '../mmfi/petr_mmfi_p1_s1_m1_bimamba_encoder.py'

dataset_type = 'opera.WiPose18Dataset'
data_root = '/home/xl/CSI/Person-in-WiFi-3D-repo/data/wipose18_official/Wi-Pose'

train_pipeline = [
    dict(
        type='opera.DefaultFormatBundle',
        extra_keys=['gt_keypoints', 'gt_labels']),
    dict(
        type='mmdet.Collect',
        keys=['img', 'gt_bboxes', 'gt_labels', 'gt_keypoints', 'gt_areas'],
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
                extra_keys=['gt_keypoints', 'gt_labels']),
            dict(type='mmdet.Collect', keys=['img'], meta_keys=[])
        ])
]

wipose_common = dict(
    image_size=(640, 480),
    normalize_csi=False,
    query_selection='top_score',
    confidence_threshold=0.0)

data = dict(
    samples_per_gpu=32,
    workers_per_gpu=2,
    train=dict(
        _delete_=True,
        type=dataset_type,
        dataset_root=data_root,
        pipeline=train_pipeline,
        mode='train',
        **wipose_common),
    val=dict(
        _delete_=True,
        type=dataset_type,
        dataset_root=data_root,
        pipeline=test_pipeline,
        mode='test',
        samples_per_gpu=1,
        **wipose_common),
    test=dict(
        _delete_=True,
        type=dataset_type,
        dataset_root=data_root,
        pipeline=test_pipeline,
        mode='test',
        samples_per_gpu=1,
        **wipose_common))

# WiPose CSI is (Tx=3, Rx=3, subcarrier=30, time=5). The linear stem sees
# 45 link-time tokens, each represented by 30 amplitude features.
model = dict(
    input_dim=30,
    bbox_head=dict(
        num_keypoints=18,
        coordinate_dims=2,
        transformer=dict(
            num_keypoints=18,
            coordinate_dims=2,
            decoder=dict(num_keypoints=18, coordinate_dims=2),
            refine_decoder=dict(coordinate_dims=2))),
    train_cfg=dict(assigner=dict(coordinate_dims=2)))

evaluation = dict(
    interval=1, metric='mpjpe', save_best='mpjpe', rule='less')
checkpoint_config = dict(interval=5, max_keep_ckpts=10)
lr_config = dict(policy='step', step=[60])
runner = dict(type='EpochBasedRunner', max_epochs=80)

work_dir = 'result/wipose_m1_bimamba_2d_pck'
