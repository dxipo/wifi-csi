dataset_type = 'opera.MMFiPoseDataset'
data_root = '/home/xl/Downloads/MMFi_Dataset/MMFi_unzipped'

mmfi_common = dict(
    protocol='protocol2',
    split_to_use='random_split',
    random_ratio=0.8,
    random_seed=0,
    use_phase=True,
    normalize_csi=True)

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

data = dict(
    samples_per_gpu=32,
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
auto_scale_lr = dict(enable=False, base_batch_size=32)

model = dict(
    type='opera.PETR',
    input_stem='conv2d',
    stem_in_channels=6,
    stem_hidden_channels=64,
    stem_out_channels=256,
    backbone=dict(
        type='mmdet.ResNet',
        depth=50,
        num_stages=4,
        out_indices=(1, 2, 3),
        frozen_stages=1,
        norm_cfg=dict(type='BN', requires_grad=False),
        norm_eval=True,
        style='pytorch',
        init_cfg=None),
    neck=dict(
        type='mmdet.ChannelMapper',
        in_channels=[512, 1024, 2048],
        kernel_size=1,
        out_channels=256,
        act_cfg=None,
        norm_cfg=dict(type='GN', num_groups=32),
        num_outs=4),
    bbox_head=dict(
        type='opera.PETRHead',
        num_query=100,
        num_classes=1,
        in_channels=2048,
        sync_cls_avg_factor=True,
        with_kpt_refine=True,
        as_two_stage=True,
        num_keypoints=17,
        transformer=dict(
            type='opera.PETRTransformer',
            num_keypoints=17,
            encoder=dict(
                type='mmcv.DetrTransformerEncoder',
                num_layers=6,
                transformerlayers=dict(
                    type='mmcv.BaseTransformerLayer',
                    attn_cfgs=dict(
                        type='mmcv.MultiheadAttention',
                        embed_dims=256,
                        num_heads=8,
                        dropout=0.1),
                    feedforward_channels=1024,
                    ffn_dropout=0.1,
                    operation_order=('self_attn', 'norm', 'ffn', 'norm'))),
            decoder=dict(
                type='opera.PetrTransformerDecoder',
                num_layers=3,
                num_keypoints=17,
                return_intermediate=True,
                transformerlayers=dict(
                    type='mmcv.DetrTransformerDecoderLayer',
                    attn_cfgs=[
                        dict(
                            type='mmcv.MultiheadAttention',
                            embed_dims=256,
                            num_heads=8,
                            dropout=0.1),
                        dict(
                            type='mmcv.MultiheadAttention',
                            embed_dims=256,
                            num_heads=8,
                            dropout=0.1)
                    ],
                    feedforward_channels=1024,
                    ffn_dropout=0.1,
                    operation_order=('self_attn', 'norm', 'cross_attn', 'norm',
                                     'ffn', 'norm'))),
            refine_decoder=dict(
                type='opera.PetrRefineTransformerDecoder',
                num_layers=2,
                return_intermediate=True,
                transformerlayers=dict(
                    type='mmcv.DetrTransformerDecoderLayer',
                    attn_cfgs=[
                        dict(
                            type='mmcv.MultiheadAttention',
                            embed_dims=256,
                            num_heads=8,
                            dropout=0.1),
                        dict(
                            type='mmcv.MultiheadAttention',
                            embed_dims=256,
                            num_heads=8,
                            dropout=0.1)
                    ],
                    feedforward_channels=1024,
                    ffn_dropout=0.1,
                    operation_order=('self_attn', 'norm', 'cross_attn', 'norm',
                                     'ffn', 'norm')))),
        positional_encoding=dict(
            type='mmcv.SinePositionalEncoding',
            num_feats=128,
            normalize=True,
            offset=-0.5),
        loss_cls=dict(
            type='mmdet.FocalLoss',
            use_sigmoid=True,
            gamma=2.0,
            alpha=0.25,
            loss_weight=4.0),
        loss_kpt=dict(type='mmdet.MSELoss', loss_weight=70.0),
        loss_kpt_rpn=dict(type='mmdet.MSELoss', loss_weight=70.0),
        loss_oks=dict(type='opera.OKSLoss', loss_weight=0.0),
        loss_hm=dict(type='opera.CenterFocalLoss', loss_weight=0.0),
        loss_kpt_refine=dict(type='mmdet.MSELoss', loss_weight=70.0),
        loss_oks_refine=dict(type='opera.OKSLoss', loss_weight=0.0)),
    train_cfg=dict(
        assigner=dict(
            type='opera.PoseHungarianAssigner',
            cls_cost=dict(type='mmdet.FocalLossCost', weight=4.0),
            kpt_cost=dict(type='opera.KptMSECost', weight=70.0),
            oks_cost=dict(type='opera.OksCost', weight=0.0))),
    test_cfg=dict(max_per_img=100))

optimizer = dict(type='AdamW', lr=1e-4, weight_decay=0.0001)
optimizer_config = dict(grad_clip=dict(max_norm=0.1, norm_type=2))
lr_config = dict(policy='step', step=[60])
runner = dict(type='EpochBasedRunner', max_epochs=80)
find_unused_parameters = True
work_dir = 'result/mmfi_b0_3d_baseline'
auto_resume = False
gpu_ids = range(0, 1)
