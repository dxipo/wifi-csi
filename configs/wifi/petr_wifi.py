dataset_type = 'opera.WifiPoseDataset'
data_root = '../data/wifipose'
# train_pipeline = [
#     dict(
#         type='opera.DefaultFormatBundle',
#         extra_keys=['gt_keypoints', 'gt_labels']),
#     dict(
#         type='mmdet.Collect',
#         keys=['img', 'gt_bboxes', 'gt_labels', 'gt_keypoints', 'gt_areas'],
#         meta_keys=[])
# ]

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
    samples_per_gpu=32, #32 256
    workers_per_gpu=2, #2 8
    train=dict(
        type='opera.WifiPoseDataset',
        dataset_root='/home/xl/CSI/Person-in-WiFi-3D-repo/data/wifipose/all_single_train_data_hold_out',
        pipeline=[
            dict(
                type='opera.DefaultFormatBundle',
                extra_keys=['gt_keypoints', 'gt_labels']),
            dict(
                type='mmdet.Collect',
                keys=[
                    'img', 'gt_bboxes', 'gt_labels', 'gt_keypoints', 'gt_areas'
                ],
                meta_keys=[])
        ],
        mode='train'),
    val=dict(
        type='opera.WifiPoseDataset',
        dataset_root='/home/xl/CSI/Person-in-WiFi-3D-repo/data/wifipose/all_single_test_data_hold_out',
        pipeline=[
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
        ],
        mode='test'),
    test=dict(
        type='opera.WifiPoseDataset',
        dataset_root='/home/xl/CSI/Person-in-WiFi-3D-repo/data/wifipose/all_single_test_data_hold_out',
        pipeline=[
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
        ],
        mode='test'))

#fp16 = dict(loss_scale='dynamic')# new added
evaluation = dict(interval=1, metric='mpjpe')
checkpoint_config = dict(interval=5, max_keep_ckpts=100)
# checkpoint_config = dict(
#     interval=5,              # 每5个epoch存一次常规checkpoint
#     max_keep_ckpts=20,       # 常规checkpoint最多保留20个（覆盖最近100个epoch）
#     save_best='mpjpe',       # 额外保存“最优mpjpe”的模型
#     rule='less'              # mpjpe越小越好
# )
#evaluation = dict(interval=1, metric='mpjpe')

#log_config = dict(interval=10, hooks=[dict(type='TextLoggerHook')])
log_config = dict(
    interval=10,
    hooks=[
        dict(type='TextLoggerHook'),
        dict(type='TensorboardLoggerHook')
    ]
)
custom_hooks = [dict(type='NumClassCheckHook')]
dist_params = dict(backend='nccl')
log_level = 'INFO'
load_from = None
resume_from = None
workflow = [('train', 1)]
opencv_num_threads = 0
mp_start_method = 'fork'
auto_scale_lr = dict(enable=False, base_batch_size=16)
model = dict(
    type='opera.PETR',
    backbone=dict(
        type='mmdet.ResNet',
        depth=50,
        num_stages=4,
        out_indices=(1, 2, 3),
        frozen_stages=1,
        norm_cfg=dict(type='BN', requires_grad=False),
        norm_eval=True,
        style='pytorch',
        init_cfg=dict(type='Pretrained', checkpoint='torchvision://resnet50')),
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
        #with_kpt_refine=True,
        with_kpt_refine=True,
        as_two_stage=True,
        num_keypoints=14,
        transformer=dict(
            type='opera.PETRTransformer',
            num_keypoints=14,
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
                num_keypoints=14,
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
        #loss_kpt=dict(type='mmdet.MSELoss', loss_weight=70.0),
        loss_kpt=dict(type='mmdet.MSELoss', loss_weight=10.0),
        loss_kpt_rpn=dict(type='mmdet.MSELoss', loss_weight=10.0),
        # loss_oks=dict(type='opera.OKSLoss', loss_weight=2.0),
        # loss_hm=dict(type='opera.CenterFocalLoss', loss_weight=4.0),
        # loss_kpt_refine=dict(type='mmdet.MSELoss', loss_weight=70.0),
        # loss_oks_refine=dict(type='opera.OKSLoss', loss_weight=3.0)),

        loss_oks=dict(type='opera.OKSLoss', loss_weight=0.0),
        loss_hm=dict(type='opera.CenterFocalLoss', loss_weight=0.0),
        loss_kpt_refine=dict(type='mmdet.MSELoss', loss_weight=0.0),
        loss_oks_refine=dict(type='opera.OKSLoss', loss_weight=0.0)),



    train_cfg=dict(
        assigner=dict(
            type='opera.PoseHungarianAssigner',
            cls_cost=dict(type='mmdet.FocalLossCost', weight=4.0),
            #kpt_cost=dict(type='opera.KptMSECost', weight=70.0),
            kpt_cost=dict(type='opera.KptMSECost', weight=10.0),
            #oks_cost=dict(type='opera.OksCost', weight=7.0))),
            oks_cost=dict(type='opera.OksCost', weight=0.0))),
    test_cfg=dict(max_per_img=100))
optimizer = dict(
    type='AdamW',
    lr=2e-05, # 2e-05 3e
    weight_decay=0.0001,
    paramwise_cfg=dict(
        custom_keys=dict(
            backbone=dict(lr_mult=0.1),
            sampling_offsets=dict(lr_mult=0.1),
            reference_points=dict(lr_mult=0.1))))
optimizer_config = dict(grad_clip=dict(max_norm=0.1, norm_type=2)) # max_norm=0.1 1.0
lr_config = dict(policy='step', step=[400]) #400
runner = dict(type='EpochBasedRunner', max_epochs=450) #450
find_unused_parameters = True
work_dir = '/home/xl/CSI/Person-in-WiFi-3D-repo/result/wifipose_2d_baseline_test_recover'
auto_resume = False
#gpu_ids = range(0, 3)
gpu_ids = range(0, 1)
