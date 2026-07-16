_base_ = './petr_wifi.py'

# Person-in-WiFi-3D M1: replace only the six-layer Transformer encoder with
# shared-weight bidirectional Mamba. Input preprocessing, PETR decoders,
# queries, losses, and batch size remain aligned with the original baseline.
data_root = '/home/xl/CSI/Person-in-WiFi-3D-repo/data/wifipose'
data = dict(
    train=dict(dataset_root=f'{data_root}/train_data'),
    val=dict(dataset_root=f'{data_root}/test_data'),
    test=dict(dataset_root=f'{data_root}/test_data'))

model = dict(
    bbox_head=dict(
        transformer=dict(
            encoder=dict(
                _delete_=True,
                type='opera.MambaEncoder',
                embed_dims=256,
                num_layers=6,
                d_state=16,
                d_conv=4,
                expand=2,
                dropout=0.1,
                bidirectional=True,
                final_norm=True))))

lr_config = dict(policy='step', step=[60])
runner = dict(type='EpochBasedRunner', max_epochs=80)

evaluation = dict(
    interval=1,
    metric='mpjpe',
    save_best='mpjpe',
    rule='less')
checkpoint_config = dict(interval=5, max_keep_ckpts=10)
log_config = dict(
    interval=50,
    hooks=[
        dict(type='TextLoggerHook'),
        dict(type='TensorboardLoggerHook')
    ])

work_dir = 'result/person_wifi_m1_bimamba_encoder_80ep'
