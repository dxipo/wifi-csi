_base_ = './petr_wifi.py'

data = dict(
    train=dict(dataset_root='data/wifipose/train_data'),
    val=dict(dataset_root='data/wifipose/test_data'),
    test=dict(dataset_root='data/wifipose/test_data'))

model = dict(
    bbox_head=dict(
        dn_cfg=dict(
            enabled=True,
            num_groups=5,
            root_indices=[5, 7],
            root_noise_std=0.03,
            joint_noise_std=0.02,
            scale_noise_std=0.03)))

log_config = dict(
    interval=50,
    hooks=[
        dict(type='TextLoggerHook'),
        dict(type='TensorboardLoggerHook')
    ])

work_dir = 'result/origin_paper_dn_p1_pose_denoising'
