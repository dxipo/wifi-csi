_base_ = './petr_mmfi_b0_3d_baseline.py'

sdp_offline_root = 'data/mmfi_sdp_imagelike_ctx7'

sdp_imagelike_common = dict(
    preprocess='sdp_imagelike_offline',
    use_phase=False,
    normalize_csi=True,
    sdp_context_radius=7,
    sdp_window_size=8,
    sdp_stride=3,
    sdp_n_delta=6,
    sdp_layout='lagwindow',
    sdp_use_hampel=True,
    sdp_hampel_window=2,
    sdp_hampel_sigma=3.0,
    sdp_use_moving_average=True,
    sdp_ma_window=3,
    sdp_acf_unbiased=False,
    sdp_positive_clip=True,
    sdp_zero_column_fill='uniform',
    sdp_offline_ext='.npy',
    strict_sdp_offline=True)

data = dict(
    train=dict(
        **sdp_imagelike_common,
        sdp_offline_dir=sdp_offline_root + '/train'),
    val=dict(
        **sdp_imagelike_common,
        sdp_offline_dir=sdp_offline_root + '/val'),
    test=dict(
        **sdp_imagelike_common,
        sdp_offline_dir=sdp_offline_root + '/val'))

model = dict(
    input_stem='conv_patch',
    stem_in_channels=3,
    patch_kernel_size=(12, 24),
    patch_stride=(12, 24),
    patch_padding=0,
    patch_out_dim=256)

optimizer = dict(type='AdamW', lr=1e-4, weight_decay=0.0001)
lr_config = dict(policy='step', step=[60])
runner = dict(type='EpochBasedRunner', max_epochs=80)

work_dir = 'result/mmfi_b2_sdp_imagelike_no_translator'
