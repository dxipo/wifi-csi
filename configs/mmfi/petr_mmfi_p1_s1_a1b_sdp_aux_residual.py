_base_ = './petr_mmfi_b0_p1_s1_official_like.py'

# A1b: keep the proven MMFi origin time-token input as the main path, and add
# centered power-ACF SDP as an auxiliary residual feature branch.
#
# Input per MMFi frame:
#   raw main token dim = 114 DWT amplitude + 114 sanitized phase = 228
#   SDP aux token dim  = 114 subcarriers x 6 lag bins = 684
#   total token dim    = 912
#   token count        = 3 antennas x 10 center-frame packets = 30
#
# The raw branch keeps the parameter name "head" in PETR, so it can load the
# best official-like baseline checkpoint directly. The SDP branch is initialized
# separately and injected with a small learnable residual coefficient.
mmfi_p1_s1_a1b_common = dict(
    protocol='protocol1',
    split_to_use='sample_random_split',
    random_ratio=0.75,
    random_seed=0,
    preprocess='sdp_power_acf_aux_time_token',
    use_phase=True,
    normalize_csi=False,
    origin_linear_layout='time_token',
    sdp_context_radius=7,
    sdp_window_size=60,
    sdp_stride=1,
    sdp_n_delta=6,
    sdp_layout='lagwindow',
    sdp_use_hampel=False,
    sdp_hampel_window=2,
    sdp_hampel_sigma=3.0,
    sdp_use_moving_average=True,
    sdp_ma_window=3,
    sdp_acf_unbiased=False,
    sdp_positive_clip=True,
    sdp_zero_column_fill='uniform',
    sdp_center_aligned=True)

data = dict(
    train=dict(**mmfi_p1_s1_a1b_common),
    val=dict(**mmfi_p1_s1_a1b_common),
    test=dict(**mmfi_p1_s1_a1b_common))

model = dict(
    input_stem='sdp_aux_residual',
    input_dim=912,
    raw_input_dim=228,
    sdp_input_dim=684,
    sdp_aux_init=0.05)

optimizer = dict(
    type='AdamW',
    lr=2e-6,
    weight_decay=0.0001,
    paramwise_cfg=dict(
        custom_keys=dict(
            sdp_head=dict(lr_mult=10.0),
            sdp_aux_alpha=dict(lr_mult=10.0, decay_mult=0.0))))

lr_config = dict(policy='step', step=[15])
runner = dict(type='EpochBasedRunner', max_epochs=20)
checkpoint_config = dict(interval=5, max_keep_ckpts=10)

load_from = 'result/mmfi_b0_p1_s1_official_like/best_mpjpe_epoch_77.pth'
work_dir = 'result/mmfi_p1_s1_a1b_sdp_aux_residual'
