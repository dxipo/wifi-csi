_base_ = './petr_mmfi_b0_p1_s1_official_like.py'

# Full fusion experiment for MMFi P1-S1.
#
# This is the fair from-scratch counterpart of A1b:
#   - no baseline checkpoint is loaded
#   - train for the same 80 epochs as the official-like baseline
#   - use the same base learning rate and schedule as the baseline
#   - keep the proven time-token organization
#
# Input per MMFi frame:
#   raw main token dim = 114 DWT amplitude + 114 sanitized phase = 228
#   SDP aux token dim  = 114 subcarriers x 6 lag bins = 684
#   total token dim    = 912
#   token count        = 3 antennas x 10 center-frame packets = 30
mmfi_p1_s1_a1b_fulltrain_common = dict(
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
    train=dict(**mmfi_p1_s1_a1b_fulltrain_common),
    val=dict(**mmfi_p1_s1_a1b_fulltrain_common),
    test=dict(**mmfi_p1_s1_a1b_fulltrain_common))

model = dict(
    input_stem='sdp_aux_residual',
    input_dim=912,
    raw_input_dim=228,
    sdp_input_dim=684,
    sdp_aux_init=1.0)

optimizer = dict(type='AdamW', lr=2e-5, weight_decay=0.0001)
lr_config = dict(policy='step', step=[60])
runner = dict(type='EpochBasedRunner', max_epochs=80)
checkpoint_config = dict(interval=5, max_keep_ckpts=10)

load_from = None
resume_from = None
work_dir = 'result/mmfi_p1_s1_a1b_sdp_aux_fulltrain'
