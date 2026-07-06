_base_ = './petr_mmfi_b0_p1_s1_official_like.py'

# A1: replace the origin amplitude branch with a centered power-ACF SDP branch,
# while keeping the original sanitized phase branch and the proven time-token
# organization.
#
# Input per MMFi frame:
#   CSIamp/CSIphase: (3 antennas, 114 subcarriers, 10 packets)
#
# Centered context:
#   15 frames x 10 packets = 150 packets
#
# Output tokenization:
#   tokens       = 3 antennas x 10 center-frame packets = 30
#   token dim    = 114 subcarriers x 6 lag bins + 114 phase = 798
#   query count  = 30, unchanged from the official-like time-token baseline
mmfi_p1_s1_a1_common = dict(
    protocol='protocol1',
    split_to_use='sample_random_split',
    random_ratio=0.75,
    random_seed=0,
    preprocess='sdp_power_acf_phase_time_token',
    use_phase=True,
    normalize_csi=False,
    origin_linear_layout='time_token',
    sdp_context_radius=7,
    sdp_window_size=60,
    sdp_stride=1,
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
    sdp_center_aligned=True)

data = dict(
    train=dict(**mmfi_p1_s1_a1_common),
    val=dict(**mmfi_p1_s1_a1_common),
    test=dict(**mmfi_p1_s1_a1_common))

model = dict(input_dim=798)

work_dir = 'result/mmfi_p1_s1_a1_power_acf_sdp_phase'
