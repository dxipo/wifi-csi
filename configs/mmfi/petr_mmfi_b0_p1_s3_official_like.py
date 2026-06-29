_base_ = './petr_mmfi_b0_correct_time_token.py'

# MM-Fi paper-style P1-S3 benchmark:
# P1: daily actions.
# S3: cross-environment split.
# The local split uses E01-E03 for training and E04 for validation/test.
# Keep the origin-paper aligned time-token CSI representation unchanged:
# (3 antennas, 10 time packets, 114 * 2 CSI features), Linear(228, 256).
mmfi_p1_s3_common = dict(
    protocol='protocol1',
    split_to_use='cross_scene_split',
    preprocess='origin',
    use_phase=True,
    normalize_csi=False,
    origin_linear_layout='time_token')

data = dict(
    train=dict(**mmfi_p1_s3_common),
    val=dict(**mmfi_p1_s3_common),
    test=dict(**mmfi_p1_s3_common))

work_dir = 'result/mmfi_b0_p1_s3_official_like'
