_base_ = './petr_mmfi_b0_correct_time_token.py'

# MM-Fi paper-style P1-S1 benchmark:
# P1: daily actions.
# S1: random split with a 3:1 train/test ratio.
# The local dataset implementation names this sample-level split
# "sample_random_split" to distinguish it from the older subject-level
# random_split used by previous experiments.
mmfi_p1_s1_common = dict(
    protocol='protocol1',
    split_to_use='sample_random_split',
    random_ratio=0.75,
    random_seed=0,
    preprocess='origin',
    use_phase=True,
    normalize_csi=False,
    origin_linear_layout='time_token')

data = dict(
    train=dict(**mmfi_p1_s1_common),
    val=dict(**mmfi_p1_s1_common),
    test=dict(**mmfi_p1_s1_common))

work_dir = 'result/mmfi_b0_p1_s1_official_like'
