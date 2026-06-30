_base_ = './skeleton_teacher_t0.py'

# Reuse the trained T0 2D-skeleton teacher, but export tokens for the
# corrected MMFi P1-S1 split used by petr_mmfi_b0_p1_s1_official_like.py.
mmfi_p1_s1_common = dict(
    protocol='protocol1',
    split_to_use='sample_random_split',
    random_ratio=0.75,
    random_seed=0,
    use_phase=True,
    normalize_csi=False,
    return_pose2d=True,
    pose2d_window=9,
    pose2d_normalize=True,
    pose2d_confidence=True)

data = dict(
    train=dict(**mmfi_p1_s1_common),
    val=dict(**mmfi_p1_s1_common),
    test=dict(**mmfi_p1_s1_common))
