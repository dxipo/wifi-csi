_base_ = './petr_mmfi_b0_p1_s1_topscore_eval.py'

# Amplitude-only ablation under the unchanged P1-S1 strict top-score setup.
# CSIamp: (3, 114, 10) -> 30 time-link tokens, each with 114 features.
data = dict(
    train=dict(use_phase=False),
    val=dict(use_phase=False),
    test=dict(use_phase=False))

model = dict(input_dim=114)

work_dir = 'result/mmfi_p1_s1_amp_only_strict_query_eval'
