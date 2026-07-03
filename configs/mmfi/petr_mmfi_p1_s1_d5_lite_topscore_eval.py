_base_ = './petr_mmfi_p1_s1_d5_lite_distill_corrected.py'

# Strict inference-time evaluation for the corrected P1-S1 D5-lite model.
# The default MMFi evaluation uses query_selection='oracle', which selects the
# query closest to GT. This config selects the highest-score prediction only.
data = dict(
    workers_per_gpu=0,
    train=dict(query_selection='top_score'),
    val=dict(query_selection='top_score'),
    test=dict(query_selection='top_score'))

load_from = None
resume_from = None
work_dir = 'result/mmfi_p1_s1_d5_lite_topscore_eval'
