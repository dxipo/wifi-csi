_base_ = './petr_mmfi_b0_p1_s1_official_like.py'

# Strict evaluation for the P1-S1 baseline.
# The default MMFi evaluation in this repo uses query_selection='oracle',
# which chooses the query closest to GT. This config selects the highest-score
# prediction only, matching inference-time availability.
data = dict(
    workers_per_gpu=0,
    train=dict(query_selection='top_score'),
    val=dict(query_selection='top_score'),
    test=dict(query_selection='top_score'))

work_dir = 'result/mmfi_b0_p1_s1_topscore_eval'
