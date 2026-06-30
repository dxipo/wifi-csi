_base_ = './petr_mmfi_b0_p1_s3_official_like.py'

# Strict evaluation for the P1-S3 baseline.
# Select the highest-score prediction instead of choosing the query closest
# to GT during evaluation.
data = dict(
    workers_per_gpu=0,
    train=dict(query_selection='top_score'),
    val=dict(query_selection='top_score'),
    test=dict(query_selection='top_score'))

work_dir = 'result/mmfi_b0_p1_s3_topscore_eval'
