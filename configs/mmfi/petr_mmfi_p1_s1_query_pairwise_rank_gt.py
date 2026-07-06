_base_ = './petr_mmfi_b0_p1_s1_official_like.py'

# Q2: GT-supervised pairwise query ranking.
# For two queries i and j, if query i has lower MPJPE than query j by at
# least query_quality_rank_delta, the score branch is trained to rank i higher.
# Inference still uses CSI only and selects the top-score query.

data = dict(
    train=dict(query_selection='top_score'),
    val=dict(query_selection='top_score'),
    test=dict(query_selection='top_score'))

model = dict(
    bbox_head=dict(
        query_quality_loss_weight=0.2,
        query_quality_loss_mode='pairwise',
        query_quality_loss_type='softplus',
        query_quality_rank_delta=0.005,
        query_quality_rank_margin=0.0))

optimizer = dict(type='AdamW', lr=5e-6, weight_decay=0.0001)
lr_config = dict(policy='step', step=[10])
runner = dict(type='EpochBasedRunner', max_epochs=15)

evaluation = dict(interval=1, metric='mpjpe', save_best='mpjpe', rule='less')
checkpoint_config = dict(interval=1, max_keep_ckpts=5)

load_from = 'result/mmfi_b0_p1_s1_official_like/best_mpjpe_epoch_77.pth'
resume_from = None
work_dir = 'result/mmfi_p1_s1_query_pairwise_rank_gt'
