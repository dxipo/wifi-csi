_base_ = './petr_mmfi_b0_p1_s1_official_like.py'

# Q3: GT-supervised listwise query ranking.
# Convert per-query pose error to a target distribution:
#   p_gt = softmax(-query_mpjpe / query_quality_tau)
# and train the score distribution:
#   p_score = softmax(score / query_quality_score_tau)
# with KL divergence. Inference remains CSI-only and selects top-score query.

data = dict(
    train=dict(query_selection='top_score'),
    val=dict(query_selection='top_score'),
    test=dict(query_selection='top_score'))

model = dict(
    bbox_head=dict(
        query_quality_loss_weight=0.05,
        query_quality_loss_mode='listwise',
        query_quality_loss_type='kl',
        query_quality_tau=0.15,
        query_quality_score_tau=4.0))

optimizer = dict(type='AdamW', lr=5e-6, weight_decay=0.0001)
lr_config = dict(policy='step', step=[10])
runner = dict(type='EpochBasedRunner', max_epochs=15)

evaluation = dict(interval=1, metric='mpjpe', save_best='mpjpe', rule='less')
checkpoint_config = dict(interval=1, max_keep_ckpts=5)

load_from = 'result/mmfi_b0_p1_s1_official_like/best_mpjpe_epoch_77.pth'
resume_from = None
work_dir = 'result/mmfi_p1_s1_query_listwise_rank_gt'
