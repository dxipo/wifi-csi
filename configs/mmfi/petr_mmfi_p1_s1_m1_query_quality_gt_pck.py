_base_ = './petr_mmfi_p1_s1_m1_bimamba_encoder.py'

# Combine the M1 bidirectional Mamba encoder with GT-supervised query-quality
# calibration. Evaluation remains strict: the selected query is the one with
# the highest predicted score and never uses GT at inference time.
data = dict(
    train=dict(query_selection='top_score'),
    val=dict(query_selection='top_score'),
    test=dict(query_selection='top_score'))

model = dict(
    bbox_head=dict(
        query_quality_loss_weight=0.2,
        query_quality_tau=0.15,
        query_quality_loss_type='bce'))

optimizer = dict(type='AdamW', lr=5e-6, weight_decay=0.0001)
lr_config = dict(policy='step', step=[10])
runner = dict(type='EpochBasedRunner', max_epochs=15)

evaluation = dict(interval=1, metric='mpjpe', save_best='mpjpe', rule='less')
checkpoint_config = dict(interval=1, max_keep_ckpts=5)

load_from = ('/home/xl/CSI/Person-in-WiFi-3D-repo/result/'
             'mmfi_p1_s1_m1_bimamba_encoder/best_mpjpe_epoch_75.pth')
resume_from = None
work_dir = 'result/mmfi_p1_s1_m1_query_quality_gt_pck'
