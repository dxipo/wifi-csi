_base_ = './petr_mmfi_b0_3d_baseline.py'

# Origin-paper aligned tokenization for MMFi:
# CSIamp/CSIphase: (3, 114, 10)
# -> denoised amplitude/phase
# -> (3 antennas, 10 time packets, 114 * 2 CSI features)
# -> 30 tokens, each with 228 dimensions.
mmfi_correct_time_token = dict(
    preprocess='origin',
    use_phase=True,
    normalize_csi=False,
    origin_linear_layout='time_token')

data = dict(
    train=dict(**mmfi_correct_time_token),
    val=dict(**mmfi_correct_time_token),
    test=dict(**mmfi_correct_time_token))

model = dict(
    input_stem='linear',
    input_dim=228,
    stem_out_channels=256,
    bbox_head=dict(
        num_query=30,
        transformer=dict(two_stage_num_proposals=30)),
    test_cfg=dict(max_per_img=30))

optimizer = dict(type='AdamW', lr=2e-5, weight_decay=0.0001)
lr_config = dict(policy='step', step=[60])
runner = dict(type='EpochBasedRunner', max_epochs=80)

work_dir = 'result/mmfi_b0_correct_time_token'
