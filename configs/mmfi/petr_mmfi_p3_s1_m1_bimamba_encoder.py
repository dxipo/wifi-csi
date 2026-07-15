_base_ = './petr_mmfi_p1_s1_m1_bimamba_encoder.py'

# P3-S1 keeps the M1 BiMamba model and sample-level 3:1 split unchanged,
# while expanding the benchmark from P1 daily actions to all 27 MMFi actions.
data = dict(
    train=dict(protocol='protocol3'),
    val=dict(protocol='protocol3'),
    test=dict(protocol='protocol3'))

work_dir = 'result/mmfi_p3_s1_m1_bimamba_encoder'
