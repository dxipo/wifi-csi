_base_ = './petr_mmfi_p1_s1_m0_mamba_encoder.py'

# M1 scans the same 30 CSI tokens in both directions with shared Mamba
# parameters. Mean fusion isolates bidirectional context without increasing
# the encoder parameter count.
model = dict(
    bbox_head=dict(
        transformer=dict(
            encoder=dict(
                bidirectional=True))))

work_dir = 'result/mmfi_p1_s1_m1_bimamba_encoder'
