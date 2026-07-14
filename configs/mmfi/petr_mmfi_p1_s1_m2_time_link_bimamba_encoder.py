_base_ = './petr_mmfi_p1_s1_m1_bimamba_encoder.py'

# M2 applies the shared bidirectional Mamba mixer to both antenna-major
# (time-first within each antenna) and time-major (link-first within each time
# step) token orders. Both outputs are restored to antenna-major positions and
# averaged, so M1 and M2 have identical trainable parameter counts.
model = dict(
    bbox_head=dict(
        transformer=dict(
            encoder=dict(
                multi_order=True,
                num_antennas=3,
                num_time_steps=10))))

work_dir = 'result/mmfi_p1_s1_m2_time_link_bimamba_encoder'
