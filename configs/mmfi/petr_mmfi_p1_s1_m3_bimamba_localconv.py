_base_ = './petr_mmfi_p1_s1_m1_bimamba_encoder.py'

# M3 keeps M1's time-first shared-weight BiMamba and adds a lightweight
# depthwise 3x3 convolution over the 3-antenna x 10-time token grid. A small
# per-channel residual scale lets the network adopt local context gradually.
model = dict(
    bbox_head=dict(
        transformer=dict(
            encoder=dict(
                use_local_conv=True,
                local_kernel_size=3,
                local_init_scale=0.1,
                num_antennas=3,
                num_time_steps=10))))

work_dir = 'result/mmfi_p1_s1_m3_bimamba_localconv'
