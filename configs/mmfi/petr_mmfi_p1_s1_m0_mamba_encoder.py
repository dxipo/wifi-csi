_base_ = './petr_mmfi_b0_p1_s1_official_like.py'

# M0 changes only the six-layer CSI encoder. The existing 30-token ordering,
# PETR decoders, losses, data split, and optimizer remain unchanged.
data = dict(
    train=dict(query_selection='top_score'),
    val=dict(query_selection='top_score'),
    test=dict(query_selection='top_score'))

model = dict(
    bbox_head=dict(
        transformer=dict(
            encoder=dict(
                _delete_=True,
                type='opera.MambaEncoder',
                embed_dims=256,
                num_layers=6,
                d_state=16,
                d_conv=4,
                expand=2,
                dropout=0.1,
                final_norm=True))))

work_dir = 'result/mmfi_p1_s1_m0_mamba_encoder'
