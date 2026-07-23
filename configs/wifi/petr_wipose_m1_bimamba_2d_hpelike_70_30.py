_base_ = './petr_wipose_m1_bimamba_2d_pck.py'

# HPE-Li reports a packet-level random 70/30 split on all WiPose samples.
# Merge the released Train/Test directories before applying the fixed split.
hpelike_split = dict(
    split_strategy='packet_random',
    random_ratio=0.7,
    random_seed=0)

data = dict(
    train=dict(**hpelike_split),
    val=dict(**hpelike_split),
    test=dict(**hpelike_split))

work_dir = 'result/wipose_m1_bimamba_2d_hpelike_70_30'
