_base_ = './petr_wipose_m1_bimamba_2d_hpelike_70_30.py'

# W1 single-person control: remove query ranking and proposal diversity while
# keeping the M1 encoder, HPE-Li-like 70/30 split, losses, and schedule fixed.
model = dict(
    bbox_head=dict(
        num_query=1,
        transformer=dict(two_stage_num_proposals=1)),
    test_cfg=dict(max_per_img=1))

work_dir = 'result/wipose_w1_singlequery_m1_hpelike_70_30'
