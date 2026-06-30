_base_ = './petr_mmfi_b0_p1_s1_official_like.py'

# Single-query control under the same P1-S1 split and time-token input.
# This removes multi-query selection from the model itself.
data = dict(
    train=dict(query_selection='top_score'),
    val=dict(query_selection='top_score'),
    test=dict(query_selection='top_score'))

model = dict(
    bbox_head=dict(
        num_query=1,
        transformer=dict(two_stage_num_proposals=1)),
    test_cfg=dict(max_per_img=1))

work_dir = 'result/mmfi_b0_p1_s1_numquery1_topscore'
