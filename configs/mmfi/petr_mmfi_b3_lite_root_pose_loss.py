_base_ = './petr_mmfi_b2_sdp_imagelike_no_translator.py'

model = dict(
    bbox_head=dict(
        rel_pose_loss_weight=0.0,
        root_pose_loss_weight=0.2,
        axis_pose_loss_weight=0.0))

work_dir = 'result/mmfi_b3_lite_root_pose_loss'
