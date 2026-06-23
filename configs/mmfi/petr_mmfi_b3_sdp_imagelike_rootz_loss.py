_base_ = './petr_mmfi_b2_sdp_imagelike_no_translator.py'

model = dict(
    bbox_head=dict(
        rel_pose_loss_weight=2.0,
        root_pose_loss_weight=1.0,
        axis_pose_loss_weight=1.0,
        axis_loss_weights=(1.0, 1.0, 2.0)))

work_dir = 'result/mmfi_b3_sdp_imagelike_rootz_loss'
