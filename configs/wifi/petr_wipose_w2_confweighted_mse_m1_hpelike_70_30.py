_base_ = './petr_wipose_w1_singlequery_m1_hpelike_70_30.py'

# W2 uses the AlphaPose label confidence as the x/y MSE weight. Joints below
# 0.1 confidence are ignored; remaining confidence values are clipped to 1.
data = dict(
    train=dict(use_keypoint_confidence=True))

model = dict(
    bbox_head=dict(
        use_keypoint_confidence=True,
        keypoint_confidence_threshold=0.1))

work_dir = 'result/wipose_w2_confweighted_mse_m1_hpelike_70_30'
