_base_ = './petr_mmfi_d2_joint_distill_no_stage1.py'

data = dict(
    train=dict(
        shuffle_teacher_tokens=True,
        shuffle_teacher_seed=42))

work_dir = 'result/mmfi_d3_shuffled_token_control'
