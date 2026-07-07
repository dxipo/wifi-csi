_base_ = './petr_wifi.py'

data_root = 'data/wifipose'

data = dict(
    train=dict(dataset_root='data/wifipose/train_data'),
    val=dict(
        dataset_root='data/wifipose/test_data',
        query_selection='top_score',
        query_topk='num_gt'),
    test=dict(
        dataset_root='data/wifipose/test_data',
        query_selection='top_score',
        query_topk='num_gt'))

work_dir = 'result/origin_paper_top_score_eval'
