import argparse
import os

import mmcv
import numpy as np
import torch
from mmcv import Config
from mmcv.runner import load_checkpoint

from opera.datasets import build_dataset
from opera.models import build_model


def parse_args():
    parser = argparse.ArgumentParser(
        description='Export MMFi 2D-skeleton teacher tokens for CSI distillation.')
    parser.add_argument('config', help='teacher config path')
    parser.add_argument('--checkpoint', default=None, help='teacher checkpoint')
    parser.add_argument(
        '--split',
        default='train',
        choices=['train', 'val', 'test'],
        help='dataset split to export')
    parser.add_argument(
        '--out-dir',
        default='data/mmfi_teacher_tokens/t0',
        help='output token root; split name will be appended')
    parser.add_argument('--batch-size', type=int, default=256)
    parser.add_argument('--device', default='cuda:0')
    return parser.parse_args()


def build_split_dataset(cfg, split):
    data_cfg = cfg.data.train if split == 'train' else cfg.data.val
    data_cfg = data_cfg.copy()
    data_cfg['return_pose2d'] = True
    data_cfg['pose2d_confidence'] = True
    data_cfg['pipeline'] = []
    return build_dataset(data_cfg)


def main():
    args = parse_args()
    cfg = Config.fromfile(args.config)
    checkpoint = args.checkpoint or os.path.join(cfg.work_dir, 'latest.pth')
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')

    dataset = build_split_dataset(cfg, args.split)
    model = build_model(cfg.model)
    load_checkpoint(model, checkpoint, map_location='cpu')
    model.to(device)
    model.eval()

    out_dir = os.path.join(args.out_dir, args.split)
    mmcv.mkdir_or_exist(out_dir)

    batch_pose = []
    batch_infos = []
    total = len(dataset.data_infos)

    def flush():
        if not batch_pose:
            return
        pose = torch.from_numpy(np.stack(batch_pose, axis=0)).float().to(device)
        with torch.no_grad():
            tokens, pred_3d = model.extract_teacher_tokens(pose)
        tokens = tokens.detach().cpu().numpy().astype(np.float32)
        pred_3d = pred_3d.detach().cpu().numpy().astype(np.float32)
        for idx, info in enumerate(batch_infos):
            name = dataset.teacher_token_filename(info)
            np.savez_compressed(
                os.path.join(out_dir, name + '.npz'),
                tokens=tokens[idx],
                pred_3d=pred_3d[idx],
                sample_id=info['sample_id'])
        batch_pose.clear()
        batch_infos.clear()

    for idx, info in enumerate(dataset.data_infos):
        pose2d_seq, _ = dataset.load_pose2d_sequence(info)
        batch_pose.append(pose2d_seq)
        batch_infos.append(info)
        if len(batch_pose) >= args.batch_size:
            flush()
        if (idx + 1) % 1000 == 0:
            print(f'exported {idx + 1}/{total}')
    flush()
    print(f'exported {total} samples to {out_dir}')


if __name__ == '__main__':
    main()
