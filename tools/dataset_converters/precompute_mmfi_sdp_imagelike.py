import argparse
import os
import time
import multiprocessing as mp
from collections import OrderedDict

import mmcv
import numpy as np
import scipy.io as scio
from mmcv import Config, ConfigDict

from opera.datasets import build_dataset


_DATASET = None
_OUT_DIR = None
_EXT = None
_SKIP_EXISTING = False


def parse_args():
    parser = argparse.ArgumentParser(
        description='Precompute MMFi SDP image-like CSI features.')
    parser.add_argument('config', help='MMFi SDP experiment config path')
    parser.add_argument(
        '--out-dir',
        default='data/mmfi_sdp_imagelike_ctx7',
        help='output root; split name will be appended')
    parser.add_argument(
        '--splits',
        nargs='+',
        default=['train', 'val'],
        choices=['train', 'val', 'test'],
        help='dataset splits to precompute')
    parser.add_argument(
        '--num-workers',
        type=int,
        default=4,
        help='worker processes per split')
    parser.add_argument(
        '--chunksize',
        type=int,
        default=16,
        help='multiprocessing imap chunksize')
    parser.add_argument('--max-samples', type=int, default=None)
    parser.add_argument('--skip-existing', action='store_true')
    parser.add_argument(
        '--per-sample',
        action='store_true',
        help='process each sample independently instead of caching action CSI')
    parser.add_argument('--print-interval', type=int, default=500)
    return parser.parse_args()


def plain_cfg(obj):
    if isinstance(obj, (ConfigDict, dict)):
        return {key: plain_cfg(value) for key, value in obj.items()}
    if isinstance(obj, list):
        return [plain_cfg(value) for value in obj]
    if isinstance(obj, tuple):
        return tuple(plain_cfg(value) for value in obj)
    return obj


def get_split_cfg(cfg, split, max_samples=None):
    if split == 'train':
        data_cfg = cfg.data.train.copy()
    elif split == 'val':
        data_cfg = cfg.data.val.copy()
    elif split == 'test':
        data_cfg = cfg.data.test.copy()
    else:
        raise ValueError(f'Unsupported split: {split}')

    data_cfg['pipeline'] = []
    data_cfg['preprocess'] = 'sdp_imagelike'
    data_cfg.pop('sdp_offline_dir', None)
    data_cfg.pop('sdp_offline_ext', None)
    data_cfg.pop('strict_sdp_offline', None)
    if max_samples is not None:
        data_cfg['max_samples'] = max_samples
    return plain_cfg(data_cfg)


def init_worker(data_cfg, out_dir, ext, skip_existing):
    global _DATASET, _OUT_DIR, _EXT, _SKIP_EXISTING
    _DATASET = build_dataset(data_cfg)
    _OUT_DIR = out_dir
    _EXT = ext
    _SKIP_EXISTING = skip_existing


def process_index(index):
    info = _DATASET.data_infos[index]
    out_path = get_out_path(info)
    if _SKIP_EXISTING and os.path.exists(out_path):
        return 'skipped', None

    feature = _DATASET.load_sdp_imagelike(info).numpy().astype(np.float32)
    save_feature(out_path, feature)
    return 'saved', feature.shape


def get_out_path(info):
    filename = _DATASET.sdp_offline_filename(info) + _EXT
    return os.path.join(_OUT_DIR, filename)


def save_feature(out_path, feature):
    tmp_path = f'{out_path}.tmp.{os.getpid()}'
    with open(tmp_path, 'wb') as f:
        np.save(f, np.ascontiguousarray(feature.astype(np.float32)))
    os.replace(tmp_path, out_path)


def build_groups(data_infos):
    groups = OrderedDict()
    for index, info in enumerate(data_infos):
        key = (info['scene'], info['subject'], info['action'])
        groups.setdefault(key, []).append(index)
    return list(groups.values())


def load_amp_for_info(info):
    mat = scio.loadmat(info['csi_path'])
    amp = mat['CSIamp'].astype(np.float32)
    return _DATASET.replace_invalid(amp)


def load_action_amp_cache(info):
    action_dir = os.path.join(_DATASET.data_root, info['scene'], info['subject'],
                              info['action'])
    csi_dir = os.path.join(action_dir, 'wifi-csi')
    gt = np.load(info['gt_path'], mmap_mode='r')
    amps = []
    for frame_idx in range(gt.shape[0]):
        csi_path = os.path.join(csi_dir, f'frame{frame_idx + 1:03d}.mat')
        if not os.path.exists(csi_path) or os.path.getsize(csi_path) == 0:
            amps.append(None)
            continue
        mat = scio.loadmat(csi_path)
        amp = mat['CSIamp'].astype(np.float32)
        amps.append(_DATASET.replace_invalid(amp))
    return amps


def build_centered_amp_from_cache(info, amps):
    radius = int(_DATASET.sdp_cfg['context_radius'])
    num_frames = len(amps)
    center_amp = amps[info['frame_idx']]
    if center_amp is None:
        center_amp = load_amp_for_info(info)

    frames = []
    for offset in range(-radius, radius + 1):
        frame_idx = int(np.clip(info['frame_idx'] + offset, 0, num_frames - 1))
        amp = amps[frame_idx]
        if amp is None:
            amp = center_amp
        frames.append(amp)
    return np.concatenate(frames, axis=-1).astype(np.float32)


def process_group(indices):
    pending = []
    skipped = 0
    for index in indices:
        info = _DATASET.data_infos[index]
        out_path = get_out_path(info)
        if _SKIP_EXISTING and os.path.exists(out_path):
            skipped += 1
        else:
            pending.append((info, out_path))

    if not pending:
        return 'group', 0, skipped, len(indices), None

    amps = load_action_amp_cache(pending[0][0])
    saved = 0
    shape = None
    for info, out_path in pending:
        amp = build_centered_amp_from_cache(info, amps)
        feature = _DATASET.extract_sdp_imagelike_from_amp(amp)
        if _DATASET.normalize_csi:
            feature = _DATASET.normalize_feature(feature)
        save_feature(out_path, feature)
        saved += 1
        shape = feature.shape
    return 'group', saved, skipped, len(indices), shape


def format_seconds(seconds):
    seconds = int(seconds)
    hours, rem = divmod(seconds, 3600)
    minutes, seconds = divmod(rem, 60)
    if hours:
        return f'{hours}h{minutes:02d}m{seconds:02d}s'
    if minutes:
        return f'{minutes}m{seconds:02d}s'
    return f'{seconds}s'


def export_split(cfg, split, args):
    out_dir = os.path.join(args.out_dir, split)
    mmcv.mkdir_or_exist(out_dir)

    data_cfg = get_split_cfg(cfg, split, max_samples=args.max_samples)
    dataset = build_dataset(data_cfg)
    total = len(dataset.data_infos)
    print(f'[{split}] total samples: {total}, out_dir: {out_dir}')
    if total == 0:
        return

    saved = 0
    skipped = 0
    start = time.time()

    if args.per_sample:
        tasks = range(total)
        worker_fn = process_index
        task_chunksize = args.chunksize
        progress_mode = 'sample'
    else:
        tasks = build_groups(dataset.data_infos)
        worker_fn = process_group
        task_chunksize = 1
        progress_mode = 'group'
        print(f'[{split}] action groups: {len(tasks)}')

    if args.num_workers <= 1:
        init_worker(data_cfg, out_dir, '.npy', args.skip_existing)
        iterator = map(worker_fn, tasks)
        pool = None
    else:
        ctx = mp.get_context('fork')
        pool = ctx.Pool(
            args.num_workers,
            initializer=init_worker,
            initargs=(data_cfg, out_dir, '.npy', args.skip_existing))
        iterator = pool.imap_unordered(
            worker_fn, tasks, chunksize=task_chunksize)

    try:
        done = 0
        last_print = 0
        for task_result in iterator:
            if progress_mode == 'sample':
                status, shape = task_result
                count = 1
            else:
                _, group_saved, group_skipped, count, shape = task_result
                status = 'group'
                saved += group_saved
                skipped += group_skipped

            done += count
            if status == 'saved':
                saved += 1
            elif status == 'skipped':
                skipped += 1

            should_print = (
                done == 1 or done == total or
                done - last_print >= max(args.print_interval, 1))
            if should_print:
                last_print = done
                elapsed = time.time() - start
                rate = done / max(elapsed, 1e-6)
                eta = (total - done) / max(rate, 1e-6)
                shape_msg = f', shape={shape}' if shape is not None else ''
                print(
                    f'[{split}] {done}/{total} saved={saved} skipped={skipped} '
                    f'rate={rate:.2f}/s eta={format_seconds(eta)}{shape_msg}',
                    flush=True)
    except BaseException:
        if pool is not None:
            pool.terminate()
            pool.join()
            pool = None
        raise
    finally:
        if pool is not None:
            pool.close()
            pool.join()

    elapsed = time.time() - start
    print(
        f'[{split}] done. saved={saved}, skipped={skipped}, '
        f'elapsed={format_seconds(elapsed)}, out_dir={out_dir}')


def main():
    args = parse_args()
    cfg = Config.fromfile(args.config)
    for split in args.splits:
        export_split(cfg, split, args)


if __name__ == '__main__':
    main()
