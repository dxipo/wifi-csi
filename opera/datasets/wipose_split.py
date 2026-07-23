import glob
import os

import numpy as np


def build_wipose_data_infos(dataset_root,
                            mode,
                            split_strategy='official',
                            random_ratio=0.8,
                            random_seed=0):
    """Return WiPose sample paths for the requested reproducible split."""
    is_train = mode == 'train'
    if split_strategy == 'official':
        split = 'Train' if is_train else 'Test'
        sample_paths = sorted(glob.glob(
            os.path.join(dataset_root, split, '*.mat')))
    elif split_strategy == 'packet_random':
        if not 0.0 < random_ratio < 1.0:
            raise ValueError(
                f'random_ratio must be in (0, 1), got {random_ratio}')
        sample_paths = []
        for split in ('Train', 'Test'):
            sample_paths.extend(glob.glob(
                os.path.join(dataset_root, split, '*.mat')))
        sample_paths = sorted(sample_paths)

        rng = np.random.RandomState(random_seed)
        indices = rng.permutation(len(sample_paths))
        # Avoid losing an exact integer split to binary floating-point error,
        # for example 0.7 * 166600 evaluating just below 116620.
        split_index = int(np.floor(
            random_ratio * len(sample_paths) + 1e-8))
        indices = indices[:split_index] if is_train else indices[split_index:]
        sample_paths = [sample_paths[index] for index in indices]
    else:
        raise ValueError(f'Unsupported WiPose split strategy: {split_strategy}')

    if not sample_paths:
        raise RuntimeError(
            f'No WiPose samples found under {dataset_root} for '
            f'{split_strategy}/{mode}')
    return sample_paths
