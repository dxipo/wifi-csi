import glob
import os
from collections import OrderedDict

import h5py
import numpy as np
import torch
from mmdet.datasets.pipelines import Compose
from torch.utils.data import Dataset

from .builder import DATASETS


@DATASETS.register_module()
class WiPose18Dataset(Dataset):
    """Official Wi-Pose dataset with 18 AlphaPose joints and 2D labels."""

    CLASSES = ('person', )
    JOINT_NAMES = (
        'Nose', 'Neck', 'R.Shoulder', 'R.Elbow', 'R.Wrist',
        'L.Shoulder', 'L.Elbow', 'L.Wrist', 'R.Hip', 'R.Knee',
        'R.Ankle', 'L.Hip', 'L.Knee', 'L.Ankle', 'R.Eye', 'L.Eye',
        'R.Ear', 'L.Ear')

    def __init__(self,
                 dataset_root,
                 pipeline,
                 mode,
                 image_size=(640, 480),
                 normalize_csi=False,
                 query_selection='top_score',
                 confidence_threshold=0.0,
                 pck_thresholds=(20, 30, 40, 50),
                 max_samples=None,
                 **kwargs):
        self.data_root = dataset_root
        self.pipeline = Compose(pipeline)
        self.mode = mode
        self.image_width = float(image_size[0])
        self.image_height = float(image_size[1])
        self.normalize_csi = normalize_csi
        self.query_selection = query_selection
        self.confidence_threshold = float(confidence_threshold)
        self.pck_thresholds = tuple(int(x) for x in pck_thresholds)

        split = 'Train' if mode == 'train' else 'Test'
        split_dir = os.path.join(self.data_root, split)
        if not os.path.isdir(split_dir):
            raise FileNotFoundError(f'WiPose split directory not found: {split_dir}')
        self.data_infos = sorted(glob.glob(os.path.join(split_dir, '*.mat')))
        if max_samples is not None:
            self.data_infos = self.data_infos[:int(max_samples)]
        if not self.data_infos:
            raise RuntimeError(f'No WiPose .mat samples found under {split_dir}')
        self.flag = np.zeros(len(self.data_infos), dtype=np.uint8)

    def __len__(self):
        return len(self.data_infos)

    def __getitem__(self, index):
        csi, joints, _ = self.load_sample(index)
        scale = np.array(
            [self.image_width, self.image_height], dtype=np.float32)
        joints_normalized = joints / scale

        result = dict(
            img=torch.from_numpy(csi).float(),
            gt_keypoints=torch.from_numpy(joints_normalized[None]).float(),
            gt_labels=np.zeros(1, dtype=np.int64),
            gt_bboxes=torch.empty((0, 4), dtype=torch.float32),
            gt_areas=torch.empty((0,), dtype=torch.float32),
            img_name=os.path.basename(self.data_infos[index]))
        return self.pipeline(result)

    def load_sample(self, index):
        with h5py.File(self.data_infos[index], 'r') as mat:
            csi = self._read_numeric(mat['CSI'])
            skeleton = np.asarray(mat['SkeletonPoints'][()]).reshape(-1)

        if csi.shape == (5, 30, 3, 3):
            csi = np.transpose(csi, (2, 3, 1, 0))
        if csi.shape != (3, 3, 30, 5):
            raise ValueError(
                f'Expected WiPose CSI shape (3,3,30,5), got {csi.shape} '
                f'in {self.data_infos[index]}')
        csi = np.abs(csi).astype(np.float32)
        csi = self._replace_invalid(csi)
        if self.normalize_csi:
            csi = (csi - csi.mean()) / (csi.std() + 1e-6)

        # (Tx, Rx, subcarrier, time) -> 45 link-time tokens x 30 features.
        csi = np.transpose(csi, (0, 1, 3, 2))

        if skeleton.size != len(self.JOINT_NAMES) * 3:
            raise ValueError(
                f'Expected 54 WiPose label values, got {skeleton.size} '
                f'in {self.data_infos[index]}')
        skeleton = skeleton.reshape(3, len(self.JOINT_NAMES)).T
        joints = skeleton[:, :2].astype(np.float32)
        confidence = skeleton[:, 2].astype(np.float32)
        return np.ascontiguousarray(csi), joints, confidence

    @staticmethod
    def _read_numeric(dataset):
        value = dataset[()]
        if value.dtype.fields and {'real', 'imag'} <= set(value.dtype.fields):
            value = value['real'] + 1j * value['imag']
        return np.asarray(value)

    @staticmethod
    def _replace_invalid(value):
        value = value.copy()
        finite = np.isfinite(value)
        if finite.all():
            return value
        replacement = value[finite].mean() if finite.any() else 0.0
        value[~finite] = replacement
        return value

    def evaluate(self,
                 results,
                 metric='pck',
                 logger=None,
                 jsonfile_prefix=None,
                 classwise=False,
                 proposal_nums=(100, 300, 1000),
                 iou_thrs=None,
                 metric_items=None):
        errors = []
        valid_masks = []
        scale = np.array(
            [self.image_width, self.image_height], dtype=np.float32)

        for index, result in enumerate(results):
            _, gt, confidence = self.load_sample(index)
            pred = self.select_prediction(result)
            if pred is None:
                continue
            pred = pred[..., :2].astype(np.float32) * scale
            valid = (np.isfinite(gt).all(axis=-1) &
                     np.isfinite(pred).all(axis=-1) &
                     (confidence >= self.confidence_threshold))
            errors.append(np.linalg.norm(pred - gt, axis=-1))
            valid_masks.append(valid)

        if not errors:
            return OrderedDict(
                [('mpjpe_pixel', np.nan)] +
                [(f'pck{threshold}', np.nan)
                 for threshold in self.pck_thresholds])

        errors = np.stack(errors)
        valid_masks = np.stack(valid_masks)
        metrics = OrderedDict()
        metrics['mpjpe_pixel'] = float(errors[valid_masks].mean())
        for threshold in self.pck_thresholds:
            correct = errors <= threshold
            metrics[f'pck{threshold}'] = float(
                correct[valid_masks].mean() * 100.0)
            for joint_index, joint_name in enumerate(self.JOINT_NAMES):
                joint_valid = valid_masks[:, joint_index]
                value = (correct[joint_valid, joint_index].mean() * 100.0
                         if joint_valid.any() else np.nan)
                metrics[f'pck{threshold}/{joint_name}'] = float(value)
        return metrics

    @staticmethod
    def select_prediction(result):
        if not isinstance(result, (list, tuple)) or len(result) < 2:
            return None
        det_bboxes = result[0][0]
        pred_keypoints = result[1][0]
        if pred_keypoints is None or len(pred_keypoints) == 0:
            return None
        if (det_bboxes is not None and len(det_bboxes) == len(pred_keypoints)
                and det_bboxes.shape[-1] >= 5):
            index = int(np.argmax(det_bboxes[:, 4]))
        else:
            index = 0
        return pred_keypoints[index]
