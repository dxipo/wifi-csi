import os
from collections import OrderedDict

import numpy as np
import pywt
import scipy.io as scio
import torch
from torch.utils.data import Dataset as dataset

from mmdet.datasets.pipelines import Compose

from .builder import DATASETS


ALL_SUBJECTS = [
    'S01', 'S02', 'S03', 'S04', 'S05', 'S06', 'S07', 'S08', 'S09', 'S10',
    'S11', 'S12', 'S13', 'S14', 'S15', 'S16', 'S17', 'S18', 'S19', 'S20',
    'S21', 'S22', 'S23', 'S24', 'S25', 'S26', 'S27', 'S28', 'S29', 'S30',
    'S31', 'S32', 'S33', 'S34', 'S35', 'S36', 'S37', 'S38', 'S39', 'S40'
]

ALL_ACTIONS = [
    'A01', 'A02', 'A03', 'A04', 'A05', 'A06', 'A07', 'A08', 'A09',
    'A10', 'A11', 'A12', 'A13', 'A14', 'A15', 'A16', 'A17', 'A18',
    'A19', 'A20', 'A21', 'A22', 'A23', 'A24', 'A25', 'A26', 'A27'
]

PROTOCOL_ACTIONS = {
    'protocol1': [
        'A02', 'A03', 'A04', 'A05', 'A13', 'A14', 'A17', 'A18',
        'A19', 'A20', 'A21', 'A22', 'A23', 'A27'
    ],
    'protocol2': [
        'A01', 'A06', 'A07', 'A08', 'A09', 'A10', 'A11', 'A12',
        'A15', 'A16', 'A24', 'A25', 'A26'
    ],
    'protocol3': ALL_ACTIONS,
}

CROSS_SUBJECT_TRAIN = [
    'S01', 'S02', 'S03', 'S04', 'S06', 'S07', 'S08', 'S09',
    'S11', 'S12', 'S13', 'S14', 'S16', 'S17', 'S18', 'S19',
    'S21', 'S22', 'S23', 'S24', 'S26', 'S27', 'S28', 'S29',
    'S31', 'S32', 'S33', 'S34', 'S36', 'S37', 'S38', 'S39'
]

CROSS_SUBJECT_VAL = ['S05', 'S10', 'S15', 'S20', 'S25', 'S30', 'S35', 'S40']


def subject_to_scene(subject):
    subject_idx = int(subject[1:])
    scene_idx = (subject_idx - 1) // 10 + 1
    return f'E{scene_idx:02d}'


def actions_for_protocol(protocol):
    if protocol not in PROTOCOL_ACTIONS:
        raise ValueError(f'Unsupported MMFi protocol: {protocol}')
    return PROTOCOL_ACTIONS[protocol]


def build_subject_action_map(protocol='protocol2',
                             split_to_use='random_split',
                             split='train',
                             random_ratio=0.8,
                             random_seed=0,
                             subjects=None,
                             actions=None):
    """Build subject -> actions map following the official MMFi split style."""
    protocol_actions = list(actions or actions_for_protocol(protocol))
    split = 'val' if split in ('val', 'validation', 'test') else 'train'

    if split_to_use == 'random_split':
        result = {}
        seed = random_seed
        for action in protocol_actions:
            rng = np.random.RandomState(seed)
            indices = rng.permutation(len(ALL_SUBJECTS))
            split_idx = int(np.floor(random_ratio * len(ALL_SUBJECTS)))
            train_subjects = np.array(ALL_SUBJECTS)[indices[:split_idx]]
            val_subjects = np.array(ALL_SUBJECTS)[indices[split_idx:]]
            selected = train_subjects if split == 'train' else val_subjects
            for subject in selected.tolist():
                result.setdefault(subject, []).append(action)
            seed += 1
        return result

    if split_to_use == 'cross_scene_split':
        selected = ALL_SUBJECTS[:30] if split == 'train' else ALL_SUBJECTS[30:]
        return {subject: protocol_actions for subject in selected}

    if split_to_use == 'cross_subject_split':
        selected = CROSS_SUBJECT_TRAIN if split == 'train' else CROSS_SUBJECT_VAL
        return {subject: protocol_actions for subject in selected}

    if split_to_use == 'manual_split':
        if subjects is None:
            selected_subjects = ALL_SUBJECTS
        else:
            selected_subjects = subjects
        return {subject: protocol_actions for subject in selected_subjects}

    raise ValueError(f'Unsupported split_to_use: {split_to_use}')


@DATASETS.register_module()
class MMFiPoseDataset(dataset):
    CLASSES = ('person', )

    def __init__(self,
                 dataset_root,
                 pipeline,
                 mode,
                 protocol='protocol2',
                 split_to_use='random_split',
                 random_ratio=0.8,
                 random_seed=0,
                 use_phase=True,
                 normalize_csi=True,
                 preprocess='raw',
                 origin_linear_layout='time_token',
                 return_pose2d=False,
                 pose2d_window=9,
                 pose2d_normalize=True,
                 pose2d_confidence=False,
                 teacher_token_dir=None,
                 teacher_token_ext='.npz',
                 teacher_token_key='tokens',
                 strict_teacher_tokens=True,
                 shuffle_teacher_tokens=False,
                 shuffle_teacher_seed=42,
                 max_samples=None,
                 subjects=None,
                 actions=None,
                 **kwargs):
        self.data_root = dataset_root
        self.pipeline = Compose(pipeline)
        self.mode = mode
        self.protocol = protocol
        self.split_to_use = split_to_use
        self.random_ratio = random_ratio
        self.random_seed = random_seed
        self.use_phase = use_phase
        self.normalize_csi = normalize_csi
        self.preprocess = preprocess
        self.origin_linear_layout = origin_linear_layout
        self.return_pose2d = return_pose2d
        self.pose2d_window = pose2d_window
        self.pose2d_normalize = pose2d_normalize
        self.pose2d_confidence = pose2d_confidence
        self.teacher_token_dir = teacher_token_dir
        self.teacher_token_ext = teacher_token_ext
        self.teacher_token_key = teacher_token_key
        self.strict_teacher_tokens = strict_teacher_tokens
        self.shuffle_teacher_tokens = shuffle_teacher_tokens
        self.shuffle_teacher_seed = shuffle_teacher_seed
        self.max_samples = max_samples
        self.subject_action_map = build_subject_action_map(
            protocol=protocol,
            split_to_use=split_to_use,
            split=mode,
            random_ratio=random_ratio,
            random_seed=random_seed,
            subjects=subjects,
            actions=actions)
        self.data_infos = self.load_data_infos()
        self._set_group_flag()

    def __len__(self):
        return len(self.data_infos)

    def __getitem__(self, index):
        result = self.prepare_sample(index)
        return self.pipeline(result)

    def _set_group_flag(self):
        self.flag = np.zeros(len(self), dtype=np.uint8)

    def load_data_infos(self):
        infos = []
        for subject, actions in sorted(self.subject_action_map.items()):
            scene = subject_to_scene(subject)
            for action in actions:
                action_dir = os.path.join(self.data_root, scene, subject, action)
                gt_path = os.path.join(action_dir, 'ground_truth.npy')
                csi_dir = os.path.join(action_dir, 'wifi-csi')
                if not os.path.exists(gt_path) or not os.path.isdir(csi_dir):
                    continue
                gt = np.load(gt_path, mmap_mode='r')
                for frame_idx in range(gt.shape[0]):
                    frame_name = f'frame{frame_idx + 1:03d}'
                    csi_path = os.path.join(csi_dir, frame_name + '.mat')
                    if not os.path.exists(csi_path) or os.path.getsize(csi_path) == 0:
                        continue
                    infos.append(
                        dict(
                            scene=scene,
                            subject=subject,
                            action=action,
                            frame_idx=frame_idx,
                            frame_name=frame_name,
                            csi_path=csi_path,
                            gt_path=gt_path,
                            sample_id=f'{scene}/{subject}/{action}/{frame_name}'))
                    if self.max_samples is not None and len(infos) >= self.max_samples:
                        return infos
        return infos

    def prepare_sample(self, index):
        info = self.data_infos[index]
        csi = self.load_csi(info['csi_path'])
        keypoint = np.load(info['gt_path'])[info['frame_idx']].astype(np.float32)
        keypoint = torch.from_numpy(keypoint[None, ...]).float()

        gt_labels = np.zeros(1, dtype=np.int64)
        gt_bboxes = torch.empty((0, 4), dtype=torch.float32)
        gt_areas = torch.empty((0,), dtype=torch.float32)
        result = dict(
            img=csi,
            gt_keypoints=keypoint,
            gt_labels=gt_labels,
            gt_bboxes=gt_bboxes,
            gt_areas=gt_areas,
            img_name=info['sample_id'])
        if self.return_pose2d:
            pose2d_seq, pose2d_mask = self.load_pose2d_sequence(info)
            result['pose2d_seq'] = torch.from_numpy(pose2d_seq).float()
            result['pose2d_mask'] = torch.from_numpy(pose2d_mask).float()
        if self.teacher_token_dir is not None:
            teacher = self.load_teacher_tokens(info)
            result['teacher_tokens'] = torch.from_numpy(teacher).float()
        return result

    def load_csi(self, csi_path):
        mat = scio.loadmat(csi_path)
        if self.preprocess == 'origin':
            csi = self.load_origin_style_csi(mat)
            return torch.from_numpy(csi).float()
        if self.preprocess != 'raw':
            raise ValueError(f'Unsupported MMFi CSI preprocess: {self.preprocess}')

        amp = mat['CSIamp'].astype(np.float32)
        features = [self.clean_csi(amp)]
        if self.use_phase:
            phase = mat['CSIphase'].astype(np.float32)
            features.append(self.clean_csi(phase))
        csi = np.concatenate(features, axis=0)
        return torch.from_numpy(csi).float()

    def load_pose2d_sequence(self, info):
        if self.pose2d_window % 2 != 1:
            raise ValueError('pose2d_window must be odd so a center frame exists.')
        radius = self.pose2d_window // 2
        action_dir = os.path.join(self.data_root, info['scene'], info['subject'],
                                  info['action'])
        rgb_dir = os.path.join(action_dir, 'rgb')
        gt = np.load(info['gt_path'], mmap_mode='r')
        num_frames = gt.shape[0]
        frames = []
        masks = []
        for offset in range(-radius, radius + 1):
            frame_idx = int(np.clip(info['frame_idx'] + offset, 0,
                                    num_frames - 1))
            pose_path = os.path.join(rgb_dir, f'frame{frame_idx + 1:03d}.npy')
            pose = self.load_pose2d_file(pose_path)
            mask = self.pose2d_valid_mask(pose)
            frames.append(pose)
            masks.append(mask)
        pose2d = np.stack(frames, axis=0).astype(np.float32)
        mask = np.stack(masks, axis=0).astype(np.float32)
        if self.pose2d_normalize:
            pose2d = self.normalize_pose2d_sequence(pose2d, mask)
        if self.pose2d_confidence:
            pose2d = np.concatenate((pose2d, mask[..., None]), axis=-1)
        return pose2d.astype(np.float32), mask.astype(np.float32)

    def load_pose2d_file(self, pose_path):
        if not os.path.exists(pose_path):
            return np.zeros((17, 2), dtype=np.float32)
        pose = np.load(pose_path).astype(np.float32)
        if pose.shape[-1] > 2:
            pose = pose[..., :2]
        return pose.reshape(17, 2).astype(np.float32)

    @staticmethod
    def pose2d_valid_mask(pose):
        finite = np.isfinite(pose).all(axis=-1)
        nonzero = np.abs(pose).sum(axis=-1) > 1e-6
        return (finite & nonzero).astype(np.float32)

    @staticmethod
    def normalize_pose2d_sequence(pose2d, mask, eps=1e-6):
        out = pose2d.copy().astype(np.float32)
        for t in range(out.shape[0]):
            valid = mask[t] > 0
            if valid.sum() < 2:
                out[t] = 0.0
                continue
            if valid[11] and valid[12]:
                center = (out[t, 11] + out[t, 12]) / 2.0
            else:
                center = out[t, valid].mean(axis=0)
            valid_xy = out[t, valid]
            min_xy = valid_xy.min(axis=0)
            max_xy = valid_xy.max(axis=0)
            scale = float(np.max(max_xy - min_xy))
            if scale < eps:
                scale = float(np.linalg.norm(out[t, valid] - center, axis=-1).mean())
            scale = max(scale, eps)
            out[t] = (out[t] - center) / scale
            out[t, ~valid] = 0.0
        return out

    def load_teacher_tokens(self, info):
        token_path = os.path.join(
            self.teacher_token_dir,
            self.teacher_token_filename(info) + self.teacher_token_ext)
        if not os.path.exists(token_path):
            if self.strict_teacher_tokens:
                raise FileNotFoundError(f'Missing teacher token file: {token_path}')
            return np.zeros((18, 256), dtype=np.float32)
        data = np.load(token_path)
        if isinstance(data, np.lib.npyio.NpzFile):
            tokens = data[self.teacher_token_key] if self.teacher_token_key in data else data[data.files[0]]
        else:
            tokens = data
        tokens = tokens.astype(np.float32)
        if self.shuffle_teacher_tokens:
            sample_offset = sum(ord(ch) for ch in info['sample_id'])
            rng = np.random.RandomState(self.shuffle_teacher_seed + sample_offset)
            tokens = tokens[rng.permutation(tokens.shape[0])]
        return tokens

    @staticmethod
    def teacher_token_filename(info):
        return info['sample_id'].replace('/', '__')

    def load_origin_style_csi(self, mat):
        amp = self.replace_invalid(mat['CSIamp'].astype(np.float32))
        phase = self.replace_invalid(mat['CSIphase'].astype(np.float32))
        complex_csi = amp.astype(np.float64) * np.exp(1j * phase.astype(np.float64))

        csi_amp = self.dwt_amp(complex_csi)
        if self.use_phase:
            csi_phase = np.angle(self.phase_deno(complex_csi))
            if self.origin_linear_layout == 'time_token':
                csi = np.concatenate((csi_amp, csi_phase), axis=1)
                csi = np.transpose(csi, (0, 2, 1))
            elif self.origin_linear_layout == 'subcarrier_token':
                csi = np.concatenate((csi_amp, csi_phase), axis=2)
            else:
                raise ValueError(
                    f'Unsupported origin_linear_layout: {self.origin_linear_layout}')
        else:
            if self.origin_linear_layout == 'time_token':
                csi = np.transpose(csi_amp, (0, 2, 1))
            elif self.origin_linear_layout == 'subcarrier_token':
                csi = csi_amp
            else:
                raise ValueError(
                    f'Unsupported origin_linear_layout: {self.origin_linear_layout}')

        csi = csi.astype(np.float32)
        if self.normalize_csi:
            csi = self.normalize_feature(csi)
        return csi.astype(np.float32)

    def clean_csi(self, csi):
        csi = self.replace_invalid(csi)
        if self.normalize_csi:
            csi = self.normalize_feature(csi)
        return csi

    def replace_invalid(self, csi):
        csi = csi.copy()
        csi[np.isinf(csi)] = np.nan
        if np.isnan(csi).any():
            mean = np.nanmean(csi)
            if np.isnan(mean):
                mean = 0.0
            csi[np.isnan(csi)] = mean
        return csi

    @staticmethod
    def normalize_feature(csi):
        mean = csi.mean()
        std = csi.std()
        return (csi - mean) / (std + 1e-6)

    def dwt_amp(self, csi):
        wavelet = pywt.Wavelet('dB11')
        coeffs = pywt.wavedec(np.abs(csi), wavelet, 'sym', axis=-1)
        csi_amp = pywt.waverec(coeffs, wavelet, axis=-1)
        return self.match_last_dim(csi_amp, csi.shape[-1])

    def phase_deno(self, csi):
        if csi.ndim != 3:
            raise ValueError(
                f'MMFi origin-style phase denoise expects (3,N,T), got {csi.shape}')
        return self.csi_sanitization(csi)

    def csi_sanitization(self, csi_rx):
        if csi_rx.shape[0] != 3:
            raise ValueError(
                f'MMFi origin-style phase sanitization expects 3 channels, got {csi_rx.shape}')
        one_csi = csi_rx[0, :, :]
        two_csi = csi_rx[1, :, :]
        three_csi = csi_rx[2, :, :]
        pi = np.pi
        num_antennas = 3
        num_subcarriers = one_csi.shape[0]
        num_packets = one_csi.shape[1]
        subcarrier_interval = 312.5 * 2
        csi_phase = np.zeros((num_antennas, num_subcarriers, num_packets))
        ai = np.tile(
            2 * pi * subcarrier_interval * np.arange(num_subcarriers),
            num_antennas)
        bi = np.ones(num_antennas * num_subcarriers)
        temp = np.tile(np.arange(num_subcarriers), num_antennas).reshape(
            num_antennas, num_subcarriers)

        for packet_idx in range(num_packets):
            csi_phase[0, :, packet_idx] = np.unwrap(
                np.angle(one_csi[:, packet_idx]))
            csi_phase[1, :, packet_idx] = np.unwrap(
                csi_phase[0, :, packet_idx] +
                np.angle(two_csi[:, packet_idx] * np.conj(one_csi[:, packet_idx])))
            csi_phase[2, :, packet_idx] = np.unwrap(
                csi_phase[1, :, packet_idx] +
                np.angle(three_csi[:, packet_idx] * np.conj(two_csi[:, packet_idx])))

            ci = np.concatenate((
                csi_phase[0, :, packet_idx],
                csi_phase[1, :, packet_idx],
                csi_phase[2, :, packet_idx]))
            A = np.dot(ai, ai)
            B = np.dot(ai, bi)
            C = np.dot(bi, bi)
            D = np.dot(ai, ci)
            E = np.dot(bi, ci)
            denom = A * C - B ** 2
            rho_opt = (B * E - C * D) / denom
            beta_opt = (B * D - A * E) / denom
            csi_phase[:, :, packet_idx] = (
                csi_phase[:, :, packet_idx] +
                2 * pi * subcarrier_interval * temp * rho_opt + beta_opt)

        antenna_one = np.abs(one_csi) * np.exp(1j * csi_phase[0, :, :])
        antenna_two = np.abs(two_csi) * np.exp(1j * csi_phase[1, :, :])
        antenna_three = np.abs(three_csi) * np.exp(1j * csi_phase[2, :, :])
        return np.concatenate((
            np.expand_dims(antenna_one, axis=0),
            np.expand_dims(antenna_two, axis=0),
            np.expand_dims(antenna_three, axis=0)))

    @staticmethod
    def match_last_dim(array, target_size):
        if array.shape[-1] == target_size:
            return array
        if array.shape[-1] > target_size:
            return array[..., :target_size]
        pad_width = [(0, 0)] * array.ndim
        pad_width[-1] = (0, target_size - array.shape[-1])
        return np.pad(array, pad_width, mode='edge')

    def get_gt_keypoints(self, index):
        info = self.data_infos[index]
        return np.load(info['gt_path'])[info['frame_idx']].astype(np.float32)

    def evaluate(self,
                 results,
                 metric='mpjpe',
                 logger=None,
                 jsonfile_prefix=None,
                 classwise=False,
                 proposal_nums=(100, 300, 1000),
                 iou_thrs=None,
                 metric_items=None):
        mpjpe_abs = []
        mpjpe_pelvis = []
        pa_mpjpe = []
        mpjpe_x = []
        mpjpe_y = []
        mpjpe_z = []

        for index, result in enumerate(results):
            gt = self.get_gt_keypoints(index)
            pred = self.select_prediction(result, gt)
            if pred is None:
                continue
            err = np.linalg.norm(pred - gt, axis=-1)
            mpjpe_abs.append(err.mean() * 1000.0)
            mpjpe_x.append(np.abs(pred[:, 0] - gt[:, 0]).mean() * 1000.0)
            mpjpe_y.append(np.abs(pred[:, 1] - gt[:, 1]).mean() * 1000.0)
            mpjpe_z.append(np.abs(pred[:, 2] - gt[:, 2]).mean() * 1000.0)

            pred_rel = pred - self.pelvis(pred)
            gt_rel = gt - self.pelvis(gt)
            mpjpe_pelvis.append(np.linalg.norm(pred_rel - gt_rel, axis=-1).mean() * 1000.0)

            pred_pa = self.compute_similarity_transform(pred, gt)
            pa_mpjpe.append(np.linalg.norm(pred_pa - gt, axis=-1).mean() * 1000.0)

        if len(mpjpe_abs) == 0:
            return OrderedDict(
                mpjpe=np.nan,
                mpjpe_abs=np.nan,
                mpjpe_pelvis=np.nan,
                pa_mpjpe=np.nan,
                mpjpe_x=np.nan,
                mpjpe_y=np.nan,
                mpjpe_z=np.nan)

        result = OrderedDict()
        result['mpjpe'] = float(np.mean(mpjpe_abs))
        result['mpjpe_abs'] = float(np.mean(mpjpe_abs))
        result['mpjpe_pelvis'] = float(np.mean(mpjpe_pelvis))
        result['pa_mpjpe'] = float(np.mean(pa_mpjpe))
        result['mpjpe_x'] = float(np.mean(mpjpe_x))
        result['mpjpe_y'] = float(np.mean(mpjpe_y))
        result['mpjpe_z'] = float(np.mean(mpjpe_z))
        return result

    def select_prediction(self, result, gt):
        if not isinstance(result, (list, tuple)) or len(result) < 2:
            return None
        pred_keypoints = result[1][0]
        if pred_keypoints is None or len(pred_keypoints) == 0:
            return None
        distances = np.linalg.norm(pred_keypoints - gt[None, ...], axis=-1).mean(axis=-1)
        return pred_keypoints[int(np.argmin(distances))].astype(np.float32)

    @staticmethod
    def pelvis(keypoints):
        return ((keypoints[11] + keypoints[12]) / 2.0)[None, :]

    @staticmethod
    def compute_similarity_transform(pred, gt):
        pred_t = pred.T
        gt_t = gt.T
        mu_pred = pred_t.mean(axis=1, keepdims=True)
        mu_gt = gt_t.mean(axis=1, keepdims=True)
        pred_centered = pred_t - mu_pred
        gt_centered = gt_t - mu_gt
        var_pred = np.sum(pred_centered ** 2)
        if var_pred < 1e-12:
            return pred.copy()
        cov = pred_centered @ gt_centered.T
        u, s, vh = np.linalg.svd(cov)
        v = vh.T
        z = np.eye(3, dtype=np.float32)
        z[-1, -1] = np.sign(np.linalg.det(v @ u.T))
        rotation = v @ z @ u.T
        scale = np.trace(rotation @ cov) / var_pred
        translation = mu_gt - scale * rotation @ mu_pred
        aligned = scale * rotation @ pred_t + translation
        return aligned.T.astype(np.float32)
