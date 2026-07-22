#!/usr/bin/env python3
"""Compute WiPose pixel PCK from saved model predictions."""

import argparse
import glob
import json
import os

import h5py
import mmcv
import numpy as np


JOINT_NAMES = (
    'Nose', 'Neck', 'R.Shoulder', 'R.Elbow', 'R.Wrist',
    'L.Shoulder', 'L.Elbow', 'L.Wrist', 'R.Hip', 'R.Knee',
    'R.Ankle', 'L.Hip', 'L.Knee', 'L.Ankle', 'R.Eye', 'L.Eye',
    'R.Ear', 'L.Ear')


def parse_args():
    parser = argparse.ArgumentParser(
        description='Compute overall and per-joint WiPose PCK after training.')
    parser.add_argument(
        'predictions', help='Pickle output produced by tools/test.py --out')
    parser.add_argument(
        '--dataset-root',
        default=(
            '/home/xl/CSI/Person-in-WiFi-3D-repo/data/'
            'wipose18_official/Wi-Pose'))
    parser.add_argument('--split', default='Test', choices=['Train', 'Test'])
    parser.add_argument(
        '--thresholds', type=float, nargs='+', default=[20, 30, 40, 50])
    parser.add_argument(
        '--image-size', type=float, nargs=2, default=[640, 480],
        metavar=('WIDTH', 'HEIGHT'))
    parser.add_argument('--confidence-threshold', type=float, default=0.0)
    parser.add_argument(
        '--selection', choices=['top_score', 'oracle'], default='top_score',
        help='Use top_score for reported results; oracle is diagnostic only.')
    parser.add_argument('--output-json')
    return parser.parse_args()


def load_ground_truth(path):
    with h5py.File(path, 'r') as mat:
        skeleton = np.asarray(mat['SkeletonPoints'][()]).reshape(-1)
    if skeleton.size != len(JOINT_NAMES) * 3:
        raise ValueError(f'Expected 54 label values, got {skeleton.size}: {path}')
    skeleton = skeleton.reshape(3, len(JOINT_NAMES)).T
    return (skeleton[:, :2].astype(np.float32),
            skeleton[:, 2].astype(np.float32))


def unpack_predictions(result):
    if not isinstance(result, (list, tuple)) or len(result) < 2:
        return None, None
    det_bboxes = result[0][0]
    keypoints = result[1][0]
    if keypoints is None or len(keypoints) == 0:
        return None, None
    return det_bboxes, np.asarray(keypoints)


def select_prediction(det_bboxes, keypoints, gt_normalized, selection):
    if selection == 'oracle':
        distances = np.linalg.norm(
            keypoints[..., :2] - gt_normalized[None], axis=-1).mean(axis=-1)
        index = int(np.argmin(distances))
    elif (det_bboxes is not None and len(det_bboxes) == len(keypoints)
          and det_bboxes.shape[-1] >= 5):
        index = int(np.argmax(det_bboxes[:, 4]))
    else:
        index = 0
    return keypoints[index, :, :2].astype(np.float32)


def evaluate(args):
    results = mmcv.load(args.predictions)
    sample_paths = sorted(glob.glob(
        os.path.join(args.dataset_root, args.split, '*.mat')))
    if len(results) != len(sample_paths):
        raise ValueError(
            f'Prediction/sample count mismatch: {len(results)} vs '
            f'{len(sample_paths)}')

    scale = np.asarray(args.image_size, dtype=np.float32)
    errors = []
    valid_masks = []
    for path, result in zip(sample_paths, results):
        gt, confidence = load_ground_truth(path)
        det_bboxes, keypoints = unpack_predictions(result)
        if keypoints is None:
            continue
        pred = select_prediction(
            det_bboxes, keypoints, gt / scale, args.selection) * scale
        valid = (np.isfinite(gt).all(axis=-1)
                 & np.isfinite(pred).all(axis=-1)
                 & (confidence >= args.confidence_threshold))
        errors.append(np.linalg.norm(pred - gt, axis=-1))
        valid_masks.append(valid)

    if not errors:
        raise RuntimeError('No valid predictions were found.')
    errors = np.stack(errors)
    valid_masks = np.stack(valid_masks)
    if not valid_masks.any():
        raise RuntimeError('No keypoints passed the confidence threshold.')

    metrics = {
        'selection': args.selection,
        'samples': int(errors.shape[0]),
        'valid_keypoints': int(valid_masks.sum()),
        'mpjpe': float(errors[valid_masks].mean()),
        'pck': {},
        'per_joint_pck': {},
    }
    for threshold in args.thresholds:
        name = f'pck{threshold:g}'
        correct = errors <= threshold
        metrics['pck'][name] = float(correct[valid_masks].mean() * 100.0)
        joint_values = {}
        for joint_index, joint_name in enumerate(JOINT_NAMES):
            joint_valid = valid_masks[:, joint_index]
            value = (correct[joint_valid, joint_index].mean() * 100.0
                     if joint_valid.any() else np.nan)
            joint_values[joint_name] = float(value)
        metrics['per_joint_pck'][name] = joint_values
    return metrics


def print_metrics(metrics):
    print(f"Selection: {metrics['selection']}")
    print(f"Samples: {metrics['samples']}")
    print(f"Valid keypoints: {metrics['valid_keypoints']}")
    print(f"MPJPE (pixel): {metrics['mpjpe']:.4f}")
    headers = list(metrics['pck'])
    print('\nKeypoint\t' + '\t'.join(headers))
    for joint_name in JOINT_NAMES:
        values = [
            metrics['per_joint_pck'][name][joint_name] for name in headers]
        print(joint_name + '\t' + '\t'.join(f'{x:.2f}' for x in values))
    averages = [metrics['pck'][name] for name in headers]
    print('Average\t' + '\t'.join(f'{x:.2f}' for x in averages))


def main():
    args = parse_args()
    metrics = evaluate(args)
    print_metrics(metrics)
    if args.output_json:
        output_dir = os.path.dirname(os.path.abspath(args.output_json))
        os.makedirs(output_dir, exist_ok=True)
        with open(args.output_json, 'w', encoding='utf-8') as output_file:
            json.dump(metrics, output_file, indent=2, ensure_ascii=True)
        print(f'\nSaved JSON: {args.output_json}')


if __name__ == '__main__':
    main()
