# Copyright (c) Hikvision Research Institute. All rights reserved.
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmdet.models.detectors.base import BaseDetector

from opera.core.keypoint import bbox_kpt2result
from ..builder import DETECTORS


@DETECTORS.register_module()
class SkeletonPoseTeacher(BaseDetector):
    """2D skeleton sequence teacher for 3D pose token distillation.

    The model treats a 2D skeleton sequence as an image-like tensor with
    channels x/y[/mask], temporal positions as height, and joints as width.
    It predicts the center-frame 3D pose and exposes pose tokens for offline
    distillation into the CSI-only student.
    """

    def __init__(self,
                 num_keypoints=17,
                 in_channels=3,
                 embed_dims=256,
                 num_layers=4,
                 num_heads=8,
                 feedforward_channels=512,
                 dropout=0.1,
                 max_seq_len=31,
                 loss_abs_weight=1.0,
                 loss_rel_weight=1.0,
                 pelvis_indices=(11, 12),
                 init_cfg=None,
                 train_cfg=None,
                 test_cfg=None):
        super(SkeletonPoseTeacher, self).__init__(init_cfg)
        self.num_keypoints = num_keypoints
        self.in_channels = in_channels
        self.embed_dims = embed_dims
        self.loss_abs_weight = loss_abs_weight
        self.loss_rel_weight = loss_rel_weight
        self.pelvis_indices = pelvis_indices

        self.pose_stem = nn.Sequential(
            nn.Conv2d(in_channels, embed_dims, kernel_size=3, padding=1),
            nn.BatchNorm2d(embed_dims),
            nn.ReLU(inplace=True),
            nn.Conv2d(embed_dims, embed_dims, kernel_size=1),
            nn.ReLU(inplace=True))
        self.time_embed = nn.Parameter(torch.zeros(max_seq_len, embed_dims))
        self.joint_embed = nn.Parameter(torch.zeros(num_keypoints, embed_dims))
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dims,
            nhead=num_heads,
            dim_feedforward=feedforward_channels,
            dropout=dropout,
            activation='relu')
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers)
        self.token_norm = nn.LayerNorm(embed_dims)
        self.pose_head = nn.Sequential(
            nn.Linear(embed_dims, embed_dims),
            nn.ReLU(inplace=True),
            nn.Linear(embed_dims, 3))

    def init_weights(self):
        nn.init.normal_(self.time_embed, std=0.02)
        nn.init.normal_(self.joint_embed, std=0.02)
        for module in self.modules():
            if isinstance(module, (nn.Linear, nn.Conv2d)):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

    def extract_feat(self, imgs):
        return self.encode_pose(imgs)[0]

    def forward_train(self,
                      img,
                      img_metas,
                      gt_keypoints,
                      pose2d_seq,
                      pose2d_mask=None,
                      **kwargs):
        pose_tokens, pred_3d = self.encode_pose(pose2d_seq)
        target = self._format_gt_keypoints(gt_keypoints, pred_3d.device)

        loss_abs = F.smooth_l1_loss(pred_3d, target, reduction='mean')
        pred_rel = pred_3d - self._pelvis(pred_3d)
        target_rel = target - self._pelvis(target)
        loss_rel = F.smooth_l1_loss(pred_rel, target_rel, reduction='mean')

        return dict(
            loss_pose_abs=loss_abs * self.loss_abs_weight,
            loss_pose_rel=loss_rel * self.loss_rel_weight,
            pose_token_std=pose_tokens.detach().std())

    def simple_test(self, img, img_metas, pose2d_seq=None, rescale=False, **kwargs):
        if pose2d_seq is None:
            raise ValueError('SkeletonPoseTeacher.simple_test requires pose2d_seq.')
        _, pred_3d = self.encode_pose(pose2d_seq)
        pred_3d = pred_3d.detach()
        results = []
        for batch_idx in range(pred_3d.size(0)):
            kpts = pred_3d[batch_idx:batch_idx + 1]
            bboxes = kpts.new_zeros((1, 5))
            bboxes[:, 4] = 1.0
            labels = kpts.new_zeros((1,), dtype=torch.long)
            results.append(
                bbox_kpt2result(bboxes, labels, kpts, num_classes=1))
        return results

    def aug_test(self, imgs, img_metas, **kwargs):
        return self.simple_test(imgs[0], img_metas[0], **kwargs)

    @torch.no_grad()
    def extract_teacher_tokens(self, pose2d_seq):
        pose_tokens, pred_3d = self.encode_pose(pose2d_seq)
        return pose_tokens, pred_3d

    def encode_pose(self, pose2d_seq):
        pose2d_seq = self._stack_tensor(pose2d_seq)
        if pose2d_seq.dim() != 4:
            raise ValueError(
                f'pose2d_seq must have shape (B,T,K,C), got {pose2d_seq.shape}')
        if pose2d_seq.size(-1) != self.in_channels:
            raise ValueError(
                f'Expected pose2d_seq channel={self.in_channels}, got {pose2d_seq.size(-1)}')
        bsz, seq_len, num_keypoints, _ = pose2d_seq.shape
        if num_keypoints != self.num_keypoints:
            raise ValueError(
                f'Expected {self.num_keypoints} keypoints, got {num_keypoints}')
        if seq_len > self.time_embed.size(0):
            raise ValueError(
                f'seq_len={seq_len} exceeds max_seq_len={self.time_embed.size(0)}')

        x = pose2d_seq.permute(0, 3, 1, 2).contiguous()
        feat = self.pose_stem(x).permute(0, 2, 3, 1).contiguous()
        pos = (self.time_embed[:seq_len, None, :] +
               self.joint_embed[None, :num_keypoints, :])
        feat = feat + pos[None, :, :, :]
        feat = feat.view(bsz, seq_len * num_keypoints, self.embed_dims)
        memory = self.encoder(feat.transpose(0, 1)).transpose(0, 1)
        memory = memory.view(bsz, seq_len, num_keypoints, self.embed_dims)

        center_idx = seq_len // 2
        joint_tokens = memory[:, center_idx]
        global_token = memory.mean(dim=(1, 2), keepdim=False).unsqueeze(1)
        pose_tokens = self.token_norm(torch.cat([global_token, joint_tokens], dim=1))
        pred_3d = self.pose_head(pose_tokens[:, 1:])
        return pose_tokens, pred_3d

    @staticmethod
    def _stack_tensor(value):
        if isinstance(value, (list, tuple)):
            if len(value) == 1 and isinstance(value[0], (list, tuple)):
                value = value[0]
            value = torch.stack([v if torch.is_tensor(v) else torch.as_tensor(v)
                                 for v in value], dim=0)
        return value.float()

    @staticmethod
    def _format_gt_keypoints(gt_keypoints, device):
        if isinstance(gt_keypoints, (list, tuple)):
            items = []
            for item in gt_keypoints:
                tensor = item if torch.is_tensor(item) else torch.as_tensor(item)
                if tensor.dim() == 3:
                    tensor = tensor[0]
                items.append(tensor)
            gt = torch.stack(items, dim=0)
        else:
            gt = gt_keypoints
            if gt.dim() == 4:
                gt = gt[:, 0]
            elif gt.dim() == 3 and gt.size(0) == 1:
                gt = gt[None, 0]
        return gt.to(device=device, dtype=torch.float32)

    def _pelvis(self, keypoints):
        left, right = self.pelvis_indices
        return ((keypoints[:, left] + keypoints[:, right]) * 0.5).unsqueeze(1)
