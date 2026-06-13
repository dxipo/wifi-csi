# Copyright (c) Hikvision Research Institute. All rights reserved.
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmdet.models.detectors.single_stage import SingleStageDetector

from ..builder import DETECTORS
from .petr import PETR


@DETECTORS.register_module()
class PETRDistill(PETR):
    """PETR student with pose-token distillation from a 2D skeleton teacher."""

    def __init__(self,
                 *args,
                 distill_stage='stage2',
                 distill_num_tokens=18,
                 teacher_embed_dims=256,
                 distill_loss_weight=0.1,
                 relation_loss_weight=0.03,
                 distill_num_heads=8,
                 **kwargs):
        super(PETRDistill, self).__init__(*args, **kwargs)
        self.distill_stage = distill_stage
        self.distill_num_tokens = distill_num_tokens
        self.teacher_embed_dims = teacher_embed_dims
        self.distill_loss_weight = distill_loss_weight
        self.relation_loss_weight = relation_loss_weight

        self.distill_queries = nn.Parameter(
            torch.randn(distill_num_tokens, 256) * 0.02)
        self.distill_attn = nn.MultiheadAttention(
            embed_dim=256, num_heads=distill_num_heads, dropout=0.1)
        self.distill_norm = nn.LayerNorm(256)
        self.distill_ffn = nn.Sequential(
            nn.Linear(256, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, 256))
        self.student_projector = nn.Linear(256, teacher_embed_dims)
        self.teacher_norm = nn.LayerNorm(teacher_embed_dims)
        self.student_norm = nn.LayerNorm(teacher_embed_dims)

    def init_weights(self):
        super(PETRDistill, self).init_weights()
        nn.init.normal_(self.distill_queries, std=0.02)
        nn.init.xavier_uniform_(self.student_projector.weight)
        nn.init.constant_(self.student_projector.bias, 0)

    def forward_train(self,
                      img,
                      img_metas,
                      gt_bboxes,
                      gt_labels,
                      gt_keypoints,
                      gt_areas,
                      teacher_tokens=None,
                      gt_bboxes_ignore=None):
        super(SingleStageDetector, self).forward_train(img, img_metas)
        csi_tokens = self.extract_csi_tokens(img)

        losses = {}
        if self.distill_stage == 'stage1':
            losses.update(self.forward_distill(csi_tokens, teacher_tokens))
            return losses

        losses.update(
            self.bbox_head.forward_train(csi_tokens, img_metas, gt_bboxes,
                                         gt_labels, gt_keypoints, gt_areas,
                                         gt_bboxes_ignore))
        if teacher_tokens is not None and self.distill_loss_weight > 0:
            losses.update(self.forward_distill(csi_tokens, teacher_tokens))
        return losses

    def forward_distill(self, csi_tokens, teacher_tokens):
        if teacher_tokens is None:
            raise ValueError(
                'PETRDistill requires teacher_tokens when distill_stage=stage1 '
                'or when distillation losses are enabled.')
        teacher_tokens = self._stack_teacher_tokens(teacher_tokens, csi_tokens.device)
        student_tokens = self.extract_student_pose_tokens(csi_tokens)
        if student_tokens.shape[:2] != teacher_tokens.shape[:2]:
            raise ValueError(
                f'Student tokens {student_tokens.shape} and teacher tokens '
                f'{teacher_tokens.shape} are not aligned.')

        student = self.student_norm(self.student_projector(student_tokens))
        teacher = self.teacher_norm(teacher_tokens.detach())
        loss_token = F.smooth_l1_loss(student, teacher, reduction='mean')
        loss_rel = F.mse_loss(
            self.token_relation(student),
            self.token_relation(teacher),
            reduction='mean')
        cosine = F.cosine_similarity(
            student.flatten(1), teacher.flatten(1), dim=1).mean()
        return dict(
            loss_token_distill=loss_token * self.distill_loss_weight,
            loss_relation_distill=loss_rel * self.relation_loss_weight,
            distill_token_cos=cosine.detach())

    def extract_student_pose_tokens(self, csi_tokens):
        batch_size = csi_tokens.size(0)
        query = self.distill_queries[:, None, :].expand(-1, batch_size, -1)
        memory = csi_tokens.transpose(0, 1).contiguous()
        attended, _ = self.distill_attn(query, memory, memory)
        attended = attended.transpose(0, 1).contiguous()
        attended = self.distill_norm(attended + self.distill_ffn(attended))
        return attended

    @staticmethod
    def token_relation(tokens, eps=1e-6):
        tokens = F.normalize(tokens, dim=-1, eps=eps)
        return torch.matmul(tokens, tokens.transpose(1, 2))

    @staticmethod
    def _stack_teacher_tokens(teacher_tokens, device):
        if isinstance(teacher_tokens, (list, tuple)):
            if len(teacher_tokens) == 1 and isinstance(teacher_tokens[0], (list, tuple)):
                teacher_tokens = teacher_tokens[0]
            teacher_tokens = torch.stack(
                [x if torch.is_tensor(x) else torch.as_tensor(x)
                 for x in teacher_tokens],
                dim=0)
        return teacher_tokens.to(device=device, dtype=torch.float32)
