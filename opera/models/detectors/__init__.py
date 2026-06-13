# Copyright (c) Hikvision Research Institute. All rights reserved.
from .inspose import InsPose
from .petr import PETR
from .petr_distill import PETRDistill
from .skeleton_pose_teacher import SkeletonPoseTeacher
from .soit import SOIT

__all__ = ['InsPose', 'PETR', 'PETRDistill', 'SkeletonPoseTeacher', 'SOIT']
