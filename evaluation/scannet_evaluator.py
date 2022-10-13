# Copyright (c) Gorilla-Lab. All rights reserved.
import os
import os.path as osp
from typing import List, Union

import numpy as np

from .basic import DatasetEvaluators, SemanticEvaluator, InstanceEvaluator

CLASS_LABELS = [
    "wall", "floor", "cabinet", "bed", "chair", "sofa", "table", "door",
    "window", "bookshelf", "picture", "counter", "desk", "curtain",
    "refrigerator", "shower curtain", "toilet", "sink", "bathtub",
    "otherfurniture"
]
CLASS_IDS = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 14, 16, 24, 28, 33, 34, 36, 39]


class ScanNetSemanticEvaluator(SemanticEvaluator):
    def __init__(self,
                 dataset_root,
                 class_labels: List[str]=CLASS_LABELS,
                 class_ids: Union[np.ndarray, List[int]]=CLASS_IDS,
                 logger=None,
                 **kwargs):
        super().__init__(class_labels=class_labels,
                         class_ids=class_ids,
                         logger=logger,
                         **kwargs)
        self.dataset_root = dataset_root

    def process(self, inputs, outputs):
        """
        Args:
            inputs: the inputs to a model.
            outputs: the outputs of a model. It is either list of semantic segmentation predictions
                or list of dicts with key "sem_seg" that contains semantic
                segmentation prediction in the same format.
        """
        if not isinstance(inputs, List):
            inputs = [inputs]
        if not isinstance(outputs, List):
            outputs = [outputs]
        for input, output in zip(inputs, outputs):
            scene_name = input["scene_name"]
            semantic_gt = self.read_gt(self.dataset_root, scene_name)
            semantic_pred = output["semantic_pred"].cpu().clone().numpy()
            semantic_pred = self.class_ids[semantic_pred]
            self.fill_confusion(semantic_pred, semantic_gt)

    @staticmethod
    def read_gt(origin_root, scene_name):
        label = np.loadtxt(os.path.join(origin_root, scene_name + "_sem.txt"))
        label = label.astype(np.int32)
        return label


# ---------- Label info ---------- #
FOREGROUND_CLASS_LABELS = [
    "cabinet", "bed", "chair", "sofa", "table", "door", "window", "bookshelf",
    "picture", "counter", "desk", "curtain", "refrigerator", "shower curtain",
    "toilet", "sink", "bathtub", "otherfurniture"
]
FOREGROUND_CLASS_IDS = np.array(
    [3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 14, 16, 24, 28, 33, 34, 36, 39])


class ScanNetInstanceEvaluator(InstanceEvaluator):
    """
    Evaluate semantic segmentation metrics.
    """
    def __init__(self,
                dataset_root: str,
                class_labels: List[str]=FOREGROUND_CLASS_LABELS,
                class_ids: List[int]=FOREGROUND_CLASS_IDS,
                logger=None,
                **kwargs):
        """
        Args:
            ignore_label: deprecated argument
        """
        super().__init__(class_labels=class_labels,
                         class_ids=class_ids,
                         logger=logger,
                         **kwargs)
        self._dataset_root = dataset_root

    def process(self, inputs, outputs):
        """
        Args:
            inputs: the inputs to a model.
                It is a list of dicts.
            outputs: the outputs of a model. It is either list of semantic segmentation predictions
                or list of dicts with key "sem_seg" that contains semantic
                segmentation prediction in the same format.
        """
        if not isinstance(inputs, List):
            inputs = [inputs]
        if not isinstance(outputs, List):
            outputs = [outputs]
        for input, output in zip(inputs, outputs):
            scene_name = input["scene_name"]
            gt_file = osp.join(self._dataset_root, scene_name + "_ins.txt")
            gt_ids = np.loadtxt(gt_file)
            self.assign(scene_name, output, gt_ids)     # InstanceEvaluator.assign

ScanNetEvaluator = DatasetEvaluators([ScanNetSemanticEvaluator, ScanNetInstanceEvaluator])
