# Copyright (c) Gorilla-Lab. All rights reserved.
import os.path as osp
from typing import List, Union

import numpy as np

from .basic import DatasetEvaluators, SemanticEvaluator, InstanceEvaluator

CLASS_LABELS = [
    "ceiling", "floor", "wall", "beam", "column", "window", "door",
    "table", "chair", "sofa", "bookcase", "board", "clutter"
]

CLASS_IDS = np.arange(len(CLASS_LABELS))

INS_CLASS_IDS = np.arange(len(CLASS_LABELS)) + 1  # when generating GT txt for S3DIS, the semantic label is mapped to [1-13]

class S3DISSemanticEvaluator(SemanticEvaluator):
    def __init__(self,
                 class_labels: List[str]=CLASS_LABELS,
                 class_ids: Union[np.ndarray, List[int]]=CLASS_IDS,
                 logger=None,
                 **kwargs):
        super().__init__(class_labels=class_labels,
                         class_ids=class_ids,
                         logger=logger,
                         **kwargs)

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
            semantic_pred = output["semantic_pred"].cpu().clone().numpy()
            semantic_gt = output["semantic_gt"].cpu().clone().numpy()
            self.fill_confusion(semantic_pred, semantic_gt)



class S3DISInstanceEvaluator(InstanceEvaluator):
    """
    Evaluate semantic segmentation metrics.
    """
    def __init__(self,
                dataset_root: str,
                class_labels: List[str]=CLASS_LABELS,
                class_ids: List[int]=INS_CLASS_IDS,
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
            gt_file = osp.join(self._dataset_root, scene_name + ".txt")
            gt_ids = np.loadtxt(gt_file)
            self.assign(scene_name, output, gt_ids)


S3DISEvaluator = DatasetEvaluators([S3DISSemanticEvaluator, S3DISInstanceEvaluator])
