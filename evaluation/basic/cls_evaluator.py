import itertools
from typing import Sequence

import torch
import numpy as np

from .evaluator import DatasetEvaluator
from .metric_classification import accuracy, accuracy_for_each_class
import utils

# modify from https://github.com/Megvii-BaseDetection/cvpods/blob/master/cvpods/evaluation/classification_evaluation.py
class ClassificationEvaluator(DatasetEvaluator):
    """
    Evaluate semantic segmentation metrics.
    """
    def __init__(self,
                 class_labels: Sequence[str],
                 class_ids: Sequence[int],
                 top_k: Sequence[int],
                 **kwargs,):
        """
        Args:
            ignore_label: deprecated argument
        """
        super().__init__(
            class_labels=class_labels,
            class_ids=class_ids,)
        self._top_k = top_k
        self.reset()

    def reset(self):
        self._predictions = []
        self._labels = []

    def match(self,
              prediction: np.ndarray,
              label: np.ndarray):
        self._predictions.append(prediction)
        self._labels.append(label)
        

    def evaluate(self, show_per_class: bool=True):
        self._predictions = torch.cat(self._predictions).view(-1, self.num_classes) # [N, num_classes]
        self._labels = torch.cat(self._labels).view(-1) # [N]

        # calcualate instance accuracy
        acc = accuracy(self._predictions, self._labels, self._top_k)
        
        acc_dict = {}
        for i, k in enumerate(self._top_k):
            acc_dict[f"Top_{k} Acc"] = acc[i]
            
        acc_table = utils.create_small_table(acc_dict)
        self.logger.info("Evaluation results for classification:")
        for line in acc_table.split("\n"):
            self.logger.info(line)
        self.logger.info("")

        if show_per_class:
            totals, corrects = accuracy_for_each_class(self._predictions, self._labels.view(-1, 1), self.num_classes) # [num_classes]
            corrects_per_class = (corrects * 100)/ totals # [num_classes]

            self.logger.info("Top_1 Acc of each class")
            # tabulate it
            N_COLS = min(8, len(self.class_labels) * 2)
            acc_per_class = [(self.class_labels[i], float(corrects_per_class[i])) for i in range(len(self.class_labels))]
            acc_flatten = utils.concat_list(acc_per_class)
            results_2d = itertools.zip_longest(*[acc_flatten[i::N_COLS] for i in range(N_COLS)])
            acc_table = utils.table(
                results_2d,
                headers=["class", "Acc"] * (N_COLS // 2),
            )
            for line in acc_table.split("\n"):
                self.logger.info(line)
            self.logger.info(f"mean: {corrects_per_class.mean():.4f}")
            self.logger.info("")

        return acc
