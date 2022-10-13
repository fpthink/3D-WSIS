from typing import Sequence

import numpy as np

from .evaluator import DatasetEvaluator
import utils

class SemanticEvaluator(DatasetEvaluator):
    """
    Evaluate semantic segmentation metrics.
    """
    def __init__(self,
                 class_labels: Sequence[str],
                 class_ids: Sequence[int],
                 logger=None,
                 ignore: Sequence[int]=[],
                 **kwargs,):
        """
        Args:
            ignore_label: deprecated argument
        """
        super().__init__(
            class_labels=class_labels,
            class_ids=class_ids,
            logger=logger)
        self.ignore = ignore
        self.include = [i for i in self.class_ids if i not in self.ignore]
        self.reset()

    def reset(self):
        max_id = self.class_ids.max() + 1
        self.confusion = np.zeros((max_id + 1, max_id + 1), dtype=np.int64)

    def fill_confusion(self,
                       pred: np.ndarray,
                       gt: np.ndarray):
        np.add.at(self.confusion, (gt.flatten(), pred.flatten()), 1)

    def prase_iou(self):
        # clone to avoid modifying the real deal
        conf = np.zeros_like(self.confusion)
        for i in self.include:
            conf[i, self.include] = self.confusion[i, self.include]

        # get the clean stats
        tp = conf.diagonal()
        fp = conf.sum(axis=1) - tp
        fn = conf.sum(axis=0) - tp

        intersection = tp
        union = tp + fp + fn
        union = np.maximum(union, 1)

        return tp, fp, fn, intersection, union  # returns "iou mean", "iou per class" ALL CLASSES

    def print_result(self):
        # calculate ious
        tp, fp, fn, intersection, union = self.prase_iou()
        ious = (intersection / union) * 100

        # build IoU table
        haeders = ["class", "IoU", "TP/(TP+FP+FN)"]
        results = []
        self.logger.info("Evaluation results for semantic segmentation:")
        
        max_length = max(15, max(map(lambda x: len(x), self.class_labels)))
        for class_id in self.include:
            results.append((
                self.id_to_label[class_id].ljust(max_length, " "),
                ious[class_id],
                f"({intersection[class_id]:>6d}/{union[class_id]:<6d})"))

        acc_table = utils.table(
            results,
            headers=haeders,
            stralign="left")
        for line in acc_table.split("\n"):
            self.logger.info(line)
        self.logger.info(f"mean: {np.nanmean(ious[self.include]):.1f}")
        self.logger.info("")

    def evaluate(self):
        # print semantic segmentation result(IoU)
        self.print_result()

        # return confusion matrix
        return self.confusion



