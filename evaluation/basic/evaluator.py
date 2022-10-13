# Copyright (c) Facebook, Inc. and its affiliates.
import logging
from typing import Sequence
from collections import OrderedDict

import numpy as np

import utils

class DatasetEvaluator:
    r"""
    Base class for a dataset evaluator.

    The function :func:`inference_on_dataset` runs the model over
    all samples in the dataset, and have a DatasetEvaluator to process the inputs/outputs.

    This class will accumulate information of the inputs/outputs (by :meth:`process`),
    and produce evaluation results in the end (by :meth:`evaluate`).
    """
    def __init__(self,
                 class_labels: Sequence[str]=[],
                 class_ids: Sequence[int]=[],
                 logger=None):
        # self.logger = logging.getLogger(__name__)     # le
        self.logger = logger
        self.class_labels = class_labels
        self.class_ids = np.array(class_ids)
        self.num_classes = len(class_labels)
        assert len(self.class_labels) == len(self.class_ids), (
            f"all classe labels are {self.class_labels}, length is {len(self.class_labels)}\n"
            f"all class ids are {self.class_ids}, length is {len(self.class_ids)}\n"
            f"their length do not match")
        self.id_to_label = {class_id: class_label for \
            (class_id, class_label) in zip(class_ids, class_labels)}

    def reset(self):
        r"""
        Preparation for a new round of evaluation.
        Should be called before starting a round of evaluation.
        """
        pass

    def process(self, inputs, outputs):
        r"""
        Process the pair of inputs and outputs.
        If they contain batches, the pairs can be consumed one-by-one using `zip`:

        .. code-block:: python

            for input_, output in zip(inputs, outputs):
                # do evaluation on single input/output pair
                ...

        Args:
            inputs (list): the inputs that's used to call the model.
            outputs (list): the return value of `model(inputs)`
        """
        pass

    def evaluate(self):
        r"""
        Evaluate/summarize the performance, after processing all input/output pairs.

        Returns:
            dict:
                A new evaluator class can return a dict of arbitrary format
                as long as the user can process the results.
                In our train_net.py, we expect the following format:

                * key: the name of the task (e.g., bbox)
                * value: a dict of {metric name: score}, e.g.: {"AP50": 80}
        """
        pass


class DatasetEvaluators():
    r"""
    Wrapper class to combine multiple :class:`DatasetEvaluator` instances.

    This class dispatches every evaluation call to
    all of its :class:`DatasetEvaluator`.
    """
    def __init__(self, evaluators: Sequence[DatasetEvaluator]):
        r"""
        Args:
            evaluators (list): the evaluators to combine.
        """
        super().__init__()
        self._evaluators = evaluators
        self.logger = logging.getLogger(__name__)

    def reset(self):
        for evaluator in self._evaluators:
            evaluator.reset()

    def process(self, inputs, outputs):
        for evaluator in self._evaluators:
            evaluator.process(inputs, outputs)

    def evaluate(self):
        results = OrderedDict()
        for evaluator in self._evaluators:
            result = evaluator.evaluate()
            if utils.is_main_process() and result is not None:
                for k, v in result.items():
                    assert (
                        k not in results
                    ), f"Different evaluators produce results with the same key {k}"
                    results[k] = v
        return results
