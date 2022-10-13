# Copyright (c) Gorilla-Lab. All rights reserved.
import json
from typing import Dict, List

import numpy as np

class VertInstance(object):
    instance_id = 0
    instance_count = 0
    med_dist = -1
    dist_conf = 0.0

    def __init__(self,
                 instance_ids: np.ndarray,
                 instance_id: int):
        r"""
        construct an instance mask

        Args:
            instance_ids (np.ndarray, [N]): instance ids of each vertex
            instance_id (int): id of instance
                the id of instance contain two part: sem_id and obj_id
                instance_id = sem_id * 1000 + obj_id
                sem_id = instance_id // 1000
                obj_id = instance_id %  1000
        """
        if (instance_id == -1):
            return
        self.instance_id = int(instance_id)
        self.gt_mask = (instance_ids == instance_id)
        self.instance_count = int((instance_ids == instance_id).sum())

    @property
    def label_id(self):
        r"""get the semantic label id"""
        return int(self.instance_id // 1000)

    @property
    def dict(self):
        r"""return a dict represent instance"""
        dict = {}
        dict["gt_mask"] = self.gt_mask
        dict["instance_id"] = self.instance_id
        dict["label_id"] = self.label_id
        dict["instance_count"] = self.instance_count
        dict["med_dist"] = self.med_dist
        dict["dist_conf"] = self.dist_conf
        return dict

    def __str__(self):
        return "(" + str(self.instance_id) + ")"

    @staticmethod
    def get_instances(instance_ids: np.ndarray,
                      class_ids: np.ndarray,
                      class_labels: List[str],
                      id2label: Dict) -> Dict:
        r"""
        get the dict represents all instance for point cloud

        Args:
            instance_ids (np.ndarray, N): instance ids of each vertex
            class_ids (np.ndarray, num_classes): class ids
            class_labels (List[str]): class name list
            id2label (Dict): map class id to class name

        Returns:
            Dict: instance dicts for each class
        """
        assert len(class_labels) == len(class_ids)
        instances = {}
        for label in class_labels:
            instances[label] = []
        # traverse all instances
        inst_ids = np.unique(instance_ids)
        for id in inst_ids:
            # skip 0 and negative instance id (background points)
            if id <= 0:
                continue
            # get instance
            inst = VertInstance(instance_ids, id)
            # record in correspond class dict
            if inst.label_id in class_ids:
                instances[id2label[inst.label_id]].append(inst.dict)
        return instances
