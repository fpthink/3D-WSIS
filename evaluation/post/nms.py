# import gorilla


# Copyright (c) Gorilla-Lab. All rights reserved.
import numpy as np


def non_max_suppression(ious: np.ndarray,
                        scores: np.ndarray,
                        threshold: float
) -> np.ndarray:
    r"""non max suppression for nparray (have given the ious map)

    Args:
        ious (np.ndarray): (N, N) ious map
        scores (np.ndarray): (N) iou
        threshold (float): iou threshold

    Returns:
        np.ndarray: keep dix
    """
    ixs = scores.argsort()[::-1]
    keep = []
    while len(ixs) > 0:
        i = ixs[0]
        keep.append(i)
        iou = ious[i, ixs[1:]]
        remove_ixs = np.where(iou > threshold)[0] + 1
        ixs = np.delete(ixs, remove_ixs)
        ixs = np.delete(ixs, 0)
    return np.array(keep, dtype=np.int32)


