# Copyright (c) Gorilla-Lab. All rights reserved.
from collections import defaultdict
from typing import Dict, List, Optional, Sequence

import torch
import torch.distributed as dist
import numpy as np
from tensorboardX import SummaryWriter

from .comm import master_only

class TensorBoardWriter:
    def __init__(self, logdir: str, **kwargs):
        self.logdir = logdir
        self.writer = SummaryWriter(logdir, **kwargs)
        self.buffer = LogBuffer()
        self.step = 0

    def clear(self):
        """clear the buffer"""
        self.buffer.clear()

    @master_only
    def update(self, content: Dict, global_step: Optional[int]=None):
        """"update the buffer according to given directory"""
        self.buffer.update(content)
        # write immediately
        if global_step is not None:
            self.write(global_step)

    @master_only
    def write(self, global_step: Optional[int]=None):
        """write according to buffer"""
        self.buffer.average()
        scalar_type = (int, float, torch.Tensor, np.ndarray)
        # using the integrate step
        if global_step is None:
            self.step += 1
            global_step = self.step
        else:
            self.step = global_step
        for key, value in self.buffer.output.items():
            if isinstance(value, scalar_type):
                self.add_scalar(key, value, global_step)
            elif isinstance(value, dict):
                self.add_scalars(key, value, global_step)
            else:
                raise TypeError(f"The type of scalar must be "
                                f"`int`, `float`, `ndarray` or "
                                f"`Tensor`, ` or `dict` "
                                f"but got {type(value)}")
        self.clear()

    # NOTE: the add_scalar and add_scalars is the wrapper of tensorboard
    #       we support the origin API for using
    @master_only
    def add_text(self,
                  tag,
                  string,
                  global_step,
                  **kwargs):
        r"""the wrapper API of SummaryWriter.add_text"""
        self.writer.add_text(tag, string, global_step, **kwargs)

    # NOTE: the add_scalar and add_scalars is the wrapper of tensorboard
    #       we support the origin API for using
    @master_only
    def add_scalar(self,
                  tag,
                  scalar_value,
                  global_step,
                  **kwargs):
        r"""the wrapper API of SummaryWriter.add_scalar"""
        self.writer.add_scalar(tag, scalar_value, global_step, **kwargs)

    @master_only
    def add_scalars(self,
                    tag,
                    scalar_value,
                    global_step,
                    **kwargs):
        r"""the wrapper API of SummaryWriter.add_scalars"""
        self.writer.add_scalars(tag, scalar_value, global_step, **kwargs)

    def __str__(self) -> str:
        msg = f"writer's logdir: {self.logdir}\n"
        msg += f"current buffer: \n{str(self.buffer)}"

        return msg


class LogBuffer:
    def __init__(self):
        self._val_history = defaultdict(HistoryBuffer)
        self._output = {}

    @property
    def values(self):
        return self._val_history

    @property
    def output(self):
        return self._output

    def clear(self):
        self._val_history.clear()
        self.clear_output()

    def clear_output(self):
        self._output.clear()

    def update(self, content: Dict):
        assert isinstance(content, dict)
        for key, var in content.items():
            scalar_type = (int, float, torch.Tensor, np.ndarray)
            if isinstance(var, Sequence) and len(var) == 2:
                var = list(var) # change tuple
                if isinstance(var[0], scalar_type):
                    var[0] = float(var[0])
                elif isinstance(var[0], Sequence):
                    var[0] = np.sum(np.array(var[0]))
                else:
                    raise TypeError(f"get invalid type of var '{type(var[0])}'")
                self._val_history[key].update(*var)
            elif isinstance(var, scalar_type):
                var = float(var)
                self._val_history[key].update(var)
            else:
                raise TypeError(f"var must be a Sequence with length of 2, "
                                f"int, float, ndarray or Tensor scalar, "
                                f"but got {type(var)}")

    def average(self, n=0):
        r"""Average latest n values or all values."""
        assert n >= 0
        for key in self._val_history:
            self._output[key] = self._val_history[key].average(n)

    def summation(self, n=0):
        r"""Average latest n values or all values."""
        assert n >= 0
        for key in self._val_history:
            self._output[key] = self._val_history[key].summation(n)

    def get(self, name):
        r"""Get the values of name"""
        return self._val_history.get(name, None)

    @property
    def avg(self):
        avg_dict = {}
        for key in self._val_history:
            avg_dict[key] = self._val_history[key].avg
        return avg_dict

    @property
    def sum(self):
        sum_dict = {}
        for key in self._val_history:
            sum_dict[key] = self._val_history[key].sum
        return sum_dict
        
    @property
    def latest(self):
        latest_dict = {}
        for key in self._val_history:
            latest_dict[key] = self._val_history[key].latest
        return latest_dict

    def __str__(self) -> str:
        msg = ""

        for key, value in self._val_history.items():
            msg += f"{key}:\n"
            msg += str(value)

        return msg


class HistoryBuffer:
    r"""
    Track a series of scalar values and provide access to smoothed values over a
    window or the global average of the series.
    """

    def __init__(self) -> None:
        self.clear()

    def clear(self) -> None:
        self._values: List[float] = []
        self._nums: List[float] = []
        self._count: int = 0
        self._global_avg: float = 0
        self._global_sum: float = 0

    def update(self,
               value: float,
               num: Optional[float] = None) -> None:
        r"""
        Add a new scalar value and the number of counter. If the length
        of the buffer exceeds self._max_length, the oldest element will be
        removed from the buffer.
        """
        self._values.append(value)
        if num is None:
            num = 1
        self._nums.append(num)

        self._count += 1
        self._global_sum = sum(map(lambda x: x[0] * x[1], zip(self._values, self._nums)))
        self._global_avg = self._global_sum / sum(self._nums)

    def median(self, window_size: int) -> float:
        r"""
        Return the median of the latest `window_size` values in the buffer.
        """
        return np.median(self._values[-window_size:])

    def summation(self, window_size: int) -> float:
        r"""
        Return the summation of the latest `window_size` values in the buffer.
        """
        _sum = sum(map(lambda x: x[0] * x[1], zip(self._values[-window_size:],
                                                  self._nums[-window_size:])))
        return _sum

    def average(self, window_size: int) -> float:
        r"""
        Return the mean of the latest `window_size` values in the buffer.
        """
        return self.summation(window_size) / sum(self._nums[-window_size:])

    @property
    def latest(self) -> float:
        r"""
        Return the latest scalar value added to the buffer.
        """
        return self._values[-1]

    @property
    def avg(self) -> float:
        r"""
        Return the mean of all the elements in the buffer. Note that this
        includes those getting removed due to limited buffer storage.
        """
        return self._global_avg

    @property
    def sum(self) -> float:
        r"""
        Return the summation of all the elements in the buffer. Note that this
        includes those getting removed due to limited buffer storage.
        """
        return self._global_sum

    @property
    def values(self) -> List[float]:
        r"""
        Returns:
            number: content of the current buffer.
        """
        return self._values

    @property
    def nums(self) -> List[float]:
        r"""
        Returns:
            number: content of the current buffer.
        """
        return self._nums

    def __str__(self) -> str:
        msg = "\n".join([f"    values: {self.values}",
                         f"    nums:   {self.nums}",
                         f"    count:  {self._count}",
                         f"    avg:    {self.avg}",
                         f"    sum:    {self.sum}\n"])

        return msg

