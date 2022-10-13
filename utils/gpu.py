# Copyright (c) Gorilla-Lab. All rights reserved.
import os
import time
import warnings

import pynvml
import torch
from gpustat import GPUStatCollection

from .misc import convert_list_str

# init
pynvml.nvmlInit()
NUM_GPUS = pynvml.nvmlDeviceGetCount()


def get_free_gpu(mode="memory", memory_need=11000) -> list:
    r"""Get free gpu according to mode (process-free or memory-free).

    Args:
        mode (str, optional): memory-free or process-free. Defaults to "memory".
        memory_need (int): The memory you need, used if mode=='memory'. Defaults to 10000.

    Returns:
        list: free gpu ids
    """
    assert mode in ["memory", "process"], f"mode must be 'memory' or 'process', but got {mode}"
    if mode == "memory":
        assert memory_need is not None, "'memory_need' if None, 'memory' mode must give the free memory you want to apply for"
        memory_need = int(memory_need)
        assert memory_need > 0, "'memory_need' you want must be positive"
    gpu_stats = GPUStatCollection.new_query()
    gpu_free_id_list = []

    for idx, gpu_stat in enumerate(gpu_stats):
        if gpu_check_condition(gpu_stat, mode, memory_need):
            gpu_free_id_list.append(idx)
            print(f"gpu[{idx}]: {gpu_stat.memory_free}MB")
    return gpu_free_id_list


def gpu_check_condition(gpu_stat, mode, memory_need) -> bool:
    r"""Check gpu is free or not.

    Args:
        gpu_stat (obj:`core`): gpustat to check
        mode (str): memory-free or process-free.
        memory_need (int): The memory you need, used if mode=='memory'

    Returns:
        bool: gpu is free or not
    """
    if mode == "memory":
        return gpu_stat.memory_free > memory_need
    elif mode == "process":
        for process in gpu_stat.processes:
            if process["command"] == "python": return False
        return True
    else: return False


def supervise_gpu(num_gpu=1, mode="memory", memory_need=11000) -> list:
    r"""Supervise gpu for you need

    Args:
        num_gpu (int, optional): The number of gpu you need. Defaults to 1.
        mode (str, optional): memory-free or process-free. Defaults to "memory".
        memory_need (int, optional): The memory you need, used if mode=='memory'. Defaults to 10000.

    Returns:
        list: free gpu id list
    """
    gpu_free_id_list = []
    if num_gpu> NUM_GPUS:
        warnings.warn(f"num_gpu: {num_gpu} > all_num_gpu: {NUM_GPUS} we surplus this num")
        num_gpu %= NUM_GPUS
    while len(gpu_free_id_list) < num_gpu:
        time.sleep(2)
        gpu_free_id_list = get_free_gpu(mode, memory_need)
    used_gpu_id_list = gpu_free_id_list[:num_gpu]
    return used_gpu_id_list


def set_cuda_visible_devices(gpu_ids=None, num_gpu=1, mode="memory", memory_need=11000):
    r"""Set cuda visible devices automatically

    Args:
        gpu_ids (str | list, optional): specified gpus. Defaults to None.
        mode (str, optional): memory-free or process-free. Defaults to "memory".
        memory_need (int, optional): The memory you need, used if mode=='memory'.
        num_gpu (int, optional):
            the num of gpus you want to use. Defaults to 1.
            (useless if `gpu_ids` is not None)
    """
    if gpu_ids is not None: # specified gpus
        if not isinstance(gpu_ids, list):
            gpu_ids = [gpu_ids]
    elif gpu_ids is None and num_gpu >= 1: # not specify gpus
        gpu_ids = supervise_gpu(num_gpu, mode, memory_need)
    else:
        raise ValueError(f"`num_gpu` is invalid: {num_gpu}, it must '>=1'")
    
    # just for single machine multi gpu setting, the ngpus_per_node is
    # all gpus in this machine
    gpu_ids = ",".join(convert_list_str(gpu_ids))
    print(f"set CUDA_VISIBLE_DEVICES as {gpu_ids}")
    # return  gpu_ids
    os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu_ids
