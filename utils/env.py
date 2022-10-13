# Copyright (c) Facebook, Inc. and its affiliates.
import os
import os.path as osp
import logging
import sys
import re
import random
import subprocess
import importlib
from collections import defaultdict
from typing import Optional

import torch
import torchvision
import numpy as np
from tabulate import tabulate

from .comm import get_dist_info

__all__ = ["collect_env_info"]


def collect_torch_env():
    try:
        import torch.__config__

        return torch.__config__.show()
    except ImportError:
        # compatible with older versions of pytorch
        from torch.utils.collect_env import get_pretty_env_info

        return get_pretty_env_info()


def get_env_module():
    var_name = "GORILLA_ENV_MODULE"
    return var_name, os.environ.get(var_name, "<not set>")


def detect_compute_compatibility(CUDA_HOME, so_file):
    try:
        cuobjdump = osp.join(CUDA_HOME, "bin", "cuobjdump")
        if osp.isfile(cuobjdump):
            output = subprocess.check_output(f"'{cuobjdump}' --list-elf 'so_file{so_file}'",
                                             shell=True)
            output = output.decode("utf-8").strip().split("\n")
            arch = []
            for line in output:
                line = re.findall(r"\.sm_([0-9]*)\.", line)[0]
                arch.append(".".join(line))
            arch = sorted(set(arch))
            return ", ".join(arch)
        else:
            return so_file + "; cannot find cuobjdump"
    except Exception:
        # unhandled failure
        return so_file


def collect_env_info() -> str:
    has_gpu = torch.cuda.is_available()  # true for both CUDA & ROCM
    torch_version = torch.__version__

    # NOTE: the use of CUDA_HOME and ROCM_HOME requires the CUDA/ROCM build deps, though in
    # theory gorilla should be made runnable with only the corresponding runtimes
    from torch.utils.cpp_extension import CUDA_HOME

    has_rocm = False
    if tuple(map(int, torch_version.split(".")[:2])) >= (1, 5):
        from torch.utils.cpp_extension import ROCM_HOME

        if (getattr(torch.version, "hip", None) is not None) and (ROCM_HOME
                                                                  is not None):
            has_rocm = True
    has_cuda = has_gpu and (not has_rocm)

    data = []
    data.append(("sys.platform", sys.platform))
    data.append(("Python", sys.version.replace("\n", "")))
    data.append(("numpy", np.__version__))

    try:
        import gorilla  # noqa

        data.append(
            ("gorilla",
             gorilla.__version__ + " @" + osp.dirname(gorilla.__file__)))
    except ImportError:
        data.append(("gorilla", "failed to import"))

    data.append(get_env_module())
    data.append(
        ("PyTorch", torch_version + " @" + osp.dirname(torch.__file__)))
    data.append(("PyTorch debug build", torch.version.debug))

    data.append(("GPU available", has_gpu))
    if has_gpu:
        devices = defaultdict(list)
        for k in range(torch.cuda.device_count()):
            cap = ".".join(
                (str(x) for x in torch.cuda.get_device_capability(k)))
            name = torch.cuda.get_device_name(k) + f" (arch={cap})"
            devices[name].append(str(k))
        for name, devids in devices.items():
            data.append(("GPU " + ",".join(devids), name))

        if has_rocm:
            msg = " - invalid!" if not osp.isdir(ROCM_HOME) else ""
            data.append(("ROCM_HOME", str(ROCM_HOME) + msg))
        else:
            msg = " - invalid!" if not osp.isdir(CUDA_HOME) else ""
            data.append(("CUDA_HOME", str(CUDA_HOME) + msg))

            cuda_arch_list = os.environ.get("TORCH_CUDA_ARCH_LIST", None)
            if cuda_arch_list:
                data.append(("TORCH_CUDA_ARCH_LIST", cuda_arch_list))

    try:
        data.append((
            "torchvision",
            str(torchvision.__version__) + " @" +
            osp.dirname(torchvision.__file__),
        ))
        if has_cuda:
            try:
                torchvision_C = importlib.util.find_spec(
                    "torchvision._C").origin
                msg = detect_compute_compatibility(CUDA_HOME, torchvision_C)
                data.append(("torchvision arch flags", msg))
            except ImportError:
                data.append(("torchvision._C", "Not found"))
    except AttributeError:
        data.append(("torchvision", "unknown"))

    try:
        import cv2

        data.append(("OpenCV", cv2.__version__))
    except ImportError:
        data.append(("cv2", "Not found"))
    env_str = tabulate(data) + "\n"
    env_str += collect_torch_env()
    return env_str


def set_random_seed(seed: int,
                    deterministic: bool=True,
                    use_rank_shift: bool=False,
                    **kwargs) -> None:
    r"""Set random seed.
    Args:
        seed (int): Seed to be used.
        deterministic (bool): Whether to set the deterministic option for
            CUDNN backend, i.e., set `torch.backends.cudnn.deterministic`
            to True and `torch.backends.cudnn.benchmark` to False.
            Default: True.
        rank_shift (bool): Whether to add rank number to the random seed to
            have different random seed in different threads. Default: False.
    """
    logger = logging.getLogger(__name__)
    logger.info(f"set random seed: {seed}")
        
    if use_rank_shift:
        rank, _ = get_dist_info()
        seed += rank
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
