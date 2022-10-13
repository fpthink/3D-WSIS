# Copyright (c) Gorilla-Lab. All rights reserved.
import functools
import os
import os.path as osp
import logging
from typing import Dict, Iterable, Optional
from collections import OrderedDict
import termcolor

import torch.distributed as dist
from tabulate import tabulate

from .timer import timestamp
from .misc import convert_list_str


def collect_logger(root: str="log",
                   prefix: Optional[str]=None,
                   suffix: Optional[str]=None,
                   log_name: Optional[str]=None,
                   log_file: Optional[str]=None,
                   show_name: bool=True,
                   **kwargs):
    r"""Author: liang.zhihao
    A easy combination of get_log_dir and get_logger, use the timestamp
    as log file's name

    Args:
        root (str, optional): the root directory of logger. Defaults to "log".
        prefix (str, optional): the extra prefix. Defaults to None.
        suffix (str, optional): the extra suffix. Defaults to None.
        log_name (str, optional):
            name of log file, if given None, the name of log_file is time_stamp.
            Defaults to None.
        log_file (str, optional):
            the path of log_file, the highest priority, if given, directly init the logger.
            Defaults to None.
        show_name: (bool, optional):
            show the name of logger prefix.
            Defaults to True.

    Returns:
        [str, logging.Logger]: the log dir and the logger
    """
    # get the timestamp
    time_stamp = timestamp()
    # get the log_file
    if log_file is None:
        log_dir = get_log_dir(root,
                            prefix,
                            suffix,
                            **kwargs)
        
        if log_name is None:
            log_file = osp.join(log_dir, f"{time_stamp}.log")
        else:
            log_file = osp.join(log_dir, f"{log_name}.log")
    else:
        log_dir = osp.dirname(log_file)
    
    if not log_dir.startswith("."):
        log_dir = f"./{log_dir}"

    if not log_file.startswith("."):
        log_file = f"./{log_file}"

    logger = get_logger(log_file, timestamp=time_stamp, show_name=show_name)

    return log_dir, logger


def get_log_dir(root: str="log",
                prefix: str=None,
                suffix: str=None,
                **kwargs) -> str:
    r"""Author: liang.zhihao
    Get log dir according to the given params key-value pair

    Args:
        root (str, optional): the root directory of logger. Defaults to "log".
        prefix (str, optional): the extra prefix. Defaults to None.
        suffix (str, optional): the extra suffix. Defaults to None.

    Example:
        >>> import gorilla
        >>> # dynamic concatenate
        >>> gorilla.get_log_dir(lr=0.001, bs=4)
        "log/lr_0.001_bs_4"
        >>> gorilla.get_log_dir(lr=0.001, bs=4, optim="Adam")
        "log/lr_0.001_bs_4_Adam"
        >>> gorilla.get_log_dir(lr=0.001, bs=4, optim="Adam", suffix="test") # add the suffix
        "log/lr_0.001_bs_4_Adam_test"
        >>> gorilla.get_log_dir(lr=0.001, bs=4, optim="Adam", prefix="new") # add the prefix
        "log/new_lr_0.001_bs_4_Adam_test"
        >>> gorilla.get_log_dir(lr=0.001, bs=4, optim="Adam", prefix="new", suffix="test") # add both prefix and suffix 
        "log/lr_0.001_bs_4_Adam_test"

    Returns:
        str: the directory path of log
    """
    # concatenate the given parameters
    args_dict = OrderedDict(kwargs)
    params = []
    for key, value in args_dict.items():
        params.extend([key, value])
    
    # deal with prefix
    if prefix is not None:
        params.insert(0, prefix)
    
    # deal with suffix
    if suffix is not None:
        params.append(suffix)
    
    # convert all parameters as str
    params = convert_list_str(params)

    # get log dir and make
    sub_log_dir = "_".join(params)
    log_dir = osp.join(root, sub_log_dir)
    os.makedirs(log_dir, exist_ok=True)

    return log_dir


@functools.lru_cache()  # so that calling setup_logger multiple times won't add many handlers
def get_logger(log_file: str=None,
               name: str="project",
               log_level: int=logging.INFO,
               timestamp: Optional[str]=None,
               abbrev_name: Optional[str]=None,
               show_name: bool=True,
               color: bool=True) -> logging.Logger:
    r"""Initialize and get a logger by name.
        If the logger has not been initialized, this method will initialize the
        logger by adding one or two handlers, otherwise the initialized logger will
        be directly returned. During initialization, a StreamHandler will always be
        added. If `log_file` is specified and the process rank is 0, a FileHandler
        will also be added.

    Args:
        name (str): Logger name. Default to 'gorilla'
        log_file (str | None): The log filename. If specified, a FileHandler
            will be added to the logger.
        log_level (int): The logger level. Note that only the process of
            rank 0 is affected, and other processes will set the level to
            "Error" thus be silent most of the time.
        timestamp (str, optional): The timestamp of logger.
            Defaults to None
        abbrev_name (str): an abbreviation of the module, to avoid long names in logs.
            Set to "" to not log the root module in logs.
            By default, will be "gorilla" and leave other
            modules unchanged.
    Returns:
        logging.Logger: The expected logger.
    """
    logger = logging.getLogger(name)
    logger.timestamp = timestamp
    logger.parent = None

    try:
        # piror rich handler
        from rich.logging import RichHandler
        handlers = [RichHandler(rich_tracebacks=True, show_level=False, show_time=False)]
        # fake colored refer to termcolor's colored API
        colored = lambda text, color, on_color=None, attrs=None: text # NOTE: fix the conflict between rich and colored
    except:
        stream_handler = logging.StreamHandler()
        handlers = [stream_handler]
        colored = termcolor.colored

    if dist.is_available() and dist.is_initialized():
        rank = dist.get_rank()
    else:
        rank = 0

    # only rank 0 will add a FileHandler
    if rank == 0 and log_file is not None:
        log_dir = os.path.join(".", osp.dirname(log_file))
        if not osp.isdir(log_dir):
            os.makedirs(log_dir)
        file_handler = logging.FileHandler(log_file, "w")
        handlers.append(file_handler)
        
    # detectron2 style
    if abbrev_name is None:
        abbrev_name = name
    if color:
        if show_name:
            prefix = colored("[%(asctime)s %(name)s]", "green") + " %(levelname)s: %(message)s"
        else:
            prefix = colored("[%(asctime)s]", "green") + " %(levelname)s: %(message)s"
        formatter = _ColorfulFormatter(
            prefix,
            datefmt="%m/%d %H:%M:%S",
            root_name=name,
            abbrev_name=abbrev_name,
            colored=colored,
        )
    else:
        if show_name:
            prefix = "[%(asctime)s %(name)s] %(levelname)s: %(message)s"
        else:
            prefix = "[%(asctime)s] %(levelname)s: %(message)s"
        formatter = logging.Formatter(prefix, datefmt="%m/%d %H:%M:%S")

    # # mmcv style
    # formatter = logging.Formatter(
    #     "%(asctime)s - %(name)s - %(levelname)s - %(message)s")

    for handler in handlers:
        handler.setFormatter(formatter)
        handler.setLevel(log_level)
        logger.addHandler(handler)

    if rank == 0:
        logger.setLevel(log_level)
    else:
        logger.setLevel(logging.ERROR)

    return logger


def derive_logger(name: str,
                  parent: str="project") -> logging.Logger:
    r"""
    drive a logger, whose parent is decided by the `parent` name
    in order to intialize logger by `__name__` more convenient
    (process the message by parent's handlers)

    Args:
        name (str): the name of initialized logger
        parent (str, optional): the name of parent logger. Defaults to "gorilla".

    Raises:
        KeyError: parent logger does not exist

    Returns:
        logging.Logger: initilalized logger
    """
    if parent not in logging.Logger.manager.loggerDict.keys():
        raise KeyError(f"the parent logger-{parent} are not initialized")
    logger = logging.getLogger(name)
    logger.parent = logging.getLogger(parent)

    return logger


def print_log(msg: str,
              logger: Optional[logging.Logger]=None,
              level: int=logging.INFO):
    r"""Print a log message.

    Args:
        msg (str): The message to be logged.
        logger (logging.Logger | str | None): The logger to be used.
            Some special loggers are:
            - "silent": no message will be printed.
            - other str: the logger obtained with `get_logger(log_file)`.
            - None: The `print()` method will be used to print log messages.
        level (int): Logging level. Only available when `logger` is a Logger
            object or "root".
    """
    if logger is None:
        print(msg)
    elif isinstance(logger, logging.Logger):
        logger.log(level, msg)
    elif logger == "silent":
        pass
    elif isinstance(logger, str):
        _logger = get_logger(name=logger)
        _logger.log(level, msg)
    else:
        raise TypeError(
            f"logger should be either a logging.Logger object, str, "
            f"'silent' or None, but got {type(logger)}")

# modify from detectron2 https://github.com/facebookresearch/detectron2/blob/master/detectron2/utils/logger.py
class _ColorfulFormatter(logging.Formatter):
    def __init__(self, *args, **kwargs):
        self._root_name = kwargs.pop("root_name") + "."
        self._abbrev_name = kwargs.pop("abbrev_name", "")
        self._colored = kwargs.pop("colored", termcolor.colored)
        if len(self._abbrev_name):
            self._abbrev_name = self._abbrev_name + "."
        super(_ColorfulFormatter, self).__init__(*args, **kwargs)

    def formatMessage(self, record):
        record.name = record.name.replace(self._root_name, self._abbrev_name)
        log = super(_ColorfulFormatter, self).formatMessage(record)
        if record.levelno == logging.WARNING:
            prefix = self._colored("WARNING", "red", attrs=["blink"])
        elif record.levelno == logging.ERROR or record.levelno == logging.CRITICAL:
            prefix = self._colored("ERROR", "red", attrs=["blink", "underline"])
        else:
            return log
        return prefix + " " + log


# modify from https://github.com/Megvii-BaseDetection/cvpods/blob/master/cvpods/utils/dump/logger.py
def create_small_table(small_dict: Dict,
                       tablefmt: str="psql",
                       **kwargs):
    """
    Create a small table using the keys of small_dict as headers. This is only
    suitable for small dictionaries.
    Args:
        small_dict (dict): a result dictionary of only a few items.
    Returns:
        str: the table as a string.
    """
    keys, values = tuple(zip(*small_dict.items()))
    table = tabulate(
        [values],
        headers=keys,
        tablefmt=tablefmt,
        floatfmt=".3f",
        stralign="center",
        numalign="center",
        **kwargs
    )
    return table

def table(data: Iterable,
          headers: Iterable,
          tablefmt: str="psql",
          stralign: str="center",
          numalign="center",
          floatfmt=".3f",
          **kwargs):
    r"""
    a lite wrapper of tabulate, given the default tablefmt/floatfmt/stralign/numalign
    """
    if not isinstance(data, Iterable):
        data = [data]
    if not isinstance(headers, Iterable):
        headers = [headers]

    table = tabulate(
        data,
        headers=headers,
        tablefmt=tablefmt,
        floatfmt=floatfmt,
        stralign=stralign,
        numalign=numalign,
        **kwargs
    )
    return table
