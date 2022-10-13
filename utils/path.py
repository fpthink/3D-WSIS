# Copyright (c) Open-MMLab. All rights reserved.

import os
import os.path as osp
from pathlib import Path

def is_filepath(x) -> bool:
    r"""Path check function.

    Args:
        x (str | obj:`Path`): path address

    Returns:
        bool: is a path or not
    """
    return isinstance(x, (str, Path))


def check_file(filepath: [str, Path], msg_tmpl="file `{}` not exist or is a directory"):
    r"""Check path exists and file or not.

    Args:
        filepath (str | obj:`Path`): path address
        msg_tmpl (str, optional): error message pattern. Defaults to "file `{}` not exist or is a directory".
    """
    if isinstance(filepath, str):
        if not osp.isfile(filepath):
            raise FileNotFoundError(msg_tmpl.format(filepath))
    elif isinstance(filepath, Path):
        if not filepath.is_file():
            raise FileNotFoundError(msg_tmpl.format(str(filepath)))


def check_dir(dir_path, msg_tmpl="dir `{}` not exist or is a directory"):
    r"""Check path exists and file or not.

    Args:
        dir_path (str | obj:`Path`): path address
        msg_tmpl (str, optional): error message pattern. Defaults to "dir `{}` not exist or is a directory".
    """
    if isinstance(dir_path, str):
        if not osp.isdir(dir_path):
            raise FileNotFoundError(msg_tmpl.format(dir_path))
    elif isinstance(dir_path, Path):
        if not dir_path.is_dir():
            raise FileNotFoundError(msg_tmpl.format(dir_path))


def fopen(filepath, *args, **kwargs):
    """File open wrapper function

    Args:
        filepath (str | obj:`Path`): path address

    Raises:
        ValueError: not a avaliable file

    Returns:
        open session: open file session
    """
    try:
        check_file(filepath)
    except:
        raise ValueError("`filepath` should be a string or a Path and not a directory")
    return open(str(filepath), *args, **kwargs)


def mkdir_or_exist(dir_name, mode=0o777):
    """Use makedirs to wrap mkdir function

    Args:
        dir_name (str | obj:`Path`): directory addres
        mode (CODE, optional): determine user permissions. Defaults to 0o777.
    """
    if dir_name == "":
        return
    dir_name = osp.expanduser(dir_name)
    os.makedirs(dir_name, mode=mode, exist_ok=True)


def symlink(src, dst, overwrite=True, **kwargs):
    r"""Soft link function

    Args:
        src (str | obj:`Path`): source directory address
        dst (str | obj:`Path`): destination directory address
        overwrite (bool, optional): overwrite destination directory ot not. Defaults to True.
    """
    if osp.lexists(dst) and overwrite:
        os.remove(dst)
    os.symlink(src, dst, **kwargs)


def scandir(dir_path, suffix=None, recursive=False):
    r"""Scan a directory to find the interested files.

    Args:
        dir_path (str | obj:`Path`): Path of the directory.
        suffix (str | tuple(str), optional): 
            File suffix that we are interested in. Default: None.
        recursive (bool, optional): 
            If set to True, recursively scan the directory. Default: False.
    Returns:
        A generator for all the interested files with relative pathes.
    """
    if isinstance(dir_path, (str, Path)):
        dir_path = str(dir_path)
    else:
        raise TypeError(f"`dir_path` must be a string or Path object, but got {type(dir_path)}")

    if (suffix is not None) and not isinstance(suffix, (str, tuple)):
        raise TypeError(f"`suffix` must be a string or tuple of strings, but got {type(suffix)}")

    root = dir_path

    def _scandir(dir_path, suffix, recursive):
        for entry in os.scandir(dir_path):
            if not entry.name.startswith('.') and entry.is_file():
                rel_path = osp.relpath(entry.path, root)
                if suffix is None:
                    yield rel_path
                elif rel_path.endswith(suffix):
                    yield rel_path
            else:
                if recursive:
                    yield from _scandir(
                        entry.path, suffix=suffix, recursive=recursive)
                else:
                    continue

    return _scandir(dir_path, suffix=suffix, recursive=recursive)


def find_vcs_root(path, markers=(".git", )):
    r"""Finds the root directory (including itself) of specified markers.
    
    Args:
        path (str): Path of directory or file.
        markers (list[str], optional): List of file or directory names.
    Returns:
        The directory contained one of the markers or None if not found.
    """
    if osp.isfile(path):
        path = osp.dirname(path)

    prev, cur = None, osp.abspath(osp.expanduser(path))
    while cur != prev:
        if any(osp.exists(osp.join(cur, marker)) for marker in markers):
            return cur
        prev, cur = cur, osp.split(cur)[0]
    return None

    