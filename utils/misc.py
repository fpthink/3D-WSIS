# Copyright (c) Open-MMLab. All rights reserved.
import functools
import operator
import subprocess
import warnings
from collections import abc
from importlib import import_module
from inspect import getfullargspec
from six.moves import map, zip
from typing import Dict, List, Callable, Iterable, Optional, Sequence, Union


def import_modules_from_strings(imports, allow_failed_imports=False):
    """Import modules from the given list of strings.
    Args:
        imports (list | str | None): The given module names to be imported.
        allow_failed_imports (bool): If True, the failed imports will return
            None. Otherwise, an ImportError is raise. Default: False.
    Returns:
        list[module] | module | None: The imported modules.
    Examples:
        >>> osp, sys = import_modules_from_strings(
        ...     ['os.path', 'sys'])
        >>> import os.path as osp_
        >>> import sys as sys_
        >>> assert osp == osp_
        >>> assert sys == sys_
    """
    if not imports:
        return
    single_import = False
    if isinstance(imports, str):
        single_import = True
        imports = [imports]
    if not isinstance(imports, list):
        raise TypeError(
            f'custom_imports must be a list but got type {type(imports)}')
    imported = []
    for imp in imports:
        if not isinstance(imp, str):
            raise TypeError(
                f'{imp} is of type {type(imp)} and cannot be imported.')
        try:
            imported_tmp = import_module(imp)
        except ImportError:
            if allow_failed_imports:
                warnings.warn(f'{imp} failed to import and is ignored.',
                              UserWarning)
                imported_tmp = None
            else:
                raise ImportError
        imported.append(imported_tmp)
    if single_import:
        imported = imported[0]
    return imported


def convert_list(input_list: List, type: Callable):
    return list(map(type, input_list))

convert_list_str = functools.partial(convert_list, type=str)
convert_list_int = functools.partial(convert_list, type=int)
convert_list_float = functools.partial(convert_list, type=float)


def iter_cast(inputs: Iterable,
              dst_type: Callable,
              return_type: Optional[Callable]=None):
    r"""Cast elements of an iterable object into some type.

    Args:
        inputs (Iterable): The input object.
        dst_type (type): Destination type.
        return_type (type, optional): If specified, the output object will be
            converted to this type, otherwise an iterator.
    Returns:
        iterator or specified type: The converted object.
    """
    if not isinstance(inputs, abc.Iterable):
        raise TypeError("inputs must be an iterable object")
    if not isinstance(dst_type, type):
        raise TypeError("`dst_type` must be a valid type")

    out_iterable = map(dst_type, inputs)

    if return_type is None:
        return out_iterable
    else:
        return return_type(out_iterable)


list_cast = functools.partial(iter_cast, return_type=list)
tuple_cast = functools.partial(iter_cast, return_type=tuple)


def is_seq_of(seq: Sequence,
              expected_type: Callable,
              seq_type: Optional[Callable]=None) -> bool:
    r"""Check whether it is a sequence of some type.

    Args:
        seq (Sequence): The sequence to be checked.
        expected_type (type): Expected type of sequence items.
        seq_type (type, optional): Expected sequence type.
    Returns:
        bool: Whether the sequence is valid.
    """
    exp_seq_type = abc.Sequence
    if seq_type is not None:
        assert isinstance(seq_type, type), "`seq_type` must be a valid type"
        exp_seq_type = seq_type
    if not isinstance(seq, exp_seq_type):
        return False
    for item in seq:
        if not isinstance(item, expected_type):
            return False
    return True


is_list_of = functools.partial(is_seq_of, seq_type=list)
is_tuple_of = functools.partial(is_seq_of, seq_type=tuple)


def slice_list(in_list: List, lens: Union[int, List]) -> list:
    r"""Slice a list into several sub lists by a list of given length.

    Args:
        in_list (list): The list to be sliced.
        lens(int or list): The expected length of each out list.
    Returns:
        list: A list of sliced list.
    """
    if isinstance(lens, int):
        assert len(in_list) % lens == 0
        lens = [lens] * int(len(in_list) / lens)
    if not isinstance(lens, list):
        raise TypeError("'indices' must be an integer or a list of integers")
    elif sum(lens) != len(in_list):
        raise ValueError(f"sum of lens and list length does not "
                         f"match: {sum(lens)} != {len(in_list)}")

    out_list = []
    idx = 0
    for l in lens:
        out_list.append(in_list[idx:idx + l])
        idx += l
    return out_list


def concat_list(in_list: List) -> list:
    r"""Concatenate a list of list into a single list.

    Args:
        in_list (list): The list of list to be merged.
    Returns:
        list: The concatenated flat list.
    """
    # return list(itertools.chain(*in_list))
    return functools.reduce(operator.iconcat, in_list, [])


def check_prerequisites(prerequisites: Union[str, List[str]],
                        checker: Callable,
                        msg_tmpl: Optional[str]=None):
    r"""A decorator factory to check if prerequisites are satisfied.
    Args:
        prerequisites (str of list[str]): Prerequisites to be checked.
        checker (callable): The checker method that returns True if a
            prerequisite is meet, False otherwise.
        msg_tmpl (str | None, optional): The message template with two variables.
    Returns:
        decorator: A specific decorator.
    """
    if msg_tmpl is None:
        msg_tmpl = ("Prerequisites '{}' are required "
                    "in method '{}' but not found, "
                    "please install them first.")

    def wrap(func):
        @functools.wraps(func)
        def wrapped_func(*args, **kwargs):
            requirements = [prerequisites] if isinstance(prerequisites, str) \
                           else list_cast(prerequisites, str)
            missing = []
            for item in requirements:
                if not checker(item):
                    missing.append(item)
            if missing:
                print(msg_tmpl.format(", ".join(missing), func.__name__))
                raise RuntimeError("Prerequisites not meet.")
            else:
                return func(*args, **kwargs)

        return wrapped_func

    return wrap


def _check_py_package(package: str) -> bool:
    r"""Check whether package can be import

    Args:
        package (str): Package"s name

    Returns:
        bool: The `package` can be import or not
    """
    assert isinstance(package,
                      str), "`package` must be the string of package's name"
    try:
        import_module(package)
    except ImportError:
        return False
    else:
        return True


def _check_executable(cmd: str  ) -> bool:
    r"""Check whether cmd can be executed

    Args:
        cmd (str): Cmd content

    Returns:
        bool: The `cmd` can be executed or not
    """
    if subprocess.call(f"which {cmd}", shell=True) != 0:
        return False
    else:
        return True


requires_package = functools.partial(check_prerequisites,
                                     checker=_check_py_package)
requires_executable = functools.partial(check_prerequisites,
                                        checker=_check_executable)


# NOTE: use to maintain
def deprecated_api_warning(name_dict: Dict, cls_name: Optional[str]=None):
    r"""A decorator to check if some argments are deprecate and try to replace
    deprecate src_arg_name to dst_arg_name.

    Args:
        name_dict(dict):
            key (str): Deprecate argument names.
            val (str): Expected argument names.
    Returns:
        func: New function.
    """
    def api_warning_wrapper(old_func):
        @functools.wraps(old_func)
        def new_func(*args, **kwargs):
            # get the arg spec of the decorated method
            args_info = getfullargspec(old_func)
            # get name of the function
            func_name = old_func.__name__
            if cls_name is not None:
                func_name = f"{cls_name}.{func_name}"
            if args:
                arg_names = args_info.args[:len(args)]
                for src_arg_name, dst_arg_name in name_dict.items():
                    if src_arg_name in arg_names:
                        warnings.warn(f"'{src_arg_name}' is deprecated in `{func_name}`, "
                                      f"please use `{dst_arg_name}` instead")
                        arg_names[arg_names.index(src_arg_name)] = dst_arg_name
            if kwargs:
                for src_arg_name, dst_arg_name in name_dict.items():
                    if src_arg_name in kwargs:
                        warnings.warn(f"`{src_arg_name}` is deprecated in `{func_name}`, "
                                      f"please use `{dst_arg_name}` ")
                        kwargs[dst_arg_name] = kwargs.pop(src_arg_name)

            # apply converted arguments to the decorated method
            output = old_func(*args, **kwargs)
            return output

        return new_func

    return api_warning_wrapper


def multi_apply(func: Callable, *args, **kwargs):
    r"""Apply function to a list of arguments.
    Note:
        This function applies the ``func`` to multiple inputs and
        map the multiple outputs of the ``func`` into different
        list. Each list contains the same type of outputs corresponding
        to different inputs.
    Args:
        func (Callable): A function that will be applied to a list of
            arguments
    Returns:
        tuple(list): A tuple containing multiple list, each list contains \
            a kind of returned results by the function
    """
    pfunc = functools.partial(func, **kwargs) if kwargs else func
    map_results = map(pfunc, *args)
    return tuple(map(list, zip(*map_results)))

def is_power2(num: int) -> bool:
    r"""Author: liang.zhihao

    Args:
        num (int): input positive number

    Returns:
        bool: is the power of 2 or not
    """
    return num != 0 and ((num & (num - 1)) == 0)

def is_multiple(num: Union[int, float], multiple: Union[int, float]) -> bool:
    r"""Author: liang.zhihao

    Args:
        num (int, float): input number
        multiple (int, float): input multiple number

    Returns:
        bool: can num be multiply by multiple or not
    """
    return num != 0. and num % multiple == 0.

