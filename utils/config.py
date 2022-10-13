# Copyright (c) Open-MMLab. All rights reserved.
import re
import ast
import sys
import yaml
import shutil
import tempfile
import warnings
import os.path as osp
from typing import Optional
from argparse import Action, ArgumentParser, Namespace

from collections import abc
from importlib import import_module
from addict import Dict
from yapf.yapflib.yapf_api import FormatCode

from .path import check_file
from .misc import import_modules_from_strings

BASE_KEY = "_base_"
DELETE_KEY = "_delete_"
RESERVED_KEYS = ["filename", "text", "pretty_text"]


class ConfigDict(Dict):
    r"""ConfigDict based on Dict, which use to convert the config
        file into config dict
    """
    def __missing__(self, name):
        raise KeyError(name)

    def __getattr__(self, name):
        try:
            value = super(ConfigDict, self).__getattr__(name)
        except KeyError:
            ex = AttributeError(f"`{self.__class__.__name__}` object has no attribute `{name}`")
        except Exception as e:
            ex = e
        else:
            return value
        raise ex


def add_args(parser: ArgumentParser,
             cfg: Dict,
             prefix: str="") -> ArgumentParser:
    for k, v in cfg.items():
        if isinstance(v, str):
            parser.add_argument("--" + prefix + k)
        elif isinstance(v, int):
            parser.add_argument("--" + prefix + k, type=int)
        elif isinstance(v, float):
            parser.add_argument("--" + prefix + k, type=float)
        elif isinstance(v, bool):
            parser.add_argument("--" + prefix + k, action="store_true")
        elif isinstance(v, dict):
            add_args(parser, v, prefix + k + ".")
        elif isinstance(v, abc.Iterable):
            parser.add_argument("--" + prefix + k, type=type(v[0]), nargs="+")
        else:
            print(f"cannot parse key {prefix + k} of type {type(v)}")
    return parser


class Config(object):
    r"""A facility for config and config files.
        It supports common file formats as configs: python/json/yaml. The interface
        is the same as a dict object and also allows access config values as
        attributes.

    Example:
        >>> cfg = Config(dict(a=1, b=dict(b1=[0, 1])))
        >>> cfg.a
        1
        >>> cfg.b
        {"b1": [0, 1]}
        >>> cfg.b.b1
        [0, 1]
        >>> cfg = Config.fromfile("./configs/test.py")
        >>> cfg.filename
        "/home/gorilla_lab/code/gorilla/configs/test.py"
        >>> cfg.item4
        "test"
        >>> cfg
        "Config [path: /home/gorilla_lab/code/gorilla/configs/test.py]: "
        "{"item1": [1, 2], "item2": {"a": 0}, "item3": True, "item4": "test"}"
    """
    @staticmethod
    def _validate_py_syntax(filename: str):
        with open(filename) as f:
            content = f.read()
        try:
            ast.parse(content)
        except SyntaxError as e:
            raise SyntaxError(f"There are syntax errors in config file {filename}: {e}")

    @staticmethod
    def _substitute_predefined_vars(filename: str, temp_config_name: str):
        file_dirname = osp.dirname(filename)
        file_basename = osp.basename(filename)
        file_basename_no_extension = osp.splitext(file_basename)[0]
        file_extname = osp.splitext(filename)[1]

        support_templates = dict(
            file_dir_name=file_dirname,
            file_base_name=file_basename,
            file_base_name_no_extension=file_basename_no_extension,
            file_ext_name=file_extname)

        with open(filename, "r") as f:
            config_file = f.read()

        for key, value in support_templates.items():
            regexp = r"\{\{\s*" + str(key) + r"\s*\}\}"
            config_file = re.sub(regexp, value, config_file)

        with open(temp_config_name, "w") as tmp_config_file:
            tmp_config_file.write(config_file)

    @staticmethod
    def _file2dict(filename: str, use_predefined_variables: bool=True):
        filename = osp.abspath(osp.expanduser(filename))
        check_file(filename)
        file_extname = osp.splitext(filename)[1]
        if file_extname not in [".py", ".json", ".yaml", ".yml"]:
            raise IOError("Only py/yml/yaml/json type are supported now!")

        with tempfile.TemporaryDirectory() as temp_config_dir:
            temp_config_file = tempfile.NamedTemporaryFile(dir=temp_config_dir,
                                                           suffix=file_extname)

            temp_config_name = osp.basename(temp_config_file.name)
            # Substitute predefined variables
            if use_predefined_variables:
                Config._substitute_predefined_vars(filename,
                                                   temp_config_file.name)
            else:
                shutil.copyfile(filename, temp_config_file.name)

            if filename.endswith(".py"):
                temp_module_name = osp.splitext(temp_config_name)[0]
                sys.path.insert(0, temp_config_dir)
                Config._validate_py_syntax(filename)
                mod = import_module(temp_module_name)
                sys.path.pop(0)
                cfg_dict = {
                    name: value
                    for name, value in mod.__dict__.items()
                    if not name.startswith("__")
                }
                # delete imported module
                del sys.modules[temp_module_name]

            elif filename.endswith((".yml", ".yaml", ".json")):
                from .io import load
                cfg_dict = load(temp_config_file.name)
            # close temp file
            temp_config_file.close()

        cfg_text = filename + "\n"
        with open(filename, "r") as f:
            cfg_text += f.read()

        # here cfg_dict is still the same as content in --config file,
        # and the code block below read 4 sub-config file then merge into one.
        # BASE_KEY == "_base_"
        if BASE_KEY in cfg_dict:
            cfg_dir = osp.dirname(filename)
            base_filename = cfg_dict.pop(BASE_KEY)
            base_filename = base_filename if isinstance(
                base_filename, list) else [base_filename]

            cfg_dict_list = list()
            cfg_text_list = list()
            for f in base_filename:
                _cfg_dict, _cfg_text = Config._file2dict(osp.join(cfg_dir, f))
                cfg_dict_list.append(_cfg_dict)
                cfg_text_list.append(_cfg_text)

            base_cfg_dict = dict()
            for c in cfg_dict_list:
                if len(base_cfg_dict.keys() & c.keys()) > 0:
                    # e.g. sub-config file about dataset should not overlap with
                    # the one about model
                    raise KeyError("Duplicate key is not allowed among bases")
                base_cfg_dict.update(c)

            cfg_dict = Config._merge_a_into_b(cfg_dict, base_cfg_dict)

            # merge cfg_text
            cfg_text_list.append(cfg_text)
            cfg_text = "\n".join(cfg_text_list)

        return cfg_dict, cfg_text

    @staticmethod
    def _merge_a_into_b(a, b, allow_list_keys=False):
        r"""merge dict ``a`` into dict ``b`` (non-inplace).
        Values in ``a`` will overwrite ``b``. ``b`` is copied first to avoid
        in-place modifications.
        Args:
            a (dict): The source dict to be merged into ``b``.
            b (dict): The origin dict to be fetch keys from ``a``.
            allow_list_keys (bool): If True, int string keys (e.g. "0", "1")
              are allowed in source ``a`` and will replace the element of the
              corresponding index in b if b is a list. Default: False.
        Returns:
            dict: The modified dict of ``b`` using ``a``.
        Examples:
            # Normally merge a into b.
            >>> Config._merge_a_into_b(
            ...     dict(obj=dict(a=2)), dict(obj=dict(a=1)))
            {"obj": {"a": 2}}
            # Delete b first and merge a into b.
            >>> Config._merge_a_into_b(
            ...     dict(obj=dict(_delete_=True, a=2)), dict(obj=dict(a=1)))
            {"obj": {"a": 2}}
            # b is a list
            >>> Config._merge_a_into_b(
            ...     {"0": dict(a=2)}, [dict(a=1), dict(b=2)], True)
            [{"a": 2}, {"b": 2}]
        """
        b = b.copy()
        for k, v in a.items():
            if allow_list_keys and k.isdigit() and isinstance(b, list):
                k = int(k)
                if len(b) <= k:
                    raise KeyError(f"Index {k} exceeds the length of list {b}")
                b[k] = Config._merge_a_into_b(v, b[k], allow_list_keys)
            elif isinstance(v,
                            dict) and k in b and not v.pop(DELETE_KEY, False):
                allowed_types = (dict, list) if allow_list_keys else dict
                if not isinstance(b[k], allowed_types):
                    raise TypeError(
                        f"{k}={v} in child config cannot inherit from base "
                        f"because {k} is a dict in the child config but is of "
                        f"type {type(b[k])} in base config. You may set "
                        f"`{DELETE_KEY}=True` to ignore the base config")
                b[k] = Config._merge_a_into_b(v, b[k], allow_list_keys)
            else:
                b[k] = v
        return b

    @staticmethod
    def fromfile(filename: str,
                 use_predefined_variables: bool=True,
                 import_custom_modules=True):
        r"""cfg_text is the text content read from 5 files, and cfg_dict is
            a dict resolved by the text content.
        """
        cfg_dict, cfg_text = Config._file2dict(filename,
                                               use_predefined_variables)
        if import_custom_modules and cfg_dict.get('custom_imports', None):
            import_modules_from_strings(**cfg_dict['custom_imports'])
        return Config(cfg_dict, cfg_text=cfg_text, filename=filename)

    @staticmethod
    def fromstring(cfg_str, file_format):
        """Generate config from config str.
        Args:
            cfg_str (str): Config str.
            file_format (str): Config file format corresponding to the
               config str. Only py/yml/yaml/json type are supported now!
        Returns:
            obj:`Config`: Config obj.
        """
        if file_format not in [".py", ".json", ".yaml", ".yml"]:
            raise IOError("Only py/yml/yaml/json type are supported now!")
        if file_format != ".py" and "dict(" in cfg_str:
            # check if users specify a wrong suffix for python
            warnings.warn(
                "Please check 'file_format', the file format may be .py")

        with tempfile.NamedTemporaryFile("w", suffix=file_format) as temp_file:
            temp_file.write(cfg_str)
            temp_file.flush()
            cfg = Config.fromfile(temp_file.name)
        return cfg

    @staticmethod
    def auto_argparser(description: Optional[str]=None):
        r"""Generate argparser from config file automatically (experimental)"""
        partial_parser = ArgumentParser(description=description)
        partial_parser.add_argument("config", help="config file path")
        cfg_file = partial_parser.parse_known_args()[0].config
        cfg = Config.fromfile(cfg_file)
        parser = ArgumentParser(description=description)
        parser.add_argument("config", help="config file path")
        add_args(parser, cfg)
        return parser, cfg

    def __init__(self,
                 cfg_dict: Optional[Dict]=None,
                 cfg_text: Optional[str]=None,
                 filename: Optional[str]=None):
        if cfg_dict is None:
            cfg_dict = dict()
        elif not isinstance(cfg_dict, dict):
            raise TypeError(f"cfg_dict must be a dict, "
                            f"but got {type(cfg_dict)}")
        for key in cfg_dict:
            if key in RESERVED_KEYS:
                raise KeyError(f"{key} is reserved for config file")

        super(Config, self).__setattr__("_cfg_dict", ConfigDict(cfg_dict))
        super(Config, self).__setattr__("_filename", filename)
        if cfg_text:
            text = cfg_text
        elif filename:
            with open(filename, "r") as f:
                text = f.read()
        else:
            text = ""
        super(Config, self).__setattr__("_text", text)


    @property
    def filename(self) -> str:
        return self._filename

    @property
    def text(self) -> str:
        return self._text

    @property
    def pretty_text(self) -> str:

        indent = 4

        def _indent(s_:str, num_spaces: int) -> str:
            s = s_.split("\n")
            if len(s) == 1:
                return s_
            first = s.pop(0)
            s = [(num_spaces * " ") + line for line in s]
            s = "\n".join(s)
            s = first + "\n" + s
            return s

        def _format_basic_types(k, v, use_mapping=False):
            if isinstance(v, str):
                v_str = f"'{v}'"
            else:
                v_str = str(v)

            if use_mapping:
                k_str = f"'{k}'" if isinstance(k, str) else str(k)
                attr_str = f"{k_str}: {v_str}"
            else:
                attr_str = f"{k}={v_str}"
            attr_str = _indent(attr_str, indent)

            return attr_str

        def _format_list(k, v, use_mapping=False):
            # check if all items in the list are dict
            if all(isinstance(_, dict) for _ in v):
                v_str = "[\n"
                v_str += "\n".join(
                    f"dict({_indent(_format_dict(v_), indent)}),"
                    for v_ in v).rstrip(",")
                if use_mapping:
                    k_str = f"'{k}'" if isinstance(k, str) else str(k)
                    attr_str = f"{k_str}: {v_str}"
                else:
                    attr_str = f"{k}={v_str}"
                attr_str = _indent(attr_str, indent) + "]"
            else:
                attr_str = _format_basic_types(k, v, use_mapping)
            return attr_str

        def _contain_invalid_identifier(dict_str):
            contain_invalid_identifier = False
            for key_name in dict_str:
                contain_invalid_identifier |= \
                    (not str(key_name).isidentifier())
            return contain_invalid_identifier

        def _format_dict(input_dict, outest_level=False):
            r = ""
            s = []

            use_mapping = _contain_invalid_identifier(input_dict)
            if use_mapping:
                r += "{"
            for idx, (k, v) in enumerate(input_dict.items()):
                is_last = idx >= len(input_dict) - 1
                end = "" if outest_level or is_last else ","
                if isinstance(v, dict):
                    v_str = "\n" + _format_dict(v)
                    if use_mapping:
                        k_str = f"'{k}'" if isinstance(k,
                                                               str) else str(k)
                        attr_str = f"{k_str}: dict({v_str}"
                    else:
                        attr_str = f"{k}=dict({v_str}"
                    attr_str = _indent(attr_str, indent) + ")" + end
                elif isinstance(v, list):
                    attr_str = _format_list(k, v, use_mapping) + end
                else:
                    attr_str = _format_basic_types(k, v, use_mapping) + end

                s.append(attr_str)
            r += "\n".join(s)
            if use_mapping:
                r += "}"
            return r

        cfg_dict = self._cfg_dict.to_dict()
        text = _format_dict(cfg_dict, outest_level=True)
        # copied from setup.cfg
        yapf_style = dict(based_on_style="pep8",
                          blank_line_before_nested_class_or_def=True,
                          split_before_expression_after_opening_paren=True)

        # NOTE: avoid show
        # print("text:\n", text)
        text, _ = FormatCode(text, style_config=yapf_style, verify=True)

        return text

    def _pretty(self, d: Dict, depth: int=0) -> None:
        for key, value in dict(d).items():
            if key in ["content"]:
                continue
            self.content += f"{'    ' * depth}{key}: "
            if isinstance(value, dict):
                self.content += "\n"
                self._pretty(value, depth + 1)
            else:
                self.content += f"{value}\n"

    def __repr__(self) -> str:
        self.content = f"Config (path: {self.filename})\n"
        self._pretty(self._cfg_dict)
        return self.content

    def __len__(self) -> int:
        return len(self._cfg_dict)

    def __getattr__(self, name: str):
        return getattr(self._cfg_dict, name)

    def __getitem__(self, name: str):
        return self._cfg_dict.__getitem__(name)

    def __setattr__(self, name: str, value: Dict):
        if isinstance(value, dict):
            value = ConfigDict(value)
        self._cfg_dict.__setattr__(name, value)

    def __setitem__(self, name: str, value: Dict):
        if isinstance(value, dict):
            value = ConfigDict(value)
        self._cfg_dict.__setitem__(name, value)

    def __iter__(self):
        return iter(self._cfg_dict)

    def dump(self, file: Optional[str]=None, **kwargs):
        cfg_dict = super(Config, self).__getattribute__("_cfg_dict").to_dict()
        if self.filename.endswith(".py"):
            if file is None:
                return self.pretty_text
            else:
                with open(file, "w") as f:
                    f.write(self.pretty_text)
        else:
            from .io import dump
            if file is None:
                file_format = self.filename.split(".")[-1]
                return dump(cfg_dict, file_format=file_format, **kwargs)
            else:
                dump(cfg_dict, file, **kwargs)

    def merge_from_dict(self, options: Dict, allow_list_keys: bool=True):
        r"""Merge list into cfg_dict.
        Merge the dict parsed by MultipleKVAction into this cfg.
        Examples:
            >>> options = {"model.backbone.depth": 50,
            ...            "model.backbone.with_cp":True}
            >>> cfg = Config(dict(model=dict(backbone=dict(type="ResNet"))))
            >>> cfg.merge_from_dict(options)
            >>> cfg_dict = super(Config, self).__getattribute__("_cfg_dict")
            >>> assert cfg_dict == dict(
            ...     model=dict(backbone=dict(depth=50, with_cp=True)))
            
            # Merge list element
            >>> cfg = Config(dict(pipeline=[
            ...     dict(type="LoadImage"), dict(type="LoadAnnotations")]))
            >>> options = dict(pipeline={"0": dict(type="SelfLoadImage")})
            >>> cfg.merge_from_dict(options, allow_list_keys=True)
            >>> cfg_dict = super(Config, self).__getattribute__("_cfg_dict")
            >>> assert cfg_dict == dict(pipeline=[
            ...     dict(type="SelfLoadImage"), dict(type="LoadAnnotations")])
        Args:
            options (dict): dict of configs to merge from.
            allow_list_keys (bool): If True, int string keys (e.g. '0', '1')
              are allowed in ``options`` and will replace the element of the
              corresponding index in the config if the config is a list.
              Default: True.
        """
        option_cfg_dict = {}
        for full_key, v in options.items():
            if v is None:  # handle the case when a parameter simultaneously appears in argparse and config file
                continue
            d = option_cfg_dict
            key_list = full_key.split(".")
            for subkey in key_list[:-1]:
                d.setdefault(subkey, ConfigDict())
                d = d[subkey]
            subkey = key_list[-1]
            d[subkey] = v

        cfg_dict = super(Config, self).__getattribute__("_cfg_dict")
        super(Config, self).__setattr__(
            "_cfg_dict",
            Config._merge_a_into_b(
                option_cfg_dict, cfg_dict, allow_list_keys=allow_list_keys))


def merge_cfg_and_args(cfg: Optional[Config]=None,
                       args: Optional[Namespace]=None) -> Config:
    r"""merge args and cfg into a Config by calling 'merge_from_dict' func

    Args:
        cfg (Config, optional): Config from cfg file.
        args (Namespace, optional): Argument parameters input.

    Returns:
        Config: Merged Config
    """
    assert cfg is not None or args is not None, "'cfg' or 'args' can not be None simultaneously"

    if cfg is None:
        cfg = Config()
    else:
        assert isinstance(cfg, Config), f"'cfg' must be None or gorilla.Config, but got {type(cfg)}"
    if args is None:
        args = Namespace()
    else:
        assert isinstance(args, Namespace), f"'args' must be None or argsparse.Namespace, but got {type(args)}"

    # convert namespace into dict
    args_dict = vars(args)
    cfg.merge_from_dict(args_dict)
    return cfg


# TODO: read it
class DictAction(Action):
    r"""
    argparse action to split an argument into KEY=VALUE form
    on the first = and append to a dictionary. List options can
    be passed as comma separated values, i.e 'KEY=V1,V2,V3', or with explicit
    brackets, i.e. 'KEY=[V1,V2,V3]'. It also support nested brackets to build
    list/tuple values. e.g. 'KEY=[(V1,V2),(V3,V4)]'
    """

    @staticmethod
    def _parse_int_float_bool(val):
        try:
            return int(val)
        except ValueError:
            pass
        try:
            return float(val)
        except ValueError:
            pass
        if val.lower() in ['true', 'false']:
            return True if val.lower() == 'true' else False
        return val

    @staticmethod
    def _parse_iterable(val):
        """Parse iterable values in the string.
        All elements inside '()' or '[]' are treated as iterable values.
        Args:
            val (str): Value string.
        Returns:
            list | tuple: The expanded list or tuple from the string.
        Examples:
            >>> DictAction._parse_iterable('1,2,3')
            [1, 2, 3]
            >>> DictAction._parse_iterable('[a, b, c]')
            ['a', 'b', 'c']
            >>> DictAction._parse_iterable('[(1, 2, 3), [a, b], c]')
            [(1, 2, 3), ['a', 'b], 'c']
        """

        def find_next_comma(string):
            """Find the position of next comma in the string.
            If no ',' is found in the string, return the string length. All
            chars inside '()' and '[]' are treated as one element and thus ','
            inside these brackets are ignored.
            """
            assert (string.count('(') == string.count(')')) and (
                    string.count('[') == string.count(']')), \
                f'Imbalanced brackets exist in {string}'
            end = len(string)
            for idx, char in enumerate(string):
                pre = string[:idx]
                # The string before this ',' is balanced
                if ((char == ',') and (pre.count('(') == pre.count(')'))
                        and (pre.count('[') == pre.count(']'))):
                    end = idx
                    break
            return end

        # Strip ' and " characters and replace whitespace.
        val = val.strip('\'\"').replace(' ', '')
        is_tuple = False
        if val.startswith('(') and val.endswith(')'):
            is_tuple = True
            val = val[1:-1]
        elif val.startswith('[') and val.endswith(']'):
            val = val[1:-1]
        elif ',' not in val:
            # val is a single value
            return DictAction._parse_int_float_bool(val)

        values = []
        while len(val) > 0:
            comma_idx = find_next_comma(val)
            element = DictAction._parse_iterable(val[:comma_idx])
            values.append(element)
            val = val[comma_idx + 1:]
        if is_tuple:
            values = tuple(values)
        return values

    def __call__(self, parser, namespace, values, option_string=None):
        options = {}
        for kv in values:
            key, val = kv.split('=', maxsplit=1)
            options[key] = self._parse_iterable(val)
        setattr(namespace, self.dest, options)

