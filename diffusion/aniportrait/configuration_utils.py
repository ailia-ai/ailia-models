# coding=utf-8
# Copyright 2023 The HuggingFace Inc. team.
# Copyright (c) 2022, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

""" ConfigMixin base class and utilities."""

import functools
import inspect
from collections import OrderedDict
from typing import Any, Dict


class FrozenDict(OrderedDict):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        for key, value in self.items():
            setattr(self, key, value)

        self.__frozen = True


class ConfigMixin:
    def register_to_config(self, **kwargs):
        if not hasattr(self, "_internal_dict"):
            internal_dict = kwargs
        else:
            internal_dict = {**self._internal_dict, **kwargs}

        self._internal_dict = FrozenDict(internal_dict)

    @classmethod
    def from_config(cls, config_dict, **kwargs):
        init_dict, _, hidden_dict = cls.extract_init_dict(config_dict, **kwargs)

        # Return model and optionally state and/or unused_kwargs
        model = cls(**init_dict)

        # make sure to also save config parameters that might be used for compatible classes
        model.register_to_config(**hidden_dict)

        return model

    @staticmethod
    def _get_init_keys(cls):
        return set(dict(inspect.signature(cls.__init__).parameters).keys())

    @classmethod
    def extract_init_dict(cls, config_dict, **kwargs):
        # 0. Copy origin config dict
        original_dict = dict(config_dict.items())

        # 1. Retrieve expected config attributes from __init__ signature
        expected_keys = cls._get_init_keys(cls)
        expected_keys.remove("self")

        # remove private attributes
        config_dict = {k: v for k, v in config_dict.items() if not k.startswith("_")}

        # 3. Create keyword arguments that will be passed to __init__ from expected keyword arguments
        init_dict = {}
        for key in expected_keys:
            # if config param is passed to kwarg and is present in config dict
            # it should overwrite existing config dict key
            if key in kwargs and key in config_dict:
                config_dict[key] = kwargs.pop(key)

            if key in kwargs:
                # overwrite key
                init_dict[key] = kwargs.pop(key)
            elif key in config_dict:
                # use value from config dict
                init_dict[key] = config_dict.pop(key)

        # 6. Define unused keyword arguments
        unused_kwargs = {**config_dict, **kwargs}

        # 7. Define "hidden" config parameters that were saved for compatible classes
        hidden_config_dict = {k: v for k, v in original_dict.items() if k not in init_dict}

        return init_dict, unused_kwargs, hidden_config_dict

    @property
    def config(self) -> Dict[str, Any]:
        """
        Returns the config of the class as a frozen dictionary

        Returns:
            `Dict[str, Any]`: Config of the class.
        """
        return self._internal_dict


def register_to_config(init):
    r"""
    Decorator to apply on the init of classes inheriting from [`ConfigMixin`] so that all the arguments are
    automatically sent to `self.register_for_config`. To ignore a specific argument accepted by the init but that
    shouldn't be registered in the config, use the `ignore_for_config` class variable

    Warning: Once decorated, all private arguments (beginning with an underscore) are trashed and not sent to the init!
    """

    @functools.wraps(init)
    def inner_init(self, *args, **kwargs):
        # Ignore private kwargs in the init.
        init_kwargs = {k: v for k, v in kwargs.items() if not k.startswith("_")}
        config_init_kwargs = {k: v for k, v in kwargs.items() if k.startswith("_")}

        # Get positional arguments aligned with kwargs
        new_kwargs = {}
        signature = inspect.signature(init)
        parameters = {
            name: p.default for i, (name, p) in enumerate(signature.parameters.items()) if i > 0
        }
        for arg, name in zip(args, parameters.keys()):
            new_kwargs[name] = arg

        # Then add all kwargs
        new_kwargs.update(
            {
                k: init_kwargs.get(k, default)
                for k, default in parameters.items()
                if k not in new_kwargs
            }
        )

        new_kwargs = {**config_init_kwargs, **new_kwargs}
        getattr(self, "register_to_config")(**new_kwargs)
        init(self, *args, **init_kwargs)

    return inner_init
