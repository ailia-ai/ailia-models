import itertools
from typing import Any, Dict, List, Tuple, Union

import numpy as np


class Instances:
    def __init__(self, image_size: Tuple[int, int], **kwargs: Any):
        """
        Args:
            image_size (height, width): the spatial size of the image.
            kwargs: fields to add to this `Instances`.
        """
        self._image_size = image_size
        self._fields: Dict[str, Any] = {}
        for k, v in kwargs.items():
            self.set(k, v)

    @property
    def image_size(self) -> Tuple[int, int]:
        """
        Returns:
            tuple: height, width
        """
        return self._image_size

    def __setattr__(self, name: str, val: Any) -> None:
        if name.startswith("_"):
            super().__setattr__(name, val)
        else:
            self.set(name, val)

    def __getattr__(self, name: str) -> Any:
        return self._fields[name]

    def set(self, name: str, value: Any) -> None:
        self._fields[name] = value

    def has(self, name: str) -> bool:
        return name in self._fields

    def remove(self, name: str) -> None:
        del self._fields[name]

    def get(self, name: str) -> Any:
        return self._fields[name]

    def __getitem__(self, item: Union[int, slice, np.ndarray]) -> "Instances":
        """
        Args:
            item: an index-like object and will be used to index all the fields.
        Returns:
            If `item` is a string, return the data in the corresponding field.
            Otherwise, returns an `Instances` where all fields are indexed by `item`.
        """
        if type(item) == int:
            if item >= len(self) or item < -len(self):
                raise IndexError("Instances index out of range!")
            else:
                item = slice(item, None, len(self))

        ret = Instances(self._image_size)
        for k, v in self._fields.items():
            # if index by numpy boolean array
            if k == "kalman_models" and isinstance(item, np.ndarray):
                ret_list = []
                for i, if_true in enumerate(item):
                    if if_true:
                        ret_list.append(self.kalman_models[i])
                ret.set(k, ret_list)
            else:
                ret.set(k, v[item])
        return ret

    def __len__(self) -> int:
        for v in self._fields.values():
            # use __len__ because len() has to be int and is not friendly to tracing
            return v.__len__()
        raise NotImplementedError("Empty Instances does not support __len__!")

    @staticmethod
    def cat(instance_lists: List["Instances"]) -> "Instances":
        """
        Args:
            instance_lists (list[Instances])
        Returns:
            Instances
        """
        if len(instance_lists) == 1:
            return instance_lists[0]

        image_size = instance_lists[0].image_size
        ret = Instances(image_size)
        for k in instance_lists[0]._fields.keys():
            values = [i.get(k) for i in instance_lists]
            v0 = values[0]
            if isinstance(v0, np.ndarray):
                values = np.concatenate(values, axis=0)
            elif isinstance(v0, list):
                values = list(itertools.chain(*values))
            elif hasattr(type(v0), "cat"):
                values = type(v0).cat(values)
            else:
                raise ValueError(
                    "Unsupported type {} for concatenation".format(type(v0))
                )
            ret.set(k, values)
        return ret
