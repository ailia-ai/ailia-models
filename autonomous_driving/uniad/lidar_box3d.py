# Copyright (c) OpenMMLab. All rights reserved.
from abc import abstractmethod

import numpy as np


class BaseInstance3DBoxes(object):
    """Base class for 3D Boxes.

    Note:
        The box is bottom centered, i.e. the relative position of origin in
        the box is (0.5, 0.5, 0).

    Args:
        tensor (torch.Tensor | np.ndarray | list): a N x box_dim matrix.
        box_dim (int): Number of the dimension of a box.
            Each row is (x, y, z, x_size, y_size, z_size, yaw).
            Defaults to 7.
        with_yaw (bool): Whether the box is with yaw rotation.
            If False, the value of yaw will be set to 0 as minmax boxes.
            Defaults to True.
        origin (tuple[float], optional): Relative position of the box origin.
            Defaults to (0.5, 0.5, 0). This will guide the box be converted to
            (0.5, 0.5, 0) mode.

    Attributes:
        tensor (torch.Tensor): Float matrix of N x box_dim.
        box_dim (int): Integer indicating the dimension of a box.
            Each row is (x, y, z, x_size, y_size, z_size, yaw, ...).
        with_yaw (bool): If True, the value of yaw will be set to 0 as minmax
            boxes.
    """

    def __init__(self, tensor, box_dim=7, with_yaw=True, origin=(0.5, 0.5, 0)):
        tensor = np.asarray(tensor, dtype=np.float32)
        if tensor.size == 0:
            tensor = tensor.reshape((0, box_dim)).astype(np.float32)
        # assert tensor.ndim == 2 and tensor.shape[-1] == box_dim, tensor.shape

        if tensor.shape[-1] == 6:
            # If the dimension of boxes is 6, we expand box_dim by padding
            # 0 as a fake yaw and set with_yaw to False.
            assert box_dim == 6
            fake_rot = np.zeros((tensor.shape[0], 1), dtype=tensor.dtype)
            tensor = np.concatenate((tensor, fake_rot), axis=-1)
            self.box_dim = box_dim + 1
            self.with_yaw = False
        else:
            self.box_dim = box_dim
            self.with_yaw = with_yaw
        self.tensor = tensor.copy()

        if origin != (0.5, 0.5, 0):
            dst = np.array((0.5, 0.5, 0))
            src = np.array(origin)
            self.tensor[:, :3] += self.tensor[:, 3:6] * (dst - src)

    @property
    def volume(self):
        """np.array: A vector with volume of each box."""
        return self.tensor[:, 3] * self.tensor[:, 4] * self.tensor[:, 5]

    @property
    def dims(self):
        """np.array: Size dimensions of each box in shape (N, 3)."""
        return self.tensor[:, 3:6]

    @property
    def yaw(self):
        """np.array: A vector with yaw of each box in shape (N, )."""
        return self.tensor[:, 6]

    @property
    def bottom_center(self):
        """np.array: A tensor with center of each box in shape (N, 3)."""
        return self.tensor[:, :3]

    @abstractmethod
    def rotate(self, angle, points=None):
        """Rotate boxes with points (optional) with the given angle or rotation
        matrix.

        Args:
            angle (float | torch.Tensor | np.ndarray):
                Rotation angle or rotation matrix.
            points (torch.Tensor | numpy.ndarray |
                :obj:`BasePoints`, optional):
                Points to rotate. Defaults to None.
        """
        pass

    def translate(self, trans_vector):
        """Translate boxes with the given translation vector.

        Args:
            trans_vector (np.array): Translation vector of size (1, 3).
        """
        if not isinstance(trans_vector, np.ndarray):
            trans_vector = np.array(trans_vector)
        self.tensor[:, :3] += trans_vector

    def clone(self):
        """Clone the Boxes.

        Returns:
            :obj:`BaseInstance3DBoxes`: Box object with the same properties
                as self.
        """
        original_type = type(self)
        return original_type(
            self.tensor.copy(), box_dim=self.box_dim, with_yaw=self.with_yaw
        )


class LiDARInstance3DBoxes(BaseInstance3DBoxes):
    """3D boxes of instances in LIDAR coordinates.

    Coordinates in LiDAR:

    .. code-block:: none

                                up z    x front (yaw=0)
                                   ^   ^
                                   |  /
                                   | /
       (yaw=0.5*pi) left y <------ 0

    The relative coordinate of bottom center in a LiDAR box is (0.5, 0.5, 0),
    and the yaw is around the z axis, thus the rotation axis=2.
    The yaw is 0 at the positive direction of x axis, and increases from
    the positive direction of x to the positive direction of y.

    A refactor is ongoing to make the three coordinate systems
    easier to understand and convert between each other.

    Attributes:
        tensor (np.ndarray): Float matrix of N x box_dim.
        box_dim (int): Integer indicating the dimension of a box.
            Each row is (x, y, z, x_size, y_size, z_size, yaw, ...).
        with_yaw (bool): If True, the value of yaw will be set to 0 as minmax
            boxes.
    """

    YAW_AXIS = 2

    @property
    def gravity_center(self):
        """np.array: A tensor with center of each box in shape (N, 3)."""
        bottom_center = self.bottom_center
        gravity_center = np.zeros_like(bottom_center)
        gravity_center[:, :2] = bottom_center[:, :2]
        gravity_center[:, 2] = bottom_center[:, 2] + self.tensor[:, 5] * 0.5
        return gravity_center
