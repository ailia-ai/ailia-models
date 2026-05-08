import cv2
import numpy as np


def get_3rd_point(a, b):
    direct = a - b
    return b + np.array([-direct[1], direct[0]], dtype=np.float32)


def get_dir(src_point, rot_rad):
    sn, cs = np.sin(rot_rad), np.cos(rot_rad)
    return [src_point[0] * cs - src_point[1] * sn,
            src_point[0] * sn + src_point[1] * cs]


def get_affine_transform(center, scale, rot, output_size,
                         shift=np.array([0, 0], dtype=np.float32), inv=False):
    """Compute affine transform matrix, matching original repo."""
    if not isinstance(scale, np.ndarray):
        scale = np.array([scale, scale], dtype=np.float32)

    scale_tmp = scale * 200.0
    src_w = scale_tmp[0]
    dst_w, dst_h = output_size

    rot_rad = np.pi * rot / 180
    src_dir = get_dir([0, src_w * -0.5], rot_rad)
    dst_dir = np.array([0, dst_w * -0.5], dtype=np.float32)

    src = np.zeros((3, 2), dtype=np.float32)
    dst = np.zeros((3, 2), dtype=np.float32)
    src[0, :] = center + scale_tmp * shift
    src[1, :] = center + np.array(src_dir) + scale_tmp * shift
    dst[0, :] = [dst_w * 0.5, dst_h * 0.5]
    dst[1, :] = np.array([dst_w * 0.5, dst_h * 0.5]) + dst_dir

    src[2, :] = get_3rd_point(src[0, :], src[1, :])
    dst[2, :] = get_3rd_point(dst[0, :], dst[1, :])

    if inv:
        trans = cv2.getAffineTransform(np.float32(dst), np.float32(src))
    else:
        trans = cv2.getAffineTransform(np.float32(src), np.float32(dst))
    return trans


def _get_transform(center, scale, output_size):
    """Build 3x3 transform matrix matching original repo's get_transform."""
    if isinstance(scale, np.ndarray):
        scale = float(scale[0])
    h = 200.0 * scale
    t = np.zeros((3, 3), dtype=np.float64)
    t[0, 0] = output_size[0] / h
    t[1, 1] = output_size[1] / h
    t[0, 2] = output_size[0] * (-center[0] / h + 0.5)
    t[1, 2] = output_size[1] * (-center[1] / h + 0.5)
    t[2, 2] = 1
    return t


def transform_preds(coords, center, scale, output_size):
    """Map heatmap coordinates back to original image space.

    Uses the same inverse transform as the original repo's transform_pixel.
    coords: (K, 2), 1-indexed heatmap coordinates
    """
    if not isinstance(scale, (np.ndarray, list)):
        scale = np.array([scale, scale], dtype=np.float32)

    t = _get_transform(center, scale, output_size)
    t_inv = np.linalg.inv(t)

    result = np.zeros_like(coords)
    for i in range(len(coords)):
        # original: transform_pixel uses pt - 1 (0-indexed input), result + 1
        pt = np.array([coords[i, 0] - 1, coords[i, 1] - 1, 1.0])
        new_pt = t_inv @ pt
        result[i, 0] = new_pt[0] + 1
        result[i, 1] = new_pt[1] + 1

    return result.astype(np.float32)
