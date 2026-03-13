"""
Reference: Kiuikit, https://github.com/ashawkey/kiuikit
"""

import numpy as np
import cv2

def safe_normalize(x: np.ndarray, eps=1e-20) -> np.ndarray:
    length = np.sqrt(np.maximum(np.sum(x * x, axis=-1, keepdims=True), eps))
    return x / length

def look_at(campos, target, opengl=True):
    if not opengl:
        # forward is camera --> target
        forward_vector = safe_normalize(target - campos)
        up_vector = np.array([0, 1, 0], dtype=np.float32)
        right_vector = safe_normalize(np.cross(forward_vector, up_vector))
        up_vector = safe_normalize(np.cross(right_vector, forward_vector))
    else:
        # forward is target --> camera
        forward_vector = safe_normalize(campos - target)
        up_vector = np.array([0, 1, 0], dtype=np.float32)
        right_vector = safe_normalize(np.cross(up_vector, forward_vector))
        up_vector = safe_normalize(np.cross(forward_vector, right_vector))
    R = np.stack([right_vector, up_vector, forward_vector], axis=1)
    return R

def orbit_camera(elevation, azimuth, radius=1, is_degree=True, target=None, opengl=True):
    if is_degree:
        elevation = np.deg2rad(elevation)
        azimuth = np.deg2rad(azimuth)
    x = radius * np.cos(elevation) * np.sin(azimuth)
    y = - radius * np.sin(elevation)
    z = radius * np.cos(elevation) * np.cos(azimuth)
    if target is None:
        target = np.zeros([3], dtype=np.float32)
    campos = np.array([x, y, z]) + target  # [3]
    T = np.eye(4, dtype=np.float32)
    T[:3, :3] = look_at(campos, target, opengl)
    T[:3, 3] = campos
    return T

def recenter(image: np.ndarray, mask: np.ndarray, border_ratio: float = 0.2) -> np.ndarray:
    return_int = False
    if image.dtype == np.uint8:
        image = image.astype(np.float32) / 255
        return_int = True
    
    H, W, C = image.shape
    size = max(H, W)

    # default to white bg if rgb, but use 0 if rgba
    if C == 3:
        result = np.ones((size, size, C), dtype=np.float32)
    else:
        result = np.zeros((size, size, C), dtype=np.float32)
            
    coords = np.nonzero(mask)
    x_min, x_max = coords[0].min(), coords[0].max()
    y_min, y_max = coords[1].min(), coords[1].max()
    h = x_max - x_min
    w = y_max - y_min
    desired_size = int(size * (1 - border_ratio))
    scale = desired_size / max(h, w)
    h2 = int(h * scale)
    w2 = int(w * scale)
    x2_min = (size - h2) // 2
    x2_max = x2_min + h2
    y2_min = (size - w2) // 2
    y2_max = y2_min + w2
    result[x2_min:x2_max, y2_min:y2_max] = cv2.resize(image[x_min:x_max, y_min:y_max], (w2, h2), interpolation=cv2.INTER_AREA)

    if return_int:
        result = (result * 255).astype(np.uint8)

    return result
