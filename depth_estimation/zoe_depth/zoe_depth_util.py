from typing import Tuple

import cv2
import matplotlib
import numpy as np
from einops import rearrange
from PIL import Image


def get_params(arch):
    weight_path = f"{arch}.onnx"
    model_path = f"{arch}.onnx.prototxt"
    return weight_path, model_path


def preprocess(img_orig, input_size) -> Tuple[np.ndarray, np.ndarray]:
    resized_img = cv2.resize(img_orig, input_size).astype(np.float32)
    resized_img /= 255.0
    resized_img_reversed = resized_img[..., ::-1]
    resized_img = rearrange(resized_img, "h w c -> 1 c h w")
    resized_img_reversed = rearrange(resized_img_reversed, "h w c -> 1 c h w")
    return resized_img, resized_img_reversed


def postprocess(
    pred,
    original_width: int,
    original_height: int,
    vmin: int = 0,
    vmax: int = 10,
    cmap: str = "magma_r",
) -> np.ndarray:
    invalid_mask = pred == -99
    mask = np.logical_not(invalid_mask)

    if vmin is None:
        vmin = np.percentile(pred[mask], 2)
    if vmax is None:
        vmax = np.percentile(pred[mask], 85)

    pred = (pred - vmin) / (vmax - vmin)
    pred[invalid_mask] = np.nan
    if hasattr(matplotlib, "colormaps"):
        cmapper = matplotlib.colormaps[cmap]
    else:
        cmapper = matplotlib.cm.get_cmap(cmap)
    pred = cmapper(pred, bytes=True)
    img = pred[...]
    img[invalid_mask] = (128, 128, 128, 0)
    img = cv2.resize(img, (original_width, original_height))
    return img


def save(
    pred: np.ndarray,
    output_filename: str,
):
    Image.fromarray(pred).save(output_filename)
