import sys
import cv2
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from typing import List, Optional, Union

import ailia

sys.path.append('../../util')
from detector_utils import load_image  # noqa

class LightGlueRunner:
    def __init__(
        self,
        lightglue_path: str,
        extractor_path=None,
        env_id=None,
        memory_mode=None
    ):
        self.extractor = (
            ailia.Net(None,extractor_path,env_id=env_id,memory_mode=memory_mode)
        )
        self.lightglue = ailia.Net(None,lightglue_path,env_id=env_id,memory_mode=memory_mode)

    def run(self, image0: np.ndarray, image1: np.ndarray, scales0, scales1):
        if self.extractor is None:
             return None,None
        else:
            kpts0, scores0, desc0 = self.extractor.run(image0)
            kpts1, scores1, desc1 = self.extractor.run(image1)

            matches0, matches1, mscores0, mscores1 = self.lightglue.run(
                    [self.normalize_keypoints(
                        kpts0, image0.shape[2], image0.shape[3]
                    ),
                    self.normalize_keypoints(
                        kpts1, image1.shape[2], image1.shape[3]
                    ),
                    desc0,
                    desc1]
            )
            m_kpts0, m_kpts1 = self.post_process(
                kpts0, kpts1, matches0, scales0, scales1
            )
            return m_kpts0, m_kpts1

    @staticmethod
    def normalize_keypoints(kpts: np.ndarray, h: int, w: int) -> np.ndarray:
        size = np.array([w, h])
        shift = size / 2
        scale = size.max() / 2
        kpts = (kpts - shift) / scale
        return kpts.astype(np.float32)

    @staticmethod
    def post_process(kpts0, kpts1, matches0, scales0, scales1):
        kpts0 = (kpts0 + 0.5) / scales0 - 0.5
        kpts1 = (kpts1 + 0.5) / scales1 - 0.5
        # create match indices
        valid = matches0[0] > -1
        matches = np.stack([np.where(valid)[0], matches0[0][valid]], -1)
        m_kpts0, m_kpts1 = kpts0[0][matches[..., 0]], kpts1[0][matches[..., 1]]
        return m_kpts0, m_kpts1

def read_image(path: str, grayscale: bool = False) -> np.ndarray:
    """Read an image from path as RGB or grayscale"""
    mode = cv2.IMREAD_GRAYSCALE if grayscale else cv2.IMREAD_COLOR
    image = cv2.imread(path, mode)
    if image is None:
        raise IOError(f"Could not read image at {path}.")
    if not grayscale:
        image = image[..., ::-1]
    return image


def normalize_image(image: np.ndarray) -> np.ndarray:
    """Normalize the image tensor and reorder the dimensions."""
    if image.ndim == 3:
        image = image.transpose((2, 0, 1))  # HxWxC to CxHxW
    elif image.ndim == 2:
        image = image[None]  # add channel axis
    else:
        raise ValueError(f"Not an image: {image.shape}")
    return image / 255.0


def resize_image(
    image: np.ndarray,
    size: Union[List[int], int],
    fn: str,
    interp: Optional[str] = "area",
) -> np.ndarray:
    """Resize an image to a fixed size, or according to max or min edge."""
    h, w = image.shape[:2]

    fn = {"max": max, "min": min}[fn]
    if isinstance(size, int):
        scale = size / fn(h, w)
        h_new, w_new = int(round(h * scale)), int(round(w * scale))
        scale = (w_new / w, h_new / h)
    elif isinstance(size, (tuple, list)):
        h_new, w_new = size
        scale = (w_new / w, h_new / h)
    else:
        raise ValueError(f"Incorrect new size: {size}")
    mode = {
        "linear": cv2.INTER_LINEAR,
        "cubic": cv2.INTER_CUBIC,
        "nearest": cv2.INTER_NEAREST,
        "area": cv2.INTER_AREA,
    }[interp]
    return cv2.resize(image, (w_new, h_new), interpolation=mode), scale


def load(
    path: str,
    grayscale: bool = False,
    resize: int = None,
    fn: str = "max",
    interp: str = "area",
):
    img = read_image(path, grayscale=grayscale)
    scales = [1, 1]
    if resize is not None:
        img, scales = resize_image(img, resize, fn=fn, interp=interp)
    return normalize_image(img)[None].astype(np.float32), np.asarray(scales)


def rgb_to_grayscale(image: np.ndarray) -> np.ndarray:
    """Convert an RGB image to grayscale."""
    scale = np.array([0.299, 0.587, 0.114], dtype=image.dtype).reshape(3, 1, 1)
    image = (image * scale).sum(axis=-3, keepdims=True)
    return image

def plot_images(imgs, titles=None, cmaps='gray', dpi=100, pad=.5,
                adaptive=True):
    """Plot a set of images horizontally.
    Args:
        imgs: a list of NumPy or PyTorch images, RGB (H, W, 3) or mono (H, W).
        titles: a list of strings, as titles for each image.
        cmaps: colormaps for monochrome images.
        adaptive: whether the figure size should fit the image aspect ratios.
    """
    n = len(imgs)
    if not isinstance(cmaps, (list, tuple)):
        cmaps = [cmaps] * n

    if adaptive:
        ratios = [i.shape[1] / i.shape[0] for i in imgs]  # W / H
    else:
        ratios = [4/3] * n
    figsize = [sum(ratios)*4.5, 4.5]
    fig, ax = plt.subplots(
        1, n, figsize=figsize, dpi=dpi, gridspec_kw={'width_ratios': ratios})
    if n == 1:
        ax = [ax]
    for i in range(n):
        ax[i].imshow(imgs[i], cmap=plt.get_cmap(cmaps[i]))
        ax[i].get_yaxis().set_ticks([])
        ax[i].get_xaxis().set_ticks([])
        ax[i].set_axis_off()
        for spine in ax[i].spines.values():  # remove frame
            spine.set_visible(False)
        if titles:
            ax[i].set_title(titles[i])
    fig.tight_layout(pad=pad)


def plot_keypoints(kpts, colors='lime', ps=4, axes=None, a=1.0):
    """Plot keypoints for existing images.
    Args:
        kpts: list of ndarrays of size (N, 2).
        colors: string, or list of list of tuples (one for each keypoints).
        ps: size of the keypoints as float.
    """
    if not isinstance(colors, list):
        colors = [colors] * len(kpts)
    if not isinstance(a, list):
        a = [a] * len(kpts)
    if axes is None:
        axes = plt.gcf().axes
    for ax, k, c, alpha in zip(axes, kpts, colors, a):
        ax.scatter(k[:, 0], k[:, 1], c=c, s=ps, linewidths=0, alpha=alpha)


def plot_matches(kpts0, kpts1, color=None, lw=1.5, ps=4, a=1., labels=None,
                 axes=None):
    """Plot matches for a pair of existing images.
    Args:
        kpts0, kpts1: corresponding keypoints of size (N, 2).
        color: color of each match, string or RGB tuple. Random if not given.
        lw: width of the lines.
        ps: size of the end points (no endpoint if ps=0)
        indices: indices of the images to draw the matches on.
        a: alpha opacity of the match lines.
    """
    fig = plt.gcf()
    if axes is None:
        ax = fig.axes
        ax0, ax1 = ax[0], ax[1]
    else:
        ax0, ax1 = axes

    assert len(kpts0) == len(kpts1)
    if color is None:
        color = matplotlib.cm.hsv(np.random.rand(len(kpts0))).tolist()
    elif len(color) > 0 and not isinstance(color[0], (tuple, list)):
        color = [color] * len(kpts0)

    if lw > 0:
        for i in range(len(kpts0)):
            l = matplotlib.patches.ConnectionPatch(
                xyA=(kpts0[i, 0], kpts0[i, 1]), xyB=(kpts1[i, 0], kpts1[i, 1]),
                coordsA=ax0.transData, coordsB=ax1.transData,
                axesA=ax0, axesB=ax1,
                zorder=1, color=color[i], linewidth=lw, clip_on=True,
                alpha=a, label=None if labels is None else labels[i],
                picker=5.0)
            l.set_annotation_clip(True)
            fig.add_artist(l)

    # freeze the axes to prevent the transform to change
    ax0.autoscale(enable=False)
    ax1.autoscale(enable=False)

    if ps > 0:
        ax0.scatter(kpts0[:, 0], kpts0[:, 1], c=color, s=ps)
        ax1.scatter(kpts1[:, 0], kpts1[:, 1], c=color, s=ps)

def save_plot(path, **kw):
    """Save the current figure without any white margin."""
    plt.savefig(path, bbox_inches='tight', pad_inches=0, **kw)
