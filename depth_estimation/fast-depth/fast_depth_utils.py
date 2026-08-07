import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

import transforms.transforms as transforms

IHEIGHT, IWIDTH = 480, 640  # raw image size

cmap = plt.cm.viridis


def colored_depthmap(depth, d_min=None, d_max=None):
    if d_min is None:
        d_min = np.min(depth)
    if d_max is None:
        d_max = np.max(depth)
    depth_relative = (depth - d_min) / (d_max - d_min)
    return 255 * cmap(depth_relative)[:, :, :3]  # H, W, C


def save_image(img_merge, filename):
    img_merge = Image.fromarray(img_merge.astype("uint8"))
    img_merge.save(filename)


def transform(rgb, output_size):
    transformer = transforms.Compose(
        [
            transforms.Resize((int(IWIDTH * (250.0 / IHEIGHT)), 250)),
            transforms.CenterCrop((228, 304)),
            transforms.Resize(output_size),
        ]
    )
    rgb_np = transformer(rgb)
    rgb_np = np.asarray(rgb_np, dtype="float") / 255

    return rgb_np
