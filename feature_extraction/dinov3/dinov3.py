import math
import os
import sys
import time
from logging import getLogger

import ailia
import cv2
import numpy as np
from PIL import Image as PILImage

# import original modules
sys.path.append("../../util")
from arg_utils import get_base_parser, get_savepath, update_parser  # noqa: E402
from image_utils import imread  # noqa: E402
from math_utils import sigmoid  # noqa: E402
from model_utils import check_and_download_models  # noqa: E402
from webcamera_utils import get_capture, get_writer  # noqa: E402

logger = getLogger(__name__)


# ======================
# Parameters
# ======================

VITS16_WEIGHT_PATH = "dinov3_vits16.onnx"
VITS16_MODEL_PATH = "dinov3_vits16.onnx.prototxt"
VITB16_WEIGHT_PATH = "dinov3_vitb16.onnx"
VITB16_MODEL_PATH = "dinov3_vitb16.onnx.prototxt"
VITL16_WEIGHT_PATH = "dinov3_vitl16.onnx"
VITL16_MODEL_PATH = "dinov3_vitl16.onnx.prototxt"
VITH16PLUS_WEIGHT_PATH = "dinov3_vith16plus.onnx"
VITH16PLUS_MODEL_PATH = "dinov3_vith16plus.onnx.prototxt"
CONVNEXT_TINY_WEIGHT_PATH = "dinov3_convnext_tiny.onnx"
CONVNEXT_TINY_MODEL_PATH = "dinov3_convnext_tiny.onnx.prototxt"

REMOTE_PATH = "https://storage.googleapis.com/ailia-models/dinov3/"

MODEL_PARAMS = {
    "vits16": {
        "weight": VITS16_WEIGHT_PATH,
        "model": VITS16_MODEL_PATH,
        "num_prefix_tokens": 5,  # cls token + 4 register tokens
        "patch_size": 16,
    },
    "vitb16": {
        "weight": VITB16_WEIGHT_PATH,
        "model": VITB16_MODEL_PATH,
        "num_prefix_tokens": 5,
        "patch_size": 16,
    },
    "vitl16": {
        "weight": VITL16_WEIGHT_PATH,
        "model": VITL16_MODEL_PATH,
        "num_prefix_tokens": 5,
        "patch_size": 16,
    },
    "vith16plus": {
        "weight": VITH16PLUS_WEIGHT_PATH,
        "model": VITH16PLUS_MODEL_PATH,
        "num_prefix_tokens": 5,
        "patch_size": 16,
    },
    "convnext_tiny": {
        "weight": CONVNEXT_TINY_WEIGHT_PATH,
        "model": CONVNEXT_TINY_MODEL_PATH,
        "num_prefix_tokens": 1,  # cls token only, no register tokens
        "patch_size": 32,  # total stride
    },
}

IMAGE_PATH = "coco_000000039769.jpg"
SAVE_IMAGE_PATH = "output.png"

# 224 (pretraining default) gives only a 14x14 / 7x7 patch grid — too coarse for dense feature viz
DEFAULT_RESOLUTION = 896

PCA_IMAGE_SIZE = 768  # height for aspect-ratio-preserving resize in PCA / matching mode

# Tracking mode parameters (matching segmentation_tracking.ipynb)
TRACKING_NEIGHBORHOOD_SIZE = 12  # circle radius in patch units
TRACKING_NEIGHBORHOOD_SHAPE = "circle"
TRACKING_TOPK = 5
TRACKING_TEMPERATURE = 0.2
TRACKING_MAX_CONTEXT = 7  # rolling queue length (first frame always added on top)

# tab10 palette in BGR for tracking overlay (class 1, 2, 3, ...)
TRACKING_COLORS_BGR = [
    (180, 119, 31),  # tab:blue   #1f77b4
    (14, 127, 255),  # tab:orange #ff7f0e
    (44, 160, 44),  # tab:green  #2ca02c
    (40, 39, 214),  # tab:red    #d62728
    (189, 103, 148),  # tab:purple #9467bd
    (75, 86, 140),  # tab:brown  #8c564b
    (194, 119, 227),  # tab:pink   #e377c2
    (127, 127, 127),  # tab:gray   #7f7f7f
    (34, 189, 188),  # tab:olive  #bcbd22
    (207, 190, 23),  # tab:cyan   #17becf
]

MATCHING_DIST_THRESHOLD_SQ = (
    PCA_IMAGE_SIZE * 0.047
) ** 2  # ~4.7% of preprocessed height suppression radius

# False: equal-height panels (default, both images at PCA_IMAGE_SIZE rows)
# True:  equal-width panels  (matches matplotlib subplot behaviour — narrower image appears taller)
SPARSE_EQUAL_WIDTH = False

IMAGE_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGE_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


# ======================
# Argument Parser Config
# ======================

parser = get_base_parser("DINOv3", IMAGE_PATH, SAVE_IMAGE_PATH)
parser.add_argument(
    "-m",
    "--model_type",
    default="convnext_tiny",
    choices=list(MODEL_PARAMS.keys()),
    help="model type",
)
parser.add_argument(
    "--resolution",
    type=int,
    default=DEFAULT_RESOLUTION,
    help="inference resolution (height=width). the model has no fixed input "
    "size, so any value works; higher gives a denser patch grid.",
)
parser.add_argument(
    "--mode",
    default="similarity",
    choices=["pca", "similarity", "tracking", "matching"],
    help=(
        "output mode: "
        "pca=PCA patch features (rainbow, requires --mask for foreground-only), "
        "similarity=patch similarity map, "
        "tracking=segmentation tracking (video), "
        "matching=dense+sparse cross-image correspondences (requires --image2)"
    ),
)
parser.add_argument(
    "--point",
    type=int,
    nargs=2,
    action="append",
    default=None,
    metavar=("X", "Y"),
    help=(
        "reference point (x y, in the original image's pixel coordinates). "
        "mode 3: builds a patch similarity map; can be repeated for multiple panels "
        "(_0, _1, ... suffix). "
        "mode 4: center of the initial circular mask when --mask is not given. "
        "defaults to image center."
    ),
)
parser.add_argument(
    "--mask",
    type=str,
    default=None,
    metavar="PATH",
    help="foreground mask image (grayscale, >127=foreground). "
    "pca: applies PCA to foreground patches only, background is black. "
    "tracking: initial mask for frame 0.",
)
parser.add_argument(
    "--image2",
    type=str,
    default=None,
    metavar="PATH",
    help="right image path (matching mode only)",
)
parser.add_argument(
    "--mask2",
    type=str,
    default=None,
    metavar="PATH",
    help="right image foreground mask (grayscale, >127=foreground; matching mode only)",
)
parser.add_argument("--onnx", action="store_true", help="execute onnxruntime version.")
args = update_parser(parser)


# ======================
# Classes
# ======================


class FrameDirCapture:
    """cv2.VideoCapture-compatible wrapper for a directory of image files."""

    _EXTS = ("*.jpg", "*.jpeg", "*.png", "*.bmp", "*.JPG", "*.JPEG", "*.PNG", "*.BMP")

    def __init__(self, dir_path):
        import glob

        files = []
        for ext in self._EXTS:
            files.extend(glob.glob(os.path.join(dir_path, ext)))

        def _sort_key(p):
            stem = os.path.splitext(os.path.basename(p))[0]
            try:
                return (0, int(stem))
            except ValueError:
                return (1, stem)

        self._files = sorted(set(files), key=_sort_key)
        self._idx = 0
        if self._files:
            first = cv2.imread(self._files[0])
            self._h, self._w = (
                (first.shape[0], first.shape[1]) if first is not None else (0, 0)
            )
        else:
            self._h = self._w = 0

    def isOpened(self):
        return len(self._files) > 0

    def read(self):
        if self._idx >= len(self._files):
            return False, None
        frame = cv2.imread(self._files[self._idx])
        self._idx += 1
        return (True, frame) if frame is not None else (False, None)

    def get(self, prop_id):
        if prop_id == cv2.CAP_PROP_FRAME_HEIGHT:
            return float(self._h)
        if prop_id == cv2.CAP_PROP_FRAME_WIDTH:
            return float(self._w)
        return 0.0

    def release(self):
        pass


# ======================
# Utils
# ======================


def load_mask_gray(mask_path):
    img = cv2.imread(mask_path, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise FileNotFoundError(f"Mask file not found: {mask_path}")
    if img.ndim == 3 and img.shape[2] == 4:
        return img[:, :, 3]  # RGBA: alpha channel is the foreground mask
    elif img.ndim == 3:
        return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return img


# ======================
# Rendering
# ======================


def render_panel(similarity_map, point, img_w, img_h):
    # no photo blend: mixing colormap with image produces muddy rainbow, not clean contrast
    # INTER_CUBIC not INTER_NEAREST: official panels show smooth texture, not hard block edges
    heatmap = (similarity_map * 255).clip(0, 255).astype(np.uint8)
    heatmap = cv2.resize(heatmap, (img_w, img_h), interpolation=cv2.INTER_CUBIC)
    panel = cv2.applyColorMap(heatmap, cv2.COLORMAP_VIRIDIS)

    if point is None:
        point_xy = (img_w // 2, img_h // 2)
    else:
        point_xy = (int(point[0]), int(point[1]))
    cv2.drawMarker(
        panel,
        point_xy,
        (0, 0, 255),
        markerType=cv2.MARKER_CROSS,
        markerSize=16,
        thickness=2,
    )

    return panel


def render_pca(patch_tokens, h_patches, w_patches, img_w, img_h, mask_path=None):
    from sklearn.decomposition import PCA as SklearnPCA

    if mask_path is not None:
        mask_img = load_mask_gray(mask_path)
        mask_img = cv2.resize(
            mask_img, (w_patches, h_patches), interpolation=cv2.INTER_NEAREST
        )
        fg = (mask_img > 127).reshape(-1)
    else:
        fg = np.ones(len(patch_tokens), dtype=bool)

    pca = SklearnPCA(n_components=3, whiten=True)
    pca.fit(patch_tokens[fg])
    projected = pca.transform(patch_tokens)  # (N, 3)

    colored = sigmoid(projected * 2.0)  # (N, 3) in [0, 1]
    colored[~fg] = 0.0

    rgb = (colored * 255).clip(0, 255).astype(np.uint8).reshape(h_patches, w_patches, 3)
    bgr = rgb[:, :, ::-1].copy()
    return cv2.resize(bgr, (img_w, img_h), interpolation=cv2.INTER_NEAREST)


def render_dense_matching(col_l, col_r, patch_size):
    def to_bgr_blocks(col, ps):
        img = np.repeat(np.repeat(col, ps, axis=0), ps, axis=1)
        return (img[:, :, ::-1] * 255).clip(0, 255).astype(np.uint8)

    img_l = to_bgr_blocks(col_l, patch_size)
    img_r = to_bgr_blocks(col_r, patch_size)

    max_h = max(img_l.shape[0], img_r.shape[0])
    if img_l.shape[0] < max_h:
        pad = np.zeros((max_h - img_l.shape[0], img_l.shape[1], 3), dtype=np.uint8)
        img_l = np.concatenate([img_l, pad], axis=0)
    if img_r.shape[0] < max_h:
        pad = np.zeros((max_h - img_r.shape[0], img_r.shape[1], 3), dtype=np.uint8)
        img_r = np.concatenate([img_r, pad], axis=0)

    return np.concatenate([img_l, img_r], axis=1)


def render_sparse_matching(
    img_left_bgr,
    img_right_bgr,
    locs_left,
    locs_right,
    indices_keep,
    col_l,
    patch_size,
    w_prep_l,
    w_prep_r,
    h_prep_l,
    h_prep_r,
):
    h_l, w_l = img_left_bgr.shape[:2]
    h_r, w_r = img_right_bgr.shape[:2]

    if SPARSE_EQUAL_WIDTH:
        # Equal-width panels: both images at the same display width.
        # Matches matplotlib subplot(1,2,*) — narrower aspect → taller panel.
        display_w = max(w_prep_l, w_prep_r)
        scale_lx = scale_ly = display_w / w_prep_l
        scale_rx = scale_ry = display_w / w_prep_r
        disp_h_l = max(1, int(round(h_prep_l * scale_ly)))
        disp_h_r = max(1, int(round(h_prep_r * scale_ry)))
        img_l_disp = cv2.resize(img_left_bgr, (display_w, disp_h_l))
        img_r_disp = cv2.resize(img_right_bgr, (display_w, disp_h_r))
        max_h = max(disp_h_l, disp_h_r)
        canvas = np.zeros((max_h, display_w * 2, 3), dtype=np.uint8)
        canvas[:disp_h_l, :display_w] = img_l_disp
        canvas[:disp_h_r, display_w:] = img_r_disp
        pt_offset = display_w
    else:
        # Equal-height panels (default): both images at preprocessed height.
        # Locs are in preprocessed pixel space, so scale == 1.0 and coords map exactly.
        img_l_disp = cv2.resize(img_left_bgr, (w_prep_l, h_prep_l))
        img_r_disp = cv2.resize(img_right_bgr, (w_prep_r, h_prep_r))
        canvas = np.concatenate([img_l_disp, img_r_disp], axis=1)
        scale_lx = scale_ly = 1.0
        scale_rx = scale_ry = 1.0
        pt_offset = w_prep_l

    for i in indices_keep:
        row_l, col_l_px = locs_left[i]
        row_r, col_r_px = locs_right[i]

        pt_l = (int(col_l_px * scale_lx), int(row_l * scale_ly))
        pt_r = (int(col_r_px * scale_rx) + pt_offset, int(row_r * scale_ry))

        pr = int(row_l / patch_size)
        pc = int(col_l_px / patch_size)
        color_rgb = col_l[pr, pc]
        color_bgr = (
            int(color_rgb[2] * 255),
            int(color_rgb[1] * 255),
            int(color_rgb[0] * 255),
        )

        cv2.line(canvas, pt_l, pt_r, color_bgr, thickness=1, lineType=cv2.LINE_AA)

    return canvas


# ======================
# Similarity mode
# ======================


def similarity_map_for_point(patch_tokens, grid_size, point, img_w, img_h):
    if point is None:
        fx, fy = 0.5, 0.5
    else:
        fx, fy = point[0] / img_w, point[1] / img_h

    grid_x = min(max(int(fx * grid_size), 0), grid_size - 1)
    grid_y = min(max(int(fy * grid_size), 0), grid_size - 1)
    ref_index = grid_y * grid_size + grid_x

    # patch_tokens: (num_patches, channels)
    norm = patch_tokens / (np.linalg.norm(patch_tokens, axis=-1, keepdims=True) + 1e-6)
    similarity = (norm @ norm[ref_index]).reshape(grid_size, grid_size)

    # min-max instead of fixed (cos+1)/2: real similarities rarely reach -1,
    # so the fixed mapping leaves the background washed out rather than dark
    sim_min, sim_max = similarity.min(), similarity.max()
    similarity = (similarity - sim_min) / (sim_max - sim_min + 1e-6)

    return similarity, (grid_x, grid_y)


# ======================
# Matching mode
# ======================


def preprocess_mask_patch(mask_path, h_patches, w_patches, patch_size):
    if mask_path is None:
        return np.ones((h_patches, w_patches), dtype=np.float32)
    mask_gray = load_mask_gray(mask_path)
    mask_resized = cv2.resize(
        mask_gray,
        (w_patches * patch_size, h_patches * patch_size),
        interpolation=cv2.INTER_NEAREST,
    )
    arr = mask_resized.astype(np.float32) / 255.0
    return arr.reshape(h_patches, patch_size, w_patches, patch_size).mean(axis=(1, 3))


def get_normalized_patch_features(
    last_hidden_state, h_patches, w_patches, num_prefix_tokens
):
    dim = last_hidden_state.shape[-1]
    patch_tokens = last_hidden_state[0, num_prefix_tokens:]  # (N, dim)
    feat = patch_tokens.reshape(h_patches, w_patches, dim).transpose(
        2, 0, 1
    )  # (dim, H, W)
    norm = np.linalg.norm(feat, axis=0, keepdims=True) + 1e-6
    return feat / norm


def stratify_points(pts_yx, threshold_sq):
    n = len(pts_yx)
    if n == 0:
        return np.array([], dtype=int)
    diff = pts_yx[:, np.newaxis, :] - pts_yx[np.newaxis, :, :]
    dists = (diff**2).sum(axis=-1)
    np.fill_diagonal(dists, threshold_sq + 1.0)
    keep = np.ones(n, dtype=bool)
    while True:
        counts = (dists <= threshold_sq).sum(axis=1)
        counts[~keep] = 0
        if counts.max() == 0:
            break
        idx = int(counts.argmax())
        keep[idx] = False
        dists[idx, :] = threshold_sq + 1.0
        dists[:, idx] = threshold_sq + 1.0
    return np.where(keep)[0]


# ======================
# Tracking mode
# ======================


def preprocess_tracking(img_bgr, short_side, patch_size):
    """ResizeToMultiple preprocessing for tracking mode (BICUBIC, aspect-ratio-preserving).
    Returns (pixel_values [1,3,H,W], h_patches, w_patches)."""
    h, w = img_bgr.shape[:2]
    multiple = patch_size
    if w > h:  # landscape: height is the short side
        new_h = math.ceil(short_side / multiple) * multiple
        new_w = math.ceil(w * new_h / h / multiple) * multiple
    else:
        new_w = math.ceil(short_side / multiple) * multiple
        new_h = math.ceil(h * new_w / w / multiple) * multiple

    pil_img = PILImage.fromarray(img_bgr[:, :, ::-1])  # BGR → RGB
    pil_resized = pil_img.resize((new_w, new_h), PILImage.BICUBIC)

    arr = np.array(pil_resized, dtype=np.float32) / 255.0
    arr = (arr - IMAGE_MEAN) / IMAGE_STD
    pixel_values = arr.transpose(2, 0, 1)[np.newaxis]

    h_patches = new_h // patch_size
    w_patches = new_w // patch_size
    return pixel_values, h_patches, w_patches


def load_mask_multiclass(mask_path):
    """Load mask and remap pixel values to consecutive class indices 0..M-1.

    Returns (mask_indices: np.uint8 [H, W], num_classes: int).
    0 is always background; non-zero unique values are remapped to 1, 2, 3, ...
    """
    pil = PILImage.open(mask_path)
    arr = np.array(pil if pil.mode == "P" else pil.convert("L"), dtype=np.uint8)
    unique_vals = np.unique(arr)
    fg_vals = unique_vals[unique_vals > 0]
    if len(fg_vals) == 0:
        return arr, 2
    remap = np.zeros(256, dtype=np.uint8)
    for i, v in enumerate(fg_vals):
        remap[v] = i + 1
    return remap[arr], int(len(fg_vals)) + 1


def postprocess_probs_tracking(probs_mhw):
    """Per-channel min-max normalization. probs_mhw: (M, H, W) → (M, H, W)."""
    out = probs_mhw.copy()
    for m in range(out.shape[0]):
        vmin, vmax = out[m].min(), out[m].max()
        if vmax > vmin:
            out[m] = (out[m] - vmin) / (vmax - vmin)
        else:
            out[m][:] = 0.0
    return out


def make_neighborhood_mask_np(h, w, size, shape="circle"):
    """Returns [h, w, h, w] bool mask. mask[i,j,u,v] is True when (u,v) is within
    neighborhood of (i,j)."""
    rows = np.arange(h, dtype=np.float32)
    cols = np.arange(w, dtype=np.float32)
    ii, jj = np.meshgrid(rows, cols, indexing="ij")
    ij = np.stack([ii, jj], axis=-1)  # [h, w, 2]
    diff = ij[:, :, np.newaxis, np.newaxis, :] - ij[np.newaxis, np.newaxis, :, :, :]
    if shape == "circle":
        dist = np.linalg.norm(diff, axis=-1)
    else:
        dist = np.abs(diff).max(axis=-1)
    return dist <= size


def propagate_mask(
    F_curr,
    context_features,
    context_probs,
    neighborhood_mask,
    topk=TRACKING_TOPK,
    temperature=TRACKING_TEMPERATURE,
):
    """
    Propagates segmentation probabilities from context frames to the current frame.

    F_curr:            (h, w, D) – L2-normalized current-frame patch features
    context_features:  list of t arrays, each (h, w, D)
    context_probs:     list of t arrays, each (h, w, M)
    neighborhood_mask: (h, w, h, w) bool
    Returns:           (h, w, M)
    """
    h, w, D = F_curr.shape
    t = len(context_features)
    M = context_probs[0].shape[-1]

    ctx_feat_arr = np.stack(context_features, axis=0)  # (t, h, w, D)
    ctx_prob_arr = np.stack(context_probs, axis=0)  # (t, h, w, M)

    curr_flat = F_curr.reshape(h * w, D)
    ctx_flat = ctx_feat_arr.reshape(t * h * w, D)
    ctx_prob_flat = ctx_prob_arr.reshape(t * h * w, M)

    dot = curr_flat @ ctx_flat.T  # (hw, thw)

    nbr_flat = neighborhood_mask.reshape(h * w, h * w)  # (hw, hw)
    nbr_broadcast = np.tile(nbr_flat, (1, t))  # (hw, thw)
    dot[~nbr_broadcast] = -np.inf

    kth = np.partition(dot, kth=-topk, axis=-1)[:, -topk]
    dot = np.where(dot >= kth[:, np.newaxis], dot, -np.inf)

    row_max = np.where(np.isfinite(dot), dot, -np.inf).max(axis=-1, keepdims=True)
    row_max = np.where(np.isinf(row_max), 0.0, row_max)
    exp_dot = np.exp((dot - row_max) / temperature)
    exp_dot[~np.isfinite(dot)] = 0.0
    weights = exp_dot / (exp_dot.sum(axis=-1, keepdims=True) + 1e-10)

    probs = weights @ ctx_prob_flat  # (hw, M)
    probs /= probs.sum(axis=-1, keepdims=True) + 1e-10

    return probs.reshape(h, w, M)


# ======================
# Main functions
# ======================


def preprocess(img, image_size):
    img = img[:, :, ::-1]  # BGR -> RGB

    # rescale before resize: keeps bilinear interpolation in float precision (not uint8)
    img = img.astype(np.float32) / 255.0
    img = cv2.resize(img, (image_size, image_size), interpolation=cv2.INTER_LINEAR)
    img = (img - IMAGE_MEAN) / IMAGE_STD

    img = img.transpose(2, 0, 1)  # HWC -> CHW
    img = np.expand_dims(img, axis=0).astype(np.float32)

    return img


def preprocess_pca(img, patch_size):
    h, w = img.shape[:2]
    h_patches = PCA_IMAGE_SIZE // patch_size
    w_patches = int((w * PCA_IMAGE_SIZE) / (h * patch_size))
    new_h = h_patches * patch_size
    new_w = w_patches * patch_size

    # PIL BILINEAR in uint8 space matches TF.resize(PIL_image, antialias=True) exactly
    pil_img = PILImage.fromarray(img[:, :, ::-1])  # BGR -> RGB uint8
    img_resized = (
        np.array(pil_img.resize((new_w, new_h), PILImage.BILINEAR), dtype=np.float32)
        / 255.0
    )
    img_norm = (img_resized - IMAGE_MEAN) / IMAGE_STD
    pixel_values = img_norm.transpose(2, 0, 1)[np.newaxis].astype(np.float32)
    return pixel_values, h_patches, w_patches


def predict(net, img, image_size):
    pixel_values = preprocess(img, image_size)

    if not args.onnx:
        output = net.predict([pixel_values])
    else:
        output = net.run(None, {"pixel_values": pixel_values})
    last_hidden_state, pooler_output = output

    return last_hidden_state, pooler_output


def recognize_from_image(net):
    params = MODEL_PARAMS[args.model_type]
    num_prefix_tokens = params["num_prefix_tokens"]
    image_size = args.resolution
    grid_size = image_size // params["patch_size"]

    for image_path in args.input:
        logger.info(image_path)

        img = imread(image_path)
        img_h, img_w = img.shape[:2]

        logger.info("Start inference...")
        if args.benchmark:
            logger.info("BENCHMARK mode")
            total_time_estimation = 0
            for i in range(args.benchmark_count):
                start = int(round(time.time() * 1000))
                last_hidden_state, pooler_output = predict(net, img, image_size)
                end = int(round(time.time() * 1000))
                estimation_time = end - start

                logger.info(f"\tailia processing estimation time {estimation_time} ms")
                if i != 0:
                    total_time_estimation = total_time_estimation + estimation_time

            logger.info(
                f"\taverage time estimation {total_time_estimation / (args.benchmark_count - 1)} ms"
            )

        savepath = get_savepath(args.savepath, image_path)

        last_hidden_state, pooler_output = predict(net, img, image_size)

        logger.info(f"model_type: {args.model_type}")
        logger.info(f"last_hidden_state: shape={last_hidden_state.shape}")
        logger.info(
            f"pooler_output: shape={pooler_output.shape}, "
            f"norm={np.linalg.norm(pooler_output):.4f}"
        )

        patch_tokens = last_hidden_state[0, num_prefix_tokens:, :]
        points = args.point if args.point else [None]
        if len(points) > 1:
            base, ext = os.path.splitext(savepath)

        for i, point in enumerate(points):
            similarity_map, ref_grid = similarity_map_for_point(
                patch_tokens, grid_size, point, img_w, img_h
            )
            logger.info(f"reference patch (grid coords): {ref_grid}")

            panel = render_panel(similarity_map, point, img_w, img_h)

            out_path = f"{base}_{i}{ext}" if len(points) > 1 else savepath
            logger.info(f"saved at : {out_path}")
            cv2.imwrite(out_path, panel)

    logger.info("Script finished successfully.")


def recognize_from_image_pca(net):
    params = MODEL_PARAMS[args.model_type]
    num_prefix_tokens = params["num_prefix_tokens"]

    for image_path in args.input:
        logger.info(image_path)

        img = imread(image_path)
        img_h, img_w = img.shape[:2]

        logger.info("Start inference...")

        pixel_values, h_patches, w_patches = preprocess_pca(img, params["patch_size"])
        if not args.onnx:
            output = net.predict([pixel_values])
        else:
            output = net.run(None, {"pixel_values": pixel_values})
        last_hidden_state, pooler_output = output
        patch_tokens = last_hidden_state[0, num_prefix_tokens:, :]
        panel = render_pca(
            patch_tokens, h_patches, w_patches, img_w, img_h, mask_path=args.mask
        )

        savepath = get_savepath(args.savepath, image_path)
        logger.info(f"saved at : {savepath}")
        cv2.imwrite(savepath, panel)

    logger.info("Script finished successfully.")


def recognize_from_video(net):
    params = MODEL_PARAMS[args.model_type]
    num_prefix_tokens = params["num_prefix_tokens"]
    patch_size = params["patch_size"]
    short_side = args.resolution  # used as ResizeToMultiple short_side

    video_file = args.video if args.video else args.input[0]
    if os.path.isdir(video_file):
        capture = FrameDirCapture(video_file)
    else:
        capture = get_capture(video_file)
    assert capture.isOpened(), "Cannot capture source"

    if args.savepath != SAVE_IMAGE_PATH:
        f_h = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        f_w = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        writer = get_writer(args.savepath, f_h, f_w * 2)  # side-by-side
    else:
        writer = None

    first_feats = None
    first_probs = None
    num_classes = None
    features_queue = []
    probs_queue = []
    neighborhood_mask = None
    h_patches = None
    w_patches = None

    has_display = bool(os.environ.get("DISPLAY"))
    frame_idx = 0
    frame_shown = False

    while True:
        ret, frame = capture.read()
        if not ret:
            break
        if has_display:
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            if (
                frame_shown
                and cv2.getWindowProperty("frame", cv2.WND_PROP_VISIBLE) == 0
            ):
                break

        img_h, img_w = frame.shape[:2]

        # ResizeToMultiple preprocessing (BICUBIC, aspect-ratio-preserving)
        pixel_values, h_p, w_p = preprocess_tracking(frame, short_side, patch_size)

        if not args.onnx:
            output = net.predict([pixel_values])
        else:
            output = net.run(None, {"pixel_values": pixel_values})
        last_hidden_state, _ = output

        patch_tokens = last_hidden_state[0, num_prefix_tokens:]  # (h*w, D)
        dim = patch_tokens.shape[-1]
        F_curr = patch_tokens.reshape(h_p, w_p, dim)
        F_curr = F_curr / (np.linalg.norm(F_curr, axis=-1, keepdims=True) + 1e-6)

        if frame_idx == 0:
            h_patches, w_patches = h_p, w_p
            logger.info(
                f"Feature grid: {h_patches}x{w_patches} (short_side={short_side})"
            )

            neighborhood_mask = make_neighborhood_mask_np(
                h_patches,
                w_patches,
                TRACKING_NEIGHBORHOOD_SIZE,
                TRACKING_NEIGHBORHOOD_SHAPE,
            )

            if args.mask:
                mask_indices, num_classes = load_mask_multiclass(args.mask)
                mask_resized = cv2.resize(
                    mask_indices,
                    (w_patches, h_patches),
                    interpolation=cv2.INTER_NEAREST,
                )
                current_probs = np.eye(num_classes, dtype=np.float32)[
                    mask_resized
                ]  # (h, w, M)
            else:
                num_classes = 2
                point = args.point[0] if args.point else None
                px = point[0] if point is not None else img_w // 2
                py = point[1] if point is not None else img_h // 2
                gx = min(max(int(px / img_w * w_patches), 0), w_patches - 1)
                gy = min(max(int(py / img_h * h_patches), 0), h_patches - 1)
                radius = max(1, min(h_patches, w_patches) // 8)
                gy_idx, gx_idx = np.ogrid[:h_patches, :w_patches]
                fg = ((gx_idx - gx) ** 2 + (gy_idx - gy) ** 2 <= radius**2).astype(
                    np.float32
                )
                current_probs = np.stack([1.0 - fg, fg], axis=-1)  # (h, w, 2)

            first_feats = F_curr.copy()
            first_probs = current_probs.copy()

            left = frame
            probs_hw_0 = current_probs.transpose(2, 0, 1)
            probs_full_0 = np.stack(
                [
                    cv2.resize(
                        probs_hw_0[m], (img_w, img_h), interpolation=cv2.INTER_NEAREST
                    )
                    for m in range(num_classes)
                ],
                axis=0,
            )
            probs_full_0 = postprocess_probs_tracking(probs_full_0)
            src_mask = probs_full_0.argmax(axis=0).astype(np.uint8)
            right = np.zeros((img_h, img_w, 3), dtype=np.uint8)
            for m in range(1, num_classes):
                right[src_mask == m] = TRACKING_COLORS_BGR[
                    (m - 1) % len(TRACKING_COLORS_BGR)
                ]
        else:
            ctx_feats = [first_feats] + features_queue
            ctx_probs = [first_probs] + probs_queue

            current_probs = propagate_mask(
                F_curr, ctx_feats, ctx_probs, neighborhood_mask
            )

            features_queue.append(F_curr.copy())
            probs_queue.append(current_probs.copy())
            if len(features_queue) > TRACKING_MAX_CONTEXT:
                features_queue.pop(0)
            if len(probs_queue) > TRACKING_MAX_CONTEXT:
                probs_queue.pop(0)

            probs_hw = current_probs.transpose(2, 0, 1)
            probs_full = np.stack(
                [
                    cv2.resize(
                        probs_hw[m], (img_w, img_h), interpolation=cv2.INTER_NEAREST
                    )
                    for m in range(num_classes)
                ],
                axis=0,
            )
            probs_full = postprocess_probs_tracking(probs_full)
            pred = probs_full.argmax(axis=0).astype(np.uint8)

            left = frame
            right = np.zeros((img_h, img_w, 3), dtype=np.uint8)
            for m in range(1, num_classes):
                right[pred == m] = TRACKING_COLORS_BGR[
                    (m - 1) % len(TRACKING_COLORS_BGR)
                ]

        panel = np.concatenate([left, right], axis=1)

        if has_display:
            cv2.imshow("frame", panel)
            frame_shown = True

        if writer is not None:
            writer.write(panel)

        frame_idx += 1

    capture.release()
    if has_display:
        cv2.destroyAllWindows()
    if writer is not None:
        writer.release()

    logger.info("Script finished successfully.")


def recognize_from_image_matching(net):
    from sklearn.decomposition import PCA as SklearnPCA

    if args.image2 is None:
        raise ValueError("--image2 is required for matching mode")

    params = MODEL_PARAMS[args.model_type]
    num_prefix_tokens = params["num_prefix_tokens"]
    patch_size = params["patch_size"]

    img_l = imread(args.input[0])
    img_r = imread(args.image2)

    pv_l, h_l, w_l = preprocess_pca(img_l, patch_size)
    pv_r, h_r, w_r = preprocess_pca(img_r, patch_size)

    logger.info("Running inference on left image...")
    if not args.onnx:
        lhs_l, _ = net.predict([pv_l])
    else:
        lhs_l, _ = net.run(None, {"pixel_values": pv_l})

    logger.info("Running inference on right image...")
    if not args.onnx:
        lhs_r, _ = net.predict([pv_r])
    else:
        lhs_r, _ = net.run(None, {"pixel_values": pv_r})

    feat_l = get_normalized_patch_features(lhs_l, h_l, w_l, num_prefix_tokens)
    feat_r = get_normalized_patch_features(lhs_r, h_r, w_r, num_prefix_tokens)

    mask_l = preprocess_mask_patch(args.mask, h_l, w_l, patch_size)
    mask_r = preprocess_mask_patch(args.mask2, h_r, w_r, patch_size)

    FG = 0.5
    dim = feat_l.shape[0]
    x_l = feat_l.reshape(dim, -1).T  # (N_l, dim)
    x_r = feat_r.reshape(dim, -1).T  # (N_r, dim)

    # ---- Dense: PCA on left foreground patches, transform both ----
    fg_l_mask = mask_l.reshape(-1) > FG
    pca = SklearnPCA(n_components=3, whiten=True)
    pca.fit(x_l[fg_l_mask])

    def project_and_color(x, mask):
        proj = pca.transform(x).reshape(mask.shape[0], mask.shape[1], 3)
        col = sigmoid(proj * 2.0)
        col[mask <= FG] = 0.0
        return col

    col_l = project_and_color(x_l, mask_l)
    col_r = project_and_color(x_r, mask_r)

    dense_panel = render_dense_matching(col_l, col_r, patch_size)

    # ---- Sparse: cosine-sim argmax, foreground filter, stratify ----
    N_l = h_l * w_l
    rows_l = np.arange(N_l) // w_l
    cols_l = np.arange(N_l) % w_l
    locs_l = (np.stack([rows_l, cols_l], axis=1).astype(np.float32) + 0.5) * patch_size

    heatmaps = x_l @ feat_r.reshape(dim, -1)  # (N_l, N_r)
    best_r = heatmaps.argmax(axis=1)
    rows_r = best_r // w_r
    cols_r = best_r % w_r
    locs_r = (np.stack([rows_r, cols_r], axis=1).astype(np.float32) + 0.5) * patch_size

    fg_r_for_match = mask_r.reshape(-1)[best_r] > FG
    fg_sel = (mask_l.reshape(-1) > FG) & fg_r_for_match
    locs_l_fg = locs_l[fg_sel]
    locs_r_fg = locs_r[fg_sel]

    h_prep_l = h_l * patch_size
    h_prep_r = h_r * patch_size

    # Stratify in original image space (scale converts preprocessed locs to original pixel coords)
    indices_keep = stratify_points(locs_l_fg, MATCHING_DIST_THRESHOLD_SQ)
    logger.info(
        f"Sparse correspondences: {len(locs_l_fg)} fg matches → {len(indices_keep)} after stratification"
    )

    sparse_panel = render_sparse_matching(
        img_l,
        img_r,
        locs_l_fg,
        locs_r_fg,
        indices_keep,
        col_l,
        patch_size,
        w_prep_l=w_l * patch_size,
        w_prep_r=w_r * patch_size,
        h_prep_l=h_prep_l,
        h_prep_r=h_prep_r,
    )

    # ---- Save ----
    savepath = get_savepath(args.savepath, args.input[0])
    base, ext = os.path.splitext(savepath)
    dense_path = savepath
    sparse_path = f"{base}_sparse{ext}"

    cv2.imwrite(dense_path, dense_panel)
    logger.info(f"Dense matching saved: {dense_path}")
    cv2.imwrite(sparse_path, sparse_panel)
    logger.info(f"Sparse matching saved: {sparse_path}")


def main():
    params = MODEL_PARAMS[args.model_type]
    weight_path, model_path = params["weight"], params["model"]

    # model files check and download
    check_and_download_models(weight_path, model_path, REMOTE_PATH)

    env_id = args.env_id

    # initialize
    if not args.onnx:
        net = ailia.Net(model_path, weight_path, env_id=env_id)
    else:
        import onnxruntime

        providers = ["CPUExecutionProvider", "CUDAExecutionProvider"]
        net = onnxruntime.InferenceSession(weight_path, providers=providers)

    if args.mode == "tracking":
        recognize_from_video(net)
    elif args.mode == "matching":
        recognize_from_image_matching(net)
    elif args.mode == "pca":
        recognize_from_image_pca(net)
    else:
        recognize_from_image(net)


if __name__ == "__main__":
    main()
