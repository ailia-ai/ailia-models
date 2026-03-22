import sys
import time
from functools import partial
from logging import getLogger

import ailia
import cv2
import numpy as np

# import original modules
sys.path.append("../../util")
from arg_utils import get_base_parser, get_savepath, update_parser
from detector_utils import load_image
from model_utils import check_and_download_models
from resize_utils import tv_resize

logger = getLogger(__name__)


# ======================
# Parameters
# ======================

WEIGHT_MOGE = "moge.onnx"
WEIGHT_SS_COND = "ss_condition_embedder.onnx"
MODEL_MOGE = WEIGHT_MOGE + ".prototxt"
MODEL_SS_COND = WEIGHT_SS_COND + ".prototxt"
REMOTE_PATH = "https://storage.googleapis.com/ailia-models/sam-3d-objects/"

IMAGE_PATH = "input.png"
MASK_PATH = "mask.png"
SAVE_PATH = "output.ply"

IMAGE_SIZE = 518  # DINOv2 input resolution


# ======================
# Argument Parser Config
# ======================

parser = get_base_parser(
    "SAM 3D Objects — single image to 3D Gaussian Splatting", IMAGE_PATH, SAVE_PATH
)
parser.add_argument(
    "--mask",
    type=str,
    default=MASK_PATH,
    help="Path to a separate mask image (white=object).",
)
parser.add_argument(
    "--seed",
    type=int,
    default=42,
    help="Random seed for reproducible sampling.",
)
parser.add_argument(
    "--onnx",
    action="store_true",
    help="Execute onnxruntime version.",
)
args = update_parser(parser)


# ======================
# Secondary Functions
# ======================


def load_mask(path):
    """Load mask image, handling both single-channel and multi-channel inputs."""
    mask = load_image(path) > 0
    return mask[..., -1] if mask.ndim == 3 else mask


def _crop(arr, top, left, height, width):
    """Numpy equivalent of torchvision.transforms.functional.crop."""
    return arr[top : top + height, left : left + width]


def _pad_square(arr, fill=0.0):
    """Centre-pad [H,W,...] to square."""
    H, W = arr.shape[:2]
    M = max(H, W)
    ph, pw = (M - H) // 2, (M - W) // 2
    ph2, pw2 = M - H - ph, M - W - pw
    pad_spec = (
        ((ph, ph2), (pw, pw2)) if arr.ndim == 2 else ((ph, ph2), (pw, pw2), (0, 0))
    )
    return np.pad(arr, pad_spec, constant_values=fill)


def _resize(arr, size, bilinear=True):
    interp = cv2.INTER_LINEAR if bilinear else cv2.INTER_NEAREST
    return tv_resize(arr.astype(np.float32), (size, size), interpolation=interp)


def _normalize_pointmap(pointmap, mask, shift=None, scale=None):
    """
    ObjectCentricSSI normalisation (use_scene_scale=True).
    pointmap : [H, W, 3]  float32  (may contain NaN)
    mask     : [H, W]     float32  binary [0, 1]
    Returns  : (normalised pointmap, shift[3], scale[3])
    """
    h, w = pointmap.shape[:2]
    mask_resized = cv2.resize(
        mask.astype(np.float32), (w, h), interpolation=cv2.INTER_NEAREST
    )

    flat = pointmap.reshape(-1, 3).T  # [3, N]
    mask_flat = mask_resized.reshape(-1) > 0.5

    mask_pts = flat[:, mask_flat]  # [3, N_mask]
    valid = np.isfinite(mask_pts).all(axis=0)
    mask_pts = mask_pts[:, valid]

    if mask_pts.shape[1] == 0:
        default_shift = np.zeros(3, dtype=np.float32)
        default_scale = np.ones(3, dtype=np.float32)
        return pointmap, default_shift, default_scale

    _shift = np.nanmedian(mask_pts, axis=1)  # [3]
    centered = flat - _shift[:, None]  # [3, N]
    max_dims = np.nanmax(np.abs(centered), axis=0)  # [N]
    _scale_scalar = float(np.nanmedian(max_dims))
    _scale = np.full(3, _scale_scalar, dtype=np.float32)

    shift = np.asarray(
        shift if shift is not None else _shift, dtype=np.float32
    ).reshape(3)
    scale = np.asarray(
        scale if scale is not None else _scale, dtype=np.float32
    ).reshape(3)

    if np.any(scale == 0.0) or not np.all(np.isfinite(scale)):
        return pointmap, _shift, _scale

    normalized = (pointmap - shift) / scale
    return normalized.astype(np.float32), shift, scale


def _compute_mask_bbox(mask, box_size_factor=1.2):
    """Tight bounding box around mask with factor expansion."""
    indices = np.argwhere(mask > 0)
    if len(indices) == 0:
        # Handle empty mask case
        return (0, 0, 0, 0)
    y_min, x_min = indices.min(0)
    y_max, x_max = indices.max(0)
    cx, cy = (x_min + x_max) / 2.0, (y_min + y_max) / 2.0
    size = int(max(x_max - x_min, y_max - y_min, 2) * box_size_factor)
    return (
        int(cx - size // 2),
        int(cy - size // 2),
        int(cx + size // 2),
        int(cy + size // 2),
    )


def _nearest_downsample(points, downsample_size):
    """
    points: np.ndarray, shape = (1, H, W, C)
    downsample_size: (out_h, out_w)

    return:
        np.ndarray, shape = (1, out_h, out_w, C)
    """
    assert points.ndim == 4, f"expected 4D array, got {points.ndim}D"
    assert points.shape[0] == 1, f"expected batch size 1, got {points.shape[0]}"

    _, in_h, in_w, _ = points.shape
    out_h, out_w = downsample_size

    # PyTorch F.interpolate(..., mode='nearest') 相当の最近傍インデックス
    y_idx = np.floor(np.arange(out_h) * in_h / out_h).astype(int)
    x_idx = np.floor(np.arange(out_w) * in_w / out_w).astype(int)

    return points[:, y_idx[:, None], x_idx[None, :], :]


def _normalized_view_plane_uv(width, height, dtype=np.float32):
    """Build centered UV coordinates normalized by image half-diagonal."""
    half_w = width / 2.0
    half_h = height / 2.0
    half_diag = (height**2 + width**2) ** 0.5 / 2.0
    xs = (np.arange(width, dtype=np.float64) + 0.5 - half_w) / half_diag
    ys = (np.arange(height, dtype=np.float64) + 0.5 - half_h) / half_diag
    uv_x, uv_y = np.meshgrid(xs, ys)
    return np.stack([uv_x, uv_y], axis=-1).astype(dtype)


def _solve_optimal_focal_shift(uv: np.ndarray, xyz: np.ndarray):
    "Solve `min |focal * xy / (z + shift) - uv|` with respect to shift and focal"
    from scipy.optimize import least_squares

    uv, xy, z = uv.reshape(-1, 2), xyz[..., :2].reshape(-1, 2), xyz[..., 2].reshape(-1)

    def fn(uv: np.ndarray, xy: np.ndarray, z: np.ndarray, shift: np.ndarray):
        xy_proj = xy / (z + shift)[:, None]
        f = (xy_proj * uv).sum() / np.square(xy_proj).sum()
        err = (f * xy_proj - uv).ravel()
        return err

    solution = least_squares(partial(fn, uv, xy, z), x0=0, ftol=1e-3, method="lm")
    optim_shift = solution["x"].squeeze().astype(np.float32)

    xy_proj = xy / (z + optim_shift)[:, None]
    optim_focal = (xy_proj * uv).sum() / np.square(xy_proj).sum()

    return optim_shift, optim_focal


def recover_focal_shift(
    points: np.ndarray,
    mask: np.ndarray,
    downsample_size=(64, 64),
):
    """
    Recover the depth map and FoV from a point map with unknown z shift and focal.

    Note that it assumes:
    - the optical center is at the center of the map
    - the map is undistorted
    - the map is isometric in the x and y directions

    ### Parameters:
    - `points: np.ndarray` of shape (..., H, W, 3)
    - `mask: np.ndarray` of shape (..., H, W), binary mask
    - `downsample_size: Tuple[int, int]` in (height, width), the size of the downsampled map. Downsampling produces approximate solution and is efficient for large maps.

    ### Returns:
    - `focal`: np.ndarray of shape (...) the estimated focal length, relative to the half diagonal of the map
    - `shift`: np.ndarray of shape (...) Z-axis shift to translate the point map to camera space
    """
    shape = points.shape
    height, width = points.shape[-3], points.shape[-2]

    points = points.reshape(-1, *shape[-3:])
    mask = mask.reshape(-1, *shape[-3:-1])
    uv = _normalized_view_plane_uv(width, height, dtype=points.dtype)  # (H, W, 2)

    points_lr = _nearest_downsample(points, downsample_size)
    uv_lr = _nearest_downsample(uv[None, ...], downsample_size)[0, ...]
    mask_lr = (
        _nearest_downsample(mask.astype(np.float32)[..., None], downsample_size)[..., 0]
        > 0.0
    )

    optim_shift, optim_focal = [], []
    for i in range(points.shape[0]):
        points_lr_i_np = points_lr[i][mask_lr[i]]
        uv_lr_i_np = uv_lr[mask_lr[i]]
        if uv_lr_i_np.shape[0] < 2:
            optim_focal.append(1)
            optim_shift.append(0)
            continue
        optim_shift_i, optim_focal_i = _solve_optimal_focal_shift(
            uv_lr_i_np, points_lr_i_np
        )
        optim_focal.append(float(optim_focal_i))
        optim_shift.append(float(optim_shift_i))
    optim_shift = np.asarray(optim_shift, dtype=points.dtype).reshape(shape[:-3])
    optim_focal = np.asarray(optim_focal, dtype=points.dtype).reshape(shape[:-3])

    return optim_focal, optim_shift


# ======================
# Main functions
# ======================


def preprocess(rgba, pointmap):
    """
    Full preprocessing pipeline.

    rgba : np.ndarray  [H, W, 4]  uint8  (mask in alpha channel)
    pointmap: np.ndarray  [H, W, 3]  float32
    Returns : (ss_inputs, slat_inputs) dicts ready for model inference.
    """
    rgb_image = rgba[:, :, :3].astype(np.float32) / 255.0
    rgb_image_mask = (rgba[:, :, 3] > 0).astype(np.float32)
    pointmap = pointmap.transpose(1, 2, 0)

    # ObjectCentricSSI normalisation using object mask
    processed_pointmap, return_shift, return_scale = _normalize_pointmap(
        pointmap, rgb_image_mask
    )

    # Object-centric crop (box_size_factor=1.2, no extra padding)
    bbox = _compute_mask_bbox(rgb_image_mask, box_size_factor=1.2)
    x1, y1, x2, y2 = bbox
    top, left, height, width = y1, x1, y2 - y1, x2 - x1
    rgb_crop = _crop(rgb_image, top, left, height, width)
    msk_crop = _crop(rgb_image_mask, top, left, height, width)
    pm_crop = _crop(processed_pointmap, top, left, height, width)

    # Pad to square
    rgb_sq = _pad_square(rgb_crop, fill=0.0)
    msk_sq = _pad_square(msk_crop, fill=0.0)
    pm_sq = _pad_square(pm_crop, fill=0.0)

    # Resize to target sizes
    processed_rgb = _resize(rgb_sq, IMAGE_SIZE, bilinear=True)
    processed_mask = _resize(msk_sq, IMAGE_SIZE, bilinear=False)
    processed_pm = _resize(pm_sq, IMAGE_SIZE, bilinear=False)

    # Full-image (uncropped) versions: pad to square then resize
    rgb_full = _resize(_pad_square(rgb_image, fill=0.0), IMAGE_SIZE, bilinear=True)
    msk_full = _resize(
        _pad_square(rgb_image_mask, fill=0.0), IMAGE_SIZE, bilinear=False
    )

    rgb_pointmap, _, _ = _normalize_pointmap(
        pointmap, msk_full, shift=return_shift, scale=return_scale
    )
    pm_full = _resize(_pad_square(rgb_pointmap, fill=0.0), IMAGE_SIZE, bilinear=False)

    result = {
        "image": image_518.transpose(2, 0, 1),
        "mask": mask_518[..., None].transpose(2, 0, 1),
        "rgb_image": rgb_full_518.transpose(2, 0, 1),
        "rgb_image_mask": msk_full_518[..., None].transpose(2, 0, 1),
        "pointmap": pm_518.transpose(2, 0, 1),
        "rgb_pointmap": pm_full_518.transpose(2, 0, 1),
    }
    return result


def run_moge(image, moge_model):
    """
    Run MoGe depth estimation.

    image       : [3, H, W]  float32
    moge_model  : ONNX InferenceSession or ailia.Net
    Returns     : [H, W, 3]  float32  point map in camera space
    """
    net = moge_model
    if not args.onnx:
        output = net.predict([image[None, ...]])
    else:
        output = net.run(None, {"image": image[None, ...]})
    points, mask = output  # [1, H, W, 3], [1, H, W]

    mask_binary = mask > 0.5

    # Get camera-space point map. (Focal here is the focal length relative to half the image diagonal)
    focal, shift = recover_focal_shift(points, mask_binary)

    # Reproduce force_projection=True: recompute XYZ from depth + intrinsics
    # Add shift to z-axis: points += [0, 0, shift]
    shift_offset = np.stack(
        [np.zeros_like(shift), np.zeros_like(shift), shift], axis=-1
    )  # [*, 3]
    points = points + shift_offset[..., None, None, :]

    # Apply mask (set invalid points to inf)
    points = np.where(mask_binary[..., None], points, np.inf)

    return points.squeeze(0)  # [H, W, 3]


def compute_pointmap(rgba_img, moge_model):
    """
    Compute pointmap from an RGBA image.

    rgba_img   : np.ndarray  [H, W, 4]  uint8  (RGB + alpha mask)
    moge_model : ONNX InferenceSession or ailia.Net
    Returns    : dict with pts_color [3, H, W] and pointmap [3, H, W]
    """
    # RGB only: extract and normalize to [0, 1]
    rgb_img = rgba_img[:, :, :3].astype(np.float32) / 255.0
    rgb_img_chw = rgb_img.transpose(2, 0, 1)

    # Get depth map from MoGe: [H, W, 3]
    pointmap_hwc = run_moge(rgb_img_chw, moge_model)

    # Camera convention: MoGe (x-right, y-down, z-fwd) -> PyTorch3D (x-left, y-up, z-fwd)
    pointmap_hwc = pointmap_hwc * np.array([-1.0, -1.0, 1.0], dtype=np.float32)

    # Convert to CHW for downstream processing
    pointmap_chw = pointmap_hwc.transpose(2, 0, 1)
    return {
        "pts_color": rgb_img_chw,
        "pointmap": pointmap_chw,
    }


def inference(models, img, mask):
    """Run full SAM-3D-Objects pipeline for one image and mask."""
    img = img[:, :, ::-1]  # BGR -> RGB
    mask = mask.astype(np.uint8) * 255
    rgba_img = np.concatenate([img, mask[:, :, None]], axis=2)

    rng = np.random.default_rng(args.seed)

    logger.info("Computing pointmap ...")
    pointmap_dict = compute_pointmap(rgba_img, models["moge"])
    pointmap = pointmap_dict["pointmap"]

    logger.info("Preprocessing ...")
    input_dict = preprocess(rgba_img, pointmap)


def recognize_from_image(models):
    """Loop over all input images."""
    for image_path in args.input:
        logger.info(image_path)

        img = load_image(image_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
        mask = load_mask(args.mask)

        logger.info("Start inference...")
        if args.benchmark:
            logger.info("BENCHMARK mode")
            total_time = 0
            for i in range(args.benchmark_count):
                start = int(round(time.time() * 1000))
                gaussians = inference(models, img, mask)
                end = int(round(time.time() * 1000))
                logger.info(f"\tailia processing estimation time {end - start} ms")
                if i != 0:
                    total_time += end - start
            logger.info(
                f"\taverage time estimation {total_time / (args.benchmark_count - 1)} ms"
            )
        else:
            gaussians = inference(models, img, mask)

        savepath = get_savepath(args.savepath, image_path, ext=".ply")
        save_ply(gaussians, savepath)
        logger.info(f"saved at : {savepath}")


def main():
    # Download models if needed
    check_and_download_models(WEIGHT_MOGE, MODEL_MOGE, REMOTE_PATH)
    check_and_download_models(WEIGHT_SS_COND, MODEL_SS_COND, REMOTE_PATH)

    env_id = args.env_id

    if not args.onnx:
        # Use ailia backend
        moge_model = ailia.Net(MODEL_MOGE, WEIGHT_MOGE, env_id=env_id)
        ss_cond = ailia.Net(MODEL_SS_COND, WEIGHT_SS_COND, env_id=env_id)
    else:
        # Use ONNX Runtime backend
        import onnxruntime

        providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]

        moge_model = onnxruntime.InferenceSession(WEIGHT_MOGE, providers=providers)
        ss_cond = onnxruntime.InferenceSession(WEIGHT_SS_COND, providers=providers)

    models = {
        "moge": moge_model,
        "ss_cond": ss_cond,
    }

    recognize_from_image(models)

    logger.info("Script finished successfully.")


if __name__ == "__main__":
    main()
