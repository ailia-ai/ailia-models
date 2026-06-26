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

MATCHING_DIST_THRESHOLD_SQ = (
    100.0**2
)  # 100px suppression radius in original image space

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
    choices=["pca", "similarity", "matching"],
    help=(
        "output mode: "
        "pca=PCA patch features (rainbow, requires --mask for foreground-only), "
        "similarity=patch similarity map, "
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
):
    h_l, w_l = img_left_bgr.shape[:2]
    h_r, w_r = img_right_bgr.shape[:2]

    if SPARSE_EQUAL_WIDTH:
        # Equal-width panels: both images at the same display width.
        # Matches matplotlib subplot(1,2,*) — narrower aspect → taller panel.
        display_w = max(w_prep_l, w_prep_r)
        scale_l = display_w / w_prep_l
        scale_r = display_w / w_prep_r
        disp_h_l = max(1, int(round(PCA_IMAGE_SIZE * scale_l)))
        disp_h_r = max(1, int(round(PCA_IMAGE_SIZE * scale_r)))
        img_l_disp = cv2.resize(img_left_bgr, (display_w, disp_h_l))
        img_r_disp = cv2.resize(img_right_bgr, (display_w, disp_h_r))
        max_h = max(disp_h_l, disp_h_r)
        canvas = np.zeros((max_h, display_w * 2, 3), dtype=np.uint8)
        canvas[:disp_h_l, :display_w] = img_l_disp
        canvas[:disp_h_r, display_w:] = img_r_disp
        pt_offset = display_w
    else:
        # Equal-height panels (default): both images at PCA_IMAGE_SIZE rows.
        display_h = PCA_IMAGE_SIZE
        disp_scale = display_h / PCA_IMAGE_SIZE  # == 1.0; kept for clarity
        disp_w_l = max(1, int(round(w_l * display_h / h_l)))
        disp_w_r = max(1, int(round(w_r * display_h / h_r)))
        img_l_disp = cv2.resize(img_left_bgr, (disp_w_l, display_h))
        img_r_disp = cv2.resize(img_right_bgr, (disp_w_r, display_h))
        canvas = np.concatenate([img_l_disp, img_r_disp], axis=1)
        scale_l = scale_r = disp_scale
        pt_offset = disp_w_l

    for i in indices_keep:
        row_l, col_l_px = locs_left[i]
        row_r, col_r_px = locs_right[i]

        pt_l = (int(col_l_px * scale_l), int(row_l * scale_l))
        pt_r = (int(col_r_px * scale_r) + pt_offset, int(row_r * scale_r))

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

        if args.mode == "pca":
            pixel_values, h_patches, w_patches = preprocess_pca(
                img, params["patch_size"]
            )
            if not args.onnx:
                output = net.predict([pixel_values])
            else:
                output = net.run(None, {"pixel_values": pixel_values})
            last_hidden_state, pooler_output = output
            patch_tokens = last_hidden_state[0, num_prefix_tokens:, :]
            panel = render_pca(
                patch_tokens, h_patches, w_patches, img_w, img_h, mask_path=args.mask
            )
            logger.info(f"saved at : {savepath}")
            cv2.imwrite(savepath, panel)

        else:  # similarity
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

    # Stratify in original image space (scale converts 768px locs to original pixel coords)
    scale_l = img_l.shape[0] / PCA_IMAGE_SIZE
    indices_keep = stratify_points(locs_l_fg * scale_l, MATCHING_DIST_THRESHOLD_SQ)
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

    if args.mode == "matching":
        recognize_from_image_matching(net)
    else:
        recognize_from_image(net)


if __name__ == "__main__":
    main()
