import itertools
import math
import sys
import time
from logging import getLogger

import ailia
import cv2
import numpy as np
from PIL import Image

sys.path.append("../../util")
from arg_utils import get_base_parser, get_savepath, update_parser
from detector_utils import load_image
from image_utils import normalize_image
from model_utils import check_and_download_file, check_and_download_models

logger = getLogger(__name__)

# ======================
# Parameters
# ======================

WEIGHT_SAM_ENC_PATH = "sam_image_encoder.onnx"
MODEL_SAM_ENC_PATH = "sam_image_encoder.onnx.prototxt"
DATA_SAM_ENC_PATH = "sam_image_encoder_weights.pb"
WEIGHT_SAM_DEC_PATH = "sam_mask_decoder.onnx"
MODEL_SAM_DEC_PATH = "sam_mask_decoder.onnx.prototxt"
WEIGHT_SEG_ADE20K_PATH = "segformer_ade20k.onnx"
MODEL_SEG_ADE20K_PATH = "segformer_ade20k.onnx.prototxt"
WEIGHT_SEG_CITY_PATH = "segformer_cityscapes.onnx"
MODEL_SEG_CITY_PATH = "segformer_cityscapes.onnx.prototxt"
REMOTE_PATH = "https://storage.googleapis.com/ailia-models/semantic-segment-anything/"

IMAGE_PATH = "demo.png"
SAVE_IMAGE_PATH = "output.png"

# ======================
# Argument Parser Config
# ======================

parser = get_base_parser("Semantic Segment Anything", IMAGE_PATH, SAVE_IMAGE_PATH)
parser.add_argument(
    "-m",
    "--model_type",
    default="ade20k",
    choices=("ade20k", "cityscapes"),
    help="Segformer dataset to use for semantic labels.",
)
parser.add_argument("--onnx", action="store_true", help="Execute onnxruntime version.")
args = update_parser(parser)


# ======================
# Utilities
# ======================


def generate_crop_boxes(im_h, im_w, n_layers, overlap_ratio=512 / 1500):
    """Replicates SAM amg.generate_crop_boxes."""
    crop_boxes = [[0, 0, im_w, im_h]]
    layer_idxs = [0]
    short_side = min(im_h, im_w)

    for i_layer in range(n_layers):
        n_crops = 2 ** (i_layer + 1)
        overlap = int(overlap_ratio * short_side * (2.0 / n_crops))

        crop_w = math.ceil((overlap * (n_crops - 1) + im_w) / n_crops)
        crop_h = math.ceil((overlap * (n_crops - 1) + im_h) / n_crops)

        x0s = [int((crop_w - overlap) * i) for i in range(n_crops)]
        y0s = [int((crop_h - overlap) * i) for i in range(n_crops)]

        for x0, y0 in itertools.product(x0s, y0s):
            box = [x0, y0, min(x0 + crop_w, im_w), min(y0 + crop_h, im_h)]
            crop_boxes.append(box)
            layer_idxs.append(i_layer + 1)

    return crop_boxes, layer_idxs


def build_point_grid(n_per_side):
    """Return (n²,2) normalized [0,1] grid of (x,y) points."""
    offset = 1.0 / (2 * n_per_side)
    pts = np.linspace(offset, 1.0 - offset, n_per_side)
    grid_x, grid_y = np.meshgrid(pts, pts)
    return np.stack([grid_x.ravel(), grid_y.ravel()], axis=-1)


def mask_to_full_image(mask_256, crop_box, new_h, new_w, im_h, im_w):
    """Convert 256×256 SAM logit mask to a boolean mask in full image space.

    Replicates SAM's postprocess_masks (two-step bilinear):
      1. 256×256 → SAM_TARGET×SAM_TARGET
      2. crop padding → [new_h, new_w]
      3. resize → crop region size in original image
    """
    x0, y0, x1, y1 = crop_box
    crop_h_orig = y1 - y0
    crop_w_orig = x1 - x0

    # Step 1: upsample 256×256 → 1024×1024
    mask_1024 = np.array(
        Image.fromarray(mask_256.astype(np.float32)).resize(
            (SAM_TARGET, SAM_TARGET), Image.BILINEAR
        )
    )

    # Step 2: remove padding → [new_h, new_w]
    mask_1024 = mask_1024[:new_h, :new_w]

    # Step 3: resize to crop region in original image
    mask_crop = np.array(
        Image.fromarray(mask_1024.astype(np.float32)).resize(
            (crop_w_orig, crop_h_orig), Image.BILINEAR
        )
    )

    # Place in full image
    full = np.zeros((im_h, im_w), dtype=np.float32)
    full[y0:y1, x0:x1] = mask_crop
    return full


def calculate_stability_score(masks_logits, mask_threshold, offset):
    """Ratio of high-confidence pixels to any-positive pixels."""
    pos_strong = (masks_logits > (mask_threshold + offset)).sum((-2, -1))
    pos_any = (masks_logits > (mask_threshold - offset)).sum((-2, -1))
    return pos_strong / (pos_any + 1e-6)


def is_box_near_crop_edge(boxes, crop_box, orig_box, atol=20.0):
    """Return bool array (N,): True if box is near crop edge but not image edge."""
    crop_edge = np.array(crop_box, dtype=np.float32)
    orig_edge = np.array(orig_box, dtype=np.float32)
    near_crop = np.abs(boxes - crop_edge[None]) <= atol
    near_orig = np.abs(boxes - orig_edge[None]) <= atol
    return np.any(near_crop & ~near_orig, axis=1)


# ======================
# SAM preprocessing helpers
# ======================


def get_preprocess_shape(h, w, long_side):
    scale = long_side / max(h, w)
    new_h = int(h * scale + 0.5)
    new_w = int(w * scale + 0.5)
    return new_h, new_w


def preprocess_sam(img_rgb):
    """Resize, normalize and pad image for SAM encoder. Returns (tensor, new_h, new_w)."""
    im_h, im_w = img_rgb.shape[:2]
    new_h, new_w = get_preprocess_shape(im_h, im_w, SAM_TARGET)

    img = np.array(Image.fromarray(img_rgb).resize((new_w, new_h), Image.BILINEAR))
    img = normalize_image(img, "ImageNet")

    pad_h = SAM_TARGET - new_h
    pad_w = SAM_TARGET - new_w
    img = np.pad(img, ((0, pad_h), (0, pad_w), (0, 0)))

    img = img.transpose(2, 0, 1)[None].astype(np.float32)
    return img, new_h, new_w


# ======================
# SAM inference
# ======================


def run_sam_encoder(models, image):
    input_image, new_h, new_w = preprocess_sam(image)

    enc = models["sam_enc"]
    if not args.onnx:
        output = enc.predict([input_image])
    else:
        output = enc.run(None, {"image": input_image})
    image_embedding = output[0]

    return image_embedding, new_h, new_w


def process_crop(models, img_rgb, crop_box, point_grid, im_h, im_w):
    """
    Run SAM encoder + decoder on one crop. Returns (bin_masks, scores).
    Each mask is boolean (im_h, im_w) in full-image coordinates.
    """
    x0, y0, x1, y1 = crop_box
    crop_img = img_rgb[y0:y1, x0:x1]

    image_embedding, new_h, new_w = run_sam_encoder(models, crop_img)

    # Scale normalized [0,1] grid to encoder input space
    pts_sam = np.zeros_like(point_grid)
    pts_sam[:, 0] = point_grid[:, 0] * new_w  # x in encoder input space
    pts_sam[:, 1] = point_grid[:, 1] * new_h  # y in encoder input space

    dec = models["sam_dec"]
    masks = []
    scores = []

    BATCH_SIZE = 64
    for i in range(0, len(pts_sam), BATCH_SIZE):
        batch_pts = pts_sam[i : i + BATCH_SIZE]  # (B, 2)
        B = len(batch_pts)
        coords = batch_pts[:, None, :].astype(np.float32)  # (B, 1, 2)
        labels = np.ones((B, 1), dtype=np.float32)  # (B, 1) foreground
        emb = np.repeat(image_embedding, B, axis=0)  # (B, C, H, W)

        if not args.onnx:
            output = dec.predict([emb, coords, labels])
        else:
            output = dec.run(
                None,
                {
                    "image_embedding": emb,
                    "point_coords": coords,
                    "point_labels": labels,
                },
            )
        batch_masks, batch_iou_preds = output  # (B,3,256,256), (B,3)

        flat_logits = batch_masks.reshape(
            -1, batch_masks.shape[-2], batch_masks.shape[-1]
        )  # (B*3, 256, 256)
        flat_scores = batch_iou_preds.reshape(-1)  # (B*3,)

        full_logits = np.stack(
            [
                mask_to_full_image(m, crop_box, new_h, new_w, im_h, im_w)
                for m in flat_logits
            ]
        )  # (B*3, im_h, im_w)

        # Filter by predicted IoU
        keep_mask = flat_scores > PRED_IOU_THRESH
        full_logits = full_logits[keep_mask]
        flat_scores = flat_scores[keep_mask]

        # Calculate stability score
        data_stab = calculate_stability_score(full_logits, 0.0, STABILITY_SCORE_OFFSET)
        keep_mask = data_stab >= STABILITY_SCORE_THRESH
        full_logits = full_logits[keep_mask]
        flat_scores = flat_scores[keep_mask]

        # Threshold masks and calculate boxes
        batch_masks = full_logits > 0

        # Filter boxes that touch crop boundaries (but not image boundaries)
        boxes = masks_to_boxes(batch_masks)
        keep_mask = ~is_box_near_crop_edge(boxes, crop_box, [0, 0, im_w, im_h])
        masks.extend(batch_masks[keep_mask])
        scores.extend(flat_scores[keep_mask].tolist())

    masks = np.array(masks)  # (N, im_h, im_w)
    scores = np.array(scores)  # (N,)

    if len(masks) == 0:
        return masks, scores

    return masks, scores


def generate_all_masks(models, img_rgb):
    """Full automatic mask generation matching SAM's SamAutomaticMaskGenerator."""
    im_h, im_w = img_rgb.shape[:2]

    crop_boxes, layer_idxs = generate_crop_boxes(im_h, im_w, CROP_N_LAYERS)

    # Build point grids for all layers
    point_grids = [
        build_point_grid(max(1, POINTS_PER_SIDE // (CROP_N_POINTS_DOWNSCALE_FACTOR**i)))
        for i in range(CROP_N_LAYERS + 1)
    ]

    all_masks = []
    all_scores = []
    for crop_box, layer_idx in zip(crop_boxes, layer_idxs):
        logger.info(f"  Crop {crop_box}, layer {layer_idx}")
        masks, scores = process_crop(
            models, img_rgb, crop_box, point_grids[layer_idx], im_h, im_w
        )
        all_masks.extend(masks)
        all_scores.extend(scores)


# ======================
# Main
# ======================


def predict(models, img):
    logger.info("Generating SAM masks...")
    masks, scores = generate_all_masks(models, img)
    logger.info(f"{len(masks)} masks after filtering")


def recognize_from_image(models):
    for image_path in args.input:
        logger.info(image_path)

        img = load_image(image_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)

        logger.info("Start inference...")
        if args.benchmark:
            logger.info("BENCHMARK mode")
            total_time = 0
            for i in range(args.benchmark_count):
                start = int(round(time.time() * 1000))
                predict(models, img)
                end = int(round(time.time() * 1000))
                t = end - start
                logger.info(f"\tailia processing estimation time {t} ms")
                if i != 0:
                    total_time += t
            logger.info(
                f"\taverage time estimation {total_time / (args.benchmark_count - 1)} ms"
            )
        else:
            predict(models, img)

    logger.info("Script finished successfully.")


def main():
    check_and_download_models(WEIGHT_SAM_ENC_PATH, MODEL_SAM_ENC_PATH, REMOTE_PATH)
    check_and_download_file(DATA_SAM_ENC_PATH, REMOTE_PATH)
    check_and_download_models(WEIGHT_SAM_DEC_PATH, MODEL_SAM_DEC_PATH, REMOTE_PATH)

    if args.model_type == "ade20k":
        check_and_download_models(
            WEIGHT_SEG_ADE20K_PATH, MODEL_SEG_ADE20K_PATH, REMOTE_PATH
        )
    else:
        check_and_download_models(
            WEIGHT_SEG_CITY_PATH, MODEL_SEG_CITY_PATH, REMOTE_PATH
        )

    env_id = args.env_id

    if not args.onnx:
        memory_mode = ailia.get_memory_mode(
            reduce_constant=True,
            ignore_input_with_initializer=True,
            reduce_interstage=False,
            reuse_interstage=True,
        )
        sam_enc = ailia.Net(
            MODEL_SAM_ENC_PATH,
            WEIGHT_SAM_ENC_PATH,
            env_id=env_id,
            memory_mode=memory_mode,
        )
        sam_dec = ailia.Net(
            MODEL_SAM_DEC_PATH,
            WEIGHT_SAM_DEC_PATH,
            env_id=env_id,
            memory_mode=memory_mode,
        )
        if args.model_type == "ade20k":
            segformer = ailia.Net(
                MODEL_SEG_ADE20K_PATH,
                WEIGHT_SEG_ADE20K_PATH,
                env_id=env_id,
                memory_mode=memory_mode,
            )
        else:
            segformer = ailia.Net(
                MODEL_SEG_CITY_PATH,
                WEIGHT_SEG_CITY_PATH,
                env_id=env_id,
                memory_mode=memory_mode,
            )
    else:
        import onnxruntime

        providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
        sam_enc = onnxruntime.InferenceSession(WEIGHT_SAM_ENC_PATH, providers=providers)
        sam_dec = onnxruntime.InferenceSession(WEIGHT_SAM_DEC_PATH, providers=providers)
        if args.model_type == "ade20k":
            segformer = onnxruntime.InferenceSession(
                WEIGHT_SEG_ADE20K_PATH, providers=providers
            )
        else:
            segformer = onnxruntime.InferenceSession(
                WEIGHT_SEG_CITY_PATH, providers=providers
            )

    models = {
        "sam_enc": sam_enc,
        "sam_dec": sam_dec,
        "segformer": segformer,
    }

    recognize_from_image(models)


if __name__ == "__main__":
    main()
