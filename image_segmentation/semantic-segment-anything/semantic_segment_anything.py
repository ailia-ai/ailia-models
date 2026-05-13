import gc
import itertools
import math
import os
import sys
import time
from logging import getLogger

import ailia
import cv2
import numpy as np
import spacy
from PIL import Image
from tqdm import tqdm
from transformers import BertTokenizerFast, CLIPTokenizerFast

sys.path.append("../../util")
from arg_utils import get_base_parser, get_savepath, update_parser
from detector_utils import load_image
from image_utils import normalize_image
from math_utils import sigmoid, softmax
from model_utils import check_and_download_file, check_and_download_models

CLIP_BACKEND = "builtin"  # "builtin" | "local"

if CLIP_BACKEND == "builtin":
    import clip_inference_builtin as clip_inference
else:
    import clip_inference_local as clip_inference


logger = getLogger(__name__)

_nlp = spacy.load("en_core_web_sm")


# ======================
# Parameters
# ======================

WEIGHT_SAM_ENC_PATH = "sam_image_encoder.onnx"
MODEL_SAM_ENC_PATH = "sam_image_encoder.onnx.prototxt"
DATA_SAM_ENC_PATH = "sam_image_encoder_weights.pb"
WEIGHT_SAM_DEC_PATH = "sam_mask_decoder.onnx"
MODEL_SAM_DEC_PATH = "sam_mask_decoder.onnx.prototxt"
WEIGHT_CLIP_VIS_PATH = clip_inference.VIS_WEIGHT
MODEL_CLIP_VIS_PATH = clip_inference.VIS_MODEL
WEIGHT_CLIP_TXT_PATH = clip_inference.TXT_WEIGHT
MODEL_CLIP_TXT_PATH = clip_inference.TXT_MODEL
WEIGHT_ONEFORMER_ADE20K_PATH = "oneformer_ade20k.onnx"
MODEL_ONEFORMER_ADE20K_PATH = "oneformer_ade20k.onnx.prototxt"
WEIGHT_ONEFORMER_COCO_PATH = "oneformer_coco.onnx"
MODEL_ONEFORMER_COCO_PATH = "oneformer_coco.onnx.prototxt"
WEIGHT_CLIPSEG_PATH = "clipseg.onnx"
MODEL_CLIPSEG_PATH = "clipseg.onnx.prototxt"
WEIGHT_BLIP_VIS_PATH = "blip_vision_encoder.onnx"
MODEL_BLIP_VIS_PATH = "blip_vision_encoder.onnx.prototxt"
WEIGHT_BLIP_DEC_PATH = "blip_text_decoder.onnx"
MODEL_BLIP_DEC_PATH = "blip_text_decoder.onnx.prototxt"
REMOTE_PATH = "https://storage.googleapis.com/ailia-models/semantic-segment-anything/"
CLIP_REMOTE_PATH = clip_inference.REMOTE_PATH

IMAGE_PATH = "sa_10013862.jpg"
SAVE_IMAGE_PATH = "output.png"

SAM_TARGET = 1024
SAM_PIXEL_MEAN = np.array([123.675, 116.28, 103.53], dtype=np.float32)
SAM_PIXEL_STD = np.array([58.395, 57.12, 57.375], dtype=np.float32)

# SAM auto-mask generation parameters
POINTS_PER_SIDE = 32
PRED_IOU_THRESH = 0.88
STABILITY_SCORE_THRESH = 0.95
STABILITY_SCORE_OFFSET = 1.0
CROP_N_LAYERS = 0
CROP_N_POINTS_DOWNSCALE_FACTOR = 1
MIN_MASK_REGION_AREA = 0
BOX_NMS_THRESH = 0.7
CROP_NMS_THRESH = 0.7

# OneFormer native training resolution per dataset
ONEFORMER_SHORT_EDGE = 800
ONEFORMER_MAX_SIZE = 1333

# Crop scales (matching semantic_annotation_pipeline)
SCALE_SMALL = 1.2
SCALE_LARGE = 1.6  # for BLIP
SCALE_HUGE = 1.6  # for CLIPSeg

# BLIP generation token IDs (blip-image-captioning-large)
BLIP_BOS_TOKEN_ID = 30522
BLIP_SEP_TOKEN_ID = 102  # used as eos in generation
BLIP_MAX_NEW_TOKENS = 20

# Top-K class candidates from each OneFormer model per mask
TOP_K_PER_MODEL = 1

# Number of CLIP top candidates passed to CLIPSeg
CLIP_TOP_K = 3

# CLIP normalization constants
CLIP_MEAN = np.array([0.48145466, 0.4578275, 0.40821073], dtype=np.float32)
CLIP_STD = np.array([0.26862954, 0.26130258, 0.27577711], dtype=np.float32)

# fmt: off
ADE20K_CLASSES = [
    'wall', 'building', 'sky', 'floor', 'tree', 'ceiling', 'road', 'bed',
    'windowpane', 'grass', 'cabinet', 'sidewalk', 'person', 'earth', 'door',
    'table', 'mountain', 'plant', 'curtain', 'chair', 'car', 'water',
    'painting', 'sofa', 'shelf', 'house', 'sea', 'mirror', 'rug', 'field',
    'armchair', 'seat', 'fence', 'desk', 'rock', 'wardrobe', 'lamp',
    'bathtub', 'railing', 'cushion', 'base', 'box', 'column', 'signboard',
    'chest of drawers', 'counter', 'sand', 'sink', 'skyscraper', 'fireplace',
    'refrigerator', 'grandstand', 'path', 'stairs', 'runway', 'case',
    'pool table', 'pillow', 'screen door', 'stairway', 'river', 'bridge',
    'bookcase', 'blind', 'coffee table', 'toilet', 'flower', 'book',
    'hill', 'bench', 'countertop', 'stove', 'palm', 'kitchen island',
    'computer', 'swivel chair', 'boat', 'bar', 'arcade machine', 'hovel',
    'bus', 'towel', 'light', 'truck', 'tower', 'chandelier', 'awning',
    'streetlight', 'booth', 'television receiver', 'airplane', 'dirt track',
    'apparel', 'pole', 'land', 'bannister', 'escalator', 'ottoman',
    'bottle', 'buffet', 'poster', 'stage', 'van', 'ship', 'fountain',
    'conveyer belt', 'canopy', 'washer', 'plaything', 'swimming pool',
    'stool', 'barrel', 'basket', 'waterfall', 'tent', 'bag', 'minibike',
    'cradle', 'oven', 'ball', 'food', 'step', 'tank', 'trade name',
    'microwave', 'pot', 'animal', 'bicycle', 'lake', 'dishwasher',
    'screen', 'blanket', 'sculpture', 'hood', 'sconce', 'vase',
    'traffic light', 'tray', 'ashcan', 'fan', 'pier', 'crt screen',
    'plate', 'monitor', 'bulletin board', 'shower', 'radiator', 'glass',
    'clock', 'flag',
]

# COCO panoptic classes (refined_id2label, indices 0-132)
COCO_CLASSES = [
    'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train',
    'truck', 'boat', 'traffic light', 'fire hydrant', 'stop sign',
    'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep',
    'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella',
    'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard',
    'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard',
    'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup', 'fork',
    'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange',
    'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair',
    'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'tv',
    'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave',
    'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase',
    'scissors', 'teddy bear', 'hair drier', 'toothbrush', 'banner',
    'blanket', 'bridge', 'cardboard', 'counter', 'curtain', 'door',
    'floor-wood', 'flower', 'fruit', 'gravel', 'house', 'light', 'mirror',
    'net', 'pillow', 'platform', 'playingfield', 'railroad', 'river',
    'road', 'roof', 'sand', 'sea', 'shelf', 'snow', 'stairs', 'tent',
    'towel', 'wall-brick', 'wall-stone', 'wall-tile', 'wall', 'water',
    'window-blind', 'window', 'tree', 'fence', 'ceiling', 'sky', 'cabinet',
    'table', 'floor', 'pavement', 'mountain', 'grass', 'dirt', 'paper',
    'food', 'building', 'rock', 'wall', 'rug',
]
# fmt: on

# ======================
# Argument Parser Config
# ======================

parser = get_base_parser("Semantic Segment Anything", IMAGE_PATH, SAVE_IMAGE_PATH)
parser.add_argument("--onnx", action="store_true", help="Execute onnxruntime version.")
args = update_parser(parser)


# ======================
# LazyModel
# ======================


class LazyModel:
    """Defers model loading until the first predict/run call."""

    def __init__(self, loader_fn, name=""):
        self._loader_fn = loader_fn
        self._name = name
        self._net = None

    def load(self):
        if self._net is None:
            logger.info(f"Loading model: {self._name}")
            self._net = self._loader_fn()
        return self._net

    def unload(self):
        if self._net is not None:
            self._net = None
            gc.collect()

    def predict(self, *args, **kwargs):
        return self.load().predict(*args, **kwargs)

    def run(self, *args, **kwargs):
        return self.load().run(*args, **kwargs)


# ======================
# Visualization
# ======================


def draw_result(img_bgr, masks, class_names):
    """Draw per-mask results matching mmdet imshow_det_bboxes behavior."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.collections import PatchCollection
    from matplotlib.patches import Polygon

    EPS = 1e-2

    N = len(masks)
    img_rgb = img_bgr[:, :, ::-1].copy().astype(np.uint8)
    height, width = img_rgb.shape[:2]

    state = np.random.get_state()
    np.random.seed(42)
    colors = np.random.randint(0, 256, size=(N, 3)).astype(np.uint8)
    np.random.set_state(state)

    text_color_mpl = (0.0, 1.0, 0.0)

    taken_colors = set([0, 0, 0])
    polygons = []
    for i, mask in enumerate(masks):
        mask_u8 = np.ascontiguousarray(mask.astype(np.uint8))
        contours, _ = cv2.findContours(mask_u8, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)
        for c in contours:
            pts = c.reshape(-1, 2)
            if len(pts) >= 2:
                polygons.append(Polygon(pts))

        color_mask = colors[i].copy()
        while tuple(color_mask.tolist()) in taken_colors:
            bias = np.random.randint(-30, 31, size=3)
            color_mask = np.clip(color_mask.astype(np.int32) + bias, 0, 255).astype(
                np.uint8
            )
        taken_colors.add(tuple(color_mask.tolist()))

        mask_bool = mask.astype(bool)
        img_rgb[mask_bool] = np.clip(
            img_rgb[mask_bool] * 0.2 + color_mask * 0.8, 0, 255
        ).astype(np.uint8)

    # Create matplotlib figure (same sizing as imshow_det_bboxes)
    fig = plt.figure("", frameon=False)
    plt.title("")
    canvas = fig.canvas
    dpi = fig.get_dpi()
    fig.set_size_inches((width + EPS) / dpi, (height + EPS) / dpi)
    plt.subplots_adjust(left=0, right=1, bottom=0, top=1)
    ax = plt.gca()
    ax.axis("off")

    # White contour edges (PatchCollection with edgecolors='w', alpha=0.8)
    if polygons:
        p = PatchCollection(
            polygons, facecolor="none", edgecolors="w", linewidths=1, alpha=0.8
        )
        ax.add_collection(p)

    for mask, name in zip(masks, class_names):
        mask_u8 = mask.astype(np.uint8)
        _, _, stats, centroids = cv2.connectedComponentsWithStats(
            mask_u8, connectivity=8
        )
        if len(stats) < 2:
            continue
        largest_id = int(np.argmax(stats[1:, -1])) + 1
        pos = centroids[largest_id]
        area = float(stats[largest_id, -1])
        scale = float(np.clip(0.5 + (area - 800) / (30000 - 800), 0.5, 1.0))

        ax.text(
            pos[0],
            pos[1],
            name,
            bbox={"facecolor": "black", "alpha": 0.8, "pad": 0.7, "edgecolor": "none"},
            color=text_color_mpl,
            fontsize=25 * scale,
            verticalalignment="top",
            horizontalalignment="center",
        )

    plt.imshow(img_rgb)

    stream, _ = canvas.print_to_buffer()
    buffer = np.frombuffer(stream, dtype="uint8")
    if sys.platform == "darwin":
        width, height = canvas.get_width_height(physical=True)
    img_rgba = buffer.reshape(height, width, 4)
    rgb, _ = np.split(img_rgba, [3], axis=2)
    result_bgr = rgb.astype(np.uint8)[:, :, ::-1].copy()

    plt.close()
    return result_bgr


# ======================
# SAM utilities
# ======================


def generate_crop_boxes(im_size, n_layers, overlap_ratio):
    crop_boxes, layer_idxs = [], []
    im_h, im_w = im_size
    short_side = min(im_h, im_w)

    crop_boxes.append([0, 0, im_w, im_h])
    layer_idxs.append(0)

    def crop_len(orig_len, n_crops, overlap):
        return int(math.ceil((overlap * (n_crops - 1) + orig_len) / n_crops))

    for i_layer in range(n_layers):
        n_crops_per_side = 2 ** (i_layer + 1)
        overlap = int(overlap_ratio * short_side * (2 / n_crops_per_side))

        crop_w = crop_len(im_w, n_crops_per_side, overlap)
        crop_h = crop_len(im_h, n_crops_per_side, overlap)

        crop_box_x0 = [int((crop_w - overlap) * i) for i in range(n_crops_per_side)]
        crop_box_y0 = [int((crop_h - overlap) * i) for i in range(n_crops_per_side)]

        for x0, y0 in itertools.product(crop_box_x0, crop_box_y0):
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


def remove_small_regions(mask, area_thresh, mode):
    """Remove small disconnected regions and holes in a mask (requires cv2)."""
    assert mode in ["holes", "islands"]
    correct_holes = mode == "holes"
    working_mask = (correct_holes ^ mask).astype(np.uint8)
    n_labels, regions, stats, _ = cv2.connectedComponentsWithStats(working_mask, 8)
    sizes = stats[:, -1][1:]
    small_regions = [i + 1 for i, s in enumerate(sizes) if s < area_thresh]
    if len(small_regions) == 0:
        return mask, False
    fill_labels = [0] + small_regions
    if not correct_holes:
        fill_labels = [i for i in range(n_labels) if i not in fill_labels]
        if len(fill_labels) == 0:
            fill_labels = [int(np.argmax(sizes)) + 1]
    mask = np.isin(regions, fill_labels)
    return mask, True


def postprocess_small_regions(masks, scores, boxes, min_area, nms_thresh):
    """Remove small disconnected regions/holes, then re-NMS to drop new duplicates."""
    if len(masks) == 0:
        return masks, scores, boxes

    new_masks = []
    nms_scores = []
    for mask in masks:
        mask, changed = remove_small_regions(mask, min_area, mode="holes")
        unchanged = not changed
        mask, changed = remove_small_regions(mask, min_area, mode="islands")
        unchanged = unchanged and not changed
        new_masks.append(mask)
        # prefer masks that needed no postprocessing
        nms_scores.append(1.0 if unchanged else 0.0)

    new_boxes = masks_to_boxes(new_masks)
    nms_scores = np.array(nms_scores)
    kept_idx = box_nms(new_boxes, nms_scores, nms_thresh)

    masks = [new_masks[i] for i in kept_idx]
    scores = scores[kept_idx]
    boxes = new_boxes[kept_idx]
    return masks, scores, boxes


def masks_to_boxes(masks):
    """Convert list of boolean (H,W) masks to (N,4) XYXY boxes."""
    boxes = np.zeros((len(masks), 4), dtype=np.float32)
    for i, mask in enumerate(masks):
        ys, xs = np.where(mask)
        if len(xs):
            boxes[i] = [xs.min(), ys.min(), xs.max(), ys.max()]
    return boxes


def box_nms(boxes, scores, iou_threshold):
    """Greedy box-based NMS, returns kept indices (sorted by score desc)."""
    if len(boxes) == 0:
        return []
    x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    order = np.argsort(scores)[::-1]
    kept = []
    suppressed = np.zeros(len(boxes), dtype=bool)
    for i in order:
        if suppressed[i]:
            continue
        kept.append(int(i))
        ix1, iy1, ix2, iy2 = x1[i], y1[i], x2[i], y2[i]
        inter_x1 = np.maximum(ix1, x1[order])
        inter_y1 = np.maximum(iy1, y1[order])
        inter_x2 = np.minimum(ix2, x2[order])
        inter_y2 = np.minimum(iy2, y2[order])
        inter_w = np.maximum(0, inter_x2 - inter_x1 + 1)
        inter_h = np.maximum(0, inter_y2 - inter_y1 + 1)
        inter = inter_w * inter_h
        iou = inter / (areas[i] + areas[order] - inter + 1e-6)
        suppress_mask = iou > iou_threshold
        suppress_mask[0] = False
        suppressed[order[suppress_mask]] = True
    return kept


# ======================
# SAM preprocessing / inference
# ======================


def get_preprocess_shape(h, w, long_side):
    scale = long_side / max(h, w)
    new_h = int(h * scale + 0.5)
    new_w = int(w * scale + 0.5)
    return new_h, new_w


def preprocess_sam(img):
    im_h, im_w = img.shape[:2]
    new_h, new_w = get_preprocess_shape(im_h, im_w, SAM_TARGET)

    img = np.array(Image.fromarray(img).resize((new_w, new_h), Image.BILINEAR))
    img = (
        img.astype(np.float32) - SAM_PIXEL_MEAN
    ) / SAM_PIXEL_STD  # Sam.preprocess() L167

    pad_h = SAM_TARGET - new_h
    pad_w = SAM_TARGET - new_w
    img = np.pad(img, ((0, pad_h), (0, pad_w), (0, 0)))

    img = img.transpose(2, 0, 1)[None].astype(np.float32)
    return img, new_h, new_w


def run_sam_encoder(models, image):
    input_image, new_h, new_w = preprocess_sam(image)

    enc = models["sam_enc"]
    if not args.onnx:
        output = enc.predict([input_image])
    else:
        output = enc.run(None, {"image": input_image})
    image_embedding = output[0]

    return image_embedding, new_h, new_w


def process_crop(models, img, crop_box, point_grid, im_h, im_w):
    x0, y0, x1, y1 = crop_box
    crop_img = img[y0:y1, x0:x1]

    image_embedding, new_h, new_w = run_sam_encoder(models, crop_img)

    pts_sam = np.zeros_like(point_grid)
    pts_sam[:, 0] = point_grid[:, 0] * new_w
    pts_sam[:, 1] = point_grid[:, 1] * new_h

    dec = models["sam_dec"]
    dec.load()

    mask_list = []
    score_list = []
    box_list = []

    BATCH_SIZE = 64
    for i in tqdm(range(0, len(pts_sam), BATCH_SIZE), desc="Processing batches"):
        batch_pts = pts_sam[i : i + BATCH_SIZE]
        B = len(batch_pts)
        coords = batch_pts[:, None, :].astype(np.float32)
        labels = np.ones((B, 1), dtype=np.float32)
        emb = np.repeat(image_embedding, B, axis=0)

        if not args.onnx:
            output = dec.predict(
                [
                    emb,
                    coords,
                    labels,
                    np.array(new_h, dtype=np.int64),
                    np.array(new_w, dtype=np.int64),
                    np.array(crop_box, dtype=np.int64),
                ]
            )
        else:
            output = dec.run(
                None,
                {
                    "image_embedding": emb,
                    "point_coords": coords,
                    "point_labels": labels,
                    "pp_new_h": np.array(new_h, dtype=np.int64),
                    "pp_new_w": np.array(new_w, dtype=np.int64),
                    "pp_crop_box": np.array(crop_box, dtype=np.int64),
                },
            )
        batch_masks, batch_iou_preds = output

        masks = batch_masks.reshape(-1, batch_masks.shape[-2], batch_masks.shape[-1])
        iou_preds = batch_iou_preds.reshape(-1)

        # Filter by predicted IoU
        keep_mask = iou_preds > PRED_IOU_THRESH
        masks = masks[keep_mask]
        iou_preds = iou_preds[keep_mask]
        if len(masks) == 0:
            continue

        # Calculate stability score
        stability_score = calculate_stability_score(masks, 0.0, STABILITY_SCORE_OFFSET)
        keep_mask = stability_score >= STABILITY_SCORE_THRESH
        masks = masks[keep_mask]
        iou_preds = iou_preds[keep_mask]

        # Threshold masks and calculate boxes
        x0, y0, x1, y1 = crop_box
        crop_bool = masks > 0
        boxes_crop = masks_to_boxes(crop_bool)

        # Filter boxes that touch crop boundaries
        boxes = boxes_crop + np.array([x0, y0, x0, y0], dtype=boxes_crop.dtype)
        keep_mask = ~is_box_near_crop_edge(boxes, crop_box, [0, 0, im_w, im_h])
        crop_bool = crop_bool[keep_mask]
        boxes = boxes[keep_mask]
        iou_preds = iou_preds[keep_mask]

        # Upscale crop masks into full image space
        full_masks = np.zeros((len(crop_bool), im_h, im_w), dtype=bool)
        full_masks[:, y0:y1, x0:x1] = crop_bool

        mask_list.extend(full_masks)
        score_list.extend(iou_preds)
        box_list.extend(boxes)

    masks = np.array(mask_list)
    scores = np.array(score_list)
    boxes = np.array(box_list)

    if len(masks) == 0:
        return masks, scores, boxes

    # NMS
    if len(masks) > 1:
        kept_idx = box_nms(boxes, scores, BOX_NMS_THRESH)
        masks = masks[kept_idx]
        scores = scores[kept_idx]
        boxes = boxes[kept_idx]

    return masks, scores, boxes


def generate_all_masks(models, img):
    im_h, im_w = img.shape[:2]

    crop_boxes, layer_idxs = generate_crop_boxes(
        (im_h, im_w), CROP_N_LAYERS, 512 / 1500
    )

    # Build point grids for all layers
    point_grids = [
        build_point_grid(max(1, POINTS_PER_SIDE // (CROP_N_POINTS_DOWNSCALE_FACTOR**i)))
        for i in range(CROP_N_LAYERS + 1)
    ]

    all_masks = []
    all_scores = []
    all_boxes = []
    all_crop_boxes = []
    for crop_box, layer_idx in zip(crop_boxes, layer_idxs):
        logger.info(f"  Crop {crop_box}, layer {layer_idx}")
        masks, scores, boxes = process_crop(
            models, img, crop_box, point_grids[layer_idx], im_h, im_w
        )
        all_masks.extend(masks)
        all_scores.extend(scores)
        all_boxes.extend(boxes)
        all_crop_boxes.extend([crop_box] * len(masks))

    if len(all_masks) == 0:
        return [], np.array([]), np.array([])

    all_scores = np.array(all_scores)
    all_boxes = np.array(all_boxes)

    # Remove duplicate masks between crops
    if len(crop_boxes) > 1 and len(all_masks) > 0:
        crop_boxes_arr = np.array(all_crop_boxes, dtype=np.float32)
        crop_areas = (crop_boxes_arr[:, 2] - crop_boxes_arr[:, 0]) * (
            crop_boxes_arr[:, 3] - crop_boxes_arr[:, 1]
        )
        crop_scores = 1.0 / np.maximum(crop_areas, 1.0)
        kept_idx = box_nms(all_boxes, crop_scores, CROP_NMS_THRESH)
        all_masks = [all_masks[i] for i in kept_idx]
        all_scores = all_scores[kept_idx]
        all_boxes = all_boxes[kept_idx]

    return all_masks, all_scores, all_boxes


# ======================
# OneFormer preprocessing / inference
# ======================


def preprocess_oneformer(img):
    im_h, im_w = img.shape[:2]
    scale = ONEFORMER_SHORT_EDGE / min(im_h, im_w)
    new_h = int(round(im_h * scale))
    new_w = int(round(im_w * scale))
    if max(new_h, new_w) > ONEFORMER_MAX_SIZE:
        scale = ONEFORMER_MAX_SIZE / max(new_h, new_w)
        new_h = int(round(new_h * scale))
        new_w = int(round(new_w * scale))
    img = np.array(Image.fromarray(img).resize((new_w, new_h), Image.BILINEAR))
    img = normalize_image(img, "ImageNet")
    return img.transpose(2, 0, 1)[None].astype(np.float32)


def resize_bilinear(feat, out_h, out_w):
    """Bilinear resize (C, in_h, in_w) -> (C, out_h, out_w), align_corners=False."""
    c, in_h, in_w = feat.shape
    y = (np.arange(out_h, dtype=np.float32) + 0.5) * in_h / out_h - 0.5
    x = (np.arange(out_w, dtype=np.float32) + 0.5) * in_w / out_w - 0.5
    y = np.clip(y, 0.0, in_h - 1.0)
    x = np.clip(x, 0.0, in_w - 1.0)
    y0 = np.floor(y).astype(np.int64).clip(0, in_h - 1)
    x0 = np.floor(x).astype(np.int64).clip(0, in_w - 1)
    y1 = (y0 + 1).clip(0, in_h - 1)
    x1 = (x0 + 1).clip(0, in_w - 1)
    wy = (y - y0).astype(np.float32)
    wx = (x - x0).astype(np.float32)
    wa = (1 - wy)[:, None] * (1 - wx)[None, :]
    wb = (1 - wy)[:, None] * wx[None, :]
    wc = wy[:, None] * (1 - wx)[None, :]
    wd = wy[:, None] * wx[None, :]
    return (
        feat[:, y0[:, None], x0[None, :]] * wa
        + feat[:, y0[:, None], x1[None, :]] * wb
        + feat[:, y1[:, None], x0[None, :]] * wc
        + feat[:, y1[:, None], x1[None, :]] * wd
    )


def oneformer_postprocess(
    class_queries_logits, masks_queries_logits, target_h, target_w
):
    """OneFormer outputs -> (H, W) class ID map."""
    # Remove null class (last entry), matching transformers post_process_semantic_segmentation
    masks_classes = softmax(class_queries_logits, axis=-1)[..., :-1]  # (1, Q, C)
    masks_probs = sigmoid(masks_queries_logits)  # (1, Q, H', W')
    segmentation = np.einsum(
        "bqc,bqhw->bchw", masks_classes, masks_probs
    )  # (1, C, H', W')
    seg_up = resize_bilinear(segmentation[0], target_h, target_w)  # (C, H, W)
    return np.argmax(seg_up, axis=0).astype(np.int32)


def run_oneformer(net, img):
    im_h, im_w = img.shape[:2]
    img_tensor = preprocess_oneformer(img)

    if not args.onnx:
        output = net.predict([img_tensor])
    else:
        output = net.run(None, {"pixel_values": img_tensor})
    class_queries_logits, masks_queries_logits = output

    return oneformer_postprocess(class_queries_logits, masks_queries_logits, im_h, im_w)


# ======================
# CLIP text tokenizer
# ======================


def clip_tokenize(models, texts):
    tok = models["clip_tokenizer"]
    enc = tok(
        texts,
        return_tensors="np",
        padding=True,
        truncation=True,
        max_length=77,
    )
    return enc["input_ids"].astype(np.int64), enc["attention_mask"].astype(np.int64)


# ======================
# CLIP vision / text inference
# ======================


def preprocess_clip_image(img):
    img = Image.fromarray(img).resize((224, 224), Image.BICUBIC)
    img = np.array(img, dtype=np.float32) / 255.0
    img = (img - CLIP_MEAN) / CLIP_STD
    return img.transpose(2, 0, 1)[None].astype(np.float32)


def run_clip_text(models, texts):
    """Encode list of texts -> (N, D) L2-normalised embeddings."""
    return clip_inference.run_text(
        models["clip_txt"], texts, args.onnx, models.get("clip_tokenizer")
    )


def run_clip_vision(models, img):
    """Encode image patch -> (D,) L2-normalised embedding."""
    img_tensor = preprocess_clip_image(img)
    return clip_inference.run_vision(models["clip_vis"], img_tensor, args.onnx)


# ======================
# CLIPSeg inference
# ======================


def preprocess_clipseg_image(img_np, n_classes):
    """Resize to 512x512, normalise with CLIP stats -> (n, 3, 512, 512)."""
    img = cv2.resize(img_np, (512, 512), interpolation=cv2.INTER_LINEAR)
    img = img.astype(np.float32) / 255.0
    img = (img - CLIP_MEAN) / CLIP_STD
    tile = img.transpose(2, 0, 1)  # (3, 512, 512)
    return np.repeat(tile[None], n_classes, axis=0).astype(np.float32)


def clipseg_segmentation(models, img_crop, class_names):
    """Run CLIPSeg on a crop for each class_name -> (N, H, W) logits (crop size)."""
    h, w = img_crop.shape[:2]
    n = len(class_names)
    input_ids, attention_mask = clip_tokenize(models, class_names)
    pixel_values = preprocess_clipseg_image(img_crop, n)

    net = models["clipseg"]
    if not args.onnx:
        output = net.predict([input_ids, attention_mask, pixel_values])
    else:
        output = net.run(
            None,
            {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "pixel_values": pixel_values,
            },
        )
    logits = output[0]  # (N, H', W')
    return resize_bilinear(logits, h, w)  # (N, H, W)


# ======================
# BLIP inference
# ======================


def preprocess_blip_image(img_bgr_np):
    """(H,W,3) uint8 BGR -> (1,3,384,384) float32 with CLIP/BLIP normalisation.
    NOTE: passes BGR as-is (no channel swap) to match original pipeline.py behavior,
    where mmcv BGR patches are fed directly to BlipProcessor without RGB conversion."""
    img = Image.fromarray(img_bgr_np).resize((384, 384), Image.BICUBIC)
    img = np.array(img, dtype=np.float32) / 255.0
    img = (img - CLIP_MEAN) / CLIP_STD
    return img.transpose(2, 0, 1)[None].astype(np.float32)


def get_noun_phrases(text):
    return [chunk.text for chunk in _nlp(text).noun_chunks]


def open_vocabulary_classification_blip(models, pixel_values):
    """Greedy decode a caption from pixel_values (1,3,384,384) using BLIP ONNX."""
    vis_net = models["blip_vis"]
    if not args.onnx:
        image_embeds = vis_net.predict([pixel_values])[0]
    else:
        image_embeds = vis_net.run(None, {"pixel_values": pixel_values})[0]

    img_seq_len = image_embeds.shape[1]
    encoder_attention_mask = np.ones((1, img_seq_len), dtype=np.int64)

    dec_net = models["blip_dec"]
    input_ids = np.array([[BLIP_BOS_TOKEN_ID]], dtype=np.int64)

    for _ in range(BLIP_MAX_NEW_TOKENS):
        attention_mask = np.ones_like(input_ids, dtype=np.int64)
        if not args.onnx:
            logits = dec_net.predict(
                [input_ids, attention_mask, image_embeds, encoder_attention_mask]
            )[0]
        else:
            logits = dec_net.run(
                None,
                {
                    "input_ids": input_ids,
                    "attention_mask": attention_mask,
                    "encoder_hidden_states": image_embeds,
                    "encoder_attention_mask": encoder_attention_mask,
                },
            )[0]
        next_token = int(np.argmax(logits[0, -1, :]))
        if next_token == BLIP_SEP_TOKEN_ID:
            break
        input_ids = np.concatenate(
            [input_ids, np.array([[next_token]], dtype=np.int64)], axis=1
        )

    tokenizer = models["blip_tokenizer"]
    caption = tokenizer.decode(input_ids[0].tolist(), skip_special_tokens=True)
    return get_noun_phrases(caption)


# ======================
# Crop utility
# ======================


def imcrop_scale(img, bbox_xyxy, scale):
    """Crop img at bbox (x1,y1,x2,y2) expanded by scale. Matches mmcv.imcrop."""
    x1, y1, x2, y2 = (
        int(bbox_xyxy[0]),
        int(bbox_xyxy[1]),
        int(bbox_xyxy[2]),
        int(bbox_xyxy[3]),
    )
    w = float(x2 - x1 + 1)
    h = float(y2 - y1 + 1)
    dw = w * (scale - 1) * 0.5
    dh = h * (scale - 1) * 0.5
    nx1 = int(x1 - dw)
    ny1 = int(y1 - dh)
    nx2 = int(x2 + dw)
    ny2 = int(y2 + dh)
    nx1 = max(0, min(img.shape[1] - 1, nx1))
    ny1 = max(0, min(img.shape[0] - 1, ny1))
    nx2 = max(0, min(img.shape[1] - 1, nx2))
    ny2 = max(0, min(img.shape[0] - 1, ny2))
    patch = img[ny1 : ny2 + 1, nx1 : nx2 + 1]
    if patch.size == 0:
        return img[0:1, 0:1]
    return patch


# ======================
# Main predict function  (semantic_annotation_pipeline equivalent)
# ======================


def predict(models, img):
    # --- SAM: generate masks -----------------------------------------------
    logger.info("Generating SAM masks...")
    masks, scores, boxes = generate_all_masks(models, img)
    masks, scores, boxes = postprocess_small_regions(
        masks, scores, boxes, MIN_MASK_REGION_AREA, max(BOX_NMS_THRESH, CROP_NMS_THRESH)
    )
    models["sam_enc"].unload()
    models["sam_dec"].unload()
    logger.info(f"{len(masks)} masks after filtering")

    if not masks:
        return [], []

    # Sort by area descending (largest first)
    areas = np.array([m.sum() for m in masks])
    order = np.argsort(areas)[::-1]
    masks = [masks[i] for i in order]
    boxes = boxes[order]

    # --- OneFormer: full-image semantic class maps --------------------------
    logger.info("Running OneFormer COCO...")
    net = models["oneformer_coco"]
    coco_ids = run_oneformer(net, img)  # (H, W), values 0-132
    models["oneformer_coco"].unload()

    logger.info("Running OneFormer ADE20K...")
    net = models["oneformer_ade20k"]
    ade20k_ids = run_oneformer(net, img)  # (H, W), values 0-149
    models["oneformer_ade20k"].unload()

    # --- Pre-encode all class names with CLIP text encoder -----------------
    logger.info("Encoding class names with CLIP text encoder...")
    all_classes = list(dict.fromkeys(ADE20K_CLASSES + COCO_CLASSES))  # deduped
    BATCH = 64
    text_embs_list = []
    for i in range(0, len(all_classes), BATCH):
        text_embs_list.append(run_clip_text(models, all_classes[i : i + BATCH]))
    all_text_embs = np.concatenate(text_embs_list, axis=0)  # (N_all, D)
    # Build name->embedding cache; clip_txt stays loaded for novel BLIP nouns
    name_to_emb = {n: all_text_embs[i] for i, n in enumerate(all_classes)}

    # --- Per-mask annotation -----------------------------------------------
    logger.info("Annotating masks with CLIP + CLIPSeg...")
    models["blip_vis"].load()
    models["blip_dec"].load()
    models["clip_vis"].load()
    models["clipseg"].load()
    mask_class_names = []

    for mask, box in tqdm(zip(masks, boxes), total=len(masks), desc="Annotating"):
        if not mask.any():
            mask_class_names.append("background")
            continue

        # Top-k candidates from each OneFormer model within this mask
        ade20k_votes = np.bincount(
            ade20k_ids[mask].astype(np.int32), minlength=len(ADE20K_CLASSES)
        )
        coco_votes = np.bincount(
            coco_ids[mask].astype(np.int32), minlength=len(COCO_CLASSES)
        )
        ade20k_top = np.argsort(ade20k_votes)[::-1][:TOP_K_PER_MODEL]
        coco_top = np.argsort(coco_votes)[::-1][:TOP_K_PER_MODEL]

        local_candidates = list(
            dict.fromkeys(
                [ADE20K_CLASSES[i] for i in ade20k_top]
                + [COCO_CLASSES[i] for i in coco_top if i < len(COCO_CLASSES)]
            )
        )

        # BLIP: add open-vocabulary candidates from image captioning
        patch_large = imcrop_scale(img, box.tolist(), SCALE_LARGE)
        if patch_large.size > 0:
            try:
                pv = preprocess_blip_image(patch_large)
                blip_nouns = open_vocabulary_classification_blip(models, pv)
                local_candidates = list(dict.fromkeys(local_candidates + blip_nouns))
            except Exception:
                pass

        if not local_candidates:
            mask_class_names.append("background")
            continue

        # CLIP: rank local candidates using a small crop
        patch_small = imcrop_scale(img, box.tolist(), SCALE_SMALL)
        if patch_small.size == 0:
            mask_class_names.append(local_candidates[0])
            continue

        img_emb = run_clip_vision(models, patch_small)  # (D,)

        # Encode any novel candidates (e.g. BLIP noun phrases) not pre-encoded
        novel = [n for n in local_candidates if n not in name_to_emb]
        if novel:
            novel_embs = run_clip_text(models, novel)
            for n, e in zip(novel, novel_embs):
                name_to_emb[n] = e

        local_embs = np.stack([name_to_emb[n] for n in local_candidates], axis=0)
        sims = img_emb @ local_embs.T  # (len(local),)
        n_top = min(CLIP_TOP_K, len(local_candidates))
        top_k_classes = [local_candidates[i] for i in np.argsort(sims)[::-1][:n_top]]

        if len(top_k_classes) == 1:
            mask_class_names.append(top_k_classes[0])
            continue

        # CLIPSeg: segment a larger crop, vote within the valid mask region
        patch_huge = imcrop_scale(img, box.tolist(), SCALE_HUGE)
        if patch_huge.size == 0:
            mask_class_names.append(top_k_classes[0])
            continue

        # Crop the boolean mask to the same region as patch_huge
        valid_mask_huge = imcrop_scale(
            mask.astype(np.uint8), box.tolist(), SCALE_HUGE
        ).astype(bool)

        logits = clipseg_segmentation(models, patch_huge, top_k_classes)  # (k, h, w)

        # Align valid_mask_huge to logits spatial size if needed
        lh, lw = logits.shape[1], logits.shape[2]
        if valid_mask_huge.shape != (lh, lw):
            valid_mask_huge = np.array(
                Image.fromarray(valid_mask_huge.astype(np.uint8)).resize(
                    (lw, lh), Image.NEAREST
                )
            ).astype(bool)

        pred_ids = np.argmax(logits, axis=0)  # (h, w)
        valid_preds = pred_ids[valid_mask_huge]
        if len(valid_preds) == 0:
            mask_class_names.append(top_k_classes[0])
            continue

        winner = int(
            np.bincount(
                valid_preds.astype(np.int32), minlength=len(top_k_classes)
            ).argmax()
        )
        mask_class_names.append(top_k_classes[winner])

    models["clip_txt"].unload()
    return masks, mask_class_names


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
                semantc_mask, class_names = predict(models, img)
                end = int(round(time.time() * 1000))
                t = end - start
                logger.info(f"\tailia processing estimation time {t} ms")
                if i != 0:
                    total_time += t
            logger.info(
                f"\taverage time estimation {total_time / (args.benchmark_count - 1)} ms"
            )
        else:
            semantc_mask, class_names = predict(models, img)

        res_img = draw_result(img, semantc_mask, class_names)

        savepath = get_savepath(args.savepath, image_path, ext=".png")
        logger.info(f"saved at : {savepath}")
        cv2.imwrite(savepath, res_img)

    logger.info("Script finished successfully.")


def main():
    check_and_download_models(WEIGHT_SAM_ENC_PATH, MODEL_SAM_ENC_PATH, REMOTE_PATH)
    check_and_download_file(DATA_SAM_ENC_PATH, REMOTE_PATH)
    check_and_download_models(WEIGHT_SAM_DEC_PATH, MODEL_SAM_DEC_PATH, REMOTE_PATH)
    check_and_download_models(
        WEIGHT_CLIP_VIS_PATH, MODEL_CLIP_VIS_PATH, CLIP_REMOTE_PATH
    )
    check_and_download_models(
        WEIGHT_CLIP_TXT_PATH, MODEL_CLIP_TXT_PATH, CLIP_REMOTE_PATH
    )
    check_and_download_models(
        WEIGHT_ONEFORMER_ADE20K_PATH, MODEL_ONEFORMER_ADE20K_PATH, REMOTE_PATH
    )
    check_and_download_models(
        WEIGHT_ONEFORMER_COCO_PATH, MODEL_ONEFORMER_COCO_PATH, REMOTE_PATH
    )
    check_and_download_models(WEIGHT_CLIPSEG_PATH, MODEL_CLIPSEG_PATH, REMOTE_PATH)
    check_and_download_models(WEIGHT_BLIP_VIS_PATH, MODEL_BLIP_VIS_PATH, REMOTE_PATH)
    check_and_download_models(WEIGHT_BLIP_DEC_PATH, MODEL_BLIP_DEC_PATH, REMOTE_PATH)

    env_id = args.env_id

    if not args.onnx:
        memory_mode = ailia.get_memory_mode(
            reduce_constant=True,
            ignore_input_with_initializer=True,
            reduce_interstage=False,
            reuse_interstage=True,
        )

        def _net(model_path, weight_path):
            return ailia.Net(
                model_path, weight_path, env_id=env_id, memory_mode=memory_mode
            )

        sam_enc = LazyModel(
            lambda: _net(MODEL_SAM_ENC_PATH, WEIGHT_SAM_ENC_PATH), "sam_enc"
        )
        sam_dec = LazyModel(
            lambda: _net(MODEL_SAM_DEC_PATH, WEIGHT_SAM_DEC_PATH), "sam_dec"
        )
        clip_vis = LazyModel(
            lambda: _net(MODEL_CLIP_VIS_PATH, WEIGHT_CLIP_VIS_PATH), "clip_vis"
        )
        clip_txt = LazyModel(
            lambda: _net(MODEL_CLIP_TXT_PATH, WEIGHT_CLIP_TXT_PATH), "clip_txt"
        )
        oneformer_ade20k = LazyModel(
            lambda: _net(MODEL_ONEFORMER_ADE20K_PATH, WEIGHT_ONEFORMER_ADE20K_PATH),
            "oneformer_ade20k",
        )
        oneformer_coco = LazyModel(
            lambda: _net(MODEL_ONEFORMER_COCO_PATH, WEIGHT_ONEFORMER_COCO_PATH),
            "oneformer_coco",
        )
        clipseg = LazyModel(
            lambda: _net(MODEL_CLIPSEG_PATH, WEIGHT_CLIPSEG_PATH), "clipseg"
        )
        blip_vis = LazyModel(
            lambda: _net(MODEL_BLIP_VIS_PATH, WEIGHT_BLIP_VIS_PATH), "blip_vis"
        )
        blip_dec = LazyModel(
            lambda: _net(MODEL_BLIP_DEC_PATH, WEIGHT_BLIP_DEC_PATH), "blip_dec"
        )
    else:
        import onnxruntime

        sess_options = onnxruntime.SessionOptions()
        sess_options.enable_mem_pattern = False
        providers = [
            ("CUDAExecutionProvider", {"arena_extend_strategy": "kSameAsRequested"}),
            "CPUExecutionProvider",
        ]

        def _sess(path):
            return onnxruntime.InferenceSession(
                path, sess_options=sess_options, providers=providers
            )

        sam_enc = LazyModel(lambda: _sess(WEIGHT_SAM_ENC_PATH), "sam_enc")
        sam_dec = LazyModel(lambda: _sess(WEIGHT_SAM_DEC_PATH), "sam_dec")
        clip_vis = LazyModel(lambda: _sess(WEIGHT_CLIP_VIS_PATH), "clip_vis")
        clip_txt = LazyModel(lambda: _sess(WEIGHT_CLIP_TXT_PATH), "clip_txt")
        oneformer_ade20k = LazyModel(
            lambda: _sess(WEIGHT_ONEFORMER_ADE20K_PATH), "oneformer_ade20k"
        )
        oneformer_coco = LazyModel(
            lambda: _sess(WEIGHT_ONEFORMER_COCO_PATH), "oneformer_coco"
        )
        clipseg = LazyModel(lambda: _sess(WEIGHT_CLIPSEG_PATH), "clipseg")
        blip_vis = LazyModel(lambda: _sess(WEIGHT_BLIP_VIS_PATH), "blip_vis")
        blip_dec = LazyModel(lambda: _sess(WEIGHT_BLIP_DEC_PATH), "blip_dec")

    blip_tokenizer = BertTokenizerFast.from_pretrained("./blip_tokenizer")
    clip_tokenizer = CLIPTokenizerFast.from_pretrained("./clip_tokenizer")

    models = {
        "sam_enc": sam_enc,
        "sam_dec": sam_dec,
        "clip_vis": clip_vis,
        "clip_txt": clip_txt,
        "oneformer_ade20k": oneformer_ade20k,
        "oneformer_coco": oneformer_coco,
        "clipseg": clipseg,
        "blip_vis": blip_vis,
        "blip_dec": blip_dec,
        "blip_tokenizer": blip_tokenizer,
        "clip_tokenizer": clip_tokenizer,
    }

    recognize_from_image(models)


if __name__ == "__main__":
    main()
