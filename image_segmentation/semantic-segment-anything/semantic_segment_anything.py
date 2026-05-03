import itertools
import math
import sys
import time
from logging import getLogger

import ailia
import cv2
import numpy as np
from PIL import Image
from tqdm import tqdm

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

SAM_TARGET = 1024

# SAM auto-mask generation parameters
POINTS_PER_SIDE = 64
PRED_IOU_THRESH = 0.86
STABILITY_SCORE_THRESH = 0.92
STABILITY_SCORE_OFFSET = 1.0
CROP_N_LAYERS = 1
CROP_N_POINTS_DOWNSCALE_FACTOR = 2
MIN_MASK_REGION_AREA = 100
BOX_NMS_THRESH = 0.7
CROP_NMS_THRESH = 0.7

# Segformer native training resolution per dataset
SEGFORMER_SIZE = {"ade20k": (640, 640), "cityscapes": (1024, 1024)}

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

CITYSCAPES_CLASSES = [
    'road', 'sidewalk', 'building', 'wall', 'fence', 'pole',
    'traffic light', 'traffic sign', 'vegetation', 'terrain', 'sky',
    'person', 'rider', 'car', 'truck', 'bus', 'train', 'motorcycle', 'bicycle',
]
# fmt: on

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
# Visualization
# ======================


def draw_result(img_bgr, semantc_mask, class_names):
    """Replicate mmdet imshow_det_bboxes for semantic masks.

    Matches the original pipeline.py call:
        imshow_det_bboxes(img, bboxes=None, labels=np.arange(N),
                          segms=np.stack(bitmasks), class_names=names,
                          font_size=25, show=False, out_file=...)
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.collections import PatchCollection
    from matplotlib.patches import Polygon

    EPS = 1e-2

    # Build per-class bitmasks in torch.unique order (sorted ascending)
    unique_class_ids = np.unique(semantc_mask)
    N = len(unique_class_ids)

    semantic_bitmasks = []
    semantic_class_names_local = []
    for class_id in unique_class_ids:
        semantic_bitmasks.append((semantc_mask == class_id).astype(np.uint8))
        semantic_class_names_local.append(class_names[int(class_id)])

    # mask colors: get_palette(None, N) → np.random.seed(42), randint(0,256,(N,3))
    state = np.random.get_state()
    np.random.seed(42)
    colors = np.random.randint(0, 256, size=(N, 3)).astype(np.uint8)
    np.random.set_state(state)

    # text color: get_palette('green', N) → mmcv.color_val('green')=(0,255,0) BGR
    # [::-1] → (0,255,0) RGB, palette_val divides by 255 → (0, 1.0, 0)
    text_color_mpl = (0.0, 1.0, 0.0)

    # Convert BGR→RGB (imshow_det_bboxes calls mmcv.bgr2rgb at start)
    img_rgb = img_bgr[:, :, ::-1].copy().astype(np.uint8)
    height, width = img_rgb.shape[:2]

    # draw_masks: alpha=0.8, with_edge=True
    # taken_colors = set([0, 0, 0]) = {0} in Python (a set with one int)
    # tuple(color_mask) is never in {0}, so color bias never fires
    taken_colors = {0}
    polygons = []
    for i, mask in enumerate(semantic_bitmasks):
        # bitmap_to_polygon equivalent
        contours, _ = cv2.findContours(
            np.ascontiguousarray(mask), cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE
        )
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

    # Text labels: centroid of largest connected component, adaptive font size
    for i, (mask, name) in enumerate(
        zip(semantic_bitmasks, semantic_class_names_local)
    ):
        _, _, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)
        if len(stats) < 2:
            continue
        largest_id = int(np.argmax(stats[1:, -1])) + 1
        pos = centroids[largest_id]  # (x, y)
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
    import cv2

    assert mode in ["holes", "islands"]
    correct_holes = mode == "holes"
    working_mask = (correct_holes ^ mask).astype(np.uint8)
    n_labels, regions, stats, _ = cv2.connectedComponentsWithStats(working_mask, 8)
    sizes = stats[:, -1][1:]  # row 0 is background label
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
        suppress_mask[0] = False  # don't suppress self (order[0] == i)
        suppressed[order[suppress_mask]] = True
    return kept


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
# Segformer preprocessing
# ======================


def preprocess_segformer(img_rgb, model_type):
    """Resize with PIL BILINEAR (matching SegformerFeatureExtractor) and normalize."""
    target_h, target_w = SEGFORMER_SIZE[model_type]
    img = np.array(
        Image.fromarray(img_rgb).resize((target_w, target_h), Image.BILINEAR)
    )
    img = normalize_image(img, "ImageNet")
    img = img.transpose(2, 0, 1)[None].astype(np.float32)
    return img


def resize_logits_align_corners(logits, out_h, out_w):
    """
    Upsample (C, in_h, in_w) logits to (C, out_h, out_w) using bilinear
    interpolation with align_corners=True, matching:
        F.interpolate(logits, size=(h,w), mode='bilinear', align_corners=True)
    """
    c, in_h, in_w = logits.shape
    # align_corners=True coordinate mapping: i * (in-1)/(out-1)
    y = np.arange(out_h, dtype=np.float32) * ((in_h - 1) / max(out_h - 1, 1))
    x = np.arange(out_w, dtype=np.float32) * ((in_w - 1) / max(out_w - 1, 1))
    y0 = np.floor(y).astype(np.int64).clip(0, in_h - 1)
    x0 = np.floor(x).astype(np.int64).clip(0, in_w - 1)
    y1 = (y0 + 1).clip(0, in_h - 1)
    x1 = (x0 + 1).clip(0, in_w - 1)
    wy = (y - y0).astype(np.float32)  # (out_h,)
    wx = (x - x0).astype(np.float32)  # (out_w,)
    # weights: (out_h, out_w)
    wa = (1 - wy)[:, None] * (1 - wx)[None, :]
    wb = (1 - wy)[:, None] * wx[None, :]
    wc = wy[:, None] * (1 - wx)[None, :]
    wd = wy[:, None] * wx[None, :]
    out = (
        logits[:, y0[:, None], x0[None, :]] * wa
        + logits[:, y0[:, None], x1[None, :]] * wb
        + logits[:, y1[:, None], x0[None, :]] * wc
        + logits[:, y1[:, None], x1[None, :]] * wd
    )
    return out


# ======================
# Segformer inference
# ======================


def run_segformer(models, img):
    """Returns (H, W) class index map at original image resolution."""
    im_h, im_w = img.shape[:2]
    img_tensor = preprocess_segformer(img, args.model_type)

    net = models["segformer"]
    if not args.onnx:
        output = net.predict([img_tensor])
    else:
        output = net.run(None, {"pixel_values": img_tensor})
    logits = output[0]
    logits = logits[0]  # (C, logit_h, logit_w)

    # Upsample logits to original size BEFORE argmax (matching segformer.py:
    #   F.interpolate(logits, size=(h,w), mode='bilinear', align_corners=True))
    logits_up = resize_logits_align_corners(logits, im_h, im_w)
    class_ids = np.argmax(logits_up, axis=0).astype(np.uint8)

    return class_ids


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
    Run SAM encoder + decoder on one crop. Returns (bin_masks, scores, boxes).
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
    mask_list = []
    score_list = []
    box_list = []

    BATCH_SIZE = 64
    for i in tqdm(range(0, len(pts_sam), BATCH_SIZE), desc="Processing batches"):
        batch_pts = pts_sam[i : i + BATCH_SIZE]  # (B, 2)
        B = len(batch_pts)
        coords = batch_pts[:, None, :].astype(np.float32)  # (B, 1, 2)
        labels = np.ones((B, 1), dtype=np.float32)  # (B, 1) foreground
        emb = np.repeat(image_embedding, B, axis=0)  # (B, C, H, W)

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
        batch_masks, batch_iou_preds = output  # (B*3, crop_h, crop_w), (B, 3)

        masks = batch_masks.reshape(
            -1, batch_masks.shape[-2], batch_masks.shape[-1]
        )  # (B*3, crop_h, crop_w)
        iou_preds = batch_iou_preds.reshape(-1)  # (B*3,)

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
        crop_bool = masks > 0  # (M, crop_h, crop_w)
        boxes_crop = masks_to_boxes(crop_bool)  # (M, 4) in crop coords

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

    masks = np.array(mask_list)  # (N, im_h, im_w)
    scores = np.array(score_list)  # (N,)
    boxes = np.array(box_list)  # (N, 4)

    if len(masks) == 0:
        return masks, scores, boxes

    # NMS
    if len(masks) > 1:
        kept_idx = box_nms(boxes, scores, BOX_NMS_THRESH)
        masks = masks[kept_idx]
        scores = scores[kept_idx]
        boxes = boxes[kept_idx]

    return masks, scores, boxes


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
    all_boxes = []
    all_crop_boxes = []
    for crop_box, layer_idx in zip(crop_boxes, layer_idxs):
        logger.info(f"  Crop {crop_box}, layer {layer_idx}")
        masks, scores, boxes = process_crop(
            models, img_rgb, crop_box, point_grids[layer_idx], im_h, im_w
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
# Main
# ======================


def predict(models, img):
    class_names = ADE20K_CLASSES if args.model_type == "ade20k" else CITYSCAPES_CLASSES

    logger.info("Running Segformer...")
    class_ids = run_segformer(models, img)  # (H, W)

    # Start semantic mask from Segformer result
    semantc_mask = class_ids.copy().astype(np.int32)

    logger.info("Generating SAM masks...")
    masks, scores, boxes = generate_all_masks(models, img)
    masks, scores, boxes = postprocess_small_regions(
        masks,
        scores,
        boxes,
        MIN_MASK_REGION_AREA,
        max(BOX_NMS_THRESH, CROP_NMS_THRESH),
    )
    logger.info(f"{len(masks)} masks after filtering")

    # Sort by area descending (largest first)
    if masks:
        areas = np.array([m.sum() for m in masks])
        order = np.argsort(areas)[::-1]
        masks = [masks[i] for i in order]

    # Vote class per SAM mask and update semantic_mask
    for mask in masks:
        if not mask.any():
            continue
        propose_classes = class_ids[mask].astype(np.int32)
        unique_classes = np.unique(propose_classes)
        if len(unique_classes) == 1:
            semantc_mask[mask] = unique_classes[0]
        else:
            top1 = int(np.bincount(propose_classes).argmax())
            semantc_mask[mask] = top1

    return semantc_mask, class_names


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
