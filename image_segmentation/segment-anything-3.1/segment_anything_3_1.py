import gc
import glob
import os
import sys
import tempfile
import time
from logging import getLogger

import ailia
import cv2
import numpy as np
from PIL import Image
from tqdm import tqdm

sys.path.append("../../util")
from arg_utils import get_base_parser, get_savepath, update_parser
from detector_utils import hsv_to_rgb, load_image
from math_utils import sigmoid
from model_utils import check_and_download_file, check_and_download_models
from simple_tokenizer import SimpleTokenizer
from webcamera_utils import get_capture, get_writer

logger = getLogger(__name__)

# ======================
# Parameters
# ======================

# Grounding models (image + video init)
WEIGHT_ENC_PATH = "sam3.1_image_encoder.onnx"
WEIGHT_ENC_OPT_PATH = "sam3.1_image_encoder.opt.onnx"
WEIGHT_GND_PATH = "sam3.1_grounding.onnx"
MODEL_ENC_PATH = "sam3.1_image_encoder.onnx.prototxt"
MODEL_ENC_OPT_PATH = "sam3.1_image_encoder.opt.onnx.prototxt"
MODEL_GND_PATH = "sam3.1_grounding.onnx.prototxt"

# Tracker models (video mode)
WEIGHT_PE_PATH = "sam3.1_prompt_encoder.onnx"
WEIGHT_DEC_PATH = "sam3.1_mask_decoder.onnx"
WEIGHT_TDEC_PATH = (
    "sam3.1_tracking_mask_decoder.onnx"  # MultiplexMaskDecoder for tracking
)
WEIGHT_MENC_PATH = "sam3.1_memory_encoder.onnx"
WEIGHT_MATTN_PATH = "sam3.1_memory_attention.onnx"

WEIGHT_PROJ_PATH = "sam3.1_obj_ptr_proj.onnx"
WEIGHT_IPROJ_PATH = "sam3.1_interactive_obj_ptr_proj.onnx"  # for init frame
WEIGHT_TPOS_PATH = "sam3.1_obj_ptr_tpos_proj.onnx"
MODEL_PE_PATH = "sam3.1_prompt_encoder.onnx.prototxt"
MODEL_DEC_PATH = "sam3.1_mask_decoder.onnx.prototxt"
MODEL_TDEC_PATH = "sam3.1_tracking_mask_decoder.onnx.prototxt"
MODEL_MENC_PATH = "sam3.1_memory_encoder.onnx.prototxt"
MODEL_MATTN_PATH = "sam3.1_memory_attention.onnx.prototxt"
MODEL_PROJ_PATH = "sam3.1_obj_ptr_proj.onnx.prototxt"
MODEL_IPROJ_PATH = "sam3.1_interactive_obj_ptr_proj.onnx.prototxt"
MODEL_TPOS_PATH = "sam3.1_obj_ptr_tpos_proj.onnx.prototxt"

TPOS_ENC_PATH = "npy/sam3.1_maskmem_tpos_enc.npy"
NO_OBJ_EMBED_PATH = "npy/sam3.1_no_obj_embed_spatial.npy"
NO_OBJ_PTR_LINEAR_W_PATH = "npy/sam3.1_no_obj_ptr_linear_weight.npy"
NO_OBJ_PTR_LINEAR_B_PATH = "npy/sam3.1_no_obj_ptr_linear_bias.npy"
VALID_EMBED_PATH = "npy/sam3.1_output_valid_embed.npy"

BPE_PATH = "bpe_simple_vocab_16e6.txt.gz"
REMOTE_PATH = "https://storage.googleapis.com/ailia-models/segment-anything-3.1/"

IMAGE_PATH = "test_image.jpg"
SAVE_IMAGE_PATH = "output.png"

IMAGE_SIZE = 1008
CONTEXT_LENGTH = 32
CONFIDENCE_THRESHOLD = 0.5

# Tracker constants
MEMORY_MASK_SIZE = 1152  # interpol_size for memory_encoder masks
MASK_CHANNELS = 32  # multiplex_count(16) × 2
HW = 5184  # 72 × 72
NUM_MASKMEM = 7  # 1 conditioning + 6 non-cond frames
MAX_OBJ_PTRS = 16
MULTIPLEX_COUNT = 16  # number of object slots per bucket (SAM 3.1 multiplex)
SIGMOID_SCALE_FOR_MEM_ENC = 2.0  # scale applied after sigmoid in memory encoder mask
SIGMOID_BIAS_FOR_MEM_ENC = -1.0  # bias applied after sigmoid in memory encoder mask
NO_OBJ_SCORE = -1024.0  # mask logit used when object is not appearing
OBJ_SCORE_THRESHOLD = 0  # obj_score threshold: below this the object is not appearing

# ======================
# Argument Parser Config
# ======================

parser = get_base_parser("SAM3.1", IMAGE_PATH, SAVE_IMAGE_PATH, fp16_support=False)
parser.add_argument(
    "--caption",
    type=str,
    default="shoe",
    help="Text prompt for grounded segmentation.",
)
parser.add_argument(
    "--threshold",
    type=float,
    default=CONFIDENCE_THRESHOLD,
    help="Confidence threshold for instance selection.",
)
parser.add_argument(
    "--box",
    nargs=4,
    type=float,
    metavar=("X1", "Y1", "X2", "Y2"),
    action="append",
    help="Bounding box prompt in pixel coords [x1 y1 x2 y2].",
)
parser.add_argument(
    "--box_label",
    nargs="+",
    type=int,
    default=None,
    metavar="LABEL",
    help="Label per --box (1=positive, 0=negative). Defaults to all positive.",
)
parser.add_argument(
    "--point",
    nargs=2,
    type=float,
    metavar=("X", "Y"),
    action="append",
    help="Point prompt in pixel coords [x y] for video tracking. Can be specified multiple times.",
)
parser.add_argument(
    "--point_label",
    nargs="+",
    type=int,
    default=None,
    metavar="LABEL",
    help="Label per --point (1=positive, 0=negative). Defaults to all positive.",
)
parser.add_argument("--onnx", action="store_true", help="execute onnxruntime version.")
parser.add_argument(
    "--normal",
    action="store_true",
    help="Use the non-optimized image encoder (sam3.1_image_encoder.onnx) instead of the .opt variant.",
)
parser.add_argument(
    "--tracking",
    action="store_true",
    help="Enable SAM3.1 memory-based tracking mode (requires --video).",
)
args = update_parser(parser)


# ======================
# Classes
# ======================


class FrameDirCapture:
    """cv2.VideoCapture-compatible wrapper for a directory of image files.

    Files are sorted by name, so zero-padded filenames (e.g. 000001.jpg) work correctly.
    """

    _EXTS = ("*.jpg", "*.jpeg", "*.png", "*.bmp", "*.JPG", "*.JPEG", "*.PNG", "*.BMP")

    def __init__(self, dir_path):
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
        if frame is None:
            return False, None
        return True, frame

    def get(self, prop_id):
        if prop_id == cv2.CAP_PROP_FRAME_HEIGHT:
            return float(self._h)
        if prop_id == cv2.CAP_PROP_FRAME_WIDTH:
            return float(self._w)
        if prop_id == cv2.CAP_PROP_FRAME_COUNT:
            return float(len(self._files))
        return 0.0

    def release(self):
        pass


class LazyModel:
    """Defers model loading until the first predict/run call."""

    def __init__(self, loader_fn, name=""):
        self._loader_fn = loader_fn
        self._name = name
        self._net = None

    def load(self):
        if self._net is None:
            if not args.tracking:
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
# Tokenizer
# ======================

_tokenizer = None


def get_tokenizer():
    global _tokenizer
    if _tokenizer is None:
        _tokenizer = SimpleTokenizer(BPE_PATH)
    return _tokenizer


def tokenize(text):
    tok = get_tokenizer()
    sot = tok.encoder["<|startoftext|>"]
    eot = tok.encoder["<|endoftext|>"]
    tokens = [sot] + tok.encode(text) + [eot]
    result = np.zeros((1, CONTEXT_LENGTH), dtype=np.int64)
    n = min(len(tokens), CONTEXT_LENGTH)
    result[0, :n] = tokens[:n]
    if len(tokens) > CONTEXT_LENGTH:
        result[0, CONTEXT_LENGTH - 1] = eot
    return result


# ======================
# Visualization
# ======================


def show_mask(mask, img, color):
    color = np.array(color[:3], dtype=np.uint8).reshape(1, 1, 3)
    h, w = mask.shape
    mask_3d = mask.reshape(h, w, 1).astype(np.float32)
    masked = img.astype(np.float32) * (1 - mask_3d * 0.4) + mask_3d * color * 0.4
    return np.clip(masked, 0, 255).astype(np.uint8)


def draw_predictions(image, boxes, scores, masks, label, obj_ids=None):
    n = len(scores)
    if obj_ids is not None:
        colors = [
            hsv_to_rgb(int(256 * ((int(obj_id) - 1) % 32) / 32), 200, 200)
            for obj_id in obj_ids
        ]
    else:
        colors = [hsv_to_rgb(int(256 * i / max(n, 1)), 200, 200) for i in range(n)]

    for i in range(n):
        if i < len(masks):
            image = show_mask(masks[i], image, colors[i])

    for i, (box, score) in enumerate(zip(boxes, scores)):
        x1, y1, x2, y2 = box.astype(int)
        color = colors[i][:3]
        cv2.rectangle(image, (x1, y1), (x2, y2), color=color, thickness=2)
        if obj_ids is not None:
            text = f"(id={obj_ids[i]})"
        else:
            text = f"{label}: {score:.2f}"
        cv2.putText(
            image,
            text,
            (x1, max(y1 - 5, 12)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            color,
            1,
            cv2.LINE_AA,
        )

    return image


def render_frame(
    frame_path,
    scores,
    det_boxes,
    masks,
    bank_idxs,
    display_label,
    writer,
    has_display,
    frame_shown_ref,
):
    """Load frame, overlay predictions, display/write."""
    frame = cv2.imread(frame_path)
    obj_ids = [i + 1 for i in bank_idxs]
    res_img = frame.copy()
    res_img = draw_predictions(
        res_img, det_boxes, scores, masks, display_label, obj_ids=obj_ids
    )
    if has_display:
        cv2.imshow("frame", res_img)
        frame_shown_ref[0] = True
    if writer is not None:
        writer.write(res_img.astype(np.uint8))


def collect_frame_paths(video_src):
    """
    Returns sorted list of frame file paths.
    For directories: JPEG/PNG files sorted by numeric stem.
    For video files: decodes all frames to a temp directory.
    """
    if os.path.isdir(video_src):
        paths = sorted(
            [
                p
                for p in glob.glob(os.path.join(video_src, "*"))
                if p.lower().endswith((".jpg", ".jpeg", ".png"))
            ],
            key=lambda p: (
                int(os.path.splitext(os.path.basename(p))[0])
                if os.path.splitext(os.path.basename(p))[0].isdigit()
                else p
            ),
        )
        if not paths:
            raise FileNotFoundError(f"No JPEG/PNG frames found in {video_src}")
        return paths

    # Video file: decode to temp directory
    cap = get_capture(video_src)
    assert cap.isOpened(), f"Cannot open video: {video_src}"
    tmpdir = tempfile.mkdtemp(prefix="sam3p1_frames_")
    frame_idx = 0
    paths = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        p = os.path.join(tmpdir, f"{frame_idx:05d}.jpg")
        cv2.imwrite(p, frame)
        paths.append(p)
        frame_idx += 1
    cap.release()
    logger.info("Decoded %d frames to %s", len(paths), tmpdir)
    return paths


# ======================
# Secondary Functions
# ======================


def build_box_inputs(boxes, box_labels, orig_h, orig_w):
    """Convert pixel [x1,y1,x2,y2] boxes to the tensors expected by sam3.1_grounding.onnx."""
    n = len(boxes)
    coords = np.array(boxes, dtype=np.float32)  # (N, 4) [x1,y1,x2,y2]
    cx = (coords[:, 0] + coords[:, 2]) / 2.0 / orig_w
    cy = (coords[:, 1] + coords[:, 3]) / 2.0 / orig_h
    bw = (coords[:, 2] - coords[:, 0]) / orig_w
    bh = (coords[:, 3] - coords[:, 1]) / orig_h
    cxcywh = np.stack([cx, cy, bw, bh], axis=-1)  # (N, 4)

    box_coords = cxcywh[:, np.newaxis, :]  # (N, 1, 4)

    if box_labels is None:
        lbls = np.ones(n, dtype=np.int64)
    else:
        lbls = np.array(box_labels[:n], dtype=np.int64)
    box_labels_arr = lbls[:, np.newaxis]  # (N, 1)

    box_mask = np.zeros((1, n), dtype=bool)

    return box_coords, box_labels_arr, box_mask


# ======================
# Grounding functions
# ======================


def preprocess(img):
    img_rgb_u8 = img[:, :, ::-1]  # BGR→RGB, uint8
    pil_img = Image.fromarray(img_rgb_u8)
    pil_resized = pil_img.resize((IMAGE_SIZE, IMAGE_SIZE), Image.BILINEAR)
    # /255 → float16 storage (quantize) → float16 normalize → float32
    img_f16 = (np.array(pil_resized, dtype=np.float32) / 255.0).astype(np.float16)
    img_norm = ((img_f16 - np.float16(0.5)) / np.float16(0.5)).astype(np.float32)
    img_chw = img_norm.transpose(2, 0, 1)[None]  # [1,3,H,W]
    return img_chw


def postprocess(
    pred_masks, pred_boxes, pred_logits, presence_logit_dec, orig_h, orig_w, threshold
):
    out_probs = sigmoid(pred_logits[0, :, 0])  # [200]
    presence_score = sigmoid(presence_logit_dec[0, 0])
    out_probs = out_probs * presence_score

    keep = out_probs > threshold
    scores = out_probs[keep]
    masks_raw = pred_masks[0][keep]  # [K, mask_h, mask_w]
    boxes_cxcywh = pred_boxes[0][keep]  # [K, 4]

    if len(scores) == 0:
        empty = np.zeros((0, orig_h, orig_w), dtype=bool)
        return scores, np.zeros((0, 4)), empty

    cx, cy, bw, bh = (
        boxes_cxcywh[:, 0],
        boxes_cxcywh[:, 1],
        boxes_cxcywh[:, 2],
        boxes_cxcywh[:, 3],
    )
    xyxy = np.stack([cx - bw / 2, cy - bh / 2, cx + bw / 2, cy + bh / 2], axis=-1)
    xyxy = xyxy * np.array([orig_w, orig_h, orig_w, orig_h])

    resized_masks = np.stack(
        [
            sigmoid(cv2.resize(m, (orig_w, orig_h), interpolation=cv2.INTER_LINEAR))
            > 0.5
            for m in masks_raw
        ]
    )

    return scores, xyxy, resized_masks


def run_encoder(models, img_input):
    """Run image encoder.

    Returns 10 outputs:
      [0] fpn0,      [1] fpn1,      [2] fpn2      — detection path (for grounding)
      [3] pos0,      [4] pos1,      [5] pos2
      [6] prop_fpn0, [7] prop_fpn1, [8] prop_fpn2 — propagation path (for tracking)
      [9] prop_pos2
    """
    encoder = models["encoder"]
    if not args.onnx:
        enc_out = encoder.predict([img_input])
    else:
        enc_out = encoder.run(None, {"image": img_input})
    return enc_out


def run_grounding(
    models, fpn0, fpn1, fpn2, pos2, text_tokens, box_coords, box_labels_arr, box_mask
):
    """Run grounding model."""
    # In opt+ailia mode, empty (0-length) blobs are not supported by MatMul,
    # so substitute a single dummy box marked as padding via box_mask=True.
    # In --normal mode or --onnx (ORT) mode, keep the original empty inputs.
    if (
        not args.normal
        and not args.onnx
        and box_coords.shape[0] == 0
    ):
        box_coords = np.zeros((1, 1, 4), dtype=np.float32)
        box_labels_arr = np.zeros((1, 1), dtype=np.int64)
        box_mask = np.ones((1, 1), dtype=bool)

    grounder = models["grounder"]
    if not args.onnx:
        gnd_out = grounder.predict(
            [fpn0, fpn1, fpn2, pos2, text_tokens, box_coords, box_labels_arr, box_mask]
        )
    else:
        gnd_out = grounder.run(
            None,
            {
                "fpn0": fpn0,
                "fpn1": fpn1,
                "fpn2": fpn2,
                "pos2": pos2,
                "text_tokens": text_tokens,
                "box_coords": box_coords,
                "box_labels": box_labels_arr,
                "box_mask": box_mask,
            },
        )
    return gnd_out  # pred_masks, pred_boxes, pred_logits, presence_logit_dec


def predict(models, img, caption, threshold, boxes=None, box_labels=None):
    orig_h, orig_w = img.shape[:2]
    img_input = preprocess(img)

    use_visual = boxes is not None and len(boxes) > 0

    if use_visual:
        text_tokens = tokenize("visual")
        box_coords, box_labels_arr, box_mask = build_box_inputs(
            boxes, box_labels, orig_h, orig_w
        )
    else:
        text_tokens = tokenize(caption)
        box_coords = np.zeros((0, 1, 4), dtype=np.float32)
        box_labels_arr = np.zeros((0, 1), dtype=np.int64)
        box_mask = np.zeros((1, 0), dtype=bool)

    enc_out = run_encoder(models, img_input)
    fpn0, fpn1, fpn2, pos0, pos1, pos2, *_ = enc_out

    gnd_out = run_grounding(
        models,
        fpn0,
        fpn1,
        fpn2,
        pos2,
        text_tokens,
        box_coords,
        box_labels_arr,
        box_mask,
    )
    pred_masks, pred_boxes, pred_logits, presence_logit_dec = gnd_out

    return postprocess(
        pred_masks,
        pred_boxes,
        pred_logits,
        presence_logit_dec,
        orig_h,
        orig_w,
        threshold,
    )


# ======================
# Main functions
# ======================


def recognize_from_image(models):
    for key in ("encoder", "grounder"):
        m = models.get(key)
        if m is not None and hasattr(m, "load"):
            m.load()

    caption = args.caption
    threshold = args.threshold
    boxes = args.box
    box_labels = args.box_label

    use_visual = boxes is not None and len(boxes) > 0
    display_label = "visual" if use_visual else caption

    if use_visual:
        logger.info(
            "Visual prompt: %d box(es), labels=%s",
            len(boxes),
            box_labels if box_labels else "[all positive]",
        )
    else:
        logger.info("Caption: '%s'  threshold: %s", caption, threshold)

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
                output = predict(models, img, caption, threshold, boxes, box_labels)
                end = int(round(time.time() * 1000))
                elapsed = end - start
                logger.info(f"\tailia processing estimation time {elapsed} ms")
                if i != 0:
                    total_time += elapsed
            logger.info(
                f"\taverage time estimation {total_time / (args.benchmark_count - 1)} ms"
            )
        else:
            output = predict(models, img, caption, threshold, boxes, box_labels)

        scores, det_boxes, masks = output
        logger.info(f"Detected {len(scores)} instance(s)")

        res_img = img.copy()
        res_img = draw_predictions(res_img, det_boxes, scores, masks, display_label)

        savepath = get_savepath(args.savepath, image_path, ext=".png")
        cv2.imwrite(savepath, res_img)
        logger.info(f"saved at : {savepath}")

    logger.info("Script finished successfully.")


def recognize_from_video(models):
    """Per-frame grounding on video — runs predict() independently for each frame."""
    for key in ("encoder", "grounder"):
        m = models.get(key)
        if m is not None and hasattr(m, "load"):
            m.load()

    caption = args.caption
    threshold = args.threshold
    boxes = args.box
    box_labels = args.box_label

    use_visual = boxes is not None and len(boxes) > 0
    display_label = "visual" if use_visual else caption

    video_file = args.video if args.video else args.input[0]
    if os.path.isdir(video_file):
        capture = FrameDirCapture(video_file)
    else:
        capture = get_capture(video_file)
    assert capture.isOpened(), "Cannot capture source"

    if args.savepath != SAVE_IMAGE_PATH:
        f_h = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        f_w = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        writer = get_writer(args.savepath, f_h, f_w)
    else:
        writer = None

    frame_shown = False
    while True:
        ret, frame = capture.read()
        if (cv2.waitKey(1) & 0xFF == ord("q")) or not ret:
            break
        if frame_shown and cv2.getWindowProperty("frame", cv2.WND_PROP_VISIBLE) == 0:
            break

        scores, det_boxes, masks = predict(
            models, frame, caption, threshold, boxes, box_labels
        )

        res_img = frame.copy()
        res_img = draw_predictions(res_img, det_boxes, scores, masks, display_label)

        cv2.imshow("frame", res_img)
        frame_shown = True

        if writer is not None:
            writer.write(res_img.astype(np.uint8))

    capture.release()
    cv2.destroyAllWindows()
    if writer is not None:
        writer.release()

    logger.info("Script finished successfully.")


def recognize_from_tracking(models):
    """SAM3.1 memory-based multi-person tracking.

    Stage 1: add_prompt (frame 0)
    Stage 2: propagate_in_video forward (frames 1 → N)
    Stage 3: propagate_in_video reverse (frames N → 0, stub)
    """
    from video_tracking import (  # avoid circular import (video_tracking imports sam3p1)
        Sam3Tracker,
    )

    video_src = args.video if args.video else args.input[0]
    try:
        int(video_src)
        logger.error("Webcam input not supported in tracking mode.")
        sys.exit(1)
    except (ValueError, TypeError):
        pass

    if args.point:
        logger.info("Tracking init: point prompt — %d point(s)", len(args.point))
        display_label = "tracked"
    elif args.box:
        logger.info("Tracking init: box prompt")
        display_label = "tracked"
    else:
        logger.info("Tracking init: text grounding — caption='%s'", args.caption)
        display_label = args.caption

    maskmem_tpos_enc = np.load(TPOS_ENC_PATH)
    no_obj_params = (
        np.load(NO_OBJ_EMBED_PATH),
        np.load(NO_OBJ_PTR_LINEAR_W_PATH),
        np.load(NO_OBJ_PTR_LINEAR_B_PATH),
    )

    logger.info("Loading frame paths from %s …", video_src)
    frame_paths = collect_frame_paths(video_src)
    logger.info("%d frames found", len(frame_paths))

    frame0_img = cv2.imread(frame_paths[0])
    f_h, f_w = frame0_img.shape[:2]
    writer = (
        get_writer(args.savepath, f_h, f_w)
        if args.savepath != SAVE_IMAGE_PATH
        else None
    )
    saving_to_file = args.savepath != SAVE_IMAGE_PATH
    has_display = bool(os.environ.get("DISPLAY")) and not saving_to_file
    frame_shown = [False]

    pbar = tqdm(total=len(frame_paths), desc="Tracking", unit="frame")

    # Stage 1: add_prompt (frame 0)
    tracker = Sam3Tracker(
        models, maskmem_tpos_enc, no_obj_params, threshold=args.threshold
    )
    if args.point or args.box:
        scores0, boxes0, masks0 = tracker.add_prompt_interactive(
            frame0_img,
            points=args.point,
            point_labels=args.point_label,
            box=args.box[0] if args.box else None,
        )
    else:
        scores0, boxes0, masks0 = tracker.add_prompt(frame0_img, args.caption)

    bank_idxs0 = list(range(1, len(tracker.memory_banks) + 1))
    render_frame(
        frame_paths[0],
        scores0,
        boxes0,
        masks0,
        bank_idxs0,
        display_label,
        writer,
        has_display,
        frame_shown,
    )
    pbar.update(1)

    # Stage 2: propagate_in_video (forward: frames 1 → N)
    for frame_idx, scores, det_boxes, masks, obj_ids in tracker.propagate_in_video(
        frame_paths, start_frame=1
    ):
        if has_display and (cv2.waitKey(1) & 0xFF == ord("q")):
            break
        if (
            has_display
            and frame_shown[0]
            and cv2.getWindowProperty("frame", cv2.WND_PROP_VISIBLE) == 0
        ):
            break

        render_frame(
            frame_paths[frame_idx],
            scores,
            det_boxes,
            masks,
            obj_ids,
            display_label,
            writer,
            has_display,
            frame_shown,
        )
        pbar.update(1)

    # Stage 3: reverse propagation (stub)

    pbar.close()
    if has_display:
        cv2.destroyAllWindows()
    if writer is not None:
        writer.release()

    logger.info("Script finished successfully.")


def main():
    weight_enc_path = WEIGHT_ENC_PATH if args.normal else WEIGHT_ENC_OPT_PATH
    model_enc_path = MODEL_ENC_PATH if args.normal else MODEL_ENC_OPT_PATH
    check_and_download_models(weight_enc_path, model_enc_path, REMOTE_PATH)
    check_and_download_models(WEIGHT_GND_PATH, MODEL_GND_PATH, REMOTE_PATH)
    check_and_download_file(BPE_PATH, REMOTE_PATH)

    use_tracking = args.tracking

    if use_tracking:
        for w, m in [
            (WEIGHT_PE_PATH, MODEL_PE_PATH),
            (WEIGHT_DEC_PATH, MODEL_DEC_PATH),
            (WEIGHT_TDEC_PATH, MODEL_TDEC_PATH),
            (WEIGHT_MENC_PATH, MODEL_MENC_PATH),
            (WEIGHT_MATTN_PATH, MODEL_MATTN_PATH),
            (WEIGHT_PROJ_PATH, MODEL_PROJ_PATH),
            (WEIGHT_IPROJ_PATH, MODEL_IPROJ_PATH),
            (WEIGHT_TPOS_PATH, MODEL_TPOS_PATH),
        ]:
            check_and_download_models(w, m, REMOTE_PATH)

    env_id = args.env_id

    if not args.onnx:
        memory_mode = ailia.get_memory_mode(
            reduce_constant=True,
            ignore_input_with_initializer=True,
            reduce_interstage=False,
            reuse_interstage=True,
        )
        encoder = ailia.Net(
            model_enc_path, weight_enc_path, env_id=env_id, memory_mode=memory_mode
        )
        grounder = ailia.Net(
            MODEL_GND_PATH, WEIGHT_GND_PATH, env_id=env_id, memory_mode=memory_mode
        )
        models = dict(encoder=encoder, grounder=grounder)
        if use_tracking:
            models["prompt_enc"] = ailia.Net(
                MODEL_PE_PATH, WEIGHT_PE_PATH, env_id=env_id, memory_mode=memory_mode
            )
            models["mask_dec"] = ailia.Net(
                MODEL_DEC_PATH, WEIGHT_DEC_PATH, env_id=env_id, memory_mode=memory_mode
            )
            models["track_dec"] = ailia.Net(
                MODEL_TDEC_PATH,
                WEIGHT_TDEC_PATH,
                env_id=env_id,
                memory_mode=memory_mode,
            )
            models["mem_enc"] = ailia.Net(
                MODEL_MENC_PATH,
                WEIGHT_MENC_PATH,
                env_id=env_id,
                memory_mode=memory_mode,
            )
            models["mem_attn"] = ailia.Net(
                MODEL_MATTN_PATH,
                WEIGHT_MATTN_PATH,
                env_id=env_id,
                memory_mode=memory_mode,
            )
            models["obj_proj"] = ailia.Net(
                MODEL_PROJ_PATH,
                WEIGHT_PROJ_PATH,
                env_id=env_id,
                memory_mode=memory_mode,
            )
            models["iobj_proj"] = ailia.Net(
                MODEL_IPROJ_PATH,
                WEIGHT_IPROJ_PATH,
                env_id=env_id,
                memory_mode=memory_mode,
            )
            models["tpos_proj"] = ailia.Net(
                MODEL_TPOS_PATH,
                WEIGHT_TPOS_PATH,
                env_id=env_id,
                memory_mode=memory_mode,
            )
    else:
        import onnxruntime

        cuda = 0 < ailia.get_gpu_environment_id()
        providers = (
            ["CUDAExecutionProvider", "CPUExecutionProvider"]
            if cuda
            else ["CPUExecutionProvider"]
        )
        # All models use LazyModel so unused models can be unloaded to free VRAM.
        # memory_attention activation buffers grow with T_mem (up to ~6 GB at T_mem_max).
        # Encoder's BFCArena caches large buffers and accumulates VRAM frame-over-frame,
        # so encoder must also be unloaded before each memory_attention call.
        models = {}
        models["encoder"] = LazyModel(
            lambda: onnxruntime.InferenceSession(weight_enc_path, providers=providers),
            "encoder",
        )
        models["grounder"] = LazyModel(
            lambda: onnxruntime.InferenceSession(WEIGHT_GND_PATH, providers=providers),
            "grounder",
        )
        if use_tracking:
            models["prompt_enc"] = LazyModel(
                lambda: onnxruntime.InferenceSession(
                    WEIGHT_PE_PATH, providers=providers
                ),
                "prompt_enc",
            )
            models["mask_dec"] = LazyModel(
                lambda: onnxruntime.InferenceSession(
                    WEIGHT_DEC_PATH, providers=providers
                ),
                "mask_dec",
            )
            models["track_dec"] = LazyModel(
                lambda: onnxruntime.InferenceSession(
                    WEIGHT_TDEC_PATH, providers=providers
                ),
                "track_dec",
            )
            models["mem_enc"] = LazyModel(
                lambda: onnxruntime.InferenceSession(
                    WEIGHT_MENC_PATH, providers=providers
                ),
                "mem_enc",
            )
            models["mem_attn"] = LazyModel(
                lambda: onnxruntime.InferenceSession(
                    WEIGHT_MATTN_PATH, providers=providers
                ),
                "mem_attn",
            )
            models["obj_proj"] = LazyModel(
                lambda: onnxruntime.InferenceSession(
                    WEIGHT_PROJ_PATH, providers=providers
                ),
                "obj_proj",
            )
            models["iobj_proj"] = LazyModel(
                lambda: onnxruntime.InferenceSession(
                    WEIGHT_IPROJ_PATH, providers=providers
                ),
                "iobj_proj",
            )
            models["tpos_proj"] = LazyModel(
                lambda: onnxruntime.InferenceSession(
                    WEIGHT_TPOS_PATH, providers=providers
                ),
                "tpos_proj",
            )

    if args.tracking:
        recognize_from_tracking(models)
    elif args.video is not None:
        recognize_from_video(models)
    else:
        recognize_from_image(models)


if __name__ == "__main__":
    main()
