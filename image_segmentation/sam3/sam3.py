import sys
import time
from logging import getLogger

import ailia
import cv2
import numpy as np

sys.path.append("../../util")
from arg_utils import get_base_parser, get_savepath, update_parser  # noqa
from detector_utils import hsv_to_rgb, load_image  # noqa
from math_utils import sigmoid  # noqa
from model_utils import check_and_download_file, check_and_download_models  # noqa
from simple_tokenizer import SimpleTokenizer
from webcamera_utils import get_capture, get_writer  # noqa

logger = getLogger(__name__)

# ======================
# Parameters
# ======================

WEIGHT_ENC_PATH = "sam3_image_encoder.onnx"
WEIGHT_GND_PATH = "sam3_grounding.onnx"
WEIGHT_GND_VIS_PATH = "sam3_grounding_visual.onnx"
MODEL_ENC_PATH = "sam3_image_encoder.onnx.prototxt"
MODEL_GND_PATH = "sam3_grounding.onnx.prototxt"
MODEL_GND_VIS_PATH = "sam3_grounding_visual.onnx.prototxt"
DATA_ENC_PATH = "sam3_image_encoder_weights.pb"
DATA_GND_PATH = "sam3_grounding_weights.pb"
DATA_GND_VIS_PATH = "sam3_grounding_visual_weights.pb"
BPE_PATH = "bpe_simple_vocab_16e6.txt.gz"
REMOTE_PATH = "https://storage.googleapis.com/ailia-models/sam3/"

IMAGE_PATH = "test_image.jpg"
SAVE_IMAGE_PATH = "output.png"

IMAGE_SIZE = 1008
CONTEXT_LENGTH = 32
CONFIDENCE_THRESHOLD = 0.5
MAX_BOXES = 8  # fixed slot count in sam3_grounding_visual.onnx

# ======================
# Argument Parser Config
# ======================

parser = get_base_parser("SAM3", IMAGE_PATH, SAVE_IMAGE_PATH, fp16_support=False)
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
    help=(
        "Bounding box visual prompt in pixel coords [x1 y1 x2 y2]. "
        "Can be specified up to %d times." % MAX_BOXES
    ),
)
parser.add_argument(
    "--box_label",
    nargs="+",
    type=int,
    default=None,
    metavar="LABEL",
    help="Label per --box (1=positive, 0=negative). Defaults to all positive.",
)
parser.add_argument("--onnx", action="store_true", help="execute onnxruntime version.")
args = update_parser(parser)

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
# Secondary Functions
# ======================


def show_mask(mask, img, color):
    color = np.array(color[:3], dtype=np.uint8).reshape(1, 1, 3)
    h, w = mask.shape
    mask_3d = mask.reshape(h, w, 1).astype(np.float32)
    masked = img.astype(np.float32) * (1 - mask_3d * 0.4) + mask_3d * color * 0.4
    return np.clip(masked, 0, 255).astype(np.uint8)


def draw_predictions(image, boxes, scores, masks, label):
    n = len(scores)
    colors = [hsv_to_rgb(int(256 * i / max(n, 1)), 200, 200) for i in range(n)]

    for i in range(n):
        if i < len(masks):
            image = show_mask(masks[i], image, colors[i])

    for i, (box, score) in enumerate(zip(boxes, scores)):
        x1, y1, x2, y2 = box.astype(int)
        color = colors[i][:3]
        cv2.rectangle(image, (x1, y1), (x2, y2), color=color, thickness=2)
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


def draw_input_boxes(image, boxes, box_labels):
    """Overlay the input bounding box prompts on the image for debugging."""
    for i, box in enumerate(boxes):
        x1, y1, x2, y2 = [int(v) for v in box]
        label = box_labels[i] if box_labels is not None and i < len(box_labels) else 1
        color = (0, 255, 0) if label == 1 else (0, 0, 255)
        cv2.rectangle(image, (x1, y1), (x2, y2), color=color, thickness=1)
    return image


# ======================
# Main functions
# ======================


def preprocess(img):
    img_rgb = img[:, :, ::-1].astype(np.float32) / 255.0  # BGR→RGB, [0,1]
    img_resized = cv2.resize(
        img_rgb, (IMAGE_SIZE, IMAGE_SIZE), interpolation=cv2.INTER_LINEAR
    )
    img_norm = (img_resized - 0.5) / 0.5  # mean=0.5, std=0.5
    img_chw = img_norm.transpose(2, 0, 1)[None].astype(np.float32)  # [1,3,H,W]
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

    # [cx, cy, w, h] normalized → [x0, y0, x1, y1] pixels
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


def build_box_inputs(boxes, box_labels, orig_h, orig_w):
    """Convert pixel [x1,y1,x2,y2] boxes to the padded tensors expected by
    sam3_grounding_visual.onnx.

    Returns:
        box_coords  : float32 (MAX_BOXES, 1, 4)  [cx,cy,w,h] normalized
        box_labels_arr : int64 (MAX_BOXES, 1)    1=positive, 0=negative
        box_mask    : bool    (1, MAX_BOXES)      True = unused slot
    """
    n = len(boxes)
    if n > MAX_BOXES:
        logger.warning("Too many boxes (%d); truncating to %d.", n, MAX_BOXES)
        boxes = boxes[:MAX_BOXES]
        if box_labels is not None:
            box_labels = box_labels[:MAX_BOXES]
        n = MAX_BOXES

    coords = np.array(boxes, dtype=np.float32)  # (n, 4) [x1,y1,x2,y2]
    cx = (coords[:, 0] + coords[:, 2]) / 2.0 / orig_w
    cy = (coords[:, 1] + coords[:, 3]) / 2.0 / orig_h
    bw = (coords[:, 2] - coords[:, 0]) / orig_w
    bh = (coords[:, 3] - coords[:, 1]) / orig_h
    cxcywh = np.stack([cx, cy, bw, bh], axis=-1)  # (n, 4)

    # Pad to MAX_BOXES
    pad = MAX_BOXES - n
    if pad > 0:
        cxcywh = np.concatenate([cxcywh, np.zeros((pad, 4), np.float32)], axis=0)

    box_coords = cxcywh[:, np.newaxis, :]  # (MAX_BOXES, 1, 4)

    if box_labels is None:
        lbls = np.ones(n, dtype=np.int64)
    else:
        lbls = np.array(box_labels[:n], dtype=np.int64)
    if pad > 0:
        lbls = np.concatenate([lbls, np.zeros(pad, dtype=np.int64)], axis=0)
    box_labels_arr = lbls[:, np.newaxis]  # (MAX_BOXES, 1)

    box_mask = np.zeros((1, MAX_BOXES), dtype=bool)
    box_mask[0, n:] = True  # mark padding slots as unused

    return box_coords, box_labels_arr, box_mask


def predict(models, img, caption, threshold, boxes=None, box_labels=None):
    orig_h, orig_w = img.shape[:2]
    img_input = preprocess(img)

    use_visual = boxes is not None and len(boxes) > 0

    if use_visual:
        text_tokens = tokenize("visual")
        box_coords, box_labels_arr, box_mask = build_box_inputs(
            boxes, box_labels, orig_h, orig_w
        )
        grounder = models["grounder"]
    else:
        text_tokens = tokenize(caption)
        grounder = models["grounder"]

    encoder = models["encoder"]
    if not args.onnx:
        enc_out = encoder.predict([img_input])
    else:
        enc_out = encoder.run(None, {"image": img_input})
    fpn0, fpn1, fpn2, pos0, pos1, pos2 = enc_out

    if use_visual:
        if not args.onnx:
            gnd_out = grounder.predict(
                [
                    fpn0,
                    fpn1,
                    fpn2,
                    pos0,
                    pos1,
                    pos2,
                    text_tokens,
                    box_coords,
                    box_labels_arr,
                    box_mask,
                ]
            )
        else:
            gnd_out = grounder.run(
                None,
                {
                    "fpn0": fpn0,
                    "fpn1": fpn1,
                    "fpn2": fpn2,
                    "pos0": pos0,
                    "pos1": pos1,
                    "pos2": pos2,
                    "text_tokens": text_tokens,
                    "box_coords": box_coords,
                    "box_labels": box_labels_arr,
                    "box_mask": box_mask,
                },
            )
    else:
        if not args.onnx:
            gnd_out = grounder.predict(
                [fpn0, fpn1, fpn2, pos0, pos1, pos2, text_tokens]
            )
        else:
            gnd_out = grounder.run(
                None,
                {
                    "fpn0": fpn0,
                    "fpn1": fpn1,
                    "fpn2": fpn2,
                    "pos0": pos0,
                    "pos1": pos1,
                    "pos2": pos2,
                    "text_tokens": text_tokens,
                },
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


def recognize_from_image(models):
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
        # if use_visual:
        #     res_img = draw_input_boxes(res_img, boxes, box_labels)
        res_img = draw_predictions(res_img, det_boxes, scores, masks, display_label)

        savepath = get_savepath(args.savepath, image_path, ext=".png")
        cv2.imwrite(savepath, res_img)
        logger.info(f"saved at : {savepath}")

    logger.info("Script finished successfully.")


def recognize_from_video(models):
    caption = args.caption
    threshold = args.threshold
    boxes = args.box
    box_labels = args.box_label

    use_visual = boxes is not None and len(boxes) > 0
    display_label = "visual" if use_visual else caption

    video_file = args.video if args.video else args.input[0]
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
        # if use_visual:
        #     res_img = draw_input_boxes(res_img, boxes, box_labels)
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


def main():
    use_visual = args.box is not None and len(args.box) > 0

    check_and_download_models(WEIGHT_ENC_PATH, MODEL_ENC_PATH, REMOTE_PATH)
    check_and_download_file(DATA_ENC_PATH, REMOTE_PATH)
    check_and_download_file(BPE_PATH, REMOTE_PATH)

    if use_visual:
        check_and_download_models(WEIGHT_GND_VIS_PATH, MODEL_GND_VIS_PATH, REMOTE_PATH)
        check_and_download_file(DATA_GND_VIS_PATH, REMOTE_PATH)
    else:
        check_and_download_models(WEIGHT_GND_PATH, MODEL_GND_PATH, REMOTE_PATH)
        check_and_download_file(DATA_GND_PATH, REMOTE_PATH)

    env_id = args.env_id

    if not args.onnx:
        memory_mode = ailia.get_memory_mode(
            reduce_constant=True,
            ignore_input_with_initializer=True,
            reduce_interstage=False,
            reuse_interstage=True,
        )
        encoder = ailia.Net(
            MODEL_ENC_PATH, WEIGHT_ENC_PATH, env_id=env_id, memory_mode=memory_mode
        )
        grounder = ailia.Net(
            MODEL_GND_VIS_PATH if use_visual else MODEL_GND_PATH,
            WEIGHT_GND_VIS_PATH if use_visual else WEIGHT_GND_PATH,
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
        encoder = onnxruntime.InferenceSession(WEIGHT_ENC_PATH, providers=providers)
        grounder = onnxruntime.InferenceSession(
            WEIGHT_GND_VIS_PATH if use_visual else WEIGHT_GND_PATH,
            providers=providers,
        )

    models = dict(encoder=encoder, grounder=grounder)

    if args.video is not None:
        recognize_from_video(models)
    else:
        recognize_from_image(models)


if __name__ == "__main__":
    main()
