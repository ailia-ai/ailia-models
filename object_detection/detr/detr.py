import sys
import time

import ailia
import cv2
import numpy as np
from PIL import Image

sys.path.append("../../util")
from logging import getLogger

from arg_utils import get_base_parser, get_savepath, update_parser
from detector_utils import load_image, plot_results
from image_utils import normalize_image
from math_utils import sigmoid, softmax
from model_utils import check_and_download_models
from webcamera_utils import get_capture, get_writer

logger = getLogger(__name__)


# ======================
# Parameters
# ======================

WEIGHT_DET_PATH = "detr-r50-e632da11.onnx"
MODEL_DET_PATH = "detr-r50-e632da11.onnx.prototxt"
WEIGHT_PAN_PATH = "detr_resnet101_panoptic.onnx"
MODEL_PAN_PATH = "detr_resnet101_panoptic.onnx.prototxt"

REMOTE_PATH = "https://storage.googleapis.com/ailia-models/detr/"

IMAGE_PATH = "input.jpg"
SAVE_IMAGE_PATH = "output.png"

THRESHOLD = 0.7

# COCO 91-class labels (index = COCO category ID, 0=N/A, 91=no-object in model)
#fmt: off
COCO_CATEGORY = [
    'N/A', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
    'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'N/A',
    'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse',
    'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'N/A',
    'backpack', 'umbrella', 'N/A', 'N/A', 'handbag', 'tie', 'suitcase',
    'frisbee', 'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat',
    'baseball glove', 'skateboard', 'surfboard', 'tennis racket', 'bottle',
    'N/A', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana',
    'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
    'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'N/A',
    'dining table', 'N/A', 'N/A', 'toilet', 'N/A', 'tv', 'laptop', 'mouse',
    'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster',
    'sink', 'refrigerator', 'N/A', 'book', 'clock', 'vase', 'scissors',
    'teddy bear', 'hair drier', 'toothbrush',
]

# COCO panoptic labels: things (IDs 1-90) + stuff (IDs 92-200)
COCO_PANOPTIC_CATEGORY = {
    1: 'person', 2: 'bicycle', 3: 'car', 4: 'motorcycle', 5: 'airplane',
    6: 'bus', 7: 'train', 8: 'truck', 9: 'boat', 10: 'traffic light',
    11: 'fire hydrant', 13: 'stop sign', 14: 'parking meter', 15: 'bench',
    16: 'bird', 17: 'cat', 18: 'dog', 19: 'horse', 20: 'sheep', 21: 'cow',
    22: 'elephant', 23: 'bear', 24: 'zebra', 25: 'giraffe', 27: 'backpack',
    28: 'umbrella', 31: 'handbag', 32: 'tie', 33: 'suitcase', 34: 'frisbee',
    35: 'skis', 36: 'snowboard', 37: 'sports ball', 38: 'kite',
    39: 'baseball bat', 40: 'baseball glove', 41: 'skateboard',
    42: 'surfboard', 43: 'tennis racket', 44: 'bottle', 46: 'wine glass',
    47: 'cup', 48: 'fork', 49: 'knife', 50: 'spoon', 51: 'bowl',
    52: 'banana', 53: 'apple', 54: 'sandwich', 55: 'orange', 56: 'broccoli',
    57: 'carrot', 58: 'hot dog', 59: 'pizza', 60: 'donut', 61: 'cake',
    62: 'chair', 63: 'couch', 64: 'potted plant', 65: 'bed',
    67: 'dining table', 70: 'toilet', 72: 'tv', 73: 'laptop', 74: 'mouse',
    75: 'remote', 76: 'keyboard', 77: 'cell phone', 78: 'microwave',
    79: 'oven', 80: 'toaster', 81: 'sink', 82: 'refrigerator', 84: 'book',
    85: 'clock', 86: 'vase', 87: 'scissors', 88: 'teddy bear',
    89: 'hair drier', 90: 'toothbrush',
    92: 'banner', 93: 'blanket', 94: 'bridge', 95: 'cardboard',
    96: 'counter', 97: 'curtain', 98: 'door', 99: 'floor-wood',
    100: 'flower', 101: 'fruit', 102: 'gravel', 103: 'house', 104: 'light',
    105: 'mirror', 106: 'net', 107: 'pillow', 108: 'plastic',
    109: 'platform', 110: 'playingfield', 111: 'railroad', 112: 'river',
    113: 'road', 114: 'roof', 115: 'sand', 116: 'sea', 117: 'shelf',
    118: 'snow', 119: 'stairs', 120: 'tent', 121: 'textile', 122: 'towel',
    123: 'wall-brick', 124: 'wall-concrete', 125: 'wall-panel',
    126: 'wall-stone', 127: 'wall-tile', 128: 'wall-wood',
    129: 'water', 130: 'window-blind', 131: 'window',
    132: 'tree', 133: 'fence', 134: 'ceiling', 135: 'sky',
    136: 'cabinet', 137: 'table', 138: 'floor', 139: 'pavement',
    140: 'mountain', 141: 'grass', 142: 'dirt', 143: 'paper',
    144: 'food', 145: 'building', 146: 'rock', 147: 'wall', 148: 'rug',
}
#fmt: on


# ======================
# Argument Parser Config
# ======================

parser = get_base_parser("DETR", IMAGE_PATH, SAVE_IMAGE_PATH)
parser.add_argument(
    "-th",
    "--threshold",
    default=THRESHOLD,
    type=float,
    help="Detection score threshold.",
)
parser.add_argument(
    "--segment",
    action="store_true",
    help="Use panoptic segmentation model (detr_resnet101_panoptic) instead of the default object detection model (detr-r50).",
)
parser.add_argument("--onnx", action="store_true", help="Execute with onnxruntime.")
args = update_parser(parser)


# ======================
# Secondary functions
# ======================


def preprocess(img_bgr):
    img_rgb = img_bgr[..., ::-1]  # BGR -> RGB
    pil = Image.fromarray(img_rgb)

    h, w = img_bgr.shape[:2]
    scale = 800 / min(h, w)
    if max(h, w) * scale > 1333:
        scale = 1333 / max(h, w)
    new_h = int(round(h * scale))
    new_w = int(round(w * scale))
    pil_r = pil.resize((new_w, new_h), Image.BILINEAR)

    img_arr = np.array(pil_r).astype(np.float32)
    img_norm = normalize_image(img_arr, normalize_type="ImageNet")
    pixel_values = img_norm.transpose(2, 0, 1)[np.newaxis].astype(np.float32)
    # pixel_mask=False means valid pixel (inverted DETR convention in this export)
    pixel_mask = np.zeros((1, new_h, new_w), dtype=bool)

    return pixel_values, pixel_mask


def box_cxcywh_to_xyxy(boxes):
    cx, cy, w, h = boxes[..., 0], boxes[..., 1], boxes[..., 2], boxes[..., 3]
    return np.stack([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], axis=-1)


def postprocess_detection(pred_logits, pred_boxes, orig_h, orig_w):
    probs = softmax(pred_logits[0], axis=-1)
    # last class is "no object"; object score = 1 - p(no_object)
    scores = 1.0 - probs[:, -1]
    keep = scores > args.threshold
    if not keep.any():
        return []

    scores_k = scores[keep]
    labels_k = np.argmax(probs[keep, :-1], axis=-1)
    boxes_k = box_cxcywh_to_xyxy(pred_boxes[0][keep])

    detect_objects = []
    for score, label, box in zip(scores_k, labels_k, boxes_k):
        x1, y1, x2, y2 = box
        x1 = float(np.clip(x1, 0, 1))
        y1 = float(np.clip(y1, 0, 1))
        x2 = float(np.clip(x2, 0, 1))
        y2 = float(np.clip(y2, 0, 1))
        r = ailia.DetectorObject(
            category=int(label),
            prob=float(score),
            x=x1,
            y=y1,
            w=x2 - x1,
            h=y2 - y1,
        )
        detect_objects.append(r)
    return detect_objects


def draw_segmentation(detect_objects, segm_masks, img):
    h, w = img.shape[:2]
    count = len(detect_objects)
    fontScale = w / 2048

    for idx in range(count):
        color_bgr = [
            int(c)
            for c in __import__("colorsys").hsv_to_rgb(idx / (count + 1), 1.0, 1.0)
        ]
        color_bgr = [int(c * 255) for c in color_bgr]

        mask = np.repeat(np.expand_dims(segm_masks[idx], 2), 3, axis=2).astype(bool)
        fill = np.zeros_like(img[:, :, :3])
        fill[:] = color_bgr[::-1]  # RGB→BGR for OpenCV
        img[:, :, :3][mask] = (img[:, :, :3][mask] * 0.6 + fill[mask] * 0.4).astype(
            np.uint8
        )

        obj = detect_objects[idx]
        text = f"{obj.category} {obj.prob:.2f}"
        textsize = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, fontScale, 1)[0]
        x1 = max(0, int(w * obj.x))
        y1 = max(textsize[1] + 4, int(h * obj.y))
        cv2.rectangle(
            img,
            (x1, y1 - textsize[1] - 3),
            (x1 + textsize[0] + 3, y1 + 1),
            color_bgr[::-1],
            thickness=-1,
        )
        cv2.putText(
            img, text, (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, fontScale, (255, 255, 255), 1
        )

    return img


def postprocess_panoptic(pred_logits, pred_boxes, pred_masks, orig_h, orig_w):
    probs = softmax(pred_logits[0], axis=-1)
    scores = 1.0 - probs[:, -1]
    keep = scores > args.threshold
    if not keep.any():
        return [], []

    scores_k = scores[keep]
    labels_k = np.argmax(probs[keep, :-1], axis=-1)
    boxes_k = box_cxcywh_to_xyxy(pred_boxes[0][keep])
    masks_k = pred_masks[0][keep]  # (N, H', W')

    detect_objects = []
    segm_masks = []
    for score, label, box, mask_raw in zip(scores_k, labels_k, boxes_k, masks_k):
        x1, y1, x2, y2 = box
        x1 = float(np.clip(x1, 0, 1))
        y1 = float(np.clip(y1, 0, 1))
        x2 = float(np.clip(x2, 0, 1))
        y2 = float(np.clip(y2, 0, 1))
        label_name = COCO_PANOPTIC_CATEGORY.get(int(label), f"category_{int(label)}")
        r = ailia.DetectorObject(
            category=label_name,
            prob=float(score),
            x=x1,
            y=y1,
            w=x2 - x1,
            h=y2 - y1,
        )
        detect_objects.append(r)

        mask_prob = sigmoid(mask_raw)
        mask_bin = (
            cv2.resize(mask_prob, (orig_w, orig_h), interpolation=cv2.INTER_LINEAR)
            >= 0.5
        )
        segm_masks.append(mask_bin)

    return detect_objects, segm_masks


def predict(net, img):
    pixel_values, pixel_mask = preprocess(img)

    if not args.onnx:
        # ailia does not support bool tensors; pass pixel_mask as int64 (0=valid, 1=masked)
        output = net.predict(
            {"pixel_values": pixel_values, "pixel_mask": pixel_mask.astype(np.int64)}
        )
    else:
        output = net.run(
            [x.name for x in net.get_outputs()],
            {"pixel_values": pixel_values, "pixel_mask": pixel_mask},
        )

    orig_h, orig_w = img.shape[:2]
    if not args.segment:
        pred_logits, pred_boxes = output
        return postprocess_detection(pred_logits, pred_boxes, orig_h, orig_w)
    else:
        pred_logits, pred_boxes, pred_masks = output
        return postprocess_panoptic(pred_logits, pred_boxes, pred_masks, orig_h, orig_w)


# ======================
# Main functions
# ======================


def recognize_from_image(net):
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
                result = predict(net, img)
                end = int(round(time.time() * 1000))
                if i != 0:
                    total_time += end - start
                logger.info(f"\tailia processing time {end - start} ms")
            logger.info(f"\taverage time {total_time / (args.benchmark_count - 1)} ms")
        else:
            result = predict(net, img)

        if not args.segment:
            detect_objects = result
            res_img = plot_results(detect_objects, img, category=COCO_CATEGORY)
        else:
            detect_objects, segm_masks = result
            res_img = draw_segmentation(detect_objects, segm_masks, img)

        savepath = get_savepath(args.savepath, image_path, ext=".png")
        logger.info(f"saved at : {savepath}")
        cv2.imwrite(savepath, res_img)

    logger.info("Script finished successfully.")


def recognize_from_video(net):
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

        result = predict(net, frame)

        if not args.segment:
            detect_objects = result
            res_img = plot_results(detect_objects, frame, category=COCO_CATEGORY)
        else:
            detect_objects, segm_masks = result
            res_img = draw_segmentation(detect_objects, segm_masks, frame)

        cv2.imshow("frame", res_img)
        frame_shown = True

        if writer is not None:
            res_img = res_img.astype(np.uint8)
            writer.write(res_img)

    capture.release()
    cv2.destroyAllWindows()
    if writer is not None:
        writer.release()

    logger.info("Script finished successfully.")


def main():
    if not args.segment:
        weight_path = WEIGHT_DET_PATH
        model_path = MODEL_DET_PATH
    else:
        weight_path = WEIGHT_PAN_PATH
        model_path = MODEL_PAN_PATH

    check_and_download_models(weight_path, model_path, REMOTE_PATH)

    if not args.onnx:
        memory_mode = ailia.get_memory_mode(
            reduce_constant=True,
            ignore_input_with_initializer=True,
            reduce_interstage=False,
            reuse_interstage=True,
        )
        net = ailia.Net(
            model_path, weight_path, env_id=args.env_id, memory_mode=memory_mode
        )
    else:
        import onnxruntime

        cuda = 0 < ailia.get_gpu_environment_id()
        providers = (
            ["CUDAExecutionProvider", "CPUExecutionProvider"]
            if cuda
            else ["CPUExecutionProvider"]
        )
        net = onnxruntime.InferenceSession(weight_path, providers=providers)

    if args.video is not None:
        recognize_from_video(net)
    else:
        recognize_from_image(net)


if __name__ == "__main__":
    main()
