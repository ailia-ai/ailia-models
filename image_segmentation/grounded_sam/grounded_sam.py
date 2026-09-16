import sys
import os
import time

import numpy as np
import cv2
from PIL import Image

import ailia


# import original modules
sys.path.append("../../util")
from arg_utils import get_base_parser, update_parser, get_savepath  # noqa
from model_utils import check_and_download_models, check_and_download_file  # noqa
from image_utils import normalize_image  # noqa
from detector_utils import load_image, plot_results, hsv_to_rgb  # noqa
from nms_utils import nms_boxes  # noqa
from load_model import load_groudingdino

# logger
from logging import getLogger  # noqa


top_path = os.path.dirname(
    os.path.dirname(
        os.path.dirname(
            os.path.abspath(__file__),
        )
    )
)


logger = getLogger(__name__)


# ======================
# Parameters
# ======================

WEIGHT_PATH = "sam_vit_h_4b8939.onnx"
MODEL_PATH = "sam_vit_h_4b8939.onnx.prototxt"
DATA_PATH = "sam_vit_h_4b8939_weights.pb"
WEIGHT_GDINO_PATH = "groundingdino_swint_ogc.onnx"
MODEL_GDINO_PATH = "groundingdino_swint_ogc.onnx.prototxt"
REMOTE_PATH = "https://storage.googleapis.com/ailia-models/grounded-sam/"
REMOTE_GDINO_PATH = "https://storage.googleapis.com/ailia-models/groundingdino/"

IMAGE_PATH = "demo.jpg"
SAVE_IMAGE_PATH = "output.png"

# ======================
# Arguemnt Parser Config
# ======================

parser = get_base_parser("Grounded-SAM", IMAGE_PATH, SAVE_IMAGE_PATH)
parser.add_argument(
    "--caption",
    type=str,
    default="The running dog.",
    help="Text prompt.",
)
parser.add_argument(
    '--disable_ailia_tokenizer',
    action='store_true',
    help='disable ailia tokenizer.'
)
parser.add_argument("--onnx", action="store_true", help="execute onnxruntime version.")
args = update_parser(parser)


# ======================
# Secondaty Functions
# ======================


def setup_groudingdino(tokenizer, grounding_dino):
    mod = load_groudingdino(args)

    def predict(img, caption):
        models = {
            "tokenizer": tokenizer,
            "net": grounding_dino,
        }
        boxes, logits, phrases = mod.predict(models, img, caption)
        return boxes, logits, phrases

    return predict


def generate_masks_with_special_tokens(input_ids, special_tokens_list):
    """Generate attention mask between each pair of special tokens
    Args:
        input_ids (np.ndarray): input ids. Shape: [bs, num_token]
        special_tokens_mask (list): special tokens mask.
    Returns:
        np.ndarray: attention mask between each special tokens.
    """
    bs, num_token = input_ids.shape
    # special_tokens_mask: bs, num_token. 1 for special tokens. 0 for normal tokens
    special_tokens_mask = np.zeros((bs, num_token), dtype=bool)
    for special_token in special_tokens_list:
        special_tokens_mask |= input_ids == special_token

    # idxs: each row is a list of indices of special tokens
    idxs = np.nonzero(special_tokens_mask)

    # generate attention mask and positional ids
    attention_mask = np.eye(num_token, dtype=bool)[None, ...].repeat(bs, axis=0)
    position_ids = np.zeros((bs, num_token), dtype=int)
    previous_col = 0
    for i in range(len(idxs[0])):
        row, col = idxs[0][i], idxs[1][i]
        if (col == 0) or (col == num_token - 1):
            attention_mask[row, col, col] = True
            position_ids[row, col] = 0
        else:
            attention_mask[
                row, previous_col + 1 : col + 1, previous_col + 1 : col + 1
            ] = True
            position_ids[row, previous_col + 1 : col + 1] = np.arange(
                0,
                col - previous_col,
            )
            c2t_maski = np.zeros((num_token), dtype=bool)
            c2t_maski[previous_col + 1 : col] = True
        previous_col = col

    return attention_mask, position_ids


def draw_predictions(image, boxes, logits, phrases, masks):
    height, width, _ = image.shape

    colors = [
        hsv_to_rgb(256 * i / (len(boxes) + 1), 255, 255) for i in range(len(boxes))
    ]

    # draw masks
    colored_mask = np.array(image, copy=True, dtype=np.uint8)
    area = np.array([np.sum(mask) for mask in masks])
    for detection_idx in np.flip(np.argsort(area)):
        mask = masks[detection_idx]
        colored_mask[mask] = colors[detection_idx][:3]

    opacity = 0.5
    image = cv2.addWeighted(colored_mask, opacity, image, 1 - opacity, 0)
    image = image.astype(np.uint8)

    boxes = boxes * np.array([width, height, width, height])
    cx, cy, w, h = np.split(boxes, 4, axis=-1)
    x1 = cx - 0.5 * w
    y1 = cy - 0.5 * h
    x2 = cx + 0.5 * w
    y2 = cy + 0.5 * h
    xyxy = np.concatenate((x1, y1, x2, y2), axis=-1)

    mode_ailia = True
    if mode_ailia:
        detect_objects = []
        for i in range(len(xyxy)):
            x1, y1, x2, y2 = xyxy[i].astype(int)
            r = ailia.DetectorObject(
                category=phrases[i],
                prob=logits[i],
                x=x1 / width,
                y=y1 / height,
                w=(x2 - x1) / width,
                h=(y2 - y1) / height,
            )
            detect_objects.append(r)

        res_img = plot_results(detect_objects, image)
        return res_img

    labels = [f"{phrase} {logit:.2f}" for phrase, logit in zip(phrases, logits)]

    font = cv2.FONT_HERSHEY_SIMPLEX
    thickness = 2
    text_scale = 0.5
    text_thickness = 1
    text_padding = 10
    for i in range(len(xyxy)):
        x1, y1, x2, y2 = xyxy[i].astype(int)
        color = colors[i]
        cv2.rectangle(
            img=image,
            pt1=(x1, y1),
            pt2=(x2, y2),
            color=color,
            thickness=thickness,
        )

        text = labels[i]

        text_width, text_height = cv2.getTextSize(
            text=text,
            fontFace=font,
            fontScale=text_scale,
            thickness=text_thickness,
        )[0]

        text_x = x1 + text_padding
        text_y = y1 - text_padding

        text_background_x1 = x1
        text_background_y1 = y1 - 2 * text_padding - text_height

        text_background_x2 = x1 + 2 * text_padding + text_width
        text_background_y2 = y1

        cv2.rectangle(
            img=image,
            pt1=(text_background_x1, text_background_y1),
            pt2=(text_background_x2, text_background_y2),
            color=color,
            thickness=cv2.FILLED,
        )
        cv2.putText(
            img=image,
            text=text,
            org=(text_x, text_y),
            fontFace=font,
            fontScale=text_scale,
            color=(0, 0, 0),
            thickness=text_thickness,
            lineType=cv2.LINE_AA,
        )

    return image


# ======================
# Main functions
# ======================


def preprocess_shape(h, w, long_side_length):
    scale = long_side_length * 1.0 / max(h, w)
    new_h, new_w = h * scale, w * scale
    new_w = int(new_w + 0.5)
    new_h = int(new_h + 0.5)

    return new_h, new_w


def preprocess(img, target_size):
    img = np.array(
        Image.fromarray(img).resize(target_size[::-1], Image.Resampling.BILINEAR)
    )

    img = normalize_image(img, normalize_type="ImageNet")

    img_size = 1024
    pad_h = img_size - target_size[0]
    pad_w = img_size - target_size[1]
    img = np.pad(img, ((0, pad_h), (0, pad_w), (0, 0)))

    img = img.transpose(2, 0, 1)  # HWC -> CHW
    img = np.expand_dims(img, axis=0)
    img = img.astype(np.float32)

    return img


def segment(net, img, xyxy):
    height, width, _ = img.shape
    img = img[:, :, ::-1]  # BGR -> RGB

    target_length = 1024
    target_size = preprocess_shape(height, width, target_length)
    img = preprocess(img, target_size)

    result_masks = []
    for box in xyxy:
        # Transform
        coords = box.reshape(-1, 2, 2)
        coords[..., 0] = coords[..., 0] * (target_size[1] / width)
        coords[..., 1] = coords[..., 1] * (target_size[0] / height)
        box = box.reshape(-1, 4)
        box = box[None, :]

        # feedforward
        input_size = np.array(target_size, dtype=int)
        original_size = np.array((height, width), dtype=int)
        if not args.onnx:
            output = net.predict([img, box, input_size, original_size])
        else:
            output = net.run(
                None,
                {
                    "image": img,
                    "box": box,
                    "input_size": input_size,
                    "original_size": original_size,
                },
            )
        masks, iou_predictions, low_res_masks = output

        masks = masks[0]
        scores = iou_predictions[0]
        logits = low_res_masks[0]
        index = np.argmax(scores)
        result_masks.append(masks[index])

    return np.array(result_masks)


def predict(models, img, caption):
    height, width, _ = img.shape

    predict_grounding_dino = models["grounding_dino"]
    boxes, logits, phrases = predict_grounding_dino(img, caption)

    _boxes = boxes * np.array([width, height, width, height], dtype=np.float32)
    cx, cy, w, h = np.split(_boxes, 4, axis=-1)
    x1 = cx - 0.5 * w
    y1 = cy - 0.5 * h
    x2 = cx + 0.5 * w
    y2 = cy + 0.5 * h
    xyxy = np.concatenate((x1, y1, x2, y2), axis=-1)

    # NMS
    NMS_THRESHOLD = 0.8
    nms_idx = nms_boxes(xyxy, logits, NMS_THRESHOLD)

    boxes = boxes[nms_idx]
    xyxy = xyxy[nms_idx]
    logits = logits[nms_idx]
    phrases = [p for i, p in enumerate(phrases) if i in nms_idx]

    net = models["sam"]
    masks = segment(net, img, xyxy)

    return boxes, logits, phrases, masks


def recognize_from_image(models):
    caption = args.caption
    if not caption.endswith("."):
        caption = caption + "."

    logger.info("Caption: " + caption)

    # input image loop
    for image_path in args.input:
        logger.info(image_path)

        # prepare input data
        img = load_image(image_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)

        # inference
        logger.info("Start inference...")
        if args.benchmark:
            logger.info("BENCHMARK mode")
            total_time_estimation = 0
            for i in range(args.benchmark_count):
                start = int(round(time.time() * 1000))
                output = predict(models, img, caption)
                end = int(round(time.time() * 1000))
                estimation_time = end - start

                # Loggin
                logger.info(f"\tailia processing estimation time {estimation_time} ms")
                if i != 0:
                    total_time_estimation = total_time_estimation + estimation_time

            logger.info(
                f"\taverage time estimation {total_time_estimation / (args.benchmark_count - 1)} ms"
            )
        else:
            output = predict(models, img, caption)

        boxes, logits, phrases, masks = output

        logger.info("detected %d instances" % len(boxes))

        # draw prediction
        res_img = draw_predictions(img, boxes, logits, phrases, masks)

        # plot result
        savepath = get_savepath(args.savepath, image_path, ext=".png")
        logger.info(f"saved at : {savepath}")
        cv2.imwrite(savepath, res_img)

    logger.info("Script finished successfully.")


def main():
    # model files check and download
    check_and_download_models(WEIGHT_PATH, MODEL_PATH, REMOTE_PATH)
    check_and_download_file(DATA_PATH, REMOTE_PATH)
    check_and_download_models(WEIGHT_GDINO_PATH, MODEL_GDINO_PATH, REMOTE_GDINO_PATH)

    env_id = args.env_id

    # initialize
    if not args.onnx:
        grounding_dino = ailia.Net(MODEL_GDINO_PATH, WEIGHT_GDINO_PATH, env_id=env_id, memory_mode=args.memory_mode)
        net = ailia.Net(MODEL_PATH, WEIGHT_PATH, env_id=env_id, memory_mode=args.memory_mode)
    else:
        import onnxruntime

        grounding_dino = onnxruntime.InferenceSession(WEIGHT_GDINO_PATH)
        net = onnxruntime.InferenceSession(WEIGHT_PATH)

    if args.disable_ailia_tokenizer:
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained("tokenizer")
    else:
        #ailia Tokenizer 1.3
        #from ailia_tokenizer import BertUncasedTokenizer
        #tokenizer = BertUncasedTokenizer.from_pretrained("./tokenizer/vocab.txt")

        #ailia Tokenizer 1.4
        from ailia_tokenizer import BertTokenizer
        tokenizer = BertTokenizer.from_pretrained("./tokenizer/")

    tokenizer.specical_tokens = tokenizer.convert_tokens_to_ids(
        ["[CLS]", "[SEP]", ".", "?"]
    )

    predict_grounding_dino = setup_groudingdino(tokenizer, grounding_dino)

    models = {
        "tokenizer": tokenizer,
        "grounding_dino": predict_grounding_dino,
        "sam": net,
    }

    recognize_from_image(models)


if __name__ == "__main__":
    main()
