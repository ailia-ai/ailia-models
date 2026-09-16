import sys
import os
import time
from copy import deepcopy
from collections import OrderedDict
from logging import getLogger

import numpy as np
import cv2
from PIL import Image

import ailia

# import original modules
sys.path.append("../../util")
from arg_utils import get_base_parser, update_parser, get_savepath  # noqa
from model_utils import urlretrieve, progress_print, check_and_download_models  # noqa
from image_utils import normalize_image  # noqa
from detector_utils import load_image  # noqa
from webcamera_utils import get_capture, get_writer  # noqa

logger = getLogger(__name__)

# ======================
# Parameters
# ======================

WEIGHT_ENC_PATH = "edge_sam_3x_encoder.onnx"
WEIGHT_DEC_PATH = "edge_sam_3x_decoder.onnx"
MODEL_ENC_PATH = "edge_sam_3x_encoder.onnx.prototxt"
MODEL_DEC_PATH = "edge_sam_3x_decoder.onnx.prototxt"
REMOTE_PATH = "https://storage.googleapis.com/ailia-models/edge_sam/"

IMAGE_PATH = "truck.jpg"
SAVE_IMAGE_PATH = "output.png"

POINT = (500, 375)

TARGET_LENGTH = 1024

# ======================
# Arguemnt Parser Config
# ======================

parser = get_base_parser("Edge SAM", IMAGE_PATH, SAVE_IMAGE_PATH, fp16_support = False)
parser.add_argument(
    "-p",
    "--pos",
    action="append",
    type=int,
    metavar="X",
    nargs=2,
    help="Positive coordinate specified by x,y.",
)
parser.add_argument(
    "--neg",
    action="append",
    type=int,
    metavar="X",
    nargs=2,
    help="Negative coordinate specified by x,y.",
)
parser.add_argument(
    "--box",
    type=int,
    metavar="X",
    nargs=4,
    help="Box coordinate specified by x1,y1,x2,y2.",
)
parser.add_argument(
    "--num_multimask_outputs",
    type=int,
    default=1,
    choices=(1, 3, 4),
    help="number of output masks",
)
parser.add_argument("--onnx", action="store_true", help="execute onnxruntime version.")
parser.add_argument("--gui", action="store_true", help="Open mouse click GUI.")
args = update_parser(parser)


# ======================
# Secondaty Functions
# ======================


def apply_coords(coords, h, w):
    scale = TARGET_LENGTH / max(h, w)
    coords = deepcopy(coords).astype(np.float32)
    coords[..., 0] = coords[..., 0] * scale
    coords[..., 1] = coords[..., 1] * scale

    return coords


def show_mask(mask, img):
    color = np.array([255, 144, 30], dtype=np.uint8)
    color = color.reshape(1, 1, -1)

    h, w = mask.shape[-2:]
    mask = mask.reshape(h, w, 1)

    mask_image = mask * color
    img = (img * (1 - mask)) + (img * mask) * 0.6 + mask_image * 0.4

    img = np.clip(img, 0, 255).astype(np.uint8)
    return img


def show_points(coords, labels, img):
    pos_points = coords[labels == 1]
    neg_points = coords[labels == 0]

    for p in pos_points:
        cv2.drawMarker(
            img,
            p,
            (0, 255, 0),
            markerType=cv2.MARKER_TILTED_CROSS,
            line_type=cv2.LINE_AA,
            markerSize=30,
            thickness=5,
        )
    for p in neg_points:
        cv2.drawMarker(
            img,
            p,
            (0, 0, 255),
            markerType=cv2.MARKER_TILTED_CROSS,
            line_type=cv2.LINE_AA,
            markerSize=30,
            thickness=5,
        )

    return img


def show_box(box, img):
    cv2.rectangle(
        img,
        box[0],
        box[1],
        color=(2, 118, 2),
        thickness=3,
        lineType=cv2.LINE_4,
        shift=0,
    )

    return img


def show_gui(models, image_path):
    # prepare input data
    img = load_image(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
    overlay_img = img.copy()

    coord_list = []
    label_list = []

    def predict_on_click(event, x, y, flags, param):
        nonlocal overlay_img  # クリックごとに更新する
        if event == cv2.EVENT_LBUTTONDOWN:
            coord_list.append((x, y))
            label_list.append(1)
            coord_input = np.array(coord_list)
            label_input = np.array(label_list)
            output = predict(models, img, coord_input, label_input)
            masks, scores = output

            mask = np.expand_dims(masks[scores.argmax()], axis=0)
            overlay_img = show_mask(mask, img)
            overlay_img = show_points(coord_input, label_input, overlay_img)
            print("x:", x, "y:", y)

    # 画像のウインドウに名前をつけ、コールバック関数をセット
    winname = "Mouse click GUI"
    cv2.namedWindow(winname)
    cv2.setMouseCallback(winname, predict_on_click)

    while True:
        cv2.imshow(winname, overlay_img)  # 更新した画像を表示
        # ESCキーで終了
        if cv2.waitKey(20) & 0xFF == 27 or not cv2.getWindowProperty(
            winname, cv2.WND_PROP_VISIBLE
        ):
            break
        time.sleep(0.1)

    cv2.destroyAllWindows()


# ======================
# Main functions
# ======================


def preprocess(img):
    im_h, im_w, _ = img.shape

    scale = TARGET_LENGTH / max(im_h, im_w)
    ow = int(im_w * scale)
    oh = int(im_h * scale)
    img = np.array(Image.fromarray(img).resize((ow, oh), Image.Resampling.BICUBIC))

    img = normalize_image(img, normalize_type="ImageNet")

    pad_h = TARGET_LENGTH - oh
    pad_w = TARGET_LENGTH - ow
    img = np.pad(img, ((0, pad_h), (0, pad_w), (0, 0)), "constant")

    img = img.transpose((2, 0, 1))  # HWC -> CHW
    img = np.expand_dims(img, axis=0)
    img = img.astype(np.float32)

    return img, (oh, ow)


def postprocess_masks(
    mask: np.ndarray, input_h: int, input_w: int, orig_h: int, orig_w: int
):
    mask = mask.squeeze(0).transpose(1, 2, 0)
    mask = cv2.resize(
        mask, (TARGET_LENGTH, TARGET_LENGTH), interpolation=cv2.INTER_LINEAR
    )
    mask = mask[:input_h, :input_w, :]
    mask = cv2.resize(mask, (orig_w, orig_h), interpolation=cv2.INTER_LINEAR)
    mask = mask.transpose(2, 0, 1)[None, :, :, :]
    return mask


def predict(models, img, point_coords, point_labels, num_multimask_outputs=1):
    im_h, im_w, _ = img.shape
    img = img[:, :, ::-1]  # BGR -> RGB
    img, input_size = preprocess(img)

    encoder = models["encoder"]
    if not args.onnx:
        output = encoder.predict([img])
    else:
        output = encoder.run(None, {"image": img})
    features = output[0]

    point_coords = np.array(point_coords, dtype=np.float32)[None]
    point_labels = np.array(point_labels, dtype=np.float32)[None]
    point_coords = apply_coords(point_coords, im_h, im_w)

    decoder = models["decoder"]
    if not args.onnx:
        output = decoder.predict([features, point_coords, point_labels])
    else:
        output = decoder.run(
            None,
            {
                "image_embeddings": features,
                "point_coords": point_coords,
                "point_labels": point_labels,
            },
        )
    scores, low_res_masks = output
    masks = postprocess_masks(low_res_masks, *input_size, im_h, im_w)

    mask_threshold = 0.0
    masks = masks > mask_threshold

    # Select the correct mask or masks for output
    if num_multimask_outputs == 4:
        mask_slice = slice(0, None)
    elif num_multimask_outputs == 3:
        mask_slice = slice(1, None)
    elif num_multimask_outputs == 1:
        mask_slice = slice(0, 1)
    else:
        raise ValueError

    masks = masks[:, mask_slice, :, :].squeeze(0)
    scores = scores[:, mask_slice].squeeze(0)

    return masks, scores


def recognize_from_image(models):
    pos_points = args.pos
    neg_points = args.neg
    box = args.box
    num_multimask_outputs = args.num_multimask_outputs

    if pos_points is None:
        if neg_points is None and box is None:
            pos_points = [POINT]
        else:
            pos_points = []
    if neg_points is None:
        neg_points = []
    if box is not None:
        box = np.array(box).reshape(2, 2)

    lf = "\n"
    logger.info(f"Positive coordinate: {pos_points}")
    logger.info(f"Negative coordinate: {neg_points}")
    logger.info(f"Box coordinate: {lf if box is not None else ''}{box}")

    coord_list = []
    label_list = []
    if box is not None:
        coord_list.append(box)
        label_list.append(np.array([2, 3]))
    if pos_points:
        coord_list.append(np.array(pos_points))
        label_list.append(np.ones(len(pos_points)))
    if neg_points:
        coord_list.append(np.array(neg_points))
        label_list.append(np.zeros(len(neg_points)))

    coord_list = np.concatenate(coord_list, axis=0)
    label_list = np.concatenate(label_list, axis=0)

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
                output = predict(
                    models, img, coord_list, label_list, num_multimask_outputs
                )
                end = int(round(time.time() * 1000))
                estimation_time = end - start

                # Logging
                logger.info(f"\tailia processing estimation time {estimation_time} ms")
                if i != 0:
                    total_time_estimation = total_time_estimation + estimation_time

            logger.info(
                f"\taverage time estimation {total_time_estimation / (args.benchmark_count - 1)} ms"
            )
        else:
            output = predict(models, img, coord_list, label_list, num_multimask_outputs)

        masks, scores = output

        mask = np.expand_dims(masks[scores.argmax()], axis=0)
        res_img = show_mask(mask, img)

        sel = label_list < 2
        coord_list = coord_list[sel]
        label_list = label_list[sel]
        if 0 < len(label_list):
            res_img = show_points(coord_list, label_list, res_img)
        if box is not None:
            res_img = show_box(box, res_img)

        # plot result
        savepath = get_savepath(args.savepath, image_path, ext=".png")
        cv2.imwrite(savepath, res_img)
        logger.info(f"saved at : {savepath}")


def main():
    # model files check and download
    check_and_download_models(WEIGHT_ENC_PATH, MODEL_ENC_PATH, REMOTE_PATH)
    check_and_download_models(WEIGHT_DEC_PATH, MODEL_DEC_PATH, REMOTE_PATH)

    env_id = args.env_id

    # initialize
    if not args.onnx:
        encoder = ailia.Net(MODEL_ENC_PATH, WEIGHT_ENC_PATH, env_id=env_id, memory_mode=args.memory_mode)
        decoder = ailia.Net(MODEL_DEC_PATH, WEIGHT_DEC_PATH, env_id=env_id, memory_mode=args.memory_mode)
    else:
        import onnxruntime

        providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
        encoder = onnxruntime.InferenceSession(WEIGHT_ENC_PATH, providers=providers)
        decoder = onnxruntime.InferenceSession(WEIGHT_DEC_PATH, providers=providers)

    models = dict(
        encoder=encoder,
        decoder=decoder,
    )

    if not args.gui:
        recognize_from_image(models)
    else:
        show_gui(models, args.input[0])


if __name__ == "__main__":
    main()
