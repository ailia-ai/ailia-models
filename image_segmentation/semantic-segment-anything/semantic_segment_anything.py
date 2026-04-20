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
# SAM auto-mask generation
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


def generate_all_masks(models, img_rgb):
    """Full automatic mask generation matching SAM's SamAutomaticMaskGenerator."""
    im_h, im_w = img_rgb.shape[:2]

    crop_boxes, layer_idxs = generate_crop_boxes(im_h, im_w, CROP_N_LAYERS)


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

        sam_enc = onnxruntime.InferenceSession(WEIGHT_SAM_ENC_PATH)
        sam_dec = onnxruntime.InferenceSession(WEIGHT_SAM_DEC_PATH)
        if args.model_type == "ade20k":
            segformer = onnxruntime.InferenceSession(WEIGHT_SEG_ADE20K_PATH)
        else:
            segformer = onnxruntime.InferenceSession(WEIGHT_SEG_CITY_PATH)

    models = {
        "sam_enc": sam_enc,
        "sam_dec": sam_dec,
        "segformer": segformer,
    }

    recognize_from_image(models)


if __name__ == "__main__":
    main()
