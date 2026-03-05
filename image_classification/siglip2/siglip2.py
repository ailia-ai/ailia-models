import sys
import time
from logging import getLogger

import numpy as np
import cv2
from PIL import Image

import ailia

# import original modules
sys.path.append("../../util")
from arg_utils import get_base_parser, update_parser
from model_utils import check_and_download_file, check_and_download_models
from detector_utils import load_image
from math_utils import softmax


logger = getLogger(__name__)


# ======================
# Parameters
# ======================

WEIGHT_BASE_P16_224_PATH = "siglip2-base-patch16-224.onnx"
WEIGHT_LARGE_P16_256_PATH = "siglip2-large-patch16-256.onnx"
WEIGHT_GIANT_P16_256_PATH = "siglip2-giant-opt-patch16-256.onnx"
MODEL_BASE_P16_224_PATH = "siglip2-base-patch16-224.onnx.prototxt"
MODEL_LARGE_P16_256_PATH = "siglip2-large-patch16-256.onnx.prototxt"
MODEL_GIANT_P16_256_PATH = "siglip2-giant-opt-patch16-256.onnx.prototxt"
PB_LARGE_P16_256_PATH = "siglip2-large-patch16-256_weights.pb"
PB_GIANT_P16_256_PATH = "siglip2-giant-opt-patch16-256_weights.pb"
REMOTE_PATH = "https://storage.googleapis.com/ailia-models/siglip2/"

IMAGE_PATH = "demo.jpg"


# ======================
# Arguemnt Parser Config
# ======================

parser = get_base_parser("SigLIP2", IMAGE_PATH, None)
parser.add_argument(
    "-t",
    "--text",
    dest="text_inputs",
    type=str,
    action="append",
    help="Input text. (can be specified multiple times)",
)
parser.add_argument(
    "-m",
    "--model_type",
    default="base-patch16-224",
    choices=("base-patch16-224", "large-patch16-256", "giant-patch16-256"),
    help="model type",
)
parser.add_argument(
    "--disable_ailia_tokenizer", action="store_true", help="disable ailia tokenizer."
)
parser.add_argument("--onnx", action="store_true", help="execute onnxruntime version.")
args = update_parser(parser)


# ======================
# Main functions
# ======================


def preprocess(img, model_type):
    dic_size = {
        "base-patch16-224": 224,
        "large-patch16-256": 256,
        "giant-patch16-256": 256,
    }
    img_size = dic_size[model_type]

    img = img[:, :, ::-1]  # BGR -> RBG

    # resize
    img = np.array(
        Image.fromarray(img).resize((img_size, img_size), Image.Resampling.BILINEAR)
    )

    rescale_factor = 0.00392156862745098
    img = img.astype(np.float32) * rescale_factor
    img = (img - 0.5) / 0.5

    img = img.transpose(2, 0, 1)  # HWC -> CHW
    img = np.expand_dims(img, axis=0)

    return img


def postprocess_result(logits_per_image, top_k: int = 5):
    probs = softmax(logits_per_image, axis=1)[0]

    top_labels = np.argsort(-probs)[: min(top_k, probs.shape[0])]
    top_probs = probs[top_labels]

    return top_labels, top_probs


def predict(models, img, input_ids):
    pixel_values = preprocess(img, models["model_type"])

    # feedforward
    net = models["net"]
    if not args.onnx:
        output = net.predict([input_ids, pixel_values])
    else:
        output = net.run(None, {"input_ids": input_ids, "pixel_values": pixel_values})
    logits_per_image = output[0]

    return logits_per_image


def recognize_from_image(models):
    input_labels = args.text_inputs
    if input_labels is None:
        input_labels = ["2 cats", "a plane", "a remote", "3 dogs"]

    text_descriptions = [f"This is a photo of a {label}" for label in input_labels]

    tokenizer = models["tokenizer"]
    encoded = tokenizer(
        text_descriptions,
        return_tensors="np",
        padding=True,
        truncation=True,
    )
    input_ids = encoded["input_ids"]
    if not args.disable_ailia_tokenizer:
        input_ids = input_ids[:,1:] # remove bos token

    net = models["net"]

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
                logits_per_image = predict(models, img, input_ids)
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
            logits_per_image = predict(models, img, input_ids)

        top_labels, top_probs = postprocess_result(logits_per_image)

        # Show results
        a = [(input_labels[x], y) for x, y in zip(top_labels, top_probs)]
        for idx, (label, score) in enumerate(a):
            print(f"{idx + 1}: {label} - {score * 100 :.2f}%")

    logger.info("Script finished successfully.")


def main():
    # model files check and download
    dic_model = {
        "base-patch16-224": (WEIGHT_BASE_P16_224_PATH, MODEL_BASE_P16_224_PATH, None),
        "large-patch16-256": (
            WEIGHT_LARGE_P16_256_PATH,
            MODEL_LARGE_P16_256_PATH,
            PB_LARGE_P16_256_PATH,
        ),
        "giant-patch16-256": (
            WEIGHT_GIANT_P16_256_PATH,
            MODEL_GIANT_P16_256_PATH,
            PB_GIANT_P16_256_PATH,
        ),
    }
    model_type = args.model_type
    WEIGTH_PATH, MODEL_PATH, PB_PATH = dic_model[model_type]

    check_and_download_models(WEIGTH_PATH, MODEL_PATH, REMOTE_PATH)
    if PB_PATH:
        check_and_download_file(PB_PATH, REMOTE_PATH)

    env_id = args.env_id

    # initialize
    if not args.onnx:
        net = ailia.Net(MODEL_PATH, WEIGTH_PATH, env_id=env_id)
    else:
        import onnxruntime

        net = onnxruntime.InferenceSession(WEIGTH_PATH)

    if args.disable_ailia_tokenizer:
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(
            "tokenizer" if model_type == "giant-patch16-256" else "tokenizer-giant"
        )
    else:
        from ailia_tokenizer import GemmaTokenizer

        tokenizer = GemmaTokenizer.from_pretrained("tokenizer" if model_type == "giant-patch16-256" else "tokenizer-giant")

    models = dict(model_type=model_type, tokenizer=tokenizer, net=net)

    recognize_from_image(models)


if __name__ == "__main__":
    main()
