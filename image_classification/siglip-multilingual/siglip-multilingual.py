import string
import sys
import time
from logging import getLogger

import ailia
import cv2
import numpy as np
from PIL import Image

# import original modules
sys.path.append("../../util")
from arg_utils import get_base_parser, update_parser
from detector_utils import load_image
from math_utils import sigmoid
from model_utils import check_and_download_models

logger = getLogger(__name__)


# ======================
# Parameters
# ======================

WEIGHT_PATH = "siglip-base-patch16-256-multilingual.onnx"
MODEL_PATH = "siglip-base-patch16-256-multilingual.onnx.prototxt"
REMOTE_PATH = (
    "https://storage.googleapis.com/ailia-models/siglip-multilingual/"
)

IMAGE_PATH = "demo.jpg"
IMAGE_SIZE = 256
SEQ_LENGTH = 64

# transformers' zero-shot-image-classification pipeline formats candidate
# labels through this template before tokenizing.
HYPOTHESIS_TEMPLATE = "This is a photo of {}."

# SiglipTokenizer has no attention_mask and pads with eos_token_id (1), not
# ailia_tokenizer's default 0. The exported model expects this pad id.
PAD_TOKEN_ID = 1


# ======================
# Arguemnt Parser Config
# ======================

parser = get_base_parser("SigLIP (base-sized model, multilingual)", IMAGE_PATH, None)
parser.add_argument(
    "-t",
    "--text",
    dest="text_inputs",
    type=str,
    action="append",
    help="Candidate label, e.g. 'a plane' (not a full sentence; can be "
    f"specified multiple times). Formatted via {HYPOTHESIS_TEMPLATE!r} "
    "before being fed to the model.",
)
parser.add_argument(
    "--disable_ailia_tokenizer", action="store_true", help="disable ailia tokenizer."
)
parser.add_argument("--onnx", action="store_true", help="execute onnxruntime version.")
args = update_parser(parser)


# ======================
# Secondary Functions
# ======================


def canonicalize_text(text):
    # Mirrors SiglipTokenizer.canonicalize_text(): lowercase + strip
    # punctuation. ailia_tokenizer doesn't do this on its own.
    text = text.lower()
    return text.translate(str.maketrans("", "", string.punctuation))


# ======================
# Main functions
# ======================


def preprocess(img):
    img = img[:, :, ::-1]  # BGR -> RGB

    # resize (preprocessor_config.json: resample=3 -> PIL BICUBIC)
    img = np.array(
        Image.fromarray(img).resize((IMAGE_SIZE, IMAGE_SIZE), Image.Resampling.BICUBIC)
    )

    rescale_factor = 0.00392156862745098
    img = img.astype(np.float32) * rescale_factor
    img = (img - 0.5) / 0.5

    img = img.transpose(2, 0, 1)  # HWC -> CHW
    img = np.expand_dims(img, axis=0)

    return img


def tokenize(tokenizer, texts):
    texts = [canonicalize_text(text) for text in texts]
    encoded = tokenizer(
        texts, padding="max_length", max_length=SEQ_LENGTH, return_tensors="np"
    )
    input_ids = np.array(encoded["input_ids"], dtype=np.int64)
    if "attention_mask" in encoded.keys():
        # ailia_tokenizer pads with 0; fix up to PAD_TOKEN_ID.
        attention_mask = np.array(encoded["attention_mask"])
        input_ids[attention_mask == 0] = PAD_TOKEN_ID

    return input_ids


def postprocess_result(logits_per_image, top_k: int = 5):
    probs = sigmoid(logits_per_image)[0]

    top_labels = np.argsort(-probs)[: min(top_k, probs.shape[0])]
    top_probs = probs[top_labels]

    return top_labels, top_probs


def predict(net, img, input_ids):
    pixel_values = preprocess(img)

    # feedforward
    if not args.onnx:
        output = net.predict([pixel_values, input_ids])
    else:
        output = net.run(None, {"pixel_values": pixel_values, "input_ids": input_ids})
    logits_per_image, logits_per_text, image_embeds, text_embeds = output

    return logits_per_image


def recognize_from_image(net, tokenizer):
    input_labels = args.text_inputs
    if input_labels is None:
        input_labels = [
            "2 cats",
            "a plane",
            "a remote",
        ]

    texts = [HYPOTHESIS_TEMPLATE.format(label) for label in input_labels]
    input_ids = tokenize(tokenizer, texts)

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
                logits_per_image = predict(net, img, input_ids)
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
            logits_per_image = predict(net, img, input_ids)

        top_labels, top_probs = postprocess_result(logits_per_image)

        # Show results
        a = [(input_labels[x], y) for x, y in zip(top_labels, top_probs)]
        for idx, (label, score) in enumerate(a):
            print(f"{idx + 1}: {label} - {score * 100 :.2f}%")

    logger.info("Script finished successfully.")


def main():
    # model files check and download
    check_and_download_models(WEIGHT_PATH, MODEL_PATH, REMOTE_PATH)

    env_id = args.env_id

    # initialize
    if not args.onnx:
        net = ailia.Net(MODEL_PATH, WEIGHT_PATH, env_id=env_id)
    else:
        import onnxruntime

        net = onnxruntime.InferenceSession(WEIGHT_PATH)

    if args.disable_ailia_tokenizer:
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained("tokenizer")
    else:
        from ailia_tokenizer import T5Tokenizer

        tokenizer = T5Tokenizer.from_pretrained("tokenizer")

    recognize_from_image(net, tokenizer)


if __name__ == "__main__":
    main()
