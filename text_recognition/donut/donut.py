import sys
import time
import json
import re
from logging import getLogger
from typing import List

import cv2
import numpy as np
from PIL import Image

import ailia

# import original modules
sys.path.append("../../util")
from detector_utils import load_image
from model_utils import check_and_download_models
from image_utils import normalize_image
from arg_utils import get_base_parser, update_parser

logger = getLogger(__name__)


# ======================
# Parameters
# ======================
DEFAULT_MODEL = "donut-base-finetuned-cord-v2"

REMOTE_PATH = "https://storage.googleapis.com/ailia-models/donut/"

IMAGE_PATH = "cord_sample_receipt1.png"

COPY_BLOB_DATA = True


# ======================
# Arguemnt Parser Config
# ======================

parser = get_base_parser("Donut: Document Understanding Transformer", IMAGE_PATH, None)
parser.add_argument("--model_name", default=DEFAULT_MODEL, help="model name")
parser.add_argument("--onnx", action="store_true", help="execute onnxruntime version.")
args = update_parser(parser)


# ======================
# Utils
# ======================


def token2json(tokenizer, tokens, is_inner_value=False):
    """
    Convert a (generated) token seuqnce into an ordered JSON format
    """
    output = dict()

    while tokens:
        start_token = re.search(r"<s_(.*?)>", tokens, re.IGNORECASE)
        if start_token is None:
            break
        key = start_token.group(1)
        end_token = re.search(rf"</s_{key}>", tokens, re.IGNORECASE)
        start_token = start_token.group()
        if end_token is None:
            tokens = tokens.replace(start_token, "")
        else:
            end_token = end_token.group()
            start_token_escaped = re.escape(start_token)
            end_token_escaped = re.escape(end_token)
            content = re.search(
                f"{start_token_escaped}(.*?){end_token_escaped}", tokens, re.IGNORECASE
            )
            if content is not None:
                content = content.group(1).strip()
                if r"<s_" in content and r"</s_" in content:  # non-leaf node
                    value = token2json(tokenizer, content, is_inner_value=True)
                    if value:
                        if len(value) == 1:
                            value = value[0]
                        output[key] = value
                else:  # leaf nodes
                    output[key] = []
                    for leaf in content.split(r"<sep/>"):
                        leaf = leaf.strip()
                        if (
                            leaf in tokenizer.get_added_vocab()
                            and leaf[0] == "<"
                            and leaf[-2:] == "/>"
                        ):
                            leaf = leaf[1:-2]  # for categorical special tokens
                        output[key].append(leaf)
                    if len(output[key]) == 1:
                        output[key] = output[key][0]

            tokens = tokens[tokens.find(end_token) + len(end_token) :].strip()
            if tokens[:6] == r"<sep/>":  # non-leaf nodes
                return [output] + token2json(tokenizer, tokens[6:], is_inner_value=True)

    if len(output):
        return [output] if is_inner_value else output
    else:
        return [] if is_inner_value else {"text_sequence": tokens}


# ======================
# Main functions
# ======================


def preprocess(img):
    input_size = (1280, 960)
    im_h, im_w, _ = img.shape

    size = min(input_size)
    short, long = (im_w, im_h) if im_w <= im_h else (im_h, im_w)
    new_short, new_long = size, size * long // short
    ow, oh = (new_short, new_long) if im_w <= im_h else (new_long, new_short)
    if ow != im_w or oh != im_h:
        img = np.array(Image.fromarray(img).resize((ow, oh), Image.Resampling.BILINEAR))

    # PIL.thumbnail
    scale = min(input_size[1] / ow, input_size[0] / oh)
    ow = int(ow * scale)
    oh = int(oh * scale)
    img = np.array(Image.fromarray(img).resize((ow, oh), Image.Resampling.BICUBIC))

    delta_width = input_size[1] - ow
    delta_height = input_size[0] - oh
    pad_width = delta_width // 2
    pad_height = delta_height // 2
    img = np.pad(
        img,
        (
            (pad_height, delta_height - pad_height),
            (pad_width, delta_width - pad_width),
            (0, 0),
        ),
        "constant",
    )

    img = normalize_image(img, normalize_type="ImageNet")

    img = img.transpose(2, 0, 1)  # HWC -> CHW
    img = np.expand_dims(img, axis=0)
    img = img.astype(np.float16)

    return img


def forward(
    net,
    input_ids: np.ndarray,
    attention_mask: np.ndarray,
    encoder_hidden_states: np.ndarray,
    past_key_values: List[np.ndarray],
    blob_copy: bool,
):
    if not args.onnx:
        if not blob_copy:
            output = net.predict(
                [
                    input_ids,
                    attention_mask,
                    encoder_hidden_states,
                    *past_key_values,
                ]
            )
            logits, new_past_key_values = output[0], output[1:]
        else:
            NUM_KV = 8
            key_shapes = [
                net.get_blob_shape(
                    net.find_blob_index_by_name("key_cache_out" + str(i))
                )
                for i in range(NUM_KV)
            ]
            value_shapes = [
                net.get_blob_shape(
                    net.find_blob_index_by_name("value_cache_out" + str(i))
                )
                for i in range(NUM_KV)
            ]
            net.set_input_blob_data(input_ids, net.find_blob_index_by_name("input_ids"))
            net.set_input_blob_data(
                attention_mask, net.find_blob_index_by_name("attention_mask")
            )
            net.set_input_blob_data(
                encoder_hidden_states,
                net.find_blob_index_by_name("encoder_hidden_states"),
            )
            for i in range(NUM_KV):
                net.set_input_blob_shape(
                    key_shapes[i], net.find_blob_index_by_name("key_cache" + str(i))
                )
                net.set_input_blob_shape(
                    value_shapes[i], net.find_blob_index_by_name("value_cache" + str(i))
                )
                net.copy_blob_data("key_cache" + str(i), "key_cache_out" + str(i))
                net.copy_blob_data("value_cache" + str(i), "value_cache_out" + str(i))
            net.update()
            logits = net.get_blob_data(net.find_blob_index_by_name("logits"))
            new_past_key_values = [
                net.get_blob_data(net.find_blob_index_by_name("key_cache_out0"))
            ]
    else:
        key_cache = {"key_cache%d" % i: past_key_values[i * 2] for i in range(8)}
        value_cache = {
            "value_cache%d" % i: past_key_values[i * 2 + 1] for i in range(8)
        }
        output = net.run(
            None,
            {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "encoder_hidden_states": encoder_hidden_states,
                **key_cache,
                **value_cache,
            },
        )
        logits, new_past_key_values = output[0], output[1:]

    return logits, new_past_key_values


def logits_processor(scores):
    bad_words_id_length_1 = 3
    bad_words_mask = np.zeros(scores.shape[1])
    bad_words_mask[bad_words_id_length_1] = 1
    bad_words_mask = np.expand_dims(bad_words_mask, axis=0)

    scores = np.where(bad_words_mask, -np.inf, scores)

    return scores


def stopping_criteria(input_ids: np.array) -> np.array:
    max_length = 768
    cur_len = input_ids.shape[-1]
    is_done = cur_len >= max_length
    is_done = np.full(input_ids.shape[0], is_done)

    eos_token_id = np.array([151643])
    is_done = is_done | np.isin(input_ids[:, -1], eos_token_id)

    return is_done


def greedy_search(models, input_ids, last_hidden_state):
    eos_token_id = 2

    # keep track of which sequences are already finished
    batch_size, cur_len = input_ids.shape
    unfinished_sequences = np.ones(batch_size, dtype=int)

    past_key_values = [np.zeros((1, 16, 0, 64), dtype=np.float16)] * 16

    blob_copy = False
    while True:
        # prepare model inputs
        if 0 < past_key_values[0].shape[1]:
            model_input_ids = input_ids[:, -1:]
        else:
            model_input_ids = input_ids
        pad_token_id = 1
        attention_mask = (input_ids != pad_token_id).astype(np.int64)

        if args.benchmark:
            start = int(round(time.time() * 1000))

        net = models["net"]
        logits, past_key_values = forward(
            net,
            model_input_ids,
            attention_mask,
            last_hidden_state,
            past_key_values,
            blob_copy,
        )
        blob_copy = True if COPY_BLOB_DATA else False

        if args.benchmark:
            end = int(round(time.time() * 1000))
            estimation_time = end - start
            logger.info(f"\tdecode time {estimation_time} ms")

        next_token_logits = logits[:, -1, :]

        # pre-process distribution
        next_token_scores = logits_processor(next_token_logits)

        # token selection
        next_tokens = np.argmax(next_token_scores, axis=-1)

        # finished sentences should have their next token be a padding token
        next_tokens = next_tokens * unfinished_sequences + pad_token_id * (
            1 - unfinished_sequences
        )

        # update generated ids, model inputs, and length for next step
        input_ids = np.concatenate([input_ids, next_tokens[:, None]], axis=-1)
        cur_len += 1

        unfinished_sequences = unfinished_sequences * (next_tokens != eos_token_id)
        if np.max(unfinished_sequences) == 0 or stopping_criteria(input_ids):
            break

    return input_ids


def predict(models, img):
    img = img[:, :, ::-1]  # BGR -> RGB
    img = preprocess(img)

    task_prompt = "<s_cord-v2>"
    tokenizer = models["tokenizer"]
    input_ids = tokenizer(task_prompt, add_special_tokens=False, return_tensors="np")[
        "input_ids"
    ]

    net = models["enc"]
    if not args.onnx:
        output = net.predict([img])
    else:
        output = net.run(None, {"x": img})
    last_hidden_state = output[0]

    sequences = greedy_search(models, input_ids, last_hidden_state)

    output = []
    for seq in tokenizer.batch_decode(sequences):
        seq = seq.replace(tokenizer.eos_token, "").replace(tokenizer.pad_token, "")
        seq = re.sub(
            r"<.*?>", "", seq, count=1
        ).strip()  # remove first task start token
        output.append(token2json(tokenizer, seq))

    return output[0]


def recognize_from_image(models):
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
                output = predict(models, img)
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
            output = predict(models, img)

        print(json.dumps(output, indent=4, ensure_ascii=False))

    logger.info("Script finished successfully.")


def main():
    model_name = args.model_name
    WEIGHT_PATH = f"{model_name}.onnx"
    WEIGHT_ENC_PATH = f"{model_name}_encoder.onnx"
    MODEL_PATH = f"{model_name}.onnx.prototxt"
    MODEL_ENC_PATH = f"{model_name}_encoder.onnx.prototxt"

    # model files check and download
    check_and_download_models(WEIGHT_PATH, MODEL_PATH, REMOTE_PATH)
    check_and_download_models(WEIGHT_ENC_PATH, MODEL_ENC_PATH, REMOTE_PATH)

    env_id = args.env_id

    # initialize
    if not args.onnx:
        mem_mode = ailia.get_memory_mode(
            reduce_constant=True, ignore_input_with_initializer=True,
            reduce_interstage=False, reuse_interstage=True)
        net = ailia.Net(MODEL_PATH, WEIGHT_PATH, env_id=env_id, memory_mode=mem_mode)
        enc = ailia.Net(MODEL_ENC_PATH, WEIGHT_ENC_PATH, env_id=env_id, memory_mode=mem_mode)
    else:
        import onnxruntime

        providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
        net = onnxruntime.InferenceSession(WEIGHT_PATH, providers=providers)
        enc = onnxruntime.InferenceSession(WEIGHT_ENC_PATH, providers=providers)

    args.disable_ailia_tokenizer = True
    if args.disable_ailia_tokenizer:
        import transformers

        tokenizer = transformers.XLMRobertaTokenizer.from_pretrained("./tokenizer")
    else:
        raise NotImplementedError

    models = dict(
        tokenizer=tokenizer,
        enc=enc,
        net=net,
    )

    recognize_from_image(models)


if __name__ == "__main__":
    main()
