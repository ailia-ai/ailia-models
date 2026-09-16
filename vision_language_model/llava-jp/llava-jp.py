import sys
import time
from typing import List

# logger
from logging import getLogger  # noqa

import numpy as np
import cv2
from PIL import Image

import ailia

# import original modules
sys.path.append("../../util")
from arg_utils import get_base_parser, update_parser  # noqa
from model_utils import check_and_download_models, check_and_download_file  # noqa
from detector_utils import load_image  # noqa
from math_utils import softmax

from logit_process import logits_processor

logger = getLogger(__name__)

# ======================
# Parameters
# ======================

REMOTE_PATH = "https://storage.googleapis.com/ailia-models/llava-jp/"

IMAGE_PATH = "sample.jpg"


# ======================
# Arguemnt Parser Config
# ======================

parser = get_base_parser("LLaVA-JP", IMAGE_PATH, None, large_model=True)
parser.add_argument(
    "-p",
    "--prompt",
    type=str,
    default="猫の隣には何がありますか？",
    help="prompt",
)
parser.add_argument(
    "--disable_ailia_tokenizer", action="store_true", help="disable ailia tokenizer."
)
parser.add_argument(
    "-im",
    "--intermediate",
    action="store_true",
    help="print intermediate results.",
)
parser.add_argument("--onnx", action="store_true", help="execute onnxruntime version.")
args = update_parser(parser)


# ======================
# Model selection
# ======================


WEIGHT_PATH = "llava-jp-1.3b-v1.1.onnx"
WEIGHT_PB_PATH = "llava-jp-1.3b-v1.1_weights.pb"
WEIGHT_ENC_PATH = "encode_images.onnx"
MODEL_PATH = "llava-jp-1.3b-v1.1.onnx.prototxt"
MODEL_ENC_PATH = "encode_images.onnx.prototxt"

IMG_SIZE = 768

COPY_BLOB_DATA = True

SYSTEM_PROMPT = "これは好奇心旺盛なユーザーと人工知能システムのチャットです。システムはユーザーの質問に親切、詳細、丁寧に答える。"
IMAGE_TOKEN_INDEX = -200
STOP_STR = "<EOD|LLM-jp>"


# ======================
# Secondary Functions
# ======================


class TextStreamer:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        self.token_cache = []
        self.print_len = 0
        self.stop_str = STOP_STR

    def put(self, value):
        self.token_cache.extend(value.tolist())
        output_text = self.tokenizer.decode(self.token_cache)

        if output_text.endswith(self.stop_str) or output_text.endswith("\n"):
            printable_text = output_text[self.print_len :].replace(self.stop_str, "")
            self.token_cache = []
            self.print_len = 0
        elif output_text != "":
            printable_text = output_text[self.print_len :]
            self.print_len += len(printable_text)

        print(printable_text, flush=True, end="")

    def end(self):
        # Flush the cache, if it exists
        if len(self.token_cache) > 0:
            text = self.tokenizer.decode(self.token_cache)
            printable_text = text[self.print_len :]
            self.token_cache = []
            self.print_len = 0
        else:
            printable_text = ""

        print(printable_text, flush=True, end=None)


# ======================
# Main functions
# ======================


def preprocess(images):
    images = [im[:, :, ::-1] for im in images]  # BGR -> RGB

    height = width = IMG_SIZE
    images = [
        np.array(Image.fromarray(img).resize((width, height), Image.Resampling.BICUBIC))
        for img in images
    ]
    scale = 0.00392156862745098
    images = [(img.astype(np.float64) * scale).astype(np.float32) for img in images]

    mean = np.array([0.5, 0.5, 0.5], dtype=np.float32)
    std = np.array([0.5, 0.5, 0.5], dtype=np.float32)
    images = [(img - mean) / std for img in images]

    images = [img.transpose(2, 0, 1) for img in images]
    images = np.array(images, dtype=np.float32)

    return images


def prepare_inputs_for_multimodal(
    net_vis, input_ids, position_ids, attention_mask, past_key_values, images
):
    if input_ids.shape[1] == 1:
        target_shape = past_key_values[-1][-1].shape[-2] + 1
        attention_mask = np.concatenate(
            (
                attention_mask,
                np.ones(
                    (
                        attention_mask.shape[0],
                        target_shape - attention_mask.shape[1],
                    ),
                    dtype=attention_mask.dtype,
                ),
            ),
            axis=1,
        )
        position_ids = np.expand_dims(np.sum(attention_mask, axis=1), axis=-1) - 1
        return (
            input_ids,
            position_ids,
            past_key_values,
            None,
        )

    if args.benchmark:
        start = int(round(time.time() * 1000))

    if not args.onnx:
        output = net_vis.predict(
            [
                input_ids,
                attention_mask,
                images,
            ]
        )
    else:
        output = net_vis.run(
            None,
            {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "images": images,
            },
        )
    input_embeds, image_features = output

    if args.benchmark:
        end = int(round(time.time() * 1000))
        estimation_time = end - start
        logger.info(f"\tencode time {estimation_time} ms")

    cur_input_ids = input_ids[0]

    num_images = np.sum(cur_input_ids == IMAGE_TOKEN_INDEX)
    if num_images == 0:
        cur_image_features = image_features[cur_image_idx]
        cur_input_embeds_1 = input_embeds
        cur_input_embeds = np.concatenate(
            [cur_input_embeds_1, cur_image_features[:0]], axis=0
        )
        new_input_embeds = cur_input_embeds
    else:
        # ex. input_ids -> input_ids_noim
        # [1 2 3 -200 4 5 6] -> [1 2 3], [4 5 6]
        image_token_indices = (
            [-1]
            + np.where(cur_input_ids == IMAGE_TOKEN_INDEX)[0].tolist()
            + [cur_input_ids.shape[0]]
        )
        split_sizes = [
            image_token_indices[i + 1] - (image_token_indices[i] + 1)
            for i in range(len(image_token_indices) - 1)
        ]

        # 分割位置の計算
        split_indices = np.cumsum(split_sizes[:-1])
        input_embeds_no_im = np.split(input_embeds, split_indices, axis=0)

        # IMAGE_TOKEN_INDEXの部分を画像特徴量に置き換える
        new_input_embeds = []
        cur_image_idx = 0
        for i in range(num_images + 1):
            new_input_embeds.append(input_embeds_no_im[i])
            if i < num_images:
                cur_image_features = image_features[cur_image_idx]
                cur_image_idx += 1
                new_input_embeds.append(cur_image_features)

        new_input_embeds = np.concatenate(new_input_embeds)

    tokenizer_model_max_length = 1532
    new_input_embeds = new_input_embeds[:tokenizer_model_max_length]

    max_len = new_input_embeds.shape[0]

    new_input_embeds = np.expand_dims(new_input_embeds, axis=0)
    attention_mask = np.zeros((1, max_len), dtype=attention_mask.dtype)
    position_ids = np.zeros((1, max_len), dtype=position_ids.dtype)
    attention_mask[..., :max_len] = True
    position_ids[..., :max_len] = np.arange(0, max_len, dtype=position_ids.dtype)

    return (
        None,
        position_ids,
        past_key_values,
        new_input_embeds,
    )


def forward(
    models,
    input_ids: np.ndarray,
    position_ids: np.ndarray,
    attention_mask: np.ndarray,
    images: np.ndarray,
    past_key_values: List[np.ndarray],
    blob_copy,
):
    visual = models["visual"]
    (
        input_ids,
        position_ids,
        past_key_values,
        inputs_embeds,
    ) = prepare_inputs_for_multimodal(
        visual, input_ids, position_ids, attention_mask, past_key_values, images
    )

    if input_ids is None:
        input_ids = np.zeros((1, 0), dtype=np.int64)
    if inputs_embeds is None:
        inputs_embeds = np.zeros((1, 0, 2048), dtype=np.float32)

    net = models["net"]
    if not args.onnx:
        if not blob_copy:
            output = net.predict(
                [
                    input_ids,
                    position_ids,
                    inputs_embeds,
                    *past_key_values,
                ]
            )
            logits, new_past_key_values = output[0], output[1:]
        else:
            NUM_KV = 24
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
                inputs_embeds, net.find_blob_index_by_name("inputs_embeds")
            )
            net.set_input_blob_data(
                position_ids, net.find_blob_index_by_name("position_ids")
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
        output = net.run(
            None,
            {
                "input_ids": input_ids,
                "position_ids": position_ids,
                "inputs_embeds": inputs_embeds,
                "key_cache0": past_key_values[0],
                "value_cache0": past_key_values[1],
                "key_cache1": past_key_values[2],
                "value_cache1": past_key_values[3],
                "key_cache2": past_key_values[4],
                "value_cache2": past_key_values[5],
                "key_cache3": past_key_values[6],
                "value_cache3": past_key_values[7],
                "key_cache4": past_key_values[8],
                "value_cache4": past_key_values[9],
                "key_cache5": past_key_values[10],
                "value_cache5": past_key_values[11],
                "key_cache6": past_key_values[12],
                "value_cache6": past_key_values[13],
                "key_cache7": past_key_values[14],
                "value_cache7": past_key_values[15],
                "key_cache8": past_key_values[16],
                "value_cache8": past_key_values[17],
                "key_cache9": past_key_values[18],
                "value_cache9": past_key_values[19],
                "key_cache10": past_key_values[20],
                "value_cache10": past_key_values[21],
                "key_cache11": past_key_values[22],
                "value_cache11": past_key_values[23],
                "key_cache12": past_key_values[24],
                "value_cache12": past_key_values[25],
                "key_cache13": past_key_values[26],
                "value_cache13": past_key_values[27],
                "key_cache14": past_key_values[28],
                "value_cache14": past_key_values[29],
                "key_cache15": past_key_values[30],
                "value_cache15": past_key_values[31],
                "key_cache16": past_key_values[32],
                "value_cache16": past_key_values[33],
                "key_cache17": past_key_values[34],
                "value_cache17": past_key_values[35],
                "key_cache18": past_key_values[36],
                "value_cache18": past_key_values[37],
                "key_cache19": past_key_values[38],
                "value_cache19": past_key_values[39],
                "key_cache20": past_key_values[40],
                "value_cache20": past_key_values[41],
                "key_cache21": past_key_values[42],
                "value_cache21": past_key_values[43],
                "key_cache22": past_key_values[44],
                "value_cache22": past_key_values[45],
                "key_cache23": past_key_values[46],
                "value_cache23": past_key_values[47],
            },
        )
        logits, new_past_key_values = output[0], output[1:]

    return logits, new_past_key_values


def stopping_criteria(input_ids: np.array) -> np.array:
    max_length = 310
    cur_len = input_ids.shape[-1]
    is_done = cur_len >= max_length
    is_done = np.full(input_ids.shape[0], is_done)

    eos_token_id = np.array([7])
    is_done = is_done | np.isin(input_ids[:, -1], eos_token_id)

    return is_done


def sample(models, input_ids, attention_mask, images, intermediate=False):
    pad_token_id = 7

    if intermediate:
        streamer = TextStreamer(models["tokenizer"])
    else:
        streamer = None

    past_key_values = [np.zeros((1, 16, 0, 128), dtype=np.float32)] * 48

    # keep track of which sequences are already finished
    batch_size, cur_len = input_ids.shape
    this_peer_finished = False
    unfinished_sequences = np.ones(batch_size, dtype=int)
    cache_position = (
        np.cumsum(np.ones_like(input_ids[0, :], dtype=np.int64), axis=0) - 1
    )

    blob_copy = False
    while True:
        # prepare model inputs
        if 0 < past_key_values[0].shape[-2]:
            model_input_ids = input_ids[:, cache_position]
        else:
            model_input_ids = input_ids
        position_ids = attention_mask.astype(np.int32).cumsum(axis=-1) - 1
        if 0 < past_key_values[0].shape[-2]:
            position_ids = position_ids[:, -model_input_ids.shape[1] :]

        if args.benchmark:
            start = int(round(time.time() * 1000))

        logits, past_key_values = forward(
            models,
            model_input_ids,
            position_ids,
            attention_mask,
            images,
            past_key_values,
            blob_copy,
        )
        blob_copy = True if COPY_BLOB_DATA else False

        if args.benchmark:
            end = int(round(time.time() * 1000))
            estimation_time = end - start
            logger.info(f"\tdecode time {estimation_time} ms")

        attention_mask = np.concatenate(
            [attention_mask, np.ones((attention_mask.shape[0], 1), dtype=int)],
            axis=-1,
        )
        cache_position = cache_position[-1:] + 1

        next_token_logits = logits[:, -1, :]

        # pre-process distribution
        next_token_scores = logits_processor(input_ids, next_token_logits)

        # token selection
        probs = softmax(next_token_scores, axis=-1)
        next_tokens = np.random.choice(len(probs[0]), size=1, p=probs[0])

        # finished sentences should have their next token be a padding token
        next_tokens = next_tokens * unfinished_sequences + pad_token_id * (
            1 - unfinished_sequences
        )

        # update generated ids, model inputs, and length for next step
        input_ids = np.concatenate([input_ids, next_tokens[:, None]], axis=-1)

        if streamer:
            streamer.put(next_tokens)

        unfinished_sequences = unfinished_sequences & ~stopping_criteria(input_ids)
        this_peer_finished = np.max(unfinished_sequences) == 0
        cur_len += 1

        if this_peer_finished:
            break

    if streamer is not None:
        streamer.end()

    return input_ids


def predict(models, images, message, intermediate=False):
    images = preprocess(images)

    prompt = "".join(["<image>\n"] * len(images)) + message
    messages = [["ユーザー", prompt], ["システム", None]]

    seps = [" ", STOP_STR]
    ret = SYSTEM_PROMPT + seps[0]
    for i, (role, message) in enumerate(messages):
        if message:
            ret += role + ": " + message + seps[i % 2]
        else:
            ret += role + ": "
    prompt = ret

    tokenizer = models["tokenizer"]
    prompt_chunks = [tokenizer(chunk).input_ids for chunk in prompt.split("<image>")]

    input_ids = []
    offset = 0
    if (
        len(prompt_chunks) > 0
        and len(prompt_chunks[0]) > 0
        and prompt_chunks[0][0] == tokenizer.bos_token_id
    ):
        offset = 1
        input_ids.append(prompt_chunks[0][0])

    sep = [IMAGE_TOKEN_INDEX] * (offset + 1)
    for x in [
        ele
        for sublist in zip(prompt_chunks, [sep] * len(prompt_chunks))
        for ele in sublist
    ][:-1]:
        input_ids.extend(x[offset:])
    input_ids = np.array([input_ids], dtype=np.int64)
    input_ids = input_ids[:, :-1]  # </sep>がinputの最後に入るので削除する
    attention_mask = np.ones(input_ids.shape[:2], dtype=np.int64)

    output = sample(
        models, input_ids, attention_mask, images, intermediate=intermediate
    )
    output_text = tokenizer.decode(output[0][len(input_ids[0]) :])
    output_text = output_text.replace(STOP_STR, "")

    return output_text


def recognize(models):
    prompt = args.prompt
    intermediate = args.intermediate

    logger.info("Prompt: %s" % prompt)

    # input image loop
    images = []
    for image_path in args.input:
        # prepare input data
        img = load_image(image_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
        images.append(img)

    # inference
    logger.info("Start inference...")
    if args.benchmark:
        logger.info("BENCHMARK mode")
        total_time_estimation = 0
        for i in range(args.benchmark_count):
            start = int(round(time.time() * 1000))
            output_text = predict(models, images, prompt, intermediate=intermediate)
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
        output_text = predict(models, images, prompt, intermediate=intermediate)

    if not intermediate:
        print(output_text)

    logger.info("Script finished successfully.")


def main():
    check_and_download_models(WEIGHT_PATH, MODEL_PATH, REMOTE_PATH)
    check_and_download_models(WEIGHT_ENC_PATH, MODEL_ENC_PATH, REMOTE_PATH)
    check_and_download_file(WEIGHT_PB_PATH, REMOTE_PATH)

    env_id = args.env_id

    # initialize
    if not args.onnx:
        visual = ailia.Net(
            MODEL_ENC_PATH, WEIGHT_ENC_PATH, env_id=env_id, memory_mode=args.memory_mode
        )
        net = ailia.Net(MODEL_PATH, WEIGHT_PATH, env_id=env_id, memory_mode=args.memory_mode)
    else:
        import onnxruntime

        providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]

        visual = onnxruntime.InferenceSession(WEIGHT_ENC_PATH, providers=providers)
        net = onnxruntime.InferenceSession(WEIGHT_PATH, providers=providers)

    args.disable_ailia_tokenizer = True
    if args.disable_ailia_tokenizer:
        import transformers

        tokenizer = transformers.AutoTokenizer.from_pretrained("./tokenizer")
    else:
        raise NotImplementedError

    models = {
        "tokenizer": tokenizer,
        "visual": visual,
        "net": net,
    }

    # generate
    recognize(models)


if __name__ == "__main__":
    main()
