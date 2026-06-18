import sys
import time
from typing import List, Tuple
from io import StringIO
import platform

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

REMOTE_PATH = "https://storage.googleapis.com/ailia-models/qwen2_vl/"

IMAGE_PATH = "demo.jpeg"
SAVE_IMAGE_PATH = "output.png"

COPY_BLOB_DATA = True
INTERMEDIATE = True


# ======================
# Arguemnt Parser Config
# ======================

parser = get_base_parser("Qwen2-VL", IMAGE_PATH, SAVE_IMAGE_PATH, large_model = True)
parser.add_argument(
    "-p",
    "--prompt",
    type=str,
    default="Describe this image.",
    help="prompt",
)
parser.add_argument(
    "--min_pixels",
    type=int,
    default=None,
    help="min_pixels",
)
parser.add_argument(
    "--max_pixels",
    type=int,
    default=None,
    help="max_pixels",
)
parser.add_argument(
    "--total_pixels",
    type=int,
    default=None,
    help="total_pixels",
)
parser.add_argument(
    "--fps",
    type=int,
    default=None,
    help="fps",
)
parser.add_argument(
    "--disable_ailia_tokenizer", action="store_true", help="disable ailia tokenizer."
)
parser.add_argument(
    "--fp16", action="store_true", help="use fp16 model.",
)
parser.add_argument(
    "--quantize", type=str, default=None, choices=["int4"],
    help="use int4 quantized model.",
)
parser.add_argument(
    "--temperature",
    type=float,
    default=0.01,
    help="temperature from generation_config.json",
)
parser.add_argument(
    "--top_p",
    type=float,
    default=0.001,
    help="top_p from generation_config.json",
)
parser.add_argument(
    "--top_k",
    type=int,
    default=1,
    help="top_k from generation_config.json",
)
parser.add_argument(
    "--max_length",
    type=int,
    default=256,
    help="max_length for generation",
)
parser.add_argument(
    "--normal", action="store_true", help="use normal model (default : opt model)."
)
parser.add_argument("--onnx", action="store_true", help="execute onnxruntime version.")
args = update_parser(parser)


# ======================
# Model selection
# ======================

FP16 = ""
if args.fp16:
    FP16 = "_fp16"

OPT = ".opt"
if args.normal:
    OPT = ""

if args.quantize == "int4":
    WEIGHT_PATH = "Qwen2-VL-2B_int4.onnx"
    WEIGHT_VIS_PATH = "Qwen2-VL-2B_vis_int4.onnx"
    MODEL_PATH = "Qwen2-VL-2B_int4.onnx.prototxt"
    MODEL_VIS_PATH = "Qwen2-VL-2B_vis_int4.onnx.prototxt"
    PB_PATH = None
    PB_VIS_PATH = None
else:
    WEIGHT_PATH = "Qwen2-VL-2B" + FP16 + ".onnx"
    WEIGHT_VIS_PATH = "Qwen2-VL-2B_vis" + FP16 + OPT + ".onnx"
    MODEL_PATH = "Qwen2-VL-2B" + FP16 + ".onnx.prototxt"
    MODEL_VIS_PATH = "Qwen2-VL-2B_vis" + FP16 + OPT + ".onnx.prototxt"
    if args.fp16:
        PB_PATH = "Qwen2-VL-2B_weights_fp16.pb"
        PB_VIS_PATH = None
    else:
        PB_PATH = "Qwen2-VL-2B_weights.pb"
        PB_VIS_PATH = "Qwen2-VL-2B_vis_weights.pb"


# ======================
# Secondary Functions
# ======================


def smart_resize(
    height: int,
    width: int,
    factor: int = 28,
    min_pixels: int = 4 * 28 * 28,
    max_pixels: int = 16384 * 28 * 28,
) -> Tuple[int, int]:
    def round_by_factor(number: int, factor: int) -> int:
        return round(number / factor) * factor

    def ceil_by_factor(number: int, factor: int) -> int:
        return np.ceil(number / factor) * factor

    def floor_by_factor(number: int, factor: int) -> int:
        return np.floor(number / factor) * factor

    h_bar = max(factor, round_by_factor(height, factor))
    w_bar = max(factor, round_by_factor(width, factor))
    if h_bar * w_bar > max_pixels:
        beta = np.sqrt((height * width) / max_pixels)
        h_bar = floor_by_factor(height / beta, factor)
        w_bar = floor_by_factor(width / beta, factor)
    elif h_bar * w_bar < min_pixels:
        beta = np.sqrt(min_pixels / (height * width))
        h_bar = ceil_by_factor(height * beta, factor)
        w_bar = ceil_by_factor(width * beta, factor)

    return int(h_bar), int(w_bar)


def fetch_image(image_path: str) -> np.ndarray:
    img = load_image(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGRA2RGB)

    height, width, _ = img.shape

    MIN_PIXELS = 4 * 28 * 28
    MAX_PIXELS = 16384 * 28 * 28
    min_pixels = args.min_pixels or MIN_PIXELS
    max_pixels = args.max_pixels or MAX_PIXELS
    resized_height, resized_width = smart_resize(
        height, width, min_pixels=min_pixels, max_pixels=max_pixels
    )

    img = np.array(Image.fromarray(img).resize((resized_width, resized_height)))

    return img


def fetch_video(video_path: str) -> np.ndarray:
    cap = cv2.VideoCapture(video_path)
    video_fps = cap.get(cv2.CAP_PROP_FPS)
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2RGB)
        frames.append(frame)

    total_frames = len(frames)
    fps = args.fps or 2
    min_frames = 4
    max_frames = 30
    nframes = total_frames / video_fps * fps
    nframes = min(max(nframes, min_frames), max_frames)
    nframes = round(nframes / 2) * 2
    no = np.linspace(0, total_frames - 1, nframes)
    frames = [x[1] for x in filter(lambda x: x[0] in no, enumerate(frames))]

    height, width, _ = frames[0].shape

    VIDEO_MIN_PIXELS = 128 * 28 * 28
    VIDEO_MAX_PIXELS = 768 * 28 * 28
    VIDEO_TOTAL_PIXELS = 24576 * 28 * 28
    FRAME_FACTOR = 2
    min_pixels = args.min_pixels or VIDEO_MIN_PIXELS
    total_pixels = args.total_pixels or VIDEO_TOTAL_PIXELS
    max_pixels = args.max_pixels or max(
        min(VIDEO_MAX_PIXELS, total_pixels / nframes * FRAME_FACTOR),
        int(min_pixels * 1.05),
    )
    resized_height, resized_width = smart_resize(
        height,
        width,
        min_pixels=min_pixels,
        max_pixels=max_pixels,
    )

    frames = [
        np.array(
            Image.fromarray(frame).resize(
                (resized_width, resized_height),
                Image.Resampling.BICUBIC,
            )
        )
        for frame in frames
    ]

    return frames


# ======================
# Main functions
# ======================


def preprocess(images):
    height, width, _ = images[0].shape

    max_pixels = 12845056
    min_pixels = 56 * 56
    patch_size = 14
    merge_size = 2
    factor = patch_size * merge_size

    h_bar = round(height / factor) * factor
    w_bar = round(width / factor) * factor
    if h_bar * w_bar > max_pixels:
        beta = np.sqrt((height * width) / max_pixels)
        h_bar = np.floor(height / beta / factor) * factor
        w_bar = np.floor(width / beta / factor) * factor
    elif h_bar * w_bar < min_pixels:
        beta = np.sqrt(min_pixels / (height * width))
        h_bar = np.ceil(height * beta / factor) * factor
        w_bar = np.ceil(width * beta / factor) * factor
    resized_height, resized_width = h_bar, w_bar

    patches = []
    for img in images:
        img = np.array(
            Image.fromarray(img).resize(
                (resized_width, resized_height), Image.Resampling.BICUBIC
            )
        )

        mean = np.array([0.48145467, 0.4578275, 0.40821072], dtype=np.float32)
        std = np.array([0.26862955, 0.2613026, 0.2757771], dtype=np.float32)
        img = img / 255
        img = (img - mean) / std

        img = img.transpose(2, 0, 1)  # HWC -> CHW
        # img = np.expand_dims(img, axis=0)
        img = img.astype(np.float32)

        patches.append(img)

    patches = np.array(patches)

    temporal_patch_size = 2
    if patches.shape[0] == 1:
        patches = np.tile(patches, (temporal_patch_size, 1, 1, 1))

    channel = patches.shape[1]
    grid_t = patches.shape[0] // temporal_patch_size
    grid_h, grid_w = resized_height // patch_size, resized_width // patch_size
    patches = patches.reshape(
        grid_t,
        temporal_patch_size,
        channel,
        grid_h // merge_size,
        merge_size,
        patch_size,
        grid_w // merge_size,
        merge_size,
        patch_size,
    )
    patches = patches.transpose(0, 3, 6, 4, 7, 2, 1, 5, 8)
    flatten_patches = patches.reshape(
        grid_t * grid_h * grid_w,
        channel * temporal_patch_size * patch_size * patch_size,
    )

    return flatten_patches, (grid_t, grid_h, grid_w)


def forward(
    net,
    input_ids: np.ndarray,
    inputs_embeds: np.ndarray,
    position_ids: np.ndarray,
    attention_mask: np.ndarray,
    past_key_values: List[np.ndarray],
    first_run,
):
    if not args.onnx:
        if first_run or COPY_BLOB_DATA == False:
            output = net.predict(
                [input_ids, inputs_embeds, position_ids, attention_mask, *past_key_values]
            )
            logits, new_past_key_values = output[0], output[1:]
        else:
            NUM_KV = 28
            key_shapes = []
            value_shapes = []
            for i in range(NUM_KV):
                key_shapes.append(net.get_blob_shape(net.find_blob_index_by_name("key_cache_out"+str(i))))
                value_shapes.append(net.get_blob_shape(net.find_blob_index_by_name("value_cache_out"+str(i))))
            net.set_input_blob_data( input_ids, net.find_blob_index_by_name("input_ids"))
            net.set_input_blob_data( inputs_embeds, net.find_blob_index_by_name("inputs_embeds"))
            net.set_input_blob_data( position_ids, net.find_blob_index_by_name("position_ids"))
            net.set_input_blob_data( attention_mask, net.find_blob_index_by_name("attention_mask"))
            for i in range(NUM_KV):
                net.set_input_blob_shape(key_shapes[i], net.find_blob_index_by_name("key_cache"+str(i)))
                net.set_input_blob_shape(value_shapes[i], net.find_blob_index_by_name("value_cache"+str(i)))
                net.copy_blob_data( "key_cache"+str(i), "key_cache_out"+str(i))
                net.copy_blob_data( "value_cache"+str(i), "value_cache_out"+str(i))
            net.update()
            logits = net.get_blob_data( net.find_blob_index_by_name("logits") )
            new_past_key_values = None
    else:
        output = net.run(
            None,
            {
                "input_ids": input_ids.astype(np.int64),
                "inputs_embeds": inputs_embeds,
                "position_ids": position_ids.astype(np.int64),
                "attention_mask": attention_mask.astype(np.int64),
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
                "key_cache24": past_key_values[48],
                "value_cache24": past_key_values[49],
                "key_cache25": past_key_values[50],
                "value_cache25": past_key_values[51],
                "key_cache26": past_key_values[52],
                "value_cache26": past_key_values[53],
                "key_cache27": past_key_values[54],
                "value_cache27": past_key_values[55],
            },
        )
        logits, new_past_key_values = output[0], output[1:]

    return logits, new_past_key_values


def get_rope_index(input_ids, image_grid_thw, video_grid_thw, attention_mask):
    spatial_merge_size = 2
    image_token_id = 151655
    video_token_id = 151656
    vision_start_token_id = 151652
    mrope_position_deltas = []

    total_input_ids = input_ids
    position_ids = np.ones(
        (3, input_ids.shape[0], input_ids.shape[1]),
        dtype=input_ids.dtype,
    )
    image_index, video_index = 0, 0
    for i, input_ids in enumerate(total_input_ids):
        input_ids = input_ids[attention_mask[i] == 1]
        image_nums, video_nums = 0, 0
        vision_start_indices = np.argwhere(input_ids == vision_start_token_id).squeeze(
            1
        )
        vision_tokens = input_ids[vision_start_indices + 1]
        image_nums = np.sum(vision_tokens == image_token_id)
        video_nums = np.sum(vision_tokens == video_token_id)
        input_tokens = input_ids.tolist()
        llm_pos_ids_list: list = []
        st = 0
        remain_images, remain_videos = image_nums, video_nums
        for _ in range(image_nums + video_nums):
            if image_token_id in input_tokens and remain_images > 0:
                ed_image = input_tokens.index(image_token_id, st)
            else:
                ed_image = len(input_tokens) + 1
            if video_token_id in input_tokens and remain_videos > 0:
                ed_video = input_tokens.index(video_token_id, st)
            else:
                ed_video = len(input_tokens) + 1
            if ed_image < ed_video:
                t, h, w = (
                    image_grid_thw[image_index][0],
                    image_grid_thw[image_index][1],
                    image_grid_thw[image_index][2],
                )
                image_index += 1
                remain_images -= 1
                ed = ed_image
            else:
                t, h, w = (
                    video_grid_thw[video_index][0],
                    video_grid_thw[video_index][1],
                    video_grid_thw[video_index][2],
                )
                video_index += 1
                remain_videos -= 1
                ed = ed_video
            llm_grid_t, llm_grid_h, llm_grid_w = (
                t.item(),
                h.item() // spatial_merge_size,
                w.item() // spatial_merge_size,
            )
            text_len = ed - st

            st_idx = llm_pos_ids_list[-1].max() + 1 if len(llm_pos_ids_list) > 0 else 0
            llm_pos_ids_list.append(
                np.tile(np.arange(text_len).reshape(1, -1), (3, 1)) + st_idx
            )

            t_index = np.tile(
                np.arange(llm_grid_t).reshape(-1, 1), (1, llm_grid_h * llm_grid_w)
            ).flatten()
            h_index = np.tile(
                np.arange(llm_grid_h).reshape(1, -1, 1),
                (llm_grid_t, 1, llm_grid_w),
            ).flatten()
            w_index = np.tile(
                np.arange(llm_grid_w).reshape(1, 1, -1),
                (llm_grid_t, llm_grid_h, 1),
            ).flatten()
            llm_pos_ids_list.append(
                np.stack([t_index, h_index, w_index]) + text_len + st_idx
            )
            st = ed + llm_grid_t * llm_grid_h * llm_grid_w

        if st < len(input_tokens):
            st_idx = llm_pos_ids_list[-1].max() + 1 if len(llm_pos_ids_list) > 0 else 0
            text_len = len(input_tokens) - st
            llm_pos_ids_list.append(
                np.tile(np.arange(text_len).reshape(1, -1), (3, 1)) + st_idx
            )

        llm_positions = np.concatenate(llm_pos_ids_list, axis=1).reshape(3, -1)
        position_ids[..., i, attention_mask[i] == 1] = llm_positions
        mrope_position_deltas.append(llm_positions.max() + 1 - len(total_input_ids[i]))

    mrope_position_deltas = np.expand_dims(np.array(mrope_position_deltas), axis=1)
    return position_ids, mrope_position_deltas


def stopping_criteria(input_ids: np.array, max_length) -> np.array:
    cur_len = input_ids.shape[-1]
    is_done = cur_len >= max_length
    is_done = np.full(input_ids.shape[0], is_done)

    eos_token_id = np.array([151645, 151643])
    is_done = is_done | np.isin(input_ids[:, -1], eos_token_id)

    return is_done


def tokenizer_decode(input_ids, generated_ids, tokenizer, intermediate):
    generated_ids_trimmed = [
        out_ids[len(in_ids) :] for in_ids, out_ids in zip(input_ids, generated_ids)
    ]
    try:
        if args.disable_ailia_tokenizer:
            output_text = tokenizer.batch_decode(
                generated_ids_trimmed,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )
        else:
            output_text = tokenizer.batch_decode(
                generated_ids_trimmed,
                skip_special_tokens=True,
                #clean_up_tokenization_spaces=False,
            )
    except UnicodeDecodeError:
        if intermediate:
            return [""]
        raise
    return output_text


def sample(
    models,
    input_ids,
    pixel_values,
    attention_mask,
    image_grid_thw,
    video_grid_thw,
    tokenizer,
):
    pad_token_id = 151643
    image_token_id = 151655
    video_token_id = 151656
    if INTERMEDIATE:
        initial_ids = input_ids.copy()

    pixel_values = (
        pixel_values
        if pixel_values is not None
        # dummy for no image
        else np.zeros((16, 1176), dtype=np.float32)
    )
    image_token_id = (
        np.array([image_token_id])
        if image_grid_thw is not None
        else (
            np.array([video_token_id])
            if video_grid_thw is not None
            # dummy for no image
            else np.array([-1], dtype=int)
        )
    )
    image_grid_thw = (
        image_grid_thw
        if image_grid_thw is not None
        else (
            video_grid_thw
            if video_grid_thw is not None
            # dummy for no image
            else np.array([[1, 4, 4]], dtype=int)
        )
    )

    if args.benchmark:
        start = int(round(time.time() * 1000))

    if INTERMEDIATE:
        print("Encoding..." + "\n\u001B[2A")
        before_text = ""

    net = models["visual"]
    if not args.onnx:
        output = net.predict([input_ids, pixel_values, image_grid_thw, image_token_id])
    else:
        output = net.run(
            None,
            {
                "input_ids": input_ids.astype(np.int64),
                "pixel_values": pixel_values,
                "image_grid_thw": image_grid_thw.astype(np.int64),
                "image_token_id": image_token_id.astype(np.int64),
            },
        )
    inputs_embeds = output[0]
    past_key_values = [
        np.zeros((1, 2, 0, 128), dtype=np.float32) for _ in range(28 * 2)
    ]

    if args.benchmark:
        end = int(round(time.time() * 1000))
        estimation_time = end - start
        logger.info(f"\tencode time {estimation_time} ms")

    # keep track of which sequences are already finished
    batch_size, cur_len = input_ids.shape
    this_peer_finished = False
    unfinished_sequences = np.ones(batch_size, dtype=int)
    cache_position = (
        np.cumsum(np.ones_like(input_ids[0, :], dtype=np.int64), axis=0) - 1
    )
    rope_deltas = None
    max_length = args.max_length + input_ids.shape[1]

    net = models["net"]
    first_run = True
    while True:
        # prepare model inputs
        model_input_ids = input_ids
        if model_input_ids.shape[1] != cache_position.shape[0]:
            model_input_ids = model_input_ids[:, cache_position]
        batch_size, seq_length = model_input_ids.shape
        if cache_position[0] == 0:
            position_ids, rope_deltas = get_rope_index(
                model_input_ids, image_grid_thw, video_grid_thw, attention_mask
            )
        else:
            delta = cache_position[0] + rope_deltas if rope_deltas is not None else 0
            position_ids = np.arange(seq_length)
            position_ids = np.tile(position_ids.reshape(1, -1), (batch_size, 1))
            position_ids = position_ids + delta
            position_ids = np.tile(np.expand_dims(position_ids, axis=0), (3, 1, 1))
        if cache_position[0] != 0:
            # Disable inputs_embeds parameter if cache is used
            inputs_embeds = inputs_embeds[:0, :1, :]

        if args.benchmark:
            start = int(round(time.time() * 1000))

        logits, past_key_values = forward(
            net,
            model_input_ids,
            inputs_embeds,
            position_ids,
            attention_mask,
            past_key_values,
            first_run
        )
        first_run = False

        if args.benchmark:
            end = int(round(time.time() * 1000))
            estimation_time = end - start
            logger.info(f"\tdecode time {estimation_time} ms")

        next_token_logits = logits[:, -1, :]

        # pre-process distribution
        next_token_scores = logits_processor(input_ids, next_token_logits, args.temperature, args.top_p, args.top_k)

        # token selection
        probs = softmax(next_token_scores, axis=-1)
        next_tokens = np.random.choice(len(probs[0]), size=1, p=probs[0])

        # finished sentences should have their next token be a padding token
        next_tokens = next_tokens * unfinished_sequences + pad_token_id * (
            1 - unfinished_sequences
        )

        # update generated ids, model inputs, and length for next step
        input_ids = np.concatenate([input_ids, next_tokens[:, None]], axis=-1)

        attention_mask = np.concatenate(
            [attention_mask, np.ones((attention_mask.shape[0], 1), dtype=int)],
            axis=-1,
        )
        cache_position = cache_position[-1:] + 1

        unfinished_sequences = unfinished_sequences & ~stopping_criteria(
            input_ids, max_length
        )
        this_peer_finished = np.max(unfinished_sequences) == 0
        cur_len += 1

        if this_peer_finished:
            break

        if INTERMEDIATE and not args.benchmark:
            output_text = tokenizer_decode(initial_ids, input_ids, tokenizer, True)[0]
            if output_text.startswith(before_text):
                deltaText = output_text[len(before_text):]
            else:
                deltaText = output_text
            print(deltaText, end="")
            sys.stdout.flush()
            if output_text != "":
                before_text = output_text

    return input_ids


def predict(models, messages):
    buf = StringIO()
    for i, message in enumerate(messages):
        if i == 0 and message["role"] != "system":
            buf.write("<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n")
        buf.write("<|im_start|>")
        buf.write(f'{message["role"]}\n')
        if isinstance(message["content"], str):
            buf.write(f'{message["content"]}')
            buf.write("<|im_end|>\n")
        else:
            for content in message["content"]:
                if content["type"] == "image" and "image" in content:
                    buf.write("<|vision_start|><|image_pad|><|vision_end|>")
                elif content["type"] == "video" and "video" in content:
                    buf.write("<|vision_start|><|video_pad|><|vision_end|>")
                elif "text" in content:
                    buf.write(f'{content["text"]}')
            buf.write("<|im_end|>\n")
    buf.write("<|im_start|>assistant\n")
    text = buf.getvalue()

    image_inputs = []
    video_inputs = []
    for message in messages:
        for ele in message["content"]:
            if "image" in ele:
                img = fetch_image(ele["image"])
                image_inputs.append(img)
            if "video" in ele:
                video = fetch_video(ele["video"])
                video_inputs.append(video)

    pixel_values = None
    image_grid_thw = None
    video_grid_thw = None
    if image_inputs:
        pixel_values = []
        vision_grid_thws = []
        for img in image_inputs:
            patches, vision_grid_thw = preprocess([img])
            pixel_values.extend(patches)
            vision_grid_thws.append(vision_grid_thw)
        pixel_values = np.array(pixel_values)
        image_grid_thw = np.array(vision_grid_thws)
    if video_inputs:
        pixel_values = []
        vision_grid_thws = []
        for images in video_inputs:
            patches, vision_grid_thw = preprocess(images)
            pixel_values.extend(patches)
            vision_grid_thws.append(vision_grid_thw)
        pixel_values = np.array(pixel_values)
        video_grid_thw = np.array(vision_grid_thws)

    merge_length = 4
    if image_inputs:
        index = 0
        while "<|image_pad|>" in text:
            text = text.replace(
                "<|image_pad|>",
                "<|placeholder|>" * (image_grid_thw[index].prod() // merge_length),
                1,
            )
            index += 1
        text = text.replace("<|placeholder|>", "<|image_pad|>")
    if video_inputs:
        index = 0
        while "<|video_pad|>" in text:
            text = text.replace(
                "<|video_pad|>",
                "<|placeholder|>" * (video_grid_thw[index].prod() // merge_length),
                1,
            )
            index += 1
        text = text.replace("<|placeholder|>", "<|video_pad|>")

    text = [text]

    tokenizer = models["tokenizer"]
    if args.disable_ailia_tokenizer:
        text_inputs = tokenizer(
            text,
            return_tensors="np",
            padding=True,
            padding_side="left",
        )
    else:
        text_inputs = tokenizer(
            text,
            return_tensors="np",
            padding=True,
            #padding_side="left",
        )

    input_ids = text_inputs["input_ids"]
    attention_mask = text_inputs["attention_mask"]

    generated_ids = sample(
        models,
        input_ids,
        pixel_values,
        attention_mask,
        image_grid_thw,
        video_grid_thw,
        tokenizer,
    )

    output_text = tokenizer_decode(input_ids, generated_ids, tokenizer, False)

    return output_text[0]


def recognize(models):
    prompt = args.prompt
    logger.info("Prompt: %s" % prompt)

    content = []
    if args.video is not None:
        content.append({"type": "video", "video": args.video})
    else:
        for input_path in args.input:
            content.append({"type": "image", "image": input_path})
    content.append({"type": "text", "text": prompt})

    messages = [{"role": "user", "content": content}]

    # inference
    logger.info("Start inference...")
    if args.benchmark:
        logger.info("BENCHMARK mode")
        start = int(round(time.time() * 1000))
        output_text = predict(models, messages)
        end = int(round(time.time() * 1000))
        estimation_time = end - start
        logger.info(f"\tailia processing estimation time {estimation_time} ms")
    else:
        output_text = predict(models, messages)

    if INTERMEDIATE and not args.benchmark:
        print("")
    else:
        print(output_text)

    logger.info("Script finished successfully.")


def main():
    check_and_download_models(WEIGHT_PATH, MODEL_PATH, REMOTE_PATH)
    check_and_download_models(WEIGHT_VIS_PATH, MODEL_VIS_PATH, REMOTE_PATH)
    if PB_PATH is not None:
        check_and_download_file(PB_PATH, REMOTE_PATH)
    if PB_VIS_PATH is not None:
        check_and_download_file(PB_VIS_PATH, REMOTE_PATH)
    
    env_id = args.env_id

    # initialize
    if not args.onnx:
        memory_mode = ailia.get_memory_mode(
            reduce_constant=True,
            ignore_input_with_initializer=True,
            reduce_interstage=False,
            reuse_interstage=True,
        )
        visual = ailia.Net(
            MODEL_VIS_PATH, WEIGHT_VIS_PATH, env_id=env_id, memory_mode=memory_mode
        )
        net = ailia.Net(MODEL_PATH, WEIGHT_PATH, env_id=env_id, memory_mode=memory_mode)
    else:
        import onnxruntime

        providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]

        visual = onnxruntime.InferenceSession(WEIGHT_VIS_PATH, providers=providers)

        sess_options = onnxruntime.SessionOptions()
        if args.quantize is not None:
            sess_options.graph_optimization_level = (
                onnxruntime.GraphOptimizationLevel.ORT_DISABLE_ALL
            )
        net = onnxruntime.InferenceSession(
            WEIGHT_PATH, sess_options=sess_options, providers=providers
        )

    #args.disable_ailia_tokenizer = True
    if args.disable_ailia_tokenizer:
        import transformers

        tokenizer = transformers.Qwen2TokenizerFast.from_pretrained("./tokenizer")
    else:
        from ailia_tokenizer import GPT2Tokenizer

        tokenizer = GPT2Tokenizer.from_pretrained("./tokenizer")
        tokenizer.add_special_tokens({"additional_special_tokens":['<|end_of_text|>', '<|im_start|>', '<|im_end|>', '<|object_ref_start|>', '<|object_ref_end|>', '<|box_start|>', '<|box_end|>', '<|quad_start|>', '<|quad_end|>', '<|vision_start|>', '<|vision_end|>', '<|vision_pad|>', '<|image_pad|>', '<|video_pad|>']})

    models = {
        "tokenizer": tokenizer,
        "visual": visual,
        "net": net,
    }

    # generate
    recognize(models)


if __name__ == "__main__":
    main()
