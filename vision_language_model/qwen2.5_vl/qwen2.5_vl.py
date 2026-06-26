import math
import sys
import time
from collections import defaultdict
from io import StringIO
from logging import getLogger
from typing import Any, Dict, List, Tuple, Union

import ailia
import cv2
import numpy as np
from PIL import Image

# import original modules
sys.path.append("../../util")
from arg_utils import get_base_parser, update_parser
from detector_utils import load_image
from math_utils import softmax
from model_utils import check_and_download_file, check_and_download_models

logger = getLogger(__name__)

# ======================
# Parameters
# ======================

REMOTE_PATH = "https://storage.googleapis.com/ailia-models/qwen2.5_vl/"

IMAGE_PATH = "demo.jpeg"

COPY_BLOB_DATA = True
INTERMEDIATE = True


# ======================
# Arguemnt Parser Config
# ======================

parser = get_base_parser("Qwen2.5-VL", IMAGE_PATH, None, large_model=True)
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
    "--temperature",
    type=float,
    default=1e-6,
    help="temperature from generation_config.json",
)
parser.add_argument(
    "--top_k",
    type=int,
    default=50,
    help="top_k from generation_config.json",
)
parser.add_argument(
    "--max_length",
    type=int,
    default=3730,
    help="max_length for generation",
)
parser.add_argument(
    "--disable_ailia_tokenizer", action="store_true", help="disable ailia tokenizer."
)
parser.add_argument(
    "--quantize", type=str, default=None, choices=["int4"],
    help="use int4 quantized model.",
)
parser.add_argument("--onnx", action="store_true", help="execute onnxruntime version.")
args = update_parser(parser)


# ======================
# Model selection
# ======================

if args.quantize == "int4":
    WEIGHT_PATH = "Qwen2.5-VL-3B_language_model_int4.onnx"
    MODEL_PATH = "Qwen2.5-VL-3B_language_model_int4.onnx.prototxt"
    WEIGHT_VISION_PATH = "Qwen2.5-VL-3B_vision_encoder_int4.onnx"
    MODEL_VISION_PATH = "Qwen2.5-VL-3B_vision_encoder_int4.onnx.prototxt"
    PB_PATH = "Qwen2.5-VL-3B_language_model_int4_weights.pb"
    PB_VISION_PATH = None
else:
    WEIGHT_PATH = "Qwen2.5-VL-3B_language_model.onnx"
    MODEL_PATH = "Qwen2.5-VL-3B_language_model.onnx.prototxt"
    WEIGHT_VISION_PATH = "Qwen2.5-VL-3B_vision_encoder.onnx"
    MODEL_VISION_PATH = "Qwen2.5-VL-3B_vision_encoder.onnx.prototxt"
    PB_PATH = "Qwen2.5-VL-3B_language_model_weights.pb"
    PB_VISION_PATH = "Qwen2.5-VL-3B_vision_encoder_weights.pb"

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
        return math.ceil(number / factor) * factor

    def floor_by_factor(number: int, factor: int) -> int:
        return math.floor(number / factor) * factor

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

    return h_bar, w_bar


def smart_nframes(
    total_frames: int,
    video_fps: Union[int, float],
) -> int:
    """calculate the number of frames for video used for model inputs."""
    FPS_MAX_FRAMES = 768
    FRAME_FACTOR = 2

    def floor_by_factor(number: int, factor: int) -> int:
        return math.floor(number / factor) * factor

    fps = args.fps or 2
    min_frames = 4
    max_frames = floor_by_factor(min(FPS_MAX_FRAMES, total_frames), FRAME_FACTOR)

    nframes = total_frames / video_fps * fps
    nframes = min(min(max(nframes, min_frames), max_frames), total_frames)
    nframes = floor_by_factor(nframes, FRAME_FACTOR)

    return nframes


def fetch_image(image_path: str, image_patch_size: int = 14) -> np.ndarray:
    img = load_image(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGRA2RGB)

    height, width, _ = img.shape

    SPATIAL_MERGE_SIZE = 2
    IMAGE_MIN_TOKEN_NUM = 4
    IMAGE_MAX_TOKEN_NUM = 16384

    MIN_PIXELS = IMAGE_MIN_TOKEN_NUM * image_patch_size**2
    MAX_PIXELS = IMAGE_MAX_TOKEN_NUM * image_patch_size**2
    min_pixels = args.min_pixels or MIN_PIXELS
    max_pixels = args.max_pixels or MAX_PIXELS
    patch_factor = int(image_patch_size * SPATIAL_MERGE_SIZE)
    resized_height, resized_width = smart_resize(
        height, width, factor=patch_factor, min_pixels=min_pixels, max_pixels=max_pixels
    )

    img = np.array(Image.fromarray(img).resize((resized_width, resized_height)))

    return img


def fetch_video(video_path: str, image_patch_size: int = 14) -> np.ndarray:
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
    nframes = smart_nframes(total_frames=total_frames, video_fps=video_fps)
    no = np.linspace(0, total_frames - 1, nframes).round().astype(int)
    frames = [x[1] for x in filter(lambda x: x[0] in no, enumerate(frames))]
    sample_fps = nframes / max(total_frames, 1e-6) * video_fps

    height, width, _ = frames[0].shape

    SPATIAL_MERGE_SIZE = 2
    VIDEO_MIN_TOKEN_NUM = 128
    VIDEO_MAX_TOKEN_NUM = 768
    MODEL_SEQ_LEN = 128000

    image_factor = image_patch_size * SPATIAL_MERGE_SIZE
    VIDEO_MIN_PIXELS = VIDEO_MIN_TOKEN_NUM * image_factor * image_factor
    VIDEO_MAX_PIXELS = VIDEO_MAX_TOKEN_NUM * image_factor * image_factor
    FRAME_FACTOR = 2
    min_pixels = args.min_pixels or VIDEO_MIN_PIXELS
    total_pixels = (
        args.total_pixels or MODEL_SEQ_LEN * image_factor * image_factor * 0.9
    )
    max_pixels = args.max_pixels or max(
        min(VIDEO_MAX_PIXELS, total_pixels / nframes * FRAME_FACTOR),
        int(min_pixels * 1.05),
    )
    resized_height, resized_width = smart_resize(
        height,
        width,
        factor=image_factor,
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
    video = np.stack(frames)
    video = video.transpose(0, 3, 1, 2)  # TCHW

    return video, sample_fps


def apply_chat_template(messages: List[dict]) -> str:
    buf = StringIO()
    image_count = 0
    video_count = 0
    add_vision_id = False  # Set to True if you want "Picture N:" / "Video N:" labels

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
                # Check for image: type=='image' or 'image' in content or 'image_url' in content
                if (
                    content.get("type") == "image"
                    or "image" in content
                    or "image_url" in content
                ):
                    image_count += 1
                    if add_vision_id:
                        buf.write(f"Picture {image_count}: ")
                    buf.write("<|vision_start|><|image_pad|><|vision_end|>")
                # Check for video: type=='video' or 'video' in content
                elif content.get("type") == "video" or "video" in content:
                    video_count += 1
                    if add_vision_id:
                        buf.write(f"Video {video_count}: ")
                    buf.write("<|vision_start|><|video_pad|><|vision_end|>")
                # Check for text
                elif "text" in content:
                    buf.write(f'{content["text"]}')
            buf.write("<|im_end|>\n")
    buf.write("<|im_start|>assistant\n")
    text = buf.getvalue()
    return text


def extract_vision_info(
    conversations: Union[List[Dict[str, Any]], List[List[Dict[str, Any]]]],
) -> List[Dict[str, Any]]:
    vision_infos = []
    if isinstance(conversations[0], dict):
        conversations = [conversations]
    for conversation in conversations:
        for message in conversation:
            if isinstance(message["content"], list):
                for ele in message["content"]:
                    if (
                        "image" in ele
                        or "image_url" in ele
                        or "video" in ele
                        or ele.get("type", "text") in ("image", "image_url", "video")
                    ):
                        vision_infos.append(ele)
    return vision_infos


def process_vision_info(
    conversations: Union[List[Dict[str, Any]], List[List[Dict[str, Any]]]],
    image_patch_size: int = 14,
):
    vision_infos = extract_vision_info(conversations)
    ## Read images or videos
    image_inputs = []
    video_inputs = []
    video_sample_fps_list = []
    for vision_info in vision_infos:
        if "image" in vision_info:
            image_inputs.append(
                fetch_image(vision_info["image"], image_patch_size=image_patch_size)
            )
        elif "video" in vision_info:
            video_input, video_sample_fps = fetch_video(
                vision_info["video"], image_patch_size=image_patch_size
            )
            video_sample_fps_list.append(video_sample_fps)
            video_inputs.append(video_input)
        else:
            raise ValueError("image, image_url or video should in content.")
    if len(image_inputs) == 0:
        image_inputs = None
    if len(video_inputs) == 0:
        video_inputs = None

    return image_inputs, video_inputs


# ======================
# Main functions
# ======================


def preprocess(images):
    longest_edge = 12845056
    shortest_edge = 56 * 56
    patch_size = 14
    merge_size = 2

    def _group_images_by_shape(images):
        """Helper function to flatten a single level of nested image structures and group by shape."""
        grouped_images = defaultdict(list)
        grouped_images_index = []
        for image in images:
            shape = image.shape[1:]
            grouped_images[shape].append(image)
            grouped_images_index.append((shape, len(grouped_images[shape]) - 1))

        # Stack images with the same shape
        grouped_images = {
            shape: np.stack(images_list, axis=0)
            for shape, images_list in grouped_images.items()
        }

        return grouped_images, grouped_images_index

    grouped_images, grouped_images_index = _group_images_by_shape(images)
    resized_images_grouped = {}
    for shape, stacked_images in grouped_images.items():
        height, width = stacked_images.shape[-2:]

        resized_height, resized_width = smart_resize(
            height,
            width,
            factor=patch_size * merge_size,
            min_pixels=shortest_edge,
            max_pixels=longest_edge,
        )
        resized_images = [
            np.array(
                Image.fromarray(img.transpose(1, 2, 0)).resize(
                    (resized_width, resized_height), Image.Resampling.BICUBIC
                )
            ).transpose(2, 0, 1)
            for img in stacked_images
        ]
        resized_images_grouped[shape] = np.stack(resized_images, axis=0)

    resized_images = [
        resized_images_grouped[shape][idx] for shape, idx in grouped_images_index
    ]

    temporal_patch_size = 2

    grouped_images, grouped_images_index = _group_images_by_shape(resized_images)
    processed_images_grouped = {}
    processed_grids = {}
    for shape, stacked_images in grouped_images.items():
        resized_height, resized_width = stacked_images.shape[-2:]
        # Fused rescale and normalize
        mean = np.array([0.48145467, 0.4578275, 0.40821072], dtype=np.float32)
        std = np.array([0.26862955, 0.2613026, 0.2757771], dtype=np.float32)

        # Rescale from 0-255 to 0-1
        patches = stacked_images.astype(np.float32) / 255.0

        # Normalize with mean and std
        mean = mean.reshape(1, 3, 1, 1)
        std = std.reshape(1, 3, 1, 1)
        patches = (patches - mean) / std
        if patches.ndim == 4:
            # add a temporal dimension if we have images
            patches = np.expand_dims(patches, axis=1)
        if patches.shape[1] % temporal_patch_size != 0:
            repeats = np.repeat(
                patches[:, -1:, :, :, :], temporal_patch_size - 1, axis=1
            )
            patches = np.concatenate([patches, repeats], axis=1)
        batch_size, grid_t, channel = patches.shape[:3]
        grid_t = grid_t // temporal_patch_size
        grid_h, grid_w = resized_height // patch_size, resized_width // patch_size

        patches = patches.reshape(
            batch_size,
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
        # Reorder dimensions to group grid and patch information for subsequent flattening.
        # (batch, grid_t, grid_h, grid_w, merge_h, merge_w, channel, temp_patch_size, patch_h, patch_w)
        patches = np.transpose(patches, (0, 1, 4, 7, 5, 8, 3, 2, 6, 9))
        flatten_patches = patches.reshape(
            batch_size,
            grid_t * grid_h * grid_w,
            channel * temporal_patch_size * patch_size * patch_size,
        )

        processed_images_grouped[shape] = flatten_patches
        processed_grids[shape] = [[grid_t, grid_h, grid_w]] * batch_size

    processed_images = [
        processed_images_grouped[shape][idx] for shape, idx in grouped_images_index
    ]
    processed_grids = [
        processed_grids[shape][idx] for shape, idx in grouped_images_index
    ]
    pixel_values = np.concatenate(processed_images, axis=0)
    image_grid_thw = np.array(processed_grids)

    data = {"pixel_values": pixel_values, "image_grid_thw": image_grid_thw}

    return data


def video_processor(videos):
    longest_edge = 12845056
    shortest_edge = 56 * 56
    patch_size = 14
    merge_size = 2

    def _group_videos_by_shape(videos):
        """
        Groups videos by shape.
        Returns a dictionary with the shape as key and a list of videos with that shape as value,
        and a list with the index of the video in the original list as key and the shape and index in the grouped list as value.
        """
        grouped_videos = {}
        grouped_videos_index = []
        for video in videos:
            shape = video.shape[-2::]
            num_frames = video.shape[0]  # video format BTCHW
            shape = (num_frames, *shape)
            if shape not in grouped_videos:
                grouped_videos[shape] = []
            grouped_videos[shape].append(video)
            grouped_videos_index.append((shape, len(grouped_videos[shape]) - 1))
        # stack videos with the same size and number of frames
        grouped_videos = {
            shape: np.stack(videos, axis=0) for shape, videos in grouped_videos.items()
        }
        return grouped_videos, grouped_videos_index

    grouped_videos, grouped_videos_index = _group_videos_by_shape(videos)
    resized_videos_grouped = {}
    for shape, stacked_videos in grouped_videos.items():
        height, width = stacked_videos[0].shape[-2:]

        resized_height, resized_width = smart_resize(
            height,
            width,
            factor=patch_size * merge_size,
            min_pixels=shortest_edge,
            max_pixels=longest_edge,
        )
        resized_videos = [
            np.stack(
                [
                    np.array(
                        Image.fromarray(frame.transpose(1, 2, 0)).resize(
                            (resized_width, resized_height), Image.Resampling.BICUBIC
                        )
                    ).transpose(2, 0, 1)
                    for frame in video
                ],
                axis=0,
            )
            for video in stacked_videos
        ]
        resized_videos_grouped[shape] = np.stack(resized_videos, axis=0)

    resized_videos = [
        resized_videos_grouped[shape][idx] for shape, idx in grouped_videos_index
    ]

    temporal_patch_size = 2

    grouped_videos, grouped_videos_index = _group_videos_by_shape(resized_videos)
    processed_videos_grouped = {}
    processed_grids = {}
    for shape, stacked_videos in grouped_videos.items():
        resized_height, resized_width = stacked_videos[0].shape[-2:]
        # Fused rescale and normalize
        mean = np.array([0.48145467, 0.4578275, 0.40821072], dtype=np.float32)
        std = np.array([0.26862955, 0.2613026, 0.2757771], dtype=np.float32)

        # Rescale from 0-255 to 0-1
        patches = stacked_videos.astype(np.float32) / 255.0

        # Normalize with mean and std
        mean = mean.reshape(1, 3, 1, 1)
        std = std.reshape(1, 3, 1, 1)
        patches = (patches - mean) / std

        # Check that videos have `num_frames` divisible by `temporal_patch_size`
        if patches.shape[1] % temporal_patch_size != 0:
            repeats = np.repeat(
                patches[:, -1:, :, :, :], temporal_patch_size - 1, axis=1
            )
            patches = np.concatenate([patches, repeats], axis=1)
        batch_size, grid_t, channel = patches.shape[:3]
        grid_t = grid_t // temporal_patch_size
        grid_h, grid_w = resized_height // patch_size, resized_width // patch_size

        patches = patches.reshape(
            batch_size,
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
        # Reorder dimensions to group grid and patch information for subsequent flattening.
        # (batch, grid_t, grid_h, grid_w, merge_h, merge_w, channel, temp_patch_size, patch_h, patch_w)
        patches = np.transpose(patches, (0, 1, 4, 7, 5, 8, 3, 2, 6, 9))
        flatten_patches = patches.reshape(
            batch_size,
            grid_t * grid_h * grid_w,
            channel * temporal_patch_size * patch_size * patch_size,
        )

        processed_videos_grouped[shape] = flatten_patches
        processed_grids[shape] = [[grid_t, grid_h, grid_w]] * batch_size

    processed_videos = [
        processed_videos_grouped[shape][idx] for shape, idx in grouped_videos_index
    ]
    processed_grids = [
        processed_grids[shape][idx] for shape, idx in grouped_videos_index
    ]
    pixel_values_videos = np.concatenate(processed_videos, axis=0)
    video_grid_thw = np.array(processed_grids)

    data = {
        "pixel_values_videos": pixel_values_videos,
        "video_grid_thw": video_grid_thw,
    }

    return data


def logits_processor(input_ids, scores):
    top_k = args.top_k
    temperature = args.temperature

    penalty = 1.05
    # Convert to numpy if needed (assuming scores and input_ids are already numpy arrays)
    score = np.take_along_axis(scores, input_ids, axis=1)
    # if score < 0 then repetition penalty has to be multiplied to reduce the token probabilities
    score = np.where(score < 0, score * penalty, score / penalty)
    scores = scores.copy()
    np.put_along_axis(scores, input_ids, score, axis=1)

    scores = scores / temperature

    top_k = min(top_k, scores.shape[-1])  # Safety check
    # Remove all tokens with a probability less than the last token of the top-k
    top_k_values = np.partition(scores, -top_k, axis=-1)[..., -top_k:]
    top_k_min = np.min(top_k_values, axis=-1, keepdims=True)
    indices_to_remove = scores < top_k_min
    scores = np.where(indices_to_remove, -float("inf"), scores)

    return scores


def forward(
    net,
    input_ids: np.ndarray,
    inputs_embeds: np.ndarray,
    position_ids: np.ndarray,
    past_key_values: List[np.ndarray],
    first_run,
):
    if not args.onnx:
        if first_run or COPY_BLOB_DATA == False:
            output = net.predict(
                [
                    input_ids,
                    inputs_embeds,
                    position_ids,
                    *past_key_values,
                ]
            )
            logits, new_past_key_values = output[0], output[1:]
        else:
            NUM_KV = 36
            key_shapes = []
            value_shapes = []
            for i in range(NUM_KV):
                key_shapes.append(
                    net.get_blob_shape(net.find_blob_index_by_name(f"present.{i}.key"))
                )
                value_shapes.append(
                    net.get_blob_shape(
                        net.find_blob_index_by_name(f"present.{i}.value")
                    )
                )
            net.set_input_blob_data(input_ids, net.find_blob_index_by_name("input_ids"))
            net.set_input_blob_data(
                inputs_embeds, net.find_blob_index_by_name("inputs_embeds")
            )
            net.set_input_blob_data(
                position_ids, net.find_blob_index_by_name("position_ids")
            )
            for i in range(NUM_KV):
                net.set_input_blob_shape(
                    key_shapes[i],
                    net.find_blob_index_by_name(f"past_key_values.{i}.key"),
                )
                net.set_input_blob_shape(
                    value_shapes[i],
                    net.find_blob_index_by_name(f"past_key_values.{i}.value"),
                )
                net.copy_blob_data(f"past_key_values.{i}.key", f"present.{i}.key")
                net.copy_blob_data(f"past_key_values.{i}.value", f"present.{i}.value")
            net.update()
            logits = net.get_blob_data(net.find_blob_index_by_name("logits"))
            new_past_key_values = None
    else:
        onnx_inputs = {
            "input_ids": input_ids,
            "inputs_embeds": inputs_embeds,
            "position_ids": position_ids,
        }
        for i in range(len(past_key_values) // 2):
            onnx_inputs[f"past_key_values.{i}.key"] = past_key_values[i * 2]
            onnx_inputs[f"past_key_values.{i}.value"] = past_key_values[i * 2 + 1]

        output = net.run(None, onnx_inputs)
        logits, new_past_key_values = output[0], output[1:]

    return logits, new_past_key_values


def get_rope_index(input_ids, image_grid_thw, video_grid_thw, attention_mask):
    spatial_merge_size = 2
    image_token_id = 151655
    video_token_id = 151656
    vision_start_token_id = 151652
    mrope_position_deltas = []

    total_input_ids = input_ids
    attention_mask = attention_mask == 1
    position_ids = np.ones(
        (3, input_ids.shape[0], input_ids.shape[1]),
        dtype=input_ids.dtype,
    )
    image_index, video_index = 0, 0
    for i, input_ids in enumerate(total_input_ids):
        input_ids = input_ids[attention_mask[i]]
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
                second_per_grid_t = 0
                image_index += 1
                remain_images -= 1
                ed = ed_image
            else:
                t, h, w = (
                    video_grid_thw[video_index][0],
                    video_grid_thw[video_index][1],
                    video_grid_thw[video_index][2],
                )
                second_per_grid_t = 1.0
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

            range_tensor = np.arange(llm_grid_t).reshape(-1, 1)
            expanded_range = np.tile(range_tensor, (1, llm_grid_h * llm_grid_w))

            # normalize type
            second_per_grid_t = np.array(second_per_grid_t, dtype=range_tensor.dtype)

            tokens_per_second = 0.03125  # self.config.vision_config.tokens_per_second
            time_tensor = expanded_range * second_per_grid_t * tokens_per_second

            time_tensor_long = time_tensor.astype(np.int64)
            t_index = time_tensor_long.flatten()

            h_index = np.tile(
                np.arange(llm_grid_h).reshape(1, -1, 1), (llm_grid_t, 1, llm_grid_w)
            ).flatten()
            w_index = np.tile(
                np.arange(llm_grid_w).reshape(1, 1, -1), (llm_grid_t, llm_grid_h, 1)
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
        position_ids[..., i, attention_mask[i]] = llm_positions
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
                # clean_up_tokenization_spaces=False,
            )
    except UnicodeDecodeError:
        if intermediate:
            return [""]
        raise
    return output_text


def sample(
    models,
    input_ids,
    attention_mask,
    pixel_values=None,
    pixel_values_videos=None,
    image_grid_thw=None,
    video_grid_thw=None,
):
    pad_token_id = 151643
    image_token_id = 151655
    video_token_id = 151656
    max_length = args.max_length

    if INTERMEDIATE:
        initial_ids = input_ids.copy()

    if args.benchmark:
        start = int(round(time.time() * 1000))

    if INTERMEDIATE:
        print("Encoding..." + "\n\u001b[2A")
        before_text = ""

    inputs_embeds = np.zeros((1, 0, 2048), dtype=np.float32)

    net = models["vision_encoder"]
    if pixel_values is not None:
        if not args.onnx:
            output = net.predict(
                [
                    input_ids,
                    inputs_embeds,
                    pixel_values,
                    image_grid_thw,
                    np.array(image_token_id),
                ]
            )
        else:
            output = net.run(
                None,
                {
                    "input_ids": input_ids,
                    "inputs_embeds": inputs_embeds,
                    "pixel_values": pixel_values,
                    "image_grid_thw": image_grid_thw,
                    "image_token_id": np.array(image_token_id),
                },
            )
        inputs_embeds = output[0]
    if pixel_values_videos is not None:
        if not args.onnx:
            output = net.predict(
                [
                    input_ids,
                    inputs_embeds,
                    pixel_values_videos,
                    video_grid_thw,
                    np.array(video_token_id),
                ]
            )
        else:
            output = net.run(
                None,
                {
                    "input_ids": input_ids,
                    "inputs_embeds": inputs_embeds,
                    "pixel_values": pixel_values_videos,
                    "image_grid_thw": video_grid_thw,
                    "image_token_id": np.array(video_token_id),
                },
            )
        inputs_embeds = output[0]

    past_key_values = [
        np.zeros((1, 2, 0, 128), dtype=np.float32) for _ in range(36 * 2)
    ]

    if args.benchmark:
        end = int(round(time.time() * 1000))
        estimation_time = end - start
        logger.info(f"\tencode time {estimation_time} ms")

    # keep track of which sequences are already finished
    batch_size, cur_len = input_ids.shape
    this_peer_finished = False
    unfinished_sequences = np.ones(batch_size, dtype=int)
    cache_position = np.cumsum(np.ones(cur_len, dtype=np.int64), axis=0) - 1

    tokenizer = models["tokenizer"]
    net = models["language_model"]
    first_run = True
    rope_deltas = None
    while not this_peer_finished:
        # prepare model inputs
        model_input_ids = input_ids
        if model_input_ids.shape[1] != cache_position.shape[0]:
            model_input_ids = model_input_ids[:, cache_position]

        position_ids = np.cumsum(attention_mask.astype(np.int64), axis=-1) - 1
        position_ids[attention_mask == 0] = 1
        current_input_length = model_input_ids.shape[1]
        position_ids = position_ids[:, -current_input_length:]
        if cache_position[0] == 0:
            vision_positions, rope_deltas = get_rope_index(
                model_input_ids, image_grid_thw, video_grid_thw, attention_mask
            )
        else:
            batch_size, seq_length = position_ids.shape
            vision_position_ids = np.arange(seq_length, dtype=np.int64)
            vision_position_ids = np.tile(
                vision_position_ids.reshape(1, 1, -1), (3, batch_size, 1)
            )
            delta = cache_position[0] + rope_deltas
            delta = np.repeat(delta, batch_size // delta.shape[0], axis=0)
            vision_positions = vision_position_ids + np.broadcast_to(
                delta, vision_position_ids.shape
            )

        text_positions = position_ids[None, ...]
        position_ids = np.concatenate([text_positions, vision_positions], axis=0)

        if args.benchmark:
            start = int(round(time.time() * 1000))

        logits, past_key_values = forward(
            net,
            model_input_ids,
            inputs_embeds,
            position_ids,
            past_key_values,
            first_run,
        )
        first_run = False
        inputs_embeds = inputs_embeds[:, :0, :]

        if args.benchmark:
            end = int(round(time.time() * 1000))
            estimation_time = end - start
            logger.info(f"\tdecode time {estimation_time} ms")

        # update_model_kwargs_for_generation
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
        batch_size, vocab_size = probs.shape
        next_tokens = np.array(
            [np.random.choice(vocab_size, p=probs[i]) for i in range(batch_size)]
        )

        # finished sentences should have their next token be a padding token
        next_tokens = next_tokens * unfinished_sequences + pad_token_id * (
            1 - unfinished_sequences
        )

        # update generated ids, model inputs, and length for next step
        input_ids = np.concatenate([input_ids, next_tokens[:, None]], axis=-1)

        if INTERMEDIATE:
            output_text = tokenizer_decode(initial_ids, input_ids, tokenizer, True)[0]
            if output_text.startswith(before_text):
                deltaText = output_text[len(before_text) :]
            else:
                deltaText = output_text
            print(deltaText, end="")
            sys.stdout.flush()
            if output_text != "":
                before_text = output_text

        unfinished_sequences = unfinished_sequences & ~stopping_criteria(
            input_ids, max_length=max_length
        )
        this_peer_finished = np.max(unfinished_sequences) == 0

    return input_ids


def predict(models, messages):
    text = apply_chat_template(messages)
    images, videos = process_vision_info(messages)

    image_inputs = videos_inputs = {}
    if images is not None:
        images = [img.transpose(2, 0, 1) for img in images]
        image_inputs = preprocess(images)
        image_grid_thw = image_inputs["image_grid_thw"]
    if videos is not None:
        videos_inputs = video_processor(videos)
        video_grid_thw = videos_inputs["video_grid_thw"]

        fps = 2.0
        temporal_patch_size = 2
        second_per_grid_ts = [temporal_patch_size / fps] * len(video_grid_thw)
        videos_inputs.update({"second_per_grid_ts": second_per_grid_ts})

    text = [text]

    merge_size = 2

    image_token = "<|image_pad|>"
    if images is not None:
        merge_length = merge_size**2
        index = 0
        for i in range(len(text)):
            while image_token in text[i]:
                num_image_tokens = image_grid_thw[index].prod() // merge_length
                text[i] = text[i].replace(
                    image_token, "<|placeholder|>" * num_image_tokens, 1
                )
                index += 1
            text[i] = text[i].replace("<|placeholder|>", image_token)

    video_token = "<|video_pad|>"
    if videos is not None:
        merge_length = merge_size**2
        index = 0
        for i in range(len(text)):
            while video_token in text[i]:
                num_video_tokens = video_grid_thw[index].prod() // merge_length
                text[i] = text[i].replace(
                    video_token, "<|placeholder|>" * num_video_tokens, 1
                )
                index += 1
            text[i] = text[i].replace("<|placeholder|>", video_token)

    tokenizer = models["tokenizer"]
    if args.disable_ailia_tokenizer:
        text_inputs = tokenizer(
            text,
            return_tensors="np",
            padding=True,
        )
    else:
        text_inputs = tokenizer(
            text,
            return_tensors="np",
            padding=True,
        )

    input_ids = text_inputs["input_ids"]
    attention_mask = text_inputs["attention_mask"]
    generated_ids = sample(
        models,
        input_ids=input_ids,
        attention_mask=attention_mask,
        pixel_values=image_inputs.get("pixel_values"),
        pixel_values_videos=videos_inputs.get("pixel_values_videos"),
        image_grid_thw=image_inputs.get("image_grid_thw"),
        video_grid_thw=videos_inputs.get("video_grid_thw"),
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
        total_time_estimation = 0
        for i in range(args.benchmark_count):
            start = int(round(time.time() * 1000))
            output_text = predict(models, messages)
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
        output_text = predict(models, messages)

    if INTERMEDIATE:
        print("")
    else:
        print(output_text)

    logger.info("Script finished successfully.")


def main():
    check_and_download_models(WEIGHT_PATH, MODEL_PATH, REMOTE_PATH)
    check_and_download_models(WEIGHT_VISION_PATH, MODEL_VISION_PATH, REMOTE_PATH)
    if PB_PATH is not None:
        check_and_download_file(PB_PATH, REMOTE_PATH)
    if PB_VISION_PATH is not None:
        check_and_download_file(PB_VISION_PATH, REMOTE_PATH)

    env_id = args.env_id

    # initialize
    if not args.onnx:
        memory_mode = ailia.get_memory_mode(
            reduce_constant=True,
            ignore_input_with_initializer=True,
            reduce_interstage=False,
            reuse_interstage=True,
        )
        vision_encoder = ailia.Net(
            MODEL_VISION_PATH,
            WEIGHT_VISION_PATH,
            env_id=env_id,
            memory_mode=memory_mode,
        )
        language_model = ailia.Net(
            MODEL_PATH, WEIGHT_PATH, env_id=env_id, memory_mode=memory_mode
        )
    else:
        import onnxruntime

        providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]

        vision_encoder = onnxruntime.InferenceSession(
            WEIGHT_VISION_PATH, providers=providers
        )
        sess_options = onnxruntime.SessionOptions()
        if args.quantize is not None:
            sess_options.graph_optimization_level = (
                onnxruntime.GraphOptimizationLevel.ORT_DISABLE_ALL
            )
        language_model = onnxruntime.InferenceSession(
            WEIGHT_PATH, sess_options=sess_options, providers=providers
        )

    if args.disable_ailia_tokenizer:
        import transformers

        tokenizer = transformers.Qwen2TokenizerFast.from_pretrained("./tokenizer")
    else:
        from ailia_tokenizer import GPT2Tokenizer

        tokenizer = GPT2Tokenizer.from_pretrained("./tokenizer")
        tokenizer.add_special_tokens(
            {
                "additional_special_tokens": [
                    "<|end_of_text|>",
                    "<|im_start|>",
                    "<|im_end|>",
                    "<|object_ref_start|>",
                    "<|object_ref_end|>",
                    "<|box_start|>",
                    "<|box_end|>",
                    "<|quad_start|>",
                    "<|quad_end|>",
                    "<|vision_start|>",
                    "<|vision_end|>",
                    "<|vision_pad|>",
                    "<|image_pad|>",
                    "<|video_pad|>",
                ]
            }
        )

    models = {
        "tokenizer": tokenizer,
        "vision_encoder": vision_encoder,
        "language_model": language_model,
    }

    # generate
    recognize(models)


if __name__ == "__main__":
    main()
