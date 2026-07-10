import math
import sys
import time
from io import StringIO
from logging import getLogger

import numpy as np
from PIL import Image as PILImage

sys.path.append("../../util")
from arg_utils import get_base_parser, update_parser
from math_utils import softmax
from model_utils import check_and_download_file, check_and_download_models

logger = getLogger(__name__)

# ======================
# Parameters
# ======================

REMOTE_PATH = "https://storage.googleapis.com/ailia-models/qwen3_vl/"
IMAGE_PATH = "demo.jpeg"

# Model config constants shared by both 4b and 8b (verified equal via
# transformers.AutoConfig: num_hidden_layers/num_key_value_heads/head_dim/
# vision num_position_embeddings/spatial_merge_size/vocab & token ids all
# match; only the language model hidden_size differs, see MODEL_TYPE below).
NUM_KV_HEADS = 8
HEAD_DIM = 128
NUM_LM_LAYERS = 36
SPATIAL_MERGE_SIZE = 2
NUM_GRID_PER_SIDE = 48  # int(sqrt(num_position_embeddings=2304))
IMAGE_TOKEN_ID = 151655
VIDEO_TOKEN_ID = 151656
EOS_TOKEN_IDS = {151645, 151643}

# Vision preprocessing constants
PATCH_SIZE = 16
TEMPORAL_PATCH_SIZE = 2
IMAGE_FACTOR = PATCH_SIZE * SPATIAL_MERGE_SIZE  # 32
IMAGE_MIN_PIXELS = 65536  # shortest_edge from processor config
IMAGE_MAX_PIXELS = 16777216  # longest_edge from processor config
IMAGE_MEAN = np.array([0.5, 0.5, 0.5], dtype=np.float32)
IMAGE_STD = np.array([0.5, 0.5, 0.5], dtype=np.float32)

INTERMEDIATE = True
COPY_BLOB_DATA = True

# ======================
# Argument Parser Config
# ======================

parser = get_base_parser("Qwen3-VL", IMAGE_PATH, None, large_model=True)
parser.add_argument(
    "-p",
    "--prompt",
    type=str,
    default="Describe this image.",
    help="prompt text",
)
parser.add_argument(
    "--max_new_tokens",
    type=int,
    default=1024,
    help="maximum number of tokens to generate",
)
parser.add_argument(
    "--temperature",
    type=float,
    default=0.7,
    help="sampling temperature",
)
parser.add_argument(
    "--top_k",
    type=int,
    default=20,
    help="top-k sampling",
)
parser.add_argument(
    "--top_p",
    type=float,
    default=0.8,
    help="nucleus sampling threshold",
)
parser.add_argument(
    "--repetition_penalty",
    type=float,
    default=1.0,
    help="repetition penalty",
)
parser.add_argument(
    "--disable_ailia_tokenizer",
    action="store_true",
    help="use transformers tokenizer instead of ailia tokenizer",
)
parser.add_argument(
    "--onnx",
    action="store_true",
    help="use onnxruntime",
)
parser.add_argument(
    "--fp16",
    action="store_true",
    help="use fp16 language model",
)
parser.add_argument(
    "--model_type",
    default="8b",
    choices=["4b", "8b"],
    help="Qwen3-VL model type: 4b or 8b (default: 8b)",
)
args = update_parser(parser)


# fp16 is 8B-only: the 4B fp32 LM (~15GB) already fits in GPU memory, so no
# fp16 variant is exported for it.
if args.fp16 and args.model_type == "4b":
    logger.error("--fp16 is not supported for --model_type 4b")
    sys.exit(1)

if args.model_type == "4b":
    HIDDEN_SIZE = 2560
    WEIGHT_VIS_PATH = "qwen3_vl_4b_instruct_vision_encoder.onnx"
    MODEL_VIS_PATH = "qwen3_vl_4b_instruct_vision_encoder.onnx.prototxt"
    PB_VIS_PATH = None  # 4b vision encoder is under 2GB, so ONNX export keeps
    WEIGHT_EMBED_PATH = "qwen3_vl_4b_instruct_embed_tokens.npy"
    WEIGHT_LM_PATH = "qwen3_vl_4b_instruct_language_model.onnx"
    MODEL_LM_PATH = "qwen3_vl_4b_instruct_language_model.onnx.prototxt"
    PB_LM_PATH = "qwen3_vl_4b_instruct_language_model_weights.pb"
else:
    HIDDEN_SIZE = 4096
    WEIGHT_VIS_PATH = "qwen3_vl_8b_instruct_vision_encoder.onnx"
    MODEL_VIS_PATH = "qwen3_vl_8b_instruct_vision_encoder.onnx.prototxt"
    PB_VIS_PATH = "qwen3_vl_8b_instruct_vision_encoder_weights.pb"
    WEIGHT_EMBED_PATH = "qwen3_vl_8b_instruct_embed_tokens.npy"
    WEIGHT_LM_PATH = "qwen3_vl_8b_instruct_language_model.onnx"
    MODEL_LM_PATH = "qwen3_vl_8b_instruct_language_model.onnx.prototxt"
    PB_LM_PATH = "qwen3_vl_8b_instruct_language_model_weights.pb"
    WEIGHT_LM_FP16_PATH = "qwen3_vl_8b_instruct_language_model_fp16.onnx"
    MODEL_LM_FP16_PATH = "qwen3_vl_8b_instruct_language_model_fp16.onnx.prototxt"
    PB_LM_FP16_PATH = "qwen3_vl_8b_instruct_language_model_weights_fp16.pb"


LM_DTYPE = np.float16 if args.fp16 else np.float32


# ======================
# Preprocessing utilities
# ======================


def load_embed_tokens():
    """Load embed_tokens.weight from the pre-saved .npy file.

    tie_word_embeddings=false, so embed_tokens.weight != lm_head.weight.
    The embed_tokens weights are saved separately during export.
    """
    return np.load(WEIGHT_EMBED_PATH)


# ── image preprocessing ────────────────────────────────────────────────────────


def smart_resize(
    height,
    width,
    factor=IMAGE_FACTOR,
    min_pixels=IMAGE_MIN_PIXELS,
    max_pixels=IMAGE_MAX_PIXELS,
):
    def round_by(n, f):
        return round(n / f) * f

    def ceil_by(n, f):
        return math.ceil(n / f) * f

    def floor_by(n, f):
        return math.floor(n / f) * f

    h = max(factor, round_by(height, factor))
    w = max(factor, round_by(width, factor))
    if h * w > max_pixels:
        beta = math.sqrt((height * width) / max_pixels)
        h = floor_by(height / beta, factor)
        w = floor_by(width / beta, factor)
    elif h * w < min_pixels:
        beta = math.sqrt(min_pixels / (height * width))
        h = ceil_by(height * beta, factor)
        w = ceil_by(width * beta, factor)
    return h, w


def preprocess_image(img):
    """Convert a PIL image to pixel_values [N, 1536] and image_grid_thw [1, 3]."""
    img_np = np.array(img.convert("RGB"))
    h, w = img_np.shape[:2]

    res_h, res_w = smart_resize(h, w)
    img_resized = np.array(
        PILImage.fromarray(img_np).resize((res_w, res_h), PILImage.Resampling.BICUBIC)
    )

    grid_h = res_h // PATCH_SIZE
    grid_w = res_w // PATCH_SIZE
    grid_t = 1

    # Normalize: (x/255 - 0.5) / 0.5
    patches = img_resized.astype(np.float32) / 255.0
    patches = (patches - IMAGE_MEAN) / IMAGE_STD
    patches = patches.transpose(2, 0, 1)  # [3, H, W]

    # Expand to [1, 2, 3, H, W]: batch, temporal (repeat single frame)
    patches = patches[None, None, :, :, :]
    patches = np.concatenate([patches, patches], axis=1)  # temporal_patch_size=2

    # Reshape into merge blocks then flatten
    m = SPATIAL_MERGE_SIZE
    patches = patches.reshape(
        1,
        grid_t,
        TEMPORAL_PATCH_SIZE,
        3,
        grid_h // m,
        m,
        PATCH_SIZE,
        grid_w // m,
        m,
        PATCH_SIZE,
    )
    patches = np.transpose(patches, (0, 1, 4, 7, 5, 8, 3, 2, 6, 9))
    pixel_values = patches.reshape(
        1,
        grid_t * grid_h * grid_w,
        3 * TEMPORAL_PATCH_SIZE * PATCH_SIZE * PATCH_SIZE,
    )[
        0
    ]  # [N, 1536]

    image_grid_thw = np.array([[grid_t, grid_h, grid_w]], dtype=np.int64)
    return pixel_values.astype(np.float32), image_grid_thw


def build_vision_bilinear(grid_thw):
    """Bilinear interpolation indices/weights for the vision encoder position embedding.

    Returns bilinear_indices [4, N] int32 and bilinear_weights [4, N] float32.
    Equivalent to transformers.vision_utils.get_vision_bilinear_indices_and_weights.
    """
    side = NUM_GRID_PER_SIDE
    idx_parts = [[] for _ in range(4)]
    wgt_parts = [[] for _ in range(4)]

    for thw in grid_thw:
        t, h, w = int(thw[0]), int(thw[1]), int(thw[2])

        h_grid = np.linspace(0, side - 1, h, dtype=np.float32)
        w_grid = np.linspace(0, side - 1, w, dtype=np.float32)

        h_fl = h_grid.astype(np.int32)
        w_fl = w_grid.astype(np.int32)
        h_ce = np.minimum(h_fl + 1, side - 1)
        w_ce = np.minimum(w_fl + 1, side - 1)
        h_fr = h_grid - h_fl
        w_fr = w_grid - w_fl

        h_fl_off = h_fl * side
        h_ce_off = h_ce * side

        c00 = h_fl_off[:, None] + w_fl[None, :]  # [h, w]
        c01 = h_fl_off[:, None] + w_ce[None, :]
        c10 = h_ce_off[:, None] + w_fl[None, :]
        c11 = h_ce_off[:, None] + w_ce[None, :]

        w00 = ((1 - h_fr)[:, None] * (1 - w_fr)[None, :]).astype(np.float32)
        w01 = ((1 - h_fr)[:, None] * w_fr[None, :]).astype(np.float32)
        w10 = (h_fr[:, None] * (1 - w_fr)[None, :]).astype(np.float32)
        w11 = (h_fr[:, None] * w_fr[None, :]).astype(np.float32)

        m = SPATIAL_MERGE_SIZE

        def to_merge(arr):
            # Reorder [h, w] → merge block order (h//m, w//m, m, m) → flatten
            return arr.reshape(h // m, m, w // m, m).transpose(0, 2, 1, 3).flatten()

        for i, (c, wg) in enumerate(
            [
                (to_merge(c00), to_merge(w00)),
                (to_merge(c01), to_merge(w01)),
                (to_merge(c10), to_merge(w10)),
                (to_merge(c11), to_merge(w11)),
            ]
        ):
            idx_parts[i].append(np.tile(c, t))
            wgt_parts[i].append(np.tile(wg, t))

    bilinear_indices = np.stack(
        [np.concatenate(idx_parts[i]) for i in range(4)]
    ).astype(np.int32)
    bilinear_weights = np.stack([np.concatenate(wgt_parts[i]) for i in range(4)])
    return bilinear_indices, bilinear_weights


def build_vision_position_ids(grid_thw):
    """(row, col) position IDs for vision rotary embeddings.

    Returns position_ids [N, 2] int64, ordered by merge blocks.
    Equivalent to transformers.vision_utils.get_vision_position_ids.
    """
    pos_list = []
    m = SPATIAL_MERGE_SIZE
    for thw in grid_thw:
        t, h, w = int(thw[0]), int(thw[1]), int(thw[2])

        # hpos: [h, w] — value = row index
        hpos = np.arange(h, dtype=np.int64)[:, None] * np.ones(w, dtype=np.int64)
        hpos = hpos.reshape(h // m, m, w // m, m).transpose(0, 2, 1, 3).flatten()

        # wpos: [h, w] — value = col index
        wpos = np.ones(h, dtype=np.int64)[:, None] * np.arange(w, dtype=np.int64)
        wpos = wpos.reshape(h // m, m, w // m, m).transpose(0, 2, 1, 3).flatten()

        frame = np.stack([hpos, wpos], axis=-1)  # [h*w, 2]
        pos_list.append(np.tile(frame, (t, 1)))  # [t*h*w, 2]

    return np.concatenate(pos_list, axis=0).astype(np.int64)


# ── text preprocessing ─────────────────────────────────────────────────────────


def apply_chat_template(messages):
    buf = StringIO()

    for message in messages:
        buf.write(f'<|im_start|>{message["role"]}\n')
        if isinstance(message["content"], str):
            buf.write(message["content"])
            buf.write("<|im_end|>\n")
        else:
            for part in message["content"]:
                if part.get("type") == "image" or "image" in part:
                    buf.write("<|vision_start|><|image_pad|><|vision_end|>")
                elif part.get("type") == "video" or "video" in part:
                    buf.write("<|vision_start|><|video_pad|><|vision_end|>")
                elif "text" in part:
                    buf.write(part["text"])
            buf.write("<|im_end|>\n")

    buf.write("<|im_start|>assistant\n")
    return buf.getvalue()


def expand_image_tokens(text, image_grid_thw):
    """Replace each <|image_pad|> placeholder with M copies (M = merged token count)."""
    merge_length = SPATIAL_MERGE_SIZE**2
    for thw in image_grid_thw:
        num_tokens = int(thw[0]) * int(thw[1]) * int(thw[2]) // merge_length
        text = text.replace(
            "<|image_pad|>",
            "<|placeholder|>" * num_tokens,
            1,
        )
    return text.replace("<|placeholder|>", "<|image_pad|>")


# ── MRoPE position IDs ─────────────────────────────────────────────────────────


def get_rope_index(input_ids, mm_token_type_ids, image_grid_thw, attention_mask):
    """Compute T/H/W MRoPE position IDs for the language model.

    Returns
    -------
    thw_pos     : np.ndarray [3, batch, seq]  T, H, W coordinates
    rope_deltas : np.ndarray [batch, 1]       position offset for decode step
    """
    batch_size, seq_len = input_ids.shape
    position_ids = np.zeros((3, batch_size, seq_len), dtype=np.int64)
    rope_deltas = []

    grid_iter = iter(image_grid_thw) if image_grid_thw is not None else iter([])

    for batch_idx in range(batch_size):
        token_types = mm_token_type_ids[batch_idx]
        mask = (
            attention_mask[batch_idx].astype(bool)
            if attention_mask is not None
            else np.ones(seq_len, dtype=bool)
        )
        token_types_masked = token_types[mask]

        # Group consecutive tokens by modality (0=text, 1=image, 2=video)
        groups = []
        if len(token_types_masked) > 0:
            cur, start = int(token_types_masked[0]), 0
            for i in range(1, len(token_types_masked)):
                v = int(token_types_masked[i])
                if v != cur:
                    groups.append((cur, start, i))
                    cur, start = v, i
            groups.append((cur, start, len(token_types_masked)))

        current_pos = 0
        pos_parts = []

        for mod_type, gstart, gend in groups:
            if mod_type == 0:  # text
                n = gend - gstart
                pos = np.arange(n, dtype=np.int64) + current_pos
                pos_parts.append(np.tile(pos[None, :], (3, 1)))
                current_pos += n
            else:  # image=1 or video=2
                thw = next(grid_iter)
                lg_t = int(thw[0])
                lg_h = int(thw[1]) // SPATIAL_MERGE_SIZE
                lg_w = int(thw[2]) // SPATIAL_MERGE_SIZE

                pt = np.arange(lg_t, dtype=np.int64)
                ph = np.arange(lg_h, dtype=np.int64) + current_pos
                pw = np.arange(lg_w, dtype=np.int64) + current_pos

                # Expand to [lg_t * lg_h * lg_w] each — matches PyTorch repeat patterns
                pw = np.tile(pw, lg_h * lg_t)
                ph = np.tile(np.repeat(ph, lg_w), lg_t)
                pt = np.repeat(pt, lg_h * lg_w) + current_pos

                pos_parts.append(np.stack([pt, ph, pw], axis=0))
                current_pos += max(lg_h, lg_w)

        if pos_parts:
            llm_positions = np.concatenate(pos_parts, axis=1)
        else:
            llm_positions = np.zeros((3, int(mask.sum())), dtype=np.int64)

        position_ids[:, batch_idx, mask] = llm_positions
        rope_deltas.append(int(llm_positions.max()) + 1 - int(mask.sum()))

    return position_ids, np.array(rope_deltas, dtype=np.int64)[:, None]


# ======================
# Main functions
# ======================


def preprocess(image_path, prompt, tokenizer):
    img = PILImage.open(image_path).convert("RGB")
    pixel_values, image_grid_thw = preprocess_image(img)
    bilinear_indices, bilinear_weights = build_vision_bilinear(image_grid_thw)
    position_ids_vis = build_vision_position_ids(image_grid_thw)

    text = apply_chat_template(
        [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": img},
                    {"type": "text", "text": prompt},
                ],
            }
        ]
    )
    text = expand_image_tokens(text, image_grid_thw)

    token_ids = tokenizer.encode(text)
    input_ids = np.array([token_ids], dtype=np.int64)  # [1, seq]
    attention_mask = np.ones((1, len(token_ids)), dtype=np.int64)  # [1, seq]

    # mm_token_type_ids: 1 at image token positions, 0 elsewhere
    mm_token_type_ids = (input_ids == IMAGE_TOKEN_ID).astype(np.int64)

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "mm_token_type_ids": mm_token_type_ids,
        "pixel_values": pixel_values,
        "image_grid_thw": image_grid_thw,
        "bilinear_indices": bilinear_indices,
        "bilinear_weights": bilinear_weights,
        "position_ids_vis": position_ids_vis,
        "tokenizer": tokenizer,
    }


def run_vision_encoder(models, data, use_onnx=False):
    vis_net = models["vision_encoder"]
    feeds = {
        "pixel_values": data["pixel_values"],
        "bilinear_indices": data["bilinear_indices"],
        "bilinear_weights": data["bilinear_weights"],
        "position_ids": data["position_ids_vis"],
    }
    if use_onnx:
        out = vis_net.run(
            ["image_embeds", "deepstack_0", "deepstack_1", "deepstack_2"],
            feeds,
        )
    else:
        out = vis_net.predict(list(feeds.values()))
    return out[0], out[1], out[2], out[3]


def build_lm_inputs(data, image_embeds, embed_tokens):
    """Construct inputs_embeds [1,seq,4096], position_ids [4,1,seq], visual_pos_masks [1,seq]."""
    input_ids = data["input_ids"]
    mm_token_type_ids = data["mm_token_type_ids"]
    image_grid_thw = data["image_grid_thw"]
    attention_mask = data["attention_mask"]

    token_embeds = embed_tokens[input_ids[0]].astype(LM_DTYPE, copy=True)  # [seq, 4096]
    image_positions = np.where(input_ids[0] == IMAGE_TOKEN_ID)[0]
    token_embeds[image_positions] = image_embeds.astype(LM_DTYPE, copy=False)
    inputs_embeds = token_embeds[None, :, :]  # [1, seq, 4096]

    visual_pos_masks = mm_token_type_ids == 1  # [1, seq] bool

    thw_pos, rope_deltas = get_rope_index(
        input_ids, mm_token_type_ids, image_grid_thw, attention_mask
    )
    seq_len = input_ids.shape[1]
    text_pos = np.arange(seq_len, dtype=np.int64).reshape(1, 1, seq_len)
    position_ids = np.concatenate([text_pos, thw_pos], axis=0)  # [4, 1, seq]

    return inputs_embeds, position_ids, visual_pos_masks, rope_deltas


def lm_step(models, feeds, first_run=True, use_onnx=False):
    lm_net = models["language_model"]
    out_names = (
        ["logits"]
        + [f"present.{i}.key" for i in range(NUM_LM_LAYERS)]
        + [f"present.{i}.value" for i in range(NUM_LM_LAYERS)]
    )
    if use_onnx:
        out = lm_net.run(out_names, feeds)
        return out[0], out[1 : 1 + NUM_LM_LAYERS], out[1 + NUM_LM_LAYERS :]

    if first_run or not COPY_BLOB_DATA:
        out = lm_net.predict(list(feeds.values()))
        return out[0], out[1 : 1 + NUM_LM_LAYERS], out[1 + NUM_LM_LAYERS :]

    # Decode step with copy_blob_data: avoids re-uploading large KV caches
    key_shapes = [
        lm_net.get_blob_shape(lm_net.find_blob_index_by_name(f"present.{i}.key"))
        for i in range(NUM_LM_LAYERS)
    ]
    value_shapes = [
        lm_net.get_blob_shape(lm_net.find_blob_index_by_name(f"present.{i}.value"))
        for i in range(NUM_LM_LAYERS)
    ]

    for name in [
        "inputs_embeds",
        "position_ids",
        "visual_pos_masks",
        "deepstack_0",
        "deepstack_1",
        "deepstack_2",
    ]:
        lm_net.set_input_blob_data(feeds[name], lm_net.find_blob_index_by_name(name))

    for i in range(NUM_LM_LAYERS):
        lm_net.set_input_blob_shape(
            key_shapes[i], lm_net.find_blob_index_by_name(f"past_key_values.{i}.key")
        )
        lm_net.set_input_blob_shape(
            value_shapes[i],
            lm_net.find_blob_index_by_name(f"past_key_values.{i}.value"),
        )
        lm_net.copy_blob_data(f"past_key_values.{i}.key", f"present.{i}.key")
        lm_net.copy_blob_data(f"past_key_values.{i}.value", f"present.{i}.value")

    lm_net.update()
    logits = lm_net.get_blob_data(lm_net.find_blob_index_by_name("logits"))
    return logits, None, None


def logits_processor(
    generated_ids, scores, temperature, top_k, top_p, repetition_penalty
):
    # RepetitionPenaltyLogitsProcessor
    if repetition_penalty != 1.0 and generated_ids:
        all_ids = np.array(generated_ids, dtype=np.int32).reshape(1, -1)
        score = np.take_along_axis(scores, all_ids, axis=1)
        score = np.where(
            score < 0, score * repetition_penalty, score / repetition_penalty
        )
        scores = scores.copy()
        np.put_along_axis(scores, all_ids, score, axis=1)

    # TemperatureLogitsWarper
    scores = scores / temperature

    # TopKLogitsWarper
    k = min(top_k, scores.shape[-1])
    kth = np.partition(scores, -k, axis=-1)[..., -k : -k + 1]
    scores = np.where(scores < kth, -np.inf, scores)

    # TopPLogitsWarper
    if top_p < 1.0:
        sorted_idx = np.argsort(scores, axis=-1)
        sorted_scores = np.take_along_axis(scores, sorted_idx, axis=-1)
        cumprobs = softmax(sorted_scores, axis=-1).cumsum(axis=-1)
        to_remove = cumprobs <= (1.0 - top_p)
        to_remove[:, -1] = False
        sorted_scores = sorted_scores.copy()
        sorted_scores[to_remove] = -np.inf
        scores = np.empty_like(scores)
        np.put_along_axis(scores, sorted_idx, sorted_scores, axis=-1)

    return scores


def generate(models, data, embed_tokens):
    use_onnx = args.onnx
    max_new_tokens = args.max_new_tokens
    benchmark = args.benchmark
    temperature = args.temperature
    top_k = args.top_k
    top_p = args.top_p
    repetition_penalty = args.repetition_penalty
    tokenizer = data["tokenizer"]

    logger.info("Running vision encoder...")
    if benchmark:
        t0 = int(round(time.time() * 1000))

    image_embeds, ds0, ds1, ds2 = run_vision_encoder(models, data, use_onnx)

    if benchmark:
        logger.info(f"\tvision encoder: {int(round(time.time() * 1000)) - t0} ms")

    inputs_embeds, position_ids, visual_pos_masks, rope_deltas = build_lm_inputs(
        data,
        image_embeds.astype(LM_DTYPE, copy=False),
        embed_tokens,
    )
    rope_delta = int(rope_deltas[0, 0])
    EMPTY_DS = np.zeros((0, HIDDEN_SIZE), dtype=LM_DTYPE)

    # --- Prefill ---
    logger.info("Prefill...")
    if INTERMEDIATE:
        print("Generating..." + "\n\x1b[2A")
        before_text = ""

    if benchmark:
        t0 = int(round(time.time() * 1000))

    feeds = {
        "inputs_embeds": inputs_embeds,
        "position_ids": position_ids.astype(np.int64),
        "visual_pos_masks": visual_pos_masks,
        "deepstack_0": ds0.astype(LM_DTYPE, copy=False),
        "deepstack_1": ds1.astype(LM_DTYPE, copy=False),
        "deepstack_2": ds2.astype(LM_DTYPE, copy=False),
    }
    for i in range(NUM_LM_LAYERS):
        feeds[f"past_key_values.{i}.key"] = np.zeros(
            (1, NUM_KV_HEADS, 0, HEAD_DIM), dtype=LM_DTYPE
        )
        feeds[f"past_key_values.{i}.value"] = np.zeros(
            (1, NUM_KV_HEADS, 0, HEAD_DIM), dtype=LM_DTYPE
        )

    logits, past_keys, past_values = lm_step(
        models, feeds, first_run=True, use_onnx=use_onnx
    )

    if benchmark:
        logger.info(f"\tprefill: {int(round(time.time() * 1000)) - t0} ms")

    scores = logits_processor(
        [], logits[0, 0:1, :], temperature, top_k, top_p, repetition_penalty
    )
    probs = softmax(scores, axis=-1)
    next_token = int(np.random.choice(probs.shape[-1], p=probs[0]))
    generated = [next_token]
    past_len = inputs_embeds.shape[1]

    # --- Decode loop ---
    for step in range(max_new_tokens - 1):
        if next_token in EOS_TOKEN_IDS:
            break

        if INTERMEDIATE:
            try:
                text = tokenizer.decode(generated, skip_special_tokens=True)
            except UnicodeDecodeError:
                text = before_text
            delta = text[len(before_text) :] if text.startswith(before_text) else text
            print(delta, end="", flush=True)
            before_text = text

        if benchmark and step == 0:
            t0 = int(round(time.time() * 1000))

        next_embed = embed_tokens[[next_token]][None, :, :].astype(LM_DTYPE)

        pos_dec = np.full((4, 1, 1), past_len + rope_delta, dtype=np.int64)
        pos_dec[0, 0, 0] = past_len

        feeds_dec = {
            "inputs_embeds": next_embed,
            "position_ids": pos_dec,
            "visual_pos_masks": np.zeros((1, 1), dtype=bool),
            "deepstack_0": EMPTY_DS,
            "deepstack_1": EMPTY_DS,
            "deepstack_2": EMPTY_DS,
        }
        if past_keys is not None:
            for i in range(NUM_LM_LAYERS):
                feeds_dec[f"past_key_values.{i}.key"] = past_keys[i]
                feeds_dec[f"past_key_values.{i}.value"] = past_values[i]

        logits, past_keys, past_values = lm_step(
            models, feeds_dec, first_run=False, use_onnx=use_onnx
        )

        if benchmark and step == 0:
            logger.info(f"\t1st decode: {int(round(time.time() * 1000)) - t0} ms")

        scores = logits_processor(
            generated, logits[0, 0:1, :], temperature, top_k, top_p, repetition_penalty
        )
        probs = softmax(scores, axis=-1)
        next_token = int(np.random.choice(probs.shape[-1], p=probs[0]))
        generated.append(next_token)
        past_len += 1

    return tokenizer.decode(generated, skip_special_tokens=True)


def predict(models, embed_tokens, image_path, prompt):
    logger.info(f"Image: {image_path}")
    logger.info(f"Prompt: {prompt}")

    data = preprocess(image_path, prompt, models["tokenizer"])
    output_text = generate(models, data, embed_tokens)

    if INTERMEDIATE:
        print("")
    else:
        print(output_text)

    return output_text


def recognize(models, embed_tokens):
    logger.info("Start inference...")
    if args.benchmark:
        logger.info("BENCHMARK mode")
        total = 0
        for i in range(args.benchmark_count):
            t0 = int(round(time.time() * 1000))
            predict(models, embed_tokens, args.input[0], args.prompt)
            elapsed = int(round(time.time() * 1000)) - t0
            logger.info(f"\tailia processing time {elapsed} ms")
            if i != 0:
                total += elapsed
        logger.info(f"\taverage time {total / (args.benchmark_count - 1)} ms")
    else:
        predict(models, embed_tokens, args.input[0], args.prompt)

    logger.info("Script finished successfully.")


def main():
    if args.fp16:
        weight_lm_path = WEIGHT_LM_FP16_PATH
        model_lm_path = MODEL_LM_FP16_PATH
        pb_lm_path = PB_LM_FP16_PATH
    else:
        weight_lm_path = WEIGHT_LM_PATH
        model_lm_path = MODEL_LM_PATH
        pb_lm_path = PB_LM_PATH

    check_and_download_models(WEIGHT_VIS_PATH, MODEL_VIS_PATH, REMOTE_PATH)
    check_and_download_models(weight_lm_path, model_lm_path, REMOTE_PATH)
    if PB_VIS_PATH is not None:
        check_and_download_file(PB_VIS_PATH, REMOTE_PATH)
    check_and_download_file(pb_lm_path, REMOTE_PATH)
    check_and_download_file(WEIGHT_EMBED_PATH, REMOTE_PATH)

    logger.info("Loading embed_tokens...")
    embed_tokens = load_embed_tokens()
    logger.info(f"embed_tokens shape: {embed_tokens.shape}")

    env_id = args.env_id

    if args.disable_ailia_tokenizer:
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained("./tokenizer")
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

    if not args.onnx:
        import ailia

        memory_mode = ailia.get_memory_mode(
            reduce_constant=True,
            ignore_input_with_initializer=True,
            reduce_interstage=False,
            reuse_interstage=True,
        )
        vision_encoder = ailia.Net(
            MODEL_VIS_PATH, WEIGHT_VIS_PATH, env_id=env_id, memory_mode=memory_mode
        )
        language_model = ailia.Net(
            model_lm_path, weight_lm_path, env_id=env_id, memory_mode=memory_mode
        )
    else:
        import onnxruntime

        providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
        vision_encoder = onnxruntime.InferenceSession(
            WEIGHT_VIS_PATH, providers=providers
        )
        language_model = onnxruntime.InferenceSession(
            weight_lm_path, providers=providers
        )

    models = {
        "vision_encoder": vision_encoder,
        "language_model": language_model,
        "tokenizer": tokenizer,
    }

    recognize(models, embed_tokens)


if __name__ == "__main__":
    main()
