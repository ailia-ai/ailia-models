import gc
import sys
import time
from logging import getLogger

import ailia
import cv2
import numpy as np
from ailia_tokenizer import GPT2Tokenizer

# import original modules
sys.path.append("../../util")
from arg_utils import get_base_parser, update_parser  # noqa
from detector_utils import load_image  # noqa
from model_utils import check_and_download_file, check_and_download_models  # noqa
from resize_utils import tv_resize  # noqa

logger = getLogger(__name__)

# ======================
# Model selection
# ======================

WEIGHT_PATH = "colqwen2-v0.1.onnx"
MODEL_PATH = "colqwen2-v0.1.onnx.prototxt"
PB_PATH = "colqwen2-v0.1_weights.pb"

# The token embedding and the vision encoder have the same weights as Qwen2-VL-2B,
# so its vision encoder model is used as is.
WEIGHT_VIS_PATH = "Qwen2-VL-2B_vis.onnx"
MODEL_VIS_PATH = "Qwen2-VL-2B_vis.onnx.prototxt"
WEIGHT_VIS_OPT_PATH = "Qwen2-VL-2B_vis.opt.onnx"
MODEL_VIS_OPT_PATH = "Qwen2-VL-2B_vis.opt.onnx.prototxt"
PB_VIS_PATH = "Qwen2-VL-2B_vis_weights.pb"

REMOTE_PATH = "https://storage.googleapis.com/ailia-models/colqwen2/"
REMOTE_PATH_VIS = "https://storage.googleapis.com/ailia-models/qwen2_vl/"


# ======================
# Parameters
# ======================

IMAGE_PATHS = ["doc1.jpg", "doc2.jpg", "doc3.jpg", "doc4.jpg"]
QUERIES = [
    "What is the variable represented on the y-axis of the graph?",
    "Total outlay is maximum in which year?",
]


# ======================
# Argument Parser Config
# ======================

parser = get_base_parser("ColQwen2", IMAGE_PATHS, None)
parser.add_argument(
    "-q",
    "--query",
    nargs="+",
    default=QUERIES,
    help="Text queries to search the document images with.",
)
parser.add_argument(
    "--disable_ailia_tokenizer", action="store_true", help="disable ailia tokenizer."
)
parser.add_argument(
    "--normal",
    action="store_true",
    help="use normal vision encoder model (default : opt model).",
)
parser.add_argument("--onnx", action="store_true", help="execute onnxruntime version.")
args = update_parser(parser)

# ======================
# Secondary Functions
# ======================

IMAGE_TOKEN_ID = 151655  # <|image_pad|>

PATCH_SIZE = 14
MERGE_SIZE = 2
TEMPORAL_PATCH_SIZE = 2
MIN_PIXELS = 3136
MAX_PIXELS = 602112  # 768 visual tokens at most
IMAGE_MEAN = np.array([0.48145466, 0.4578275, 0.40821073], dtype=np.float32)
IMAGE_STD = np.array([0.26862954, 0.26130258, 0.27577711], dtype=np.float32)

# Prompt formats the model was trained with
DOCUMENT_PROMPT = (
    "<|im_start|>user\n<|vision_start|>{image_pad}<|vision_end|>"
    "Describe the image.<|im_end|><|endoftext|>"
)
QUERY_PREFIX = "Query: "
QUERY_AUGMENTATION = "<|endoftext|>" * 10


class LazyModel:
    """Defers model loading until the first predict/run call."""

    def __init__(self, loader_fn, name=""):
        self._loader_fn = loader_fn
        self._name = name
        self._net = None

    def load(self):
        if self._net is None:
            logger.info(f"Loading model: {self._name}")
            self._net = self._loader_fn()
        return self._net

    def unload(self):
        if self._net is not None:
            self._net = None
            gc.collect()

    def predict(self, *args, **kwargs):
        return self.load().predict(*args, **kwargs)

    def run(self, *args, **kwargs):
        return self.load().run(*args, **kwargs)


def smart_resize(height, width, factor, min_pixels, max_pixels):
    h_bar = round(height / factor) * factor
    w_bar = round(width / factor) * factor
    if h_bar * w_bar > max_pixels:
        beta = np.sqrt((height * width) / max_pixels)
        h_bar = max(factor, int(np.floor(height / beta / factor)) * factor)
        w_bar = max(factor, int(np.floor(width / beta / factor)) * factor)
    elif h_bar * w_bar < min_pixels:
        beta = np.sqrt(min_pixels / (height * width))
        h_bar = int(np.ceil(height * beta / factor)) * factor
        w_bar = int(np.ceil(width * beta / factor)) * factor
    return h_bar, w_bar


def preprocess(img):
    """resize (uint8, bicubic) -> rescale/normalize -> flatten to patches"""
    height, width, _ = img.shape
    resized_height, resized_width = smart_resize(
        height,
        width,
        factor=PATCH_SIZE * MERGE_SIZE,
        min_pixels=MIN_PIXELS,
        max_pixels=MAX_PIXELS,
    )
    img = tv_resize(
        img, (resized_height, resized_width), interpolation="bicubic", antialias=True
    )

    img = (img / 255 - IMAGE_MEAN) / IMAGE_STD
    img = img.transpose(2, 0, 1).astype(np.float32)  # HWC -> CHW

    # a single image is repeated along the temporal axis
    channel = img.shape[0]
    grid_h, grid_w = resized_height // PATCH_SIZE, resized_width // PATCH_SIZE
    patches = np.tile(img[None], (TEMPORAL_PATCH_SIZE, 1, 1, 1))
    patches = patches.reshape(
        TEMPORAL_PATCH_SIZE,
        channel,
        grid_h // MERGE_SIZE,
        MERGE_SIZE,
        PATCH_SIZE,
        grid_w // MERGE_SIZE,
        MERGE_SIZE,
        PATCH_SIZE,
    )
    patches = patches.transpose(2, 5, 3, 6, 1, 0, 4, 7)
    pixel_values = patches.reshape(
        grid_h * grid_w,
        channel * TEMPORAL_PATCH_SIZE * PATCH_SIZE * PATCH_SIZE,
    )
    image_grid_thw = np.array([[1, grid_h, grid_w]], dtype=np.int64)

    return pixel_values, image_grid_thw


def tokenize(tokenizer, text):
    if args.disable_ailia_tokenizer:
        input_ids = tokenizer(text, return_tensors="np")["input_ids"]
    else:
        input_ids = np.array([tokenizer(text)["input_ids"]])
    return input_ids.astype(np.int64)


def get_rope_index(input_ids, image_grid_thw):
    """3D (temporal, height, width) position ids of the M-RoPE for a single image prompt.

    Text tokens use the same position on the 3 axes. The image tokens use the
    position in the grid of merged patches, offset by the preceding text length.
    """
    tokens = input_ids[0]
    image_start = int(np.argmax(tokens == IMAGE_TOKEN_ID))
    _, h, w = image_grid_thw[0]
    llm_grid_h, llm_grid_w = h // MERGE_SIZE, w // MERGE_SIZE
    num_image_tokens = llm_grid_h * llm_grid_w

    text_before = np.tile(np.arange(image_start), (3, 1))
    t_index = np.zeros(num_image_tokens, dtype=np.int64)
    h_index = np.repeat(np.arange(llm_grid_h), llm_grid_w)
    w_index = np.tile(np.arange(llm_grid_w), llm_grid_h)
    image = np.stack([t_index, h_index, w_index]) + image_start
    text_len = len(tokens) - image_start - num_image_tokens
    st_idx = image.max() + 1
    text_after = np.tile(np.arange(text_len), (3, 1)) + st_idx

    position_ids = np.concatenate([text_before, image, text_after], axis=1)
    return position_ids[:, None, :].astype(np.int64)  # (3, 1, seq)


def maxsim(query_embeddings, document_embeddings):
    """Late interaction (ColBERT) score: sum over query tokens of the max similarity"""
    return (
        np.einsum("qd,sd->qs", query_embeddings, document_embeddings).max(axis=1).sum()
    )


# ======================
# Main functions
# ======================


def embed(
    models, input_ids, pixel_values, image_grid_thw, image_token_id, position_ids
):
    visual = models["visual"]
    net = models["net"]

    # token embedding + vision encoder (Qwen2-VL-2B)
    if not args.onnx:
        output = visual.predict(
            [input_ids, pixel_values, image_grid_thw, image_token_id]
        )
    else:
        output = visual.run(
            None,
            {
                "input_ids": input_ids,
                "pixel_values": pixel_values,
                "image_grid_thw": image_grid_thw,
                "image_token_id": image_token_id,
            },
        )
    (inputs_embeds,) = output

    # language model + projection + L2 normalization (ColQwen2)
    attention_mask = np.ones(input_ids.shape, dtype=np.int64)
    if not args.onnx:
        output = net.predict([inputs_embeds, position_ids, attention_mask])
    else:
        output = net.run(
            None,
            {
                "inputs_embeds": inputs_embeds,
                "position_ids": position_ids,
                "attention_mask": attention_mask,
            },
        )
    (embeddings,) = output

    return embeddings[0]  # (seq, 128)


def encode_document(models, image_path):
    img = load_image(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGRA2RGB)
    pixel_values, image_grid_thw = preprocess(img)

    num_image_tokens = int(image_grid_thw[0].prod()) // (MERGE_SIZE**2)
    text = DOCUMENT_PROMPT.format(image_pad="<|image_pad|>" * num_image_tokens)
    input_ids = tokenize(models["tokenizer"], text)
    position_ids = get_rope_index(input_ids, image_grid_thw)
    image_token_id = np.array([IMAGE_TOKEN_ID], dtype=np.int64)

    embeddings = embed(
        models, input_ids, pixel_values, image_grid_thw, image_token_id, position_ids
    )

    # reload the vision encoder for each image to avoid an ailia issue
    if not args.onnx:
        models["visual"].unload()

    return embeddings


def encode_query(models, query):
    text = QUERY_PREFIX + query + QUERY_AUGMENTATION
    input_ids = tokenize(models["tokenizer"], text)
    position_ids = np.tile(np.arange(input_ids.shape[1]), (3, 1, 1)).astype(np.int64)

    # the vision encoder model needs a dummy image when there is no image
    pixel_values = np.zeros((16, 1176), dtype=np.float32)
    image_grid_thw = np.array([[1, 4, 4]], dtype=np.int64)
    image_token_id = np.array([-1], dtype=np.int64)

    return embed(
        models, input_ids, pixel_values, image_grid_thw, image_token_id, position_ids
    )


def predict(models, image_paths, queries):
    document_embeddings = []
    for image_path in image_paths:
        document_embeddings.append(encode_document(models, image_path))
        logger.info(f"document: {image_path} {document_embeddings[-1].shape}")
    query_embeddings = []
    for query in queries:
        query_embeddings.append(encode_query(models, query))
        logger.info(f"query: {query} {query_embeddings[-1].shape}")

    scores = np.array(
        [[maxsim(q, d) for d in document_embeddings] for q in query_embeddings]
    )
    return scores


def recognize(models):
    image_paths = args.input if isinstance(args.input, list) else [args.input]
    queries = args.query

    # inference
    logger.info("Start inference...")
    if args.benchmark:
        logger.info("BENCHMARK mode")
        for i in range(args.benchmark_count):
            start = int(round(time.time() * 1000))
            scores = predict(models, image_paths, queries)
            end = int(round(time.time() * 1000))
            logger.info(f"\tailia processing time {end - start} ms")
    else:
        scores = predict(models, image_paths, queries)

    logger.info("Scores (rows: queries, columns: documents)")
    logger.info("\n" + np.array2string(scores, precision=4))
    for query, row in zip(queries, scores):
        logger.info(f"Query: {query}")
        for rank, i in enumerate(np.argsort(-row)):
            logger.info(f"  [{rank + 1}] {image_paths[i]} ({row[i]:.4f})")

    logger.info("Script finished successfully.")


def main():
    if args.normal:
        weight_vis_path, model_vis_path = WEIGHT_VIS_PATH, MODEL_VIS_PATH
    else:
        weight_vis_path, model_vis_path = WEIGHT_VIS_OPT_PATH, MODEL_VIS_OPT_PATH

    # model files check and download
    check_and_download_models(WEIGHT_PATH, MODEL_PATH, REMOTE_PATH)
    check_and_download_file(PB_PATH, REMOTE_PATH)
    check_and_download_models(weight_vis_path, model_vis_path, REMOTE_PATH_VIS)
    check_and_download_file(PB_VIS_PATH, REMOTE_PATH_VIS)

    env_id = args.env_id

    memory_mode = ailia.get_memory_mode(
        reduce_constant=True,
        ignore_input_with_initializer=True,
        reduce_interstage=False,
        reuse_interstage=True,
    )
    cuda = 0 < ailia.get_gpu_environment_id()
    providers = (
        ["CUDAExecutionProvider", "CPUExecutionProvider"]
        if cuda
        else ["CPUExecutionProvider"]
    )

    def load_net(model_path, weight_path):
        if not args.onnx:
            return ailia.Net(
                model_path, weight_path, env_id=env_id, memory_mode=memory_mode
            )
        else:
            import onnxruntime

            return onnxruntime.InferenceSession(weight_path, providers=providers)

    if args.disable_ailia_tokenizer:
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained("./tokenizer")
    else:
        tokenizer = GPT2Tokenizer.from_pretrained("./tokenizer")
        # Pass added_tokens_decoder of tokenizer_config.json sorted by id.
        # Omitting a token shifts all the ids that follow it.
        tokenizer.add_special_tokens(
            {
                "additional_special_tokens": [
                    "<|endoftext|>",
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
        "visual": LazyModel(
            lambda: load_net(model_vis_path, weight_vis_path), "vision encoder"
        ),
        "net": load_net(MODEL_PATH, WEIGHT_PATH),
    }

    recognize(models)


if __name__ == "__main__":
    main()
