import math
import os
import sys
import time
from logging import getLogger

import ailia
import cv2
import librosa
import numpy as np
from PIL import Image

# import original modules
sys.path.append("../../util")
from arg_utils import get_base_parser, update_parser  # noqa
from model_utils import check_and_download_models  # noqa
from resize_utils import tv_resize  # noqa

logger = getLogger(__name__)

# ======================
# Parameters
# ======================

WEIGHT_PATH = "embeddinggemma-2.onnx"
MODEL_PATH = "embeddinggemma-2.onnx.prototxt"
WEIGHT_VISION_PATH = "embeddinggemma-2_vision.onnx"
MODEL_VISION_PATH = "embeddinggemma-2_vision.onnx.prototxt"
WEIGHT_AUDIO_PATH = "embeddinggemma-2_audio.onnx"
MODEL_AUDIO_PATH = "embeddinggemma-2_audio.onnx.prototxt"
REMOTE_PATH = "https://storage.googleapis.com/ailia-models/embeddinggemma-2/"

QUERIES = [
    "cats sleeping on a couch",
    "a president's speech about serving your country",
    "a turtle swimming in the ocean",
]
IMAGE_PATH = "cats.jpg"
AUDIO_PATH = "jfk.wav"
VIDEO_PATH = "sea-turtle.mp4"
VIDEO_EXTENSIONS = (".mp4", ".avi", ".mov", ".mkv", ".webm")
AUDIO_EXTENSIONS = (".wav", ".mp3", ".flac", ".ogg", ".m4a", ".aac")

# ======================
# Argument Parser Config
# ======================

parser = get_base_parser("EmbeddingGemma 2", None, None, fp16_support=False)
parser.add_argument(
    "-i",
    "--input",
    nargs="+",
    action="extend",
    default=None,
    help=(
        "Documents to search given as image, video and audio files. "
        "The type is determined by the extension. "
        "The sample image, audio and video are used when neither --input nor --document is given."
    ),
)
parser.add_argument(
    "-q",
    "--query",
    nargs="+",
    action="extend",
    default=None,
    help=(
        "Text queries to search the documents with. "
        "The sample queries are used when not given."
    ),
)
parser.add_argument(
    "-d",
    "--document",
    nargs="+",
    action="extend",
    default=None,
    help="Documents to search given as texts.",
)
parser.add_argument(
    "--truncate_dim",
    type=int,
    default=None,
    choices=[128, 256, 512, 768],
    help="Truncate the embeddings to this dimension (Matryoshka).",
)
parser.add_argument(
    "--ffmpeg",
    action="store_true",
    help="decode and resample the audio files with ffmpeg instead of librosa.",
)
parser.add_argument(
    "--disable_ailia_tokenizer", action="store_true", help="disable ailia tokenizer."
)
parser.add_argument("--onnx", action="store_true", help="execute onnxruntime version.")
args = update_parser(parser, check_input_type=False)


# ======================
# Secondary Functions
# ======================

QUERY_PROMPT = "task: search result | query: "
DOCUMENT_PROMPT = "title: none | text: "

TEXT_HIDDEN_SIZE = 512

BOI_TOKEN = "<|image>"
EOI_TOKEN = "<image|>"
IMAGE_TOKEN = "<|image|>"
VIDEO_TOKEN = "<|video|>"
BOA_TOKEN = "<|audio>"
EOA_TOKEN = "<audio|>"
AUDIO_TOKEN = "<|audio|>"

PATCH_SIZE = 16
POOLING_KERNEL_SIZE = 3
IMAGE_MAX_SOFT_TOKENS = 280
VIDEO_MAX_SOFT_TOKENS = 140
VIDEO_FPS = 1
VIDEO_MAX_FRAMES = 32

SAMPLING_RATE = 16000
FRAME_LENGTH = 320
HOP_LENGTH = 160
FFT_LENGTH = 512
NUM_MEL_BINS = 128
MEL_FLOOR = 1e-3
AUDIO_MAX_LENGTH = 480000
AUDIO_PAD_TO_MULTIPLE_OF = 128


def get_aspect_ratio_preserving_size(height, width, max_patches):
    """Largest size within max_patches whose sides are multiples of 48 (patch x pooling kernel)."""
    total_px = height * width
    target_px = max_patches * (PATCH_SIZE**2)
    factor = math.sqrt(target_px / total_px)
    ideal_height = factor * height
    ideal_width = factor * width
    side_mult = POOLING_KERNEL_SIZE * PATCH_SIZE

    target_height = int(math.floor(ideal_height / side_mult)) * side_mult
    target_width = int(math.floor(ideal_width / side_mult)) * side_mult

    max_side_length = (max_patches // POOLING_KERNEL_SIZE**2) * side_mult
    if target_height == 0:
        target_height = side_mult
        target_width = min(int(math.floor(width / height)) * side_mult, max_side_length)
    elif target_width == 0:
        target_width = side_mult
        target_height = min(
            int(math.floor(height / width)) * side_mult, max_side_length
        )

    return target_height, target_width


def preprocess_image(img, max_soft_tokens):
    """resize (uint8, bicubic) -> rescale -> flatten to patches padded to max patches

    img: RGB uint8 (H, W, 3)
    """
    max_patches = max_soft_tokens * POOLING_KERNEL_SIZE**2
    height, width, _ = img.shape
    target_height, target_width = get_aspect_ratio_preserving_size(
        height, width, max_patches
    )
    if (target_height, target_width) != (height, width):
        img = tv_resize(
            img, (target_height, target_width), interpolation="bicubic", antialias=True
        )

    img = img.astype(np.float32) * np.float32(1 / 255)
    img = img.transpose(2, 0, 1)  # HWC -> CHW

    channel = img.shape[0]
    patch_height = target_height // PATCH_SIZE
    patch_width = target_width // PATCH_SIZE
    patches = img.reshape(channel, patch_height, PATCH_SIZE, patch_width, PATCH_SIZE)
    patches = patches.transpose(1, 3, 2, 4, 0)
    patches = patches.reshape(patch_height * patch_width, -1)
    num_soft_tokens = patches.shape[0] // POOLING_KERNEL_SIZE**2

    # (x, y) position of each patch, padding patches are (-1, -1)
    grid_x, grid_y = np.meshgrid(np.arange(patch_width), np.arange(patch_height))
    positions = np.stack([grid_x, grid_y], axis=-1).reshape(-1, 2)

    pad = max_patches - patches.shape[0]
    pixel_values = np.pad(patches, ((0, pad), (0, 0)))
    position_ids = np.pad(positions, ((0, pad), (0, 0)), constant_values=-1)

    return pixel_values, position_ids.astype(np.int64), num_soft_tokens


def load_video_frames(video_path):
    """Frames sampled at VIDEO_FPS, at most VIDEO_MAX_FRAMES (uniformly re-sampled)."""
    capture = cv2.VideoCapture(video_path)
    fps = capture.get(cv2.CAP_PROP_FPS)
    total_num_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_num_frames / fps

    step = fps / VIDEO_FPS
    num_sampled = max(1, int(duration * VIDEO_FPS))
    indices = [min(total_num_frames - 1, int(i * step)) for i in range(num_sampled)]
    if len(indices) > VIDEO_MAX_FRAMES:
        linspace_idx = np.linspace(0, len(indices) - 1, VIDEO_MAX_FRAMES, dtype=int)
        indices = [indices[i] for i in linspace_idx]

    frames = []
    for index in indices:
        capture.set(cv2.CAP_PROP_POS_FRAMES, index)
        ret, frame = capture.read()
        if not ret:
            break
        frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    capture.release()

    return frames


def hertz_to_mel(freq):
    return 2595.0 * np.log10(1.0 + (freq / 700.0))


def mel_to_hertz(mels):
    return 700.0 * (np.power(10, mels / 2595.0) - 1.0)


def mel_filter_bank():
    """Triangular HTK mel filters (no normalization) of shape (FFT_LENGTH // 2 + 1, NUM_MEL_BINS)."""
    mel_freqs = np.linspace(
        hertz_to_mel(0.0), hertz_to_mel(SAMPLING_RATE / 2), NUM_MEL_BINS + 2
    )
    filter_freqs = mel_to_hertz(mel_freqs)
    fft_freqs = np.linspace(0, SAMPLING_RATE // 2, FFT_LENGTH // 2 + 1)

    filter_diff = np.diff(filter_freqs)
    slopes = np.expand_dims(filter_freqs, 0) - np.expand_dims(fft_freqs, 1)
    down_slopes = -slopes[:, :-2] / filter_diff[:-1]
    up_slopes = slopes[:, 2:] / filter_diff[1:]
    return np.maximum(np.zeros(1), np.minimum(down_slopes, up_slopes))


def extract_audio_features(waveform):
    """log-mel spectrogram (frames, 128) and its valid-frame mask"""
    waveform = waveform[:AUDIO_MAX_LENGTH]
    length = len(waveform)
    padded_length = (
        math.ceil(length / AUDIO_PAD_TO_MULTIPLE_OF) * AUDIO_PAD_TO_MULTIPLE_OF
    )
    attention_mask = np.zeros(padded_length, dtype=np.int32)
    attention_mask[:length] = 1
    waveform = np.pad(waveform, (0, padded_length - length))

    # semicausal padding, frames of FRAME_LENGTH + 1 samples
    pad_left = FRAME_LENGTH // 2
    waveform = np.pad(waveform, (pad_left, 0))
    attention_mask = np.pad(attention_mask, (pad_left, 0))
    frame_size = FRAME_LENGTH + 1
    num_frames = (len(waveform) - frame_size) // HOP_LENGTH + 1
    frames = np.lib.stride_tricks.as_strided(
        waveform,
        shape=(num_frames, frame_size),
        strides=(waveform.strides[0] * HOP_LENGTH, waveform.strides[0]),
    )
    frames = frames[:, :-1]

    # periodic hann window
    window = np.hanning(FRAME_LENGTH + 1)[:-1].astype(np.float32)
    stft = np.fft.rfft(frames * window, n=FFT_LENGTH, axis=-1)
    mel_spec = np.matmul(np.abs(stft), mel_filter_bank())
    log_mel_spec = np.log(mel_spec + MEL_FLOOR).astype(np.float32)

    frame_end_indices = np.arange(num_frames) * HOP_LENGTH + frame_size - 1
    mask = attention_mask[frame_end_indices].astype(bool)
    input_features = log_mel_spec * mask[:, None]

    return input_features, mask


def num_audio_soft_tokens(mask):
    """Number of valid frames after the two stride-2 convolutions of the audio encoder."""
    for _ in range(2):
        t_out = (len(mask) + 2 - 3) // 2 + 1
        mask = mask[::2][:t_out]
    return int(mask.sum())


def tokenize(tokenizer, text):
    if args.disable_ailia_tokenizer:
        input_ids = tokenizer(text, return_tensors="np")["input_ids"]
    else:
        input_ids = np.array([tokenizer(text)["input_ids"]])
    return input_ids.astype(np.int64)


# ======================
# Main functions
# ======================


def run_vision(models, pixel_values, position_ids):
    net = models["vision"]
    if not args.onnx:
        output = net.predict([pixel_values, position_ids])
    else:
        output = net.run(
            None, {"pixel_values": pixel_values, "pixel_position_ids": position_ids}
        )
    (vision_features,) = output
    return vision_features


def run_audio(models, input_features, input_features_mask):
    net = models["audio"]
    if not args.onnx:
        output = net.predict([input_features, input_features_mask])
    else:
        output = net.run(
            None,
            {
                "input_features": input_features,
                "input_features_mask": input_features_mask,
            },
        )
    (audio_features,) = output
    return audio_features


def embed(models, text, image_features=None, video_features=None, audio_features=None):
    """Sentence embedding of the text where the placeholders are filled with the media features."""
    input_ids = tokenize(models["tokenizer"], text)
    attention_mask = np.ones(input_ids.shape, dtype=np.int64)

    # unused modalities take a dummy row, which is ignored when there is no placeholder
    dummy = np.zeros((1, TEXT_HIDDEN_SIZE), dtype=np.float32)
    image_features = dummy if image_features is None else image_features
    video_features = dummy if video_features is None else video_features
    audio_features = dummy if audio_features is None else audio_features

    net = models["net"]
    if not args.onnx:
        output = net.predict(
            [input_ids, attention_mask, image_features, video_features, audio_features]
        )
    else:
        output = net.run(
            None,
            {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "image_features": image_features,
                "video_features": video_features,
                "audio_features": audio_features,
            },
        )
    last_hidden_state, sentence_embedding = output

    return sentence_embedding[0]


def encode_text(models, text, prompt):
    return embed(models, prompt + text)


def encode_image(models, image_path):
    img = np.array(Image.open(image_path).convert("RGB"))
    pixel_values, position_ids, num_soft_tokens = preprocess_image(
        img, IMAGE_MAX_SOFT_TOKENS
    )
    image_features = run_vision(models, pixel_values[None], position_ids[None])

    text = BOI_TOKEN + IMAGE_TOKEN * num_soft_tokens + EOI_TOKEN
    return embed(models, text, image_features=image_features)


def encode_video(models, video_path):
    frames = load_video_frames(video_path)
    pixel_values, position_ids = [], []
    for frame in frames:
        pv, pos, num_soft_tokens = preprocess_image(frame, VIDEO_MAX_SOFT_TOKENS)
        pixel_values.append(pv)
        position_ids.append(pos)
    video_features = run_vision(models, np.stack(pixel_values), np.stack(position_ids))

    text = (BOI_TOKEN + VIDEO_TOKEN * num_soft_tokens + EOI_TOKEN) * len(frames)
    return embed(models, text, video_features=video_features)


def load_audio(audio_path):
    """Mono waveform at SAMPLING_RATE"""
    if not args.ffmpeg:
        waveform, _ = librosa.load(audio_path, sr=SAMPLING_RATE, mono=True)
    else:
        import subprocess

        # differs slightly from librosa in the resampler and the downmix ((L + R) / sqrt(2))
        command = [
            "ffmpeg", "-v", "error", "-i", audio_path,
            "-ac", "1", "-ar", str(SAMPLING_RATE), "-f", "f32le", "-",
        ]  # fmt: skip
        result = subprocess.run(command, capture_output=True, check=True)
        waveform = np.frombuffer(result.stdout, dtype=np.float32)
    return waveform


def encode_audio(models, audio_path):
    waveform = load_audio(audio_path)
    input_features, mask = extract_audio_features(waveform)
    audio_features = run_audio(models, input_features[None], mask[None])

    text = BOA_TOKEN + AUDIO_TOKEN * num_audio_soft_tokens(mask) + EOA_TOKEN
    return embed(models, text, audio_features=audio_features)


def normalize(embeddings):
    norms = np.linalg.norm(embeddings, ord=2, axis=-1, keepdims=True)
    return embeddings / np.maximum(norms, 1e-12)


def predict(models, queries, documents):
    query_embeddings = []
    for query in queries:
        query_embeddings.append(encode_text(models, query, QUERY_PROMPT))
        logger.info(f"query: {query}")
    document_embeddings = []
    for kind, document in documents:
        if kind == "text":
            emb = encode_text(models, document, DOCUMENT_PROMPT)
        elif kind == "image":
            emb = encode_image(models, document)
        elif kind == "video":
            emb = encode_video(models, document)
        else:
            emb = encode_audio(models, document)
        document_embeddings.append(emb)
        logger.info(f"{kind}: {document}")

    query_embeddings = np.stack(query_embeddings)
    document_embeddings = np.stack(document_embeddings)
    if args.truncate_dim is not None:
        query_embeddings = normalize(query_embeddings[:, : args.truncate_dim])
        document_embeddings = normalize(document_embeddings[:, : args.truncate_dim])

    scores = query_embeddings @ document_embeddings.T
    return scores


def recognize(models, documents):
    queries = args.query or QUERIES

    # inference
    logger.info("Start inference...")
    if args.benchmark:
        logger.info("BENCHMARK mode")
        for i in range(args.benchmark_count):
            start = int(round(time.time() * 1000))
            scores = predict(models, queries, documents)
            end = int(round(time.time() * 1000))
            logger.info(f"\tailia processing time {end - start} ms")
    else:
        scores = predict(models, queries, documents)

    logger.info("Similarity (rows: queries, columns: documents)")
    logger.info("\n" + np.array2string(scores, precision=4))
    for query, row in zip(queries, scores):
        logger.info(f"Query: {query}")
        for rank, i in enumerate(np.argsort(-row)):
            kind, document = documents[i]
            logger.info(f"  [{rank + 1}] ({row[i]:.4f}) {kind}: {document}")

    logger.info("Script finished successfully.")


def main():
    documents = [("text", d) for d in args.document or []]
    for path in args.input or []:
        ext = os.path.splitext(path)[1].lower()
        if ext in VIDEO_EXTENSIONS:
            documents.append(("video", path))
        elif ext in AUDIO_EXTENSIONS:
            documents.append(("audio", path))
        else:
            documents.append(("image", path))
    if not documents:
        documents = [
            ("image", IMAGE_PATH),
            ("audio", AUDIO_PATH),
            ("video", VIDEO_PATH),
        ]
    kinds = {kind for kind, _ in documents}
    use_vision = bool(kinds & {"image", "video"})
    use_audio = "audio" in kinds

    # model files check and download
    check_and_download_models(WEIGHT_PATH, MODEL_PATH, REMOTE_PATH)
    if use_vision:
        check_and_download_models(WEIGHT_VISION_PATH, MODEL_VISION_PATH, REMOTE_PATH)
    if use_audio:
        check_and_download_models(WEIGHT_AUDIO_PATH, MODEL_AUDIO_PATH, REMOTE_PATH)

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
        from ailia_tokenizer import GemmaTokenizer

        tokenizer = GemmaTokenizer.from_pretrained("./tokenizer")

    models = {
        "tokenizer": tokenizer,
        "net": load_net(MODEL_PATH, WEIGHT_PATH),
    }
    if use_vision:
        models["vision"] = load_net(MODEL_VISION_PATH, WEIGHT_VISION_PATH)
    if use_audio:
        models["audio"] = load_net(MODEL_AUDIO_PATH, WEIGHT_AUDIO_PATH)

    recognize(models, documents)


if __name__ == "__main__":
    main()
