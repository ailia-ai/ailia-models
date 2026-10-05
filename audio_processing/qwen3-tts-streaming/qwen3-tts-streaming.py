import os
import sys
import json
import time
import queue
import threading

import numpy as np

# import original modules
sys.path.append('../../util')
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from arg_utils import get_base_parser, update_parser  # noqa: E402
from model_utils import check_and_download_models, check_and_download_file  # noqa: E402
# ailia 1.7: consecutive DNN layers are recorded once as a CUDA graph and replayed
# (DnnSegment), which removes the per layer overhead of the ~600 small layers of a
# talker step (40 -> 11 ms on RTX 3080). It is opt-in through this variable and
# needs the default memory mode (the inter-stage memory optimizer disables it).
os.environ.setdefault("AILIA_ENABLE_DNN_SEGMENT", "1")
try:
    import ailia
    AILIA_EXIST = True
except ImportError:
    AILIA_EXIST = False

from live_text_source import LiveTextSource, token_offsets_fn  # noqa: E402

# logger
from logging import getLogger   # noqa: E402
logger = getLogger(__name__)

# ======================
# PARAMETERS
# ======================

OUTPUT_WAV_PATH = 'output.wav'
INPUT_TEXT_PATH = 'input.txt'
REFERENCE_WAV_PATH = 'reference.wav'
REFERENCE_TEXT = "かしこまりました。"
UTTERANCE_END = "。！？!?"
REMOTE_PATH = "https://storage.googleapis.com/ailia-models/qwen3-tts-streaming/"
TOKENIZER_REMOTE_PATH = "https://storage.googleapis.com/ailia-models/qwen3-tts/"   # same tokenizer as qwen3-tts

SAMPLE_RATE = 24000
FRAME_SAMPLES = 1920        # decoder upsample rate: 24000 Hz / 12.5 frames per second


# ======================
# Arguemnt Parser Config
# ======================
parser = get_base_parser(
    'Qwen3-TTS streaming (CustomVoice): text in, audio out, both incremental',
    INPUT_TEXT_PATH,
    OUTPUT_WAV_PATH,
    large_model=True,
)
parser.add_argument(
    '-p', '--parameter_num', default='0.6B', help='[0.6B, 1.7B]'
)
parser.add_argument(
    '-m', '--model', default='custom_voice', choices=['custom_voice', 'base'],
    help='custom_voice: predefined speakers (--speaker); base: voice clone from --ref_audio / --ref_text'
)
parser.add_argument('--ref_audio', default=REFERENCE_WAV_PATH, help='base: reference audio to clone the voice of')
parser.add_argument('--ref_text', default=REFERENCE_TEXT,
                    help='base: transcript of the reference audio (in-context voice clone). An empty string '
                         'selects the speaker-embedding-only clone (x_vector_only_mode of Qwen3-TTS): no '
                         'transcript, no wait for the first tokens, the latency of CustomVoice, lower similarity')
parser.add_argument(
    '--speaker', default='Ono_Anna',
    help='CustomVoice speaker: Ono_Anna, Vivian, Serena, Ryan, Aiden, Sohee, Eric, Dylan, Uncle_Fu'
)
parser.add_argument(
    '--language', default='Japanese',
    help='Auto, or chinese, english, japanese, korean, german, french, russian, portuguese, spanish, italian'
)
parser.add_argument(
    '--commit', default='token', choices=['token', 'punct'],
    help='when text becomes available to the talker: token = per talker token, 3 held back (default, earliest '
         'speech); punct = at spaces and 、。！？ etc. (tokenizes exactly like the whole text)'
)
# how the text arrives: -i is a text file (or the text itself); it is fed in pieces at
# --cps characters per second, echoed to the console as it goes in, and spoken as it arrives
parser.add_argument(
    '--cps', type=float, default=30.0,
    help='the input text is pushed in pieces of --piece characters at this many characters per second, like an LLM would'
)
parser.add_argument('--piece', type=int, default=2, help='characters per push')
parser.add_argument(
    '--max_chars', type=int, default=150,
    help='the text is spoken as consecutive utterances cut at 。！？ or line ends, each at most this long '
         '(the fixed KV buffer talker holds about 40 s of speech)'
)
parser.add_argument(
    '--replay', default=None,
    help='JSON list of texts, each pushed as Qwen2.5 LLM token deltas at --rate tokens per second'
)
parser.add_argument('--rate', type=float, default=30.0, help='tokens per second for --replay')
# how the audio leaves
parser.add_argument('--play', action='store_true', help='play on the default audio device (needs sounddevice)')
parser.add_argument('--quiet', action='store_true', help='do not echo the text to the console as it is fed in')
parser.add_argument('--prebuffer', type=float, default=0.5, help='seconds of audio to queue before playback starts')
parser.add_argument('--results', default=None, help='append one JSON line of timings per utterance to this file')
# generation
parser.add_argument('--temperature', type=float, default=0.9, help='talker sampling temperature, 0 = greedy')
parser.add_argument('--top_k', type=int, default=50, help='talker top-k')
parser.add_argument('--subtalker_temperature', type=float, default=0.9, help='code predictor temperature, 0 = greedy')
parser.add_argument('--emit_every', type=int, default=8, help='emit a PCM chunk every N frames (N * 80 ms)')
parser.add_argument('--decode_window', type=int, default=80, help='frames decoded for each chunk (context + chunk)')
parser.add_argument('--first_chunk_emit', type=int, default=0, help='frames in the first chunk, 0 = same as --emit_every')
parser.add_argument('--seed', type=int, default=None, help='random seed')
# runtime
parser.add_argument('--onnx', action='store_true', help='execute onnxruntime version.')
parser.add_argument(
    '--cuda_graph', default='talker,code_predictor',
    help='onnxruntime + CUDA: which fixed-shape graphs to replay as CUDA graphs (comma list, or none)'
)
parser.add_argument('--disable_ailia_tokenizer', action='store_true', help='disable ailia tokenizer.')
parser.add_argument('--profile', action='store_true', help='use profile model')
args = update_parser(parser, check_input_type=False)

if args.parameter_num not in ("0.6B", "1.7B"):
    logger.error("invalid parameter_num")
    sys.exit()
parameter_num = args.parameter_num
if not AILIA_EXIST and not args.onnx:
    logger.info("ailia is not installed, running onnxruntime")
    args.onnx = True

MODEL_SUFFIX = {"base": "", "custom_voice": "_custom_voice"}[args.model]
CONFIG_PATH = f"config_{parameter_num}{MODEL_SUFFIX}.json"

# The ONNX split is the one of audio_processing/qwen3-tts (ailia-models). The
# Base (voice clone) files are the same ones that sample uses; the CustomVoice
# files come from export/export_onnx.py --model custom_voice and carry a
# _custom_voice suffix. CustomVoice's speaker is a row of the codec embedding
# table, so it has no encoder.
#   encoder           (base only) reference audio -> codec frames + speaker embedding
#   prompt            text token ids -> talker hidden (text embedding + projection)
#   codec_embedding   the 16 codec tables, read out once at start up
#   talker ..._static  auto regressive body with a fixed KV buffer (export --static):
#                     onnxruntime updates the cache in place and replays it as a
#                     CUDA graph, ailia copies present -> past inside the device
#   code_predictor    groups 1..15 of a frame in one call, top-k (50) sampling inside
#                     (export --only code_predictor_frame, simplify_frame.py, then
#                     convert_to_fp16.py, which names the result qwen3_tts_code_predictor_...)
#   decoder           codec frames -> waveform
def model_path(name, extra=""):
    # the models are the fp16 ones of export/convert_to_fp16.py (weights and
    # MatMul / Conv in fp16, every other tensor fp32); only the encoder stays
    # fp32, its outputs are codebook indices which fp16 flips
    if name != "encoder":
        extra += "_fp16"
    weight = f"qwen3_tts_{name}_{parameter_num}{MODEL_SUFFIX}{extra}.onnx"
    return weight, weight + ".prototxt"


WEIGHT_PATH_ENCODER, MODEL_PATH_ENCODER = model_path("encoder")
WEIGHT_PATH_PROMPT, MODEL_PATH_PROMPT = model_path("prompt")
WEIGHT_PATH_CODEC_EMBEDDING, MODEL_PATH_CODEC_EMBEDDING = model_path("codec_embedding")
WEIGHT_PATH_DECODER, MODEL_PATH_DECODER = model_path("decoder")
WEIGHT_PATH_TALKER_STATIC, MODEL_PATH_TALKER_STATIC = model_path("talker", "_static")
WEIGHT_PATH_CODE_PREDICTOR_FRAME, MODEL_PATH_CODE_PREDICTOR_FRAME = model_path("code_predictor")

onnx_list = [
    (WEIGHT_PATH_PROMPT, MODEL_PATH_PROMPT),
    (WEIGHT_PATH_CODEC_EMBEDDING, MODEL_PATH_CODEC_EMBEDDING),
    (WEIGHT_PATH_TALKER_STATIC, MODEL_PATH_TALKER_STATIC),
    (WEIGHT_PATH_CODE_PREDICTOR_FRAME, MODEL_PATH_CODE_PREDICTOR_FRAME),
    (WEIGHT_PATH_DECODER, MODEL_PATH_DECODER),
]
if args.model == "base":
    onnx_list.append((WEIGHT_PATH_ENCODER, MODEL_PATH_ENCODER))
# weights over the 2GB protobuf limit live next to the ONNX in a .data file (the 1.7B talker)
external_data_list = [WEIGHT_PATH_TALKER_STATIC + ".data"] if parameter_num == "1.7B" else []

TOKENIZER_DIR = "./tokenizer"
VOCAB_PATH = os.path.join(TOKENIZER_DIR, "vocab.json")
MERGES_PATH = os.path.join(TOKENIZER_DIR, "merges.txt")
TOKENIZER_SPECIAL_TOKENS = [
    '<|endoftext|>', '<|im_start|>', '<|im_end|>',
    '<|object_ref_start|>', '<|object_ref_end|>',
    '<|box_start|>', '<|box_end|>',
    '<|quad_start|>', '<|quad_end|>',
    '<|vision_start|>', '<|vision_end|>',
    '<|vision_pad|>', '<|image_pad|>', '<|video_pad|>',
]

# the ailia path needs 1.7.0 or later (fused code predictor graph: TopK etc.)
if not args.onnx:
    _v = [int(x) for x in ailia.get_version().split(".")[:2]]
    if _v < [1, 7]:
        logger.error(f"ailia SDK {ailia.get_version()} is too old: 1.7.0 or later is required (or run with --onnx)")
        sys.exit()


# ======================
# Helpers
# ======================
class Benchmark:
    """Per model call counts and total time, reported once at the end (-b)."""

    def __init__(self, enabled=True):
        self.enabled = enabled
        self.records = {}

    def add(self, name, seconds):
        calls, total = self.records.get(name, (0, 0.0))
        self.records[name] = (calls + 1, total + seconds)

    def reset(self):
        self.records = {}

    def report(self):
        for name, (calls, total) in self.records.items():
            logger.info("\t{} processing time {:.0f} ms ({} calls, {:.1f} ms/call)".format(
                name, total * 1000, calls, total * 1000 / calls))

    def as_dict(self):
        return {k: [c, round(t * 1000 / c, 2)] for k, (c, t) in self.records.items()}


def top_k_top_p(logits, top_k=0, top_p=1.0):
    if 0 < top_k < logits.shape[-1]:
        kth = np.partition(logits, -top_k)[-top_k]
        logits = np.where(logits < kth, -np.inf, logits)
    if top_p < 1.0:
        order = np.argsort(-logits)
        sorted_logits = logits[order]
        probs = np.exp(sorted_logits - sorted_logits.max())
        probs /= probs.sum()
        remove = np.cumsum(probs) > top_p
        remove[0] = False
        sorted_logits[remove] = -np.inf
        logits = np.empty_like(sorted_logits)
        logits[order] = sorted_logits
    return logits


def sample_token(logits, rng, temperature=0.9, top_k=50, top_p=1.0, suppress=None):
    """One token id from a 1D logits vector; temperature 0 is greedy."""
    logits = logits.astype(np.float64, copy=True)
    if suppress is not None:
        logits[suppress] = -np.inf
    if temperature <= 0:
        return int(np.argmax(logits))
    logits = top_k_top_p(logits / temperature, top_k, top_p)
    probs = np.exp(logits - logits.max())
    probs /= probs.sum()
    return int(rng.choice(len(probs), p=probs))


# mel spectrogram fed to the speaker encoder (as in audio_processing/qwen3-tts)
MEL_N_FFT, MEL_NUM_MELS, MEL_HOP, MEL_WIN, MEL_FMIN, MEL_FMAX = 1024, 128, 256, 1024, 0, 12000


def mel_spectrogram(y, sr):
    import librosa
    mel_basis = librosa.filters.mel(sr=sr, n_fft=MEL_N_FFT, n_mels=MEL_NUM_MELS, fmin=MEL_FMIN, fmax=MEL_FMAX)
    padding = (MEL_N_FFT - MEL_HOP) // 2
    y = np.pad(y[None], [(0, 0), (padding, padding)], mode="reflect")
    spec = np.abs(np.squeeze(librosa.stft(y, n_fft=MEL_N_FFT, hop_length=MEL_HOP, win_length=MEL_WIN,
                                          window=np.hanning(MEL_WIN), center=False, pad_mode="reflect")))
    spec = np.log(np.clip(np.dot(mel_basis, spec), 1e-5, None)).T          # (T, 128)
    return spec[None].astype(np.float32)


def create_tokenizer():
    """(encode function, tokenizer): the tokenizer also gives LiveTextSource its token offsets."""
    if args.disable_ailia_tokenizer:
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_DIR)
        return lambda text: tokenizer(text)["input_ids"], tokenizer
    from ailia_tokenizer import GPT2Tokenizer
    tokenizer = GPT2Tokenizer.from_pretrained(TOKENIZER_DIR)
    tokenizer.add_special_tokens({"additional_special_tokens": TOKENIZER_SPECIAL_TOKENS})
    return lambda text: list(tokenizer.encode(text)), tokenizer


# ======================
# onnxruntime
# ======================
def onnx_providers(env_id, cuda_graph=False):
    """CUDA first when the ailia env_id selected is a GPU (or AUTO), else CPU only."""
    gpu = not AILIA_EXIST or env_id == ailia.ENVIRONMENT_AUTO or ailia.get_environment(env_id).type == "GPU"
    if not gpu:
        return ["CPUExecutionProvider"]
    options = {}
    if cuda_graph:
        options["enable_cuda_graph"] = "1"
    return [("CUDAExecutionProvider", options), "CPUExecutionProvider"]


class OnnxNet:
    """One onnxruntime session; run(*inputs) in graph input order."""

    def __init__(self, weight, providers, bench, name):
        import onnxruntime as ort
        opts = ort.SessionOptions()
        opts.log_severity_level = 3
        self.session = ort.InferenceSession(weight, opts, providers=providers)
        self.input_names = [i.name for i in self.session.get_inputs()]
        self.output_names = [o.name for o in self.session.get_outputs()]
        self.device = "cuda" if self.session.get_providers()[0] == "CUDAExecutionProvider" else "cpu"
        self.bench, self.name = bench, name

    def run(self, *inputs):
        t = time.perf_counter()
        out = self.session.run(None, dict(zip(self.input_names, inputs)))
        self.bench.add(self.name, time.perf_counter() - t)
        return out


class OnnxStaticNet(OnnxNet):
    """A fixed-shape graph with every input and output in buffers bound once.

    Each call only copies the new input values into the same device buffers, so
    on CUDA the session can be replayed as a CUDA graph: the launch overhead of
    the hundreds of small kernels a step is made of goes away, which is what
    torch.compile's reduce-overhead mode does for the PyTorch version.
    """

    def __init__(self, weight, providers, bench, name, example_inputs):
        import onnxruntime as ort
        super().__init__(weight, providers, bench, name)
        self.binding = self.session.io_binding()
        self.in_vals = []
        for in_name, arr in zip(self.input_names, example_inputs):
            v = ort.OrtValue.ortvalue_from_numpy(np.ascontiguousarray(arr), self.device, 0)
            self.in_vals.append(v)
            self.binding.bind_ortvalue_input(in_name, v)
        for out_name in self.output_names:
            self.binding.bind_output(out_name, self.device, 0)
        self.session.run_with_iobinding(self.binding)
        # the buffers onnxruntime allocated on the first run stay the outputs
        self.out_vals = self.binding.get_outputs()
        for out_name, v in zip(self.output_names, self.out_vals):
            self.binding.bind_ortvalue_output(out_name, v)

    def run(self, *inputs):
        t = time.perf_counter()
        for v, arr in zip(self.in_vals, inputs):
            v.update_inplace(np.ascontiguousarray(arr))
        self.session.run_with_iobinding(self.binding)
        out = [v.numpy() for v in self.out_vals]
        self.bench.add(self.name, time.perf_counter() - t)
        return out


class OnnxStaticTalker(OnnxNet):
    """The fixed KV buffer talker (export --static), one frame per step.

    Every layer's present_pkv output is bound to the buffer its past_pkv input
    reads: the graph's cache write is elementwise (past * keep + spread), so the
    cache is updated in place and never leaves the device (CUDA, or the host
    on the CPU provider). The prefill is fed position by position, which is
    identical to a batched prefill (causal attention) and keeps the one
    captured CUDA graph.
    """

    def __init__(self, weight, providers, bench, hidden):
        import onnxruntime as ort
        super().__init__(weight, providers, bench, "talker")
        shape = self.session.get_inputs()[4].shape          # past_pkv_0 [1, kv, buf, dim]
        self.kv_heads, self.buf, self.head_dim = int(shape[1]), int(shape[2]), int(shape[3])
        self.num_cache = len(self.input_names) - 4
        mk = lambda arr: ort.OrtValue.ortvalue_from_numpy(np.ascontiguousarray(arr), self.device, 0)
        self.x = mk(np.zeros((1, 1, hidden), np.float32))
        self.mask = mk(np.zeros((1, 1, 1, self.buf), np.float32))
        self.pos = mk(np.zeros((1, 1), np.int64))
        self.cpos = mk(np.zeros(1, np.int64))
        self.cache = [mk(np.zeros((1, self.kv_heads, self.buf, self.head_dim), np.float32))
                      for _ in range(self.num_cache)]
        b = self.session.io_binding()
        for name, v in zip(self.input_names, [self.x, self.mask, self.pos, self.cpos] + self.cache):
            b.bind_ortvalue_input(name, v)
        # logits and last_hidden first: get_outputs() returns values in binding order
        for name in self.output_names[:2]:
            b.bind_output(name, self.device, 0)
        for name, v in zip(self.output_names[2:], self.cache):
            b.bind_ortvalue_output(name, v)                 # in place cache update
        self.session.run_with_iobinding(b)
        self.outs = b.get_outputs()[:2]
        for name, v in zip(self.output_names[:2], self.outs):
            b.bind_ortvalue_output(name, v)
        self.binding = b

    def prefill(self, inputs_embeds, start=0):
        """Feed positions start.. one by one; positions before start keep their cache."""
        for i in range(inputs_embeds.shape[1]):
            logits, last_hidden = self.step(inputs_embeds[:, i:i + 1], start + i)
        return logits, last_hidden

    def step(self, x, position):
        t = time.perf_counter()
        mask = np.zeros((1, 1, 1, self.buf), np.float32)
        mask[..., position + 1:] = -np.inf
        self.x.update_inplace(np.ascontiguousarray(x, dtype=np.float32))
        self.mask.update_inplace(mask)
        self.pos.update_inplace(np.array([[position]], np.int64))
        self.cpos.update_inplace(np.array([position], np.int64))
        self.session.run_with_iobinding(self.binding)
        out = [v.numpy() for v in self.outs]
        self.bench.add(self.name, time.perf_counter() - t)
        return out


# ======================
# ailia SDK
# ======================
class AiliaNet:
    def __init__(self, model, weight, memory_mode, env_id, bench, name):
        self.net = ailia.Net(stream=model, weight=weight, memory_mode=memory_mode, env_id=env_id)
        if args.profile:
            self.net.set_profile_mode(True)
        self.bench, self.name = bench, name

    def run(self, *inputs):
        t = time.perf_counter()
        for index, value in enumerate(inputs):
            self.net.set_input_blob_shape(value.shape, index)
        out = self.net.run(list(inputs))
        self.bench.add(self.name, time.perf_counter() - t)
        self.check_finite(out)
        return out

    def check_finite(self, outputs):
        """--debug: report the first model whose output has NaN/Inf (FP16 overflow hunting)."""
        if args.debug:
            for i, o in enumerate(outputs):
                if o.dtype.kind == "f" and not np.isfinite(o).all():
                    logger.warning(f"{self.name}: output {i} has NaN/Inf")
                    break


class AiliaStaticTalker(AiliaNet):
    """The same fixed KV buffer talker on ailia, one position per call like the
    onnxruntime one, so that every input keeps the same shape ([1, 1, H] etc.)
    and the network is recorded as one CUDA graph. The first position of an
    utterance passes zero caches in; every later call copies each present_pkv
    blob onto its past_pkv blob inside ailia (copy_blob_data, fixed
    [1, kv, buf, dim] buffers, ~1 ms for 56 blobs) and only logits and
    last_hidden reach Python.
    """

    def __init__(self, model, weight, memory_mode, env_id, bench, kv_heads, head_dim):
        super().__init__(model, weight, memory_mode, env_id, bench, "talker")
        inputs = self.net.get_input_blob_list()
        self.num_cache = len(inputs) - 4
        self.kv_heads, self.head_dim = kv_heads, head_dim
        self.buf = int(self.net.get_blob_shape(inputs[4])[2])      # the buffer length is fixed in the graph

    def prefill(self, inputs_embeds, start=0):
        """Feed positions start.. one by one; positions before start keep their cache."""
        for i in range(inputs_embeds.shape[1]):
            logits, last_hidden = self.step(inputs_embeds[:, i:i + 1], start + i)
        return logits, last_hidden

    def step(self, x, position):
        t = time.perf_counter()
        net = self.net
        input_blobs, output_blobs = net.get_input_blob_list(), net.get_output_blob_list()
        mask = np.zeros((1, 1, 1, self.buf), np.float32)
        mask[..., position + 1:] = -np.inf
        fixed = [np.ascontiguousarray(x, dtype=np.float32), mask,
                 np.array([[position]], np.int64), np.array([position], np.int64)]
        if position == 0:
            # start of an utterance: the cache buffers come in as zeros
            inputs = fixed + [np.zeros((1, self.kv_heads, self.buf, self.head_dim), np.float32)
                              for _ in range(self.num_cache)]
            for index, value in enumerate(inputs):
                net.set_input_blob_shape(value.shape, index)
            out = net.run(inputs)[:2]
        else:
            for index, value in enumerate(fixed):
                net.set_input_blob_data(value, input_blobs[index])
            for i in range(self.num_cache):
                net.copy_blob_data(input_blobs[4 + i], output_blobs[2 + i], net)
            net.update()
            out = [net.get_blob_data(output_blobs[i]) for i in range(2)]
        self.bench.add(self.name, time.perf_counter() - t)
        self.check_finite(out)
        return out


# ======================
# Voice
# ======================
class Voice:
    """A voice as the talker sees it: the speaker embedding and, for the in-context
    clone, the reference codec frames with the reference text.

    It also caches, per language, the utterance-independent prefix of the talker
    prompt: role | codec tags, speaker, codec pad | reference text (+) reference
    codec. The talker's KV buffer is written in place at each position, so once
    this prefix has been prefilled for a voice and language it stays valid in
    the buffer: the next utterance of the same voice only prefills its own text
    positions after it (talker.prefix_key says what the buffer holds).
    """

    def __init__(self, name, speaker_embed, frames=(), text=None, codec_icl=None):
        self.name = name
        self.speaker_embed = speaker_embed          # [1, 1, H]
        self.frames = list(frames)                  # reference codec frames (ICL) or []
        self.text = text                            # projected reference text [1, n_ref, H] (ICL) or None
        self.codec_icl = codec_icl                  # codec_bos + reference frames [1, n_icl, H] (ICL) or None
        self._prefix = {}

    @property
    def icl(self):
        return self.codec_icl is not None

    @property
    def need_tokens(self):
        """Tokens of the utterance that must be committed before the prefill."""
        return self.codec_icl.shape[1] - self.text.shape[1] if self.icl else 1

    def prefix(self, tts, language_id):
        """(embeddings [1, P, H], key) of the utterance-independent prompt prefix."""
        if language_id not in self._prefix:
            tc = tts.cfg["talker_config"]
            if language_id is None:
                tags = [tc["codec_nothink_id"], tc["codec_think_bos_id"], tc["codec_think_eos_id"]]
            else:
                tags = [tc["codec_think_id"], tc["codec_think_bos_id"], language_id, tc["codec_think_eos_id"]]
            codec_input = np.concatenate(
                [tts.codec_table[tags][None], self.speaker_embed,
                 tts.codec_table[[tc["codec_pad_id"], tc["codec_bos_id"]]][None]], axis=1)
            n = codec_input.shape[1]
            head = np.concatenate([np.repeat(tts.tts_pad, n - 2, axis=1), tts.tts_bos], axis=1) + codec_input[:, :-1]
            parts = [tts.role_embed, head]
            if self.icl:
                n_ref = self.text.shape[1]
                parts.append(self.text + self.codec_icl[:, :n_ref])     # reference text on its codec frames
            embeds = np.concatenate(parts, axis=1).astype(np.float32)
            self._prefix[language_id] = (embeds, codec_input[:, -1:], (self.name, language_id))
        return self._prefix[language_id]


# ======================
# Model
# ======================
class Qwen3TTSStreaming:
    def __init__(self, memory_mode, env_id):
        self.cfg = json.load(open(CONFIG_PATH, encoding="utf-8"))
        tc = self.cfg["talker_config"]
        self.hidden = tc["hidden_size"]
        self.num_groups = tc["num_code_groups"]                         # 16
        self.talker_vocab = tc["vocab_size"]                            # 3072
        self.group_vocab = tc["code_predictor_config"]["vocab_size"]    # 2048
        self.zero_row = self.talker_vocab + (self.num_groups - 1) * self.group_vocab
        self.rng = np.random.default_rng(args.seed)
        self.bench = Benchmark()
        self.encode, self.tokenizer = create_tokenizer()
        self.decode_window_frames = args.decode_window
        self.cuda_graph = tuple(x for x in args.cuda_graph.split(",") if x and x != "none")

        self.encoder = None
        if args.onnx:
            self._load_onnx(env_id)
        else:
            self._load_ailia(memory_mode, env_id)

        # The 16 codec embedding tables, read out of the graph once so a lookup in
        # the decode loop is a numpy gather. Rows: the talker's 3072, then 15 x 2048,
        # then one zero row.
        n_rows = self.zero_row + 1
        rows = np.full((n_rows, self.num_groups), self.zero_row, dtype=np.int64)
        rows[:, 0] = np.arange(n_rows)
        self.codec_table = self.codec_embedding.run(rows)[0][0].astype(np.float32)
        # special text embeddings and the role prefix, one prompt call
        role = self.encode("<|im_start|>assistant\n")
        assert len(role) == 3, role
        ids = [self.cfg["tts_bos_token_id"], self.cfg["tts_eos_token_id"], self.cfg["tts_pad_token_id"]] + role
        proj = self.prompt.run(np.array([ids], dtype=np.int64))[0]
        self.tts_bos, self.tts_eos, self.tts_pad = proj[:, 0:1], proj[:, 1:2], proj[:, 2:3]
        self.role_embed = proj[:, 3:]

        eos_ids = {tc["codec_eos_token_id"], 2150, 2157, 151670, self.cfg["tts_eos_token_id"],
                   self.cfg["im_end_token_id"], 151643}
        self.eos_ids = eos_ids
        self.suppress = [i for i in range(self.talker_vocab - 1024, self.talker_vocab) if i not in eos_ids]
        self.codec_eos = [i for i in eos_ids if i < self.talker_vocab]
        self.voices = {}
        self.clone_voice = self.voice_clone_prompt(args.ref_audio, args.ref_text) if args.model == "base" else None
        self.bench.reset()

    def _load_onnx(self, env_id):
        prov = onnx_providers(env_id)
        self.providers = prov
        # the text embedding table is 1.3GB and a call is a gather: it stays on the CPU
        self.prompt = OnnxNet(WEIGHT_PATH_PROMPT, ["CPUExecutionProvider"], self.bench, "prompt")
        self.codec_embedding = OnnxNet(WEIGHT_PATH_CODEC_EMBEDDING, ["CPUExecutionProvider"], self.bench, "codec_embedding")
        if args.model == "base":
            self.encoder = OnnxNet(WEIGHT_PATH_ENCODER, prov, self.bench, "encoder")
        cuda = prov[0] != "CPUExecutionProvider"
        # talker: the fixed buffer graph, replayed as a CUDA graph on CUDA
        self.talker = OnnxStaticTalker(
            WEIGHT_PATH_TALKER_STATIC, onnx_providers(env_id, cuda_graph=cuda and "talker" in self.cuda_graph),
            self.bench, self.hidden)
        # code predictor: the fused one call graph, every shape static
        self.code_predictor = OnnxStaticNet(
            WEIGHT_PATH_CODE_PREDICTOR_FRAME,
            onnx_providers(env_id, cuda_graph=cuda and "code_predictor" in self.cuda_graph), self.bench,
            "code_predictor",
            [np.zeros((1, 1, self.hidden), np.float32), np.zeros(1, np.int64),
             np.zeros((self.num_groups - 1, self.group_vocab), np.float32), np.ones(1, np.float32)])
        # decoder: one fixed input shape (see decode_window), so cuDNN plans once.
        # Its own CUDA graph gave wrong output / crashes on onnxruntime 1.18 and is
        # not used.
        self.decoder = OnnxNet(WEIGHT_PATH_DECODER, prov, self.bench, "decoder")

    def _load_ailia(self, memory_mode, env_id):
        self.providers = ["ailia env_id={}".format(env_id)]
        # Two memory modes. The talker step and the code predictor have fixed shapes
        # and run every frame: with the default memory mode ailia 1.7 records their
        # layers as CUDA graphs (DnnSegment, AILIA_ENABLE_DNN_SEGMENT=1) and replays
        # them, 40 -> 14 ms and 69 -> 21 ms a call. The decoder keeps the inter-stage
        # memory optimisation (no segment): recorded as a segment its 80-frame
        # convolutions took seconds per call, with it 50 ms. The one-shot models
        # (prompt, codec embedding, encoder) have varying input lengths and use the
        # same mode.
        static = memory_mode
        varying = ailia.get_memory_mode(reduce_constant=True, ignore_input_with_initializer=True,
                                        reduce_interstage=False, reuse_interstage=True)
        mk = lambda model, weight, name, mode=varying, env=env_id: AiliaNet(model, weight, mode, env, self.bench, name)
        # the two embedding tables are gathers (1 ms): on the CPU they do not take
        # the GPU memory (0.8 GB) that the 1.7B models need
        cpu_env = ailia.get_cpu_environment_id() if hasattr(ailia, "get_cpu_environment_id") else 0
        self.prompt = mk(MODEL_PATH_PROMPT, WEIGHT_PATH_PROMPT, "prompt", env=cpu_env)
        self.codec_embedding = mk(MODEL_PATH_CODEC_EMBEDDING, WEIGHT_PATH_CODEC_EMBEDDING, "codec_embedding", env=cpu_env)
        if args.model == "base":
            self.encoder = mk(MODEL_PATH_ENCODER, WEIGHT_PATH_ENCODER, "encoder")
        tc = self.cfg["talker_config"]
        self.talker = AiliaStaticTalker(MODEL_PATH_TALKER_STATIC, WEIGHT_PATH_TALKER_STATIC, static, env_id,
                                        self.bench, tc["num_key_value_heads"], tc["head_dim"])
        # the fused one call graph (15 steps + Gumbel-max sampling inside), same inputs as on onnxruntime
        self.code_predictor = mk(MODEL_PATH_CODE_PREDICTOR_FRAME, WEIGHT_PATH_CODE_PREDICTOR_FRAME, "code_predictor", static)
        self.decoder = mk(MODEL_PATH_DECODER, WEIGHT_PATH_DECODER, "decoder")

    # ---------------- embeddings ----------------
    def group_embed(self, group, token):
        row = token if group == 0 else self.talker_vocab + (group - 1) * self.group_vocab + token
        return self.codec_table[row]

    def frame_embed(self, tokens):
        return sum(self.group_embed(g, t) for g, t in enumerate(tokens))

    def embed_committed(self, chunks, is_utterance_start=False):
        """Chunks from LiveTextSource carry their own leading space, so each is tokenized as is."""
        parts = []
        for i, c in enumerate(chunks):
            text = c.lstrip() if (is_utterance_start and i == 0) else c
            ids = self.encode(text) if text else []
            if ids:
                parts.append(self.prompt.run(np.array([ids], dtype=np.int64))[0])
        return np.concatenate(parts, axis=1) if parts else None

    # ---------------- voice clone ----------------
    def voice_clone_prompt(self, ref_audio, ref_text):
        """Encode the reference once: its codec frames, speaker embedding and the
        projected reference text, i.e. everything the ICL prompt is built from."""
        import librosa
        wav, sr = librosa.load(ref_audio, sr=SAMPLE_RATE, mono=True)
        codes, spk = self.encoder.run(wav[None, None].astype(np.float32), mel_spectrogram(wav, sr))
        frames = [[int(t) for t in f] for f in codes[0, : self.num_groups].T]   # [T][16]
        speaker = spk.reshape(1, 1, -1).astype(np.float32)
        if not ref_text.strip():
            # x_vector_only_mode of the reference implementation: the speaker embedding
            # takes the place of a CustomVoice speaker, the reference codes are unused
            logger.info("reference: {:.1f} s, speaker embedding only (no transcript)".format(len(wav) / sr))
            return Voice("clone:" + ref_audio, speaker)
        ids = self.encode(f"<|im_start|>assistant\n{ref_text}<|im_end|>\n")
        text = self.prompt.run(np.array([ids[3:-2]], dtype=np.int64))[0]           # [1, n_ref_text, H]
        codec_icl = np.concatenate(
            [self.codec_table[[self.cfg["talker_config"]["codec_bos_id"]]][None],
             np.stack([self.frame_embed(f) for f in frames])[None]], axis=1)       # codec_bos + ref frames
        logger.info("reference: {:.1f} s, {} codec frames, {} text tokens -> the first {} tokens of each "
                    "utterance are needed before the prefill".format(
                        len(wav) / sr, len(frames), text.shape[1], codec_icl.shape[1] - text.shape[1]))
        return Voice("clone:" + ref_audio, speaker, frames, text, codec_icl)

    def speaker_voice(self, speaker, lang):
        """A CustomVoice speaker: a row of the codec embedding table. Returns (voice, language_id override)."""
        tc = self.cfg["talker_config"]
        spk = speaker.lower()
        if spk not in tc["spk_id"]:
            raise ValueError(f"unknown speaker {speaker}, choose from {list(tc['spk_id'])}")
        if spk not in self.voices:
            self.voices[spk] = Voice(spk, self.codec_table[tc["spk_id"][spk]][None, None])
        dialect = tc["spk_is_dialect"].get(spk)
        override = tc["codec_language_id"][dialect] if lang in ("chinese", "auto") and dialect not in (False, None) else None
        return self.voices[spk], override

    # ---------------- code predictor ----------------
    def predict_groups(self, token, past_hidden, temperature):
        """Groups 1..15 of the frame whose group 0 is `token`, and the frame's summed embedding.

        Sampling happens inside the graph with the Gumbel-max trick:
        argmax(top_50(logits / T) + g), g = -log(-log(u)), the same distribution
        torch.multinomial draws from over the top-k softmax with the reference
        defaults (top_k 50, top_p 1.0); top_k is a constant of the graph, since
        the acoustic groups gain nothing from changing it. Greedy: noise 0, T 1."""
        if temperature > 0:
            u = self.rng.uniform(1e-10, 1.0, (self.num_groups - 1, self.group_vocab)).astype(np.float32)
            noise = -np.log(-np.log(np.clip(u, 1e-10, 1.0 - 1e-7)))
            temp = np.array([temperature], np.float32)
        else:
            noise, temp = np.zeros((self.num_groups - 1, self.group_vocab), np.float32), np.ones(1, np.float32)
        groups, frame_embed = self.code_predictor.run(
            past_hidden.astype(np.float32), np.array([token], np.int64), noise, temp)
        return [int(g) for g in groups], frame_embed[0, 0]

    # ---------------- decoder ----------------
    def decode_window(self, codes):
        """codes: list of up to decode_window_frames 16-token frames -> their waveform.

        The decoder always sees a window of exactly decode_window_frames frames:
        a shorter one (the first chunks of an utterance, the final flush) is padded
        on the right with its last frame and the output is cut to the real frames.
        The decoder is causal, so the padding does not change them (max 1% of the
        signal rms in fp16); one input shape means one cuDNN algorithm search and
        one CUDA graph instead of one per window length on both runtimes."""
        n = len(codes)
        W = self.decode_window_frames
        padded = list(codes) + [codes[-1]] * (W - n)
        arr = np.array(padded, dtype=np.int64).T[None]             # [1, 16, W]
        wav = self.decoder.run(arr)[0][0, 0].astype(np.float32)
        if n == W:
            return wav
        return wav[: len(wav) - (W - n) * FRAME_SAMPLES]           # the decoder trims a constant tail

    # ---------------- the live text loop ----------------
    def stream(self, source, speaker, language,
               temperature=0.9, top_k=50, sub_temperature=0.9,
               emit_every_frames=8, decode_window_frames=80, first_chunk_emit_every=0,
               first_chunk_decode_window=48, first_chunk_frames=48,
               wind_down_frames=200, wind_down_multiplier=6.0, max_frames=10000, on_event=None):
        """Yields PCM float32 chunks at SAMPLE_RATE. A numpy port of
        Qwen3-TTS-streaming-input's stream_generate_pcm_live_text:
        prefill once the first text has committed, then one talker step per frame,
        pausing (no forward) whenever the text has been caught up with, suppressing
        EOS until the source is closed, and emitting a chunk every emit_every_frames
        from a window of the last decode_window_frames frames."""
        tc = self.cfg["talker_config"]
        ev = on_event or (lambda *a: None)

        # --- wait for the first committed text ---
        # Base reads text only where it is aligned with the reference codec frames
        # (trailing text alone is ignored), so the positions the reference text
        # leaves free have to be filled with the first tokens of this utterance
        # before the prefill; CustomVoice starts on the first token.
        lang = (language or "auto").lower()
        language_id = None if lang == "auto" else tc["codec_language_id"][lang]
        if self.clone_voice is not None:
            voice = self.clone_voice
        else:
            voice, override = self.speaker_voice(speaker, lang)
            language_id = override if override is not None else language_id
        first, closed = None, False
        while True:
            chunks, closed = source.poll()
            rows = self.embed_committed(chunks, is_utterance_start=first is None)
            if rows is not None:
                first = rows if first is None else np.concatenate([first, rows], axis=1)
            if (first is not None and first.shape[1] >= voice.need_tokens) or closed:
                break
            source.wait()
        if first is None:
            raise ValueError("the text source closed without any text")
        ev("first_commit")
        eos_appended = False

        # --- conditioning: [prefix: role | codec tags, speaker, codec pad | reference] + this utterance ---
        prefix, codec_bos, prefix_key = voice.prefix(self, language_id)
        if not voice.icl:
            # CustomVoice, or the speaker-embedding-only clone: the first text token
            # sits on codec_bos, the rest is trailing
            utterance = first[:, :1] + codec_bos
            trailing = first[:, 1:]
        else:
            # Base (ICL): this utterance's text (+ eos when complete) position by
            # position on the reference frames the reference text left free, padded
            # with tts_pad if shorter; what does not fit is consumed as trailing text
            n_ref = voice.text.shape[1]
            codec_rest = voice.codec_icl[:, n_ref:]
            text = first
            if closed:
                text = np.concatenate([text, self.tts_eos], axis=1)
                eos_appended = True
            n_rest = codec_rest.shape[1]
            if text.shape[1] < n_rest:
                text = np.concatenate([text, np.repeat(self.tts_pad, n_rest - text.shape[1], axis=1)], axis=1)
            utterance = text[:, :n_rest] + codec_rest
            trailing = text[:, n_rest:]
        ref_frames = voice.frames
        utterance = utterance.astype(np.float32)

        def pick(logits, allow_eos):
            sup = self.suppress if allow_eos else self.suppress + self.codec_eos
            return sample_token(logits, self.rng, temperature, top_k, 1.0, sup)

        # --- prefill: the prefix only when the buffer does not hold it already ---
        prefill_len = prefix.shape[1] + utterance.shape[1]
        if prefill_len >= self.talker.buf:
            raise RuntimeError(f"prompt of {prefill_len} positions does not fit the talker's KV buffer ({self.talker.buf})")
        if getattr(self.talker, "prefix_key", None) != prefix_key:
            self.talker.prefill(prefix)
            self.talker.prefix_key = prefix_key
            ev("prefix_prefill")
        logits, last_hidden = self.talker.prefill(utterance, start=prefix.shape[1])
        past_hidden = last_hidden.astype(np.float32)
        position = prefill_len
        generation_step = 0
        token = pick(logits[0, -1], closed and generation_step >= trailing.shape[1])

        codes = list(ref_frames)
        n_ctx = len(codes)
        frames_since_emit, total_emitted = 0, n_ctx
        frames_since_closed = real_frames = 0
        ended = False

        for _ in range(max_frames):
            g = generation_step
            new_chunks, closed = source.poll()
            new_rows = self.embed_committed(new_chunks)
            if closed and not eos_appended:
                new_rows = self.tts_eos if new_rows is None else np.concatenate([new_rows, self.tts_eos], axis=1)
                eos_appended = True
            # positions already passed with padding are backfilled so new text lands at g
            if trailing.shape[1] < g:
                trailing = np.concatenate([trailing, np.repeat(self.tts_pad, g - trailing.shape[1], axis=1)], axis=1)
            if new_rows is not None and new_rows.shape[1] > 0:
                trailing = np.concatenate([trailing, new_rows], axis=1)
            caught_up = trailing.shape[1] <= g

            frames_since_closed = frames_since_closed + 1 if (closed and caught_up) else 0
            if not caught_up:
                real_frames += 1
            if closed and frames_since_closed > max(wind_down_frames, int(wind_down_multiplier * real_frames)):
                ended = True
                break
            if not closed and caught_up:
                ev("pause")             # the talker has no clock: wait for text, then continue
                source.wait()
                continue
            if token in self.eos_ids:
                ended = True
                break

            groups, frame_embed = self.predict_groups(token, past_hidden, sub_temperature)
            frame = [token] + groups
            codes.append(frame)
            text_feedback = trailing[:, g:g + 1] if g < trailing.shape[1] else self.tts_pad
            x = (frame_embed[None, None] + text_feedback).astype(np.float32)
            if position >= self.talker.buf:
                raise RuntimeError(
                    f"utterance longer than the talker's fixed KV buffer ({self.talker.buf} positions); "
                    "export the static talker with a larger --max_seq_len")
            logits, last_hidden = self.talker.step(x, position)
            past_hidden = last_hidden.astype(np.float32)
            position += 1
            generation_step += 1
            token = pick(logits[0, -1], closed and generation_step >= trailing.shape[1])
            ev("frame", len(codes) - n_ctx)

            # --- emit audio ---
            frames_since_emit += 1
            if first_chunk_emit_every > 0 and len(codes) - n_ctx < first_chunk_frames:
                cur_emit, cur_window = first_chunk_emit_every, first_chunk_decode_window
            else:
                cur_emit, cur_window = emit_every_frames, decode_window_frames
            if frames_since_emit < cur_emit:
                continue
            frames_since_emit = 0
            wav = self.decode_window(codes[-cur_window:])
            total_emitted = len(codes)
            yield wav[-FRAME_SAMPLES * cur_emit:]

        # --- flush the frames not emitted yet, decoded with context in front ---
        remaining = len(codes) - total_emitted
        if remaining > 0:
            context = min(total_emitted, decode_window_frames - remaining)
            wav = self.decode_window(codes[total_emitted - context:])
            yield wav[context * FRAME_SAMPLES:]
        if not ended and not closed:
            raise RuntimeError(f"max_frames={max_frames} reached while text was still expected")


# ======================
# Drivers
# ======================
class Timeline:
    def __init__(self):
        self.t0 = time.perf_counter()
        self.ev = {}

    def now(self):
        return time.perf_counter() - self.t0

    def mark(self, k):
        self.ev.setdefault(k, round(self.now(), 3))


class Player:
    """Plays chunks on the audio device (--play) and, either way, keeps a real-time
    clock to count underruns: moments when playback caught up with the audio made
    so far (the "gaps" of the evaluation reports)."""

    def __init__(self, tl, play=False, prebuffer_s=0.0):
        self.tl, self.pre = tl, prebuffer_s
        self.fed = 0.0; self.started = False; self.start = 0.0; self.underruns = 0; self.gap = 0.0
        self.stream = None; self.chunks = []; self.q = None
        if play:
            import sounddevice as sd
            self.stream = sd.OutputStream(samplerate=SAMPLE_RATE, channels=1, dtype="float32")
            self.stream.start()
            # the device write blocks until its buffer has room, so it runs on its
            # own thread: generation must be free to run ahead of playback
            self.q = queue.Queue()
            self.thread = threading.Thread(target=self._play, daemon=True)
            self.thread.start()

    def _play(self):
        while True:
            chunk = self.q.get()
            if chunk is None:
                return
            self.stream.write(np.ascontiguousarray(chunk))

    def feed(self, chunk):
        self.chunks.append(chunk)
        now = self.tl.now()
        if self.started:
            played = now - self.start
            if played > self.fed:
                self.underruns += 1; self.gap += played - self.fed; self.start += played - self.fed
        self.fed += len(chunk) / SAMPLE_RATE
        if not self.started and self.fed >= self.pre:
            self.started = True; self.start = now; self.tl.mark("play_start")
        if self.q is not None:
            self.q.put(chunk)

    def close(self):
        if self.stream is not None:
            self.q.put(None); self.thread.join()      # let the queued audio play out
            self.stream.stop(); self.stream.close()
        return np.concatenate(self.chunks) if self.chunks else np.zeros(0, np.float32)


def new_source(tts):
    if args.commit == "token":
        return LiveTextSource(tokenize=token_offsets_fn(tts.tokenizer), holdback_tokens=3)
    return LiveTextSource()


def run_one(tts, source, tl, label="", player=None):
    """Generate one utterance from `source`, play/collect it, and return (timings, audio).
    With a shared `player` the audio of consecutive utterances is one continuous stream."""
    pl = player or Player(tl, play=args.play, prebuffer_s=args.prebuffer)
    n0, u0, g0 = len(pl.chunks), pl.underruns, pl.gap
    frames = {"n": 0, "pauses": 0}

    def on_event(kind, *a):
        if kind == "first_commit":
            tl.mark("first_commit")
        elif kind == "pause":
            frames["pauses"] += 1
        elif kind == "prefix_prefill":
            tl.mark("prefix_prefill")        # the voice's prompt prefix was (re)filled this utterance
        elif kind == "frame":
            frames["n"] = a[0]
            tl.mark("first_frame")

    err = None
    try:
        for chunk in tts.stream(source, args.speaker, args.language,
                                temperature=args.temperature, top_k=args.top_k,
                                sub_temperature=args.subtalker_temperature,
                                emit_every_frames=args.emit_every, decode_window_frames=args.decode_window,
                                first_chunk_emit_every=args.first_chunk_emit, on_event=on_event):
            tl.mark("first_pcm")
            pl.feed(chunk)
    except Exception as e:  # noqa: BLE001 -- report, keep the session alive
        err = repr(e)
    tl.mark("gen_done")
    if player is None:
        audio = pl.close()
    else:
        audio = np.concatenate(pl.chunks[n0:]) if len(pl.chunks) > n0 else np.zeros(0, np.float32)
    dur = len(audio) / SAMPLE_RATE
    gen = tl.ev.get("gen_done", 0) - tl.ev.get("first_pcm", 0)
    r = dict(label=label, audio_s=round(dur, 2), frames=frames["n"], pauses=frames["pauses"],
             rtf=round(gen / dur, 2) if dur else None, underruns=pl.underruns - u0, gap_s=round(pl.gap - g0, 2),
             ev=tl.ev, error=err)
    # with a shared player the text being echoed owns the console: per utterance lines go to --debug
    (logger.debug if player is not None else logger.info)(json.dumps(r, ensure_ascii=False))
    return r, audio


def llm_tokenizer():
    from huggingface_hub import snapshot_download
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained(snapshot_download("Qwen/Qwen2.5-1.5B-Instruct", local_files_only=True))


def replay_llm(tts, ltok, text, tl, rate):
    """Push `text` as the token deltas a Qwen2.5 LLM would stream, at `rate` tokens/s,
    the first one 0.10 s in (the LLM's time to first token measured earlier)."""
    ids = ltok(text)["input_ids"]
    deltas, prev = [], ""
    for i in range(len(ids)):
        cur = ltok.decode(ids[: i + 1])
        if not cur.endswith("�"):
            deltas.append(cur[len(prev):]); prev = cur
    s = new_source(tts)

    def prod():
        t = 0.10
        for d in deltas:
            dt = t - tl.now()
            if dt > 0:
                time.sleep(dt)
            tl.mark("first_push"); s.push(d); t += 1.0 / rate
        tl.mark("text_done"); s.close()
    threading.Thread(target=prod, daemon=True).start()
    return s


def split_utterances(text, max_chars):
    """Utterances: whole sentences (ending in 。！？ or at a line end) packed up to
    max_chars; a sentence longer than that is cut at 、 or, failing that, at max_chars."""
    sentences = []
    for line in text.split("\n"):
        cur = ""
        for ch in line:
            cur += ch
            if ch in UTTERANCE_END:
                sentences.append(cur); cur = ""
        if cur.strip():
            sentences.append(cur)
    out, cur = [], ""
    for s in sentences:
        while len(s) > max_chars:
            cut = s.rfind("、", 0, max_chars) + 1 or max_chars
            if cur:
                out.append(cur); cur = ""
            out.append(s[:cut]); s = s[cut:]
        if cur and len(cur) + len(s) > max_chars:
            out.append(cur); cur = ""
        cur += s
    if cur.strip():
        out.append(cur)
    return out


def speak_text(tts, text):
    """Feed `text` in pieces at --cps, echo each piece as it goes in, and speak it as
    it arrives: one LiveTextSource per utterance, generated back to back into one
    Player, while the producer thread keeps feeding the following utterances."""
    utterances = split_utterances(text, args.max_chars)
    sources = queue.Queue()
    tl = Timeline()
    pl = Player(tl, play=args.play, prebuffer_s=args.prebuffer)
    echo = (lambda s: None) if args.quiet else (lambda s: (sys.stdout.write(s), sys.stdout.flush()))

    def prod():
        for u in utterances:
            src = new_source(tts); sources.put((src, Timeline()))
            for i in range(0, len(u), args.piece):
                src.push(u[i:i + args.piece]); echo(u[i:i + args.piece]); time.sleep(args.piece / args.cps)
            src.close(); echo("\n")
        sources.put(None)
    threading.Thread(target=prod, daemon=True).start()

    results = []
    while True:
        item = sources.get()
        if item is None:
            break
        src, utl = item
        r, _ = run_one(tts, src, utl, label=f"utterance{len(results)}", player=pl)
        results.append(r)
    audio = pl.close()
    logger.info("spoke {} utterances, {:.1f} s of audio in {:.1f} s, play start {:.2f} s, underruns {} ({:.2f} s)".format(
        len(results), len(audio) / SAMPLE_RATE, tl.now(), tl.ev.get("play_start", 0), pl.underruns, pl.gap))
    return results, audio


def main():
    # model files check and download
    for onnx, prototxt in onnx_list:
        check_and_download_models(onnx, prototxt, REMOTE_PATH)
    for f in external_data_list:
        check_and_download_file(f, REMOTE_PATH)
    os.makedirs(TOKENIZER_DIR, exist_ok=True)
    check_and_download_file(VOCAB_PATH, TOKENIZER_REMOTE_PATH)
    check_and_download_file(MERGES_PATH, TOKENIZER_REMOTE_PATH)

    memory_mode = None
    if not args.onnx:
        # reduce_interstage / reuse_interstage would block the DnnSegment CUDA graphs
        memory_mode = ailia.get_memory_mode(reduce_constant=True, ignore_input_with_initializer=True)

    t = time.perf_counter()
    tts = Qwen3TTSStreaming(memory_mode, args.env_id)
    logger.info("loaded in {:.1f}s: {}".format(time.perf_counter() - t, ", ".join(map(str, tts.providers))))
    logger.info("model: {} {}{}".format(
        parameter_num, args.model,
        f" (voice clone of {args.ref_audio})" if args.model == "base" else f" (speaker {args.speaker})"))
    logger.info("talker: fixed KV buffer of {} positions{}".format(
        tts.talker.buf, " (CUDA graph)" if args.onnx and "talker" in tts.cuda_graph and tts.providers[0] != "CPUExecutionProvider" else ""))

    # warm up (captures the CUDA graphs on the second run of each)
    s = new_source(tts); s.push("こんにちは、テストです。"); s.close()
    for _ in tts.stream(s, args.speaker, args.language):
        pass
    tts.bench.reset()

    results, audios = [], []
    if args.replay:
        texts = json.load(open(args.replay, encoding="utf-8"))
        ltok = llm_tokenizer()
        for i, text in enumerate(texts):
            tl = Timeline()
            r, audio = run_one(tts, replay_llm(tts, ltok, text, tl, args.rate), tl, label=f"replay{i}@{args.rate:g}tok/s")
            results.append(r); audios.append(audio)
    else:
        inputs = args.input if isinstance(args.input, list) else [args.input]
        for inp in inputs:
            # a text file, or the text itself
            text = open(inp, encoding="utf-8").read() if os.path.isfile(inp) else inp
            logger.info(f"speaking {inp} ({len(text)} characters) fed at {args.cps:g} characters/s")
            rs, audio = speak_text(tts, text)
            results += rs; audios.append(audio)

    if args.benchmark:
        tts.bench.report()
    if args.profile and not args.onnx:
        for name in ("prompt", "codec_embedding", "talker", "code_predictor", "decoder"):
            print(name + " : ")
            print(getattr(tts, name).net.get_summary())

    import soundfile as sf
    base, ext = os.path.splitext(args.savepath)
    for i, a in enumerate(audios):
        path = f"{base}_{i}{ext}" if len(audios) > 1 else args.savepath
        sf.write(path, a, SAMPLE_RATE)
        logger.info(f"Saved as {path}.")
    if args.results:
        with open(args.results, "a", encoding="utf-8") as f:
            for r in results:
                r.update(runtime="onnxruntime" if args.onnx else "ailia", parameter_num=parameter_num, model=args.model,
                         commit=args.commit, cuda_graph=args.cuda_graph if args.onnx else None,
                         emit_every=args.emit_every, decode_window=args.decode_window,
                         timing=tts.bench.as_dict())
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
    logger.info('Script finished successfully.')


if __name__ == '__main__':
    main()
