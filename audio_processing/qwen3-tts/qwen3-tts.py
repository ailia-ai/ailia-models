import ailia
import time
import sys
import argparse
import re

import numpy as np
import soundfile as sf
import librosa
import librosa.filters
from scipy.io.wavfile import write, read
from librosa.util import normalize
from librosa.filters import mel as librosa_mel_fn
import os
import json
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, List, Optional


# import original modules
sys.path.append('../../util')
from arg_utils import get_base_parser, update_parser, get_savepath  # noqa: E402
from model_utils import check_and_download_models, check_and_download_file  # noqa: E402

# logger
from logging import getLogger   # noqa: E402
logger = getLogger(__name__)

# ======================
# PARAMETERS
# ======================

OUTPUT_WAV_PATH = 'output.wav'
TEXT_STR = "Good one. Okay, fine, I'm just gonna leave this sock monkey here. Goodbye."
INPUT_WAV_PATH = "clone_2.wav"
INPUT_TEXT_STR = "Okay. Yeah. I resent you. I love you. I respect you. But you know what? You blew it! And thanks to you."
REMOTE_PATH = "https://storage.googleapis.com/ailia-models/qwen3-tts/"


# ======================
# Arguemnt Parser Config
# ======================
parser = get_base_parser(
    'QWEN3 TTS', 
    TEXT_STR, 
    OUTPUT_WAV_PATH,
    large_model = True
)

# 参照音声ファイルのパス
parser.add_argument(
    '--ref_audio', type=str, default=INPUT_WAV_PATH,
    help='Reference audio file path for Voice Clone mode (e.g. clone_2.wav)'
)

# 参照音声のキャプション（書き起こし）テキスト
parser.add_argument(
    '--ref_text', type=str, default=INPUT_TEXT_STR,
    help='Reference text for Voice Clone mode (e.g. Okay. Yeah.)'
)


parser.add_argument(
    '-m', '--model',
    default='base',
    help='[base]'
)
parser.add_argument(
    '-p', '--parameter_num',
    default='0.6B',
    help='[0.6B, 1.7B]'
)
parser.add_argument(
    '--language', type=str, default='Auto',
    help='Target language. "Auto" for auto detection, or one of: '
         'chinese, english, japanese, korean, german, french, russian, '
         'portuguese, spanish, italian'
)
parser.add_argument(
    '--temperature', type=float, default=0.9,
    help='Sampling temperature for the talker. 0 for greedy decoding. (default: 0.9, matches official)'
)
parser.add_argument(
    '--top_k', type=int, default=50,
    help='Top-k sampling for the talker. (default: 50, matches official)'
)
parser.add_argument(
    '--repetition_penalty', type=float, default=1.05,
    help='Repetition penalty for the talker. (default: 1.05)'
)
parser.add_argument(
    '--subtalker_temperature', type=float, default=0.9,
    help='Sampling temperature for the subtalker (code predictor). 0 for greedy. (default: 0.9, matches official)'
)
parser.add_argument(
    '--subtalker_top_k', type=int, default=50,
    help='Top-k sampling for the subtalker (code predictor). (default: 50, matches official)'
)
parser.add_argument(
    '--seed', type=int, default=None,
    help='Random seed for reproducible sampling. (default: None)'
)
parser.add_argument(
    '--disable_ailia_tokenizer', action='store_true', help='disable ailia tokenizer.'
)
parser.add_argument(
    '--onnx', action='store_true', help='execute onnxruntime version.'
)
parser.add_argument(
    '--profile', action='store_true', help='use profile model'
)
args = update_parser(parser, check_input_type=False)

if  not args.model == "base":
    logger.error("unknown model")
    sys.exit()

if args.parameter_num == "0.6B" or args.parameter_num == "1.7B":
    parameter_num = args.parameter_num
else:
    logger.error("invalid parameter_num")
    sys.exit()

CONFIG_PATH = f"config_{parameter_num}.json"


WEIGHT_PATH_SPEAKER_ENCODER  = f"qwen3_tts_speaker_encoder_{parameter_num}.onnx"
WEIGHT_PATH_SPEAKER_ENCODER_DATA  = f"qwen3_tts_speaker_encoder_{parameter_num}.onnx.data"
MODEL_PATH_SPEAKER_ENCODER   = WEIGHT_PATH_SPEAKER_ENCODER + ".prototxt"
WEIGHT_PATH_TALKER_IO        = f"qwen3_tts_talker_io_units_{parameter_num}.onnx"
WEIGHT_PATH_TALKER_IO_DATA   = f"qwen3_tts_talker_io_units_{parameter_num}.onnx.data"
MODEL_PATH_TALKER_IO         =  WEIGHT_PATH_TALKER_IO + ".prototxt"
WEIGHT_PATH_TALKER_DECODER   = f"qwen3_tts_talker_decoder_{parameter_num}.onnx"
WEIGHT_PATH_TALKER_DECODER_DATA = WEIGHT_PATH_TALKER_DECODER + ".data"
MODEL_PATH_TALKER_DECODER    = WEIGHT_PATH_TALKER_DECODER + ".prototxt"
WEIGHT_PATH_TOKENIZER_ENCODER = f"qwen3_tts_tokenizer_encoder_{parameter_num}.onnx"
MODEL_PATH_TOKENIZER_ENCODER  = WEIGHT_PATH_TOKENIZER_ENCODER + ".prototxt"
WEIGHT_PATH_TOKENIZER_DECODER = f"qwen3_tts_tokenizer_decoder_{parameter_num}.onnx"
WEIGHT_PATH_TOKENIZER_DECODER_DATA = f"qwen3_tts_tokenizer_decoder_{parameter_num}.onnx.data"
MODEL_PATH_TOKENIZER_DECODER  = WEIGHT_PATH_TOKENIZER_DECODER + ".prototxt"
WEIGHT_PATH_TEXT_EMB          = f"qwen3_tts_text_embedding_{parameter_num}.npy"
WEIGHT_PATH_CODEC_EMB         = f"qwen3_tts_codec_embeddings_{parameter_num}.npy"
WEIGHT_PATH_SUBTALKER_DECODER   = f"qwen3_tts_subtalker_decoder_{parameter_num}.onnx"
MODEL_PATH_SUBTALKER_DECODER   = WEIGHT_PATH_SUBTALKER_DECODER + ".prototxt"
WEIGHT_PATH_SUBTALKER_LM_HEADS  = f"qwen3_tts_subtalker_lm_heads_{parameter_num}.npy"
WEIGHT_PATH_SUBTALKER_CODEC_EMB = f"qwen3_tts_subtalker_codec_emb_{parameter_num}.npy"
onnx_list = [
    (WEIGHT_PATH_SPEAKER_ENCODER,MODEL_PATH_SPEAKER_ENCODER),
    (WEIGHT_PATH_TALKER_IO,MODEL_PATH_TALKER_IO),
    (WEIGHT_PATH_TALKER_DECODER, MODEL_PATH_TALKER_DECODER),
    (WEIGHT_PATH_TOKENIZER_ENCODER,MODEL_PATH_TOKENIZER_ENCODER),
    (WEIGHT_PATH_TOKENIZER_DECODER,MODEL_PATH_TOKENIZER_DECODER),
    (WEIGHT_PATH_SUBTALKER_DECODER,MODEL_PATH_SUBTALKER_DECODER),
]
file_list = [
    WEIGHT_PATH_TEXT_EMB,
    WEIGHT_PATH_CODEC_EMB,
    WEIGHT_PATH_SUBTALKER_LM_HEADS,
    WEIGHT_PATH_SUBTALKER_CODEC_EMB,
]
# ONNX files whose weights are stored in a separate external data file
if parameter_num == "0.6B":
    file_list += [
        WEIGHT_PATH_SPEAKER_ENCODER_DATA,
        WEIGHT_PATH_TALKER_IO_DATA,
        WEIGHT_PATH_TOKENIZER_DECODER_DATA,
    ]
else:
    # the 1.7B talker exceeds the 2GB protobuf limit
    file_list += [
        WEIGHT_PATH_TOKENIZER_DECODER_DATA,
        WEIGHT_PATH_TALKER_DECODER_DATA,
    ]

# copy_blob_data で KV cache を ailia 内部だけで受け渡すには 1.2.15 以降が必要
version = ailia.get_version().split(".")
AILIA_VERSION_MAJOR = int(version[0])
AILIA_VERSION_MINOR = int(version[1])
AILIA_VERSION_REVISION = int(version[2])
COPY_BLOB_DATA = not (
    AILIA_VERSION_MAJOR <= 1
    and AILIA_VERSION_MINOR <= 2
    and AILIA_VERSION_REVISION < 15
)

#Parameters reqired to create mell spectograms
sampling_rate = 24000
segment_size = 8192
num_mels = 128
num_freq = 1025
n_fft = 1024
hop_size = 256
win_size = 1024

fmin = 0
fmax = 12000
MAX_WAV_VALUE = 32768.0

class Benchmark:
    """-b オプション用の計測。

    talker と subtalker はトークンごとに何度も呼ばれるので、1回ごとに出力する
    かわりにモデルごとの呼び出し回数と合計時間を積んで最後にまとめて出力する。
    """

    def __init__(self, enabled):
        self.enabled = enabled
        self.records = {}

    @contextmanager
    def measure(self, name):
        if not self.enabled:
            yield
            return
        start = int(round(time.time() * 1000))
        yield
        elapsed = int(round(time.time() * 1000)) - start
        calls, total = self.records.get(name, (0, 0))
        self.records[name] = (calls + 1, total + elapsed)

    def report(self):
        if not self.enabled:
            return
        for name, (calls, total) in self.records.items():
            if calls == 1:
                logger.info("\t{} processing time {} ms".format(name, total))
            else:
                logger.info("\t{} processing time {} ms ({} calls, {:.1f} ms/call)".format(
                    name, total, calls, total / calls))


def _sample_token(logits_1d: np.ndarray, temperature: float = 0.9, top_k: int = 50) -> int:
    """temperature + top-k サンプリング。temperature=0 で greedy。"""
    if temperature == 0.0:
        return int(np.argmax(logits_1d))

    logits = logits_1d.astype(np.float64) / temperature
 
    # top-k マスク
    if top_k > 0 and top_k < len(logits):
        top_k_indices = np.argpartition(logits, -top_k)[-top_k:]
        mask = np.full_like(logits, -np.inf)
        mask[top_k_indices] = logits[top_k_indices]
        logits = mask
 
    # softmax (数値安定)
    logits -= np.max(logits)
    probs = np.exp(logits)
    probs /= probs.sum()
 
    return int(np.random.choice(len(probs), p=probs))

def dynamic_range_compression(x, C=1, clip_val=1e-5):
    return np.log(np.clip(x, clip_val, None) * C)

def mel_spectrogram(y, n_fft, num_mels, sr, hop_size, win_size, fmin, fmax, center=False):
    if np.min(y) < -1.:
        print('min value is ', np.min(y))
    if np.max(y) > 1.:
        print('max value is ', np.max(y))

    mel_basis = {}
    hann_window = {}
    if fmax not in mel_basis:
        mel_X = librosa_mel_fn(sr=sr, n_fft=n_fft, n_mels=num_mels, fmin=fmin, fmax=fmax)
       
        mel_basis[str(fmax)] = mel_X
        hann_window[str(1)] = np.hanning(win_size)
    padding = (n_fft - hop_size) // 2
    y = np.pad(y, [(0, 0),(padding, padding)], mode='reflect')
    y_for_stft = np.squeeze(y)


    spec = librosa.stft(y, n_fft=n_fft, hop_length=hop_size, win_length=win_size, window=hann_window["1"],
                       center=False, pad_mode='reflect')
    spec = np.squeeze(spec)

    spec = np.abs(spec)
    
    spec = np.dot(mel_basis[str(fmax)], spec,)

    spec = dynamic_range_compression(spec)
    spec = spec.T # (Time, 128)
    return np.expand_dims(spec, 0).astype(np.float32) # (1, Time, 128)

def load_qwen_config(config_path=CONFIG_PATH):
    with open(config_path, 'r') as f:
        raw = json.load(f)
    tc = raw["talker_config"]
    rs = tc.get("rope_scaling", {})
    return {
        # ---- トップレベル (text tokenizer 側) ----
        "tts_bos_id":         raw["tts_bos_token_id"],    # 151672
        "tts_eos_id":         raw["tts_eos_token_id"],    # 151673
        "tts_pad_id":         raw["tts_pad_token_id"],    # 151671
        "im_start_id":        raw["im_start_token_id"],   # 151644
        "assistant_id":       raw["assistant_token_id"],  # 77091
        # ---- talker_config (codec tokenizer 側) ----
        "codec_bos_id":       tc["codec_bos_id"],         # 2149
        "codec_eos_id":       tc["codec_eos_token_id"],   # 2150 
        "codec_pad_id":       tc["codec_pad_id"],         # 2148
        "codec_nothink_id":   tc["codec_nothink_id"],     # 2155
        "codec_think_id":     tc["codec_think_id"],       # 2154
        "codec_think_bos_id": tc["codec_think_bos_id"],   # 2156
        "codec_think_eos_id": tc["codec_think_eos_id"],   # 2157
        "codec_language_id":  tc.get("codec_language_id", {}),
        # ---- アーキテクチャ ----
        "num_kv_heads":       tc.get("num_key_value_heads", 8),   # 8
        "head_dim":           tc.get("head_dim", 128),            # 128
        "rope_theta":         tc["rope_theta"],                   # 1000000
        "mrope_section":      rs.get("mrope_section", [24, 20, 20]),
        # talker の hidden size (0.6B: 1024, 1.7B: 2048)
        "hidden_size":        tc["hidden_size"],
        # text embedding の次元 (0.6B / 1.7B ともに 2048)
        "text_hidden_size":   tc["text_hidden_size"],
        # subtalker が 1 トークンあたりに走る位置数 (talker hidden + 15 グループ)
        "num_code_groups":    tc["num_code_groups"],
    }
TOKENIZER_DIR = "./tokenizer"

# ailia tokenizer (GPT2Tokenizer) で利用する追加特殊トークン (token id 151643-151656 の順)
TOKENIZER_SPECIAL_TOKENS = [
    '<|endoftext|>', '<|im_start|>', '<|im_end|>',
    '<|object_ref_start|>', '<|object_ref_end|>',
    '<|box_start|>', '<|box_end|>',
    '<|quad_start|>', '<|quad_end|>',
    '<|vision_start|>', '<|vision_end|>',
    '<|vision_pad|>', '<|image_pad|>', '<|video_pad|>',
]


def _ensure_vocab_and_merges():
    """ailia tokenizer 用に vocab.json / merges.txt を tokenizer.json から生成する。"""
    vocab_path = os.path.join(TOKENIZER_DIR, "vocab.json")
    merges_path = os.path.join(TOKENIZER_DIR, "merges.txt")
    if os.path.exists(vocab_path) and os.path.exists(merges_path):
        return

    with open(os.path.join(TOKENIZER_DIR, "tokenizer.json"), "r", encoding="utf-8") as f:
        model = json.load(f)["model"]

    with open(vocab_path, "w", encoding="utf-8") as f:
        json.dump(model["vocab"], f, ensure_ascii=False)

    with open(merges_path, "w", encoding="utf-8") as f:
        f.write("#version: 0.2\n")
        for merge in model["merges"]:
            if isinstance(merge, list):
                merge = " ".join(merge)
            f.write(merge + "\n")


def create_tokenizer():
    if args.disable_ailia_tokenizer:
        from transformers import AutoTokenizer
        return AutoTokenizer.from_pretrained(TOKENIZER_DIR)
    else:
        from ailia_tokenizer import GPT2Tokenizer
        _ensure_vocab_and_merges()
        tokenizer = GPT2Tokenizer.from_pretrained(TOKENIZER_DIR)
        tokenizer.add_special_tokens(
            {"additional_special_tokens": TOKENIZER_SPECIAL_TOKENS}
        )
        return tokenizer


@dataclass
class VoiceClonePromptItem:
    ref_code: Any 
    ref_spk_embedding: Any
    x_vector_only_mode: bool
    icl_mode: bool
    ref_text: str = ""

def _sample_token(logits_1d: np.ndarray, temperature: float = 0.9, top_k: int = 50) -> int:
    """temperature + top-k サンプリング。temperature=0 で greedy。"""
    if temperature == 0.0:
        return int(np.argmax(logits_1d))
    logits = logits_1d.astype(np.float64) / temperature
    if 0 < top_k < len(logits):
        top_k_idx = np.argpartition(logits, -top_k)[-top_k:]
        mask = np.full_like(logits, -np.inf)
        mask[top_k_idx] = logits[top_k_idx]
        logits = mask
    logits -= np.max(logits)
    probs = np.exp(logits)
    probs /= probs.sum()
    return int(np.random.choice(len(probs), p=probs))


class OnnxNet:
    """--onnx 用に onnxruntime を ailia.Net と同じ形で使えるようにするラッパー。"""

    def __init__(self, weight):
        import onnxruntime
        self.session = onnxruntime.InferenceSession(
            weight, providers=["CPUExecutionProvider"]
        )
        self.input_names = [i.name for i in self.session.get_inputs()]
        self.input_types = [i.type for i in self.session.get_inputs()]

    def get_input_blob_list(self):
        return list(range(len(self.input_names)))

    def set_input_blob_shape(self, shape, index):
        # onnxruntime は入力そのものから shape を決めるので何もしない
        pass

    def run(self, inputs):
        if not isinstance(inputs, (list, tuple)):
            inputs = [inputs]
        feed = {}
        for name, dtype, value in zip(self.input_names, self.input_types, inputs):
            value = np.asarray(value)
            if dtype == "tensor(int64)":
                value = value.astype(np.int64)
            elif dtype == "tensor(float)":
                value = value.astype(np.float32)
            feed[name] = value
        return self.session.run(None, feed)


def create_net(model_path, weight_path, memory_mode, env_id):
    if args.onnx:
        return OnnxNet(weight_path)
    return ailia.Net(
        stream=model_path, weight=weight_path, memory_mode=memory_mode, env_id=env_id
    )


class Qwen3TTS:
    def __init__(self, memory_mode, env_id):
        self.cfg = load_qwen_config()
        self.hidden_size = self.cfg["hidden_size"]
        # talker_io は text_projection と codec_head を 1 つの ONNX にまとめて
        # いるため、片方だけ使うときは他方の入力にゼロを渡す
        self.zero_hidden = np.zeros((1, 1, self.hidden_size), dtype=np.float32)
        self.speaker_encoder   = create_net(MODEL_PATH_SPEAKER_ENCODER, WEIGHT_PATH_SPEAKER_ENCODER,   memory_mode, env_id)
        self.talker_io         = create_net(MODEL_PATH_TALKER_IO, WEIGHT_PATH_TALKER_IO,         memory_mode, env_id)
        self.talker_decoder    = create_net(MODEL_PATH_TALKER_DECODER, WEIGHT_PATH_TALKER_DECODER,    memory_mode, env_id)
        self.tokenizer_encoder = create_net(MODEL_PATH_TOKENIZER_ENCODER, WEIGHT_PATH_TOKENIZER_ENCODER, memory_mode, env_id)
        self.tokenizer_decoder = create_net(MODEL_PATH_TOKENIZER_DECODER, WEIGHT_PATH_TOKENIZER_DECODER, memory_mode, env_id)
        self.text_emb_weight   = np.load(WEIGHT_PATH_TEXT_EMB)
        self.codec_emb_weight  = np.load(WEIGHT_PATH_CODEC_EMB)
        self.text_tokenizer    = create_tokenizer()
        self.subtalker_decoder  = create_net(MODEL_PATH_SUBTALKER_DECODER, WEIGHT_PATH_SUBTALKER_DECODER, memory_mode, env_id)
        self.text_emb_weight    = np.load(WEIGHT_PATH_TEXT_EMB)
        self.codec_emb_weight   = np.load(WEIGHT_PATH_CODEC_EMB)
        self.subtalker_lm_heads  = np.load(WEIGHT_PATH_SUBTALKER_LM_HEADS)   # [15, 2048, subtalker hidden]
        self.subtalker_codec_emb = np.load(WEIGHT_PATH_SUBTALKER_CODEC_EMB)  # [15, 2048, talker hidden]
        self.text_tokenizer      = create_tokenizer()
        # KV cache を持つ層数は ONNX の入力数から求める (inputs_embeds,
        # attention_mask, position_ids の 3 入力 + 層ごとに key/value の 2 入力)
        self.NUM_LAYERS     = (len(self.talker_decoder.get_input_blob_list()) - 3) // 2
        self.NUM_SUB_LAYERS = (len(self.subtalker_decoder.get_input_blob_list()) - 3) // 2
        # onnxruntime には blob をコピーする API がないので ailia のときだけ使う
        self.use_copy_blob_data = COPY_BLOB_DATA and not args.onnx
        self.benchmark = Benchmark(args.benchmark)
        self.nets = {
            "speaker_encoder":   self.speaker_encoder,
            "tokenizer_encoder": self.tokenizer_encoder,
            "tokenizer_decoder": self.tokenizer_decoder,
            "talker_io_units":   self.talker_io,
            "talker_decoder":    self.talker_decoder,
            "subtalker_decoder": self.subtalker_decoder,
        }
        if args.profile and not args.onnx:
            for net in self.nets.values():
                net.set_profile_mode(True)
        # subtalker は固定長 cache なので mask も position ごとに固定長で作る
        self.NUM_CODE_GROUPS = self.cfg["num_code_groups"]
        self._sub_attn = [
            self.generate_subtalker_mask(p) for p in range(self.NUM_CODE_GROUPS)
        ]
        self._sub_pos  = [
            np.array([[p]], dtype=np.int64) for p in range(self.NUM_CODE_GROUPS)
        ]

    # ------------------------------------------------------------------
    # generate_attention_mask: 4D causal mask を作る
    #   prefill:  generate_attention_mask(seq_len)
    #   decode:   generate_attention_mask(1, past_len)
    # ------------------------------------------------------------------
    def generate_attention_mask(self, seq_len, past_len=0):
        total_len = seq_len + past_len
        mask = np.full((seq_len, total_len), -np.inf, dtype=np.float32)
        for i in range(seq_len):
            mask[i, : past_len + i + 1] = 0.0
        return mask[np.newaxis, np.newaxis, :, :]   # [1, 1, seq_len, total_len]

    # ------------------------------------------------------------------
    # generate_subtalker_mask: 固定長 cache 用の 4D mask
    #   position までのスロットだけ参照可能にする [1, 1, 1, num_code_groups]
    # ------------------------------------------------------------------
    def generate_subtalker_mask(self, position):
        mask = np.full((1, 1, 1, self.NUM_CODE_GROUPS), -np.inf, dtype=np.float32)
        mask[..., : position + 1] = 0.0
        return mask

    # ------------------------------------------------------------------
    # _run_decoder: talker / subtalker 共通の 1 ステップ実行
    #   inputs_embeds : [1, seq, hidden_size]
    #   attention_mask: [1, 1, seq, total]
    #   position_ids  : [1, seq]  int64
    #   kv_caches     : list of num_layers*2 tensors [1, num_kv_heads, past, head_dim]
    #                   None を渡すと、前ステップの出力ブロブを ailia 内部で
    #                   そのまま次の入力ブロブへコピーして使う (copy_blob_data)。
    #                   KV cache を Python 側に取り出して渡し直す往復が消えるので
    #                   デコードが進んで cache が伸びるほど効果が大きい。
    # returns (last_hidden [1, seq, hidden_size], next kv_caches or None)
    # ------------------------------------------------------------------
    def _run_decoder(self, name, net, num_layers, inputs_embeds, attention_mask, position_ids, kv_caches):
        with self.benchmark.measure(name):
            return self._run_decoder_impl(
                net, num_layers, inputs_embeds, attention_mask, position_ids, kv_caches
            )

    def _run_decoder_impl(self, net, num_layers, inputs_embeds, attention_mask, position_ids, kv_caches):
        if kv_caches is not None:
            net.set_input_blob_shape(inputs_embeds.shape,   0)
            net.set_input_blob_shape(attention_mask.shape,  1)
            net.set_input_blob_shape(position_ids.shape,    2)
            for i in range(num_layers * 2):
                net.set_input_blob_shape(kv_caches[i].shape, 3 + i)

            outputs = net.run(
                [inputs_embeds, attention_mask, position_ids] + kv_caches
            )
            last_hidden = outputs[0]                    # [1, seq, hidden_size]
            if self.use_copy_blob_data:
                # 次のステップからは ailia 内部の出力ブロブをコピーする
                return last_hidden, None
            return last_hidden, [outputs[1 + i] for i in range(num_layers * 2)]

        input_blobs  = net.get_input_blob_list()
        output_blobs = net.get_output_blob_list()
        # 入力の shape を変えると ailia が全体の形状を再推論し、コピー元にする
        # 出力ブロブの shape も変わってしまうので、先に全部読み出しておく
        kv_shapes = [
            net.get_blob_shape(output_blobs[1 + i]) for i in range(num_layers * 2)
        ]
        for index, value in enumerate([inputs_embeds, attention_mask, position_ids]):
            net.set_input_blob_data(value, input_blobs[index])
        for i in range(num_layers * 2):
            dst = input_blobs[3 + i]
            # subtalker は固定長 cache なので shape は毎回同じ。設定し直すと
            # ailia が形状を再推論してしまうので、変わったときだけ呼ぶ
            if net.get_blob_shape(dst) != kv_shapes[i]:
                net.set_input_blob_shape(kv_shapes[i], dst)
            net.copy_blob_data(dst, output_blobs[1 + i], net)
        net.update()
        return net.get_blob_data(output_blobs[0]), None

    # ------------------------------------------------------------------
    # _run_talker_io: text_projection と codec_head をまとめた ONNX を呼ぶ
    #   使わない側の入力にはゼロを渡す (self.zero_hidden / dummy text)
    # returns (projected_text, logits)
    # ------------------------------------------------------------------
    def _run_talker_io(self, text_features, last_hidden_states):
        with self.benchmark.measure("talker_io_units"):
            self.talker_io.set_input_blob_shape(text_features.shape, 0)
            self.talker_io.set_input_blob_shape(last_hidden_states.shape, 1)
            return self.talker_io.run([text_features, last_hidden_states])

    def _run_talker_decoder(self, inputs_embeds, attention_mask, position_ids, kv_caches):
        return self._run_decoder(
            "talker_decoder", self.talker_decoder, self.NUM_LAYERS,
            inputs_embeds, attention_mask, position_ids, kv_caches
        )

    def _run_subtalker_decoder(self, inputs_embeds, attention_mask, position_ids, kv_caches):
        return self._run_decoder(
            "subtalker_decoder", self.subtalker_decoder, self.NUM_SUB_LAYERS,
            inputs_embeds, attention_mask, position_ids, kv_caches
        )


    # ------------------------------------------------------------------
    # _predict_subgroups: グループ 1〜15 を subtalker で順に予測する
    #   subtalker の ONNX は num_code_groups スロットの固定長 cache を持ち、
    #   position が指すスロットだけを書き換える。全ステップが seq=1 の同じ
    #   shape になるので、ailia の形状再推論が起きない。
    #     position 0        : talker の hidden
    #     position k+1      : グループ k の埋め込み → グループ k+1 を予測
    # ------------------------------------------------------------------
    def _predict_subgroups(self, group0_token: int, past_hidden: np.ndarray,
                           temperature: float = 0.9, top_k: int = 50) -> list:
        NSL  = self.NUM_SUB_LAYERS
        NKV  = self.cfg["num_kv_heads"]
        HDIM = self.cfg["head_dim"]
        sub_kv = [
            np.zeros((1, NKV, self.NUM_CODE_GROUPS, HDIM), dtype=np.float32)
            for _ in range(NSL * 2)
        ]

        # position 0: talker の hidden を cache に入れる (出力は使わない)
        _, sub_kv = self._run_subtalker_decoder(
            past_hidden.astype(np.float32), self._sub_attn[0], self._sub_pos[0], sub_kv
        )

        group_tokens = []
        for k in range(self.NUM_CODE_GROUPS - 1):
            if k == 0:
                emb = self.codec_emb_weight[0, group0_token, :]
            else:
                emb = self.subtalker_codec_emb[k - 1, group_tokens[k - 1], :]
            emb = emb[np.newaxis, np.newaxis, :].astype(np.float32)

            hidden, sub_kv = self._run_subtalker_decoder(
                emb, self._sub_attn[k + 1], self._sub_pos[k + 1], sub_kv
            )
            group_tokens.append(_sample_token(
                self.subtalker_lm_heads[k] @ hidden[0, -1, :], temperature, top_k
            ))

        return group_tokens

    # ------------------------------------------------------------------
    # create_voice_clone_prompt 
    # ------------------------------------------------------------------
    def create_voice_clone_prompt(self, ref_audio, ref_text=None, x_vector_only_mode=False):
        wav, sr = librosa.load(ref_audio, sr=24000, mono=True)
        wav_input = wav[np.newaxis, np.newaxis, :].astype(np.float32)

        with self.benchmark.measure("tokenizer_encoder"):
            self.tokenizer_encoder.set_input_blob_shape(wav_input.shape, 0)
            ref_code = self.tokenizer_encoder.run([wav_input])[0]
        ref_code = np.squeeze(ref_code, axis=0)  # [16, T]

        mel     = mel_spectrogram(wav[np.newaxis, :], n_fft, num_mels, 24000, hop_size, win_size, fmin, fmax)
        with self.benchmark.measure("speaker_encoder"):
            spk_emb = self.speaker_encoder.run([mel])[0]
        spk_emb = np.squeeze(spk_emb, axis=0)  # [hidden_size]

        return [VoiceClonePromptItem(
            ref_code          = None if x_vector_only_mode else ref_code,
            ref_spk_embedding = spk_emb,
            x_vector_only_mode= bool(x_vector_only_mode),
            icl_mode          = bool(not x_vector_only_mode),
            ref_text          = ref_text,
        )]

    # ------------------------------------------------------------------
    # generate_icl_prompt 
    # ------------------------------------------------------------------
    def generate_icl_prompt(self, text_id, ref_id, ref_code, tts_pad_embed, tts_eos_embed):
        combined_text_ids = np.concatenate([ref_id, text_id], axis=1)
        text_emb = self.text_emb_weight[combined_text_ids].astype(np.float32)
 
        text_proj = self._run_talker_io(text_emb, self.zero_hidden)[0]
        text_proj = np.concatenate([text_proj, tts_eos_embed], axis=1)   # [1, text_lens, hidden_size]
 
        ref_T = ref_code.shape[1]
        codec_sum_emb = np.zeros((ref_T, self.hidden_size), dtype=np.float32)
        # ★ Group 0: main talker embedding (codec_emb_weight[0])
        # ★ Groups 1-15: subtalker embedding (subtalker_codec_emb[i-1])
        #   PyTorch: i==0 → talker.get_input_embeddings()
        #            i>0  → code_predictor.get_input_embeddings()[i-1]
        codec_sum_emb += self.codec_emb_weight[0, ref_code[0, :], :]
        for i in range(1, 16):
            codec_sum_emb += self.subtalker_codec_emb[i - 1, ref_code[i, :], :]
        codec_sum_emb = np.expand_dims(codec_sum_emb, axis=0)
 
        bos_id          = self.cfg["codec_bos_id"]
        codec_bos_emb   = self.codec_emb_weight[0][[bos_id]]
        codec_bos_emb   = np.expand_dims(codec_bos_emb, axis=0)
        codec_final_emb = np.concatenate([codec_bos_emb, codec_sum_emb], axis=1)  # [1, codec_lens, hidden_size]
 
        text_lens  = text_proj.shape[1]
        codec_lens = codec_final_emb.shape[1]
 
 
        if codec_lens >= text_lens:
            # 通常ケース: codec の方が長い → テキストをパディング
            num_pads  = codec_lens - text_lens
            if num_pads > 0:
                pads      = np.tile(tts_pad_embed, (1, num_pads, 1))
                text_proj = np.concatenate([text_proj, pads], axis=1)
            icl_embed            = text_proj + codec_final_emb   # [1, codec_lens, hidden_size]
            trailing_text_hidden = tts_pad_embed                  # [1, 1, hidden_size] → 全ステップ tts_pad_embed
        else:
            # 稀ケース: text の方が長い → codec 長でカット、余りをtrailing に
            icl_embed            = text_proj[:, :codec_lens, :] + codec_final_emb
            trailing_text_hidden = text_proj[:, codec_lens:, :]   # オーバーフロー分
 
 
        return icl_embed, trailing_text_hidden

    # ------------------------------------------------------------------
    # predict
    # ------------------------------------------------------------------
    def predict(self, text, ref_audio=None, ref_text=None, language="Auto",
                temperature=0.9, top_k=50, repetition_penalty=1.05,
                subtalker_temperature=0.9, subtalker_top_k=50):
        cfg = self.cfg   # 短縮エイリアス

        # ---- 言語ID 解決 ----
        #   "Auto" → language_id=None (自動判定: nothink パス)
        #   明示指定 → codec_language_id から取得 (think パス)
        if language is None or language.lower() == "auto":
            language_id = None
        else:
            if language.lower() not in cfg["codec_language_id"]:
                logger.error(
                    f"Unsupported language: {language}. "
                    f"Supported: Auto, {', '.join(cfg['codec_language_id'].keys())}"
                )
                sys.exit()
            language_id = cfg["codec_language_id"][language.lower()]

        # ---- 参照音声 ----
        #   speaker embedding / ref_code は create_voice_clone_prompt() で
        #   一括して計算する (ここで二重に encode しない)。

        if ref_text is not None:
            ref_text_formatted = f"<|im_start|>assistant\n{ref_text}<|im_end|>\n"
            ref_ids = np.array(self.text_tokenizer.encode(ref_text_formatted),
                               dtype=np.int64)[np.newaxis, :]

        # ---- テキスト投影 ----
        prompt_text = "<|im_start|>assistant\n"
        body_text   = f"{text}<|im_end|>\n<|im_start|>assistant\n"
        all_text_ids = self.text_tokenizer.encode(prompt_text) + self.text_tokenizer.encode(body_text)
        all_text_emb = np.expand_dims(self.text_emb_weight[all_text_ids], axis=0).astype(np.float32)

        projected_text = self._run_talker_io(all_text_emb, self.zero_hidden)[0]

        # ---- voice clone prompt ----
        voice_clone_prompt = self.create_voice_clone_prompt(ref_audio, ref_text)[0]
        spk_emb = voice_clone_prompt.ref_spk_embedding

        # ---- special embed (tts bos/eos/pad) ----
        special_ids  = [cfg["tts_bos_id"], cfg["tts_eos_id"], cfg["tts_pad_id"]]   # ★ config
        special_embs = self.text_emb_weight[special_ids][np.newaxis, :, :].astype(np.float32)
        projected_specials = self._run_talker_io(special_embs, self.zero_hidden)[0]
        tts_bos_embed = projected_specials[:, 0:1, :]
        tts_eos_embed = projected_specials[:, 1:2, :]
        tts_pad_embed = projected_specials[:, 2:3, :]

        # ---- codec tag 埋め込み ----
        #   Auto (language_id=None): [nothink, think_bos, think_eos]            (3 tokens)
        #   明示指定:                [think,   think_bos, language_id, think_eos] (4 tokens)
        if language_id is None:
            tag_ids_0 = [cfg["codec_nothink_id"],    # 2155
                         cfg["codec_think_bos_id"],  # 2156
                         cfg["codec_think_eos_id"]]  # 2157
        else:
            tag_ids_0 = [cfg["codec_think_id"],      # 2154
                         cfg["codec_think_bos_id"],  # 2156
                         language_id,                # e.g. japanese=2058
                         cfg["codec_think_eos_id"]]  # 2157
        tag_ids_1 = [cfg["codec_pad_id"],        # 2148
                     cfg["codec_bos_id"]]         # 2149
        tag_part0   = self.codec_emb_weight[0][tag_ids_0]          # [3 or 4, hidden_size]
        tag_part_spk = spk_emb[np.newaxis, :]                      # [1, hidden_size]
        tag_part1   = self.codec_emb_weight[0][tag_ids_1]          # [2, hidden_size]
        codec_tag_emb = np.expand_dims(
            np.concatenate([tag_part0, tag_part_spk, tag_part1], axis=0), axis=0
        )  # [1, 6 or 7, hidden_size]

        # ---- role 投影 ----
        role_ids  = [cfg["im_start_id"], cfg["assistant_id"], 198]  # 198=\n
        role_proj = self._run_talker_io(
            self.text_emb_weight[role_ids][np.newaxis, :].astype(np.float32),
            self.zero_hidden
        )[0]  # [1, 3, hidden_size]

        # ---- tag 投影 ----
        #   tts_pad * (codec_tag_len - 2) + tts_bos  → codec_tag_emb[:, :-1] と同じ長さ
        num_tts_pad   = codec_tag_emb.shape[1] - 2
        tag_base_ids  = [cfg["tts_pad_id"]] * num_tts_pad + [cfg["tts_bos_id"]]
        tag_base_proj = self._run_talker_io(
            self.text_emb_weight[tag_base_ids][np.newaxis, :].astype(np.float32),
            self.zero_hidden
        )[0]  # [1, 5, hidden_size]
        tags_combined = tag_base_proj + codec_tag_emb[:, :-1, :]  # [1, 5, hidden_size]

        talker_input_embed = np.concatenate([role_proj, tags_combined], axis=1)  # [1, 8, hidden_size]

        # ---- ICL prompt ----
        all_text_ids_np = np.array(all_text_ids, dtype=np.int64)[np.newaxis, :]
        icl_input_embed, trailing_text_hidden = self.generate_icl_prompt(
            text_id      = all_text_ids_np[:, 3:-5],
            ref_id       = ref_ids[:, 3:-2],
            ref_code     = voice_clone_prompt.ref_code,
            tts_pad_embed= tts_pad_embed,
            tts_eos_embed= tts_eos_embed,
        )
        talker_input_embed = np.concatenate([talker_input_embed, icl_input_embed], axis=1)

        # ==============================================================
        # PREFILL
        # ==============================================================
        NL          = self.NUM_LAYERS
        NKV         = cfg["num_kv_heads"]   # 8
        HDIM        = cfg["head_dim"]        # 128
        prefill_len = talker_input_embed.shape[1]

        attn_mask   = self.generate_attention_mask(prefill_len)          # [1,1,prefill,prefill]
        position_ids = np.arange(prefill_len, dtype=np.int64)[np.newaxis, :]  # [1, prefill]
        kv_caches   = [np.zeros((1, NKV, 0, HDIM), dtype=np.float32) for _ in range(NL * 2)]

        finite_vals = attn_mask[np.isfinite(attn_mask)]


        last_hidden, kv_caches = self._run_talker_decoder(
            talker_input_embed.astype(np.float32), attn_mask, position_ids, kv_caches
        )

        # ==============================================================
        # 最初のトークン予測 (prefill の最終 hidden から)
        # ==============================================================
        dummy_text = np.zeros((1, 1, self.cfg["text_hidden_size"]), dtype=np.float32)
        _, logits = self._run_talker_io(dummy_text, last_hidden)


        # テキストフィードバック用の pad embed
        pad_token_id  = self.text_tokenizer.encode("<|endoftext|>")[0]
        pad_emb_2048  = self.text_emb_weight[[pad_token_id]][np.newaxis, :, :].astype(np.float32)

        EOS_TOKEN_ID  = cfg["codec_eos_id"]   # 2150
        MAX_NEW_TOKENS = 2048
        rep_penalty    = repetition_penalty

        curr_token_id = _sample_token(logits[0, -1, :], temperature, top_k)
        past_hidden = last_hidden[:, -1:, :].astype(np.float32)

        # ==============================================================
        # DECODE ループ
        # ==============================================================
        generated_ids  = []
        all_codes_list = []  
        token_counts   = {}
 
        for step in range(MAX_NEW_TOKENS):
            if curr_token_id == EOS_TOKEN_ID:
                print(f"[DECODE] EOS at step {step}")
                break
 
            # ── subtalker でグループ 1〜15 を予測 ──────────────────
            sub_tokens     = self._predict_subgroups(
                curr_token_id, past_hidden,
                temperature=subtalker_temperature, top_k=subtalker_top_k,
            )
            all_group_tokens = [curr_token_id] + sub_tokens   # len=16
 
            generated_ids.append(curr_token_id)
            all_codes_list.append(all_group_tokens)
            token_counts[curr_token_id] = token_counts.get(curr_token_id, 0) + 1
 
            # ── 16グループの codec 埋め込みを合算 ──────────────────
            codec_sum_emb = np.zeros((1, 1, self.hidden_size), dtype=np.float32)
            codec_sum_emb += self.codec_emb_weight[0, all_group_tokens[0], :][np.newaxis, np.newaxis, :]
            for g in range(1, 16):
                codec_sum_emb += self.subtalker_codec_emb[g - 1, all_group_tokens[g], :][np.newaxis, np.newaxis, :]

 
            # ── テキストフィードバック ──────────────────────────────
            text_idx      = step
            if step < trailing_text_hidden.shape[1]:
                text_feedback = trailing_text_hidden[:, step:step+1, :]
            else:
                text_feedback = tts_pad_embed
 
            current_input = (codec_sum_emb + text_feedback).astype(np.float32)  # [1, 1, hidden_size]
 
            # ── main talker decode ─────────────────────────────────
            decode_pos  = prefill_len + step
            attn_mask_d = self.generate_attention_mask(1, decode_pos)
            pos_ids_d   = np.array([[decode_pos]], dtype=np.int64)
 
            current_hidden, kv_caches = self._run_talker_decoder(
                current_input, attn_mask_d, pos_ids_d, kv_caches
            )
 
            # 次ステップの subtalker 用 past_hidden
            past_hidden = current_hidden.astype(np.float32)   # [1, 1, hidden_size]
 
            # ── 次の group0 トークン予測 ───────────────────────────
            _, logits = self._run_talker_io(dummy_text, current_hidden)
 
            logits_1d = logits[0, -1, :].copy()
            for tid, cnt in token_counts.items():   # repetition penalty
                logits_1d[tid] = (logits_1d[tid] / rep_penalty
                                  if logits_1d[tid] > 0
                                  else logits_1d[tid] * rep_penalty)
 
            curr_token_id = _sample_token(logits_1d, temperature, top_k)
  
        # ==============================================================
        # AUDIO 生成: tokenizer_decoder で波形に変換
        # ==============================================================
        T = len(generated_ids)
        if T == 0:
            return np.zeros(1, dtype=np.float32)
 
        # [1, 16, T]: 全16グループに値を格納
        all_codes = np.zeros((1, 16, T), dtype=np.int64)
        for t, group_toks in enumerate(all_codes_list):
            for g in range(16):
                all_codes[0, g, t] = group_toks[g]

        # ── 公式準拠: 参照codecを前に連結してdecodeし、先頭を比率カット ──
        #   (model.py generate_voice_clone L612-631)
        #   decoder は causal なので、参照フレームを前置きして文脈を与え、
        #   生成した波形の先頭(参照分)を比率で切り落とす。
        ref_code = voice_clone_prompt.ref_code   # [>=16, T_ref] or None
        if ref_code is not None:
            # 生成コード(all_codes)は16コードブック。ICLプロンプトと同様に
            # 先頭16行を使う (encoder出力は16より多い場合がある)。
            ref_code16 = ref_code[:16, :]
            ref_T = ref_code16.shape[1]
            codes_for_decode = np.concatenate(
                [ref_code16[np.newaxis, :, :].astype(np.int64), all_codes], axis=2
            )  # [1, 16, T_ref + T]
        else:
            ref_T = 0
            codes_for_decode = all_codes

        with self.benchmark.measure("tokenizer_decoder"):
            self.tokenizer_decoder.set_input_blob_shape(codes_for_decode.shape, 0)
            wav = np.squeeze(self.tokenizer_decoder.run([codes_for_decode])[0])  # [L]

        if ref_T > 0:
            total_T = codes_for_decode.shape[2]
            cut = int(ref_T / max(total_T, 1) * wav.shape[0])
            wav = wav[cut:]

        return wav
    
def main():
    for onnx, prototxt in onnx_list:
        check_and_download_models(onnx, prototxt, REMOTE_PATH)
    for f in file_list:
        check_and_download_file(f, REMOTE_PATH) 
    memory_mode = None
    if not args.onnx:
        memory_mode = ailia.get_memory_mode(
        reduce_constant=True,
        ignore_input_with_initializer=True,
        reduce_interstage=False,
        reuse_interstage=True
        )
    # 0. 乱数シード (サンプリングの再現性用)
    if args.seed is not None:
        np.random.seed(args.seed)

    # 1. セットアップ
    tts_engine = Qwen3TTS(memory_mode, args.env_id)

    # 2. 検証用データの指定
    wav_text = args.ref_text

    # 3. 推論実行
    print("Generating speech...")
    if args.benchmark:
        start = int(round(time.time() * 1000))
    waveform = tts_engine.predict(
        args.input, ref_audio=args.ref_audio, ref_text=wav_text,
        language=args.language,
        temperature=args.temperature, top_k=args.top_k,
        repetition_penalty=args.repetition_penalty,
        subtalker_temperature=args.subtalker_temperature,
        subtalker_top_k=args.subtalker_top_k,
    )
    if args.benchmark:
        end = int(round(time.time() * 1000))
        tts_engine.benchmark.report()
        logger.info("\ttotal processing time {} ms".format(end - start))

    if args.profile and not args.onnx:
        for name, net in tts_engine.nets.items():
            print(name + " : ")
            print(net.get_summary())

    # 4. 保存
    sf.write(args.savepath, waveform.squeeze(), 24000)
    print(f"Saved as {args.savepath}.")


if __name__ == "__main__":
    main()
