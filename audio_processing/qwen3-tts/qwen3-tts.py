import os
import ailia
import time
import sys

import numpy as np
import soundfile as sf
import librosa
from librosa.filters import mel as librosa_mel_fn
import json
from contextlib import contextmanager


# import original modules
sys.path.append('../../util')
from arg_utils import get_base_parser, update_parser  # noqa: E402
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
    help='Sampling temperature for the code predictor. 0 for greedy. (default: 0.9, matches official)'
)
parser.add_argument(
    '--subtalker_top_k', type=int, default=50,
    help='Top-k sampling for the code predictor. (default: 50, matches official)'
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

# ONNX は 6 つで、weight はすべてどれかのグラフに入っている。サンプル側は配列の
# 組み立てとトークンのサンプリングだけを行う。
#   encoder           参照音声 -> codec トークン + speaker embedding
#   prompt            テキストのトークン ID -> talker の hidden への投影
#   codec_embedding   codec 埋め込みテーブル (16 グループ分をまとめて 1 つ)
#   talker            自己回帰本体 (出力ヘッド入り)
#   code_predictor    グループ 1〜15 の予測 (15 個の出力ヘッド入り)
#   tokenizer_decoder codec トークン -> 波形
WEIGHT_PATH_ENCODER           = f"qwen3_tts_encoder_{parameter_num}.onnx"
MODEL_PATH_ENCODER            = WEIGHT_PATH_ENCODER + ".prototxt"
WEIGHT_PATH_PROMPT            = f"qwen3_tts_prompt_{parameter_num}.onnx"
MODEL_PATH_PROMPT             = WEIGHT_PATH_PROMPT + ".prototxt"
WEIGHT_PATH_CODEC_EMBEDDING   = f"qwen3_tts_codec_embedding_{parameter_num}.onnx"
MODEL_PATH_CODEC_EMBEDDING    = WEIGHT_PATH_CODEC_EMBEDDING + ".prototxt"
WEIGHT_PATH_TALKER            = f"qwen3_tts_talker_{parameter_num}.onnx"
MODEL_PATH_TALKER             = WEIGHT_PATH_TALKER + ".prototxt"
WEIGHT_PATH_CODE_PREDICTOR    = f"qwen3_tts_code_predictor_{parameter_num}.onnx"
MODEL_PATH_CODE_PREDICTOR     = WEIGHT_PATH_CODE_PREDICTOR + ".prototxt"
WEIGHT_PATH_TOKENIZER_DECODER = f"qwen3_tts_tokenizer_decoder_{parameter_num}.onnx"
MODEL_PATH_TOKENIZER_DECODER  = WEIGHT_PATH_TOKENIZER_DECODER + ".prototxt"

onnx_list = [
    (WEIGHT_PATH_ENCODER, MODEL_PATH_ENCODER),
    (WEIGHT_PATH_PROMPT, MODEL_PATH_PROMPT),
    (WEIGHT_PATH_CODEC_EMBEDDING, MODEL_PATH_CODEC_EMBEDDING),
    (WEIGHT_PATH_TALKER, MODEL_PATH_TALKER),
    (WEIGHT_PATH_CODE_PREDICTOR, MODEL_PATH_CODE_PREDICTOR),
    (WEIGHT_PATH_TOKENIZER_DECODER, MODEL_PATH_TOKENIZER_DECODER),
]

# 2GB の protobuf 制限に収まらないモデルは weight を .onnx.data に分けている
EXTERNAL_DATA = {
    "0.6B": [WEIGHT_PATH_PROMPT, WEIGHT_PATH_TOKENIZER_DECODER],
    "1.7B": [WEIGHT_PATH_PROMPT, WEIGHT_PATH_TALKER, WEIGHT_PATH_TOKENIZER_DECODER],
}
file_list = [name + ".data" for name in EXTERNAL_DATA[parameter_num]]

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

# speaker encoder に渡す mel spectrogram のパラメータ
num_mels = 128
n_fft = 1024
hop_size = 256
win_size = 1024
fmin = 0
fmax = 12000


class Benchmark:
    """-b オプション用の計測。

    talker と code predictor はトークンごとに何度も呼ばれるので、1回ごとに出力
    するかわりにモデルごとの呼び出し回数と合計時間を積んで最後にまとめて出力する。
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
    if 0 < top_k < len(logits):
        top_k_idx = np.argpartition(logits, -top_k)[-top_k:]
        mask = np.full_like(logits, -np.inf)
        mask[top_k_idx] = logits[top_k_idx]
        logits = mask
    logits -= np.max(logits)
    probs = np.exp(logits)
    probs /= probs.sum()
    return int(np.random.choice(len(probs), p=probs))


def dynamic_range_compression(x, C=1, clip_val=1e-5):
    return np.log(np.clip(x, clip_val, None) * C)

def mel_spectrogram(y, n_fft, num_mels, sr, hop_size, win_size, fmin, fmax):
    if np.min(y) < -1.:
        logger.warning('min value is {}'.format(np.min(y)))
    if np.max(y) > 1.:
        logger.warning('max value is {}'.format(np.max(y)))

    mel_basis = librosa_mel_fn(sr=sr, n_fft=n_fft, n_mels=num_mels, fmin=fmin, fmax=fmax)
    padding = (n_fft - hop_size) // 2
    y = np.pad(y, [(0, 0), (padding, padding)], mode='reflect')

    spec = np.abs(np.squeeze(librosa.stft(
        y, n_fft=n_fft, hop_length=hop_size, win_length=win_size,
        window=np.hanning(win_size), center=False, pad_mode='reflect')))
    spec = dynamic_range_compression(np.dot(mel_basis, spec)).T   # (Time, 128)
    return np.expand_dims(spec, 0).astype(np.float32)             # (1, Time, 128)

def load_qwen_config(config_path=CONFIG_PATH):
    with open(config_path, 'r') as f:
        raw = json.load(f)
    tc = raw["talker_config"]
    cp = tc["code_predictor_config"]
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
        "num_code_groups":    tc["num_code_groups"],               # 16
        "codec_vocab_size":   tc["vocab_size"],                    # 3072
        "group_vocab_size":   cp["vocab_size"],                    # 2048
        "num_kv_heads":       tc.get("num_key_value_heads", 8),    # 8
        "head_dim":           tc.get("head_dim", 128),             # 128
        "sub_num_kv_heads":   cp.get("num_key_value_heads", 8),    # 8
        "sub_head_dim":       cp.get("head_dim", 128),             # 128
        # talker の hidden size (0.6B: 1024, 1.7B: 2048)
        "hidden_size":        tc["hidden_size"],
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
        self.num_code_groups = self.cfg["num_code_groups"]
        # codec_embedding の入力は 1 つにまとまったテーブルの行番号。先頭 3072 行
        # が talker のグループ 0 用、そのあと 2048 行ずつがグループ 1〜15 用、
        # 最後の 1 行がゼロ (足すものが無いグループ用)。
        self.talker_vocab_size = self.cfg["codec_vocab_size"]
        self.group_vocab_size = self.cfg["group_vocab_size"]
        self.zero_row = (self.talker_vocab_size
                         + (self.num_code_groups - 1) * self.group_vocab_size)
        # code predictor のヘッドは 15 個をつないだ 1 つの行列になっているので、
        # ステップごとに使う 2048 行を渡す
        self.head_rows = [
            np.arange(k * self.group_vocab_size, (k + 1) * self.group_vocab_size,
                      dtype=np.int64)
            for k in range(self.num_code_groups - 1)
        ]
        self.encoder           = create_net(MODEL_PATH_ENCODER, WEIGHT_PATH_ENCODER, memory_mode, env_id)
        self.prompt            = create_net(MODEL_PATH_PROMPT, WEIGHT_PATH_PROMPT, memory_mode, env_id)
        self.codec_embedding   = create_net(MODEL_PATH_CODEC_EMBEDDING, WEIGHT_PATH_CODEC_EMBEDDING, memory_mode, env_id)
        self.talker            = create_net(MODEL_PATH_TALKER, WEIGHT_PATH_TALKER, memory_mode, env_id)
        self.code_predictor    = create_net(MODEL_PATH_CODE_PREDICTOR, WEIGHT_PATH_CODE_PREDICTOR, memory_mode, env_id)
        self.tokenizer_decoder = create_net(MODEL_PATH_TOKENIZER_DECODER, WEIGHT_PATH_TOKENIZER_DECODER, memory_mode, env_id)
        self.text_tokenizer    = create_tokenizer()
        # KV cache を持つ層数は ONNX の入力数から求める。KV cache 以外の入力は
        # talker が 3 個 (inputs_embeds, attention_mask, position_ids)、
        # code_predictor が 4 個 (head_rows が加わる)。
        self.NUM_LAYERS     = (len(self.talker.get_input_blob_list()) - 3) // 2
        self.NUM_SUB_LAYERS = (len(self.code_predictor.get_input_blob_list()) - 4) // 2
        # onnxruntime には blob をコピーする API がないので ailia のときだけ使う
        self.use_copy_blob_data = COPY_BLOB_DATA and not args.onnx
        self.benchmark = Benchmark(args.benchmark)
        self.nets = {
            "encoder":           self.encoder,
            "prompt":            self.prompt,
            "codec_embedding":   self.codec_embedding,
            "talker":            self.talker,
            "code_predictor":    self.code_predictor,
            "tokenizer_decoder": self.tokenizer_decoder,
        }
        if args.profile and not args.onnx:
            for net in self.nets.values():
                net.set_profile_mode(True)
        num_groups = self.num_code_groups
        self._sub_attn_prefill = self.generate_attention_mask(2)          # [1,1,2,2]
        self._sub_pos_prefill  = np.array([[0, 1]], dtype=np.int64)
        self._sub_attn_decode  = [self.generate_attention_mask(1, 1 + k)
                                  for k in range(1, num_groups - 1)]
        self._sub_pos_decode   = [np.array([[1 + k]], dtype=np.int64)
                                  for k in range(1, num_groups - 1)]

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
    # _run_decoder: talker / code predictor 共通の 1 ステップ実行
    #   inputs      : KV cache 以外の入力のリスト
    #   num_outputs : KV cache 以外の出力の数 (talker は logits と last_hidden の
    #                 2 個、code predictor は logits の 1 個)
    #   kv_caches   : num_layers*2 個の [1, num_kv_heads, past, head_dim]
    #                 None を渡すと、前ステップの出力ブロブを ailia 内部で
    #                 そのまま次の入力ブロブへコピーして使う (copy_blob_data)。
    #                 KV cache を Python 側に取り出して渡し直す往復が消えるので
    #                 デコードが進んで cache が伸びるほど効果が大きい。
    # returns (KV cache 以外の出力のリスト, next kv_caches or None)
    # ------------------------------------------------------------------
    def _run_decoder(self, name, net, num_layers, num_outputs, inputs, kv_caches):
        with self.benchmark.measure(name):
            return self._run_decoder_impl(net, num_layers, num_outputs, inputs, kv_caches)

    def _run_decoder_impl(self, net, num_layers, num_outputs, inputs, kv_caches):
        n = len(inputs)   # KV cache 以外の入力の数
        if kv_caches is not None:
            for index, value in enumerate(inputs):
                net.set_input_blob_shape(value.shape, index)
            for i in range(num_layers * 2):
                net.set_input_blob_shape(kv_caches[i].shape, n + i)

            outputs = net.run(inputs + kv_caches)
            if self.use_copy_blob_data:
                # 次のステップからは ailia 内部の出力ブロブをコピーする
                return list(outputs[:num_outputs]), None
            return (list(outputs[:num_outputs]),
                    [outputs[num_outputs + i] for i in range(num_layers * 2)])

        input_blobs  = net.get_input_blob_list()
        output_blobs = net.get_output_blob_list()
        # 入力の shape を変えると ailia が全体の形状を再推論し、コピー元にする
        # 出力ブロブの shape も変わってしまうので、先に全部読み出しておく
        kv_shapes = [
            net.get_blob_shape(output_blobs[num_outputs + i]) for i in range(num_layers * 2)
        ]
        for index, value in enumerate(inputs):
            net.set_input_blob_data(value, input_blobs[index])
        for i in range(num_layers * 2):
            net.set_input_blob_shape(kv_shapes[i], input_blobs[n + i])
            net.copy_blob_data(input_blobs[n + i], output_blobs[num_outputs + i], net)
        net.update()
        return [net.get_blob_data(output_blobs[i]) for i in range(num_outputs)], None

    # ------------------------------------------------------------------
    # _run_talker: 自己回帰本体を 1 ステップ進める
    #   prefill は組み立てたプロンプト、decode は 16 グループの codec 埋め込みを
    #   合算して text_feedback を足したものを inputs_embeds に渡す。
    # returns (logits [1,1,codec_vocab], last_hidden [1,1,hidden], next kv)
    # ------------------------------------------------------------------
    def _run_talker(self, inputs_embeds, attention_mask, position_ids, kv_caches):
        outputs, kv_caches = self._run_decoder(
            "talker", self.talker, self.NUM_LAYERS, 2,
            [inputs_embeds, attention_mask, position_ids], kv_caches,
        )
        return outputs[0], outputs[1], kv_caches

    def _run_code_predictor(self, inputs_embeds, head_rows, attention_mask,
                            position_ids, kv_caches):
        outputs, kv_caches = self._run_decoder(
            "code_predictor", self.code_predictor, self.NUM_SUB_LAYERS, 1,
            [inputs_embeds, head_rows, attention_mask, position_ids], kv_caches
        )
        return outputs[0], kv_caches

    # ------------------------------------------------------------------
    # _run_codec_embedding: codec 埋め込みテーブルを引く
    #   codec_rows [n, 16] を渡すと、各位置の 16 行分を合算した [1, n, H] が返る
    # ------------------------------------------------------------------
    def _run_codec_embedding(self, codec_rows):
        with self.benchmark.measure("codec_embedding"):
            self.codec_embedding.set_input_blob_shape(codec_rows.shape, 0)
            return self.codec_embedding.run([codec_rows])[0]

    def codec_row(self, group, token):
        """グループ group のトークン token が入っている行番号。"""
        if group == 0:
            return token
        return self.talker_vocab_size + (group - 1) * self.group_vocab_size + token

    def frame_rows(self, group_tokens):
        """1 フレーム 16 グループ分の行番号。"""
        return [self.codec_row(group, token)
                for group, token in enumerate(group_tokens)]

    def group_rows(self, group, token):
        """1 グループだけの行番号。残りはゼロ行を指す。"""
        return [self.codec_row(group, token)] + [self.zero_row] * (self.num_code_groups - 1)

    # ------------------------------------------------------------------
    # _run_prompt: テキストのトークン ID をまとめて 1 回で投影する
    #   必要な ID を全部つないで渡し、返ってきた埋め込みを用途ごとに切り出す。
    # returns projected_text [1, text_len, H]
    # ------------------------------------------------------------------
    def _run_prompt(self, text_tokens):
        with self.benchmark.measure("prompt"):
            self.prompt.set_input_blob_shape(text_tokens.shape, 0)
            return self.prompt.run([text_tokens])[0]

    # ------------------------------------------------------------------
    # _predict_subgroups: グループ 1〜15 を code predictor で順に予測する
    #   code predictor の ONNX は 15 個の lm_head を内部に持つので、使う
    #   2048 行を head_rows で指定して logits を受け取る。
    #     position 0   : talker の hidden
    #     position k+1 : グループ k の埋め込み → グループ k+1 の logits
    # ------------------------------------------------------------------
    def _predict_subgroups(self, group0_token: int, past_hidden: np.ndarray,
                           temperature: float = 0.9, top_k: int = 50) -> list:
        NSL  = self.NUM_SUB_LAYERS
        NKV  = self.cfg["sub_num_kv_heads"]
        HDIM = self.cfg["sub_head_dim"]
        sub_kv = [np.zeros((1, NKV, 0, HDIM), dtype=np.float32) for _ in range(NSL * 2)]

        # ── Prefill (seq=2): position 0 は hidden、position 1 はグループ 0 ──
        group0_emb = self._run_codec_embedding(
            np.array([self.group_rows(0, group0_token)], dtype=np.int64))
        prefill_emb = np.concatenate(
            [past_hidden, group0_emb], axis=1
        ).astype(np.float32)

        logits, sub_kv = self._run_code_predictor(
            prefill_emb, self.head_rows[0], self._sub_attn_prefill,
            self._sub_pos_prefill, sub_kv
        )
        group_tokens = [_sample_token(logits[0, -1, :], temperature, top_k)]

        # ── Decode (seq=1 × 14) ──
        for k in range(1, self.num_code_groups - 1):
            embed = self._run_codec_embedding(
                np.array([self.group_rows(k, group_tokens[k - 1])], dtype=np.int64)
            ).astype(np.float32)
            logits, sub_kv = self._run_code_predictor(
                embed, self.head_rows[k],
                self._sub_attn_decode[k - 1], self._sub_pos_decode[k - 1], sub_kv
            )
            group_tokens.append(_sample_token(logits[0, -1, :], temperature, top_k))

        return group_tokens

    # ------------------------------------------------------------------
    # create_voice_clone_prompt
    #   参照音声から codec トークンと speaker embedding を 1 回の推論で得る
    # returns (ref_code [16, T_ref], speaker embedding [1, 1, H])
    # ------------------------------------------------------------------
    def create_voice_clone_prompt(self, ref_audio):
        wav, sr = librosa.load(ref_audio, sr=24000, mono=True)
        wav_input = wav[np.newaxis, np.newaxis, :].astype(np.float32)
        mel = mel_spectrogram(wav[np.newaxis, :], n_fft, num_mels, 24000,
                              hop_size, win_size, fmin, fmax)

        with self.benchmark.measure("encoder"):
            self.encoder.set_input_blob_shape(wav_input.shape, 0)
            self.encoder.set_input_blob_shape(mel.shape, 1)
            ref_code, spk_emb = self.encoder.run([wav_input, mel])
        # encoder は 32 コードブック出すが、talker が使うのは先頭 16 グループ
        ref_code = ref_code[0, : self.num_code_groups, :].astype(np.int64)  # [16, T]
        spk_emb = spk_emb.reshape(1, 1, -1).astype(np.float32)              # [1, 1, H]

        return ref_code, spk_emb

    # ------------------------------------------------------------------
    # generate_icl_prompt: テキストと参照 codec を足し合わせて ICL プロンプトを作る
    #   text_proj     : 参照テキスト + 生成テキストの投影 [1, text_lens, H]
    #   codec_bos_emb : codec BOS の埋め込み [1, 1, H]
    #   ref_codec_sum : 参照フレームの 16 グループ合算 [1, ref_lens, H]
    # ------------------------------------------------------------------
    def generate_icl_prompt(self, text_proj, codec_bos_emb, ref_codec_sum,
                            tts_pad_embed, tts_eos_embed):
        text_proj = np.concatenate([text_proj, tts_eos_embed], axis=1)
        codec_final_emb = np.concatenate([codec_bos_emb, ref_codec_sum], axis=1)

        text_lens  = text_proj.shape[1]
        codec_lens = codec_final_emb.shape[1]

        if codec_lens >= text_lens:
            # 通常ケース: codec の方が長い → テキストをパディング
            num_pads = codec_lens - text_lens
            if num_pads > 0:
                pads      = np.tile(tts_pad_embed, (1, num_pads, 1))
                text_proj = np.concatenate([text_proj, pads], axis=1)
            icl_embed            = text_proj + codec_final_emb   # [1, codec_lens, H]
            trailing_text_hidden = tts_pad_embed                 # 全ステップ tts_pad_embed
        else:
            # 稀ケース: text の方が長い → codec 長でカット、余りを trailing に
            icl_embed            = text_proj[:, :codec_lens, :] + codec_final_emb
            trailing_text_hidden = text_proj[:, codec_lens:, :]  # オーバーフロー分

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

        # ---- 参照音声: codec トークンと speaker embedding ----
        ref_code, spk_emb = self.create_voice_clone_prompt(ref_audio)

        # ---- テキストのトークン化 ----
        if ref_text is None:
            logger.error("--ref_text is required for voice cloning")
            sys.exit()
        ref_text_formatted = f"<|im_start|>assistant\n{ref_text}<|im_end|>\n"
        ref_ids = self.text_tokenizer.encode(ref_text_formatted)
        prompt_text = "<|im_start|>assistant\n"
        body_text   = f"{text}<|im_end|>\n<|im_start|>assistant\n"
        all_text_ids = (self.text_tokenizer.encode(prompt_text)
                        + self.text_tokenizer.encode(body_text))

        # ---- codec tag のトークン ID ----
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
        # codec_tag_emb は tag_ids_0 + speaker + [codec_pad, codec_bos] の並びで、
        # そこから 1 つ短い長さの tts_pad*n + tts_bos を足し合わせる
        num_tts_pad = len(tag_ids_0) + 1
        codec_ids = tag_ids_0 + [cfg["codec_pad_id"], cfg["codec_bos_id"]]

        # ---- テキスト側の埋め込みを 1 回の prompt 推論でまとめて取得 ----
        #   special (2) + role (3) + tag base (num_tts_pad+1) + ICL テキスト
        special_ids  = [cfg["tts_eos_id"], cfg["tts_pad_id"]]
        role_ids     = [cfg["im_start_id"], cfg["assistant_id"], 198]  # 198=\n
        tag_base_ids = [cfg["tts_pad_id"]] * num_tts_pad + [cfg["tts_bos_id"]]
        icl_text_ids = ref_ids[3:-2] + all_text_ids[3:-5]
        text_ids     = special_ids + role_ids + tag_base_ids + icl_text_ids

        projected = self._run_prompt(np.array([text_ids], dtype=np.int64))

        # ---- codec 側の埋め込みも 1 回でまとめて取得 ----
        #   tag は 1 グループだけなので残りをゼロ行に、参照フレームは 16 グループ分
        codec_rows = [self.group_rows(0, codec_id) for codec_id in codec_ids] + [
            self.frame_rows(ref_code[:, frame]) for frame in range(ref_code.shape[1])
        ]
        codec_embeds = self._run_codec_embedding(np.array(codec_rows, dtype=np.int64))
        ref_codec_sum = codec_embeds[:, len(codec_ids):, :]
        codec_embeds = codec_embeds[:, : len(codec_ids), :]

        tts_eos_embed = projected[:, 0:1, :]
        tts_pad_embed = projected[:, 1:2, :]
        offset = len(special_ids)
        role_proj = projected[:, offset:offset + len(role_ids), :]
        offset += len(role_ids)
        tag_base_proj = projected[:, offset:offset + len(tag_base_ids), :]
        offset += len(tag_base_ids)
        text_proj = projected[:, offset:, :]

        codec_bos_emb = codec_embeds[:, -1:, :]
        codec_tag_emb = np.concatenate(
            [codec_embeds[:, :len(tag_ids_0), :], spk_emb,
             codec_embeds[:, len(tag_ids_0):, :]], axis=1
        )   # [1, len(tag_ids_0) + 3, H]

        tags_combined = tag_base_proj + codec_tag_emb[:, :-1, :]
        talker_input_embed = np.concatenate([role_proj, tags_combined], axis=1)

        # ---- ICL prompt ----
        icl_input_embed, trailing_text_hidden = self.generate_icl_prompt(
            text_proj     = text_proj,
            codec_bos_emb = codec_bos_emb,
            ref_codec_sum = ref_codec_sum,
            tts_pad_embed = tts_pad_embed,
            tts_eos_embed = tts_eos_embed,
        )
        talker_input_embed = np.concatenate(
            [talker_input_embed, icl_input_embed], axis=1
        ).astype(np.float32)

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

        logits, last_hidden, kv_caches = self._run_talker(
            talker_input_embed, attn_mask, position_ids, kv_caches
        )

        EOS_TOKEN_ID   = cfg["codec_eos_id"]   # 2150
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
                logger.info(f"EOS at step {step}")
                break

            # ── code predictor でグループ 1〜15 を予測 ──────────────
            sub_tokens = self._predict_subgroups(
                curr_token_id, past_hidden,
                temperature=subtalker_temperature, top_k=subtalker_top_k,
            )
            all_group_tokens = [curr_token_id] + sub_tokens   # len=16

            generated_ids.append(curr_token_id)
            all_codes_list.append(all_group_tokens)
            token_counts[curr_token_id] = token_counts.get(curr_token_id, 0) + 1

            # ── テキストフィードバック ──────────────────────────────
            if step < trailing_text_hidden.shape[1]:
                text_feedback = trailing_text_hidden[:, step:step + 1, :]
            else:
                text_feedback = tts_pad_embed

            # ── main talker decode ─────────────────────────────────
            #   16 グループの codec 埋め込みを合算し、テキストを足したものが入力
            frame_emb = self._run_codec_embedding(
                np.array([self.frame_rows(all_group_tokens)], dtype=np.int64))
            current_input = (frame_emb + text_feedback).astype(np.float32)

            decode_pos  = prefill_len + step
            attn_mask_d = self.generate_attention_mask(1, decode_pos)
            pos_ids_d   = np.array([[decode_pos]], dtype=np.int64)

            logits, past_hidden, kv_caches = self._run_talker(
                current_input, attn_mask_d, pos_ids_d, kv_caches
            )
            past_hidden = past_hidden.astype(np.float32)   # [1, 1, hidden_size]

            # ── 次の group0 トークン予測 ───────────────────────────
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
        all_codes = np.array(all_codes_list, dtype=np.int64).T[np.newaxis, :, :]

        # ── 公式準拠: 参照codecを前に連結してdecodeし、先頭を比率カット ──
        #   (model.py generate_voice_clone L612-631)
        #   decoder は causal なので、参照フレームを前置きして文脈を与え、
        #   生成した波形の先頭(参照分)を比率で切り落とす。
        ref_T = ref_code.shape[1]
        codes_for_decode = np.concatenate(
            [ref_code[np.newaxis, :, :], all_codes], axis=2
        )  # [1, 16, T_ref + T]

        with self.benchmark.measure("tokenizer_decoder"):
            self.tokenizer_decoder.set_input_blob_shape(codes_for_decode.shape, 0)
            wav = np.squeeze(self.tokenizer_decoder.run([codes_for_decode])[0])  # [L]

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
    logger.info("Generating speech...")
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
    logger.info(f"Saved as {args.savepath}.")


if __name__ == "__main__":
    main()
