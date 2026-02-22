import ailia
import numpy as np
import yaml
import sys
import matplotlib.pyplot as plt
import os
import scipy.io.wavfile as wavfile
import re
from string import punctuation
from logging import getLogger
import onnx

# ===========================
# Settings
# ===========================

# リポジトリのルートにあるutilsを参照できるようにする
sys.path.append('../../util')
from arg_utils import get_base_parser, update_parser, get_savepath  # noqa
from model_utils import check_and_download_models  # noqa

sys.path.append(".")
from g2p_en import G2p
from pypinyin import pinyin, Style
from text import text_to_sequence

logger = getLogger(__name__)

# モデル設定
WEIGHT_PATH_FS2 = 'ljspeech.onnx'
MODEL_PATH_FS2 = 'ljspeech.onnx.prototxt'
WEIGHT_PATH_HIFI = 'hifigan.onnx'
MODEL_PATH_HIFI = 'hifigan.onnx.prototxt'
REMOTE_PATH = "https://storage.googleapis.com/ailia-models/fastspeech2"

PREPROCESS_CONFIG = "config/LJSpeech/preprocess.yaml"

# ===========================
# Arguments
# ===========================
parser = get_base_parser(
    'FastSpeech2 (Ailia Inference)',
    None,
    'output.wav'
)
parser.add_argument(
    '-t', '--text',
    type=str,
    default="Ailia SDK makes it easy to deploy deep learning models.",
    help='raw text to synthesize, for single-sentence mode only'
)
parser.add_argument(
    '--speaker_id',
    type=int,
    default=0,
    help='speaker ID for multi-speaker synthesis, for single-sentence mode only'
)
parser.add_argument(
    '--pitch_control',
    type=float,
    default=1.0,
    help='control the pitch of the whole utterance, larger value for higher pitch'
)
parser.add_argument(
    '--energy_control',
    type=float,
    default=1.0,
    help='control the energy of the whole utterance, larger value for larger volume'
)
parser.add_argument(
    '--duration_control',
    type=float,
    default=1.0,
    help='control the speed of the whole utterance, larger value for slower speaking rate'
)
parser.add_argument(
    '--preprocess_config',
    type=str,
    default=PREPROCESS_CONFIG,
    help='path to preprocess.yaml'
)
parser.add_argument(
    '--onnx_fs2',
    default=WEIGHT_PATH_FS2,
    help='Path to FastSpeech2 ONNX file.'
)
parser.add_argument(
    '--onnx_hifi',
    default=WEIGHT_PATH_HIFI,
    help='Path to HiFi-GAN ONNX file.'
)
parser.add_argument(
    '--output_dir',
    type=str,
    default=None,
    help='output directory for generated audio files'
)
args = update_parser(parser)

# ===========================
# 1. 前処理（元リポジトリ synthesize.py と同一）
# ===========================
def preprocess_english(text, preprocess_config):
    text = text.rstrip(punctuation)
    lexicon = read_lexicon(preprocess_config["path"]["lexicon_path"])

    g2p = G2p()
    phones = []
    words = re.split(r"([,;.\-\?\!\s+])", text)
    for w in words:
        if w.lower() in lexicon:
            phones += lexicon[w.lower()]
        else:
            phones += list(filter(lambda p: p != " ", g2p(w)))
    phones = "{" + "}{".join(phones) + "}"
    phones = re.sub(r"\{[^\w\s]?\}", "{sp}", phones)
    phones = phones.replace("}{", " ")

    print("Raw Text Sequence: {}".format(text))
    print("Phoneme Sequence: {}".format(phones))

    sequence = np.array(
        text_to_sequence(
            phones, preprocess_config["preprocessing"]["text"]["text_cleaners"]
        )
    )
    return sequence

def read_lexicon(lex_path):
    lexicon = {}
    if not os.path.exists(lex_path):
        print(f"Warning: Lexicon file not found at {lex_path}. Skipping lexicon.")
        return lexicon

    with open(lex_path, encoding='utf-8') as f:
        for line in f:
            temp = re.split(r"\s+", line.strip("\n"))
            word = temp[0]
            phones = temp[1:]
            if word.lower() not in lexicon:
                lexicon[word.lower()] = phones
    return lexicon

def preprocess_mandarin(text, preprocess_config):
    lexicon = read_lexicon(preprocess_config["path"]["lexicon_path"])

    phones = []
    pinyins = [
        p[0]
        for p in pinyin(
            text, style=Style.TONE3, strict=False, neutral_tone_with_five=True
        )
    ]
    for p in pinyins:
        if p in lexicon:
            phones += lexicon[p]
        else:
            phones.append("sp")

    phones = "{" + " ".join(phones) + "}"
    print("Raw Text Sequence: {}".format(text))
    print("Phoneme Sequence: {}".format(phones))
    sequence = np.array(
        text_to_sequence(
            phones, preprocess_config["preprocessing"]["text"]["text_cleaners"]
        )
    )

    return np.array(sequence)

def preprocess_text(text, preprocess_config, preprocess_config_path):
    cleaners = preprocess_config["preprocessing"]["text"]["text_cleaners"]
    is_mandarin = ('mandarin_cleaners' in cleaners or 'pinyin_cleaners' in cleaners)
    if 'AISHELL3' in preprocess_config_path.upper():
        is_mandarin = True

    if is_mandarin:
        print("Detected language: Mandarin")
        return preprocess_mandarin(text, preprocess_config)
    else:
        print("Detected language: English")
        return preprocess_english(text, preprocess_config)

# ===========================
# 3. メイン関数
# ===========================
def infer():
    # モデルのダウンロード
    try:
        check_and_download_models(args.onnx_fs2, args.onnx_fs2 + ".prototxt", REMOTE_PATH)
    except Exception:
        pass
    try:
        check_and_download_models(args.onnx_hifi, args.onnx_hifi + ".prototxt", REMOTE_PATH)
    except Exception:
        pass

    # -------------------------------------------
    # ロード
    # -------------------------------------------
    if not os.path.exists(args.onnx_fs2) or not os.path.exists(args.onnx_hifi):
        logger.error("Error: ONNX file not found.")
        return

    logger.info("Loading Config...")
    try:
        preprocess_config = yaml.load(open(args.preprocess_config, "r"), Loader=yaml.FullLoader)
    except Exception as e:
        logger.error(f"Error loading config: {e}")
        return

    logger.info("Loading ONNX Models (ailia SDK)...")
    try:
        env_id = args.env_id
        fs2_net = ailia.Net(None, args.onnx_fs2, env_id=env_id)
        hifi_net = ailia.Net(None, args.onnx_hifi, env_id=env_id)
    except Exception as e:
        logger.error(f"Error initializing ailia: {e}")
        return

    # ONNXモデルの出力名を取得（ailiaSDKでは直接取得できないため、ONNXファイルから読み込む）
    onnx_model = onnx.load(args.onnx_fs2)
    fs2_output_names = [output.name for output in onnx_model.graph.output]

    # -------------------------------------------
    # 入力準備（元リポジトリ synthesize.py と同じ: actual length で渡す）
    # -------------------------------------------
    logger.info(f"Input Text: {args.text}")
    sequence = preprocess_text(args.text, preprocess_config, args.preprocess_config)

    real_len = len(sequence)
    logger.info(f"Sequence Length: {real_len}")

    texts = np.array([sequence], dtype=np.int64)  # (1, actual_len) パディングなし
    src_lens = np.array([real_len], dtype=np.int64)

    # ONNXモデルの入力名を取得
    fs2_input_names = [inp.name for inp in onnx_model.graph.input
                       if inp.name not in [n.name for n in onnx_model.graph.initializer]]

    # max_src_len: 元リポジトリと同じく actual length を渡す
    max_src_len = np.array(real_len, dtype=np.int64)

    # speakersの形状を確認して適切に設定
    if "speakers" in fs2_input_names:
        for inp in onnx_model.graph.input:
            if inp.name == "speakers":
                dims = [d.dim_value if d.dim_value > 0 else d.dim_param
                        for d in inp.type.tensor_type.shape.dim]
                if len(dims) == 2:
                    speakers = np.array([[args.speaker_id]], dtype=np.int64)
                else:
                    speakers = np.array([args.speaker_id], dtype=np.int64)
                break
    else:
        speakers = None

    p_control = np.array(args.pitch_control, dtype=np.float32)
    e_control = np.array(args.energy_control, dtype=np.float32)
    d_control = np.array(args.duration_control, dtype=np.float32)

    # -------------------------------------------
    # FastSpeech2 推論
    # -------------------------------------------
    logger.info("Running FastSpeech2...")

    inputs = {
        "texts": texts,
        "src_lens": src_lens,
        "max_src_len": max_src_len,
    }
    for ctrl in ["p_control", "e_control", "d_control"]:
        if ctrl in fs2_input_names:
            inputs[ctrl] = locals()[ctrl]
    if speakers is not None:
        inputs["speakers"] = speakers

    # 入力形状のデバッグ情報
    logger.info("\n=== FastSpeech2 Input Shapes ===")
    for k, v in inputs.items():
        logger.info(f"  {k:20s}: {v.shape if hasattr(v, 'shape') else type(v)}")
    logger.info("=" * 40)

    try:
        fs2_res = fs2_net.predict(inputs)
    except Exception as e:
        logger.error(f"FastSpeech2 inference failed: {e}")
        return

    # -------------------------------------------
    # 結果の切り出し
    # -------------------------------------------
    try:
        d_rounded_index = fs2_output_names.index("d_rounded")
        postnet_index = fs2_output_names.index("postnet_output")
    except ValueError:
        d_rounded_index = 5
        postnet_index = 1

    mel_output_whole = fs2_res[postnet_index]  # [1, MaxLen, 80]
    d_rounded = fs2_res[d_rounded_index]       # [1, MaxLen]

    # 元のリポジトリと同じ処理：mel_lenを計算（synth_samplesと同様）
    valid_durations = d_rounded[0, :real_len]
    mel_len = int(np.sum(valid_durations))

    logger.info(f"Generated Mel Length: {mel_len}")

    # 元のリポジトリと同じ処理：mel_lenで切り出す（バッファなし）
    mel_output = mel_output_whole[:, :mel_len, :]

    # -------------------------------------------
    # HiFi-GAN 推論（元のリポジトリと同じ処理）
    # -------------------------------------------
    logger.info("Running HiFi-GAN...")

    # 元のリポジトリと同じ処理：[1, MelLen, 80] -> [1, 80, MelLen]
    # synth_samplesでは predictions[1].transpose(1, 2) を使用
    mel_input = mel_output.transpose(0, 2, 1).astype(np.float32)

    hop_length = preprocess_config["preprocessing"]["stft"]["hop_length"]
    logger.info(f"HiFi-GAN input: mel_input shape = {mel_input.shape}")

    try:
        audio_res = hifi_net.predict([mel_input])
        wav = audio_res[0].squeeze()
    except Exception as e:
        logger.error(f"HiFi-GAN inference failed: {e}")
        return

    audio_len = mel_len * hop_length
    if len(wav) > audio_len:
        wav = wav[:audio_len]
        logger.info(f"Trimmed audio to {audio_len} samples (mel_len={mel_len} * hop_length={hop_length})")

    # -------------------------------------------
    # 保存（元のリポジトリと同じ処理）
    # -------------------------------------------
    # 元のリポジトリのvocoder_inferと同じ処理：
    # wavs = wavs.cpu().numpy() * max_wav_value
    MAX_WAV_VALUE = preprocess_config["preprocessing"]["audio"]["max_wav_value"]
    wav = wav * MAX_WAV_VALUE
    wav = wav.astype('int16')

    if args.output_dir is not None:
        output_dir = args.output_dir
    else:
        output_dir = "onnx/result/ailia"

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    wav_path = os.path.join(output_dir, "output_ailia.wav")
    plot_path = os.path.join(output_dir, "output_mel_ailia.png")

    sampling_rate = preprocess_config["preprocessing"]["audio"]["sampling_rate"]
    wavfile.write(wav_path, sampling_rate, wav)
    logger.info(f"Saved Audio: {wav_path}")

    plt.figure(figsize=(10, 4))
    plt.imshow(mel_output[0].T, aspect="auto", origin="lower")
    plt.title(f"Generated Mel (Len: {mel_output.shape[1]})")
    plt.colorbar()
    plt.tight_layout()
    plt.savefig(plot_path)
    plt.close()
    logger.info(f"Saved Plot: {plot_path}")

if __name__ == "__main__":
    infer()
