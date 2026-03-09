import ailia
import numpy as np
import yaml
import sys
import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import os
import scipy.io.wavfile as wavfile
import re
from string import punctuation
from logging import getLogger
import onnx

sys.path.append('../../util')
from arg_utils import get_base_parser, update_parser, get_savepath  # noqa
from model_utils import check_and_download_models  # noqa

sys.path.append(".")
from g2p_en import G2p
from pypinyin import pinyin, Style
from text import text_to_sequence

logger = getLogger(__name__)

# モデル設定
REMOTE_PATH = "https://storage.googleapis.com/ailia-models/fastspeech2/"

parser = get_base_parser(
    'FastSpeech2',
    None,
    'output.wav'
)
parser.add_argument(
    '--text',
    type=str,
    default="Hello World",
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
# こちらの引数について、元のリポジトリでも、推論の際に変更しても効果がありませんでした。
# 形式上残してありますが、推論結果には全く影響しません。
parser.add_argument(
    '--energy_control',
    type=float,
    default=1.0,
    help='control the energy of the whole utterance, larger value for larger volume'
)
# 以下２つの引数は推論時に適用されることが確認できています。
parser.add_argument(
    '--duration_control',
    type=float,
    default=1.0,
    help='control the speed of the whole utterance, larger value for slower speaking rate'
)
# 以下引数の指定。デフォルトはLJSpeech.
parser.add_argument(
    '-o', '--output',
    type=str,
    default=None,
    help='output wav path (if omitted, --savepath is used)'
)
parser.add_argument(
    '-m', '--model_name',
    default='ljspeech',
    choices=["ljspeech", "LibriTTS", "AISHELL-3"],
)
args = update_parser(parser)
if args.output is None:
    args.output = args.savepath

if args.model_name == 'ljspeech':
    WEIGHT_PATH_FS2 = 'ljspeech.onnx'
    WEIGHT_PATH_HIFI = "hifigan_ljspeech.onnx"
    PREPROCESS_CONFIG = "config/LJSpeech/preprocess.yaml"
elif args.model_name == 'LibriTTS':
    WEIGHT_PATH_FS2 = 'libritts.onnx'
    WEIGHT_PATH_HIFI = "hifigan.onnx"
    PREPROCESS_CONFIG = "config/LibriTTS/preprocess.yaml"
elif args.model_name == 'AISHELL-3':
    WEIGHT_PATH_FS2 = 'aishell3.onnx'
    WEIGHT_PATH_HIFI = "hifigan.onnx"
    PREPROCESS_CONFIG = "config/AISHELL3/preprocess.yaml"

def expand_phoneme(values, durations):
    out = []
    for v, d in zip(values, durations):
        out += [float(v)] * max(0, int(d))
    return np.array(out)


def plot_mel(data, stats, titles):
    fig, axes = plt.subplots(len(data), 1, squeeze=False)
    if titles is None:
        titles = [None for _ in range(len(data))]
    pitch_min, pitch_max, pitch_mean, pitch_std, energy_min, energy_max = stats
    pitch_min = pitch_min * pitch_std + pitch_mean
    pitch_max = pitch_max * pitch_std + pitch_mean

    def add_axis(fig, old_ax):
        ax = fig.add_axes(old_ax.get_position(), anchor="W")
        ax.set_facecolor("None")
        return ax

    for i in range(len(data)):
        mel, pitch, energy = data[i]
        pitch = pitch * pitch_std + pitch_mean
        axes[i][0].imshow(mel, origin="lower")
        axes[i][0].set_aspect(2.5, adjustable="box")
        axes[i][0].set_ylim(0, mel.shape[0])
        axes[i][0].set_title(titles[i], fontsize="medium")
        axes[i][0].tick_params(labelsize="x-small", left=False, labelleft=False)
        axes[i][0].set_anchor("W")

        ax1 = add_axis(fig, axes[i][0])
        ax1.plot(pitch, color="tomato")
        ax1.set_xlim(0, mel.shape[1])
        ax1.set_ylim(0, pitch_max)
        ax1.set_ylabel("F0", color="tomato")
        ax1.tick_params(
            labelsize="x-small", colors="tomato", bottom=False, labelbottom=False
        )

        ax2 = add_axis(fig, axes[i][0])
        ax2.plot(energy, color="darkviolet")
        ax2.set_xlim(0, mel.shape[1])
        ax2.set_ylim(energy_min, energy_max)
        ax2.set_ylabel("Energy", color="darkviolet")
        ax2.yaxis.set_label_position("right")
        ax2.tick_params(
            labelsize="x-small",
            colors="darkviolet",
            bottom=False,
            labelbottom=False,
            left=False,
            labelleft=False,
            right=True,
            labelright=True,
        )

    return fig


# 前処理
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

# 前処理の際に、preprocess_configのパスからデータセット名を判定
def select_hifigan(preprocess_config_path):
    """preprocess_configのパスからデータセット名を判定し、適切なHiFi-GANを選択"""
    dataset = os.path.basename(os.path.dirname(preprocess_config_path))
    if dataset == "LJSpeech":
        return "hifigan_ljspeech.onnx"
    else:
        return "hifigan.onnx"


def infer():
    # モデルのダウンロード
    check_and_download_models(WEIGHT_PATH_FS2, WEIGHT_PATH_FS2 + ".prototxt", REMOTE_PATH)
    check_and_download_models(WEIGHT_PATH_HIFI, WEIGHT_PATH_HIFI + ".prototxt", REMOTE_PATH)

    # -------------------------------------------
    # ロード
    # -------------------------------------------
    preprocess_config = yaml.load(open(PREPROCESS_CONFIG, "r"), Loader=yaml.FullLoader)

    logger.info("Loading ONNX Models (ailia SDK)...")
    try:
        env_id = args.env_id
        fs2_net = ailia.Net(WEIGHT_PATH_FS2 + ".prototxt", WEIGHT_PATH_FS2, env_id=env_id)
        hifi_net = ailia.Net(WEIGHT_PATH_HIFI + ".prototxt", WEIGHT_PATH_HIFI, env_id=env_id)
    except Exception as e:
        logger.error(f"Error initializing ailia: {e}")
        return

    # ONNXモデルの出力名を取得
    onnx_model = onnx.load(WEIGHT_PATH_FS2)
    fs2_output_names = [output.name for output in onnx_model.graph.output]

    # 入力
    logger.info(f"Input Text: {args.text}")
    sequence = preprocess_text(args.text, preprocess_config, PREPROCESS_CONFIG)

    real_len = len(sequence)
    logger.info(f"Sequence Length: {real_len}")

    texts = np.array([sequence], dtype=np.int64) 
    src_lens = np.array([real_len], dtype=np.int64)

    # ONNXモデルの入力名を取得
    fs2_input_names = [inp.name for inp in onnx_model.graph.input
                       if inp.name not in [n.name for n in onnx_model.graph.initializer]]


    max_src_len = np.array(real_len, dtype=np.int64)


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

    # 推論
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


    try:
        d_rounded_index = fs2_output_names.index("d_rounded")
        postnet_index = fs2_output_names.index("postnet_output")
    except ValueError:
        d_rounded_index = 5
        postnet_index = 1

    mel_output_whole = fs2_res[postnet_index]  # [1, MaxLen, 80]
    d_rounded = fs2_res[d_rounded_index]       # [1, MaxLen]


    valid_durations = d_rounded[0, :real_len]
    mel_len = int(np.sum(valid_durations))

    logger.info(f"Generated Mel Length: {mel_len}")

    mel_output = mel_output_whole[:, :mel_len, :]


    logger.info("Running HiFi-GAN...")
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


    MAX_WAV_VALUE = preprocess_config["preprocessing"]["audio"]["max_wav_value"]
    wav = (wav * MAX_WAV_VALUE).astype("int16")

    # レビュー指摘に合わせ、常に指定パスへwavを保存する。
    wav_path = args.output
    output_dir = os.path.dirname(wav_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    plot_path = os.path.splitext(wav_path)[0] + ".png"

    sampling_rate = preprocess_config["preprocessing"]["audio"]["sampling_rate"]
    wavfile.write(wav_path, sampling_rate, wav)
    logger.info("Saved Audio: {}".format(wav_path))

    try:
        p_pred_index = fs2_output_names.index("p_predictions")
        e_pred_index = fs2_output_names.index("e_predictions")
    except ValueError:
        p_pred_index, e_pred_index = 2, 3

    pitch_pred = fs2_res[p_pred_index][0, :real_len]
    energy_pred = fs2_res[e_pred_index][0, :real_len]
    duration_arr = d_rounded[0, :real_len]

    pitch_feature = preprocess_config["preprocessing"]["pitch"]["feature"]
    energy_feature = preprocess_config["preprocessing"]["energy"]["feature"]

    pitch_plot = expand_phoneme(pitch_pred, duration_arr) if pitch_feature == "phoneme_level" else pitch_pred[:mel_len]
    energy_plot = expand_phoneme(energy_pred, duration_arr) if energy_feature == "phoneme_level" else energy_pred[:mel_len]

    stats_path = os.path.join(preprocess_config["path"]["preprocessed_path"], "stats.json")
    with open(stats_path) as f:
        stats = json.load(f)
    stats_values = stats["pitch"] + stats["energy"][:2]

    mel_plot = mel_output[0, :mel_len].T  # (80, mel_len)
    fig = plot_mel([(mel_plot, pitch_plot, energy_plot)], stats_values, ["Synthetized Spectrogram"])
    plt.savefig(plot_path)
    plt.close()
    logger.info("Saved Plot:  {}".format(plot_path))

if __name__ == "__main__":
    infer()
