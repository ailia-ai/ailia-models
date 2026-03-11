import os
import time
import sys
import platform

# logger
from logging import getLogger  # noqa: E402

import numpy as np
import soundfile
import librosa

# import original modules
sys.path.append("../../util")
from arg_utils import get_base_parser, update_parser  # noqa: E402
from model_utils import check_and_download_models  # noqa: E402

import ailia
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "gpt-sovits-v2"))
from text import cleaned_text_to_sequence
from text.cleaner import clean_text


logger = getLogger(__name__)

# ======================
# PARAMETERS
# ======================

REF_WAV_PATH = "reference_audio_captured_by_ax.wav"
REF_TEXT = "水をマレーシアから買わなくてはならない。"
SAVE_WAV_PATH = "output.wav"
REMOTE_PATH = "https://storage.googleapis.com/ailia-models/gpt-sovits-v2-pro/"
REMOTE_PATH_T2S = REMOTE_PATH
WEIGHT_PATH_SSL = "cnhubert.onnx"
WEIGHT_PATH_T2S_ENCODER = "t2s_encoder.onnx"
WEIGHT_PATH_T2S_FIRST_DECODER = "t2s_fsdec.onnx"
WEIGHT_PATH_T2S_STAGE_DECODER = "t2s_sdec.opt.onnx"
WEIGHT_PATH_VITS = "vits.onnx"
WEIGHT_PATH_SV = "sv.onnx"
WEIGHT_PATH_BERT = "chinese-roberta.onnx"
MODEL_PATH_SSL = WEIGHT_PATH_SSL + ".prototxt"
MODEL_PATH_T2S_ENCODER = WEIGHT_PATH_T2S_ENCODER + ".prototxt"
MODEL_PATH_T2S_FIRST_DECODER = WEIGHT_PATH_T2S_FIRST_DECODER + ".prototxt"
MODEL_PATH_T2S_STAGE_DECODER = WEIGHT_PATH_T2S_STAGE_DECODER + ".prototxt"
MODEL_PATH_VITS = WEIGHT_PATH_VITS + ".prototxt"
MODEL_PATH_SV = WEIGHT_PATH_SV + ".prototxt"
MODEL_PATH_BERT = WEIGHT_PATH_BERT + ".prototxt"


# ======================
# Arguemnt Parser Config
# ======================

parser = get_base_parser("GPT-SoVits-V2-Pro", None, SAVE_WAV_PATH)
# overwrite
parser.add_argument(
    "--input",
    "-i",
    metavar="TEXT",
    default="ax株式会社ではAIの実用化のための技術を開発しています。",
    help="input text",
)
parser.add_argument(
    "--text_language", "-tl", default="ja", choices=("ja", "en", "zh"), help="[ja, en, zh]"
)
parser.add_argument(
    "--ref_audio",
    "-ra",
    metavar="TEXT",
    default=REF_WAV_PATH,
    help="ref audio",
)
parser.add_argument(
    "--ref_text",
    "-rt",
    metavar="TEXT",
    default=REF_TEXT,
    help="ref text",
)
parser.add_argument(
    "--ref_language", "-rl", default="ja", choices=("ja", "en", "zh"), help="[ja, en, zh]"
)
parser.add_argument("--top_k", type=int, default=15, help="top_k")
parser.add_argument("--top_p", type=float, default=1.0, help="top_p")
parser.add_argument("--temperature", type=float, default=1.0, help="temperature")
parser.add_argument("--speed", type=float, default=1.0, help="Speech rate")
parser.add_argument("--onnx", action="store_true", help="use onnx runtime")
parser.add_argument("--profile", action="store_true", help="use profile model")
parser.add_argument("--distill", type=str, default=None, help="use distill model", choices=(None, "tiny", "base", "small"))
args = update_parser(parser, check_input_type=False)


splits = {
    # fmt: off
    "，", "。", "？", "！", ",", ".", "?", "!", "~", ":", "：", "—", "…",
    # fmt: on
}

version = ailia.get_version().split(".")
AILIA_VERSION_MAJOR = int(version[0])
AILIA_VERSION_MINOR = int(version[1])
AILIA_VERSION_REVISION = int(version[2])
COPY_BLOB_DATA = not (
    AILIA_VERSION_MAJOR <= 1
    and AILIA_VERSION_MINOR <= 2
    and AILIA_VERSION_REVISION < 15
)


# ======================
# Distill models
# ======================

if args.distill is not None:
    REMOTE_PATH_T2S = "https://storage.googleapis.com/ailia-models/gpt-sovits-v2-pro-distill/"
    MODEL_SIZE = args.distill
    WEIGHT_PATH_T2S_ENCODER = "t2s_encoder_distill_"+MODEL_SIZE+".onnx"
    WEIGHT_PATH_T2S_FIRST_DECODER = "t2s_fsdec_distill_"+MODEL_SIZE+".onnx"
    WEIGHT_PATH_T2S_STAGE_DECODER = "t2s_sdec_distill_"+MODEL_SIZE+".opt.onnx"
    MODEL_PATH_T2S_ENCODER = None
    MODEL_PATH_T2S_FIRST_DECODER = None
    MODEL_PATH_T2S_STAGE_DECODER = None

# ======================
# Secondary Functions
# ======================


def split(todo_text):
    todo_text = todo_text.replace("……", "。").replace("——", "，")
    if todo_text[-1] not in splits:
        todo_text += "。"
    i_split_head = i_split_tail = 0
    len_text = len(todo_text)
    todo_texts = []
    while 1:
        if i_split_head >= len_text:
            break  # 结尾一定有标点，所以直接跳出即可，最后一段在上次已加入
        if todo_text[i_split_head] in splits:
            i_split_head += 1
            todo_texts.append(todo_text[i_split_tail:i_split_head])
            i_split_tail = i_split_head
        else:
            i_split_head += 1
    return todo_texts


def cut(inp):
    punctuation = set(["!", "?", "…", ",", ".", "-", " "])

    inp = inp.strip("\n")
    inps = split(inp)
    split_idx = list(range(0, len(inps), 4))
    split_idx[-1] = None
    if len(split_idx) > 1:
        opts = []
        for idx in range(len(split_idx) - 1):
            opts.append("".join(inps[split_idx[idx] : split_idx[idx + 1]]))
    else:
        opts = [inp]
    opts = [item for item in opts if not set(item).issubset(punctuation)]
    return "\n".join(opts)


def process_text(texts):
    _text = []
    if all(text in [None, " ", "\n", ""] for text in texts):
        raise ValueError("Please enter valid text.")
    for text in texts:
        if text in [None, " ", ""]:
            pass
        else:
            _text.append(text)
    return _text


def merge_short_text_in_array(texts, threshold):
    if (len(texts)) < 2:
        return texts
    result = []
    text = ""
    for ele in texts:
        text += ele
        if len(text) >= threshold:
            result.append(text)
            text = ""
    if len(text) > 0:
        if len(result) == 0:
            result.append(text)
        else:
            result[len(result) - 1] += text
    return result


# ======================
# Main Logic
# ======================


class T2SModel:
    def __init__(self, sess_encoder, sess_fsdec, sess_sdec):
        self.hz = 50
        self.max_sec = 54
        self.top_k = 5
        self.early_stop_num = np.array([self.hz * self.max_sec])
        self.sess_encoder = sess_encoder
        self.sess_fsdec = sess_fsdec
        self.sess_sdec = sess_sdec

    def forward(
        self,
        ref_seq,
        text_seq,
        ref_bert,
        text_bert,
        ssl_content,
        top_k=20,
        top_p=0.6,
        temperature=0.6,
        repetition_penalty=1.35,
    ):
        early_stop_num = self.early_stop_num

        top_k = np.array([top_k], dtype=np.int64)
        top_p = np.array([top_p], dtype=np.float32)
        temperature = np.array([temperature], dtype=np.float32)
        repetition_penalty = np.array([repetition_penalty], dtype=np.float32)

        EOS = 1024

        if args.benchmark:
            start = int(round(time.time() * 1000))
        if args.onnx:
            x, prompts = self.sess_encoder.run(
                None,
                {
                    "ref_seq": ref_seq,
                    "text_seq": text_seq,
                    "ref_bert": ref_bert,
                    "text_bert": text_bert,
                    "ssl_content": ssl_content,
                },
            )
        else:
            x, prompts = self.sess_encoder.run(
                {
                    "ref_seq": ref_seq,
                    "text_seq": text_seq,
                    "ref_bert": ref_bert,
                    "text_bert": text_bert,
                    "ssl_content": ssl_content,
                }
            )
        if args.benchmark:
            end = int(round(time.time() * 1000))
            logger.info("\tsencoder processing time {} ms".format(end - start))

        prefix_len = prompts.shape[1]

        if args.benchmark:
            start = int(round(time.time() * 1000))
        if args.onnx:
            y, k, v, y_emb, x_example = self.sess_fsdec.run(
                None,
                {
                    "x": x,
                    "prompts": prompts,
                    "top_k": top_k,
                    "top_p": top_p,
                    "temperature": temperature,
                    "repetition_penalty": repetition_penalty,
                },
            )
        else:
            y, k, v, y_emb, x_example = self.sess_fsdec.run(
                {
                    "x": x,
                    "prompts": prompts,
                    "top_k": top_k,
                    "top_p": top_p,
                    "temperature": temperature,
                    "repetition_penalty": repetition_penalty,
                }
            )
        if args.benchmark:
            end = int(round(time.time() * 1000))
            logger.info("\tfsdec processing time {} ms".format(end - start))

        stop = False
        for idx in range(1, 1500):
            if args.benchmark:
                start = int(round(time.time() * 1000))
            if args.onnx:
                y, k, v, y_emb, logits, samples = self.sess_sdec.run(
                    None,
                    {
                        "iy": y,
                        "ik": k,
                        "iv": v,
                        "iy_emb": y_emb,
                        "ix_example": x_example,
                        "top_k": top_k,
                        "top_p": top_p,
                        "temperature": temperature,
                        "repetition_penalty": repetition_penalty,
                    },
                )
            else:
                if idx == 1:
                    y, k, v, y_emb, logits, samples = self.sess_sdec.run(
                        {
                            "iy": y,
                            "ik": k,
                            "iv": v,
                            "iy_emb": y_emb,
                            "ix_example": x_example,
                            "top_k": top_k,
                            "top_p": top_p,
                            "temperature": temperature,
                            "repetition_penalty": repetition_penalty,
                        }
                    )
                    kv_base_shape = k.shape
                else:
                    input_blob_idx = self.sess_sdec.get_input_blob_list()
                    output_blob_idx = self.sess_sdec.get_output_blob_list()
                    self.sess_sdec.set_input_blob_data(y, 0)
                    if COPY_BLOB_DATA:
                        kv_shape = (
                            kv_base_shape[0],
                            kv_base_shape[1] + idx - 2,
                            kv_base_shape[2],
                            kv_base_shape[3],
                        )
                        self.sess_sdec.set_input_blob_shape(kv_shape, 1)
                        self.sess_sdec.set_input_blob_shape(kv_shape, 2)
                        self.sess_sdec.copy_blob_data(
                            input_blob_idx[1], output_blob_idx[1], self.sess_sdec
                        )
                        self.sess_sdec.copy_blob_data(
                            input_blob_idx[2], output_blob_idx[2], self.sess_sdec
                        )
                    else:
                        self.sess_sdec.set_input_blob_data(k, 1)
                        self.sess_sdec.set_input_blob_data(v, 2)
                    self.sess_sdec.set_input_blob_data(y_emb, 3)
                    self.sess_sdec.set_input_blob_data(x_example, 4)
                    self.sess_sdec.set_input_blob_data(top_k, 5)
                    self.sess_sdec.set_input_blob_data(top_p, 6)
                    self.sess_sdec.set_input_blob_data(temperature, 7)
                    self.sess_sdec.set_input_blob_data(repetition_penalty, 8)
                    self.sess_sdec.update()
                    y = self.sess_sdec.get_blob_data(output_blob_idx[0])
                    if not COPY_BLOB_DATA:
                        k = self.sess_sdec.get_blob_data(output_blob_idx[1])
                        v = self.sess_sdec.get_blob_data(output_blob_idx[2])
                    y_emb = self.sess_sdec.get_blob_data(output_blob_idx[3])
                    logits = self.sess_sdec.get_blob_data(output_blob_idx[4])
                    samples = self.sess_sdec.get_blob_data(output_blob_idx[5])

            if args.benchmark:
                end = int(round(time.time() * 1000))
                logger.info("\tsdec processing time {} ms".format(end - start))
            if early_stop_num != -1 and (y.shape[1] - prefix_len) > early_stop_num:
                stop = True
            if np.argmax(logits, axis=-1)[0] == EOS or samples[0, 0] == EOS:
                stop = True
            if stop:
                break
        y[0, -1] = 0

        return y[np.newaxis, :, -idx:-1]


class GptSoVits:
    def __init__(self, t2s: T2SModel, sess):
        self.t2s = t2s
        self.sess = sess

    def forward(
        self,
        ref_seq,
        text_seq,
        ref_bert,
        text_bert,
        ref_audio,
        ssl_content,
        sv_emb,
        top_k=20,
        top_p=0.6,
        temperature=0.6,
        repetition_penalty=1.35,
        speed=1.0,
    ):
        pred_semantic = self.t2s.forward(
            ref_seq,
            text_seq,
            ref_bert,
            text_bert,
            ssl_content,
            top_k=top_k,
            top_p=top_p,
            temperature=temperature,
            repetition_penalty=repetition_penalty,
        )
        if args.benchmark:
            start = int(round(time.time() * 1000))
        if args.onnx:
            audio1 = self.sess.run(
                None,
                {
                    "text_seq": text_seq,
                    "pred_semantic": pred_semantic,
                    "ref_audio": ref_audio,
                    "sv_emb": sv_emb,
                },
            )
        else:
            audio1 = self.sess.run(
                {
                    "text_seq": text_seq,
                    "pred_semantic": pred_semantic,
                    "ref_audio": ref_audio,
                    "sv_emb": sv_emb,
                }
            )
        if args.benchmark:
            end = int(round(time.time() * 1000))
            logger.info("\tvits processing time {} ms".format(end - start))
        return audio1[0]


class SSLModel:
    def __init__(self, sess):
        self.sess = sess

    def forward(self, ref_audio_16k):
        if args.benchmark:
            start = int(round(time.time() * 1000))
        if args.onnx:
            last_hidden_state = self.sess.run(None, {"ref_audio_16k": ref_audio_16k})
        else:
            last_hidden_state = self.sess.run({"ref_audio_16k": ref_audio_16k})
        if args.benchmark:
            end = int(round(time.time() * 1000))
            logger.info("\tssl processing time {} ms".format(end - start))
        return last_hidden_state[0]


def compute_kaldi_fbank(waveform, sample_rate=16000, num_mel_bins=80):
    """Compute Kaldi-compatible fbank features using numpy.

    Args:
        waveform: numpy array of shape (num_samples,), float32, range [-1, 1]
        sample_rate: sample rate in Hz
        num_mel_bins: number of mel filterbank bins

    Returns:
        fbank_feat: numpy array of shape (num_frames, num_mel_bins)
    """
    frame_length_ms = 25.0
    frame_shift_ms = 10.0
    preemph_coeff = 0.97
    low_freq = 20.0
    high_freq = sample_rate / 2.0

    frame_length = int(sample_rate * frame_length_ms / 1000)
    frame_shift = int(sample_rate * frame_shift_ms / 1000)

    # next power of 2
    n_fft = 1
    while n_fft < frame_length:
        n_fft *= 2

    # Frame the signal
    num_samples = len(waveform)
    num_frames = 1 + (num_samples - frame_length) // frame_shift
    if num_frames < 1:
        return np.zeros((0, num_mel_bins), dtype=np.float32)

    indices = np.arange(frame_length)[None, :] + np.arange(num_frames)[:, None] * frame_shift
    frames = waveform[indices]

    # Remove DC offset
    frames = frames - np.mean(frames, axis=1, keepdims=True)

    # Compute raw energy (before preemphasis)
    raw_energy = np.sum(frames ** 2, axis=1)
    raw_energy = np.maximum(raw_energy, np.finfo(np.float32).eps)
    log_energy = np.log(raw_energy)

    # Preemphasis
    preemph_frames = np.copy(frames)
    preemph_frames[:, 1:] = frames[:, 1:] - preemph_coeff * frames[:, :-1]
    frames = preemph_frames

    # Povey window (hann^0.85)
    window = np.hanning(frame_length).astype(np.float32) ** 0.85
    frames = frames * window[None, :]

    # Zero-pad to n_fft
    if n_fft > frame_length:
        frames = np.pad(frames, ((0, 0), (0, n_fft - frame_length)))

    # FFT
    spectrum = np.abs(np.fft.rfft(frames, n=n_fft))
    power_spectrum = spectrum ** 2

    # Mel filterbank
    num_fft_bins = n_fft // 2
    mel_low = 1127.0 * np.log(1.0 + low_freq / 700.0)
    mel_high = 1127.0 * np.log(1.0 + high_freq / 700.0)
    mel_delta = (mel_high - mel_low) / (num_mel_bins + 1)

    bins = np.arange(num_mel_bins)
    left_mel = mel_low + bins * mel_delta
    center_mel = mel_low + (bins + 1) * mel_delta
    right_mel = mel_low + (bins + 2) * mel_delta

    fft_bin_width = sample_rate / n_fft
    mel_freqs = 1127.0 * np.log(1.0 + fft_bin_width * np.arange(num_fft_bins) / 700.0)

    # (num_mel_bins, num_fft_bins)
    up_slope = (mel_freqs[None, :] - left_mel[:, None]) / (center_mel[:, None] - left_mel[:, None])
    down_slope = (right_mel[:, None] - mel_freqs[None, :]) / (right_mel[:, None] - center_mel[:, None])
    mel_banks = np.maximum(0.0, np.minimum(up_slope, down_slope)).astype(np.float32)

    # Pad to match spectrum size (num_fft_bins + 1)
    mel_banks = np.pad(mel_banks, ((0, 0), (0, 1)))

    # Apply mel filterbank
    mel_energies = power_spectrum @ mel_banks.T
    mel_energies = np.maximum(mel_energies, np.finfo(np.float32).eps)
    log_mel = np.log(mel_energies)

    return log_mel.astype(np.float32)


class SVModel:
    """Speaker Verification model (ERes2NetV2) for computing sv_emb."""

    def __init__(self, sess):
        self.sess = sess

    def forward(self, ref_audio_16k):
        """Compute speaker verification embedding from 16kHz audio.

        Args:
            ref_audio_16k: numpy array of shape (num_samples,), 16kHz audio

        Returns:
            sv_emb: numpy array of shape (1, 20480)
        """
        if args.benchmark:
            start = int(round(time.time() * 1000))

        fbank_feat = compute_kaldi_fbank(ref_audio_16k, sample_rate=16000, num_mel_bins=80)
        fbank_feat = fbank_feat[np.newaxis, :, :]  # (1, T, 80)

        if args.onnx:
            sv_emb = self.sess.run(None, {"fbank_feat": fbank_feat})
        else:
            sv_emb = self.sess.run({"fbank_feat": fbank_feat})

        if args.benchmark:
            end = int(round(time.time() * 1000))
            logger.info("\tsv processing time {} ms".format(end - start))
        return sv_emb[0]


def get_phones_and_bert(text, language, bert_model=None, bert_tokenizer=None, final=False):
    if language == "en":
        try:
            import LangSegment

            LangSegment.setfilters(["en"])
            formattext = " ".join(tmp["text"] for tmp in LangSegment.getTexts(text))
        except ImportError:
            formattext = text
    else:
        formattext = text
    while "  " in formattext:
        formattext = formattext.replace("  ", " ")

    phones, word2ph, norm_text = clean_text(formattext, language)
    phones = cleaned_text_to_sequence(phones)

    if language == "zh" and word2ph is not None and bert_model is not None:
        from text.bert_feature import get_bert_feature
        bert = get_bert_feature(norm_text, word2ph, bert_tokenizer, bert_model, args.onnx).T
        # .T: (num_phones, 1024) -> (1024, num_phones)
    else:
        bert = np.zeros((1024, len(phones)), dtype=np.float32)

    if not final and len(phones) < 6:
        return get_phones_and_bert("." + text, language, bert_model, bert_tokenizer, final=True)

    return phones, bert, norm_text


def generate_voice(ssl, t2s_encoder, t2s_first_decoder, t2s_stage_decoder, vits, sv, bert=None):
    gpt = T2SModel(
        t2s_encoder,
        t2s_first_decoder,
        t2s_stage_decoder,
    )
    gpt_sovits = GptSoVits(gpt, vits)
    ssl = SSLModel(ssl)
    sv = SVModel(sv)

    use_zh = args.text_language == "zh" or args.ref_language == "zh"
    bert_tokenizer = None
    if use_zh:
        from ailia_tokenizer import BertTokenizer
        bert_tokenizer = BertTokenizer.from_pretrained("tokenizer/")

    input_audio = args.ref_audio
    ref_text = args.ref_text
    ref_language = args.ref_language
    text = args.input
    text_language = args.text_language
    top_k = args.top_k
    top_p = args.top_p
    temperature = args.temperature
    speed = args.speed

    ref_text = ref_text.strip("\n")
    if ref_text[-1] not in splits:
        ref_text += "。" if ref_language != "en" else "."
    logger.info("Actual Input Reference Text: %s" % ref_text)

    text = text.strip("\n")
    logger.info("Actual Input Target Text: %s" % text)

    vits_hps_data_sampling_rate = 32000
    zero_wav = np.zeros(int(vits_hps_data_sampling_rate * 0.3), dtype=np.float16)

    ref_audio, sr = librosa.load(input_audio, sr=vits_hps_data_sampling_rate)

    ref_audio_16k = librosa.resample(ref_audio, orig_sr=sr, target_sr=16000)
    if ref_audio_16k.shape[0] > 160000 or ref_audio_16k.shape[0] < 48000:
        logger.warning(
            "Reference audio is outside the 3-10 second range, please choose another one!"
        )

    # Compute speaker verification embedding from 16kHz audio
    sv_emb = sv.forward(ref_audio_16k.astype(np.float32))

    # padding for hubert input
    ref_audio_16k = np.concatenate([ref_audio_16k, zero_wav], axis=0)
    ref_audio_16k = ref_audio_16k[np.newaxis, :]
    ssl_content = ssl.forward(ref_audio_16k)

    text = cut(text)  # Slice once every 4 sentences
    while "\n\n" in text:
        text = text.replace("\n\n", "\n")
    logger.info("Actual Input Target Text (after sentence segmentation): %s" % text)
    texts = text.split("\n")
    texts = process_text(texts)
    texts = merge_short_text_in_array(texts, 5)

    ref_seq, ref_bert, _ = get_phones_and_bert(ref_text, ref_language, bert, bert_tokenizer)
    ref_seq = np.array(ref_seq)[np.newaxis, :]

    ref_audio = ref_audio[np.newaxis, :]

    audio_opt = []
    for i_text, text in enumerate(texts):
        if len(text.strip()) == 0:
            continue
        if text[-1] not in splits:
            text += "。" if text_language != "en" else "."

        logger.info("Actual Input Target Text (per sentence): %s" % text)
        text_seq, text_bert, norm_text = get_phones_and_bert(text, text_language, bert, bert_tokenizer)
        text_seq = np.array(text_seq)[np.newaxis, :]
        logger.info("Processed text from the frontend (per sentence): %s" % norm_text)

        audio = gpt_sovits.forward(
            ref_seq,
            text_seq,
            ref_bert.T,
            text_bert.T,
            ref_audio,
            ssl_content,
            sv_emb,
            top_k=top_k,
            top_p=top_p,
            temperature=temperature,
            speed=speed,
        )

        max_audio = np.abs(audio).max()
        if max_audio > 1:
            audio /= max_audio
        audio_opt.append(audio)
        audio_opt.append(zero_wav)

    audio = (np.concatenate(audio_opt, 0) * 32768).astype(np.int16)

    savepath = args.savepath
    logger.info(f"saved at : {savepath}")
    soundfile.write(savepath, audio, vits_hps_data_sampling_rate)

    logger.info("Script finished successfully.")


def main():
    use_zh = args.text_language == "zh" or args.ref_language == "zh"

    # model files check and download
    check_and_download_models(WEIGHT_PATH_SSL, MODEL_PATH_SSL, REMOTE_PATH)
    check_and_download_models(WEIGHT_PATH_VITS, MODEL_PATH_VITS, REMOTE_PATH)
    check_and_download_models(WEIGHT_PATH_SV, MODEL_PATH_SV, REMOTE_PATH)
    check_and_download_models(
        WEIGHT_PATH_T2S_ENCODER, MODEL_PATH_T2S_ENCODER, REMOTE_PATH_T2S
    )
    check_and_download_models(
        WEIGHT_PATH_T2S_FIRST_DECODER, MODEL_PATH_T2S_FIRST_DECODER, REMOTE_PATH_T2S
    )
    check_and_download_models(
        WEIGHT_PATH_T2S_STAGE_DECODER, MODEL_PATH_T2S_STAGE_DECODER, REMOTE_PATH_T2S
    )
    if use_zh:
        check_and_download_models(WEIGHT_PATH_BERT, MODEL_PATH_BERT, REMOTE_PATH)

    env_id = args.env_id

    if args.onnx:
        import onnxruntime

        ssl = onnxruntime.InferenceSession(WEIGHT_PATH_SSL)
        t2s_encoder = onnxruntime.InferenceSession(WEIGHT_PATH_T2S_ENCODER)
        t2s_first_decoder = onnxruntime.InferenceSession(WEIGHT_PATH_T2S_FIRST_DECODER)
        t2s_stage_decoder = onnxruntime.InferenceSession(WEIGHT_PATH_T2S_STAGE_DECODER)
        vits = onnxruntime.InferenceSession(WEIGHT_PATH_VITS)
        sv_net = onnxruntime.InferenceSession(WEIGHT_PATH_SV)
        bert_net = onnxruntime.InferenceSession(WEIGHT_PATH_BERT) if use_zh else None
    else:
        memory_mode = ailia.get_memory_mode(
            reduce_constant=True,
            ignore_input_with_initializer=True,
            reduce_interstage=False,
            reuse_interstage=True,
        )
        ssl = ailia.Net(
            weight=WEIGHT_PATH_SSL,
            stream=MODEL_PATH_SSL,
            memory_mode=memory_mode,
            env_id=env_id,
        )
        t2s_encoder = ailia.Net(
            weight=WEIGHT_PATH_T2S_ENCODER,
            stream=MODEL_PATH_T2S_ENCODER,
            memory_mode=memory_mode,
            env_id=env_id,
        )
        t2s_first_decoder = ailia.Net(
            weight=WEIGHT_PATH_T2S_FIRST_DECODER,
            stream=MODEL_PATH_T2S_FIRST_DECODER,
            memory_mode=memory_mode,
            env_id=env_id,
        )
        t2s_stage_decoder = ailia.Net(
            weight=WEIGHT_PATH_T2S_STAGE_DECODER,
            stream=MODEL_PATH_T2S_STAGE_DECODER,
            memory_mode=memory_mode,
            env_id=env_id,
        )
        vits = ailia.Net(
            weight=WEIGHT_PATH_VITS,
            stream=MODEL_PATH_VITS,
            memory_mode=memory_mode,
            env_id=env_id,
        )
        sv_net = ailia.Net(
            weight=WEIGHT_PATH_SV,
            stream=MODEL_PATH_SV,
            memory_mode=memory_mode,
            env_id=env_id,
        )
        bert_net = ailia.Net(
            weight=WEIGHT_PATH_BERT,
            stream=MODEL_PATH_BERT,
            memory_mode=memory_mode,
            env_id=env_id,
        ) if use_zh else None
        if args.profile:
            ssl.set_profile_mode(True)
            t2s_encoder.set_profile_mode(True)
            t2s_first_decoder.set_profile_mode(True)
            t2s_stage_decoder.set_profile_mode(True)
            vits.set_profile_mode(True)
            sv_net.set_profile_mode(True)
            if bert_net is not None:
                bert_net.set_profile_mode(True)
        pf = platform.system()
        if pf == "Darwin":
            if args.env_id == 2:
                logger.info(
                    "This model not optimized for macOS GPU currently. Please try -e 1 option to improve inference speed."
                )

    if args.benchmark:
        start = int(round(time.time() * 1000))

    generate_voice(ssl, t2s_encoder, t2s_first_decoder, t2s_stage_decoder, vits, sv_net, bert_net)

    if args.benchmark:
        end = int(round(time.time() * 1000))
        logger.info("\ttotal processing time {} ms".format(end - start))

    if args.profile:
        print("ssl : ")
        print(ssl.get_summary())
        print("t2s_encoder : ")
        print(t2s_encoder.get_summary())
        print("t2s_first_decoder : ")
        print(t2s_first_decoder.get_summary())
        print("t2s_stage_decoder : ")
        print(t2s_stage_decoder.get_summary())
        print("vits : ")
        print(vits.get_summary())
        print("sv : ")
        print(sv_net.get_summary())
        if bert_net is not None:
            print("bert : ")
            print(bert_net.get_summary())


if __name__ == "__main__":
    main()
