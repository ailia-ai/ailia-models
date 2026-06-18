import time
import sys

# logger
from logging import getLogger

import librosa
import numpy as np
import soundfile
import platform

# import original modules
sys.path.append("../../util")
from arg_utils import get_base_parser, update_parser
from model_utils import check_and_download_models

import ailia
from text import cleaned_text_to_sequence
from text.cleaner import clean_text


logger = getLogger(__name__)

# ======================
# PARAMETERS
# ======================

REMOTE_PATH = "https://storage.googleapis.com/ailia-models/gpt-sovits-v3/"
WEIGHT_PATH_SSL = "cnhubert.onnx"
WEIGHT_PATH_T2S_ENCODER = "t2s_encoder.onnx"
WEIGHT_PATH_T2S_FIRST_DECODER = "t2s_fsdec.onnx"
WEIGHT_PATH_T2S_STAGE_DECODER = "t2s_sdec.opt.onnx"
WEIGHT_PATH_VQ = "vq_model.onnx"
WEIGHT_PATH_CFM = "vq_cfm.onnx"
WEIGHT_PATH_VGAN = "bigvgan_model.onnx"
WEIGHT_PATH_BERT = "chinese-roberta.onnx"
MODEL_PATH_SSL = WEIGHT_PATH_SSL + ".prototxt"
MODEL_PATH_T2S_ENCODER = WEIGHT_PATH_T2S_ENCODER + ".prototxt"
MODEL_PATH_T2S_FIRST_DECODER = WEIGHT_PATH_T2S_FIRST_DECODER + ".prototxt"
MODEL_PATH_T2S_STAGE_DECODER = WEIGHT_PATH_T2S_STAGE_DECODER + ".prototxt"
MODEL_PATH_VQ = WEIGHT_PATH_VQ + ".prototxt"
MODEL_PATH_CFM = WEIGHT_PATH_CFM + ".prototxt"
MODEL_PATH_VGAN = WEIGHT_PATH_VGAN + ".prototxt"
MODEL_PATH_BERT = WEIGHT_PATH_BERT + ".prototxt"

REF_WAV_PATH = "reference_audio_captured_by_ax.wav"
REF_TEXT = "水をマレーシアから買わなくてはならない。"
SAVE_WAV_PATH = "output.wav"


# ======================
# Arguemnt Parser Config
# ======================

parser = get_base_parser("GPT-SoVits", None, SAVE_WAV_PATH)
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
parser.add_argument(
    "--sample_steps", type=int, default=4,
    help="Number of CFM sampling steps (original: 32, default: 4)"
)
parser.add_argument("--onnx", action="store_true", help="use onnx runtime")
parser.add_argument("--profile", action="store_true", help="use profile model")
args = update_parser(parser, check_input_type=False)


COPY_BLOB_DATA = True

splits = {
    # fmt: off
    "，", "。", "？", "！", ",", ".", "?", "!", "~", ":", "：", "—", "…",
    # fmt: on
}


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


def get_spepc(audio):
    maxx = np.abs(audio).max()
    if maxx > 1:
        audio /= min(2, maxx)
    spec = spectrogram(
        audio,
        n_fft=2048,
        win_size=2048,
        hop_size=640,
        num_mels=100,
        sampling_rate=32000,
        center=False,
        eps=1e-6,
        mel_filter=False,
    )
    return spec


def spectrogram(
    y,
    n_fft,
    num_mels,
    sampling_rate,
    hop_size,
    win_size,
    fmin=0,
    fmax=1,
    center=False,
    eps=1e-9,
    mel_filter=True,
):
    p = int((n_fft - hop_size) / 2)
    y = np.pad(y, pad_width=((0, 0), (p, p)), mode="reflect")

    spec = librosa.stft(
        y,
        n_fft=n_fft,
        hop_length=hop_size,
        win_length=win_size,
        window="hann",
        center=center,
        pad_mode="reflect",
    )
    spec = np.sqrt(np.abs(spec) ** 2 + eps)

    if mel_filter:
        mel_basis = librosa.filters.mel(
            sr=sampling_rate, n_fft=n_fft, n_mels=num_mels, fmin=fmin, fmax=fmax
        )
        spec = np.matmul(mel_basis, spec)
        spec = np.log(np.clip(spec, a_min=1e-5, a_max=None))

    return spec


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

        if not args.onnx:
            x, prompts = self.sess_encoder.run(
                {
                    "ref_seq": ref_seq,
                    "text_seq": text_seq,
                    "ref_bert": ref_bert,
                    "text_bert": text_bert,
                    "ssl_content": ssl_content,
                }
            )
        else:
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

        if args.benchmark:
            end = int(round(time.time() * 1000))
            logger.info("\tsencoder processing time {} ms".format(end - start))

        prefix_len = prompts.shape[1]

        if args.benchmark:
            start = int(round(time.time() * 1000))

        if not args.onnx:
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
        else:
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

        if args.benchmark:
            end = int(round(time.time() * 1000))
            logger.info("\tfsdec processing time {} ms".format(end - start))

        logger.info(f"T2S Decoding...")

        stop = False
        for idx in range(1, 1500):
            if args.benchmark:
                start = int(round(time.time() * 1000))

            if not args.onnx:
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
            else:
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

            if args.benchmark:
                end = int(round(time.time() * 1000))
                logger.info("\tsdec processing time {} ms".format(end - start))

            if early_stop_num != -1 and (y.shape[1] - prefix_len) > early_stop_num:
                stop = True
            if np.argmax(logits, axis=-1)[0] == EOS or samples[0, 0] == EOS:
                stop = True
            if stop:
                logger.info(f"T2S Decoding EOS [{prefix_len} -> {y.shape[1]}]")
                break
        y[0, -1] = 0

        return y[np.newaxis, :, -idx:-1], prompts


class GptSoVits:
    def __init__(self, t2s: T2SModel, vq, vq_cfm, vgan):
        self.t2s = t2s
        self.vq = vq
        self.vq_cfm = vq_cfm
        self.vgan = vgan

    def cfm_inference(self, mu, x_lens, prompt, n_timesteps, temperature=1.0):
        """Forward diffusion"""
        B, T, _ = mu.shape
        in_channels = 100
        x = np.random.randn(B, in_channels, T) * temperature
        x = x.astype(mu.dtype)
        prompt_len = prompt.shape[-1]
        prompt_x = np.zeros_like(x, dtype=mu.dtype)
        prompt_x[..., :prompt_len] = prompt[..., :prompt_len]
        x[..., :prompt_len] = 0
        mu = mu.transpose(0, 2, 1)
        t = 0
        d = 1 / n_timesteps
        for i in range(n_timesteps):
            if args.benchmark:
                start = int(round(time.time() * 1000))

            t_tensor = np.ones(x.shape[0], dtype=mu.dtype) * t
            d_tensor = np.ones(x.shape[0], dtype=mu.dtype) * d
            if not args.onnx:
                self.vq_cfm.set_input_blob_data(x, 0)
                self.vq_cfm.set_input_blob_data(prompt_x, 1)
                self.vq_cfm.set_input_blob_data(x_lens, 2)
                self.vq_cfm.set_input_blob_data(t_tensor, 3)
                self.vq_cfm.set_input_blob_data(d_tensor, 4)
                self.vq_cfm.set_input_blob_data(mu, 5)
                self.vq_cfm.update()
                v_pred = self.vq_cfm.get_blob_data("output")
            else:
                output = self.vq_cfm.run(
                    None,
                    {
                        "x": x,
                        "cond": prompt_x,
                        "x_lens": x_lens,
                        "time": t_tensor,
                        "dt_base_bootstrap": d_tensor,
                        "text": mu,
                    },
                )
                v_pred = output[0]

            if args.benchmark:
                end = int(round(time.time() * 1000))
                logger.info("\tcfm processing time {} ms".format(end - start))

            v_pred = v_pred.transpose(0, 2, 1)
            x = x + d * v_pred
            t = t + d
            x[:, :, :prompt_len] = 0

        return x

    def forward(
        self,
        ref_seq,
        text_seq,
        ref_bert,
        text_bert,
        ref_audio,
        ssl_content,
        top_k=20,
        top_p=0.6,
        temperature=0.6,
        repetition_penalty=1.35,
        speed=1.0,
    ):
        sample_steps = args.sample_steps  # original: 32

        pred_semantic, prompt = self.t2s.forward(
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

        refer = get_spepc(ref_audio).astype(np.float16)
        ge_0 = np.zeros((0, 512, 1), dtype=np.float16)

        if not args.onnx:
            output = self.vq.run(
                {
                    "codes": prompt[None, ...],
                    "text": ref_seq,
                    "refer": refer,
                    "ge": ge_0,
                    "speed": np.array(1, dtype=np.float32),
                },
            )
        else:
            output = self.vq.run(
                None,
                {
                    "codes": prompt[None, ...],
                    "text": ref_seq,
                    "refer": refer,
                    "ge": ge_0,
                    "speed": np.array(1, dtype=np.float32),
                },
            )
        fea_ref, ge = output

        ref_audio_24k = librosa.resample(ref_audio, orig_sr=32000, target_sr=24000)
        mel2 = spectrogram(
            ref_audio_24k,
            n_fft=1024,
            win_size=1024,
            hop_size=256,
            num_mels=100,
            sampling_rate=24000,
            fmin=0,
            fmax=None,
            center=False,
        )

        spec_min = -12
        spec_max = 2
        # norm_spec
        mel2 = (mel2 - spec_min) / (spec_max - spec_min) * 2 - 1

        T_min = min(mel2.shape[2], fea_ref.shape[2])
        mel2 = mel2[:, :, :T_min]
        fea_ref = fea_ref[:, :, :T_min]
        if T_min > 468:
            mel2 = mel2[:, :, -468:]
            fea_ref = fea_ref[:, :, -468:]
            T_min = 468
        chunk_len = 934 - T_min

        if not args.onnx:
            output = self.vq.run(
                {
                    "codes": pred_semantic,
                    "text": text_seq,
                    "refer": refer,
                    "ge": ge,
                    "speed": np.array(speed, dtype=np.float32),
                },
            )
        else:
            output = self.vq.run(
                None,
                {
                    "codes": pred_semantic,
                    "text": text_seq,
                    "refer": refer,
                    "ge": ge,
                    "speed": np.array(speed, dtype=np.float32),
                },
            )
        fea_todo, _ = output

        logger.info("vq_model cfm inference...")

        cfm_resss = []
        idx = 0
        while True:
            fea_todo_chunk = fea_todo[:, :, idx : idx + chunk_len]
            if fea_todo_chunk.shape[-1] == 0:
                break
            idx += chunk_len
            fea = np.concatenate([fea_ref, fea_todo_chunk], axis=2).transpose(0, 2, 1)
            cfm_res = self.cfm_inference(
                fea, np.array([fea.shape[1]]), mel2, sample_steps
            )
            cfm_res = cfm_res[:, :, mel2.shape[2] :]
            mel2 = cfm_res[:, :, -T_min:]
            fea_ref = fea_todo_chunk[:, :, -T_min:]
            cfm_resss.append(cfm_res)

        cmf_res = np.concatenate(cfm_resss, axis=2)
        # denorm_spec
        cmf_res = (cmf_res + 1) / 2 * (spec_max - spec_min) + spec_min

        logger.info("bigvgan inference...")

        if not args.onnx:
            output = self.vgan.run(
                {"x": cmf_res},
            )
        else:
            output = self.vgan.run(
                None,
                {"x": cmf_res},
            )
        audio = output[0]

        return audio[0][0]


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


def generate_voice(ssl, models, bert=None):
    gpt = T2SModel(
        models["t2s_encoder"], models["t2s_first_decoder"], models["t2s_stage_decoder"]
    )
    gpt_sovits = GptSoVits(gpt, models["vq"], models["vq_cfm"], models["vgan"])
    ssl = SSLModel(ssl)

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

    # hubertの入力のみpaddingする
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
    for _, text in enumerate(texts):
        # 解决输入目标文本的空行导致报错的问题
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
    soundfile.write(savepath, audio, 24000)

    logger.info("Script finished successfully.")


def main():
    use_zh = args.text_language == "zh" or args.ref_language == "zh"

    # model files check and download
    check_and_download_models(WEIGHT_PATH_SSL, MODEL_PATH_SSL, REMOTE_PATH)
    check_and_download_models(
        WEIGHT_PATH_T2S_ENCODER, MODEL_PATH_T2S_ENCODER, REMOTE_PATH
    )
    check_and_download_models(
        WEIGHT_PATH_T2S_FIRST_DECODER, MODEL_PATH_T2S_FIRST_DECODER, REMOTE_PATH
    )
    check_and_download_models(
        WEIGHT_PATH_T2S_STAGE_DECODER, MODEL_PATH_T2S_STAGE_DECODER, REMOTE_PATH
    )
    check_and_download_models(WEIGHT_PATH_VQ, MODEL_PATH_VQ, REMOTE_PATH)
    check_and_download_models(WEIGHT_PATH_CFM, MODEL_PATH_CFM, REMOTE_PATH)
    check_and_download_models(WEIGHT_PATH_VGAN, MODEL_PATH_VGAN, REMOTE_PATH)
    if use_zh:
        BERT_REMOTE_PATH = "https://storage.googleapis.com/ailia-models/gpt-sovits-v3/"
        check_and_download_models(WEIGHT_PATH_BERT, MODEL_PATH_BERT, BERT_REMOTE_PATH)

    env_id = args.env_id

    if not args.onnx:
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
        vq = ailia.Net(
            weight=WEIGHT_PATH_VQ,
            stream=MODEL_PATH_VQ,
            memory_mode=memory_mode,
            env_id=env_id,
        )
        vq_cfm = ailia.Net(
            weight=WEIGHT_PATH_CFM,
            stream=MODEL_PATH_CFM,
            memory_mode=memory_mode,
            env_id=env_id,
        )
        vgan = ailia.Net(
            weight=WEIGHT_PATH_VGAN,
            stream=MODEL_PATH_VGAN,
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
            vq.set_profile_mode(True)
            vq_cfm.set_profile_mode(True)
            vgan.set_profile_mode(True)
            if bert_net is not None:
                bert_net.set_profile_mode(True)
        pf = platform.system()
        if pf == "Darwin":
            if args.env_id == 2:
                logger.info(
                    "This model not optimized for macOS GPU currently. Please try -e 1 option to improve inference speed."
                )
    else:
        import onnxruntime

        ssl = onnxruntime.InferenceSession(WEIGHT_PATH_SSL)
        t2s_encoder = onnxruntime.InferenceSession(WEIGHT_PATH_T2S_ENCODER)
        t2s_first_decoder = onnxruntime.InferenceSession(WEIGHT_PATH_T2S_FIRST_DECODER)
        t2s_stage_decoder = onnxruntime.InferenceSession(WEIGHT_PATH_T2S_STAGE_DECODER)
        vq = onnxruntime.InferenceSession(WEIGHT_PATH_VQ)
        vq_cfm = onnxruntime.InferenceSession(WEIGHT_PATH_CFM)
        vgan = onnxruntime.InferenceSession(WEIGHT_PATH_VGAN)
        bert_net = onnxruntime.InferenceSession(WEIGHT_PATH_BERT) if use_zh else None

    models = dict(
        ssl=ssl,
        t2s_encoder=t2s_encoder,
        t2s_first_decoder=t2s_first_decoder,
        t2s_stage_decoder=t2s_stage_decoder,
        vq=vq,
        vq_cfm=vq_cfm,
        vgan=vgan,
    )

    if args.benchmark:
        start = int(round(time.time() * 1000))

    generate_voice(ssl, models, bert_net)

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
        print("vq : ")
        print(vq.get_summary())
        print("vq_cfm : ")
        print(vq_cfm.get_summary())
        print("vgan : ")
        print(vgan.get_summary())
        if bert_net is not None:
            print("bert : ")
            print(bert_net.get_summary())


if __name__ == "__main__":
    main()
