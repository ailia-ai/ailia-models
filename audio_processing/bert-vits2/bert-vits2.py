import time
import sys
import os
import shutil

import numpy as np
import librosa

import ailia  # noqa: E402
sys.path.append('../../util')
from arg_utils import get_base_parser, update_parser  # noqa: E402
from model_utils import check_and_download_models, check_and_download_file  # noqa: E402

from vits2utils import text_normalize, g2p, intersperse, text2sep_kata, cleaned_text_to_sequence
from clap_feature_extraction import feature_extractor

# logger
from logging import getLogger   # noqa: E402
logger = getLogger(__name__)


class BenchmarkWrapper:
    def __init__(self, net, name):
        self.net = net
        self.name = name

    def predict(self, *args, **kwargs):
        start = int(round(time.time() * 1000))
        result = self.net.predict(*args, **kwargs)
        end = int(round(time.time() * 1000))
        logger.info("\t{} predict time {} ms".format(self.name, end - start))
        return result

    def __getattr__(self, name):
        return getattr(self.net, name)

from scipy.io.wavfile import write


# ======================
# Argument Parser Config
# ======================
DEFAULT_INPUT = '吾輩は猫である'
DEFAULT_EMO = "私はいまとても嬉しいです"
DEFAULT_OUTPUT = 'result.wav'
parser = get_base_parser('Bert-VITS2', None, DEFAULT_OUTPUT, fp16_support=False)

parser.add_argument(
    '--text',
    type=str,
    default=DEFAULT_INPUT,
    help='Text to be converted to audio'
)

parser.add_argument(
    '--emo',
    type=str,
    default=DEFAULT_EMO,
    help='Emotion text to be used for styling the audio'
)

parser.add_argument(
    '--emo-audio',
    type=str,
    default=None,
    help='Path to the audio file that is used to modify the emotion'
)


parser.add_argument(
    '--style-text',
    type=str,
    default="",
    help='Style text to be used for styling the audio'
)

parser.add_argument(
    '--sid',
    type=int,
    default=340,
    help='Speaker ID'
)

parser.add_argument(
    '--disable_ailia_tokenizer',
    action='store_true',
    help='disable ailia tokenizer.'
)

parser.add_argument(
    '--seed', default=1000, type=int,
    help='random seed'
)

args = update_parser(parser)


# ======================
# PARAMETERS
# ======================

PATHS = {
    'enc': 'BertVits2.2PT_enc_p.onnx',
    'emb_g': 'BertVits2.2PT_emb.onnx',
    'dp': 'BertVits2.2PT_dp.onnx',
    'sdp': 'BertVits2.2PT_sdp.onnx',
    'flow': 'BertVits2.2PT_flow.onnx',
    'dec': 'BertVits2.2PT_dec.onnx',
    'clap': 'emo_clap.onnx',
    'bert': 'debertav2lc.onnx',
    'clap_audio': 'emo_clap_audio.onnx'
}

MODEL_PATHS = {
    'enc': 'BertVits2.2PT_enc_p.onnx.prototxt',
    'emb_g': 'BertVits2.2PT_emb.onnx.prototxt',
    'dp': 'BertVits2.2PT_dp.onnx.prototxt',
    'sdp': 'BertVits2.2PT_sdp.onnx.prototxt',
    'flow': 'BertVits2.2PT_flow.onnx.prototxt',
    'dec': 'BertVits2.2PT_dec.onnx.prototxt',
    'clap': 'emo_clap.onnx.prototxt',
    'bert': 'debertav2lc.onnx.prototxt',
    'clap_audio': 'emo_clap_audio.onnx.prototxt'
}

MODEL_NAMES = ['enc', 'emb_g', 'dp', 'sdp', 'flow', 'dec', 'clap', 'bert', 'clap_audio']

REMOTE_PATH = "https://storage.googleapis.com/ailia-models/bert-vits2/"


# ======================
# helper functions
# ======================
def get_bert_features(bert, tokenizer, text, word2ph, style_text=None, style_weight=0.7):

    text = "".join(text2sep_kata(text)[0])
    if style_text:
        style_text = "".join(text2sep_kata(style_text)[0])
    
    inputs = dict(tokenizer(text, return_tensors="np"))
    res = bert.predict((inputs['input_ids'], inputs['attention_mask']))[0][0]

    if style_text:
        style_inputs = dict(tokenizer(style_text, return_tensors="np"))
        style_res = bert.predict((style_inputs['input_ids'], style_inputs['attention_mask']))[0][0]
        style_res_mean = style_res.mean(0)

    assert len(word2ph) == len(text) + 2
    phone_level_feature = []
    for i in range(len(word2ph)):
        if style_text:
            repeat_feature = (
                np.tile(res[i][:,None],reps=word2ph[i]) * (1 - style_weight)
                + np.tile(style_res_mean[:,None],reps=word2ph[i]) * style_weight
            )
        else:
            repeat_feature = np.tile(res[i][:,None], word2ph[i])
        phone_level_feature.append(repeat_feature)

    phone_level_feature = np.concatenate(phone_level_feature, axis=1)

    return phone_level_feature[None]

def prepare_inputs(text, style_text, sid, emo, emo_audio_path, bert, tokenizer, clap_text, clap_tokenizer, clap_audio):
    text = text_normalize(text)
    phones, tones, word2ph = g2p(text, tokenizer)
    x, tones, lang_ids = cleaned_text_to_sequence(phones, tones,'JP')

    x = intersperse(x, 0)
    tones = intersperse(tones, 0)
    lang_ids = intersperse(lang_ids, 0)

    for i in range(len(word2ph)):
        word2ph[i] = word2ph[i] * 2
    word2ph[0] += 1
    
    bert = get_bert_features(bert, tokenizer, text, word2ph, style_text)

    x = np.array(x)[None]
    tones = np.array(tones)[None]
    lang_ids = np.array(lang_ids)[None]
    sid = np.array([sid])

    if emo_audio_path is not None:
        emo_audio, _ = librosa.load(emo_audio_path, sr=48000, mono=True)
        input_feature, longer = feature_extractor(emo_audio)
        input_feature = input_feature[None]
        longer = np.asarray(longer)[None]
        emo_embed = clap_audio.predict(
            (input_feature, longer)
        )[0]
    elif emo is not None:
        emo_input = clap_tokenizer(emo, return_tensors="np")
        emo_embed = clap_text.predict({
            'input_ids': emo_input['input_ids'],
            'attention_mask': emo_input['attention_mask']
        })[0]
    else:
        raise ValueError('At least one of emo or emo_audio must be present')

    return x, tones, lang_ids, bert, sid, emo_embed

def generate_path(duration, mask):
    """
    duration: [b, 1, t_x]
    mask: [b, 1, t_y, t_x]
    """

    b, _, t_y, t_x = mask.shape
    cum_duration = np.cumsum(duration, -1)

    cum_duration_flat = cum_duration.reshape(b * t_x)
    path = sequence_mask(cum_duration_flat, t_y)
    path = path.reshape(b, t_x, t_y)
    path = path ^ np.pad(path, ((0, 0), (1, 0), (0, 0)))[:, :-1]
    path = np.expand_dims(path, 1).transpose(0, 1, 3, 2)
    return path


def sequence_mask(length, max_length=None):
    if max_length is None:
        max_length = length.max()
    x = np.arange(max_length, dtype=length.dtype)
    return np.expand_dims(x, 0) < np.expand_dims(length, 1)

# ======================
# Main function
# ======================

def predict(inputs, models, noise_scale=0.667, length_scale=1.0, noise_scale_w=0.8, sdp_ratio=0.0):
    x, tone, language, bert, sid, emo = inputs['x'], inputs['tone'], inputs['language'], inputs['bert'], inputs['sid'], inputs['emo']
    enc, emb_g, dp, sdp, flow, dec = models['enc'], models['emb_g'], models['dp'], models['sdp'], models['flow'], models['dec']
    g = emb_g.predict({
        'sid': sid
    })[0]
    g = np.expand_dims(g, -1)
    enc_rtn = enc.predict({
        'x': x,
        'x_lengths': np.array([x.shape[1]]),
        'tone': tone,
        'language': language,
        'bert': bert,#np.zeros_like(bert_zh),
        'emo': emo,
        'g':g
    })

    
    x, m_p, logs_p, x_mask = enc_rtn[0], enc_rtn[1], enc_rtn[2], enc_rtn[3]
    logw = sdp.predict(
        {
            'x': x,
            'x_mask': x_mask,
            'g': g
        }
    )[0] * (sdp_ratio) + dp.predict({
            'x': x,
            'x_mask': x_mask,
            'g': g
        })[0] * (1 - sdp_ratio)
    w = np.exp(logw) * x_mask * length_scale
    w_ceil = np.ceil(w)
    y_lengths = np.clip(np.sum(w_ceil, (1, 2)), a_min=1.0, a_max=100000).astype(
        np.int64
    )
    y_mask = np.expand_dims(sequence_mask(y_lengths, None), 1)
    attn_mask = np.expand_dims(x_mask, 2) * np.expand_dims(y_mask, -1)
    attn = generate_path(w_ceil, attn_mask)
    m_p = np.matmul(attn.squeeze(1), m_p.transpose(0, 2, 1)).transpose(
        0, 2, 1
    )  # [b, t', t], [b, t, d] -> [b, d, t']
    logs_p = np.matmul(attn.squeeze(1), logs_p.transpose(0, 2, 1)).transpose(
        0, 2, 1
    )  # [b, t', t], [b, t, d] -> [b, d, t']
    z_p = (
        m_p
        + np.random.randn(m_p.shape[0], m_p.shape[1], m_p.shape[2])
        * np.exp(logs_p)
        * noise_scale
    )
    z = flow.predict({
            "z_p": z_p.astype(np.float32),
            "y_mask": y_mask.astype(np.float32),
            "g": g,
    })[0]
    return dec.predict({"z_in": (z * y_mask), "g": g})[0]

def infer(models):
    text = args.text
    emo = args.emo
    emo_audio_path = args.emo_audio
    style_text = args.style_text
    sid = args.sid

    phones, tones, lang_ids, bert, sid, emo = prepare_inputs(
        text,
        style_text,
        sid,
        emo,
        emo_audio_path,
        models['bert'],
        models['bert_tokenizer'],
        models['clap'],
        models['clap_tokenizer'],
        models['clap_audio']
    )
    inputs = {
        "x": phones,
        "tone": tones,
        "language": lang_ids,
        "bert": bert,
        "sid": sid,
        "emo": emo
    }

    raw_wav = predict(inputs, models)

    write(filename=args.savepath, rate=44100, data=raw_wav)

def main():
    # load models
    if args.disable_ailia_tokenizer:
        from transformers import AutoTokenizer
        bert_tokenizer = AutoTokenizer.from_pretrained("ku-nlp/deberta-v2-large-japanese-char-wwm")
        clap_tokenizer = AutoTokenizer.from_pretrained('laion/clap-htsat-fused')
    else:
        # Originally it is Juman++, but because it is a char tokenizer, any previous Japanese word segmentation should work.
        import ailia_tokenizer
        DICT_REMOTE_PATH = "https://storage.googleapis.com/ailia-models/bert_maskedlm/"
        check_and_download_file("unidic-lite.zip", DICT_REMOTE_PATH)
        if not os.path.exists("unidic-lite"):
            shutil.unpack_archive('unidic-lite.zip', '')
        bert_tokenizer = ailia_tokenizer.BertJapaneseCharacterTokenizer.from_pretrained(dict_path = 'unidic-lite', pretrained_model_name_or_path = "./tokenizer/deberta-v2-large-japanese-char-wwm")
        clap_tokenizer = ailia_tokenizer.RobertaTokenizer.from_pretrained('./tokenizer/clap-htsat-fused')
    models = {'bert_tokenizer': bert_tokenizer,'clap_tokenizer': clap_tokenizer}

    for m in MODEL_NAMES:
        check_and_download_models(
            PATHS[m], MODEL_PATHS[m], REMOTE_PATH
        )
        net = ailia.Net(MODEL_PATHS[m], PATHS[m], args.env_id)
        if args.benchmark:
            net = BenchmarkWrapper(net, m)
        models[m] = net

    if args.benchmark:
        start = int(round(time.time() * 1000))
    infer(models)
    if args.benchmark:
        end = int(round(time.time() * 1000))
        logger.info("\ttotal processing time {} ms".format(end-start))

    logger.info('Script finished successfully.')


if __name__ == "__main__":
    np.random.seed(args.seed)
    main()
