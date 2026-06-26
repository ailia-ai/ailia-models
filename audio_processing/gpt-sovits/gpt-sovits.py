import time
import sys
import platform

import numpy as np
import soundfile as sf

# import original modules
sys.path.append('../../util')
from arg_utils import get_base_parser, update_parser, get_savepath  # noqa: E402
from model_utils import check_and_download_models  # noqa: E402
from scipy.io.wavfile import write
# logger
from logging import getLogger   # noqa: E402
logger = getLogger(__name__)

from text import cleaned_text_to_sequence
import soundfile
import librosa

# ======================
# PARAMETERS
# ======================

SAVE_WAV_PATH = 'output.wav'
REMOTE_PATH = 'https://storage.googleapis.com/ailia-models/gpt-sovits/'

# ======================
# Arguemnt Parser Config
# ======================

parser = get_base_parser( 'GPT-SoVits', None, SAVE_WAV_PATH)
# overwrite
parser.add_argument(
    '--input', '-i', metavar='TEXT', default="ax株式会社ではAIの実用化のための技術を開発しています。",
    help='input text'
)
parser.add_argument(
    '--text_language', '-tl',
    default='ja',
    help='[ja, en, zh]'
)
parser.add_argument(
    '--ref_audio', '-ra', metavar='TEXT', default="reference_audio_captured_by_ax.wav",
    help='ref audio'
)
parser.add_argument(
    '--ref_text', '-rt', metavar='TEXT', default="水をマレーシアから買わなくてはならない。",
    help='ref text'
)
parser.add_argument(
    '--ref_language', '-rl',
    default='ja',
    help='[ja, en, zh]'
)
parser.add_argument(
    '--onnx', action='store_true',
    help='use onnx runtime'
)
parser.add_argument(
    '--normal', action='store_true',
    help='use normal model'
)
parser.add_argument(
    '--profile', action='store_true',
    help='use profile model'
)
parser.add_argument(
    '--ailia_voice', action='store_true',
    help='use ailia voice for G2P'
)
parser.add_argument(
    "--fp16", action="store_true", help="use fp16 model (default : fp32 model)."
)
args = update_parser(parser, check_input_type=False)

FP16_PREFIX = ""
if args.fp16:
    FP16_PREFIX = "_fp16"

WEIGHT_PATH_SSL = 'cnhubert' + FP16_PREFIX + '.onnx'
WEIGHT_PATH_T2S_ENCODER = 't2s_encoder' + FP16_PREFIX + '.onnx'
WEIGHT_PATH_T2S_FIRST_DECODER = 't2s_fsdec' + FP16_PREFIX + '.onnx'
if args.normal:
    WEIGHT_PATH_T2S_STAGE_DECODER = 't2s_sdec.onnx'
else:
    WEIGHT_PATH_T2S_STAGE_DECODER = 't2s_sdec' + FP16_PREFIX + '.opt3.onnx'
WEIGHT_PATH_VITS = 'vits' + FP16_PREFIX + '.onnx'
WEIGHT_PATH_BERT = 'chinese-roberta.onnx'

MODEL_PATH_SSL = WEIGHT_PATH_SSL + '.prototxt'
MODEL_PATH_T2S_ENCODER = WEIGHT_PATH_T2S_ENCODER + '.prototxt'
MODEL_PATH_T2S_FIRST_DECODER = WEIGHT_PATH_T2S_FIRST_DECODER + '.prototxt'
MODEL_PATH_T2S_STAGE_DECODER = WEIGHT_PATH_T2S_STAGE_DECODER + '.prototxt'
MODEL_PATH_VITS = WEIGHT_PATH_VITS + '.prototxt'
MODEL_PATH_BERT = WEIGHT_PATH_BERT + '.prototxt'

# ======================
# Mode
# ======================

if not args.onnx:
    import ailia
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
# Logic
# ======================

class T2SModel():
    def __init__(self, sess_encoder, sess_fsdec, sess_sdec):
        self.hz = 50
        self.max_sec = 54
        self.top_k = 5
        self.early_stop_num = np.array([self.hz * self.max_sec])
        self.sess_encoder = sess_encoder
        self.sess_fsdec = sess_fsdec
        self.sess_sdec = sess_sdec

    def forward(self, ref_seq, text_seq, ref_bert, text_bert, ssl_content):
        early_stop_num = self.early_stop_num

        top_k = np.array([5], dtype=np.int64)
        top_p = np.array([1.0], dtype=np.float32)
        temperature = np.array([1.0], dtype=np.float32)
        repetition_penalty = np.array([1.35], dtype=np.float32)

        EOS = 1024

        if args.benchmark:
            start = int(round(time.time() * 1000))
        if args.onnx:
            x, prompts = self.sess_encoder.run(None, {"ref_seq":ref_seq, "text_seq":text_seq, "ref_bert":ref_bert, "text_bert":text_bert, "ssl_content":ssl_content})
        else:
            x, prompts = self.sess_encoder.run({"ref_seq":ref_seq, "text_seq":text_seq, "ref_bert":ref_bert, "text_bert":text_bert, "ssl_content":ssl_content})
        if args.benchmark:
            end = int(round(time.time() * 1000))
            logger.info("\tsencoder processing time {} ms".format(end-start))

        prefix_len = prompts.shape[1]

        if args.benchmark:
            start = int(round(time.time() * 1000))
        if args.onnx:
            y, k, v, y_emb, x_example = self.sess_fsdec.run(None, {"x":x, "prompts":prompts, "top_k":top_k, "top_p":top_p, "temperature":temperature, "repetition_penalty":repetition_penalty})
        else:
            y, k, v, y_emb, x_example = self.sess_fsdec.run({"x":x, "prompts":prompts, "top_k":top_k, "top_p":top_p, "temperature":temperature, "repetition_penalty":repetition_penalty})
        if args.benchmark:
            end = int(round(time.time() * 1000))
            logger.info("\tfsdec processing time {} ms".format(end-start))

        stop = False
        for idx in range(1, 1500):
            if args.benchmark:
                start = int(round(time.time() * 1000))
            if args.onnx:
                y, k, v, y_emb, logits, samples = self.sess_sdec.run(None, {"iy":y, "ik":k, "iv":v, "iy_emb":y_emb, "ix_example":x_example, "top_k":top_k, "top_p":top_p, "temperature":temperature, "repetition_penalty":repetition_penalty})
            else:
                if idx == 1:
                    y, k, v, y_emb, logits, samples = self.sess_sdec.run({"iy":y, "ik":k, "iv":v, "iy_emb":y_emb, "ix_example":x_example, "top_k":top_k, "top_p":top_p, "temperature":temperature, "repetition_penalty":repetition_penalty})
                    kv_base_shape = k.shape
                else:
                    input_blob_idx = self.sess_sdec.get_input_blob_list()
                    output_blob_idx = self.sess_sdec.get_output_blob_list()
                    self.sess_sdec.set_input_blob_data(y, 0)
                    if COPY_BLOB_DATA:
                        kv_shape = (kv_base_shape[0], kv_base_shape[1] + idx - 2, kv_base_shape[2], kv_base_shape[3])
                        self.sess_sdec.set_input_blob_shape(kv_shape, 1)
                        self.sess_sdec.set_input_blob_shape(kv_shape, 2)
                        self.sess_sdec.copy_blob_data(input_blob_idx[1], output_blob_idx[1], self.sess_sdec)
                        self.sess_sdec.copy_blob_data(input_blob_idx[2], output_blob_idx[2], self.sess_sdec)
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
                logger.info("\tsdec processing time {} ms".format(end-start))
            if early_stop_num != -1 and (y.shape[1] - prefix_len) > early_stop_num:
                stop = True
            if np.argmax(logits, axis=-1)[0] == EOS or samples[0, 0] == EOS:
                stop = True
            if stop:
                break
        y[0, -1] = 0

        return y[np.newaxis, :, -idx:-1]


class GptSoVits():
    def __init__(self, t2s, sess):
        self.t2s = t2s
        self.sess = sess
    
    def forward(self, ref_seq, text_seq, ref_bert, text_bert, ref_audio, ssl_content):
        pred_semantic = self.t2s.forward(ref_seq, text_seq, ref_bert, text_bert, ssl_content)
        if args.benchmark:
            start = int(round(time.time() * 1000))
        if args.onnx:
            audio1 = self.sess.run(None, {
                "text_seq" : text_seq,
                "pred_semantic" : pred_semantic, 
                "ref_audio" : ref_audio
            })
        else:
            audio1 = self.sess.run({
                "text_seq" : text_seq,
                "pred_semantic" : pred_semantic, 
                "ref_audio" : ref_audio
            })
        if args.benchmark:
            end = int(round(time.time() * 1000))
            logger.info("\tvits processing time {} ms".format(end-start))
        return audio1[0]


class SSLModel():
    def __init__(self, sess):
        self.sess = sess

    def forward(self, ref_audio_16k):
        if args.benchmark:
            start = int(round(time.time() * 1000))
        if args.onnx:
            last_hidden_state = self.sess.run(None, {
                "ref_audio_16k" : ref_audio_16k
            })
        else:
            last_hidden_state = self.sess.run({
                "ref_audio_16k" : ref_audio_16k
            })
        if args.benchmark:
            end = int(round(time.time() * 1000))
            logger.info("\tssl processing time {} ms".format(end-start))
        return last_hidden_state[0]


def generate_voice(ssl, t2s_encoder, t2s_first_decoder, t2s_stage_decoder, vits, bert=None):
    gpt = T2SModel(t2s_encoder, t2s_first_decoder, t2s_stage_decoder,)
    gpt_sovits = GptSoVits(gpt, vits)
    ssl = SSLModel(ssl)

    input_audio = args.ref_audio

    use_zh = args.text_language == "zh" or args.ref_language == "zh"

    if use_zh:
        import text.chinese as chinese
        from text.bert_feature import get_bert_feature
        from ailia_tokenizer import BertTokenizer
        bert_tokenizer = BertTokenizer.from_pretrained("tokenizer/")

    if args.ailia_voice:
        import ailia_voice
        voice = ailia_voice.G2P()
        voice.initialize_model(model_path = "./models/")
    else:
        import text.japanese as japanese
        import text.english as english

    ref_word2ph = None
    if args.ref_language == "zh":
        ref_norm_text = chinese.text_normalize(args.ref_text)
        ref_phones, ref_word2ph = chinese.g2p(ref_norm_text)
    elif args.ref_language == "ja":
        if args.ailia_voice:
            ref_phones = voice.g2p(args.ref_text, ailia_voice.AILIA_VOICE_G2P_TYPE_GPT_SOVITS_JA).split(" ")[:-1]
        else:
            ref_phones = japanese.g2p(args.ref_text)
    else:
        if args.ailia_voice:
            ref_phones = voice.g2p(args.ref_text, ailia_voice.AILIA_VOICE_G2P_TYPE_GPT_SOVITS_EN).split(" ")[:-1]
        else:
            ref_phones = english.g2p(args.ref_text)
    ref_seq = np.array([cleaned_text_to_sequence(ref_phones)], dtype=np.int64)

    text_word2ph = None
    if args.text_language == "zh":
        text_norm_text = chinese.text_normalize(args.input)
        text_phones, text_word2ph = chinese.g2p(text_norm_text)
    elif args.text_language == "ja":
        if args.ailia_voice:
            text_phones = voice.g2p(args.input, ailia_voice.AILIA_VOICE_G2P_TYPE_GPT_SOVITS_JA).split(" ")[:-1]
        else:
            text_phones = japanese.g2p(args.input)
    else:
        if args.ailia_voice:
            text_phones = voice.g2p(args.input, ailia_voice.AILIA_VOICE_G2P_TYPE_GPT_SOVITS_EN).split(" ")[:-1]
        else:
            text_phones = english.g2p(args.input)
    text_seq = np.array([cleaned_text_to_sequence(text_phones)], dtype=np.int64)

    # BERT features: real for zh, zeros for ja/en
    if args.ref_language == "zh" and ref_word2ph is not None:
        ref_bert = get_bert_feature(ref_norm_text, ref_word2ph, bert_tokenizer, bert, args.onnx)
    else:
        ref_bert = np.zeros((ref_seq.shape[1], 1024), dtype=np.float32)

    if args.text_language == "zh" and text_word2ph is not None:
        text_bert = get_bert_feature(text_norm_text, text_word2ph, bert_tokenizer, bert, args.onnx)
    else:
        text_bert = np.zeros((text_seq.shape[1], 1024), dtype=np.float32)
    
    vits_hps_data_sampling_rate = 32000

    zero_wav = np.zeros(
        int(vits_hps_data_sampling_rate * 0.3),
        dtype=np.float32,
    )
    wav16k, sr = librosa.load(input_audio, sr=16000)
    wav16k = np.concatenate([wav16k, zero_wav], axis=0)
    wav16k = wav16k[np.newaxis, :]
    ref_audio_16k = wav16k # hubertの入力のみpaddingする

    wav32k, sr = librosa.load(input_audio, sr=vits_hps_data_sampling_rate)
    wav32k = wav32k[np.newaxis, :]

    ssl_content = ssl.forward(ref_audio_16k)

    a = gpt_sovits.forward(ref_seq, text_seq, ref_bert, text_bert, wav32k, ssl_content)
    if args.fp16:
        a = a.astype(np.float32)

    savepath = args.savepath
    logger.info(f'saved at : {savepath}')

    soundfile.write(savepath, a, vits_hps_data_sampling_rate)

    logger.info('Script finished successfully.')


def main():
    use_zh = args.text_language == "zh" or args.ref_language == "zh"

    # model files check and download
    check_and_download_models(WEIGHT_PATH_SSL, MODEL_PATH_SSL, REMOTE_PATH)
    check_and_download_models(WEIGHT_PATH_T2S_ENCODER, MODEL_PATH_T2S_ENCODER, REMOTE_PATH)
    check_and_download_models(WEIGHT_PATH_T2S_FIRST_DECODER, MODEL_PATH_T2S_FIRST_DECODER, REMOTE_PATH)
    check_and_download_models(WEIGHT_PATH_T2S_STAGE_DECODER, MODEL_PATH_T2S_STAGE_DECODER, REMOTE_PATH)
    check_and_download_models(WEIGHT_PATH_VITS, MODEL_PATH_VITS, REMOTE_PATH)
    if use_zh:
        check_and_download_models(WEIGHT_PATH_BERT, MODEL_PATH_BERT, REMOTE_PATH)

    #env_id = args.env_id

    if args.onnx:
        import onnxruntime
        providers = ["CPUExecutionProvider"]
        #providers = ["CUDAExecutionProvider"]
        ssl = onnxruntime.InferenceSession(WEIGHT_PATH_SSL, providers=providers)
        t2s_encoder = onnxruntime.InferenceSession(WEIGHT_PATH_T2S_ENCODER, providers=providers)
        t2s_first_decoder = onnxruntime.InferenceSession(WEIGHT_PATH_T2S_FIRST_DECODER, providers=providers)
        t2s_stage_decoder = onnxruntime.InferenceSession(WEIGHT_PATH_T2S_STAGE_DECODER, providers=providers)
        vits = onnxruntime.InferenceSession(WEIGHT_PATH_VITS, providers=providers)
        bert = onnxruntime.InferenceSession(WEIGHT_PATH_BERT, providers=providers) if use_zh else None
    else:
        import ailia
        memory_mode = ailia.get_memory_mode(reduce_constant=True, ignore_input_with_initializer=True, reduce_interstage=False, reuse_interstage=True)
        ssl = ailia.Net(weight = WEIGHT_PATH_SSL, stream = MODEL_PATH_SSL, memory_mode = memory_mode, env_id = args.env_id)
        t2s_encoder = ailia.Net(weight = WEIGHT_PATH_T2S_ENCODER, stream = MODEL_PATH_T2S_ENCODER, memory_mode = memory_mode, env_id = args.env_id)
        t2s_first_decoder = ailia.Net(weight = WEIGHT_PATH_T2S_FIRST_DECODER, stream = MODEL_PATH_T2S_FIRST_DECODER, memory_mode = memory_mode, env_id = args.env_id)
        t2s_stage_decoder = ailia.Net(weight = WEIGHT_PATH_T2S_STAGE_DECODER, stream = MODEL_PATH_T2S_STAGE_DECODER, memory_mode = memory_mode, env_id = args.env_id)
        vits = ailia.Net(weight = WEIGHT_PATH_VITS, stream = MODEL_PATH_VITS, memory_mode = memory_mode, env_id = args.env_id)
        bert = ailia.Net(weight = WEIGHT_PATH_BERT, stream = MODEL_PATH_BERT, memory_mode = memory_mode, env_id = args.env_id) if use_zh else None
        if args.profile:
            ssl.set_profile_mode(True)
            t2s_encoder.set_profile_mode(True)
            t2s_first_decoder.set_profile_mode(True)
            t2s_stage_decoder.set_profile_mode(True)
            vits.set_profile_mode(True)
            if bert is not None:
                bert.set_profile_mode(True)
        pf = platform.system()
        if pf == "Darwin":
            if args.env_id == 2:
                logger.info(
                    "This model not optimized for macOS GPU currently. Please try -e 1 option to improve inference speed."
                )

    if args.benchmark:
        start = int(round(time.time() * 1000))

    generate_voice(ssl, t2s_encoder, t2s_first_decoder, t2s_stage_decoder, vits, bert)

    if args.benchmark:
        end = int(round(time.time() * 1000))
        logger.info("\ttotal processing time {} ms".format(end-start))

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
        if bert is not None:
            print("bert : ")
            print(bert.get_summary())


if __name__ == '__main__':
    main()
