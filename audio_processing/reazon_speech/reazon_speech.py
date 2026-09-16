import sys
import time
from logging import getLogger

import yaml
import numpy as np
import librosa

import ailia

# import original modules
sys.path.append('../../util')
from arg_utils import get_base_parser, update_parser, get_savepath  # noqa
from model_utils import check_and_download_models  # noqa
from image_utils import normalize_image  # noqa
from detector_utils import load_image  # noqa
from webcamera_utils import get_capture, get_writer  # noqa

from beam_search import BatchBeamSearch
import transformer_decoder
import seq_rnn_lm
import ctc_prefix_score
from transformer_decoder import TransformerDecoder
from seq_rnn_lm import SequentialRNNLM
from ctc_prefix_score import CTCPrefixScorer

logger = getLogger(__name__)

# ======================
# Parameters
# ======================

WEIGHT_ENC_PATH = 'reazonspeech-espnet-v1-encoder.onnx'
MODEL_ENC_PATH = 'reazonspeech-espnet-v1-encoder.onnx.prototxt'
WEIGHT_DEC_PATH = 'reazonspeech-espnet-v1-decoder.onnx'
MODEL_DEC_PATH = 'reazonspeech-espnet-v1-decoder.onnx.prototxt'
WEIGHT_LM_PATH = 'reazonspeech-espnet-v1-lm.onnx'
MODEL_LM_PATH = 'reazonspeech-espnet-v1-lm.onnx.prototxt'
WEIGHT_CTC_PATH = 'reazonspeech-espnet-v1-ctc.onnx'
MODEL_CTC_PATH = 'reazonspeech-espnet-v1-ctc.onnx.prototxt'
REMOTE_PATH = 'https://storage.googleapis.com/ailia-models/reason_speech/'

WAV_PATH = 'speech-001.wav'
SAVE_TEXT_PATH = 'output.txt'

# ======================
# Arguemnt Parser Config
# ======================

parser = get_base_parser(
    'ReazonSpeech', WAV_PATH, SAVE_TEXT_PATH, input_ftype='audio'
)
parser.add_argument('--beam_size', type=int, default=20, help="Beam size")
parser.add_argument('--nbest', type=int, default=1, help="Output N-best hypotheses")
parser.add_argument('--penalty', type=float, default=0.0, help="Insertion penalty")
parser.add_argument(
    "--maxlenratio", type=float, default=0.0,
    help="Input length ratio to obtain max output length. "
         "If maxlenratio=0.0 (default), it uses a end-detect "
         "function "
         "to automatically find maximum hypothesis lengths."
         "If maxlenratio<0.0, its absolute value is interpreted"
         "as a constant max output length",
)
parser.add_argument(
    "--minlenratio", type=float, default=0.0,
    help="Input length ratio to obtain min output length",
)
parser.add_argument(
    "--ctc_weight",
    type=float,
    default=0.5,
    help="CTC weight in joint decoding",
)
parser.add_argument("--lm_weight", type=float, default=1.0, help="RNNLM weight")
parser.add_argument("--ngram_weight", type=float, default=0.9, help="ngram weight")
parser.add_argument(
    '--onnx',
    action='store_true',
    help='execute onnxruntime version.'
)
args = update_parser(parser)


# ======================
# Secondaty Functions
# ======================

class CharTokenizer(object):
    def __init__(
            self,
            space_symbol: str = "<space>"):
        self.space_symbol = space_symbol

    def text2tokens(self, line: str):
        tokens = []
        while len(line) != 0:
            t = line[0]
            if t == " ":
                t = self.space_symbol
            tokens.append(t)
            line = line[1:]
        return tokens

    def tokens2text(self, tokens) -> str:
        tokens = [t if t != self.space_symbol else " " for t in tokens]
        return "".join(tokens)


# ======================
# Main functions
# ======================

def decode(mod, enc):
    nbest = args.nbest
    maxlenratio = args.maxlenratio
    minlenratio = args.minlenratio

    beam_search = mod['beam_search']
    token_list = mod['token_list']
    tokenizer = mod['tokenizer']

    nbest_hyps = beam_search.forward(
        x=enc, maxlenratio=maxlenratio, minlenratio=minlenratio
    )

    nbest_hyps = nbest_hyps[: nbest]

    results = []
    for hyp in nbest_hyps:
        # remove sos/eos and get results
        token_int = hyp.yseq[1:-1].tolist()

        # remove blank symbol id, which is assumed to be 0
        token_int = list(filter(lambda x: x != 0, token_int))

        # Change integer-ids to tokens
        token = [token_list[i] for i in token_int]

        text = tokenizer.tokens2text(token)
        results.append((text, token, token_int, hyp))

    return results


def predict(mod, speech):
    speech = np.expand_dims(speech, axis=0)
    lengths = np.ones([1], dtype=int) * speech.shape[1]  # lengths: (1,)
    # speech = np.load("speech-001.npy")
    # lengths = np.load("lengths-001.npy")
    # speech = np.load("speech-002.npy")
    # lengths = np.load("lengths-002.npy")

    net = mod['net']

    # feedforward
    if not args.onnx:
        output = net.predict([speech, lengths])
    else:
        output = net.run(None, {'speech': speech, 'lengths': lengths})
    enc, _ = output

    results = decode(mod, enc[0])

    return results


def recognize_from_audio(mod):
    # input audio loop
    for audio_path in args.input:
        logger.info(audio_path)

        # prepare input data
        speech, rate = librosa.load(audio_path, sr=16000)

        # max 30 sec
        max_len = 16000 * 30
        if speech.shape[0] >= max_len:
            speech = speech[0:max_len]
            logger.warning('Cut to 30 seconds of maximum recognition seconds.')

        # inference
        logger.info('Start inference...')
        if args.benchmark:
            logger.info('BENCHMARK mode')
            total_time_estimation = 0
            for i in range(args.benchmark_count):
                start = int(round(time.time() * 1000))
                output = predict(mod, speech)
                end = int(round(time.time() * 1000))
                estimation_time = (end - start)

                # Logging
                logger.info(f'\tailia processing estimation time {estimation_time} ms')
                if i != 0:
                    total_time_estimation = total_time_estimation + estimation_time

            logger.info(f'\taverage time estimation {total_time_estimation / (args.benchmark_count - 1)} ms')
        else:
            output = predict(mod, speech)

        for res in output:
            logger.info(f"{res[0]}")

    logger.info('Script finished successfully.')


def main():
    # model files check and download
    check_and_download_models(WEIGHT_ENC_PATH, MODEL_ENC_PATH, REMOTE_PATH)
    check_and_download_models(WEIGHT_DEC_PATH, MODEL_DEC_PATH, REMOTE_PATH)
    check_and_download_models(WEIGHT_LM_PATH, MODEL_LM_PATH, REMOTE_PATH)
    check_and_download_models(WEIGHT_CTC_PATH, MODEL_CTC_PATH, REMOTE_PATH)

    env_id = args.env_id

    # initialize
    if not args.onnx:
        net = ailia.Net(MODEL_ENC_PATH, WEIGHT_ENC_PATH, env_id=env_id, memory_mode=args.memory_mode)
        decoder = ailia.Net(MODEL_DEC_PATH, WEIGHT_DEC_PATH, env_id=env_id, memory_mode=args.memory_mode)
        lm_net = ailia.Net(MODEL_LM_PATH, WEIGHT_LM_PATH, env_id=env_id, memory_mode=args.memory_mode)
        ctc = ailia.Net(MODEL_CTC_PATH, WEIGHT_CTC_PATH, env_id=env_id, memory_mode=args.memory_mode)
    else:
        import onnxruntime
        net = onnxruntime.InferenceSession(WEIGHT_ENC_PATH)
        decoder = onnxruntime.InferenceSession(WEIGHT_DEC_PATH)
        lm_net = onnxruntime.InferenceSession(WEIGHT_LM_PATH)
        ctc = onnxruntime.InferenceSession(WEIGHT_CTC_PATH)
        
        transformer_decoder.onnx = True
        seq_rnn_lm.onnx = True
        ctc_prefix_score.onnx = True

    with open('config.yaml') as file:
        config = yaml.safe_load(file)

    beam_size = args.beam_size
    ctc_weight = args.ctc_weight
    lm_weight = args.lm_weight
    ngram_weight = args.ngram_weight
    penalty = args.penalty

    weights = {
        'decoder': 1.0 - ctc_weight, 'ctc': ctc_weight,
        'lm': lm_weight, 'ngram': ngram_weight, 'length_bonus': penalty
    }
    sos = eos = 2601
    token_list = config['token_list']

    decoder = TransformerDecoder(
        decoder=decoder,
        num_blocks=6)
    lm = SequentialRNNLM(lm_net)
    ctc = CTCPrefixScorer(ctc, eos)
    scorers = {
        'decoder': decoder,
        'lm': lm,
        'ctc': ctc,
    }
    beam_search = BatchBeamSearch(
        beam_size=beam_size,
        weights=weights,
        scorers=scorers,
        sos=sos,
        eos=eos,
        vocab_size=len(token_list),
        token_list=token_list,
        pre_beam_score_key='full'
    )
    tokenizer = CharTokenizer(space_symbol='<space>')

    mod = {
        'net': net,
        'beam_search': beam_search,
        'token_list': token_list,
        'tokenizer': tokenizer,
    }

    recognize_from_audio(mod)


if __name__ == '__main__':
    main()
