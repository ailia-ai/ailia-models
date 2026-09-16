import sys
import time
from logging import getLogger
import json

import random

import librosa
import numpy as np

import ailia

# import original modules
sys.path.append('../../util')
from arg_utils import get_base_parser, update_parser, get_savepath  # noqa
from model_utils import check_and_download_models  # noqa

logger = getLogger(__name__)

# ======================
# Parameters
# ======================

CAPTION_WEIGHT_PATH_2023 = 'msclap_2023_caption.onnx'
AUDIO_WEIGHT_PATH_2023 = 'msclap_2023_audio.onnx'

CAPTION_MODEL_PATH_2023 = 'msclap_2023_caption.onnx.prototxt'
AUDIO_MODEL_PATH_2023 = 'msclap_2023_audio.onnx.prototxt'

CAPTION_WEIGHT_PATH_2022 = 'msclap_2022_caption.onnx'
AUDIO_WEIGHT_PATH_2022 = 'msclap_2022_audio.onnx'

CAPTION_MODEL_PATH_2022 = 'msclap_2022_caption.onnx.prototxt'
AUDIO_MODEL_PATH_2022 = 'msclap_2022_audio.onnx.prototxt'

REMOTE_PATH = "https://storage.googleapis.com/ailia-models/msclap/"

# ======================
# Arguemnt Parser Config
# ======================

parser = get_base_parser(
    'msclap', None, None
)

parser.add_argument(
    "-a", "--audio", type=str,
    default="input.wav",
    help="Input audio file path."
)

parser.add_argument(
    "-t", "--text", type=str,
    default="captions.txt",
    help="Input text caption file path"
)

parser.add_argument(
    "-v", "--version", type=str,
    default="2023",
    help="Version of the CLAP model (2022 or 2023)."
)

parser.add_argument(
    '-w', '--write_json',
    action='store_true',
    help='Flag to output results to json file.'
)
parser.add_argument(
    '--disable_ailia_tokenizer',
    action='store_true',
    help='disable ailia tokenizer.'
)
args = update_parser(parser, check_input_type=False)

# ======================
# Helper functions
# ======================

def read_audio(audio_path):
    r"""Loads audio file or array and returns a numpy tensor"""
    # Randomly sample a segment of audio_duration from the clip or pad to match duration
    audio_time_series, sample_rate = librosa.load(audio_path, sr=None)
    return audio_time_series, sample_rate

def resample_audio(audio_time_series, sample_rate, resample_rate):
    resample_rate = 44100
    if resample_rate != sample_rate:
        audio_time_series = librosa.resample(
            audio_time_series,
            orig_sr=sample_rate,
            target_sr=resample_rate,
            res_type = 'sinc_best'
        )
    return audio_time_series, resample_rate
    

def resize_audio(audio_time_series, sample_rate, audio_duration, resample=False):
    r"""Loads audio file and returns raw audio."""
    # Randomly sample a segment of audio_duration from the clip or pad to match duration
    audio_time_series = audio_time_series.reshape(-1)
    # audio_time_series is shorter than predefined audio duration,
    # so audio_time_series is extended
    if audio_duration*sample_rate >= audio_time_series.shape[0]:
        repeat_factor = int(np.ceil((audio_duration*sample_rate) /
                                    audio_time_series.shape[0]))
        # Repeat audio_time_series by repeat_factor to match audio_duration
        audio_time_series = np.tile(audio_time_series,repeat_factor)
        # remove excess part of audio_time_series
        audio_time_series = audio_time_series[0:audio_duration*sample_rate]
    else:
        # audio_time_series is longer than predefined audio duration,
        # so audio_time_series is trimmed
        start_index = random.randrange(
            audio_time_series.shape[0] - audio_duration*sample_rate)
        audio_time_series = audio_time_series[start_index:start_index +
                                              audio_duration*sample_rate]
    return audio_time_series

def get_audio_embeddings(wav_input, sample_rate, model, version="2023"):
    if version in ('2023', '2022'):
        wav_input = resample_audio(wav_input, sample_rate, 44100)[0]
        wav_input = resize_audio(wav_input, 44100, 7)[None] 
    return model['audio_model'].predict(wav_input)

def get_caption_embeddings(text_input, model, version="2023"):

    # preprocesing
    if version == '2023':
        text_input = [t + ' <|endoftext|>' for t in text_input]
    tokenized = dict(model['tokenizer'](text_input, padding = True, return_tensors = 'np'))

    # inference
    model_input = (tokenized['input_ids'], tokenized['attention_mask'])
    return model['caption_model'].predict(model_input)[0]

def cossim(v1, v2):
    return np.sum(v1 * v2, axis = -1) / (np.sum(v1 ** 2, axis = -1) ** 0.5 * np.sum(v2 ** 2, axis = -1) ** 0.5)

def print_sorted_dict(d):
    m_len = max([len(k) for k in d.keys()])
    for k, v in sorted(d.items(), key=lambda x: x[1], reverse=True):
        pad = ' ' * (m_len - len(k) + 4)
        print(f'{pad + k}: {v}')

def save_sorted_dict_as_json(d):
    m_len = max([len(k) for k in d.keys()])
    result = []
    for k, v in sorted(d.items(), key=lambda x: x[1], reverse=True):
        result.append({"caption": k, "similarity": float(v)})
    with open('output.json', 'w', encoding='utf-8') as f:
        json.dump(result, f, indent=2)

# ======================
# Main functions
# ======================

def inference(model, input_text, input_wav, sample_rate, version):
    # get embeddings
    audio_embeddings = get_audio_embeddings(input_wav, sample_rate, model, version)
    caption_embeddings = get_caption_embeddings(input_text, model, version)

    return cossim(audio_embeddings, caption_embeddings)

def estimate_best_caption(model):
    # load inputs
    #input_text = CAPTIONS
    with open(args.text, 'r') as f:
        input_text = f.read().splitlines()
    #input_text = args.input.split('.')
    
    input_wav, sample_rate = read_audio(args.audio)
    input_wav = input_wav[None]

    logger.info("input_text: %s" % input_text)

    # inference
    logger.info('inference has started...')
    if args.benchmark:
        logger.info('BENCHMARK mode')
        total_time_estimation = 0
        for i in range(args.benchmark_count):
            start = int(round(time.time() * 1000))
            output = inference(model, input_text, input_wav, sample_rate, args.version)
            end = int(round(time.time() * 1000))
            estimation_time = (end - start)

            # Logging
            logger.info(f'\tailia processing estimation time {estimation_time} ms')
            if i != 0:
                total_time_estimation = total_time_estimation + estimation_time

        logger.info(f'\taverage time estimation {total_time_estimation / (args.benchmark_count - 1)} ms')
    else:
        output = inference(model, input_text, input_wav, sample_rate, args.version)

    print(f"Similarity: ")
    print_sorted_dict(dict(zip(input_text, output)))

    if args.write_json:
        save_sorted_dict_as_json(dict(zip(input_text, output)))

    logger.info('Script finished successfully.')


def main():
    # model files check and download
    if args.version == '2023':
        check_and_download_models(
            CAPTION_WEIGHT_PATH_2023,
            CAPTION_MODEL_PATH_2023,
            REMOTE_PATH
        )
        check_and_download_models(
            AUDIO_WEIGHT_PATH_2023,
            AUDIO_MODEL_PATH_2023,
            REMOTE_PATH
        )
    elif args.version == '2022':
        check_and_download_models(
            CAPTION_WEIGHT_PATH_2022,
            CAPTION_MODEL_PATH_2022,
            REMOTE_PATH
        )
        check_and_download_models(
            AUDIO_WEIGHT_PATH_2022,
            AUDIO_MODEL_PATH_2022,
            REMOTE_PATH
        )

    env_id = args.env_id

    # disable FP16
    if "FP16" in ailia.get_environment(args.env_id).props or sys.platform == 'Darwin':
        logger.warning('This model do not work on FP16. So use CPU mode.')
        env_id = 0

    # initialize
    if args.version == '2023':
        caption_model = ailia.Net(CAPTION_MODEL_PATH_2023, CAPTION_WEIGHT_PATH_2023, env_id=env_id, memory_mode=args.memory_mode)
        audio_model = ailia.Net(AUDIO_MODEL_PATH_2023, AUDIO_WEIGHT_PATH_2023, env_id=env_id, memory_mode=args.memory_mode)
        if args.disable_ailia_tokenizer:
            from transformers import AutoTokenizer
            tokenizer = AutoTokenizer.from_pretrained('gpt2')
            tokenizer.add_special_tokens({'pad_token': '!'})
        else:
            from ailia_tokenizer import GPT2Tokenizer
            tokenizer = GPT2Tokenizer.from_pretrained('./tokenizer_gpt2/')
            tokenizer.add_special_tokens({'pad_token': '!'})
            #tokenizer._pad_token_id = 0
    elif args.version == '2022':
        caption_model = ailia.Net(CAPTION_MODEL_PATH_2022, CAPTION_WEIGHT_PATH_2022, env_id=env_id, memory_mode=args.memory_mode)
        audio_model = ailia.Net(AUDIO_MODEL_PATH_2022, AUDIO_WEIGHT_PATH_2022, env_id=env_id, memory_mode=args.memory_mode)
        if args.disable_ailia_tokenizer:
            from transformers import AutoTokenizer
            tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
        else:
            from ailia_tokenizer import BertTokenizer
            tokenizer = BertTokenizer.from_pretrained('./tokenizer_bert/')

    model = {
        'caption_model':caption_model,
        'audio_model':audio_model,
        'tokenizer':tokenizer
    }

    estimate_best_caption(model)

if __name__ == '__main__':
    main()