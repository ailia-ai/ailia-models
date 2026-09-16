import sys
import time
from logging import getLogger
import os

import numpy as np
import soundfile as sf

import ailia

# import original modules
sys.path.append('../../util')
from microphone_utils import start_microphone_input  # noqa
from model_utils import check_and_download_models  # noqa
from arg_utils import get_base_parser, get_savepath, update_parser  # noqa

from utils.generation import generate_audio

logger = getLogger(__name__)

# ======================
# Parameters
# ======================

REMOTE_PATH = 'https://storage.googleapis.com/ailia-models/vall-e-x/'

SAMPLE_RATE = 16000

# ======================
# PARAMETERS
# ======================

SAVE_WAV_PATH = 'output.wav'

# ======================
# Arguemnt Parser Config
# ======================

parser = get_base_parser( 'VALL-E-X', None, SAVE_WAV_PATH)
# overwrite
parser.add_argument(
    '--input', '-i', metavar='TEXT', default=None,
    help='input text'
)
parser.add_argument(
    '--audio', '-a', default=None,
    help='input audio context'
)
parser.add_argument(
    '--transcript', '-t', default=None,
    help='input audio transcript'
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
    '--top_k', '-top_k', default=100,
    help='top_k filtering for output token sampling. official value is -100 (top_k is disabled). ailia modified this value for stability.'
)
parser.add_argument(
    '--seed', type=int, default=1023,
    help='random seed'
)
args = update_parser(parser, check_input_type=False)

# ======================
# Models
# ======================

WEIGHT_NAR_PREDICT_LAYERS_PATH = "nar_predict_layers.onnx"
WEIGHT_AR_AUDIO_EMBEDDING_PATH = "ar_audio_embedding.onnx"
WEIGHT_AR_LANGUAGE_EMBEDDING_PATH = "ar_language_embedding.onnx"
WEIGHT_AR_TEXT_EMBEDDING_PATH = "ar_text_embedding.onnx"
WEIGHT_NAR_AUDIO_EMBEDDING_LAYERS_PATH = "nar_audio_embedding_layers.onnx"
WEIGHT_NAR_AUDIO_EMBEDDING_PATH = "nar_audio_embedding.onnx"
WEIGHT_NAR_LANGUAGE_EMBEDDING_PATH = "nar_language_embedding.onnx"
WEIGHT_NAR_TEXT_EMBEDDING_PATH = "nar_text_embedding.onnx"
if args.normal:
    WEIGHT_DECODER_PATH = "ar_decoder.onnx"
    WEIGHT_NAR_DECODER_PATH = "nar_decoder.onnx"
else:
    WEIGHT_DECODER_PATH = "ar_decoder.opt.onnx"
    WEIGHT_NAR_DECODER_PATH = "nar_decoder.opt.onnx"
WEIGHT_ENCODEC_PATH = "encodec.onnx"
WEIGHT_VOCOS_PATH = "vocos.onnx"
WEIGHT_POSITION_EMBEDDING_PATH = "position_embedding.onnx"

ALL_MODELS = [
    WEIGHT_NAR_DECODER_PATH,
    WEIGHT_NAR_PREDICT_LAYERS_PATH,
    WEIGHT_AR_AUDIO_EMBEDDING_PATH,
    WEIGHT_AR_LANGUAGE_EMBEDDING_PATH,
    WEIGHT_AR_TEXT_EMBEDDING_PATH,
    WEIGHT_NAR_AUDIO_EMBEDDING_PATH,
    WEIGHT_NAR_AUDIO_EMBEDDING_LAYERS_PATH,
    WEIGHT_NAR_LANGUAGE_EMBEDDING_PATH,
    WEIGHT_NAR_TEXT_EMBEDDING_PATH,
    WEIGHT_DECODER_PATH,
    WEIGHT_ENCODEC_PATH,
    WEIGHT_VOCOS_PATH,
    WEIGHT_POSITION_EMBEDDING_PATH
]

# ======================
# Parameters
# ======================

if args.onnx:
    import onnxruntime
else:
    import ailia

if args.input:
    text = args.input
else:
    text ="音声合成のテストを行なっています。"

if args.seed:
    np.random.seed(args.seed)

sampling_rate = 24000

def generate_voice(models):
    # onnx
    logger.info("Input text : " + text)

    if args.benchmark:
        start = int(round(time.time() * 1000))

    model_name = None

    if args.audio != None:
        model_name = "jsut"
        #args.audio = "BASIC5000_0001.wav"
        #args.transcript = "水をマレーシアから買わなくてはならないのです"

        os.makedirs("customs", exist_ok=True)
        from utils.prompt_making import make_prompt
        make_prompt(name=model_name, audio_prompt_path=args.audio, transcript=args.transcript, models=models, ort=args.onnx)

    output = generate_audio(text, prompt=model_name, language='auto', accent='no-accent', benchmark=args.benchmark, models=models, ort=args.onnx, logger = logger, top_k = args.top_k)
    #print(output.shape)

    if args.benchmark:
        end = int(round(time.time() * 1000))
        estimation_time = (end - start)
        logger.info(f'total processing time {estimation_time} ms')

    # export to audio
    savepath = args.savepath
    logger.info(f'saved at : {savepath}')
    sf.write(savepath, output.astype(np.float32), sampling_rate)
    logger.info('Script finished successfully.')

def main():
    # model files check and download
    os.makedirs("onnx", exist_ok=True)
    for model in ALL_MODELS:
        check_and_download_models("./onnx/"+model, "./onnx/"+model+".prototxt", REMOTE_PATH)

    if ("MPS" in ailia.get_environment(args.env_id).name):
        logger.warning('This model is slow on MPS. Please try to use -e 1.')

    models = {}

    if args.onnx:
        for model in ALL_MODELS:
            net = onnxruntime.InferenceSession( "./onnx/"+model, providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
            models[model] = net
    else:
        for model in ALL_MODELS:
            net = ailia.Net(stream = "./onnx/"+model + ".prototxt", weight = "./onnx/"+model, memory_mode = args.memory_mode, env_id = args.env_id)
            if args.profile:
                net.set_profile_mode(True)
            models[model] = net

    generate_voice(models)

    if args.profile and not args.onnx:
        for model in ALL_MODELS:
            if model == "ar_decoder.onnx" or model == "ar_decoder.opt.onnx":
                print(model)
                print(models[model].get_summary())

if __name__ == '__main__':
    main()
