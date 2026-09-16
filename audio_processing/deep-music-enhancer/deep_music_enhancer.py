import time
import sys
import argparse

import numpy as np

import ailia  # noqa: E402

# import original modules
sys.path.append('../../util')
from arg_utils import get_base_parser, update_parser, get_savepath  # noqa: E402
from model_utils import check_and_download_models  # noqa: E402

# logger
from logging import getLogger   # noqa: E402
logger = getLogger(__name__)

import os
from tqdm import tqdm
import matplotlib.pyplot as plt
from scipy.io import wavfile
from deep_music_enhancer_utils import (
    read_audio,
    SingleSong
)


# ======================
# PARAMETERS
# ======================
WAV_PATH = 'input.wav'
SAVE_WAV_PATH = 'output.wav'

WEIGHT_PATH_RESNET = 'resnet.onnx'
MODEL_PATH_RESNET = 'resnet.onnx.prototxt'
WEIGHT_PATH_RESNET_BN = 'resnetbn.onnx'
MODEL_PATH_RESNET_BN = 'resnetbn.onnx.prototxt'
WEIGHT_PATH_RESNET_DA = 'resnetda.onnx'
MODEL_PATH_RESNET_DA = 'resnetda.onnx.prototxt'
WEIGHT_PATH_RESNET_DO = 'resnetdo.onnx'
MODEL_PATH_RESNET_DO = 'resnetdo.onnx.prototxt'

WEIGHT_PATH_UNET = 'unet.onnx'
MODEL_PATH_UNET = 'unet.onnx.prototxt'
WEIGHT_PATH_UNET_BN = 'unetbn.onnx'
MODEL_PATH_UNET_BN = 'unetbn.onnx.prototxt'
WEIGHT_PATH_UNET_DA = 'unetda.onnx'
MODEL_PATH_UNET_DA = 'unetda.onnx.prototxt'
WEIGHT_PATH_UNET_DO = 'unetdo.onnx'
MODEL_PATH_UNET_DO = 'unetdo.onnx.prototxt'

REMOTE_PATH = 'https://storage.googleapis.com/ailia-models/deep-music-enhancer/'


# ======================
# Arguemnt Parser Config
# ======================
parser = get_base_parser(
    'On Filter Generalization for Music Bandwidth Extension Using Deep Neural Networks', 
    WAV_PATH, 
    SAVE_WAV_PATH
)
# overwrite
parser.add_argument(
    '--input', '-i', metavar='WAV', default=WAV_PATH,
    help='input audio'
)
parser.add_argument(
    '--ailia_audio', action='store_true',
    help='use ailia audio library'
)
parser.add_argument(
    '--vis', action='store_true',
    help='save visualized spectrogram'
)
parser.add_argument(
    '--model', type=str, default='unet',
    choices=[
        'resnet', 'resnet_bn', 'resnet_da', 'resnet_do',
        'unet', 'unet_bn', 'unet_da', 'unet_do'
    ],
)
args = update_parser(parser, check_input_type=False)


# ======================
# Main function
# ======================
def audio_bandwidth_extension(net):
    FILTERS_TEST = [('cheby1', 6), ('butter', 6)]
    c_SAMPLE_RATE = 44100
    c_WAV_SAMPLE_LEN = 8192 
    cutoff = 11025
    duration = None
    start = 0

    for filter_ in FILTERS_TEST:
        input_name = args.input[0]
        input_name_without_ext = os.path.splitext(os.path.basename(input_name))[0]
        hq_path = input_name

        logger.info('filter: {}, input_name: {}'.format(filter_, input_name))

        # create dataset
        song_data = SingleSong(
            c_WAV_SAMPLE_LEN, 
            filter_, 
            hq_path,
            cutoff=cutoff, 
            duration=duration, 
            start=start
        )

        y_full = song_data.preallocate()  # preallocation to keep individual output chunks

        idx_start_chunk = 0 # model works on chunks of audio, these are concatenated later

        for i in tqdm(range(len(song_data))):
            x, t = song_data[i]
            x = x[np.newaxis, :, :]

            y = net.predict(x)

            idx_end_chunk = idx_start_chunk + y.shape[0]
            y_full[idx_start_chunk:idx_end_chunk] = y
            idx_start_chunk = idx_end_chunk

        y_full = np.concatenate(y_full, axis=-1) # create full song out of chunks

        x_full, t_full = song_data.get_full_signals()
        y_full = np.clip(y_full, -1, 1 - np.finfo(np.float32).eps)

        # save audio
        wavfile.write(args.savepath, c_SAMPLE_RATE, y_full.T)

        # save spec
        if args.vis:
            _, _, _, _ = plt.specgram(x_full.T[:c_SAMPLE_RATE*5, 0], Fs=c_SAMPLE_RATE)
            plt.savefig('{}_{}_input_spec.png'.format(input_name_without_ext, filter_[0]))
            _, _, _, _ = plt.specgram(y_full.T[:c_SAMPLE_RATE*5, 0], Fs=c_SAMPLE_RATE)
            plt.savefig('{}_{}_output_spec.png'.format(input_name_without_ext, filter_[0]))

    logger.info('Script finished successfully.')


def main():
    # model files check and download
    if args.model == 'resnet':
        weight_path, model_path = WEIGHT_PATH_RESNET, MODEL_PATH_RESNET
    elif args.model == 'resnet_bn':
        weight_path, model_path = WEIGHT_PATH_RESNET_BN, MODEL_PATH_RESNET_BN
    elif args.model == 'resnet_da':
        weight_path, model_path = WEIGHT_PATH_RESNET_DA, MODEL_PATH_RESNET_DA
    elif args.model == 'resnet_do':
        weight_path, model_path = WEIGHT_PATH_RESNET_DO, MODEL_PATH_RESNET_DO
    elif args.model == 'unet':
        weight_path, model_path = WEIGHT_PATH_UNET, MODEL_PATH_UNET
    elif args.model == 'unet_bn':
        weight_path, model_path = WEIGHT_PATH_UNET_BN, MODEL_PATH_UNET_BN
    elif args.model == 'unet_da':
        weight_path, model_path = WEIGHT_PATH_UNET_DA, MODEL_PATH_UNET_DA
    elif args.model == 'unet_do':
        weight_path, model_path = WEIGHT_PATH_UNET_DO, MODEL_PATH_UNET_DO

    env_id = args.env_id

    check_and_download_models(weight_path, model_path, REMOTE_PATH)
    net = ailia.Net(model_path, weight_path, env_id=env_id, memory_mode=args.memory_mode)

    audio_bandwidth_extension(net)


if __name__ == "__main__":
    main()
