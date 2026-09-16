import os
import sys
import time
import random

import numpy as np
import librosa

import ailia

# import original modules
sys.path.append('../../util')
from arg_utils import get_base_parser, update_parser  # noqa: E402
from model_utils import check_and_download_models  # noqa: E402

# logger
from logging import getLogger  # noqa: E402

logger = getLogger(__name__)

# ======================
# Parameters
# ======================

WEIGHT_IDENT_PATH = './proposed_iden.onnx'
MODEL_IDENT_PATH = './proposed_iden.onnx.prototxt'
WEIGHT_CLASSIFIER_PATH = './proposed_classifier.onnx'
MODEL_CLASSIFIER_PATH = './proposed_classifier.onnx.prototxt'
WEIGHT_VERI_PATH = './proposed_veri.onnx'
MODEL_VERI_PATH = './proposed_veri.onnx.prototxt'
REMOTE_PATH = \
    'https://storage.googleapis.com/ailia-models/auto_speech/'

WAVE_PATH = "wav/id10283/oGZsanLiXsY/00004.wav"

# Audio
SAMPLING_RATE = 16000

# Mel-filterbank
WINDOW_LENGTH = 25  # In milliseconds
WINDOW_STEP = 10  # In milliseconds
N_FFT = 512

# Audio volume normalization
AUDIO_NORM_TARGET_dBFS = -30

THRESHOLD = 0.26

INT16_MAX = (2 ** 15) - 1

# ======================
# Arguemnt Parser Config
# ======================

parser = get_base_parser(
    'AutoSpeech', WAVE_PATH, None, input_ftype='audio'
)
parser.add_argument(
    '-i1', '--input1', metavar='WAV', default=None,
    help='Specify an wav file to compare with the input2 wav. (verification mode)'
)
parser.add_argument(
    '-i2', '--input2', metavar='WAV', default=None,
    help='Specify an wav file to compare with the input1 wav. (verification mode)'
)
parser.add_argument(
    '-th', '--threshold',
    default=THRESHOLD, type=float,
    help='The similar threshold for verification.'
)
args = update_parser(parser)


# ======================
# Secondaty Functions
# ======================

def read_wave(path):
    # prepare input data
    wav, source_sr = librosa.load(path, sr=None)
    # Resample the wav if needed
    if source_sr is not None and source_sr != SAMPLING_RATE:
        wav = librosa.resample(wav, source_sr, SAMPLING_RATE)

    return wav


def voxceleb1_ids():
    with open("VoxCeleb1_ids.txt") as f:
        ids = [x.strip() for x in f]

    return ids


def normalize_volume(wav, target_dBFS, increase_only=False, decrease_only=False):
    if increase_only and decrease_only:
        raise ValueError("Both increase only and decrease only are set")
    rms = np.sqrt(np.mean((wav * INT16_MAX) ** 2))
    wave_dBFS = 20 * np.log10(rms / INT16_MAX)
    dBFS_change = target_dBFS - wave_dBFS
    if dBFS_change < 0 and increase_only or dBFS_change > 0 and decrease_only:
        return wav
    return wav * (10 ** (dBFS_change / 20))


def wav_to_spectrogram(wav):
    frames = np.abs(librosa.core.stft(
        wav,
        n_fft=N_FFT,
        hop_length=int(SAMPLING_RATE * WINDOW_STEP / 1000),
        win_length=int(SAMPLING_RATE * WINDOW_LENGTH / 1000),
    ))
    return frames.astype(np.float32).T


def generate_sequence(feature, partial_n_frames, shift=None):
    while feature.shape[0] <= partial_n_frames:
        feature = np.repeat(feature, 2, axis=0)
    if shift is None:
        shift = partial_n_frames // 2
    test_sequence = []
    start = 0
    while start + partial_n_frames <= feature.shape[0]:
        test_sequence.append(feature[start: start + partial_n_frames])
        start += shift
    test_sequence = np.stack(test_sequence, axis=0)
    return test_sequence


def cosine_similar(a, b, data_is_normalized=False):
    if not data_is_normalized:
        a = np.asarray(a) / np.linalg.norm(a, axis=1, keepdims=True)
        b = np.asarray(b) / np.linalg.norm(b, axis=1, keepdims=True)
    return np.dot(a, b.T)


# ======================
# Main functions
# ======================

def preprocess(wav):
    wav = normalize_volume(wav, AUDIO_NORM_TARGET_dBFS, increase_only=True)
    feature = wav_to_spectrogram(wav)

    sequence = generate_sequence(feature, partial_n_frames=300)

    mean = np.load('mean.npy')
    std = np.load('std.npy')
    sequence = (sequence - mean) / std
    # if random.random() < 0.5:
    #     sequence = np.flip(sequence, axis=0).copy()

    return sequence


def predict(wav, net, net_classifier=None):
    # initial preprocesses
    sequence = preprocess(wav)

    # feedforward
    output = net.predict([sequence])
    output = output[0]

    output = np.mean(output, axis=0, keepdims=True)

    if not net_classifier:
        return output

    output = net_classifier.predict([output])
    output = output[0]

    idx = np.argsort(output[0])[::-1]

    return idx


def eval_identification(net, net_classifier):
    ids = voxceleb1_ids()

    for input_path in args.input:
        logger.info(f'input: {input_path}')

        # prepare input data
        wav = read_wave(input_path)

        # inference
        logger.info('Start inference...')
        if args.benchmark:
            logger.info('BENCHMARK mode')
            for i in range(5):
                start = int(round(time.time() * 1000))
                idx = predict(wav, net, net_classifier)
                end = int(round(time.time() * 1000))
                logger.info(f'\tailia processing time {end - start} ms')
        else:
            idx = predict(wav, net, net_classifier)

        logger.info(' Top5: %s' % ', '.join([ids[i] for i in idx[:5]]))

    logger.info('Script finished successfully.')


def eval_verification(net):
    threshold = args.threshold
    input1 = args.input1
    input2 = args.input2

    if input1 is None:
        logger.error('input1 is not specified')
        sys.exit(-1)
    elif not os.path.isfile(input1):
        logger.error('specified input1 is not file path nor directory path')
        sys.exit(-1)
    if input2 is None:
        logger.error('input2 is not specified')
        sys.exit(-1)
    elif not os.path.isfile(input2):
        logger.error('specified input2 is not file path nor directory path')
        sys.exit(-1)

    logger.info(f'input1: {input1}')
    logger.info(f'input2: {input2}')

    # prepare input data
    wav1 = read_wave(args.input1)
    wav2 = read_wave(args.input2)

    # inference
    logger.info('Start inference...')
    if args.benchmark:
        logger.info('BENCHMARK mode')
        for i in range(5):
            start = int(round(time.time() * 1000))
            output = predict(wav1, net)
            output2 = predict(wav2, net)
            end = int(round(time.time() * 1000))
            logger.info(f'\tailia processing time {end - start} ms')
    else:
        output = predict(wav1, net)
        output2 = predict(wav2, net)

    similar = cosine_similar(output, output2)
    logger.info(' similar: %.8f' % similar[0])
    logger.info(' verification: %s (threshold: %.3f)' %
                ('match' if similar[0] >= threshold else 'unmatch', threshold))

    logger.info('Script finished successfully.')


def main():
    # model files check and download
    if args.input1 or args.input2:
        check_and_download_models(WEIGHT_VERI_PATH, MODEL_VERI_PATH, REMOTE_PATH)
    else:
        logger.info('Checking identification model...')
        check_and_download_models(WEIGHT_IDENT_PATH, MODEL_IDENT_PATH, REMOTE_PATH)
        logger.info('Checking classification model...')
        check_and_download_models(WEIGHT_CLASSIFIER_PATH, MODEL_CLASSIFIER_PATH, REMOTE_PATH)

    env_id = args.env_id

    if args.input1 or args.input2:
        net = ailia.Net(MODEL_VERI_PATH, WEIGHT_VERI_PATH, env_id=env_id, memory_mode=args.memory_mode)

        eval_verification(net)
    else:
        # initialize
        net = ailia.Net(MODEL_IDENT_PATH, WEIGHT_IDENT_PATH, env_id=env_id, memory_mode=args.memory_mode)
        net_classifier = ailia.Net(MODEL_CLASSIFIER_PATH, WEIGHT_CLASSIFIER_PATH, env_id=env_id, memory_mode=args.memory_mode)

        eval_identification(net, net_classifier)


if __name__ == '__main__':
    main()
