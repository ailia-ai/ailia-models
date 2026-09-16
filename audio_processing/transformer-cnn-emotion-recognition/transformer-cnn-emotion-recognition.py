import time
import sys

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
# PARAMETERS
# ======================
# https://smartlaboratory.org/ravdess/
WAVE_PATH = "03-01-01-01-01-01-01.wav"

WEIGHT_PATH = "parallel_is_all_you_want_ep428.onnx"
MODEL_PATH = "parallel_is_all_you_want_ep428.onnx.prototxt"
REMOTE_PATH = "https://storage.googleapis.com/ailia-models/parallel_is_all_you_want/"

LABELS = [
    "surprised", "neutral", "calm", "happy",
    "sad", "angry", "fearful", "disgust",
]

# ======================
# Arguemnt Parser Config
# ======================
parser = get_base_parser(
    'Parallel_is_All_You_Want.', WAVE_PATH, None, input_ftype='audio')
args = update_parser(parser)


# ======================
# Utils
# ======================

def get_waveforms(file, sample_rate=48000):
    # load an individual sample audio file
    # read the full 3 seconds of the file, cut off the first 0.5s of silence; native sample rate = 48k
    # don't need to store the sample rate that librosa.load returns
    waveform, _ = librosa.load(file, duration=3, offset=0.5, sr=sample_rate)

    # make sure waveform vectors are homogenous by defining explicitly
    waveform_homo = np.zeros((int(sample_rate * 3, )))
    waveform_homo[:len(waveform)] = waveform

    # return a single file's waveform
    return waveform_homo


def feature_mfcc(
        waveform,
        sample_rate,
        n_mfcc=40,
        fft=1024,
        winlen=512,
        window='hamming',
        # hop=256, # increases # of time steps; was not helpful
        mels=128):
    # Compute the MFCCs for all STFT frames
    # 40 mel filterbanks (n_mfcc) = 40 coefficients
    mfc_coefficients = librosa.feature.mfcc(
        y=waveform,
        sr=sample_rate,
        n_mfcc=n_mfcc,
        n_fft=fft,
        win_length=winlen,
        window=window,
        # hop_length=hop,
        n_mels=mels,
        fmax=sample_rate / 2
    )

    return mfc_coefficients


def preprocess(x):
    c, h, w = x.shape
    x = x.reshape(-1)

    mean = np.load('std_mean.npy')
    var = np.load('std_var.npy')
    x = (x - mean) / np.sqrt(var)

    x = x.reshape((c, h, w))

    return x


# ======================
# Main function
# ======================

def predict(net, waveform):
    sample_rate = 48000
    features = feature_mfcc(waveform, sample_rate)
    features = np.expand_dims(features, axis=0)

    x = preprocess(features)

    # feedforward
    x = np.expand_dims(x, axis=0)
    output = net.predict([x])

    output_logits, output_softmax = output
    output_softmax = output_softmax[0]

    label = np.argmax(output_softmax)
    conf = output_softmax[label]

    return label, conf


def recognize_from_audio(net):
    for input_path in args.input:
        logger.info(f'input: {input_path}')

        # load audio
        waveform = get_waveforms(input_path)

        # inference
        logger.info('Start inference...')
        if args.benchmark:
            logger.info('BENCHMARK mode')
            total_time_estimation = 0
            for i in range(args.benchmark_count):
                start = int(round(time.time() * 1000))
                label, conf = predict(net, waveform)
                end = int(round(time.time() * 1000))
                estimation_time = (end - start)

                # Loggin
                logger.info(f'\tailia processing estimation time {estimation_time} ms')
                if i != 0:
                    total_time_estimation = total_time_estimation + estimation_time

        else:
            label, conf = predict(net, waveform)

        label = LABELS[label]
        logger.info("Emotion: %s" % label)
        logger.info("Confidence: %s" % conf)

    logger.info('Script finished successfully.')


def main():
    # model files check and download
    check_and_download_models(WEIGHT_PATH, MODEL_PATH, REMOTE_PATH)

    # initialize
    net = ailia.Net(MODEL_PATH, WEIGHT_PATH, env_id=args.env_id, memory_mode=args.memory_mode)

    recognize_from_audio(net)


if __name__ == "__main__":
    main()
