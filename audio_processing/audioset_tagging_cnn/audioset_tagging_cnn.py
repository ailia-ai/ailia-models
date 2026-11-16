import os
import sys
import time
import librosa
import argparse
import utilities
import numpy as np
import matplotlib.pyplot as plt
import json

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

WEIGHT_TAGGING_PATH = './audio_tagging.onnx'
MODEL_TAGGING_PATH = './audio_tagging.onnx.prototxt'
WEIGHT_DETECTION_PATH = './sound_event_detection.onnx'
MODEL_DETECTION_PATH = './sound_event_detection.onnx.prototxt'
REMOTE_PATH = 'https://storage.googleapis.com/ailia-models/audioset_tagging_cnn/'

WAVE_PATH = "R9_ZSCveAHg_7s.wav"
SAVE_PATH = "output.png"


# ======================
# Arguemnt Parser Config
# ======================

parser = get_base_parser(
    'audioset_tagging_cnn', WAVE_PATH, None, input_ftype='audio'
)
parser.add_argument('--mode', type=str, default="audio_tagging", choices=["audio_tagging", "sound_event_detection"])
parser.add_argument('--sample_rate', type=int, default=32000)
parser.add_argument('--window_size', type=int, default=1024)
parser.add_argument('--hop_size', type=int, default=320)
parser.add_argument('--mel_bins', type=int, default=64)
parser.add_argument('--fmin', type=int, default=50)
parser.add_argument('--fmax', type=int, default=14000) 
parser.add_argument('-w', '--write_json', action='store_true', help='Flag to output results to json file.')

args = parser.parse_args()


def audio_tagging(args,model):
    """Inference audio tagging result of an audio clip.
    """

    # Arugments & parameters
    sample_rate = args.sample_rate
    window_size = args.window_size
    hop_size = args.hop_size
    mel_bins = args.mel_bins
    fmin = args.fmin
    fmax = args.fmax
    audio_path = args.input
    
    classes_num = utilities.classes_num
    labels = utilities.labels

    # Load audio
    (waveform, _) = librosa.core.load(audio_path, sr=sample_rate, mono=True)

    waveform = waveform[None, :]    # (1, audio_length)

    clipwise_output = model.run(waveform)[0][0]

    sorted_indexes = np.argsort(clipwise_output)[::-1]

    # Print audio tagging top probabilities
    for k in range(10):
        print('{}: {:.3f}'.format(np.array(labels)[sorted_indexes[k]], 
            clipwise_output[sorted_indexes[k]]))
    if args.write_json:
        result = []
        for k in range(10):
            result.append({
                'label': np.array(labels)[sorted_indexes[k]],
                'prob': float(clipwise_output[sorted_indexes[k]])
            })
        with open('output_tagging.json', 'w') as f:
            json.dump(result, f, indent=2)

    return clipwise_output, labels


def sound_event_detection(args,model):
    """Inference sound event detection result of an audio clip."""

    # Arugments & parameters
    sample_rate = args.sample_rate
    window_size = args.window_size
    hop_size = args.hop_size
    mel_bins = args.mel_bins
    fmin = args.fmin
    fmax = args.fmax
    audio_path = args.input

    classes_num = utilities.classes_num
    labels = utilities.labels
    frames_per_second = sample_rate // hop_size

    # Load audio
    (waveform, _) = librosa.core.load(audio_path, sr=sample_rate, mono=True)

    waveform = waveform[None, :]    # (1, audio_length)

    framewise_output = model.run(waveform)[0][0]

    print('Sound event detection result (time_steps x classes_num): {}'.format(
        framewise_output.shape))

    sorted_indexes = np.argsort(np.max(framewise_output, axis=0))[::-1]

    top_k = 10  # Show top results
    top_result_mat = framewise_output[:, sorted_indexes[0 : top_k]]    
    """(time_steps, top_k)"""

    # Plot result    
    stft = librosa.core.stft(y=waveform[0], n_fft=window_size, 
        hop_length=hop_size, window='hann', center=True)
    frames_num = stft.shape[-1]

    fig, axs = plt.subplots(2, 1, sharex=True, figsize=(10, 4))
    axs[0].matshow(np.log(np.abs(stft)), origin='lower', aspect='auto', cmap='jet')
    axs[0].set_ylabel('Frequency bins')
    axs[0].set_title('Log spectrogram')
    axs[1].matshow(top_result_mat.T, origin='upper', aspect='auto', cmap='jet', vmin=0, vmax=1)
    axs[1].xaxis.set_ticks(np.arange(0, frames_num, frames_per_second))
    axs[1].xaxis.set_ticklabels(np.arange(0, frames_num / frames_per_second))
    axs[1].yaxis.set_ticks(np.arange(0, top_k))
    axs[1].yaxis.set_ticklabels(np.array(labels)[sorted_indexes[0 : top_k]])
    axs[1].yaxis.grid(color='k', linestyle='solid', linewidth=0.3, alpha=0.3)
    axs[1].set_xlabel('Seconds')
    axs[1].xaxis.set_ticks_position('bottom')

    plt.tight_layout()
    if args.savepath is None:
        plt.savefig(SAVE_PATH)
    else:
        plt.savefig(args.savepath)

    return framewise_output, labels

## ======================
## Main function
## ======================

def main(func):
    # model files check and download

    # create instance
    if args.mode == "audio_tagging":
        model = ailia.Net(None,WEIGHT_TAGGING_PATH, env_id=args.env_id)
    elif args.mode == "sound_event_detection":
        model = ailia.Net(None,WEIGHT_DETECTION_PATH, env_id=args.env_id)

    logger.info('Start inference...')
    if args.benchmark:
        logger.info('BENCHMARK mode')
        for c in range(5):
            start = int(round(time.time() * 1000))
            func(args,model)
            end = int(round(time.time() * 1000))
            logger.info("\tailia processing time {} ms".format(end-start))
    else:
        func(args,model)

    logger.info('Script finished successfully.')



if __name__ == '__main__':
    check_and_download_models(WEIGHT_TAGGING_PATH, MODEL_TAGGING_PATH, REMOTE_PATH)
    check_and_download_models(WEIGHT_DETECTION_PATH, MODEL_DETECTION_PATH, REMOTE_PATH)
    
    if args.mode == 'audio_tagging':
        main(audio_tagging)
    elif args.mode == 'sound_event_detection':
        main(sound_event_detection)
    else:
        raise Exception('Error argument!')


