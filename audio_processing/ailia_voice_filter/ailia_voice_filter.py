import sys
import time
from logging import getLogger

import ailia
import librosa
import numpy as np
import soundfile as sf

# import original modules
sys.path.append('../../util')
from arg_utils import get_base_parser, get_savepath, update_parser  # noqa: E402
from model_utils import check_and_download_file  # noqa: E402

from ailia_voice_filter_utils import (  # noqa: E402
    MODELS,
    SPEAKER_RATE,
    GraphStream,
    compute_dvector,
)

logger = getLogger(__name__)


# ======================
# Parameters
# ======================

WEIGHT_SPEAKER_PATH = 'speaker_encoder.onnx'
REMOTE_PATH = 'https://storage.googleapis.com/ailia-models/ailia_voice_filter/'

WAVE_PATH = 'mixed.wav'
REFERENCE_PATH = 'reference.wav'
SAVE_PATH = 'output.wav'


# ======================
# Arguemnt Parser Config
# ======================

parser = get_base_parser(
    'ailia Voice Filter', WAVE_PATH, SAVE_PATH, input_ftype='audio'
)
parser.add_argument(
    '-m', '--model', metavar='MODEL',
    default='voicefilter', choices=tuple(MODELS),
    help='model to use: voicefilter (keeps the enrolled speaker and removes '
         'everything else) or deepfilternet (denoising only)'
)
parser.add_argument(
    '-r', '--reference_file', metavar='REFERENCE',
    default=None, type=str,
    help='clean audio of the speaker to keep (default: %s). Required by '
         'voicefilter, refused by deepfilternet' % REFERENCE_PATH
)
args = update_parser(parser)


# ======================
# Main functions
# ======================

def enrol(stream, reference_path, env_id):
    """Turn a sample of the speaker's voice into the d-vector."""
    check_and_download_file(WEIGHT_SPEAKER_PATH, REMOTE_PATH)
    encoder = ailia.Net(None, WEIGHT_SPEAKER_PATH, env_id=env_id)

    logger.info('reference : %s' % reference_path)
    audio, _ = librosa.load(reference_path, sr=SPEAKER_RATE, mono=True)
    dvector = compute_dvector(encoder, audio)
    stream.enrol(dvector)


def enhance(stream, wav_path, save_path):
    audio, _ = librosa.load(wav_path, sr=stream.rate, mono=True)
    audio = audio.astype(np.float32)

    logger.info('input : %s' % wav_path)
    logger.info('Start inference...')
    if args.benchmark:
        logger.info('BENCHMARK mode')
        for i in range(5):
            start = int(round(time.time() * 1000))
            enhanced = stream.run(audio)
            end = int(round(time.time() * 1000))
            logger.info('\tailia processing time {} ms'.format(end - start))
    else:
        enhanced = stream.run(audio)

    sf.write(save_path, enhanced, stream.rate)

    before = 10 * np.log10(np.mean(audio.astype(np.float64) ** 2) + 1e-12)
    after = 10 * np.log10(np.mean(enhanced.astype(np.float64) ** 2) + 1e-12)
    logger.info('\t%.1f sec  input %.2f dBFS  output %.2f dBFS'
                % (len(audio) / stream.rate, before, after))
    logger.info('saved at : %s' % save_path)


def main():
    spec = MODELS[args.model]

    # model files check and download
    check_and_download_file(spec['graph'], REMOTE_PATH)

    env_id = args.env_id
    net = ailia.Net(None, spec['graph'], env_id=env_id)
    stream = GraphStream(net, args.model)

    if stream.conditioned:
        enrol(stream, args.reference_file or REFERENCE_PATH, env_id)
    elif args.reference_file is not None:
        logger.error('%s cannot be told a speaker; drop --reference_file'
                     % spec['name'])
        sys.exit(1)

    logger.info('%s  %d Hz  hop %d  latency %d samples'
                % (spec['name'], stream.rate, stream.hop, stream.latency))

    for wav_path in args.input:
        save_path = get_savepath(args.savepath, wav_path, ext='.wav')
        enhance(stream, wav_path, save_path)

    logger.info('Script finished successfully.')


if __name__ == '__main__':
    main()
