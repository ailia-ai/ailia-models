import sys
import time

import numpy as np
import librosa

import ailia

# import original modules
sys.path.append('../../util')
from arg_utils import get_base_parser, update_parser  # noqa: E402
from model_utils import check_and_download_models  # noqa: E402

from kaldifeat import compute_fbank_feats

# logger
from logging import getLogger  # noqa: E402

logger = getLogger(__name__)

use_torch = True

try:
    import torch
    import torchaudio.compliance.kaldi as kaldi
except ImportError:
    if use_torch:
        logger.warning("The torchaudio is not installed, use another means.")
    use_torch = False

# ======================
# Parameters
# ======================

WEIGHT_VOX_PATH = 'voxceleb_resnet34.onnx'
MODEL_VOX_PATH = 'voxceleb_resnet34.onnx.prototxt'
WEIGHT_CNC_PATH = 'cnceleb_resnet34.onnx'
MODEL_CNC_PATH = 'cnceleb_resnet34.onnx.prototxt'
REMOTE_PATH = 'https://storage.googleapis.com/ailia-models/wespeaker/'

THRESHOLD = 0.7

# ======================
# Arguemnt Parser Config
# ======================

parser = get_base_parser(
    'WeSpeaker', None, None, input_ftype='audio'
)
parser.add_argument(
    '-i1', '--input1', metavar='WAV', default=None,
    help='Specify an wav file to compare with the input2 wav.'
)
parser.add_argument(
    '-i2', '--input2', metavar='WAV', default=None,
    help='Specify an wav file to compare with the input1 wav.'
)
parser.add_argument(
    '-th', '--threshold',
    default=THRESHOLD, type=float,
    help='The similar threshold for verification.'
)
parser.add_argument(
    '-en', '--english', action='store_true',
    help='Language is English.'
)
parser.add_argument(
    '-cn', '--chinese', action='store_true',
    help='Language is Chinese.'
)
parser.add_argument(
    '--onnx',
    action='store_true',
    help='execute onnxruntime version.'
)
args = update_parser(parser)


# ======================
# Secondaty Functions
# ======================

def read_audio(wav_path, resample_rate=16000):
    waveform, sample_rate = librosa.load(wav_path, sr=None)

    # Resample the wav if needed
    if resample_rate is not None and sample_rate != resample_rate:
        waveform = ailia.audio.resample(waveform, org_sr=sample_rate, target_sr=resample_rate)

    return waveform, sample_rate


def cosine_score(emb1, emb2):
    """ Compute cosine score between emb1 and emb2.
    Args:
        emb1(numpy.ndarray): embedding of speaker-1
        emb2(numpy.ndarray): embedding of speaker-2
    Return:
        score(float): cosine score
    """
    return np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2))


# ======================
# Main functions
# ======================

def extract_fbank_features(
        waveform: np.ndarray, sample_rate, cmn=True) -> np.ndarray:
    """
    the waveform should not be normalized before feature extraction.
    """
    num_mel_bins = 80
    frame_length = 25
    frame_shift = 10
    dither = 0.0
    waveform = (waveform * (1 << 15))  # 16-bit signed integers
    if use_torch:
        waveform = torch.from_numpy(waveform).unsqueeze(0)
        mat = kaldi.fbank(
            waveform,
            num_mel_bins=num_mel_bins,
            frame_length=frame_length,
            frame_shift=frame_shift,
            dither=dither,
            sample_frequency=sample_rate,
            window_type='hamming',
            use_energy=False)
        feats = mat.numpy()
    else:
        feats = compute_fbank_feats(
            waveform,
            dither=dither,
            frame_length=frame_length,
            frame_shift=frame_shift,
            num_mel_bins=num_mel_bins,
        )

    if cmn:
        # CMN, without CVN
        feats = feats - np.mean(feats, axis=0)

    return feats


def predict(waveform, net, sample_rate):
    # initial preprocesses
    feats = extract_fbank_features(waveform, sample_rate)
    feats = np.expand_dims(feats, axis=0)
    feats = feats.astype(np.float32)

    # feedforward
    if not args.onnx:
        output = net.predict([feats])
    else:
        output = net.run(None, {'feats': feats})
    embeddings = output[0]

    return embeddings


def speaker_verification(net):
    input1 = args.input1 or (args.input[0] if args.input else None)
    input2 = args.input2
    threshold = args.threshold

    if input1 is None or input2 is None:
        logger.error('Please specified input1 and input2 audios')
        sys.exit(-1)

    logger.info(f'input1: {input1}')
    logger.info(f'input2: {input2}')

    # prepare input data
    wavform1, sample_rate1 = read_audio(input1)
    wavform2, sample_rate2 = read_audio(input2)

    # inference
    logger.info('Start inference...')
    if args.benchmark:
        logger.info('BENCHMARK mode')
        for i in range(5):
            start = int(round(time.time() * 1000))
            emb1 = predict(wavform1, net, sample_rate1)
            emb2 = predict(wavform2, net, sample_rate2)
            end = int(round(time.time() * 1000))
            logger.info(f'\tailia processing time {end - start} ms')
    else:
        emb1 = predict(wavform1, net, sample_rate1)
        emb2 = predict(wavform2, net, sample_rate2)

    cos_score = cosine_score(emb1[0], emb2[0])
    cos_score = (cos_score + 1) / 2.0

    logger.info('The speakers are {:.1f}% similar'.format(cos_score * 100))
    logger.info('Welcome, human!' if cos_score >= threshold else 'Warning! stranger!')

    logger.info('Script finished successfully.')


def main():
    # model files check and download
    check_and_download_models(WEIGHT_VOX_PATH, MODEL_VOX_PATH, REMOTE_PATH)
    check_and_download_models(WEIGHT_CNC_PATH, MODEL_CNC_PATH, REMOTE_PATH)
    MODEL_PATH, WEIGHT_PATH = (MODEL_VOX_PATH, WEIGHT_VOX_PATH) \
        if args.english or not args.chinese else (MODEL_CNC_PATH, WEIGHT_CNC_PATH)

    env_id = args.env_id

    # initialize
    if not args.onnx:
        net = ailia.Net(MODEL_PATH, WEIGHT_PATH, env_id=env_id, memory_mode=args.memory_mode)
    else:
        import onnxruntime
        cuda = 0 < ailia.get_gpu_environment_id()
        providers = ['CUDAExecutionProvider', 'CPUExecutionProvider'] if cuda else ['CPUExecutionProvider']
        net = onnxruntime.InferenceSession(WEIGHT_PATH, providers=providers)

    speaker_verification(net)


if __name__ == '__main__':
    main()
