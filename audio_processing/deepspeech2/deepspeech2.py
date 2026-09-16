import sys
import time

import numpy as np
import pyaudio

import ailia
# import original moduls
sys.path.append('../../util')
from arg_utils import get_base_parser, update_parser, get_savepath  # noqa: E402
from model_utils import check_and_download_models  # noqa: E402

# logger
from logging import getLogger   # noqa: E402
logger = getLogger(__name__)


# ======================
# Parameters
# ======================
MODEL_LISTS = [
    'an4_pretrained_v2', 'librispeech_pretrained_v2', 'ted_pretrained_v2'
]

DEFAULT_MODEL = 'librispeech_pretrained_v2'

WEIGHT_PATH = 'librispeech_pretrained_v2.onnx'
MODEL_PATH = 'librispeech_pretrained_v2.onnx.prototxt'
REMOTE_PATH = 'https://storage.googleapis.com/ailia-models/deepspeech2/'

WAV_PATH = './1221-135766-0000.wav'
SAVE_TEXT_PATH = 'output.txt'

SAMPLING_RATE = 16000
WIN_LENGTH = int(SAMPLING_RATE * 0.02)
HOP_LENGTH = int(SAMPLING_RATE * 0.01)

LABELS = list('_\'ABCDEFGHIJKLMNOPQRSTUVWXYZ ')
int_to_char = dict([(i, c) for (i, c) in enumerate(LABELS)])
BRANK_LABEL_INDEX = 0

# BeamCTCDecoder parameter
LM_PATH = '3-gram.pruned.3e-7.arpa'
ALPHA = 1.97
BETA = 4.36
CUTOFF_TOP_N = 40
CUTOFF_PROB = 1.0
NUM_PROCESS = 1
BEAM_WIDTH = 128

# pyaudio
CHUNK = 1024
FORMAT = pyaudio.paInt16
CHANNELS = 1
RECODING_SAMPING_RATE = 48000
THRESHOLD = 0.02

# ======================
# Arguemnt Parser Config
# ======================
parser = get_base_parser(
    'deepspeech2', WAV_PATH, SAVE_TEXT_PATH, input_ftype='audio'
)
# overwrite
parser.add_argument(
    '-i', '--input', metavar='WAV',
    default=WAV_PATH,
    help='The input wav path.',
)
parser.add_argument(
    '-V',
    action='store_true',
    help='use microphone input',
)
parser.add_argument(
    '-d', '--beamdecode',
    action='store_true',
    help='use beam decoder',
)
parser.add_argument(
    '-a', '--arch', metavar='WEIGHT',
    default=DEFAULT_MODEL, choices=MODEL_LISTS,
    help='model lists: ' + ' | '.join(MODEL_LISTS)
)
parser.add_argument(
    '--ailia_audio', action='store_true',
    help='use ailia audio library'
)
args = update_parser(parser)

if args.ailia_audio:
  import ailia.audio
  import soundfile as sf
else:
  import librosa
  import torch

# ======================
# Utils
# ======================
def create_spectrogram(wav):
    if args.ailia_audio:
        stft = ailia.audio.spectrogram(
            wav,
            fft_n=WIN_LENGTH,
            hop_n=HOP_LENGTH,
            win_n=WIN_LENGTH,
            win_type="hamming",
        )
        stft, _ = ailia.audio.magphase(stft)
    else:
        stft = librosa.stft(
            wav,
            n_fft=WIN_LENGTH,
            win_length=WIN_LENGTH,
            hop_length=HOP_LENGTH,
            window='hamming',
        )
        stft, _ = librosa.magphase(stft)

    spectrogram = np.log1p(stft)
    spec_length = np.array(([stft.shape[1]-1]))

    mean = spectrogram.mean()
    std = spectrogram.std()
    spectrogram -= mean
    spectrogram /= std
    
    spectrogram = np.log1p(spectrogram)
    spectrogram = spectrogram[np.newaxis, np.newaxis, :, :]

    return (spectrogram, spec_length)


def record_microphone_input():
    logger.info('Ready...')
    time.sleep(1)
    p = pyaudio.PyAudio()

    stream = p.open(
        format=FORMAT,
        channels=CHANNELS,
        rate=RECODING_SAMPING_RATE,
        input=True,
        frames_per_buffer=CHUNK,
    )

    # time.sleep(1)
    logger.info("Please speak something")

    frames = []
    count_uv = 0

    stream.start_stream()
    while True:
        data = np.frombuffer(stream.read(CHUNK), dtype=np.int16) / 32768.0
        if data.max() > THRESHOLD:
            frames.extend(data)
            count_uv = 0
        elif len(frames) > 0:
            count_uv += 1
            if count_uv > 48:
                break
            frames.extend(data)

    # logger.info("Translating")

    stream.stop_stream()
    stream.close()
    p.terminate()

    wav = np.array(frames)
    if args.ailia_audio:
        return ailia.audio.resample(wav, RECODING_SAMPING_RATE, SAMPLING_RATE)
    else:
        return librosa.resample(wav, RECODING_SAMPING_RATE, SAMPLING_RATE)


def decode(sequence, size=None):
    sequence = np.argmax(sequence, -1)

    text = ''
    size = int(size[0]) if size is not None else len(sequence)
    for i in range(size):
        char = int_to_char[sequence[i]]
        if char != int_to_char[BRANK_LABEL_INDEX]:
            if i != 0 and char == int_to_char[sequence[i - 1]]:
                pass
            else:
                text += char
    return text.lower()


def beam_ctc_decode(sequence, size=None, decoder=None):
    """
    Decode using language model
    """
    out, scores, offsets, seq_len = decoder.decode(sequence, size)

    # results = []
    for b, batch in enumerate(out):
        utterances = []
        for p, utt in enumerate(batch):
            size = seq_len[0][p]
            if size > 0:
                transcript = ''.join(
                    map(lambda x: int_to_char[x.item()], utt[0:size])
                )
            else:
                transcript = ''
        utterances.append(transcript)

    return utterances[0].lower()


# ======================
# Main functions
# ======================
def wavfile_input_recognition():
    if args.beamdecode:
        try:
            from ctcdecode import CTCBeamDecoder
        except ImportError:
            raise ImportError("BeamCTCDecoder requires paddledecoder package.")

        decoder = CTCBeamDecoder(
            LABELS,
            LM_PATH,
            ALPHA,
            BETA,
            CUTOFF_TOP_N,
            CUTOFF_PROB,
            BEAM_WIDTH,
            NUM_PROCESS,
            BRANK_LABEL_INDEX,
        )

    # net initialize
    net = ailia.Net(MODEL_PATH, WEIGHT_PATH, env_id=args.env_id, memory_mode=args.memory_mode)

    for soundf_path in args.input:
        logger.info(soundf_path)
        if args.ailia_audio:
            wav,sr = sf.read(soundf_path)
            wav = ailia.audio.resample(wav,sr,SAMPLING_RATE)
        else:
            wav = librosa.load(soundf_path, sr=SAMPLING_RATE)[0]
        spectrogram = create_spectrogram(wav)
        net.set_input_shape(spectrogram[0].shape)

        # inference
        logger.info('Start inference...')
        if args.benchmark:
            logger.info('BENCHMARK mode')
            for c in range(5):
                start = int(round(time.time() * 1000))
                preds_ailia, output_length = net.predict(spectrogram)
                end = int(round(time.time() * 1000))
                logger.info("\tailia processing time {} ms".format(end-start))
        else:
            # Deep Speech output: output_probability, output_length
            preds_ailia, output_length = net.predict(spectrogram)

        if args.beamdecode:
            text = beam_ctc_decode(
                torch.from_numpy(preds_ailia),
                torch.from_numpy(output_length),
                decoder,
            )
        else:
            text = decode(preds_ailia[0], output_length)

        savepath = get_savepath(args.savepath, soundf_path, ext='.txt')
        logger.info(f'Results saved at : {savepath}')
        with open(savepath, 'w', encoding='utf-8') as f:
            f.write(text)
        logger.info(f'predict sentence:\n{text}')
    logger.info('Script finished successfully.')


# ======================
# microphone input mode
# ======================
def microphone_input_recognition():
    if args.beamdecode:
        try:
            from ctcdecode import CTCBeamDecoder
        except ImportError:
            raise ImportError("BeamCTCDecoder requires paddledecoder package.")

        decoder = CTCBeamDecoder(
            LABELS,
            LM_PATH,
            ALPHA,
            BETA,
            CUTOFF_TOP_N,
            CUTOFF_PROB,
            BEAM_WIDTH,
            NUM_PROCESS,
            BRANK_LABEL_INDEX,
        )

    while True:
        wav = record_microphone_input()
        spectrogram = create_spectrogram(wav)

        # net initialize
        net = ailia.Net(MODEL_PATH, WEIGHT_PATH, env_id=args.env_id, memory_mode=args.memory_mode)
        net.set_input_shape(spectrogram[0].shape)

        # inference
        logger.info('Translating...')
        # Deep Speech output: output_probability, output_length
        preds_ailia, output_length = net.predict(spectrogram)

        if args.beamdecode:
            text = beam_ctc_decode(
                torch.from_numpy(preds_ailia),
                torch.from_numpy(output_length),
                decoder,
            )
        else:
            text = decode(preds_ailia[0], output_length)

        logger.info(f'predict sentence:\n{text}\n')
        time.sleep(1)


def main():
    global WEIGHT_PATH, MODEL_PATH
    if args.arch != WEIGHT_PATH:
        WEIGHT_PATH = args.arch + '.onnx'
        MODEL_PATH = WEIGHT_PATH + '.prototxt'

    check_and_download_models(WEIGHT_PATH, MODEL_PATH, REMOTE_PATH)
    check_and_download_models(LM_PATH, LM_PATH, REMOTE_PATH)

    # microphone input mode
    if args.V:
        try:
            microphone_input_recognition()
        except KeyboardInterrupt:
            logger.info('script finished successfully.')

    # sound file input mode
    else:
        wavfile_input_recognition()


if __name__ == "__main__":
    main()
