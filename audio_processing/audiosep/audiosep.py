import sys
import time
import math
from logging import getLogger

import scipy
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

QUERY_WEIGHT_PATH = 'audiosep_text.onnx'
SEPNET_WEIGHT_PATH = 'audiosep_resunet.onnx'

QUERY_MODEL_PATH = 'audiosep_text.onnx.prototxt'
SEPNET_MODEL_PATH = 'audiosep_resunet.onnx.prototxt'

REMOTE_PATH = "https://storage.googleapis.com/ailia-models/audiosep/"

# ======================
# Arguemnt Parser Config
# ======================

parser = get_base_parser(
    'audiosep', "input.wav", None
)

parser.add_argument(
    "-p", "--prompt", metavar="TEXT", type=str,
    default="water drops",
    help="Text query."
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

"""
Functions below are taken from https://github.com/Audio-AGI/AudioSep, which was released under MIT license.
Modified to be run with numpy arrays instead of torch tensors
"""
def preprocess_mag(mag):
    #batch normalize self.bn0 = nn.BatchNorm2d(window_size // 2 + 1, momentum=momentum)
    mag = np.transpose(mag, (0,3,2,1))
    mag = (mag - np.mean(mag, axis=(2,3), keepdims=True)) / (np.std(mag, axis=(2,3), keepdims=True) + 1e-5)
    mag = np.transpose(mag, (0,3,2,1))
    p = math.ceil(mag.shape[2] / 2**5) * 2**5 - mag.shape[2]
    mag = np.pad(mag, ((0,0),(0,0),(0,p),(0,0)))
    mag = mag[:,:,:,0:mag.shape[-1]-1]
    return mag

def spectrogram_phase(input, eps=0.):
    D = librosa.stft(
         input,
         n_fft=2048,
         hop_length=320,
         win_length=2048,
         window='hann',
         center=True,
         pad_mode='reflect'
    )
    real = np.real(D)
    imag = np.imag(D)
    mag = np.clip(real ** 2 + imag ** 2, eps, np.inf) ** 0.5
    cos = real / mag# normalize
    sin = imag / mag# normalize
    return mag, cos, sin

def wav_to_spectrogram(input, eps=1e-10):
        """Waveform to spectrogram.

        Args:
          input: (batch_size, segment_samples, channels_num)

        Outputs:
          output: (batch_size, channels_num, time_steps, freq_bins)
        """
        sp_list = []
        cos_list = []
        sin_list = []
        channels_num = input.shape[1]
        for channel in range(channels_num):
            mag, cos, sin = spectrogram_phase(input[:, channel, :], eps=eps)
            sp_list.append(mag)
            cos_list.append(cos)
            sin_list.append(sin)

        sps = np.concatenate(sp_list, axis=1)
        coss = np.concatenate(cos_list, axis=1)
        sins = np.concatenate(sin_list, axis=1)
        return sps, coss, sins

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def feature_maps_to_wav(
        input_tensor,
        sp,
        sin_in,
        cos_in,
        audio_length,
    ):
        batch_size, _, time_steps, freq_bins = input_tensor.shape

        x = input_tensor.reshape(
            batch_size,
            1,
            1,
            3,
            time_steps,
            freq_bins,
        )
        # x: (batch_size, target_sources_num, output_channels, self.K, time_steps, freq_bins)

        mask_mag = sigmoid(x[:, :, :, 0, :, :])
        _mask_real = np.tanh(x[:, :, :, 1, :, :])
        _mask_imag = np.tanh(x[:, :, :, 2, :, :])
        # linear_mag = torch.tanh(x[:, :, :, 3, :, :])
        _, phase = librosa.magphase(_mask_real + 1j*_mask_imag)
        #norm = (np.real(phase)**2 + np.imag(phase)**2)**0.5
        mask_cos = np.real(phase)
        mask_sin = np.imag(phase)

        # Y = |Y|cos∠Y + j|Y|sin∠Y
        #   = |Y|cos(∠X + ∠M) + j|Y|sin(∠X + ∠M)
        #   = |Y|(cos∠X cos∠M - sin∠X sin∠M) + j|Y|(sin∠X cos∠M + cos∠X sin∠M)
        out_cos = (
            cos_in[:, None, :, :, :] * mask_cos - sin_in[:, None, :, :, :] * mask_sin
        )
        out_sin = (
            sin_in[:, None, :, :, :] * mask_cos + cos_in[:, None, :, :, :] * mask_sin
        )
        # out_cos: (batch_size, target_sources_num, output_channels, time_steps, freq_bins)
        # out_sin: (batch_size, target_sources_num, output_channels, time_steps, freq_bins)

        # Calculate |Y|.
        out_mag = np.max(sp[:, None, :, :, :] * mask_mag, 0)
        # out_mag = F.relu_(sp[:, None, :, :, :] * mask_mag + linear_mag)
        # out_mag: (batch_size, target_sources_num, output_channels, time_steps, freq_bins)

        # Calculate Y_{real} and Y_{imag} for ISTFT.
        out_real = out_mag * out_cos
        out_imag = out_mag * out_sin
        # out_real, out_imag: (batch_size, target_sources_num, output_channels, time_steps, freq_bins)

        # Reformat shape to (N, 1, time_steps, freq_bins) for ISTFT where
        # N = batch_size * target_sources_num * output_channels
        shape = (
            batch_size,
            1,
            time_steps,
            freq_bins,
        )
        out_real = out_real.reshape(shape)
        out_imag = out_imag.reshape(shape)

        x = librosa.istft(
             (out_real +  1j * out_imag)[0,0].astype('complex64').transpose((1,0)),
             n_fft = 2048,
             hop_length = 320,
             win_length = 2048,
             window = 'hann',
             center = True,
             length = audio_length,
            )

        return x

# ======================
# Main functions
# ======================

def inference(model, input_text, input_wav):
    # tokenize
    tokenizer = model['tokenizer']
    text_prompt_tkn = dict(tokenizer(input_text, return_tensors = 'np', padding = True))
    text_prompt_tkn = (text_prompt_tkn['input_ids'], text_prompt_tkn['attention_mask'])

    # prepare audio input
    mag, cosin, sinin = wav_to_spectrogram(input_wav)
    orig_len = mag.shape[-1]
    mag = mag.transpose((0,2,1))[None]
    cosin = cosin.transpose((0,2,1))[None]
    sinin = sinin.transpose((0,2,1))[None]

    # preprocess
    mag_in = preprocess_mag(mag)

    # inference
    query = model['querynet'].predict(text_prompt_tkn)[0]

    output = model['sepnet'].predict((query, mag_in))[0]

    # postprocess
    output = output[:,:,:orig_len,:]# trim to original length

    output_wav = feature_maps_to_wav(output, mag, sinin, cosin, input_wav.shape[-1])

    return output_wav

def split_audio(model):
    input_text = args.prompt
    input_wav = librosa.load(args.input[0], sr=32000, mono=True)[0][None,None,:]

    logger.info("input_text: %s" % input_text)

    # inference
    logger.info('inference has started...')
    if args.benchmark:
        logger.info('BENCHMARK mode')
        total_time_estimation = 0
        for i in range(args.benchmark_count):
            start = int(round(time.time() * 1000))
            output = inference(model, input_text, input_wav)
            end = int(round(time.time() * 1000))
            estimation_time = (end - start)

            # Logging
            logger.info(f'\tailia processing estimation time {estimation_time} ms')
            if i != 0:
                total_time_estimation = total_time_estimation + estimation_time

        logger.info(f'\taverage time estimation {total_time_estimation / (args.benchmark_count - 1)} ms')
    else:
        output = inference(model, input_text, input_wav)

    # save output
    if args.savepath is None:
        sp = 'output.wav'
    else:
        sp = args.savepath
    scipy.io.wavfile.write(sp, 32000, np.round(output * 32767).astype(np.int16))

    logger.info(f"Separated audio has been saved to {sp}")

    logger.info('Script finished successfully.')


def main():
    # model files check and download
    check_and_download_models(QUERY_WEIGHT_PATH, QUERY_MODEL_PATH, REMOTE_PATH)
    check_and_download_models(SEPNET_WEIGHT_PATH, SEPNET_MODEL_PATH, REMOTE_PATH)

    env_id = args.env_id

    # initialize
    querynet = ailia.Net(None, QUERY_WEIGHT_PATH, env_id=env_id)
    sepnet = ailia.Net(None, SEPNET_WEIGHT_PATH, env_id=env_id)
    if args.disable_ailia_tokenizer:
        from transformers import RobertaTokenizer
        tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
    else:
        import ailia_tokenizer
        tokenizer = ailia_tokenizer.RobertaTokenizer.from_pretrained('./tokenizer/')
    model = {
        'querynet': querynet,
        'sepnet':sepnet,
        'tokenizer':tokenizer
    }

    split_audio(model)

if __name__ == '__main__':
    main()
