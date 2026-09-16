import os
import sys
import io

import numpy as np
import cv2

import torch
import torchaudio
import soundfile as sf

import ailia

# import original modules
sys.path.append('../../util')
from arg_utils import get_base_parser, update_parser, get_savepath  # noqa
from model_utils import check_and_download_models, urlretrieve, progress_print  # noqa
# logger
from logging import getLogger  # noqa

import df

logger = getLogger(__name__)

# ======================
# Parameters
# ======================

WEIGHT_UNET_PATH = 'unet.onnx'
WEIGHT_UNET_PB_PATH = 'unet_weights.pb'
MODEL_UNET_PATH = 'unet.onnx.prototxt'
WEIGHT_TEXT_ENCODER_PATH = 'text_encoder.onnx'
MODEL_TEXT_ENCODER_PATH = 'text_encoder.onnx.prototxt'
WEIGHT_VAE_DECODER_PATH = 'vae_decoder.onnx'
MODEL_VAE_DECODER_PATH = 'vae_decoder.onnx.prototxt'

REMOTE_PATH = 'https://storage.googleapis.com/ailia-models/riffusion/'

SAVE_WAV_PATH = 'output.wav'

# ======================
# Arguemnt Parser Config
# ======================

parser = get_base_parser(
    'Riffusion', None, SAVE_WAV_PATH, fp16_support=False
)
parser.add_argument(
    "-i", "--input", metavar="TEXT", type=str,
    default="jazzy rapping from paris",
    help="the prompt to render"
)
parser.add_argument(
    "--seed", type=int, default=42,
    help="random seed",
)
parser.add_argument(
    "--width", type=int, default=512,
    help="width",
)
parser.add_argument(
    '--onnx',
    action='store_true',
    help='execute onnxruntime version.'
)
parser.add_argument(
    '--disable_ailia_tokenizer',
    action='store_true',
    help='disable ailia tokenizer.'
)
args = update_parser(parser, check_input_type=False)


# ======================
# Secondaty Functions
# ======================

def audio_filters(segment):
    """
    Apply post-processing filters to the audio segment to compress it and
    keep at a -10 dBFS level.
    """
    # TODO(hayk): Come up with a principled strategy for these filters and experiment end-to-end.
    # TODO(hayk): Is this going to make audio unbalanced between sequential clips?
    import pydub

    desired_db = -12
    segment = segment.apply_gain(desired_db - segment.dBFS)

    segment = pydub.effects.normalize(
        segment,
        headroom=0.1,
    )

    return segment


# ======================
# Main functions
# ======================

class SpectrogramConverter:
    """
    Convert between audio segments and spectrogram tensors using torchaudio.

    In this class a "spectrogram" is defined as a (batch, time, frequency) tensor with float values
    that represent the amplitude of the frequency at that time bucket (in the frequency domain).
    Frequencies are given in the perceptul Mel scale defined by the params. A more specific term
    used in some functions is "mel amplitudes".

    The spectrogram computed from `spectrogram_from_audio` is complex valued, but it only
    returns the amplitude, because the phase is chaotic and hard to learn. The function
    `audio_from_spectrogram` is an approximate inverse of `spectrogram_from_audio`, which
    approximates the phase information using the Griffin-Lim algorithm.

    Each channel in the audio is treated independently, and the spectrogram has a batch dimension
    equal to the number of channels in the input audio segment.

    Both the Griffin Lim algorithm and the Mel scaling process are lossy.

    For more information, see https://pytorch.org/audio/stable/transforms.html
    """

    def __init__(self, params):
        self.p = params

        # https://pytorch.org/audio/stable/generated/torchaudio.transforms.GriffinLim.html
        self.inverse_spectrogram_func = torchaudio.transforms.GriffinLim(
            n_fft=params["n_fft"],
            n_iter=params["num_griffin_lim_iters"],
            win_length=params["win_length"],
            hop_length=params["hop_length"],
            window_fn=torch.hann_window,
            power=1.0,
            wkwargs=None,
            momentum=0.99,
            length=None,
            rand_init=True,
        )

        # https://pytorch.org/audio/stable/generated/torchaudio.transforms.InverseMelScale.html
        self.inverse_mel_scaler = torchaudio.transforms.InverseMelScale(
            n_stft=params["n_fft"] // 2 + 1,
            n_mels=params["num_frequencies"],
            sample_rate=params["sample_rate"],
            f_min=params["min_frequency"],
            f_max=params["max_frequency"],
            #max_iter=params["max_mel_iters"], # removed from latest torch audio
            #tolerance_loss=1e-5,
            #tolerance_change=1e-8,
            #sgdargs=None,
            norm=params["mel_scale_norm"],
            mel_scale=params["mel_scale_type"],
        )

    def audio_from_spectrogram(
            self,
            spectrogram: np.ndarray):
        """
        Reconstruct an audio segment from a spectrogram.

        Args:
            spectrogram: (batch, frequency, time)
        """
        # Move to device
        amplitudes_mel = torch.from_numpy(spectrogram)

        # Reconstruct the waveform
        waveform = self.waveform_from_mel_amplitudes(amplitudes_mel)
        waveform = waveform.cpu().numpy()

        # Normalize volume to fit in int16
        normalize = True
        if normalize:
            waveform *= np.iinfo(np.int16).max / np.max(np.abs(waveform))

        # Transpose and convert to int16
        samples = waveform.transpose(1, 0)
        samples = samples.astype(np.int16)

        return samples

    def waveform_from_mel_amplitudes(
            self,
            amplitudes_mel: torch.Tensor):
        """
        Torch-only function to approximately reconstruct a waveform from Mel-scale amplitudes.

        Args:
            amplitudes_mel: (batch, frequency, time)

        Returns:
            waveform: (batch, samples)
        """
        # Convert from mel scale to linear
        amplitudes_linear = self.inverse_mel_scaler(amplitudes_mel)

        # Run the approximate algorithm to compute the phase and recover the waveform
        waveform = self.inverse_spectrogram_func(amplitudes_linear)
        return waveform


def spectrogram_from_image(
        image: np.ndarray,
        power: float = 0.25,
        stereo: bool = False,
        max_value: float = 30e6) -> np.ndarray:
    """
    Compute a spectrogram magnitude array from a spectrogram image.

    This is the inverse of image_from_spectrogram, except for discretization error from
    quantizing to uint8.

    Args:
        image: (frequency, time, channels)
        power: The power curve applied to the spectrogram
        stereo: Whether the spectrogram encodes stereo data
        max_value: The max value of the original spectrogram. In practice doesn't matter.

    Returns:
        spectrogram: (channels, frequency, time)
    """

    # Flip Y
    image = image[::-1, :, :]

    # Munge channels into a numpy array of (channels, frequency, time)
    data = image.transpose(2, 0, 1)
    if stereo:
        # Take the G and B channels as done in image_from_spectrogram
        data = data[[1, 2], :, :]
    else:
        data = data[0:1, :, :]

    # Convert to floats
    data = data.astype(np.float32)

    # Invert
    data = 255 - data

    # Rescale to 0-1
    data = data / 255

    # Reverse the power curve
    data = np.power(data, 1 / power)

    # Rescale to max value
    data = data * max_value

    return data


def audio_from_spectrogram_image(
        converter,
        image,
        max_value: float = 30e6):
    """
    Reconstruct an audio segment from a spectrogram image.

    Args:
        image: Spectrogram image (in pillow format)
        apply_filters: Apply post-processing to improve the reconstructed audio
        max_value: Scaled max amplitude of the spectrogram. Shouldn't matter.
    """
    spectrogram = spectrogram_from_image(
        image,
        max_value=max_value,
        power=converter.p["power_for_image"],
        stereo=converter.p["stereo"],
    )

    samples = converter.audio_from_spectrogram(spectrogram)

    return samples


def recognize_from_text(pipe):
    prompt = args.input if isinstance(args.input, str) else args.input[0]
    negative_prompt = ""
    num_inference_steps = 30
    guidance = 7.0
    width = args.width
    logger.info("prompt: %s" % prompt)

    logger.info('Start inference...')

    image = pipe.forward(
        prompt=prompt,
        num_inference_steps=num_inference_steps,
        guidance_scale=guidance,
        negative_prompt=negative_prompt or None,
        width=width,
        height=512,
    )
    image = (image[0] * 255).astype(np.uint8)

    sample_rate = 44100
    converter = SpectrogramConverter(params=dict(
        n_fft=17640,
        hop_length=441,
        win_length=4410,
        num_frequencies=512,
        sample_rate=sample_rate,
        min_frequency=0,
        max_frequency=10000,
        max_mel_iters=200,
        mel_scale_norm=None,
        mel_scale_type="htk",
        num_griffin_lim_iters=32,
        power_for_image=0.25,
        stereo=False,
    ))

    audio_savepath = get_savepath(args.savepath, "", ext='.wav')
    p, _ = os.path.splitext(audio_savepath)
    img_savepath = p + ".png"
    logger.info(f'saved at : {img_savepath}')
    cv2.imwrite(img_savepath, image)

    samples = audio_from_spectrogram_image(converter, image)
    try:
        from scipy.io import wavfile
        import pydub

        # Write to the bytes of a WAV file
        wav_bytes = io.BytesIO()
        wavfile.write(wav_bytes, sample_rate, samples)
        wav_bytes.seek(0)

        # Read into pydub
        segment = pydub.AudioSegment.from_wav(wav_bytes)

        # Optionally apply post-processing filters
        apply_filters = True
        if apply_filters:
            segment = audio_filters(segment)

        logger.info(f'saved at : {audio_savepath}')
        segment.export(audio_savepath, format="wav")

    except ModuleNotFoundError:
        logger.info(f'saved at : {audio_savepath}')
        sf.write(audio_savepath, samples, sample_rate, 'PCM_16', format='WAV')

    logger.info('Script finished successfully.')


def main():
    check_and_download_models(WEIGHT_UNET_PATH, MODEL_UNET_PATH, REMOTE_PATH)
    check_and_download_models(WEIGHT_TEXT_ENCODER_PATH, MODEL_TEXT_ENCODER_PATH, REMOTE_PATH)
    check_and_download_models(WEIGHT_VAE_DECODER_PATH, MODEL_VAE_DECODER_PATH, REMOTE_PATH)

    if not os.path.exists(WEIGHT_UNET_PB_PATH):
        urlretrieve(
            REMOTE_PATH + WEIGHT_UNET_PB_PATH,
            WEIGHT_UNET_PB_PATH,
            progress_print,
        )

    seed = args.seed
    if seed is not None:
        np.random.seed(seed)

    env_id = args.env_id

    # initialize
    if not args.onnx:
        net = ailia.Net(MODEL_UNET_PATH, WEIGHT_UNET_PATH, env_id=env_id, memory_mode=args.memory_mode)
        text_encoder = ailia.Net(MODEL_TEXT_ENCODER_PATH, WEIGHT_TEXT_ENCODER_PATH, env_id=env_id, memory_mode=args.memory_mode)
        vae_decoder = ailia.Net(MODEL_VAE_DECODER_PATH, WEIGHT_VAE_DECODER_PATH, env_id=env_id, memory_mode=args.memory_mode)
    else:
        import onnxruntime
        cuda = 0 < ailia.get_gpu_environment_id()
        providers = ['CUDAExecutionProvider', 'CPUExecutionProvider'] if cuda else ['CPUExecutionProvider']

        net = onnxruntime.InferenceSession(WEIGHT_UNET_PATH, providers=providers)
        text_encoder = onnxruntime.InferenceSession(WEIGHT_TEXT_ENCODER_PATH, providers=providers)
        vae_decoder = onnxruntime.InferenceSession(WEIGHT_VAE_DECODER_PATH, providers=providers)

    if args.disable_ailia_tokenizer:
        import transformers
        tokenizer = transformers.CLIPTokenizer.from_pretrained(
            "./tokenizer"
        )
    else:
        import ailia_tokenizer
        tokenizer = ailia_tokenizer.CLIPTokenizer.from_pretrained()
        tokenizer.model_max_length = 77

    scheduler = df.schedulers.DPMSolverMultistepScheduler.from_config({
        "num_train_timesteps": 1000,
        "beta_start": 0.00085,
        "beta_end": 0.012,
        "beta_schedule": "scaled_linear",
        "trained_betas": None,
        "skip_prk_steps": True,
        "set_alpha_to_one": False,
        "prediction_type": "epsilon",
        "steps_offset": 1,
    })

    pipe = df.StableDiffusion(
        vae_decoder=vae_decoder,
        text_encoder=text_encoder,
        tokenizer=tokenizer,
        unet=net,
        scheduler=scheduler,
        use_onnx=args.onnx,
    )

    # generate
    recognize_from_text(pipe)


if __name__ == '__main__':
    main()
