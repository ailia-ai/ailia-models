import numpy as np
import ailia.audio
import soundfile as sf

# hard-coded audio hyperparameters
SAMPLE_RATE = 16000
N_FFT = 400
HOP_LENGTH = 160
CHUNK_LENGTH = 30
N_SAMPLES = CHUNK_LENGTH * SAMPLE_RATE  # 480000: number of samples in a chunk
N_FRAMES = (N_SAMPLES // HOP_LENGTH)  # 3000: number of frames in a mel spectrogram input


def load_audio(file: str, sr: int = SAMPLE_RATE):
    # prepare input data
    wav, source_sr = sf.read(file)
    # convert to mono
    if len(wav.shape) >= 2 and wav.shape[1] == 2:
        wav = np.mean(wav, axis=1)
    # Resample the wav if needed
    if source_sr is not None and source_sr != sr:
        wav = ailia.audio.resample(wav, org_sr=source_sr, target_sr=sr)
    return wav


def pad_or_trim(array, length=N_SAMPLES, axis=-1):
    """
    Pad or trim the audio array to N_SAMPLES, as expected by the encoder.
    """
    if array.shape[axis] > length:
        array = array.take(indices=range(length), axis=axis)

    if array.shape[axis] < length:
        pad_widths = [(0, 0)] * array.ndim
        pad_widths[axis] = (0, length - array.shape[axis])
        array = np.pad(array, pad_widths)

    return array


def log_mel_spectrogram(audio, n_mels: int = 80, padding: int = 0):
    """
    Compute the log-Mel spectrogram of

    Parameters
    ----------
    audio: np.ndarray
    n_mels: int
        The number of Mel-frequency filters, only 80 is supported
    padding: int
        Number of zero samples to pad to the right

    Returns
    -------
    A Tensor that contains the Mel spectrogram, shape = (80, n_frames)
    """
    if padding > 0:
        audio = np.pad(audio, (0, padding))

    mel_spec = ailia.audio.mel_spectrogram(
        audio, sample_rate=SAMPLE_RATE, fft_n=N_FFT, hop_n=HOP_LENGTH,
        win_type="hann", center_mode=1, power=2.0, mel_n=n_mels)

    log_spec = np.log10(np.clip(mel_spec, 1e-10, None))
    log_spec = np.maximum(log_spec, np.max(log_spec) - 8.0)
    log_spec = (log_spec + 4.0) / 4.0

    return log_spec
