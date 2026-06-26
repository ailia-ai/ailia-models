import librosa
import numpy as np
import tqdm

# hard-coded audio hyperparameters
SAMPLE_RATE = 16000
N_FFT = 400
N_MELS = 80
HOP_LENGTH = 160
CHUNK_LENGTH = 30
N_SAMPLES = CHUNK_LENGTH * SAMPLE_RATE  # 480000: number of samples in a chunk
N_FRAMES = N_SAMPLES // HOP_LENGTH  # 3000: number of frames in a mel spectrogram input


def pad_or_trim(array, length: int = N_SAMPLES, *, axis: int = -1):
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


def log_mel_spectrogram(audio, n_mels: int = N_MELS):
    """
    Compute the log-Mel spectrogram of
    """
    stft = librosa.stft(
        y=audio,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH,
        window="hann",
        pad_mode="reflect",
    )
    magnitudes = np.abs(stft[:, :-1]) ** 2

    filters = librosa.filters.mel(sr=SAMPLE_RATE, n_fft=N_FFT, n_mels=n_mels)
    mel_spec = filters @ magnitudes

    log_spec = np.log10(np.clip(mel_spec, 1e-10, None))
    log_spec = np.maximum(log_spec, np.max(log_spec) - 8.0)
    log_spec = (log_spec + 4.0) / 4.0

    return log_spec


def transcribe(
    model,
    audio,
    verbose: bool = True,
    use_onnx: bool = False,
):
    """
    Transcribe an audio file using Whisper
    """
    mel = log_mel_spectrogram(audio)

    all_segments = []

    def add_segment(*, start: float, end: float, encoder_embeddings):
        all_segments.append(
            {
                "start": start,
                "end": end,
                "encoder_embeddings": encoder_embeddings,
            }
        )

    # show the progress bar when verbose is False (otherwise the transcribed text will be printed)
    num_frames = mel.shape[-1]
    seek = 0
    sample_skip = 3000
    with tqdm.tqdm(total=num_frames, unit="frames", disable=not verbose) as pbar:
        while seek < num_frames:
            # seek是开始的帧数
            end_seek = min(seek + sample_skip, num_frames)
            segment = pad_or_trim(mel[:, seek : seek + sample_skip], N_FRAMES)
            segment = segment.astype(np.float16)

            single = segment.ndim == 2
            if single:
                segment = np.expand_dims(segment, axis=0)

            if not use_onnx:
                output = model.predict([segment])
            else:
                output = model.run(None, {"segment": segment})
            embeddings = output[0]

            encoder_embeddings = embeddings
            add_segment(
                start=seek,
                end=end_seek,
                encoder_embeddings=encoder_embeddings,
            )
            seek += sample_skip

            pbar.update(sample_skip)

    return dict(segments=all_segments)
