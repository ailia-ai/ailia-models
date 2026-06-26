import numpy as np
import librosa
from librosa.filters import mel as librosa_mel_fn

MAX_WAV_VALUE = 32768.0


def load_wav(file, target_sr):
    # prepare input data
    audio, source_sr = librosa.load(file, sr=None)
    # Resample the wav if needed
    if source_sr is not None and source_sr != target_sr:
        audio = librosa.resample(audio, orig_sr=source_sr, target_sr=target_sr)

    return audio.astype(np.float32)


def mel_spectrogram(
    y, n_fft, num_mels, sampling_rate, hop_size, win_size, fmin, fmax, center=False
):
    pad_size = int((n_fft - hop_size) / 2)
    y = np.pad(y, ((0, 0), (pad_size, pad_size)), mode="reflect")

    stft = librosa.stft(
        y=y,
        n_fft=n_fft,
        hop_length=hop_size,
        win_length=win_size,
        window="hann",
        center=center,
        pad_mode="reflect",
    )
    spec = np.abs(stft) + 1e-9

    mel = librosa_mel_fn(
        sr=sampling_rate, n_fft=n_fft, n_mels=num_mels, fmin=fmin, fmax=fmax
    )
    mel_spec = np.matmul(mel, spec)

    log_spec = np.log(np.clip(mel_spec, 1e-5, None))

    return log_spec


def log_mel_spectrogram(
    audio: np.ndarray,
    n_mels: int = 80,
    n_fft: int = 400,
    padding: int = 0,
    sr: int = 16000,
    hop_length: int = 160,
):
    # Apply padding if needed
    if padding > 0:
        audio = np.pad(audio, (0, padding))

    # Compute STFT
    stft = librosa.stft(
        audio,
        n_fft=n_fft,
        hop_length=hop_length,
        window="hann",
        pad_mode="reflect",
    )
    # Compute magnitude spectrogram
    magnitudes = np.abs(stft) ** 2

    # Compute mel filter bank
    mel_filters = librosa.filters.mel(sr=sr, n_fft=n_fft, n_mels=n_mels)
    # Apply mel filter bank to magnitude spectrogram
    mel_spec = np.matmul(mel_filters, magnitudes)

    # Convert to log scale
    log_spec = np.log10(np.maximum(mel_spec, 1e-10))
    # Normalize log spectrogram
    log_spec = np.maximum(log_spec, log_spec.max() - 8.0)
    log_spec = (log_spec + 4.0) / 4.0

    return log_spec


def compute_fbank(signal, sr=16000):
    """
    librosa でなるべく上記の Kaldi 設定に近づける
    """
    # 25ms=400サンプル, 10ms=160サンプル, FFTサイズ400 (2のべき乗化しない)
    S = librosa.feature.melspectrogram(
        y=signal,
        sr=sr,
        n_fft=400,
        hop_length=160,
        win_length=400,
        window="hamming",  # hamming
        center=False,  # snip_edgesに近い
        power=2.0,  # パワースペクトル
        fmin=0.0,
        fmax=sr / 2,
        n_mels=80,
        htk=True,  # HTK互換
        norm=None,  # Slaney正規化をオフに
    )
    log_S = np.log(np.maximum(S, 1e-10))
    feat = log_S.T
    feat = feat - feat.mean(axis=0, keepdims=True)
    return feat
