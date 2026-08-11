# adapted from Keith Ito's tacotron implementation (MIT License)
# https://github.com/keithito/tacotron/blob/master/util/audio.py

import librosa
import numpy as np

SAMPLE_RATE = 16000
N_FFT = 1200
HOP_LENGTH = 160
WIN_LENGTH = 400
MIN_LEVEL_DB = -100.0
REF_LEVEL_DB = 20.0
NUM_MELS = 40


class Audio:
    def get_mel(self, y, n_fft):
        mel_basis = librosa.filters.mel(
            sr=SAMPLE_RATE,
            n_fft=n_fft,
            n_mels=NUM_MELS,
        )
        y = librosa.core.stft(
            y=y,
            n_fft=n_fft,
            hop_length=HOP_LENGTH,
            win_length=WIN_LENGTH,
            window="hann",
        )
        magnitudes = np.abs(y) ** 2
        mel = np.log10(np.dot(mel_basis, magnitudes) + 1e-6)

        return mel

    def wav2spec(self, y):
        D = self.stft(y)
        S = self.amp_to_db(np.abs(D)) - REF_LEVEL_DB
        S, D = self.normalize(S), np.angle(D)
        S, D = S.T, D.T  # to make [time, freq]

        return S, D

    def spec2wav(self, spectrogram, phase):
        spectrogram, phase = spectrogram.T, phase.T
        # used during inference only
        # spectrogram: enhanced output
        # phase: use noisy input's phase, so no GLA is required
        S = self.db_to_amp(self.denormalize(spectrogram) + REF_LEVEL_DB)

        return self.istft(S, phase)

    def stft(self, y):
        return librosa.stft(
            y=y,
            n_fft=N_FFT,
            hop_length=HOP_LENGTH,
            win_length=WIN_LENGTH,
        )

    def istft(self, mag, phase):
        stft_matrix = mag * np.exp(1j * phase)
        return librosa.istft(stft_matrix, hop_length=HOP_LENGTH, win_length=WIN_LENGTH)

    def amp_to_db(self, x):
        return 20.0 * np.log10(np.maximum(1e-5, x))

    def db_to_amp(self, x):
        return np.power(10.0, x * 0.05)

    def normalize(self, S):
        return np.clip(S / -MIN_LEVEL_DB, -1.0, 0.0) + 1.0

    def denormalize(self, S):
        return (np.clip(S, 0.0, 1.0) - 1.0) * -MIN_LEVEL_DB
