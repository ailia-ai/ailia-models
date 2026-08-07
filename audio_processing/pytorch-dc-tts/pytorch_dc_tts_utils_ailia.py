import os
import re
import copy
import scipy.io.wavfile
import unicodedata
import numpy as np
import ailia.audio as ailia_audio

"""Hyper parameters"""
class HParams:
    n_fft = 2048 # fft points (samples)
    power = 1.5  # Exponent for amplifying the predicted magnitude
    n_iter = 50  # Number of inversion iterations
    preemphasis = .97
    max_db = 100
    ref_db = 20
    sr = 22050  # Sampling rate
    frame_shift = 0.0125  # seconds
    frame_length = 0.05  # seconds
    hop_length = int(sr * frame_shift)  # samples. =276.
    win_length = int(sr * frame_length)  # samples. =1102.
hp = HParams()


"""Save to wav"""
def save_to_wav(mag, filename):
    """Generate and save an audio file from the given linear spectrogram using Griffin-Lim."""
    wav = spectrogram2wav(mag)
    scipy.io.wavfile.write(filename, hp.sr, wav)


def spectrogram2wav(mag):
    '''# Generate wave file from linear magnitude spectrogram
    Args:
      mag: A numpy array of (T, 1+n_fft//2)
    Returns:
      wav: A 1-D numpy array.
    '''
    # transpose
    mag = mag.T

    # de-noramlize
    mag = (np.clip(mag, 0, 1) * hp.max_db) - hp.max_db + hp.ref_db

    # to amplitude
    mag = np.power(10.0, mag * 0.05)

    # wav reconstruction
    wav = griffin_lim(mag ** hp.power)

    # de-preemphasis
    wav = ailia_audio.linerfilter(np.array([1]), np.array([1, -hp.preemphasis]), wav)

    # trim
    wav, _ = ailia_audio.trim(wav)

    return wav.astype(np.float32)


def griffin_lim(spectrogram):
    '''Applies Griffin-Lim's raw.'''
    X_best = copy.deepcopy(spectrogram)
    for i in range(hp.n_iter):
        X_t = invert_spectrogram(X_best)
        est = ailia_audio.spectrogram(X_t, fft_n=hp.n_fft, hop_n=hp.hop_length, win_n = hp.win_length) 
        phase = est / np.maximum(1e-8, np.abs(est))
        X_best = spectrogram * phase
    X_t = invert_spectrogram(X_best)
    y = np.real(X_t)

    return y


def invert_spectrogram(spectrogram):
    '''Applies inverse fft.
    Args:
      spectrogram: [1+n_fft//2, t]
    '''
    return ailia_audio.inverse_spectrogram(spectrogram, hop_n=hp.hop_length, win_n=hp.win_length, win_type="hann")



"""Get test data"""
def get_test_data(sentence, max_n):
    normalized_sentences = [text_normalize(line).strip() + "E" for line in sentence]  # text normalization, E: EOS
    texts = np.zeros((len(normalized_sentences), max_n + 1), np.int64)
    vocab = "PE abcdefghijklmnopqrstuvwxyz'.?"  # P: Padding, E: EOS.
    char2idx = {char: idx for idx, char in enumerate(vocab)}
    for i, sent in enumerate(normalized_sentences):
        texts[i, :len(sent)] = [char2idx[char] for char in sent]
    return texts


def text_normalize(text):
    vocab = "PE abcdefghijklmnopqrstuvwxyz'.?"  # P: Padding, E: EOS.
    text = ''.join(char for char in unicodedata.normalize('NFD', text)
                   if unicodedata.category(char) != 'Mn')  # Strip accents

    text = text.lower()
    text = re.sub("[^{}]".format(vocab), " ", text)
    text = re.sub("[ ]+", " ", text)
    return text

