"""
Pure Python fallback for librosa / soundfile on WoA (wav files only).

On Windows on ARM (WoA) librosa and soundfile cannot be installed because
they depend on binary wheels (numba, soxr, libsndfile) that are not provided
for that platform. This module reimplements the small subset of their APIs
used by the samples with numpy and the standard library only, limited to
wav file input / output.

Call `enable_if_needed()` once before `import librosa` / `import soundfile`
and, only when the real package is missing, a compatible module is
registered in sys.modules so the samples keep working without any change:

    import woa_librosa
    woa_librosa.enable_if_needed()

    import librosa                      # -> pure Python fallback on WoA
    y, sr = librosa.load('input.wav', mono=True)

Supported APIs:
    librosa.load / librosa.resample / librosa.to_mono /
    librosa.get_samplerate / librosa.get_duration
    soundfile.read / soundfile.write
"""

import sys
import types
import struct
import importlib.util

import numpy as np

from logging import getLogger
logger = getLogger(__name__)


# =============================================================================
# wav reading (RIFF parser; the stdlib wave module cannot read float wav)
# =============================================================================
_WAVE_FORMAT_PCM = 1
_WAVE_FORMAT_IEEE_FLOAT = 3
_WAVE_FORMAT_EXTENSIBLE = 0xFFFE


def _read_wav(path):
    """Read a wav file and return (float32 array of shape (frames, channels), rate)."""
    with open(path, 'rb') as f:
        data = f.read()

    if len(data) < 12 or data[0:4] != b'RIFF' or data[8:12] != b'WAVE':
        raise ValueError(
            'woa_librosa: %r is not a wav file '
            '(this pure Python fallback only supports wav input)' % path)

    fmt_body = None
    raw = None
    pos = 12
    while pos + 8 <= len(data):
        chunk_id = data[pos:pos + 4]
        chunk_size = struct.unpack('<I', data[pos + 4:pos + 8])[0]
        body = data[pos + 8:pos + 8 + chunk_size]
        if chunk_id == b'fmt ':
            fmt_body = body
        elif chunk_id == b'data':
            raw = body
        # chunks are word aligned
        pos += 8 + chunk_size + (chunk_size & 1)

    if fmt_body is None or len(fmt_body) < 16 or raw is None:
        raise ValueError('woa_librosa: %r has no fmt/data chunk' % path)

    audio_format, channels, rate, _, _, bits = \
        struct.unpack('<HHIIHH', fmt_body[:16])
    if audio_format == _WAVE_FORMAT_EXTENSIBLE and len(fmt_body) >= 26:
        # the real format is the first 2 bytes of the SubFormat GUID
        audio_format = struct.unpack('<H', fmt_body[24:26])[0]

    if audio_format == _WAVE_FORMAT_PCM:
        if bits == 8:
            y = np.frombuffer(raw, dtype=np.uint8).astype(np.float32)
            y = (y - 128.0) / 128.0
        elif bits == 16:
            n = len(raw) // 2
            y = np.frombuffer(raw[:n * 2], dtype='<i2').astype(np.float32)
            y /= 32768.0
        elif bits == 24:
            a = np.frombuffer(raw, dtype=np.uint8)
            n = len(a) // 3
            a = a[:n * 3].reshape(-1, 3).astype(np.int32)
            v = a[:, 0] | (a[:, 1] << 8) | (a[:, 2] << 16)
            v -= (v & 0x800000) << 1  # sign extend
            y = v.astype(np.float32) / 8388608.0
        elif bits == 32:
            n = len(raw) // 4
            y = np.frombuffer(raw[:n * 4], dtype='<i4').astype(np.float32)
            y /= 2147483648.0
        else:
            raise ValueError('woa_librosa: unsupported PCM bit depth %d' % bits)
    elif audio_format == _WAVE_FORMAT_IEEE_FLOAT:
        if bits == 32:
            n = len(raw) // 4
            y = np.frombuffer(raw[:n * 4], dtype='<f4').astype(np.float32)
        elif bits == 64:
            n = len(raw) // 8
            y = np.frombuffer(raw[:n * 8], dtype='<f8').astype(np.float32)
        else:
            raise ValueError('woa_librosa: unsupported float bit depth %d' % bits)
    else:
        raise ValueError(
            'woa_librosa: unsupported wav format 0x%X' % audio_format)

    channels = max(int(channels), 1)
    frames = len(y) // channels
    return y[:frames * channels].reshape(-1, channels), int(rate)


# =============================================================================
# librosa compatible API
# =============================================================================
def to_mono(y):
    y = np.asarray(y)
    if y.ndim > 1:
        y = np.mean(y, axis=tuple(range(y.ndim - 1)))
    return y


def resample(y, orig_sr=None, target_sr=None, res_type=None, fix=True,
             scale=False, axis=-1, **kwargs):
    """FFT based resampling (equivalent to scipy.signal.resample)."""
    if orig_sr is None or target_sr is None:
        raise TypeError('woa_librosa.resample: orig_sr and target_sr are required')

    y = np.asarray(y)
    if int(orig_sr) == int(target_sr):
        return y

    n_in = y.shape[axis]
    n_out = int(np.ceil(n_in * float(target_sr) / float(orig_sr)))

    y_moved = np.moveaxis(y, axis, -1)
    spec = np.fft.rfft(y_moved, axis=-1)
    bins_out = n_out // 2 + 1
    if bins_out <= spec.shape[-1]:
        spec = spec[..., :bins_out]
    else:
        pad = np.zeros(spec.shape[:-1] + (bins_out - spec.shape[-1],),
                       dtype=spec.dtype)
        spec = np.concatenate([spec, pad], axis=-1)
    out = np.fft.irfft(spec, n=n_out, axis=-1) * (float(n_out) / float(n_in))

    if scale:
        out /= np.sqrt(float(n_out) / float(n_in))
    out = np.moveaxis(out, -1, axis)
    return np.ascontiguousarray(out, dtype=y.dtype if y.dtype.kind == 'f' else np.float32)


def load(path, sr=22050, mono=True, offset=0.0, duration=None,
         dtype=np.float32, res_type=None):
    """librosa.load compatible wav loader (wav files only)."""
    data, native_sr = _read_wav(path)  # (frames, channels)

    # librosa truncates (not rounds) offset / duration to whole frames
    start = int(offset * native_sr) if offset else 0
    if duration is not None:
        data = data[start:start + int(duration * native_sr)]
    elif start:
        data = data[start:]

    # librosa returns (n,) for mono and (channels, n) for multichannel
    y = data.T
    if y.shape[0] == 1:
        y = y[0]
    elif mono:
        y = to_mono(y)

    if sr is not None and int(sr) != int(native_sr):
        y = resample(y, orig_sr=native_sr, target_sr=sr)
    else:
        sr = native_sr

    return np.ascontiguousarray(y, dtype=dtype), int(sr)


def get_samplerate(path):
    return _read_wav(path)[1]


def get_duration(y=None, sr=22050, path=None, filename=None, **kwargs):
    if y is not None:
        y = np.asarray(y)
        return y.shape[-1] / float(sr)
    target = path if path is not None else filename
    if target is None:
        raise ValueError('woa_librosa.get_duration: y or path is required')
    data, native_sr = _read_wav(target)
    return data.shape[0] / float(native_sr)


# =============================================================================
# soundfile compatible API
# =============================================================================
def read(file, frames=-1, start=0, stop=None, dtype='float64',
         always_2d=False, **kwargs):
    """soundfile.read compatible wav reader (wav files only)."""
    data, rate = _read_wav(file)  # (frames, channels)

    if stop is not None:
        data = data[start:stop]
    elif frames is not None and frames >= 0:
        data = data[start:start + frames]
    elif start:
        data = data[start:]

    if not always_2d and data.shape[1] == 1:
        data = data[:, 0]
    return data.astype(dtype), rate


def write(file, data, samplerate, subtype=None, endian=None, format=None,
          closefd=True):
    """soundfile.write compatible wav writer (PCM_16 / FLOAT wav only)."""
    data = np.asarray(data)
    if data.ndim == 1:
        data = data[:, np.newaxis]
    frames, channels = data.shape

    if subtype is None:
        subtype = 'FLOAT' if data.dtype.kind == 'f' else 'PCM_16'
    subtype = subtype.upper()

    if subtype == 'PCM_16':
        if data.dtype.kind == 'f':
            pcm = (np.clip(data, -1.0, 1.0) * 32767.0).astype('<i2')
        else:
            pcm = data.astype('<i2')
        payload = pcm.tobytes()
        audio_format, bits = _WAVE_FORMAT_PCM, 16
    elif subtype == 'FLOAT':
        payload = data.astype('<f4').tobytes()
        audio_format, bits = _WAVE_FORMAT_IEEE_FLOAT, 32
    else:
        raise ValueError(
            'woa_librosa: unsupported subtype %r '
            '(this pure Python fallback supports PCM_16 and FLOAT)' % subtype)

    block_align = channels * bits // 8
    header = struct.pack(
        '<4sI4s4sIHHIIHH4sI',
        b'RIFF', 36 + len(payload), b'WAVE',
        b'fmt ', 16, audio_format, channels, int(samplerate),
        int(samplerate) * block_align, block_align, bits,
        b'data', len(payload))
    with open(file, 'wb') as f:
        f.write(header)
        f.write(payload)


# =============================================================================
# activation
# =============================================================================
_LIBROSA_API = {
    'load': load,
    'resample': resample,
    'to_mono': to_mono,
    'get_samplerate': get_samplerate,
    'get_duration': get_duration,
}

_SOUNDFILE_API = {
    'read': read,
    'write': write,
}


def _install(name, api):
    if name in sys.modules:
        return
    mod = types.ModuleType(name)
    mod.__dict__.update(api)
    mod.__version__ = '0.0.0+woa_librosa_fallback'
    sys.modules[name] = mod
    logger.info(
        '%s is not installed: using the pure Python wav-only fallback '
        'of woa_librosa.' % name)


def _is_missing(name):
    try:
        return importlib.util.find_spec(name) is None
    except Exception:
        return True


def enable():
    """Register the fallback modules as librosa / soundfile."""
    _install('librosa', _LIBROSA_API)
    _install('soundfile', _SOUNDFILE_API)


def enable_if_needed():
    """Register the fallback only for packages that are not installed."""
    if _is_missing('librosa'):
        _install('librosa', _LIBROSA_API)
    if _is_missing('soundfile'):
        _install('soundfile', _SOUNDFILE_API)
