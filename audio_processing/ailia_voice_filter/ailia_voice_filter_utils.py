"""Streaming helpers for the ailia Voice Filter sample.

Everything that is not inside the graph lives here: the analysis window, the
STFT, the state bookkeeping and the speaker encoder's slicing.  The browser
demo keeps the same numbers in a JSON contract next to each graph; they are
inlined below so the sample needs nothing but the three ONNX files.

``GraphStream.push`` takes exactly one hop of samples and returns exactly one
hop, so the caller never has to know whether the transform is inside the graph
(DeepFilterNet3) or its own job (the conditioned model).
"""

import numpy as np

# ======================
# Model contracts
# ======================

# The graphs are exported for streaming, so every recurrent tensor is an
# explicit input and output.  `state_shapes` is what a fresh call starts from.
_STATE_SHAPES = {
    'erb_cache': (1, 1, 2, 32),
    'df_cache': (1, 2, 2, 96),
    'convp_cache': (1, 64, 4, 96),
    'erb_mean': (1, 32),
    'unit_mean': (1, 96),
    'enc_h': (1, 1, 256),
    'erb_h': (2, 1, 256),
    'df_h': (2, 1, 256),
    'spec_queue': (1, 4, 481, 2),
}

# The two running normalisers do not start at zero; these are DeepFilterNet's
# own initial values, the same ones the offline model is primed with.
_INITIAL = {
    'erb_mean': np.linspace(-60.0, -90.0, 32, dtype=np.float32),
    'unit_mean': np.linspace(1e-3, 1e-4, 96, dtype=np.float32),
}

# Six of the states are transient.  For the first `warmup_frames` frames after
# a reset the graph's own values for them must be thrown away and the previous
# ones sent again -- the offline model never sees those frames.
_NETWORK_STATES = (
    'erb_cache', 'df_cache', 'convp_cache', 'enc_h', 'erb_h', 'df_h',
)

MODELS = {
    'voicefilter': {
        'graph': 'voicefilter_stream.onnx',
        'name': 'ailia Voice Filter (48 kHz, speaker conditioned)',
        'rate': 48000,
        'conditioned': True,
        'n_fft': 960,
        'hop': 480,
        'win': 960,
        'window': 'vorbis',
        # The graph works on a scaled spectrum; undo it on the way out.
        'spectrum_scale': 1.0 / 960.0,
        'dvector': 256,
        'warmup_frames': 2,
        'startup_pad': 480,
        # Measured by cross-correlating the output against the input, on three
        # signals: a single peak at 960 every time.  The browser demo's
        # contract says 1920, but it only prints that figure -- it streams and
        # never realigns -- so the error never showed there.
        'latency_samples': 960,
        'state_shapes': dict(_STATE_SHAPES),
        'initial': dict(_INITIAL),
    },
    'deepfilternet': {
        'graph': 'deepfilternet_stream.onnx',
        'name': 'DeepFilterNet3 (48 kHz, denoising only)',
        'rate': 48000,
        'conditioned': False,
        'n_fft': 960,
        'hop': 480,
        'warmup_frames': 2,
        # A hop of samples in and a hop out: the transform is inside the graph,
        # so it carries the two extra buffers the overlap-add needs.
        'delay_hops': 3,
        'state_shapes': dict(
            _STATE_SHAPES,
            input_tail=(1, 480),
            overlap=(1, 480),
        ),
        'initial': dict(_INITIAL),
    },
}


def vorbis_window(n_fft):
    """``sin(pi/2 * sin^2(pi (n + 0.5) / N))`` -- what the 48 kHz line uses.

    At half overlap its squares already sum to unity, so the window-square
    normalisation in the overlap-add is a no-op for this model.
    """
    index = np.arange(n_fft)
    return np.sin(np.pi / 2 * np.sin(np.pi * (index + 0.5) / n_fft) ** 2)


# ======================
# Streaming graph
# ======================

class GraphStream:
    """One streaming graph, driven a hop at a time."""

    def __init__(self, net, model):
        self.net = net
        self.spec = MODELS[model]
        self.inputs = [net.get_blob_name(i) for i in net.get_input_blob_list()]
        self.outputs = [net.get_blob_name(i) for i in net.get_output_blob_list()]

        self.rate = self.spec['rate']
        self.hop = self.spec['hop']
        self.n_fft = self.spec['n_fft']
        self.conditioned = self.spec['conditioned']
        self.dvector_size = self.spec.get('dvector', 256)
        self.warmup = self.spec['warmup_frames']
        self.scale = self.spec.get('spectrum_scale', 1.0)

        # A spectrum in and out, or a hop of samples in and out.
        self.spectral = 'real' in self.inputs
        self.window = self._window() if self.spectral else None
        # torch.stft(center=True) puts this many zeros in front of the signal.
        self.startup = self.spec.get('startup_pad', self.n_fft // 2)
        # Until n_fft samples have arrived no frame comes out at all.
        self._silent = max(0, -(-(self.n_fft - self.startup) // self.hop) - 1)

        self.dvector = None
        self.reset()

    # ------------------------------------------------------------- setup

    def _window(self):
        """The analysis window, centred in the frame the way torch centres it."""
        length = self.spec.get('win', self.n_fft)
        shape = self.spec['window']
        if shape == 'vorbis':
            short = vorbis_window(length).astype(np.float64)
        elif shape == 'hann':
            short = np.hanning(length + 1)[:-1].astype(np.float64)
        else:
            raise ValueError('unknown analysis window %r' % shape)
        window = np.zeros(self.n_fft, dtype=np.float64)
        pad = (self.n_fft - length) // 2
        window[pad:pad + length] = short
        return window

    @property
    def latency(self):
        """Samples the output lags the input by."""
        if 'latency_samples' in self.spec:
            return self.spec['latency_samples']
        return self.spec.get('delay_hops', 0) * self.hop

    def enrol(self, dvector):
        vector = np.asarray(dvector, dtype=np.float32).reshape(-1)
        if vector.shape[0] != self.dvector_size:
            raise ValueError(
                'the d-vector has %d dimensions, the graph wants %d'
                % (vector.shape[0], self.dvector_size))
        self.dvector = vector[None]

    def reset(self):
        """Back to the state a fresh call starts in."""
        self.state = {name: np.zeros(shape, dtype=np.float32)
                      for name, shape in self.spec['state_shapes'].items()}
        for name, value in self.spec['initial'].items():
            self.state[name] = np.asarray(
                value, dtype=np.float32).reshape(self.state[name].shape)
        self.index = 0
        # The analysis buffer starts on `startup_pad` zeros.  A frame is cut
        # every time n_fft samples are available, then it advances by one hop.
        self._buffer = np.zeros(self.startup, dtype=np.float32)
        # One accumulator the length of a frame, and the same for the squared
        # window: overlap-add only reconstructs when the squared windows sum to
        # one, so the emitted hop is divided by that sum.
        self._overlap = np.zeros(self.n_fft, dtype=np.float64)
        self._weight = np.zeros(self.n_fft, dtype=np.float64)

    # ----------------------------------------------------------- one hop

    def _run(self, feed):
        for name in self.inputs:
            if name.endswith('_in') and name[:-3] in self.state:
                feed[name] = self.state[name[:-3]]
        feed = {k: v for k, v in feed.items() if k in self.inputs}
        produced = dict(zip(self.outputs, self.net.run(feed)))
        for name, value in produced.items():
            if not name.endswith('_out'):
                continue
            key = name[:-4]
            if key not in self.state:
                continue
            if self.index < self.warmup and key in _NETWORK_STATES:
                continue
            self.state[key] = value
        self.index += 1
        return produced

    def push(self, hop):
        """One hop of samples in, one hop out.  Call it in order."""
        hop = np.asarray(hop, dtype=np.float32)
        if len(hop) != self.hop:
            raise ValueError('expected %d samples, got %d' % (self.hop, len(hop)))
        if self.conditioned and self.dvector is None:
            raise RuntimeError('%s: enrol() before pushing' % self.spec['name'])

        if not self.spectral:
            produced = self._run({'hop': hop[None]})
            return np.asarray(produced['out'], dtype=np.float32).reshape(-1)

        self._buffer = np.concatenate([self._buffer, hop])
        pieces = []
        while len(self._buffer) >= self.n_fft:
            frame = self._buffer[:self.n_fft].astype(np.float64) * self.window
            self._buffer = self._buffer[self.hop:]
            spectrum = np.fft.rfft(frame) * self.scale
            feed = {
                'real': spectrum.real.astype(np.float32)[None],
                'imaginary': spectrum.imag.astype(np.float32)[None],
            }
            if self.dvector is not None and 'dvector' in self.inputs:
                feed['dvector'] = self.dvector
            produced = self._run(feed)
            real = np.asarray(produced['out_real'], dtype=np.float64).reshape(-1)
            imaginary = np.asarray(
                produced['out_imaginary'], dtype=np.float64).reshape(-1)
            rebuilt = np.fft.irfft((real + 1j * imaginary) / self.scale,
                                   n=self.n_fft) * self.window

            self._overlap += rebuilt
            self._weight += self.window ** 2
            pieces.append(self._overlap[:self.hop]
                          / np.maximum(self._weight[:self.hop], 1e-8))
            self._overlap = np.concatenate(
                [self._overlap[self.hop:], np.zeros(self.hop)])
            self._weight = np.concatenate(
                [self._weight[self.hop:], np.zeros(self.hop)])

        # One push is still one hop.  The opening frames that cannot be formed
        # yet return silence, and `skew` counts them.
        out = np.concatenate(pieces) if pieces else np.zeros(0)
        if len(out) < self.hop:
            out = np.concatenate([out, np.zeros(self.hop - len(out))])
        return out[:self.hop].astype(np.float32)

    # ------------------------------------------------------- whole signal

    @property
    def lead(self):
        """Samples of our own padding that come out before the first real one.

        The first frame covers ``[-startup_pad, -startup_pad + n_fft)``, so the
        first hop out reconstructs the padding rather than the signal, and
        while fewer than n_fft samples have arrived no frame comes out at all.
        Zero when the graph does its own transform.
        """
        if not self.spectral:
            return 0
        return self._silent * self.hop + self.startup

    @property
    def skew(self):
        return self.lead + self.latency

    def run(self, audio):
        """The same path as ``push``, over a whole signal, skew removed."""
        self.reset()
        audio = np.asarray(audio, dtype=np.float32)
        wanted = len(audio) + self.skew
        pad = (-wanted) % self.hop
        padded = np.concatenate([audio, np.zeros(self.skew + pad, np.float32)])
        pieces = [self.push(padded[i * self.hop:(i + 1) * self.hop])
                  for i in range(len(padded) // self.hop)]
        out = np.concatenate(pieces) if pieces else np.zeros(0, np.float32)
        return out[self.skew:][:len(audio)]


# ======================
# Speaker encoder
# ======================

SPEAKER_RATE = 16000
SPEAKER_PARTIAL = 25600      # 1.6 s
SPEAKER_EMBEDDING = 256
SPEAKER_TARGET_DBFS = -30.0


def compute_dvector(net, audio):
    """A 256-d GE2E d-vector for 16 kHz mono audio.

    The STFT and the mel filterbank are inside the graph, so nothing here
    reproduces librosa.  What is left of ``embed_utterance`` is slicing:
    partials of 1.6 s at 50% overlap, averaged, then L2 normalised.
    """
    wav = np.asarray(audio, dtype=np.float32).reshape(-1).copy()

    # -30 dBFS, increase only, as the encoder's own normalise_volume does.
    power = max(float(np.mean(wav ** 2)) if wav.size else 0.0, 1e-12)
    change = SPEAKER_TARGET_DBFS - 10 * np.log10(power)
    if change > 0:
        wav *= 10 ** (change / 20)

    if len(wav) < SPEAKER_PARTIAL:
        wav = np.concatenate([wav, np.zeros(SPEAKER_PARTIAL - len(wav), np.float32)])

    stride = SPEAKER_PARTIAL // 2
    partials = [wav[start:start + SPEAKER_PARTIAL]
                for start in range(0, len(wav) - SPEAKER_PARTIAL + 1, stride)]
    embeddings = [np.asarray(net.run(partial[None])[0]).reshape(-1)
                  for partial in partials]

    mean = np.mean(embeddings, axis=0)
    return (mean / max(float(np.linalg.norm(mean)), 1e-12)).astype(np.float32)
