import math
import queue
import signal
import sys
import time
from logging import getLogger

import ailia
import librosa
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import soundfile as sf  # noqa: E402

# import original modules
sys.path.append("../../util")
from arg_utils import get_base_parser, get_savepath, update_parser  # noqa: E402
from math_utils import sigmoid  # noqa: E402
from microphone_utils import start_microphone_input  # noqa: E402
from model_utils import check_and_download_models  # noqa: E402

logger = getLogger(__name__)

# ======================
# Parameters
# ======================

WEIGHT_PATH = "nemotron-3-diarization.onnx"
MODEL_PATH = "nemotron-3-diarization.onnx.prototxt"
SILENCE_EMBEDS_PATH = "silence_embeds.npy"
REMOTE_PATH = "https://storage.googleapis.com/ailia-models/nemotron-3-diarization/"

AUDIO_PATH = "diarization_example.mp3"
SAVE_IMAGE_PATH = "output.png"

SAMPLE_RATE = 16000
N_FFT = 512
HOP_LENGTH = 160
WIN_LENGTH = 400
NUM_MEL_BINS = 128
PREEMPHASIS = 0.97
LOG_ZERO_GUARD_VALUE = 2**-24

SUBSAMPLING_FACTOR = 8
HIDDEN_SIZE = 512
NUM_SPEAKERS = 8

# offline: (chunk_length, chunk_right_context, fifo_length, speaker_cache_update_period)
OFFLINE_CONFIG = (340, 40, 40, 300)
# streaming: (chunk_length, chunk_right_context) in encoder frames (80 ms)
STREAMING_MODES = {
    "low_latency": (9, 4),
    "very_low_latency": (6, 2),
    "ultra_low_latency": (3, 1),
}
STREAMING_FIFO_LENGTH = 264
STREAMING_UPDATE_PERIOD = 222

SPEAKER_CACHE_LENGTH = 264
SILENCE_FRAMES_PER_SPEAKER = 1
PREDICTION_SCORE_THRESHOLD = 0.25
LATEST_FRAMES_SCORE_BOOST = 0.05
MIN_POSITIVE_SCORES_RATE = 0.5
STRONG_BOOST_RATE = 0.75
WEAK_BOOST_RATE = 1.5

THRESHOLD = 0.5

# ======================
# Arguemnt Parser Config
# ======================

parser = get_base_parser(
    "Nemotron 3 Diarization", AUDIO_PATH, SAVE_IMAGE_PATH, input_ftype="audio"
)
parser.add_argument(
    "--latency",
    default="offline",
    choices=("offline",) + tuple(STREAMING_MODES),
    help="offline: whole audio at once (30.4 s buffer) | "
    "low_latency (1.04 s) | very_low_latency (0.64 s) | ultra_low_latency (0.32 s): "
    "streaming, the audio is fed chunk by chunk",
)
parser.add_argument(
    "-V",
    action="store_true",
    help="use microphone input (streaming, low_latency unless another streaming --latency is given)",
)
parser.add_argument(
    "-th",
    "--threshold",
    default=THRESHOLD,
    type=float,
    help="speaker probability above which a frame counts as speech.",
)
parser.add_argument("--onnx", action="store_true", help="execute onnxruntime version.")
args = update_parser(parser)


# ======================
# Secondary Functions
# ======================


def read_wave(path):
    audio, sr = sf.read(path, dtype="float32", always_2d=True)
    audio = audio.mean(axis=1)  # multi channel -> mono
    if sr != SAMPLE_RATE:
        logger.info(f"resampling {sr}Hz -> {SAMPLE_RATE}Hz")
        audio = librosa.resample(audio, orig_sr=sr, target_sr=SAMPLE_RATE)

    return np.ascontiguousarray(audio)


MEL_FILTERS = librosa.filters.mel(
    sr=SAMPLE_RATE,
    n_fft=N_FFT,
    n_mels=NUM_MEL_BINS,
    fmin=0.0,
    fmax=SAMPLE_RATE / 2,
    norm="slaney",
).astype(np.float32)

# symmetric hann window, zero-padded to n_fft on both sides
WINDOW = np.pad(
    np.hanning(WIN_LENGTH),
    ((N_FFT - WIN_LENGTH) // 2, (N_FFT - WIN_LENGTH + 1) // 2),
).astype(np.float32)


def extract_features(audio, center=True, is_last_audio_chunk=False):
    """audio: [T] -> log-mel [frames, 128], attention_mask [frames]

    center=True pads n_fft // 2 on both sides (offline and the first streaming chunk).
    The later streaming chunks start n_fft // 2 before their first frame and are not padded,
    which gives the same frames as a centered pass over the whole audio.
    """
    x = audio.astype(np.float32)
    x = np.concatenate([x[:1], x[1:] - PREEMPHASIS * x[:-1]])

    audio_length = len(x)
    if is_last_audio_chunk and not center:
        # the last frame of a centered pass reaches n_fft // 2 - hop past the audio
        num_padding = N_FFT // 2 - HOP_LENGTH
        x = np.pad(x, (0, num_padding))
        audio_length += num_padding

    if center:
        x = np.pad(x, (N_FFT // 2, N_FFT // 2))
    num_frames = 1 + (len(x) - N_FFT) // HOP_LENGTH
    frames = np.lib.stride_tricks.as_strided(
        x,
        shape=(num_frames, N_FFT),
        strides=(x.strides[0] * HOP_LENGTH, x.strides[0]),
    )
    spec = np.fft.rfft(frames * WINDOW, axis=-1)
    power = (spec.real**2 + spec.imag**2).astype(np.float32)

    mel_spec = np.log(power @ MEL_FILTERS.T + LOG_ZERO_GUARD_VALUE)

    if center:
        features_length = audio_length // HOP_LENGTH
    else:
        features_length = (audio_length - N_FFT) // HOP_LENGTH + 1
    attention_mask = np.arange(num_frames) < features_length
    mel_spec = mel_spec * attention_mask[:, None]

    return mel_spec.astype(np.float32), attention_mask.astype(np.int64)


def topk_indices(x, k, axis):
    # only -inf / +inf can tie, which does not change the selected set
    return np.argpartition(-x, k - 1, axis=axis).take(np.arange(k), axis=axis)


class SpeakerCache:
    """Arrival-Order Speaker Cache and FIFO queue of the encoder input frames"""

    def __init__(self, fifo_length, update_period, silence_embeds):
        self.fifo_length = fifo_length
        self.update_period = update_period
        self.silence_embeds = silence_embeds

        budget = SPEAKER_CACHE_LENGTH // NUM_SPEAKERS - SILENCE_FRAMES_PER_SPEAKER
        self.min_positive_scores = math.floor(budget * MIN_POSITIVE_SCORES_RATE)
        self.num_strong_boosted_frames = math.floor(budget * STRONG_BOOST_RATE)
        self.num_weak_boosted_frames = math.floor(budget * WEAK_BOOST_RATE)

        self.embeds = np.zeros((SPEAKER_CACHE_LENGTH, HIDDEN_SIZE), dtype=np.float32)
        self.probs = np.zeros((SPEAKER_CACHE_LENGTH, NUM_SPEAKERS), dtype=np.float32)
        self.fifo = np.zeros((fifo_length, HIDDEN_SIZE), dtype=np.float32)
        self.num_cache_frames = 0
        self.num_fifo_frames = 0
        self.is_compressed = False

    def get_embeds(self):
        return np.concatenate(
            [self.embeds[: self.num_cache_frames], self.fifo[: self.num_fifo_frames]]
        )

    def update(self, chunk_input_embeds, chunk_logits, num_chunk_frames, mask):
        """Pushes the chunk frames to the FIFO, moving the oldest ones to the speaker cache
        when it overflows (the look-ahead frames are fed again at the next step)."""
        num_cache_frames, num_fifo_frames = self.num_cache_frames, self.num_fifo_frames

        # speaker probabilities at the encoder frame rate, zeroed on the padding frames
        probs = sigmoid(chunk_logits)
        num_frames = probs.shape[0] // SUBSAMPLING_FACTOR
        probs = probs.reshape(num_frames, SUBSAMPLING_FACTOR, -1).mean(axis=1)
        probs = probs * mask[:, None]

        chunk_start = num_cache_frames + num_fifo_frames
        chunk_embeds = chunk_input_embeds[chunk_start : chunk_start + num_chunk_frames]
        fifo_embeds = np.concatenate([self.fifo[:num_fifo_frames], chunk_embeds])

        num_popped = 0
        if len(fifo_embeds) > self.fifo_length:
            num_popped = max(self.update_period, len(fifo_embeds) - self.fifo_length)
            num_popped = min(num_popped, len(fifo_embeds))
        if num_popped:
            fifo_probs = probs[num_cache_frames : num_cache_frames + len(fifo_embeds)]
            # a compressed cache is out of order, so only its stored probs are valid
            if self.is_compressed:
                stored_probs = self.probs[:num_cache_frames]
            else:
                stored_probs = probs[:num_cache_frames]
            cache_embeds = np.concatenate(
                [self.embeds[:num_cache_frames], fifo_embeds[:num_popped]]
            )
            cache_probs = np.concatenate([stored_probs, fifo_probs[:num_popped]])
            fifo_embeds = fifo_embeds[num_popped:]

            if len(cache_embeds) > SPEAKER_CACHE_LENGTH:
                cache_embeds, cache_probs = self.compress(cache_embeds, cache_probs)
                self.is_compressed = True
            self.num_cache_frames = len(cache_embeds)
            self.embeds[: self.num_cache_frames] = cache_embeds
            self.probs[: self.num_cache_frames] = cache_probs

        self.num_fifo_frames = len(fifo_embeds)
        self.fifo[: self.num_fifo_frames] = fifo_embeds

    def get_frame_scores(self, probs):
        log_probs = np.log(np.maximum(probs, PREDICTION_SCORE_THRESHOLD))
        log_complements = np.log(np.maximum(1.0 - probs, PREDICTION_SCORE_THRESHOLD))
        scores = (
            log_probs
            - log_complements
            + log_complements.sum(axis=-1, keepdims=True)
            - math.log(0.5)
        )

        is_speech = probs > 0.5
        scores = np.where(is_speech, scores, -np.inf)
        is_positive = scores > 0
        has_enough_positive = (
            is_positive.sum(axis=0, keepdims=True) >= self.min_positive_scores
        )
        scores = np.where(
            ~is_positive & is_speech & has_enough_positive, -np.inf, scores
        )
        return scores

    def boost_scores(self, scores, num_boosted, boost):
        indices = topk_indices(scores, num_boosted, axis=0)
        boosted = np.take_along_axis(scores, indices, axis=0) + boost
        np.put_along_axis(scores, indices, boosted, axis=0)
        return scores

    def compress(self, embeds, probs):
        """Keeps the most important frames of each speaker, in their original order,
        with reserved slots filled with the silence embedding."""
        num_frames, num_speakers = probs.shape

        scores = self.get_frame_scores(probs)
        # frames beyond the cache capacity are the ones popped from the FIFO
        scores[SPEAKER_CACHE_LENGTH:] += LATEST_FRAMES_SCORE_BOOST

        scores = self.boost_scores(
            scores, self.num_strong_boosted_frames, boost=-2.0 * math.log(0.5)
        )
        scores = self.boost_scores(
            scores, self.num_weak_boosted_frames, boost=-math.log(0.5)
        )
        scores = np.pad(
            scores, ((0, SILENCE_FRAMES_PER_SPEAKER), (0, 0)), constant_values=np.inf
        )
        embeds = np.concatenate([embeds, self.silence_embeds[None]])
        probs = np.pad(probs, ((0, 1), (0, 0)))

        num_scored_frames = num_frames + SILENCE_FRAMES_PER_SPEAKER
        sentinel = num_scored_frames * num_speakers
        flat_scores = scores.T.reshape(-1)

        indices = topk_indices(flat_scores, SPEAKER_CACHE_LENGTH, axis=0)
        indices = np.where(flat_scores[indices] == -np.inf, sentinel, indices)
        indices = np.sort(indices)
        frame_indices = np.where(
            indices == sentinel,
            num_frames,
            np.minimum(indices % num_scored_frames, num_frames),
        )
        return embeds[frame_indices], probs[frame_indices]


def extract_speaker_segments(probs, attention_mask=None, threshold=THRESHOLD):
    """probs: [frames, 8] -> [(start [s], end [s], speaker), ...] sorted by start time"""
    frame_duration = HOP_LENGTH / SAMPLE_RATE
    active = probs > threshold
    if attention_mask is not None:
        active = active & attention_mask.astype(bool)[:, None]
    active = active.astype(np.int64)
    boundary = np.zeros((1, active.shape[1]), dtype=np.int64)
    changes = np.diff(np.concatenate([boundary, active, boundary]), axis=0)

    segments = []
    for speaker in range(active.shape[1]):
        starts = np.nonzero(changes[:, speaker] == 1)[0]
        ends = np.nonzero(changes[:, speaker] == -1)[0]
        segments.extend(
            (round(start * frame_duration, 2), round(end * frame_duration, 2), speaker)
            for start, end in zip(starts, ends)
        )
    segments.sort(key=lambda segment: (segment[0], segment[2]))
    return segments


def plot_result(audio, probs, segments, savepath):
    duration = len(audio) / SAMPLE_RATE
    colors = plt.get_cmap("tab10")

    fig, axes = plt.subplots(3, 1, figsize=(16, 8), sharex=True)
    t = np.arange(len(audio)) / SAMPLE_RATE
    axes[0].plot(t, audio, linewidth=0.3, color="gray")
    axes[0].set_ylabel("waveform")

    frame_t = (np.arange(len(probs)) + 0.5) * HOP_LENGTH / SAMPLE_RATE
    for k in range(NUM_SPEAKERS):
        if probs[:, k].max() > 0.1:
            axes[1].plot(frame_t, probs[:, k], color=colors(k), label=f"speaker_{k}")
    axes[1].set_ylabel("speaker prob")
    axes[1].set_ylim(0, 1.05)
    axes[1].legend(loc="upper right", fontsize=8, ncol=4)

    for start, end, spk in segments:
        axes[2].barh(spk, end - start, left=start, height=0.6, color=colors(spk))
    axes[2].set_yticks(range(NUM_SPEAKERS))
    axes[2].set_yticklabels([f"speaker_{k}" for k in range(NUM_SPEAKERS)])
    axes[2].invert_yaxis()
    axes[2].set_xlabel("time [s]")
    axes[2].set_xlim(0, duration)

    fig.tight_layout()
    fig.savefig(savepath, dpi=100)
    plt.close(fig)


# ======================
# Main functions
# ======================


def forward(
    models,
    input_features,
    attention_mask,
    speaker_cache,
    chunk_length,
    chunk_right_context,
):
    """input_features: [frames, 128] -> logits [frames, 8]

    The input is split into chunks of chunk_length encoder frames, each attending to
    chunk_right_context look-ahead frames and to the frames held in speaker_cache.
    """
    num_frames = len(input_features)
    num_embeds = -(-num_frames // SUBSAMPLING_FACTOR)
    num_chunk_embeds = (
        num_embeds - chunk_right_context if chunk_length is None else num_embeds
    )
    if chunk_length is None:
        chunk_length = num_chunk_embeds

    logits = []
    for start_idx in range(0, num_chunk_embeds, chunk_length):
        end_idx = min(start_idx + chunk_length, num_chunk_embeds)
        num_chunk_frames = end_idx - start_idx
        mel_slice = slice(
            start_idx * SUBSAMPLING_FACTOR,
            min(end_idx + chunk_right_context, num_embeds) * SUBSAMPLING_FACTOR,
        )
        chunk_features = input_features[None, mel_slice]
        chunk_mask = attention_mask[None, mel_slice]

        cached_embeds = speaker_cache.get_embeds()
        cached_length = len(cached_embeds)

        # feedforward
        net = models["net"]
        if not args.onnx:
            output = net.predict(
                {
                    "input_features": chunk_features,
                    "attention_mask": chunk_mask,
                    "cached_embeds": cached_embeds[None],
                }
            )
        else:
            output = net.run(
                None,
                {
                    "input_features": chunk_features,
                    "attention_mask": chunk_mask,
                    "cached_embeds": cached_embeds[None],
                },
            )
        chunk_embeds, chunk_logits = output
        chunk_embeds, chunk_logits = chunk_embeds[0], chunk_logits[0]

        chunk_input_embeds = np.concatenate([cached_embeds, chunk_embeds])
        step_mask = np.concatenate(
            [
                np.ones(cached_length, dtype=np.float32),
                chunk_mask[0, ::SUBSAMPLING_FACTOR].astype(np.float32),
            ]
        )
        speaker_cache.update(
            chunk_input_embeds, chunk_logits, num_chunk_frames, step_mask
        )

        start = cached_length * SUBSAMPLING_FACTOR
        end = (cached_length + num_chunk_frames) * SUBSAMPLING_FACTOR
        logits.append(chunk_logits[start:end])

    # with no look-ahead, the last encoder frame may be the padding of the frame stacking
    return np.concatenate(logits)[:num_frames]


def predict_offline(models, audio):
    input_features, attention_mask = extract_features(audio)

    chunk_length, chunk_right_context, fifo_length, update_period = OFFLINE_CONFIG
    speaker_cache = SpeakerCache(fifo_length, update_period, models["silence_embeds"])
    logits = forward(
        models,
        input_features,
        attention_mask,
        speaker_cache,
        chunk_length,
        chunk_right_context,
    )
    return sigmoid(logits), attention_mask


class StreamingSession:
    """Diarizes audio fed piece by piece, running the model each time a chunk and its
    look-ahead have arrived."""

    def __init__(self, models, mode):
        self.models = models
        self.chunk_length, self.chunk_right_context = STREAMING_MODES[mode]
        num_mel_frames_per_chunk = (
            self.chunk_length + self.chunk_right_context
        ) * SUBSAMPLING_FACTOR
        self.num_mel_frames_per_step = self.chunk_length * SUBSAMPLING_FACTOR
        self.num_samples_first_chunk = (
            num_mel_frames_per_chunk - 1
        ) * HOP_LENGTH + WIN_LENGTH // 2
        self.num_samples_per_chunk = num_mel_frames_per_chunk * HOP_LENGTH + WIN_LENGTH

        self.speaker_cache = SpeakerCache(
            STREAMING_FIFO_LENGTH, STREAMING_UPDATE_PERIOD, models["silence_embeds"]
        )
        self.buffer = np.zeros(0, dtype=np.float32)
        self.buffer_start = 0  # sample index of buffer[0] in the whole stream
        self.mel_frame_idx = None  # first mel frame of the next chunk

    def audio_chunk_start(self, mel_frame_idx):
        # an uncentered chunk starts n_fft // 2 before its first frame
        return mel_frame_idx * HOP_LENGTH - N_FFT // 2

    def run_chunk(self, start, end, is_first_audio_chunk, is_last_audio_chunk):
        audio_chunk = self.buffer[start - self.buffer_start : end - self.buffer_start]
        input_features, attention_mask = extract_features(
            audio_chunk,
            center=is_first_audio_chunk,
            is_last_audio_chunk=is_last_audio_chunk,
        )
        num_frames = int(attention_mask.sum())
        if num_frames == 0:
            return np.zeros((0, NUM_SPEAKERS), dtype=np.float32)
        input_features = input_features[:num_frames]
        attention_mask = attention_mask[:num_frames]
        num_lookahead = 0 if is_last_audio_chunk else self.chunk_right_context
        logits = forward(
            self.models,
            input_features,
            attention_mask,
            self.speaker_cache,
            None,
            num_lookahead,
        )

        # drop the samples no later chunk reads
        if self.mel_frame_idx is None:
            self.mel_frame_idx = self.num_mel_frames_per_step
        else:
            self.mel_frame_idx += self.num_mel_frames_per_step
        next_start = self.audio_chunk_start(self.mel_frame_idx)
        self.buffer = self.buffer[next_start - self.buffer_start :]
        self.buffer_start = next_start
        return logits

    def feed(self, audio):
        """audio: [T] -> speaker probabilities [frames, 8] decided by the new audio"""
        self.buffer = np.concatenate([self.buffer, audio.astype(np.float32)])
        num_samples = self.buffer_start + len(self.buffer)

        logits = [np.zeros((0, NUM_SPEAKERS), dtype=np.float32)]
        while True:
            if self.mel_frame_idx is None:
                start, end = 0, self.num_samples_first_chunk
            else:
                start = self.audio_chunk_start(self.mel_frame_idx)
                end = start + self.num_samples_per_chunk
            if num_samples < end:
                break
            logits.append(self.run_chunk(start, end, self.mel_frame_idx is None, False))
        return sigmoid(np.concatenate(logits))

    def finish(self):
        """Scores the frames left in the buffer, the last ones of the stream."""
        num_samples = self.buffer_start + len(self.buffer)
        if self.mel_frame_idx is None:
            # shorter than the first chunk: the whole audio is one centered chunk without look-ahead
            logits = self.run_chunk(0, num_samples, True, True)
        else:
            start = self.audio_chunk_start(self.mel_frame_idx)
            logits = self.run_chunk(start, num_samples, False, True)
        return sigmoid(logits)


def predict_streaming(models, audio, mode):
    session = StreamingSession(models, mode)
    probs = [session.feed(audio), session.finish()]
    return np.concatenate(probs), None


def predict(models, audio):
    if args.latency == "offline":
        return predict_offline(models, audio)
    return predict_streaming(models, audio, args.latency)


def recognize_from_audio(models):
    for soundf_path in args.input:
        logger.info(soundf_path)

        audio = read_wave(soundf_path)

        # inference
        logger.info(f"Start inference... (latency: {args.latency})")
        if args.benchmark:
            logger.info("BENCHMARK mode")
            total_time_estimation = 0
            for i in range(args.benchmark_count):
                start = int(round(time.time() * 1000))
                probs, attention_mask = predict(models, audio)
                end = int(round(time.time() * 1000))
                estimation_time = end - start

                # Loggin
                logger.info(f"\tailia processing estimation time {estimation_time} ms")
                if i != 0:
                    total_time_estimation = total_time_estimation + estimation_time

            logger.info(
                f"\taverage time estimation {total_time_estimation / (args.benchmark_count - 1)} ms"
            )
        else:
            probs, attention_mask = predict(models, audio)

        segments = extract_speaker_segments(probs, attention_mask, args.threshold)
        for start, end, spk in segments:
            logger.info(f"speaker_{spk}: {start:.2f}s - {end:.2f}s")

        savepath = get_savepath(args.savepath, soundf_path, ext=".png")
        plot_result(audio, probs, segments, savepath)
        logger.info(f"saved at : {savepath}")

    logger.info("Script finished successfully.")


def recognize_from_microphone(models, mic_info):
    p = mic_info["p"]
    que = mic_info["que"]
    fin = mic_info["fin"]

    mode = args.latency if args.latency in STREAMING_MODES else "low_latency"
    logger.info(f"Streaming mode: {mode}")
    session = StreamingSession(models, mode)
    frame_duration = HOP_LENGTH / SAMPLE_RATE

    recorded, probs = [], []
    num_decided_frames = 0

    def log_decided(new_probs):
        nonlocal num_decided_frames
        start = num_decided_frames * frame_duration
        num_decided_frames += len(new_probs)
        end = num_decided_frames * frame_duration
        speakers = np.nonzero((new_probs > args.threshold).any(axis=0))[0]
        label = ", ".join(f"speaker_{k}" for k in speakers) or "-"
        logger.info(f"[{start:7.2f}s - {end:7.2f}s] {label}")

    def process(wav):
        logger.info("captured! len: %s" % (wav.shape[0] / SAMPLE_RATE))
        # the captured speech is appended to one stream, so the times are those of the speech only
        recorded.append(wav)
        new_probs = session.feed(wav)
        if len(new_probs):
            probs.append(new_probs)
            log_decided(new_probs)

    # Ctrl+C only requests the stop, so that a captured audio is always processed as a whole
    stop_requested = False

    def request_stop(signum, frame):
        nonlocal stop_requested
        stop_requested = True

    default_handler = signal.signal(signal.SIGINT, request_stop)
    try:
        cout = True
        while not stop_requested:
            if cout:
                logger.info("Please speak something")
                cout = False
            try:
                wav = que.get(timeout=0.1)
            except queue.Empty:
                if not p.is_alive():
                    break
                continue
            process(wav)
            cout = True
    finally:
        fin.set()
        signal.signal(signal.SIGINT, default_handler)

    # the audio still in the queue belongs to the stream as well
    while True:
        try:
            wav = que.get(timeout=0.1)
        except queue.Empty:
            break
        process(wav)

    if recorded:
        new_probs = session.finish()
        probs.append(new_probs)
        log_decided(new_probs)

        audio = np.concatenate(recorded)
        probs = np.concatenate(probs)
        segments = extract_speaker_segments(probs, None, args.threshold)
        for start, end, spk in segments:
            logger.info(f"speaker_{spk}: {start:.2f}s - {end:.2f}s")
        plot_result(audio, probs, segments, args.savepath)
        logger.info(f"saved at : {args.savepath}")

    logger.info("script finished successfully.")


def main():
    # model files check and download
    check_and_download_models(WEIGHT_PATH, MODEL_PATH, REMOTE_PATH)

    mic_info = None
    if args.V:
        # in microphone input mode, start thread before load the model.
        mic_info = start_microphone_input(SAMPLE_RATE, sc=False, speaker=False)

    if not args.onnx:
        net = ailia.Net(MODEL_PATH, WEIGHT_PATH, env_id=args.env_id)
    else:
        import onnxruntime

        cuda = 0 < ailia.get_gpu_environment_id()
        providers = (
            ["CUDAExecutionProvider", "CPUExecutionProvider"]
            if cuda
            else ["CPUExecutionProvider"]
        )
        net = onnxruntime.InferenceSession(WEIGHT_PATH, providers=providers)

    models = {
        "net": net,
        "silence_embeds": np.load(SILENCE_EMBEDS_PATH),
    }

    if args.V:
        # microphone input mode
        recognize_from_microphone(models, mic_info)
    else:
        recognize_from_audio(models)


if __name__ == "__main__":
    main()
