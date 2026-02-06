import sys
import time
from collections import namedtuple
from logging import getLogger

import ailia
import numpy as np
import soundfile as sf
from nemo.collections.common import tokenizers
from tqdm import tqdm

# import original modules
# isort : on
sys.path.append("../../util")
from arg_utils import get_base_parser, get_savepath, update_parser  # noqa
from audio_utils import load_audio  # noqa
from math_utils import softmax

# isort : on
from microphone_utils import start_microphone_input
from model_utils import check_and_download_models

logger = getLogger(__name__)

# ======================
# Parameters
# ======================

WEIGHT_PATH = "parakeet-tdt-0.6b-v2.onnx"
MODEL_PATH = "parakeet-tdt-0.6b-v2.onnx.prototxt"
WEIGHT_ENCODER_PROJECTION_PATH = "parakeet-tdt-0.6b-v2_encoder_projection.onnx"
WEIGHT_PREDICTOR_PATH = "parakeet-tdt-0.6b-v2_predictor.onnx"
WEIGHT_JOINT_PATH = "parakeet-tdt-0.6b-v2_joint.onnx"
DURATIONS_PATH = "parakeet-tdt-0.6b-v2_durations.npy"

WAV_PATH = "demo.wav"
SAVE_TEXT_PATH = "output.txt"

# ======================
# Argument Parser Config
# ======================

parser = get_base_parser("Parakeet TDT", WAV_PATH, SAVE_TEXT_PATH, input_ftype="audio")
parser.add_argument(
    "--temperature", type=float, default=0, help="temperature to use for sampling"
)
parser.add_argument("--onnx", action="store_true", help="execute onnxruntime version.")
args = update_parser(parser)

# if args.ailia_audio:
#     from ailia_audio_utils import (
#         CHUNK_LENGTH,
#         HOP_LENGTH,
#         N_FRAMES,
#         N_SAMPLES,
#         SAMPLE_RATE,
#         load_audio,
#         log_mel_spectrogram,
#         pad_or_trim,
#     )
# else:
#     from audio_utils import (
#         CHUNK_LENGTH,
#         HOP_LENGTH,
#         N_FRAMES,
#         N_SAMPLES,
#         SAMPLE_RATE,
#         load_audio,
#         log_mel_spectrogram,
#         pad_or_trim,
#     )


# ======================
# Workaround
# ======================


# ======================
# Models
# ======================


REMOTE_PATH = "https://storage.googleapis.com/ailia-models/parakeet-tdt/"


# ======================
# Secondaty Functions
# ======================


def soundfile_info(path):
    """Get audio file information using soundfile."""
    info_ = sf.info(path)
    return dict(
        channels=info_.channels,
        frames=info_.frames,
        samplerate=info_.samplerate,
        duration=info_.duration,
    )


def load_audio_from_cut(cut, tokenizer):
    """
    Load audio from a cut and prepare batch data (ported from lhotse collate_audio pattern).

    Args:
        cut: Cut dict with audio_filepath and duration
        tokenizer: Tokenizer for text processing

    Returns:
        tuple: (audio_array, token_array) as numpy arrays
    """
    sample_rate = 16000  # Assume 16kHz
    audio_filepath = cut["audio_filepath"]

    wav = load_audio(audio_filepath, sr=sample_rate)

    # Handle cut (time segment) if start/duration provided
    start = cut.get("start", 0)
    duration = cut.get("duration")
    if start > 0 or duration is not None:
        start_sample = int(start * sample_rate)
        if duration is not None:
            end_sample = start_sample + int(duration * sample_rate)
            wav = wav[start_sample:end_sample]
        else:
            wav = wav[start_sample:]

    # Get text and tokenize
    text = cut.get("text", "")
    if hasattr(cut, "tokens"):
        tokens = cut.tokens
    else:
        tokens = tokenizer.text_to_ids(text) if text else []

    return np.array(wav, dtype=np.float32), np.array(tokens, dtype=np.int64)


def collate_vectors(tensors, padding_value=0, dtype=np.float32):
    """
    Collate 1-D tensors/arrays into a single padded array (right padding).

    Args:
        tensors: List of 1-D numpy arrays
        padding_value: Value to use for padding
        dtype: Data type for output array

    Returns:
        Padded 2-D numpy array of shape (batch, max_len)
    """
    if not tensors:
        return np.array([], dtype=dtype)

    # Convert to numpy arrays if needed
    tensors = [t if isinstance(t, np.ndarray) else np.array(t) for t in tensors]
    assert all(len(t.shape) == 1 for t in tensors), "Expected only 1-D tensors."

    # Find longest sequence
    max_len = max(t.shape[0] for t in tensors)

    # Create padded result array initialized with padding value
    result = np.ones((len(tensors), max_len), dtype=dtype) * padding_value
    for i, t in enumerate(tensors):
        # Pad right: copy original data to the left side
        result[i, : t.shape[0]] = t

    return result


def collate_audio(cuts, tokenizer, pad_value=0.0):
    """
    Load audio from cuts and collate into padded numpy arrays (following NeMo lhotse pattern).
    Implementation matches /usr/local/lib/python3.10/dist-packages/nemo/collections/asr/data/audio_to_text_lhotse.py L52-68

    Args:
        cuts: List of cut dicts with audio_filepath and duration
        tokenizer: Tokenizer for text processing
        pad_value: Padding value (default: 0.0)

    Returns:
        tuple: (audio_padded, audio_lens_arr, tokens_padded, token_lens)
    """
    if not cuts:
        return (
            np.array([]),
            np.array([], dtype=np.int64),
            np.array([]),
            np.array([], dtype=np.int64),
        )

    # Step 1: Load audio and tokens from each cut (load_audio called only once per file)
    audio_list = []
    tokens_list = []
    audio_lens = []

    for cut in cuts:
        audio, tokens = load_audio_from_cut(cut, tokenizer)
        audio_list.append(audio)
        tokens_list.append(tokens)
        audio_lens.append(len(audio))

    # Step 2: Collate audio using collate_vectors (right padding with pad_value)
    audio_padded = collate_vectors(
        audio_list, padding_value=pad_value, dtype=np.float32
    )
    audio_lens_arr = np.array(audio_lens, dtype=np.int64)

    # Step 3: Collate tokens using collate_vectors (right padding with -100)
    token_lens = np.array([len(t) for t in tokens_list], dtype=np.int64)
    tokens_padded = collate_vectors(tokens_list, padding_value=-100, dtype=np.int64)

    return audio_padded, audio_lens_arr, tokens_padded, token_lens


def data_sampler(audio_files, max_duration=None, max_cuts=None, drop_last=False):
    batch = []
    batch_duration = 0.0
    longest_seen = 0.0
    for path in audio_files:
        audio_info = soundfile_info(path)
        cut = dict(
            audio_filepath=path,
            sampling_rate=audio_info["samplerate"],
            num_samples=audio_info["frames"],
            duration=audio_info["duration"],
            channel_ids=list(range(audio_info["channels"])),
            text="",
        )
        cut_duration = cut.get("duration", 0.0)

        # Add cut to batch
        batch.append(cut)
        batch_duration += cut_duration
        longest_seen = max(longest_seen, cut_duration)

        # Check if adding this cut would exceed constraints
        would_exceed_duration = (
            max_duration is not None and batch_duration > max_duration
        )
        would_exceed_cuts = max_cuts is not None and len(batch) >= max_cuts

        if batch and (would_exceed_duration or would_exceed_cuts):
            # Yield current batch and start new one
            yield batch
            batch = []
            batch_duration = 0.0

    # Yield remaining batch
    if batch and not drop_last:
        yield batch


def data_loader(
    tokenizer,
    audio_files,
    max_duration=None,
    max_cuts=None,
    drop_last=False,
):
    """
    Data loader: sampler generates cuts with DurationBatcher, dataset fetches data.

    Args:
        tokenizer: NeMo tokenizer
        audio_files: List of audio file paths or manifest path (str ending with .json)
        max_duration: Maximum total audio duration in seconds per batch (optional)
        max_cuts: Maximum number of cuts per batch (optional)
        drop_last: Drop last incomplete batch if True (default: False)

    Yields:
        tuple: (audio, audio_lens, tokens, token_lens)
    """
    sampler = data_sampler(
        audio_files,
        max_duration=max_duration,
        max_cuts=max_cuts,
        drop_last=drop_last,
    )

    for batch_cuts in sampler:
        # Load audio and collate (load_audio called only once per file inside collate_audio)
        audio_padded, audio_lens_arr, tokens_padded, token_lens = collate_audio(
            batch_cuts, tokenizer, pad_value=0.0
        )

        yield (audio_padded, audio_lens_arr, tokens_padded, token_lens)


# ======================
# Main functions
# ======================


def predict(models, input_signal, input_signal_length):
    input_signal_length = input_signal_length.astype(np.int32)

    # feedforward
    net = models["net"]
    if not args.onnx:
        output = net.predict([input_signal, input_signal_length])
    else:
        output = net.run(
            None,
            {
                "input_signal": input_signal,
                "input_signal_length": input_signal_length,
            },
        )
    encoded, encoded_len = output

    return encoded, encoded_len


def decode_full(models, encoder_output, encoder_output_length):
    encoder_output_length = encoder_output_length.astype(np.int64)

    # Load models and metadata
    encoder_projection = models["encoder_projection"]
    predictor = models["predictor"]
    joint_net = models["joint"]
    _blank_index = models["blank_index"]
    model_durations = models["durations"]
    num_durations = len(model_durations)

    # Step 1: Project encoder output
    if not args.onnx:
        output = encoder_projection.predict([encoder_output])
    else:
        output = encoder_projection.run(None, {"encoder_output": encoder_output})
    encoder_output_projected = output[0]

    # Step 2: Initialize decoder state
    batch_size = encoder_output.shape[0]
    labels = np.full((batch_size, 1), _blank_index, dtype=np.int64)
    state_0 = np.zeros((2, batch_size, 640), dtype=np.float32)
    state_1 = np.zeros((2, batch_size, 640), dtype=np.float32)

    # Get initial decoder output
    if not args.onnx:
        output = predictor.predict([labels, state_0, state_1])
    else:
        output = predictor.run(
            None, {"labels": labels, "state_0": state_0, "state_1": state_1}
        )
    decoder_output = output[0]
    state_0 = output[1]
    state_1 = output[2]

    # Step 3: Get initial joint output
    batch_indices = np.arange(batch_size, dtype=np.int64)
    last_timesteps = np.maximum(encoder_output_length - 1, 0)
    time_indices = np.zeros(batch_size, dtype=np.int64)
    safe_time_indices = np.minimum(time_indices, last_timesteps)

    encoder_output_frame = np.expand_dims(
        encoder_output_projected[batch_indices, safe_time_indices], axis=1
    )
    if not args.onnx:
        output = joint_net.predict([encoder_output_frame, decoder_output])
    else:
        output = joint_net.run(
            None,
            {
                "encoder_output": encoder_output_frame,
                "decoder_output": decoder_output,
            },
        )
    logits = output[0]

    token_logits = logits[:, :-num_durations]
    duration_logits = logits[:, -num_durations:]

    batch_size, max_time, _ = encoder_output.shape

    # Initialize batch indices and time indices
    batch_indices = np.arange(batch_size, dtype=np.int64)
    last_timesteps = np.maximum(encoder_output_length - 1, 0)
    time_indices = np.zeros(batch_size, dtype=np.int64)
    safe_time_indices = np.zeros(
        batch_size, dtype=np.int64
    )  # min(0, last_timesteps) = 0
    time_indices_current_labels = np.zeros(batch_size, dtype=np.int64)

    active_mask = time_indices < encoder_output_length
    active_mask_prev = active_mask.copy()
    advance_mask = np.zeros(batch_size, dtype=bool)

    # loop while there are active utterances
    while active_mask.any():
        # stage 1: get joint output, iteratively seeking for non-blank labels
        # blank label in `labels` tensor means "end of hypothesis" (for this index)
        active_mask_prev = active_mask.copy()

        # stage 1.1: get first joint output
        labels = np.argmax(token_logits, axis=-1)
        scores = np.max(token_logits, axis=-1)

        jump_durations_indices = np.argmax(duration_logits, axis=-1)
        durations = model_durations[jump_durations_indices]

        # search for non-blank labels using joint, advancing time indices for blank labels
        # checking max_symbols is not needed, since we already forced advancing time indices for such cases
        blank_mask = labels == _blank_index
        # for blank labels force duration >= 1
        mask_fill = np.logical_and(durations == 0, blank_mask)
        durations[mask_fill] = 1
        time_indices_current_labels = time_indices.copy()

        # advance_mask is a mask for current batch for searching non-blank labels;
        # each element is True if non-blank symbol is not yet found AND we can increase the time index
        time_indices += durations * active_mask
        safe_time_indices[:] = np.minimum(time_indices, last_timesteps)
        active_mask[:] = time_indices < encoder_output_length
        advance_mask[:] = np.logical_and(active_mask, blank_mask)

        # stage 1.2: inner loop - find next non-blank labels (if exist)
        while advance_mask.any():
            # same as: time_indices_current_labels[advance_mask] = time_indices[advance_mask], but non-blocking
            # store current time indices to use further for storing the results
            time_indices_current_labels[:] = np.where(
                advance_mask, time_indices, time_indices_current_labels
            )
            encoder_output_frame = np.expand_dims(
                encoder_output_projected[batch_indices, safe_time_indices], axis=1
            )
            if not args.onnx:
                output = joint_net.predict([encoder_output_frame, decoder_output])
            else:
                output = joint_net.run(
                    None,
                    {
                        "encoder_output": encoder_output_frame,
                        "decoder_output": decoder_output,
                    },
                )
            logits = output[0]

            # get labels (greedy) and scores from current logits, replace labels/scores with new
            # labels[advance_mask] are blank, and we are looking for non-blank labels
            more_logits_slice = logits[:, :-num_durations]
            more_labels = np.argmax(more_logits_slice, axis=-1)
            more_scores = np.max(more_logits_slice, axis=-1)

            # same as: labels[advance_mask] = more_labels[advance_mask], but non-blocking
            labels = np.where(advance_mask, more_labels, labels)
            # same as: scores[advance_mask] = more_scores[advance_mask], but non-blocking
            scores = np.where(advance_mask, more_scores, scores)
            jump_durations_indices = np.argmax(logits[:, -num_durations:], axis=-1)
            durations = model_durations[jump_durations_indices]

            blank_mask = labels == self._blank_index
            # for blank labels force duration >= 1
            mask_fill = np.logical_and(durations == 0, blank_mask)
            durations[mask_fill] = 1
            # same as time_indices[advance_mask] += durations[advance_mask], but non-blocking
            time_indices = np.where(
                advance_mask, time_indices + durations, time_indices
            )
            safe_time_indices[:] = np.minimum(time_indices, last_timesteps)
            active_mask[:] = time_indices < encoder_output_length
            advance_mask[:] = np.logical_and(active_mask, blank_mask)

        # NB: difference between RNN-T and TDT here, at the end of utterance:
        # For RNN-T, if we found a non-blank label, the utterance is active (need to find blank to stop decoding)
        # For TDT, we could find a non-blank label, add duration, and the utterance may become inactive
        found_labels_mask = np.logical_and(active_mask_prev, labels != _blank_index)

        # Store found labels (simplified - just collect them)
        decoded_labels = []
        for i in range(batch_size):
            if found_labels_mask[i]:
                decoded_labels.append(int(labels[i]))

        # stage 3: get decoder (prediction network) output with found labels
        # preserve state/decoder_output for inactive elements
        prev_state_0 = state_0.copy()
        prev_state_1 = state_1.copy()
        prev_decoder_output = decoder_output.copy()

        labels_input = labels.reshape(batch_size, 1)
        if not args.onnx:
            output = predictor.predict([labels_input, state_0, state_1])
        else:
            output = predictor.run(
                None, {"labels": labels_input, "state_0": state_0, "state_1": state_1}
            )
        decoder_output = output[0]
        state_0 = output[1]
        state_1 = output[2]

        # preserve correct states/outputs for inactive elements
        for i in range(batch_size):
            if not found_labels_mask[i]:
                state_0[:, i, :] = prev_state_0[:, i, :]
                state_1[:, i, :] = prev_state_1[:, i, :]
                decoder_output[i] = prev_decoder_output[i]

    # Return decoded labels
    return decoded_labels


def transcribe_post_processing(models, encoder_output, encoded_lengths):
    # Apply optional preprocessing
    encoder_output = encoder_output.transpose(0, 2, 1)  # (B, T, D)

    decoded_tokens = decode_full(models, encoder_output, encoded_lengths)

    return decoded_tokens


def recognize_from_audio(models, audio_files):
    tokenizer = models["tokenizer"]

    audio_files = ["2086-149220-0033.wav"]
    dloader = data_loader(tokenizer, audio_files, max_cuts=2)
    for batch in tqdm(dloader, desc="Transcribing"):
        encoded, encoded_len = predict(
            models, input_signal=batch[0], input_signal_length=batch[1]
        )

        transcribe_post_processing(models, encoded, encoded_len)

    logger.info("Script finished successfully.")


def main():
    # check_and_download_models(WEIGHT_ENC_PATH, MODEL_ENC_PATH, REMOTE_PATH)
    # check_and_download_models(WEIGHT_DEC_PATH, MODEL_DEC_PATH, REMOTE_PATH)
    # check_and_download_file(WEIGTH_ENC_LARGE_PB_PATH, REMOTE_PATH)

    env_id = args.env_id

    # initialize
    if not args.onnx:
        net = ailia.Net(MODEL_PATH, WEIGHT_PATH, env_id=env_id)
        encoder_projection = ailia.Net(
            None, WEIGHT_ENCODER_PROJECTION_PATH, env_id=env_id
        )
        predictor = ailia.Net(None, WEIGHT_PREDICTOR_PATH, env_id=env_id)
        joint = ailia.Net(None, WEIGHT_JOINT_PATH, env_id=env_id)
    else:
        import onnxruntime

        cuda = 0 < ailia.get_gpu_environment_id()
        providers = (
            ["CUDAExecutionProvider", "CPUExecutionProvider"]
            if cuda
            else ["CPUExecutionProvider"]
        )
        net = onnxruntime.InferenceSession(WEIGHT_PATH, providers=providers)
        encoder_projection = onnxruntime.InferenceSession(
            WEIGHT_ENCODER_PROJECTION_PATH, providers=providers
        )
        predictor = onnxruntime.InferenceSession(
            WEIGHT_PREDICTOR_PATH, providers=providers
        )
        joint = onnxruntime.InferenceSession(WEIGHT_JOINT_PATH, providers=providers)

    model_path = "tokenizer/tokenizer.model"
    tokenizer = tokenizers.SentencePieceTokenizer(model_path=model_path)

    models = {
        "tokenizer": tokenizer,
        "net": net,
        "encoder_projection": encoder_projection,
        "predictor": predictor,
        "joint": joint,
        "blank_index": 1024,
        # "durations": durations,
        "durations": np.arange(5, dtype=np.int64),
    }

    # Support both manifest.json and direct audio file list
    audio_files = args.input if args.input else "manifest.json"
    recognize_from_audio(models, audio_files)


if __name__ == "__main__":
    main()
