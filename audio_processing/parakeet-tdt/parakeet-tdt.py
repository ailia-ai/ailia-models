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
from microphone_utils import start_microphone_input  # noqa
from model_utils import check_and_download_file, check_and_download_models  # noqa

logger = getLogger(__name__)

# ======================
# Parameters
# ======================

WEIGHT_PATH = "parakeet-tdt-0.6b-v2.onnx"
MODEL_PATH = "parakeet-tdt-0.6b-v2.onnx.prototxt"
WEIGHT_DEC_PATH = "parakeet-tdt-0.6b-v2_decoder.onnx"
MODEL_DEC_PATH = "parakeet-tdt-0.6b-v2_decoder.onnx.prototxt"

WAV_PATH = "demo.wav"
SAVE_TEXT_PATH = "output.txt"

# ======================
# Argument Parser Config
# ======================

parser = get_base_parser("Parakeet TDT", WAV_PATH, SAVE_TEXT_PATH, input_ftype="audio")
parser.add_argument(
    "-V",
    action="store_true",
    help="use microphone input",
)
parser.add_argument(
    "-m",
    "--model_type",
    default="small",
    choices=(
        "tiny",
        "base",
        "small",
        "medium",
        "large",
        "large-v3",
        "turbo",
    ),
    help="model type",
)
parser.add_argument(
    "--temperature", type=float, default=0, help="temperature to use for sampling"
)
parser.add_argument(
    "--best_of",
    type=float,
    default=5,
    help="number of candidates when sampling with non-zero temperature",
)
parser.add_argument(
    "--beam_size",
    type=int,
    default=None,  # modified for ailia models, official whisper specifies 5
    help="number of beams in beam search, only applicable when temperature is zero, None means use greedy search",
)
parser.add_argument(
    "--patience",
    type=float,
    default=None,
    help="optional patience value to use in beam decoding,"
    " as in https://arxiv.org/abs/2204.05424, the default (1.0) is equivalent to conventional beam search",
)
parser.add_argument(
    "--length_penalty",
    type=float,
    default=None,
    help="optional token length penalty coefficient (alpha)"
    " as in https://arxiv.org/abs/1609.08144, uses simple lengt normalization by default",
)
parser.add_argument(
    "--suppress_tokens",
    type=str,
    default="-1",
    help="comma-separated list of token ids to suppress during sampling;"
    " '-1' will suppress most special characters except common punctuations",
)
parser.add_argument(
    "--temperature_increment_on_fallback",
    type=float,
    default=0.2,
    help="temperature to increase when falling back when the decoding fails to meet either of the thresholds below",
)
parser.add_argument(
    "--compression_ratio_threshold",
    type=float,
    default=2.4,
    help="if the gzip compression ratio is higher than this value, treat the decoding as failed",
)
parser.add_argument(
    "--logprob_threshold",
    type=float,
    default=-1.0,
    help="if the average log probability is lower than this value, treat the decoding as failed",
)
parser.add_argument(
    "--no_speech_threshold",
    type=float,
    default=0.6,
    help="if the probability of the <|nospeech|> token is higher than this value"
    " AND the decoding has failed due to `logprob_threshold`, consider the segment as silence",
)
parser.add_argument("--onnx", action="store_true", help="execute onnxruntime version.")
parser.add_argument(
    "--dynamic_kv_cache", action="store_true", help="execute dynamic kv_cache version."
)
parser.add_argument("--debug", action="store_true", help="display progress.")
parser.add_argument("--profile", action="store_true", help="display profile.")
parser.add_argument("--ailia_audio", action="store_true", help="use ailia audio.")
parser.add_argument(
    "--disable_ailia_tokenizer", action="store_true", help="disable ailia tokenizer."
)
parser.add_argument(
    "--normal", action="store_true", help="use normal model (default : opt model)."
)
parser.add_argument(
    "--task",
    default="transcribe",
    choices=("transcribe", "translate"),
    help="task type",
)
parser.add_argument("--memory_mode", default=-1, type=int, help="memory mode")
parser.add_argument("--prompt", default=None, help="prompt for word vocabulary")
parser.add_argument(
    "--intermediate", action="store_true", help="display intermediate state."
)
parser.add_argument(
    "--fp16", action="store_true", help="use fp16 model (default : fp32 model)."
)
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

if not args.onnx:
    import ailia

    # ailia SDK 1.2.13のAILIA UNSETTLED SHAPEの抑制、1.2.14では不要
    version = ailia.get_version().split(".")
    AILIA_VERSION_MAJOR = int(version[0])
    AILIA_VERSION_MINOR = int(version[1])
    AILIA_VERSION_REVISION = int(version[2])
    REQUIRE_CONSTANT_SHAPE_BETWEEN_INFERENCE = (
        AILIA_VERSION_MAJOR <= 1
        and AILIA_VERSION_MINOR <= 2
        and AILIA_VERSION_REVISION < 14
    )
    COPY_BLOB_DATA_ENABLE = not (
        AILIA_VERSION_MAJOR <= 1
        and AILIA_VERSION_MINOR <= 2
        and AILIA_VERSION_REVISION < 15
    )
    LAYER_NORM_ENABLE = not (
        AILIA_VERSION_MAJOR <= 1
        and AILIA_VERSION_MINOR <= 2
        and AILIA_VERSION_REVISION < 16
    )
    SAVE_ENC_SHAPE = ()
    SAVE_DEC_SHAPE = ()

    if args.memory_mode == -1:
        args.memory_mode = ailia.get_memory_mode(
            reduce_constant=True,
            ignore_input_with_initializer=True,
            reduce_interstage=False,
            reuse_interstage=True,
        )
    if (args.memory_mode & 16) != 0:
        ailia.set_temporary_cache_path("./")
else:
    LAYER_NORM_ENABLE = False
    if args.fp16:
        LAYER_NORM_ENABLE = True

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
    encoder_output_length = encoder_output_length.astype(np.int32)

    # encoder_output = np.load("encoder_output.npy")

    # feedforward
    net = models["decoder"]
    if not args.onnx:
        output = net.predict([encoder_output, encoder_output_length])
    else:
        output = net.run(None, {"x": encoder_output, "out_len": encoder_output_length})
    (
        token_logits,
        duration_logits,
        encoder_output_projected,
        decoder_output,
        active_mask,
        time_indices,
        model_durations,
    ) = output

    _blank_index = 1024

    batch_size, max_time, _ = encoder_output.shape

    # Initialize batch indices and time indices
    batch_indices = np.arange(batch_size, dtype=np.int64)
    last_timesteps = np.maximum(encoder_output_length - 1, 0)
    time_indices = np.zeros(batch_size, dtype=np.int64)
    safe_time_indices = np.zeros(
        batch_size, dtype=np.int64
    )  # min(0, last_timesteps) = 0
    time_indices_current_labels = np.zeros(batch_size, dtype=np.int64)

    active_mask = active_mask.astype(bool)
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
            encoder_slice = encoder_output_projected[batch_indices, safe_time_indices]
            encoder_slice_exp = np.expand_dims(encoder_slice, 1)
            joint_output = self.joint.joint_after_projection(
                encoder_slice_exp,
                decoder_output,
            )
            logits = np.squeeze(np.squeeze(joint_output, 1), 1)

            # get labels (greedy) and scores from current logits, replace labels/scores with new
            # labels[advance_mask] are blank, and we are looking for non-blank labels
            more_logits_slice = logits[:, :-num_durations]
            more_labels = np.argmax(more_logits_slice, axis=-1)
            more_scores = np.max(more_logits_slice, axis=-1)

            if self.fusion_models is not None:
                logits_with_fusion = logits.copy()
                for fusion_idx, fusion_scores in enumerate(fusion_scores_list):
                    # combined scores with fusion model - without blank
                    logits_with_fusion[:, : -num_durations - 1] += (
                        self.fusion_models_alpha[fusion_idx] * fusion_scores
                    )
                # get max scores and labels without blank
                fusion_logits_slice = logits_with_fusion[:, : -num_durations - 1]
                more_labels_w_fusion = np.argmax(fusion_logits_slice, axis=-1)
                more_scores_w_fusion = np.max(fusion_logits_slice, axis=-1)
                # preserve "blank" / "non-blank" category
                more_labels = np.where(
                    more_labels == self._blank_index, more_labels, more_labels_w_fusion
                )

            # same as: labels[advance_mask] = more_labels[advance_mask], but non-blocking
            labels = np.where(advance_mask, more_labels, labels)
            # same as: scores[advance_mask] = more_scores[advance_mask], but non-blocking
            scores = np.where(advance_mask, more_scores, scores)
            jump_durations_indices = np.argmax(logits[:, -num_durations:], axis=-1)
            durations = model_durations[jump_durations_indices]

            if use_alignments:
                alignments.add_results_masked_(
                    active_mask=advance_mask,
                    time_indices=time_indices_current_labels,
                    logits=logits if self.preserve_alignments else None,
                    labels=more_labels if self.preserve_alignments else None,
                    confidence=self._get_frame_confidence(
                        logits=logits, num_durations=num_durations
                    ),
                )

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
        found_labels_mask = np.logical_and(
            active_mask_prev, labels != self._blank_index
        )
        # store hypotheses
        if self.max_symbols is not None:
            # pre-allocated memory, no need for checks
            batched_hyps.add_results_masked_no_checks_(
                active_mask=found_labels_mask,
                labels=labels,
                time_indices=time_indices_current_labels,
                scores=scores,
                token_durations=durations if self.include_duration else None,
            )
        else:
            # auto-adjusted storage
            batched_hyps.add_results_masked_(
                active_mask=found_labels_mask,
                labels=labels,
                time_indices=time_indices_current_labels,
                scores=scores,
                token_durations=durations if self.include_duration else None,
            )

        # stage 3: get decoder (prediction network) output with found labels
        # NB: if active_mask is False, this step is redundant;
        # but such check will require device-to-host synchronization, so we avoid it
        # preserve state/decoder_output for inactive elements
        prev_state = state
        prev_decoder_output = decoder_output
        labels_exp = np.expand_dims(labels, 1)
        decoder_output, state, *_ = self.decoder.predict(
            labels_exp, state, add_sos=False, batch_size=batch_size
        )
        decoder_output = self.joint.project_prednet(
            decoder_output
        )  # do not recalculate joint projection

        # preserve correct states/outputs for inactive elements
        self.decoder.batch_replace_states_mask(
            src_states=prev_state,
            dst_states=state,
            mask=~found_labels_mask,
        )
        found_mask_exp = np.expand_dims(np.expand_dims(found_labels_mask, -1), -1)
        decoder_output = np.where(found_mask_exp, decoder_output, prev_decoder_output)

        # stage 4: to avoid infinite looping, go to the next frame after max_symbols emission
        if self.max_symbols is not None:
            # if labels are non-blank (not end-of-utterance), check that last observed timestep with label:
            # if it is equal to the current time index, and number of observations is >= max_symbols, force blank
            force_blank_mask = np.logical_and(
                active_mask,
                np.logical_and(
                    np.logical_and(
                        labels != self._blank_index,
                        batched_hyps.last_timestamp_lasts >= self.max_symbols,
                    ),
                    batched_hyps.last_timestamp == time_indices,
                ),
            )
            time_indices += force_blank_mask  # emit blank => advance time indices
            # update safe_time_indices, non-blocking
            safe_time_indices[:] = np.minimum(time_indices, last_timesteps)
            # same as: active_mask = time_indices < encoder_output_length
            active_mask[:] = time_indices < encoder_output_length

    # fix timestamps for iterative decoding
    if prev_batched_state is not None:
        prev_decoded_exp = np.expand_dims(prev_batched_state.decoded_lengths, 1)
        batched_hyps.timestamps += prev_decoded_exp
        if use_alignments:
            alignments.timestamps += prev_decoded_exp
    # NB: last labels can not exist (nothing decoded on this step).
    # return the last labels from the previous state in this case
    last_labels = batched_hyps.get_last_labels(pad_id=self._SOS)
    decoding_state = BatchedLabelLoopingState(
        predictor_states=state,
        predictor_outputs=decoder_output,
        labels=(
            np.where(last_labels == self._SOS, prev_batched_state.labels, last_labels)
            if prev_batched_state is not None
            else last_labels
        ),
        decoded_lengths=(
            encoder_output_length.copy()
            if prev_batched_state is None
            else encoder_output_length + prev_batched_state.decoded_lengths
        ),
        fusion_states_list=fusion_states_list,
        time_jumps=time_indices - encoder_output_length,
    )
    if use_alignments:
        return batched_hyps, alignments, decoding_state

    return batched_hyps, None, decoding_state


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
        print(encoded, encoded_len)

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
        decoder = ailia.Net(MODEL_DEC_PATH, WEIGHT_DEC_PATH, env_id=env_id)
    else:
        import onnxruntime

        cuda = 0 < ailia.get_gpu_environment_id()
        providers = (
            ["CUDAExecutionProvider", "CPUExecutionProvider"]
            if cuda
            else ["CPUExecutionProvider"]
        )
        net = onnxruntime.InferenceSession(WEIGHT_PATH, providers=providers)
        decoder = onnxruntime.InferenceSession(WEIGHT_DEC_PATH, providers=providers)

    model_path = "tokenizer/tokenizer.model"
    tokenizer = tokenizers.SentencePieceTokenizer(model_path=model_path)

    models = {"tokenizer": tokenizer, "net": net, "decoder": decoder}

    # Support both manifest.json and direct audio file list
    audio_files = args.input if args.input else "manifest.json"
    recognize_from_audio(models, audio_files)


if __name__ == "__main__":
    main()
