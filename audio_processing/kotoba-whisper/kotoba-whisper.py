import sys
import time
from typing import List
from logging import getLogger

import numpy as np
from scipy.special import log_softmax, logsumexp
import librosa


# import original modules
sys.path.append("../../util")
from arg_utils import get_base_parser, update_parser  # noqa
from model_utils import check_and_download_models, check_and_download_file  # noqa

import ailia

from audio_utils import SAMPLE_RATE, extract_fbank_features
from tokenizer_utils import decode_asr


logger = getLogger(__name__)

# ======================
# Parameters
# ======================

WEIGHT_ENC_PATH = "kotoba-whisper-v1.0_encoder.onnx"
WEIGHT_ENC_PB_PATH = "kotoba-whisper-v1.0_encoder_weights.pb"
MODEL_ENC_PATH = "kotoba-whisper-v1.0_encoder.onnx.prototxt"
WEIGHT_DEC_PATH = "kotoba-whisper-v1.0_decoder.onnx"
MODEL_DEC_PATH = "kotoba-whisper-v1.0_decoder.onnx.prototxt"
REMOTE_PATH = "https://storage.googleapis.com/ailia-models/kotoba-whisper/"

WAV_PATH = "demo.wav"
SAVE_TEXT_PATH = "output.txt"

flg_ffmpeg = False

# ======================
# Arguemnt Parser Config
# ======================

parser = get_base_parser(
    "Kotoba-Whisper", WAV_PATH, SAVE_TEXT_PATH, input_ftype="audio"
)
parser.add_argument(
    "--chunk_length", type=int, default=None, help="the chunk size for chunking."
)
parser.add_argument(
    "--batch_size",
    type=int,
    default=1,
    help="the size of the batch to use, for inference.",
)
parser.add_argument(
    '--disable_ailia_tokenizer',
    action='store_true',
    help='disable ailia tokenizer.'
)
parser.add_argument("--onnx", action="store_true", help="execute onnxruntime version.")
args = update_parser(parser, check_input_type=False)


# ======================
# Secondaty Functions
# ======================


def load_audio(file: str, sr: int = SAMPLE_RATE):
    if flg_ffmpeg:
        import ffmpeg

        try:
            out, _ = (
                ffmpeg.input(file, threads=0)
                .output("-", format="f32le", acodec="pcm_f32le", ac=1, ar=sr)
                .run(cmd="ffmpeg", capture_stdout=True, capture_stderr=True)
            )
        except ffmpeg.Error as e:
            raise RuntimeError("ffmpeg command failed: %s" % e.stderr.decode())
        audio = np.frombuffer(out, np.float32)
    else:
        # prepare input data
        audio, source_sr = librosa.load(file, sr=None)
        # Resample the audio if needed
        if source_sr is not None and source_sr != sr:
            audio = librosa.resample(audio, orig_sr=source_sr, target_sr=sr)

    return audio


def feature_extractor(raw_speech, padding="max_length"):
    raw_speech = np.asarray([raw_speech]).T

    max_length = 480000
    if padding == "longest":
        max_length = len(raw_speech)
    else:
        raw_speech = raw_speech[:max_length]

    if len(raw_speech) < max_length:
        difference = max_length - len(raw_speech)
        raw_speech = np.pad(
            raw_speech,
            ((0, difference), (0, 0)),
            "constant",
        )

    raw_speech = np.expand_dims(raw_speech, axis=0)
    raw_speech = raw_speech.transpose(2, 0, 1)

    features = extract_fbank_features(raw_speech[0])
    features = features.astype(np.float32)

    return features


def chunk_iter(inputs, chunk_len, stride_left, stride_right):
    inputs_len = inputs.shape[0]
    step = chunk_len - stride_left - stride_right
    for chunk_start_idx in range(0, inputs_len, step):
        chunk_end_idx = chunk_start_idx + chunk_len
        chunk = inputs[chunk_start_idx:chunk_end_idx]
        features = feature_extractor(chunk)

        _stride_left = 0 if chunk_start_idx == 0 else stride_left
        # all right strides must be full, otherwise it is the last item
        is_last = (
            chunk_end_idx > inputs_len
            if stride_right > 0
            else chunk_end_idx >= inputs_len
        )
        _stride_right = 0 if is_last else stride_right

        chunk_len = chunk.shape[0]
        stride = (chunk_len, _stride_left, _stride_right)

        if chunk.shape[0] > _stride_left:
            yield {"input_features": features, "stride": stride}
        if is_last:
            break


# ======================
# Main functions
# ======================


def preprocess(inputs, chunk_length_s=0):
    if chunk_length_s:
        stride_length_s = chunk_length_s / 6

        chunk_len = chunk_length_s * SAMPLE_RATE
        stride_left = stride_right = int(round(stride_length_s * SAMPLE_RATE))

        if chunk_len < stride_left + stride_right:
            raise ValueError("Chunk length must be superior to stride length")

        for item in chunk_iter(
            inputs,
            chunk_len,
            stride_left,
            stride_right,
        ):
            yield item
    else:
        n_samples = 480000
        if inputs.shape[0] > n_samples:
            features = feature_extractor(
                inputs,
                padding="longest",
            )
        else:
            features = feature_extractor(inputs)
        yield {"input_features": features}


def decode(
    net,
    encoder_hidden_states: np.ndarray,
    input_ids: np.ndarray,
    past_key_values: List[np.ndarray],
):
    if args.benchmark:
        start = int(round(time.time() * 1000))

    if not args.onnx:
        decoder_output = net.predict(
            [
                input_ids,
                encoder_hidden_states,
                past_key_values[0],
                past_key_values[1],
                past_key_values[2],
                past_key_values[3],
                past_key_values[4],
                past_key_values[5],
                past_key_values[6],
                past_key_values[7],
            ]
        )
    else:
        decoder_output = net.run(
            None,
            {
                "input_ids": input_ids,
                "encoder_hidden_states": encoder_hidden_states,
                "past_key_values.0.decoder.key": past_key_values[0],
                "past_key_values.0.decoder.value": past_key_values[1],
                "past_key_values.0.encoder.key": past_key_values[2],
                "past_key_values.0.encoder.value": past_key_values[3],
                "past_key_values.1.decoder.key": past_key_values[4],
                "past_key_values.1.decoder.value": past_key_values[5],
                "past_key_values.1.encoder.key": past_key_values[6],
                "past_key_values.1.encoder.value": past_key_values[7],
            },
        )

    if args.benchmark:
        end = int(round(time.time() * 1000))
        estimation_time = end - start
        logger.info(f"\tdecoder processing time {estimation_time} ms")

    logits, new_past_key_values = decoder_output[0], decoder_output[1:]

    return logits, new_past_key_values


def stopping_criteria(input_ids: np.array) -> bool:
    is_done = np.full((input_ids.shape[0],), False)

    # MaxLengthCriteria
    max_length = 448
    cur_len = input_ids.shape[-1]
    is_done = is_done | cur_len >= max_length

    # EosTokenCriteria
    eos_token_id = 50257
    is_done = is_done | np.isin(input_ids[:, -1], [eos_token_id])

    return is_done


def greedy_search(net, input_ids, encoder_hidden_states):
    pad_token_id = 50257
    suppress_tokens = [
        # fmt: off
        1,     2,     7,     8,     9,     10,    14,    25,    26,    27,
        28,    29,    31,    58,    59,    60,    61,    62,    63,    90,
        91,    92,    93,    359,   503,   522,   542,   873,   893,   902,
        918,   922,   931,   1350,  1853,  1982,  2460,  2627,  3246,  3253,
        3268,  3536,  3846,  3961,  4183,  4667,  6585,  6647,  7273,  9061,
        9383,  10428, 10929, 11938, 12033, 12331, 12562, 13793, 14157, 14635,
        15265, 15618, 16553, 16604, 18362, 18956, 20075, 21675, 22520, 26130,
        26161, 26435, 28279, 29464, 31650, 32302, 32470, 36865, 42863, 47425,
        49870, 50254, 50258, 50359, 50360, 50361, 50362, 50363
        # fmt: on
    ]
    begin_index = 3
    begin_suppress_tokens = [220, 50257]
    no_timestamps_token_id = 50364
    timestamp_begin = 50365
    eos_token_id = 50257
    max_initial_timestamp_index = 50

    batch_size, cur_len = input_ids.shape
    past_key_values = [np.zeros((batch_size, 20, 0, 64), dtype=np.float16)] * 8

    # keep track of which sequences are already finished
    this_peer_finished = False
    unfinished_sequences = np.ones(batch_size, dtype=int)

    while not this_peer_finished:
        # prepare model inputs
        past_length = past_key_values[0].shape[2]
        decoder_input_ids = input_ids[:, past_length:]

        # forward pass to get next token
        logits, past_key_values = decode(
            net,
            encoder_hidden_states,
            decoder_input_ids,
            past_key_values,
        )
        next_tokens_scores = logits[:, -1, :]

        # SuppressTokensLogitsProcessor
        next_tokens_scores[:, suppress_tokens] = -float("inf")
        # SuppressTokensAtBeginLogitsProcessor
        if input_ids.shape[1] == begin_index:
            next_tokens_scores[:, begin_suppress_tokens] = -float("inf")

        # WhisperTimeStampLogitsProcessor
        next_tokens_scores[:, no_timestamps_token_id] = -float("inf")
        # timestamps have to appear in pairs, except directly before eos_token; mask logits accordingly
        for k in range(input_ids.shape[0]):
            sampled_tokens = input_ids[k, begin_index:]
            seq = sampled_tokens.tolist()

            last_was_timestamp = len(seq) >= 1 and seq[-1] >= timestamp_begin
            penultimate_was_timestamp = len(seq) < 2 or seq[-2] >= timestamp_begin

            if last_was_timestamp:
                if penultimate_was_timestamp:  # has to be non-timestamp
                    next_tokens_scores[k, timestamp_begin:] = -float("inf")
                else:  # cannot be normal text tokens
                    next_tokens_scores[k, :eos_token_id] = -float("inf")

            timestamps = sampled_tokens[
                np.greater_equal(sampled_tokens, timestamp_begin)
            ]
            if timestamps.size > 0:
                # `timestamps` shouldn't decrease; forbid timestamp tokens smaller than the last
                # The following lines of code are copied from: https://github.com/openai/whisper/pull/914/files#r1137085090
                if last_was_timestamp and not penultimate_was_timestamp:
                    timestamp_last = timestamps[-1]
                else:
                    # Avoid to emit <|0.00|> again
                    timestamp_last = timestamps[-1] + 1

                next_tokens_scores[k, timestamp_begin:timestamp_last] = -float("inf")
        # apply the `max_initial_timestamp` option
        if input_ids.shape[1] == begin_index:
            next_tokens_scores[:, :timestamp_begin] = -float("inf")

            if max_initial_timestamp_index is not None:
                last_allowed = timestamp_begin + max_initial_timestamp_index
                next_tokens_scores[:, last_allowed + 1 :] = -float("inf")
        # if sum of probability over timestamps is above any other token, sample timestamp
        logprobs = log_softmax(next_tokens_scores, axis=-1)
        for k in range(input_ids.shape[0]):
            timestamp_logprob = logsumexp(logprobs[k, timestamp_begin:], axis=-1)
            max_text_token_logprob = np.max(logprobs[k, :timestamp_begin])
            if timestamp_logprob > max_text_token_logprob:
                next_tokens_scores[k, :timestamp_begin] = -float("inf")

        # argmax
        next_tokens = np.argmax(next_tokens_scores, axis=-1)

        # finished sentences should have their next token be a padding token
        next_tokens = next_tokens * unfinished_sequences + pad_token_id * (
            1 - unfinished_sequences
        )

        # update generated ids, model inputs, and length for next step
        input_ids = np.concatenate([input_ids, next_tokens[:, None]], axis=-1)

        is_stopping = stopping_criteria(input_ids)
        unfinished_sequences = unfinished_sequences & ~is_stopping
        if np.max(unfinished_sequences) == 0:
            this_peer_finished = True

    return input_ids


def forward(models, input_features):
    if args.benchmark:
        start = int(round(time.time() * 1000))

    # encoder
    net = models["enc"]
    if not args.onnx:
        output = net.run(input_features)
    else:
        output = net.run(None, {"input_features": input_features})
    last_hidden_state = output[0]
    last_hidden_state = last_hidden_state.astype(np.float16)

    if args.benchmark:
        end = int(round(time.time() * 1000))
        estimation_time = end - start
        logger.info(f"\tencoder processing time {estimation_time} ms")

    # language: japanese
    # task: transcribe
    init_tokens = np.array([[50258, 50266, 50360]])

    batch_size = input_features.shape[0]
    decoder_input_ids = np.repeat(init_tokens[:, ...], batch_size, axis=0)

    net = models["dec"]
    outputs = greedy_search(net, decoder_input_ids, last_hidden_state)

    return outputs


def generate(models, input_features):
    num_segment_frames = 3000
    total_input_frames = input_features.shape[-1]
    is_shortform = total_input_frames <= num_segment_frames

    if is_shortform:
        outputs = forward(models, input_features)
    else:
        # global generate variables
        time_precision = 0.02
        input_stride = 2

        # global longform generation variables
        timestamp_begin = 50365
        max_frames = np.ones((1,), dtype=int) * total_input_frames
        seek = np.zeros((1,), dtype=int)

        # Preppare running variables, list for generation
        current_segments = [[]]

        def retrieve_segment(
            seek_sequence,
            seek_outputs,
            time_offset,
            timestamp_begin,
            seek_num_frames,
            time_precision,
            input_stride,
            prev_idx,
            idx,
        ):
            # find the predicted "end of segment" predictions of Whisper
            # "end of segment" predictions occur whenever Whisper predicts a timestamp token
            timestamp_tokens = np.greater_equal(seek_sequence, timestamp_begin)
            single_timestamp_ending = timestamp_tokens[-2:].tolist() == [False, True]
            timestamp_segment_indices = np.where(
                timestamp_tokens[:-1] & timestamp_tokens[1:]
            )[0]
            np.add(timestamp_segment_indices, 1, out=timestamp_segment_indices)

            # If whisper predicted a "end of segment" via a timestep token, let's go ever each
            # "end of segment" prediction and slice the decoding into segments accordingly
            if len(timestamp_segment_indices) > 0:
                # if the output contains two consecutive timestamp tokens
                slices = timestamp_segment_indices.tolist()
                segments = []
                if single_timestamp_ending:
                    slices.append(len(seek_sequence))

                last_slice = 0
                # Add each segment to list of all segments
                for current_slice in slices:
                    sliced_tokens = seek_sequence[last_slice:current_slice]
                    start_timestamp_pos = sliced_tokens[0].item() - timestamp_begin
                    end_timestamp_pos = sliced_tokens[-1].item() - timestamp_begin
                    segments.append(
                        {
                            "start": time_offset[prev_idx]
                            + start_timestamp_pos * time_precision,
                            "end": time_offset[prev_idx]
                            + end_timestamp_pos * time_precision,
                            "tokens": sliced_tokens,
                            "result": seek_outputs[idx],
                        }
                    )
                    logger.info(models["tokenizer"].decode(sliced_tokens))
                    last_slice = current_slice

                if single_timestamp_ending:
                    # single timestamp at the end means no speech after the last timestamp.
                    segment_offset = seek_num_frames[prev_idx]
                else:
                    # otherwise, ignore the unfinished segment and seek to the last timestamp
                    # here we throw away all predictions after the last predicted "end of segment"
                    # since we are cutting right in the middle of an audio
                    last_timestamp_pos = (
                        seek_sequence[last_slice - 1].item() - timestamp_begin
                    )
                    segment_offset = last_timestamp_pos * input_stride
            else:
                # If whisper does not predict any "end of segment" token, then
                # the whole decoding is considered a segment and we add it to the list of segments
                timestamps = seek_sequence[timestamp_tokens.nonzero()[0]]
                last_timestamp_pos = seek_num_frames[prev_idx]
                if timestamps.size > 0 and timestamps[-1].item() != timestamp_begin:
                    # no consecutive timestamps but it has a timestamp; use the last one.
                    last_timestamp_pos = timestamps[-1].item() - timestamp_begin
                segments = [
                    {
                        "start": time_offset[prev_idx],
                        "end": time_offset[prev_idx]
                        + last_timestamp_pos * time_precision,
                        "tokens": seek_sequence,
                        "result": seek_outputs[idx],
                    }
                ]
                logger.info(models["tokenizer"].decode(seek_sequence))
                segment_offset = seek_num_frames[prev_idx]

            return segments, segment_offset

        eos_token_id = 50257
        pad_token_id = 50257

        # Transcribe audio until we reach the end of all input audios
        while (seek < max_frames).any():
            time_offset = seek * time_precision / input_stride
            seek_num_frames = np.clip(max_frames - seek, None, num_segment_frames)

            # cut out next 30s segment from input features
            segment_input = input_features[:, :, seek[0] : seek[0] + seek_num_frames[0]]
            if segment_input.shape[-1] < num_segment_frames:
                # pad to 3000 if necessary
                segment_input = np.pad(
                    segment_input,
                    pad_width=(
                        (0, 0),
                        (0, 0),
                        (0, num_segment_frames - segment_input.shape[-1]),
                    ),
                )

            seek_outputs = forward(models, segment_input)
            seek_sequences = seek_outputs = seek_outputs[:, 3:]

            seek_sequence_list = []
            seek_outputs_list = []

            for i, seek_sequence in enumerate(seek_sequences):
                # make sure we cut a predicted EOS token if we are not finished with the generation yet
                is_not_final = (seek[0] + num_segment_frames) < max_frames[0]
                # remove eos token id
                if is_not_final and seek_sequence[-1] == eos_token_id:
                    seek_sequence = seek_sequence[:-1]

                # remove all padding tokens
                if seek_sequence[-1] == pad_token_id:
                    num_paddings = np.sum(seek_sequence == pad_token_id)
                    seek_sequence = seek_sequence[:-num_paddings]

                seek_sequence_list.append(seek_sequence)
                seek_outputs_list.append(seek_outputs[i])

            seek_sequences = seek_sequence_list
            seek_outputs = seek_outputs_list

            # 6.9 In every generated sequence, split by timestamp tokens and extract segments
            for i, seek_sequence in enumerate(seek_sequences):
                prev_i = 0

                segments, segment_offset = retrieve_segment(
                    seek_sequence=seek_sequence,
                    seek_outputs=seek_outputs,
                    time_offset=time_offset,
                    timestamp_begin=timestamp_begin,
                    seek_num_frames=seek_num_frames,
                    time_precision=time_precision,
                    input_stride=input_stride,
                    prev_idx=prev_i,
                    idx=i,
                )

                current_segments[i] += segments
                seek[i] += segment_offset

        max_total_length = 0
        sequences = []
        for current_segment_list in current_segments:
            if (
                current_segment_list is not None
                and len([d["tokens"] for d in current_segment_list]) > 0
            ):
                sequence = np.concatenate(
                    [d["tokens"] for d in current_segment_list], axis=-1
                )
                sequences.append(sequence)
                max_total_length = max(max_total_length, len(sequences[-1]))
            else:
                sequences.append(np.array([]))
        for i in range(len(current_segments)):
            pad_length = max_total_length - len(sequences[i])
            pad = (0, pad_length)

            sequences[i] = np.pad(
                sequences[i], pad_width=pad, constant_values=pad_token_id
            )
        tokens = np.stack(sequences, axis=0)
        outputs = tokens

    return outputs


def predict(models, audio, chunk_length_s=0):
    batch_size = args.batch_size

    processed = preprocess(audio, chunk_length_s)

    model_outputs = []

    # pack_iter
    accumulator = []
    for item in processed:
        accumulator.append(item)
        if batch_size <= len(accumulator):
            input_features = np.concatenate(
                [item.pop("input_features") for item in accumulator], axis=0
            )
            tokens = generate(models, input_features)
            for i, item in enumerate(accumulator):
                item["tokens"] = tokens[i : i + 1]
                model_outputs.append(item)
            accumulator.clear()

    if 0 < len(accumulator):
        input_features = np.concatenate(
            [item.pop("input_features") for item in accumulator], axis=0
        )
        tokens = generate(models, input_features)
        for i, item in enumerate(accumulator):
            item["tokens"] = tokens[i : i + 1]
            model_outputs.append(item)

    if chunk_length_s:
        for item in model_outputs:
            chunk_len, stride_left, stride_right = item["stride"]
            # Go back in seconds
            chunk_len /= SAMPLE_RATE
            stride_left /= SAMPLE_RATE
            stride_right /= SAMPLE_RATE
            item["stride"] = chunk_len, stride_left, stride_right

    # postprocess
    tokenizer = models["tokenizer"]
    time_precision = 0.02
    if args.disable_ailia_tokenizer:
        text, optional = tokenizer._decode_asr(
            model_outputs,
            return_timestamps=True,
            return_language=None,
            time_precision=time_precision)
    else:
        text, optional = decode_asr(
            tokenizer,
            model_outputs,
            return_timestamps=True,
            return_language=None,
            time_precision=time_precision,
            logger=logger)

    return {"text": text, **optional}


def recognize_from_audio(models):
    chunk_length_s = args.chunk_length

    # input image loop
    for audio_path in args.input:
        logger.info(audio_path)

        # prepare input data
        audio = load_audio(audio_path)

        # inference
        logger.info("Start inference...")

        if args.benchmark:
            start = int(round(time.time() * 1000))

        result = predict(models, audio, chunk_length_s=chunk_length_s)

        if args.benchmark:
            end = int(round(time.time() * 1000))
            estimation_time = end - start
            logger.info(f"\ttotal processing time {estimation_time} ms")

    logger.info("Final output.")
    logger.info(result["text"])

    logger.info("Script finished successfully.")


def main():
    # model files check and download
    check_and_download_models(WEIGHT_ENC_PATH, MODEL_ENC_PATH, REMOTE_PATH)
    check_and_download_models(WEIGHT_DEC_PATH, MODEL_DEC_PATH, REMOTE_PATH)
    check_and_download_file(WEIGHT_ENC_PB_PATH, REMOTE_PATH)

    env_id = args.env_id

    # initialize
    if not args.onnx:
        enc_net = ailia.Net(
            MODEL_ENC_PATH, WEIGHT_ENC_PATH, env_id=env_id, memory_mode=args.memory_mode
        )
        dec_net = ailia.Net(
            MODEL_DEC_PATH, WEIGHT_DEC_PATH, env_id=env_id, memory_mode=args.memory_mode
        )
    else:
        import onnxruntime

        providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
        enc_net = onnxruntime.InferenceSession(WEIGHT_ENC_PATH, providers=providers)
        dec_net = onnxruntime.InferenceSession(WEIGHT_DEC_PATH, providers=providers)

    if args.disable_ailia_tokenizer:
        from transformers import WhisperTokenizerFast
        tokenizer = WhisperTokenizerFast.from_pretrained("tokenizer")
    else:
        from ailia_tokenizer import WhisperTokenizer
        tokenizer = WhisperTokenizer.from_pretrained("./tokenizer/")

    models = {
        "enc": enc_net,
        "dec": dec_net,
        "tokenizer": tokenizer,
    }

    recognize_from_audio(models)


if __name__ == "__main__":
    main()
