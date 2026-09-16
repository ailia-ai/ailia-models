import sys
import time
from typing import List
from logging import getLogger

import numpy as np
import soundfile as sf

# import original modules
sys.path.append('../../util')
from arg_utils import get_base_parser, update_parser  # noqa
from model_utils import check_and_download_models, check_and_download_file  # noqa
from detector_utils import load_image  # noqa

import ailia

from audio_utils import mel_filter_bank, window_function, spectrogram
from tokenizer_utils import decode_asr

logger = getLogger(__name__)

# ======================
# Parameters
# ======================

WEIGHT_ENC_PATH = 'distil-large-v2_encoder.onnx'
WEIGHT_ENC_PB_PATH = 'distil-large-v2_encoder_weights.pb'
MODEL_ENC_PATH = 'distil-large-v2_encoder.onnx.prototxt'
WEIGHT_DEC_PATH = 'distil-large-v2_decoder.onnx'
MODEL_DEC_PATH = 'distil-large-v2_decoder.onnx.prototxt'
REMOTE_PATH = 'https://storage.googleapis.com/ailia-models/distil-whisper/'

WAV_PATH = 'demo.wav'
SAVE_TEXT_PATH = 'output.txt'

SAMPLE_RATE = 16000

# ======================
# Arguemnt Parser Config
# ======================

parser = get_base_parser(
    'Distil-Whisper', WAV_PATH, SAVE_TEXT_PATH, input_ftype='audio'
)
parser.add_argument(
    '--chunk_length', type=int, default=None,
    help='the chunk size for chunking.'
)
parser.add_argument(
    '--onnx',
    action='store_true',
    help='execute onnxruntime version.'
)
parser.add_argument(
    '--disable_ailia_tokenizer',
    action='store_true',
    help='disable ailia tokenizer.'
)
args = update_parser(parser, check_input_type=False)


# ======================
# Secondaty Functions
# ======================

def feature_extractor(raw_speech):
    raw_speech = np.asarray([raw_speech]).T

    max_length = 480000
    raw_speech = raw_speech[:max_length]

    if len(raw_speech) < max_length:
        difference = max_length - len(raw_speech)
        raw_speech = np.pad(
            raw_speech, ((0, difference), (0, 0)), "constant",
        )

    raw_speech = np.expand_dims(raw_speech, axis=0)
    raw_speech = raw_speech.transpose(2, 0, 1)

    n_fft = 400
    hop_length = 160
    feature_size = 80

    if not hasattr(preprocess, 'mel_filters'):
        preprocess.mel_filters = \
            mel_filter_bank(
                num_frequency_bins=1 + n_fft // 2,
                num_mel_filters=feature_size,
                min_frequency=0.0,
                max_frequency=8000.0,
                sampling_rate=SAMPLE_RATE,
                norm="slaney",
                mel_scale="slaney",
            )

    features = []
    for i, waveform in enumerate(raw_speech[0]):
        log_spec = spectrogram(
            waveform,
            window_function(n_fft, "hann"),
            frame_length=n_fft,
            hop_length=hop_length,
            power=2.0,
            mel_filters=preprocess.mel_filters,
        )
        log_spec = log_spec[:, :-1]
        log_spec = np.maximum(log_spec, log_spec.max() - 8.0)
        log_spec = (log_spec + 4.0) / 4.0

        features.append(log_spec)

    features = np.array(features, dtype=np.float32)

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
        is_last = chunk_end_idx > inputs_len if stride_right > 0 else chunk_end_idx >= inputs_len
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

        for item in chunk_iter(
                inputs, chunk_len, stride_left, stride_right,
        ):
            yield item
    else:
        features = feature_extractor(inputs)
        yield {"input_features": features}


def decode(
        net,
        encoder_hidden_states: np.ndarray,
        input_ids: np.ndarray,
        past_key_values: List[np.ndarray]):
    if args.benchmark:
        start = int(round(time.time() * 1000))

    if not args.onnx:
        decoder_output = net.predict([
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
        ])
    else:
        decoder_output = net.run(
            None, {
                'input_ids': input_ids,
                'encoder_hidden_states': encoder_hidden_states,
                'past_key_values.0.decoder.key': past_key_values[0],
                'past_key_values.0.decoder.value': past_key_values[1],
                'past_key_values.0.encoder.key': past_key_values[2],
                'past_key_values.0.encoder.value': past_key_values[3],
                'past_key_values.1.decoder.key': past_key_values[4],
                'past_key_values.1.decoder.value': past_key_values[5],
                'past_key_values.1.encoder.key': past_key_values[6],
                'past_key_values.1.encoder.value': past_key_values[7],
            }
        )

    if args.benchmark:
        end = int(round(time.time() * 1000))
        estimation_time = (end - start)
        logger.info(f'\tdecoder processing time {estimation_time} ms')

    logits, new_past_key_values = decoder_output[0], decoder_output[1:]

    return logits, new_past_key_values


def stopping_criteria(
        input_ids: np.array) -> bool:
    max_length = 129

    cur_len = input_ids.shape[-1]
    is_done = cur_len >= max_length
    return is_done


def greedy_search(net, encoder_hidden_states):
    decoder_start_token_id = 50258
    eos_token_id = np.array([50257])
    pad_token_id = 50257

    suppress_tokens = [
        1, 2, 7, 8, 9, 10, 14, 25, 26, 27, 28, 29, 31, 58, 59, 60, 61, 62, 63, 90, 91, 92, 93,
        359, 503, 522, 542, 873, 893, 902, 918, 922, 931, 1350, 1853, 1982, 2460, 2627,
        3246, 3253, 3268, 3536, 3846, 3961, 4183, 4667, 6585, 6647, 7273, 9061, 9383,
        10428, 10929, 11938, 12033, 12331, 12562, 13793, 14157, 14635, 15265, 15618, 16553,
        16604, 18362, 18956, 20075, 21675, 22520, 26130, 26161, 26435, 28279, 29464, 31650,
        32302, 32470, 36865, 42863, 47425, 49870, 50254, 50258, 50360, 50361, 50362
    ]
    begin_index = 4
    begin_suppress_tokens = [220, 50257]
    force_token_map = {1: 50259, 2: 50359, 3: 50363}

    shape = encoder_hidden_states.shape[:-1]
    batch_size = shape[0]

    # input_ids = np.ones(shape, dtype=int) * -100
    input_ids = np.ones((batch_size, 1), dtype=int) * decoder_start_token_id
    past_key_values = [np.zeros((batch_size, 20, 0, 64), dtype=np.float16)] * 8

    # keep track of which sequences are already finished
    unfinished_sequences = np.ones(input_ids.shape[0], dtype=int)

    this_peer_finished = False  # used by synced_gpus only
    while True:
        logits, past_key_values = decode(
            net, encoder_hidden_states, input_ids[:, -1:], past_key_values
        )
        next_tokens_scores = logits[:, -1, :]

        # SuppressTokensLogitsProcessor
        next_tokens_scores[:, suppress_tokens] = -float("inf")
        # SuppressTokensAtBeginLogitsProcessor
        if input_ids.shape[1] == begin_index:
            next_tokens_scores[:, begin_suppress_tokens] = -float("inf")
        # ForceTokensLogitsProcessor
        generation_idx = input_ids.shape[-1]
        current_token = force_token_map.get(generation_idx, None)
        if current_token is not None:
            next_tokens_scores[:, :] = -float("inf")
            next_tokens_scores[:, current_token] = 0

        # argmax
        next_tokens = np.argmax(next_tokens_scores, axis=-1)

        # finished sentences should have their next token be a padding token
        next_tokens = next_tokens * unfinished_sequences + pad_token_id * (1 - unfinished_sequences)

        # update generated ids, model inputs, and length for next step
        input_ids = np.concatenate([input_ids, next_tokens[:, None]], axis=-1)

        # if eos_token was found in one sentence, set sentence to finished
        unfinished_sequences = unfinished_sequences * np.prod(
            np.tile(next_tokens, (eos_token_id.shape[0], 1)) != eos_token_id[:, None],
            axis=0
        )

        # stop when each sentence is finished
        if np.max(unfinished_sequences) == 0:
            this_peer_finished = True

        # stop if we exceed the maximum length
        if stopping_criteria(input_ids):
            this_peer_finished = True

        if this_peer_finished:
            break

    return input_ids


def predict(models, wav, chunk_length_s=0):
    processed = preprocess(wav, chunk_length_s)

    model_outputs = []
    for item in processed:
        if args.benchmark:
            start = int(round(time.time() * 1000))

        input_features = item.pop("input_features")

        net = models['enc']
        if not args.onnx:
            output = net.run(input_features)
        else:
            output = net.run(None, {'input_features': input_features})
        last_hidden_state = output[0]
        last_hidden_state = last_hidden_state.astype(np.float16)

        if args.benchmark:
            end = int(round(time.time() * 1000))
            estimation_time = (end - start)
            logger.info(f'\tencoder processing time {estimation_time} ms')

        net = models['dec']
        tokens = greedy_search(net, last_hidden_state)

        item["tokens"] = tokens

        if "stride" in item:
            chunk_len, stride_left, stride_right = item["stride"]
            # Go back in seconds
            chunk_len /= SAMPLE_RATE
            stride_left /= SAMPLE_RATE
            stride_right /= SAMPLE_RATE
            item["stride"] = chunk_len, stride_left, stride_right

        model_outputs.append(item)

    tokenizer = models['tokenizer']
    time_precision = 0.02

    if args.disable_ailia_tokenizer:
        text, optional = tokenizer._decode_asr(
            model_outputs,
            return_timestamps=None,
            return_language=None,
            time_precision=time_precision)
    else:
        text, optional = decode_asr(
            tokenizer,
            model_outputs,
            return_timestamps=None,
            return_language=None,
            time_precision=time_precision)
    text = text.strip()

    return text


def recognize_from_audio(models):
    chunk_length_s = args.chunk_length

    # input image loop
    for audio_path in args.input:
        logger.info(audio_path)

        # prepare input data
        wav, _ = sf.read(audio_path)

        # inference
        logger.info('Start inference...')

        if args.benchmark:
            start = int(round(time.time() * 1000))

        text = predict(models, wav, chunk_length_s=chunk_length_s)

        if args.benchmark:
            end = int(round(time.time() * 1000))
            estimation_time = (end - start)
            logger.info(f'\ttotal processing time {estimation_time} ms')

    logger.info(text)

    logger.info('Script finished successfully.')


def main():
    # model files check and download
    check_and_download_models(WEIGHT_ENC_PATH, MODEL_ENC_PATH, REMOTE_PATH)
    check_and_download_models(WEIGHT_DEC_PATH, MODEL_DEC_PATH, REMOTE_PATH)
    check_and_download_file(WEIGHT_ENC_PB_PATH, REMOTE_PATH)

    env_id = args.env_id

    # initialize
    if not args.onnx:
        enc_net = ailia.Net(MODEL_ENC_PATH, WEIGHT_ENC_PATH, env_id=env_id, memory_mode=args.memory_mode)
        dec_net = ailia.Net(MODEL_DEC_PATH, WEIGHT_DEC_PATH, env_id=env_id, memory_mode=args.memory_mode)
    else:
        import onnxruntime
        cuda = 0 < ailia.get_gpu_environment_id()
        providers = ['CUDAExecutionProvider', 'CPUExecutionProvider'] if cuda else ['CPUExecutionProvider']
        enc_net = onnxruntime.InferenceSession(WEIGHT_ENC_PATH, providers=providers)
        dec_net = onnxruntime.InferenceSession(WEIGHT_DEC_PATH, providers=providers)

    if args.disable_ailia_tokenizer:
        from transformers import WhisperTokenizer
        tokenizer = WhisperTokenizer.from_pretrained("tokenizer")
    else:
        from ailia_tokenizer import WhisperTokenizer
        tokenizer = WhisperTokenizer.from_pretrained("./tokenizer/")

    models = {
        'enc': enc_net,
        'dec': dec_net,
        'tokenizer': tokenizer,
    }

    recognize_from_audio(models)


if __name__ == '__main__':
    main()
