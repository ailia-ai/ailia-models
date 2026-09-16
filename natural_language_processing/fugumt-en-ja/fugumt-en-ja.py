import sys
import time
from logging import getLogger

import numpy as np
from scipy.special import log_softmax

import ailia

# import original modules
sys.path.append('../../util')
from arg_utils import get_base_parser, update_parser, get_savepath  # noqa
from model_utils import check_and_download_models  # noqa

from beam_search import BeamSearchScorer

logger = getLogger(__name__)

# ======================
# Parameters
# ======================

WEIGHT_PATH = 'seq2seq-lm-with-past.onnx'
MODEL_PATH = 'seq2seq-lm-with-past.onnx.prototxt'
REMOTE_PATH = 'https://storage.googleapis.com/ailia-models/fugumt/'

# ======================
# Arguemnt Parser Config
# ======================

parser = get_base_parser(
    'FuguMT', None, None
)
parser.add_argument(
    "-i", "--input", metavar="TEXT", type=str,
    default="This is a cat.",
    help="Input text."
)
parser.add_argument(
    '--disable_ailia_tokenizer',
    action='store_true',
    help='disable ailia tokenizer.'
)
parser.add_argument(
    '--onnx',
    action='store_true',
    help='execute onnxruntime version.'
)
args = update_parser(parser, check_input_type=False)


# ======================
# Import tokenizer
# ======================

if args.disable_ailia_tokenizer:
    from transformers import MarianTokenizer
else:
    from ailia_tokenizer import MarianTokenizer


# ======================
# Secondaty Functions
# ======================

def logits_processor(input_ids, scores):
    max_length = 512
    eos_token_id = [0]

    cur_len = input_ids.shape[-1]
    if cur_len == max_length - 1:
        num_tokens = scores.shape[1]
        scores[:, [i for i in range(num_tokens) if i not in eos_token_id]] = -float("inf")
        for i in eos_token_id:
            scores[:, i] = 0

    return scores


def reorder_cache(past, beam_idx):
    reordered_past = []
    for i in range(6):
        # cached cross_attention states don't have to be reordered -> they are always the same
        layer_past = past[i * 4:(i + 1) * 4]
        reordered_past.extend([np.take(past_state, beam_idx, axis=0) for past_state in layer_past[:2]])
        reordered_past.extend(layer_past[2:])
    return reordered_past


# ======================
# Main functions
# ======================

def forward(net, input_ids, attention_mask, decoder_input_ids, past_key_values):
    if not args.onnx:
        output = net.predict([
            input_ids, attention_mask, decoder_input_ids,
            past_key_values[0], past_key_values[1], past_key_values[2], past_key_values[3],
            past_key_values[4], past_key_values[5], past_key_values[6], past_key_values[7],
            past_key_values[8], past_key_values[9], past_key_values[10], past_key_values[11],
            past_key_values[12], past_key_values[13], past_key_values[14], past_key_values[15],
            past_key_values[16], past_key_values[17], past_key_values[18], past_key_values[19],
            past_key_values[20], past_key_values[21], past_key_values[22], past_key_values[23],
        ])
    else:
        output = net.run(None, {
            'input_ids': input_ids, 'attention_mask': attention_mask,
            'decoder_input_ids': decoder_input_ids,
            'past_key_values.0.decoder.key': past_key_values[0],
            'past_key_values.0.decoder.value': past_key_values[1],
            'past_key_values.0.encoder.key': past_key_values[2],
            'past_key_values.0.encoder.value': past_key_values[3],
            'past_key_values.1.decoder.key': past_key_values[4],
            'past_key_values.1.decoder.value': past_key_values[5],
            'past_key_values.1.encoder.key': past_key_values[6],
            'past_key_values.1.encoder.value': past_key_values[7],
            'past_key_values.2.decoder.key': past_key_values[8],
            'past_key_values.2.decoder.value': past_key_values[9],
            'past_key_values.2.encoder.key': past_key_values[10],
            'past_key_values.2.encoder.value': past_key_values[11],
            'past_key_values.3.decoder.key': past_key_values[12],
            'past_key_values.3.decoder.value': past_key_values[13],
            'past_key_values.3.encoder.key': past_key_values[14],
            'past_key_values.3.encoder.value': past_key_values[15],
            'past_key_values.4.decoder.key': past_key_values[16],
            'past_key_values.4.decoder.value': past_key_values[17],
            'past_key_values.4.encoder.key': past_key_values[18],
            'past_key_values.4.encoder.value': past_key_values[19],
            'past_key_values.5.decoder.key': past_key_values[20],
            'past_key_values.5.decoder.value': past_key_values[21],
            'past_key_values.5.encoder.key': past_key_values[22],
            'past_key_values.5.encoder.value': past_key_values[23],
        })
    logits, *past_key_values, _ = output

    return logits, past_key_values


def beam_search(net, model_inputs):
    batch_size = 1
    num_beams = 12
    max_length = 512

    decoder_start_token_id = pad_token_id = 32000
    eos_token_id = 0

    # prepare beam search scorer
    length_penalty = 1.0
    early_stopping = False
    num_return_sequences = 1
    beam_scorer = BeamSearchScorer(
        batch_size=1,
        num_beams=num_beams,
        length_penalty=length_penalty,
        do_early_stopping=early_stopping,
        num_beam_hyps_to_keep=num_return_sequences,
    )

    input_ids = np.array([model_inputs["input_ids"]] * num_beams, dtype=int)
    attention_mask = np.array([model_inputs["attention_mask"]] * num_beams, dtype=int)
    decoder_input_ids = np.ones((num_beams, 1), dtype=int) * decoder_start_token_id
    past_key_values = [np.zeros((num_beams, 8, 0, 64), dtype=np.float32)] * 24

    batch_beam_size, cur_len = decoder_input_ids.shape

    # initialise score of first beam with 0 and the rest with -1e9. This makes sure that only tokens
    # of the first beam are considered to avoid sampling the exact same tokens across all beams.
    beam_scores = np.zeros((batch_size, num_beams))
    beam_scores[:, 1:] = -1e9
    beam_scores = beam_scores.reshape((batch_size * num_beams))

    while True:
        logits, past_key_values = forward(
            net, input_ids, attention_mask, decoder_input_ids[:, -1:], past_key_values
        )

        next_token_logits = logits[:, -1, :]
        next_token_logits[:, pad_token_id] = float("-inf")
        next_token_scores = log_softmax(next_token_logits, axis=-1)
        next_token_scores_processed = logits_processor(decoder_input_ids, next_token_scores)

        next_token_scores = \
            next_token_scores_processed \
            + np.broadcast_to(beam_scores[:, None], next_token_scores.shape)

        # reshape for beam search
        vocab_size = next_token_scores.shape[-1]
        next_token_scores = next_token_scores.reshape(batch_size, num_beams * vocab_size)

        # Sample 2 next tokens for each beam (so we have some spare tokens and match output of beam search)
        next_tokens = np.argsort(-next_token_scores, axis=1)[:, :2 * num_beams]
        next_token_scores = np.stack([next_token_scores[i, next_tokens[i]] for i in range(len(next_tokens))])

        next_indices = next_tokens.astype(int) // vocab_size
        next_tokens = next_tokens % vocab_size

        # stateless
        beam_outputs = beam_scorer.process(
            decoder_input_ids,
            next_token_scores,
            next_tokens,
            next_indices,
            pad_token_id=pad_token_id,
            eos_token_id=eos_token_id,
            beam_indices=None,
        )
        beam_scores = beam_outputs["next_beam_scores"]
        beam_next_tokens = beam_outputs["next_beam_tokens"]
        beam_idx = beam_outputs["next_beam_indices"]

        decoder_input_ids = np.concatenate([
            decoder_input_ids[beam_idx, :], np.expand_dims(beam_next_tokens, axis=-1)
        ], axis=-1)

        past_key_values = reorder_cache(past_key_values, beam_idx)

        cur_len = cur_len + 1

        if beam_scorer.is_done or decoder_input_ids.shape[-1] > max_length:
            break

    sequence_outputs = beam_scorer.finalize(
        input_ids,
        beam_scores,
        next_tokens,
        next_indices,
        pad_token_id=pad_token_id,
        eos_token_id=eos_token_id,
        max_length=max_length,
        beam_indices=None,
    )

    return sequence_outputs["sequences"][0]


def predict(mod, input_text):
    tokenizer = mod["tokenizer"]
    net = mod["net"]

    model_inputs = tokenizer(input_text)
    output_ids = beam_search(net, model_inputs)

    translation_text = tokenizer.decode(
        output_ids,
        skip_special_tokens=True,
        #clean_up_tokenization_spaces=False,
    )

    return translation_text


def recognize_from_text(mod):
    input_text = args.input

    logger.info("input_text: %s" % input_text)

    # inference
    logger.info('Start inference...')
    if args.benchmark:
        logger.info('BENCHMARK mode')
        total_time_estimation = 0
        for i in range(args.benchmark_count):
            start = int(round(time.time() * 1000))
            output = predict(mod, input_text)
            end = int(round(time.time() * 1000))
            estimation_time = (end - start)

            # Logging
            logger.info(f'\tailia processing estimation time {estimation_time} ms')
            if i != 0:
                total_time_estimation = total_time_estimation + estimation_time

        logger.info(f'\taverage time estimation {total_time_estimation / (args.benchmark_count - 1)} ms')
    else:
        output = predict(mod, input_text)

    logger.info(f"translation_text: {output}")

    logger.info('Script finished successfully.')


def main():
    # model files check and download
    check_and_download_models(WEIGHT_PATH, MODEL_PATH, REMOTE_PATH)

    if args.disable_ailia_tokenizer:
        tokenizer = MarianTokenizer.from_pretrained("tokenizer")
    else:
        tokenizer = MarianTokenizer.from_pretrained("./tokenizer/")

    env_id = args.env_id

    # initialize
    if not args.onnx:
        net = ailia.Net(MODEL_PATH, WEIGHT_PATH, env_id=env_id, memory_mode=args.memory_mode)
    else:
        import onnxruntime
        cuda = 0 < ailia.get_gpu_environment_id()
        providers = ['CUDAExecutionProvider', 'CPUExecutionProvider'] if cuda else ['CPUExecutionProvider']
        net = onnxruntime.InferenceSession(WEIGHT_PATH, providers=providers)

    mod = {
        "tokenizer": tokenizer,
        "net": net,
    }

    recognize_from_text(mod)


if __name__ == '__main__':
    main()
