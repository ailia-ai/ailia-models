import sys
import time
from logging import getLogger

import numpy as np

import ailia

# import original modules
sys.path.append('../../util')
from arg_utils import get_base_parser, update_parser, get_savepath  # noqa
from model_utils import check_and_download_models  # noqa

logger = getLogger(__name__)

# ======================
# Parameters
# ======================

ENCODER_WEIGHT_PATH = 't5_base_japanese_summarization_enc.onnx'
ENCODER_MODEL_PATH = 't5_base_japanese_summarization_enc.onnx.prototxt'

DECODER_WEIGHT_PATH = 't5_base_japanese_summarization_dec.onnx'
DECODER_MODEL_PATH = 't5_base_japanese_summarization_dec.onnx.prototxt'

REMOTE_PATH = "https://storage.googleapis.com/ailia-models/t5_base_japanese_summarization/"

# ======================
# Arguemnt Parser Config
# ======================

parser = get_base_parser(
    't5_base_japanese_summarization', None, None, fp16_support=False
)

parser.add_argument(
    "-f", "--file", metavar="PATH", type=str,
    default="input.txt",
    help="Input file path."
)

parser.add_argument(
    "-i", "--input", metavar="TEXT", type=str,
    default=None,
    help="Input text."
)

parser.add_argument(
    "-sc", "--score", action = 'store_true'
)

parser.add_argument(
    '-o', '--onnx', action='store_true',
    help="Option to use onnxrutime to run or not."
)

parser.add_argument(
    '--seed', type=int,
    help='random seed'
)

parser.add_argument(
    '--disable_ailia_tokenizer',
    action='store_true',
    help='disable ailia tokenizer.'
)

args = update_parser(parser, check_input_type=False)

if args.seed:
    np.random.seed(args.seed)

# ======================
# Helper functions
# ======================

def handle_subwords(token):
    r"""
    Description:
        Get rid of subwords '##'.
    About tokenizer subwords:
        See: https://huggingface.co/docs/transformers/tokenizer_summary
    """
    if len(token) > 2 and token[0:2] == '##':
        token = token[2:]
    return token

def softmax_np(x: np.ndarray):
    e_x = np.exp(x - np.max(x))  # subtract max to stabilize
    return e_x / e_x.sum(axis=0)

def search(query : list, key : list):
    """search key in query using boyer-moore algorithm and return its index"""
    # create skip table
    skip = {}
    for i, k in enumerate(key):
        skip[k] = len(key) - i - 1

    # search
    i = len(key) - 1
    while i < len(query):
        j = len(key) - 1
        while query[i] == key[j]:
            if j == 0:
                return i
            i -= 1
            j -= 1
        i += skip.get(query[i], len(key))
    return -1

def check_repeat_ngram(tokens, new_token, detect_size):
    """
    Description:
        Detect repeating ngrams longer than detect_size.
        This function should be called upon every new token generation.
    Args:
        tokens (list): List of tokens generated so far.
        new_token (int): Newly generated token.
        detect_size (int): Size of ngram to detect.
    """
    rep_idx = search(tokens, tokens[-detect_size+1:]+new_token)
    return rep_idx

"""
top k top p filtering algorithm
Modified by Takumi Ibayashi.
"""
def top_k_top_p_filtering(logits: np.ndarray, top_k:int=0, top_p:float=0.0, filter_value:float=-float("Inf")):
    """
    Filter a distribution of logits using top-k and/or nucleus (top-p) filtering
    Args:
        logits: logits distribution shape (vocabulary size)
        top_k > 0: keep only top k tokens with highest probability (top-k filtering).
        top_p > 0.0: keep the top tokens with cumulative probability >= top_p (nucleus filtering).
            Nucleus filtering is described in Holtzman et al. (http://arxiv.org/abs/1904.09751)
    From: https://gist.github.com/thomwolf/1a5a29f6962089e871b94cbd09daf317
    """
    assert (
        logits.ndim == 1
    )  # batch size 1 for now - could be updated for more but the code would be less clear
    top_k = min(top_k, logits.shape[-1])  # Safety check
    if top_k > 0:
        # Remove all tokens with a probability less than the last token of the top-k
        indices_to_remove = logits < np.partition(logits, -top_k)[-top_k]
        logits[indices_to_remove] = filter_value

    if top_p > 0.0:
        sorted_indices = np.argsort(logits)[::-1]
        sorted_logits = logits[sorted_indices]
        cumulative_probs = np.cumsum(np.exp(sorted_logits) / np.sum(np.exp(sorted_logits)))

        # Remove tokens with cumulative probability above the threshold
        sorted_indices_to_remove = cumulative_probs > top_p
        # Shift the indices to the right to keep also the first token above the threshold
        sorted_indices_to_remove[1:] = sorted_indices_to_remove[:-1]
        sorted_indices_to_remove[0] = 0

        indices_to_remove = sorted_indices[sorted_indices_to_remove]
        logits[indices_to_remove] = filter_value
    return logits

"""
model wrapper
"""
class T5Model:
    """
    This class is based on `GenerativeT5` from `models` in `onnxt5`.
    Modified by Takumi Ibayashi.
    """
    def __init__(self, encoder, decoder_with_lm_head, tokenizer):
        super().__init__()
        self.encoder = encoder
        self.decoder_with_lm_head = decoder_with_lm_head
        self.tokenizer = tokenizer

    def estimate(
        self, prompt: str, max_length: int, temperature:float=1.0, repetition_penalty:float=1.0, top_k:int=50, top_p:int=0, max_context_length: int=512, 
    ):
        """
        Generate a text output given a prompt using the model.

        Args:
            prompt (str): The initial text input to the model, which it uses as a 
                starting point to generate the subsequent text.
            max_length (int): The maximum length of the text to be generated.
            temperature (float, optional): This controls the randomness in the model's 
                text generation. A higher temperature value results in more random output. 
                If the temperature is very small, it will approach greedy decoding. 
                Defaults to 1.0.
            top_k (int, optional): parameter for top k filtering algorithm
            top_p (int, optional): parameter for top p filtering algorithm
            repetition_penalty (float, optional): This increases the model's likelihood 
                to generate diverse output by discouraging it from repeating the same 
                token. Defaults to 1.0.
            max_context_length (int, optional): The maximum length of the context to be 
                used in generation. Defaults to 512.
        """
        new_tokens = np.array([], dtype=np.float32)
        new_logits = []

        # generate tokens with tokenizer
        enc = self.tokenizer.encode_plus(#encode tokens
            text=prompt,
            max_length=512,
            truncation=True,
        )

        enc_input =(
            np.array(enc['input_ids'])[None,:],#prepare input
            np.array(enc['attention_mask'])[None,:],
        )

        # encode tokens
        encoder_outputs_prompt = self.encoder.run(enc_input)[0]

        # reset token
        token = np.zeros((1,1), dtype=np.int64)
        for _ in range(max_length):
            # decode tokens
            outputs = np.array(
                self.decoder_with_lm_head.run(
                    (token, encoder_outputs_prompt, np.ones_like(token)),
                )[0][0]
            )
            next_token_logits = outputs[-1, :] / (temperature if temperature > 0 else 1.0)

            # `1` means end of sentence. (EOS token)
            if int(next_token_logits.argmax()) in (1,0):
                break

            new_logits.append(next_token_logits)
            for _ in set(token.reshape(-1).tolist()):
                next_token_logits[_] /= repetition_penalty

            # select next token
            if temperature == 0:
                # greedy sampling: this methods always choose the most probable token.
                next_token = np.expand_dims(np.argmax(next_token_logits), axis=0)
            else:
                # Top-k and top-p filtering: this methods enhance text diversity and creativity.
                filtered_logits = top_k_top_p_filtering(next_token_logits, top_k=top_k, top_p=top_p)
                probs = softmax_np(filtered_logits)
                next_token = np.expand_dims(np.random.choice(np.arange(len(probs)), p=probs), axis=0)
            token = np.concatenate((token, np.expand_dims(next_token, axis=0)), axis=1)
            new_tokens = np.concatenate((new_tokens, next_token), axis=0)
        return self.tokenizer.decode(new_tokens.astype('int'), skip_special_tokens = True), new_logits

# ======================
# Main functions
# ======================


def summarize(model):
    input_text = args.input
    input_path = args.file
    if input_text is None:
        input_text = open(input_path, "r", encoding="utf-8").read()

    logger.info("input_text: %s" % input_text)

    # inference
    logger.info('inference has started...')
    if args.benchmark:
        logger.info('BENCHMARK mode')
        total_time_estimation = 0
        for i in range(args.benchmark_count):
            start = int(round(time.time() * 1000))

            out, _ = model.estimate(input_text, max_length = 512, top_p = 0.93, repetition_penalty=1.5)

            end = int(round(time.time() * 1000))
            estimation_time = (end - start)

            # Logging
            logger.info(f'\tailia processing estimation time {estimation_time} ms')
            if i != 0:
                total_time_estimation = total_time_estimation + estimation_time

        logger.info(f'\taverage time estimation {total_time_estimation / (args.benchmark_count - 1)} ms')
    else:
        #prediction = predict(model, input_text)
        out, _ = model.estimate(input_text, max_length = 512, top_p = 0.93, repetition_penalty=1.5)
        logger.info('summarization of input text:')
        logger.info(f'{out}')

        # save output        
        if args.savepath is not None:
            save_path = get_savepath(args.savepath, input_path)
            with open(save_path, "w", encoding="utf-8") as fo:
                fo.write(out)
        
    

    logger.info('Script finished successfully.')


def main():
    # model files check and download
    check_and_download_models(ENCODER_WEIGHT_PATH, ENCODER_MODEL_PATH, REMOTE_PATH)
    check_and_download_models(DECODER_WEIGHT_PATH, DECODER_MODEL_PATH, REMOTE_PATH)

    if args.disable_ailia_tokenizer:
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained("sonoisa/t5-base-japanese", is_fast=True)
    else:
        from ailia_tokenizer import T5Tokenizer
        tokenizer = T5Tokenizer.from_pretrained("./tokenizer/")

    env_id = args.env_id

    # initialize
    encoder = ailia.Net(ENCODER_MODEL_PATH, ENCODER_WEIGHT_PATH, env_id = env_id, memory_mode = args.memory_mode)
    decoder = ailia.Net(DECODER_MODEL_PATH, DECODER_WEIGHT_PATH, env_id = env_id, memory_mode = args.memory_mode)

    model = T5Model(encoder, decoder, tokenizer)

    summarize(model)

if __name__ == '__main__':
    main()
