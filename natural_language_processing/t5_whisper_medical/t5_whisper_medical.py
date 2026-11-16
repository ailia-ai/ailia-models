import logging
import sys
import time

import numpy as np
from tqdm import trange
sys.path.append('../../util')
from arg_utils import get_base_parser, update_parser, get_savepath  # noqa: E402
from model_utils import check_and_download_models  # noqa

# logger
logger = logging.getLogger(__name__)

"""
params
"""
ENCODER_ONNX_PATH = "t5_whisper_medical-encoder.obf.onnx"
ENCODER_PROTOTXT_PATH = "t5_whisper_medical-encoder.onnx.prototxt"
DECODER_ONNX_PATH = "t5_whisper_medical-decoder-with-lm-head.obf.onnx"
DECODER_PROTOTXT_PATH = "t5_whisper_medical-decoder-with-lm-head.onnx.prototxt"
REMOTE_PATH = "https://storage.googleapis.com/ailia-models/t5_whisper_medical/"

MAX_SOURCE_LENGTH = 384
DEFAULT_INPUT_PATH = "./input.txt"
DEFAULT_OUTPUT_PATH = "./output.txt"


"""
parse args
"""
parser = get_base_parser(
    description="T5 whisper medical error correction",
    default_input=DEFAULT_INPUT_PATH,
    default_save=DEFAULT_OUTPUT_PATH,
    input_ftype="text",
)
parser.add_argument(
    '-o', '--onnx', action='store_true',
    help="Option to use onnxrutime to run or not."
)
parser.add_argument(
    '--disable_ailia_tokenizer',
    action='store_true',
    help='disable ailia tokenizer.'
)
args = update_parser(parser)


def softmax_np(x: np.ndarray):
    e_x = np.exp(x - np.max(x))  # subtract max to stabilize
    return e_x / e_x.sum(axis=0)

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
        token = np.expand_dims(np.array(self.tokenizer(prompt)['input_ids'])[:max_context_length - 1], axis=0)

        # encode tokens
        if args.onnx:
            encoder_outputs_prompt = self.encoder.run(None, {"input_ids": token})[0]
        else:
            encoder_outputs_prompt = self.encoder.run({"input_ids": token})[0]

        # reset token
        token = np.zeros((1,1), dtype=np.int64)
        for _ in trange(max_length):
            # decode tokens
            if args.onnx:
                outputs = np.array(
                    self.decoder_with_lm_head.run(
                        None,
                        {"input_ids": token,"encoder_hidden_states": encoder_outputs_prompt},
                    )[0][0]
                )
            else:
                outputs = np.array(
                    self.decoder_with_lm_head.run(
                        {"input_ids": token,"encoder_hidden_states": encoder_outputs_prompt},
                    )[0][0]
                )
            next_token_logits = outputs[-1, :] / (temperature if temperature > 0 else 1.0)

            # `1` means end of sentence. (EOS token)
            if int(next_token_logits.argmax()) == 1:
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
        return self.tokenizer.decode(new_tokens), new_logits

"""
pre process functions
"""
def preprocess_body(text: str) -> str:
    prefix = "医療用語の訂正: "
    return prefix + text

def main(args):
    # download onnx and prototxt
    check_and_download_models(ENCODER_ONNX_PATH, ENCODER_PROTOTXT_PATH, REMOTE_PATH)
    check_and_download_models(DECODER_ONNX_PATH, DECODER_PROTOTXT_PATH, REMOTE_PATH)

    # load model
    if args.disable_ailia_tokenizer:
        from transformers import T5Tokenizer
        tokenizer = T5Tokenizer.from_pretrained("./tokenizer/", is_fast=False)
    else:
        from ailia_tokenizer import T5Tokenizer
        tokenizer = T5Tokenizer.from_pretrained("./tokenizer/")

    if args.onnx:
        from onnxruntime import InferenceSession
        encoder_sess = InferenceSession(ENCODER_ONNX_PATH)
        decoder_sess = InferenceSession(DECODER_ONNX_PATH)
        model = T5Model(encoder_sess, decoder_sess, tokenizer)
    else:
        import ailia
        encoder_sess = ailia.Net(ENCODER_PROTOTXT_PATH, ENCODER_ONNX_PATH, env_id=args.env_id)
        decoder_sess = ailia.Net(DECODER_PROTOTXT_PATH, DECODER_ONNX_PATH, env_id=args.env_id)
        model = T5Model(encoder_sess, decoder_sess, tokenizer)

    if args.benchmark:
        logger.info('BENCHMARK mode')
        with open(args.input[0], "r", encoding="utf-8") as fi:
            body = fi.read()
        body_preprocessed = preprocess_body(body)
        for c in range(5):
            start = int(round(time.time() * 1000))
            _, _ = model.estimate(body_preprocessed, 384, temperature=0.0, top_k=50, top_p=0)
            end = int(round(time.time() * 1000))
            logger.info("\tailia processing time {} ms".format(end-start))
    else:
        for input_path in args.input:
            # load input file
            with open(input_path, "r", encoding="utf-8") as fi:
                body = fi.read()

            # pre process
            body_preprocessed = preprocess_body(body)

            # execute prediction
            most_plausible_title, _ = model.estimate(body_preprocessed, 384, temperature=0.0, top_k=50, top_p=0)
            logger.info("%s", most_plausible_title)
            save_path = get_savepath(args.savepath, input_path)
            with open(save_path, "w", encoding="utf-8") as fo:
                fo.write(most_plausible_title + "\n")


if __name__ == '__main__':
    main(args)
