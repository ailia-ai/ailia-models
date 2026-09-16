import sys
import time
import re
from functools import partial
from logging import getLogger

import numpy as np

import ailia

# import original modules
sys.path.append("../../util")
from arg_utils import get_base_parser, update_parser  # noqa
from model_utils import check_and_download_models  # noqa

from beam_searcher import S2SBeamSearcher

logger = getLogger(__name__)


# ======================
# Parameters
# ======================

WEIGHT_PATH = "soundchoice-g2p_atn.onnx"
MODEL_PATH = "soundchoice-g2p_atn.onnx.prototxt"
WEIGHT_EMB_PATH = "soundchoice-g2p_emb.onnx"
MODEL_EMB_PATH = "soundchoice-g2p_emb.onnx.prototxt"
WEIGHT_BEAM_PATH = "rnn_beam_searcher.onnx"
MODEL_BEAM_PATH = "rnn_beam_searcher.onnx.prototxt"
REMOTE_PATH = "https://storage.googleapis.com/ailia-models/soundchoice-g2p/"

lab2ind = {
    # fmt: off
    '<bos>': 0, '<eos>': 1, '<unk>': 2, 'A': 3, 'B': 4, 'C': 5, 'D': 6, 'E': 7, 'F': 8, 'G': 9, 'H': 10, 'I': 11, 'J': 12, 'K': 13, 'L': 14, 'M': 15, 'N': 16, 'O': 17, 'P': 18, 'Q': 19, 'R': 20, 'S': 21, 'T': 22, 'U': 23, 'V': 24, 'W': 25, 'X': 26, 'Y': 27, 'Z': 28, "'": 29, ' ': 30
    # fmt: on
}

ind2lab = {
    # fmt: off
    0: '<bos>', 1: '<eos>', 2: '<unk>', 3: 'AA', 4: 'AE', 5: 'AH', 6: 'AO', 7: 'AW', 8: 'AY', 9: 'B', 
    10: 'CH', 11: 'D', 12: 'DH', 13: 'EH', 14: 'ER', 15: 'EY', 16: 'F', 17: 'G', 18: 'HH', 19: 'IH', 
    20: 'IY', 21: 'JH', 22: 'K', 23: 'L', 24: 'M', 25: 'N', 26: 'NG', 27: 'OW', 28: 'OY', 29: 'P', 
    30: 'R', 31: 'S', 32: 'SH', 33: 'T', 34: 'TH', 35: 'UH', 36: 'UW', 37: 'V', 38: 'W', 39: 'Y', 
    40: 'Z', 41: 'ZH', 42: ' '
    # fmt: on
}


# ======================
# Arguemnt Parser Config
# ======================

parser = get_base_parser("SoundChoice: Grapheme-to-Phoneme", None, None)
parser.add_argument(
    "-i",
    "--input",
    type=str,
    default="To be or not to be, that is the question",
    help="Input text.",
)
parser.add_argument(
    '--disable_ailia_tokenizer',
    action='store_true',
    help='disable ailia tokenizer.'
)
parser.add_argument("--onnx", action="store_true", help="execute onnxruntime version.")
args = update_parser(parser, check_input_type=False)


# ======================
# Secondary Functions
# ======================


def expand_to_chars(emb, seq, seq_len, word_separator):
    """Expands word embeddings to a sequence of character
    embeddings, assigning each character the word embedding
    of the word to which it belongs

    Arguments
    ---------
    emb: torch.Tensor
        a tensor of word embeddings
    seq: torch.Tensor
        a tensor of character embeddings
    seq_len: torch.Tensor
        a tensor of character embedding lengths
    word_separator: torch.Tensor
        the word separator being used

    Returns
    -------
    char_word_emb: torch.Tensor
        a combined character + word embedding tensor
    """
    word_boundaries = seq == word_separator
    words = np.cumsum(word_boundaries, axis=-1)

    char_word_emb = np.zeros((emb.shape[0], seq.shape[-1], emb.shape[-1]))
    seq_len_idx = (seq_len * seq.shape[-1]).astype(int)
    for idx, (item, item_length) in enumerate(zip(words, seq_len_idx)):
        char_word_emb[idx] = emb[idx, item]
        char_word_emb[idx, item_length:, :] = 0
        char_word_emb[idx, word_boundaries[idx], :] = 0

    return char_word_emb


def clean_pipeline(txt, graphemes):
    """
    Cleans incoming text, removing any characters not on the
    accepted list of graphemes and converting to uppercase

    Arguments
    ---------
    txt: str
        the text to clean up
    graphemes: list
        a list of graphemes

    Returns
    -------
    str:
        A wrapped transformation function
    """
    RE_MULTI_SPACE = re.compile(r"\s{2,}")

    result = txt.upper()
    result = "".join(char for char in result if char in graphemes)
    result = RE_MULTI_SPACE.sub(" ", result)
    return result


def grapheme_pipeline(char, uppercase=True):
    """Encodes a grapheme sequence

    Arguments
    ---------
    graphemes: list
        a list of available graphemes
    uppercase: bool
        whether or not to convert items to uppercase

    Returns
    -------
    grapheme_list: list
        a raw list of graphemes, excluding any non-matching
        labels
    grapheme_encoded_list: list
        a list of graphemes encoded as integers
    grapheme_encoded: torch.Tensor
    """

    if uppercase:
        char = char.upper()
    grapheme_list = [
        # grapheme for grapheme in char if grapheme in grapheme_encoder.lab2ind
        grapheme
        for grapheme in char
        if grapheme in lab2ind
    ]

    def encode_label(label):
        """Encode label to int

        Arguments
        ---------
        label : hashable
            Label to encode, must exist in the mapping.
        Returns
        -------
        int
            Corresponding encoded int value.
        """
        try:
            return lab2ind[label]
        except KeyError:
            unk_label = "<unk>"
            return lab2ind[unk_label]

    grapheme_encoded_list = [encode_label(label) for label in grapheme_list]

    bos_label = "<bos>"
    grapheme_encoded = np.array([lab2ind[bos_label]] + list(grapheme_encoded_list))
    grapheme_len = np.array(len(grapheme_encoded))

    return grapheme_list, grapheme_encoded_list, grapheme_encoded, grapheme_len


def word_emb_pipeline(
    models,
    txt,
    grapheme_encoded,
    grapheme_encoded_len,
):
    raw_word_emb = embeddings(models, txt)
    word_separator_idx = lab2ind[" "]

    char_word_emb = expand_to_chars(
        emb=raw_word_emb[None, ...],
        seq=grapheme_encoded[None, ...],
        seq_len=grapheme_encoded_len[None, ...],
        word_separator=word_separator_idx,
    )
    char_word_emb = np.squeeze(char_word_emb)

    return char_word_emb


# ======================
# Main functions
# ======================


def embeddings(models, sentence):
    tokenizer = models["tokenizer"]
    encoded = tokenizer.encode_plus(sentence, return_tensors="np")
    input_ids = encoded["input_ids"]
    attention_mask = encoded["attention_mask"]
    token_type_ids = encoded["token_type_ids"]

    # feedforward
    net = models["emb"]
    if not args.onnx:
        output = net.predict([input_ids, attention_mask, token_type_ids])
    else:
        output = net.run(
            None,
            {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "token_type_ids": token_type_ids,
            },
        )
    hidden_states = output[0]

    token_ids_word = np.array(
        [idx for idx, word_id in enumerate(encoded.word_ids()) if word_id is not None],
    )

    # get_hidden_states
    layers = [-4, -3, -2, -1]
    output = np.sum(hidden_states[layers], axis=0)
    output = np.squeeze(output)
    output = output[token_ids_word]

    return output


def encode_input(models, input_text):
    intermediate = {}

    graphemes = [
        # fmt: off
        'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', "'", ' '
        # fmt: on
    ]
    partial_clean_pipeline = partial(clean_pipeline, graphemes=graphemes)
    txt_cleaned = partial_clean_pipeline(input_text)

    grapheme_list, grapheme_encoded_list, grapheme_encoded, grapheme_len = (
        grapheme_pipeline(txt_cleaned)
    )
    intermediate["grapheme_list"] = grapheme_list
    intermediate["grapheme_encoded_list"] = grapheme_encoded_list
    intermediate["grapheme_encoded"] = grapheme_encoded

    word_emb = word_emb_pipeline(models, input_text, grapheme_encoded, grapheme_len)
    intermediate["word_emb"] = word_emb

    return intermediate


def compute_outputs(net, p_seq, encoder_outputs):
    hyps, scores, *_ = S2SBeamSearcher(net, args.onnx).forward(encoder_outputs)

    def decode_ndim(x):
        try:
            decoded = []
            for subtensor in x:
                decoded.append(decode_ndim(subtensor))
            return decoded
        except TypeError:  # Not an iterable, bottom level!
            return ind2lab[int(x)]

    phonemes = decode_ndim(hyps)
    return phonemes


def predict(models, input_text):
    model_input = encode_input(models, input_text)
    grapheme_encoded = model_input["grapheme_encoded"][None, ...]
    word_emb = model_input["word_emb"][None, ...].astype(np.float32)

    # feedforward
    net = models["net"]
    if not args.onnx:
        output = net.predict([grapheme_encoded, word_emb])
    else:
        output = net.run(
            None, {"grapheme_encoded": grapheme_encoded, "word_emb": word_emb}
        )
    p_seq, encoder_outputs, _ = output

    net = models["beam"]
    phonemes = compute_outputs(net, p_seq, encoder_outputs)

    return phonemes[0]


def recognize_from_text(models):
    input_text = args.input

    logger.info("Input text: " + input_text)

    # inference
    logger.info("Start inference...")
    if args.benchmark:
        logger.info("BENCHMARK mode")
        total_time_estimation = 0
        for i in range(args.benchmark_count):
            start = int(round(time.time() * 1000))
            phonemes = predict(models, input_text)
            end = int(round(time.time() * 1000))
            estimation_time = end - start

            # Logging
            logger.info(f"\tailia processing estimation time {estimation_time} ms")
            if i != 0:
                total_time_estimation = total_time_estimation + estimation_time

        logger.info(
            f"\taverage time estimation {total_time_estimation / (args.benchmark_count - 1)} ms"
        )
    else:
        phonemes = predict(models, input_text)

    print("-".join(phonemes))

    logger.info("Script finished successfully.")


def main():
    # model files check and download
    check_and_download_models(WEIGHT_PATH, MODEL_PATH, REMOTE_PATH)
    check_and_download_models(WEIGHT_EMB_PATH, MODEL_EMB_PATH, REMOTE_PATH)
    check_and_download_models(WEIGHT_BEAM_PATH, MODEL_BEAM_PATH, REMOTE_PATH)

    env_id = args.env_id

    # initialize
    if not args.onnx:
        net = ailia.Net(MODEL_PATH, WEIGHT_PATH, env_id=env_id, memory_mode=args.memory_mode)
        emb = ailia.Net(MODEL_EMB_PATH, WEIGHT_EMB_PATH, env_id=env_id, memory_mode=args.memory_mode)
        beam = ailia.Net(MODEL_BEAM_PATH, WEIGHT_BEAM_PATH, env_id=env_id, memory_mode=args.memory_mode)
    else:
        import onnxruntime

        providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
        net = onnxruntime.InferenceSession(WEIGHT_PATH, providers=providers)
        emb = onnxruntime.InferenceSession(WEIGHT_EMB_PATH, providers=providers)
        beam = onnxruntime.InferenceSession(WEIGHT_BEAM_PATH, providers=providers)

    if args.disable_ailia_tokenizer:
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained("tokenizer")
    else:
        from ailia_tokenizer import BertTokenizer
        tokenizer = BertTokenizer.from_pretrained("./tokenizer/")

    models = {
        "tokenizer": tokenizer,
        "net": net,
        "emb": emb,
        "beam": beam,
    }

    recognize_from_text(models)


if __name__ == "__main__":
    main()
