import time
import sys

import numpy as np
from typing import Dict, List, Tuple, Union

class SquadExample:
    def __init__(self, question, context):
        self.question_text = question
        self.context_text = context

class SquadFeatures:
    def __init__(self, input_ids, attention_mask, token_type_ids, p_mask, encoding):
        self.input_ids = input_ids
        self.attention_mask = attention_mask
        self.token_type_ids = token_type_ids
        self.p_mask = p_mask
        self.encoding = encoding

import ailia

sys.path.append('../../util')
from arg_utils import get_base_parser, update_parser  # noqa: E402
from model_utils import check_and_download_models  # noqa: E402

# logger
from logging import getLogger   # noqa: E402
logger = getLogger(__name__)


# ======================
# Arguemnt Parser Config
# ======================

DEFAULT_QUESTION = "What is ailia SDK ?"
DEFAULT_CONTEXT = ("ailia SDK is a highly performant single inference engine "
                   "for multiple platforms and hardware")

#test
#DEFAULT_QUESTION = 'Why is model conversion important?'
#DEFAULT_CONTEXT = 'The option to convert models between FARM and transformers gives freedom to the user and let people easily switch between frameworks.'

parser = get_base_parser('bert question answering.', None, None)
parser.add_argument(
    '--question', '-q', metavar='TEXT', default=DEFAULT_QUESTION,
    help='input question'
)
parser.add_argument(
    '--context', '-c', metavar='TEXT', default=DEFAULT_CONTEXT,
    help='input context'
)
parser.add_argument(
    '--disable_ailia_tokenizer',
    action='store_true',
    help='disable ailia tokenizer.'
)
args = update_parser(parser, check_input_type=False)


# ======================
# PARAMETERS
# ======================

WEIGHT_PATH = "roberta-base-squad2.onnx"
MODEL_PATH = "roberta-base-squad2.onnx.prototxt"
REMOTE_PATH = \
    "https://storage.googleapis.com/ailia-models/bert_question_answering/"


# ======================
# Utils
# ======================

# code from https://github.com/patil-suraj/onnx_transformers
# Apache license

def create_sample(
    question: Union[str, List[str]], context: Union[str, List[str]]
) -> Union[SquadExample, List[SquadExample]]:
    """
    QuestionAnsweringPipeline leverages the :class:`~transformers.SquadExample` internally.
    This helper method encapsulate all the logic for converting question(s) and context(s) to
    :class:`~transformers.SquadExample`.

    We currently support extractive question answering.

    Arguments:
        question (:obj:`str` or :obj:`List[str]`): The question(s) asked.
        context (:obj:`str` or :obj:`List[str]`): The context(s) in which we will look for the answer.

    Returns:
        One or a list of :class:`~transformers.SquadExample`: The corresponding
        :class:`~transformers.SquadExample` grouping question and context.
    """
    if isinstance(question, list):
        return [SquadExample(q, c) for q, c in zip(question, context)]
    else:
        return SquadExample(question, context)


def decode(start: np.ndarray, end: np.ndarray, topk: int, max_answer_len: int) -> Tuple:
    """
    Take the output of any :obj:`ModelForQuestionAnswering` and will generate probalities for each span to be
    the actual answer.

    In addition, it filters out some unwanted/impossible cases like answer len being greater than
    max_answer_len or answer end position being before the starting position.
    The method supports output the k-best answer through the topk argument.

    Args:
        start (:obj:`np.ndarray`): Individual start probabilities for each token.
        end (:obj:`np.ndarray`): Individual end probabilities for each token.
        topk (:obj:`int`): Indicates how many possible answer span(s) to extract from the model output.
        max_answer_len (:obj:`int`): Maximum size of the answer to extract from the model's output.
    """
    # Ensure we have batch axis
    if start.ndim == 1:
        start = start[None]

    if end.ndim == 1:
        end = end[None]

    # Compute the score of each tuple(start, end) to be the real answer
    outer = np.matmul(np.expand_dims(start, -1), np.expand_dims(end, 1))

    # Remove candidate with end < start and end - start > max_answer_len
    candidates = np.tril(np.triu(outer), max_answer_len - 1)

    #  Inspired by Chen & al. (https://github.com/facebookresearch/DrQA)
    scores_flat = candidates.flatten()
    if topk == 1:
        idx_sort = [np.argmax(scores_flat)]
    elif len(scores_flat) < topk:
        idx_sort = np.argsort(-scores_flat)
    else:
        idx = np.argpartition(-scores_flat, topk)[0:topk]
        idx_sort = idx[np.argsort(-scores_flat[idx])]

    start, end = np.unravel_index(idx_sort, candidates.shape)[1:]
    return start, end, candidates[0, start, end]


def span_to_answer(tokenizer, text: str, start: int, end: int) -> Dict[str, Union[str, int]]:
    """
    When decoding from token probalities, this method maps token indexes to actual word in
    the initial context.

    Args:
        text (:obj:`str`): The actual context to extract the answer from.
        start (:obj:`int`): The answer starting token index.
        end (:obj:`int`): The answer end token index.

    Returns:
        Dictionary like :obj:`{'answer': str, 'start': int, 'end': int}`
    """
    words = []
    token_idx = char_start_idx = char_end_idx = chars_idx = 0

    for i, word in enumerate(text.split(" ")):
        token = tokenizer.tokenize(word)

        # Append words if they are in the span
        if start <= token_idx <= end:
            if token_idx == start:
                char_start_idx = chars_idx

            if token_idx == end:
                char_end_idx = chars_idx + len(word)

            words += [word]

        # Stop if we went over the end of the answer
        if token_idx > end:
            break

        # Append the subtokenization length to the running index
        token_idx += len(token)
        chars_idx += len(word) + 1

    # Join text with spaces
    return {
        "answer": " ".join(words),
        "start": max(0, char_start_idx),
        "end": min(len(text), char_end_idx),
    }

# ======================
# Extract features
# ======================

#Reference
#https://github.com/huggingface/transformers/blob/master/src/transformers/pipelines/question_answering.py

def extract_feature_transformer4(example, tokenizer):
    # for transformer3, we can not use squad_convert_examples_to_features for is_fast model

    # Define the side we want to truncate / pad and the text/pair sorting
    padding = "do_not_pad"
    max_seq_len = 384
    doc_stride =128
    question_first = tokenizer.padding_side == "right"

    if args.disable_ailia_tokenizer:
        encoded_inputs = tokenizer(
            text=example.question_text if question_first else example.context_text,
            text_pair=example.context_text if question_first else example.question_text,
            padding=padding,
            truncation="only_second" if question_first else "only_first",
            max_length=max_seq_len,
            stride=doc_stride,
            return_tensors="np",
            return_token_type_ids=True,
            return_overflowing_tokens=True,
            return_offsets_mapping=True,
            return_special_tokens_mask=True,
        )
    else:
        encoded_inputs = tokenizer(
            text=example.question_text if question_first else example.context_text,
            text_pair=example.context_text if question_first else example.question_text,
            padding=padding,
            truncation="only_second" if question_first else "only_first",
            max_length=max_seq_len,
            #stride=doc_stride,
            return_tensors="np",
            return_token_type_ids=True,
            #return_overflowing_tokens=True,
            #return_offsets_mapping=True,
            #return_special_tokens_mask=True,
        )

    # When the input is too long, it's converted in a batch of inputs with overflowing tokens
    # and a stride of overlap between the inputs. If a batch of inputs is given, a special output
    # "overflow_to_sample_mapping" indicate which member of the encoded batch belong to which original batch sample.
    # Here we tokenize examples one-by-one so we don't need to use "overflow_to_sample_mapping".
    # "num_span" is the number of output samples generated from the overflowing tokens.
    num_spans = len(encoded_inputs["input_ids"])

    # p_mask: mask with 1 for token than cannot be in the answer (0 for token which can be in an answer)
    # We put 0 on the tokens from the context and 1 everywhere else (question and special tokens)
    p_mask = np.asarray(
        [
            [tok != 1 if question_first else 0 for tok in encoded_inputs.sequence_ids(span_id)]
            for span_id in range(num_spans)
        ]
    )

    # keep the cls_token unmasked (some models use it to indicate unanswerable questions)
    if tokenizer.cls_token_id is not None:
        cls_index = np.nonzero(encoded_inputs["input_ids"] == tokenizer.cls_token_id)
        p_mask[cls_index] = 0

    features = []
    for span_idx in range(num_spans):
        input_ids_span_idx = encoded_inputs["input_ids"][span_idx]
        attention_mask_span_idx = (
            encoded_inputs["attention_mask"][span_idx] if "attention_mask" in encoded_inputs.keys() else None
        )
        token_type_ids_span_idx = (
            encoded_inputs["token_type_ids"][span_idx] if "token_type_ids" in encoded_inputs.keys() else None
        )
        submask = p_mask[span_idx]
        if isinstance(submask, np.ndarray):
            submask = submask.tolist()
        features.append(
            SquadFeatures(
                input_ids=input_ids_span_idx,
                attention_mask=attention_mask_span_idx,
                token_type_ids=token_type_ids_span_idx,
                p_mask=submask,
                encoding=encoded_inputs[span_idx] if args.disable_ailia_tokenizer else encoded_inputs,
            )
        )
    return features


def convert_the_answer_back_to_the_original_text_transformer4(answers, example, feature, tokenizer, starts, ends, scores):
    # Convert the answer (tokens) back to the original text
    # Score: score from the model
    # Start: Index of the first character of the answer in the context string
    # End: Index of the character following the last character of the answer in the context string
    # Answer: Plain text of the answer
    question_first = bool(tokenizer.padding_side == "right")
    enc = feature.encoding

    # Encoding was *not* padded, input_ids *might*.
    # It doesn't make a difference unless we're padding on
    # the left hand side, since now we have different offsets
    # everywhere.
    if tokenizer.padding_side == "left":
        offset = (feature.input_ids == tokenizer.pad_token_id).numpy().sum()
    else:
        offset = 0

    # Sometimes the max probability token is in the middle of a word so:
    # - we start by finding the right word containing the token with `token_to_word`
    # - then we convert this word in a character span with `word_to_chars`
    sequence_index = 1 if question_first else 0
    for s, e, score in zip(starts, ends, scores):
        s = s - offset
        e = e - offset
        try:
            start_word = enc.token_to_word(s)
            end_word = enc.token_to_word(e)
            start_index = enc.word_to_chars(start_word, sequence_index=sequence_index)[0]
            end_index = enc.word_to_chars(end_word, sequence_index=sequence_index)[1]
        except Exception:
            # Some tokenizers don't really handle words. Keep to offsets then.
            start_index = enc.offsets[s][0]
            end_index = enc.offsets[e][1]

        answers.append(
            {
                "score": score.item(),
                "start": start_index,
                "end": end_index,
                "answer": example.context_text[start_index:end_index],
            }
        )


# ======================
# Main function
# ======================
def main():
    # model files check and download
    check_and_download_models(WEIGHT_PATH, MODEL_PATH, REMOTE_PATH)

    inputs = {
        "question": args.question,
        "context": args.context,
    }

    logger.info("Question : " + str(args.question))
    logger.info("Context : " + str(args.context))

    # Set defaults values
    handle_impossible_answer = False
    topk = 1
    max_answer_len = 15

    if args.disable_ailia_tokenizer:
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained('deepset/roberta-base-squad2')
        import transformers
        TRANSFORMER_VERSION=int(transformers.__version__.split(".")[0])
        if TRANSFORMER_VERSION <= 3:
            logger.error("Transformer must be greater than 4.")
            return
    else:
        # not supported yet
        from ailia_tokenizer import RobertaTokenizer
        tokenizer = RobertaTokenizer.from_pretrained('tokenizer/')
        tokenizer.padding_side = "right"
        tokenizer.cls_token_id = 0
        tokenizer.model_input_names = ["input_ids", "attention_mask"]
    
   
    # Convert inputs to features
    examples = []

    if True:  # for i, item in enumerate(inputs):
        item = inputs
        logger.debug(item)
        if isinstance(item, dict):
            if any(k not in item for k in ["question", "context"]):
                raise KeyError("You need to provide a dictionary with keys "
                               "{question:..., context:...}")

            example = create_sample(**item)
            examples.append(example)

    features_list = [
        extract_feature_transformer4(example,tokenizer) for example in examples
    ]

    all_answers = []
    for features, example in zip(features_list, examples):
        model_input_names = tokenizer.model_input_names + ["input_ids"]
        fw_args = {k: [feature.__dict__[k] for feature in features]
                   for k in model_input_names}
        fw_args = {k: np.array(v) for (k, v) in fw_args.items()}

        logger.debug("Input" + str(fw_args))
        logger.debug("Shape" + str(fw_args["input_ids"].shape))

        net = ailia.Net(MODEL_PATH, WEIGHT_PATH, env_id=args.env_id, memory_mode=args.memory_mode)
        net.set_input_shape(fw_args["input_ids"].shape)
        if args.benchmark:
            logger.info('BENCHMARK mode')
            for i in range(5):
                start = int(round(time.time() * 1000))
                outputs = net.predict(fw_args)
                end = int(round(time.time() * 1000))
                logger.info(
                    "\tailia processing time {} ms".format(end - start)
                )
        else:
            outputs = net.predict(fw_args)

        logger.debug("Output"+str(outputs))
        start, end = outputs[0:2]

        min_null_score = 1000000  # large and positive
        answers = []
        for (feature, start_, end_) in zip(features, start, end):
            # Ensure padded tokens & question tokens cannot belong
            # to the set of candidate answers.
            undesired_tokens = np.abs(np.array(feature.p_mask) - 1) & \
                feature.attention_mask

            # Generate mask
            undesired_tokens_mask = undesired_tokens == 0.0

            # Make sure non-context indexes in the tensor cannot contribute
            # to the softmax
            start_ = np.where(undesired_tokens_mask, -10000.0, start_)
            end_ = np.where(undesired_tokens_mask, -10000.0, end_)

            # Normalize logits and spans to retrieve the answer
            start_ = np.exp(
                start_ - np.log(np.sum(np.exp(start_), axis=-1, keepdims=True))
            )
            end_ = np.exp(
                end_ - np.log(np.sum(np.exp(end_), axis=-1, keepdims=True))
            )

            if handle_impossible_answer:
                min_null_score = min(
                    min_null_score, (start_[0] * end_[0]).item()
                )

            # Mask CLS
            start_[0] = end_[0] = 0.0

            starts, ends, scores = decode(start_, end_, topk, max_answer_len)
            convert_the_answer_back_to_the_original_text_transformer4(answers, example, feature, tokenizer, starts, ends, scores)

        if handle_impossible_answer:
            answers.append(
                {"score": min_null_score, "start": 0, "end": 0, "answer": ""}
            )

        answers = sorted(
            answers, key=lambda x: x["score"], reverse=True
        )[:topk]
        all_answers += answers

    logger.info("Answer : "+str(all_answers))
    logger.info('Script finished successfully.')


if __name__ == "__main__":
    main()
