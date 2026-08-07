import re

import numpy as np
import nltk

# logger
from logging import getLogger  # noqa

logger = getLogger(__name__)

# for nltk<=3.8.1
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
# for nltk>=3.8.2
nltk.download('punkt_tab')
nltk.download('averaged_perceptron_tagger_eng')


def run_ner(caption):
    noun_phrases = find_noun_phrases(caption)
    noun_phrases = [remove_punctuation(phrase) for phrase in noun_phrases]
    noun_phrases = [phrase for phrase in noun_phrases if phrase != '']
    relevant_phrases = noun_phrases
    labels = noun_phrases

    tokens_positive = []

    for entity, label in zip(relevant_phrases, labels):
        try:
            # search all occurrences and mark them as different entities
            for m in re.finditer(entity, caption.lower()):
                tokens_positive.append([[m.start(), m.end()]])
        except:
            logger.info("noun entities:", noun_phrases)
            logger.info("entity:", entity)
            logger.info("caption:", caption.lower())

    return tokens_positive, labels


def create_positive_map_label_to_token_from_positive_map(positive_map, plus=0):
    positive_map_label_to_token = {}
    for i in range(len(positive_map)):
        positive_map_label_to_token[i + plus] = np.nonzero(positive_map[i])[0].tolist()

    return positive_map_label_to_token


def create_positive_map(tokenized, tokens_positive):
    """construct a map such that positive_map[i,j] = True iff box i is associated to token j"""
    positive_map = np.zeros((len(tokens_positive), 256))

    for j, tok_list in enumerate(tokens_positive):
        for (beg, end) in tok_list:
            try:
                beg_pos = tokenized.char_to_token(beg)
                end_pos = tokenized.char_to_token(end - 1)
            except Exception as e:
                logger.info("beg:", beg, "end:", end)
                logger.info("token_positive:", tokens_positive)
                # print("beg_pos:", beg_pos, "end_pos:", end_pos)
                raise e

            if beg_pos is None:
                try:
                    beg_pos = tokenized.char_to_token(beg + 1)
                    if beg_pos is None:
                        beg_pos = tokenized.char_to_token(beg + 2)
                except:
                    beg_pos = None

            if end_pos is None:
                try:
                    end_pos = tokenized.char_to_token(end - 2)
                    if end_pos is None:
                        end_pos = tokenized.char_to_token(end - 3)
                except:
                    end_pos = None

            if beg_pos is None or end_pos is None:
                continue

            assert beg_pos is not None and end_pos is not None
            positive_map[j, beg_pos: end_pos + 1] = 1

    return positive_map / (positive_map.sum(-1)[:, None] + 1e-6)


def find_noun_phrases(caption: str):
    caption = caption.lower()
    tokens = nltk.word_tokenize(caption)
    pos_tags = nltk.pos_tag(tokens)

    grammar = "NP: {<DT>?<JJ.*>*<NN.*>+}"
    cp = nltk.RegexpParser(grammar)
    result = cp.parse(pos_tags)

    noun_phrases = list()
    for subtree in result.subtrees():
        if subtree.label() == 'NP':
            noun_phrases.append(' '.join(t[0] for t in subtree.leaves()))

    return noun_phrases


def remove_punctuation(text: str) -> str:
    punct = [
        '|', ':', ';', '@', '(', ')', '[', ']', '{', '}', '^',
        '\'', '\"', '’', '`', '?', '$', '%', '#', '!', '&', '*', '+', ',', '.'
    ]
    for p in punct:
        text = text.replace(p, '')
    return text.strip()
