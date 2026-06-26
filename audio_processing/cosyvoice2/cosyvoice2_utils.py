import re
import regex

import inflect

from logging import getLogger
logger = getLogger(__name__)

# whether contain chinese character
def contains_chinese(text):
    return bool(text_normalize.chinese_char_pattern.search(text))


# remove blank between chinese character
def replace_blank(text: str):
    out_str = []
    for i, c in enumerate(text):
        if c == " ":
            if (text[i + 1].isascii() and text[i + 1] != " ") and (
                text[i - 1].isascii() and text[i - 1] != " "
            ):
                out_str.append(c)
        else:
            out_str.append(c)
    return "".join(out_str)


# replace special symbol
def replace_corner_mark(text):
    text = text.replace("²", "平方")
    text = text.replace("³", "立方")
    return text


# remove meaningless symbol
def remove_bracket(text):
    text = text.replace("（", "").replace("）", "")
    text = text.replace("【", "").replace("】", "")
    text = text.replace("`", "").replace("`", "")
    text = text.replace("——", " ")
    return text


# spell Arabic numerals
def spell_out_number(text: str, inflect_parser):
    new_text = []
    st = None
    for i, c in enumerate(text):
        if not c.isdigit():
            if st is not None:
                num_str = inflect_parser.number_to_words(text[st:i])
                new_text.append(num_str)
                st = None
            new_text.append(c)
        else:
            if st is None:
                st = i
    if st is not None and st < len(text):
        num_str = inflect_parser.number_to_words(text[st:])
        new_text.append(num_str)
    return "".join(new_text)


# split paragrah logic：
# 1. per sentence max len token_max_n, min len token_min_n, merge if last sentence len less than merge_len
# 2. cal sentence len according to lang
# 3. split sentence according to puncatation
def split_paragraph(
    text: str,
    tokenize,
    lang="zh",
    token_max_n=80,
    token_min_n=60,
    merge_len=20,
    comma_split=False,
):
    def calc_utt_length(_text: str):
        if lang == "zh":
            return len(_text)
        else:
            return len(tokenize(_text))

    def should_merge(_text: str):
        if lang == "zh":
            return len(_text) < merge_len
        else:
            return len(tokenize(_text)) < merge_len

    if lang == "zh":
        pounc = ["。", "？", "！", "；", "：", "、", ".", "?", "!", ";"]
    else:
        pounc = [".", "?", "!", ";", ":"]
    if comma_split:
        pounc.extend(["，", ","])

    if text[-1] not in pounc:
        if lang == "zh":
            text += "。"
        else:
            text += "."

    st = 0
    utts = []
    for i, c in enumerate(text):
        if c in pounc:
            if len(text[st:i]) > 0:
                utts.append(text[st:i] + c)
            if i + 1 < len(text) and text[i + 1] in ['"', "”"]:
                tmp = utts.pop(-1)
                utts.append(tmp + text[i + 1])
                st = i + 2
            else:
                st = i + 1

    final_utts = []
    cur_utt = ""
    for utt in utts:
        if (
            calc_utt_length(cur_utt + utt) > token_max_n
            and calc_utt_length(cur_utt) > token_min_n
        ):
            final_utts.append(cur_utt)
            cur_utt = ""
        cur_utt = cur_utt + utt
    if len(cur_utt) > 0:
        if should_merge(cur_utt) and len(final_utts) != 0:
            final_utts[-1] = final_utts[-1] + cur_utt
        else:
            final_utts.append(cur_utt)

    return final_utts


def is_only_punctuation(text):
    # Regular expression: Match strings that consist only of punctuation marks or are empty.
    punctuation_pattern = r"^[\p{P}\p{S}]*$"
    return bool(regex.fullmatch(punctuation_pattern, text))


def text_normalize(tokenizer, text, split=True):
    text = text.strip()

    if hasattr(text_normalize, "chinese_char_pattern") is False:
        text_normalize.chinese_char_pattern = re.compile(r"[\u4e00-\u9fff]+")

    def encode(text, **kwargs):
        tokens = tokenizer([text], return_tensors="np")
        tokens = tokens["input_ids"][0].tolist()
        return tokens

    if contains_chinese(text):
        if hasattr(text_normalize, "zh_tn_model") is False:
            try:
                from tn.chinese.normalizer import Normalizer as ZhNormalizer
                text_normalize.zh_tn_model = ZhNormalizer(
                    remove_erhua=False, full_to_half=False, overwrite_cache=True
                )
            except:
                logger.warning("tn.chinese.normalizer load failed. normalize disabled.")
        if hasattr(text_normalize, "zh_tn_model") is True:
            text = text_normalize.zh_tn_model.normalize(text)
        text = text.replace("\n", "")
        text = replace_blank(text)
        text = replace_corner_mark(text)
        text = text.replace(".", "。")
        text = text.replace(" - ", "，")
        text = remove_bracket(text)
        text = re.sub(r"[，,、]+$", "。", text)
        texts = list(
            split_paragraph(
                text,
                encode,
                "zh",
                token_max_n=80,
                token_min_n=60,
                merge_len=20,
                comma_split=False,
            )
        )
    else:
        if hasattr(text_normalize, "en_tn_model") is None:
            try:
                from tn.english.normalizer import Normalizer as EnNormalizer
                text_normalize.en_tn_model = EnNormalizer()
                text_normalize.inflect_parser = inflect.engine()
            except:
                logger.warning("tn.english.normalizer load failed. normalize disabled.")
        if hasattr(text_normalize, "en_tn_model") is True:
            text = text_normalize.en_tn_model.normalize(text)
            text = spell_out_number(text, text_normalize.inflect_parser)
        texts = list(
            split_paragraph(
                text,
                encode,
                "en",
                token_max_n=80,
                token_min_n=60,
                merge_len=20,
                comma_split=False,
            )
        )
    texts = [i for i in texts if not is_only_punctuation(i)]
    return texts if split is True else text
