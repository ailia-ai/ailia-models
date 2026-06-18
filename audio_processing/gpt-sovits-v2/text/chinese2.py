import os
import re

import cn2an
from pypinyin import lazy_pinyin, Style
from pypinyin.contrib.tone_convert import to_normal, to_finals_tone3, to_initials, to_finals

from text.symbols2 import punctuation
from text.tone_sandhi import ToneSandhi
from text.zh_normalization.text_normlization import TextNormalizer

normalizer = lambda x: cn2an.transform(x, "an2cn")

current_file_path = os.path.dirname(__file__)
pinyin_to_symbol_map = {
    line.split("\t")[0]: line.strip().split("\t")[1]
    for line in open(os.path.join(current_file_path, "opencpop-strict.txt")).readlines()
}

import jieba.posseg as psg
import logging
logging.getLogger("jieba").setLevel(logging.WARNING)

# is_g2pw_str = os.environ.get("is_g2pw", "True")
# is_g2pw = False#True if is_g2pw_str.lower() == 'true' else False
is_g2pw = True
if is_g2pw:
    print("Using g2pw for pinyin inference")
    from text.g2pw import G2PWPinyin, correct_pronunciation
    parent_directory = os.path.dirname(current_file_path)
    g2pw = G2PWPinyin(
        model_dir=os.path.join(parent_directory, "text", "G2PWModel"),
        model_source=os.path.join(parent_directory, "tokenizer"),
        v_to_u=False,
        neutral_tone_with_five=True,
    )

rep_map = {
    "\uFF1A": ",",
    "\uFF1B": ",",
    "\uFF0C": ",",
    "\u3002": ".",
    "\uFF01": "!",
    "\uFF1F": "?",
    "\n": ".",
    "\u00B7": ",",
    "\u3001": ",",
    "...": "\u2026",
    "$": ".",
    "/": ",",
    "\u2014": "-",
    "~": "\u2026",
    "\uFF5E": "\u2026",
}

tone_modifier = ToneSandhi()


def replace_punctuation(text):
    text = text.replace("\u55EF", "\u6069").replace("\u5463", "\u6BCD")
    pattern = re.compile("|".join(re.escape(p) for p in rep_map.keys()))

    replaced_text = pattern.sub(lambda x: rep_map[x.group()], text)

    replaced_text = re.sub(
        r"[^\u4e00-\u9fa5" + "".join(punctuation) + r"]+", "", replaced_text
    )

    return replaced_text


def g2p(text):
    pattern = r"(?<=[{0}])\s*".format("".join(punctuation))
    sentences = [i for i in re.split(pattern, text) if i.strip() != ""]
    phones, word2ph = _g2p(sentences)
    return phones, word2ph


def _get_initials_finals(word):
    initials = []
    finals = []

    orig_initials = lazy_pinyin(word, neutral_tone_with_five=True, style=Style.INITIALS)
    orig_finals = lazy_pinyin(
        word, neutral_tone_with_five=True, style=Style.FINALS_TONE3
    )

    for c, v in zip(orig_initials, orig_finals):
        initials.append(c)
        finals.append(v)
    return initials, finals


must_erhua = {
    "\u5C0F\u9662\u513F", "\u80E1\u540C\u513F", "\u8303\u513F",
    "\u8001\u6C49\u513F", "\u6492\u6B22\u513F", "\u5BFB\u8001\u793C\u513F",
    "\u59A5\u59A5\u513F", "\u5AB3\u5987\u513F",
}
not_erhua = {
    "\u8650\u513F", "\u4E3A\u513F", "\u62A4\u513F", "\u7792\u513F",
    "\u6551\u513F", "\u66FF\u513F", "\u6709\u513F", "\u4E00\u513F",
    "\u6211\u513F", "\u4FFA\u513F", "\u59BB\u513F", "\u62D0\u513F",
    "\u804B\u513F", "\u4E5E\u513F", "\u60A3\u513F", "\u5E7C\u513F",
    "\u5B64\u513F", "\u5A74\u513F", "\u5A74\u5E7C\u513F",
    "\u8FDE\u4F53\u513F", "\u8111\u7621\u513F", "\u6D41\u6D6A\u513F",
    "\u4F53\u5F31\u513F", "\u6DF7\u8840\u513F", "\u871C\u96EA\u513F",
    "\u8235\u513F", "\u7956\u513F", "\u7F8E\u513F", "\u5E94\u91C7\u513F",
    "\u53EF\u513F", "\u4F84\u513F", "\u5B59\u513F", "\u4F84\u5B59\u513F",
    "\u5973\u513F", "\u7537\u513F", "\u7EA2\u5B69\u513F", "\u82B1\u513F",
    "\u866B\u513F", "\u9A6C\u513F", "\u9E1F\u513F", "\u732A\u513F",
    "\u732B\u513F", "\u72D7\u513F", "\u5C11\u513F",
}


def _merge_erhua(initials, finals, word, pos):
    """
    Do erhua.
    """
    # fix er1
    for i, phn in enumerate(finals):
        if i == len(finals) - 1 and word[i] == "\u513F" and phn == 'er1':
            finals[i] = 'er2'

    if word not in must_erhua and (word in not_erhua or
                                        pos in {"a", "j", "nr"}):
        return initials, finals

    if len(finals) != len(word):
        return initials, finals

    assert len(finals) == len(word)

    new_initials = []
    new_finals = []
    for i, phn in enumerate(finals):
        if i == len(finals) - 1 and word[i] == "\u513F" and phn in {
                "er2", "er5"
        } and word[-2:] not in not_erhua and new_finals:
            phn = "er" + new_finals[-1][-1]

        new_initials.append(initials[i])
        new_finals.append(phn)

    return new_initials, new_finals


def _g2p(segments):
    phones_list = []
    word2ph = []
    for seg in segments:
        pinyins = []
        # Replace all English words in the sentence
        seg = re.sub("[a-zA-Z]+", "", seg)
        seg_cut = psg.lcut(seg)
        seg_cut = tone_modifier.pre_merge_for_modify(seg_cut)
        initials = []
        finals = []

        if not is_g2pw:
            for word, pos in seg_cut:
                if pos == "eng":
                    continue
                sub_initials, sub_finals = _get_initials_finals(word)
                sub_finals = tone_modifier.modified_tone(word, pos, sub_finals)
                sub_initials, sub_finals = _merge_erhua(sub_initials, sub_finals, word, pos)
                initials.append(sub_initials)
                finals.append(sub_finals)
            initials = sum(initials, [])
            finals = sum(finals, [])
        else:
            # g2pw uses sentence-level inference
            pinyins = g2pw.lazy_pinyin(seg, neutral_tone_with_five=True, style=Style.TONE3)

            pre_word_length = 0
            for word, pos in seg_cut:
                sub_initials = []
                sub_finals = []
                now_word_length = pre_word_length + len(word)

                if pos == 'eng':
                    pre_word_length = now_word_length
                    continue

                word_pinyins = pinyins[pre_word_length:now_word_length]

                # Polyphonic character disambiguation
                word_pinyins = correct_pronunciation(word, word_pinyins)

                for pinyin in word_pinyins:
                    if pinyin[0].isalpha():
                        sub_initials.append(to_initials(pinyin))
                        sub_finals.append(to_finals_tone3(pinyin, neutral_tone_with_five=True))
                    else:
                        sub_initials.append(pinyin)
                        sub_finals.append(pinyin)

                pre_word_length = now_word_length
                sub_finals = tone_modifier.modified_tone(word, pos, sub_finals)
                sub_initials, sub_finals = _merge_erhua(sub_initials, sub_finals, word, pos)
                initials.append(sub_initials)
                finals.append(sub_finals)

            initials = sum(initials, [])
            finals = sum(finals, [])

        for c, v in zip(initials, finals):
            raw_pinyin = c + v
            if c == v:
                assert c in punctuation
                phone = [c]
                word2ph.append(1)
            else:
                v_without_tone = v[:-1]
                tone = v[-1]

                pinyin = c + v_without_tone
                assert tone in "12345"

                if c:
                    v_rep_map = {
                        "uei": "ui",
                        "iou": "iu",
                        "uen": "un",
                    }
                    if v_without_tone in v_rep_map.keys():
                        pinyin = c + v_rep_map[v_without_tone]
                else:
                    pinyin_rep_map = {
                        "ing": "ying",
                        "i": "yi",
                        "in": "yin",
                        "u": "wu",
                    }
                    if pinyin in pinyin_rep_map.keys():
                        pinyin = pinyin_rep_map[pinyin]
                    else:
                        single_rep_map = {
                            "v": "yu",
                            "e": "e",
                            "i": "y",
                            "u": "w",
                        }
                        if pinyin[0] in single_rep_map.keys():
                            pinyin = single_rep_map[pinyin[0]] + pinyin[1:]

                assert pinyin in pinyin_to_symbol_map.keys(), (pinyin, seg, raw_pinyin)
                new_c, new_v = pinyin_to_symbol_map[pinyin].split(" ")
                new_v = new_v + tone
                phone = [new_c, new_v]
                word2ph.append(len(phone))

            phones_list += phone
    return phones_list, word2ph


def replace_punctuation_with_en(text):
    text = text.replace("\u55EF", "\u6069").replace("\u5463", "\u6BCD")
    pattern = re.compile("|".join(re.escape(p) for p in rep_map.keys()))

    replaced_text = pattern.sub(lambda x: rep_map[x.group()], text)

    replaced_text = re.sub(
        r"[^\u4e00-\u9fa5A-Za-z" + "".join(punctuation) + r"]+", "", replaced_text
    )

    return replaced_text


def replace_consecutive_punctuation(text):
    punctuations = ''.join(re.escape(p) for p in punctuation)
    pattern = f'([{punctuations}])([{punctuations}])+'
    result = re.sub(pattern, r'\1', text)
    return result


def text_normalize(text):
    tx = TextNormalizer()
    sentences = tx.normalize(text)
    dest_text = ""
    for sentence in sentences:
        dest_text += replace_punctuation(sentence)

    dest_text = replace_consecutive_punctuation(dest_text)
    return dest_text


def mix_text_normalize(text):
    tx = TextNormalizer()
    sentences = tx.normalize(text)
    dest_text = ""
    for sentence in sentences:
        dest_text += replace_punctuation_with_en(sentence)

    dest_text = replace_consecutive_punctuation(dest_text)
    return dest_text


if __name__ == "__main__":
    text = "\u4F60\u597D"
    text = text_normalize(text)
    print(g2p(text))
