import os

import numpy as np
from typing import List, Optional, Tuple, Union
import regex as re

import json

from languages import LANGUAGES, TO_LANGUAGE_CODE

def get_tokenizer(multilingual: bool,
        num_languages: int = 99,
        task: Optional[str] = None,  # Literal["transcribe", "translate", None]
        language: Optional[str] = None):
    if multilingual:
        tokenizer_name = "multilingual"
        task = task or "transcribe"
        language = language or "en"
    else:
        tokenizer_name = "gpt2"
        task = None
        language = None

    _assets = os.path.join(os.path.dirname(os.path.abspath(__file__)), "assets")
    tokenizer = AiliaTokenizer()
    tokenizer.build_tokenizer(
        os.path.join(_assets, "multilingual/vocab.json"),
        os.path.join(_assets, "multilingual/merges.txt"),
        tokenizer_name, task, language, num_languages)
    return tokenizer

# Uses some code from Apache licensed transformers
# https://github.com/huggingface/transformers/blob/main/LICENSE

def get_pairs(word):
    """
    Return set of symbol pairs in a word.
    Word is represented as tuple of symbols (symbols being variable-length strings).
    """
    pairs = set()
    prev_char = word[0]
    for char in word[1:]:
        pairs.add((prev_char, char))
        prev_char = char
    return pairs

class AiliaTokenizer:
    # dictionary
    vocab = {}
    byte_encoder = None
    byte_decode = None

    # state
    language = None

    # sequence
    sot_sequence = None
    all_language_tokens_list = None

    # special tokens
    sot = None
    sot_prev = None
    sot_lm = None
    no_speech = None
    eot = None
    no_timestamps = None
    translate = None
    transcribe = None
    timestamp_begin = None

    def build_tokenizer(self, vocab_path, merges_path, tokenizer_name, task, language, num_languages):
        self.language = language

        # load vocab
        json_open = open(vocab_path, 'r', encoding='utf-8')
        json_load = json.load(json_open)
        for key in json_load.keys():
            self.vocab[json_load[key]] = key
        self.byte_encoder = self.bytes_to_unicode()
        self.byte_decoder = {v: k for k, v in self.byte_encoder.items()}

        #specials = [
        #"<|startoftranscript|>",
        #*[f"<|{lang}|>" for lang in LANGUAGES.keys()],
        #"<|translate|>",
        #"<|transcribe|>",
        #"<|startoflm|>",
        #"<|startofprev|>",
        #"<|nospeech|>",
        #"<|notimestamps|>",
        #]

        # load special token ids
        if tokenizer_name == "multilingual":
            multilingal = 1
        else:
            multilingal = 0
        self.eot  = 50256 + multilingal
        self.sot  = 50257 + multilingal
        language_tokens = []
        i = self.sot + 1
        for lang in list(LANGUAGES.keys())[:num_languages]:
            language_tokens.append(i)
            i = i + 1
        self.translate = 50357 + (num_languages - 99) + multilingal
        self.transcribe = 50358 + (num_languages - 99) + multilingal
        self.sot_lm = 50359 + (num_languages - 99) + multilingal
        self.sot_prev = 50360 + (num_languages - 99) + multilingal
        self.no_speech = 50361 + (num_languages - 99) + multilingal
        self.no_timestamps = 50362 + (num_languages - 99) + multilingal
        self.timestamp_begin = 50363 + (num_languages - 99) + multilingal

        self.all_language_tokens_list = language_tokens

        langs = tuple(list(LANGUAGES.keys())[:num_languages])
        sot_sequence = [self.sot]
        if language is not None:
            sot_sequence.append(self.sot + 1 + langs.index(language))
        if task is not None:
            sot_sequence.append(self.transcribe if task == "transcribe" else self.translate)
        self.sot_sequence = sot_sequence

        # for encoder
        with open(merges_path, encoding="utf-8") as merges_handle:
            bpe_merges = merges_handle.read().split("\n")[1:-1]
        bpe_merges = [tuple(merge.split()) for merge in bpe_merges]
        self.bpe_ranks = dict(zip(bpe_merges, range(len(bpe_merges))))
        self.cache = {}

    def bytes_to_unicode(self):
        """
        Returns list of utf-8 byte and a mapping to unicode strings. We specifically avoids mapping to whitespace/control
        characters the bpe code barfs on.
        The reversible bpe codes work on unicode strings. This means you need a large # of unicode characters in your vocab
        if you want to avoid UNKs. When you're at something like a 10B token dataset you end up needing around 5K for
        decent coverage. This is a significant percentage of your normal, say, 32K bpe vocab. To avoid that, we want lookup
        tables between utf-8 bytes and unicode strings.
        """
        bs = (
            list(range(ord("!"), ord("~") + 1)) + list(range(ord("¡"), ord("¬") + 1)) + list(range(ord("®"), ord("ÿ") + 1))
        )
        cs = bs[:]
        n = 0
        for b in range(2**8):
            if b not in bs:
                bs.append(b)
                cs.append(2**8 + n)
                n += 1
        cs = [chr(n) for n in cs]
        return dict(zip(bs, cs))

    def convert_tokens_to_string(self, tokens):
        """Converts a sequence of tokens (string) in a single string."""
        text = "".join(tokens)
        text = bytearray([self.byte_decoder[c] for c in text]).decode("utf-8", errors="ignore")
        return text

    def bpe(self, token):
        if token in self.cache:
            return self.cache[token]
        word = tuple(token)
        pairs = get_pairs(word)

        if not pairs:
            return token

        while True:
            bigram = min(pairs, key=lambda pair: self.bpe_ranks.get(pair, float("inf")))
            if bigram not in self.bpe_ranks:
                break
            first, second = bigram
            new_word = []
            i = 0
            while i < len(word):
                try:
                    j = word.index(first, i)
                except ValueError:
                    new_word.extend(word[i:])
                    break
                else:
                    new_word.extend(word[i:j])
                    i = j

                if word[i] == first and i < len(word) - 1 and word[i + 1] == second:
                    new_word.append(first + second)
                    i += 2
                else:
                    new_word.append(word[i])
                    i += 1
            new_word = tuple(new_word)
            word = new_word
            if len(word) == 1:
                break
            else:
                pairs = get_pairs(word)
        word = " ".join(word)
        self.cache[token] = word
        return word

    def tokenize(self, text):
        """Tokenize a string."""
        # Should have added re.IGNORECASE so BPE merges can happen for capitalized versions of contractions
        pat = re.compile(r"""'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+""")
        bpe_tokens = []
        for token in re.findall(pat, text):
            token = "".join(
                self.byte_encoder[b] for b in token.encode("utf-8")
            )  # Maps all our bytes to unicode strings, avoiding control tokens of the BPE (spaces in our case)
            bpe_tokens.extend(bpe_token for bpe_token in self.bpe(token).split(" "))
        return bpe_tokens

    def encode(self, text):
        tokenized = self.tokenize(text)
        tokens = []
        for token in tokenized:
            for i in range(len(self.vocab)):
                if self.vocab[i] == token:
                    tokens.append(i)
        return tokens

    def decode(self, tokens):
        tokens = [token for token in tokens if token < self.eot]
        vocab_tokens = []
        for token in tokens:
            vocab_tokens.append(self.vocab[token])
        return self.convert_tokens_to_string(vocab_tokens)

    @property
    def language_token(self) -> int:
        i = 0
        for lang in LANGUAGES.keys():
            if lang == self.language:
                return self.all_language_tokens_list[i]
            i = i + 1
        raise KeyError(f"Language {self.language} not found in tokenizer.")

    @property
    def all_language_tokens(self) -> Tuple[int]:
        return self.all_language_tokens_list

    @property
    def all_language_codes(self) -> Tuple[str]:
        return tuple(l for l in LANGUAGES.keys())

    @property
    def non_speech_tokens(self) -> Tuple[int]:
        """
        Returns the list of tokens to suppress in order to avoid any speaker tags or non-speech
        annotations, to prevent sampling texts that are not actually spoken in the audio, e.g.

        - ♪♪♪
        - ( SPEAKING FOREIGN LANGUAGE )
        - [DAVID] Hey there,

        keeping basic punctuations like commas, periods, question marks, exclamation points, etc.
        """
        return (1, 2, 7, 8, 9, 10, 14, 25, 26, 27, 28, 29, 31, 58, 59, 60, 61, 62, 63, 90, 91, 92, 93, 359, 503, 522, 542, 873, 893, 902, 918, 922, 931, 1350, 1853, 1982, 2460, 2627, 3246, 3253, 3268, 3536, 3846, 3961, 4183, 4667, 6585, 6647, 7273, 9061, 9383, 10428, 10929, 11938, 12033, 12331, 12562, 13793, 14157, 14635, 15265, 15618, 16553, 16604, 18362, 18956, 20075, 21675, 22520, 26130, 26161, 26435, 28279, 29464, 31650, 32302, 32470, 36865, 42863, 47425, 49870, 50254)
