# coding=utf-8
# SPDX-License-Identifier: Apache-2.0
"""
Thread-safe text buffer between a producer (an LLM, a keyboard) and the
synthesis loop. The producer calls push() with fragments of any size and
close() when there is no more; the loop calls poll() to take the text that can
be committed, and wait() to sleep until something new arrives.

Text is committed up to the last point where cutting cannot change how the
whole text tokenizes: before a whitespace run, or after one of BOUNDARY_CHARS
(Japanese / Chinese punctuation, so text without spaces streams clause by
clause). With `tokenize` given, it is committed at talker token boundaries
instead, holding back the last `holdback_tokens` tokens that might still merge
with text to come. Committed chunks carry their own leading space, so the
consumer tokenizes each chunk as is.

This is the simplified form of Qwen3-TTS-streaming-input's LiveTextSource:
the end of the text is only ever signalled by close(), there is no idle timeout
that guesses it.
"""
import re
import threading
from typing import Callable, List, Optional, Sequence, Tuple

BOUNDARY_CHARS = "、。，．！？：；…"
_WS_RUN = re.compile(r"\s+")


class LiveTextSource:
    def __init__(self, boundary_chars: str = BOUNDARY_CHARS,
                 tokenize: Optional[Callable[[str], Sequence[Tuple[int, int]]]] = None,
                 holdback_tokens: int = 3):
        self._boundary = frozenset(boundary_chars)
        self._tokenize = tokenize
        self._holdback = max(1, holdback_tokens)
        self._cv = threading.Condition()
        self._buffer = ""
        self._closed = False

    def push(self, fragment: str) -> None:
        with self._cv:
            if not self._closed:
                self._buffer += fragment
                self._cv.notify_all()

    def close(self) -> None:
        with self._cv:
            self._closed = True
            self._cv.notify_all()

    def is_closed(self) -> bool:
        with self._cv:
            return self._closed

    def poll(self) -> Tuple[List[str], bool]:
        """(committed chunks, closed). After close() everything left is committed."""
        with self._cv:
            if self._closed:
                text, self._buffer = self._buffer, ""
                return self._chunk(text), True
            cut = self._cut(self._buffer)
            if cut <= 0:
                return [], False
            text, self._buffer = self._buffer[:cut], self._buffer[cut:]
            return self._chunk(text), False

    def wait(self, timeout: Optional[float] = None) -> None:
        """Block until poll() would return something (or close()), or timeout."""
        with self._cv:
            self._cv.wait_for(lambda: self._closed or self._cut(self._buffer) > 0, timeout)

    @staticmethod
    def _chunk(text: str) -> List[str]:
        text = _WS_RUN.sub(" ", text)
        return [text] if text.strip() else []

    def _cut(self, text: str) -> int:
        """Largest i such that text[:i] can be committed; 0 if none yet."""
        if self._tokenize is not None:
            offsets = self._tokenize(text)
            return offsets[-1 - self._holdback][1] if len(offsets) > self._holdback else 0
        for i in range(len(text) - 1, 0, -1):
            if text[i] in self._boundary:
                return i + 1
            if text[i].isspace() and not text[i - 1].isspace():
                return i   # cut before the space: it belongs to the next chunk
        return 0


def token_offsets_fn(tokenizer) -> Callable[[str], Sequence[Tuple[int, int]]]:
    """Adapt a tokenizer to the `tokenize` argument: character offsets of the tokens.

    A Hugging Face fast tokenizer gives them directly. Any other tokenizer with
    encode / decode (ailia_tokenizer's GPT2Tokenizer) gets them by decoding the
    growing prefix of ids: a token's end is how much of the text the ids up to
    it reproduce, so a token that ends in the middle of a multi-byte character
    ends where the previous one did and the following one closes the character.
    Only the end of the token holdback_tokens from the last is ever used."""
    if getattr(tokenizer, "is_fast", False):
        def offsets(text: str):
            return tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)["offset_mapping"]
        return offsets

    def offsets(text: str):
        ids = list(tokenizer.encode(text, add_special_tokens=False))
        result, start = [], 0
        for k in range(len(ids)):
            try:
                prefix = tokenizer.decode(ids[:k + 1])
            except UnicodeDecodeError:          # the ids so far end inside a multi-byte character
                result.append((start, start))
                continue
            end = 0
            for end in range(min(len(prefix), len(text)), 0, -1):
                if text.startswith(prefix[:end]):
                    break
            else:
                end = 0
            end = max(end, start)
            result.append((start, end))
            start = end
        return result
    return offsets
