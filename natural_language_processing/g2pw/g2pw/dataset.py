import numpy as np
from g2pw.utils import tokenize_and_map


def get_phoneme_labels(polyphonic_chars):
    labels = sorted(list(set([phoneme for char, phoneme in polyphonic_chars])))
    char2phonemes = {}
    for char, phoneme in polyphonic_chars:
        if char not in char2phonemes:
            char2phonemes[char] = []
        char2phonemes[char].append(labels.index(phoneme))
    return labels, char2phonemes


class TextDataset:
    def __init__(self, tokenizer, labels, char2phonemes, chars, texts, query_ids,
                 use_mask=False, window_size=None, max_len=512):
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.window_size = window_size

        self.labels = labels
        self.char2phonemes = char2phonemes
        self.chars = chars
        self.texts = texts
        self.query_ids = query_ids

        self.use_mask = use_mask

        if window_size is not None:
            self.truncated_texts, self.truncated_query_ids = self._truncate_texts(self.window_size, texts, query_ids)

    def _truncate_texts(self, window_size, texts, query_ids):
        truncated_texts = []
        truncated_query_ids = []
        for text, query_id in zip(texts, query_ids):
            start = max(0, query_id - window_size // 2)
            end = min(len(text), query_id + window_size // 2)
            truncated_text = text[start:end]
            truncated_texts.append(truncated_text)

            truncated_query_id = query_id - start
            truncated_query_ids.append(truncated_query_id)
        return truncated_texts, truncated_query_ids

    def _truncate(self, max_len, text, query_id, tokens, text2token, token2text):
        truncate_len = max_len - 2
        if len(tokens) <= truncate_len:
            return (text, query_id, tokens, text2token, token2text)

        token_position = text2token[query_id]

        token_start = token_position - truncate_len // 2
        token_end = token_start + truncate_len
        font_exceed_dist = -token_start
        back_exceed_dist = token_end - len(tokens)
        if font_exceed_dist > 0:
            token_start += font_exceed_dist
            token_end += font_exceed_dist
        elif back_exceed_dist > 0:
            token_start -= back_exceed_dist
            token_end -= back_exceed_dist

        start = token2text[token_start][0]
        end = token2text[token_end - 1][1]

        return (
            text[start:end],
            query_id - start,
            tokens[token_start:token_end],
            [i - token_start if i is not None else None for i in text2token[start:end]],
            [(s - start, e - start) for s, e in token2text[token_start:token_end]]
        )

    def __getitem__(self, idx):
        text = (self.truncated_texts if self.window_size else self.texts)[idx].lower()
        query_id = (self.truncated_query_ids if self.window_size else self.query_ids)[idx]

        try:
            tokens, text2token, token2text = tokenize_and_map(self.tokenizer, text)
        except Exception:
            print(f'warning: text "{text}" is invalid')
            return self[(idx + 1) % len(self)]

        text, query_id, tokens, text2token, token2text = self._truncate(self.max_len, text, query_id, tokens, text2token, token2text)

        processed_tokens = ['[CLS]'] + tokens + ['[SEP]']

        input_ids = np.array(self.tokenizer.convert_tokens_to_ids(processed_tokens), dtype=np.int64)
        token_type_ids = np.array([0] * len(processed_tokens), dtype=np.int64)
        attention_mask = np.array([1] * len(processed_tokens), dtype=np.int64)

        query_char = text[query_id]
        phoneme_mask = [1 if i in self.char2phonemes[query_char] else 0 for i in range(len(self.labels))] \
            if self.use_mask else [1] * len(self.labels)
        char_id = self.chars.index(query_char)
        position_id = text2token[query_id] + 1  # [CLS] token locate at first place

        outputs = {
            'input_ids': input_ids,
            'token_type_ids': token_type_ids,
            'attention_mask': attention_mask,
            'phoneme_mask': phoneme_mask,
            'char_id': char_id,
            'position_id': position_id,
        }
        return outputs

    def __len__(self):
        return len(self.texts)

    def create_mini_batch(self, samples):
        def _agg(name):
            return [sample[name] for sample in samples]

        # numpyによるpad_sequenceの実装
        def pad_sequence_np(sequences, padding_value=0):
            max_len = max([len(s) for s in sequences])
            # (Batch, MaxLen)
            out = np.full((len(sequences), max_len), padding_value, dtype=np.int64)
            for i, seq in enumerate(sequences):
                out[i, :len(seq)] = seq
            return out

        input_ids = pad_sequence_np(_agg('input_ids'))
        token_type_ids = pad_sequence_np(_agg('token_type_ids'))
        attention_mask = pad_sequence_np(_agg('attention_mask'))
        
        phoneme_mask = np.array(_agg('phoneme_mask'), dtype=np.float32)
        char_ids = np.array(_agg('char_id'), dtype=np.int64)
        position_ids = np.array(_agg('position_id'), dtype=np.int64)

        batch_output = {
            'input_ids': input_ids,
            'token_type_ids': token_type_ids,
            'attention_mask': attention_mask,
            'phoneme_mask': phoneme_mask,
            'char_ids': char_ids,
            'position_ids': position_ids
        }

        return batch_output