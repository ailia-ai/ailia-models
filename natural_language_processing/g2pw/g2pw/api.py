import os
import json

from transformers import BertTokenizer
import numpy as np

from g2pw.dataset import TextDataset, get_phoneme_labels
from g2pw.utils import load_config


def predict(onnx_session, dataloader_or_generator, labels):
    all_preds = []
    for data in dataloader_or_generator:
        input_ids, token_type_ids, attention_mask, phoneme_mask, char_ids, position_ids = \
            [data[name] for name in ('input_ids', 'token_type_ids', 'attention_mask', 'phoneme_mask', 'char_ids', 'position_ids')]

        probs = onnx_session.run(
            [],
            {
                'input_ids': input_ids,
                'token_type_ids': token_type_ids,
                'attention_mask': attention_mask,
                'phoneme_mask': phoneme_mask,
                'char_ids': char_ids,
                'position_ids': position_ids
            }
        )[0]

        preds = np.argmax(probs, axis=-1)
        all_preds += [labels[pred] for pred in preds.tolist()]
    return all_preds


class G2PWConverter:
    def __init__(self, model_dir='G2PWModel/', style='bopomofo', model_source=None, batch_size=None,
                 enable_non_tradional_chinese=False, onnx_session=None):
        required_files = (
            'config.py',
            'POLYPHONIC_CHARS.txt',
            'MONOPHONIC_CHARS.txt',
            'version',
        )
        missing_files = [f for f in required_files if not os.path.exists(os.path.join(model_dir, f))]
        if missing_files:
            raise FileNotFoundError(
                f"Offline mode: required model files are missing in '{model_dir}': {', '.join(missing_files)}"
            )

        if onnx_session is None:
            raise ValueError("onnx_session is required. This implementation does not use onnxruntime.")
        self.session_g2pw = onnx_session

        self.config = load_config(os.path.join(model_dir, 'config.py'), use_default=True)

        self.batch_size = batch_size if batch_size else self.config.batch_size
        self.model_source = model_source if model_source else self.config.model_source
        self.tokenizer = BertTokenizer.from_pretrained(self.model_source)

        polyphonic_chars_path = os.path.join(model_dir, 'POLYPHONIC_CHARS.txt')
        monophonic_chars_path = os.path.join(model_dir, 'MONOPHONIC_CHARS.txt')
        self.polyphonic_chars = [line.split('\t') for line in open(polyphonic_chars_path, 'r', encoding='utf-8').read().strip().split('\n')]
        self.monophonic_chars = [line.split('\t') for line in open(monophonic_chars_path, 'r', encoding='utf-8').read().strip().split('\n')]
        self.labels, self.char2phonemes = get_phoneme_labels(self.polyphonic_chars)
        self.chars = sorted(list(self.char2phonemes.keys()))

        with open(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                               'bopomofo_to_pinyin_wo_tune_dict.json'), 'r', encoding='utf-8') as fr:
            self.bopomofo_convert_dict = json.load(fr)
        self.style_convert_func = {
            'bopomofo': lambda x: x,
            'pinyin': self._convert_bopomofo_to_pinyin,
        }[style]

        with open(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                               'char_bopomofo_dict.json'), 'r', encoding='utf-8') as fr:
            self.char_bopomofo_dict = json.load(fr)

        self.enable_non_tradional_chinese = enable_non_tradional_chinese
        if self.enable_non_tradional_chinese:
            self.s2t_dict = {}
            for line in open(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                    'bert-base-chinese_s2t_dict.txt'), 'r', encoding='utf-8').read().strip().split('\n'):
                s_char, t_char = line.split('\t')
                self.s2t_dict[s_char] = t_char

    def _convert_bopomofo_to_pinyin(self, bopomofo):
        tone = bopomofo[-1]
        assert tone in '12345'
        component = self.bopomofo_convert_dict.get(bopomofo[:-1])
        if component:
            return component + tone
        else:
            print(f'Warning: "{bopomofo}" cannot convert to pinyin')
            return None

    def _convert_s2t(self, sentence):
        return ''.join([self.s2t_dict.get(char, char) for char in sentence])

    def __call__(self, sentences):
        if isinstance(sentences, str):
            sentences = [sentences]

        if self.enable_non_tradional_chinese:
            translated_sentences = []
            for sent in sentences:
                translated_sent = self._convert_s2t(sent)
                assert len(translated_sent) == len(sent)
                translated_sentences.append(translated_sent)
            sentences = translated_sentences

        texts, query_ids, sent_ids, partial_results = self._prepare_data(sentences)
        if len(texts) == 0:
            # sentences no polyphonic words
            return partial_results

        dataset = TextDataset(self.tokenizer, self.labels, self.char2phonemes, self.chars, texts, query_ids,
                              use_mask=self.config.use_mask, window_size=self.config.window_size)

        # DataLoaderの代わりにジェネレータ関数を定義
        def batch_generator():
            n_samples = len(dataset)
            for i in range(0, n_samples, self.batch_size):
                batch_samples = [dataset[j] for j in range(i, min(i + self.batch_size, n_samples))]
                yield dataset.create_mini_batch(batch_samples)

        preds = predict(self.session_g2pw, batch_generator(), self.labels)

        results = partial_results
        for sent_id, query_id, pred in zip(sent_ids, query_ids, preds):
            results[sent_id][query_id] = self.style_convert_func(pred)

        return results

    def _prepare_data(self, sentences):
        polyphonic_chars = set(self.chars)
        monophonic_chars_dict = {
            char: phoneme for char, phoneme in self.monophonic_chars
        }
        texts, query_ids, sent_ids, partial_results = [], [], [], []
        for sent_id, sent in enumerate(sentences):
            partial_result = [None] * len(sent)
            for i, char in enumerate(sent):
                if char in polyphonic_chars:
                    texts.append(sent)
                    query_ids.append(i)
                    sent_ids.append(sent_id)
                elif char in monophonic_chars_dict:
                    partial_result[i] =  self.style_convert_func(monophonic_chars_dict[char])
                elif char in self.char_bopomofo_dict:
                    partial_result[i] =  self.style_convert_func(self.char_bopomofo_dict[char][0])
            partial_results.append(partial_result)
        return texts, query_ids, sent_ids, partial_results