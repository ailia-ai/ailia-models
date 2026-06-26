from pathlib import Path
from typing import List
from typing import Union

class SentencepiecesTokenizer:
    def __init__(self, bpemodel: Union[Path, str], ailia_tokenizer: bool = False, **kwargs):
        super().__init__(**kwargs)
        self.bpemodel = str(bpemodel)
        self.sp = None
        self.ailia_tokenizer = ailia_tokenizer
        self._build_sentence_piece_processor()

    def __repr__(self):
        return f'{self.__class__.__name__}(model="{self.bpemodel}")'

    def _build_sentence_piece_processor(self):
        # Build SentencePieceProcessor lazily.
        if self.sp is None:
            if self.ailia_tokenizer:
                from ailia_tokenizer import LlamaTokenizer
                self.tokenizer = LlamaTokenizer.from_pretrained("./tokenizer")
            else:
                import sentencepiece as spm
                self.sp = spm.SentencePieceProcessor()
                self.sp.load(self.bpemodel)

    def decode(self, line: List[int], **kwargs):
        self._build_sentence_piece_processor()
        if self.ailia_tokenizer:
            return self.tokenizer.decode(line, skip_special_tokens=True)
        else:
            return self.sp.DecodeIds(line)
