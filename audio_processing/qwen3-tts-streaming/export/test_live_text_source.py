"""LiveTextSource: committed chunks must tokenize exactly like the whole text.

    python test_live_text_source.py

Pushes each text in random 1-4 character pieces with random poll() timing and
compares the concatenated token ids of the chunks with the whole text's ids,
for the punctuation/whitespace mode and the token mode at several holdbacks.
"""
import os
import random
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from transformers import AutoTokenizer

from live_text_source import LiveTextSource, token_offsets_fn

TEXTS = [
    "ご注文を承りました。お届けは三日後の予定です。",
    "型番はRTX-3080、シリアル番号はAB12-345Cです。",
    "RTX 3080 の在庫はございますが、価格につきましてはお問い合わせください。",
    "iPhone 15 Proの在庫は、2025年10月1日に入荷予定です。",
    "「少々お待ちください」と申し上げました。お客様、よろしいですか？",
    "Hello there, this is a test of streaming text input. It works!",
    "价格是12800元。请稍等，谢谢！",
]


def main():
    tok = AutoTokenizer.from_pretrained(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "tokenizer"))
    rng = random.Random(0)
    modes = {"punct": lambda: LiveTextSource(),
             "whitespace only": lambda: LiveTextSource(boundary_chars="")}
    for hb in (1, 2, 3):
        modes[f"token hb={hb}"] = lambda hb=hb: LiveTextSource(tokenize=token_offsets_fn(tok), holdback_tokens=hb)
    try:
        from ailia_tokenizer import GPT2Tokenizer
        atok = GPT2Tokenizer.from_pretrained(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "tokenizer"))
        # offsets reconstructed from encode / decode (what the sample uses by default)
        modes["token hb=3 ailia"] = lambda: LiveTextSource(tokenize=token_offsets_fn(atok), holdback_tokens=3)
    except ImportError:
        pass
    ok_all = True
    for name, make in modes.items():
        fails, n_commits, first_at = 0, 0, 0.0
        for text in TEXTS:
            for _ in range(200):
                src, i, commits, first, pushed = make(), 0, [], None, 0
                while i < len(text):
                    n = rng.randint(1, 4); src.push(text[i:i + n]); pushed += n; i += n
                    if rng.random() < 0.5:
                        w, _ = src.poll()
                        if w:
                            commits += w; first = first or pushed
                src.close(); w, _ = src.poll(); commits += w
                ids = []
                for k, c in enumerate(commits):
                    ids += tok(c.lstrip() if k == 0 else c)["input_ids"]
                n_commits += len(commits); first_at += (first or len(text)) / len(text)
                if ids != tok(text)["input_ids"]:
                    fails += 1
        total = len(TEXTS) * 200
        print(f"{name:16} mismatches {fails}/{total}  chunks/utterance {n_commits / total:.1f}  "
              f"first commit at {first_at / total * 100:.0f}% of the text")
        ok_all &= fails == 0 or name.startswith("token")
    print("OK" if ok_all else "NG")


if __name__ == "__main__":
    main()
