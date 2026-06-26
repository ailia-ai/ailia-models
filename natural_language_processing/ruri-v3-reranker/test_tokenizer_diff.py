"""Compare input_ids produced by ailia_tokenizer vs transformers AutoTokenizer
for the same query/document pair, to see where they diverge.
"""

from ailia_tokenizer import GemmaTokenizer
from transformers import PreTrainedTokenizerFast, AutoTokenizer

QUERY = "瑠璃色はどんな色？"
DOCUMENT = (
    "瑠璃色（るりいろ）は、紫みを帯びた濃い青。名は、半貴石の瑠璃（ラピスラズリ、"
    "英: lapis lazuli）による。JIS慣用色名では「こい紫みの青」（略号 dp-pB）と定義している[1][2]。"
)


def show(name, tokenizer):
    features = tokenizer(
        [[QUERY, DOCUMENT]],
        padding=True,
        truncation=True,
        max_length=512,
        return_tensors="np",
    )
    ids = features["input_ids"][0]
    print(f"--- {name} ---")
    print("num_tokens:", len(ids))
    print("input_ids:", ids.tolist())
    print("decoded:", repr(tokenizer.decode(ids)))
    print()
    return ids


def show_single(name, tokenizer):
    ids = tokenizer([QUERY], return_tensors="np")["input_ids"][0]
    print(f"--- {name} (query only) ---")
    print("num_tokens:", len(ids))
    print("input_ids:", ids.tolist())
    print("decoded:", repr(tokenizer.decode(ids)))
    print()
    return ids


def main():
    #tok_hf = AutoTokenizer.from_pretrained("tokenizer")
    tok_hf = AutoTokenizer.from_pretrained("cl-nagoya/ruri-v3-reranker-310m")
    #tok_hf = PreTrainedTokenizerFast.from_pretrained("tokenizer")
    #tok_hf = PreTrainedTokenizerFast.from_pretrained("cl-nagoya/ruri-v3-reranker-310m")
    tok_ai = GemmaTokenizer.from_pretrained("./tokenizer")

    ids_hf_q = show_single("AutoTokenizer (--disable_ailia_tokenizer)", tok_hf)
    ids_ai_q = show_single("ailia_tokenizer.GemmaTokenizer", tok_ai)
    print("query: same length:", len(ids_hf_q) == len(ids_ai_q))
    print()

    ids_hf = show("AutoTokenizer (--disable_ailia_tokenizer)", tok_hf)
    ids_ai = show("ailia_tokenizer.GemmaTokenizer", tok_ai)

    print("same length:", len(ids_hf) == len(ids_ai))
    if len(ids_hf) == len(ids_ai):
        print("same ids:", (ids_hf == ids_ai).all())


if __name__ == "__main__":
    main()
