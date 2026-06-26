import numpy as np


def get_bert_feature(text, word2ph, tokenizer, bert_session, use_onnx=False):
    """Extract BERT features for Chinese text and map to phone-level.

    Args:
        text: Normalized Chinese text string.
        word2ph: List of integers indicating how many phones each character maps to.
        tokenizer: ailia_tokenizer.BertTokenizer instance.
        bert_session: ailia.Net or onnxruntime.InferenceSession for BERT model.
        use_onnx: If True, use onnxruntime; otherwise use ailia SDK.

    Returns:
        np.ndarray of shape (num_phones, 1024) with BERT features per phone.
    """
    inputs = tokenizer(text, return_tensors="np")
    input_ids = inputs["input_ids"].astype(np.int64)
    attention_mask = inputs["attention_mask"].astype(np.int64)
    token_type_ids = inputs["token_type_ids"].astype(np.int64)

    if use_onnx:
        hidden_states = bert_session.run(None, {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "token_type_ids": token_type_ids,
        })[0]
    else:
        hidden_states = bert_session.run({
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "token_type_ids": token_type_ids,
        })[0]

    # hidden_states shape: (1, seq_len, 1024)
    # Remove [CLS] at start and [SEP] at end
    res = hidden_states[0][1:-1]  # shape: (text_len, 1024)

    assert len(word2ph) == len(text), f"word2ph length {len(word2ph)} != text length {len(text)}"

    phone_level_feature = []
    for i in range(len(word2ph)):
        repeat_feature = np.tile(res[i], (word2ph[i], 1))
        phone_level_feature.append(repeat_feature)
    phone_level_feature = np.concatenate(phone_level_feature, axis=0)

    # Return shape (num_phones, 1024) as float32
    return phone_level_feature.astype(np.float32)
