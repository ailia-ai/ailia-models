import numpy as np


def sample_next_token(logits, rng, top_k, top_p, temperature):
    """Pick the next token id with top-k / top-p (nucleus) sampling.

    argsort is called with kind='stable' on purpose. The default introsort
    leaves the order of equal elements undefined and that order differs
    between numpy 1.x and 2.x, which would make the generated text depend on
    the installed numpy version.
    """
    logits = np.asarray(logits, dtype=np.float64)
    if temperature != 1.0:
        logits = logits / temperature

    order = np.argsort(-logits, kind='stable')
    if top_k > 0:
        order = order[:top_k]

    probs = np.exp(logits[order] - logits[order[0]])
    probs /= probs.sum()

    if 0.0 < top_p < 1.0:
        cumulative = np.cumsum(probs)
        # keep the shortest prefix whose cumulative probability reaches top_p
        keep = int(np.searchsorted(cumulative, top_p)) + 1
        order = order[:keep]
        probs = probs[:keep]
        probs /= probs.sum()

    cumulative = np.cumsum(probs)
    pos = int(np.searchsorted(cumulative, rng.random() * cumulative[-1], side='right'))
    return int(order[min(pos, len(order) - 1)])


def generate_text(tokenizer, model, span, outputlength, onnx_runtime=False, greedy = False,
                  top_k = 50, top_p = 0.95, temperature = 1.0, seed = 42):
    rng = np.random.default_rng(seed)
    # ailia_tokenizer does not expose eos_token, fall back to the T5 default
    eos_token = getattr(tokenizer, 'eos_token', None) or '</s>'

    model_input = tokenizer.encode_plus(span)
    model_input = {name : np.atleast_2d(value) for name, value in model_input.items()}

    model_input['input_ids'] = np.array(model_input['input_ids'], dtype='int64')
    model_input['attention_mask'] = np.array(model_input['attention_mask'], dtype='int64')

    if onnx_runtime:
      onnx_result = model.run(None,model_input)
    else:
      onnx_result = model.run(model_input)

    out_str = span
    for i in range(outputlength):
      if not greedy:
        index = sample_next_token(onnx_result[0][0, -1], rng, top_k, top_p, temperature)
      else:
        next_token_logits = onnx_result[0][:, -1, :]
        next_tokens = np.argmax(next_token_logits, axis=-1)
        index = next_tokens[0]

      token = tokenizer.convert_ids_to_tokens([index])[0]
      if token == eos_token:
        break
      out_str += token
      trim = 0
      input = np.append(model_input['input_ids'][:,trim:], index)
      model_input['input_ids'] = np.expand_dims(input, 0)
      attention_mask = np.append(model_input['attention_mask'][:,trim:], 1)
      model_input['attention_mask'] = np.expand_dims(attention_mask, 0)
      if onnx_runtime:
        onnx_result = model.run(None,model_input)
      else:
        onnx_result = model.run(model_input)

      if token == "<unk>":
        break

    return out_str