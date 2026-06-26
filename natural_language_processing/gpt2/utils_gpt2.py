import numpy as np
import random

def generate_text(tokenizer, ailia_model, span, outputlength, onnx_runtime=False, seed=None, temperature=1.0):
    model_input = tokenizer.encode_plus(span)
    model_input = {name : np.atleast_2d(value) for name, value in model_input.items()}

    model_input['input_ids'] = np.array(model_input['input_ids'], dtype='int64')
    model_input['attention_mask'] = np.array(model_input['attention_mask'], dtype='int64')

    if onnx_runtime:
      onnx_result = ailia_model.run(None,model_input)
    else:
      onnx_result = ailia_model.run(model_input)

    if seed is not None:
        random.seed(seed)

    out_str = span
    for i in range(outputlength):
      # pick next token logits
      logits = onnx_result[0][0, -1]

      # safe softmax with temparature
      mod_logits = logits - np.max(logits)
      work = np.exp(mod_logits / temperature)
      prob = work / np.sum(work)

      # pick top-k
      K=20
      topk_idx = np.argpartition(-prob, K)[:K]
      topk_prob = [ prob[i] for i in topk_idx ]

      # select next token with top-k probability weight
      index = random.choices(topk_idx, weights=topk_prob)[0]
      token = tokenizer.convert_ids_to_tokens([index])[0]
      out_str += token.replace('Ġ',' ')
      trim = 0
      input = np.append(model_input['input_ids'][:,trim:], index)
      model_input['input_ids'] = np.expand_dims(input, 0)
      attention_mask = np.append(model_input['attention_mask'][:,trim:], 1)
      model_input['attention_mask'] = np.expand_dims(attention_mask, 0)
      if onnx_runtime:
        onnx_result = ailia_model.run(None,model_input)
      else:
        onnx_result = ailia_model.run(model_input)

      if token == "<unk>":
        break

    return out_str