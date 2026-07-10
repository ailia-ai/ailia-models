# Qwen3-VL

## Input

- Image

  ![Input](demo.jpeg)

  (Image from https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-VL/assets/demo.jpeg)

- Prompt

  Describe this image.

## Output

- Response

  ```
  This is a heartwarming, sun-drenched photograph capturing a joyful moment between a woman and her dog on a beach at sunset.

  **Key Elements:**

  - **The Subjects:** A young woman with long dark hair, wearing a plaid shirt and dark pants, sits cross-legged in the sand. She is smiling brightly, looking at her dog. Beside her, a light-colored Labrador Retriever, wearing a colorful patterned harness, sits attentively, extending its paw to give a high-five to the woman’s hand.

  - **The Setting:** They are on a wide, sandy beach. The ocean stretches out behind them, with gentle waves rolling in. The horizon is softly lit by the setting sun, which creates a warm, golden glow and a slight lens flare in the upper right corner.

  - **The Action:** The central focus is the high-five gesture — a playful and affectionate interaction between the woman and her dog. It conveys a sense of companionship.
  ```

## Requirements

If you use `--disable_ailia_tokenizer` option, this model requires additional module.
```
pip3 install transformers
```

## Usage

Automatically downloads the onnx and prototxt files on the first run.
It is necessary to be connected to the Internet while downloading.

For the sample image,
```bash
$ python3 qwen3_vl.py
```

If you want to specify the input image, put the image path after the `--input` option.
```bash
$ python3 qwen3_vl.py --input IMAGE_PATH
```

If you want to specify the prompt, put the prompt after the `--prompt` option.
```bash
$ python3 qwen3_vl.py --prompt PROMPT
```

By adding the `--model_type` option, you can specify the model size which is selected from "4b", "8b". (default is 8b)
```bash
$ python3 qwen3_vl.py --model_type 4b
```

For the 8b model, adding the `--fp16` option runs the language model in fp16 to reduce memory usage. (not supported for 4b)
```bash
$ python3 qwen3_vl.py --model_type 8b --fp16
```

You can adjust the generation behavior with `--max_new_tokens`, `--temperature`, `--top_k`, `--top_p` and `--repetition_penalty`.
```bash
$ python3 qwen3_vl.py --max_new_tokens 512 --temperature 0.7 --top_k 20 --top_p 0.8 --repetition_penalty 1.0
```

## Reference

- [Qwen3-VL](https://github.com/QwenLM/Qwen3-VL)
- [Hugging Face - Qwen3-VL-4B-Instruct](https://huggingface.co/Qwen/Qwen3-VL-4B-Instruct)
- [Hugging Face - Qwen3-VL-8B-Instruct](https://huggingface.co/Qwen/Qwen3-VL-8B-Instruct)

## Framework

Pytorch

## Model Format

ONNX opset=17

## Netron

[qwen3_vl_4b_instruct_vision_encoder.onnx.prototxt](https://netron.app/?url=https://storage.googleapis.com/ailia-models/qwen3_vl/qwen3_vl_4b_instruct_vision_encoder.onnx.prototxt)  
[qwen3_vl_4b_instruct_language_model.onnx.prototxt](https://netron.app/?url=https://storage.googleapis.com/ailia-models/qwen3_vl/qwen3_vl_4b_instruct_language_model.onnx.prototxt)  
[qwen3_vl_8b_instruct_vision_encoder.onnx.prototxt](https://netron.app/?url=https://storage.googleapis.com/ailia-models/qwen3_vl/qwen3_vl_8b_instruct_vision_encoder.onnx.prototxt)  
[qwen3_vl_8b_instruct_language_model.onnx.prototxt](https://netron.app/?url=https://storage.googleapis.com/ailia-models/qwen3_vl/qwen3_vl_8b_instruct_language_model.onnx.prototxt)  
[qwen3_vl_8b_instruct_language_model_fp16.onnx.prototxt](https://netron.app/?url=https://storage.googleapis.com/ailia-models/qwen3_vl/qwen3_vl_8b_instruct_language_model_fp16.onnx.prototxt)
