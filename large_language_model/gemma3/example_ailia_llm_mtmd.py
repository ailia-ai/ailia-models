import ailia_llm
import os
import urllib.request

model_file_path = "gemma-3-4b-it-Q4_K_M.gguf"
mmproj_file_path = "gemma-3-4b-it-GGUF_mmproj-model-f16.gguf"
sample_image_path = "sample_image.jpg"

if not os.path.exists(model_file_path):
    print("Model file not found. Downloading...")
    urllib.request.urlretrieve(
        "https://storage.googleapis.com/ailia-models/gemma/gemma-3-4b-it-Q4_K_M.gguf",
        model_file_path
    )
if not os.path.exists(mmproj_file_path):
    print("Multimodal Projector file not found. Downloading...")
    urllib.request.urlretrieve(
        "https://storage.googleapis.com/ailia-models/gemma/gemma-3-4b-it-GGUF_mmproj-model-f16.gguf",
        mmproj_file_path
    )
if not os.path.exists(sample_image_path):
    urllib.request.urlretrieve(
        "https://storage.googleapis.com/ailia-models/misc/sample_image.jpg",
        sample_image_path
    )

print(f"Loading model : {model_file_path}")
model = ailia_llm.AiliaLLM()
model.open(model_file_path)
print(f"Loading projector : {mmproj_file_path}")
model.open_multimodal_projector(mmproj_file_path)

capabilities = model.get_multimodal_capabilities()
if not capabilities['vision']:
    print("Vision support is not available")
    exit(1)

messages = [
    {
    "role": "system", 
    "content": "あなたは画像解析エージェントです。ユーザーの質問に簡潔に回答してください。"
    },
    {
    "role": "user", 
    "content": "何が見える？: <__media__>",
    "media_data": [
        {
        "media_type": "image",
        "file_path": sample_image_path,
        "width": 320,
        "height": 400
        }
    ]
    }
]

import time
start_time = time.time()
stream = model.generate(messages)
text = ""
for delta_text in stream:
    text = text + delta_text
end_time = time.time()

print(text)

if model.context_full():
    raise Exception("Context full")

# token count
token_count = model.token_count(text)
print("Token Count", token_count)

prompt_token_count = model.prompt_token_count()
print("Prompt Token Count", prompt_token_count)

generated_token_count = model.generated_token_count()
print("Generated Token Count", generated_token_count)

# benchmark
print("Token Per Second", (prompt_token_count + generated_token_count) / (end_time - start_time))

