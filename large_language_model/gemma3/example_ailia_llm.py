import ailia_llm

import os
import urllib.request

model_file_path = "gemma-3-4b-it-Q4_K_M.gguf"
if not os.path.exists(model_file_path):
	urllib.request.urlretrieve(
		"https://storage.googleapis.com/ailia-models/gemma/gemma-3-4b-it-Q4_K_M.gguf",
		model_file_path
	)

model = ailia_llm.AiliaLLM()
model.open(model_file_path)

messages = []
messages.append({"role": "system", "content": "語尾に「わん」をつけてください。"})
messages.append({"role": "user", "content": "あなたの名前は何ですか？"})

stream = model.generate(messages)

text = ""
for delta_text in stream:
	text = text + delta_text
print(text)

if model.context_full():
	raise Exception("Context full")

messages.append({"role": "assistant", "content": text})