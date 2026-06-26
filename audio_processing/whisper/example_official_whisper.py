# Inference sample for benchmarking
# pip3 install openai-whisper

import time

model_size = "large-v3-turbo"
input_file = "ax.wav"

import os
try:
	if os.path.exists("whisper.py"):
		os.rename("whisper.py", "whisper_local.py")
	import whisper
finally:
	os.rename("whisper_local.py", "whisper.py")

whisper_small = whisper.load_model(model_size)

start = int(round(time.time() * 1000))
result = whisper_small.transcribe("axell_130.wav", language="ja", beam_size = 1, verbose=True)
print(result["text"])
end = int(round(time.time() * 1000))
estimation_time = (end - start)

print(f'\ttotal processing time {estimation_time} ms')
