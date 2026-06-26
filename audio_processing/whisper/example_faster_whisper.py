# Inference sample for benchmarking
# pip3 install faster_whisper

from faster_whisper import WhisperModel
import time

model_size = "large-v3-turbo"
input_file = "ax.wav"

# Run on CPU with FP32
model = WhisperModel(model_size, device="cpu", compute_type="float32")
# or run on GPU with INT8
# model = WhisperModel(model_size, device="cuda", compute_type="int8_float16")
# or run on CPU with INT8
# model = WhisperModel(model_size, device="cpu", compute_type="int8")

start = int(round(time.time() * 1000))
segments, info = model.transcribe(input_file, beam_size=1)
print("Detected language '%s' with probability %f" % (info.language, info.language_probability))
for segment in segments:
    print("[%.2fs -> %.2fs] %s" % (segment.start, segment.end, segment.text))
end = int(round(time.time() * 1000))
estimation_time = (end - start)

print(f'\ttotal processing time {estimation_time} ms')
