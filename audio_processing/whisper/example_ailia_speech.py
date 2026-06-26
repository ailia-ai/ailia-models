# Inference sample for benchmarking
# pip3 install ailia_speech

import ailia_speech

import librosa
import time

env_id = 1
is_fp16 = True
model_type = ailia_speech.AILIA_SPEECH_MODEL_TYPE_WHISPER_MULTILINGUAL_LARGE_V3_TURBO
input_file = "ax.wav"

audio_waveform, sampling_rate = librosa.load(input_file, mono = True)

# Infer
speech = ailia_speech.Whisper(env_id = env_id)
speech.initialize_model(model_path = "./models/", model_type = model_type, vad_type = None, diarization_type = None, is_fp16 = False)
start = int(round(time.time() * 1000))
recognized_text = speech.transcribe(audio_waveform, sampling_rate)
for text in recognized_text:
	print(text)
end = int(round(time.time() * 1000))
estimation_time = (end - start)

print(f'\ttotal processing time {estimation_time} ms')
