# Inference sample for benchmarking
# pip3 install ailia_speech

import ailia_speech

import librosa
import time

env_id = 1
is_fp16 = True
input_file_path = "ax.wav"

audio_waveform, sampling_rate = librosa.load(input_file_path, mono = True)

# Infer
speech = ailia_speech.SenseVoice(env_id = env_id)
speech.initialize_model(model_path = "./models/", model_type = ailia_speech.AILIA_SPEECH_MODEL_TYPE_SENSEVOICE_SMALL, vad_version = "6_2", is_fp16 = is_fp16)
start = int(round(time.time() * 1000))
recognized_text = speech.transcribe(audio_waveform, sampling_rate)
for text in recognized_text:
	print(text)
end = int(round(time.time() * 1000))
estimation_time = (end - start)

print(f'\ttotal processing time {estimation_time} ms')
