# pip3 install ailia_voice

import ailia_voice

# On Windows on ARM (WoA) librosa / soundfile cannot be installed. Only in
# that case, register the pure Python wav-only fallback in
# util/woa_librosa.py so the imports below keep working.
import platform
if platform.system() == 'Windows' and platform.machine().lower() in ('arm64', 'aarch64'):
	import os
	import sys
	sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'util'))
	import woa_librosa
	woa_librosa.enable_if_needed()

import librosa
import time
import soundfile

import os
import urllib.request

# Load reference audio
ref_text = "水をマレーシアから買わなくてはならない。"
ref_file_path = "reference_audio_girl.wav"
if not os.path.exists(ref_file_path):
	urllib.request.urlretrieve(
		"https://github.com/ailia-ai/ailia-models/raw/refs/heads/master/audio_processing/gpt-sovits/reference_audio_captured_by_ax.wav",
		"reference_audio_girl.wav"
	)
audio_waveform, sampling_rate = librosa.load(ref_file_path, mono=True)

# Infer
voice = ailia_voice.GPTSoVITS()
voice.initialize_model(model_path = "./models/")
voice.set_reference_audio(ref_text, ailia_voice.AILIA_VOICE_G2P_TYPE_GPT_SOVITS_JA, audio_waveform, sampling_rate)
buf, sampling_rate = voice.synthesize_voice("こんにちは。今日はいい天気ですね。", ailia_voice.AILIA_VOICE_G2P_TYPE_GPT_SOVITS_JA)
#buf, sampling_rate = voice.synthesize_voice("Hello world.", ailia_voice.AILIA_VOICE_G2P_TYPE_GPT_SOVITS_EN)

# Save result
soundfile.write("output.wav", buf, sampling_rate)