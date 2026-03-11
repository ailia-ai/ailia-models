
import sys
import time
from logging import getLogger

from funasr_ailia import SenseVoiceSmall
from funasr_ailia.utils.postprocess_utils import rich_transcription_postprocess
from funasr_ailia import Fsmn_vad_online

import soundfile
import librosa

import numpy as np

# import original modules
sys.path.append("../../util")

from model_utils import check_and_download_models, check_and_download_file  # noqa
from arg_utils import get_base_parser, get_savepath, update_parser  # noqa

logger = getLogger(__name__)

# ======================
# Parameters
# ======================

WAV_PATH = "ja.wav"
SAVE_TEXT_PATH = "output.txt"

# ======================
# Arguemnt Parser Config
# ======================

parser = get_base_parser("SenseVoice", WAV_PATH, SAVE_TEXT_PATH, input_ftype="audio", fp16_support = False)
parser.add_argument(
	"-V",
	action="store_true",
	help="use microphone input",
)
parser.add_argument(
	"--disable_ailia_audio", action="store_true", help="disable ailia_audio. (Require kaldi_native_fbank)"
)
parser.add_argument(
	"--disable_ailia_tokenizer", action="store_true", help="disable ailia_tokenizer. (Require sentencepiece)"
)
parser.add_argument(
	"--fp16", action="store_true", help="use fp16 model (default : fp32 model)."
)
parser.add_argument(
	"--onnx", action="store_true", help="use onnx runtime."
)
parser.add_argument("--finetuning", type=str, default=None, help="use finetuning model", choices=(None, "japanese"))
args = update_parser(parser)

# ======================
# Models
# ======================

REMOTE_PATH = "https://storage.googleapis.com/ailia-models/sensevoice/"
REMOTE_PATH_VAD = REMOTE_PATH

if args.fp16:
	WEIGHT_PATH = "sensevoice_small_fp16.onnx"
	VAD_WEIGHT_PATH = "speech_fsmn_vad_zh-cn-16k-common_fp16.onnx"
else:
	WEIGHT_PATH = "sensevoice_small.onnx"
	VAD_WEIGHT_PATH = "speech_fsmn_vad_zh-cn-16k-common.onnx"

MODEL_PATH = WEIGHT_PATH + ".prototxt"
VAD_MODEL_PATH = VAD_WEIGHT_PATH + ".prototxt"

if args.finetuning is not None:
	REMOTE_PATH = "https://storage.googleapis.com/ailia-models/sensevoice-finetuning/"
	WEIGHT_PATH = "sensevoice_small_" + args.finetuning + ".onnx"
	MODEL_PATH = None

def format_timestamp(seconds: float, always_include_hours: bool = False):
	assert seconds >= 0, "non-negative timestamp expected"
	milliseconds = round(seconds * 1000.0)

	hours = milliseconds // 3_600_000
	milliseconds -= hours * 3_600_000

	minutes = milliseconds // 60_000
	milliseconds -= minutes * 60_000

	seconds = milliseconds // 1_000
	milliseconds -= seconds * 1_000

	hours_marker = f"{hours}:" if always_include_hours or hours > 0 else ""

	return f"{hours_marker}{minutes:02d}:{seconds:02d}.{milliseconds:03d}"

def recognize_from_audio():
	# input audio loop
	logger.info("Start inference...")
	for audio_path in args.input:
		logger.info(audio_path)

		model = SenseVoiceSmall(env_id=args.env_id, onnx=args.onnx, ailia_audio=not args.disable_ailia_audio, ailia_tokenizer=not args.disable_ailia_tokenizer, profile=args.profile, model_file=WEIGHT_PATH)
		vad = Fsmn_vad_online(env_id=args.env_id, onnx=args.onnx, ailia_audio=not args.disable_ailia_audio, profile=args.profile, model_file=VAD_WEIGHT_PATH)

		# vad
		speech, sample_rate = soundfile.read(audio_path)
		if speech.ndim > 1:
			speech = np.mean(speech, axis=1)
		target_sr = 16000
		if sample_rate != target_sr:
			speech = librosa.resample(speech, orig_sr=sample_rate, target_sr=target_sr)

		vad_enable = True

		if vad_enable:
			speech_length = speech.shape[0]
			sample_offset = 0
			step = 1600
			param_dict = {"in_cache": []}
			start = -1
			end = -1

			start_profile = int(round(time.time() * 1000))

			for sample_offset in range(0, speech_length, min(step, speech_length - sample_offset)):
				if sample_offset + step >= speech_length - 1:
					step = speech_length - sample_offset
					is_final = True
				else:
					is_final = False
				param_dict["is_final"] = is_final
				segments_result = vad(
					audio_in=speech[sample_offset : sample_offset + step], param_dict=param_dict
				)
				for segment in segments_result:
					for s in segment:
						#print(s)
						if s[0] != -1:
							start = s[0]
						if s[1] != -1:
							end = s[1]
						if start != -1 and end != -1:
							start_int = int(start / 1000 * target_sr)
							end_int = int(end / 1000 * target_sr)
							audio = speech[start_int:end_int]

							res = model(audio, language="auto", use_itn=True)
							
							for i in res:
								print("[" + format_timestamp(start / 1000) + " --> " + format_timestamp(end / 1000) + "] " + rich_transcription_postprocess(i))

							start = -1
							end = -1

			end_profile = int(round(time.time() * 1000))
			estimation_time = end_profile - start_profile
			if args.benchmark:
				logger.info(f"\ts2t processing time {estimation_time} ms")
		else:
			# s2t
			wav_or_scp = speech
			start = int(round(time.time() * 1000))
			res = model(wav_or_scp, language="auto", use_itn=True) # 16khz
			end = int(round(time.time() * 1000))
			estimation_time = end - start
		
			print([rich_transcription_postprocess(i) for i in res])
			if args.benchmark:
				logger.info(f"\ts2t processing time {estimation_time} ms")

	logger.info("Script finished successfully.")

def recognize_from_mic():
	# input audio loop
	logger.info("Start inference...")

	model = SenseVoiceSmall(env_id=args.env_id, onnx=args.onnx, ailia_audio=not args.disable_ailia_audio, ailia_tokenizer=not args.disable_ailia_tokenizer, profile=args.profile, model_file=WEIGHT_PATH)
	vad = Fsmn_vad_online(env_id=args.env_id, onnx=args.onnx, ailia_audio=not args.disable_ailia_audio, profile=args.profile, model_file=VAD_WEIGHT_PATH)

	import sounddevice

	devices = sounddevice.query_devices()
	if len(devices) == 0:
		logger.error("No microphone devices found")
		return

	#logger.info(devices)
	default_input_device_idx = sounddevice.default.device[0]
	logger.info(f'Input device: {devices[default_input_device_idx]["name"]}')

	param_dict = {"in_cache": []}
	param_dict["is_final"] = False

	start = -1
	end = -1

	samples_per_read = int(0.1 * 16000)
	target_sr = 16000
	samples = np.zeros((0))
	
	logger.info("Waiting microphone input...")

	try:
		with sounddevice.InputStream(channels=1, dtype="float32", samplerate=16000) as mic:
			while True:
				speech, _ = mic.read(samples_per_read)
				speech = speech[:, 0]
				samples = np.concatenate([samples, speech])
				segments_result = vad(
					audio_in=speech, param_dict=param_dict
				)
				for segment in segments_result:
					for s in segment:
						if s[0] != -1:
							start = s[0]
						if s[1] != -1:
							end = s[1]
						if start != -1 and end != -1:
							start_int = int(start / 1000 * target_sr)
							end_int = int(end / 1000 * target_sr)
							audio = samples[start_int:end_int]

							res = model(audio, language="auto", use_itn=True)
							
							for i in res:
								print("[" + format_timestamp(start / 1000) + " --> " + format_timestamp(end / 1000) + "] " + rich_transcription_postprocess(i))

							start = -1
							end = -1
	except KeyboardInterrupt:
		pass

	logger.info("Script finished successfully.")

def main():
	check_and_download_models(WEIGHT_PATH, MODEL_PATH, REMOTE_PATH)
	check_and_download_models(VAD_WEIGHT_PATH, VAD_MODEL_PATH, REMOTE_PATH_VAD)
	
	if args.V:
		recognize_from_mic()
	else:
		recognize_from_audio()

if __name__ == "__main__":
	main()
