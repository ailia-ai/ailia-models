import numpy as np
import cv2, audio
import ailia
import time
import librosa

# Import original modules
import sys
sys.path.append("../../util")
from arg_utils import get_base_parser, update_parser, get_savepath  # noqa: E402
from model_utils import check_and_download_models  # NOQA: E402
from webcamera_utils import adjust_frame_size, get_capture, cut_max_square  # NOQA: E402
from image_utils import imread  # noqa: E402

sys.path.append("../../face_detection")
from blazeface import blazeface_utils as but

# Logger
from logging import getLogger  # noqa: E402

logger = getLogger(__name__)

# ======================
# Parameters
# ======================
WEIGHT_PATH = "lipgan.onnx"
MODEL_PATH = "lipgan.onnx.prototxt"
REMOTE_PATH = "https://storage.googleapis.com/ailia-models/lipgan/"

FACE_DETECTOR_WEIGHT_PATH = "../../face_detection/blazeface/blazeface.onnx"
FACE_DETECTOR_MODEL_PATH = "../../face_detection/blazeface/blazeface.onnx.prototxt"
FACE_DETECTOR_REMOTE_PATH = "https://storage.googleapis.com/ailia-models/blazeface/"

REALESRGAN_MODEL_PATH = 'RealESRGAN_anime.opt.onnx.prototxt'
REALESRGAN_WEIGHT_PATH = 'RealESRGAN_anime.opt.onnx'
REALESRGAN_REMOTE_PATH = 'https://storage.googleapis.com/ailia-models/real-esrgan/'

DLIB_WEIGHT_PATH = "mmod_human_face_detector.dat"

INPUT_IMAGE_PATH = "input.jpg"
INPUT_AUDIO_PATH = "input.wav"

SAVE_IMAGE_PATH = "output.mp4"

IMG_SIZE = 96

# ======================
# Argument Parser Config
# ======================
parser = get_base_parser(
	"LipGAN",
	INPUT_IMAGE_PATH,
	SAVE_IMAGE_PATH,
)
parser.add_argument(
	"--audio", default=INPUT_AUDIO_PATH, help="Path to input audio"
)
parser.add_argument(
	"--onnx",
	action="store_true",
	help="Execute Onnx Runtime mode.",
)
parser.add_argument(
	"--use_dlib",
	action="store_true",
	help="Use dlib models for inference.",
)
parser.add_argument(
	"--merge_audio",
	action="store_true",
	help="Merge audio file to video. Require ffmpeg.",
)
parser.add_argument(
	"--ailia_audio",
	action="store_true",
	help="Use ailia audio.",
)
parser.add_argument(
	"--realesrgan",
	action="store_true",
	help="Use realesrgan.",
)
parser.add_argument(
    "--cui",
    action="store_true",
    help="Don't display realtime preview in GUI."
)
args = update_parser(parser)

if args.use_dlib:
	import dlib, subprocess

# ======================
# Face Detection
# ======================

def detect_face_using_blazeface(images, blazeface): # image is rgb order
	FACE_DETECTOR_IMAGE_HEIGHT = 128
	FACE_DETECTOR_IMAGE_WIDTH = 128

	results = []
	for image in images:
		data = np.array(image)
		data = cv2.resize(
			data, (FACE_DETECTOR_IMAGE_WIDTH, FACE_DETECTOR_IMAGE_HEIGHT)
		)
		data = data / 127.5 - 1.0
		data = data.transpose((2, 0, 1))  # channel first
		data = data[np.newaxis, :, :, :].astype(
			np.float32
		)  # (batch_size, channel, h, w)

		preds_ailia = blazeface.predict([data])

		# postprocessing
		detected = but.postprocess(
			preds_ailia,
			anchor_path= "../../face_detection/blazeface/anchors.npy",
		)[0][0]
		ymin = int(detected[0] * image.shape[0])
		xmin = int(detected[1] * image.shape[1])
		ymax = int(detected[2] * image.shape[0])
		xmax = int(detected[3] * image.shape[1])
		results.append([xmin, ymin, xmax - xmin, ymax - ymin])
	return [results]
	
def rect_to_bb(d):
	x = d.rect.left()
	y = d.rect.top()
	w = d.rect.right() - x
	h = d.rect.bottom() - y
	return (x, y, w, h)

def calcMaxArea(rects, blazeface):
	max_cords = (-1,-1,-1,-1)
	max_area = 0
	max_rect = None
	for i in range(len(rects)):
		cur_rect = rects[i]
		if blazeface == None:
			(x,y,w,h) = rect_to_bb(cur_rect)
		else:
			(x,y,w,h) = (cur_rect[0], cur_rect[1], cur_rect[2], cur_rect[3])
		if w*h > max_area:
			max_area = w*h
			max_cords = (x,y,w,h)
			max_rect = cur_rect	
	return max_cords, max_rect
	
def face_detect(images, blazeface):
	if blazeface == None:
		detector = dlib.cnn_face_detection_model_v1(DLIB_WEIGHT_PATH)

	batch_size = face_det_batch_size

	predictions = []
	for i in range(0, len(images), batch_size):
		if blazeface == None:
			predictions.extend(detector(images[i:i + batch_size]))
		else:
			predictions.extend(detect_face_using_blazeface(images[i:i + batch_size], blazeface))

	results = []
	pady1, pady2, padx1, padx2 = pads
	for rects, image in zip(predictions, images):
		(x, y, w, h), max_rect = calcMaxArea(rects, blazeface)
		if x == -1:
			results.append([None, (-1,-1,-1,-1), False])
			continue
		y1 = max(0, y + pady1)
		y2 = min(image.shape[0], y + h + pady2)
		x1 = max(0, x + padx1)
		x2 = min(image.shape[1], x + w + padx2)
		face = image[y1:y2, x1:x2, ::-1] # RGB ---> BGR

		results.append([face, (y1, y2, x1, x2), True])
	
	if blazeface == None:
		del detector # make sure to clear GPU memory for LipGAN inference
	return results 


# ======================
# Functions
# ======================

# If video, until how many seconds of the clip to use for inference?
max_sec = 240.

# FPS of input video, ignore if image
fps = 25

# Padding (top, bottom, left, right)
pads = [0, 0, 0, 0]

# Single GPU batch size for face detection
face_det_batch_size = 1

# Single GPU batch size for LipGAN
lipgan_batch_size = 1

# Parameters
mel_step_size = 27
mel_idx_multiplier = 80./fps

def center_crop(image):
	center = image.shape
	w = min(center[0], center[1])
	h = w
	x = center[1]/2 - w/2
	y = center[0]/2 - h/2
	return image[int(y):int(y+h), int(x):int(x+w)]

def super_resolution(p, realesrgan):
	img = p / 255
	img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
	img = np.transpose(img, (2, 0, 1))
	img = np.expand_dims(img, 0)
	output_img = realesrgan.run(img)[0]
	output_img = np.squeeze(output_img)
	output_img = np.clip(output_img, 0, 1)
	p = np.transpose(output_img[[2, 1, 0], :, :], (1, 2, 0)) * 255
	return p

def composite(frame, p, coords):
	y1, y2, x1, x2 = coords
	p = cv2.resize(p, (x2 - x1, y2 - y1))

	blend_boundary = True
	if blend_boundary:
		alpha = np.zeros((p.shape[0], p.shape[1], 1))
		rx = alpha.shape[1] // 16
		ry = alpha.shape[0] // 16
		for y in range(alpha.shape[0]):
			for x in range(alpha.shape[1]):
				dx = min(min(x, alpha.shape[1] - 1 - x) / rx, 1)
				dy = max(min(min(y - alpha.shape[0] // 2, alpha.shape[0] - 1 - y) / ry, 1), 0)
				alpha[y, x, 0] = min(dx, dy)
		if args.debug:
			cv2.imwrite("alpha.png", (alpha * 255).astype(np.uint8))

	f = frame.copy()
	f[y1:y2, x1:x2] = f[y1:y2, x1:x2] * (1 - alpha) + p * alpha
	return f

def infer(frame, mel_chunk, ailia_net, blazeface, realesrgan):
	face_det_results = face_detect([frame[...,::-1]], blazeface) # BGR2RGB for CNN face detection

	face, coords, valid_frame = face_det_results[0].copy()
	if not valid_frame:
		print ("Face not detected, skipping frame {}".format(i))
		return frame

	face = cv2.resize(face, (IMG_SIZE, IMG_SIZE))

	img_batch, mel_batch = np.asarray([face]), np.asarray([mel_chunk])

	img_masked = img_batch.copy()
	img_masked[:, IMG_SIZE//2:] = 0

	if args.debug:
		output = np.zeros((96, 96 * 3, 3))
		output[0:96, 0:96, :] = img_batch[0]
		output[0:96, 96:96*2, :] = img_masked[0]

	img_batch = np.concatenate((img_batch, img_masked), axis=3) / 255.
	mel_batch = np.reshape(mel_batch, [len(mel_batch), mel_batch.shape[1], mel_batch.shape[2], 1])

	if args.benchmark:
		start = int(round(time.time() * 1000))
	pred = ailia_net.run([mel_batch, img_batch])[0]
	if args.benchmark:
		end = int(round(time.time() * 1000))
		print(f'\tailia lipgan processing time {end - start} ms')
	pred = pred * 255
	p = pred[0]

	if args.debug:
		output[0:96, 96*2:96*3, :] = p
		cv2.imshow("output", output.astype(np.uint8))
		cv2.imwrite("temp.jpg", output.astype(np.uint8))
	
	if args.realesrgan:
		if args.benchmark:
			start = int(round(time.time() * 1000))
		img = super_resolution(p, realesrgan)
		if args.benchmark:
			end = int(round(time.time() * 1000))
			print(f'\tailia realesrgan processing time {end - start} ms')

	f = composite(frame, p, coords)

	return f

def recognize(static, ailia_net, blazeface, realesrgan):
	# prepare input audio
	wav = librosa.load(args.audio, sr=16000)[0]
	mel = audio.melspectrogram(wav, args.ailia_audio)

	if np.isnan(mel.reshape(-1)).sum() > 0:
		raise ValueError('Mel contains nan!')

	mel_chunks = []
	i = 0
	while 1:
		start_idx = int(i * mel_idx_multiplier)
		if start_idx + mel_step_size > len(mel[0]):
			break
		mel_chunks.append(mel[:, start_idx : start_idx + mel_step_size])
		i += 1
	
	# process video
	if static:
		full_frames = [imread(args.input[0])]
	else:
		video_stream = get_capture(args.video)

	show_gui = not args.cui or not static
	frame_shown = False
	for i in range(len(mel_chunks)):
		if static:
			frame = full_frames[0]
			frames = full_frames
			ret = True
		else:
			ret, frame = video_stream.read()
			frame = center_crop(frame)
			frames = [frame]
			full_frames = frames

		if (show_gui and (cv2.waitKey(1) & 0xFF == ord('q'))) or not ret:
			break
		if frame_shown and cv2.getWindowProperty('frame', cv2.WND_PROP_VISIBLE) == 0:
			break

		if i == 0:
			frame_h, frame_w = full_frames[0].shape[:-1]
			out = cv2.VideoWriter(args.savepath, 
									cv2.VideoWriter_fourcc(*'mp4v'), fps, (frame_w, frame_h))

		f = infer(frame, mel_chunks[i], ailia_net, blazeface, realesrgan)
		out.write(f)

		if show_gui:
			cv2.imshow("frame", f)
			frame_shown = True
		else:
			end_char = "\n" if (i == len(mel_chunks) - 1) else ""
			print(f"\rProcessed frame: {i}", end=end_char)

	out.release()

	if show_gui:
		cv2.destroyAllWindows()
	if args.merge_audio:
		command = 'ffmpeg -i {} -i {} -strict -2 -q:v 1 {}'.format(args.audio, args.savepath, args.savepath + '_audio.mp4')
		subprocess.call(command, shell=True)

	logger.info('Script finished successfully.')

def main():
	# Check model files and download
	check_and_download_models(
		WEIGHT_PATH,
		MODEL_PATH,
		REMOTE_PATH,
	)
	ailia_net = ailia.Net(weight=WEIGHT_PATH, env_id = args.env_id, memory_mode = args.memory_mode)

	blazeface = None
	if not args.use_dlib:
		check_and_download_models(
			FACE_DETECTOR_WEIGHT_PATH,
			FACE_DETECTOR_MODEL_PATH,
			FACE_DETECTOR_REMOTE_PATH,
		)
		blazeface = ailia.Net(FACE_DETECTOR_MODEL_PATH, FACE_DETECTOR_WEIGHT_PATH, env_id = args.env_id, memory_mode = args.memory_mode)
	else:
		check_and_download_models(
			DLIB_WEIGHT_PATH,
			None,
			FACE_DETECTOR_REMOTE_PATH,
		)

	realesrgan = None
	if args.realesrgan:
		check_and_download_models(
			REALESRGAN_WEIGHT_PATH,
			REALESRGAN_MODEL_PATH,
			REALESRGAN_REMOTE_PATH,
		)
		realesrgan = ailia.Net(REALESRGAN_MODEL_PATH, REALESRGAN_WEIGHT_PATH, env_id = args.env_id, memory_mode = args.memory_mode)

	static = args.video is None

	recognize(static, ailia_net, blazeface, realesrgan)


if __name__ == "__main__":
	main()
