import os
import sys
import subprocess
import time
import argparse

import numpy as np
import cv2

import ailia

# import original modules
sys.path.append('../restyle-encoder') #import align_crop.py
sys.path.append('../../util')
sys.path.append('../../style_transfer') # import setup for face alignement (psgan)
sys.path.append('../../style_transfer/psgan') # import preprocess for face alignement (psgan)
from arg_utils import get_base_parser, update_parser, get_savepath  # noqa: E402
from model_utils import check_and_download_models  # noqa: E402
from image_utils import load_image  # noqa: E402
import webcamera_utils  # noqa: E402
from align_crop import align_face # noqa: E402

# logger
from logging import getLogger   # noqa: E402
logger = getLogger(__name__)

# ======================
# Parameters
# ======================
ENCODER_WEIGHT_PATH = 'encoder-fp16.onnx'
ENCODER_MODEL_PATH = 'encoder.onnx.prototxt'
PRETRAINED_ENCODER_WEIGHT_PATH = 'pretrained-encoder-fp16.onnx'
PRETRAINED_ENCODER_MODEL_PATH = 'pretrained-encoder.onnx.prototxt'
DECODER_WEIGHT_PATH = 'decoder-fp16.onnx'
DECODER_MODEL_PATH = 'decoder.onnx.prototxt'
REMOTE_PATH = 'https://storage.googleapis.com/ailia-models/sam/'

FACE_ALIGNMENT_WEIGHT_PATH = "../../face_recognition/face_alignment/2DFAN-4.onnx"
FACE_ALIGNMENT_MODEL_PATH = "../../face_recognition/face_alignment/2DFAN-4.onnx.prototxt"

FACE_DETECTOR_WEIGHT_PATH = "../../face_detection/blazeface/blazeface.onnx"
FACE_DETECTOR_MODEL_PATH = "../../face_detection/blazeface/blazeface.onnx.prototxt"

FACE_ALIGNMENT_REMOTE_PATH = "https://storage.googleapis.com/ailia-models/face_alignment/"
FACE_DETECTOR_REMOTE_PATH = "https://storage.googleapis.com/ailia-models/blazeface/"

face_alignment_path = [FACE_ALIGNMENT_MODEL_PATH, FACE_ALIGNMENT_WEIGHT_PATH]
face_detector_path = [FACE_DETECTOR_MODEL_PATH, FACE_DETECTOR_WEIGHT_PATH]

IMAGE_PATH = 'img/input.jpg'
SAVE_IMAGE_PATH = 'img/output.png'
ALIGNED_PATH = 'img/aligned/'

IMAGE_HEIGHT = 1024
IMAGE_WIDTH = 1024

RESIZE_HEIGHT = 256
RESIZE_WIDTH = 256


# ======================
# Arguemnt Parser Config
# ======================
parser = get_base_parser(
    'SAM', IMAGE_PATH, SAVE_IMAGE_PATH,
)
parser.add_argument(
    '-age',
    '--target_age', 
    type=str, 
    default='10,30,50,70,90',
    help='Target age for inference. Can be comma-separated list for multiple ages.')
parser.add_argument(
    '--onnx',
    action='store_true',
    help='Inference using onnx runtime'
)
parser.add_argument(
    "--use_dlib",
    action="store_true",
    help="Use dlib models for face alignment",
)
parser.add_argument(
    "opts",
    help="Modify config options using the command-line (for psgan)",
    default=None,
    nargs=argparse.REMAINDER,
)
args = update_parser(parser)


# ======================
# Utils
# ======================

def run_on_batch(inputs, net, onnx=False):
    results_batch = {}
    encoder, pretrained_encoder, decoder = net
    latent_avg = np.load('latent_avg.npy')
    for idx, image in enumerate(inputs):
        logger.info(f'Inference {idx+1}/{len(inputs)}')
        # ailia prediction 
        if not onnx:
            codes = encoder.predict(image)
            encoded_latents = pretrained_encoder.predict(image[:, :-1, :, :])
            encoded_latents = encoded_latents + latent_avg
            codes = codes + encoded_latents
            y_hat = decoder.predict([codes])[0]
        # onnx runtime prediction
        else: 
            ort_inputs = {encoder.get_inputs()[0].name: image.astype(np.float32)}
            ort_outs = encoder.run(None, ort_inputs)
            codes = ort_outs[0]

            ort_inputs = {pretrained_encoder.get_inputs()[0].name: image[:, :-1, :, :].astype(np.float32)}
            ort_outs = pretrained_encoder.run(None, ort_inputs)
            encoded_latents = ort_outs[0]

            encoded_latents = encoded_latents + latent_avg
            codes = codes + encoded_latents

            ort_inputs = {decoder.get_inputs()[0].name: codes.astype(np.float32)}
            ort_outs = decoder.run(None, ort_inputs)
            y_hat = ort_outs[0]

        results_batch[idx] = y_hat

    return results_batch

def add_aging_channel(img, age):
	target_age = int(age) / 100  # normalize aging amount to be in range [-1,1]
	img = np.concatenate((img, target_age * np.ones((1, img.shape[1], img.shape[2]))))
	return np.expand_dims(img, 0)

def np2im(var, input=False):
    var = var.astype('float32')
    var = cv2.cvtColor(
        var.transpose(1, 2, 0),
        cv2.COLOR_RGB2BGR
    )
    if not input:
        var = ((var + 1) / 2)
    var[var < 0] = 0
    var[var > 1] = 1
    var = var * 255
    return var.astype('uint8')

def post_processing(result_batch, input_img):
    for i in range(input_img.shape[0]):
        results = [np2im(result_batch[idx][0]) for idx in range(len(result_batch))]
        # save step-by-step results side-by-side
        input_im = np2im(input_img[i], input=True)
        res = np.array(results[0])
        for idx, result in enumerate(results[1:]):
            res = np.concatenate([res, result], axis=1)
        res = np.concatenate([input_im, res], axis=1)
    
    return res

# ======================
# Main functions
# ======================
def recognize_from_image(filename, net, onnx=False): 

    # face alignment
    aligned = align_face(filename, args, face_alignment_path, face_detector_path)
    if aligned is not None:
        path = os.path.join(ALIGNED_PATH, filename.split('/')[-1])
        aligned.save(path)
    else: 
        path = filename

    input_img = load_image(
        path,
        (IMAGE_HEIGHT, IMAGE_WIDTH),
        normalize_type='255',
        gen_input_ailia=True,
    )

    input_img_resized = load_image(
        path,
        (RESIZE_HEIGHT, RESIZE_WIDTH),
        normalize_type='255',
        gen_input_ailia=True,
    )
    input_img_resized = (input_img_resized * 2) - 1

    input_batch = [add_aging_channel(input_img_resized[0], age) for age in args.target_age.split(',')]
    
    # inference
    logger.info('Start inference...')
    if args.benchmark:
        logger.info('BENCHMARK mode')
        for i in range(5):
            # ailia prediction
            if not onnx:
                start = int(round(time.time() * 1000))
                result_batch = run_on_batch(input_batch, net)
                end = int(round(time.time() * 1000))
                logger.info(f'\tailia processing time {end - start} ms')
            # onnx runtime prediction
            else:
                start = int(round(time.time() * 1000))
                result_batch = run_on_batch(input_batch, net, onnx=True)
                end = int(round(time.time() * 1000))
                logger.info(f'\tonnx runtime processing time {end - start} ms')
    else:
        # ailia prediction
        if not onnx:
            result_batch = run_on_batch(input_batch, net)
        # onnx runtime prediction
        else:
            result_batch = run_on_batch(input_batch, net, onnx=True)

    # post processing
    res = post_processing(result_batch, input_img)

    # save results       
    savepath = get_savepath(args.savepath, filename)
    logger.info(f'saved at : {savepath}')
    cv2.imwrite(savepath, res)


def recognize_from_video(filename, net):

    capture = webcamera_utils.get_capture(args.video)

    # create video writer if savepath is specified as video format
    if args.savepath != SAVE_IMAGE_PATH:
        writer = webcamera_utils.get_writer(
            args.savepath, IMAGE_HEIGHT, IMAGE_WIDTH
        )
    else:
        writer = None
    
    frame_shown = False
    while(True):
        ret, frame = capture.read()
        if (cv2.waitKey(1) & 0xFF == ord('q')) or not ret:
            break
        if frame_shown and cv2.getWindowProperty('frame', cv2.WND_PROP_VISIBLE) == 0:
            break

        # Resize by padding the perimeter.
        _, input_data = webcamera_utils.preprocess_frame(
            frame, IMAGE_HEIGHT, IMAGE_WIDTH, normalize_type='255'
        )

        resized_input = cv2.resize(input_data[0].transpose(1,2,0), (RESIZE_HEIGHT, RESIZE_WIDTH))
        resized_input = np.expand_dims(resized_input.transpose(2,0,1), axis=0)
        resized_input = (resized_input * 2) - 1

        input_batch = [add_aging_channel(resized_input[0], age) for age in args.target_age.split(',')]

        result_batch = run_on_batch(input_batch, net)
            
        # post-processing
        res_img = post_processing(result_batch, input_data)
        cv2.imshow('frame', res_img)
        frame_shown = True

        # save results
        if writer is not None:
            writer.write(res_img)

    capture.release()
    cv2.destroyAllWindows()
    if writer is not None:
        writer.release()


def main():
    # model files check and download
    check_and_download_models(ENCODER_WEIGHT_PATH, ENCODER_MODEL_PATH, REMOTE_PATH)
    check_and_download_models(PRETRAINED_ENCODER_WEIGHT_PATH, PRETRAINED_ENCODER_MODEL_PATH, REMOTE_PATH)
    check_and_download_models(DECODER_WEIGHT_PATH, DECODER_MODEL_PATH, REMOTE_PATH)
    if not args.use_dlib:
        check_and_download_models(
            FACE_ALIGNMENT_WEIGHT_PATH,
            FACE_ALIGNMENT_MODEL_PATH,
            FACE_ALIGNMENT_REMOTE_PATH
        )
        check_and_download_models(
            FACE_DETECTOR_WEIGHT_PATH,
            FACE_DETECTOR_MODEL_PATH,
            FACE_DETECTOR_REMOTE_PATH
        )

    if args.video is not None:
        # net initialize
        encoder = ailia.Net(ENCODER_MODEL_PATH, ENCODER_WEIGHT_PATH, env_id=args.env_id, memory_mode=args.memory_mode)
        pretrained_encoder = ailia.Net(PRETRAINED_ENCODER_MODEL_PATH, PRETRAINED_ENCODER_WEIGHT_PATH, env_id=args.env_id, memory_mode=args.memory_mode)
        decoder = ailia.Net(DECODER_MODEL_PATH, DECODER_WEIGHT_PATH, env_id=args.env_id, memory_mode=args.memory_mode)
        net = (encoder, pretrained_encoder, decoder)
        # video mode
        recognize_from_video(SAVE_IMAGE_PATH, net)
    else:
        # image mode
        if args.onnx:
            import onnxruntime

            # onnx runtime
            if args.benchmark:
                start = int(round(time.time() * 1000))
                encoder = onnxruntime.InferenceSession(ENCODER_WEIGHT_PATH)
                pretrained_encoder = onnxruntime.InferenceSession(PRETRAINED_ENCODER_WEIGHT_PATH)
                decoder = onnxruntime.InferenceSession(DECODER_WEIGHT_PATH)
                ort_session = (encoder, pretrained_encoder, decoder)
                end = int(round(time.time() * 1000))
                logger.info(f'\tonnx runtime initializing time {end - start} ms')
            else:
                encoder = onnxruntime.InferenceSession(ENCODER_WEIGHT_PATH)
                pretrained_encoder = onnxruntime.InferenceSession(PRETRAINED_ENCODER_WEIGHT_PATH)
                decoder = onnxruntime.InferenceSession(DECODER_WEIGHT_PATH)
                ort_session = (encoder, pretrained_encoder, decoder)
            # input image loop
            for image_path in args.input:
                recognize_from_image(image_path, ort_session, onnx=True)
        else:
            # net initialize
            if args.benchmark:
                start = int(round(time.time() * 1000))
                encoder = ailia.Net(ENCODER_MODEL_PATH, ENCODER_WEIGHT_PATH, env_id=args.env_id, memory_mode=args.memory_mode)
                pretrained_encoder = ailia.Net(PRETRAINED_ENCODER_MODEL_PATH, PRETRAINED_ENCODER_WEIGHT_PATH, env_id=args.env_id, memory_mode=args.memory_mode)
                decoder = ailia.Net(DECODER_MODEL_PATH, DECODER_WEIGHT_PATH, env_id=args.env_id, memory_mode=args.memory_mode)
                net = (encoder, pretrained_encoder, decoder)
                end = int(round(time.time() * 1000))
                logger.info(f'\tailia initializing time {end - start} ms')
            else:
                encoder = ailia.Net(ENCODER_MODEL_PATH, ENCODER_WEIGHT_PATH, env_id=args.env_id, memory_mode=args.memory_mode)
                pretrained_encoder = ailia.Net(PRETRAINED_ENCODER_MODEL_PATH, PRETRAINED_ENCODER_WEIGHT_PATH, env_id=args.env_id, memory_mode=args.memory_mode)
                decoder = ailia.Net(DECODER_MODEL_PATH, DECODER_WEIGHT_PATH, env_id=args.env_id, memory_mode=args.memory_mode)
                net = (encoder, pretrained_encoder, decoder)
            # input image loop
            for image_path in args.input:
                recognize_from_image(image_path, net)
    logger.info('Script finished successfully.')


if __name__ == '__main__':
    main()
