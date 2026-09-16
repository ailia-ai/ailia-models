import sys
import cv2
import time
import numpy as np
from PIL import Image

import ailia
import onnxruntime

# import original modules
sys.path.append('../../util')
from arg_utils import get_base_parser, update_parser, get_savepath  # noqa: E402
from model_utils import check_and_download_models  # noqa: E402
from image_utils import imread  # noqa: E402
import webcamera_utils  # noqa: E402

# logger
from logging import getLogger   # noqa: E402
logger = getLogger(__name__)

import util 

# ======================
# Parameters 1
# ======================
IMAGE_PATH = 'input.jpg'
SAVE_IMAGE_PATH = 'output.png'

IMAGE_HEIGHT = 256
IMAGE_WIDTH = 256

# ======================
# Argument Parser Config
# ======================
parser = get_base_parser(
    'Controlling Vision-Language Models for Universal Image Restoration Official PyTorch Implementation of DA-CLIP. ', IMAGE_PATH, SAVE_IMAGE_PATH,
)
parser.add_argument('--onnx', action='store_true', help='execute onnxruntime version.')
args = update_parser(parser)


# ======================
# Parameters 2
# ======================
DACLIP_WEIGHT_PATH = "daclip" + '.onnx'
DACLIP_MODEL_PATH = DACLIP_WEIGHT_PATH + ".prototxt"

IR_WEIGHT_PATH = "universalIR" + '.onnx'
IR_MODEL_PATH = IR_WEIGHT_PATH + ".prototxt"

REMOTE_PATH = 'https://storage.googleapis.com/ailia-models/daclip-sde/'

max_sigma = 50
T  = 100
eps = 0.005

# ======================
# Main functions
# ======================

def clip_transform(np_image, resolution=224):
    def resize_image(np_image, resolution):
        return cv2.resize(np_image, (resolution, resolution), interpolation=cv2.INTER_CUBIC)
    
    def center_crop(pil_image, resolution):
        width, height = pil_image.size
        left = (width - resolution) / 2
        top = (height - resolution) / 2
        right = (width + resolution) / 2
        bottom = (height + resolution) / 2
        return pil_image.crop((left, top, right, bottom))
    
    np_image = cv2.resize(np_image, (resolution, resolution), interpolation=cv2.INTER_CUBIC)
    pil_image = Image.fromarray((np_image * 255).astype(np.uint8))
    pil_image = center_crop(pil_image, resolution)
    image = np.array(pil_image)
    image = np.transpose(image,(2,0,1))

    mean = np.array([0.48145466, 0.4578275, 0.40821073])
    std = np.array([0.26862954, 0.26130258, 0.27577711])
    for i in range(3):
        image[:, :, i] = (image[:, :, i] - mean[i]) / std[i]
    return image



def recognize_from_image(daclip, IR):
    sde = util.IRSDE(net=IR,onnx=args.onnx,max_sigma=max_sigma, T=T, eps=eps,)

    model = util.DenoisingModel()

    for image_path in args.input:
        # prepare input data
        logger.info('Input image: ' + image_path)
        
        # preprocessing
        img = imread(image_path)
        if img.ndim == 2:
            img = np.expand_dims(img, axis=2)
        c = img.shape[2]
        if c == 1:
            img = np.concatenate([img] * 3, 2)

        img = img / 255.

        def compute(image):
            img4clip = np.expand_dims(clip_transform(image),0).astype(np.float32)

            if args.onnx:
                first_input_name = daclip.get_inputs()[0].name
                image_context, degra_context = daclip.run([],{first_input_name:img4clip})
            else:
                image_context, degra_context = daclip.run(img4clip)

            image_context = image_context.astype(np.float32)
            degra_context = degra_context.astype(np.float32)

            image = np.transpose(image.astype(np.float32),(2, 0, 1))
            LQ_tensor = np.expand_dims(image,0)

            noisy_tensor = sde.noise_state(LQ_tensor)
            model.feed_data(noisy_tensor, LQ_tensor, text_context=degra_context, image_context=image_context)
            model.run(sde, logger=logger)

        # inference
        logger.info('Start inference...')
        if args.benchmark:
            logger.info('BENCHMARK mode')
            for i in range(5):
                start = int(round(time.time() * 1000))
                compute(img)
                end = int(round(time.time() * 1000))
                logger.info(f'\tailia processing time {end - start} ms')
        else:
            compute(img)

        # postprocessing

        visuals = model.get_current_visuals(need_GT=False)
        output = util.tensor2img(visuals["Output"].squeeze())
        output_img = output[:, :, [2, 1, 0]]

        savepath = get_savepath(args.savepath, image_path)
        logger.info(f'saved at : {savepath}')
        cv2.imwrite(savepath, output_img)
        
    logger.info('Script finished successfully.')

def recognize_from_video(daclip, IR):
    sde = util.IRSDE(net=IR,onnx=args.onnx,max_sigma=max_sigma, T=T, eps=eps,)

    model = util.DenoisingModel()
 
    capture = webcamera_utils.get_capture(args.video)

    # create video writer if savepath is specified as video format
    if args.savepath != SAVE_IMAGE_PATH:
        f_h = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        f_w = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        writer = webcamera_utils.get_writer(args.savepath, f_h, f_w)
    else:
        writer = None

    def compute(image):
        img4clip = np.expand_dims(clip_transform(image),0).astype(np.float32)

        if args.onnx:
            first_input_name = daclip.get_inputs()[0].name
            image_context, degra_context = daclip.run([],{first_input_name:img4clip})
        else:
            image_context, degra_context = daclip.run(img4clip)[0]

        image_context = image_context.astype(np.float32)
        degra_context = degra_context.astype(np.float32)

        image = np.transpose(image.astype(np.float32),(2, 0, 1))
        LQ_tensor = np.expand_dims(image,0)

        noisy_tensor = sde.noise_state(LQ_tensor)
        model.feed_data(noisy_tensor, LQ_tensor, text_context=degra_context, image_context=image_context)
        model.run(sde)


    frame_shown = False
    while (True): 
        ret, frame = capture.read()
        if (cv2.waitKey(1) & 0xFF == ord('q')) or not ret:
            break
        if frame_shown and cv2.getWindowProperty('output', cv2.WND_PROP_VISIBLE) == 0:
            break

        # resize with keep aspect
        frame,resized_img = webcamera_utils.adjust_frame_size(frame, IMAGE_HEIGHT, IMAGE_WIDTH)

        # preprocessing
        img = frame

        if img.ndim == 2:
            img = np.expand_dims(img, axis=2)
        c = img.shape[2]
        if c == 1:
            img = np.concatenate([img] * 3, 2)

        img = img / 255.

        compute(img)

        visuals = model.get_current_visuals(need_GT=False)
        output = util.tensor2img(visuals["Output"].squeeze())
        out_img = output[:, :, [2, 1, 0]]

        cv2.imshow('output', out_img)
        frame_shown = True

        # save results
        if writer is not None:
            writer.write(out_img)

    capture.release()
    cv2.destroyAllWindows()
    if writer is not None:
        writer.release()
    logger.info('Script finished successfully.')


def main():
    # model files check and download
    check_and_download_models(IR_WEIGHT_PATH, IR_MODEL_PATH, REMOTE_PATH)
    check_and_download_models(DACLIP_WEIGHT_PATH, DACLIP_MODEL_PATH, REMOTE_PATH)

    if args.onnx:
        daclip = onnxruntime.InferenceSession(DACLIP_WEIGHT_PATH)
        IR = onnxruntime.InferenceSession(IR_WEIGHT_PATH)
    else:
        daclip = ailia.Net(DACLIP_MODEL_PATH, DACLIP_WEIGHT_PATH, args.env_id, memory_mode=args.memory_mode)
        IR = ailia.Net(IR_MODEL_PATH, IR_WEIGHT_PATH, args.env_id, memory_mode=args.memory_mode)

    if args.video is not None:
        # video mode
        recognize_from_video(daclip, IR)
    else:
        # image mode
        recognize_from_image(daclip, IR)


if __name__ == '__main__':
    main()
