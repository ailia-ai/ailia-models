import sys

import time
import numpy as np
import cv2
import imageio

import ailia

# import original modules
sys.path.append('../../util')
from arg_utils import get_base_parser, update_parser, get_savepath  # noqa: E402
from model_utils import check_and_download_models  # noqa: E402
from image_utils import load_image  # noqa: E402
import webcamera_utils  # noqa: E402

# logger
from logging import getLogger   # noqa: E402
logger = getLogger(__name__)


# ======================
# Parameters 1
# ======================
IMAGE_PATH = 'input.png'
SAVE_IMAGE_PATH = 'output.png'

# ======================
# Argument Parser Config
# ======================
parser = get_base_parser(
    'Single Image Super-Resolution with HAN', IMAGE_PATH, SAVE_IMAGE_PATH, fp16_support=False
)
parser.add_argument(
    '-n', '--normal', action='store_true',
    help=('By default, the optimized model is used, but with this option, ' +
          'you can switch to the normal (not optimized) model.')
)
parser.add_argument(
    '--scale', default=2, type=int, choices=[2, 3, 4, 8],
    help=('Super-resolution scale. By default 2 (generates an image with twice the resolution).')
)
parser.add_argument(
    '--blur', action='store_true',
    help=('By default, uses the model trained on images degraded with the Bicubic (BI) Degradation Model, ' + 
          'but with this option, you can switch to the model trained on images degraded with the Blur-downscale Degradation Model (BD). ' +
          'A scale of 3 can only be used in combination with this option.')
)
args = update_parser(parser)


# ======================
# Parameters 2
# ======================
if args.blur:
    args.scale = 3
    if not args.normal:
        WEIGHT_PATH = 'han_BDX3.opt.onnx'
        MODEL_PATH = 'han_BDX3.opt.onnx.prototxt'
    else:
        WEIGHT_PATH = 'han_BDX3.onnx'
        MODEL_PATH = 'han_BDX3.onnx.prototxt'
else:
    if args.scale == 2:
        if not args.normal:
            WEIGHT_PATH = 'han_BIX2.opt.onnx'
            MODEL_PATH = 'han_BIX2.opt.onnx.prototxt'
        else:
            WEIGHT_PATH = 'han_BIX2.onnx'
            MODEL_PATH = 'han_BIX2.onnx.prototxt'
    elif args.scale == 3:
        if not args.normal:
            WEIGHT_PATH = 'han_BIX3.opt.onnx'
            MODEL_PATH = 'han_BIX3.opt.onnx.prototxt'
        else:
            WEIGHT_PATH = 'han_BIX3.onnx'
            MODEL_PATH = 'han_BIX3.onnx.prototxt'
    elif args.scale == 4:
        if not args.normal:
            WEIGHT_PATH = 'han_BIX4.opt.onnx'
            MODEL_PATH = 'han_BIX4.opt.onnx.prototxt'
        else:
            WEIGHT_PATH = 'han_BIX4.onnx'
            MODEL_PATH = 'han_BIX4.onnx.prototxt'
    elif args.scale == 8:
        if not args.normal:
            WEIGHT_PATH = 'han_BIX8.opt.onnx'
            MODEL_PATH = 'han_BIX8.opt.onnx.prototxt'
        else:
            WEIGHT_PATH = 'han_BIX8.onnx'
            MODEL_PATH = 'han_BIX8.onnx.prototxt'
    else:
        logger.info('Incorrect scale (choose from 2, 3, 4 or 8).')
        exit(-1)

REMOTE_PATH = 'https://storage.googleapis.com/ailia-models/han/'


# ======================
# Main functions
# ======================
def quantize(img, rgb_range):
    pixel_range = 255 / rgb_range
    aux = img * pixel_range
    aux = np.clip(aux, 0, 255)
    aux = np.around(aux)
    return aux / pixel_range

def recognize_from_image(net):
    # input image loop
    for image_path in args.input:
        # prepare input data
        logger.info('Input image: ' + image_path)

        # preprocessing
        img = imageio.imread(image_path)
        if img.ndim == 2:
            img = np.expand_dims(img, axis=2)
        c = img.shape[2]
        if c == 1:
            img = np.concatenate([img] * 3, 2)
        elif c == 4:
            img = img[:,:,0:3]

        img = np.ascontiguousarray(img.transpose((2, 0, 1)))
        img = img.astype(np.float32)

        img = img[np.newaxis, :, :, :] # (batch_size, channel, h, w)

        # Ailia Net input
        net.set_input_shape((1, 3, img.shape[3], img.shape[2]))

        # inference
        logger.info('Start inference...')
        if args.benchmark:
            logger.info('BENCHMARK mode')
            for i in range(5):
                start = int(round(time.time() * 1000))
                preds_ailia = net.predict(img)
                end = int(round(time.time() * 1000))
                logger.info(f'\tailia processing time {end - start} ms')
        else:
            preds_ailia = net.predict(img)

        # postprocessing
        output_img = quantize(preds_ailia[0], 255)
        output_img = output_img.astype(np.uint8).transpose(1, 2, 0)
        savepath = get_savepath(args.savepath, image_path)
        logger.info(f'saved at : {savepath}')
        imageio.imwrite(savepath, output_img)
        
    logger.info('Script finished successfully.')

def recognize_from_video(net):
    capture = webcamera_utils.get_capture(args.video)

    # create video writer if savepath is specified as video format
    if args.savepath != SAVE_IMAGE_PATH:
        f_h = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT) * int(args.scale))
        f_w = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH) * int(args.scale))
        writer = webcamera_utils.get_writer(args.savepath, f_h, f_w)
    else:
        writer = None

    frame_shown = False
    while (True):
        ret, frame = capture.read()
        if (cv2.waitKey(1) & 0xFF == ord('q')) or not ret:
            break
        if frame_shown and cv2.getWindowProperty('output', cv2.WND_PROP_VISIBLE) == 0:
            break

        IMAGE_HEIGHT, IMAGE_WIDTH = frame.shape[0], frame.shape[1]

        # resize with keep aspect
        frame,resized_img = webcamera_utils.adjust_frame_size(frame, IMAGE_HEIGHT, IMAGE_WIDTH)

        img = np.ascontiguousarray(resized_img.transpose((2, 0, 1)))
        img = img[np.newaxis, :, :, :] # (batch_size, channel, h, w)
        img = img.astype(np.float32)

        net.set_input_shape((1, 3, IMAGE_HEIGHT, IMAGE_WIDTH))

        output = net.run(img)

        out_img = quantize(output[0][0], 255)
        out_img = out_img.astype(np.uint8).transpose(1, 2, 0)
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
    check_and_download_models(WEIGHT_PATH, MODEL_PATH, REMOTE_PATH)

    # net initialize
    env_id = args.env_id
    net = ailia.Net(MODEL_PATH, WEIGHT_PATH, env_id=env_id, memory_mode=args.memory_mode)
    logger.info('Model: ' + WEIGHT_PATH[:-5])
    logger.info('Scale: ' + str(args.scale))
    
    if args.video is not None:
        # video mode
        recognize_from_video(net)
    else:
        # image mode
        recognize_from_image(net)


if __name__ == '__main__':
    main()
