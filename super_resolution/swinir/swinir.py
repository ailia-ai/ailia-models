import sys, os
import time
import argparse
import json

import numpy as np
import cv2

import ailia

# import original modules
sys.path.append('../../util')
from arg_utils import get_base_parser, update_parser, get_savepath  # noqa: E402
from model_utils import check_and_download_models  # noqa: E402
import webcamera_utils  # noqa: E402

# logger
from logging import getLogger   # noqa: E402
logger = getLogger(__name__)


# ======================
# Parameters
# ======================

# classical
IMAGE_CLASSICAL_PATH = 'input_classical.png'
WEIGHT_CLASSICAL_PATH = '001_classicalSR_DIV2K_s48w8_SwinIR-M_x2.onnx'
MODEL_CLASSICAL_PATH = '001_classicalSR_DIV2K_s48w8_SwinIR-M_x2.onnx.prototxt'
# lightweight
IMAGE_LIGHTWEIGHT_PATH = 'input_lightweight.png'
WEIGHT_LIGHTWEIGHT_PATH = '002_lightweightSR_DIV2K_s64w8_SwinIR-S_x2.onnx'
MODEL_LIGHTWEIGHT_PATH = '002_lightweightSR_DIV2K_s64w8_SwinIR-S_x2.onnx.prototxt'
# real
IMAGE_REAL_PATH = 'input_real.png'
WEIGHT_REAL_PATH = '003_realSR_BSRGAN_DFO_s64w8_SwinIR-M_x4_GAN.onnx'
MODEL_REAL_PATH = '003_realSR_BSRGAN_DFO_s64w8_SwinIR-M_x4_GAN.onnx.prototxt'
# gray
IMAGE_GRAY_PATH = 'input_gray.png'
WEIGHT_GRAY_PATH = '004_grayDN_DFWB_s128w8_SwinIR-M_noise25.onnx'
MODEL_GRAY_PATH = '004_grayDN_DFWB_s128w8_SwinIR-M_noise25.onnx.prototxt'
# color
IMAGE_COLOR_PATH = 'input_color.png'
WEIGHT_COLOR_PATH = '005_colorDN_DFWB_s128w8_SwinIR-M_noise25.onnx'
MODEL_COLOR_PATH = '005_colorDN_DFWB_s128w8_SwinIR-M_noise25.onnx.prototxt'
# jpeg
IMAGE_JPEG_PATH = 'input_jpeg.jpeg'
WEIGHT_JPEG_PATH = '006_CAR_DFWB_s126w7_SwinIR-M_jpeg10.onnx'
MODEL_JPEG_PATH = '006_CAR_DFWB_s126w7_SwinIR-M_jpeg10.onnx.prototxt'

SAVE_IMAGE_PATH = 'output.png'
REMOTE_PATH = 'https://storage.googleapis.com/ailia-models/swinir/'


# ======================
# Arguemnt Parser Config
# ======================

parser = get_base_parser(
    'SwinIR: Image Restoration Using Swin Transformer', IMAGE_CLASSICAL_PATH, SAVE_IMAGE_PATH, large_model=True
)
parser.add_argument(
    '--onnx',
    action='store_true',
    help='execute onnxruntime version.'
)
parser.add_argument(
    '--model_name',
    default='classical',
    choices=['classical', 'lightweight', 'real', 'gray', 'color', 'jpeg']
)
args = update_parser(parser)


# ======================
# Utils
# ======================
def add_noise(img, noise_param=50):
    height, width = img.shape[0], img.shape[1]
    std = np.random.uniform(0, noise_param)
    noise = np.random.normal(0, std, (height, width, 3))
    noise_img = np.array(img) + noise
    noise_img = np.clip(noise_img, 0, 255).astype(np.uint8)
    img = noise_img
    return img


# ======================
# Main functions
# ======================

def predict(net, input):
    if not args.onnx:
        output = net.run(input)
    else:
        output = net.run(None, {'input': input})
    return output


def recognize(img_lq, net):
    img_lq = np.transpose(img_lq if img_lq.shape[2] == 1 else img_lq[:, :, [2, 1, 0]], (2, 0, 1))  # HCW-BGR to CHW-RGB
    img_lq = img_lq[np.newaxis, :, :, :]

    if args.model_name == 'jpeg':
        window_size = 7
        tile = 70
    else:
        window_size = 8
        tile = 80

    if args.model_name in ['classical', 'lightweight']:
        scale = 2
    elif args.model_name == 'real':
        scale = 4
    else:
        scale = 1

    # pad input image to be a multiple of window_size
    h_old, w_old = img_lq.shape[2], img_lq.shape[3]
    h_pad = (h_old // window_size + 1) * window_size - h_old
    w_pad = (w_old // window_size + 1) * window_size - w_old
    img_lq = np.concatenate([img_lq, np.flip(img_lq, [2])], 2)[:, :, :h_old + h_pad, :]
    img_lq = np.concatenate([img_lq, np.flip(img_lq, [3])], 3)[:, :, :, :w_old + w_pad]

    if args.model_name == 'jpeg':
        img_lq = img_lq.squeeze()
        img_lq = img_lq.transpose((1, 2, 0))
        img_lq = cv2.cvtColor(img_lq, cv2.COLOR_BGR2GRAY)
        img_lq = img_lq[np.newaxis, np.newaxis, :, :]

    # test the image tile by tile
    b, c, h, w = img_lq.shape
    tile = min(tile, h, w)
    assert tile % window_size == 0, "tile size should be a multiple of window_size"
    tile_overlap = 32 # Overlapping of different tiles #tile_overlap = args.tile_overlap
    sf = scale
    stride = tile - tile_overlap
    h_idx_list = list(range(0, h-tile, stride)) + [h-tile]
    w_idx_list = list(range(0, w-tile, stride)) + [w-tile]
    E = np.zeros([b, c, h*sf, w*sf]) #E = torch.zeros(b, c, h*sf, w*sf).type_as(img_lq)
    W = np.zeros_like(E) #W = torch.zeros_like(E)

    print('Tile lists', h_idx_list, w_idx_list)
    logger.info('Predicting...')
    for h_idx in h_idx_list:
        for w_idx in w_idx_list:
            logger.info('Predicting h_idx = {}, w_idx = {}'.format(h_idx, w_idx))
            in_patch = img_lq[..., h_idx:h_idx+tile, w_idx:w_idx+tile]
            output = predict(net, in_patch)
            out_patch = output[0]
            out_patch_mask = np.ones_like(out_patch)
            E[..., h_idx*sf:(h_idx+tile)*sf, w_idx*sf:(w_idx+tile)*sf] += out_patch
            W[..., h_idx*sf:(h_idx+tile)*sf, w_idx*sf:(w_idx+tile)*sf] += out_patch_mask
    output = E/W
    output = output.squeeze()
    if output.ndim == 3:
        output = np.transpose(output[[2, 1, 0], :, :], (1, 2, 0))  # CHW-RGB to HCW-BGR
    output = (output * 255.0).round()
    output = np.clip(output, 0, 255)
    output = output.astype(np.uint8)  # float32 to uint8

    return output


def recognize_from_image(net):
    # input image loop
    for image_path in args.input:
        # prepare input data
        logger.info(image_path)

        if args.model_name == 'gray':
            img_lq = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE).astype(np.float32) / 255.
            img_lq = np.expand_dims(img_lq, axis=2)
        else:
            img_lq = cv2.imread(image_path, cv2.IMREAD_COLOR).astype(np.float32) / 255.

        # recognize
        output = recognize(img_lq, net)

        # save
        savepath = get_savepath(args.savepath, image_path)
        logger.info(f'saved at : {savepath}')
        cv2.imwrite(savepath, output)

    logger.info('Script finished successfully.')


def recognize_from_video(net, input_size, save_size):
    capture = webcamera_utils.get_capture(args.video)

    # create video writer if savepath is specified as video format
    if args.savepath != SAVE_IMAGE_PATH:
        writer = webcamera_utils.get_writer(args.savepath, save_size[0], save_size[1])
    else:
        writer = None

    frame_shown = False
    while(True):
        ret, frame = capture.read()
        if (cv2.waitKey(1) & 0xFF == ord('q')) or not ret:
            break
        if frame_shown and cv2.getWindowProperty('frame', cv2.WND_PROP_VISIBLE) == 0:
            break

        _, resized_image = webcamera_utils.adjust_frame_size(
            frame, input_size[0], input_size[1]
        )

        if args.model_name == 'gray':
            resized_image = cv2.cvtColor(resized_image, cv2.COLOR_BGR2GRAY)
            resized_image = resized_image[:, :, np.newaxis]

        resized_image = resized_image.astype(np.float32) / 255.

        # inference
        output = recognize(resized_image, net)

        if args.model_name == 'gray':
            output = cv2.cvtColor(output, cv2.COLOR_GRAY2RGB)

        # postprocessing
        cv2.imshow('frame', output)
        frame_shown = True

        # save results
        if writer is not None:
             writer.write(output)

    capture.release()
    cv2.destroyAllWindows()
    if writer is not None:
        writer.release()
    logger.info('Script finished successfully.')


def main():
    logger.info('model_name = {}'.format(args.model_name))
    #if args.model_name in ['color', 'jpeg']:
    #    logger.info('Script finished because seleted model is too large. Please wait future update.')
    #    exit()

    # set param
    if args.model_name == 'classical':
        model_path, weight_path = MODEL_CLASSICAL_PATH, WEIGHT_CLASSICAL_PATH
    elif args.model_name == 'lightweight':
        model_path, weight_path = MODEL_LIGHTWEIGHT_PATH, WEIGHT_LIGHTWEIGHT_PATH
    elif args.model_name == 'real':
        model_path, weight_path = MODEL_REAL_PATH, WEIGHT_REAL_PATH
    elif args.model_name == 'gray':
        model_path, weight_path = MODEL_GRAY_PATH, WEIGHT_GRAY_PATH
    elif args.model_name == 'color':
        model_path, weight_path = MODEL_COLOR_PATH, WEIGHT_COLOR_PATH
    elif args.model_name == 'jpeg':
        model_path, weight_path = MODEL_JPEG_PATH, WEIGHT_JPEG_PATH

    if args.video is None:
        default_flag = (len(args.input)==1 and args.input[0]==IMAGE_CLASSICAL_PATH)
        if args.model_name == 'lightweight':
            args.input[0] = IMAGE_LIGHTWEIGHT_PATH if default_flag else args.input[0]
        elif args.model_name == 'real':
            args.input[0] = IMAGE_REAL_PATH if default_flag else args.input[0]
        elif args.model_name == 'gray':
            args.input[0] = IMAGE_GRAY_PATH if default_flag else args.input[0]
        elif args.model_name == 'color':
            args.input[0] = IMAGE_COLOR_PATH if default_flag else args.input[0]
        elif args.model_name == 'jpeg':
            args.input[0] = IMAGE_JPEG_PATH if default_flag else args.input[0]

    if args.video is not None:
        if args.model_name == 'classical':
            input_size = (256, 256) # h, w
            save_size = (528, 528) # h, w
        elif args.model_name == 'lightweight':
            input_size = (256, 256)
            save_size = (528, 528)
        elif args.model_name == 'real':
            input_size = (256, 512)
            save_size = (1056, 2080)
        elif args.model_name == 'gray':
            input_size = (256, 256)
            save_size = (264, 264)

    # check
    check_and_download_models(weight_path, model_path, REMOTE_PATH)

    # initialize
    logger.info('initializing model...')
    if not args.onnx:
        net = ailia.Net(model_path, weight_path, env_id=args.env_id, memory_mode=args.memory_mode)
    else:
        import onnxruntime
        net = onnxruntime.InferenceSession(weight_path)

    # predict
    if args.video is not None:
        # video mode
        recognize_from_video(net, input_size, save_size)
    else:
        # image mode
        recognize_from_image(net)


if __name__ == '__main__':
    main()
