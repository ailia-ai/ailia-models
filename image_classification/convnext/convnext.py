import platform
import sys
import time

import ailia
import cv2
import numpy as np

# import original modules
sys.path.append('../../util')
# logger
from logging import getLogger  # noqa: E402

import webcamera_utils  # noqa: E402
from classifier_utils import plot_results, print_results  # noqa: E402
from image_utils import imread, load_image  # noqa: E402
from model_utils import check_and_download_models  # noqa: E402
from arg_utils import get_base_parser, update_parser  # noqa: E402

logger = getLogger(__name__)

# for ConvNeXt
import os

# ======================
# PARAMETERS 1
# ======================
IMAGE_PATH = "input_imagenet.jpg"
IMAGE_HEIGHT = 32
IMAGE_WIDTH = 32


# ======================
# Argument Parser Config
# ======================
parser = get_base_parser("ConvNeXt is ", IMAGE_PATH, None, fp16_support=False)
parser.add_argument('-m', '--model', metavar='MODEL',
                    default="base_1k", choices=['base_1k', 'small_1k', 'tiny_1k', 'cifar10'])
args = update_parser(parser)


# ==========================
# MODEL AND OTHER PARAMETERS
# ==========================
CIFAR10_MODEL_PATH   = "convnext_tiny_CIFAR-10.onnx.prototxt"
CIFAR10_WEIGHT_PATH  = "convnext_tiny_CIFAR-10.onnx"
BASE_1k_MODEL_PATH   = "convnext_base_1k_224_ema.onnx.prototxt"
BASE_1k_WEIGHT_PATH  = "convnext_base_1k_224_ema.onnx"
SMALL_1k_MODEL_PATH  = "convnext_small_1k_224_ema.onnx.prototxt"
SMALL_1k_WEIGHT_PATH = "convnext_small_1k_224_ema.onnx"
TINY_1k_MODEL_PATH   = "convnext_tiny_1k_224_ema.onnx.prototxt"
TINY_1k_WEIGHT_PATH  = "convnext_tiny_1k_224_ema.onnx"
REMOTE_PATH = "https://storage.googleapis.com/ailia-models/convnext/"


# ======================
# Main functions
# ======================
def _resize(img, size):
    h = img.shape[0]
    w = img.shape[1]
    short, long = (w, h) if w <= h else (h, w)
    requested_new_short = size if isinstance(size, int) else size[0]
    new_short, new_long = requested_new_short, int(requested_new_short * long / short)
    new_w, new_h = (new_short, new_long) if w <= h else (new_long, new_short)
    img = cv2.resize(img, dsize=(new_w, new_h))
    return img

def _center_crop(img, w, h):
    x = img.shape[1]/2 - w/2
    y = img.shape[0]/2 - h/2
    img = img[int(y):int(y+h), int(x):int(x+w)]
    return img

def _preprocess(img):
    img = _resize(img, 224)
    img = _center_crop(img, 224, 224)
    img = img / 255
    img = img.transpose((2,0,1))
    img = img[np.newaxis, :, :, :]
    return img

def recognize_from_image(net, classes):
    for image_path in args.input:
        # prepare input data
        if os.path.isfile(image_path):
            image = imread(image_path, cv2.IMREAD_COLOR)
        else:
            logger.error(f'{image_path} not found.')
            exit()

        # inference
        image = _preprocess(image)
        output = net.predict(image)

        # show results
        print_results(output, classes)

    logger.info('Script finished successfully.')


def recognize_from_video(net, classes):
    capture = webcamera_utils.get_capture(args.video)
    # create video writer if savepath is specified as video format
    if args.savepath is not None:
        f_h = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        f_w = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        writer = webcamera_utils.get_writer(args.savepath, f_h, f_w)
    else:
        writer = None

    i = 0
    while(True):
        i += 1
        ret, frame = capture.read()
        if (cv2.waitKey(1) & 0xFF == ord('q')) or not ret:
            break

        # inference
        img = _preprocess(frame)
        output = net.predict(img)
        pred_class = np.argmax(output)

        plot_results(frame, output, classes)

        cv2.imshow('frame', frame)

        # save results
        if writer is not None:
            writer.write(frame)

    capture.release()
    cv2.destroyAllWindows()
    if writer is not None:
        writer.release()

    logger.info('Script finished successfully.')

def main():
    if args.model == 'base_1k':
        # model files check and download
        check_and_download_models(BASE_1k_WEIGHT_PATH, BASE_1k_MODEL_PATH, REMOTE_PATH)
        # net initialize
        net = ailia.Net(BASE_1k_MODEL_PATH, BASE_1k_WEIGHT_PATH, env_id=args.env_id, memory_mode=args.memory_mode)
    elif args.model == 'small_1k':
        # model files check and download
        check_and_download_models(SMALL_1k_WEIGHT_PATH, SMALL_1k_MODEL_PATH, REMOTE_PATH)
        # net initialize
        net = ailia.Net(SMALL_1k_MODEL_PATH, SMALL_1k_WEIGHT_PATH, env_id=args.env_id, memory_mode=args.memory_mode)
    elif args.model == 'tiny_1k':
        # model files check and download
        check_and_download_models(TINY_1k_WEIGHT_PATH, TINY_1k_MODEL_PATH, REMOTE_PATH)
        # net initialize
        net = ailia.Net(TINY_1k_MODEL_PATH, TINY_1k_WEIGHT_PATH, env_id=args.env_id, memory_mode=args.memory_mode)
    elif args.model == 'cifar10':
        # model files check and download
        check_and_download_models(CIFAR10_WEIGHT_PATH, CIFAR10_MODEL_PATH, REMOTE_PATH)
        # net initialize
        net = ailia.Net(CIFAR10_MODEL_PATH, CIFAR10_WEIGHT_PATH, env_id=args.env_id, memory_mode=args.memory_mode)
    else:
        exit()

    if args.model in ['base_1k', 'small_1k', 'tiny_1k']:
        classes = {}
        with open('label_table.txt', 'r') as f:
             data = f.readlines()
             for d in data:
                 d = d.replace('\n', '').split('\t')
                 classes[int(d[0])] = d[2]
    else:
        classes = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

    if args.video is not None: # video mode
        recognize_from_video(net, classes)
    else: # image mode
        recognize_from_image(net, classes)

    logger.info('Script finished successfully.')

if __name__ == '__main__':
    main()
