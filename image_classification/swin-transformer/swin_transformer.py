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
from image_utils import imread, load_image  # noqa: E402
from model_utils import check_and_download_models  # noqa: E402
from arg_utils import get_base_parser, update_parser  # noqa: E402

from swin_transformer_utils import plot_results, print_results

logger = getLogger(__name__)


# ======================
# Parameters
# ======================
WEIGHT_PATH = 'swin-transformer_tiny_patch4_window7_224.onnx'
MODEL_PATH = 'swin-transformer_tiny_patch4_window7_224.onnx.prototxt'
REMOTE_PATH = 'https://storage.googleapis.com/ailia-models/swin-transformer/'

IMAGE_PATH = 'input.jpg'
IMAGE_HEIGHT = 224
IMAGE_WIDTH = 224


# ======================
# Arguemnt Parser Config
# ======================
parser = get_base_parser('Swin Transformer for Image Classification', IMAGE_PATH, None)
args = update_parser(parser)


# ======================
# Main functions
# ======================
def get_labels():
    labels = {}
    with open('label_table.txt', 'r') as f:
         data = f.readlines()
         for d in data:
             d = d.replace('\n', '').split('\t')
             labels[int(d[0])] = [d[1], d[2]]
    return labels


def recognize_from_image():
    # get labels
    labels = get_labels()

    # net initialize
    net = ailia.Net(MODEL_PATH, WEIGHT_PATH, env_id=args.env_id, memory_mode=args.memory_mode)

    # input image loop
    for i, image_path in enumerate(args.input):
        # prepare input data
        logger.info(image_path)
        im = imread(image_path)

        # resize
        h, w = im.shape[0], im.shape[1]
        if w > h:
            new_h = 224
            new_w = int(224*(w/h))
        else:
            new_h = int(224*(h/w))
            new_w = 224
        im = cv2.resize(im, dsize=(new_w, new_h))

        # center crop
        size = 224
        x = new_w/2 - size/2
        y = new_h/2 - size/2
        im = im[int(y):int(y+size), int(x):int(x+size), :]
        im = im.transpose((2, 0, 1))
        im = im[np.newaxis, :, :, :]
        input = im

        # predict
        output = net.predict(input)

        # postprocess
        output = output[0]
        print_results(output, labels)

    logger.info('Script finished successfully.')


def recognize_from_video():
    # get labels
    labels = get_labels()

    # net initialize
    net = ailia.Net(MODEL_PATH, WEIGHT_PATH, env_id=args.env_id, memory_mode=args.memory_mode)

    capture = webcamera_utils.get_capture(args.video)

    # create video writer if savepath is specified as video format
    if args.savepath is not None:
        f_h = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        f_w = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        writer = webcamera_utils.get_writer(args.savepath, f_h, f_w)
    else:
        writer = None

    frame_shown = False
    while(True):
        ret, frame = capture.read()
        if (cv2.waitKey(1) & 0xFF == ord('q')) or not ret:
            break
        if frame_shown and cv2.getWindowProperty('frame', cv2.WND_PROP_VISIBLE) == 0:
            break

        im = frame

        # resize
        h, w = im.shape[0], im.shape[1]
        if w > h:
            new_h = 224
            new_w = int(224*(w/h))
        else:
            new_h = int(224*(h/w))
            new_w = 224
        im = cv2.resize(im, dsize=(new_w, new_h))

        # center crop
        size = 224
        x = new_w/2 - size/2
        y = new_h/2 - size/2
        im = im[int(y):int(y+size), int(x):int(x+size), :]
        im = im.transpose((2, 0, 1))
        im = im[np.newaxis, :, :, :]
        input = im

        # predict
        output = net.predict(input)

        # postprocess
        output = output[0]
        plot_results(frame, output, labels)

        cv2.imshow('frame', frame)
        frame_shown = True

        # save results
        if writer is not None:
            writer.write(frame)

    capture.release()
    cv2.destroyAllWindows()
    if writer is not None:
        writer.release()
    logger.info('Script finished successfully.')


def main():
    # model files check and download
    check_and_download_models(WEIGHT_PATH, MODEL_PATH, REMOTE_PATH)

    if args.video is not None:
        # video mode
        recognize_from_video()
    else:
        # image mode
        recognize_from_image()


if __name__ == '__main__':
    main()
