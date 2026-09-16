import os
import sys
import time

import numpy as np
import cv2
from PIL import Image

import ailia

# import original modules
sys.path.append('../../util')
from arg_utils import get_base_parser, update_parser, get_savepath  # noqa: E402
from model_utils import check_and_download_models  # noqa: E402
from webcamera_utils import get_capture, get_writer  # noqa: E402
from detector_utils import load_image  # noqa: E402

# logger
from logging import getLogger  # noqa: E402

logger = getLogger(__name__)

# ======================
# Parameters
# ======================

WEIGHT_PATH = 'hand_scoremap.onnx'
MODEL_PATH = 'hand_scoremap.onnx.prototxt'
REMOTE_PATH = 'https://storage.googleapis.com/ailia-models/hand3d/'

IMAGE_PATH = 'img.png'
SAVE_IMAGE_PATH = 'output.png'
IMAGE_HEIGHT = 240
IMAGE_WIDTH = 320

# ======================
# Arguemnt Parser Config
# ======================

parser = get_base_parser('ColorHandPose3D model', IMAGE_PATH, SAVE_IMAGE_PATH)
args = update_parser(parser)


# ======================
# Main functions
# ======================

def preprocess(img):
    img = np.array(Image.fromarray(img).resize(
        (IMAGE_WIDTH, IMAGE_HEIGHT),
        resample=Image.BILINEAR))
    img = np.expand_dims((img.astype('float') / 255.0) - 0.5, axis=0)

    return img


def recognize_from_image(net):
    # input image loop
    for image_path in args.input:
        logger.info(image_path)

        # prepare input data
        img = load_image(image_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2RGB)
        img = preprocess(img)

        logger.debug(f'input image shape: {img.shape}')
        net.set_input_shape(img.shape)

        # inference
        logger.info('Start inference...')
        if args.benchmark:
            logger.info('BENCHMARK mode')
            total_time = 0
            for i in range(args.benchmark_count):
                start = int(round(time.time() * 1000))
                output = net.predict(img)
                end = int(round(time.time() * 1000))
                logger.info(f'\tailia processing time {end - start} ms')
                if i != 0:
                    total_time = total_time + (end - start)
            logger.info(f'\taverage time {total_time / (args.benchmark_count - 1)} ms')
        else:
            output = net.predict(img)

        hand_scoremap = output[0]
        hand_scoremap = np.argmax(hand_scoremap, 2) * 128

        savepath = get_savepath(args.savepath, image_path, ext='.png')
        logger.info(f'saved at : {savepath}')
        cv2.imwrite(savepath, hand_scoremap)

    logger.info('Script finished successfully.')


def recognize_from_video(net):
    capture = get_capture(args.video)

    save_h, save_w = IMAGE_HEIGHT, IMAGE_WIDTH
    output_frame = np.zeros((save_h, save_w * 2, 3))

    # create video writer if savepath is specified as video format
    if args.savepath != SAVE_IMAGE_PATH:
        logger.warning(
            'currently, video results cannot be output correctly...'
        )
        writer = get_writer(args.savepath, save_h, save_w * 2)
    else:
        writer = None

    input_shape_set = False
    frame_shown = False
    while (True):
        ret, frame = capture.read()
        if (cv2.waitKey(1) & 0xFF == ord('q')) or not ret:
            break
        if frame_shown and cv2.getWindowProperty('frame', cv2.WND_PROP_VISIBLE) == 0:
            break

        img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = preprocess(img)

        # predict
        if (not input_shape_set):
            net.set_input_shape(img.shape)
            input_shape_set = True
        output = net.predict(img)

        hand_scoremap = output[0]
        hand_scoremap = np.argmax(hand_scoremap, 2) * 128
        res_img = hand_scoremap.astype("uint8")
        res_img = cv2.cvtColor(res_img, cv2.COLOR_GRAY2BGR)

        output_frame[:, save_w:save_w * 2, :] = res_img
        output_frame[:, 0:save_w, :] = cv2.resize(frame, (IMAGE_WIDTH, IMAGE_HEIGHT))
        output_frame = output_frame.astype("uint8")

        cv2.imshow('frame', output_frame)
        frame_shown = True

        # save results
        if writer is not None:
            writer.write(output_frame)

    capture.release()
    cv2.destroyAllWindows()
    if writer is not None:
        writer.release()
    logger.info('Script finished successfully.')


def main():
    # model files check and download
    check_and_download_models(WEIGHT_PATH, MODEL_PATH, REMOTE_PATH)

    # net initialize
    net = ailia.Net(MODEL_PATH, WEIGHT_PATH, env_id=args.env_id, memory_mode=args.memory_mode)

    if args.video is not None:
        # video mode
        recognize_from_video(net)
    else:
        # image mode
        recognize_from_image(net)


if __name__ == '__main__':
    main()
