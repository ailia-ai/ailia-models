import os
import sys
import time

import ailia
import cv2
import numpy as np
from matplotlib import pyplot as plt

# import original modules
sys.path.append('../../util')
sys.path.append('../../face_detection/blazeface')
# logger
from logging import getLogger  # noqa: E402

import webcamera_utils  # noqa: E402
from blazeface_utils import compute_blazeface, crop_blazeface  # noqa: E402
from image_utils import imread, load_image  # noqa: E402
from model_utils import check_and_download_models  # noqa: E402
from arg_utils import get_base_parser, get_savepath, update_parser  # noqa: E402

logger = getLogger(__name__)


# TODO Upgrade Model
# ======================
# PARAMETERS
# ======================
WEIGHT_PATH = 'resnet_facial_feature.onnx'
MODEL_PATH = 'resnet_facial_feature.onnx.prototxt'
REMOTE_PATH = \
    "https://storage.googleapis.com/ailia-models/resnet_facial_feature/"

IMAGE_PATH = 'test.png'
SAVE_IMAGE_PATH = 'output.png'
IMAGE_HEIGHT = 226
IMAGE_WIDTH = 226

FACE_WEIGHT_PATH = 'blazeface.onnx'
FACE_MODEL_PATH = 'blazeface.onnx.prototxt'
FACE_REMOTE_PATH = "https://storage.googleapis.com/ailia-models/blazeface/"
FACE_MARGIN = 1.0


# ======================
# Arguemnt Parser Config
# ======================
parser = get_base_parser(
    'kaggle facial keypoints.', IMAGE_PATH, SAVE_IMAGE_PATH
)
args = update_parser(parser)


# ======================
# Utils
# ======================
def gen_img_from_predsailia(input_data, preds_ailia):
    fig = plt.figure(figsize=(3, 3))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.imshow(input_data.reshape(IMAGE_HEIGHT, IMAGE_WIDTH))
    points = np.vstack(np.split(preds_ailia, 15)).T * 113 + 113
    ax.plot(points[0], points[1], 'o', color='red')
    return fig


# ======================
# Main functions
# ======================
def recognize_from_image():
    # net initialize
    net = ailia.Net(MODEL_PATH, WEIGHT_PATH, env_id=args.env_id, memory_mode=args.memory_mode)

    # input image loop
    for image_path in args.input:
        # prepare input data
        logger.info(image_path)
        # prepare input data
        img = load_image(
            image_path,
            (IMAGE_HEIGHT, IMAGE_WIDTH),
            rgb=False,
            gen_input_ailia=True
        )

        # inference
        logger.info('Start inference...')
        if args.benchmark:
            logger.info('BENCHMARK mode')
            for i in range(5):
                start = int(round(time.time() * 1000))
                preds_ailia = net.predict(img)[0]
                end = int(round(time.time() * 1000))
                logger.info(f'\tailia processing time {end - start} ms')
        else:
            preds_ailia = net.predict(img)[0]

        # post-process
        savepath = get_savepath(args.savepath, image_path)
        fig = gen_img_from_predsailia(img, preds_ailia)
        logger.info(f'saved at : {savepath}')
        fig.savefig(savepath)
    logger.info('Script finished successfully.')


def recognize_from_video():
    # net initialize
    net = ailia.Net(MODEL_PATH, WEIGHT_PATH, env_id=args.env_id, memory_mode=args.memory_mode)
    detector = ailia.Net(FACE_MODEL_PATH, FACE_WEIGHT_PATH, env_id=args.env_id, memory_mode=args.memory_mode)

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

        # detect face
        detections = compute_blazeface(
            detector,
            frame,
            anchor_path='../../face_detection/blazeface/anchors.npy',
        )

        # get detected face
        if len(detections) == 0:
            crop_img = frame
        else:
            crop_img, top_left, bottom_right = crop_blazeface(
                detections[0], FACE_MARGIN, frame
            )
            if crop_img.shape[0] <= 0 or crop_img.shape[1] <= 0:
                crop_img = frame

        # preprocess
        input_image, input_data = webcamera_utils.preprocess_frame(
            crop_img, IMAGE_HEIGHT, IMAGE_WIDTH, data_rgb=False
        )

        # inference
        preds_ailia = net.predict(input_data)[0]

        # postprocessing
        fig = gen_img_from_predsailia(input_data, preds_ailia)
        fig.savefig('tmp.png')
        img = imread('tmp.png')
        cv2.imshow('frame', img)
        frame_shown = True

        # save results
        if writer is not None:
            img = cv2.resize(img, (IMAGE_WIDTH, IMAGE_HEIGHT))
            writer.write(img)

    capture.release()
    cv2.destroyAllWindows()
    if writer is not None:
        writer.release()
    os.remove('tmp.png')
    logger.info('Script finished successfully.')


def main():
    # model files check and download
    check_and_download_models(WEIGHT_PATH, MODEL_PATH, REMOTE_PATH)

    if args.video:
        check_and_download_models(
            FACE_WEIGHT_PATH, FACE_MODEL_PATH, FACE_REMOTE_PATH
        )

    if args.video is not None:
        # video mode
        recognize_from_video()
    else:
        # image mode
        recognize_from_image()


if __name__ == '__main__':
    main()
