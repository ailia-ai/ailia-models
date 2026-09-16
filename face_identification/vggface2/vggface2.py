import os
import sys
import time

import numpy as np
import cv2

import ailia

# import original modules
sys.path.append('../../util')
from arg_utils import get_base_parser, update_parser  # noqa: E402
from model_utils import check_and_download_models  # noqa: E402
from image_utils import load_image  # noqa: E402
import webcamera_utils  # noqa: E402

# logger
from logging import getLogger   # noqa: E402
logger = getLogger(__name__)


# ======================
# Parameters
# ======================
WEIGHT_PATH = 'resnet50_scratch.caffemodel'
MODEL_PATH = 'resnet50_scratch.prototxt'
REMOTE_PATH = 'https://storage.googleapis.com/ailia-models/vggface2/'

IMAGE_PATH_1 = 'couple_a.jpg'
IMAGE_PATH_2 = 'couple_c.jpg'
IMAGE_HEIGHT = 224
IMAGE_WIDTH = 224

MEAN = np.array([131.0912, 103.8827, 91.4953])  # to normalize input image
THRESHOLD = 1.00  # VGGFace2 predefined value 1~1.24
SLEEP_TIME = 0  # for video input mode


# ======================
# Arguemnt Parser Config
# ======================
parser = get_base_parser(
    'Determine if the person is the same based on VGGFace2',
    None,
    None,
)
# overwrite default argument
# NOTE: vggface2  has different usage for `--input` with other models
parser.add_argument(
    '-i', '--inputs', metavar='IMAGE',
    nargs=2,
    default=[IMAGE_PATH_1, IMAGE_PATH_2],
    help='Two image paths for calculating the face match.'
)
parser.add_argument(
    '-v', '--video', metavar=('VIDEO', 'IMAGE'),
    nargs=2,
    default=None,
    help='Determines whether the face in the video file specified by VIDEO ' +
         'and the face in the image file specified by IMAGE are the same. ' +
         'If the VIDEO argument is set to 0, the webcam input will be used.'
)
args = update_parser(parser)


# ======================
# Utils
# ======================
def distance(feature1, feature2):
    norm1 = np.sqrt(np.sum(np.abs(feature1**2)))
    norm2 = np.sqrt(np.sum(np.abs(feature2**2)))
    dist = feature1/norm1-feature2/norm2
    l2_norm = np.sqrt(np.sum(np.abs(dist**2)))
    return l2_norm


def load_and_preprocess(img_path):
    img = load_image(
        img_path,
        (IMAGE_HEIGHT, IMAGE_WIDTH),
        normalize_type='None',
        gen_input_ailia=False
    )
    return preprocess(img)


def preprocess(img, input_is_bgr=False):
    if input_is_bgr:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # normalize image
    input_data = (img.astype(float) - MEAN)
    input_data = input_data.transpose((2, 0, 1))
    input_data = input_data[np.newaxis, :, :, :]
    return input_data


# ======================
# Main functions
# ======================
def compare_images():
    # net initialize
    # get_blob_data() reads an intermediate blob (conv5_3), so REUSE_INTERSTAGE cannot be used
    net = ailia.Net(MODEL_PATH, WEIGHT_PATH, env_id=args.env_id,
                    memory_mode=ailia.get_memory_mode(reduce_constant=True))

    features = []

    # prepare input data
    for j, img_path in enumerate(args.inputs):
        input_data = load_and_preprocess(img_path)

        # inference
        logger.info('Start inference...')
        if args.benchmark and j == 0:
            # Bench mark mode is only for the first image
            logger.info('BENCHMARK mode')
            for i in range(5):
                start = int(round(time.time() * 1000))
                _ = net.predict(input_data)
                end = int(round(time.time() * 1000))
                logger.info(f'\tailia processing time {end - start} ms')
        else:
            _ = net.predict(input_data)

        blob = net.get_blob_data(net.find_blob_index_by_name('conv5_3'))
        features.append(blob)

    # get result
    fname1 = os.path.basename(args.inputs[0])
    fname2 = os.path.basename(args.inputs[1])
    dist = distance(features[0], features[1])
    logger.info(f'{fname1} vs {fname2} = {dist}')

    if dist < THRESHOLD:
        logger.info('Same person')
    else:
        logger.info('Not same person')

    logger.info('Script finished successfully.')


def compare_videoframe_image():
    # net initialize
    # get_blob_data() reads an intermediate blob (conv5_3), so REUSE_INTERSTAGE cannot be used
    net = ailia.Net(MODEL_PATH, WEIGHT_PATH, env_id=args.env_id,
                    memory_mode=ailia.get_memory_mode(reduce_constant=True))

    # img part
    fname = args.video[1]
    input_data = load_and_preprocess(fname)
    _ = net.predict(input_data)
    i_feature = net.get_blob_data(net.find_blob_index_by_name('conv5_3'))

    # video part
    capture = webcamera_utils.get_capture(args.video[0])

    # create video writer if savepath is specified as video format
    if args.savepath is not None:
        writer = webcamera_utils.get_writer(
            args.savepath, IMAGE_HEIGHT, IMAGE_WIDTH
        )

    else:
        writer = None

    frame_shown = True
    while(True):
        ret, frame = capture.read()
        if (cv2.waitKey(1) & 0xFF == ord('q')) or not ret:
            break
        if frame_shown and cv2.getWindowProperty('frame', cv2.WND_PROP_VISIBLE) == 0:
            break

        _, resized_frame = webcamera_utils.adjust_frame_size(
            frame, IMAGE_HEIGHT, IMAGE_WIDTH
        )
        input_data = preprocess(resized_frame, input_is_bgr=True)

        # inference
        _ = net.predict(input_data)
        v_feature = net.get_blob_data(net.find_blob_index_by_name('conv5_3'))

        # show result
        dist = distance(i_feature, v_feature)
        logger.info('=' * 80)
        logger.info(f'{os.path.basename(fname)} vs video frame = {dist}')

        if dist < THRESHOLD:
            logger.info('Same person')
        else:
            logger.info('Not same person')
        cv2.imshow('frame', resized_frame)
        frame_shown = False
        time.sleep(SLEEP_TIME)

        # save results
        if writer is not None:
            writer.write(resized_frame)

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
        compare_videoframe_image()
    else:
        # image mode
        compare_images()


if __name__ == '__main__':
    main()
