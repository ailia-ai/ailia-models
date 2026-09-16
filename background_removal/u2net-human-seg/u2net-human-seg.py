import sys
import time

import cv2
import numpy as np
from skimage import transform

import ailia

sys.path.append('../../util')
from arg_utils import get_base_parser, update_parser, get_savepath  # noqa: E402
from model_utils import check_and_download_models  # noqa: E402
from detector_utils import load_image  # noqa: E402C
import webcamera_utils  # noqa: E402

# logger
from logging import getLogger  # noqa: E402

logger = getLogger(__name__)

# ======================
# Parameters
# ======================

WEIGHT_PATH = 'u2net-human-seg.onnx'
MODEL_PATH = 'u2net-human-seg.onnx.prototxt'
REMOTE_PATH = 'https://storage.googleapis.com/ailia-models/u2net-human-seg/'

IMAGE_PATH = 'input.jpg'
SAVE_IMAGE_PATH = 'output.png'
IMAGE_SIZE = 320

# ======================
# Argument Parser Config
# ======================

parser = get_base_parser(
    'U^2-Net - human segmentation',
    IMAGE_PATH,
    SAVE_IMAGE_PATH,
)
parser.add_argument(
    '-c', '--composite',
    action='store_true',
    help='Composite input image and predicted alpha value'
)
args = update_parser(parser)


# ======================
# Utils
# ======================

def preprocess(img):
    img = transform.resize(img, (IMAGE_SIZE, IMAGE_SIZE), mode='constant')

    img = img / np.max(img)
    img[:, :, 0] = (img[:, :, 0] - 0.485) / 0.229
    img[:, :, 1] = (img[:, :, 1] - 0.456) / 0.224
    img[:, :, 2] = (img[:, :, 2] - 0.406) / 0.225
    img = img.astype(np.float32)

    img = img.transpose((2, 0, 1))
    img = np.expand_dims(img, axis=0)

    return img


# ======================
# Main functions
# ======================

def human_seg(net, img):
    h, w = img.shape[:2]

    # initial preprocesses
    img = preprocess(img)

    # feedforward
    output = net.predict([img])
    d1, d2, d3, d4, d5, d6, d7 = output
    pred = d1[:, 0, :, :]

    # post processes
    ma = np.max(pred)
    mi = np.min(pred)
    pred = (pred - mi) / (ma - mi)

    pred = pred.transpose(1, 2, 0)  # CHW -> HWC
    pred = cv2.resize(pred, (w, h), cv2.INTER_LINEAR)

    return pred


def recognize_from_image(net):
    # input image loop
    for image_path in args.input:
        # prepare input data
        logger.info(image_path)

        img = img_0 = load_image(image_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2RGB)

        # inference
        logger.info('Start inference...')
        if args.benchmark:
            logger.info('BENCHMARK mode')
            total_time_estimation = 0
            for i in range(args.benchmark_count):
                # Pose estimation
                start = int(round(time.time() * 1000))
                pred = human_seg(net, img)
                end = int(round(time.time() * 1000))
                estimation_time = (end - start)

                # Loggin
                logger.info(f'\tailia processing estimation time {estimation_time} ms')
                if i != 0:
                    total_time_estimation = total_time_estimation + estimation_time

            logger.info(f'\taverage time estimation {total_time_estimation / (args.benchmark_count - 1)} ms')
        else:
            # inference
            pred = human_seg(net, img)

        if not args.composite:
            res_img = pred * 255
        else:
            # composite
            img_0[:, :, 3] = pred * 255
            res_img = img_0

        # save results
        savepath = get_savepath(args.savepath, image_path, ext='.png')
        logger.info(f'saved at : {savepath}')
        cv2.imwrite(savepath, res_img)

    logger.info('Script finished successfully.')


def recognize_from_video(net):
    capture = webcamera_utils.get_capture(args.video)

    # create video writer if savepath is specified as video format
    if args.savepath != SAVE_IMAGE_PATH:
        f_h = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        f_w = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        writer = webcamera_utils.get_writer(args.savepath, f_h, f_w)
    else:
        writer = None

    frame_shown = False
    while (True):
        ret, frame = capture.read()
        if (cv2.waitKey(1) & 0xFF == ord('q')) or not ret:
            break
        if frame_shown and cv2.getWindowProperty('frame', cv2.WND_PROP_VISIBLE) == 0:
            break

        # inference
        img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pred = human_seg(net, img)

        # force composite
        frame[:, :, 0] = frame[:, :, 0] * pred + 64 * (1 - pred)
        frame[:, :, 1] = frame[:, :, 1] * pred + 177 * (1 - pred)
        frame[:, :, 2] = frame[:, :, 2] * pred

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
    check_and_download_models(WEIGHT_PATH, MODEL_PATH, REMOTE_PATH)

    # initialize
    net = ailia.Net(MODEL_PATH, WEIGHT_PATH, env_id=args.env_id, memory_mode=args.memory_mode)

    if args.video is not None:
        # video mode
        recognize_from_video(net)
    else:
        # image mode
        recognize_from_image(net)


if __name__ == '__main__':
    main()
