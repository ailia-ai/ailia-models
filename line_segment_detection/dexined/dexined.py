import sys
import time

import numpy as np
import cv2

import ailia

# import original modules
sys.path.append('../../util')
from arg_utils import get_base_parser, update_parser, get_savepath  # noqa
from model_utils import check_and_download_models  # noqa
from detector_utils import load_image  # noqa
from image_utils import normalize_image  # noqa
from math_utils import sigmoid
from webcamera_utils import get_capture, get_writer  # noqa
# logger
from logging import getLogger  # noqa

logger = getLogger(__name__)

# ======================
# Parameters
# ======================

WEIGHT_PATH = 'model.onnx'
MODEL_PATH = 'model.onnx.prototxt'
REMOTE_PATH = 'https://storage.googleapis.com/ailia-models/dexined/'

IMAGE_PATH = 'RGB_008.jpg'
SAVE_IMAGE_PATH = 'output.png'

IMAGE_SIZE = 512

# ======================
# Arguemnt Parser Config
# ======================

parser = get_base_parser(
    'DexiNed', IMAGE_PATH, SAVE_IMAGE_PATH
)
args = update_parser(parser)


# ======================
# Main functions
# ======================

def preprocess(img):
    im_h, im_w, _ = img.shape

    img = cv2.resize(img, (IMAGE_SIZE, IMAGE_SIZE))

    img = img.astype(np.float32)
    img[:, :, 0] = (img[:, :, 0] - 103.939)
    img[:, :, 1] = (img[:, :, 1] - 116.779)
    img[:, :, 2] = (img[:, :, 2] - 123.68)

    img = img.transpose(2, 0, 1)  # HWC -> CHW
    img = np.expand_dims(img, axis=0)

    return img


def post_processing(output, shape):
    h, w = shape

    preds = []
    epsilon = 1e-12
    for p in output:
        img = sigmoid(p)
        img = np.squeeze(img)
        img = (img - np.min(img)) * 255 / \
              ((np.max(img) - np.min(img)) + epsilon)
        img = img.astype(np.uint8)
        img = cv2.bitwise_not(img)
        img = cv2.resize(img, (w, h))
        preds.append(img)

    fuse = preds[-1]

    ave = np.array(preds, dtype=np.float32)
    ave = np.uint8(np.mean(ave, axis=0))

    return fuse, ave


def predict(net, img):
    h, w = img.shape[:2]

    img = preprocess(img)

    # feedforward
    output = net.predict([img])

    fuse, ave = post_processing(output, (h, w))

    return fuse, ave


def recognize_from_image(net):
    # input image loop
    for image_path in args.input:
        logger.info(image_path)

        # prepare input data
        img = load_image(image_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)

        # inference
        logger.info('Start inference...')
        if args.benchmark:
            logger.info('BENCHMARK mode')
            total_time_estimation = 0
            for i in range(args.benchmark_count):
                start = int(round(time.time() * 1000))
                out = predict(net, img)
                end = int(round(time.time() * 1000))
                estimation_time = (end - start)

                # Loggin
                logger.info(f'\tailia processing estimation time {estimation_time} ms')
                if i != 0:
                    total_time_estimation = total_time_estimation + estimation_time

            logger.info(f'\taverage time estimation {total_time_estimation / (args.benchmark_count - 1)} ms')
        else:
            out = predict(net, img)

        res_img = out[1]

        # plot result
        savepath = get_savepath(args.savepath, image_path, ext='.png')
        logger.info(f'saved at : {savepath}')
        cv2.imwrite(savepath, res_img)

    logger.info('Script finished successfully.')


def recognize_from_video(net):
    video_file = args.video if args.video else args.input[0]
    capture = get_capture(video_file)
    assert capture.isOpened(), 'Cannot capture source'

    # create video writer if savepath is specified as video format
    f_h = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    f_w = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    if args.savepath != SAVE_IMAGE_PATH:
        logger.warning(
            'currently, video results cannot be output correctly...'
        )
        writer = get_writer(args.savepath, f_h, f_w)
    else:
        writer = None

    frame_shown = False
    while True:
        ret, frame = capture.read()
        if (cv2.waitKey(1) & 0xFF == ord('q')) or not ret:
            break
        if frame_shown and cv2.getWindowProperty('frame', cv2.WND_PROP_VISIBLE) == 0:
            break

        # inference
        img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        out = predict(net, img)

        res_img = out[1]

        # show
        cv2.imshow('frame', res_img)
        frame_shown = True
        res_img = res_img.astype(np.uint8)

        # save results
        if writer is not None:
            writer.write(res_img)

    capture.release()
    cv2.destroyAllWindows()
    if writer is not None:
        writer.release()

    logger.info('Script finished successfully.')


def main():
    # model files check and download
    check_and_download_models(WEIGHT_PATH, MODEL_PATH, REMOTE_PATH)

    env_id = args.env_id

    # initialize
    net = ailia.Net(MODEL_PATH, WEIGHT_PATH, env_id=env_id, memory_mode=args.memory_mode)

    if args.video is not None:
        recognize_from_video(net)
    else:
        recognize_from_image(net)


if __name__ == '__main__':
    main()
