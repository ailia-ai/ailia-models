import sys, os
import time
import argparse

import numpy as np
import cv2

import ailia

# import original modules
sys.path.append('../../util')
from arg_utils import get_base_parser, update_parser, get_savepath  # noqa: E402
from model_utils import check_and_download_models  # noqa: E402
from detector_utils import plot_results, load_image  # noqa: E402C
from webcamera_utils import get_capture  # noqa: E402

from pixel_link_utils import decode_batch, mask_to_bboxes, draw_bbox, save_bboxes_to_json

# logger
from logging import getLogger   # noqa: E402
logger = getLogger(__name__)

# ======================
# Parameters
# ======================

WEIGHT_PATH = 'pixellink-vgg16-4s.onnx'
MODEL_PATH = 'pixellink-vgg16-4s.onnx.prototxt'
REMOTE_PATH = \
    'https://storage.googleapis.com/ailia-models/pixel_link/'

IMAGE_PATH = 'img_249.jpg'
SAVE_IMAGE_PATH = 'output.png'

# ======================
# Arguemnt Parser Config
# ======================

parser = get_base_parser('Pixel-Link model', IMAGE_PATH, SAVE_IMAGE_PATH)
parser.add_argument(
    '-w', '--write_json',
    action='store_true',
    help='Flag to output results to json file.'
)
args = update_parser(parser)


# ======================
# Secondaty Functions
# ======================


def post_processing(pixel_pos_scores, link_pos_scores, image_shape):
    mask = decode_batch(pixel_pos_scores, link_pos_scores)[0, ...]
    bboxes = mask_to_bboxes(mask, image_shape)

    return bboxes


# ======================
# Main functions
# ======================


def predict(img, net):
    img = img.astype(np.int32)

    # feedforward
    net.set_input_shape(img.shape)
    output = net.predict({'import/test/Placeholder:0': img})

    pixel_pos_scores, link_pos_scores = output
    bboxes = post_processing(pixel_pos_scores, link_pos_scores, img.shape)

    return bboxes


def recognize_from_image(filenames, net):
    for filename in filenames:
        # prepare input data
        img = load_image(filename)
        logger.info(f'input image shape: {img.shape}')

        img = cv2.cvtColor(img, cv2.COLOR_BGRA2RGB)

        # inference
        logger.info('Start inference...')
        if args.benchmark:
            logger.info('BENCHMARK mode')
            for i in range(5):
                start = int(round(time.time() * 1000))
                bboxes = predict(img, net)
                end = int(round(time.time() * 1000))
                logger.info(f'\tailia processing time {end - start} ms')
        else:
            bboxes = predict(img, net)

        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        res_img = draw_bbox(img, bboxes)

        # plot result
        savepath = get_savepath(args.savepath, filename)
        logger.info(f'saved at : {savepath}')
        cv2.imwrite(savepath, res_img)

        if args.write_json:
            json_file = '%s.json' % savepath.rsplit('.', 1)[0]
            save_bboxes_to_json(json_file, bboxes)

    logger.info('Script finished successfully.')


def recognize_from_video(video, net):
    capture = get_capture(video)

    frame_shown = False
    while (True):
        ret, frame = capture.read()
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        if frame_shown and cv2.getWindowProperty('frame', cv2.WND_PROP_VISIBLE) == 0:
            break
        if not ret:
            continue

        bboxes = predict(frame, net)

        # plot result
        res_img = draw_bbox(frame, bboxes)

        # show
        cv2.imshow('frame', res_img)
        frame_shown = True

    capture.release()
    cv2.destroyAllWindows()
    logger.info('Script finished successfully.')


def main():
    # model files check and download
    check_and_download_models(WEIGHT_PATH, MODEL_PATH, REMOTE_PATH)

    # load model
    env_id = args.env_id
    logger.info(f'env_id: {env_id}')

    # initialize
    net = ailia.Net(MODEL_PATH, WEIGHT_PATH, env_id=env_id, memory_mode=args.memory_mode)

    if args.video is not None:
        recognize_from_video(args.video, net)
    else:
        recognize_from_image(args.input, net)


if __name__ == '__main__':
    main()
