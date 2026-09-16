#ailia detector api sample
import os
import sys
import time

import ailia
import cv2
import numpy as np

from yolor_utils import COCO_CATEGORY, non_max_suppression_numpy, scale_coords

# import original modules
sys.path.append('../../util')
# logger
from logging import getLogger

import webcamera_utils
from detector_utils import plot_results, reverse_letterbox, write_predictions
from image_utils import imread  # noqa: E402
from model_utils import check_and_download_models
from arg_utils import get_base_parser, get_savepath, update_parser

logger = getLogger(__name__)

# ======================
# Arguemnt Parser Config
# ======================

IMAGE_PATH = 'input.jpg'
SAVE_IMAGE_PATH = 'output.jpg'

parser = get_base_parser('yolor model', IMAGE_PATH, SAVE_IMAGE_PATH)
parser.add_argument(
    '--model_name',
    default='yolor_w6',
    help='yolor_p6, yolor_w6]'
)
parser.add_argument(
    '-w', '--write_prediction',
    nargs='?',
    const='txt',
    choices=['txt', 'json'],
    type=str,
    help='Output results to txt or json file.'
)
args = update_parser(parser)
REMOTE_PATH = 'https://storage.googleapis.com/ailia-models/yolor/'
MODEL_NAME = args.model_name
WEIGHT_PATH = MODEL_NAME + ".opt.onnx"
MODEL_PATH = MODEL_NAME + ".opt.onnx.prototxt"

HEIGHT = 896
WIDTH = 1280
THRESHOLD = 0.4
IOU = 0.5

# ======================
# Main functions
# ======================
def recognize_from_image():
    env_id = args.env_id
    detector = ailia.Net(MODEL_PATH, WEIGHT_PATH, env_id=env_id, memory_mode=args.memory_mode)

    # input image loop
    for image_path in args.input:
        # prepare input data
        logger.debug(f'input image: {image_path}')
        raw_img = imread(image_path)
        img = cv2.resize(raw_img, dsize=(1280, 896))
        img = np.transpose(img, (2, 0, 1))
        img = np.expand_dims(img, 0)
        img = img / 255.0
        logger.debug(f'input image shape: {raw_img.shape}')

        # inference
        logger.info('Start inference...')
        if args.benchmark:
            logger.info('BENCHMARK mode')
            for i in range(5):
                start = int(round(time.time() * 1000))
                pred = detector.predict(img)
                end = int(round(time.time() * 1000))
                logger.info(f'\tailia processing time {end - start} ms')
        else:
            pred = detector.predict(img)

        pred = non_max_suppression_numpy(pred, THRESHOLD, IOU)
        for i, det in enumerate(pred):
            if det is not None and len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], raw_img.shape).round()
                img_size_h, img_size_w = raw_img.shape[:2]
                output = []
                # Write results
                for *xyxy, conf, cls in det:
                    xyxy = [int(v) for v in xyxy]
                    x1, y1, x2, y2 = xyxy
                    r = ailia.DetectorObject(
                        category=int(cls),
                        prob=conf,
                        x=x1 / img_size_w,
                        y=y1 / img_size_h,
                        w=(x2 - x1) / img_size_w,
                        h=(y2 - y1) / img_size_h,
                    )
                    output.append(r)

        detect_object = reverse_letterbox(output, raw_img, (raw_img.shape[0], raw_img.shape[1]))
        res_img = plot_results(detect_object, raw_img, COCO_CATEGORY)
        savepath = get_savepath(args.savepath, image_path)
        logger.info(f'saved at : {savepath}')
        cv2.imwrite(savepath, res_img)

        # write prediction
        if args.write_prediction is not None:
            ext = args.write_prediction
            pred_file = "%s.%s" % (savepath.rsplit('.', 1)[0], ext)
            write_predictions(pred_file, detect_object, raw_img, category=COCO_CATEGORY, file_type=ext)

    logger.info('Script finished successfully.')


def recognize_from_video():
    # net initialize
    env_id = args.env_id
    detector = ailia.Net(MODEL_PATH, WEIGHT_PATH, env_id=env_id, memory_mode=args.memory_mode)
    capture = webcamera_utils.get_capture(args.video)

    # create video writer if savepath is specified as video format
    if args.savepath != SAVE_IMAGE_PATH:
        logger.warning(
            'currently, video results cannot be output correctly...'
        )
        f_h = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        f_w = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        save_h, save_w = f_h, f_w
        writer = webcamera_utils.get_writer(args.savepath, save_h, save_w)
    else:
        writer = None

    frame_shown = False
    while (True):
        ret, frame = capture.read()
        if (cv2.waitKey(1) & 0xFF == ord('q')) or not ret:
            break
        if frame_shown and cv2.getWindowProperty('frame', cv2.WND_PROP_VISIBLE) == 0:
            break

        raw_img = frame
        img = cv2.resize(raw_img, dsize=(1280, 896))
        img = np.transpose(img, (2, 0, 1))
        img = np.expand_dims(img, 0)
        img = img / 255.0

        pred = detector.predict(img)
        pred = non_max_suppression_numpy(pred, THRESHOLD, IOU)
        for i, det in enumerate(pred):
            if det is not None and len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], raw_img.shape).round()
                img_size_h, img_size_w = raw_img.shape[:2]
                output = []
                # Write results
                for *xyxy, conf, cls in det:
                    xyxy = [int(v) for v in xyxy]
                    x1, y1, x2, y2 = xyxy
                    r = ailia.DetectorObject(
                        category=int(cls),
                        prob=conf,
                        x=x1 / img_size_w,
                        y=y1 / img_size_h,
                        w=(x2 - x1) / img_size_w,
                        h=(y2 - y1) / img_size_h,
                    )
                    output.append(r)

        detect_object = reverse_letterbox(output, raw_img, (raw_img.shape[0], raw_img.shape[1]))
        res_img = plot_results(detect_object, raw_img, COCO_CATEGORY)
        cv2.imshow('frame', res_img)
        frame_shown = True

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

    if args.video is not None:
        # video mode
        recognize_from_video()
    else:
        # image mode
        recognize_from_image()


if __name__ == '__main__':
    main()
