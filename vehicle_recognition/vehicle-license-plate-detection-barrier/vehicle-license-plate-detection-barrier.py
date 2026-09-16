import sys
import time
import random
from collections import namedtuple
import colorsys

import cv2
import json
import numpy as np

import ailia

sys.path.append('../../util')
from arg_utils import get_base_parser, update_parser, get_savepath  # noqa: E402
from model_utils import check_and_download_models  # noqa: E402
from detector_utils import load_image  # noqa: E402C
import webcamera_utils  # noqa: E402

# logger
from logging import getLogger  # noqa: E402

from this_util import *

logger = getLogger(__name__)

# ======================
# Parameters
# ======================

WEIGHT_PATH = 'vehicle-license-plate-detection-barrier-0106.onnx'
MODEL_PATH = 'vehicle-license-plate-detection-barrier-0106.onnx.prototxt'
REMOTE_PATH = 'https://storage.googleapis.com/ailia-models/vehicle-license-plate-detection-barrier/'

IMAGE_PATH = 'demo.png'
SAVE_IMAGE_PATH = 'output.png'
IMAGE_SIZE = 300
THRESHOLD = 0.5

# ======================
# Argument Parser Config
# ======================

parser = get_base_parser(
    'vehicle-license-plate-detection-barrier',
    IMAGE_PATH,
    SAVE_IMAGE_PATH,
)
parser.add_argument(
    '-th', '--threshold',
    default=THRESHOLD, type=float,
    help='The detection threshold for detection.'
)
parser.add_argument(
    '-w', '--write_json',
    action='store_true',
    help='Flag to output results to json file.'
)
args = update_parser(parser)


# ======================
# Utils
# ======================

def get_palette(n):
    rng = random.Random(0xACE)

    def dist(c1, c2):
        dh = min(abs(c1[0] - c2[0]), 1 - abs(c1[0] - c2[0])) * 2
        ds = abs(c1[1] - c2[1])
        dv = abs(c1[2] - c2[2])
        return dh * dh + ds * ds + dv * dv

    def min_distance(colors_set, color_candidate):
        distances = [dist(o, color_candidate) for o in colors_set]
        return np.min(distances)

    def hsv2rgb(h, s, v):
        return tuple(round(c * 255) for c in colorsys.hsv_to_rgb(h, s, v))

    candidates_num = 100
    hsv_colors = [(1.0, 1.0, 1.0)]

    for _ in range(1, n):
        colors_candidates = [
            (rng.random(), rng.uniform(0.8, 1.0), rng.uniform(0.5, 1.0))
            for _ in range(candidates_num)
        ]
        min_distances = [min_distance(hsv_colors, c) for c in colors_candidates]
        arg_max = np.argmax(min_distances)
        hsv_colors.append(colors_candidates[arg_max])

    palette = [hsv2rgb(*hsv) for hsv in hsv_colors]
    return palette


def draw_detections(frame, detections, palette, threshold):
    h, w = frame.shape[:2]
    for detection in detections:
        if detection.score > threshold:
            class_id = int(detection.id)
            color = palette[class_id]
            det_label = '#{}'.format(class_id)
            xmin = max(int(detection.xmin), 0)
            ymin = max(int(detection.ymin), 0)
            xmax = min(int(detection.xmax), w)
            ymax = min(int(detection.ymax), h)
            cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), color, 2)
            cv2.putText(
                frame, '{} {:.1%}'.format(det_label, detection.score),
                (xmin, ymin - 7), cv2.FONT_HERSHEY_COMPLEX, 0.6, color, 1)

    return frame


def save_result_json(file_name, frame, detections, threshold):
    h, w = frame.shape[:2]
    results = []
    for detection in detections:
        if detection.score > threshold:
            results.append({
                'class_id': detection.id,
                'score': float(detection.score),
                'xmin': float(detection.xmin),
                'ymin': float(detection.ymin),
                'xmax': float(detection.xmax),
                'ymax': float(detection.ymax)
            })
    with open(file_name, 'w') as f:
        json.dump(results, f, indent=2)


# ======================
# Main functions
# ======================

def predict(net, img):
    im_h, im_w = img.shape[:2]

    # initial preprocesses
    img = cv2.resize(img, (IMAGE_SIZE, IMAGE_SIZE))
    img = np.expand_dims(img, axis=0)

    # feedforward
    output = net.predict([img])
    conf, loc = output

    detections = detection_output(conf, loc)

    Detection = namedtuple('Detection', ['xmin', 'ymin', 'xmax', 'ymax', 'score', 'id'])
    detections = [
        Detection(xmin * im_w, ymin * im_h, xmax * im_w, ymax * im_h, score, label)
        for label, score, xmin, ymin, xmax, ymax in detections
    ]

    return detections


def recognize_from_image(net):
    threshold = args.threshold

    # input image loop
    for image_path in args.input:
        # prepare input data
        logger.info(image_path)

        img = load_image(image_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)

        img, resized_img = webcamera_utils.adjust_frame_size(img, IMAGE_SIZE, IMAGE_SIZE)

        # inference
        logger.info('Start inference...')
        if args.benchmark:
            logger.info('BENCHMARK mode')
            total_time_estimation = 0
            for i in range(args.benchmark_count):
                start = int(round(time.time() * 1000))
                detections = predict(net, img)
                end = int(round(time.time() * 1000))
                estimation_time = (end - start)

                # Loggin
                logger.info(f'\tailia processing estimation time {estimation_time} ms')
                if i != 0:
                    total_time_estimation = total_time_estimation + estimation_time

            logger.info(f'\taverage time estimation {total_time_estimation / (args.benchmark_count - 1)} ms')
        else:
            # inference
            detections = predict(net, img)

        # save results
        palette = get_palette(100)
        res_img = draw_detections(img, detections, palette, threshold)
        savepath = get_savepath(args.savepath, image_path, ext='.png')
        logger.info(f'saved at : {savepath}')
        cv2.imwrite(savepath, res_img)

        # write results
        if args.write_json:
            json_file = '%s.json' % savepath.rsplit('.', 1)[0]
            save_result_json(json_file, img, detections, threshold)

    logger.info('Script finished successfully.')


def recognize_from_video(net):
    threshold = args.threshold

    capture = webcamera_utils.get_capture(args.video)

    # create video writer if savepath is specified as video format
    if args.savepath != SAVE_IMAGE_PATH:
        f_h = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        f_w = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        writer = webcamera_utils.get_writer(args.savepath, f_h, f_w)
    else:
        writer = None

    palette = get_palette(100)
    
    frame_shown = False
    while (True):
        ret, frame = capture.read()
        if (cv2.waitKey(1) & 0xFF == ord('q')) or not ret:
            break
        if frame_shown and cv2.getWindowProperty('frame', cv2.WND_PROP_VISIBLE) == 0:
            break

        frame, resized_img = webcamera_utils.adjust_frame_size(frame, IMAGE_SIZE, IMAGE_SIZE)

        # inference
        detections = predict(net, frame)

        frame = draw_detections(frame, detections, palette, threshold)
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
