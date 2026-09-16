import sys
import time
from dataclasses import dataclass, asdict

import cv2
import numpy as np
import json

import ailia

sys.path.append('../../util')
from arg_utils import get_base_parser, update_parser, get_savepath  # noqa: E402
from model_utils import check_and_download_models  # noqa: E402
from detector_utils import plot_results, load_image  # noqa: E402C
import webcamera_utils  # noqa: E402

from detection_utils import *

# logger
from logging import getLogger  # noqa: E402

logger = getLogger(__name__)

# ======================
# Parameters
# ======================

WEIGHT_DETECTION_PATH = 'mnv2_ssd_300_2_heads.onnx'
MODEL_DETECTION_PATH = 'mnv2_ssd_300_2_heads.onnx.prototxt'
WEIGHT_REGRESSION_PATH = 'regression_model_epoch120.onnx'
MODEL_REGRESSION_PATH = 'regression_model_epoch120.onnx.prototxt'
REMOTE_PATH = 'https://storage.googleapis.com/ailia-models/3d-object-detection.pytorch/'

IMAGE_PATH = 'demo.jpg'
SAVE_IMAGE_PATH = 'output.png'
IMAGE_DETECTION_SIZE = 300
IMAGE_REGRESSION_SIZE = 224
THRESHOLD = 0.7

OBJECTRON_CLASSES = ('bike', 'book', 'bottle', 'cereal_box', 'camera', 'chair', 'cup', 'laptop', 'shoe')


@dataclass
class IOUTrackerConfig:
    time_window: int = 10
    continue_time_thresh: int = 5
    track_clear_thresh: int = 3000
    match_threshold: float = 0.4
    track_detection_iou_thresh: float = 0.5
    interpolate_time_thresh: float = 10
    detection_filter_speed: float = 0.7
    keypoints_filter_speed: float = 0.3
    add_treshold: float = .1
    no_updated_frames_treshold: int = 5
    align_kp: bool = False


# ======================
# Argument Parser Config
# ======================

parser = get_base_parser(
    '3d-object-detection.pytorch',
    IMAGE_PATH,
    SAVE_IMAGE_PATH,
)
parser.add_argument(
    '-th', '--threshold',
    default=THRESHOLD, type=float,
    help='The detection threshold for detection.'
)
parser.add_argument(
    '-n', '--num_detect',
    default=-1, type=int,
    help='The number of objects to detect.'
)
parser.add_argument(
    '--onnx',
    action='store_true',
    help='execute onnxruntime version.'
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

def preprocess(img, shape, norm=None):
    h, w = shape
    img = cv2.resize(img, (w, h), cv2.INTER_LINEAR)
    img = img.astype(np.float32) / 255

    if norm:
        img = img - np.array([0.5931, 0.4690, 0.4229], dtype=np.float32)
        img = img / np.array([0.2471, 0.2214, 0.2157], dtype=np.float32)

    img = img.transpose((2, 0, 1))
    img = np.expand_dims(img, axis=0)

    return img


def draw_detections(img, reg_detections, det_detections, ids=None, rgb=True):
    # if image in RGB space --> convert to BGR
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR) if rgb else img

    """Draws detections and labels"""
    for det_out, reg_out, _id in zip(
            det_detections, reg_detections, ids if ids else ['ID x'] * len(reg_detections)):
        left, top, right, bottom = det_out
        kp = reg_out[0]
        label = reg_out[1]
        label = OBJECTRON_CLASSES[label]

        if _id != 'ID -1':
            cv2.rectangle(img, (left, top), (right, bottom), (0, 255, 0), thickness=2)
        else:
            cv2.rectangle(img, (left, top), (right, bottom), (100, 100, 100), thickness=2)

        if kp is not None and _id != 'ID -1':
            img = draw_kp(img, kp, normalized=False)

        label_size, base_line = cv2.getTextSize(
            label, cv2.FONT_HERSHEY_SIMPLEX, 1, 1)
        top = max(top, label_size[1])
        cv2.rectangle(
            img,
            (left, top - label_size[1]), (left + label_size[0], top + base_line),
            (255, 255, 255), cv2.FILLED)
        cv2.putText(
            img, label, (left, top),
            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0))

    return img


def save_result_json(json_path, reg_detections, det_detections, ids=None):
    results = []

    for det_out, reg_out, _id in zip(
        det_detections, reg_detections, ids if ids else ['ID x'] * len(reg_detections)):
        r = {}
        r['left'], r['top'], r['right'], r['bottom'] = det_out.tolist()
        r['keypoints'] = reg_out[0].tolist()
        r['label'] = OBJECTRON_CLASSES[reg_out[1]]
        results.append(r)

    with open(json_path, 'w') as f:
        json.dump(results, f, indent=2)


def transform_kp(kp: np.array, crop_cords: tuple):
    x0, y0, x1, y1 = crop_cords
    crop_shape = (x1 - x0, y1 - y0)
    kp[:, 0] = kp[:, 0] * crop_shape[0]
    kp[:, 1] = kp[:, 1] * crop_shape[1]
    kp[:, 0] += x0
    kp[:, 1] += y0

    return kp


# ======================
# Main functions
# ======================

def predict(det_net, reg_net, img, decode=True):
    img_0 = img
    h, w = img.shape[:2]

    num_det = args.num_detect
    threshold = args.threshold

    # initial preprocesses
    img = preprocess(img, (IMAGE_DETECTION_SIZE, IMAGE_DETECTION_SIZE))

    # feedforward
    output = det_net.predict([img])
    boxes, labels = output

    reg_detections = []
    labels = labels.astype(np.int32)
    for i in range(len(boxes)):
        box = boxes[i]
        x0, y0, x1, y1, prob = box
        if prob < threshold:
            break
        x0, x1 = x0 / IMAGE_DETECTION_SIZE, x1 / IMAGE_DETECTION_SIZE
        y0, y1 = y0 / IMAGE_DETECTION_SIZE, y1 / IMAGE_DETECTION_SIZE

        x0, x1 = int(x0 * w), int(x1 * w)
        y0, y1 = int(y0 * h), int(y1 * h)
        boxes[i] = (x0, y0, x1, y1, prob)

        img = img_0[y0:y1, x0:x1]
        img = preprocess(img, (IMAGE_REGRESSION_SIZE, IMAGE_REGRESSION_SIZE), norm=True)

        # feedforward
        if not args.onnx:
            output = reg_net.predict([img])
        else:
            output = reg_net.run(
                ["cls_bbox", "label"],
                {"data": img})

        kp, label = output
        label = np.argmax(label[0])
        kp = kp[label]

        reg_detections.append((kp, label))

        if num_det!=-1 and num_det <= len(reg_detections):
            break

    n = len(reg_detections)
    boxes = boxes[:n, :4].astype(np.int32)
    kps = [out[0].reshape(9, 2) for out in reg_detections]

    if decode:
        decoded_kps = [
            transform_kp(kp, rect)
            for kp, rect in zip(kps, boxes)
        ]
        reg_detections = [(kp, out[1]) for kp, out in zip(decoded_kps, reg_detections)]

    return reg_detections, boxes


def recognize_from_image(det_net, reg_net):
    # input image loop
    for image_path in args.input:
        # prepare input data
        logger.info(image_path)

        img = load_image(image_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2RGB)

        # inference
        logger.info('Start inference...')
        if args.benchmark:
            logger.info('BENCHMARK mode')
            total_time_estimation = 0
            for i in range(args.benchmark_count):
                # Pose estimation
                start = int(round(time.time() * 1000))
                reg_detections, boxes = predict(det_net, reg_net, img)
                end = int(round(time.time() * 1000))
                estimation_time = (end - start)

                # Loggin
                logger.info(f'\tailia processing estimation time {estimation_time} ms')
                if i != 0:
                    total_time_estimation = total_time_estimation + estimation_time

            logger.info(f'\taverage time estimation {total_time_estimation / (args.benchmark_count - 1)} ms')
        else:
            # inference
            reg_detections, boxes = predict(det_net, reg_net, img)

        # save results
        res_img = draw_detections(img, reg_detections, boxes)
        savepath = get_savepath(args.savepath, image_path, ext='.png')
        logger.info(f'saved at : {savepath}')
        cv2.imwrite(savepath, res_img)

        if args.write_json:
            json_file = '%s.json' % savepath.rsplit('.', 1)[0]
            save_result_json(json_file, reg_detections, boxes)

    logger.info('Script finished successfully.')


def recognize_from_video(det_net, reg_net):
    capture = webcamera_utils.get_capture(args.video)

    # create video writer if savepath is specified as video format
    if args.savepath != SAVE_IMAGE_PATH:
        f_h = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        f_w = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        writer = webcamera_utils.get_writer(args.savepath, f_h, f_w)
    else:
        writer = None

    sct_config = IOUTrackerConfig()
    sct_config = asdict(sct_config)
    tracker = IOUTracker(**sct_config)
    
    frame_shown = False
    while (True):
        ret, frame = capture.read()
        if (cv2.waitKey(1) & 0xFF == ord('q')) or not ret:
            break
        if frame_shown and cv2.getWindowProperty('frame', cv2.WND_PROP_VISIBLE) == 0:
            break

        # inference
        img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        reg_detections, boxes = predict(det_net, reg_net, img, decode=False)

        kps = [out[0].reshape(-1) for out in reg_detections]
        tracker.process(frame, boxes, kps)
        tracked_objects = tracker.get_tracked_objects()

        # get output from tracker
        boxes = [x.rect for x in tracked_objects]
        kps = [x.kp for x in tracked_objects]
        ids = [x.label for x in tracked_objects]

        # since we work with normilized kps within the tracker, now we should transform them back
        decoded_kps = [
            transform_kp(np.array(kp).reshape(9, 2), rect)
            for kp, rect in zip(kps, boxes)]
        reg_detections = [(kp, out[1]) for kp, out in zip(decoded_kps, reg_detections)]

        frame = draw_detections(frame, reg_detections, boxes, ids, rgb=False)
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
    logger.info("=== detection model ===")
    check_and_download_models(WEIGHT_DETECTION_PATH, MODEL_DETECTION_PATH, REMOTE_PATH)
    logger.info("=== regression model ===")
    check_and_download_models(WEIGHT_REGRESSION_PATH, MODEL_REGRESSION_PATH, REMOTE_PATH)

    # initialize
    det_net = ailia.Net(MODEL_DETECTION_PATH, WEIGHT_DETECTION_PATH, env_id=args.env_id, memory_mode=args.memory_mode)
    if args.onnx:
        import onnxruntime
        reg_net = onnxruntime.InferenceSession(WEIGHT_REGRESSION_PATH)
    else:
        reg_net = ailia.Net(MODEL_REGRESSION_PATH, WEIGHT_REGRESSION_PATH, env_id=args.env_id, memory_mode=args.memory_mode)

    if args.video is not None:
        # video mode
        recognize_from_video(det_net, reg_net)
    else:
        # image mode
        recognize_from_image(det_net, reg_net)


if __name__ == '__main__':
    main()
