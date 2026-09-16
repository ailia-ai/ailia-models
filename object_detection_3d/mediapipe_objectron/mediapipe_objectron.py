import sys
import time

import cv2
import numpy as np
import json

import ailia

sys.path.append('../../util')
from arg_utils import get_base_parser, update_parser, get_savepath  # noqa: E402
from model_utils import check_and_download_models  # noqa: E402
from detector_utils import plot_results, load_image  # noqa: E402C
import webcamera_utils  # noqa: E402

from detectron_utils import *

# logger
from logging import getLogger  # noqa: E402

logger = getLogger(__name__)

# ======================
# Parameters
# ======================

WEIGHT_DETECTION_PATH = 'object_detection_ssd_mobilenetv2_oidv4_fp16.onnx'
MODEL_DETECTION_PATH = 'object_detection_ssd_mobilenetv2_oidv4_fp16.onnx.prototxt'
WEIGHT_SNEAKER_PATH = 'object_detection_3d_sneakers.onnx'
MODEL_SNEAKER_PATH = 'object_detection_3d_sneakers.onnx.prototxt'
WEIGHT_CHAIR_PATH = 'object_detection_3d_chair.onnx'
MODEL_CHAIR_PATH = 'object_detection_3d_chair.onnx.prototxt'
WEIGHT_CUP_PATH = 'object_detection_3d_cup.onnx'
MODEL_CUP_PATH = 'object_detection_3d_cup.onnx.prototxt'
WEIGHT_CAMERA_PATH = 'object_detection_3d_camera.onnx'
MODEL_CAMERA_PATH = 'object_detection_3d_camera.onnx.prototxt'
REMOTE_PATH = 'https://storage.googleapis.com/ailia-models/mediapipe_objectron/'

IMAGE_PATH = 'demo.jpg'
SAVE_IMAGE_PATH = 'output.png'
IMAGE_DETECTION_SIZE = 300
IMAGE_REGRESSION_SIZE = 224
THRESHOLD = 0.5

OBJECTRON_CLASSES = (
    '???',
    'Bicycle', 'Boot', 'Laptop', 'Person', 'Chair', 'Cattle',
    'Desk', 'Cat', 'Computer mouse', 'Computer monitor', 'Box', 'Mug',
    'Coffee cup', 'Stationary bicycle', 'Table', 'Bottle', 'High heels', 'Vehicle',
    'Footwear', 'Dog', 'Book', 'Camera', 'Car'
)

# ======================
# Argument Parser Config
# ======================

parser = get_base_parser(
    'MediaPipe Objectron',
    IMAGE_PATH,
    SAVE_IMAGE_PATH,
)
parser.add_argument(
    '-m', '--model', default='sneaker', choices=('sneaker', 'chair', 'cup', 'camera'),
    help='model type'
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
    '-w', '--write_json',
    action='store_true',
    help='Flag to output results to json file.'
)
args = update_parser(parser)


# ======================
# Utils
# ======================

def preprocess(img, shape, rang):
    h, w = img.shape[:2]
    src_pts = np.array([
        [0, h],
        [0, 0],
        [w, 0],
        [w, h],
    ], dtype=np.float32)
    h, w = shape
    dst_pts = np.array([
        [0, h],
        [0, 0],
        [w, 0],
        [w, h],
    ], dtype=np.float32)

    mat = cv2.getPerspectiveTransform(src_pts, dst_pts)
    img = cv2.warpPerspective(
        img, mat, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)

    scale = (rang[1] - rang[0]) / 255
    offset = rang[0]
    img = img.astype(np.float32) * scale + offset

    img = np.expand_dims(img, axis=0)

    return img


def pad_scale(img, shape, cx, cy, w, h):
    scale_x = 1.5
    scale_y = 1.5

    im_h, im_w = img.shape[:2]
    long_side = max(w * im_w, h * im_h)

    cx = cx * im_w
    cy = cy * im_h
    w = long_side * scale_x
    h = long_side * scale_y

    x = cx - w / 2
    y = cy - h / 2
    src_pts = np.array([
        [x, y + h],
        [x, y],
        [x + w, y],
        [x + w, y + h],
    ], dtype=np.float32)

    dst_h, dst_w = shape
    dst_pts = np.array([
        [0, dst_h],
        [0, 0],
        [dst_w, 0],
        [dst_w, dst_h],
    ], dtype=np.float32)

    mat = cv2.getPerspectiveTransform(src_pts, dst_pts)
    img = cv2.warpPerspective(
        img, mat, (dst_w, dst_h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)

    return img, x, y, w, h


def postprocess(boxes, predictions):
    num_classes = 24
    ignore_classes = (0,)
    sigmoid_score = True
    min_score_thresh = 0.5

    anchors = ssd_anchors()
    boxes = decode_boxes(boxes, anchors)

    detection_boxes = []
    detection_scores = []
    detection_classes = []
    for i, box in enumerate(boxes):
        class_id = -1
        max_score = -1e+12
        # Find the top score for box i.
        for score_idx in range(num_classes):
            if score_idx in ignore_classes:
                continue
            score = predictions[i, score_idx]
            if sigmoid_score:
                score = 1.0 / (1.0 + np.exp(-score))
            if max_score < score:
                max_score = score
                class_id = score_idx

        if max_score < min_score_thresh:
            continue

        detection_boxes.append(box)
        detection_scores.append(max_score)
        detection_classes.append(class_id)

    scores, boxes, classes = non_max_suppression(
        detection_scores, detection_boxes, detection_classes)

    for i, box in enumerate(boxes):
        x0, y0, x1, y1 = box
        w = x1 - x0
        h = y1 - y0
        boxes[i] = (x0 + w / 2, y0 + h / 2, w, h)

    return boxes, scores, classes


def draw_detections(img, reg_detections, det_detections, ids=None, rgb=True):
    # if image in RGB space --> convert to BGR
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR) if rgb else img

    """Draws detections and labels"""
    for det_out, reg_out, _id in zip(
            det_detections, reg_detections, ids if ids else ['ID x'] * len(reg_detections)):
        left, top, right, bottom = det_out
        kp = reg_out[0]
        label = reg_out[1]

        # if _id != 'ID -1':
        #     cv2.rectangle(img, (left, top), (right, bottom), (0, 255, 0), thickness=2)
        # else:
        #     cv2.rectangle(img, (left, top), (right, bottom), (100, 100, 100), thickness=2)

        if kp is not None and _id != 'ID -1':
            img = draw_kp(img, kp, normalized=False)

        # label_size, base_line = cv2.getTextSize(
        #     label, cv2.FONT_HERSHEY_SIMPLEX, 1, 1)
        # top = max(top, label_size[1])
        # cv2.rectangle(
        #     img,
        #     (left, top - label_size[1]), (left + label_size[0], top + base_line),
        #     (255, 255, 255), cv2.FILLED)
        # cv2.putText(
        #     img, label, (left, top),
        #     cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0))

    return img


def save_result_json(json_path, reg_detections, det_detections, ids=None):
    results = []

    for det_out, reg_out, _id in zip(
            det_detections, reg_detections, ids if ids else ['ID x'] * len(reg_detections)):
        kp = reg_out[0]
        label = reg_out[1]

        results.append({
            "det": det_out.tolist(),
            "kp": kp.tolist(),
            "label": label
        })

    with open(json_path, 'w') as f:
        json.dump(results, f, indent=2)


# ======================
# Main functions
# ======================


def predict(det_net, reg_net, img, labels=None):
    img_0 = img
    im_h, im_w = img.shape[:2]

    num_det = args.num_detect
    threshold = args.threshold

    # initial preprocesses
    img = preprocess(img, (IMAGE_DETECTION_SIZE, IMAGE_DETECTION_SIZE), (-1, 1))

    # feedforward
    output = det_net.predict([img])
    box_encodings, class_predictions = output
    boxes, scores, classes = postprocess(box_encodings[0], class_predictions[0])

    reg_detections = []
    det_detections = []
    for i, (box, score, cls) in enumerate(zip(boxes, scores, classes)):
        cx, cy, w, h = box
        b_x = (cx - w / 2) * im_w
        b_y = (cy - h / 2) * im_h
        b_w, b_h = w * im_w, h * im_h

        if score < threshold:
            break

        label = OBJECTRON_CLASSES[cls]
        if labels and label not in labels:
            continue

        img, x, y, w, h = pad_scale(
            img_0,
            (IMAGE_REGRESSION_SIZE, IMAGE_REGRESSION_SIZE),
            cx, cy, w, h)
        img = img / 255
        img = np.expand_dims(img, axis=0)

        # feedforward
        output = reg_net.predict([img])

        kp, prob = output
        kp = kp[0].reshape(9, 2)

        kp[:, 0] = kp[:, 0] * w / IMAGE_REGRESSION_SIZE + x
        kp[:, 1] = kp[:, 1] * h / IMAGE_REGRESSION_SIZE + y

        reg_detections.append((kp, label))
        det_detections.append((b_x, b_y, b_x + b_w, b_y + b_h))

        if num_det != -1 and num_det <= len(reg_detections):
            break

    det_detections = np.array(det_detections, dtype=np.int32)

    return reg_detections, det_detections


def recognize_from_image(det_net, reg_net, labels):
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
                start = int(round(time.time() * 1000))
                reg_detections, det_detections = predict(det_net, reg_net, img, labels)
                end = int(round(time.time() * 1000))
                estimation_time = (end - start)

                # Loggin
                logger.info(f'\tailia processing estimation time {estimation_time} ms')
                if i != 0:
                    total_time_estimation = total_time_estimation + estimation_time

            logger.info(f'\taverage time estimation {total_time_estimation / (args.benchmark_count - 1)} ms')
        else:
            # inference
            reg_detections, det_detections = predict(det_net, reg_net, img, labels)

        # save results
        res_img = draw_detections(img, reg_detections, det_detections)
        savepath = get_savepath(args.savepath, image_path, ext='.png')
        logger.info(f'saved at : {savepath}')
        cv2.imwrite(savepath, res_img)

        if args.write_json:
            json_file = '%s.json' % savepath.rsplit('.', 1)[0]
            save_result_json(json_file, reg_detections, det_detections)

    logger.info('Script finished successfully.')


def recognize_from_video(det_net, reg_net, labels):
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
        reg_detections, det_detections = predict(det_net, reg_net, img, labels)

        frame = draw_detections(frame, reg_detections, det_detections, rgb=False)
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
    dic_model = {
        'sneaker': (WEIGHT_SNEAKER_PATH, MODEL_SNEAKER_PATH, ('Footwear',)),
        'chair': (WEIGHT_CHAIR_PATH, MODEL_CHAIR_PATH, ('Chair',)),
        'cup': (WEIGHT_CUP_PATH, MODEL_CUP_PATH, ('Coffee cup', 'Mug')),
        'camera': (WEIGHT_CAMERA_PATH, MODEL_CAMERA_PATH, ('Camera',)),
    }
    weight_path, model_path, labels = dic_model[args.model]

    logger.info("=== detection model ===")
    check_and_download_models(WEIGHT_DETECTION_PATH, MODEL_DETECTION_PATH, REMOTE_PATH)
    logger.info("=== regression model ===")
    check_and_download_models(weight_path, model_path, REMOTE_PATH)

    env_id = args.env_id

    # initialize
    det_net = ailia.Net(MODEL_DETECTION_PATH, WEIGHT_DETECTION_PATH, env_id=env_id, memory_mode=args.memory_mode)
    reg_net = ailia.Net(model_path, weight_path, env_id=env_id, memory_mode=args.memory_mode)

    if args.video is not None:
        # video mode
        recognize_from_video(det_net, reg_net, labels)
    else:
        # image mode
        recognize_from_image(det_net, reg_net, labels)


if __name__ == '__main__':
    main()
