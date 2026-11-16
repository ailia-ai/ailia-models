import json
import sys
import time

import numpy as np
import cv2

import ailia

# import original modules
sys.path.append('../../util')
from arg_utils import get_base_parser, update_parser, get_savepath  # noqa: E402
from model_utils import check_and_download_models  # noqa: E402
from detector_utils import load_image  # noqa: E402
import webcamera_utils  # noqa: E402
from pose_utils import pose_object_to_dict  # noqa: E402
from pose_resnet_util import compute, keep_aspect  # noqa: E402

# logger
from logging import getLogger   # noqa: E402
logger = getLogger(__name__)


# ======================
# Parameters
# ======================
POSE_MODEL_NAME = 'pose_resnet_50_256x192'
POSE_WEIGHT_PATH = f'{POSE_MODEL_NAME}.onnx'
POSE_MODEL_PATH = f'{POSE_MODEL_NAME}.onnx.prototxt'
POSE_REMOTE_PATH = 'https://storage.googleapis.com/ailia-models/pose_resnet/'

WEIGHT_PATH = 'yolov3.opt.onnx'
MODEL_PATH = 'yolov3.opt.onnx.prototxt'
REMOTE_PATH = 'https://storage.googleapis.com/ailia-models/yolov3/'

IMAGE_PATH = 'input.jpg'
SAVE_IMAGE_PATH = 'output.png'

COCO_CATEGORY = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train",
    "truck", "boat", "traffic light", "fire hydrant", "stop sign",
    "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow",
    "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
    "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard",
    "sports ball", "kite", "baseball bat", "baseball glove", "skateboard",
    "surfboard", "tennis racket", "bottle", "wine glass", "cup", "fork",
    "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange",
    "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair",
    "couch", "potted plant", "bed", "dining table", "toilet", "tv",
    "laptop", "mouse", "remote", "keyboard", "cell phone", "microwave",
    "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase",
    "scissors", "teddy bear", "hair drier", "toothbrush"
]
THRESHOLD = 0.4
IOU = 0.45
POSE_THRESHOLD = 0.1


# ======================
# Arguemnt Parser Config
# ======================
parser = get_base_parser(
    'Simple Baseline for Pose Estimation', IMAGE_PATH, SAVE_IMAGE_PATH,
)
parser.add_argument(
    '-w', '--write_json',
    action='store_true',
    help='save result to json'
)
args = update_parser(parser)


# ======================
# Display result
# ======================
def hsv_to_rgb(h, s, v):
    bgr = cv2.cvtColor(
        np.array([[[h, s, v]]], dtype=np.uint8), cv2.COLOR_HSV2BGR)[0][0]
    return (int(bgr[0]), int(bgr[1]), int(bgr[2]), 255)


def line(input_img, person, point1, point2):
    threshold = POSE_THRESHOLD
    if person.points[point1].score > threshold and\
       person.points[point2].score > threshold:
        color = hsv_to_rgb(255*point1/ailia.POSE_KEYPOINT_CNT, 255, 255)

        x1 = int(input_img.shape[1] * person.points[point1].x)
        y1 = int(input_img.shape[0] * person.points[point1].y)
        x2 = int(input_img.shape[1] * person.points[point2].x)
        y2 = int(input_img.shape[0] * person.points[point2].y)
        cv2.line(input_img, (x1, y1), (x2, y2), color, 5)


def display_result(input_img, person):
    line(input_img, person, ailia.POSE_KEYPOINT_NOSE,
         ailia.POSE_KEYPOINT_SHOULDER_CENTER)
    line(input_img, person, ailia.POSE_KEYPOINT_SHOULDER_LEFT,
         ailia.POSE_KEYPOINT_SHOULDER_CENTER)
    line(input_img, person, ailia.POSE_KEYPOINT_SHOULDER_RIGHT,
         ailia.POSE_KEYPOINT_SHOULDER_CENTER)

    line(input_img, person, ailia.POSE_KEYPOINT_EYE_LEFT,
         ailia.POSE_KEYPOINT_NOSE)
    line(input_img, person, ailia.POSE_KEYPOINT_EYE_RIGHT,
         ailia.POSE_KEYPOINT_NOSE)
    line(input_img, person, ailia.POSE_KEYPOINT_EAR_LEFT,
         ailia.POSE_KEYPOINT_EYE_LEFT)
    line(input_img, person, ailia.POSE_KEYPOINT_EAR_RIGHT,
         ailia.POSE_KEYPOINT_EYE_RIGHT)

    line(input_img, person, ailia.POSE_KEYPOINT_ELBOW_LEFT,
         ailia.POSE_KEYPOINT_SHOULDER_LEFT)
    line(input_img, person, ailia.POSE_KEYPOINT_ELBOW_RIGHT,
         ailia.POSE_KEYPOINT_SHOULDER_RIGHT)
    line(input_img, person, ailia.POSE_KEYPOINT_WRIST_LEFT,
         ailia.POSE_KEYPOINT_ELBOW_LEFT)
    line(input_img, person, ailia.POSE_KEYPOINT_WRIST_RIGHT,
         ailia.POSE_KEYPOINT_ELBOW_RIGHT)

    line(input_img, person, ailia.POSE_KEYPOINT_BODY_CENTER,
         ailia.POSE_KEYPOINT_SHOULDER_CENTER)
    line(input_img, person, ailia.POSE_KEYPOINT_HIP_LEFT,
         ailia.POSE_KEYPOINT_BODY_CENTER)
    line(input_img, person, ailia.POSE_KEYPOINT_HIP_RIGHT,
         ailia.POSE_KEYPOINT_BODY_CENTER)

    line(input_img, person, ailia.POSE_KEYPOINT_KNEE_LEFT,
         ailia.POSE_KEYPOINT_HIP_LEFT)
    line(input_img, person, ailia.POSE_KEYPOINT_ANKLE_LEFT,
         ailia.POSE_KEYPOINT_KNEE_LEFT)
    line(input_img, person, ailia.POSE_KEYPOINT_KNEE_RIGHT,
         ailia.POSE_KEYPOINT_HIP_RIGHT)
    line(input_img, person, ailia.POSE_KEYPOINT_ANKLE_RIGHT,
         ailia.POSE_KEYPOINT_KNEE_RIGHT)


def pose_estimation(detector, pose, img):
    pose_img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
    h, w = img.shape[0], img.shape[1]
    count = detector.get_object_count()
    pose_detections = []
    for idx in range(count):
        obj = detector.get_object(idx)
        top_left = (int(w*obj.x), int(h*obj.y))
        bottom_right = (int(w*(obj.x+obj.w)), int(h*(obj.y+obj.h)))
        CATEGORY_PERSON = 0
        if obj.category != CATEGORY_PERSON:
            pose_detections.append(None)
            continue
        shape = pose.get_input_shape()
        crop_img, px1, py1, px2, py2, scale_x, scale_y = keep_aspect(
            top_left, bottom_right, pose_img, shape
        )
        offset_x = px1/pose_img.shape[1]
        offset_y = py1/pose_img.shape[0]
        detections = compute(
            pose, crop_img, offset_x, offset_y, scale_x, scale_y
        )
        pose_detections.append(detections)
    return pose_detections


def plot_results(detector, pose, img, category, pose_detections, logging=True):
    h, w = img.shape[0], img.shape[1]
    count = detector.get_object_count()
    if logging:
        logger.info(f'object_count={count}')

    for idx in range(count):
        obj = detector.get_object(idx)
        # print result
        if logging:
            logger.info(f'+ idx={idx}')
            logger.info(f'  category={obj.category}[ {category[obj.category]} ]')
            logger.info(f'  prob={obj.prob}')
            logger.info(f'  x={obj.x}')
            logger.info(f'  y={obj.y}')
            logger.info(f'  w={obj.w}')
            logger.info(f'  h={obj.h}')
        top_left = (int(w*obj.x), int(h*obj.y))
        bottom_right = (int(w*(obj.x+obj.w)), int(h*(obj.y+obj.h)))
        text_position = (int(w*obj.x)+4, int(h*(obj.y+obj.h)-8))

        # update image
        color = hsv_to_rgb(256 * obj.category / len(category), 255, 255)
        fontScale = w / 512.0
        cv2.rectangle(img, top_left, bottom_right, color, 4)

        cv2.putText(
            img,
            category[obj.category],
            text_position,
            cv2.FONT_HERSHEY_SIMPLEX,
            fontScale,
            color,
            1
        )

        CATEGORY_PERSON = 0
        if obj.category != CATEGORY_PERSON:
            continue

        # pose detection
        shape = pose.get_input_shape()
        crop_img, px1, py1, px2, py2, scale_x, scale_y  = keep_aspect(
            top_left, bottom_right, img, shape
        )
        detections = pose_detections[idx]
        cv2.rectangle(img, (px1, py1), (px2, py2), color, 1)
        display_result(img, detections)

    return img


def save_json(pose_detections, json_path):
    # pose_estimation() set None for non-person detections, so skip them.
    output = [
        pose_object_to_dict(detections)
        for detections in pose_detections
        if detections is not None
    ]

    with open(json_path, 'w') as f:
        json.dump(output, f, indent=2)


# ======================
# Main functions
# ======================
def recognize_from_image():
    # net initialize
    detector = ailia.Detector(
        MODEL_PATH,
        WEIGHT_PATH,
        len(COCO_CATEGORY),
        format=ailia.NETWORK_IMAGE_FORMAT_RGB,
        channel=ailia.NETWORK_IMAGE_CHANNEL_FIRST,
        range=ailia.NETWORK_IMAGE_RANGE_U_FP32,
        algorithm=ailia.DETECTOR_ALGORITHM_YOLOV3,
        env_id=args.env_id,
    )

    pose = ailia.Net(POSE_MODEL_PATH, POSE_WEIGHT_PATH, env_id=args.env_id)

    # input image loop
    for image_path in args.input:
        # prepare input data
        logger.info(image_path)
        img = load_image(image_path)
        logger.debug(f'input image shape: {img.shape}')

        # inference
        logger.info('Start inference...')
        detector.compute(img, THRESHOLD, IOU)

        # pose estimation
        if args.benchmark:
            logger.info('BENCHMARK mode')
            total_time = 0
            for i in range(args.benchmark_count):
                start = int(round(time.time() * 1000))
                pose_detections = pose_estimation(detector, pose, img)
                end = int(round(time.time() * 1000))
                logger.info(f'\tailia processing detection time {end - start} ms')
                if i != 0:
                    total_time = total_time + (end - start)
            logger.info(f'\taverage detection time {total_time / (args.benchmark_count-1)} ms')
        else:
            pose_detections = pose_estimation(detector, pose, img)

        # plot result
        res_img = plot_results(detector, pose, img, COCO_CATEGORY, pose_detections)
        savepath = get_savepath(args.savepath, image_path)
        logger.info(f'saved at : {savepath}')
        cv2.imwrite(savepath, res_img)

        if args.write_json:
            save_json(pose_detections, (savepath.rsplit('.', 1)[0]) + '.json')
    logger.info('Script finished successfully.')


def recognize_from_video():
    # net initialize
    detector = ailia.Detector(
        MODEL_PATH,
        WEIGHT_PATH,
        len(COCO_CATEGORY),
        format=ailia.NETWORK_IMAGE_FORMAT_RGB,
        channel=ailia.NETWORK_IMAGE_CHANNEL_FIRST,
        range=ailia.NETWORK_IMAGE_RANGE_U_FP32,
        algorithm=ailia.DETECTOR_ALGORITHM_YOLOV3,
        env_id=args.env_id,
    )

    pose = ailia.Net(POSE_MODEL_PATH, POSE_WEIGHT_PATH, env_id=args.env_id)

    capture = webcamera_utils.get_capture(args.video)
    # create video writer if savepath is specified as video format
    if args.savepath != SAVE_IMAGE_PATH:
        f_h = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        f_w = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        writer = webcamera_utils.get_writer(args.savepath, f_h, f_w)
    else:
        writer = None
    
    frame_shown = False
    while(True):
        ret, frame = capture.read()
        if (cv2.waitKey(1) & 0xFF == ord('q')) or not ret:
            break
        if frame_shown and cv2.getWindowProperty('frame', cv2.WND_PROP_VISIBLE) == 0:
            break

        img = cv2.cvtColor(frame, cv2.COLOR_BGR2BGRA)
        detector.compute(img, THRESHOLD, IOU)
        pose_detections = pose_estimation(detector, pose, img)
        res_img = plot_results(detector, pose, frame, COCO_CATEGORY, pose_detections, logging=False)
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
    check_and_download_models(
        POSE_WEIGHT_PATH, POSE_MODEL_PATH, POSE_REMOTE_PATH
    )

    if args.video is not None:
        # video mode
        recognize_from_video()
    else:
        # image mode
        recognize_from_image()


if __name__ == '__main__':
    main()
