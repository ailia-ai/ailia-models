import json
import sys
import time

import ailia
import cv2
import numpy as np

sys.path.append('../../util')
# logger
from logging import getLogger  # noqa: E402

import webcamera_utils  # noqa: E402
from image_utils import imread, load_image  # noqa: E402
from model_utils import check_and_download_models  # noqa: E402
from arg_utils import get_base_parser, get_savepath, update_parser  # noqa: E402
from pose_utils import pose_object_to_dict  # noqa: E402

logger = getLogger(__name__)


# ======================
# Parameters 1
# ======================
IMAGE_PATH = 'input.jpg'
SAVE_IMAGE_PATH = 'output.png'
IMAGE_HEIGHT = 240
IMAGE_WIDTH = 320

ALGORITHM = ailia.POSE_ALGORITHM_LW_HUMAN_POSE


# ======================
# Arguemnt Parser Config
# ======================
parser = get_base_parser(
    'Fast and accurate human pose 2D-estimation.', IMAGE_PATH, SAVE_IMAGE_PATH,
)
parser.add_argument(
    '-n', '--normal',
    action='store_true',
    help='By default, the optimized model is used, but with this option, ' +
    'you can switch to the normal (not optimized) model'
)
parser.add_argument(
    '-dw', '--detection_width',
    default=IMAGE_WIDTH, type=int,
    help='The detection width and height for yolo. (default: 416)'
)
parser.add_argument(
    '-dh', '--detection_height',
    default=IMAGE_HEIGHT, type=int,
    help='The detection height and height for yolo. (default: 416)'
)
parser.add_argument(
    '-w', '--write_json',
    action='store_true',
    help='save result to json'
)
args = update_parser(parser)


# ======================
# Parameters 2
# ======================
MODEL_NAME = 'lightweight-human-pose-estimation'
if args.normal:
    WEIGHT_PATH = f'{MODEL_NAME}.onnx'
    MODEL_PATH = f'{MODEL_NAME}.onnx.prototxt'
else:
    WEIGHT_PATH = f'{MODEL_NAME}.opt.onnx'
    MODEL_PATH = f'{MODEL_NAME}.opt.onnx.prototxt'
REMOTE_PATH = f'https://storage.googleapis.com/ailia-models/{MODEL_NAME}/'


# ======================
# Utils
# ======================
def hsv_to_rgb(h, s, v):
    bgr = cv2.cvtColor(
        np.array([[[h, s, v]]], dtype=np.uint8), cv2.COLOR_HSV2BGR
    )[0][0]
    return (int(bgr[2]), int(bgr[1]), int(bgr[0]))


def line(input_img, person, point1, point2):
    threshold = 0.3
    if person.points[point1].score > threshold and\
       person.points[point2].score > threshold:
        color = hsv_to_rgb(255*point1/ailia.POSE_KEYPOINT_CNT, 255, 255)

        x1 = int(input_img.shape[1] * person.points[point1].x)
        y1 = int(input_img.shape[0] * person.points[point1].y)
        x2 = int(input_img.shape[1] * person.points[point2].x)
        y2 = int(input_img.shape[0] * person.points[point2].y)
        cv2.line(input_img, (x1, y1), (x2, y2), color, 5)


def display_result(input_img, pose):
    count = pose.get_object_count()
    for idx in range(count):
        person = pose.get_object_pose(idx)

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


def save_json(pose, json_path):
    output = [
        pose_object_to_dict(pose.get_object_pose(idx))
        for idx in range(pose.get_object_count())
    ]

    with open(json_path, 'w') as f:
        json.dump(output, f, indent=2)


# ======================
# Main functions
# ======================
def recognize_from_image():
    # net initialize
    pose = ailia.PoseEstimator(
        MODEL_PATH, WEIGHT_PATH, env_id=args.env_id, algorithm=ALGORITHM
    )
    if args.detection_width!=IMAGE_WIDTH or args.detection_height!=IMAGE_HEIGHT:
        pose.set_input_shape((1,3,args.detection_height,args.detection_width))

    # input image loop
    for image_path in args.input:
        # prepare input data
        logger.info(image_path)
        # prepare input data
        src_img = imread(image_path)
        input_image = load_image(
            image_path,
            (args.detection_height, args.detection_width),
            normalize_type='None',
        )
        input_data = cv2.cvtColor(input_image, cv2.COLOR_RGB2BGRA)

        # inference
        logger.info('Start inference...')
        if args.benchmark:
            logger.info('BENCHMARK mode')
            total_time = 0
            for i in range(args.benchmark_count):
                start = int(round(time.time() * 1000))
                _ = pose.compute(input_data)
                end = int(round(time.time() * 1000))
                if i != 0:
                    total_time = total_time + (end - start)
                logger.info(f'\tailia processing time {end - start} ms')
            logger.info(f'\taverage time {total_time / (args.benchmark_count-1)} ms')
        else:
            _ = pose.compute(input_data)

        # postprocessing
        count = pose.get_object_count()
        logger.info(f'person_count={count}')
        display_result(src_img, pose)
        savepath = get_savepath(args.savepath, image_path)
        logger.info(f'saved at : {savepath}')
        cv2.imwrite(savepath, src_img)

        if args.write_json:
            save_json(pose, (savepath.rsplit('.', 1)[0]) + '.json')
    logger.info('Script finished successfully.')


def recognize_from_video():
    # net initialize
    pose = ailia.PoseEstimator(
        MODEL_PATH, WEIGHT_PATH, env_id=args.env_id, algorithm=ALGORITHM
    )
    if args.detection_width!=IMAGE_WIDTH or args.detection_height!=IMAGE_HEIGHT:
        pose.set_input_shape((1,3,args.detection_height,args.detection_width))

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

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2BGRA)

        # inference
        _ = pose.compute(frame)

        # postprocessing
        display_result(frame, pose)
        cv2.imshow('frame', frame)
        frame_shown = True

        # save results
        if writer is not None:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
            writer.write(frame)

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
