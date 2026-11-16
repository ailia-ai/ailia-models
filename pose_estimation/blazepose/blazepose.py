import sys
import time
import json

import ailia
import cv2
import numpy as np
from scipy.special import expit

import blazepose_utils as but

sys.path.append('../../util')
# logger
from logging import getLogger  # noqa: E402

import webcamera_utils  # noqa: E402
from image_utils import imread  # noqa: E402
from model_utils import check_and_download_models  # noqa: E402
from arg_utils import get_base_parser, get_savepath, update_parser  # noqa: E402
from dtype_utils import numpy_type_to_builtin_type  # noqa: E402

logger = getLogger(__name__)


# ======================
# Parameters 1
# ======================
IMAGE_PATH = 'girl-5204299_640.jpg'
SAVE_IMAGE_PATH = 'output.png'
IMAGE_HEIGHT = 256
IMAGE_WIDTH = 256


# ======================
# Argument Parser Config
# ======================

parser = get_base_parser(
    'BlazePose, an on-device real-time body pose tracking.',
    IMAGE_PATH,
    SAVE_IMAGE_PATH,
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
MODEL_NAME = 'blazepose'
# if args.normal:
DETECTOR_WEIGHT_PATH = f'{MODEL_NAME}_detector.onnx'
DETECTOR_MODEL_PATH = f'{MODEL_NAME}_detector.onnx.prototxt'
ESTIMATOR_WEIGHT_PATH = f'{MODEL_NAME}_estimator.onnx'
ESTIMATOR_MODEL_PATH = f'{MODEL_NAME}_estimator.onnx.prototxt'
# else:
#     WEIGHT_PATH = f'{MODEL_NAME}.opt.onnx'
#     MODEL_PATH = f'{MODEL_NAME}.opt.onnx.prototxt'
REMOTE_PATH = f'https://storage.googleapis.com/ailia-models/{MODEL_NAME}/'


# ======================
# Utils
# ======================
def hsv_to_rgb(h, s, v):
    bgr = cv2.cvtColor(
        np.array([[[h, s, v]]], dtype=np.uint8), cv2.COLOR_HSV2BGR
    )[0][0]
    return (int(bgr[2]), int(bgr[1]), int(bgr[0]))


def line(input_img, landmarks, flags, point1, point2):
    threshold = 0.5
    for i in range(len(flags)):
        landmark, flag = landmarks[i], flags[i]
        if flag > threshold:
            color = hsv_to_rgb(255*point1/but.BLAZEPOSE_KEYPOINT_CNT, 255, 255)

            x1 = int(landmark[point1, 0])
            y1 = int(landmark[point1, 1])
            x2 = int(landmark[point2, 0])
            y2 = int(landmark[point2, 1])
            cv2.line(input_img, (x1, y1), (x2, y2), color, 5)


def display_result(input_img, count, landmarks, flags):
    for _ in range(count):
        line(input_img, landmarks, flags, but.BLAZEPOSE_KEYPOINT_NOSE,
             but.BLAZEPOSE_KEYPOINT_EYE_LEFT_INNER)
        line(input_img, landmarks, flags, but.BLAZEPOSE_KEYPOINT_EYE_LEFT_INNER,
             but.BLAZEPOSE_KEYPOINT_EYE_LEFT)
        line(input_img, landmarks, flags, but.BLAZEPOSE_KEYPOINT_EYE_LEFT,
             but.BLAZEPOSE_KEYPOINT_EYE_LEFT_OUTER)
        line(input_img, landmarks, flags, but.BLAZEPOSE_KEYPOINT_EYE_LEFT_OUTER,
             but.BLAZEPOSE_KEYPOINT_EAR_LEFT)

        line(input_img, landmarks, flags, but.BLAZEPOSE_KEYPOINT_NOSE,
             but.BLAZEPOSE_KEYPOINT_EYE_RIGHT_INNER)
        line(input_img, landmarks, flags, but.BLAZEPOSE_KEYPOINT_EYE_RIGHT_INNER,
             but.BLAZEPOSE_KEYPOINT_EYE_RIGHT)
        line(input_img, landmarks, flags, but.BLAZEPOSE_KEYPOINT_EYE_RIGHT,
             but.BLAZEPOSE_KEYPOINT_EYE_RIGHT_OUTER)
        line(input_img, landmarks, flags, but.BLAZEPOSE_KEYPOINT_EYE_RIGHT_OUTER,
             but.BLAZEPOSE_KEYPOINT_EAR_RIGHT)

        line(input_img, landmarks, flags, but.BLAZEPOSE_KEYPOINT_MOUTH_LEFT,
             but.BLAZEPOSE_KEYPOINT_MOUTH_RIGHT)

        line(input_img, landmarks, flags, but.BLAZEPOSE_KEYPOINT_SHOULDER_LEFT,
             but.BLAZEPOSE_KEYPOINT_SHOULDER_RIGHT)
        line(input_img, landmarks, flags, but.BLAZEPOSE_KEYPOINT_SHOULDER_LEFT,
             but.BLAZEPOSE_KEYPOINT_ELBOW_LEFT)
        line(input_img, landmarks, flags, but.BLAZEPOSE_KEYPOINT_ELBOW_LEFT,
             but.BLAZEPOSE_KEYPOINT_WRIST_LEFT)
        line(input_img, landmarks, flags, but.BLAZEPOSE_KEYPOINT_SHOULDER_RIGHT,
             but.BLAZEPOSE_KEYPOINT_ELBOW_RIGHT)
        line(input_img, landmarks, flags, but.BLAZEPOSE_KEYPOINT_ELBOW_RIGHT,
             but.BLAZEPOSE_KEYPOINT_WRIST_RIGHT)

        line(input_img, landmarks, flags, but.BLAZEPOSE_KEYPOINT_WRIST_LEFT,
             but.BLAZEPOSE_KEYPOINT_PINKY_LEFT_KNUCKLE1)
        line(input_img, landmarks, flags, but.BLAZEPOSE_KEYPOINT_PINKY_LEFT_KNUCKLE1,
             but.BLAZEPOSE_KEYPOINT_INDEX_LEFT_KNUCKLE1)
        line(input_img, landmarks, flags, but.BLAZEPOSE_KEYPOINT_WRIST_LEFT,
             but.BLAZEPOSE_KEYPOINT_INDEX_LEFT_KNUCKLE1)
        line(input_img, landmarks, flags, but.BLAZEPOSE_KEYPOINT_WRIST_LEFT,
             but.BLAZEPOSE_KEYPOINT_THUMB_LEFT_KNUCKLE2)

        line(input_img, landmarks, flags, but.BLAZEPOSE_KEYPOINT_WRIST_RIGHT,
             but.BLAZEPOSE_KEYPOINT_PINKY_RIGHT_KNUCKLE1)
        line(input_img, landmarks, flags, but.BLAZEPOSE_KEYPOINT_PINKY_RIGHT_KNUCKLE1,
             but.BLAZEPOSE_KEYPOINT_INDEX_RIGHT_KNUCKLE1)
        line(input_img, landmarks, flags, but.BLAZEPOSE_KEYPOINT_WRIST_RIGHT,
             but.BLAZEPOSE_KEYPOINT_INDEX_RIGHT_KNUCKLE1)
        line(input_img, landmarks, flags, but.BLAZEPOSE_KEYPOINT_WRIST_RIGHT,
             but.BLAZEPOSE_KEYPOINT_THUMB_RIGHT_KNUCKLE2)

        line(input_img, landmarks, flags, but.BLAZEPOSE_KEYPOINT_SHOULDER_LEFT,
             but.BLAZEPOSE_KEYPOINT_HIP_LEFT)
        line(input_img, landmarks, flags, but.BLAZEPOSE_KEYPOINT_SHOULDER_RIGHT,
             but.BLAZEPOSE_KEYPOINT_HIP_RIGHT)
        line(input_img, landmarks, flags, but.BLAZEPOSE_KEYPOINT_HIP_LEFT,
             but.BLAZEPOSE_KEYPOINT_HIP_RIGHT)

        # Upper body: stop here

        # line(input_img, landmarks, flags, but.BLAZEPOSE_KEYPOINT_HIP_LEFT,
        #      but.BLAZEPOSE_KEYPOINT_KNEE_LEFT)
        # line(input_img, landmarks, flags, but.BLAZEPOSE_KEYPOINT_KNEE_LEFT,
        #      but.BLAZEPOSE_KEYPOINT_ANKLE_LEFT)
        # line(input_img, landmarks, flags, but.BLAZEPOSE_KEYPOINT_HIP_RIGHT,
        #      but.BLAZEPOSE_KEYPOINT_KNEE_RIGHT)
        # line(input_img, landmarks, flags, but.BLAZEPOSE_KEYPOINT_KNEE_RIGHT,
        #      but.BLAZEPOSE_KEYPOINT_ANKLE_RIGHT)

        # line(input_img, landmarks, flags, but.BLAZEPOSE_KEYPOINT_ANKLE_LEFT,
        #      but.BLAZEPOSE_KEYPOINT_HEEL_LEFT)
        # line(input_img, landmarks, flags, but.BLAZEPOSE_KEYPOINT_HEEL_LEFT,
        #      but.BLAZEPOSE_KEYPOINT_FOOT_LEFT_INDEX)
        # line(input_img, landmarks, flags, but.BLAZEPOSE_KEYPOINT_ANKLE_LEFT,
        #      but.BLAZEPOSE_KEYPOINT_FOOT_LEFT_INDEX)

        # line(input_img, landmarks, flags, but.BLAZEPOSE_KEYPOINT_ANKLE_RIGHT,
        #      but.BLAZEPOSE_KEYPOINT_HEEL_RIGHT)
        # line(input_img, landmarks, flags, but.BLAZEPOSE_KEYPOINT_HEEL_RIGHT,
        #      but.BLAZEPOSE_KEYPOINT_FOOT_RIGHT_INDEX)
        # line(input_img, landmarks, flags, but.BLAZEPOSE_KEYPOINT_ANKLE_RIGHT,
        #      but.BLAZEPOSE_KEYPOINT_FOOT_RIGHT_INDEX)


def save_json(landmarks, flags, detections, det_scores, json_path):
    output = []
    if 0 < len(landmarks):
        # detections[0] is already denormalized in estimator_preprocess(), so use it as is
        xc, yc, scale, theta = but.detection2roi(detections[0])
        for i in range(len(landmarks)):
            o = {}
            o['detection_score'] = det_scores[i]
            o['roi'] = { 'center': [xc[i], yc[i]], 'scale': scale[i], 'theta': theta[i] }
            o['flag'] = flags[i]
            o['landmarks'] = [
                {
                    'pos': [lm[0], lm[1]],
                    'z': lm[2],
                    'visibility': expit(lm[3] * but.resolution), # divided together with the coordinates on the ONNX side
                }
                for lm in landmarks[i]
            ]
            output.append(o)

    with open(json_path, 'w') as f:
        json.dump(numpy_type_to_builtin_type(output), f, indent=2)


# ======================
# Main functions
# ======================
def recognize_from_image():
    # net initialize
    detector = ailia.Net(
        DETECTOR_MODEL_PATH, DETECTOR_WEIGHT_PATH, env_id=args.env_id
    )
    estimator = ailia.Net(
        ESTIMATOR_MODEL_PATH, ESTIMATOR_WEIGHT_PATH, env_id=args.env_id
    )

    # input image loop
    for image_path in args.input:
        # prepare input data
        logger.info(image_path)
        src_img = imread(image_path)
        _, img128, scale, pad = but.resize_pad(src_img[:, :, ::-1])
        input_data = img128.astype('float32') / 255.
        input_data = np.expand_dims(np.moveaxis(input_data, -1, 0), 0)

        # inference
        logger.info('Start inference...')
        if args.benchmark:
            logger.info('BENCHMARK mode')
            total_time_detection = 0
            total_time_estimation = 0
            for i in range(args.benchmark_count):
                # Person detection
                start = int(round(time.time() * 1000))
                detector_out = detector.predict([input_data])
                detections = but.detector_postprocess(detector_out)
                count = len(detections) if detections[0].size > 0 else 0
                end = int(round(time.time() * 1000))
                detection_time = (end - start)
                # keep a copy because denormalize_detections() overwrites column 12 as a coordinate
                det_scores = detections[0][:, 12].copy()

                # Pose estimation
                start = int(round(time.time() * 1000))
                landmarks = []
                flags = []
                if count > 0:
                    img, affine, _ = but.estimator_preprocess(
                        src_img, detections, scale, pad
                    )
                    flags, normalized_landmarks, _ = estimator.predict([img])
                    landmarks = but.denormalize_landmarks(
                        normalized_landmarks, affine
                    )
                end = int(round(time.time() * 1000))
                estimation_time = (end - start)

                # Loggin
                logger.info(f'\tailia processing detection time {detection_time} ms')
                logger.info(f'\tailia processing estimation time {estimation_time} ms')

                if i != 0:
                    total_time_detection = total_time_detection + detection_time
                    total_time_estimation = total_time_estimation+ estimation_time

            logger.info(f'\taverage time detection {total_time_detection / (args.benchmark_count-1)} ms')
            logger.info(f'\taverage time estimation {total_time_estimation / (args.benchmark_count-1)} ms')
        else:
            # Person detection
            detector_out = detector.predict([input_data])
            detections = but.detector_postprocess(detector_out)
            count = len(detections) if detections[0].size != 0 else 0
            # keep a copy because denormalize_detections() overwrites column 12 as a coordinate
            det_scores = detections[0][:, 12].copy()

            # Pose estimation
            landmarks = []
            flags = []
            if count > 0:
                img, affine, _ = but.estimator_preprocess(
                    src_img, detections, scale, pad
                )
                flags, normalized_landmarks, _ = estimator.predict([img])
                landmarks = but.denormalize_landmarks(
                    normalized_landmarks, affine
                )

        # postprocessing
        logger.info(f'person_count={count}')
        display_result(src_img, count, landmarks, flags)
        savepath = get_savepath(args.savepath, image_path)
        logger.info(f'saved at : {savepath}')
        cv2.imwrite(savepath, src_img)

        if args.write_json:
            save_json(landmarks, flags, detections, det_scores, (savepath.rsplit('.', 1)[0]) + '.json')
    logger.info('Script finished successfully.')


def recognize_from_video():
    # net initialize
    detector = ailia.Net(
        DETECTOR_MODEL_PATH, DETECTOR_WEIGHT_PATH, env_id=args.env_id
    )
    estimator = ailia.Net(
        ESTIMATOR_MODEL_PATH, ESTIMATOR_WEIGHT_PATH, env_id=args.env_id
    )

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

        _, img128, scale, pad = but.resize_pad(frame[:, :, ::-1])
        input_data = img128.astype('float32') / 255.
        input_data = np.expand_dims(np.moveaxis(input_data, -1, 0), 0)

        # inference
        # Person detection
        detector_out = detector.predict([input_data])
        detections = but.detector_postprocess(detector_out)
        count = len(detections) if detections[0].size > 0 else 0

        # Pose estimation
        landmarks = []
        flags = []
        if count > 0:
            img, affine, _ = but.estimator_preprocess(
                frame, detections, scale, pad
            )
            flags, normalized_landmarks, _ = estimator.predict([img])
            landmarks = but.denormalize_landmarks(normalized_landmarks, affine)

        # postprocessing
        display_result(frame, count, landmarks, flags)
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
    pass


def main():
    # model files check and download
    check_and_download_models(
        DETECTOR_WEIGHT_PATH, DETECTOR_MODEL_PATH, REMOTE_PATH
    )
    check_and_download_models(
        ESTIMATOR_WEIGHT_PATH, ESTIMATOR_MODEL_PATH, REMOTE_PATH
    )

    if args.video is not None:
        # video mode
        recognize_from_video()
    else:
        # image mode
        recognize_from_image()


if __name__ == '__main__':
    main()
