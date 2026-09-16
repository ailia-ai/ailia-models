import sys
import time

import cv2
import numpy as np

import ailia
import blazepose_utils as but

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

MODEL_LIST = ['lite', 'full', 'heavy']
WEIGHT_LITE_PATH = 'pose_landmark_lite.onnx'
MODEL_LITE_PATH = 'pose_landmark_lite.onnx.prototxt'
WEIGHT_FULL_PATH = 'pose_landmark_full.onnx'
MODEL_FULL_PATH = 'pose_landmark_full.onnx.prototxt'
WEIGHT_HEAVY_PATH = 'pose_landmark_heavy.onnx'
MODEL_HEAVY_PATH = 'pose_landmark_heavy.onnx.prototxt'
WEIGHT_DETECTOR_PATH = 'pose_detection.onnx'
MODEL_DETECTOR_PATH = 'pose_detection.onnx.prototxt'
REMOTE_PATH = \
    'https://storage.googleapis.com/ailia-models/blazepose-fullbody/'

IMAGE_PATH = 'girl-5204299_640.jpg'
SAVE_IMAGE_PATH = 'output.png'
IMAGE_SIZE = 256

# ======================
# Argument Parser Config
# ======================

parser = get_base_parser(
    'BlazePose, an on-device real-time body pose tracking.',
    IMAGE_PATH,
    SAVE_IMAGE_PATH,
    fp16_support=False
)
parser.add_argument(
    '-m', '--model', metavar='ARCH',
    default='heavy', choices=MODEL_LIST,
    help='Set model architecture: ' + ' | '.join(MODEL_LIST)
)
parser.add_argument(
    '-th', '--threshold',
    default=0.5, type=float,
    help='The detection threshold'
)
args = update_parser(parser)


# ======================
# Utils
# ======================

def preprocess(img):
    img = cv2.resize(img, (IMAGE_SIZE, IMAGE_SIZE), interpolation=cv2.INTER_LINEAR)
    img = img.astype(np.float32) / 255
    img = np.expand_dims(img, axis=0)

    return img


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def postprocess(landmarks):
    num = len(landmarks)
    normalized_landmarks = np.zeros((num, 33, 4))
    for i in range(num):
        xx = landmarks[i]
        for j in range(33):
            x = xx[j * 5] / IMAGE_SIZE
            y = xx[j * 5 + 1] / IMAGE_SIZE
            z = xx[j * 5 + 2] / IMAGE_SIZE
            visibility = xx[j * 5 + 3]
            presence = xx[j * 5 + 4]
            normalized_landmarks[i, j] = (x, y, z, sigmoid(min(visibility, presence)))

    return normalized_landmarks


def pose_estimate(net, det_net, img):
    h, w = img.shape[:2]
    src_img = img

    logger.debug(f'input image shape: {img.shape}')

    _, img224, scale, pad = but.resize_pad(img)
    img224 = img224.astype('float32') / 255.
    img224 = np.expand_dims(img224, axis=0)

    detector_out = det_net.predict([img224])
    detections = but.detector_postprocess(detector_out)
    count = len(detections) if detections[0].size != 0 else 0

    # Pose estimation
    imgs = []
    if 0 < count:
        imgs, affine, _ = but.estimator_preprocess(
            src_img, detections, scale, pad
        )

    flags = []
    landmarks = []
    for i, img in enumerate(imgs):
        img = np.expand_dims(img, axis=0)
        output = net.predict([img])

        normalized_landmarks, f, _, _, _ = output
        normalized_landmarks = postprocess(normalized_landmarks)

        flags.append(f[0])
        landmarks.append(normalized_landmarks[0])

    if len(imgs)>=1:
        landmarks = np.stack(landmarks)
        landmarks = but.denormalize_landmarks(landmarks, affine)

    return flags, landmarks


def hsv_to_rgb(h, s, v):
    bgr = cv2.cvtColor(
        np.array([[[h, s, v]]], dtype=np.uint8), cv2.COLOR_HSV2BGR
    )[0][0]
    return (int(bgr[2]), int(bgr[1]), int(bgr[0]))


def line(input_img, landmarks, flags, point1, point2):
    threshold = args.threshold

    for i in range(len(flags)):
        landmark, flag = landmarks[i], flags[i]
        conf1 = landmark[point1, 3]
        conf2 = landmark[point2, 3]

        if flag >= threshold and conf1 >= threshold and conf2 >= threshold:
            color = hsv_to_rgb(255 * point1 / but.BLAZEPOSE_KEYPOINT_CNT, 255, 255)

            line_width = 5

            x1 = int(landmark[point1, 0])
            y1 = int(landmark[point1, 1])
            x2 = int(landmark[point2, 0])
            y2 = int(landmark[point2, 1])
            cv2.line(input_img, (x1, y1), (x2, y2), color, line_width)

def circle(input_img, landmarks, flags):
    threshold = args.threshold

    for i in range(len(flags)):
        for point1 in range(11,33):
          landmark, flag = landmarks[i], flags[i]
          conf1 = landmark[point1, 3]

          if flag >= threshold and conf1 >= threshold:
               color = hsv_to_rgb(255 * point1 / but.BLAZEPOSE_KEYPOINT_CNT, 255, 255)

               base_line_width = 5

               line_width = landmark[point1, 2]
               line_width = base_line_width - line_width/2 * 100
               line_width = max(int(line_width),1)

               x1 = int(landmark[point1, 0])
               y1 = int(landmark[point1, 1])
               cv2.circle(input_img, (x1, y1), line_width, color, thickness=2, lineType=cv2.LINE_8, shift=0)


def display_result(img, landmarks, flags):
    circle(img, landmarks, flags)

    line(img, landmarks, flags, but.BLAZEPOSE_KEYPOINT_NOSE,
         but.BLAZEPOSE_KEYPOINT_EYE_LEFT_INNER)
    line(img, landmarks, flags, but.BLAZEPOSE_KEYPOINT_EYE_LEFT_INNER,
         but.BLAZEPOSE_KEYPOINT_EYE_LEFT)
    line(img, landmarks, flags, but.BLAZEPOSE_KEYPOINT_EYE_LEFT,
         but.BLAZEPOSE_KEYPOINT_EYE_LEFT_OUTER)
    line(img, landmarks, flags, but.BLAZEPOSE_KEYPOINT_EYE_LEFT_OUTER,
         but.BLAZEPOSE_KEYPOINT_EAR_LEFT)

    line(img, landmarks, flags, but.BLAZEPOSE_KEYPOINT_NOSE,
         but.BLAZEPOSE_KEYPOINT_EYE_RIGHT_INNER)
    line(img, landmarks, flags, but.BLAZEPOSE_KEYPOINT_EYE_RIGHT_INNER,
         but.BLAZEPOSE_KEYPOINT_EYE_RIGHT)
    line(img, landmarks, flags, but.BLAZEPOSE_KEYPOINT_EYE_RIGHT,
         but.BLAZEPOSE_KEYPOINT_EYE_RIGHT_OUTER)
    line(img, landmarks, flags, but.BLAZEPOSE_KEYPOINT_EYE_RIGHT_OUTER,
         but.BLAZEPOSE_KEYPOINT_EAR_RIGHT)

    line(img, landmarks, flags, but.BLAZEPOSE_KEYPOINT_MOUTH_LEFT,
         but.BLAZEPOSE_KEYPOINT_MOUTH_RIGHT)

    line(img, landmarks, flags, but.BLAZEPOSE_KEYPOINT_SHOULDER_LEFT,
         but.BLAZEPOSE_KEYPOINT_SHOULDER_RIGHT)
    line(img, landmarks, flags, but.BLAZEPOSE_KEYPOINT_SHOULDER_LEFT,
         but.BLAZEPOSE_KEYPOINT_ELBOW_LEFT)
    line(img, landmarks, flags, but.BLAZEPOSE_KEYPOINT_ELBOW_LEFT,
         but.BLAZEPOSE_KEYPOINT_WRIST_LEFT)
    line(img, landmarks, flags, but.BLAZEPOSE_KEYPOINT_SHOULDER_RIGHT,
         but.BLAZEPOSE_KEYPOINT_ELBOW_RIGHT)
    line(img, landmarks, flags, but.BLAZEPOSE_KEYPOINT_ELBOW_RIGHT,
         but.BLAZEPOSE_KEYPOINT_WRIST_RIGHT)

    line(img, landmarks, flags, but.BLAZEPOSE_KEYPOINT_WRIST_LEFT,
         but.BLAZEPOSE_KEYPOINT_PINKY_LEFT_KNUCKLE1)
    line(img, landmarks, flags, but.BLAZEPOSE_KEYPOINT_PINKY_LEFT_KNUCKLE1,
         but.BLAZEPOSE_KEYPOINT_INDEX_LEFT_KNUCKLE1)
    line(img, landmarks, flags, but.BLAZEPOSE_KEYPOINT_WRIST_LEFT,
         but.BLAZEPOSE_KEYPOINT_INDEX_LEFT_KNUCKLE1)
    line(img, landmarks, flags, but.BLAZEPOSE_KEYPOINT_WRIST_LEFT,
         but.BLAZEPOSE_KEYPOINT_THUMB_LEFT_KNUCKLE2)

    line(img, landmarks, flags, but.BLAZEPOSE_KEYPOINT_WRIST_RIGHT,
         but.BLAZEPOSE_KEYPOINT_PINKY_RIGHT_KNUCKLE1)
    line(img, landmarks, flags, but.BLAZEPOSE_KEYPOINT_PINKY_RIGHT_KNUCKLE1,
         but.BLAZEPOSE_KEYPOINT_INDEX_RIGHT_KNUCKLE1)
    line(img, landmarks, flags, but.BLAZEPOSE_KEYPOINT_WRIST_RIGHT,
         but.BLAZEPOSE_KEYPOINT_INDEX_RIGHT_KNUCKLE1)
    line(img, landmarks, flags, but.BLAZEPOSE_KEYPOINT_WRIST_RIGHT,
         but.BLAZEPOSE_KEYPOINT_THUMB_RIGHT_KNUCKLE2)

    line(img, landmarks, flags, but.BLAZEPOSE_KEYPOINT_SHOULDER_LEFT,
         but.BLAZEPOSE_KEYPOINT_HIP_LEFT)
    line(img, landmarks, flags, but.BLAZEPOSE_KEYPOINT_SHOULDER_RIGHT,
         but.BLAZEPOSE_KEYPOINT_HIP_RIGHT)
    line(img, landmarks, flags, but.BLAZEPOSE_KEYPOINT_HIP_LEFT,
         but.BLAZEPOSE_KEYPOINT_HIP_RIGHT)

    # Upper body: stop here

    line(img, landmarks, flags, but.BLAZEPOSE_KEYPOINT_HIP_LEFT,
         but.BLAZEPOSE_KEYPOINT_KNEE_LEFT)
    line(img, landmarks, flags, but.BLAZEPOSE_KEYPOINT_KNEE_LEFT,
         but.BLAZEPOSE_KEYPOINT_ANKLE_LEFT)
    line(img, landmarks, flags, but.BLAZEPOSE_KEYPOINT_HIP_RIGHT,
         but.BLAZEPOSE_KEYPOINT_KNEE_RIGHT)
    line(img, landmarks, flags, but.BLAZEPOSE_KEYPOINT_KNEE_RIGHT,
         but.BLAZEPOSE_KEYPOINT_ANKLE_RIGHT)

    line(img, landmarks, flags, but.BLAZEPOSE_KEYPOINT_ANKLE_LEFT,
         but.BLAZEPOSE_KEYPOINT_HEEL_LEFT)
    line(img, landmarks, flags, but.BLAZEPOSE_KEYPOINT_HEEL_LEFT,
         but.BLAZEPOSE_KEYPOINT_FOOT_LEFT_INDEX)
    line(img, landmarks, flags, but.BLAZEPOSE_KEYPOINT_ANKLE_LEFT,
         but.BLAZEPOSE_KEYPOINT_FOOT_LEFT_INDEX)

    line(img, landmarks, flags, but.BLAZEPOSE_KEYPOINT_ANKLE_RIGHT,
         but.BLAZEPOSE_KEYPOINT_HEEL_RIGHT)
    line(img, landmarks, flags, but.BLAZEPOSE_KEYPOINT_HEEL_RIGHT,
         but.BLAZEPOSE_KEYPOINT_FOOT_RIGHT_INDEX)
    line(img, landmarks, flags, but.BLAZEPOSE_KEYPOINT_ANKLE_RIGHT,
         but.BLAZEPOSE_KEYPOINT_FOOT_RIGHT_INDEX)


# ======================
# Main functions
# ======================

def recognize_from_image(net, det_net):
    # input image loop
    for image_path in args.input:
        # prepare input data
        logger.info(image_path)

        img = src_img = load_image(image_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2RGB)

        # inference
        logger.info('Start inference...')
        if args.benchmark:
            logger.info('BENCHMARK mode')
            total_time_estimation = 0
            for i in range(args.benchmark_count):
                # Pose estimation
                start = int(round(time.time() * 1000))
                flags, landmarks = pose_estimate(net, det_net, img)
                end = int(round(time.time() * 1000))
                estimation_time = (end - start)

                # Loggin
                logger.info(f'\tailia processing estimation time {estimation_time} ms')
                if i != 0:
                    total_time_estimation = total_time_estimation + estimation_time

            logger.info(f'\taverage time estimation {total_time_estimation / (args.benchmark_count - 1)} ms')
        else:
            # inference
            flags, landmarks = pose_estimate(net, det_net, img)

        # plot result
        src_img = cv2.cvtColor(src_img, cv2.COLOR_BGRA2BGR)
        display_result(src_img, landmarks, flags)

        # save results
        savepath = get_savepath(args.savepath, image_path)
        logger.info(f'saved at : {savepath}')
        cv2.imwrite(savepath, src_img)

    logger.info('Script finished successfully.')


def recognize_from_video(net, det_net):
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
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        flags, landmarks = pose_estimate(net, det_net, frame_rgb)

        # plot result
        display_result(frame, landmarks, flags)
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
    # model files check and download
    logger.info('=== detector model ===')
    check_and_download_models(WEIGHT_DETECTOR_PATH, MODEL_DETECTOR_PATH, REMOTE_PATH)
    logger.info('=== blazepose model ===')
    info = {
        'lite': (WEIGHT_LITE_PATH, MODEL_LITE_PATH),
        'full': (WEIGHT_FULL_PATH, MODEL_FULL_PATH),
        'heavy': (WEIGHT_HEAVY_PATH, MODEL_HEAVY_PATH),
    }
    weight_path, model_path = info[args.model]
    check_and_download_models(weight_path, model_path, REMOTE_PATH)

    env_id = args.env_id
    
    # initialize
    det_net = ailia.Net(MODEL_DETECTOR_PATH, WEIGHT_DETECTOR_PATH, env_id=env_id, memory_mode=args.memory_mode)
    net = ailia.Net(model_path, weight_path, env_id=env_id, memory_mode=args.memory_mode)

    if args.video is not None:
        # video mode
        recognize_from_video(net, det_net)
    else:
        # image mode
        recognize_from_image(net, det_net)


if __name__ == '__main__':
    main()
