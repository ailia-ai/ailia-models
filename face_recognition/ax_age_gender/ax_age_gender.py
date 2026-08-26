import sys
import os
import time
from logging import getLogger

import ailia
import cv2
import numpy as np

sys.path.append('../../util')
import webcamera_utils  # noqa
from image_utils import imread  # noqa
from model_utils import check_and_download_models  # noqa
from arg_utils import get_base_parser, get_savepath, update_parser  # noqa

from ax_age_gender_util import (BlazeFace, AlignConfig, align_face,  # noqa
                                pose_in_range)

_this = os.path.dirname(os.path.abspath(__file__))
top_path = os.path.dirname(os.path.dirname(_this))

logger = getLogger(__name__)

# ======================
# Parameters
# ======================

WEIGHT_PATH = 'ax_age_gender_fp16.onnx'
MODEL_PATH = None
REMOTE_PATH = 'https://storage.googleapis.com/ailia-models/ax_age_gender/'

BLAZEFACE_WEIGHT_PATH = 'blazefaceback.onnx'
BLAZEFACE_MODEL_PATH = 'blazefaceback.onnx.prototxt'
BLAZEFACE_REMOTE_PATH = 'https://storage.googleapis.com/ailia-models/blazeface/'
BLAZEFACE_ANCHORS_PATH = os.path.join(
    top_path, 'face_detection/blazeface/anchorsback.npy')

FACE_MIN_SCORE_THRESH = 0.5

# preprocessing constants carried in the ONNX metadata of the model
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], np.float32)
ALIGN_CONFIG = AlignConfig(size=128, interocular=0.30, eye_y=0.40)

GENDER_NAMES = ('Male', 'Female')
MASK_NAMES = ('nomask', 'mask')

IMAGE_PATH = 'demo.jpg'
SAVE_IMAGE_PATH = 'output.png'

# ======================
# Argument Parser Config
# ======================

parser = get_base_parser(
    'ax_age_gender', IMAGE_PATH, SAVE_IMAGE_PATH,
)
parser.add_argument(
    '--no-pose-gate', action='store_true',
    help='Estimate even for head angles the model is not accurate on.'
)
args = update_parser(parser)


# ======================
# Main functions
# ======================

def estimate_age_gender(net, crop_img):
    img = cv2.cvtColor(crop_img, cv2.COLOR_BGR2RGB).astype(np.float32)
    img = (img / 255.0 - IMAGENET_MEAN) / IMAGENET_STD
    img = np.ascontiguousarray(img.transpose(2, 0, 1)[np.newaxis])

    output = net.predict([img])
    age, gender, mask = output[0], output[1], output[2]

    age = float(age[0, 0])
    gi = int(np.argmax(gender[0]))
    mi = int(np.argmax(mask[0]))
    return age, GENDER_NAMES[gi], float(gender[0][gi]), \
        MASK_NAMES[mi], float(mask[0][mi])


def recognize_image(net, detector, image):
    # detect face
    detections = detector(image, min_score=FACE_MIN_SCORE_THRESH)

    # estimate age and gender
    for det in detections:
        x1, y1, x2, y2 = detector.box(det).astype(int)

        # align face from the eye keypoints
        keypoints = detector.keypoints(det)
        crop_img, aligned_kp = align_face(image, keypoints, ALIGN_CONFIG)
        if crop_img is None:
            continue

        # skip head angles the model is not accurate on
        ok, pitch, yaw, why = pose_in_range(aligned_kp)
        if not ok and not args.no_pose_gate:
            logger.info(" skipped: %s (pitch %.2f yaw %+.2f)" % (why, pitch, yaw))
            color = (140, 140, 140)
            cv2.rectangle(image, (x1, y1), (x2, y2), color, thickness=2)
            continue

        # inference
        age, gender, gender_prob, mask, mask_prob = \
            estimate_age_gender(net, crop_img)
        logger.info(" gender is: %s (%.2f)" % (gender, gender_prob * 100))
        logger.info(" age is: %.1f" % age)
        logger.info(" mask is: %s (%.2f)" % (mask, mask_prob * 100))

        # display label
        LABEL_WIDTH = x2 - x1
        LABEL_HEIGHT = 20
        if gender == "Male":
            color = (255, 128, 128)
        else:
            color = (128, 128, 255)
        cv2.rectangle(image, (x1, y1), (x2, y2), color, thickness=2)
        cv2.rectangle(
            image,
            (x1, y1),
            (x1 + LABEL_WIDTH, y1 + LABEL_HEIGHT),
            color,
            thickness=-1,
        )

        label = "{} {}".format(gender, round(age))
        if mask == 'mask':
            label += " [mask]"
        text_position = (x1, y1 + LABEL_HEIGHT // 2)
        color = (0, 0, 0)
        fontScale = 0.5
        cv2.putText(
            image,
            label,
            text_position,
            cv2.FONT_HERSHEY_SIMPLEX,
            fontScale,
            color,
            1,
        )

    return image


def recognize_from_image(net, detector):
    # input image loop
    for image_path in args.input:
        # prepare input data
        logger.info(image_path)
        image = imread(image_path)

        # inference
        logger.info('Start inference...')
        if args.benchmark:
            logger.info('BENCHMARK mode')
            total_time_estimation = 0
            for i in range(args.benchmark_count):
                start = int(round(time.time() * 1000))
                image_out = recognize_image(net, detector, image.copy())
                end = int(round(time.time() * 1000))
                estimation_time = (end - start)

                # Logging
                logger.info(f'\tailia processing estimation time {estimation_time} ms')
                if i != 0:
                    total_time_estimation = total_time_estimation + estimation_time

            logger.info(f'\taverage time estimation {total_time_estimation / (args.benchmark_count - 1)} ms')
        else:
            image_out = recognize_image(net, detector, image.copy())

        savepath = get_savepath(args.savepath, image_path)
        logger.info(f'saved at : {savepath}')
        cv2.imwrite(savepath, image_out)

    logger.info('Script finished successfully.')


def recognize_from_video(net, detector):
    capture = webcamera_utils.get_capture(args.video)

    # create video writer if savepath is specified as video format
    if args.savepath is not None:
        f_h = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        f_w = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        writer = webcamera_utils.get_writer(args.savepath, f_h, f_w)
    else:
        writer = None

    frame_shown = False
    while True:
        ret, frame = capture.read()
        if (cv2.waitKey(1) & 0xFF == ord('q')) or not ret:
            break
        if frame_shown and cv2.getWindowProperty('frame', cv2.WND_PROP_VISIBLE) == 0:
            break

        frame = recognize_image(net, detector, frame)

        # show result
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
    logger.info('=== ax_age_gender model ===')
    check_and_download_models(
        WEIGHT_PATH, MODEL_PATH, REMOTE_PATH
    )
    logger.info('=== face detection model ===')
    check_and_download_models(
        BLAZEFACE_WEIGHT_PATH, BLAZEFACE_MODEL_PATH, BLAZEFACE_REMOTE_PATH
    )

    # load model
    env_id = args.env_id

    # net initialize
    net = ailia.Net(
        MODEL_PATH, WEIGHT_PATH, env_id=env_id
    )
    detector_net = ailia.Net(
        BLAZEFACE_MODEL_PATH, BLAZEFACE_WEIGHT_PATH, env_id=env_id
    )
    detector = BlazeFace(detector_net, BLAZEFACE_ANCHORS_PATH)

    if args.video is not None:
        # video mode
        recognize_from_video(net, detector)
    else:
        # image mode
        recognize_from_image(net, detector)


if __name__ == '__main__':
    main()
