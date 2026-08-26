import sys
import os
import time
from logging import getLogger

import ailia
import cv2
import numpy as np

sys.path.append('../../util')
import webcamera_utils  # noqa
from image_utils import imread, load_image  # noqa
from model_utils import check_and_download_models  # noqa
from arg_utils import get_base_parser, get_savepath, update_parser  # noqa
from load_model import load_face_detection_adas

_this = os.path.dirname(os.path.abspath(__file__))
top_path = os.path.dirname(os.path.dirname(_this))

sys.path.append(os.path.join(top_path, 'face_detection/blazeface'))
from blazeface_utils import crop_blazeface  # noqa

logger = getLogger(__name__)

# ======================
# Parameters
# ======================

WEIGHT_PATH = 'age-gender-recognition-retail-0013.onnx'
MODEL_PATH = 'age-gender-recognition-retail-0013.onnx.prototxt'
REMOTE_PATH = 'https://storage.googleapis.com/ailia-models/age-gender-recognition-retail/'

FACE_DETECTION_ADAS_WEIGHT_PATH = 'face-detection-adas-0001.onnx'
FACE_DETECTION_ADAS_MODEL_PATH = 'face-detection-adas-0001.onnx.prototxt'
FACE_DETECTION_ADAS_REMOTE_PATH = 'https://storage.googleapis.com/ailia-models/face-detection-adas/'
FACE_DETECTION_ADAS_PRIORBOX_PATH = 'mbox_priorbox.npy'

IMAGE_PATH = 'demo.jpg'
IMAGE_SIZE = 62

SAVE_IMAGE_PATH = 'output.png'

# ======================
# Argument Parser Config
# ======================

parser = get_base_parser(
    'age-gender-recognition', IMAGE_PATH, SAVE_IMAGE_PATH,
)
parser.add_argument(
    '-d', '--detection', action='store_true',
    help='Use face detection.'
)
args = update_parser(parser)

detection = args.detection or args.video is not None


# ======================
# Secondaty Functions
# ======================

def setup_detector(net):
    mod = load_face_detection_adas(args)

    prior_box = np.squeeze(np.load(os.path.join(
        top_path, 'face_detection/face-detection-adas', mod.PRIORBOX_PATH)))

    model_info = {
        'net': net,
        'prior_box': prior_box,
    }

    def _detector(img):
        detections = mod.predict(model_info, img)
        return detections

    return _detector


# ======================
# Main functions
# ======================

def recognize_image(net, detector, image):
    # detect face
    detections = detector(image)

    # estimate age and gender
    for obj in detections:
        # get detected face
        # make square and enlarge face bounding box for more robust operation
        # of face analytics networks (bb_enlarge_coefficient of OpenVINO demo)
        margin = 1.2
        crop_img, top_left, bottom_right = crop_blazeface(
            obj, margin, image
        )
        if crop_img.shape[0] <= 0 or crop_img.shape[1] <= 0:
            continue

        img = cv2.resize(crop_img, (IMAGE_SIZE, IMAGE_SIZE))
        img = np.expand_dims(img, axis=0)  # 次元合せ

        # inference
        output = net.predict([img])
        prob, age_conv3 = output
        prob = prob[0][0][0]
        age_conv3 = age_conv3[0][0][0][0]

        i = np.argmax(prob)
        gender = 'Female' if i == 0 else 'Male'
        age = round(age_conv3 * 100)
        logger.info(" gender is: %s (%.2f)" % (gender, prob[i] * 100))
        logger.info(" age is: %d" % age)

        # display label
        LABEL_WIDTH = bottom_right[1] - top_left[1]
        LABEL_HEIGHT = 20
        if gender == "Male":
            color = (255, 128, 128)
        else:
            color = (128, 128, 255)
        cv2.rectangle(image, top_left, bottom_right, color, thickness=2)
        cv2.rectangle(
            image,
            top_left,
            (top_left[0] + LABEL_WIDTH, top_left[1] + LABEL_HEIGHT),
            color,
            thickness=-1,
        )

        text_position = (top_left[0], top_left[1] + LABEL_HEIGHT // 2)
        color = (0, 0, 0)
        fontScale = 0.5
        cv2.putText(
            image,
            "{} {}".format(gender, age),
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

        if detection:
            image = imread(image_path)
            image = recognize_image(net, detector, image)

            savepath = get_savepath(args.savepath, image_path)
            logger.info(f'saved at : {savepath}')
            cv2.imwrite(savepath, image)
            continue

        img = load_image(
            image_path, (IMAGE_SIZE, IMAGE_SIZE),
            normalize_type='None')
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        img = np.expand_dims(img, axis=0)  # 次元合せ

        # inference
        logger.info('Start inference...')
        if args.benchmark:
            logger.info('BENCHMARK mode')
            total_time_estimation = 0
            for i in range(args.benchmark_count):
                start = int(round(time.time() * 1000))
                output = net.predict([img])
                end = int(round(time.time() * 1000))
                estimation_time = (end - start)

                # Logging
                logger.info(f'\tailia processing estimation time {estimation_time} ms')
                if i != 0:
                    total_time_estimation = total_time_estimation + estimation_time

            logger.info(f'\taverage time estimation {total_time_estimation / (args.benchmark_count - 1)} ms')
        else:
            output = net.predict([img])

        prob, age_conv3 = output
        prob = prob[0][0][0]
        age_conv3 = age_conv3[0][0][0][0]

        i = np.argmax(prob)
        logger.info(" gender is: %s (%.2f)" % ('Female' if i == 0 else 'Male', prob[i] * 100))
        logger.info(" age is: %d" % round(age_conv3 * 100))

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
    logger.info('=== age-gender-recognition model ===')
    check_and_download_models(
        WEIGHT_PATH, MODEL_PATH, REMOTE_PATH
    )

    det_weight_path = det_model_path = None
    if detection:
        logger.info('=== face detection model ===')
        det_weight_path = FACE_DETECTION_ADAS_WEIGHT_PATH
        det_model_path = FACE_DETECTION_ADAS_MODEL_PATH
        check_and_download_models(
            det_weight_path, det_model_path, FACE_DETECTION_ADAS_REMOTE_PATH
        )

    # load model
    env_id = args.env_id

    # net initialize
    net = ailia.Net(
        MODEL_PATH, WEIGHT_PATH, env_id=env_id
    )
    detector = None
    if det_weight_path:
        detector = ailia.Net(det_model_path, det_weight_path, env_id=env_id)
        detector = setup_detector(detector)

    # image mode
    if args.video is not None:
        # video mode
        recognize_from_video(net, detector)
    else:
        # image mode
        recognize_from_image(net, detector)


if __name__ == '__main__':
    main()
