"""OpenVINO version of the age-gender-recognition-retail sample.

Runs the official OpenVINO IR models (open_model_zoo) with the OpenVINO
runtime using the same preprocessing as the ailia sample
(age-gender-recognition-retail.py), so that the results of the two
implementations can be compared.

Usage:
    pip3 install openvino
    python3 example_openvino.py --input demo.jpg
    python3 example_openvino.py --input demo.jpg --detection
"""
import argparse
import os
import urllib.request
from logging import getLogger, StreamHandler, INFO

import cv2
import numpy as np
import openvino as ov

logger = getLogger(__name__)
logger.setLevel(INFO)
logger.addHandler(StreamHandler())

# ======================
# Parameters
# ======================

OMZ_REMOTE_PATH = 'https://storage.openvinotoolkit.org/repositories/' \
    'open_model_zoo/2023.0/models_bin/1/'

WEIGHT_PATH = 'age-gender-recognition-retail-0013'
FACE_DETECTION_ADAS_PATH = 'face-detection-adas-0001'

FACE_MIN_SCORE_THRESH = 0.5

IMAGE_PATH = 'demo.jpg'
IMAGE_SIZE = 62

SAVE_IMAGE_PATH = 'output_openvino.png'

# ======================
# Argument Parser Config
# ======================

parser = argparse.ArgumentParser(
    description='age-gender-recognition (OpenVINO version)')
parser.add_argument(
    '-i', '--input', default=IMAGE_PATH, type=str,
    help='Input image path.'
)
parser.add_argument(
    '-s', '--savepath', default=SAVE_IMAGE_PATH, type=str,
    help='Save path for the output image (used with --detection).'
)
parser.add_argument(
    '-d', '--detection', action='store_true',
    help='Use face detection.'
)
args = parser.parse_args()


# ======================
# Secondaty Functions
# ======================

def check_and_download_ir(model_name):
    for ext in ('.xml', '.bin'):
        file_name = model_name + ext
        if os.path.exists(file_name):
            continue
        url = OMZ_REMOTE_PATH + model_name + '/FP32/' + file_name
        logger.info(f'Downloading {file_name}...')
        urllib.request.urlretrieve(url, file_name)
    logger.info(f'{model_name} IR files are prepared!')


def detect_face(compiled, image):
    im_h, im_w, _ = image.shape
    _, _, h, w = compiled.inputs[0].shape
    img = cv2.resize(image, (w, h))
    img = img.transpose(2, 0, 1)[np.newaxis].astype(np.float32)

    # [1, 1, N, 7] : [image_id, label, conf, x_min, y_min, x_max, y_max]
    detections = compiled({0: img})[compiled.outputs[0]][0, 0]

    faces = []
    for d in detections:
        conf = d[2]
        if conf < FACE_MIN_SCORE_THRESH:
            continue
        faces.append((
            conf,
            d[3] * im_w, d[4] * im_h,
            (d[5] - d[3]) * im_w, (d[6] - d[4]) * im_h,
        ))
    return faces


def crop_face(image, x, y, w, h, margin):
    # make square and enlarge face bounding box for more robust operation
    # of face analytics networks (bb_enlarge_coefficient of OpenVINO demo)
    im_h, im_w, _ = image.shape
    cx = x + w / 2
    cy = y + h / 2
    cw = max(w, h) * margin
    fx = cx - cw / 2
    fy = cy - cw / 2
    top_left = (
        int(np.clip(fx, 0, im_w)),
        int(np.clip(fy, 0, im_h)),
    )
    bottom_right = (
        int(np.clip(fx + cw, 0, im_w)),
        int(np.clip(fy + cw, 0, im_h)),
    )
    crop_img = image[
        top_left[1]:bottom_right[1], top_left[0]:bottom_right[0], 0:3
    ]
    return crop_img, top_left, bottom_right


def estimate_age_gender(compiled, crop_img):
    img = cv2.resize(crop_img, (IMAGE_SIZE, IMAGE_SIZE))
    img = img.transpose(2, 0, 1)[np.newaxis].astype(np.float32)

    output = compiled({0: img})
    prob = np.squeeze(output['prob'])
    age_conv3 = float(np.squeeze(output['age_conv3']))

    i = int(np.argmax(prob))
    gender = 'Female' if i == 0 else 'Male'
    age = round(age_conv3 * 100)
    return gender, prob[i], age


# ======================
# Main functions
# ======================

def recognize_image(net, detector, image):
    # detect face
    faces = detect_face(detector, image)

    # estimate age and gender
    for conf, x, y, w, h in faces:
        # get detected face
        crop_img, top_left, bottom_right = crop_face(image, x, y, w, h, 1.2)
        if crop_img.shape[0] <= 0 or crop_img.shape[1] <= 0:
            continue

        gender, prob, age = estimate_age_gender(net, crop_img)
        logger.info(" gender is: %s (%.2f)" % (gender, prob * 100))
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


def main():
    # model files check and download
    logger.info('=== age-gender-recognition model ===')
    check_and_download_ir(WEIGHT_PATH)
    if args.detection:
        logger.info('=== face detection model ===')
        check_and_download_ir(FACE_DETECTION_ADAS_PATH)

    # net initialize
    core = ov.Core()
    net = core.compile_model(core.read_model(WEIGHT_PATH + '.xml'), 'CPU')

    logger.info(args.input)
    image = cv2.imread(args.input)

    if args.detection:
        detector = core.compile_model(
            core.read_model(FACE_DETECTION_ADAS_PATH + '.xml'), 'CPU')
        image = recognize_image(net, detector, image)
        logger.info(f'saved at : {args.savepath}')
        cv2.imwrite(args.savepath, image)
    else:
        gender, prob, age = estimate_age_gender(net, image)
        logger.info(" gender is: %s (%.2f)" % (gender, prob * 100))
        logger.info(" age is: %d" % age)

    logger.info('Script finished successfully.')


if __name__ == '__main__':
    main()
