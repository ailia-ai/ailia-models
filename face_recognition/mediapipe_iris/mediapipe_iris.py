import sys
import time

import ailia
import cv2
import numpy as np
import json

import mediapipe_iris_utils as iut

sys.path.append('../../util')
# logger
from logging import getLogger  # noqa: E402

from image_utils import imread, load_image  # noqa: E402
from model_utils import check_and_download_models  # noqa: E402
from arg_utils import get_base_parser, get_savepath, update_parser  # noqa: E402
from webcamera_utils import get_capture, get_writer  # noqa: E402

logger = getLogger(__name__)


# ======================
# Parameters 1
# ======================
IMAGE_PATH = 'man.jpg'
SAVE_IMAGE_PATH = 'output.png'
IMAGE_HEIGHT = 128
IMAGE_WIDTH = 128


# ======================
# Argument Parser Config
# ======================
parser = get_base_parser(
    'MediaPipe Iris, real-time iris estimation.', IMAGE_PATH, SAVE_IMAGE_PATH,
)
parser.add_argument(
    '-n', '--normal',
    action='store_true',
    help='By default, the optimized model is used, but with this option, ' +
    'you can switch to the normal (not optimized) model'
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
DETECTION_MODEL_NAME = 'blazeface'
LANDMARK_MODEL_NAME = 'facemesh'
LANDMARK2_MODEL_NAME = 'iris'
if args.normal:
    DETECTION_WEIGHT_PATH = f'{DETECTION_MODEL_NAME}.onnx'
    DETECTION_MODEL_PATH = f'{DETECTION_MODEL_NAME}.onnx.prototxt'
    LANDMARK_WEIGHT_PATH = f'{LANDMARK_MODEL_NAME}.onnx'
    LANDMARK_MODEL_PATH = f'{LANDMARK_MODEL_NAME}.onnx.prototxt'
    LANDMARK2_WEIGHT_PATH = f'{LANDMARK2_MODEL_NAME}.onnx'
    LANDMARK2_MODEL_PATH = f'{LANDMARK2_MODEL_NAME}.onnx.prototxt'
else:
    DETECTION_WEIGHT_PATH = f'{DETECTION_MODEL_NAME}.opt.onnx'
    DETECTION_MODEL_PATH = f'{DETECTION_MODEL_NAME}.opt.onnx.prototxt'
    LANDMARK_WEIGHT_PATH = f'{LANDMARK_MODEL_NAME}.opt.onnx'
    LANDMARK_MODEL_PATH = f'{LANDMARK_MODEL_NAME}.opt.onnx.prototxt'
    LANDMARK2_WEIGHT_PATH = f'{LANDMARK2_MODEL_NAME}.opt.onnx'
    LANDMARK2_MODEL_PATH = f'{LANDMARK2_MODEL_NAME}.opt.onnx.prototxt'
DETECTION_REMOTE_PATH = f'https://storage.googleapis.com/ailia-models/{DETECTION_MODEL_NAME}/'
LANDMARK_REMOTE_PATH = f'https://storage.googleapis.com/ailia-models/{LANDMARK_MODEL_NAME}/'
LANDMARK2_REMOTE_PATH = f'https://storage.googleapis.com/ailia-models/mediapipe_{LANDMARK2_MODEL_NAME}/'


# ======================
# Utils
# ======================
EYE_CONTOUR_ORDERED = [
    0, 1, 2, 3, 4, 5, 6, 7, 8, 15, 14, 13, 12, 11, 10, 9
]


def draw_landmarks(img, points, color=(0, 0, 255), size=2):
    for point in points:
        x, y = point
        x, y = int(x), int(y)
        cv2.circle(img, (x, y), size, color, thickness=cv2.FILLED)


def draw_eye_iris(
        img,
        eyes,
        iris,
        eye_color=(0, 0, 255),
        iris_color=(255, 0, 0),
        iris_pt_color=(0, 255, 0),
        size=1,
):
    """
    TODO: docstring
    """

    for i in range(2):
        pts = eyes[i, EYE_CONTOUR_ORDERED, :2].round().astype(np.int32)
        pts = pts.reshape((-1, 1, 2))
        cv2.polylines(img, [pts], True, eye_color, thickness=size)

        center = tuple(iris[i, 0])
        radius = int(np.linalg.norm(iris[i, 1] - iris[i, 0]).round())
        cv2.circle(img, center, radius, iris_color, thickness=size)
        draw_landmarks(img, iris[i], color=iris_pt_color, size=size)


def save_json(
        json_path,
        eyes,
        iris
):
    output = []
    for eyes_idx in range(len(eyes)):
        e = eyes[eyes_idx, :, :16, :2]
        i = iris[eyes_idx, :, :, :2]
        o = {}
        for eye_idx in range(2):
            o[f'contour_pts_{eye_idx}'] = e[eye_idx, EYE_CONTOUR_ORDERED, :2].tolist()
            o[f'landmarks_{eye_idx}'] = i[eye_idx].tolist()
        output.append(o)

    with open(json_path, 'w') as f:
        json.dump(output, f, indent=2)


# ======================
# Main functions
# ======================
def recognize_from_image():
    # net initialize
    detector = ailia.Net(
        DETECTION_MODEL_PATH, DETECTION_WEIGHT_PATH, env_id=args.env_id,
        memory_mode=args.memory_mode
    )
    estimator = ailia.Net(
        LANDMARK_MODEL_PATH, LANDMARK_WEIGHT_PATH, env_id=args.env_id,
        memory_mode=args.memory_mode
    )
    estimator2 = ailia.Net(
        LANDMARK2_MODEL_PATH, LANDMARK2_WEIGHT_PATH, env_id=args.env_id,
        memory_mode=args.memory_mode
    )

    # prepare input data
    # input image loop
    for image_path in args.input:
        # prepare input data
        logger.info(image_path)
        src_img = imread(image_path)
        _, img128, scale, pad = iut.resize_pad(src_img[:, :, ::-1])
        input_data = img128.astype('float32') / 127.5 - 1.0
        input_data = np.expand_dims(np.moveaxis(input_data, -1, 0), 0)

        # inference
        logger.info('Start inference...')
        if args.benchmark:
            logger.info('BENCHMARK mode')
            for _ in range(5):
                start = int(round(time.time() * 1000))
                # Face detection
                preds = detector.predict([input_data])
                detections = iut.detector_postprocess(preds)

                # Face landmark estimation
                if detections[0].size != 0:
                    imgs, affines, box = iut.estimator_preprocess(
                        src_img[:, :, ::-1], detections, scale, pad
                    )
                    estimator.set_input_shape(imgs.shape)
                    landmarks, confidences = estimator.predict([imgs])

                    # Iris landmark estimation
                    imgs2, origins = iut.iris_preprocess(imgs, landmarks)
                    estimator2.set_input_shape(imgs2.shape)
                    eyes, iris = estimator2.predict([imgs2])

                    eyes, iris = iut.iris_postprocess(
                        eyes, iris, origins, affines
                    )
                    for i in range(len(eyes)):
                        draw_eye_iris(
                            src_img,
                            eyes[i, :, :16, :2],
                            iris[i, :, :, :2],
                            size=1
                        )
                end = int(round(time.time() * 1000))
                logger.info(f'\tailia processing time {end - start} ms')
        else:
            # Face detection
            preds = detector.predict([input_data])
            detections = iut.detector_postprocess(preds)

            # Face landmark estimation
            if detections[0].size != 0:
                imgs, affines, box = iut.estimator_preprocess(
                    src_img[:, :, ::-1], detections, scale, pad
                )
                estimator.set_input_shape(imgs.shape)
                landmarks, confidences = estimator.predict([imgs])

                # Iris landmark estimation
                imgs2, origins = iut.iris_preprocess(imgs, landmarks)
                estimator2.set_input_shape(imgs2.shape)
                eyes, iris = estimator2.predict([imgs2])

                eyes, iris = iut.iris_postprocess(eyes, iris, origins, affines)
                for i in range(len(eyes)):
                    draw_eye_iris(
                        src_img, eyes[i, :, :16, :2], iris[i, :, :, :2], size=1
                    )

        savepath = get_savepath(args.savepath, image_path)
        logger.info(f'saved at : {savepath}')
        cv2.imwrite(savepath, src_img)

        if args.write_json:
            save_json((savepath.rsplit('.', 1)[0]) + '.json', eyes, iris)

    logger.info('Script finished successfully.')


def recognize_from_video():
    # net initialize
    detector = ailia.Net(
        DETECTION_MODEL_PATH, DETECTION_WEIGHT_PATH, env_id=args.env_id,
        memory_mode=args.memory_mode
    )
    estimator = ailia.Net(
        LANDMARK_MODEL_PATH, LANDMARK_WEIGHT_PATH, env_id=args.env_id,
        memory_mode=args.memory_mode
    )
    estimator2 = ailia.Net(
        LANDMARK2_MODEL_PATH, LANDMARK2_WEIGHT_PATH, env_id=args.env_id,
        memory_mode=args.memory_mode
    )

    capture = get_capture(args.video)

    # create video writer if savepath is specified as video format
    if args.savepath != SAVE_IMAGE_PATH:
        f_h = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        f_w = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        writer = get_writer(args.savepath, f_h, f_w)
    else:
        writer = None

    frame_shown = False
    while(True):
        ret, frame = capture.read()
        if (cv2.waitKey(1) & 0xFF == ord('q')) or not ret:
            break
        if frame_shown and cv2.getWindowProperty('frame', cv2.WND_PROP_VISIBLE) == 0:
            break


        _, img128, scale, pad = iut.resize_pad(frame[:,:,::-1])
        input_data = img128.astype('float32') / 127.5 - 1.0
        input_data = np.expand_dims(np.moveaxis(input_data, -1, 0), 0)

        # inference
        # Face detection
        preds = detector.predict([input_data])
        detections = iut.detector_postprocess(preds)

        # Face landmark estimation
        if detections[0].size != 0:
            imgs, affines, box = iut.estimator_preprocess(
                frame[:, :, ::-1], detections, scale, pad
            )

            dynamic_shape = False

            if dynamic_shape:
                estimator.set_input_shape(imgs.shape)
                landmarks, confidences = estimator.predict([imgs])
            else:
                landmarks = np.zeros((imgs.shape[0], 1404))
                confidences = np.zeros((imgs.shape[0], 1))
                for i in range(imgs.shape[0]):
                    landmarks[i, :], confidences[i, :] = estimator.predict(
                        [imgs[i:i+1, :, :, :]]
                    )

            # Iris landmark estimation
            imgs2, origins = iut.iris_preprocess(imgs, landmarks)

            if dynamic_shape:
                estimator2.set_input_shape(imgs2.shape)
                eyes, iris = estimator2.predict([imgs2])
            else:
                eyes = np.zeros((imgs2.shape[0], 213))
                iris = np.zeros((imgs2.shape[0], 15))
                for i in range(imgs2.shape[0]):
                    eyes[i, :], iris[i, :] = estimator2.predict(
                        [imgs2[i:i+1, :, :, :]]
                    )

            eyes, iris = iut.iris_postprocess(eyes, iris, origins, affines)
            for i in range(len(eyes)):
                draw_eye_iris(
                    frame, eyes[i, :, :16, :2], iris[i, :, :, :2], size=1
                )

        visual_img = frame
        if args.video == '0': # Flip horizontally if camera
            visual_img = np.ascontiguousarray(frame[:,::-1,:])

        cv2.imshow('frame', visual_img)
        frame_shown = True

        # save results
        if writer is not None:
            writer.write(frame)

    capture.release()
    if writer is not None:
        writer.release()
    cv2.destroyAllWindows()
    logger.info('Script finished successfully.')
    pass


def main():
    # model files check and download
    check_and_download_models(
        DETECTION_WEIGHT_PATH, DETECTION_MODEL_PATH, DETECTION_REMOTE_PATH
    )
    check_and_download_models(
        LANDMARK_WEIGHT_PATH, LANDMARK_MODEL_PATH, LANDMARK_REMOTE_PATH
    )
    check_and_download_models(
        LANDMARK2_WEIGHT_PATH, LANDMARK2_MODEL_PATH, LANDMARK2_REMOTE_PATH
    )

    if args.video is not None:
        # video mode
        recognize_from_video()
    else:
        # image mode
        recognize_from_image()


if __name__ == '__main__':
    main()
