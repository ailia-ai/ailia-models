import sys
import time

import ailia
import cv2
import numpy as np

import facemesh_utils as fut

sys.path.append('../../util')
# logger
from logging import getLogger  # noqa: E402

from image_utils import imread, load_image  # noqa: E402
from model_utils import check_and_download_models  # noqa: E402
from arg_utils import get_base_parser, get_savepath, update_parser  # noqa: E402
from webcamera_utils import get_capture, get_writer  # noqa: E402
from facemesh_const import FACEMESH_TESSELATION

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
    'Face Mesh, an on-device real-time face recognition.',
    IMAGE_PATH,
    SAVE_IMAGE_PATH,
)
parser.add_argument(
    '-m', '--model_type', default='normal', choices=('legacy', 'normal', 'refine'),
    help='facial landmark model type, legacy model uses reflectionpad2d, normal model uses constantpad2d, refine model uses attention'
)
parser.add_argument(
    '--back',
    action='store_true',
    help='By default, the front camera model is used for face detection, but with this option, ' +
    'you can switch to the back camera model'
)
args = update_parser(parser)


# ======================
# Parameters 2
# ======================
DETECTION_MODEL_NAME = 'blazeface'
if args.model_type == "refine":
    LANDMARK_MODEL_NAME = 'mediapipe_holistic'
else:
    LANDMARK_MODEL_NAME = 'facemesh'

if args.back:
    DETECTION_MODEL_DETAIL_NAME = 'blazefaceback'
else:
    DETECTION_MODEL_DETAIL_NAME = 'blazeface'

if args.model_type == "legacy":
    LANDMARK_MODEL_DETAIL_NAME = 'facemesh'
elif args.model_type == "normal":
    LANDMARK_MODEL_DETAIL_NAME = 'facemesh_constantpad2d'   # https://github.com/thepowerfuldeez/facemesh.pytorch/issues/3
elif args.model_type == "refine":
    LANDMARK_MODEL_DETAIL_NAME = 'face_landmark_with_attention'

if args.model_type == "normal":
    LANDMARK_WEIGHT_PATH = f'{LANDMARK_MODEL_DETAIL_NAME}.opt.onnx'
    LANDMARK_MODEL_PATH = f'{LANDMARK_MODEL_DETAIL_NAME}.opt.onnx.prototxt'
else:
    LANDMARK_WEIGHT_PATH = f'{LANDMARK_MODEL_DETAIL_NAME}.onnx'
    LANDMARK_MODEL_PATH = f'{LANDMARK_MODEL_DETAIL_NAME}.onnx.prototxt'

if args.back:
    DETECTION_WEIGHT_PATH = f'{DETECTION_MODEL_DETAIL_NAME}.onnx'
    DETECTION_MODEL_PATH = f'{DETECTION_MODEL_DETAIL_NAME}.onnx.prototxt'
else:
    DETECTION_WEIGHT_PATH = f'{DETECTION_MODEL_DETAIL_NAME}.opt.onnx'
    DETECTION_MODEL_PATH = f'{DETECTION_MODEL_DETAIL_NAME}.opt.onnx.prototxt'

DETECTION_REMOTE_PATH = f'https://storage.googleapis.com/ailia-models/{DETECTION_MODEL_NAME}/'
LANDMARK_REMOTE_PATH = f'https://storage.googleapis.com/ailia-models/{LANDMARK_MODEL_NAME}/'


# ======================
# Utils
# ======================
def draw_roi(img, roi):
    if len(roi) <= 0:
        return
    for i in range(roi.shape[0]):
        (x1, x2, x3, x4), (y1, y2, y3, y4) = roi[i]
        cv2.line(img, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 0), 2)
        cv2.line(img, (int(x1), int(y1)), (int(x3), int(y3)), (0, 255, 0), 2)
        cv2.line(img, (int(x2), int(y2)), (int(x4), int(y4)), (0, 0, 0), 2)
        cv2.line(img, (int(x3), int(y3)), (int(x4), int(y4)), (0, 0, 0), 2)


def draw_landmarks(img, points, color=(0, 0, 255), size=2):
    for point in points:
        x, y = point
        x, y = int(x), int(y)
        cv2.circle(img, (x, y), size, color, thickness=cv2.FILLED)


def estimate_landmarks(input_data, src_img, scale, pad, detector, fut, estimator):
    # Face detection
    preds = detector.predict([input_data])
    detections = fut.detector_postprocess(preds, back = args.back)

    # Face landmark estimation
    if detections[0].size != 0:
        imgs, affines, box = fut.estimator_preprocess(
            src_img[:, :, ::-1], detections, scale, pad
        ) # BGR -> RGB

        dynamic_input_shape = False

        if dynamic_input_shape:
            estimator.set_input_shape(imgs.shape)
            if args.model_type == "refine":
                outputs = estimator.predict([imgs[i:i+1, :, :, :]])
                landmark = outputs[4]
                confidences = outputs[0]
            else:
                landmarks, confidences = estimator.predict([imgs])
            landmarks = landmarks.reshape((imgs.shape[0],1404))
            confidences = confidences.reshape((imgs.shape[0],1))
            normalized_landmarks = landmarks / 192.0
            landmarks = fut.denormalize_landmarks(
                normalized_landmarks, affines
            )
        else:
            landmarks = np.zeros((imgs.shape[0], 1404))
            confidences = np.zeros((imgs.shape[0], 1))

            for i in range(imgs.shape[0]):
                if args.model_type == "refine":
                    outputs = estimator.predict([imgs[i:i+1, :, :, :]])
                    landmark = outputs[4]
                    confidences[i, :] = outputs[0]
                else:
                    landmark, confidences[i, :] = estimator.predict([imgs[i:i+1, :, :, :]])
                normalized_landmark = landmark / 192.0

                # postprocessing
                landmarks[i, :] = normalized_landmark
        
            landmarks = fut.denormalize_landmarks(
                landmarks, affines
            )

        return landmarks, confidences, box

    return [], [], []


def draw_face_landmarks(
        image,
        landmark_list,
        size = 1):
    for connection in FACEMESH_TESSELATION:
        start_idx = connection[0]
        end_idx = connection[1]
        sx, sy = int(landmark_list[start_idx][0]), int(landmark_list[start_idx][1])
        ex, ey = int(landmark_list[end_idx][0]), int(landmark_list[end_idx][1])
        cv2.line(
            image, (sx,sy), (ex,ey),
            (0, 255, 0), size)


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

    # input image loop
    for image_path in args.input:
        # prepare input data
        logger.info(image_path)
        src_img = imread(image_path)
        img256, img128, scale, pad = fut.resize_pad(src_img[:, :, ::-1])
        if args.back:
            input_data = img256.astype('float32') / 127.5 - 1.0
        else:
            input_data = img128.astype('float32') / 127.5 - 1.0
        input_data = np.expand_dims(np.moveaxis(input_data, -1, 0), 0)

        # inference
        logger.info('Start inference...')
        if args.benchmark:
            logger.info('BENCHMARK mode')
            total_time = 0
            for i in range(args.benchmark_count):
                start = int(round(time.time() * 1000))
                landmarks, confidences, box = estimate_landmarks(input_data, src_img, scale, pad, detector, fut, estimator)
                end = int(round(time.time() * 1000))
                if i != 0:
                    total_time = total_time + (end - start)
                logger.info(f'\tailia processing time {end - start} ms')
            logger.info(f'\taverage time {total_time / (args.benchmark_count-1)} ms')
        else:
            landmarks, confidences, box = estimate_landmarks(input_data, src_img, scale, pad, detector, fut, estimator)
        
        draw_roi(src_img, box)
        for i in range(len(landmarks)):
            landmark, confidence = landmarks[i], confidences[i]
            #draw_landmarks(src_img, landmark[:, :2], size=1)
            draw_face_landmarks(src_img, landmark[:, :2], size=1)

        savepath = get_savepath(args.savepath, image_path)
        logger.info(f'saved at : {savepath}')
        cv2.imwrite(savepath, src_img)
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

        _, img128, scale, pad = fut.resize_pad(frame[:,:,::-1])
        input_data = img128.astype('float32') / 127.5 - 1.0
        input_data = np.expand_dims(np.moveaxis(input_data, -1, 0), 0)

        landmarks, confidences, box = estimate_landmarks(input_data, frame, scale, pad, detector, fut, estimator)

        draw_roi(frame, box)

        for i in range(len(landmarks)):
            landmark, confidence = landmarks[i], confidences[i]
            # if confidence > 0:
            # Can be > 1, no idea what it represents
            #draw_landmarks(frame, landmark[:, :2], size=1)
            draw_face_landmarks(frame, landmark[:, :2], size=1)

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
    if writer is not None:
        writer.release()
    logger.info('Script finished successfully.')


def main():
    # model files check and download
    check_and_download_models(
        DETECTION_WEIGHT_PATH, DETECTION_MODEL_PATH, DETECTION_REMOTE_PATH
    )
    check_and_download_models(
        LANDMARK_WEIGHT_PATH, LANDMARK_MODEL_PATH, LANDMARK_REMOTE_PATH
    )

    if args.video is not None:
        # video mode
        recognize_from_video()
    else:
        # image mode
        recognize_from_image()


if __name__ == '__main__':
    main()
