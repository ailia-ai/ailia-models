import sys
import time

import ailia
import cv2
import numpy as np

import blazehand_utils as but

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
IMAGE_PATH = 'person_hand.jpg'
SAVE_IMAGE_PATH = 'output.png'
IMAGE_HEIGHT = 256
IMAGE_WIDTH = 256


# ======================
# Argument Parser Config
# ======================
parser = get_base_parser(
    'BlazeHand, an on-device real-time hand tracking.',
    IMAGE_PATH,
    SAVE_IMAGE_PATH,
)
parser.add_argument(
    '--hands',
    metavar='NUM_HANDS',
    type=int,
    default=2,
    help='The maximum number of hands tracked (=2 by default)'
)
args = update_parser(parser)


# ======================
# Parameters 2
# ======================
DETECTION_MODEL_NAME = 'blazepalm'
LANDMARK_MODEL_NAME = 'blazehand'
# if args.normal:
DETECTION_WEIGHT_PATH = f'{DETECTION_MODEL_NAME}.onnx'
DETECTION_MODEL_PATH = f'{DETECTION_MODEL_NAME}.onnx.prototxt'
LANDMARK_WEIGHT_PATH = f'{LANDMARK_MODEL_NAME}.onnx'
LANDMARK_MODEL_PATH = f'{LANDMARK_MODEL_NAME}.onnx.prototxt'
# else:
#     DETECTION_WEIGHT_PATH = f'{DETECTION_MODEL_NAME}.opt.onnx'
#     DETECTION_MODEL_PATH = f'{DETECTION_MODEL_NAME}.opt.onnx.prototxt'
#     LANDMARK_WEIGHT_PATH = f'{LANDMARK_MODEL_NAME}.opt.onnx'
#     LANDMARK_MODEL_PATH = f'{LANDMARK_MODEL_NAME}.opt.onnx.prototxt'
DETECTION_REMOTE_PATH = f'https://storage.googleapis.com/ailia-models/{DETECTION_MODEL_NAME}/'
LANDMARK_REMOTE_PATH = f'https://storage.googleapis.com/ailia-models/{LANDMARK_MODEL_NAME}/'


# ======================
# Utils
# ======================
def draw_landmarks(img, points, connections=[], color=(0, 0, 255), size=2):
    for connection in connections:
        x0, y0 = points[connection[0]]
        x1, y1 = points[connection[1]]
        x0, y0 = int(x0), int(y0)
        x1, y1 = int(x1), int(y1)
        cv2.line(img, (x0, y0), (x1, y1), (0, 255, 0), size)
    for point in points:
        x, y = point
        x, y = int(x), int(y)
        cv2.circle(img, (x, y), size+1, color, thickness=cv2.FILLED)


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
        img256, _, scale, pad = but.resize_pad(src_img[:, :, ::-1])
        input_data = img256.astype('float32') / 255.
        input_data = np.expand_dims(np.moveaxis(input_data, -1, 0), 0)

        # Palm detection
        preds = detector.predict([input_data])
        detections = but.detector_postprocess(preds)

        # Hand landmark estimation
        presence = [0, 0]  # [left, right]
        if detections[0].size != 0:
            imgs, affines, _ = but.estimator_preprocess(src_img, detections[0], scale, pad)
            estimator.set_input_shape(imgs.shape)

            if args.benchmark:
                logger.info('BENCHMARK mode')
                for _ in range(5):
                    start = int(round(time.time() * 1000))
                    flags, handedness, normed_landmarks = estimator.predict([imgs])
                    end = int(round(time.time() * 1000))
                    logger.info(f'\tailia processing time {end - start} ms')
            else:
                flags, handedness, normed_landmarks = estimator.predict([imgs])

            # postprocessing
            landmarks = but.denormalize_landmarks(
                normed_landmarks, affines
            )
            for i in range(len(flags)):
                landmark, flag, handed = landmarks[i], flags[i], handedness[i]
                if flag > 0.75:
                    if handed > 0.5: # Right handedness when not flipped camera input
                        presence[0] = 1
                    else:
                        presence[1] = 1
                    draw_landmarks(src_img, landmark[:,:2], but.HAND_CONNECTIONS, size=2)

        if presence[0] and presence[1]:
            hand_presence = 'Left and right'
        elif presence[0]:
            hand_presence = 'Right'
        elif presence[1]:
            hand_presence = 'Left'
        else:
            hand_presence = 'No hand'
        logger.info(f'Hand presence: {hand_presence}')
        savepath = get_savepath(args.savepath, image_path)
        logger.info(f'saved at : {savepath}')        
        cv2.imwrite(savepath, src_img)
    logger.info('Script finished successfully.')


def recognize_from_video():
    # net initialize
    detector = ailia.Net(DETECTION_MODEL_PATH, DETECTION_WEIGHT_PATH, env_id=args.env_id, memory_mode=args.memory_mode)
    estimator = ailia.Net(LANDMARK_MODEL_PATH, LANDMARK_WEIGHT_PATH, env_id=args.env_id, memory_mode=args.memory_mode)
    num_hands = args.hands
    thresh = 0.5
    tracking = False
    tracked_hands = np.array([0.0] * num_hands)
    rois = [None] * num_hands

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

        img256, _, scale, pad = but.resize_pad(frame[:, :, ::-1])
        input_data = img256.astype('float32') / 255.
        input_data = np.expand_dims(np.moveaxis(input_data, -1, 0), 0)

        # inference
        # Perform palm detection on 1st frame and if at least 1 hand has low
        # confidence (not detected)
        if np.any(tracked_hands < thresh):
            tracking = False
            # Palm detection
            preds = detector.predict([input_data])
            detections = but.detector_postprocess(preds)
            if detections[0].size > 0:
                tracking = True
                roi_imgs, affines, _ = but.estimator_preprocess(frame, detections[0][:num_hands], scale, pad)
        else:
            for i, roi in enumerate(rois):
                xc, yc, scale, theta = roi
                roi_img, affine, _ = but.extract_roi(frame, xc, yc, theta, scale)
                roi_imgs[i] = roi_img[0]
                affines[i] = affine[0]

        # Hand landmark estimation
        presence = [0, 0] # [left, right]
        if tracking:
            estimator.set_input_shape(roi_imgs.shape)
            hand_flags, handedness, normalized_landmarks = estimator.predict([roi_imgs])

            # postprocessing
            landmarks = but.denormalize_landmarks(normalized_landmarks, affines)

            tracked_hands[:] = 0
            n_imgs = len(hand_flags)
            for i in range(n_imgs):
                landmark, hand_flag, handed = landmarks[i], hand_flags[i], handedness[i]
                if hand_flag > thresh:
                    if handed > 0.5: # Right handedness when not flipped camera input
                        presence[0] = 1
                    else:
                        presence[1] = 1
                    draw_landmarks(
                        frame, landmark[:, :2], but.HAND_CONNECTIONS, size=2
                    )

                    rois[i] = but.landmarks2roi(normalized_landmarks[i], affines[i])
                tracked_hands[i] = hand_flag

        if presence[0] and presence[1]:
            text = 'Left and right'
        elif presence[0]:
            text = 'Right'
        elif presence[1]:
            text = 'Left'
        else:
            text = 'No hand'

        visual_img = frame
        if args.video == '0': # Flip horizontally if camera
            visual_img = np.ascontiguousarray(frame[:,::-1,:])

        cv2.putText(visual_img, text, (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 255), 2)
        cv2.imshow('frame', visual_img)
        frame_shown = True

        # save results
        if writer is not None:
            cv2.putText(frame, text, (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 255), 2)
            writer.write(frame)

    capture.release()
    if writer is not None:
        writer.release()
    cv2.destroyAllWindows()
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
