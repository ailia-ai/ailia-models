import sys

import cv2
import numpy as np

import ailia

# import original modules
sys.path.append('../../util')
from arg_utils import get_base_parser, update_parser  # noqa: E402
from model_utils import check_and_download_models  # noqa: E402
from webcamera_utils import get_capture, get_writer, \
    calc_adjust_fsize  # noqa: E402

# logger
from logging import getLogger   # noqa: E402
logger = getLogger(__name__)


# ======================
# Parameters
# ======================
WEIGHT_PATH = 'yolov3-hand.opt.onnx'
MODEL_PATH = 'yolov3-hand.opt.onnx.prototxt'
REMOTE_PATH = 'https://storage.googleapis.com/ailia-models/yolov3-hand/'

SAVE_PATH = ''

# NOTE: commercial model
logger.warning('The pre-trained model is not available freely')
HAND_WEIGHT_PATH = 'hand_obf.caffemodel'
HAND_MODEL_PATH = 'hand_obf.prototxt'
HAND_REMOTE_PATH = ''
HAND_ALGORITHM = ailia.POSE_ALGORITHM_ACCULUS_HAND

HAND_CATEGORY = ['hand']
THRESHOLD = 0.2
IOU = 0.45


# ======================
# Arguemnt Parser Config
# ======================
parser = get_base_parser('Acculus hand detection model', None, SAVE_PATH)
args = update_parser(parser)


# ======================
# Utils
# ======================
def hsv_to_rgb(h, s, v):
    bgr = cv2.cvtColor(
        np.array([[[h, s, v]]], dtype=np.uint8), cv2.COLOR_HSV2BGR
    )[0][0]
    return (int(bgr[2]), int(bgr[1]), int(bgr[0]))


def line(input_img, hand_keypoint, point1, point2, offset, scale):
    threshold = 0.3
    if hand_keypoint.points[point1].score > threshold and\
       hand_keypoint.points[point2].score > threshold:
        color = hsv_to_rgb(255*point2/ailia.POSE_KEYPOINT_CNT, 255, 255)

        x1 = int(hand_keypoint.points[point1].x * scale[0] + offset[0])
        y1 = int(hand_keypoint.points[point1].y * scale[1] + offset[1])
        x2 = int(hand_keypoint.points[point2].x * scale[0] + offset[0])
        y2 = int(hand_keypoint.points[point2].y * scale[1] + offset[1])

        w = int(input_img.shape[0] / 100)
        cv2.line(input_img, (x1, y1), (x2, y2), color, w)


def display_result(input_img, hand, top_left, bottom_right):
    count = hand.get_object_count()
    for idx in range(count):
        hand_keypoint = hand.get_object_hand(idx)

        offset = top_left
        scale = (bottom_right[0] - top_left[0], bottom_right[1] - top_left[1])

        line(input_img, hand_keypoint, 0, 1, offset, scale)
        line(input_img, hand_keypoint, 1, 2, offset, scale)
        line(input_img, hand_keypoint, 2, 3, offset, scale)
        line(input_img, hand_keypoint, 3, 4, offset, scale)

        line(input_img, hand_keypoint, 0, 5, offset, scale)
        line(input_img, hand_keypoint, 5, 6, offset, scale)
        line(input_img, hand_keypoint, 6, 7, offset, scale)
        line(input_img, hand_keypoint, 7, 8, offset, scale)

        line(input_img, hand_keypoint, 0, 9, offset, scale)
        line(input_img, hand_keypoint, 9, 10, offset, scale)
        line(input_img, hand_keypoint, 10, 11, offset, scale)
        line(input_img, hand_keypoint, 11, 12, offset, scale)

        line(input_img, hand_keypoint, 0, 13, offset, scale)
        line(input_img, hand_keypoint, 13, 14, offset, scale)
        line(input_img, hand_keypoint, 14, 15, offset, scale)
        line(input_img, hand_keypoint, 15, 16, offset, scale)

        line(input_img, hand_keypoint, 0, 17, offset, scale)
        line(input_img, hand_keypoint, 17, 18, offset, scale)
        line(input_img, hand_keypoint, 18, 19, offset, scale)
        line(input_img, hand_keypoint, 19, 20, offset, scale)


# ======================
# Main functions
# ======================
def recognize_from_video():
    # net initialize
    detector = ailia.Detector(
        MODEL_PATH,
        WEIGHT_PATH,
        len(HAND_CATEGORY),
        format=ailia.NETWORK_IMAGE_FORMAT_RGB,
        channel=ailia.NETWORK_IMAGE_CHANNEL_FIRST,
        range=ailia.NETWORK_IMAGE_RANGE_U_FP32,
        algorithm=ailia.DETECTOR_ALGORITHM_YOLOV3,
        env_id=args.env_id,
        memory_mode=args.memory_mode
    )

    hand = ailia.PoseEstimator(
        HAND_MODEL_PATH,
        HAND_WEIGHT_PATH,
        env_id=args.env_id,
        algorithm=HAND_ALGORITHM,
        memory_mode=args.memory_mode
    )
    hand.set_threshold(0.1)

    capture = get_capture(args.video)
    # create video writer if savepath is specified as video format
    if args.savepath != SAVE_PATH:
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

        img = cv2.cvtColor(frame, cv2.COLOR_BGR2BGRA)
        detector.compute(img, THRESHOLD, IOU)

        h, w = img.shape[0], img.shape[1]
        count = detector.get_object_count()
        for idx in range(count):
            # get detected hand
            obj = detector.get_object(idx)
            margin = 1.0
            cx = (obj.x + obj.w/2) * w
            cy = (obj.y + obj.h/2) * h
            cw = max(obj.w * w, obj.h * h) * margin
            fx = max(cx - cw/2, 0)
            fy = max(cy - cw/2, 0)
            fw = min(cw, w-fx)
            fh = min(cw, h-fy)
            top_left = (int(fx), int(fy))
            bottom_right = (int(fx+fw), int(fy+fh))

            # display detected hand
            color = hsv_to_rgb(0, 255, 255)
            cv2.rectangle(frame, top_left, bottom_right, color, 4)

            # get detected face
            crop_img = img[
                top_left[1]:bottom_right[1], top_left[0]:bottom_right[0], 0:4
            ]
            if crop_img.shape[0] <= 0 or crop_img.shape[1] <= 0:
                continue

            # inference
            _ = hand.compute(crop_img.astype(np.uint8, order='C'))

            # postprocessing
            display_result(frame, hand, top_left, bottom_right)

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
    check_and_download_models(
        HAND_WEIGHT_PATH, HAND_MODEL_PATH, HAND_REMOTE_PATH
    )
    check_and_download_models(WEIGHT_PATH, MODEL_PATH, REMOTE_PATH)

    if args.video is None:
        logger.info("Video file required.")
        return

    # video mode
    recognize_from_video()


if __name__ == '__main__':
    main()
