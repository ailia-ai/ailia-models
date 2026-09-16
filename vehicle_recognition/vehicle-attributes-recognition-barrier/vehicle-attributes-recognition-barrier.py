import sys
import time

import cv2
import numpy as np

import ailia

sys.path.append('../../util')
from arg_utils import get_base_parser, update_parser, get_savepath  # noqa: E402
from detector_utils import load_image  # noqa: E402
from model_utils import check_and_download_models  # noqa: E402
import webcamera_utils  # noqa: E402
from nms_utils import nms_between_categories  # noqa: E402

# logger
from logging import getLogger  # noqa: E402

logger = getLogger(__name__)

# ======================
# Parameters
# ======================

WEIGHT_PATH = 'vehicle-attributes-recognition-barrier-0042.onnx'
MODEL_PATH = 'vehicle-attributes-recognition-barrier-0042.onnx.prototxt'
REMOTE_PATH = 'https://storage.googleapis.com/ailia-models/vehicle-attributes-recognition-barrier/'
DT_WEIGHT_PATH = 'yolov3.opt2.onnx'
DT_MODEL_PATH = 'yolov3.opt2.onnx.prototxt'
DT_REMOTE_PATH = 'https://storage.googleapis.com/ailia-models/yolov3/'

IMAGE_PATH = 'demo.png'
IMAGE_SIZE = 72
SAVE_IMAGE_PATH = 'output.png'

COLOR_LIST = (
    'white', 'gray', 'yellow', 'red', 'green', 'blue', 'black'
)
COLOR_TABLE_LIST = (
    (255,255,255), (192,192,192), (128,255,255), (128,128,255), (128,255,128), (255,128,128), (128,128,128)
)
TYPE_LIST = (
    'car', 'van', 'truck', 'bus'
)

# yolo params
COCO_CATEGORY = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train",
    "truck", "boat", "traffic light", "fire hydrant", "stop sign",
    "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow",
    "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
    "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard",
    "sports ball", "kite", "baseball bat", "baseball glove", "skateboard",
    "surfboard", "tennis racket", "bottle", "wine glass", "cup", "fork",
    "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange",
    "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair",
    "couch", "potted plant", "bed", "dining table", "toilet", "tv",
    "laptop", "mouse", "remote", "keyboard", "cell phone", "microwave",
    "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase",
    "scissors", "teddy bear", "hair drier", "toothbrush"
]
THRESHOLD = 0.4
IOU = 0.45
DETECT_CLASSES = [2, 5, 7]

# ======================
# Argument Parser Config
# ======================

parser = get_base_parser(
    'vehicle-attributes-recognition-barrier', IMAGE_PATH, SAVE_IMAGE_PATH,
)
parser.add_argument(
    '-d', '--detection',
    action='store_true',
    help='Use object detection.'
)
parser.add_argument(
    '-th', '--threshold',
    default=THRESHOLD, type=float,
    help='The detection threshold for yolo. (default: ' + str(THRESHOLD) + ')'
)
parser.add_argument(
    '-iou', '--iou',
    default=IOU, type=float,
    help='The detection iou for yolo. (default: ' + str(IOU) + ')'
)
args = update_parser(parser)


# ======================
# Utils
# ======================

def crop(obj, margin, frame):
    w = frame.shape[1]
    h = frame.shape[0]
    cx = (obj.x + obj.w / 2) * w
    cy = (obj.y + obj.h / 2) * h
    cw = max(obj.w * w * margin, obj.h * h * margin)
    fx = max(cx - cw / 2, 0)
    fy = max(cy - cw / 2, 0)
    fw = min(cw, w - fx)
    fh = min(cw, h - fy)
    top_left = (int(fx), int(fy))
    bottom_right = (int((fx + fw)), int(fy + fh))
    crop_img = frame[
               top_left[1]:bottom_right[1],
               top_left[0]:bottom_right[0], 0:3
               ]
    return crop_img, top_left, bottom_right


# ======================
# Main functions
# ======================

def recognize_from_frame(net, detector, frame):
    # object detection
    detector.compute(frame, args.threshold, args.iou)

    frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)

    # nms
    detections = []
    for idx in range(detector.get_object_count()):
        obj = detector.get_object(idx)
        detections.append(obj)
    detections = nms_between_categories(
        detections,
        frame.shape[1],
        frame.shape[0],
        categories=DETECT_CLASSES,
        iou_threshold=IOU,
    )

    for idx in range(len(detections)):
        obj = detections[idx]
        if obj.category not in DETECT_CLASSES:
            continue

        # cropping image
        margin = 1.0
        crop_img, top_left, bottom_right = crop(
            obj, margin, frame
        )
        # inference
        img = cv2.resize(crop_img, (IMAGE_SIZE, IMAGE_SIZE))
        img = np.expand_dims(img, axis=0)  # 次元合せ

        output = net.predict([img])
        out_typ, out_clr = output
        typ = TYPE_LIST[np.argmax(out_typ)]
        clr = COLOR_LIST[np.argmax(out_clr)]
        clr_table = COLOR_TABLE_LIST[np.argmax(out_clr)]

        # draw label
        LABEL_WIDTH = bottom_right[1] - top_left[1]
        LABEL_HEIGHT = 20
        color = clr_table#(255, 128, 128)
        cv2.rectangle(frame, top_left, bottom_right, color, thickness=2)
        cv2.rectangle(
            frame,
            top_left,
            (top_left[0] + LABEL_WIDTH, top_left[1] + LABEL_HEIGHT),
            color,
            thickness=-1,
        )

        text_position = (top_left[0], top_left[1] + LABEL_HEIGHT * 3 // 4)
        color = (0, 0, 0)
        fontScale = 0.7
        cv2.putText(
            frame,
            "{} {}".format(typ, clr),
            text_position,
            cv2.FONT_HERSHEY_SIMPLEX,
            fontScale,
            color,
            1,
        )

    return frame


def recognize_from_image(net, detector):
    # input image loop
    for image_path in args.input:
        # prepare input data
        logger.info(image_path)

        img = load_image(image_path)

        if detector:
            img = recognize_from_frame(net, detector, img)
            savepath = get_savepath(args.savepath, image_path)
            logger.info(f'saved at : {savepath}')
            cv2.imwrite(savepath, img)
            continue

        img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
        img = cv2.resize(img, (IMAGE_SIZE, IMAGE_SIZE))
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

        out_typ, out_clr = output
        typ = TYPE_LIST[np.argmax(out_typ)]
        clr = COLOR_LIST[np.argmax(out_clr)]

        logger.info("- Type: %s" % typ)
        logger.info("- Color: %s" % clr)

    logger.info('Script finished successfully.')


def recognize_from_video(net, detector):
    capture = webcamera_utils.get_capture(args.video)

    # create video writer if savepath is specified as video format
    if args.savepath != SAVE_IMAGE_PATH:
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

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2BGRA)
        frame = recognize_from_frame(net, detector, frame)

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
    logger.info('Check vehicle-attributes-recognition model...')
    check_and_download_models(
        WEIGHT_PATH, MODEL_PATH, REMOTE_PATH
    )
    if args.video or args.detection:
        logger.info('Check object detection model...')
        check_and_download_models(
            DT_WEIGHT_PATH, DT_MODEL_PATH, DT_REMOTE_PATH
        )

    env_id = args.env_id

    # net initialize
    net = ailia.Net(
        MODEL_PATH, WEIGHT_PATH, env_id=env_id, memory_mode=args.memory_mode
    )
    if args.video or args.detection:
        detector = ailia.Detector(
            DT_MODEL_PATH,
            DT_WEIGHT_PATH,
            len(COCO_CATEGORY),
            format=ailia.NETWORK_IMAGE_FORMAT_RGB,
            channel=ailia.NETWORK_IMAGE_CHANNEL_FIRST,
            range=ailia.NETWORK_IMAGE_RANGE_U_FP32,
            algorithm=ailia.DETECTOR_ALGORITHM_YOLOV3,
            env_id=env_id,
            memory_mode=args.memory_mode,
        )
    else:
        detector = None

    if args.video:
        # video mode
        recognize_from_video(net, detector)
    else:
        # image mode
        recognize_from_image(net, detector)


if __name__ == '__main__':
    main()
