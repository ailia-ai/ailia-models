import os
import math
import sys
import time

import numpy as np
import cv2

import ailia

# import original modules
sys.path.append('../../util')
from arg_utils import get_base_parser, update_parser, get_savepath  # noqa: E402
from model_utils import check_and_download_models  # noqa: E402
from detector_utils import plot_results, write_predictions  # noqa: E402
from detector_utils import load_image, letterbox_convert, reverse_letterbox  # noqa: E402
import webcamera_utils  # noqa: E402

from yolov4_tiny_utils import post_processing  # noqa: E402

# logger
from logging import getLogger   # noqa: E402
logger = getLogger(__name__)


# ======================
# Parameters
# ======================
DETECTION_SIZE_LISTS = [416, 640, 1280]

REMOTE_PATH = 'https://storage.googleapis.com/ailia-models/yolov4-tiny/'

IMAGE_PATH = 'input.jpg'
SAVE_IMAGE_PATH = 'output.png'

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
THRESHOLD = 0.25
IOU = 0.45


# ======================
# Arguemnt Parser Config
# ======================
parser = get_base_parser('Yolov4-tiny model', IMAGE_PATH, SAVE_IMAGE_PATH)
parser.add_argument(
    '-th', '--threshold',
    default=THRESHOLD, type=float,
    help='The detection threshold for yolo. (default: '+str(THRESHOLD)+')'
)
parser.add_argument(
    '-iou', '--iou',
    default=IOU, type=float,
    help='The detection iou for yolo. (default: '+str(IOU)+')'
)
parser.add_argument(
    '-w', '--write_prediction',
    nargs='?',
    const='txt',
    choices=['txt', 'json'],
    type=str,
    help='Output results to txt or json file.'
)
parser.add_argument(
    '-dw', '--detection_width', metavar='DETECTION_WIDTH',
    default=DETECTION_SIZE_LISTS[0], choices=DETECTION_SIZE_LISTS, type=int,
    help='detection size lists: ' + ' | '.join(map(str,DETECTION_SIZE_LISTS))
)
parser.add_argument(
    '-dh', '--detection_height', metavar='DETECTION_HEIGHT',
    default=DETECTION_SIZE_LISTS[0], choices=DETECTION_SIZE_LISTS, type=int,
    help='detection size lists: ' + ' | '.join(map(str,DETECTION_SIZE_LISTS))
)
parser.add_argument(
    '-dt', '--detector',
    action='store_true',
    help='Use detector API (require ailia SDK 1.2.7).'
)
parser.add_argument(
    '-m', '--model',
    default="tiny",
    help='Select custom model file.'
)
args = update_parser(parser)

if args.detection_width != DETECTION_SIZE_LISTS[0] or args.detection_height!=DETECTION_SIZE_LISTS[0]:
    WEIGHT_PATH = 'yolov4-'+args.model+'_'+str(args.detection_width)+'_'+str(args.detection_height)+'.onnx'
    MODEL_PATH = 'yolov4-'+args.model+'_'+str(args.detection_width)+'_'+str(args.detection_height)+'.onnx.prototxt'
    IMAGE_HEIGHT = args.detection_height
    IMAGE_WIDTH = args.detection_width
else:
    WEIGHT_PATH = 'yolov4-'+args.model+'.onnx'
    MODEL_PATH = 'yolov4-'+args.model+'.onnx.prototxt'
    IMAGE_HEIGHT = args.detection_height
    IMAGE_WIDTH = args.detection_width

# ======================
# Main functions
# ======================
def recognize_from_image(detector):
    if args.profile:
        detector.set_profile_mode(True)

    # input image loop
    for image_path in args.input:
        # prepare input data
        logger.info(image_path)
        org_img = load_image(image_path)

        if not args.detector:
            org_img = cv2.cvtColor(org_img, cv2.COLOR_BGRA2BGR)
            logger.debug(f'input image shape: {org_img.shape}')

            img = letterbox_convert(org_img, (IMAGE_HEIGHT, IMAGE_WIDTH))

            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = np.transpose(img, [2, 0, 1])
            img = img.astype(np.float32) / 255
            img = np.expand_dims(img, 0)

        # inference
        logger.info('Start inference...')
        if args.benchmark:
            logger.info('BENCHMARK mode')
            total_time = 0
            for i in range(args.benchmark_count):
                start = int(round(time.time() * 1000))
                if args.detector:
                    detector.compute(org_img, args.threshold, args.iou)
                else:
                    output = detector.predict([img])
                end = int(round(time.time() * 1000))
                if i != 0:
                    total_time = total_time + (end - start)
                logger.info(f'\tailia processing time {end - start} ms')
            logger.info(f'\taverage time {total_time / (args.benchmark_count-1)} ms')
        else:
            if args.detector:
                detector.compute(org_img, args.threshold, args.iou)
            else:
                output = detector.predict([img])

        if not args.detector:
            detect_object = post_processing(img, args.threshold, args.iou, output)
            detect_object = reverse_letterbox(detect_object[0], org_img, (IMAGE_HEIGHT,IMAGE_WIDTH))
            res_img = plot_results(detect_object, org_img, COCO_CATEGORY)
        else:
            res_img = plot_results(detector, org_img, COCO_CATEGORY)

        # plot result
        savepath = get_savepath(args.savepath, image_path)
        logger.info(f'saved at : {savepath}')
        cv2.imwrite(savepath, res_img)

        # write prediction
        if args.write_prediction is not None:
            ext = args.write_prediction
            pred_file = "%s.%s" % (savepath.rsplit('.', 1)[0], ext)
            write_predictions(pred_file, detector if args.detector else detect_object, org_img, category=COCO_CATEGORY, file_type=ext)

    if args.profile:
        print(detector.get_summary())

    logger.info('Script finished successfully.')


def recognize_from_video(detector):
    capture = webcamera_utils.get_capture(args.video)

    # create video writer if savepath is specified as video format
    if args.savepath != SAVE_IMAGE_PATH:
        f_h = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        f_w = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        writer = webcamera_utils.get_writer(args.savepath, f_h, f_w)
    else:
        writer = None

    if args.write_prediction is not None:
        frame_count = 0
        frame_digit = int(math.log10(capture.get(cv2.CAP_PROP_FRAME_COUNT)) + 1)
        video_name = os.path.splitext(os.path.basename(args.video))[0]

    frame_shown = False
    while (True):
        ret, frame = capture.read()
        if (cv2.waitKey(1) & 0xFF == ord('q')) or not ret:
            break
        if frame_shown and cv2.getWindowProperty('frame', cv2.WND_PROP_VISIBLE) == 0:
            break

        if args.detector:
            img = cv2.cvtColor(frame, cv2.COLOR_BGR2BGRA)
            detector.compute(img, args.threshold, args.iou)
            res_img = plot_results(detector, frame, COCO_CATEGORY)
        else:
            img = letterbox_convert(frame, (IMAGE_HEIGHT, IMAGE_WIDTH))

            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = np.transpose(img, [2, 0, 1])
            img = img.astype(np.float32) / 255
            img = np.expand_dims(img, 0)

            output = detector.predict([img])
            detect_object = post_processing(
                img, args.threshold, args.iou, output
            )
            detect_object = reverse_letterbox(detect_object[0], frame, (IMAGE_HEIGHT,IMAGE_WIDTH))
            res_img = plot_results(detect_object, frame, COCO_CATEGORY)

        cv2.imshow('frame', res_img)
        frame_shown = True
        # save results
        if writer is not None:
            writer.write(res_img)

        # write prediction
        if args.write_prediction is not None:
            savepath = get_savepath(args.savepath, video_name, post_fix = '_%s' % (str(frame_count).zfill(frame_digit) + '_res'), ext='.png')
            ext = args.write_prediction
            pred_file = "%s.%s" % (savepath.rsplit('.', 1)[0], ext)
            write_predictions(pred_file, detector if args.detector else detect_object, frame, category=COCO_CATEGORY, file_type=ext)
            frame_count += 1

    capture.release()
    cv2.destroyAllWindows()
    if writer is not None:
        writer.release()
    logger.info('Script finished successfully.')


def main():
    # model files check and download
    check_and_download_models(WEIGHT_PATH, MODEL_PATH, REMOTE_PATH)

    # net initialize
    if args.detector:
        detector = ailia.Detector(
            MODEL_PATH,
            WEIGHT_PATH,
            len(COCO_CATEGORY),
            format=ailia.NETWORK_IMAGE_FORMAT_RGB,
            channel=ailia.NETWORK_IMAGE_CHANNEL_FIRST,
            range=ailia.NETWORK_IMAGE_RANGE_U_FP32,
            algorithm=ailia.DETECTOR_ALGORITHM_YOLOV4,
            env_id=args.env_id,
            memory_mode=args.memory_mode,
        )
    else:
        detector = ailia.Net(MODEL_PATH, WEIGHT_PATH, env_id=args.env_id, memory_mode=args.memory_mode)
        detector.set_input_shape((1, 3, IMAGE_HEIGHT, IMAGE_WIDTH))

    if args.video is not None:
        # video mode
        recognize_from_video(detector)
    else:
        # image mode
        recognize_from_image(detector)


if __name__ == '__main__':
    main()
