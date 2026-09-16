import math
import os
import sys
import time

import ailia
import cv2
import numpy as np

from yolov6_utils import postprocess, predictions_to_object, check_img_size, letterbox

# import original modules
sys.path.append('../../util')
# logger
from logging import getLogger

import webcamera_utils
from detector_utils import (load_image, plot_results, reverse_letterbox,
                            write_predictions)
from image_utils import imread  # noqa: E402
from model_utils import check_and_download_models
from arg_utils import get_base_parser, get_savepath, update_parser

logger = getLogger(__name__)

# os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

# ======================
# Parameters
# ======================
MODEL_PARAMS = {'yolov6n': {'input_shape': [416, 416], 'max_stride': 32},
                'yolov6s': {'input_shape': [640, 640], 'max_stride': 32},
                'yolov6t': {'input_shape': [640, 640], 'max_stride': 32}
                }

REMOTE_PATH = 'https://storage.googleapis.com/ailia-models/yolov6/'

IMAGE_PATH = 'input.jpg'
SAVE_IMAGE_PATH = 'output.jpg'

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

SCORE_THR = 0.25#0.4
NMS_THR = 0.45

# ======================
# Arguemnt Parser Config
# ======================
parser = get_base_parser('yolov6 model', IMAGE_PATH, SAVE_IMAGE_PATH)
parser.add_argument(
    '-m', '--model_name',
    default='yolov6n',
    help='[yolov6n, yolov6s, yolov6t]'
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
    '-th', '--threshold',
    default=SCORE_THR, type=float,
    help='The detection threshold for yolo. (default: '+str(SCORE_THR)+')'
)
parser.add_argument(
    '-iou', '--iou',
    default=NMS_THR, type=float,
    help='The detection iou for yolo. (default: '+str(NMS_THR)+')'
)
parser.add_argument(
    '-dw', '--detection_width',
    default=-1, type=int,
    help='The detection width and height for yolo.'
)
parser.add_argument(
    '-dh', '--detection_height',
    default=-1, type=int,
    help='The detection height and height for yolo.'
)
parser.add_argument(
    '--classes', nargs='+', 
    type=int, help='filter by class: --class 0, or --class 0 2 3'
)
parser.add_argument(
    '--agnostic-nms', 
    action='store_true', help='class-agnostic NMS'
)
args = update_parser(parser)

MODEL_NAME = args.model_name
WEIGHT_PATH = MODEL_NAME + ".onnx"
MODEL_PATH = MODEL_NAME + ".onnx.prototxt"

IMAGE_HEIGHT = MODEL_PARAMS[MODEL_NAME]['input_shape'][0]
IMAGE_WIDTH = MODEL_PARAMS[MODEL_NAME]['input_shape'][1]
STRIDE = MODEL_PARAMS[MODEL_NAME]['max_stride']

# ======================
# Main functions
# ======================
def recognize_from_image(detector):
    # input image loop
    for image_path in args.input:
        # prepare input data
        logger.debug(f'input image: {image_path}')
        raw_img = cv2.imread(image_path)  # BGR
        # Padded resize
        img = letterbox(raw_img, (IMAGE_HEIGHT, IMAGE_WIDTH))[0]
        # Convert
        img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB
        img = np.ascontiguousarray(img)
        img = img/255.0  # 0 - 255 to 0.0 - 1.0
            
        logger.debug(f'input image shape: {raw_img.shape}')

        # inference
        logger.info('Start inference...')
        if args.benchmark:
            logger.info('BENCHMARK mode')
            total_time = 0
            for i in range(args.benchmark_count):
                start = int(round(time.time() * 1000))
                output = detector.run(img[None, :, :, :])
                end = int(round(time.time() * 1000))
                if i != 0:
                    total_time = total_time + (end - start)
                logger.info(f'\tailia processing time {end - start} ms')
            logger.info(f'\taverage time {total_time / (args.benchmark_count-1)} ms')
        else:
            output = detector.run(img[None, :, :, :])
        print(output[0].shape, output[1].shape, output[2].shape)
        predictions = postprocess(output)[0]
        detect_object = predictions_to_object(predictions, raw_img, args.iou, args.threshold, img.shape)
        res_img = plot_results(detect_object, raw_img, COCO_CATEGORY)

        # plot result
        savepath = get_savepath(args.savepath, image_path)
        logger.info(f'saved at : {savepath}')
        cv2.imwrite(savepath, res_img)

        # write prediction
        if args.write_prediction is not None:
            ext = args.write_prediction
            pred_file = "%s.%s" % (savepath.rsplit('.', 1)[0], ext)
            write_predictions(pred_file, detect_object, raw_img, category=COCO_CATEGORY, file_type=ext)

    logger.info('Script finished successfully.')

def recognize_from_video(detector):
    capture = webcamera_utils.get_capture(args.video)

    # create video writer if savepath is specified as video format
    if args.savepath != SAVE_IMAGE_PATH:
        f_h = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        f_w = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        save_h, save_w = f_h, f_w
        writer = webcamera_utils.get_writer(args.savepath, save_h, save_w)
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

        raw_img = frame
        # Padded resize
        img = letterbox(raw_img, (IMAGE_HEIGHT, IMAGE_WIDTH), stride=STRIDE)[0]
        # Convert
        img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB
        img = np.ascontiguousarray(img)
        img = img/255.0  # 0 - 255 to 0.0 - 1.0
        output = detector.run(img[None, :, :, :])
        predictions = postprocess(output)[0]
        detect_object = predictions_to_object(predictions, raw_img, args.iou, args.threshold, img.shape)
        res_img = plot_results(detect_object, raw_img, COCO_CATEGORY)
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
            write_predictions(pred_file, detect_object, frame, category=COCO_CATEGORY, file_type=ext)
            frame_count += 1

    capture.release()
    cv2.destroyAllWindows()
    if writer is not None:
        writer.release()
    logger.info('Script finished successfully.')

def main():
    # model files check and download
    check_and_download_models(WEIGHT_PATH, MODEL_PATH, REMOTE_PATH)

    env_id = args.env_id

    detector = ailia.Net(MODEL_PATH, WEIGHT_PATH, env_id=env_id, memory_mode=args.memory_mode)
    if args.detection_width!=-1 or args.detection_height!=-1:
        global IMAGE_WIDTH,IMAGE_HEIGHT
        imgsz = max(args.detection_width, args.detection_height)
        imgsz = check_img_size(imgsz, s=STRIDE)  # check img_size
        IMAGE_WIDTH=imgsz
        IMAGE_HEIGHT=imgsz
    detector.set_input_shape((1,3,IMAGE_WIDTH,IMAGE_HEIGHT))

    if args.video is not None:
        # video mode
        recognize_from_video(detector)
    else:
        # image mode
        recognize_from_image(detector)


if __name__ == '__main__':
    main()
