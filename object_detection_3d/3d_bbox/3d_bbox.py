#ailia predict api sample

import argparse
import os
import sys
import time

import ailia
import cv2
import numpy as np
from matplotlib import pyplot as plt
from PIL import Image
import json

from lib_3d_bbox import ClassAverages
from lib_3d_bbox.Dataset import *
from lib_3d_bbox.Math import *
from lib_3d_bbox.Plotting import *

# import original modules
sys.path.append('../../util')
# logger
from logging import getLogger  # noqa: E402

import webcamera_utils  # noqa: E402
from detector_utils import (load_image, plot_results,  # noqa: E402
                            write_predictions)
from image_utils import imread  # noqa: E402
from model_utils import check_and_download_models  # noqa: E402
from arg_utils import get_base_parser, get_savepath, update_parser  # noqa: E402

logger = getLogger(__name__)

os.environ['KMP_DUPLICATE_LIB_OK']='TRUE'

REMOTE_PATH_YOLOV3 = 'https://storage.googleapis.com/ailia-models/yolov3/'
REMOTE_PATH_3D_BBOX = 'https://storage.googleapis.com/ailia-models/3d_bbox/'

# settings
WEIGHT_PATH_YOLOV3 = "yolov3.opt.onnx"
MODEL_PATH_YOLOV3 = "yolov3.opt.onnx.prototxt"

WEIGHT_PATH_3D_BBOX = "3d_bbox.onnx"
MODEL_PATH_3D_BBOX = "3d_bbox.onnx.prototxt"

IMAGE_PATH = 'input.png'
SAVE_IMAGE_PATH = 'output.png'

CALIB_FILE = "./lib_3d_bbox/calib_cam_to_cam.txt"

COCO_CATEGORY = [
    "Pedestrian", "bicycle", "Car", "motorcycle", "airplane", "bus", "train",
    "Truck", "boat", "traffic light", "fire hydrant", "stop sign",
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

# Default input size
HEIGHT = 224
WIDTH = 224


def plot_regressed_3d_bbox(img, cam_to_img, box_2d, dimensions, alpha, theta_ray, img_2d=None):

    # the math! returns X, the corners used for constraint
    location, X = calc_location(dimensions, cam_to_img, box_2d, alpha, theta_ray)

    orient = alpha + theta_ray

    if img_2d is not None:
        plot_2d_box(img_2d, box_2d)

    plot_3d_box(img, cam_to_img, orient, dimensions, location) # 3d boxes

    return location

# ======================
# Arguemnt Parser Config
# ======================
parser = get_base_parser('3d_bbox model', IMAGE_PATH, SAVE_IMAGE_PATH)

parser.add_argument(
    '--cui',
    action='store_true',
    help="Don't display preview in GUI."
)
parser.add_argument(
    '-w', '--write_json',
    action='store_true',
    help='Flag to output results to json file.'
)
args = update_parser(parser)

# ======================
# Main functions
# ======================
def recognize_from_image():
    averages = ClassAverages()
    angle_bins = generate_bins(2)

    env_id = args.env_id
    net_yolov3 = ailia.Detector(
            MODEL_PATH_YOLOV3,
            WEIGHT_PATH_YOLOV3,
            len(COCO_CATEGORY),
            format=ailia.NETWORK_IMAGE_FORMAT_RGB,
            channel=ailia.NETWORK_IMAGE_CHANNEL_FIRST,
            range=ailia.NETWORK_IMAGE_RANGE_U_FP32,
            algorithm=ailia.DETECTOR_ALGORITHM_YOLOV3,
            env_id=env_id,
            memory_mode=args.memory_mode,
        )
    net_3d_bbox = ailia.Net(MODEL_PATH_3D_BBOX,WEIGHT_PATH_3D_BBOX,env_id=env_id,memory_mode=args.memory_mode)
    net_yolov3.set_input_shape(WIDTH, HEIGHT)

    # input image loop
    for image_path in args.input:
        # prepare input data
        logger.debug(f'input image: {image_path}')
        img = imread(image_path)
        logger.debug(f'input image shape: {img.shape}')
        yolo_img = cv2.cvtColor(img, cv2.COLOR_BGR2BGRA)
        logger.debug(f'yolo input image shape: {yolo_img.shape}')

        # inference
        logger.info('Start inference...')
        if args.benchmark:
            logger.info('BENCHMARK mode')
            for i in range(5):
                start = int(round(time.time() * 1000))
                detections = net_yolov3.run(yolo_img, THRESHOLD, IOU)
                end = int(round(time.time() * 1000))
                logger.info(f'\tailia processing time {end - start} ms')
        else:
            detections = net_yolov3.run(yolo_img, THRESHOLD, IOU)

        saved_results = []

        for detection in detections:
            detect_class = COCO_CATEGORY[detection[0]]
            xmin = int(detection[2][0] * img.shape[1])
            ymin = int(detection[2][1] * img.shape[0])
            xmax = int(detection[2][2] * img.shape[1]) + int(detection[2][0] * img.shape[1])
            ymax = int(detection[2][3] * img.shape[0]) + int(detection[2][1] * img.shape[0])
            detect_box_2d = [(xmin, ymin),(xmax, ymax)]
            if not averages.recognized_class(detect_class):
                continue
            try:
                detectedObject = DetectedObject(img, detect_class, detect_box_2d, CALIB_FILE)
            except:
                continue

            theta_ray = detectedObject.theta_ray
            input_img = detectedObject.img
            proj_matrix = detectedObject.proj_matrix
            box_2d = detect_box_2d
            detected_class = detect_class

            input_tensor = np.expand_dims(input_img,0)
            [orient, conf, dim] = net_3d_bbox.run(input_tensor)
            orient = orient[0, :, :]
            conf = conf[0, :]
            dim = dim[0, :]

            saved_results.append({
                'orient': orient.tolist(), 'conf': conf.tolist(), 'dim': dim.tolist()
            })

            dim += averages.get_item(detected_class)

            argmax = np.argmax(conf)
            orient = orient[argmax, :]
            cos = orient[0]
            sin = orient[1]
            alpha = np.arctan2(sin, cos)
            alpha += angle_bins[argmax]
            alpha -= np.pi

            plot_regressed_3d_bbox(img, proj_matrix, box_2d, dim, alpha, theta_ray)
            if not args.cui:
                cv2.imshow('3D detections', img)

        savepath = get_savepath(args.savepath, image_path)
        logger.info(f'saved at : {savepath}')
        cv2.imwrite(savepath, img)

        if args.write_json:
            json_file = '%s.json' % savepath.rsplit('.', 1)[0]
            with open(json_file, 'w') as f:
                json.dump(saved_results, f, indent=2)

    if not args.cui and cv2.waitKey(0) != 32:  # space bar
        exit()

def recognize_from_video():
    # net initialize
    averages = ClassAverages()
    angle_bins = generate_bins(2)

    env_id = args.env_id
    net_yolov3 = ailia.Detector(
            MODEL_PATH_YOLOV3,
            WEIGHT_PATH_YOLOV3,
            len(COCO_CATEGORY),
            format=ailia.NETWORK_IMAGE_FORMAT_RGB,
            channel=ailia.NETWORK_IMAGE_CHANNEL_FIRST,
            range=ailia.NETWORK_IMAGE_RANGE_U_FP32,
            algorithm=ailia.DETECTOR_ALGORITHM_YOLOV3,
            env_id=env_id,
            memory_mode=args.memory_mode,
        )
    net_3d_bbox = ailia.Net(MODEL_PATH_3D_BBOX,WEIGHT_PATH_3D_BBOX,env_id=env_id,memory_mode=args.memory_mode)
    net_yolov3.set_input_shape(WIDTH, HEIGHT)

    capture = webcamera_utils.get_capture(args.video)

    # create video writer if savepath is specified as video format
    if args.savepath != SAVE_IMAGE_PATH:
        logger.warning(
            'currently, video results cannot be output correctly...'
        )
        f_h = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        f_w = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        save_h, save_w = f_h, f_w
        writer = webcamera_utils.get_writer(args.savepath, save_h, save_w)
    else:
        writer = None

    frame_shown = False
    while(True):
        ret, frame = capture.read()
        if (cv2.waitKey(1) & 0xFF == ord('q')) or not ret:
            break
        if frame_shown and cv2.getWindowProperty('3D detections', cv2.WND_PROP_VISIBLE) == 0:
            break

        img = frame
        yolo_img = cv2.cvtColor(img, cv2.COLOR_BGR2BGRA)

        detections = net_yolov3.run(yolo_img, THRESHOLD, IOU)
        for detection in detections:
            detect_class = COCO_CATEGORY[detection[0]]
            xmin = int(detection[2][0] * img.shape[1])
            ymin = int(detection[2][1] * img.shape[0])
            xmax = int(detection[2][2] * img.shape[1]) + int(detection[2][0] * img.shape[1])
            ymax = int(detection[2][3] * img.shape[0]) + int(detection[2][1] * img.shape[0])
            detect_box_2d = [(xmin, ymin), (xmax, ymax)]
            if not averages.recognized_class(detect_class):
                continue
            try:
                detectedObject = DetectedObject(img, detect_class, detect_box_2d, CALIB_FILE)
            except:
                continue

            theta_ray = detectedObject.theta_ray
            input_img = detectedObject.img
            proj_matrix = detectedObject.proj_matrix
            box_2d = detect_box_2d
            detected_class = detect_class

            input_tensor = np.expand_dims(input_img, 0)
            [orient, conf, dim] = net_3d_bbox.run(input_tensor)
            orient = orient[0, :, :]
            conf = conf[0, :]
            dim = dim[0, :]
            dim += averages.get_item(detected_class)

            argmax = np.argmax(conf)
            orient = orient[argmax, :]
            cos = orient[0]
            sin = orient[1]
            alpha = np.arctan2(sin, cos)
            alpha += angle_bins[argmax]
            alpha -= np.pi

            location = plot_regressed_3d_bbox(img, proj_matrix, box_2d, dim, alpha, theta_ray)

            if not args.cui:
                cv2.imshow('3D detections', img)
            frame_shown = True

        # save results
        if writer is not None:
            writer.write(img)

    capture.release()
    cv2.destroyAllWindows()
    if writer is not None:
        writer.release()
    logger.info('Script finished successfully.')

def main():
    # model files check and download
    check_and_download_models(WEIGHT_PATH_3D_BBOX, MODEL_PATH_3D_BBOX, REMOTE_PATH_3D_BBOX)
    check_and_download_models(WEIGHT_PATH_YOLOV3, MODEL_PATH_YOLOV3, REMOTE_PATH_YOLOV3)

    if args.video is not None:
        # video mode
        recognize_from_video()
    else:
        # image mode
        recognize_from_image()


if __name__ == '__main__':
    main()
