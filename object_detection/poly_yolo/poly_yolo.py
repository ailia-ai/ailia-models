import os
import sys
import cv2
import math
import time
import numpy as np

import ailia

# import original modules
sys.path.append('../../util')
# logger
from logging import getLogger
from arg_utils import get_base_parser, get_savepath, update_parser

import webcamera_utils
from image_utils import imread  # noqa: E402
from model_utils import check_and_download_models

import poly_yolo_util as yolo #or "import poly_yolo_lite as yolo" for the lite version

logger = getLogger(__name__)

# ======================
# Parameters
# ======================
REMOTE_PATH = 'https://storage.googleapis.com/ailia-models/poly_yolo/'

IMAGE_PATH = 'input.jpg'
SAVE_IMAGE_PATH = 'output.jpg'

SCORE_THR = 0.4
NMS_THR = 0.3

# ======================
# Arguemnt Parser Config
# ======================
parser = get_base_parser('poly yolo model', IMAGE_PATH, SAVE_IMAGE_PATH)

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

args = update_parser(parser)

MODEL_NAME = "poly_yolo"
WEIGHT_PATH = MODEL_NAME + ".onnx"
MODEL_PATH = MODEL_NAME + ".onnx.prototxt"

def translate_color(cls):
    if cls == 0: return (230, 25, 75)
    if cls == 1: return (60, 180, 75)
    if cls == 2: return (255, 225, 25)
    if cls == 3: return (0, 130, 200)
    if cls == 4: return (245, 130, 48)
    if cls == 5: return (145, 30, 180)
    if cls == 7: return (70, 240, 240)
    if cls == 8: return (240, 50, 230)
    if cls == 9: return (210, 245, 60)
    if cls == 10: return (250, 190, 190)
    if cls == 11: return (0, 128, 128)
    if cls == 12: return (230, 190, 255)
    if cls == 13: return (170, 110, 40)
    if cls == 14: return (255, 250, 200)
    if cls == 15: return (128, 0, 128)
    if cls == 16: return (170, 255, 195)
    if cls == 17: return (128, 128, 0)
    if cls == 18: return (255, 215, 180)
    if cls == 19: return (80, 80, 128)


def compute(model,img):
    overlay = img.copy()
    boxes   = []
    classes = []
    
    box, scores, classes, polygons = model.detect_image(img)
    
    for k in range (0, len(box)):
        boxes.append((box[k][1], box[k][0], box[k][3], box[k][2]))
        #cv2.rectangle(img, (box[k][1],box[k][0]), (box[k][3],box[k][2]), translate_color(classes[k]), 3, 1)
    
    #browse all boxes
    for b in range(0, len(boxes)):
        f              = translate_color(classes[b])    
        points_to_draw = []
        offset         = len(polygons[b])//3
        
        #filter bounding polygon vertices
        for dst in range(0, len(polygons[b])//3):
            if polygons[b][dst+offset*2] > 0.3: 
                points_to_draw.append([int(polygons[b][dst]), int(polygons[b][dst+offset])])
        
        points_to_draw = np.asarray(points_to_draw)
        points_to_draw = points_to_draw.astype(np.int32)
        if points_to_draw.shape[0]>0:
            cv2.polylines(img, [points_to_draw],True,f, thickness=2)
            cv2.fillPoly(overlay, [points_to_draw], f)
    img = cv2.addWeighted(overlay, 0.4, img, 1 - 0.4, 0)
    return img


# ======================
# Main functions
# ======================
def recognize_from_image(model):
    # input image loop
    for image_path in args.input:
        # prepare input data
        logger.debug(f'input image: {image_path}')
        raw_img = imread(image_path, cv2.IMREAD_COLOR)
        logger.debug(f'input image shape: {raw_img.shape}')

        # inference
        logger.info('Start inference...')
        if args.benchmark:
            logger.info('BENCHMARK mode')
            total_time = 0
            for i in range(args.benchmark_count):
                start = int(round(time.time() * 1000))
                output = compute(model,raw_img)
                end = int(round(time.time() * 1000))
                if i != 0:
                    total_time = total_time + (end - start)
                logger.info(f'\tailia processing time {end - start} ms')
            logger.info(f'\taverage time {total_time / (args.benchmark_count-1)} ms')
        else:
            output = compute(model,raw_img)

        # plot result
        savepath = get_savepath(args.savepath, image_path)
        logger.info(f'saved at : {savepath}')
        cv2.imwrite(savepath, output)

    logger.info('Script finished successfully.')

def recognize_from_video(model):
    capture = webcamera_utils.get_capture(args.video)

    # create video writer if savepath is specified as video format
    if args.savepath != SAVE_IMAGE_PATH:
        f_h = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        f_w = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        save_h, save_w = f_h, f_w
        writer = webcamera_utils.get_writer(args.savepath, save_h, save_w)
    else:
        writer = None

    while (True):
        ret, frame = capture.read()
        if (cv2.waitKey(1) & 0xFF == ord('q')) or not ret:
            break

        res_img = compute(model,frame)

        cv2.imshow('frame', res_img)

        # save results
        if writer is not None:
            writer.write(res_img)

    capture.release()
    cv2.destroyAllWindows()
    if writer is not None:
        writer.release()
    logger.info('Script finished successfully.')

def main():
    # model files check and download
    check_and_download_models(WEIGHT_PATH, MODEL_PATH, REMOTE_PATH)

    model = yolo.YOLO(model=ailia.Net(None,WEIGHT_PATH,memory_mode=args.memory_mode),iou=args.iou, score=args.threshold)
    if args.video is not None:
        # video mode
        recognize_from_video(model)
    else:
        # image mode
        recognize_from_image(model)


if __name__ == '__main__':
    main()

