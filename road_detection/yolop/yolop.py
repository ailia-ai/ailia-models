import argparse
import os
import sys
import time

import ailia
import cv2
import numpy as np

from yolop_utils import (letterbox_for_img, non_max_suppression)

sys.path.append('../../util')
import webcamera_utils  # noqa: E402
from model_utils import check_and_download_models  # noqa: E402
from PIL import Image
from arg_utils import get_base_parser, get_savepath, update_parser
from image_utils import imread  # noqa: E402

# ======================
# Parameters
# ======================

WEIGHT_PATH = 'yolop.onnx'
MODEL_PATH  = 'yolop.onnx.prototxt'
REMOTE_PATH = 'https://storage.googleapis.com/ailia-models/yolop/'

# logger
from logging import getLogger

logger = getLogger(__name__)

IMAGE_PATH = 'input.jpg'
SAVE_IMAGE_PATH = 'output.jpg'

# ======================
# Arguemnt Parser Config
# ======================

parser = get_base_parser('yolop model', IMAGE_PATH, SAVE_IMAGE_PATH)

parser.add_argument(
    '-m', '--model_name',
    default='yolop.onnx', type=str,
    help='model path'
)
parser.add_argument('--img-size',
    default=640, type=int, 
    help='inference size (pixels)'
)
parser.add_argument('--conf-thres',
    default=0.25, type=float, 
    help='object confidence threshold'
)
parser.add_argument('--iou-thres',
    default=0.45, type=float,
    help='IOU threshold for NMS'
)
args = update_parser(parser)
 
# ======================
# Main functions
# ======================

def resize_unscale(img, new_shape=(640, 640), color=114):
    shape = img.shape[:2]  # current shape [height, width]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    canvas = np.zeros((new_shape[0], new_shape[1], 3))
    canvas.fill(color)
    # Scale ratio (new / old) new_shape(h,w)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])

    # Compute padding
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))  # w,h
    new_unpad_w = new_unpad[0]
    new_unpad_h = new_unpad[1]
    pad_w, pad_h = new_shape[1] - new_unpad_w, new_shape[0] - new_unpad_h  # wh padding

    dw = pad_w // 2  # divide padding into 2 sides
    dh = pad_h // 2

    if shape[::-1] != new_unpad:  # resize
        img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_AREA)

    canvas[dh:dh + new_unpad_h, dw:dw + new_unpad_w, :] = img

    return canvas, r, dw, dh, new_unpad_w, new_unpad_h  # (dw,dh)


def recognize_from_video():
    capture = webcamera_utils.get_capture(args.video)
    if args.savepath != SAVE_IMAGE_PATH:
        logger.warning(
            'currently, video results cannot be output correctly...'
        )
        f_h = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        f_w = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        writer = webcamera_utils.get_writer(args.savepath, f_h, f_w)
    else:
        writer = None

    weight = args.model_name
    net = ailia.Net(None,weight,env_id=args.env_id)
    frame_shown = False
    while(True):
        ret, frame = capture.read()
        if (cv2.waitKey(1) & 0xFF == ord('q')) or not ret:
            break
        if frame_shown and cv2.getWindowProperty('frame', cv2.WND_PROP_VISIBLE) == 0:
            break

        img_det = detect(net, frame)
        cv2.imshow("frame", img_det)
        frame_shown = True

        if writer:
            writer.write(img_det)

    capture.release()
    cv2.destroyAllWindows()
    if writer is not None:
        writer.release()
    logger.info('Script finished successfully.')

def detect(net, frame):
    s = np.stack([letterbox_for_img(x, args.img_size)[0].shape for x in frame], 0)  # shapes
    rect = np.unique(s, axis=0).shape[0] == 1  # rect inference if all shapes equal
    img0 = frame.copy()
    h0, w0 = img0[0].shape[:2]
    img, _, pad = letterbox_for_img(img0[0], args.img_size, auto=rect)

    # Stack
    h, w = frame.shape[:2]
    shapes = (h0, w0), ((h / h0, w / w0), pad)

    # Convert
    img = np.ascontiguousarray(frame)

    img_det = img0
   
    img_bgr = img
    height, width, _ = img_bgr.shape

    # convert to RGB
    img_rgb = img_bgr[:, :, ::-1].copy()

    # resize & normalize
    canvas, r, dw, dh, new_unpad_w, new_unpad_h = resize_unscale(img_rgb, (640, 640))

    img = canvas.copy().astype(np.float32)  # (3,640,640) RGB
    img /= 255.0
    img[:, :, 0] -= 0.485
    img[:, :, 1] -= 0.456
    img[:, :, 2] -= 0.406
    img[:, :, 0] /= 0.229
    img[:, :, 1] /= 0.224
    img[:, :, 2] /= 0.225

    img = img.transpose(2, 0, 1)

    img = np.expand_dims(img, 0)  # (1, 3,640,640)


    # inference
    logger.info('Start inference...')
    if args.benchmark:
        logger.info('BENCHMARK mode')
        total_time = 0
        for i in range(args.benchmark_count):
            start = int(round(time.time() * 1000))
            det_out, da_seg_out, ll_seg_out = net.run(img)
            end = int(round(time.time() * 1000))
            if i != 0:
                total_time = total_time + (end - start)
            logger.info(f'\tailia processing time {end - start} ms')
        logger.info(f'\taverage time {total_time / (args.benchmark_count-1)} ms')
    else:
        det_out, da_seg_out, ll_seg_out = net.run(img)

    boxes = non_max_suppression(det_out, conf_thres=args.conf_thres, iou_thres=args.iou_thres, agnostic=False)[0]


    if boxes.shape[0] == 0:
        #print("no bounding boxes detected.")
        return img_bgr

    # scale coords to original size.
    boxes[:, 0] -= dw
    boxes[:, 1] -= dh
    boxes[:, 2] -= dw
    boxes[:, 3] -= dh
    boxes[:, :4] /= r

    #print(f"detect {boxes.shape[0]} bounding boxes.")

    img_det = img_rgb[:, :, ::-1].copy()
    for i in range(boxes.shape[0]):
        x1, y1, x2, y2, conf, label = boxes[i]
        x1, y1, x2, y2, label = int(x1), int(y1), int(x2), int(y2), int(label)
        img_det = cv2.rectangle(img_det, (x1, y1), (x2, y2), (0, 255, 0), 2, 2)

    # select da & ll segment area.
    da_seg_out = da_seg_out[:, :, dh:dh + new_unpad_h, dw:dw + new_unpad_w]
    ll_seg_out = ll_seg_out[:, :, dh:dh + new_unpad_h, dw:dw + new_unpad_w]

    da_seg_mask = np.argmax(da_seg_out, axis=1)[0]  # (?,?) (0|1)
    ll_seg_mask = np.argmax(ll_seg_out, axis=1)[0]  # (?,?) (0|1)

    color_area = np.zeros((new_unpad_h, new_unpad_w, 3), dtype=np.uint8)
    color_area[da_seg_mask == 1] = [0, 255, 0]
    color_area[ll_seg_mask == 1] = [255, 0, 0]
    color_seg = color_area

    # convert to BGR
    color_seg = color_seg[..., ::-1]
    color_mask = np.mean(color_seg, 2)
    img_merge = canvas[dh:dh + new_unpad_h, dw:dw + new_unpad_w, :]
    img_merge = img_merge[:, :, ::-1]

    # merge: resize to original size
    img_merge[color_mask != 0] = \
        img_merge[color_mask != 0] * 0.5 + color_seg[color_mask != 0] * 0.5
    img_merge = img_merge.astype(np.uint8)
    img_merge = cv2.resize(img_merge, (width, height),
                           interpolation=cv2.INTER_LINEAR)
    for i in range(boxes.shape[0]):
        x1, y1, x2, y2, conf, label = boxes[i]
        x1, y1, x2, y2, label = int(x1), int(y1), int(x2), int(y2), int(label)
        img_merge = cv2.rectangle(img_merge, (x1, y1), (x2, y2), (0, 255, 0), 2, 2)

    # da: resize to original size
    da_seg_mask = da_seg_mask * 255
    da_seg_mask = da_seg_mask.astype(np.uint8)
    da_seg_mask = cv2.resize(da_seg_mask, (width, height),
                             interpolation=cv2.INTER_LINEAR)

    # ll: resize to original size
    ll_seg_mask = ll_seg_mask * 255
    ll_seg_mask = ll_seg_mask.astype(np.uint8)
    ll_seg_mask = cv2.resize(ll_seg_mask, (width, height),
                             interpolation=cv2.INTER_LINEAR)

    img_det = img_merge
    return img_det


def recognize_from_image():
    weight = args.model_name
    net = ailia.Net(None,weight,env_id=args.env_id)

    for image_path in args.input:
        img = imread(image_path, cv2.IMREAD_COLOR)
        img_det = detect(net, img)
        save_path = get_savepath(args.savepath, image_path)
        logger.info(f'saved at : {save_path}')
        cv2.imwrite(save_path, img_det)

    logger.info('Script finished successfully.')

if __name__ == '__main__':

    check_and_download_models(WEIGHT_PATH, MODEL_PATH, REMOTE_PATH)
    if args.video is not None:
        # video mode
        recognize_from_video()
    else:
        # image mode
        recognize_from_image()


