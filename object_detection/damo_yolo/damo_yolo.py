import math
import os
import sys
import time

import ailia
import cv2
import numpy as np


import ailia
from damo_yolo_util import *
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

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

# ======================
# Parameters
# ======================

REMOTE_PATH = 'https://storage.googleapis.com/ailia-models/damoyolo/'

IMAGE_PATH = 'input.jpg'
SAVE_IMAGE_PATH = 'output.jpg'


SCORE_THR = 0.6
NMS_THR = 0.7

# ======================
# Arguemnt Parser Config
# ======================
parser = get_base_parser('damo yolo model', IMAGE_PATH, SAVE_IMAGE_PATH)
parser.add_argument(
    '-m', '--model_name',
    default='damoyolo_S',
    help='[damoyolo_T, damoyolo_S, damoyolo_M]'
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

args = update_parser(parser)

MODEL_NAME = args.model_name
WEIGHT_PATH = MODEL_NAME + ".onnx"
MODEL_PATH = MODEL_NAME + ".onnx.prototxt"

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

class Infer():
    def __init__(self,model):


        self.class_names = []
        num_classes = 80
        for i in range(num_classes):
            self.class_names.append(str(i))
        self.class_names = tuple(self.class_names)

        self.infer_size = [640,640]
        self.model = model

    def _pad_image(self, img, target_size):
        n, c, h, w = img.shape
        assert n == 1
        assert h<=target_size[0] and w<=target_size[1]
        target_size = [n, c, target_size[0], target_size[1]]
        pad_imgs = np.zeros(target_size)
        pad_imgs[:, :c, :h, :w] = np.copy(img)

        img_sizes = [img.shape[-2:]]
        pad_sizes = [pad_imgs.shape[-2:]]

        return ImageList(pad_imgs, img_sizes, pad_sizes)


    def preprocess(self, origin_img):

        img = transform_img(origin_img, 0,
                image_max_range=[640,640], flip_prob= 0.0, image_mean = None,image_std=None,
                infer_size=self.infer_size)


        img = self._pad_image(img.tensors, self.infer_size)
        # img is a image_list
        ratio = min(origin_img.shape[0] / img.image_sizes[0][0],
            origin_img.shape[1] / img.image_sizes[0][1])

        return img, ratio

    def postprocess(self, preds, origin_image=None, ratio=1.0):

        scores = preds[0]
        bboxes = preds[1]

        num_classes = 80,

        output = postprocess(scores, bboxes,
            num_classes,
            args.threshold,
            args.iou,
            origin_image)

        bboxes = output[0].bbox * ratio
        scores = output[0].get_field('scores')
        cls_inds = output[0].get_field('labels')

        return bboxes,  scores, cls_inds


    def forward(self, image):
        image, ratio = self.preprocess(image)

        image_np = np.asarray(image.tensors)
        output = self.model.run(image_np)
        bboxes, scores, cls_inds = self.postprocess(output, image, ratio=ratio)
        return bboxes, scores, cls_inds

# ======================
# Main functions
# ======================
def recognize_from_image(detector):
    # input image loop
    
    infer_engine = Infer(detector)
    for image_path in args.input:
        # prepare input data
        logger.debug(f'input image: {image_path}')

        raw_img = imread(image_path, cv2.IMREAD_COLOR)
        raw_img = cv2.cvtColor(raw_img, cv2.COLOR_BGR2RGB)


        logger.debug(f'input image shape: {raw_img.shape}')

        # inference
        logger.info('Start inference...')
        if args.benchmark:
            logger.info('BENCHMARK mode')
            total_time = 0
            for i in range(args.benchmark_count):
                start = int(round(time.time() * 1000))

                bboxes, scores, cls_inds = infer_engine.forward(raw_img)

                end = int(round(time.time() * 1000))
                if i != 0:
                    total_time = total_time + (end - start)
                logger.info(f'\tailia processing time {end - start} ms')
            logger.info(f'\taverage time {total_time / (args.benchmark_count-1)} ms')
        else:
            bboxes, scores, cls_inds = infer_engine.forward(raw_img)

        detections = []
        h,w ,_= raw_img.shape

        bboxes = xyxy2xywh(bboxes)
        for box, score , ind in zip(bboxes,scores,cls_inds):
            r = ailia.DetectorObject(
                category=int(ind-1),
                prob=score,
                x=box[0]/w,
                y=box[1]/h,
                w=box[2]/w,
                h=box[3]/h,
            )
            detections.append(r)

        raw_img = cv2.cvtColor(raw_img, cv2.COLOR_RGB2BGR)
        res_img = plot_results(detections, raw_img, COCO_CATEGORY)

        # plot result
        savepath = get_savepath(args.savepath, image_path)
        logger.info(f'saved at : {savepath}')
        cv2.imwrite(savepath, res_img)

        # write prediction
        if args.write_prediction:
            ext = args.write_prediction
            pred_file = "%s.%s" % (savepath.rsplit('.', 1)[0], ext)
            write_predictions(pred_file, detections, raw_img, category=COCO_CATEGORY, file_type=ext)

    logger.info('Script finished successfully.')

def recognize_from_video(detector):
    capture = webcamera_utils.get_capture(args.video)

    infer_engine = Infer(detector)

    # create video writer if savepath is specified as video format
    if args.savepath != SAVE_IMAGE_PATH:
        f_h = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        f_w = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        save_h, save_w = f_h, f_w
        writer = webcamera_utils.get_writer(args.savepath, save_h, save_w)
    else:
        writer = None

    if args.write_prediction:
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
        width, height = frame.shape[0:2]

        raw_img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        bboxes, scores, cls_inds = infer_engine.forward(raw_img)

        h,w ,_= raw_img.shape
        detections = []
        bboxes = xyxy2xywh(bboxes)

        for box, score , ind in zip(bboxes,scores,cls_inds):
            r = ailia.DetectorObject(
                category=ind-1,
                prob=score,
                x=box[0]/w,
                y=box[1]/h,
                w=box[2]/w,
                h=box[3]/h,
            )
            detections.append(r)

        raw_img = cv2.cvtColor(raw_img, cv2.COLOR_RGB2BGR)
        res_img = plot_results(detections, raw_img, COCO_CATEGORY)
 
        cv2.imshow('frame', res_img)
        frame_shown = True

        # save results
        if writer is not None:
            writer.write(res_img)

        # write prediction
        if args.write_prediction:
            savepath = get_savepath(args.savepath, video_name, post_fix = '_%s' % (str(frame_count).zfill(frame_digit) + '_res'), ext='.png')
            pred_file = '%s.txt' % savepath.rsplit('.', 1)[0]
            write_predictions(pred_file, detections, frame, COCO_CATEGORY)
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

    if args.video is not None:
        # video mode
        recognize_from_video(detector)
    else:
        # image mode
        recognize_from_image(detector)


if __name__ == '__main__':
    main()
