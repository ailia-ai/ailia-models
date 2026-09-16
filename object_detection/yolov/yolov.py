import math
import os
import sys
import time

import ailia
import cv2
import numpy as np

from yolov_utils import * 

sys.path.append('../../util')
# logger
from logging import getLogger

import webcamera_utils
from nms_utils import batched_nms
from detector_utils import (load_image, plot_results, reverse_letterbox)
from image_utils import imread  # noqa: E402
from model_utils import check_and_download_models
from arg_utils import get_base_parser, get_savepath, update_parser

logger = getLogger(__name__)

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

# ======================
# Parameters
# ======================
MODEL_PARAMS = {'yolov_s': {'input_shape': [640, 640]},
                'yolov_l': {'input_shape': [640, 640]},
                'yolov_x': {'input_shape': [640, 640]}}

REMOTE_PATH = 'https://storage.googleapis.com/ailia-models/yolov/'

IMAGE_PATH = 'input.jpg'
SAVE_IMAGE_PATH = 'output.jpg'


SCORE_THR = 0.4
NMS_THR = 0.45

# ======================
# Arguemnt Parser Config
# ======================
parser = get_base_parser('yolov model', IMAGE_PATH, SAVE_IMAGE_PATH)
parser.add_argument(
    '-m', '--model_name',
    default='yolov_s',
    help='[yolov_x,  yolov_l, yolov_l]'
)
parser.add_argument(
    '-w', '--write_json',
    action='store_true',
    help='Flag to output results to json file.'
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

HEIGHT = MODEL_PARAMS[MODEL_NAME]['input_shape'][0]
WIDTH = MODEL_PARAMS[MODEL_NAME]['input_shape'][1]

VID_classes = (
    'airplane', 'antelope', 'bear', 'bicycle',
    'bird', 'bus', 'car', 'cattle',
    'dog', 'domestic_cat', 'elephant', 'fox',
    'giant_panda', 'hamster', 'horse', 'lion',
    'lizard', 'monkey', 'motorcycle', 'rabbit',
    'red_panda', 'sheep', 'snake', 'squirrel',
    'tiger', 'train', 'turtle', 'watercraft',
    'whale', 'zebra'
)

def postprocess(decode_res,num_classes,nms,topK,conf):
    pred_result, pred_idx = postpro_woclass(decode_res, num_classes, nms,topK)   
    outputs,outputs_ori=postprocess_single_img(pred_result,num_classes,nms,conf)
    return outputs,outputs_ori

def postpro_woclass(prediction, num_classes, nms_thre=0.75, topK=75, features=None):
    # find topK predictions, play the same role as RPN
    '''

    Args:
        prediction: [batch,feature_num,5+clsnum]
        num_classes:
        conf_thre:
        conf_thre_high:
        nms_thre:

    Returns:
        [batch,topK,5+clsnum]
    '''
    Prenum = 750
    box_corner = np.copy(prediction)
    box_corner[:, :, 0] = prediction[:, :, 0] - prediction[:, :, 2] / 2
    box_corner[:, :, 1] = prediction[:, :, 1] - prediction[:, :, 3] / 2
    box_corner[:, :, 2] = prediction[:, :, 0] + prediction[:, :, 2] / 2
    box_corner[:, :, 3] = prediction[:, :, 1] + prediction[:, :, 3] / 2
    prediction[:, :, :4] = box_corner[:, :, :4]

    output = [None for _ in range(len(prediction))]
    output_index = [None for _ in range(len(prediction))]
    features_list = []
    for i, image_pred in enumerate(prediction):
        if not image_pred.shape[0]:
            continue

        class_conf = np.max(image_pred[:, 5: 5 + num_classes], axis=1, keepdims=True)
        class_pred = np.argmax(image_pred[:, 5: 5 + num_classes], axis=1)
        class_pred = class_pred.reshape((class_pred.shape[0],1))

        detections = np.concatenate((image_pred[:, :5], class_conf, class_pred, image_pred[:, 5: 5 + num_classes]), 1)

        conf_score = image_pred[:, 4]
        sort_idx = (-conf_score).argsort()[:Prenum]

        detections_temp = detections[sort_idx, :]

        nms_out_index = batched_nms(
            detections_temp[:, :4],
            detections_temp[:, 4] * detections_temp[:, 5],
            detections_temp[:, 6],
            nms_thre,
        )
 
        topk_idx = sort_idx[nms_out_index[:topK]]
        output[i] = detections[topk_idx, :]
        output_index[i] = topk_idx

    return output, output_index

def postprocess_single_img(prediction, num_classes, nms_thre, conf_thre=0.1):

    output_ori = [None for _ in range(len(prediction))]
    prediction_ori = prediction
    for i, detections in enumerate(prediction):

        if not detections.shape[0]:
            continue

        detections_ori = prediction_ori[i]

        conf_mask = (detections_ori[:, 4] * detections_ori[:, 5] >= conf_thre).squeeze()
        detections_ori = detections_ori[conf_mask]

        nms_out_index = batched_nms(
            detections_ori[:, :4],
            detections_ori[:, 4] * detections_ori[:, 5],
            detections_ori[:, 6],
            nms_thre,
        )

        detections_ori = detections_ori[nms_out_index]
        output_ori[i] = detections_ori
    return output_ori, output_ori



class Predictor(object):
    def __init__(
        self,
        model,
        cls_names,
    ):
        self.cls_names = cls_names
        self.num_classes = 31
        self.nmsthre = args.iou
        self.conf = args.threshold
        self.test_size = (512,512)
        self.preproc = ValTransform(legacy=False)
        self.model = model

    def inference(self, img):

        num_classes = 30
        Afternum = 30

        results=self.model.run(np.expand_dims(img[0],0 ) )
        decode_res = results[0]
 
        outputs, outputs_ori = postprocess(decode_res,num_classes,self.nmsthre,topK=Afternum,conf=self.conf)
        return outputs

    def visual(self, output,img,ratio, cls_conf=0.0):

        if output is None:
            return img
        bboxes = output[:, 0:4]

        # preprocessing: resize
        bboxes /= ratio

        cls = output[:, 6]
        scores = output[:, 4] * output[:, 5]

        #vis_res,result = vis(img, bboxes, scores, cls, cls_conf, self.cls_names)
        result = vis(img, bboxes, scores, cls, cls_conf, self.cls_names)

        detect_object = reverse_letterbox(result, img, (img.shape[0], img.shape[1]))
        res_img = plot_results(detect_object, img, VID_classes)
        return res_img

# ======================
# Main functions
# ======================
def recognize_from_image(detector):
    # input image loop
    predictor = Predictor(detector,VID_classes) 
    for image_path in args.input:
        # prepare input data
        logger.debug(f'input image: {image_path}')
        raw_img = imread(image_path, cv2.IMREAD_COLOR)

        width, height = raw_img.shape[0:2]
        ratio = min(predictor.test_size[0] / height, predictor.test_size[1] / width)

        logger.debug(f'input image shape: {raw_img.shape}')

        def compute(frame):
            frame, _ = predictor.preproc(frame, None, (512,512))
            frame = np.expand_dims(frame,0)
            outputs = predictor.inference(frame)[0]
            return outputs

        # inference
        logger.info('Start inference...')
        if args.benchmark:
            logger.info('BENCHMARK mode')
            total_time = 0
            for i in range(args.benchmark_count):
                start = int(round(time.time() * 1000))
                output = compute(raw_img)
                end = int(round(time.time() * 1000))
                if i != 0:
                    total_time = total_time + (end - start)
                logger.info(f'\tailia processing time {end - start} ms')
            logger.info(f'\taverage time {total_time / (args.benchmark_count-1)} ms')
        else:
            output = compute(raw_img)

        res_img = predictor.visual(output,raw_img,ratio,cls_conf=args.threshold)

        # plot result
        savepath = get_savepath(args.savepath, image_path)
        logger.info(f'saved at : {savepath}')
        cv2.imwrite(savepath, res_img)

        # write prediction
        if args.write_json:
            json_file = '%s.json' % savepath.rsplit('.', 1)[0]
            save_result_json(json_file, output, ratio, conf=args.threshold, class_names=VID_classes)

    logger.info('Script finished successfully.')

def recognize_from_video(detector):
    capture = webcamera_utils.get_capture(args.video)

    predictor = Predictor(detector,VID_classes) 


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
        ratio = min(predictor.test_size[0] / height, predictor.test_size[1] / width)

        raw_img = frame
        frame, _ = predictor.preproc(frame, None, (512,512))
        frame = np.expand_dims(frame,0)
        outputs = predictor.inference(frame)[0]
        res_img = predictor.visual(outputs,raw_img,ratio,cls_conf=args.threshold)
 
        cv2.imshow('frame', res_img)
        frame_shown = True

        # save results
        if writer is not None:
            writer.write(res_img)

        # write prediction
        if args.write_json:
            savepath = get_savepath(args.savepath, video_name, post_fix = '_%s' % (str(frame_count).zfill(frame_digit) + '_res'), ext='.png')
            json_file = '%s.json' % savepath.rsplit('.', 1)[0]
            save_result_json(json_file, outputs, ratio, conf=args.threshold, class_names=VID_classes)
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
