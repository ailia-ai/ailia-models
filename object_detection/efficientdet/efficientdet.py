import sys, os
import time
from logging import getLogger

import numpy as np
import cv2

import ailia

# import original modules
sys.path.append('../../util')
from arg_utils import get_base_parser, update_parser, get_savepath  # noqa: E402
from model_utils import check_and_download_models  # noqa: E402
from detector_utils import load_image, plot_results, write_predictions  # noqa: E402
from nms_utils import bb_intersection_over_union  # noqa: E402
from webcamera_utils import get_capture  # noqa: E402

from efficientdet_utils import *

logger = getLogger(__name__)

# ======================
# Parameters
# ======================

WEIGHT_EFFICIENTDET_D0_PATH = 'efficientdet-d0.onnx'
WEIGHT_EFFICIENTDET_D1_PATH = 'efficientdet-d1.onnx'
WEIGHT_EFFICIENTDET_D2_PATH = 'efficientdet-d2.onnx'
WEIGHT_EFFICIENTDET_D3_PATH = 'efficientdet-d3.onnx'
WEIGHT_EFFICIENTDET_D4_PATH = 'efficientdet-d4.onnx'
WEIGHT_EFFICIENTDET_D5_PATH = 'efficientdet-d5.onnx'
WEIGHT_EFFICIENTDET_D6_PATH = 'efficientdet-d6.onnx'
WEIGHT_EFFICIENTDET_D0HD_PATH = 'efficientdet-d0hd.onnx'
WEIGHT_EFFICIENTDET_D1HD_PATH = 'efficientdet-d1hd.onnx'
WEIGHT_EFFICIENTDET_D2HD_PATH = 'efficientdet-d2hd.onnx'
WEIGHT_EFFICIENTDET_D3HD_PATH = 'efficientdet-d3hd.onnx'
WEIGHT_EFFICIENTDET_D4HD_PATH = 'efficientdet-d4hd.onnx'
MODEL_EFFICIENTDET_D0_PATH = 'efficientdet-d0.onnx.prototxt'
MODEL_EFFICIENTDET_D1_PATH = 'efficientdet-d1.onnx.prototxt'
MODEL_EFFICIENTDET_D2_PATH = 'efficientdet-d2.onnx.prototxt'
MODEL_EFFICIENTDET_D3_PATH = 'efficientdet-d3.onnx.prototxt'
MODEL_EFFICIENTDET_D4_PATH = 'efficientdet-d4.onnx.prototxt'
MODEL_EFFICIENTDET_D5_PATH = 'efficientdet-d5.onnx.prototxt'
MODEL_EFFICIENTDET_D6_PATH = 'efficientdet-d6.onnx.prototxt'
MODEL_EFFICIENTDET_D0HD_PATH = 'efficientdet-d0hd.onnx.prototxt'
MODEL_EFFICIENTDET_D1HD_PATH = 'efficientdet-d1hd.onnx.prototxt'
MODEL_EFFICIENTDET_D2HD_PATH = 'efficientdet-d2hd.onnx.prototxt'
MODEL_EFFICIENTDET_D3HD_PATH = 'efficientdet-d3hd.onnx.prototxt'
MODEL_EFFICIENTDET_D4HD_PATH = 'efficientdet-d4hd.onnx.prototxt'
REMOTE_PATH = \
    'https://storage.googleapis.com/ailia-models/efficientdet/'

IMAGE_PATH = 'img.png'
SAVE_IMAGE_PATH = 'output.png'

obj_list = [
    'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light',
    'fire hydrant', '', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep',
    'cow', 'elephant', 'bear', 'zebra', 'giraffe', '', 'backpack', 'umbrella', '', '', 'handbag', 'tie',
    'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove',
    'skateboard', 'surfboard', 'tennis racket', 'bottle', '', 'wine glass', 'cup', 'fork', 'knife', 'spoon',
    'bowl', 'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut',
    'cake', 'chair', 'couch', 'potted plant', 'bed', '', 'dining table', '', '', 'toilet', '', 'tv',
    'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink',
    'refrigerator', '', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier',
    'toothbrush']

# ======================
# Arguemnt Parser Config
# ======================

parser = get_base_parser(
    'EfficientDet model', IMAGE_PATH, SAVE_IMAGE_PATH
)
parser.add_argument(
    '-m', '--model', type=str, default='d0',
    choices=(
        'd0', 'd1', 'd2', 'd3', 'd4', 'd5', 'd6',
        'd0hd', 'd1hd', 'd2hd', 'd3hd', 'd4hd',
    ),
    help='choice model'
)
parser.add_argument(
    '-t', '--threshold', type=float, default=0.2,
    help='threshold'
)
parser.add_argument(
    '-it', '--iou_threshold', type=float, default=0.2,
    help='iou_threshold'
)
parser.add_argument(
    '--onnx',
    action='store_true',
    help='execute onnxruntime version.'
)
parser.add_argument(
    '-w', '--write_prediction',
    nargs='?',
    const='txt',
    choices=['txt', 'json'],
    type=str,
    help='Output results to txt or json file.'
)
args = update_parser(parser)


# ======================
# Secondaty Functions
# ======================


def nms(boxes, scores, iou_threshold):
    # remove overwrapped detection
    det = []
    keep = []
    for idx in range(len(boxes)):
        obj = boxes[idx]
        is_keep = True
        for idx2 in range(len(det)):
            if not keep[idx2]:
                continue
            box_a = [det[idx2][0], det[idx2][1], det[idx2][2], det[idx2][3]]
            box_b = [obj[0], obj[1], obj[2], obj[3]]
            iou = bb_intersection_over_union(box_a, box_b)
            if iou >= iou_threshold:
                if scores[idx2] <= scores[idx]:
                    keep[idx2] = False
                else:
                    is_keep = False
        det.append(obj)
        keep.append(is_keep)

    ret = []
    for _, idx in sorted(zip(scores, range(len(boxes))), reverse=True):
        if keep[idx]:
            ret.append(idx)

    return ret


def preprocess(img, input_size=512):
    mean = (0.406, 0.456, 0.485)
    std = (0.225, 0.224, 0.229)

    img = (img / 255 - mean) / std

    img_meta = aspectaware_resize_padding(
        img[..., ::-1], input_size, input_size, means=None)
    img = img_meta[0]
    framed_metas = img_meta[1:]

    img = img.transpose(2, 0, 1)
    img = np.expand_dims(img, axis=0)

    return img, [framed_metas]


def postprocess(
        imgs,
        anchors, regression, classification,
        threshold, iou_threshold):
    transformed_anchors = bbox_transform(anchors, regression)
    transformed_anchors = clip_boxes(transformed_anchors, imgs)

    scores = np.max(classification, axis=2, keepdims=True)
    scores_over_thresh = (scores > threshold)[:, :, 0]

    out = []
    for i in range(imgs.shape[0]):
        if scores_over_thresh.sum() == 0:
            out.append({
                'rois': np.array(()),
                'class_ids': np.array(()),
                'scores': np.array(()),
            })

        classification_per = classification[i, scores_over_thresh[i, :], ...].transpose(1, 0)
        transformed_anchors_per = transformed_anchors[i, scores_over_thresh[i, :], ...]
        scores_per = scores[i, scores_over_thresh[i, :], ...]
        anchors_nms_idx = nms(transformed_anchors_per, scores_per[:, 0], iou_threshold=iou_threshold)

        if 0 < len(anchors_nms_idx):
            a = classification_per[:, anchors_nms_idx]
            scores_ = np.max(a, axis=0)
            classes_ = np.argmax(a, axis=0)
            boxes_ = transformed_anchors_per[anchors_nms_idx, :]
            out.append({
                'rois': boxes_,
                'class_ids': classes_,
                'scores': scores_,
            })
        else:
            out.append({
                'rois': np.array(()),
                'class_ids': np.array(()),
                'scores': np.array(()),
            })

    return out


def convert_to_ailia_detector_object(preds, w, h):
    i = 0
    detector_object = []
    for j in range(len(preds[i]['rois'])):
        (x1, y1, x2, y2) = preds[i]['rois'][j].astype(int)
        obj = preds[i]['class_ids'][j]
        score = float(preds[i]['scores'][j])

        r = ailia.DetectorObject(
            category=obj,
            prob=score,
            x=x1 / w,
            y=y1 / h,
            w=(x2 - x1) / w,
            h=(y2 - y1) / h,
        )

        detector_object.append(r)

    return detector_object


# ======================
# Main functions
# ======================


def predict(img, net):
    dic_input_size = {
        'd0': 512,
        'd1': 640,
        'd2': 768,
        'd3': 896,
        'd4': 1024,
        'd5': 1280,
        'd6': 1280,
        'd0hd': 1920,
        'd1hd': 1920,
        'd2hd': 1920,
        'd3hd': 1920,
        'd4hd': 1920,
    }
    input_size = dic_input_size[args.model]

    img, framed_metas = preprocess(img, input_size=input_size)

    if not args.onnx:
        net.set_input_shape(img.shape)
        output = net.predict({'imgs': img})
    else:
        output = net.run(
            ['regression', 'classification', 'anchors'],
            {'imgs': img})
    regression, classification, anchors = output

    threshold = args.threshold
    iou_threshold = args.iou_threshold
    out = postprocess(
        img,
        anchors, regression, classification,
        threshold, iou_threshold)

    out = invert_affine(framed_metas, out)

    return out


def recognize_from_image(image_path, net):
    if args.profile:
        net.set_profile_mode(True)

    # prepare input data
    img = load_image(image_path)
    logger.debug(f'input image shape: {img.shape}')

    img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)

    # inference
    logger.info('Start inference...')
    if args.benchmark:
        if not args.profile:
            net.set_profile_mode(True)
        logger.info('BENCHMARK mode')
        for i in range(5):
            start = int(round(time.time() * 1000))
            pred = predict(img, net)
            end = int(round(time.time() * 1000))
            logger.info(f'\tailia processing time {end - start} ms')
        if not args.profile:
            print(net.get_summary())
    else:
        pred = predict(img, net)

    # plot result
    detect_object = convert_to_ailia_detector_object(pred, img.shape[1], img.shape[0])
    img = plot_results(detect_object, img, obj_list)

    savepath = get_savepath(args.savepath, image_path)
    logger.info(f'saved at : {savepath}')
    cv2.imwrite(savepath, img)

    # write prediction
    if args.write_prediction is not None:
        ext = args.write_prediction
        pred_file = "%s.%s" % (savepath.rsplit('.', 1)[0], ext)
        write_predictions(pred_file, detect_object, img, category=obj_list, file_type=ext)

    if args.profile:
        print(net.get_summary())

    logger.info('Script finished successfully.')


def recognize_from_video(video, net):
    capture = get_capture(video)

    frame_shown = False
    while True:
        ret, frame = capture.read()
        if (cv2.waitKey(1) & 0xFF == ord('q')) or not ret:
            break
        if frame_shown and cv2.getWindowProperty('frame', cv2.WND_PROP_VISIBLE) == 0:
            break

        pred = predict(frame, net)

        # plot result
        detect_object = convert_to_ailia_detector_object(pred, frame.shape[1], frame.shape[0])
        img = plot_results(detect_object, frame, obj_list)

        cv2.imshow('frame', img)
        frame_shown = True

    capture.release()
    logger.info('Script finished successfully.')


def main():
    dic_model = {
        'd0': (WEIGHT_EFFICIENTDET_D0_PATH, MODEL_EFFICIENTDET_D0_PATH),
        'd1': (WEIGHT_EFFICIENTDET_D1_PATH, MODEL_EFFICIENTDET_D1_PATH),
        'd2': (WEIGHT_EFFICIENTDET_D2_PATH, MODEL_EFFICIENTDET_D2_PATH),
        'd3': (WEIGHT_EFFICIENTDET_D3_PATH, MODEL_EFFICIENTDET_D3_PATH),
        'd4': (WEIGHT_EFFICIENTDET_D4_PATH, MODEL_EFFICIENTDET_D4_PATH),
        'd5': (WEIGHT_EFFICIENTDET_D5_PATH, MODEL_EFFICIENTDET_D5_PATH),
        'd6': (WEIGHT_EFFICIENTDET_D6_PATH, MODEL_EFFICIENTDET_D6_PATH),
        'd0hd': (WEIGHT_EFFICIENTDET_D0HD_PATH, MODEL_EFFICIENTDET_D0HD_PATH),
        'd1hd': (WEIGHT_EFFICIENTDET_D1HD_PATH, MODEL_EFFICIENTDET_D1HD_PATH),
        'd2hd': (WEIGHT_EFFICIENTDET_D2HD_PATH, MODEL_EFFICIENTDET_D2HD_PATH),
        'd3hd': (WEIGHT_EFFICIENTDET_D3HD_PATH, MODEL_EFFICIENTDET_D3HD_PATH),
        'd4hd': (WEIGHT_EFFICIENTDET_D4HD_PATH, MODEL_EFFICIENTDET_D4HD_PATH),
    }
    weight_path, model_path = dic_model[args.model]

    # model files check and download
    check_and_download_models(weight_path, model_path, REMOTE_PATH)

    # initialize
    if not args.onnx:
        env_id = args.env_id
        net = ailia.Net(model_path, weight_path, env_id=env_id, memory_mode=args.memory_mode)
    else:
        import onnxruntime
        net = onnxruntime.InferenceSession(weight_path)

    if args.video is not None:
        # video mode
        recognize_from_video(args.video, net)
    else:
        # image mode
        for image_path in args.input:
            logger.info(image_path)
            recognize_from_image(image_path, net)


if __name__ == '__main__':
    main()
