import sys
import time

import numpy as np
import cv2
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from matplotlib.collections import PatchCollection

import ailia

# import original modules
sys.path.append('../../util')
from arg_utils import get_base_parser, update_parser, get_savepath  # noqa
from model_utils import check_and_download_models  # noqa
from image_utils import normalize_image  # noqa
from detector_utils import load_image, plot_results, write_predictions  # noqa
from webcamera_utils import get_capture, get_writer  # noqa
# logger
from logging import getLogger  # noqa

from picodet_utils import grid_priors, get_bboxes

logger = getLogger(__name__)

# ======================
# Parameters
# ======================

WEIGHT_S_320_PATH = 'picodet_s_320_coco.onnx'
MODEL_S_320_PATH = 'picodet_s_320_coco.onnx.prototxt'
WEIGHT_S_416_PATH = 'picodet_s_416_coco.onnx'
MODEL_S_416_PATH = 'picodet_s_416_coco.onnx.prototxt'
WEIGHT_M_416_PATH = 'picodet_m_416_coco.onnx'
MODEL_M_416_PATH = 'picodet_m_416_coco.onnx.prototxt'
WEIGHT_L_640_PATH = 'picodet_l_640_coco.onnx'
MODEL_L_640_PATH = 'picodet_l_640_coco.onnx.prototxt'
REMOTE_PATH = 'https://storage.googleapis.com/ailia-models/picodet/'

COCO_CATEGORY = (
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
)

IMAGE_PATH = 'demo.jpg'
SAVE_IMAGE_PATH = 'output.png'

THRESHOLD = 0.3
IOU = 0.6

# ======================
# Arguemnt Parser Config
# ======================

parser = get_base_parser(
    'PP-PicoDet', IMAGE_PATH, SAVE_IMAGE_PATH
)
parser.add_argument(
    '-th', '--threshold',
    default=THRESHOLD, type=float,
    help='object confidence threshold'
)
parser.add_argument(
    '-iou', '--iou',
    default=IOU, type=float,
    help='IOU threshold for NMS'
)
parser.add_argument(
    '-w', '--write_prediction',
    action='store_true',
    help='Flag to output the prediction file.'
)
parser.add_argument(
    '-m', '--model_type', default='s-416',
    choices=('s-320', 's-416', 'm-416', 'l-640'),
    help='model type'
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
# Main functions
# ======================

def preprocess(img, image_shape):
    h, w = image_shape
    im_h, im_w, _ = img.shape

    img = img[:, :, ::-1]  # BGR -> RGB
    img = cv2.resize(img, (w, h), interpolation=cv2.INTER_LINEAR)

    w_scale = w / im_w
    h_scale = h / im_h
    scale_factor = np.array([
        w_scale, h_scale, w_scale, h_scale],
        dtype=np.float32)

    img = normalize_image(img, normalize_type='ImageNet')

    divisor = 32
    pad_h = int(np.ceil(h / divisor)) * divisor
    pad_w = int(np.ceil(w / divisor)) * divisor

    padding = (0, 0, max(pad_w - w, 0), max(pad_h - h, 0))

    img = cv2.copyMakeBorder(
        img,
        padding[1], padding[3],
        padding[0], padding[2],
        cv2.BORDER_CONSTANT,
        value=0)

    img = img.transpose(2, 0, 1)  # HWC -> CHW
    img = np.expand_dims(img, axis=0)
    img = img.astype(np.float32)

    return img, scale_factor


def post_processing(img, output, out_shape, scale_factor):
    im_h, im_w = img.shape[:2]
    cls_scores = output[:4]
    bbox_preds = output[4:]

    num_levels = len(cls_scores)

    featmap_sizes = [cls_scores[i].shape[-2:] for i in range(num_levels)]
    mlvl_priors = grid_priors(num_levels, featmap_sizes)

    cls_score_list = [
        cls_scores[i][0] for i in range(num_levels)
    ]
    bbox_pred_list = [
        bbox_preds[i][0] for i in range(num_levels)
    ]

    num_classes = len(COCO_CATEGORY)
    nms_thre = args.iou

    det_bboxes, det_labels = get_bboxes(
        cls_score_list, bbox_pred_list, mlvl_priors,
        out_shape, num_classes,
        scale_factor=scale_factor, with_nms=True, nms_thre=nms_thre, score_thr=args.threshold
    )

    detections = []
    for bbox, label in zip(det_bboxes, det_labels):
        x1, y1, x2, y2, prob = bbox
        if prob < args.threshold:
            break
        r = ailia.DetectorObject(
            category=label,
            prob=bbox[4],
            x=x1 / im_w,
            y=y1 / im_h,
            w=(x2 - x1) / im_w,
            h=(y2 - y1) / im_h,
        )
        detections.append(r)

    return detections


def predict(net, img):
    dic_shape = {
        's-320': (320, 320),
        's-416': (416, 416),
        'm-416': (416, 416),
        'l-640': (640, 640),
    }
    shape = dic_shape[args.model_type]

    pp_img, scale_factor = preprocess(img, shape)

    # feedforward
    if args.benchmark:
        logger.info('BENCHMARK mode')
        total_time_estimation = 0
        for i in range(args.benchmark_count):
            start = int(round(time.time() * 1000))
            output = net.predict([pp_img])
            end = int(round(time.time() * 1000))
            estimation_time = (end - start)

            # Loggin
            logger.info(f'\tailia processing estimation time {estimation_time} ms')
            if i != 0:
                total_time_estimation = total_time_estimation + estimation_time

        logger.info(f'\taverage time estimation {total_time_estimation / (args.benchmark_count - 1)} ms')
    else:
        output = net.predict([pp_img])

    detect_object = post_processing(img, output, shape, scale_factor)

    return detect_object


def recognize_from_image(net):
    # input image loop
    for image_path in args.input:
        logger.info(image_path)

        # prepare input data
        img = load_image(image_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)

        # inference
        logger.info('Start inference...')
        detect_object = predict(net, img)
        res_img = plot_results(detect_object, img, COCO_CATEGORY)

        # plot result
        savepath = get_savepath(args.savepath, image_path, ext='.png')
        logger.info(f'saved at : {savepath}')
        cv2.imwrite(savepath, res_img)

        # write prediction
        if args.write_prediction is not None:
            ext = args.write_prediction
            pred_file = "%s.%s" % (savepath.rsplit('.', 1)[0], ext)
            write_predictions(pred_file, detect_object, img, category=COCO_CATEGORY, file_type=ext)

    logger.info('Script finished successfully.')


def recognize_from_video(net):
    video_file = args.video if args.video else args.input[0]
    capture = get_capture(video_file)
    assert capture.isOpened(), 'Cannot capture source'

    # create video writer if savepath is specified as video format
    if args.savepath != SAVE_IMAGE_PATH:
        f_h = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        f_w = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        writer = get_writer(args.savepath, f_h, f_w)
    else:
        writer = None

    score_thr = args.threshold

    frame_shown = False
    while True:
        ret, frame = capture.read()
        if (cv2.waitKey(1) & 0xFF == ord('q')) or not ret:
            break
        if frame_shown and cv2.getWindowProperty('frame', cv2.WND_PROP_VISIBLE) == 0:
            break

        # inference
        img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        detect_object = predict(net, img)

        # plot result
        res_img = plot_results(detect_object, frame, COCO_CATEGORY)

        # show
        cv2.imshow('frame', res_img)
        frame_shown = True

        # save results
        if writer is not None:
            res_img = res_img.astype(np.uint8)
            writer.write(res_img)

    capture.release()
    cv2.destroyAllWindows()
    if writer is not None:
        writer.release()

    logger.info('Script finished successfully.')


def main():
    dic_model = {
        's-320': (WEIGHT_S_320_PATH, MODEL_S_320_PATH),
        's-416': (WEIGHT_S_416_PATH, MODEL_S_416_PATH),
        'm-416': (WEIGHT_M_416_PATH, MODEL_M_416_PATH),
        'l-640': (WEIGHT_L_640_PATH, MODEL_L_640_PATH),
    }
    WEIGHT_PATH, MODEL_PATH = dic_model[args.model_type]

    # model files check and download
    check_and_download_models(WEIGHT_PATH, MODEL_PATH, REMOTE_PATH)

    env_id = args.env_id

    # initialize
    net = ailia.Net(MODEL_PATH, WEIGHT_PATH, env_id=env_id, memory_mode=args.memory_mode)

    if args.video is not None:
        recognize_from_video(net)
    else:
        recognize_from_image(net)


if __name__ == '__main__':
    main()
