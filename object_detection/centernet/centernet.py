import os
import time
import sys
import math

from collections import namedtuple

import numpy as np
import cv2

import ailia
from centernet_utils import preprocess, postprocess

# import original modules
sys.path.append('../../util')
from arg_utils import get_base_parser, update_parser, get_savepath  # noqa: E402
from model_utils import check_and_download_models  # noqa: E402
from detector_utils import plot_results, load_image, write_predictions  # noqa: E402
import webcamera_utils  # noqa: E402

# logger
from logging import getLogger   # noqa: E402
logger = getLogger(__name__)


# ======================
# Parameters
# ======================

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
THRESHOLD = 0.3  # Threshold for filteing for filtering (from 0.0 to 1.0)
K_VALUE = 40  # K value for topK function
OPSET_LISTS = ['10', '11']


# ======================
# Arguemnt Parser Config
# ======================
parser = get_base_parser('CenterNet model', IMAGE_PATH, SAVE_IMAGE_PATH)
parser.add_argument(
    '-w', '--write_prediction',
    nargs='?',
    const='txt',
    choices=['txt', 'json'],
    type=str,
    help='Output results to txt or json file.'
)
parser.add_argument(
    '-o', '--opset', metavar='OPSET',
    default='10', choices=OPSET_LISTS,
    help='opset lists: ' + ' | '.join(OPSET_LISTS)
)
args = update_parser(parser)


if args.opset == "10":
    WEIGHT_PATH = './ctdet_coco_dlav0_1x.onnx'
    MODEL_PATH = './ctdet_coco_dlav0_1x.onnx.prototxt'
else:
    WEIGHT_PATH = './ctdet_coco_dlav0_1x_opset11.onnx'
    MODEL_PATH = './ctdet_coco_dlav0_1x_opset11.onnx.prototxt'
REMOTE_PATH = 'https://storage.googleapis.com/ailia-models/centernet/'


# ======================
# Secondaty Functions
# ======================
def to_color(indx, base):
    """ return (b, r, g) tuple"""
    base2 = base * base
    b = 2 - indx / base2
    r = 2 - (indx % base2) / base
    g = 2 - (indx % base2) % base
    return b * 127, r * 127, g * 127


BASE = int(np.ceil(pow(len(COCO_CATEGORY), 1. / 3)))
COLORS = [to_color(x, BASE) for x in range(len(COCO_CATEGORY))]


# ======================
# Main functions
# ======================
def detect_objects(org_img, net):
    centernet_image_size = (512, 512)
    img = preprocess(org_img, centernet_image_size)
    net.predict(img)
    res = net.get_results()
    dets = postprocess(
        [output[0] for output in res],
        (org_img.shape[1], org_img.shape[0]),
        K_VALUE,
        THRESHOLD
    )

    boxes = []
    scores = []
    cls_inds = []

    # font_scale = 0.5
    # font = cv2.FONT_HERSHEY_SIMPLEX

    for det in dets:
        # Make sure bboxes are not out of bounds
        xmin, ymin, xmax, ymax = det[:4].astype(int)
        xmin = max(0, xmin)
        ymin = max(0, ymin)
        xmax = min(org_img.shape[1], xmax)
        ymax = min(org_img.shape[0], ymax)

        boxes.append([xmin, ymin, xmax, ymax])
        scores.append(det[4])
        cls_inds.append(det[5])

    return boxes, scores, cls_inds


def recognize_from_image(filename, detector):
    if args.profile:
        detector.set_profile_mode(True)

    # load input image
    img = load_image(filename)
    img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)

    logger.info('Start inference...')
    if args.benchmark:
        for mode in range(2):
            if mode == 0:
                logger.info('BENCHMARK mode (without post process)')
            else:
                logger.info('BENCHMARK mode (with post process)')
            zeros = np.zeros((1,3,512,512))
            total_time = 0
            for i in range(args.benchmark_count):
                start = int(round(time.time() * 1000))
                if mode == 0:
                    detector.predict(zeros)
                else:
                    boxes, scores, cls_inds = detect_objects(img, detector)
                end = int(round(time.time() * 1000))
                if i != 0:
                    total_time = total_time + (end - start)
                logger.info(f'\tailia processing time {end - start} ms')
            logger.info(f'\taverage time {total_time / (args.benchmark_count-1)} ms')
    else:
        boxes, scores, cls_inds = detect_objects(img, detector)

    try:
        logger.info('\n'.join(
            ['pos:{}, ids:{}, score:{:.3f}'.format(
                '(%.1f,%.1f,%.1f,%.1f)' % (box[0], box[1], box[2], box[3]),
                COCO_CATEGORY[int(obj_cls)],
                score
            ) for box, obj_cls, score in zip(boxes, cls_inds, scores)]
        ))
    except:
        # FIXME: do not use base 'except'
        pass

    Detection = namedtuple('Detection', ['category', 'prob', 'x', 'y', 'w', 'h'])
    ary = []
    h, w = (img.shape[0], img.shape[1])
    for i, box in enumerate(boxes):
        d = Detection(int(cls_inds[i]), scores[i], box[0]/w, box[1]/h, (box[2]-box[0])/w, (box[3]-box[1])/h)
        ary.append(d)

    im2show = plot_results(ary, img, COCO_CATEGORY)

    savepath = get_savepath(args.savepath, filename)
    logger.info(f'saved at : {savepath}')
    cv2.imwrite(savepath, im2show)

    # write prediction
    if args.write_prediction is not None:
        ext = args.write_prediction
        pred_file = "%s.%s" % (savepath.rsplit('.', 1)[0], ext)
        write_predictions(pred_file, ary, img, category=COCO_CATEGORY, file_type=ext)

    if args.profile:
        print(detector.get_summary())

    logger.info('Script finished successfully.')



def recognize_from_video(video, detector):
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
    while(True):
        ret, img = capture.read()

        # press q to end video capture
        if (cv2.waitKey(1) & 0xFF == ord('q')) or not ret:
            break
        if frame_shown and cv2.getWindowProperty('frame', cv2.WND_PROP_VISIBLE) == 0:
            break

        boxes, scores, cls_inds = detect_objects(img, detector)

        Detection = namedtuple('Detection', ['category', 'prob', 'x', 'y', 'w', 'h'])
        ary = []
        h, w = (img.shape[0], img.shape[1])
        for i, box in enumerate(boxes):
            d = Detection(int(cls_inds[i]), scores[i], box[0]/w, box[1]/h, (box[2]-box[0])/w, (box[3]-box[1])/h)
            ary.append(d)

        img = plot_results(ary, img, COCO_CATEGORY)
        cv2.imshow('frame', img)
        frame_shown = True

        # save results
        if writer is not None:
            writer.write(img)

        # write prediction
        if args.write_prediction is not None:
            savepath = get_savepath(args.savepath, video_name, post_fix = '_%s' % (str(frame_count).zfill(frame_digit) + '_res'), ext='.png')
            ext = args.write_prediction
            pred_file = "%s.%s" % (savepath.rsplit('.', 1)[0], ext)
            write_predictions(pred_file, ary, img, category=COCO_CATEGORY, file_type=ext)
            frame_count += 1

    capture.release()
    cv2.destroyAllWindows()
    if writer is not None:
        writer.release()


def main():
    # model files check and download
    check_and_download_models(WEIGHT_PATH, MODEL_PATH, REMOTE_PATH)

    # load model
    detector = ailia.Net(MODEL_PATH, WEIGHT_PATH, env_id=args.env_id, memory_mode=args.memory_mode)

    if args.video is not None:
        # video mode
        recognize_from_video(args.video, detector)
    else:
        # image mode
        # input image loop
        for image_path in args.input:
            # prepare input data
            logger.info(image_path)
            recognize_from_image(image_path, detector)

    logger.info('Script finished successfully.')


if __name__ == '__main__':
    main()
