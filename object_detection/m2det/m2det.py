import time
import sys

import numpy as np
import cv2

import ailia

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

WEIGHT_PATH = './m2det.onnx'
MODEL_PATH = './m2det.onnx.prototxt'
REMOTE_PATH = 'https://storage.googleapis.com/ailia-models/m2det/'

IMAGE_PATH = 'input.jpg'
SAVE_IMAGE_PATH = 'output.png'
IMAGE_HEIGHT = 448  # for video mode
IMAGE_WIDTH = 448  # for video mode

COCO_CATEGORY = ['__background__'] + [
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
THRESHOLD = 0.4
IOU = 0.45
KEEP_PER_CLASS = 10


# ======================
# Arguemnt Parser Config
# ======================
parser = get_base_parser('m2det model', IMAGE_PATH, SAVE_IMAGE_PATH)
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
def nms(dets, thresh):
    """Pure Python NMS baseline."""
    x1 = dets[:, 0]
    y1 = dets[:, 1]
    x2 = dets[:, 2]
    y2 = dets[:, 3]
    scores = dets[:, 4]

    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    order = scores.argsort()[::-1]

    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h
        ovr = inter / (areas[i] + areas[order[1:]] - inter)

        inds = np.where(ovr <= thresh)[0]
        order = order[inds + 1]
    return keep


def preprocess(img, resize=512, rgb_means=(104, 117, 123), swap=(2, 0, 1)):
    interp_method = cv2.INTER_LINEAR
    img = cv2.resize(
        np.array(img), (resize, resize), interpolation=interp_method
    ).astype(np.float32)
    img -= rgb_means
    # make channel first
    img = img.transpose(swap)
    return img[None, ...]


def to_color(indx, base):
    """ return (b, r, g) tuple"""
    base2 = base * base
    b = 2 - indx / base2
    r = 2 - (indx % base2) / base
    g = 2 - (indx % base2) % base
    return b * 127, r * 127, g * 127


BASE = int(np.ceil(pow(len(COCO_CATEGORY), 1. / 3)))
COLORS = [to_color(x, BASE) for x in range(len(COCO_CATEGORY))]


def convert_to_ailia_detector_object(bboxes, scores, cls_inds, w, h):
    detector_object = []
    for i, box in enumerate(bboxes):
        cls_indx = int(cls_inds[i])

        r = ailia.DetectorObject(
            category=cls_indx,
            prob=scores[i],
            x=box[0] / w,
            y=box[1] / h,
            w=(box[2] - box[0]) / w,
            h=(box[3] - box[1]) / h,
        )

        detector_object.append(r)

    return detector_object

# ======================
# Main functions
# ======================
def detect_objects(img, detector):
    # get sizes for posterior rescaling
    h, w, _ = img.shape
    scale = np.asarray([w, h, w, h])

    # initial preprocesses
    img = preprocess(img)

    # feedforward
    boxes, scores = detector.predict({'input.1': img})

    boxes = boxes[0]
    scores = scores[0]
    allboxes = []

    # filter boxes for every class
    for j in range(1, len(COCO_CATEGORY)):
        inds = np.where(scores[:, j] > THRESHOLD)[0]
        if len(inds) == 0:
            continue
        c_bboxes = boxes[inds]
        c_scores = scores[inds, j]
        c_dets = np.hstack(
            (c_bboxes, c_scores[:, np.newaxis])
        ).astype(np.float32, copy=False)
        # rank ordered iou
        # min_thresh, device_id=0 if cfg.test_cfg.cuda else None)
        keep = nms(c_dets, IOU)
        keep = keep[:KEEP_PER_CLASS]
        c_dets = c_dets[keep, :]
        allboxes.extend([_.tolist()+[j] for _ in c_dets])

    if len(allboxes) > 0:
        allboxes = np.array(allboxes)
        # split boxes and scores
        boxes = allboxes[:, :4] * scale
        scores = allboxes[:, 4]
        cls_inds = allboxes[:, 5]
        return boxes, scores, cls_inds
    else:
        return [], [], []


def recognize_from_image(filename, detector):
    if args.profile:
        detector.set_profile_mode(True)

    # load input image
    img = load_image(filename)
    img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)

    logger.info('Start inference...')
    if args.benchmark:
        logger.info('BENCHMARK mode')
        for i in range(5):
            start = int(round(time.time() * 1000))
            boxes, scores, cls_inds = detect_objects(img, detector)
            end = int(round(time.time() * 1000))
            logger.info(f'\tailia processing time {end - start} ms')
    else:
        boxes, scores, cls_inds = detect_objects(img, detector)

    try:
        logger.info('\n'.join(
            ['pos:{}, ids:{}, score:{:.3f}'.format(
                '(%.1f,%.1f,%.1f,%.1f)' % (box[0], box[1], box[2], box[3]),
                COCO_CATEGORY[int(obj_cls)], score
            ) for box, obj_cls, score in zip(boxes, cls_inds, scores)]
        ))
    except:
        # FIXME: do not use bare 'except'
        pass

    # show image
    detect_object = convert_to_ailia_detector_object(boxes, scores, cls_inds, img.shape[1], img.shape[0])
    img = plot_results(detect_object, img, COCO_CATEGORY)

    savepath = get_savepath(args.savepath, filename)
    logger.info(f'saved at : {savepath}')
    cv2.imwrite(savepath, img)

    # write prediction
    if args.write_prediction is not None:
        ext = args.write_prediction
        pred_file = "%s.%s" % (savepath.rsplit('.', 1)[0], ext)
        write_predictions(pred_file, detect_object, img, category=COCO_CATEGORY, file_type=ext)

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

    frame_shown = False
    while(True):
        ret, img = capture.read()
        # press q to end video capture
        if (cv2.waitKey(1) & 0xFF == ord('q')) or not ret:
            break
        if frame_shown and cv2.getWindowProperty('frame', cv2.WND_PROP_VISIBLE) == 0:
            break

        boxes, scores, cls_inds = detect_objects(img, detector)

        # show image
        detect_object = convert_to_ailia_detector_object(boxes, scores, cls_inds, img.shape[1], img.shape[0])
        img = plot_results(detect_object, img, COCO_CATEGORY)
        cv2.imshow('frame', img)
        frame_shown = True

        # save results
        if writer is not None:
            writer.write(img)

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
