import sys
import time

import numpy as np
import cv2

import ailia

# import original modules
sys.path.append('../../util')
from arg_utils import get_base_parser, update_parser  # noqa: E402
from image_utils import normalize_image  # noqa: E402
from model_utils import check_and_download_models  # noqa: E402
from detector_utils import load_image  # noqa: E402
import webcamera_utils  # noqa: E402

# import deepsort modules
sys.path.append('../deepsort')
from sort.tracker import Tracker
from sort.nn_matching import NearestNeighborDistanceMetric
from deepsort_utils import *

# logger
from logging import getLogger  # noqa: E402

logger = getLogger(__name__)

# ======================
# Parameters
# ======================

DT_WEIGHT_PATH = 'yolov3.opt.onnx'
DT_MODEL_PATH = 'yolov3.opt.onnx.prototxt'
EX_WEIGHT_PATH = 'deep_sort_vehicle.onnx'
EX_MODEL_PATH = 'deep_sort_vehicle.onnx.prototxt'
REMOTE_EX_PATH = 'https://storage.googleapis.com/ailia-models/deep_sort_vehicle/'
REMOTE_DT_PATH = 'https://storage.googleapis.com/ailia-models/deep_sort/'

# Deep sort model input
INPUT_HEIGHT = 128
INPUT_WIDTH = 64

# yolo params
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
THRESHOLD = 0.1  # 0.4
MIN_CONFIDENCE = 0.3
IOU = 0.45
DETECT_CLASSES = [2, 5, 7]
# DETECT_CLASSES = [2]

# Metric parameters
MAX_DISTANCE = 0.016  # threshold of matching object
NN_BUDGET = 100

# ======================
# Arguemnt Parser Config
# ======================

parser = get_base_parser('Deep SORT Vehicle', None, None, fp16_support=False)
parser.add_argument(
    '-p', '--pairimage', metavar='IMAGE',
    nargs=2,
    default=[None, None],
    help='If this option is specified, the model is set to determine '
         'if the vehicle in two images is the same vehicle or not.'
)
parser.add_argument(
    '-nd', '--no_detector', action='store_true',
    help='Do not use detector in pairimage mode.'
)
args = update_parser(parser)


# ======================
# Utils
# ======================

def resize(img, size=(INPUT_WIDTH, INPUT_HEIGHT)):
    return cv2.resize(img.astype(np.float32), size)


def euclidean_distance(x1, x2):
    x1 = x1.reshape(1, -1)
    x2 = x2.reshape(1, -1)
    a2, b2 = np.square(x1).sum(axis=1), np.square(x2).sum(axis=1)
    r2 = -2. * np.dot(x1, x2.T) + a2[:, None] + b2[None, :]
    r2 = r2[0][0]
    r2 = r2 ** 0.5
    return r2


# ======================
# Main functions
# ======================

def recognize_from_video(detector, extractor):
    results = []
    idx_frame = 0

    # tracker class instance
    metric = NearestNeighborDistanceMetric(
        "euclidean", MAX_DISTANCE, NN_BUDGET
    )
    tracker = Tracker(
        metric,
        max_iou_distance=0.7,
        max_age=70,
        n_init=3
    )

    capture = webcamera_utils.get_capture(args.video)

    # create video writer
    if args.savepath is not None:
        writer = webcamera_utils.get_writer(
            args.savepath,
            int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)),
        )
    else:
        writer = None

    logger.info('Start Inference...')
    while (True):
        idx_frame += 1
        ret, frame = capture.read()
        if (cv2.waitKey(1) & 0xFF == ord('q')) or not ret:
            break

        # In order to use ailia.Detector, the input should have 4 channels.
        input_img = cv2.cvtColor(frame, cv2.COLOR_BGR2BGRA)
        h, w = frame.shape[0], frame.shape[1]

        # do detection
        detector.compute(input_img, THRESHOLD, IOU)
        bbox_xywh, cls_conf, cls_ids = get_detector_result(detector, h, w)

        # select person class
        mask = np.isin(cls_ids, DETECT_CLASSES)
        bbox_xywh = bbox_xywh[mask]

        # bbox dilation just in case bbox too small,
        # delete this line if using a better pedestrian detector
        bbox_xywh[:, 3:] *= 1.2
        cls_conf = cls_conf[mask]

        # do tracking
        img_crops = []
        for box in bbox_xywh:
            x1, y1, x2, y2 = xywh_to_xyxy(box, h, w)
            img = frame[y1:y2, x1:x2]
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img_crops.append(img)

        if img_crops:
            # preprocess
            img_batch = np.concatenate([
                normalize_image(
                    resize(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)), '255'
                )[np.newaxis, :, :, :]
                for img in img_crops
            ], axis=0).transpose(0, 3, 1, 2)

            # TODO better to pass a batch at once
            # features = extractor.predict(img_batch)
            features = []
            for img in img_batch:
                features.append(extractor.predict(img[np.newaxis, :, :, :])[0])
            features = np.array(features)
        else:
            features = np.array([])

        bbox_tlwh = xywh_to_tlwh(bbox_xywh)
        detections = [
            Detection(bbox_tlwh[i], conf, features[i])
            for i, conf in enumerate(cls_conf) if conf > MIN_CONFIDENCE
        ]

        # run on non-maximum supression
        boxes = np.array([d.tlwh for d in detections])
        scores = np.array([d.confidence for d in detections])
        nms_max_overlap = 1.0
        indices = non_max_suppression(boxes, nms_max_overlap, scores)
        detections = [detections[i] for i in indices]

        # update tracker
        tracker.predict()
        tracker.update(detections)

        # update bbox identities
        outputs = []
        for track in tracker.tracks:
            if not track.is_confirmed() or track.time_since_update > 1:
                continue
            box = track.to_tlwh()
            x1, y1, x2, y2 = tlwh_to_xyxy(box, h, w)
            track_id = track.track_id
            outputs.append(np.array([x1, y1, x2, y2, track_id], dtype=int))
        if len(outputs) > 0:
            outputs = np.stack(outputs, axis=0)

        # draw box for visualization
        if len(outputs) > 0:
            bbox_tlwh = []
            bbox_xyxy = outputs[:, :4]
            identities = outputs[:, -1]
            frame = draw_boxes(frame, bbox_xyxy, identities)

            for bb_xyxy in bbox_xyxy:
                bbox_tlwh.append(xyxy_to_tlwh(bb_xyxy))

            results.append((idx_frame - 1, bbox_tlwh, identities))

        cv2.imshow('frame', frame)

        if writer is not None:
            writer.write(frame)

        if args.savepath is not None:
            write_results(args.savepath.split('.')[0] + '.txt', results, 'mot')
        else:
            write_results('result.txt', results, 'mot')

    capture.release()
    cv2.destroyAllWindows()
    if writer is not None:
        writer.release()

    logger.info(f'Save results to {args.savepath}')
    logger.info('Script finished successfully.')


def compare_images(detector, extractor):
    """
    This is a mode to determine if two input images have the same person
    by using the CNN model, which is used in DeepSORT to track the same person.
    It is assumed that there is always only one person in each image.
    We have not verified, and do not assume, the behavor in the case of
    multiple people. (Future work)
    """
    no_detector = args.no_detector

    # prepare input data
    input_data = []
    for i in range(len(args.pairimage)):
        input_data.append(load_image(args.pairimage[i]))

    # inference
    logger.info('Start inference...')
    features = []
    for i in range(len(input_data)):
        if not no_detector:
            # do detection
            detector.compute(input_data[i], THRESHOLD, IOU)
            h, w = input_data[i].shape[0], input_data[i].shape[1]
            bbox_xywh, cls_conf, cls_ids = get_detector_result(detector, h, w)

            # select person class
            mask = np.isin(cls_ids, DETECT_CLASSES)
            if mask.sum() == 0:
                logger.info('Detector could not detect any vehicle '
                            f'in the input image: {args.pairimage[i]}')
                logger.info('Program finished.')
                sys.exit(0)

            bbox_xywh = bbox_xywh[mask]

            # bbox dilation just in case bbox too small,
            # delete this line if using a better pedestrian detector
            bbox_xywh[:, 3:] *= 1.2
            cls_conf = cls_conf[mask]

            # image crop
            """
            [INFO] If more than one bounding box is detected,
            the one with the highest confidence is used as correct box.
            It should be noted that this works because we assume that
            the input image has only one person.
            """
            x1, y1, x2, y2 = xywh_to_xyxy(bbox_xywh[np.argmax(cls_conf)], h, w)
            src_img = cv2.cvtColor(input_data[i], cv2.COLOR_BGRA2RGB)
            img = src_img[y1:y2, x1:x2]
        else:
            img = cv2.cvtColor(input_data[i], cv2.COLOR_BGRA2RGB)

        # preprocess
        img = normalize_image(
            resize(img), '255'
        )[np.newaxis, :, :, :].transpose(0, 3, 1, 2)

        if args.benchmark:
            logger.info('BENCHMARK mode')
            for i in range(5):
                start = int(round(time.time() * 1000))
                feature = extractor.predict(img)
                end = int(round(time.time() * 1000))
                logger.info(f'\tailia processing time {end - start} ms')
        else:
            feature = extractor.predict(img)

        features.append(feature[0])

    dist = euclidean_distance(features[0], features[1])
    if dist <= MAX_DISTANCE:
        logger.info(f'{args.pairimage}: Same vehicle (distance: {dist})')
    else:
        logger.info(f'{args.pairimage}: Different vehicle (distance: {dist})')


def main():
    # model files check and download
    if not args.no_detector:
        logger.info('Check Detector...')
        check_and_download_models(DT_WEIGHT_PATH, DT_MODEL_PATH, REMOTE_DT_PATH)
    logger.info('Check Extractor...')
    check_and_download_models(EX_WEIGHT_PATH, EX_MODEL_PATH, REMOTE_EX_PATH)

    env_id = args.env_id

    # net initialize
    detector = ailia.Detector(
        DT_MODEL_PATH,
        DT_WEIGHT_PATH,
        len(COCO_CATEGORY),
        format=ailia.NETWORK_IMAGE_FORMAT_RGB,
        channel=ailia.NETWORK_IMAGE_CHANNEL_FIRST,
        range=ailia.NETWORK_IMAGE_RANGE_U_FP32,
        algorithm=ailia.DETECTOR_ALGORITHM_YOLOV3,
        env_id=env_id,
    )
    extractor = ailia.Net(EX_MODEL_PATH, EX_WEIGHT_PATH, env_id=env_id)

    if args.benchmark:
        args.no_detector = True
        args.pairimage[0] = "01840_c40752s1_00280_01.jpg"
        args.pairimage[1] = "01840_c40752s1_00307_01.jpg"

    if args.pairimage[0] is not None and args.pairimage[1] is not None:
        logger.info(
            'Checking if the vehicle in two images is the same vehicle or not.'
        )
        compare_images(detector, extractor)
    else:
        logger.info('Deep SORT started')
        if args.video is None:
            args.video = 0
        recognize_from_video(detector, extractor)


if __name__ == '__main__':
    main()
