import sys
import time

import cv2

import ailia

# import original modules
sys.path.append('../../util')
from arg_utils import get_base_parser, update_parser, get_savepath  # noqa: E402
from model_utils import check_and_download_models  # noqa: E402
import webcamera_utils  # noqa: E402
from detector_utils import plot_results, load_image, write_predictions  # noqa: E402
from nms_utils import nms_between_categories  # noqa: E402

# logger
from logging import getLogger   # noqa: E402
logger = getLogger(__name__)


# ======================
# Parameters
# ======================

MODEL_LISTS = ['yolov3-tiny', 'yolov3', 'mb2-ssd']

IMAGE_PATH = 'ferry.jpg'
SAVE_IMAGE_PATH = 'output.png'
IMAGE_HEIGHT = 416
IMAGE_WIDTH = 416

FACE_CATEGORY = ['unmasked', 'masked']

IOU = 0.45


# ======================
# Arguemnt Parser Config
# ======================
parser = get_base_parser(
    'masked face detection model', IMAGE_PATH, SAVE_IMAGE_PATH
)
parser.add_argument(
    '-a', '--arch', metavar='ARCH',
    default='yolov3-tiny', choices=MODEL_LISTS,
    help='model lists: ' + ' | '.join(MODEL_LISTS)
)
parser.add_argument(
    '-w', '--write_json',
    action='store_true',
    help='Flag to output results to json file.'
)
args = update_parser(parser)


if args.arch == "yolov3-tiny":
    WEIGHT_PATH = 'face-mask-detection-yolov3-tiny.opt.obf.onnx'
    MODEL_PATH = 'face-mask-detection-yolov3-tiny.opt.onnx.prototxt'
    RANGE = ailia.NETWORK_IMAGE_RANGE_U_FP32
    ALGORITHM = ailia.DETECTOR_ALGORITHM_YOLOV3
    THRESHOLD = 0.4
elif args.arch == "yolov3":
    WEIGHT_PATH = 'face-mask-detection-yolov3.opt.obf.onnx'
    MODEL_PATH = 'face-mask-detection-yolov3.opt.onnx.prototxt'
    RANGE = ailia.NETWORK_IMAGE_RANGE_U_FP32
    ALGORITHM = ailia.DETECTOR_ALGORITHM_YOLOV3
    THRESHOLD = 0.4
else:
    WEIGHT_PATH = 'face-mask-detection-mb2-ssd-lite.obf.onnx'
    MODEL_PATH = 'face-mask-detection-mb2-ssd-lite.onnx.prototxt'
    RANGE = ailia.NETWORK_IMAGE_RANGE_S_FP32
    ALGORITHM = ailia.DETECTOR_ALGORITHM_SSD
    THRESHOLD = 0.2
REMOTE_PATH = \
    'https://storage.googleapis.com/ailia-models/face-mask-detection/'


# ======================
# Main functions
# ======================
def recognize_from_image():
    # net initialize
    detector = ailia.Detector(
        MODEL_PATH,
        WEIGHT_PATH,
        len(FACE_CATEGORY),
        format=ailia.NETWORK_IMAGE_FORMAT_RGB,
        channel=ailia.NETWORK_IMAGE_CHANNEL_FIRST,
        range=RANGE,
        algorithm=ALGORITHM,
        env_id=args.env_id,
        memory_mode=args.memory_mode
    )

    # input image loop
    for image_path in args.input:
        # prepare input data
        logger.info(image_path)
        img = load_image(image_path)
        logger.debug(f'input image shape: {img.shape}')

        # inference
        logger.info('Start inference...')
        if args.benchmark:
            logger.info('BENCHMARK mode')
            for i in range(5):
                start = int(round(time.time() * 1000))
                detector.compute(img, THRESHOLD, IOU)
                end = int(round(time.time() * 1000))
                logger.info(f'\tailia processing time {end - start} ms')
        else:
            detector.compute(img, THRESHOLD, IOU)

        # nms
        detections = []
        for idx in range(detector.get_object_count()):
            obj = detector.get_object(idx)
            detections.append(obj)
        detections = nms_between_categories(
            detections,
            img.shape[1],
            img.shape[0],
            categories=[0, 1],
            iou_threshold=IOU,
        )

        # plot result
        res_img = plot_results(detections, img, FACE_CATEGORY)
        savepath = get_savepath(args.savepath, image_path)
        logger.info(f'saved at : {savepath}')
        cv2.imwrite(savepath, res_img)

        if args.write_json:
            json_file = '%s.json' % savepath.rsplit('.', 1)[0]
            write_predictions(json_file, detections, img, category=FACE_CATEGORY, file_type='json')

    logger.info('Script finished successfully.')


def recognize_from_video():
    # net initialize
    detector = ailia.Detector(
        MODEL_PATH,
        WEIGHT_PATH,
        len(FACE_CATEGORY),
        format=ailia.NETWORK_IMAGE_FORMAT_RGB,
        channel=ailia.NETWORK_IMAGE_CHANNEL_FIRST,
        range=RANGE,
        algorithm=ALGORITHM,
        env_id=args.env_id,
        memory_mode=args.memory_mode
    )

    capture = webcamera_utils.get_capture(args.video)

    if args.savepath != SAVE_IMAGE_PATH:
        f_h = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        f_w = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        writer = webcamera_utils.get_writer(
            args.savepath, f_h, f_w
        )
    else:
        writer = None

    frame_shown = False
    while(True):
        ret, frame = capture.read()
        if (cv2.waitKey(1) & 0xFF == ord('q')) or not ret:
            break
        if frame_shown and cv2.getWindowProperty('frame', cv2.WND_PROP_VISIBLE) == 0:
            break

        img = cv2.cvtColor(frame, cv2.COLOR_BGR2BGRA)
        detector.compute(img, THRESHOLD, IOU)

        detections = []
        for idx in range(detector.get_object_count()):
            obj = detector.get_object(idx)
            detections.append(obj)
        detections = nms_between_categories(
            detections,
            frame.shape[1],
            frame.shape[0],
            categories=[0, 1],
            iou_threshold=IOU
        )

        res_img = plot_results(detections, frame, FACE_CATEGORY, logging=False)
        cv2.imshow('frame', res_img)
        frame_shown = True

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

    if args.video is not None:
        # video mode
        recognize_from_video()
    else:
        # image mode
        recognize_from_image()


if __name__ == '__main__':
    main()
