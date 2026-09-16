import sys
import time

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
WEIGHT_PATH = 'yolov2.onnx'
MODEL_PATH = 'yolov2.onnx.prototxt'
REMOTE_PATH = 'https://storage.googleapis.com/ailia-models/yolov2/'

IMAGE_PATH = 'input.jpg'
SAVE_IMAGE_PATH = 'output.png'
IMAGE_HEIGHT = 416  # for video mode
IMAGE_WIDTH = 416  # for video mode

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
# coco anchors
ANCHORS = np.array(
    [0.57273, 0.677385, 1.87446, 2.06253, 3.33843,
     5.47434, 7.88282, 3.52778, 9.77052, 9.16828]
)
THRESHOLD = 0.2
IOU = 0.45


# ======================
# Arguemnt Parser Config
# ======================
parser = get_base_parser('Yolov2 model', IMAGE_PATH, SAVE_IMAGE_PATH)
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
def recognize_from_image():
    # net initialize
    detector = ailia.Detector(
        MODEL_PATH,
        WEIGHT_PATH,
        len(COCO_CATEGORY),
        format=ailia.NETWORK_IMAGE_FORMAT_RGB,
        channel=ailia.NETWORK_IMAGE_CHANNEL_FIRST,
        range=ailia.NETWORK_IMAGE_RANGE_S_FP32,
        algorithm=ailia.DETECTOR_ALGORITHM_YOLOV2,
        env_id=args.env_id,
        memory_mode=args.memory_mode,
    )
    detector.set_anchors(ANCHORS)
    if args.profile:
        detector.set_profile_mode(True)

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

        # plot result
        res_img = plot_results(detector, img, COCO_CATEGORY)
        savepath = get_savepath(args.savepath, image_path)
        logger.info(f'saved at : {savepath}')
        cv2.imwrite(savepath, res_img)

        # write prediction
        if args.write_prediction is not None:
            ext = args.write_prediction
            pred_file = "%s.%s" % (savepath.rsplit('.', 1)[0], ext)
            write_predictions(pred_file, detector, img, category=COCO_CATEGORY, file_type=ext)

    if args.profile:
        print(detector.get_summary())

    logger.info('Script finished successfully.')


def recognize_from_video():
    # net initialize
    detector = ailia.Detector(
        MODEL_PATH,
        WEIGHT_PATH,
        len(COCO_CATEGORY),
        format=ailia.NETWORK_IMAGE_FORMAT_RGB,
        channel=ailia.NETWORK_IMAGE_CHANNEL_FIRST,
        range=ailia.NETWORK_IMAGE_RANGE_S_FP32,
        algorithm=ailia.DETECTOR_ALGORITHM_YOLOV2,
        env_id=args.env_id,
        memory_mode=args.memory_mode,
    )
    detector.set_anchors(ANCHORS)

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
        ret, frame = capture.read()
        if (cv2.waitKey(1) & 0xFF == ord('q')) or not ret:
            break
        if frame_shown and cv2.getWindowProperty('frame', cv2.WND_PROP_VISIBLE) == 0:
            break

        img = cv2.cvtColor(frame, cv2.COLOR_RGB2BGRA)
        detector.compute(img, THRESHOLD, IOU)
        res_img = plot_results(detector, frame, COCO_CATEGORY, logging = False)
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
