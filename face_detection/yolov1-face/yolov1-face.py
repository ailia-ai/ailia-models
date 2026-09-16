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

# logger.info
from logging import getLogger   # noqa: E402
logger = getLogger(__name__)


# TODO: yolov1-face.py & yolov3-face.py is same, better to merge them ?

# ======================
# Parameters
# ======================
WEIGHT_PATH = 'yolov1-face.caffemodel'
MODEL_PATH = 'yolov1-face.prototxt'
REMOTE_PATH = 'https://storage.googleapis.com/ailia-models/yolov1-face/'

IMAGE_PATH = 'couple.jpg'
SAVE_IMAGE_PATH = 'output.png'
IMAGE_HEIGHT = 448  # for video mode
IMAGE_WIDTH = 448  # for video mode

FACE_CATEGORY = ['face']
THRESHOLD = 0.2
IOU = 0.45


# ======================
# Arguemnt Parser Config
# ======================
parser = get_base_parser(
    'Face Detection using Yolov1', IMAGE_PATH, SAVE_IMAGE_PATH
)
parser.add_argument(
    '-w', '--write_json',
    action='store_true',
    help='Flag to output results to json file.'
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
        len(FACE_CATEGORY),
        format=ailia.NETWORK_IMAGE_FORMAT_RGB,
        channel=ailia.NETWORK_IMAGE_CHANNEL_FIRST,
        range=ailia.NETWORK_IMAGE_RANGE_S_FP32,
        algorithm=ailia.DETECTOR_ALGORITHM_YOLOV1,
        env_id=args.env_id,
        memory_mode=args.memory_mode,
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

        # plot result
        res_img = plot_results(detector, img, FACE_CATEGORY)
        savepath = get_savepath(args.savepath, image_path)
        logger.info(f'saved at : {savepath}')
        cv2.imwrite(savepath, res_img)

        # write prediction
        if args.write_json:
            json_file = '%s.json' % savepath.rsplit('.', 1)[0]
            write_predictions(json_file, detector, img, category=FACE_CATEGORY, file_type='json')

    logger.info('Script finished successfully.')


def recognize_from_video():
    # net initialize
    detector = ailia.Detector(
        MODEL_PATH,
        WEIGHT_PATH,
        len(FACE_CATEGORY),
        format=ailia.NETWORK_IMAGE_FORMAT_RGB,
        channel=ailia.NETWORK_IMAGE_CHANNEL_FIRST,
        range=ailia.NETWORK_IMAGE_RANGE_S_FP32,
        algorithm=ailia.DETECTOR_ALGORITHM_YOLOV1,
        env_id=args.env_id,
        memory_mode=args.memory_mode,
    )

    capture = webcamera_utils.get_capture(args.video)

    # create video writer if savepath is specified as video format
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

        img = cv2.cvtColor(frame, cv2.COLOR_RGB2BGRA)
        detector.compute(img, THRESHOLD, IOU)
        res_img = plot_results(detector, frame, FACE_CATEGORY, logging=False)
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
