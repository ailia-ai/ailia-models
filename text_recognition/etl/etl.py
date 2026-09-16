import codecs
import sys
import time

import ailia
import cv2

# import original modules
sys.path.append('../../util')
# logger
from logging import getLogger  # noqa: E402

import webcamera_utils  # noqa: E402
from image_utils import imread  # noqa: E402
from model_utils import check_and_download_models  # noqa: E402
from arg_utils import get_base_parser, update_parser  # noqa: E402

logger = getLogger(__name__)


# ======================
# PARAMETERS
# ======================
MODEL_PATH = 'etl_BINARY_squeezenet128_20.prototxt'
WEIGHT_PATH = 'etl_BINARY_squeezenet128_20.caffemodel'
REMOTE_PATH = 'https://storage.googleapis.com/ailia-models/etl/'

IMAGE_PATH = 'font.png'
IMAGE_HEIGHT = 28
IMAGE_WIDTH = 28
ETL_PATH = 'etl_BINARY_squeezenet128_20.txt'
MAX_CLASS_COUNT = 3
SLEEP_TIME = 0  # for webcam mode


# ======================
# Arguemnt Parser Config
# ======================
parser = get_base_parser(
    'Japanese character classification model.', IMAGE_PATH, None
)
args = update_parser(parser)


# ======================
# Utils
# ======================
def preprocess_image(img):
    if img.shape[2] == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2BGRA)
    elif img.shape[2] == 1:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGRA)
    img = cv2.bitwise_not(img)
    return img


# ======================
# Main functions
# ======================
def recognize_from_image():
    # net initialize
    classifier = ailia.Classifier(
        MODEL_PATH,
        WEIGHT_PATH,
        env_id=args.env_id,
        format=ailia.NETWORK_IMAGE_FORMAT_GRAY,
        range=ailia.NETWORK_IMAGE_RANGE_U_FP32,
        memory_mode=args.memory_mode,
    )

    # input image loop
    for image_path in args.input:
        # prepare input data
        logger.info(image_path)
        etl_word = codecs.open(ETL_PATH, 'r', 'utf-8').readlines()
        img = imread(image_path, cv2.IMREAD_UNCHANGED)
        if img is None:
            logger.error("can not open "+image_path)
            continue
        img = preprocess_image(img)

        # inference
        logger.info('Start inference...')
        if args.benchmark:
            logger.info('BENCHMARK mode')
            for i in range(5):
                start = int(round(time.time() * 1000))
                classifier.compute(img, MAX_CLASS_COUNT)
                end = int(round(time.time() * 1000))
                logger.info(f'\tailia processing time {end - start} ms')
        else:
            classifier.compute(img, MAX_CLASS_COUNT)

        # get result
        count = classifier.get_class_count()
        logger.info(f'class_count: {count}')

        for idx in range(count):
            logger.info(f"+ idx={idx}")
            info = classifier.get_class(idx)
            logger.info(
                f"  category={info.category} [ {etl_word[info.category].rstrip()} ]"
            )
            logger.info(f"  prob={info.prob}")


def recognize_from_video():
    etl_word = codecs.open(ETL_PATH, 'r', 'utf-8').readlines()

    # net initialize
    classifier = ailia.Classifier(
        MODEL_PATH,
        WEIGHT_PATH,
        env_id=args.env_id,
        format=ailia.NETWORK_IMAGE_FORMAT_GRAY,
        range=ailia.NETWORK_IMAGE_RANGE_U_FP32,
        memory_mode=args.memory_mode,
    )

    capture = webcamera_utils.get_capture(args.video)

    # create video writer if savepath is specified as video format
    if args.savepath is not None:
        f_h = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        f_w = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        save_h, save_w = webcamera_utils.calc_adjust_fsize(
            f_h, f_w, IMAGE_HEIGHT, IMAGE_WIDTH
        )
        writer = webcamera_utils.get_writer(args.savepath, save_h, save_w)
    else:
        writer = None
    
    frame_shown = False
    while(True):
        ret, frame = capture.read()
        if (cv2.waitKey(1) & 0xFF == ord('q')) or not ret:
            break
        if frame_shown and cv2.getWindowProperty('frame', cv2.WND_PROP_VISIBLE) == 0:
            break
        
        in_frame, frame = webcamera_utils.adjust_frame_size(
            frame, IMAGE_HEIGHT, IMAGE_WIDTH
        )
        frame = preprocess_image(frame)

        # inference
        # compute execution time
        classifier.compute(frame, MAX_CLASS_COUNT)

        # get result
        count = classifier.get_class_count()

        logger.info('=' * 80)
        logger.info(f'class_count: {count}')
        for idx in range(count):
            logger.info(f"+ idx={idx}")
            info = classifier.get_class(idx)
            logger.info(
                f"  category={info.category} [ {etl_word[info.category].rstrip()} ]"
            )
            logger.info(f"  prob={info.prob}")
        cv2.imshow('frame', in_frame)
        frame_shown = True
        # save results
        if writer is not None:
            writer.write(in_frame)
        time.sleep(SLEEP_TIME)

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
