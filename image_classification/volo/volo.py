import sys
import time

import cv2
import numpy as np

import ailia
import volo_labels

# import original modules
sys.path.append('../../util')
from arg_utils import get_base_parser, update_parser  # noqa: E402
from model_utils import check_and_download_models  # noqa: E402
from image_utils import load_image  # noqa: E402
from classifier_utils import plot_results, print_results  # noqa: E402
import webcamera_utils  # noqa: E402

# logger
from logging import getLogger   # noqa: E402
logger = getLogger(__name__)


# ======================
# Parameters 1
# ======================
IMAGE_PATH = 'pizza.jpg'

SLEEP_TIME = 0


# ======================
# Arguemnt Parser Config
# ======================
parser = get_base_parser('Image classification model: VOLO', IMAGE_PATH, None)
parser.add_argument(
    '--arch',
    default="volo_d1_224", 
    type=str, 
    choices=["volo_d1_224" ,"volo_d1_384","volo_d2_224", "volo_d2_384","volo_d3_224","volo_d3_448", "volo_d4_224", "volo_d4_448" ,"volo_d5_224","volo_d5_448","volo_d5_512"]

)
args = update_parser(parser)

# ======================
# Parameters 2
# ======================

WEIGHT_PATH = args.arch + '.onnx'
MODEL_PATH  = args.arch + '.onnx.prototxt'
REMOTE_PATH = 'https://storage.googleapis.com/ailia-models/volo/'

IMAGE_HEIGHT = 224
IMAGE_WIDTH  = 224
# ======================
# Main functions
# ======================

def resize_and_center_crop(image, input_size, crop_pct):
    height, width = image.shape[:2]
    aspect_ratio = width / height
    target_size = input_size

    if aspect_ratio > 1.0:
        resize_width = int(width * (target_size / height))
        resize_height = target_size
    else:
        resize_width = target_size
        resize_height = int(height * (target_size / width))
    
    resized_image = cv2.resize(image, (resize_width, resize_height), interpolation=cv2.INTER_LINEAR)

    crop_size = int(round(target_size * crop_pct))
    start_x = (resize_width - crop_size) // 2
    start_y = (resize_height - crop_size) // 2

    cropped_image = resized_image[start_y:start_y+crop_size, start_x:start_x+crop_size]

    return cropped_image


def recognize_from_image():
    # net initialize
    net = ailia.Net(MODEL_PATH, WEIGHT_PATH, env_id=args.env_id, memory_mode=args.memory_mode)

    # input image loop
    for image_path in args.input:
        # prepare input data
        logger.info(image_path)
        input_data = load_image(
            image_path,
            (IMAGE_HEIGHT, IMAGE_WIDTH),
            normalize_type='ImageNet',
            gen_input_ailia=True,
        )
        input_data = input_data.astype(np.float32)

        # inference
        logger.info('Start inference...')
        if args.benchmark:
            logger.info('BENCHMARK mode')
            for i in range(args.benchmark_count):
                start = int(round(time.time() * 1000))

                preds_ailia = net.predict(input_data)

                end = int(round(time.time() * 1000))
                logger.info(f'\tailia processing time {end - start} ms')
        else:
            preds_ailia = net.predict(input_data)

        # postprocessing
        print_results(preds_ailia, volo_labels.imagenet_category)
    logger.info('Script finished successfully.')


def recognize_from_video():
    # net initialize
    net = ailia.Net(MODEL_PATH, WEIGHT_PATH, env_id=args.env_id, memory_mode=args.memory_mode)

    capture = webcamera_utils.get_capture(args.video)

    # create video writer if savepath is specified as video format
    if args.savepath is not None:
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

        _, input_data = webcamera_utils.preprocess_frame(
            frame, IMAGE_HEIGHT, IMAGE_WIDTH, normalize_type='ImageNet'
        )
        input_data = input_data.astype(np.float32)

        # inference
        preds_ailia = net.predict(input_data)

        # postprocessing
        plot_results(frame, preds_ailia, volo_labels.imagenet_category)
        cv2.imshow('frame', frame)
        frame_shown = True
        time.sleep(SLEEP_TIME)

        # save results
        if writer is not None:
            writer.write(frame)

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
