import sys
import time

import ailia
import cv2
import numpy as np

# import original modules
sys.path.append('../../util')
# logger
from logging import getLogger  # noqa: E402

import webcamera_utils  # noqa: E402 noqa: E402
from image_utils import imread  # noqa: E402
from model_utils import check_and_download_models  # noqa: E402
from arg_utils import get_base_parser, get_savepath, update_parser  # noqa: E402

logger = getLogger(__name__)

# ======================
# Parameters
# ======================
REMOTE_PATH = 'https://storage.googleapis.com/ailia-models/suim/'

WEIGHT_PATH = "suim.opt.onnx"
MODEL_PATH = "suim.opt.onnx.prototxt"

IMAGE_PATH = 'input.jpg'
SAVE_IMAGE_PATH = 'output.jpg'
HEIGHT = 256
WIDTH = 320


# ======================
# Arguemnt Parser Config
# ======================
parser = get_base_parser('suim model', IMAGE_PATH, SAVE_IMAGE_PATH)
args = update_parser(parser)

# ======================
# Visualize
# ======================

def get_color_map(num_classes):
    num_classes += 1
    cm = num_classes * [0, 0, 0]
    for i in range(0, num_classes):
        j = 0
        lab = i
        while lab:
            cm[i * 3] |= (((lab >> 0) & 1) << (7 - j))
            cm[i * 3 + 1] |= (((lab >> 1) & 1) << (7 - j))
            cm[i * 3 + 2] |= (((lab >> 2) & 1) << (7 - j))
            j += 1
            lab >>= 3

    cm = cm[3:]

    return cm

color_map = get_color_map(256)

def visualize(img, result, weight=0.6):
    cm = [
        color_map[i:i + 3] for i in range(0, len(color_map), 3)
    ]
    cm = np.array(cm).astype("uint8")

    # Use OpenCV LUT for color mapping
    c1 = cv2.LUT(result, cm[:, 0])
    c2 = cv2.LUT(result, cm[:, 1])
    c3 = cv2.LUT(result, cm[:, 2])
    pseudo_img = np.dstack((c1, c2, c3))

    vis_result = cv2.addWeighted(img, weight, pseudo_img, 1 - weight, 0)

    return vis_result

# ======================
# Main functions
# ======================
def recognize_from_image():
    # net initialize
    env_id = args.env_id
    net = ailia.Net(MODEL_PATH, WEIGHT_PATH, env_id=env_id, memory_mode=args.memory_mode)

    # input image loop
    for image_path in args.input:
        # prepare input data
        logger.debug(f'input image: {image_path}')
        img = imread(image_path)
        img = cv2.resize(img, (WIDTH, HEIGHT))
        img_data = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_data = img_data / 255.
        logger.debug(f'input image shape: {img_data.shape}')

        img_data = np.expand_dims(img_data, 0)

        # inference
        logger.info('Start inference...')
        if args.benchmark:
            logger.info('BENCHMARK mode')
            for i in range(5):
                start = int(round(time.time() * 1000))
                pred = net.predict(img_data)
                end = int(round(time.time() * 1000))
                logger.info(f'\tailia processing time {end - start} ms')
        else:
            pred = net.predict(img_data)

        # postprocessing
        pred[pred > 0.5] = 1.
        pred[pred <= 0.5] = 0.

        # save individual output masks
        pred = np.argmax(pred, axis=3).astype(np.uint8)[0]
        output = visualize(img, pred, weight=0.6)

        # save
        savepath = get_savepath(args.savepath, image_path)
        logger.info(f'saved at : {savepath}')
        cv2.imwrite(savepath,output)
    logger.info('Script finished successfully.')


def recognize_from_video():
    # net initialize
    net = ailia.Net(MODEL_PATH, WEIGHT_PATH, env_id=args.env_id, memory_mode=args.memory_mode)

    capture = webcamera_utils.get_capture(args.video)

    # create video writer if savepath is specified as video format
    f_h = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    f_w = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    if args.savepath != SAVE_IMAGE_PATH:
        logger.warning(
            'currently, video results cannot be output correctly...'
        )
        writer = webcamera_utils.get_writer(args.savepath, HEIGHT, WIDTH, rgb=False)
    else:
        writer = None
    
    frame_shown = False
    while (True):
        ret, frame = capture.read()
        if (cv2.waitKey(1) & 0xFF == ord('q')) or not ret:
            break
        if frame_shown and cv2.getWindowProperty('frame', cv2.WND_PROP_VISIBLE) == 0:
            break
        
        frame = cv2.resize(frame, (WIDTH, HEIGHT))
        input = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        input = input / 255.
        input = np.expand_dims(input, 0)

        # inference
        pred = net.predict(input)

        # postprocessing
        pred[pred > 0.5] = 1.
        pred[pred <= 0.5] = 0.

        # save individual output masks
        pred = np.argmax(pred, axis=3).astype(np.uint8)[0]
        output = visualize(frame, pred, weight=0.6)

        cv2.imshow('frame', output)
        frame_shown = True

        # save results
        if writer is not None:
            writer.write(output)

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
