import sys
import time

import ailia
import cv2

import numpy as np

# import original modules
sys.path.append('../../util')

# logger
from logging import getLogger  # noqa: E402

import webcamera_utils  # noqa: E402
from image_utils import imread  # noqa: E402
from model_utils import check_and_download_models  # noqa: E402
from arg_utils import get_base_parser, get_savepath, update_parser  # noqa: E402

from scipy.special import softmax

logger = getLogger(__name__)


# ======================
# Parameters
# ======================

WEIGHT_PATH = "hands_segmentation_pytorch.onnx"
MODEL_PATH = WEIGHT_PATH + '.prototxt'

REMOTE_PATH = "https://storage.googleapis.com/ailia-models/hands_segmentation_pytorch/"

DEFAULT_INPUT_PATH = 'sample_image.jpg'
DEFAULT_SAVE_PATH = 'output.png'

# ======================
# Arguemnt Parser Config
# ======================
parser = get_base_parser(
    'Hands Segmentation in PyTorch - A Plug and Play Model',
    DEFAULT_INPUT_PATH, DEFAULT_SAVE_PATH
)

parser.add_argument(
    '--height', type=int, default=256,
    help='height of the image to run inference on '
)

parser.add_argument(
    '--width', type=int, default=256,
    help='width of the image to run inference on'
)

parser.add_argument(
    '--overlay', action='store_true',
    help='Visualize the mask overlayed on the image'
)

args = update_parser(parser)

# ======================
# Helper functions
# ======================

def preprocess(image, h=None, w=None, mean = np.array([0.485, 0.456, 0.406]), std = np.array([0.229, 0.224, 0.225])):
    
    if h is not None and w is not None:
        image = cv2.resize(image, (w, h))
    image = (image - mean[None,None,:]) / std[None,None,:]
    return image.transpose(2, 0, 1)


def postprocess(image, logits, h, w, overlay=False):
    mask = softmax(logits[0], axis=0)[1]
    mask = cv2.resize(mask, dsize=(w, h), interpolation=cv2.INTER_LINEAR)
    
    if overlay:
        mask = (np.where(mask[:,:,None] > 0.5, np.array([1., 0, 0]) * 0.5 + image * 0.5, image) * 255).astype('uint8')
    else:
        mask = (mask[:,:,None]> 0.5).astype('uint8') * 255
    return mask

# ======================
# Main functions
# ======================

def recognize_from_image(model):
    logger.info('Start inference...')

    image_path = args.input[0]

    # prepare input data
    org_img = cv2.cvtColor(imread(image_path), cv2.COLOR_BGR2RGB) / 255.

    image = preprocess(org_img, h = args.height, w = args.width)[None]

    if args.benchmark and not (args.video is not None):
        logger.info('BENCHMARK mode')
        for i in range(5):
            start = int(round(time.time() * 1000))
            logits = model.predict(image)
            end = int(round(time.time() * 1000))
            logger.info(f'\tailia processing time {end - start} ms')
    else:
        logits = model.predict(image)

    mask = postprocess(org_img, logits, org_img.shape[0], org_img.shape[1], args.overlay)

    # visualize
    logger.info(f'saving result to {args.savepath}')
        
    mask = cv2.cvtColor(mask, cv2.COLOR_RGB2BGR)
    cv2.imwrite(args.savepath, mask)

    logger.info('Script finished successfully.')

def recognize_from_video(model):
    # net initialize

    capture = webcamera_utils.get_capture(args.video)

    frame_shown = False
    while(True):
        ret, frame = capture.read()
        if (cv2.waitKey(1) & 0xFF == ord("q")) or not ret:
            break
        if frame_shown and cv2.getWindowProperty('frame', cv2.WND_PROP_VISIBLE) == 0:
            break
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) / 255.

        # inference
        image = preprocess(frame, h = args.height, w = args.width)[None]
        logits = model.predict(image)
        mask = postprocess(frame, logits, frame.shape[0], frame.shape[1], args.overlay)

        # visualize
        cv2.imshow("frame", mask)
        frame_shown = True

    capture.release()
    logger.info('Script finished successfully.')

def main():
    # model files check and download
    check_and_download_models(WEIGHT_PATH, MODEL_PATH, REMOTE_PATH)
    
    # net initialize
    model = ailia.Net(MODEL_PATH, WEIGHT_PATH, env_id = args.env_id, memory_mode = args.memory_mode)

    if args.video is not None:
        # video mode
        recognize_from_video(model)
    else:
        # image mode
        recognize_from_image(model)


if __name__ == '__main__':
    main()