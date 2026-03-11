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

logger = getLogger(__name__)


# ======================
# Parameters
# ======================

WEIGHT_PATH = "depth_pro.onnx"
MODEL_PATH = "depth_pro.onnx.prototxt"

REMOTE_PATH = "https://storage.googleapis.com/ailia-models/depth_pro/"

IMAGE_SIZE = 1536

DEFAULT_INPUT_PATH = 'demo.png'
DEFAULT_SAVE_PATH = 'output.png'

# ======================
# Arguemnt Parser Config
# ======================
parser = get_base_parser(
    'Depth Pro', DEFAULT_INPUT_PATH, DEFAULT_SAVE_PATH, large_model = True
)

parser.add_argument(
    '-g', '--grey', action='store_true',
    help="Save image as single channel(greyscale)"
)

args = update_parser(parser)


# ======================
# Helper functions
# ======================

def preprocess(image):
    """Preprocess image for DepthPro.

    Args:
        image: BGR image (H, W, 3), uint8

    Returns:
        preprocessed: (1, 3, 1536, 1536), float32
        original height, original width
    """
    h, w = image.shape[:2]

    # Convert BGR to RGB and normalize to [0, 1]
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0

    # Resize to model input size
    image = cv2.resize(image, (IMAGE_SIZE, IMAGE_SIZE), interpolation=cv2.INTER_CUBIC)

    # Normalize with mean=0.5, std=0.5
    image = (image - 0.5) / 0.5

    # HWC -> CHW, add batch dimension
    image = image.transpose(2, 0, 1)[np.newaxis, :, :, :]
    image = image.astype(np.float32)

    return image, h, w


def post_process(predicted_depth, h, w):
    """Convert predicted depth to visualization.

    Args:
        predicted_depth: (1, 1536, 1536) depth in meters
        h: original height
        w: original width

    Returns:
        depth visualization: (h, w, 3) uint8 or (h, w) float
    """
    depth = predicted_depth[0]

    # Resize to original resolution
    depth = cv2.resize(depth, (w, h), interpolation=cv2.INTER_LINEAR)

    # Normalize for visualization
    depth_vis = (depth - depth.min()) / (depth.max() - depth.min()) * 255.0
    if not args.grey:
        depth_vis = depth_vis.astype(np.uint8)
        depth_vis = cv2.applyColorMap(depth_vis, cv2.COLORMAP_INFERNO)

    return depth_vis


# ======================
# Main functions
# ======================

def recognize_from_image(model):
    # input image loop
    logger.info('Start inference...')
    for image_path in args.input:
        # prepare input data
        org_img = imread(image_path)
        image, h, w = preprocess(org_img)

        if args.benchmark and not (args.video is not None):
            logger.info('BENCHMARK mode')
            for i in range(5):
                start = int(round(time.time() * 1000))
                output = model.predict([image])
                end = int(round(time.time() * 1000))
                logger.info(f'\tailia processing time {end - start} ms')
        else:
            output = model.predict([image])

        predicted_depth = output[0]

        depth_vis = post_process(predicted_depth, h, w)

        savepath = get_savepath(args.savepath, image_path, ext='.png')
        logger.info(f'saving result to {savepath}')
        cv2.imwrite(savepath, depth_vis)

    logger.info('Script finished successfully.')


def recognize_from_video(model):
    capture = webcamera_utils.get_capture(args.video)

    frame_shown = False
    while True:
        ret, frame = capture.read()
        if (cv2.waitKey(1) & 0xFF == ord("q")) or not ret:
            break
        if frame_shown and cv2.getWindowProperty('frame', cv2.WND_PROP_VISIBLE) == 0:
            break

        # inference
        image, h, w = preprocess(frame)
        output = model.predict([image])

        predicted_depth = output[0]

        depth_vis = post_process(predicted_depth, h, w)

        # visualize
        cv2.imshow("frame", depth_vis)
        frame_shown = True

    capture.release()
    logger.info('Script finished successfully.')


def main():
    # model files check and download
    check_and_download_models(WEIGHT_PATH, MODEL_PATH, REMOTE_PATH)

    # net initialize
    memory_mode = ailia.get_memory_mode(reduce_constant=True, ignore_input_with_initializer=True, reduce_interstage=False, reuse_interstage=True)
    model = ailia.Net(MODEL_PATH, WEIGHT_PATH, env_id=args.env_id, memory_mode=memory_mode)

    if args.video is not None:
        # video mode
        recognize_from_video(model)
    else:
        # image mode
        recognize_from_image(model)


if __name__ == '__main__':
    main()
