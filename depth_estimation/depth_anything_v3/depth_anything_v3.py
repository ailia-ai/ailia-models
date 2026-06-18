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

WEIGHT_PATH_S = "depth_anything_v3_da3_vits.onnx"
MODEL_PATH_S = "depth_anything_v3_da3_vits.onnx.prototxt"

WEIGHT_PATH_B = "depth_anything_v3_da3_vitb.onnx"
MODEL_PATH_B = "depth_anything_v3_da3_vitb.onnx.prototxt"

WEIGHT_PATH_L = "depth_anything_v3_da3_vitl.onnx"
MODEL_PATH_L = "depth_anything_v3_da3_vitl.onnx.prototxt"

REMOTE_PATH = "https://storage.googleapis.com/ailia-models/depth_anything_v3/"

DEFAULT_INPUT_PATH = 'demo.png'
DEFAULT_SAVE_PATH = 'output.png'

# ======================
# Arguemnt Parser Config
# ======================
parser = get_base_parser(
    'Depth Anything V3', DEFAULT_INPUT_PATH, DEFAULT_SAVE_PATH
)

parser.add_argument(
    '--encoder', '-ec', type=str, default='vits',
    help='model type: vits, vitb, vitl'
)

parser.add_argument(
    '-g', '--grey', action='store_true',
    help="Save image as single channel(greyscale)"
)

args = update_parser(parser)

# ======================
# Helper functions
# ======================

PROCESS_RES = 504
PATCH_SIZE = 14


def nearest_multiple(x, p):
    down = (x // p) * p
    up = down + p
    return up if abs(up - x) <= abs(x - down) else down


def preprocess(image):
    """Preprocess image following official DA3 InputProcessor.

    1. Resize longest side to PROCESS_RES
    2. Round each dimension to nearest multiple of PATCH_SIZE
    3. Normalize with ImageNet mean/std
    """
    h, w = image.shape[:2]
    longest = max(w, h)
    scale = PROCESS_RES / float(longest)
    new_w = max(1, round(w * scale))
    new_h = max(1, round(h * scale))
    interp = cv2.INTER_CUBIC if scale > 1.0 else cv2.INTER_AREA
    image = cv2.resize(image, (new_w, new_h), interpolation=interp)

    final_w = max(1, nearest_multiple(new_w, PATCH_SIZE))
    final_h = max(1, nearest_multiple(new_h, PATCH_SIZE))
    if final_w != new_w or final_h != new_h:
        upscale = (final_w > new_w) or (final_h > new_h)
        interp2 = cv2.INTER_CUBIC if upscale else cv2.INTER_AREA
        image = cv2.resize(image, (final_w, final_h), interpolation=interp2)

    # Normalize
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    image = (image - mean) / std
    image = image.transpose(2, 0, 1).astype(np.float32)
    return image


def plot_image(image, depth, savepath=None):
    if savepath is not None:
        logger.info(f'saving result to {savepath}')
        cv2.imwrite(savepath, depth)


def post_process(depth, h, w):
    # Handle (1, 1, H, W) output
    if depth.ndim == 4:
        depth = depth[0, 0]
    elif depth.ndim == 3:
        depth = depth[0]
    depth = cv2.resize(depth, dsize=(w, h), interpolation=cv2.INTER_LINEAR)
    # DA3 outputs metric depth (larger = farther).
    # Convert to inverse depth for visualization so that closer objects
    # appear brighter (matching DA3 official visualize_depth()).
    valid = depth > 0
    depth[valid] = 1.0 / depth[valid]
    depth = (depth - depth.min()) / (depth.max() - depth.min()) * 255.0
    if not args.grey:
        depth = depth.astype(np.uint8)
        depth = cv2.applyColorMap(depth, cv2.COLORMAP_INFERNO)
    return depth


# ======================
# Main functions
# ======================

def recognize_from_image(model):
    # input image loop
    logger.info('Start inference...')
    for image_path in args.input:
        # prepare input data
        org_img = cv2.cvtColor(imread(image_path), cv2.COLOR_BGR2RGB) / 255.

        image = preprocess(org_img)[None]
        if args.benchmark and not (args.video is not None):
            logger.info('BENCHMARK mode')
            for i in range(5):
                start = int(round(time.time() * 1000))
                depth = model.predict(image)
                end = int(round(time.time() * 1000))
                logger.info(f'\tailia processing time {end - start} ms')
        else:
            depth = model.predict(image)
        depth = post_process(depth, org_img.shape[0], org_img.shape[1])

        # visualize
        plot_image(org_img, depth, args.savepath)

    logger.info('Script finished successfully.')


def recognize_from_video(model):
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
        image = preprocess(frame)[None]
        depth = model.predict(image)
        depth = post_process(depth, frame.shape[0], frame.shape[1])

        # visualize
        cv2.imshow("frame", depth)
        frame_shown = True

    capture.release()
    logger.info('Script finished successfully.')


def main():
    # model files check and download
    assert args.encoder in ['vits', 'vitb', 'vitl'], \
        'encoder should be vits, vitb, or vitl'
    if args.encoder == 'vits':
        WEIGHT_PATH = WEIGHT_PATH_S
        MODEL_PATH = MODEL_PATH_S
    elif args.encoder == 'vitb':
        WEIGHT_PATH = WEIGHT_PATH_B
        MODEL_PATH = MODEL_PATH_B
    else:
        WEIGHT_PATH = WEIGHT_PATH_L
        MODEL_PATH = MODEL_PATH_L

    check_and_download_models(WEIGHT_PATH, MODEL_PATH, REMOTE_PATH)

    # net initialize
    model = ailia.Net(MODEL_PATH, WEIGHT_PATH, env_id=args.env_id)

    if args.video is not None:
        # video mode
        recognize_from_video(model)
    else:
        # image mode
        recognize_from_image(model)


if __name__ == '__main__':
    main()
