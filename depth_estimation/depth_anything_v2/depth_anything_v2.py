import sys
import time

# logger
from logging import getLogger  # noqa: E402

import ailia
import cv2
import numpy as np

# import original modules
sys.path.append("../../util")
import webcamera_utils  # noqa: E402
from arg_utils import get_base_parser, update_parser  # noqa: E402
from depth_anything_v2_util.transform import NormalizeImage, PrepareForNet, Resize
from image_utils import imread  # noqa: E402
from model_utils import check_and_download_models  # noqa: E402

logger = getLogger(__name__)


# ======================
# Parameters
# ======================

WEIGHT_PATH_S = "depth_anything_v2_depth_anything_v2_vits.onnx"
MODEL_PATH_S = "depth_anything_v2_depth_anything_v2_vits.onnx.prototxt"

WEIGHT_PATH_B = "depth_anything_v2_depth_anything_v2_vitb.onnx"
MODEL_PATH_B = "depth_anything_v2_depth_anything_v2_vitb.onnx.prototxt"

WEIGHT_PATH_L = "depth_anything_v2_depth_anything_v2_vitl.onnx"
MODEL_PATH_L = "depth_anything_v2_depth_anything_v2_vitl.onnx.prototxt"

WEIGHT_PATH_HYPERSIM_S = "depth_anything_v2_metric_hypersim_vits.onnx"
MODEL_PATH_HYPERSIM_S = "depth_anything_v2_metric_hypersim_vits.onnx.prototxt"

WEIGHT_PATH_HYPERSIM_B = "depth_anything_v2_metric_hypersim_vitb.onnx"
MODEL_PATH_HYPERSIM_B = "depth_anything_v2_metric_hypersim_vitb.onnx.prototxt"

WEIGHT_PATH_HYPERSIM_L = "depth_anything_v2_metric_hypersim_vitl.onnx"
MODEL_PATH_HYPERSIM_L = "depth_anything_v2_metric_hypersim_vitl.onnx.prototxt"

WEIGHT_PATH_VKITTI_S = "depth_anything_v2_metric_vkitti_vits.onnx"
MODEL_PATH_VKITTI_S = "depth_anything_v2_metric_vkitti_vits.onnx.prototxt"

WEIGHT_PATH_VKITTI_B = "depth_anything_v2_metric_vkitti_vitb.onnx"
MODEL_PATH_VKITTI_B = "depth_anything_v2_metric_vkitti_vitb.onnx.prototxt"

WEIGHT_PATH_VKITTI_L = "depth_anything_v2_metric_vkitti_vitl.onnx"
MODEL_PATH_VKITTI_L = "depth_anything_v2_metric_vkitti_vitl.onnx.prototxt"

REMOTE_PATH = "https://storage.googleapis.com/ailia-models/depth_anything_v2/"

METRIC_DATASETS = ["hypersim", "vkitti"]

DEFAULT_INPUT_PATH = "demo1.png"
DEFAULT_SAVE_PATH = "output.png"

# ======================
# Arguemnt Parser Config
# ======================

parser = get_base_parser("Depth Anything V2", DEFAULT_INPUT_PATH, DEFAULT_SAVE_PATH)
parser.add_argument(
    "--encoder", "-ec", type=str, default="vits", help="model type: vits, vitb, vitl"
)
parser.add_argument(
    "--metric",
    type=str,
    default=None,
    choices=METRIC_DATASETS,
    help="use metric depth model (in meters): "
    "hypersim (indoor, max 20m), vkitti (outdoor, max 80m)",
)
parser.add_argument(
    "-g", "--grey", action="store_true", help="Save image as single channel(greyscale)"
)
args = update_parser(parser)

# ======================
# Helper functions
# ======================

class get_depth_anything_v2_ts():
    def __init__(self, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)):
        self.mean = mean
        self.std = std
        self.resize = Resize(
            width=518,
            height=518,
            resize_target=False,
            keep_aspect_ratio=True,
            ensure_multiple_of=14,
            resize_method='lower_bound',
            image_interpolation_method=cv2.INTER_CUBIC,
        )
        self.normalize = NormalizeImage(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        self.prepare = PrepareForNet()

    def __call__(self, image):
        image = self.resize(image)
        image = self.normalize(image)
        image = self.prepare(image)
        return image


def plot_image(image, depth, savepath=None):
    if savepath is not None:
        logger.info(f'saving result to {savepath}')
        cv2.imwrite(savepath, depth)


def post_process(depth, h, w):
    depth = cv2.resize(depth[0], dsize=(w, h), interpolation=cv2.INTER_LINEAR)
    depth = (depth - depth.min()) / (depth.max() - depth.min()) * 255.0
    if not args.grey:
        depth = depth.astype(np.uint8)
        depth = cv2.applyColorMap(depth, cv2.COLORMAP_INFERNO)
    return depth


# ======================
# Main functions
# ======================

def recognize_from_image(model):
    da_transform = get_depth_anything_v2_ts()

    # input image loop
    logger.info('Start inference...')
    for image_path in args.input:
        # prepare input data
        org_img = cv2.cvtColor(imread(image_path), cv2.COLOR_BGR2RGB) / 255.

        image = da_transform({'image': org_img})['image'][None]
        if args.benchmark and not (args.video is not None):
            logger.info('BENCHMARK mode')
            for i in range(5):
                start = int(round(time.time() * 1000))
                depth = model.predict(image)
                end = int(round(time.time() * 1000))
                logger.info(f'\tailia processing time {end - start} ms')
        else:
            depth = model.predict(image)
        if args.metric is not None:
            logger.info(f'depth range: {depth.min():.3f} - {depth.max():.3f} m')
        depth = post_process(depth, org_img.shape[0], org_img.shape[1])

        # visualize
        plot_image(org_img, depth, args.savepath)

    logger.info('Script finished successfully.')


def recognize_from_video(model):
    da_transform = get_depth_anything_v2_ts()

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
        image = da_transform({'image': frame})['image'][None]
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
    if args.metric == 'hypersim':
        if args.encoder == 'vits':
            WEIGHT_PATH = WEIGHT_PATH_HYPERSIM_S
            MODEL_PATH = MODEL_PATH_HYPERSIM_S
        elif args.encoder == 'vitb':
            WEIGHT_PATH = WEIGHT_PATH_HYPERSIM_B
            MODEL_PATH = MODEL_PATH_HYPERSIM_B
        else:
            WEIGHT_PATH = WEIGHT_PATH_HYPERSIM_L
            MODEL_PATH = MODEL_PATH_HYPERSIM_L
    elif args.metric == 'vkitti':
        if args.encoder == 'vits':
            WEIGHT_PATH = WEIGHT_PATH_VKITTI_S
            MODEL_PATH = MODEL_PATH_VKITTI_S
        elif args.encoder == 'vitb':
            WEIGHT_PATH = WEIGHT_PATH_VKITTI_B
            MODEL_PATH = MODEL_PATH_VKITTI_B
        else:
            WEIGHT_PATH = WEIGHT_PATH_VKITTI_L
            MODEL_PATH = MODEL_PATH_VKITTI_L
    elif args.encoder == 'vits':
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
