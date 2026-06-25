import os
import sys
import time
from logging import getLogger

import ailia
import cv2
import numpy as np

# import original modules
sys.path.append("../../util")
from arg_utils import get_base_parser, get_savepath, update_parser  # noqa: E402
from image_utils import imread  # noqa: E402
from model_utils import check_and_download_models  # noqa: E402

logger = getLogger(__name__)


# ======================
# Parameters
# ======================

MODEL_PARAMS = {
    "vits16": {
        "weight": "dinov3_vits16.onnx",
        "model": "dinov3_vits16.onnx.prototxt",
        "num_prefix_tokens": 5,  # cls token + 4 register tokens
        "patch_size": 16,
    },
    "vitb16": {
        "weight": "dinov3_vitb16.onnx",
        "model": "dinov3_vitb16.onnx.prototxt",
        "num_prefix_tokens": 5,
        "patch_size": 16,
    },
    "vitl16": {
        "weight": "dinov3_vitl16.onnx",
        "model": "dinov3_vitl16.onnx.prototxt",
        "num_prefix_tokens": 5,
        "patch_size": 16,
    },
    "vith16plus": {
        "weight": "dinov3_vith16plus.onnx",
        "model": "dinov3_vith16plus.onnx.prototxt",
        "num_prefix_tokens": 5,
        "patch_size": 16,
    },
    "convnext_tiny": {
        "weight": "dinov3_convnext_tiny.onnx",
        "model": "dinov3_convnext_tiny.onnx.prototxt",
        "num_prefix_tokens": 1,  # cls token only, no register tokens
        "patch_size": 32,  # total stride
    },
}

REMOTE_PATH = "https://storage.googleapis.com/ailia-models/dinov3/"

IMAGE_PATH = "coco_000000039769.jpg"
SAVE_IMAGE_PATH = "output.png"

# 224 (pretraining default) gives only a 14x14 / 7x7 patch grid — too coarse for dense feature viz
DEFAULT_RESOLUTION = 896

IMAGE_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGE_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


# ======================
# Argument Parser Config
# ======================

parser = get_base_parser("DINOv3", IMAGE_PATH, SAVE_IMAGE_PATH)
parser.add_argument(
    "-m",
    "--model_type",
    default="convnext_tiny",
    choices=list(MODEL_PARAMS.keys()),
    help="model type",
)
parser.add_argument(
    "--resolution",
    type=int,
    default=DEFAULT_RESOLUTION,
    help="inference resolution (height=width). the model has no fixed input "
    "size, so any value works; higher gives a denser patch grid.",
)
parser.add_argument(
    "--point",
    type=int,
    nargs=2,
    action="append",
    default=None,
    metavar=("X", "Y"),
    help=(
        "reference point (x y, in the original image's pixel coordinates) "
        "used to build a patch similarity map. can be specified multiple "
        "times: one output panel is saved per point (file names get a "
        "_0, _1, ... suffix). defaults to a single point at the image center."
    ),
)
parser.add_argument("--onnx", action="store_true", help="execute onnxruntime version.")
args = update_parser(parser)


# ======================
# Main functions
# ======================


def preprocess(img, image_size):
    img = img[:, :, ::-1]  # BGR -> RGB

    # rescale before resize: keeps bilinear interpolation in float precision (not uint8)
    img = img.astype(np.float32) / 255.0
    img = cv2.resize(img, (image_size, image_size), interpolation=cv2.INTER_LINEAR)
    img = (img - IMAGE_MEAN) / IMAGE_STD

    img = img.transpose(2, 0, 1)  # HWC -> CHW
    img = np.expand_dims(img, axis=0).astype(np.float32)

    return img


def similarity_map_for_point(patch_tokens, grid_size, point, img_w, img_h):
    if point is None:
        fx, fy = 0.5, 0.5
    else:
        fx, fy = point[0] / img_w, point[1] / img_h

    grid_x = min(max(int(fx * grid_size), 0), grid_size - 1)
    grid_y = min(max(int(fy * grid_size), 0), grid_size - 1)
    ref_index = grid_y * grid_size + grid_x

    # patch_tokens: (num_patches, channels)
    norm = patch_tokens / (np.linalg.norm(patch_tokens, axis=-1, keepdims=True) + 1e-6)
    similarity = (norm @ norm[ref_index]).reshape(grid_size, grid_size)

    # min-max instead of fixed (cos+1)/2: real similarities rarely reach -1,
    # so the fixed mapping leaves the background washed out rather than dark
    sim_min, sim_max = similarity.min(), similarity.max()
    similarity = (similarity - sim_min) / (sim_max - sim_min + 1e-6)

    return similarity, (grid_x, grid_y)


def render_panel(similarity_map, point, img_w, img_h):
    # no photo blend: mixing colormap with image produces muddy rainbow, not clean contrast
    # INTER_CUBIC not INTER_NEAREST: official panels show smooth texture, not hard block edges
    heatmap = (similarity_map * 255).clip(0, 255).astype(np.uint8)
    heatmap = cv2.resize(heatmap, (img_w, img_h), interpolation=cv2.INTER_CUBIC)
    panel = cv2.applyColorMap(heatmap, cv2.COLORMAP_VIRIDIS)

    if point is None:
        point_xy = (img_w // 2, img_h // 2)
    else:
        point_xy = (int(point[0]), int(point[1]))
    cv2.drawMarker(
        panel,
        point_xy,
        (0, 0, 255),
        markerType=cv2.MARKER_CROSS,
        markerSize=16,
        thickness=2,
    )

    return panel


def predict(net, img, image_size):
    pixel_values = preprocess(img, image_size)

    if not args.onnx:
        output = net.predict([pixel_values])
    else:
        output = net.run(None, {"pixel_values": pixel_values})
    last_hidden_state, pooler_output = output

    return last_hidden_state, pooler_output


def recognize_from_image(net):
    params = MODEL_PARAMS[args.model_type]
    num_prefix_tokens = params["num_prefix_tokens"]
    image_size = args.resolution
    grid_size = image_size // params["patch_size"]

    for image_path in args.input:
        logger.info(image_path)

        img = imread(image_path)
        img_h, img_w = img.shape[:2]

        logger.info("Start inference...")
        if args.benchmark:
            logger.info("BENCHMARK mode")
            total_time_estimation = 0
            for i in range(args.benchmark_count):
                start = int(round(time.time() * 1000))
                last_hidden_state, pooler_output = predict(net, img, image_size)
                end = int(round(time.time() * 1000))
                estimation_time = end - start

                logger.info(f"\tailia processing estimation time {estimation_time} ms")
                if i != 0:
                    total_time_estimation = total_time_estimation + estimation_time

            logger.info(
                f"\taverage time estimation {total_time_estimation / (args.benchmark_count - 1)} ms"
            )
        else:
            last_hidden_state, pooler_output = predict(net, img, image_size)

        logger.info(f"model_type: {args.model_type}")
        logger.info(f"last_hidden_state: shape={last_hidden_state.shape}")
        logger.info(
            f"pooler_output: shape={pooler_output.shape}, "
            f"norm={np.linalg.norm(pooler_output):.4f}"
        )

        patch_tokens = last_hidden_state[0, num_prefix_tokens:, :]
        points = args.point if args.point else [None]

        savepath = get_savepath(args.savepath, image_path)
        if len(points) > 1:
            base, ext = os.path.splitext(savepath)

        for i, point in enumerate(points):
            similarity_map, ref_grid = similarity_map_for_point(
                patch_tokens, grid_size, point, img_w, img_h
            )
            logger.info(f"reference patch (grid coords): {ref_grid}")

            panel = render_panel(similarity_map, point, img_w, img_h)

            out_path = f"{base}_{i}{ext}" if len(points) > 1 else savepath
            logger.info(f"saved at : {out_path}")
            cv2.imwrite(out_path, panel)

    logger.info("Script finished successfully.")


def main():
    params = MODEL_PARAMS[args.model_type]
    weight_path, model_path = params["weight"], params["model"]

    # model files check and download
    check_and_download_models(weight_path, model_path, REMOTE_PATH)

    # initialize
    if not args.onnx:
        net = ailia.Net(model_path, weight_path, env_id=args.env_id)
    else:
        import onnxruntime

        net = onnxruntime.InferenceSession(weight_path)

    recognize_from_image(net)


if __name__ == "__main__":
    main()
