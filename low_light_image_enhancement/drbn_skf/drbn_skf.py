import sys
import time

import ailia
import cv2
import numpy as np
import onnxruntime
from einops import rearrange

sys.path.append('../../util')
from logging import getLogger  # noqa: E402

import webcamera_utils  # noqa: E402
from image_utils import imread  # noqa: E402
from model_utils import check_and_download_models  # noqa: E402
from arg_utils import get_base_parser, update_parser  # noqa: E402

logger = getLogger(__name__)

WEIGHT_PATH = "drbn_skf_lol.onnx"
MODEL_PATH = "drbn_skf_lol.onnx.prototxt"
REMOTE_PATH = 'https://storage.googleapis.com/ailia-models/drbn_skf/'
IMAGE_PATH = 'input.png'
SAVE_IMAGE_PATH = 'output.png'
HEIGHT_SIZE = 400
WIDTH_SIZE = 600
RGB_RANGE = 255

parser = get_base_parser(
    "DRBN_SKF",
    IMAGE_PATH,
    SAVE_IMAGE_PATH,
)
parser.add_argument(
    "-ver", 
    "--version",
    default=1,
    type=int,
    choices=[1, 2],
    help="DRBN SKF version (select 1 or 2)",
)
parser.add_argument(
    "--onnx",
    action="store_true",
    help="use onnxruntime",
)
args = update_parser(parser)

def get_model(model_path, weight_path, env_id):
    return ailia.Net(model_path, weight_path, env_id=env_id, memory_mode=args.memory_mode)


def postprocess(pred):
    pixel_range = 255 / RGB_RANGE
    out = pred * pixel_range
    out = out.clip(0, 255)
    out = np.floor(out + 0.5)
    return out


def recognize_from_image(weight_path, model_path):
    env_id = args.env_id

    if args.onnx:
        net = onnxruntime.InferenceSession(weight_path)
    else:
        net = get_model(model_path, weight_path, env_id)

    for image_path in args.input:
        logger.info(image_path)
        img = imread(image_path) / 255.
        img = img[..., ::-1]
        img = cv2.resize(img, (WIDTH_SIZE, HEIGHT_SIZE), interpolation=cv2.INTER_LANCZOS4)
        img = img[np.newaxis, :]
        img = rearrange(img, "1 h w c -> 1 c h w")
        logger.info(f"input image shape: {img.shape}")
        logger.info("Start inference ...")

        if args.benchmark:
            logger.info("BENCHMARK mode")
            for _ in range(5):
                start = int(round(time.time() * 1000))
                pred = net.run(img)[0]
                end = int(round(time.time() * 1000))
                logger.info(f"\tailia processing time {end - start} ms")
        else:
            img = img.astype(np.float32)

            if args.onnx:
                pred = net.run(["5674"], {"x.1": img})[0]
            else:
                pred = net.run(img)[0]

            pred = rearrange(pred, "1 c h w -> h w c")
            pred = pred[..., ::-1]

        pred_postprocess = postprocess(pred * 255.0)
        # save result
        logger.info(f"saved at : {args.savepath}")
        cv2.imwrite(args.savepath, pred_postprocess)

    logger.info("Script finished successfully.")


def recognize_from_video(weight_path, model_path):
    env_id = args.env_id
    net = get_model(model_path, weight_path, env_id)
    capture = webcamera_utils.get_capture(args.video)

    if args.savepath != SAVE_IMAGE_PATH:
        logger.warning(
            "currently, video results cannot be output correctry..."
        )
        writer = webcamera_utils.get_writer(args.savepath, HEIGHT_SIZE, WIDTH_SIZE, rgb=True)
    else:
        writer = None
    
    frame_shown = False
    while True:
        ret, frame = capture.read()
        if (cv2.waitKey(1) & 0xff == ord("q")) or not ret:
            break
        if frame_shown and cv2.getWindowProperty("frame", cv2.WND_PROP_VISIBLE) == 0:
            break

        img = frame / 255.
        img = img[..., ::-1]
        img = cv2.resize(img, (WIDTH_SIZE, HEIGHT_SIZE), interpolation=cv2.INTER_LANCZOS4)
        img = img[np.newaxis, :]
        img = rearrange(img, "1 h w c -> 1 c h w")
        img = img.astype(np.float32)

        if args.onnx:
            pred = net.run(["5674"], {"x.1": img})[0]
        else:
            pred = net.run(img)[0]

        pred = rearrange(pred, "1 c h w -> h w c")
        pred = pred[..., ::-1]
        pred_postprocess = postprocess(pred * 255.0)
        pred_postprocess = pred_postprocess.astype(np.uint8)

        cv2.imshow("frame", pred_postprocess)
        frame_shown = True

        if writer is not None:
            writer.write(pred_postprocess)
    
    capture.release()
    if not writer is None:
        writer.release()


def main():
    version = args.version
    weight_path= WEIGHT_PATH.replace("lol", f"lol_v{version}")
    model_path= MODEL_PATH.replace("lol", f"lol_v{version}")
    check_and_download_models(weight_path, model_path, REMOTE_PATH)

    if args.video is None:
        recognize_from_image(weight_path, model_path)
    else:
        recognize_from_video(weight_path, model_path)


if __name__ == '__main__':
    main()
