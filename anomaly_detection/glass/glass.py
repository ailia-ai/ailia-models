import os
import sys
import time
from logging import getLogger

import ailia
import cv2
import numpy as np
import scipy.ndimage as ndimage
from PIL import Image

# import original modules
sys.path.append("../../util")
from arg_utils import get_base_parser, get_savepath, update_parser
from detector_utils import load_image
from image_utils import normalize_image
from model_utils import check_and_download_models

logger = getLogger(__name__)

# ======================
# Parameters
# ======================

REMOTE_PATH = "https://storage.googleapis.com/ailia-models/glass/"

IMAGE_PATH = "./bottle_000.png"
SAVE_IMAGE_PATH = "./output.png"

BATCH_SIZE = 8
IMAGE_SIZE = 288

DATASET_NAME = [
    "mvtec_carpet",
    "mvtec_grid",
    "mvtec_leather",
    "mvtec_tile",
    "mvtec_wood",
    "mvtec_bottle",
    "mvtec_cable",
    "mvtec_capsule",
    "mvtec_hazelnut",
    "mvtec_metal_nut",
    "mvtec_pill",
    "mvtec_screw",
    "mvtec_toothbrush",
    "mvtec_transistor",
    "mvtec_zipper",
]

# ======================
# Arguemnt Parser Config
# ======================

parser = get_base_parser("GLASS", IMAGE_PATH, SAVE_IMAGE_PATH)
parser.add_argument(
    "-d",
    "--dataset_name",
    default="mvtec_bottle",
    choices=DATASET_NAME,
    help="dataset name",
)
parser.add_argument(
    "-gt",
    "--gt_dir",
    metavar="DIR",
    default="./gt_masks",
    help="directory of the ground truth mask files.",
)
parser.add_argument("--onnx", action="store_true", help="execute onnxruntime version.")
args = update_parser(parser)


# ======================
# Main functions
# ======================


def crop_img(img):
    im_h, im_w, _ = img.shape

    # resize
    short, long = (im_w, im_h) if im_w <= im_h else (im_h, im_w)
    requested_new_short = IMAGE_SIZE
    new_short, new_long = requested_new_short, int(requested_new_short * long / short)
    ow, oh = (new_short, new_long) if im_w <= im_h else (new_long, new_short)
    if ow != im_w or oh != im_h:
        img = np.array(Image.fromarray(img).resize((ow, oh), Image.Resampling.BILINEAR))

    # center_crop
    if ow > IMAGE_SIZE:
        x = int(round((ow - IMAGE_SIZE) / 2.0))
        img = img[:, x : x + IMAGE_SIZE, :]
    if oh > IMAGE_SIZE:
        y = int(round((oh - IMAGE_SIZE) / 2.0))
        img = img[y : y + IMAGE_SIZE, :, :]

    return img


def load_gt_imgs(img_paths, gt_type_dir):
    gt_imgs = []
    for i_img in range(0, len(img_paths)):
        image_path = img_paths[i_img]
        gt_img = None
        if gt_type_dir:
            fname = os.path.splitext(os.path.basename(image_path))[0]
            gt_fpath = os.path.join(gt_type_dir, fname + "_mask.png")
            if os.path.exists(gt_fpath):
                gt_img = load_image(gt_fpath)
                gt_img = cv2.cvtColor(gt_img, cv2.COLOR_BGRA2BGR)
                gt_img = crop_img(gt_img)

        gt_imgs.append(gt_img)

    return gt_imgs


def preprocess(img):
    img = img[:, :, ::-1]  # BGR -> RBG

    img = normalize_image(img, normalize_type="ImageNet")

    img = img.transpose(2, 0, 1)  # HWC -> CHW
    img = np.expand_dims(img, axis=0)
    img = img.astype(np.float32)

    return img


def predict(net, imgs):
    imgs = [preprocess(img) for img in imgs]
    imgs = np.concatenate(imgs, axis=0)

    b, _, h, w = imgs.shape
    if b < BATCH_SIZE:
        pad = np.zeros((BATCH_SIZE - b, 3, h, w), dtype=np.float32)
        imgs = np.concatenate([imgs, pad], axis=0)

    # feedforward
    if not args.onnx:
        output = net.predict([imgs])
    else:
        output = net.run(None, {"images": imgs})
    scores, masks = output

    scores = scores[:b].tolist()
    masks = masks[:b]

    smoothing = 4
    masks = [ndimage.gaussian_filter(mask, sigma=smoothing) for mask in masks]

    return scores, masks


def recognize_from_image(net):
    img_paths = args.input
    gt_type_dir = args.gt_dir
    gt_imgs = load_gt_imgs(img_paths, gt_type_dir)

    # input image loop
    for img_no in range(0, len(img_paths), BATCH_SIZE):
        # prepare input data
        imgs = []
        for image_path in img_paths[img_no : img_no + BATCH_SIZE]:
            logger.info(image_path)
            img = load_image(image_path)
            img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
            img = crop_img(img)
            imgs.append(img)

        # inference
        logger.info("Start inference...")
        if args.benchmark:
            logger.info("BENCHMARK mode")
            total_time = 0
            for i in range(args.benchmark_count):
                start = int(round(time.time() * 1000))
                scores, masks = predict(net, imgs)
                end = int(round(time.time() * 1000))
                logger.info(f"\tailia processing time {end - start} ms")
                if i != 0:
                    total_time = total_time + (end - start)
            logger.info(f"\taverage time {total_time / (args.benchmark_count - 1)} ms")
        else:
            scores, masks = predict(net, imgs)

        for i in range(len(imgs)):
            defect = imgs[i]
            segmentations = masks[i]
            target = gt_imgs[img_no + i]
            defect = cv2.resize(defect, (IMAGE_SIZE, IMAGE_SIZE))
            target = (
                cv2.resize(target, (IMAGE_SIZE, IMAGE_SIZE))
                if target is not None
                else None
            )

            mask = cv2.cvtColor(
                cv2.resize(segmentations, (defect.shape[1], defect.shape[0])),
                cv2.COLOR_GRAY2BGR,
            )
            mask = (mask * 255).astype("uint8")
            mask = cv2.applyColorMap(mask, cv2.COLORMAP_JET)

            col = [defect, target, mask] if target is not None else [defect, mask]
            res_img = np.hstack(col)
            res_img = cv2.resize(res_img, (256 * len(col), 256))

            # plot result
            savepath = get_savepath(args.savepath, img_paths[img_no + i], ext=".png")
            cv2.imwrite(savepath, res_img)
            logger.info(f"saved at : {savepath}")

    logger.info("Script finished successfully.")


def main():
    WEIGHT_PATH = f"{args.dataset_name}.onnx"
    MODEL_PATH = f"{args.dataset_name}.onnx.prototxt"

    # model files check and download
    check_and_download_models(WEIGHT_PATH, MODEL_PATH, REMOTE_PATH)

    env_id = args.env_id

    # initialize
    if not args.onnx:
        net = ailia.Net(MODEL_PATH, WEIGHT_PATH, env_id=env_id)
    else:
        import onnxruntime

        providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
        net = onnxruntime.InferenceSession(WEIGHT_PATH, providers=providers)

    recognize_from_image(net)


if __name__ == "__main__":
    main()
