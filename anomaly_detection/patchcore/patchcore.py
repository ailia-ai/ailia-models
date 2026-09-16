import os
import pickle
import sys
import time
from typing import Any, Dict, List, Optional
from sklearn.metrics import precision_recall_curve

import ailia
import cv2
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

# import original modules
sys.path.append("../../util")
# logger
from logging import getLogger  # noqa: E402

import webcamera_utils  # noqa: E402
from detector_utils import load_image  # noqa: E402
from model_utils import check_and_download_models  # noqa: E402
from arg_utils import get_base_parser, get_savepath, update_parser  # noqa: E402

from patchcore_utils import *

logger = getLogger(__name__)

# ======================
# Parameters
# ======================

REMOTE_PATH = "https://storage.googleapis.com/ailia-models/padim/"
INFER_TEST_IMAGE_PATH = "./bottle_000.png"
SAVE_INFER_TEST_IMAGE_PATH = "./output.png"
IMAGE_RESIZE = 256
KEEP_ASPECT = True

# ======================
# Arguemnt Parser Config
# ======================

parser = get_base_parser(
    "PatchCore model", INFER_TEST_IMAGE_PATH, SAVE_INFER_TEST_IMAGE_PATH
)
parser.add_argument(
    "-a",
    "--arch",
    default="wide_resnet50_2",
    choices=("resnet18", "wide_resnet50_2"),
    help="arch model.",
)
parser.add_argument(
    "-f",
    "--feat",
    metavar="PICKLE_FILE",
    default=None,
    help="train set feature pkl files.",
)
parser.add_argument("-bs", "--batch_size", default=32, help="batch size.")
parser.add_argument(
    "-tr",
    "--train_dir",
    metavar="TRAIN_DIR",
    default="./train",
    help="directory of the train files.",
)
parser.add_argument(
    "-te",
    "--test_dir",
    metavar="TEST_DIR",
    default="./test",
    help="directory of tes test files",
)
parser.add_argument(
    "-gt",
    "--gt_dir",
    metavar="DIR",
    default="./gt_masks",
    help="directory of the ground truth mask files.",
)
parser.add_argument("-th", "--threshold", type=float, default=None, help="threshold")
parser.add_argument(
    "-ag", "--aug", action="store_true", help="process with augmentation."
)
parser.add_argument(
    "-an",
    "--aug_num",
    type=int,
    default=5,
    help="specify the amplification number of augmentation.",
)
parser.add_argument(
    "-c",
    "--coreset_sampling_ratio",
    type=float,
    default=0.001,
    help="specify the coreset sampling ratio",
)
parser.add_argument(
    "-n",
    "--n_neighbors",
    type=int,
    default=9,
    help="the number of neighbors",
)
args = update_parser(parser)

# get_blob_data() reads an intermediate blob, so REUSE_INTERSTAGE cannot be used
args.memory_mode = ailia.get_memory_mode(reduce_constant=True)
logger.info(f'memory_mode: {args.memory_mode} (overridden)')


# ======================
# Main functions
# ======================


def plot_fig(
    file_list: List[str],
    test_imgs: List[np.ndarray],
    scores: np.ndarray,
    anormal_scores: np.ndarray,
    gt_imgs: Optional[np.ndarray],
    threshold: float,
    savepath: str,
):
    num = len(file_list)
    vmax = scores.max() * 255.0
    vmin = scores.min() * 255.0
    no_gt_img: bool = gt_imgs is None

    for i, (image_path, img) in enumerate(zip(file_list, test_imgs)):
        img = denormalization(img)

        if no_gt_img:
            gt = np.zeros((1, 1, 1))
        else:
            gt = gt_imgs[i].transpose(1, 2, 0).squeeze()

        heat_map, mask, vis_img = visualize(img, scores[i], threshold)

        fig_img, ax_img = plt.subplots(1, 5, figsize=(12, 3))
        fig_img.subplots_adjust(right=0.9)

        fig_img.suptitle(
            "Input : " + image_path + "  Anomaly score : " + str(anormal_scores[i])
        )
        logger.info("Anomaly score : " + str(anormal_scores[i]))

        norm = matplotlib.colors.Normalize(vmin=vmin, vmax=vmax)

        for ax_i in ax_img:
            ax_i.axes.xaxis.set_visible(False)
            ax_i.axes.yaxis.set_visible(False)

        ax_img[0].imshow(img)
        ax_img[0].title.set_text("Image")
        ax_img[1].imshow(gt, cmap="gray")
        ax_img[1].title.set_text("GroundTruth")
        ax = ax_img[2].imshow(heat_map, cmap="jet", norm=norm)
        ax_img[2].imshow(img, cmap="gray", interpolation="none")
        ax_img[2].imshow(heat_map, cmap="jet", alpha=0.5, interpolation="none")
        ax_img[2].title.set_text("Predicted heat map")
        ax_img[3].imshow(mask, cmap="gray")
        ax_img[3].title.set_text("Predicted mask")
        ax_img[4].imshow(vis_img)
        ax_img[4].title.set_text("Segmentation result")
        left = 0.92
        bottom = 0.15
        width = 0.015
        height = 1 - 2 * bottom
        rect = [left, bottom, width, height]
        cbar_ax = fig_img.add_axes(rect)
        cb = plt.colorbar(ax, shrink=0.6, cax=cbar_ax, fraction=0.046)
        cb.ax.tick_params(labelsize=8)
        font = {
            "family": "serif",
            "color": "black",
            "weight": "normal",
            "size": 8,
        }
        cb.set_label("Anomaly Score", fontdict=font)

        if "." in savepath.split("/")[-1]:
            savepath_tmp = get_savepath(savepath, image_path, ext=".png")
        else:
            filename_tmp = image_path.split("/")[-1]
            ext_tmp = "." + filename_tmp.split(".")[-1]
            filename_tmp = filename_tmp.replace(ext_tmp, ".png")

            savepath_tmp = "%s/%s" % (savepath, filename_tmp)
        logger.info(f"saved at : {savepath_tmp}")
        fig_img.savefig(savepath_tmp, dpi=100)
        plt.close()


def train_from_image_or_video(net: ailia.wrapper.Net, params: Dict[str, Any]):
    if not os.path.exists(args.train_dir):
        logger.error("Train folder doesn't exist")
        return

    # training
    train_outputs = training(
        net,
        params,
        IMAGE_RESIZE,
        KEEP_ASPECT,
        int(args.batch_size),
        args.train_dir,
        args.aug,
        args.aug_num,
        args.coreset_sampling_ratio,
        logger,
    )

    if args.feat:
        train_feat_file = args.feat
    else:
        train_dir = args.train_dir
        train_feat_file = "%s.pkl" % os.path.basename(train_dir)

    logger.info("saving train set feature to : %s ..." % train_feat_file)

    with open(train_feat_file, "wb") as f:
        pickle.dump(train_outputs, f)

    logger.info("saved.")
    return train_outputs


def load_gt_imgs(gt_type_dir: str) -> List[np.ndarray]:
    gt_imgs = []
    for i_img in range(0, len(args.input)):
        image_path = args.input[i_img]
        gt_img = None
        if gt_type_dir:
            fname = os.path.splitext(os.path.basename(image_path))[0]
            gt_fpath = os.path.join(gt_type_dir, fname + "_mask.png")
            if os.path.exists(gt_fpath):
                gt_img = load_image(gt_fpath)
                gt_img = cv2.cvtColor(gt_img, cv2.COLOR_BGRA2RGB)
                gt_img = preprocess(
                    gt_img, IMAGE_RESIZE, mask=True, keep_aspect=KEEP_ASPECT
                )
                if gt_img is not None:
                    gt_img = gt_img[0, [0]]
                else:
                    gt_img = np.zeros((1, IMAGE_SIZE, IMAGE_SIZE))

        gt_imgs.append(gt_img)

    return gt_imgs


def decide_threshold(scores, gt_imgs):
    # get optimal threshold
    gt_mask = np.asarray(gt_imgs)
    precision, recall, thresholds = precision_recall_curve(gt_mask.flatten(), scores.flatten())
    a = 2 * precision * recall
    b = precision + recall
    f1 = np.divide(a, b, out=np.zeros_like(a), where=b != 0)
    threshold = thresholds[np.argmax(f1)]
    return threshold


def decide_threshold_from_gt_image(
    net: ailia.wrapper.Net,
    params: Dict[str, Any],
    train_outputs,
    gt_imgs: List[np.ndarray],
):
    score_map = []
    for i_img in range(0, len(args.input)):
        logger.info("from (%s) " % (args.input[i_img]))

        image_path = args.input[i_img]
        img = load_image(image_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2RGB)
        img = preprocess(img, IMAGE_RESIZE, keep_aspect=KEEP_ASPECT)

        dist_tmp = infer(net, params, train_outputs, img)

        score_map.append(dist_tmp)

    scores = normalize_score_maps(score_map)
    threshold = decide_threshold(scores, gt_imgs)

    return threshold


def infer_from_image(
    net: ailia.wrapper.Net,
    params: Dict[str, Any],
    train_outputs,
    threshold: float,
    gt_imgs: Optional[List[np.ndarray]],
):
    if len(args.input) == 0:
        logger.error("Input file not found")
        return

    test_imgs = []

    score_map = []
    for i_img in range(0, len(args.input)):
        logger.info("from (%s) " % (args.input[i_img]))

        image_path = args.input[i_img]
        img = load_image(image_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2RGB)
        img = preprocess(img, IMAGE_RESIZE, keep_aspect=KEEP_ASPECT)

        test_imgs.append(img[0])

        if args.benchmark:
            logger.info("BENCHMARK mode")
            total_time = 0
            for i in range(args.benchmark_count):
                start = int(round(time.time() * 1000))
                dist_tmp = infer(net, params, train_outputs, img)
                end = int(round(time.time() * 1000))
                logger.info(f"\tailia processing time {end - start} ms")
                if i != 0:
                    total_time = total_time + (end - start)
            logger.info(f"\taverage time {total_time / (args.benchmark_count - 1)} ms")
        else:
            dist_tmp = infer(net, params, train_outputs, img)

        score_map.append(dist_tmp)

    scores = normalize_score_maps(score_map)
    anormal_scores = calculate_anormal_scores(score_map)

    # Plot gt image
    plot_fig(
        args.input, test_imgs, scores, anormal_scores, gt_imgs, threshold, args.savepath
    )


def infer_from_video(
    net: ailia.wrapper.Net,
    params: Dict[str, Any],
    train_outputs,
    threshold: float,
):
    capture = webcamera_utils.get_capture(args.video)
    if args.savepath != SAVE_INFER_TEST_IMAGE_PATH:
        f_h = int(IMAGE_SIZE)
        f_w = int(IMAGE_SIZE) * 3
        writer = webcamera_utils.get_writer(args.savepath, f_h, f_w)
    else:
        writer = None

    score_map = []

    frame_shown = False
    while True:
        ret, frame = capture.read()
        if (cv2.waitKey(1) & 0xFF == ord("q")) or not ret:
            break
        if frame_shown and cv2.getWindowProperty("frame", cv2.WND_PROP_VISIBLE) == 0:
            break

        img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = preprocess(img, IMAGE_RESIZE, keep_aspect=KEEP_ASPECT)

        dist_tmp = infer(net, params, train_outputs, img)

        score_map.append(dist_tmp)
        scores = normalize_score_maps(
            score_map
        )  # min max is calculated dynamically, please set fixed min max value from calibration data for production

        heat_map, mask, vis_img = visualize(
            denormalization(img[0]), scores[len(scores) - 1], threshold
        )
        frame = pack_visualize(heat_map, mask, vis_img, scores)

        cv2.imshow("frame", frame)
        frame_shown = True

        if writer is not None:
            writer.write(frame)

    capture.release()
    cv2.destroyAllWindows()
    if writer is not None:
        writer.release()


def train_and_infer(net: ailia.wrapper.Net, params: Dict[str, Any]):
    if args.feat:
        logger.info('loading train set feature from: %s' % args.feat)
        with open(args.feat, 'rb') as f:
            embedding_coreset = pickle.load(f)
        logger.info('loaded.')
    else:
        embedding_coreset = train_from_image_or_video(net, params)

    if args.threshold is None:
        if args.video:
            threshold = 0.5
            gt_imgs = None
            logger.info('Please set threshold manually for video mdoe')
        else:
            gt_type_dir = args.gt_dir if args.gt_dir else None
            if gt_type_dir is None or not os.path.exists(gt_type_dir):
                logger.error("Ground truth folder doesn't exist")
                return

            gt_imgs = load_gt_imgs(gt_type_dir)

            threshold = decide_threshold_from_gt_image(net, params, embedding_coreset, gt_imgs)
            logger.info('Optimal threshold: %f' % threshold)
    else:
        threshold = args.threshold
        gt_imgs = None

    if args.video:
        infer_from_video(net, params, embedding_coreset, threshold)
    else:
        infer_from_image(net, params, embedding_coreset, threshold, gt_imgs)
    logger.info('Script finished successfully.')



def main():
    # model files check and download
    weight_path, model_path, params = get_params(args.arch, args.n_neighbors)
    check_and_download_models(weight_path, model_path, REMOTE_PATH)

    # create net instance
    net = ailia.Net(model_path, weight_path, env_id=args.env_id, memory_mode=args.memory_mode)
    train_and_infer(net, params)


if __name__ == "__main__":
    main()
