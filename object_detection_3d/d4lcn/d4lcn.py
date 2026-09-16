import math
import os
import sys
import time
from io import StringIO

import ailia
import cv2
import matplotlib.pyplot as plt
import numpy as np
import json

# import original modules
sys.path.append('../../util')
# logger
from logging import getLogger  # noqa

from detector_utils import load_image  # noqa
from image_utils import imread  # noqa: E402
from model_utils import check_and_download_models  # noqa
from nms_utils import nms_boxes  # noqa
from arg_utils import get_base_parser, get_savepath, update_parser  # noqa

from d4lcn_utils import (bbox_transform_inv, convertAlpha2Rot,
                         convertRot2Alpha, hill_climb)
from instance_utils import get_2d, read_annot, read_calib_file
from points_utils import plot_3d_bbox

logger = getLogger(__name__)

# ======================
# Parameters
# ======================

WEIGHT_PATH = 'd4lcn.onnx'
MODEL_PATH = 'd4lcn.onnx.prototxt'
REMOTE_PATH = 'https://storage.googleapis.com/ailia-models/d4lcn/'

IMAGE_PATH = '000005.png'
SAVE_IMAGE_PATH = 'output.png'

IMAGE_HEIGHT = 512

# ======================
# Arguemnt Parser Config
# ======================

parser = get_base_parser(
    'D4LCN', IMAGE_PATH, SAVE_IMAGE_PATH
)
parser.add_argument(
    '--calib_path', type=str, default=None,
    help='the calibration file (Camera parameters for image) or stored directory path'
)
parser.add_argument(
    '--depth_path', type=str, default=None,
    help='the depth file (depth maps for image) or stored directory path.'
)
parser.add_argument(
    '-w', '--write_json',
    action='store_true',
    help='Flag to output results to json file.'
)
args = update_parser(parser)


# ======================
# Secondaty Functions
# ======================

def get_path(path, name):
    if path is None:
        path = "calib"
        file_path = "%s/%s.txt" % (path, name)
        if os.path.exists(file_path):
            logger.info("calib file: %s" % file_path)
            return file_path
        # else:
        #     return None
    elif os.path.isdir(path):
        file_path = "%s/%s.txt" % (path, name)
        if os.path.exists(file_path):
            logger.info("calib file: %s" % file_path)
            return file_path
    elif os.path.exists(path):
        logger.info("calib file: %s" % path)
        return path

    logger.error("calib file is not found. (path: %s)" % path)
    sys.exit(-1)


def get_depth(path, name):
    if path is None:
        path = "depth"
        file_path = "%s/%s.png" % (path, name)
        if os.path.exists(file_path):
            logger.info("depth file: %s" % file_path)
            return file_path
        # else:
        #     return None
    elif os.path.isdir(path):
        file_path = "%s/%s.png" % (path, name)
        if os.path.exists(file_path):
            logger.info("depth file: %s" % file_path)
            return file_path
    elif os.path.exists(path):
        logger.info("depth file: %s" % path)
        return path

    logger.error("depth file is not found. (path: %s)" % path)
    sys.exit(-1)


def pred_str(aboxes, p2):
    lbls = ['Car', 'Pedestrian', 'Cyclist']
    nms_topN = 40

    p2_inv = np.linalg.inv(p2)

    results = []
    results_for_json = []
    for boxind in range(0, min(nms_topN, aboxes.shape[0])):
        box = aboxes[boxind, :]
        score = box[4]
        cls = lbls[int(box[5] - 1)]

        if score < 0.75:
            continue

        # 2D box
        x1 = box[0]
        y1 = box[1]
        x2 = box[2]
        y2 = box[3]
        width = (x2 - x1 + 1)
        height = (y2 - y1 + 1)

        # plot 3D box
        x3d = box[6]
        y3d = box[7]
        z3d = box[8]
        w3d = box[9]
        h3d = box[10]
        l3d = box[11]
        ry3d = box[12]

        # Inverse matrix and scale, to 3d camera coordinate
        coord3d = p2_inv.dot(np.array([x3d * z3d, y3d * z3d, 1 * z3d, 1]))
        # convert alpha into ry3d
        ry3d = convertAlpha2Rot(ry3d, coord3d[2], coord3d[0])

        step_r = 0.3 * math.pi
        r_lim = 0.01
        box_2d = np.array([x1, y1, width, height])

        z3d, ry3d, verts_best = hill_climb(
            p2, p2_inv, box_2d, x3d, y3d, z3d, w3d, h3d, l3d, ry3d,
            step_r_init=step_r, r_lim=r_lim)

        # predict a more accurate projection
        coord3d = p2_inv.dot(np.array([x3d * z3d, y3d * z3d, 1 * z3d, 1]))
        alpha = convertRot2Alpha(ry3d, coord3d[2], coord3d[0])

        x3d = coord3d[0]
        y3d = coord3d[1]
        z3d = coord3d[2]

        y3d += h3d / 2

        results.append(
            '{} -1 -1 {:.6f} {:.6f} {:.6f} {:.6f} {:.6f} '
            '{:.6f} {:.6f} {:.6f} {:.6f} {:.6f} {:.6f} {:.6f} {:.6f}'.format(
                cls, alpha, x1, y1, x2, y2, h3d, w3d, l3d, x3d, y3d, z3d, ry3d, score))

        results_for_json.append({
            'class': cls,
            'alpha': alpha,
            'box_2d': np.array([x1, y1, x2, y2]).tolist(),
            'box_3d': np.array([h3d, w3d, l3d, x3d, y3d, z3d, ry3d]).tolist(),
            'score': score
        })

    pred_str = '\n'.join(results)
    return pred_str, json.dumps(results_for_json, indent=2)


def draw_results(img, kpts_2d):
    fig = plt.figure(figsize=(11.3, 9))
    ax = plt.subplot(111)

    fig.gca().set_axis_off()
    fig.subplots_adjust(
        top=1, bottom=0, right=1, left=0,
        hspace=0, wspace=0)
    fig.gca().xaxis.set_major_locator(plt.NullLocator())
    fig.gca().yaxis.set_major_locator(plt.NullLocator())

    height, width, _ = img.shape
    ax.imshow(img)
    ax.set_xlim([0, width])
    ax.set_ylim([0, height])
    ax.invert_yaxis()

    # plot predicted 2D screen coordinates
    for idx, kpts in enumerate(kpts_2d):
        kpts = kpts.reshape(-1, 2)
        plot_3d_bbox(ax, kpts[1:, :2], color='chartreuse', linestyle='-')

    return fig


# ======================
# Main functions
# ======================

def preprocess(img, depth):
    h, w = img.shape[:2]

    scale_factor = IMAGE_HEIGHT / h
    h = np.round(h * scale_factor).astype(int)
    w = np.round(w * scale_factor).astype(int)

    # resize
    img = cv2.resize(img.astype(np.float32), (w, h))
    depth = cv2.resize(depth.astype(np.float32), (w, h))

    # Normalize
    mean = np.array([102.9801, 115.9465, 122.7717])
    std = np.array([1., 1., 1.])
    depth_mean = np.array([4413.1606, 4413.1606, 4413.1606])
    depth_std = np.array([3270.0159, 3270.0159, 3270.0159])
    img = (img - mean) / std
    depth = (depth - depth_mean) / depth_std

    img = np.transpose(img, [2, 0, 1])
    depth = np.transpose(depth, [2, 0, 1])

    img = np.expand_dims(img, axis=0)
    depth = np.expand_dims(depth, axis=0)

    return img, depth


def locate_anchors(anchors, feat_size, stride):
    """
    Spreads each anchor shape across a feature map of size feat_size spaced by a known stride.

    Args:
        anchors (ndarray): N x 4 array describing [x1, y1, x2, y2] displacements for N anchors
        feat_size (ndarray): the downsampled resolution W x H to spread anchors across [feat_h, feat_w]
        stride (int): stride of a network

    Returns:
         ndarray: 2D array = [(W x H) x 5] array consisting of [x1, y1, x2, y2, anchor_index]
    """

    # compute rois
    shift_x = np.array(range(0, feat_size[1], 1)) * float(stride)
    shift_y = np.array(range(0, feat_size[0], 1)) * float(stride)
    [shift_x, shift_y] = np.meshgrid(shift_x, shift_y)

    rois = np.expand_dims(anchors[:, 0:4], axis=1)
    shift_x = np.expand_dims(shift_x, axis=0)
    shift_y = np.expand_dims(shift_y, axis=0)

    shift_x1 = shift_x + np.expand_dims(rois[:, :, 0], axis=2)
    shift_y1 = shift_y + np.expand_dims(rois[:, :, 1], axis=2)
    shift_x2 = shift_x + np.expand_dims(rois[:, :, 2], axis=2)
    shift_y2 = shift_y + np.expand_dims(rois[:, :, 3], axis=2)

    # compute anchor tracker
    anchor_tracker = np.zeros(shift_x1.shape, dtype=float)
    for aind in range(0, rois.shape[0]): anchor_tracker[aind, :, :] = aind

    stack_size = feat_size[0] * anchors.shape[0]

    # important to unroll according to pytorch
    shift_x1 = shift_x1.reshape(1, stack_size, feat_size[1])
    shift_y1 = shift_y1.reshape(1, stack_size, feat_size[1])
    shift_x2 = shift_x2.reshape(1, stack_size, feat_size[1])
    shift_y2 = shift_y2.reshape(1, stack_size, feat_size[1])
    anchor_tracker = anchor_tracker.reshape(1, stack_size, feat_size[1])

    shift_x1 = shift_x1.transpose(1, 2, 0).reshape(-1, 1)
    shift_y1 = shift_y1.transpose(1, 2, 0).reshape(-1, 1)
    shift_x2 = shift_x2.transpose(1, 2, 0).reshape(-1, 1)
    shift_y2 = shift_y2.transpose(1, 2, 0).reshape(-1, 1)
    anchor_tracker = anchor_tracker.transpose(1, 2, 0).reshape(-1, 1)

    rois = np.concatenate(
        (shift_x1, shift_y1, shift_x2, shift_y2, anchor_tracker),
        axis=1)

    locate_anchors.feat_h = feat_size[0]
    locate_anchors.feat_w = feat_size[1]
    locate_anchors.rois = rois

    return rois


locate_anchors.anchors = None
locate_anchors.feat_h = None
locate_anchors.feat_w = None


def post_process(anchors, bbox_2d, bbox_3d, rois, prob, scale_factor):
    bbox_x = bbox_2d[:, :, 0]
    bbox_y = bbox_2d[:, :, 1]
    bbox_w = bbox_2d[:, :, 2]
    bbox_h = bbox_2d[:, :, 3]

    bbox_x3d = bbox_3d[:, :, 0]
    bbox_y3d = bbox_3d[:, :, 1]
    bbox_z3d = bbox_3d[:, :, 2]
    bbox_w3d = bbox_3d[:, :, 3]
    bbox_h3d = bbox_3d[:, :, 4]
    bbox_l3d = bbox_3d[:, :, 5]
    bbox_ry3d = bbox_3d[:, :, 6]

    bbox_means = np.array(
        [[-0.00022546, 0.00160404, 0.06383215, -0.09315256, 0.01069604, -0.06744095,
          0.19155604, 0.05884239, -0.02122913, 0.06871941, -0.00352113]])
    bbox_stds = np.array(
        [[0.13962965, 0.1255247, 0.24738377, 0.23853353, 0.16330168, 0.13235298,
          3.62072376, 0.38246312, 0.10154974, 0.50257567, 1.85493732]])

    # detransform 3d
    bbox_x3d = bbox_x3d * bbox_stds[:, 4][0] + bbox_means[:, 4][0]
    bbox_y3d = bbox_y3d * bbox_stds[:, 5][0] + bbox_means[:, 5][0]
    bbox_z3d = bbox_z3d * bbox_stds[:, 6][0] + bbox_means[:, 6][0]
    bbox_w3d = bbox_w3d * bbox_stds[:, 7][0] + bbox_means[:, 7][0]
    bbox_h3d = bbox_h3d * bbox_stds[:, 8][0] + bbox_means[:, 8][0]
    bbox_l3d = bbox_l3d * bbox_stds[:, 9][0] + bbox_means[:, 9][0]
    bbox_ry3d = bbox_ry3d * bbox_stds[:, 10][0] + bbox_means[:, 10][0]

    # find 3d source
    tracker = rois[:, 4].astype(np.int64)
    src_3d = anchors[tracker, 4:]

    # compute 3d transform
    widths = rois[:, 2] - rois[:, 0] + 1.0
    heights = rois[:, 3] - rois[:, 1] + 1.0
    ctr_x = rois[:, 0] + 0.5 * widths
    ctr_y = rois[:, 1] + 0.5 * heights

    bbox_x3d = bbox_x3d[0, :] * widths + ctr_x
    bbox_y3d = bbox_y3d[0, :] * heights + ctr_y
    bbox_z3d = src_3d[:, 0] + bbox_z3d[0, :]
    bbox_w3d = np.exp(bbox_w3d[0, :]) * src_3d[:, 1]
    bbox_h3d = np.exp(bbox_h3d[0, :]) * src_3d[:, 2]
    bbox_l3d = np.exp(bbox_l3d[0, :]) * src_3d[:, 3]
    bbox_ry3d = src_3d[:, 4] + bbox_ry3d[0, :]

    # bundle
    coords_3d = np.stack(
        (bbox_x3d, bbox_y3d, bbox_z3d[:bbox_x3d.shape[0]], bbox_w3d[:bbox_x3d.shape[0]],
         bbox_h3d[:bbox_x3d.shape[0]], bbox_l3d[:bbox_x3d.shape[0]], bbox_ry3d[:bbox_x3d.shape[0]]),
        axis=1)

    # compile deltas pred
    deltas_2d = np.concatenate(
        (bbox_x[0, :, np.newaxis], bbox_y[0, :, np.newaxis],
         bbox_w[0, :, np.newaxis], bbox_h[0, :, np.newaxis]),
        axis=1)
    coords_2d = bbox_transform_inv(rois, deltas_2d, means=bbox_means[0, :], stds=bbox_stds[0, :])

    # scale coords
    coords_2d[:, 0:4] /= scale_factor
    coords_3d[:, 0:2] /= scale_factor

    prob = prob[0, :, :]
    cls_pred = np.argmax(prob[:, 1:], axis=1) + 1
    scores = np.amax(prob[:, 1:], axis=1)

    aboxes = np.hstack((coords_2d, scores[:, np.newaxis]))

    sorted_inds = (-aboxes[:, 4]).argsort()
    aboxes = aboxes[sorted_inds, :]
    coords_3d = coords_3d[sorted_inds, :]
    cls_pred = cls_pred[sorted_inds]
    tracker = tracker[sorted_inds]

    # pre-nms
    nms_topN = 3000
    cls_pred = cls_pred[0:min(nms_topN, cls_pred.shape[0])]
    tracker = tracker[0:min(nms_topN, tracker.shape[0])]
    aboxes = aboxes[0:min(nms_topN, aboxes.shape[0]), :]
    coords_3d = coords_3d[0:min(nms_topN, coords_3d.shape[0])]

    # nms
    nms_thres = 0.4
    keep_inds = nms_boxes(aboxes[:, 0:4], aboxes[:, 4], nms_thres)

    # stack cls prediction
    aboxes = np.hstack((aboxes, cls_pred[:, np.newaxis], coords_3d, tracker[:, np.newaxis]))

    # suppress boxes
    aboxes = aboxes[keep_inds, :]

    return aboxes


def predict(net, img, depth):
    h, w = img.shape[:2]
    img, depth = preprocess(img, depth)

    output = net.predict([img, depth])

    cls, prob, bbox_2d, bbox_3d = output

    anchors = locate_anchors.anchors
    feat_size = (32, 106)
    feat_stride = 16
    rois = locate_anchors(anchors, feat_size, feat_stride)

    scale_factor = img.shape[-2] / h
    aboxes = post_process(anchors, bbox_2d, bbox_3d, rois, prob, scale_factor)

    return aboxes


def recognize_from_image(net):
    depth_path = args.depth_path
    calib_path = args.calib_path

    # input image loop
    for image_path in args.input:
        logger.info(image_path)

        # prepare input data
        img = load_image(image_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)

        # depth
        name = os.path.splitext(os.path.basename(image_path))[0]
        path = get_depth(depth_path, name)
        depth = imread(path, cv2.IMREAD_UNCHANGED)
        depth = depth[:, :, np.newaxis]
        depth = np.tile(depth, (1, 1, 3))

        # read in calib
        path = get_path(calib_path, name)
        p2 = read_calib_file(path)

        # inference
        logger.info('Start inference...')
        if args.benchmark:
            logger.info('BENCHMARK mode')
            total_time_estimation = 0
            for i in range(args.benchmark_count):
                start = int(round(time.time() * 1000))
                aboxes = predict(net, img, depth)
                end = int(round(time.time() * 1000))
                estimation_time = (end - start)

                # Loggin
                logger.info(f'\tailia processing estimation time {estimation_time} ms')
                if i != 0:
                    total_time_estimation = total_time_estimation + estimation_time

            logger.info(f'\taverage time estimation {total_time_estimation / (args.benchmark_count - 1)} ms')
        else:
            aboxes = predict(net, img, depth)

        results_str, results_str_json = pred_str(aboxes, p2)

        buf = StringIO(results_str)
        anns = read_annot(buf)
        kpts_2d = get_2d(anns, p2[:3])

        fig = draw_results(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), kpts_2d)

        save_path = get_savepath(args.savepath, image_path, ext='.png')
        fig.savefig(save_path, dpi=100, bbox_inches='tight', pad_inches=0)
        plt.close()
        logger.info(f'saved at : {save_path}')

        if args.write_json:
            json_file = '%s.json' % save_path.rsplit('.', 1)[0]
            with open(json_file, 'w') as f:
                f.write(results_str_json)

    logger.info('Script finished successfully.')


def main():
    # model files check and download
    check_and_download_models(WEIGHT_PATH, MODEL_PATH, REMOTE_PATH)

    # anchor
    locate_anchors.anchors = np.load('anchor.npy')

    env_id = args.env_id

    # initialize
    net = ailia.Net(MODEL_PATH, WEIGHT_PATH, env_id=env_id, memory_mode=args.memory_mode)

    recognize_from_image(net)


if __name__ == '__main__':
    main()
