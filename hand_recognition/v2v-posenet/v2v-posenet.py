import sys
import time
import struct
from enum import Enum

import numpy as np
import cv2

import ailia

# import original modules
sys.path.append('../../util')
from arg_utils import get_base_parser, update_parser, get_savepath  # noqa
from model_utils import check_and_download_models  # noqa
# logger
from logging import getLogger  # noqa

from v2v_util import CUBIC_SIZE, voxelize, evaluate_keypoints

logger = getLogger(__name__)

# ======================
# Parameters
# ======================

WEIGHT_PATH = "msra-subject3-epoch15.onnx"
MODEL_PATH = "msra-subject3-epoch15.onnx.prototxt"
REMOTE_PATH = 'https://storage.googleapis.com/ailia-models/v2v-posenet/'

DEFAULT_DEPTH = 'msra_dataset/P3/1/000000_depth.bin'
SAVE_IMAGE_PATH = 'output.png'

IMG_WIDTH = 320
IMG_HEIGHT = 240
MAX_DEPTH = 700

# ======================
# Arguemnt Parser Config
# ======================

parser = get_base_parser(
    'V2V-PoseNet', DEFAULT_DEPTH, SAVE_IMAGE_PATH
)
parser.add_argument(
    '--input', '-i', default=DEFAULT_DEPTH
)
parser.add_argument(
    '--gt', '-gt', action='store_true',
    help='draw ground truth keypoints.'
)
parser.add_argument(
    '--onnx',
    action='store_true',
    help='execute onnxruntime version.'
)
args = update_parser(parser, check_input_type=False)


# ======================
# Secondaty Functions
# ======================

def load_depthmap(file_path):
    with open(file_path, mode='rb') as f:
        data = f.read()
        _, _, left, top, right, bottom = struct.unpack('I' * 6, data[:6 * 4])
        num_pixel = (right - left) * (bottom - top)
        cropped_image = struct.unpack('f' * num_pixel, data[6 * 4:])

        cropped_image = np.asarray(cropped_image).reshape(bottom - top, -1)
        depth_image = np.zeros((IMG_HEIGHT, IMG_WIDTH), dtype=np.float32)
        depth_image[top:bottom, left:right] = cropped_image
        depth_image[depth_image == 0] = MAX_DEPTH

        return depth_image


def load_center(file_path):
    a = file_path.replace('\\', '/').split('/')
    mid, fd, name = a[-3:]

    with open('msra_dataset/center_summary.csv') as f:
        a = f.readlines()
    a = [x for x in a if x.startswith('%s/%s,' % (mid, fd))]
    _, mode, b, e = a[0].strip().split(',')
    b = int(b)

    ref_pt_file = 'msra_dataset/center_%s_3_refined.txt' % mode
    with open(ref_pt_file) as f:
        ref_pt_str = [x.rstrip() for x in f]

    i = int(name.split('_')[0])
    i = b + i
    refpoint = ref_pt_str[i]
    refpoint = [float(p) for p in refpoint.split()]

    return refpoint


def get_gt_keypoints(file_path):
    a = file_path.replace('\\', '/').split('/')
    mid, fd, name = a[-3:]

    with open('msra_dataset/center_summary.csv') as f:
        a = f.readlines()
    a = [x for x in a if x.startswith('%s/%s,' % (mid, fd))]
    _, mode, b, e = a[0].strip().split(',')
    b = int(b)

    a = file_path.replace('\\', '/').rsplit('/', 1)
    a[-1] = 'joint.txt'
    annot_file = '/'.join(a)
    with open(annot_file) as f:
        lines = [line.rstrip() for line in f]

    i = int(name.split('_')[0])
    i = b + i + 1
    splitted = lines[i].split()
    joints_world = np.zeros((21, 3))
    for jid in range(21):
        joints_world[jid, 0] = float(splitted[jid * 3])
        joints_world[jid, 1] = float(splitted[jid * 3 + 1])
        joints_world[jid, 2] = -float(splitted[jid * 3 + 2])

    return joints_world


def pixel2world(x, y, z, img_width, img_height, fx, fy):
    w_x = (x - img_width / 2) * z / fx
    w_y = (img_height / 2 - y) * z / fy
    w_z = z
    return w_x, w_y, w_z


def world2pixel(x, y, z, img_width, img_height, fx, fy):
    p_x = x * fx / z + img_width / 2
    p_y = img_height / 2 - y * fy / z

    return p_x, p_y


def depthmap2points(image, fx, fy):
    h, w = image.shape
    x, y = np.meshgrid(np.arange(w) + 1, np.arange(h) + 1)
    points = np.zeros((h, w, 3), dtype=np.float32)
    points[:, :, 0], points[:, :, 1], points[:, :, 2] = pixel2world(x, y, image, w, h, fx, fy)
    return points


def points2pixels(points, img_width, img_height, fx, fy):
    pixels = np.zeros((points.shape[0], 2))
    pixels[:, 0], pixels[:, 1] = \
        world2pixel(points[:, 0], points[:, 1], points[:, 2], img_width, img_height, fx, fy)
    return pixels


def normalize_img(img, premax, com, cube):
    img[img == premax] = com[2] + (cube[2] / 2.)
    img[img == 0] = com[2] + (cube[2] / 2.)
    img[img >= com[2] + (cube[2] / 2.)] = com[2] + (cube[2] / 2.)
    img[img <= com[2] - (cube[2] / 2.)] = com[2] - (cube[2] / 2.)
    img -= com[2]
    img /= (cube[2] / 2.)

    return img


class Color(Enum):
    WHITE = (255, 255, 255)
    RED = (0, 0, 255)
    GREEN = (84, 130, 52)
    BLUE = (255, 0, 0)
    YELLOW = (17, 240, 244)
    CYAN = (255, 0, 255)


def draw_pose(img, pose):
    colors = [
        Color.WHITE, Color.GREEN, Color.GREEN, Color.GREEN, Color.WHITE, Color.BLUE, Color.BLUE, Color.BLUE,
        Color.WHITE, Color.CYAN, Color.CYAN, Color.CYAN, Color.WHITE, Color.YELLOW, Color.YELLOW, Color.YELLOW,
        Color.WHITE, Color.RED, Color.RED, Color.RED]
    colors_joint = [
        Color.WHITE, Color.GREEN, Color.GREEN, Color.GREEN, Color.GREEN,
        Color.BLUE, Color.BLUE, Color.BLUE, Color.BLUE,
        Color.CYAN, Color.CYAN, Color.CYAN, Color.CYAN,
        Color.YELLOW, Color.YELLOW, Color.YELLOW, Color.YELLOW,
        Color.RED, Color.RED, Color.RED, Color.RED]
    sketch_setting = [
        (0, 1), (1, 2), (2, 3), (3, 4), (0, 5), (5, 6), (6, 7), (7, 8),
        (0, 9), (9, 10), (10, 11), (11, 12), (0, 13), (13, 14), (14, 15), (15, 16),
        (0, 17), (17, 18), (18, 19), (19, 20)]

    for i, (x, y) in enumerate(sketch_setting):
        cv2.line(img, (int(pose[x, 0]), int(pose[x, 1])),
                 (int(pose[y, 0]), int(pose[y, 1])), colors[i].value, 1)
    for i, pt in enumerate(pose):
        radius = 5 if i in (0,) else 3 if i in (4, 8, 12, 16, 20) else 4
        cv2.circle(img, (int(pt[0]), int(pt[1])), radius, colors_joint[i].value, thickness=-1)

    return img


# ======================
# Main functions
# ======================

def predict(net, points, refpoint):
    input = voxelize(points, refpoint)
    inputs = np.expand_dims(input, axis=0)
    inputs = inputs.astype(np.float32)
    refpoint = np.array([refpoint])

    # feedforward
    if not args.onnx:
        output = net.predict([inputs])
    else:
        output = net.run(None, {
            'inputs': inputs
        })

    heatmaps = output[0]
    keypoints = evaluate_keypoints(heatmaps, refpoint)

    return keypoints[0]


def recognize_from_points(net):
    # input image loop
    for file_path in args.input:
        logger.info(file_path)

        # prepare input data
        depthmap = load_depthmap(file_path)
        try:
            refpoint = load_center(file_path)
        except (FileNotFoundError, IndexError):
            logger.error("No found reference")
            continue

        gt_keypoints = get_gt_keypoints(file_path) if args.gt else None

        fx = fy = 241.42
        points = depthmap2points(depthmap, fx, fy)
        points = points.reshape((-1, 3))
        logger.info('refpoint: %s' % refpoint)

        # inference
        logger.info('Start inference...')
        if args.benchmark:
            logger.info('BENCHMARK mode')
            total_time_estimation = 0
            for i in range(args.benchmark_count):
                start = int(round(time.time() * 1000))
                keypoints = predict(net, points, refpoint)
                end = int(round(time.time() * 1000))
                estimation_time = (end - start)

                # Logging
                logger.info(f'\tailia processing estimation time {estimation_time} ms')
                if i != 0:
                    total_time_estimation = total_time_estimation + estimation_time

            logger.info(f'\taverage time estimation {total_time_estimation / (args.benchmark_count - 1)} ms')
        else:
            keypoints = predict(net, points, refpoint)

        img = normalize_img(
            depthmap, MAX_DEPTH, refpoint,
            [CUBIC_SIZE, CUBIC_SIZE, CUBIC_SIZE])
        img = (1 - (img + 1) / 2) * 255
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)

        keypoints = gt_keypoints if gt_keypoints is not None else keypoints
        keypoints = points2pixels(keypoints, IMG_WIDTH, IMG_HEIGHT, fx, fy)
        res_img = draw_pose(img, keypoints)

        # plot result
        savepath = get_savepath(args.savepath, file_path, ext='.png')
        logger.info(f'saved at : {savepath}')
        cv2.imwrite(savepath, res_img)

    logger.info('Script finished successfully.')


def main():
    # model files check and download
    check_and_download_models(WEIGHT_PATH, MODEL_PATH, REMOTE_PATH)

    if not args.onnx:
        net = ailia.Net(MODEL_PATH, WEIGHT_PATH, env_id=args.env_id, memory_mode=args.memory_mode)
    else:
        import onnxruntime
        net = onnxruntime.InferenceSession(WEIGHT_PATH)

    recognize_from_points(net)


if __name__ == '__main__':
    main()
