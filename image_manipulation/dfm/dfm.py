import sys
import time
import io

import numpy as np
import cv2
import matplotlib.pyplot as plt
import json

import ailia

# import original modules
sys.path.append('../../util')
from arg_utils import get_base_parser, update_parser, get_savepath  # noqa
from model_utils import check_and_download_models  # noqa
from image_utils import normalize_image  # noqa
from detector_utils import load_image  # noqa
from webcamera_utils import get_capture, get_writer  # noqa
# logger
from logging import getLogger  # noqa

logger = getLogger(__name__)

# ======================
# Parameters
# ======================

WEIGHT_PATH = 'VGG19.onnx'
MODEL_PATH = 'VGG19.onnx.prototxt'
REMOTE_PATH = 'https://storage.googleapis.com/ailia-models/dfm/'

IMAGE_A_PATH = 'img_A.png'
IMAGE_B_PATH = 'img_B.png'
SAVE_IMAGE_PATH = 'output.png'

# ======================
# Arguemnt Parser Config
# ======================

parser = get_base_parser(
    'DFM', IMAGE_A_PATH, SAVE_IMAGE_PATH
)
parser.add_argument(
    '-i2', '--input2', metavar='IMAGE2', default=IMAGE_B_PATH,
    help='Pair image path of input image.'
)
parser.add_argument(
    '-kp', '--draw-keypoints', action='store_true',
    help='Save keypoints result.'
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

def dense_feature_matching(
        map_A, map_B, ratio_th, bidirectional=True):
    # normalize and reshape feature maps
    _, ch, h_A, w_A = map_A.shape
    _, _, h_B, w_B = map_B.shape

    d1 = map_A.reshape(ch, -1).T
    d1 /= np.expand_dims(np.sqrt(np.sum(np.square(d1), axis=1)), axis=1)

    d2 = map_B.reshape(ch, -1).T
    d2 /= np.expand_dims(np.sqrt(np.sum(np.square(d2), axis=1)), axis=1)

    # perform matching
    matches, scores = mnn_ratio_matcher(d1, d2, ratio_th, bidirectional)

    # form a coordinate grid and convert matching indexes to image coordinates
    x_A, y_A, = np.meshgrid(np.arange(w_A), np.arange(h_A))
    x_B, y_B, = np.meshgrid(np.arange(w_B), np.arange(h_B))

    points_A = np.stack((x_A.flatten()[matches[:, 0]], y_A.flatten()[matches[:, 0]]))
    points_B = np.stack((x_B.flatten()[matches[:, 1]], y_B.flatten()[matches[:, 1]]))

    # discard the point on image boundaries
    discard = (points_A[0, :] == 0) | (points_A[0, :] == w_A - 1) | (points_A[1, :] == 0) | (points_A[1, :] == h_A - 1) \
              | (points_B[0, :] == 0) | (points_B[0, :] == w_B - 1) | (points_B[1, :] == 0) | (
                      points_B[1, :] == h_B - 1)

    points_A = points_A[:, ~discard]
    points_B = points_B[:, ~discard]

    return points_A, points_B


def refine_points(
        points_A, points_B,
        activations_A, activations_B,
        ratio_th=0.9, bidirectional=True):
    # normalize and reshape feature maps
    d1 = activations_A[0] / np.expand_dims(np.sqrt(np.square(activations_A[0]).sum(axis=0)), axis=0)
    d2 = activations_B[0] / np.expand_dims(np.sqrt(np.square(activations_B[0]).sum(axis=0)), axis=0)

    # get number of points
    ch = d1.shape[0]
    num_input_points = points_A.shape[1]

    if num_input_points == 0:
        return points_A, points_B

    # upsample points
    points_A *= 2
    points_B *= 2

    # neighborhood to search
    neighbors = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])

    # allocate space for scores
    scores = np.zeros((num_input_points, neighbors.shape[0], neighbors.shape[0]))

    # for each point search the refined matches in given [finer] resolution
    for i, n_A in enumerate(neighbors):
        for j, n_B in enumerate(neighbors):
            # get features in the given neighborhood
            act_A = d1[:, points_A[1, :] + n_A[1], points_A[0, :] + n_A[0]].reshape(ch, -1)
            act_B = d2[:, points_B[1, :] + n_B[1], points_B[0, :] + n_B[0]].reshape(ch, -1)

            # compute mse
            scores[:, i, j] = np.sum(act_A * act_B, axis=0)

    # retrieve top 2 nearest neighbors from A2B
    score_A = -np.sort(-scores, axis=2)[:, :, :2]
    match_A = np.argsort(-scores, axis=2)[:, :, :2]

    score_A = 2 - 2 * score_A

    # compute lowe's ratio
    ratio_A2B = score_A[:, :, 0] / (score_A[:, :, 1] + 1e-8)

    # select the best match
    match_A2B = match_A[:, :, 0]
    score_A2B = score_A[:, :, 0]

    # retrieve top 2 nearest neighbors from B2A
    x = scores.transpose(0, 2, 1)
    score_B = -np.sort(-x, axis=2)[:, :, :2]
    match_B = np.argsort(-x, axis=2)[:, :, :2]

    score_B = 2 - 2 * score_B

    # compute lowe's ratio
    ratio_B2A = score_B[:, :, 0] / (score_B[:, :, 1] + 1e-8)

    # select the best match
    match_B2A = match_B[:, :, 0]

    # check for unique matches and apply ratio test
    ind_A = (np.expand_dims(np.arange(num_input_points), axis=1) * neighbors.shape[0] + match_A2B).flatten()
    ind_B = (np.expand_dims(np.arange(num_input_points), axis=1) * neighbors.shape[0] + match_B2A).flatten()

    ind = np.arange(num_input_points * neighbors.shape[0])

    # if not bidirectional, do not use ratios from B to A
    ratio_B2A[:] *= 1 if bidirectional else 0  # discard ratio21 to get the same results with matlab

    mask = np.logical_and(
        np.where(ratio_A2B > ratio_B2A, ratio_A2B, ratio_B2A) < ratio_th,
        (ind_B[ind_A] == ind).reshape(num_input_points, -1))

    # set a large SSE score for mathces above ratio threshold and not on to one (score_A2B <=4 so use 5)
    score_A2B[~mask] = 5

    # each input point can generate max two output points, so discard the two with highest SSE
    discard = np.argsort(-score_A2B, axis=1)[:, :2]

    mask[np.arange(num_input_points), discard[:, 0]] = 0
    mask[np.arange(num_input_points), discard[:, 1]] = 0

    # x & y coordiates of candidate match points of A
    x = np.repeat(np.expand_dims(points_A[0, :], axis=0), 4, axis=0).T \
        + np.repeat(np.expand_dims(neighbors[:, 0], axis=0), num_input_points, axis=0)
    y = np.repeat(np.expand_dims(points_A[1, :], axis=0), 4, axis=0).T \
        + np.repeat(np.expand_dims(neighbors[:, 1], axis=0), num_input_points, axis=0)

    refined_points_A = np.stack((x[mask], y[mask]))

    # x & y coordiates of candidate match points of A
    x = np.repeat(np.expand_dims(points_B[0, :], axis=0), 4, axis=0).T \
        + neighbors[:, 0][match_A2B]
    y = np.repeat(np.expand_dims(points_B[1, :], axis=0), 4, axis=0).T \
        + neighbors[:, 1][match_A2B]

    refined_points_B = np.stack((x[mask], y[mask]))

    # if the number of refined matches is not enough to estimate homography,
    # but number of initial matches is enough, use initial points
    if refined_points_A.shape[1] < 4 and num_input_points > refined_points_A.shape[1]:
        refined_points_A = points_A
        refined_points_B = points_B

    return refined_points_A, refined_points_B


def mnn_ratio_matcher(
        descriptors1, descriptors2, ratio=0.8, bidirectional=True):
    # Mutual NN + symmetric Lowe's ratio test matcher for L2 normalized descriptors.
    sim = descriptors1 @ descriptors2.T

    # Retrieve top 2 nearest neighbors 1->2.
    nns_sim = -np.sort(-sim, axis=1)[:, :2]
    nns = np.argsort(-sim, axis=1)[:, :2]

    nns_dist = 2 - 2 * nns_sim
    # Compute Lowe's ratio.
    ratios12 = nns_dist[:, 0] / (nns_dist[:, 1] + 1e-8)
    # Save first NN and match similarity.
    nn12 = nns[:, 0]
    match_sim = nns_sim[:, 0]

    # Retrieve top 2 nearest neighbors 1->2.
    nns_sim = -np.sort(-sim.T, axis=1)[:, :2]
    nns = np.argsort(-sim.T, axis=1)[:, :2]
    nns_dist = 2 - 2 * nns_sim
    # Compute Lowe's ratio.
    ratios21 = nns_dist[:, 0] / (nns_dist[:, 1] + 1e-8)
    # Save first NN.
    nn21 = nns[:, 0]

    # if not bidirectional, do not use ratios from 2 to 1
    ratios21[:] *= 1 if bidirectional else 0

    # Mutual NN + symmetric ratio test.
    ids1 = np.arange(0, sim.shape[0])
    mask = (ids1 == nn21[nn12]) & \
           ((ratios12 <= ratio) &
            (ratios21[nn12] <= ratio))  # discard ratios21 to get the same results with matlab
    # Final matches.
    matches = np.stack([ids1[mask], nn12[mask]], axis=-1)
    match_sim = match_sim[mask]

    return matches, match_sim


def draw_matches(img_A, img_B, keypoints0, keypoints1):
    p1s = []
    p2s = []
    dmatches = []
    for i, (x1, y1) in enumerate(keypoints0):
        x2, y2 = keypoints1[i]
        p1s.append(cv2.KeyPoint(x1, y1, 1))
        p2s.append(cv2.KeyPoint(x2, y2, 1))
        j = len(p1s) - 1
        dmatches.append(cv2.DMatch(j, j, 1))

    matched_images = cv2.drawMatches(
        img_A, p1s,
        img_B, p2s, dmatches, None)

    x = np.concatenate([img_A, img_B], axis=1)
    res_img = np.concatenate([x, matched_images], axis=0)

    return res_img


def plot_keypoints(img, pts):
    f, a = plt.subplots()
    a.set_axis_off()
    f.subplots_adjust(left=0, right=1, bottom=0, top=1)

    a.plot(pts[0, :], pts[1, :], marker='+', linestyle='none', color='red')
    a.imshow(img)

    io_buf = io.BytesIO()
    f.savefig(io_buf, format='raw')
    io_buf.seek(0)
    data = np.frombuffer(io_buf.getvalue(), dtype=np.uint8)

    w, h = f.canvas.get_width_height()
    img = data.reshape(h, w, -1)
    img = cv2.cvtColor(img, cv2.COLOR_RGBA2BGR)

    return img


def save_result_json(json_path, keypoints0, keypoints1):
    matches = []
    for i, (x1, y1) in enumerate(keypoints0):
        x2, y2 = keypoints1[i]
        matches.append({'x1': x1, 'y1': y1, 'x2': x2, 'y2': y2})
    with open(json_path, 'w') as f:
        json.dump({'matches': matches}, f, indent=2)


# ======================
# Main functions
# ======================

def preprocess(img):
    im_h, im_w, _ = img.shape

    img = normalize_image(img, normalize_type='ImageNet')

    # zero padding to make image canvas a multiple of padding_n
    padding_n = 16
    pad_right = 16 - im_w % padding_n if im_w % padding_n else 0
    pad_bottom = 16 - im_h % padding_n if im_h % padding_n else 0

    if pad_bottom or pad_right:
        pad_img = np.zeros((im_h + pad_bottom, im_w + pad_right, 3))
        pad_img[:im_h, :im_w, :] = img
        img = pad_img

    img = img.transpose(2, 0, 1)  # HWC -> CHW
    img = np.expand_dims(img, axis=0)
    img = img.astype(np.float32)

    return img, (pad_right, pad_bottom)


def predict(net, img_A, img_B):
    img_A = img_A[:, :, ::-1]  # BGR -> RGB
    img_B = img_B[:, :, ::-1]  # BGR -> RGB

    inp_A, pad_A = preprocess(img_A)
    inp_B, pad_B = preprocess(img_B)

    # feedforward
    activations_A = net.predict([inp_A])
    activations_B = net.predict([inp_B])

    # initiate warped image, its activations, initial&final estimate of homography
    H_init = np.eye(3, dtype=np.double)
    H = np.eye(3, dtype=np.double)

    # initiate matches
    ratio_th = np.array([0.9, 0.9, 0.9, 0.9, 0.95, 1.])
    bidirectional = True
    points_A, points_B = dense_feature_matching(
        activations_A[-1], activations_B[-1],
        ratio_th[-1], bidirectional)

    # upsample points (zero based)
    points_A = (points_A + 0.5) * 16 - 0.5
    points_B = (points_B + 0.5) * 16 - 0.5

    # estimate homography for initial warping
    src = points_B.T
    dst = points_A.T

    if points_A.shape[1] >= 4:
        H_init, _ = cv2.findHomography(
            src, dst, method=cv2.RANSAC,
            ransacReprojThreshold=16 * np.sqrt(2) + 1,
            maxIters=5000, confidence=0.9999)

        # opencv might return None for H, check for None
        H_init = np.eye(3, dtype=np.double) if H_init is None else H_init

    # warp image B onto image A
    img_C = cv2.warpPerspective(img_B, H_init, (img_A.shape[1], img_A.shape[0]))
    inp_C, pad_C = preprocess(img_C)

    activations_C = net.predict([inp_C])

    # initiate matches
    points_A, points_C = dense_feature_matching(
        activations_A[-2], activations_C[-2],
        ratio_th[-2], bidirectional)

    for k in range(len(activations_A) - 3, -1, -1):
        points_A, points_C = refine_points(
            points_A, points_C, activations_A[k], activations_C[k],
            ratio_th[k], bidirectional)

    # warp points form C to B (H_init is zero-based, use zero-based points)
    points_B = np.linalg.inv(H_init) @ \
               np.vstack([
                   points_C, np.ones((1, points_C.shape[1]))
               ])
    points_B = points_B[0:2, :] / points_B[2, :]

    points_A = points_A.astype(np.double)
    points_B = points_B.astype(np.double)

    # optional
    in_image = np.logical_and(
        points_A[0, :] < (inp_A.shape[3] - pad_A[0] - 16),
        points_A[1, :] < (inp_A.shape[2] - pad_A[1] - 16))
    in_image = np.logical_and(
        in_image,
        np.logical_and(
            points_B[0, :] < (inp_B.shape[3] - pad_B[0] - 16),
            points_B[1, :] < (inp_B.shape[3] - pad_B[1] - 16)))

    points_A = points_A[:, in_image]
    points_B = points_B[:, in_image]

    # estimate homography
    src = points_B.T
    dst = points_A.T

    if points_A.shape[1] >= 4:
        H, _ = cv2.findHomography(
            src, dst, method=cv2.RANSAC,
            ransacReprojThreshold=3.0,
            maxIters=5000, confidence=0.9999)

        # opencv might return None for H, check for None
        H = np.eye(3, dtype=np.double) if H is None else H

    return H, H_init, points_A, points_B


def recognize_from_image(net):
    img_B = load_image(args.input2)
    img_B = cv2.cvtColor(img_B, cv2.COLOR_BGRA2BGR)

    # input image loop
    for image_path in args.input:
        logger.info(image_path)

        # prepare input data
        img_A = load_image(image_path)
        img_A = cv2.cvtColor(img_A, cv2.COLOR_BGRA2BGR)

        # inference
        logger.info('Start inference...')
        if args.benchmark:
            logger.info('BENCHMARK mode')
            total_time_estimation = 0
            for i in range(args.benchmark_count):
                start = int(round(time.time() * 1000))
                out = predict(net, img_A, img_B)
                end = int(round(time.time() * 1000))
                estimation_time = (end - start)

                # Logging
                logger.info(f'\tailia processing estimation time {estimation_time} ms')
                if i != 0:
                    total_time_estimation = total_time_estimation + estimation_time

            logger.info(f'\taverage time estimation {total_time_estimation / (args.benchmark_count - 1)} ms')
        else:
            out = predict(net, img_A, img_B)

        H, H_init, points_A, points_B = out

        # display results
        if args.draw_keypoints:
            plot_A = plot_keypoints(img_A, points_A)
            plot_B = plot_keypoints(img_B, points_B)
            cv2.imwrite("A.png", plot_A)
            cv2.imwrite("B.png", plot_B)

        keypoints0 = points_A.T
        keypoints1 = points_B.T
        res_img = draw_matches(img_A, img_B, keypoints0, keypoints1)

        # plot result
        savepath = get_savepath(args.savepath, image_path, ext='.png')
        logger.info(f'saved at : {savepath}')
        cv2.imwrite(savepath, res_img)

        if args.write_json:
            json_file = '%s.json' % savepath.rsplit('.', 1)[0]
            save_result_json(json_file, keypoints0, keypoints1)

    logger.info('Script finished successfully.')


def main():
    # model files check and download
    check_and_download_models(WEIGHT_PATH, MODEL_PATH, REMOTE_PATH)

    env_id = args.env_id

    # initialize
    net = ailia.Net(MODEL_PATH, WEIGHT_PATH, env_id=env_id, memory_mode=args.memory_mode)

    recognize_from_image(net)


if __name__ == '__main__':
    main()
