import sys
import time
import io

import numpy as np
import cv2
import json

import ailia

# import original modules
sys.path.append('../../util')
from arg_utils import get_base_parser, update_parser, get_savepath  # noqa
from model_utils import check_and_download_models  # noqa
from image_utils import normalize_image  # noqa
from webcamera_utils import get_capture, get_writer  # noqa
# logger
from logging import getLogger  # noqa

logger = getLogger(__name__)


from scipy.special import softmax
from dfe_utils import load_image, get_sift_features, do_matching, \
estimate_for_model, robust_symmetric_epipolar_distance, rescale_and_expand


# ======================
# Parameters
# ======================

WEIGHT_INIT_PATH = 'WeightEstimatorNet_init.onnx'
MODEL_INIT_PATH = 'WeightEstimatorNet_init.onnx.prototxt'
WEIGHT_ITER_PATH = 'WeightEstimatorNet_iter.onnx'
MODEL_ITER_PATH = 'WeightEstimatorNet_iter.onnx.prototxt'
REMOTE_PATH = 'https://storage.googleapis.com/ailia-models/dfe/'

IMAGE_A_PATH = 'img_A.png'
IMAGE_B_PATH = 'img_B.png' # base
SAVE_IMAGE_PATH = 'output.png'


# ======================
# Arguemnt Parser Config
# ======================

parser = get_base_parser(
    'DFE', IMAGE_B_PATH, SAVE_IMAGE_PATH
)
parser.add_argument(
    '-i2', '--input2', metavar='IMAGE2', default=IMAGE_A_PATH,
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
# Main functions
# ======================

def draw_epipolar(img1, img2, F, x1, x2):
    imgCAM1 = img1
    imgCAM2 = img2

    ptsCAM1 = x1[0]
    ptsCAM2 = x2[0]
    ptsCAM1 = ptsCAM1.astype(int)
    ptsCAM2 = ptsCAM2.astype(int)

    #F, mask = cv2.findFundamentalMat(ptsCAM1, ptsCAM2, cv2.FM_LMEDS)

    # draw points
    for points in ptsCAM1:
        imgCAM1 = cv2.circle(imgCAM1, tuple(points), 5, (0, 0, 255), -1)

    # calc epipolar lines
    linesCAM1 = cv2.computeCorrespondEpilines(ptsCAM2, 2, F)
    linesCAM1 = linesCAM1.reshape(-1,3) #行列の変形

    # draw epipolar lines
    widthCAM1 = imgCAM1.shape[1] #画像幅
    for lines in linesCAM1:
        x0,y0 = map(int, [0,-lines[2]/lines[1]]) #左端
        x1,y1 = map(int, [widthCAM1,-(lines[2]+lines[0]*widthCAM1)/lines[1]]) #右端
        imgCAM1 = cv2.line(imgCAM1, (x0,y0), (x1,y1), (255, 255, 255), 1) #線の描画
    save1 = imgCAM1

    # draw points
    for points in ptsCAM2:
        imgCAM2 = cv2.circle(imgCAM2, tuple(points), 5, (0, 0, 255), -1)

    # calc epipolar lines
    linesCAM2 = cv2.computeCorrespondEpilines(ptsCAM1, 1, F)
    linesCAM2 = linesCAM2.reshape(-1,3) #行列の変形

    # draw epipolar lines
    widthCAM2 = imgCAM2.shape[1] #画像幅

    for lines in linesCAM2:
        x0,y0 = map(int, [0,-lines[2]/lines[1]]) #左端
        x1,y1 = map(int, [widthCAM2,-(lines[2]+lines[0]*widthCAM2)/lines[1]]) #右端
        imgCAM2 = cv2.line(imgCAM2, (x0,y0), (x1,y1), (255, 255, 255), 1) #線の描画
    save2 = imgCAM2

    return save1, save2


def save_result_json(json_file, x1, x2):
    with open(json_file, 'w') as f:
        json.dump({
            'ptsCAM1': x1[0].tolist(),
            'ptsCAM2': x2[0].tolist()
        }, f, indent=2)


def predict(pts, net_init, net_iter):
    """
    Args:
        pts (tensor): point correspondences
        side_info (tensor): side information

    Returns:
        tensor: fundamental matrix, transformation of points in first and second image
    """
    depth = 3
    side_info = np.ones((1, 1000, 3))*-1 # side_info

    pts = pts.transpose((0, 2, 1))
    pts1, pts2, rescaling_1, rescaling_2 = rescale_and_expand(pts)

    pts1 = pts1.transpose((0, 2, 1))
    pts2 = pts2.transpose((0, 2, 1))

    # init weights
    input_p_s = np.concatenate([(pts1[:,:,:2]+1)/2, (pts2[:,:,:2]+1)/2, side_info], 2).transpose(0, 2, 1)
    out_init = net_init.predict(input_p_s)
    weights = softmax(out_init, axis=2)

    out_depth = estimate_for_model(pts1, pts2, weights)
    out = [out_depth]

    # iter weights
    for _ in range(1, depth):
        residual = robust_symmetric_epipolar_distance(pts1, pts2, out_depth)

        input_p_s_w_r = np.concatenate((input_p_s, weights, residual), 1)

        out_iter = net_iter.predict(input_p_s_w_r)
        weights = softmax(out_iter, axis=2)

        out_depth = estimate_for_model(pts1, pts2, weights)
        out.append(out_depth)

    F_est = out
    F_est = rescaling_1.transpose(0, 2, 1) @ (F_est[-1] @ rescaling_2)
    F_est = F_est / F_est[:, -1, -1][:, np.newaxis, np.newaxis]
    F_out = F_est[0]

    return F_out, pts1, pts2

      
def recognize_from_image():
    env_id = args.env_id
    
    # initialize
    net_init = ailia.Net(MODEL_INIT_PATH, WEIGHT_INIT_PATH, env_id=env_id, memory_mode=args.memory_mode)
    net_iter = ailia.Net(MODEL_ITER_PATH, WEIGHT_ITER_PATH, env_id=env_id, memory_mode=args.memory_mode)

    img_A, zoom_xy_A, img_ori_A = load_image(args.input2) # base image
    sift_kp_A, sift_des_A = get_sift_features(img_ori_A, zoom_xy_A)

    # input image loop
    for i, image_path in enumerate(args.input):
        logger.info(image_path)

        # get image
        img_B, zoom_xy_B, img_ori_B = load_image(image_path)
        sift_kp_B, sift_des_B = get_sift_features(img_ori_B, zoom_xy_B)

        # get input
        matches_use_ori, weight_in, pts1, pts2, T1, T2, x1, x2 = do_matching(
            cv2.BFMatcher(normType=cv2.NORM_L2), 
            sift_des_A.copy(), sift_des_B.copy(),
            sift_kp_A.copy(), sift_kp_B.copy()
        )

        # inference
        F_out, pts1, pts2 = predict(matches_use_ori, net_init, net_iter)

        # visualize
        out_img1, out_img2 = draw_epipolar(img_A, img_B, F_out, x1, x2)

        # save
        cv2.imwrite('out_A_B{}.png'.format(i), out_img1)
        cv2.imwrite('out_B{}_A.png'.format(i), out_img2)

        if args.write_json:
            save_result_json('output.json', x1, x2)

    logger.info('Script finished successfully.')


def main():
    # model files check and download
    check_and_download_models(WEIGHT_INIT_PATH, MODEL_INIT_PATH, REMOTE_PATH)
    check_and_download_models(WEIGHT_ITER_PATH, MODEL_ITER_PATH, REMOTE_PATH)

    # inference
    recognize_from_image()


if __name__ == '__main__':
    main()
