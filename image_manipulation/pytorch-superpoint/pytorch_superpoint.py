import sys
import time
import io

import numpy as np
import cv2
import matplotlib.pyplot as plt

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

# for pytorch-superpoint
from pytorch_superpoint_utils import *


# ======================
# Parameters
# ======================

WEIGHT_PATH = 'SuperPointNet_gauss2.onnx'
MODEL_PATH = 'SuperPointNet_gauss2.onnx.prototxt'
REMOTE_PATH = 'https://storage.googleapis.com/ailia-models/pytorch-superpoint/'

IMAGE_A_PATH = 'img_A.png'
IMAGE_B_PATH = 'img_B.png'
SAVE_IMAGE_PATH = 'output.png'


# ======================
# Arguemnt Parser Config
# ======================

parser = get_base_parser(
    'pytorch-superpoint', IMAGE_A_PATH, SAVE_IMAGE_PATH
)
parser.add_argument(
    '-i2', '--input2', metavar='IMAGE2', default=IMAGE_B_PATH,
    help='Pair image path of input image.'
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

def preprocess(image):
    sizer = np.array([240, 320])
    s = max(sizer /image.shape[:2])
    image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    image = image[:int(sizer[0]/s),:int(sizer[1]/s)]
    image = cv2.resize(image, (sizer[1], sizer[0]), interpolation=cv2.INTER_AREA)
    image = image.astype('float32') / 255.0
    if image.ndim == 2:
        image = image[:,:, np.newaxis]
    image = np.transpose(image, (2, 0, 1))
    image = image[np.newaxis, :, :, :]
    return image


def predict(net, img_A, img_B):
    def run(net, img):
        out = net.run(img)
        out_semi, out_desc = out[0], out[1]
        heatmap = flattenDetection(out_semi, tensor=True) 
        return heatmap, out_desc
    
    pred = {}

    # image
    image = preprocess(img_A)
    heatmap, out_desc = run(net, image)
    pts_nms_batch = heatmap_to_pts(heatmap) # heatmap to pts
    pts = soft_argmax_points(heatmap, pts_nms_batch, patch_size=CONFIG['subpixel']['patch_size'])
    desc_sparse = desc_to_sparseDesc(pts_nms_batch, out_desc)
    outs = {"pts": pts[0], "desc": desc_sparse[0]}
    pts, desc = outs["pts"], outs["desc"]
    pred.update({"image": image.squeeze()})
    pred.update({"prob": pts.transpose(), "desc": desc.transpose()})

    # warped_image
    warped_image = preprocess(img_B)
    heatmap, out_desc = run(net, warped_image)
    pts_nms_batch = heatmap_to_pts(heatmap) # heatmap to pts
    pts = soft_argmax_points(heatmap, pts_nms_batch, patch_size=CONFIG['subpixel']['patch_size'])
    desc_sparse = desc_to_sparseDesc(pts_nms_batch, out_desc)
    outs = {"pts": pts[0], "desc": desc_sparse[0]}
    pts, desc = outs["pts"], outs["desc"]
    pred.update({"warped_image": warped_image.squeeze()})
    pred.update({"warped_prob": pts.transpose(), "warped_desc": desc.transpose()})

    return pred


def visualization(data):
    image = data['image']
    warped_image = data['warped_image']
    keypoints = data['prob'][:, [1, 0]]
    warped_keypoints = data['warped_prob'][:, [1, 0]]

    repeatibility = True
    if repeatibility:
        img = image
        pts = data['prob']
        img1 = draw_keypoints(img*255, pts.transpose())

        img = warped_image
        pts = data['warped_prob']
        img2 = draw_keypoints(img*255, pts.transpose())

        plot_imgs([img1.astype(np.uint8), img2.astype(np.uint8)], titles=['img1', 'img2'], dpi=200)
        plt.savefig('./output_keypoints.png', dpi=300, bbox_inches='tight')
        pass

    homography = True
    if homography:
        homography_thresh = [1,3,5,10,20,50]
        result = compute_homography(data, correctness_thresh=homography_thresh)

        H, W = image.shape
  
        assert result is not None
        matches, mscores = result['matches'], result['mscores']
        m_flip = flipArr(mscores[:])

        output = result
        img1 = image
        img2 = warped_image

        img1 = to3dim(img1)
        img2 = to3dim(img2)
        H = output['homography']
        
        ## plot warping
        warped_img1 = cv2.warpPerspective(img1, H, (img2.shape[1], img2.shape[0]))
        img1 = np.concatenate([img1, img1, img1], axis=2)
        warped_img1 = np.stack([warped_img1, warped_img1, warped_img1], axis=2)
        img2 = np.concatenate([img2, img2, img2], axis=2)
        plot_imgs([img1, img2, warped_img1], titles=['img1', 'img2', 'warped_img1'], dpi=200)
        plt.tight_layout()
        plt.savefig('output_warping_gray.png')

        ## plot filtered image
        img1, img2 = data['image'], data['warped_image']
        warped_img1 = cv2.warpPerspective(img1, H, (img2.shape[1], img2.shape[0]))
        plot_imgs([img1, img2, warped_img1], titles=['img1', 'img2', 'warped_img1'], dpi=200)
        plt.tight_layout()
        plt.savefig('output_warping_filtered.png')

        ## draw matches
        result['image1'] = image
        result['image2'] = warped_image
        matches = np.array(result['cv2_matches'])
        ratio = 0.2
        ran_idx = np.random.choice(matches.shape[0], int(matches.shape[0]*ratio))
        img = draw_matches_cv(result, matches[ran_idx], plot_points=True)
        plot_imgs([img], titles=["Two images feature correspondences"], dpi=200)
        plt.tight_layout()
        plt.savefig('output_warping_correspondence.png', bbox_inches='tight')
        plt.close('all')

        if args.write_json:
            save_result_json('output.json', result, matches[ran_idx])

    plotMatching = True
    if plotMatching:            
        matches = result['matches'] # np [N x 4]
        if matches.shape[0] > 0:
            ratio = 0.1
            inliers = result['inliers']

            matches_in = matches[inliers == True]
            matches_out = matches[inliers == False]

            image = data['image']
            warped_image = data['warped_image']

            ## outliers
            matches_temp, _ = get_random_m(matches_out, ratio)
            draw_matches(image, warped_image, matches_temp, lw=0.5, color='r',
                        filename='output_match_outliers.png', show=False, if_fig=True)

            ## inliers
            matches_temp, _ = get_random_m(matches_in, ratio)
            draw_matches(image, warped_image, matches_temp, lw=1.0, 
                        filename='output_match_inliers.png', show=False, if_fig=False)


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
        out = predict(net, img_A, img_B)

        # visualization
        visualization(out)

    logger.info('Script finished successfully.')


def main():
    # model files check and download
    check_and_download_models(WEIGHT_PATH, MODEL_PATH, REMOTE_PATH)

    env_id = args.env_id

    # initialize
    net = ailia.Net(MODEL_PATH, WEIGHT_PATH, env_id=env_id, memory_mode=args.memory_mode)

    if args.video:
        logger.error('This model not support video mode.')
    else:
        recognize_from_image(net)


if __name__ == '__main__':
    main()
