import sys
import time

import cv2
from PIL import Image
from skimage import color
import numpy as np
import matplotlib.pyplot as plt

import ailia

# import original modules
sys.path.append('../../util')
from arg_utils import get_base_parser, update_parser, get_savepath  # noqa: E402
from model_utils import check_and_download_models  # noqa: E402
import webcamera_utils  # noqa: E402

# logger
from logging import getLogger   # noqa: E402
logger = getLogger(__name__)

# for cnngeometric_pytorch
from skimage import io
from cnngeometric_pytorch_utils import homography_mat_from_4_pts, compose_H_matrices, compose_aff_matrices, compose_tps, GeometricTnf


# ======================
# Parameters
# ======================
WEIGHT_STREETVIEW_AFFINE_PATH = 'streetview_affine.onnx'
MODEL_STREETVIEW_AFFINE_PATH = 'streetview_affine.onnx.prototxt'
WEIGHT_STREETVIEW_HOM_PATH = 'streetview_hom.onnx'
MODEL_STREETVIEW_HOM_PATH = 'streetview_hom.onnx.prototxt'
WEIGHT_STREETVIEW_TPS_PATH = 'streetview_tps.onnx'
MODEL_STREETVIEW_TPS_PATH = 'streetview_tps.onnx.prototxt'
REMOTE_PATH = 'https://storage.googleapis.com/ailia-models/cnngeometric_pytorch/'

SOURCE_IMAGE_PATH = 'input_source.png'
TARGET_IMAGE_PATH = 'input_target.png'
SAVE_IMAGE_PATH = 'output.png'

IMAGE_WIDTH = 240
IMAGE_HEIGHT = 240


# ======================
# Arguemnt Parser Config
# ======================
parser = get_base_parser(
    'CNNGeometric PyTorch', SOURCE_IMAGE_PATH, SAVE_IMAGE_PATH, fp16_support=False
)
parser.add_argument(
    '-m', '--model_1', default='streetview_affine', choices=('streetview_affine', 'streetview_hom', 'streetview_tps'),
    help='type of 1st model'
)
parser.add_argument(
    '--model_2', choices=('streetview_affine', 'streetview_hom', 'streetview_tps'),
    help='type of 2nd model'
)
parser.add_argument(
    '--input_tgt', help='target image path'
)
parser.add_argument(
    '--video_tgt', help='target video path'
)
parser.add_argument(
    '--num_of_iters', type=int, default=3, help='number of stages to use recursively'
)
args = update_parser(parser)


# ======================
# Utils
# ======================
def preprocess_img(image):
    im_size = np.asarray(image.shape)
    image = np.expand_dims(image.transpose((2,0,1)),0)
    image = image.astype(np.float32)

    # Resize image using bilinear sampling with identity affine tnf
    affineTnf = GeometricTnf(geometric_model='affine', out_h=240, out_w=240)
    image = affineTnf(image)

    image /= 255.0
    image = image[0, :, :, :]
    mean, std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
    for i in range(3):
        image[i] = (image[i]-mean[i])/std[i]
    image = image[np.newaxis, :, :, :]

    return image


# ======================
# Main functions
# ======================
def inference_one_pair(source_image, target_image, net_1, net_2):
    if args.model_1 == 'streetview_affine':
        geoTnf = GeometricTnf(geometric_model='affine')
    elif args.model_1 == 'streetview_hom':
        geoTnf = GeometricTnf(geometric_model='hom')
    elif args.model_1 == 'streetview_tps':
        geoTnf = GeometricTnf(geometric_model='tps')

    # eval model multistage
    for it in range(args.num_of_iters):
        # First iteration
        if it==0:
            theta = net_1.predict([source_image, target_image])[0]
            if args.model_1=='streetview_hom':
                theta = homography_mat_from_4_pts(theta)
            continue

        # Compute warped image
        warped_image = None
        warped_image = geoTnf(source_image.astype(np.float32), theta)

        # Re-estimate tranformation
        theta_iter = net_1.predict([warped_image, target_image])[0]

        # update accumultated transformation
        if args.model_1 == 'streetview_hom':
            theta = compose_H_matrices(theta, homography_mat_from_4_pts(theta_iter))
        elif args.model_1 == 'streetview_affine':
            theta = compose_aff_matrices(theta, theta_iter)
        elif args.model_1 == 'streetview_tps':
            theta = compose_tps(theta, theta_iter)

    # warp one last time using final transformation
    warped_image = geoTnf(source_image, theta)

    # two stages
    if net_2 is not None:
        if args.model_2 == 'streetview_affine':
            geoTnf_2 = GeometricTnf(geometric_model='affine')
        elif args.model_2 == 'streetview_hom':
            geoTnf_2 = GeometricTnf(geometric_model='hom')
        elif args.model_2 == 'streetview_tps':
            geoTnf_2 = GeometricTnf(geometric_model='tps')

        theta_1, warped_image_1 = theta, warped_image
        theta_2 = net_2.predict([warped_image_1, target_image])[0]

        if args.model_2 == 'streetview_hom':
            theta_2 = homography_mat_from_4_pts(theta_2)

        warped_image = geoTnf_2(warped_image_1, theta_2)

    warped_image = warped_image[0]

    return warped_image


def recognize_from_image(net_1, net_2):
    # input image loop
    for image_path in args.input:
        source_image = preprocess_img(io.imread(image_path))
        if args.input_tgt is not None:
            target_image = preprocess_img(io.imread(args.input_tgt))
        else:
            target_image = preprocess_img(io.imread(TARGET_IMAGE_PATH))

        # inference
        output_image = inference_one_pair(source_image, target_image, net_1, net_2)
        output_image = output_image.transpose(1, 2, 0)

        # save
        io.imsave(args.savepath, output_image)
    logger.info('Script finished successfully.')


def recognize_from_video(net_1, net_2):
    if args.video_tgt is None:
        logger.info('--video_tgt option is required if you want to use video mode.')
        exit()

    src_capture = webcamera_utils.get_capture(args.video)
    tgt_capture = webcamera_utils.get_capture(args.video_tgt)

    # create video writer if savepath is specified as video format
    if args.savepath != SAVE_IMAGE_PATH:
        writer = webcamera_utils.get_writer(args.savepath, IMAGE_HEIGHT, IMAGE_WIDTH)
    else:
        writer = None

    frame_shown = False
    while(True):
        src_ret, src_img = src_capture.read()
        tgt_ret, tgt_img = tgt_capture.read()
        # press q to end video capture
        if (cv2.waitKey(1) & 0xFF == ord('q')) or not src_ret or not tgt_ret:
            break
        if frame_shown and cv2.getWindowProperty('frame', cv2.WND_PROP_VISIBLE) == 0:
            break
        src_img = cv2.cvtColor(src_img, cv2.COLOR_BGR2RGB)
        tgt_img = cv2.cvtColor(tgt_img, cv2.COLOR_BGR2RGB)

        src_img = preprocess_img(src_img)
        tgt_img = preprocess_img(tgt_img)

        # inference
        output_image = inference_one_pair(src_img, tgt_img, net_1, net_2)

        output_image = output_image.transpose(1, 2, 0)

        mean, std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
        output_image = output_image * std
        output_image = output_image + mean

        output_image = np.array(output_image * 255, dtype=np.uint8)
        output_image = cv2.cvtColor(output_image, cv2.COLOR_RGB2BGR)
        cv2.imshow('frame', output_image)
        frame_shown = True

        # save results
        if writer is not None:
            writer.write(output_image)

    src_capture.release()
    tgt_capture.release()
    cv2.destroyAllWindows()
    if writer is not None:
        writer.release()
    logger.info('Script finished successfully.')


def main():
    # model files check and download
    check_and_download_models(WEIGHT_STREETVIEW_AFFINE_PATH, MODEL_STREETVIEW_AFFINE_PATH, REMOTE_PATH)
    check_and_download_models(WEIGHT_STREETVIEW_HOM_PATH, MODEL_STREETVIEW_HOM_PATH, REMOTE_PATH)
    check_and_download_models(WEIGHT_STREETVIEW_TPS_PATH, MODEL_STREETVIEW_TPS_PATH, REMOTE_PATH)

    # net initialize
    if args.model_1 == 'streetview_affine':
        net_1 = ailia.Net(MODEL_STREETVIEW_AFFINE_PATH, WEIGHT_STREETVIEW_AFFINE_PATH, env_id=args.env_id, memory_mode=args.memory_mode)
    elif args.model_1 == 'streetview_hom':
        net_1 = ailia.Net(MODEL_STREETVIEW_HOM_PATH, WEIGHT_STREETVIEW_HOM_PATH, env_id=args.env_id, memory_mode=args.memory_mode)
    elif args.model_1 == 'streetview_tps':
        net_1 = ailia.Net(MODEL_STREETVIEW_TPS_PATH, WEIGHT_STREETVIEW_TPS_PATH, env_id=args.env_id, memory_mode=args.memory_mode)
    net_1.set_input_shape((1, 1, IMAGE_HEIGHT, IMAGE_WIDTH))

    # net_2 initialize
    net_2 = None
    if args.model_2 == 'streetview_affine':
        net_2 = ailia.Net(MODEL_STREETVIEW_AFFINE_PATH, WEIGHT_STREETVIEW_AFFINE_PATH, env_id=args.env_id, memory_mode=args.memory_mode)
    elif args.model_2 == 'streetview_hom':
        net_2 = ailia.Net(MODEL_STREETVIEW_HOM_PATH, WEIGHT_STREETVIEW_HOM_PATH, env_id=args.env_id, memory_mode=args.memory_mode)
    elif args.model_2 == 'streetview_tps':
        net_2 = ailia.Net(MODEL_STREETVIEW_TPS_PATH, WEIGHT_STREETVIEW_TPS_PATH, env_id=args.env_id, memory_mode=args.memory_mode)
    if args.model_2 is not None:
        net_2.set_input_shape((1, 1, IMAGE_HEIGHT, IMAGE_WIDTH))

    # inference
    if args.video is not None:
        # video mode
        recognize_from_video(net_1, net_2)
    else:
        # image mode
        recognize_from_image(net_1, net_2)


if __name__ == '__main__':
    main()
