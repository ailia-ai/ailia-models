import os
import pathlib
import sys
import time

import ailia
import cv2
import numpy as np
from matplotlib import pyplot as plt
from PIL import Image

from lib_pose_hg_3d.utils.debugger import Debugger
from lib_pose_hg_3d.utils.eval import get_preds, get_preds_3d
from lib_pose_hg_3d.utils.image import get_affine_transform, transform_preds

# import original modules
sys.path.append('../../util')
# logger
from logging import getLogger  # noqa: E402

from image_utils import imread, normalize_image  # noqa: E402
from model_utils import check_and_download_models  # noqa: E402
from arg_utils import get_base_parser, get_savepath, update_parser  # noqa: E402
from webcamera_utils import (calc_adjust_fsize, get_capture,  # noqa: E402
                             get_writer, preprocess_frame)

logger = getLogger(__name__)
os.environ['KMP_DUPLICATE_LIB_OK']='TRUE'

# ======================
# Parameters
# ======================
WEIGHT_PATH = 'pose_hg_3d.onnx'
MODEL_PATH = 'pose_hg_3d.onnx.prototxt'
REMOTE_PATH = 'https://storage.googleapis.com/ailia-models/pose_hg_3d/'

IMAGE_PATH = 'input.jpg'
SAVE_IMAGE_PATH = 'output.png'
IMAGE_HEIGHT = 256
IMAGE_WIDTH = 256
CHANNELS = 3

mean = np.array([0.485, 0.456, 0.406], np.float32).reshape(1, 1, 3)
std = np.array([0.229, 0.224, 0.225], np.float32).reshape(1, 1, 3)

# ======================
# Arguemnt Parser Config
# ======================
parser = get_base_parser('pose_hg_3d model', IMAGE_PATH, SAVE_IMAGE_PATH)
parser.add_argument(
    '--skip_plot',
    action='store_true',
    help='Skip showing 3D plot of result (for CUI)'
)
args = update_parser(parser)


# ======================
# Main functions
# ======================
def recognize_from_image():
    # net initialize
    net = ailia.Net(MODEL_PATH, WEIGHT_PATH, env_id=args.env_id, memory_mode=args.memory_mode)

    # input image loop
    for image_path in args.input:
        logger.info(image_path)

        img = imread(IMAGE_PATH)
        s = max(img.shape[0], img.shape[1]) * 1.0
        c = np.array([img.shape[1] / 2., img.shape[0] / 2.], dtype=np.float32)
        trans_input = get_affine_transform(
            c, s, 0, [IMAGE_HEIGHT, IMAGE_WIDTH])
        inp = cv2.warpAffine(img, trans_input, (IMAGE_HEIGHT, IMAGE_WIDTH),
                             flags=cv2.INTER_LINEAR)
        inp = (inp / 255. - mean) / std
        inp = inp.transpose(2, 0, 1)[np.newaxis, ...].astype(np.float32)


        logger.info(f'input image shape: {inp.shape}')
        net.set_input_shape(inp.shape)

        # inference
        logger.info('Start inference...')
        if args.benchmark:
            logger.info('BENCHMARK mode')
            for i in range(5):
                start = int(round(time.time() * 1000))
                result = net.run(inp)
                end = int(round(time.time() * 1000))
                logger.info(f'\tailia processing time {end - start} ms')
        else:
            result = net.run(inp)

        savepath = get_savepath(args.savepath, image_path, ext='.png')
        logger.info(f'saved at : {savepath}')

        pred = get_preds(result[0])[0]
        pred = transform_preds(pred, c, s, (64, 64))
        pred_3d = get_preds_3d(result[0], result[1])[0]

        debugger = Debugger()
        debugger.add_img(img)
        debugger.add_point_2d(pred, (255, 0, 0))
        debugger.add_point_3d(pred_3d, 'b')
        if not args.skip_plot:
            debugger.show_all_imgs(pause=False)
            debugger.show_3d()

        savepath_3d = pathlib.PurePath(savepath)
        savepath_3d = savepath_3d.stem+"_3dpose"+savepath_3d.suffix
        logger.info(f'saved at : {savepath_3d}')
        debugger.save_3d(savepath_3d)

        logger.info(f'saved at : {savepath}')
        debugger.save_all_imgs(savepath)

    logger.info('Script finished successfully.')


def recognize_from_video():
    # net initialize
    net = ailia.Net(MODEL_PATH, WEIGHT_PATH, env_id=args.env_id, memory_mode=args.memory_mode)

    capture = get_capture(args.video)

    # create video writer if savepath is specified as video format
    if args.savepath != SAVE_IMAGE_PATH:
        logger.warning(
            'currently, video results cannot be output correctly...'
        )
        f_h = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        f_w = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        s = max(f_h, f_w) 
        writer = get_writer(args.savepath, s, s)
    else:
        writer = None

    input_shape_set = False
    debugger = Debugger()
    while(True):
        ret, frame = capture.read()
        if (cv2.waitKey(1) & 0xFF == ord('q')) or not ret:
            break

        img, data = preprocess_frame(
            frame, IMAGE_HEIGHT, IMAGE_WIDTH, normalize_type='None'
        )

        s = max(img.shape[0], img.shape[1]) * 1.0
        c = np.array([img.shape[1] / 2., img.shape[0] / 2.], dtype=np.float32)
        trans_input = get_affine_transform(
            c, s, 0, [IMAGE_HEIGHT, IMAGE_WIDTH])
        inp = cv2.warpAffine(img, trans_input, (IMAGE_HEIGHT, IMAGE_WIDTH),
                             flags=cv2.INTER_LINEAR)
        inp = (inp / 255. - mean) / std
        inp = inp.transpose(2, 0, 1)[np.newaxis, ...].astype(np.float32)

        net.set_input_shape(inp.shape)
        result = net.run(inp)

        pred = get_preds(result[0])[0]
        pred = transform_preds(pred, c, s, (64, 64))
        pred_3d = get_preds_3d(result[0], result[1])[0]

        debugger.add_img(img)
        debugger.add_point_2d(pred, (255, 0, 0))
        debugger.add_point_3d(pred_3d, 'b')
        debugger.show_all_imgs(pause=False)
        plt.pause(.01)
        debugger.ax.clear()

        if not plt.get_fignums():
            break

        if writer is not None:
            debugger.save_video(writer)

    capture.release()
    cv2.destroyAllWindows()
    if writer is not None:
        writer.release()
    logger.info('Script finished successfully.')


def main():
    # model files check and download
    check_and_download_models(WEIGHT_PATH, MODEL_PATH, REMOTE_PATH)

    if args.video is not None:
        # video mode
        recognize_from_video()
    else:
        # image mode
        recognize_from_image()


if __name__ == '__main__':
    main()
