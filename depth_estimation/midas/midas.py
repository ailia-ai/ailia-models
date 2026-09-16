import os
import sys
import time

import ailia
import cv2
import numpy as np

# import original modules
sys.path.append('../../util')
# logger
from logging import getLogger  # noqa: E402

from image_utils import imread, normalize_image  # noqa: E402
from model_utils import check_and_download_models  # noqa: E402
from arg_utils import get_base_parser, get_savepath, update_parser  # noqa: E402
from webcamera_utils import calc_adjust_fsize  # noqa: E402
from webcamera_utils import get_capture, get_writer

logger = getLogger(__name__)


# ======================
# Parameters
# ======================
WEIGHT_v20_PATH = 'midas.onnx'
MODEL_v20_PATH = 'midas.onnx.prototxt'
WEIGHT_v21_PATH = 'midas_v2.1.onnx'
MODEL_v21_PATH = 'midas_v2.1.onnx.prototxt'
WEIGHT_v21_SMALL_PATH = 'midas_v2.1_small.onnx'
MODEL_v21_SMALL_PATH = 'midas_v2.1_small.onnx.prototxt'
REMOTE_PATH = 'https://storage.googleapis.com/ailia-models/midas/'

IMAGE_PATH = 'input.jpg'
SAVE_IMAGE_PATH = 'input_depth.png'
IMAGE_HEIGHT = 384
IMAGE_WIDTH = 384
IMAGE_HEIGHT_SMALL = 256
IMAGE_WIDTH_SMALL = 256
IMAGE_MULTIPLE_OF = 32


# ======================
# Arguemnt Parser Config
# ======================
parser = get_base_parser('MiDaS model', IMAGE_PATH, SAVE_IMAGE_PATH)
parser.add_argument(
    '-v21', '--version21', dest='v21', action='store_true',
    help='Use model version 2.1.'
)
parser.add_argument(
    '-t', '--model_type', default='large', choices=('large', 'small'),
    help='model type: large or small. small can be specified only for version 2.1 model.'
)
args = update_parser(parser)


# ======================
# Main functions
# ======================
def constrain_to_multiple_of(x, min_val=0, max_val=None):
    y = (np.round(x / IMAGE_MULTIPLE_OF) * IMAGE_MULTIPLE_OF).astype(int)
    if max_val is not None and y > max_val:
        y = (np.floor(x / IMAGE_MULTIPLE_OF) * IMAGE_MULTIPLE_OF).astype(int)
    if y < min_val:
        y = (np.ceil(x / IMAGE_MULTIPLE_OF) * IMAGE_MULTIPLE_OF).astype(int)
    return y


def midas_resize(image, target_height, target_width):
    # Resize while keep aspect ratio.
    h, w, _ = image.shape
    scale_height = target_height / h
    scale_width = target_width / w
    if scale_width < scale_height:
        scale_height = scale_width
    else:
        scale_width = scale_height
    new_height = constrain_to_multiple_of(
        scale_height * h, max_val=target_height
    )
    new_width = constrain_to_multiple_of(
        scale_width * w, max_val=target_width
    )

    return cv2.resize(
        image, (new_width, new_height), interpolation=cv2.INTER_CUBIC
    )


def midas_imread(image_path):
    if not os.path.isfile(image_path):
        logger.error(f'{image_path} not found.')
        sys.exit()
    image = imread(image_path)
    if image.ndim == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = normalize_image(image, 'ImageNet')

    h, w = (IMAGE_HEIGHT, IMAGE_WIDTH) if not args.v21 or args.model_type == 'large' \
               else (IMAGE_HEIGHT_SMALL, IMAGE_WIDTH_SMALL)
    return midas_resize(image, h, w)


def recognize_from_image(net):
    # input image loop
    for image_path in args.input:
        logger.info(image_path)

        # prepare input data
        img = midas_imread(image_path)
        img = img.transpose((2, 0, 1))  # channel first
        img = img[np.newaxis, :, :, :]

        logger.debug(f'input image shape: {img.shape}')
        net.set_input_shape(img.shape)

        # inference
        logger.info('Start inference...')
        if args.benchmark:
            logger.info('BENCHMARK mode')
            total_time = 0
            for i in range(args.benchmark_count):
                start = int(round(time.time() * 1000))
                result = net.predict(img)
                end = int(round(time.time() * 1000))
                logger.info(f'\tailia processing time {end - start} ms')
                if i != 0:
                    total_time = total_time + (end - start)
            logger.info(f'\taverage time {total_time / (args.benchmark_count-1)} ms')
        else:
            result = net.predict(img)

        depth_min = result.min()
        depth_max = result.max()
        max_val = (2 ** 16) - 1

        if depth_max - depth_min > np.finfo("float").eps:
            out = max_val * (result - depth_min) / (depth_max - depth_min)
        else:
            out = 0

        savepath = get_savepath(args.savepath, image_path, ext='.png')
        logger.info(f'saved at : {savepath}')
        cv2.imwrite(savepath, out.transpose(1, 2, 0).astype("uint16"))
    logger.info('Script finished successfully.')


def recognize_from_video(net):
    capture = get_capture(args.video)

    # allocate output buffer
    f_h = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    f_w = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))

    h, w = (IMAGE_HEIGHT, IMAGE_WIDTH) if not args.v21 or args.model_type == 'large' \
               else (IMAGE_HEIGHT_SMALL, IMAGE_WIDTH_SMALL)

    zero_frame = np.zeros((f_h,f_w,3))
    resized_img = midas_resize(zero_frame, h, w)
    save_h, save_w = resized_img.shape[0], resized_img.shape[1]

    output_frame = np.zeros((save_h,save_w*2,3))

    # create video writer if savepath is specified as video format
    if args.savepath != SAVE_IMAGE_PATH:
        logger.warning(
            'currently, video results cannot be output correctly...'
        )
        writer = get_writer(args.savepath, save_h, save_w * 2)
    else:
        writer = None

    input_shape_set = False
    frame_shown = False
    while(True):
        ret, frame = capture.read()
        if (cv2.waitKey(1) & 0xFF == ord('q')) or not ret:
            break
        if frame_shown and cv2.getWindowProperty('depth', cv2.WND_PROP_VISIBLE) == 0:
            break

        # resize to midas input size
        frame = midas_resize(frame, h, w)
        resized_img = normalize_image(frame, 'ImageNet')
        resized_img = resized_img.transpose((2, 0, 1))  # channel first
        resized_img = resized_img[np.newaxis, :, :, :]

        # predict
        if(not input_shape_set):
            net.set_input_shape(resized_img.shape)
            input_shape_set = True
        result = net.predict(resized_img)

        # normalize to 16bit
        depth_min = result.min()
        depth_max = result.max()
        max_val = (2 ** 16) - 1
        if depth_max - depth_min > np.finfo("float").eps:
            out = max_val * (result - depth_min) / (depth_max - depth_min)
        else:
            out = 0

        # convert to 8bit
        res_img = (out.transpose(1, 2, 0)/256).astype("uint8")
        res_img = cv2.cvtColor(res_img, cv2.COLOR_GRAY2BGR)

        output_frame[:,save_w:save_w*2,:]=res_img
        output_frame[:,0:save_w,:]=frame
        output_frame = output_frame.astype("uint8")

        cv2.imshow('depth', output_frame)
        frame_shown = True

        # save results
        if writer is not None:
            writer.write(output_frame)

    capture.release()
    cv2.destroyAllWindows()
    if writer is not None:
        writer.release()
    logger.info('Script finished successfully.')


def main():
    weight_path = (WEIGHT_v21_PATH if args.model_type == 'large' else WEIGHT_v21_SMALL_PATH) \
        if args.v21 else WEIGHT_v20_PATH
    model_path = (MODEL_v21_PATH if args.model_type == 'large' else MODEL_v21_SMALL_PATH) \
        if args.v21 else MODEL_v20_PATH

    # model files check and download
    check_and_download_models(weight_path, model_path, REMOTE_PATH)

    # net initialize
    net = ailia.Net(model_path, weight_path, env_id=args.env_id, memory_mode=args.memory_mode)

    if args.video is not None:
        # video mode
        recognize_from_video(net)
    else:
        # image mode
        recognize_from_image(net)


if __name__ == '__main__':
    main()
