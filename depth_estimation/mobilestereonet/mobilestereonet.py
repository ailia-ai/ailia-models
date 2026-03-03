import sys
import time

import cv2
import numpy as np

import ailia

sys.path.append('../../util')
from arg_utils import get_base_parser, update_parser, get_savepath  # noqa: E402
from model_utils import check_and_download_models  # noqa: E402
from detector_utils import load_image  # noqa: E402C
from image_utils import normalize_image  # noqa: E402C
# logger
from logging import getLogger  # noqa: E402

logger = getLogger(__name__)

# ======================
# Parameters
# ======================

WEIGHT_PATH = 'MSNet3D_SF_DS_KITTI2015.onnx'
MODEL_PATH = 'MSNet3D_SF_DS_KITTI2015.onnx.prototxt'
REMOTE_PATH = 'https://storage.googleapis.com/ailia-models/mobilestereonet/'

INPUT_PATH = 'demo'
SAVE_IMAGE_PATH = 'output.png'

IMAGE_HEIGHT = 384
IMAGE_WIDTH = 1248

# ======================
# Argument Parser Config
# ======================

parser = get_base_parser(
    'MobileStereoNet',
    INPUT_PATH,
    None,
)
parser.add_argument(
    '-i2', '--input2', default=None,
    help='The second input image path. The first input is left sight image, and second is right sight image.'
)
parser.add_argument(
    '--onnx',
    action='store_true',
    help='execute onnxruntime version.'
)
args_savepath = parser.parse_args().savepath
args = update_parser(parser)


# ======================
# Secondaty Functions
# ======================

def show_colormap(disparity, maxval=-1):
    """
    A utility function to reproduce KITTI fake colormap
    Arguments:
      - disparity: numpy float32 array of dimension HxW
      - maxval: maximum disparity value for normalization (if equal to -1, the maximum value in disparity will be used)

    Returns a numpy uint8 array of shape HxWx3.
    """

    if maxval < 0:
        maxval = np.max(disparity)

    colormap = np.asarray(
        [[0, 0, 0, 114], [0, 0, 1, 185], [1, 0, 0, 114], [1, 0, 1, 174],
         [0, 1, 0, 114], [0, 1, 1, 185], [1, 1, 0, 114], [1, 1, 1, 0]])
    weights = np.asarray([
        8.771929824561404, 5.405405405405405, 8.771929824561404, 5.747126436781609,
        8.771929824561404, 5.405405405405405, 8.771929824561404, 0])
    cumsum = np.asarray([0, 0.114, 0.299, 0.413, 0.587, 0.701, 0.8859999999999999, 0.9999999999999999])

    colored_disp = np.zeros([disparity.shape[0], disparity.shape[1], 3])
    values = np.expand_dims(np.minimum(np.maximum(disparity / maxval, 0.), 1.), -1)
    bins = np.repeat(np.repeat(np.expand_dims(np.expand_dims(cumsum, axis=0), axis=0), disparity.shape[1], axis=1),
                     disparity.shape[0], axis=0)
    diffs = np.where((np.repeat(values, 8, axis=-1) - bins) > 0, -1000, (np.repeat(values, 8, axis=-1) - bins))
    index = np.argmax(diffs, axis=-1) - 1

    w = 1 - (values[:, :, 0] - cumsum[index]) * np.asarray(weights)[index]

    colored_disp[:, :, 2] = (w * colormap[index][:, :, 0] + (1. - w) * colormap[index + 1][:, :, 0])
    colored_disp[:, :, 1] = (w * colormap[index][:, :, 1] + (1. - w) * colormap[index + 1][:, :, 1])
    colored_disp[:, :, 0] = (w * colormap[index][:, :, 2] + (1. - w) * colormap[index + 1][:, :, 2])

    return (colored_disp * np.expand_dims((disparity > 0), -1) * 255).astype(np.uint8)


# ======================
# Main functions
# ======================

def preprocess(imgs):
    new_imgs = []
    for i, img in enumerate(imgs):
        h, w = img.shape[:2]
        img = img[:, :, ::-1]  # BGR -> RGB

        # normalize
        img = normalize_image(img, normalize_type='ImageNet')

        img = img.transpose((2, 0, 1))  # HWC -> CHW

        # pad images
        top_pad = IMAGE_HEIGHT - h
        right_pad = IMAGE_WIDTH - w
        img = np.pad(img, ((0, 0), (top_pad, 0), (0, right_pad)), mode='constant', constant_values=0)

        img = np.expand_dims(img, axis=0)
        img = img.astype(np.float32)

        new_imgs.append(img)

    return new_imgs


def predict(net, left_img, right_img):
    h, w = left_img.shape[:2]

    # initial preprocesses
    imgs = (left_img, right_img)
    imgs = preprocess(imgs)

    # feedforward
    if not args.onnx:
        output = net.predict(imgs)
    else:
        output = net.run(
            None, {'left': imgs[0], 'right': imgs[1]})

    disp_ests = output[0]
    disp_ests = np.squeeze(disp_ests)
    disp_ests = disp_ests[-h:, :w]

    return disp_ests


def recognize_from_image(net):
    # Load images
    image_paths = args.input
    n_input = len(image_paths)
    if n_input == 1 and args.input2:
        image_paths.extend([args.input2])

    if len(image_paths) != 2:
        logger.error("Only two images can be specified for input.")
        sys.exit(-1)

    logger.info({k: v for k, v in zip(('left', 'right'), image_paths)})

    # prepare input data
    images = [load_image(p) for p in image_paths]
    img1, img2 = [cv2.cvtColor(im, cv2.COLOR_BGRA2BGR) for im in images]

    # inference
    logger.info('Start inference...')
    if args.benchmark:
        logger.info('BENCHMARK mode')
        total_time_estimation = 0
        for i in range(args.benchmark_count):
            start = int(round(time.time() * 1000))
            disp_est = predict(net, img1, img2)
            end = int(round(time.time() * 1000))
            estimation_time = (end - start)

            # Loggin
            logger.info(f'\tailia processing estimation time {estimation_time} ms')
            if i != 0:
                total_time_estimation = total_time_estimation + estimation_time

        logger.info(f'\taverage time estimation {total_time_estimation / (args.benchmark_count - 1)} ms')
    else:
        # inference
        disp_est = predict(net, img1, img2)

    res_img = show_colormap(disp_est)

    # save results
    savepath = get_savepath(
        args.savepath if args.savepath else SAVE_IMAGE_PATH,
        args_savepath if args_savepath != args.savepath else SAVE_IMAGE_PATH,
        post_fix='', ext='.png')
    logger.info(f'saved at : {savepath}')
    cv2.imwrite(savepath, res_img)

    logger.info('Script finished successfully.')


def main():
    # model files check and download
    check_and_download_models(WEIGHT_PATH, MODEL_PATH, REMOTE_PATH)

    # initialize
    if not args.onnx:
        memory_mode = ailia.get_memory_mode(
            reduce_constant=True, ignore_input_with_initializer=True,
            reduce_interstage=False, reuse_interstage=True)
        net = ailia.Net(MODEL_PATH, WEIGHT_PATH, env_id=args.env_id, memory_mode=memory_mode)
    else:
        import onnxruntime
        net = onnxruntime.InferenceSession(WEIGHT_PATH)

    # image mode
    recognize_from_image(net)


if __name__ == '__main__':
    main()
