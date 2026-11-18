import sys, os
import time
import cv2
import numpy as np

import ailia

# import original modules
sys.path.append('../../util')
from utils import get_base_parser, update_parser, get_savepath  # noqa: E402
from model_utils import check_and_download_models  # noqa: E402

import webcamera_utils  # noqa: E402

# logger
from logging import getLogger   # noqa: E402
logger = getLogger(__name__)

# import for ruas
import random
from PIL import Image


# ======================
# Parameters
# ======================
UPE_WEIGHT_PATH = "ruas_upe.onnx"
UPE_MODEL_PATH = "ruas_upe.onnx.prototxt"
LOL_WEIGHT_PATH = "ruas_lol.onnx"
LOL_MODEL_PATH = "ruas_lol.onnx.prototxt"
DARK_WEIGHT_PATH = "ruas_dark.onnx"
DARK_MODEL_PATH = "ruas_dark.onnx.prototxt"
REMOTE_PATH = 'https://storage.googleapis.com/ailia-models/ruas/'

IMAGE_PATH = 'input/0.png'
SAVE_IMAGE_PATH = 'output/output.png'

# Default input size
HEIGHT_SIZE = 1152
WIDTH_SIZE = 768

config_h = 320
config_w = 320


# ======================
# Arguemnt Parser Config
# ======================
parser = get_base_parser(
    'RUAS',
    IMAGE_PATH,
    SAVE_IMAGE_PATH,
)
parser.add_argument(
    '--model',
    default='upe',
    choices=['dark', 'lol', 'upe']
)
args = update_parser(parser)


def _load_images_transform(filename):
    img = Image.open(filename).convert('RGB')
    img = np.array(img)
    img = (img/ 255.).astype(np.float32)
    return img


def _preprocess(img):
    h = img.shape[0]
    w = img.shape[1]
    h_offset = random.randint(0, max(0, h - config_h - 1))
    w_offset = random.randint(0, max(0, w - config_w - 1))
    img = np.asarray(img, dtype=np.float32)
    img = np.transpose(img[:, :, :], (2, 0, 1))
    img = img[np.newaxis, :, :, :]
    return img


def _get_item(image_path):
    img = _load_images_transform(image_path)
    img = _preprocess(img)
    return img


def _save_images(tensor, filename):
    image_numpy = tensor[0]
    image_numpy = (np.transpose(image_numpy, (1, 2, 0)))
    im = Image.fromarray(np.clip(image_numpy * 255.0, 0, 255.0).astype('uint8'))
    path = './output/{}_{}'.format(args.model, filename)
    im.save(path, 'png')
    print('save {}'.format(path))


def recognize_from_image(net):
    print('image mode.')

    for image_path in args.input:
        logger.info(image_path)

        input = _get_item(image_path)
        output = net.run(input)
        u_list = output[0:4] #r_list = output[4:7]

        if args.model == 'lol':
            out_img = u_list[-1]
        elif args.model == 'upe' or args.model == 'dark':
            out_img = u_list[-2]

        filename = image_path.replace('input/', '')
        _save_images(out_img, filename)


def recognize_from_video(net):
    print('video mode.')

    capture = webcamera_utils.get_capture(args.video)

    # create video writer if savepath is specified as video format
    if args.savepath != SAVE_IMAGE_PATH:
        f_h = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        f_w = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        writer = webcamera_utils.get_writer(args.savepath, f_h, f_w)
    else:
        writer = None

    i = 0
    while(True):
        i += 1
        ret, img = capture.read()
        # press q to end video capture
        if (cv2.waitKey(1) & 0xFF == ord('q')) or not ret:
            break

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        input = _preprocess(img)
        input = (input/ 255.).astype(np.float32)
        output = net.run(input)
        u_list = output[0:4]
        r_list = output[4:7]

        if args.model == 'lol':
            out_img = u_list[-1]
        elif args.model == 'upe' or args.model == 'dark':
            out_img = u_list[-2]

        output = out_img[0]
        output = (np.transpose(output, (1, 2, 0)))
        output = np.clip(output * 255.0, 0, 255.0).astype('uint8')

        # save results
        if writer is not None:
            writer.write(output)

        print('processed {}th frame.'.format(i))

    capture.release()
    cv2.destroyAllWindows()
    if writer is not None:
        writer.release()
    logger.info('Script finished successfully.')


def main():
    if args.model == 'upe':
        check_and_download_models(UPE_WEIGHT_PATH, UPE_MODEL_PATH, REMOTE_PATH)
        net = ailia.Net(UPE_MODEL_PATH, UPE_WEIGHT_PATH, env_id=args.env_id)
    elif args.model == 'lol':
        check_and_download_models(LOL_WEIGHT_PATH, LOL_MODEL_PATH, REMOTE_PATH)
        net = ailia.Net(LOL_MODEL_PATH, LOL_WEIGHT_PATH, env_id=args.env_id)
    elif args.model == 'dark':
        check_and_download_models(DARK_WEIGHT_PATH, DARK_MODEL_PATH, REMOTE_PATH)
        net = ailia.Net(DARK_MODEL_PATH, DARK_WEIGHT_PATH, env_id=args.env_id)

    if args.video is not None: # video mode
        recognize_from_video(net)
    else: # image mode
        recognize_from_image(net)


if __name__ == '__main__':
    main()
