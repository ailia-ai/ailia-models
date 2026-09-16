import sys
import time

import cv2
import numpy as np

import ailia
# import original modules
sys.path.append('../../util')
from arg_utils import get_base_parser, update_parser, get_savepath  # noqa: E402
from model_utils import check_and_download_models  # noqa: E402
from image_utils import load_image  # noqa: E402
import webcamera_utils  # noqa: E402

# logger
from logging import getLogger   # noqa: E402
logger = getLogger(__name__)

# for fbcnn
import os.path


# ======================
# PARAMETERS
# ======================
REMOTE_PATH = "https://storage.googleapis.com/ailia-models/fbcnn/"
COLOR_WEIGHT_PATH = "fbcnn_color.onnx"
COLOR_MODEL_PATH = "fbcnn_color.onnx.prototxt"
COLOR_REAL_WEIGHT_PATH = "fbcnn_color_real.onnx"
COLOR_REAL_MODEL_PATH = "fbcnn_color_real.onnx.prototxt"
GRAY_WEIGHT_PATH = "fbcnn_gray.onnx"
GRAY_MODEL_PATH = "fbcnn_gray.onnx.prototxt"
GRAY_DOUBLEJPEG_WEIGHT_PATH = "fbcnn_gray_doublejpeg.onnx"
GRAY_DOUBLEJPEG_MODEL_PATH = "fbcnn_gray_doublejpeg.onnx.prototxt"

IMAGE_PATH = 'input.png'
SAVE_IMAGE_PATH = 'output.png'

COLOR_REAL_H = 256
COLOR_REAL_W = 256
COLOR_H = 128
COLOR_W = 128
GRAY_H = 256
GRAY_W = 256
GRAY_DOUBLEJPEG_H = 128
GRAY_DOUBLEJPEG_W = 128
GRAY_DOUBLEJPEG_SHIFT_H = 4
GRAY_DOUBLEJPEG_SHIFT_W = 4


# ======================
# Arguemnt Parser Config
# ======================
parser = get_base_parser(
    'FBCNN is model for real JPEG image restoration.',
    IMAGE_PATH,
    SAVE_IMAGE_PATH,
)
parser.add_argument('--model', '-m',
                    default='color',
                    choices=['color', 'color_real', 'gray', 'gray_doublejpeg']
                    )
args = update_parser(parser)


# ======================
# Utils
# ======================
def imread_uint(path, n_channels=3):
    # get uint8 image of size HxWxn_channles (RGB)
    # input: path
    # output: HxWx3(RGB or GGG), or HxWx1 (G)
    if n_channels == 1:
        img = cv2.imread(path, 0)  # cv2.IMREAD_GRAYSCALE
        img = np.expand_dims(img, axis=2)  # HxWx1
    elif n_channels == 3:
        img = cv2.imread(path, cv2.IMREAD_UNCHANGED)  # BGR or G
        if img.ndim == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)  # GGG
        else:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # RGB
    return img


# ======================
# Main functionss
# ======================
def predict(img, net, n_channels, qf, qf2):
    # ---------- (1) img_L ----------
    img_L = img

    if args.model in ['color', 'gray', 'gray_doublejpeg']:
        if n_channels == 3:
            img_L = cv2.cvtColor(img_L, cv2.COLOR_RGB2BGR)
        _, encimg = cv2.imencode('.jpg', img_L, [int(cv2.IMWRITE_JPEG_QUALITY), qf])
        img_L = cv2.imdecode(encimg, 0 if (n_channels==1) else 3)

        if args.model in ['gray_doublejpeg']:
            _, encimg = cv2.imencode('.jpg', img_L[GRAY_DOUBLEJPEG_SHIFT_H:, GRAY_DOUBLEJPEG_SHIFT_W:], [int(cv2.IMWRITE_JPEG_QUALITY), qf2])
            img_L = cv2.imdecode(encimg, 0) if n_channels == 1 else cv2.imdecode(encimg, 3)

        if n_channels == 3:
            img_L = cv2.cvtColor(img_L, cv2.COLOR_BGR2RGB)

    if img_L.ndim == 2:
        img_L = np.expand_dims(img_L, axis=2)
    img_L = np.ascontiguousarray(img_L)
    img_L = img_L.transpose(2, 0, 1)
    img_L = img_L.astype(np.float32)
    img_L = img_L / 255.
    img_L = img_L[np.newaxis, :, :, :]

    # ---------- (2) img_E ----------
    img_E = net.predict(img_L)
    #img_E,QF = net.predict(img_L)
    #QF = 1 - QF

    img_E = np.squeeze(img_E, 0)
    img_E = img_E.astype(np.float32)
    if img_E.ndim == 3:
        img_E = np.transpose(img_E, (1, 2, 0))

    img_E = (img_E.clip(0, 1)*255.).round()
    img_E = np.uint8(img_E)
    img = img_E

    return img

def restore_from_video(net, n_channels, qf, qf2):
    capture = webcamera_utils.get_capture(args.video)

    # create video writer if savepath is specified as video format
    if args.savepath != SAVE_IMAGE_PATH:
        f_h = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        f_w = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        if args.model in ['gray_doublejpeg']:
            f_h -= GRAY_DOUBLEJPEG_SHIFT_H
            f_w -= GRAY_DOUBLEJPEG_SHIFT_W
        writer = webcamera_utils.get_writer(args.savepath, f_h, f_w)
    else:
        writer = None

    while(True):
        ret, img = capture.read()
        if (cv2.waitKey(1) & 0xFF == ord('q')) or not ret:
            break

        # RGB -> grayscale
        if n_channels == 1:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            img = img[:, :, np.newaxis]

        img = predict(img, net, n_channels, qf, qf2)

        if n_channels == 1:
            img = img[:, :, np.newaxis]
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)

        # save results
        if writer is not None:
            writer.write(img)

    capture.release()
    cv2.destroyAllWindows()
    if writer is not None:
        writer.release()

def restore_from_image(net, n_channels, qf, qf2):
    for image_path in args.input:
        # predict
        img_name, ext = os.path.splitext(os.path.basename(image_path))
        img = imread_uint(image_path, n_channels=n_channels)
        if args.benchmark:
            logger.info('BENCHMARK mode')
            input = img
            for _ in range(5):
                start = int(round(time.time() * 1000))
                img = predict(input, net, n_channels, qf, qf2)
                end = int(round(time.time() * 1000))
                print(f'\tailia processing time {end - start} ms')
        else:
            img = predict(img, net, n_channels, qf, qf2)
        img = np.squeeze(img)
        if img.ndim == 3:
            img = img[:, :, [2, 1, 0]]

        # save
        savepath = get_savepath(args.savepath, image_path)
        cv2.imwrite(savepath, img)

        # logger
        logger.info(image_path)

def main():
    # model files check and download
    check_and_download_models(COLOR_WEIGHT_PATH, COLOR_MODEL_PATH, REMOTE_PATH)
    check_and_download_models(COLOR_REAL_WEIGHT_PATH, COLOR_REAL_MODEL_PATH, REMOTE_PATH)
    check_and_download_models(GRAY_WEIGHT_PATH, GRAY_MODEL_PATH, REMOTE_PATH)
    check_and_download_models(GRAY_DOUBLEJPEG_WEIGHT_PATH, GRAY_DOUBLEJPEG_MODEL_PATH, REMOTE_PATH)

    # net initialize
    if args.model == 'color':
        net = ailia.Net(COLOR_MODEL_PATH, COLOR_WEIGHT_PATH, env_id=args.env_id, memory_mode=args.memory_mode)
        n_channels = 3
        #qf_list = [10, 20, 30, 40, 50, 60, 70, 80, 90]
        qf = 10
        qf2 = False
    elif args.model == 'color_real':
        net = ailia.Net(COLOR_REAL_MODEL_PATH, COLOR_REAL_WEIGHT_PATH, env_id=args.env_id, memory_mode=args.memory_mode)
        n_channels = 3
        #qf_list = [5, 10, 30, 50, 70, 90]
        qf = False
        qf2 = False
    elif args.model == 'gray':
        net = ailia.Net(GRAY_MODEL_PATH, GRAY_WEIGHT_PATH, env_id=args.env_id, memory_mode=args.memory_mode)
        n_channels = 1
        #qf_list = [10, 20, 30, 40, 50, 60, 70, 80, 90]
        qf = 10
        qf2 = False
    elif args.model == 'gray_doublejpeg':
        net = ailia.Net(GRAY_DOUBLEJPEG_MODEL_PATH, GRAY_DOUBLEJPEG_WEIGHT_PATH, env_id=args.env_id, memory_mode=args.memory_mode)
        n_channels = 1
        #qf_list = [10, 30, 50]
        qf = 10
        qf2 = 30

    # predict
    if args.video is not None: # video mode
        restore_from_video(net, n_channels, qf, qf2)
    else: # image mode
        restore_from_image(net, n_channels, qf, qf2)

    logger.info('Script finished successfully.')

if __name__ == '__main__':
    main()
