import sys
import time

import numpy as np
import cv2

import ailia

# import original modules
sys.path.append('../../util')
from arg_utils import get_base_parser, update_parser, get_savepath  # noqa: E402
from detector_utils import load_image  # noqa: E402
from image_utils import normalize_image  # noqa: E402
from model_utils import check_and_download_models  # noqa: E402
import webcamera_utils  # noqa: E402

# logger
from logging import getLogger  # noqa: E402

logger = getLogger(__name__)

# ======================
# Parameters
# ======================
WEIGHT_RES_PATH = 'pspnet-hair-segmentation.onnx' #you can also use 'pspnet_resnet101.onnx'
MODEL_RES_PATH = WEIGHT_RES_PATH + '.prototxt'
WEIGHT_SQZ_PATH = 'pspnet_squeezenet.onnx'
MODEL_SQZ_PATH = WEIGHT_SQZ_PATH + '.prototxt'
REMOTE_PATH = \
    'https://storage.googleapis.com/ailia-models/pspnet-hair-segmentation/'

IMAGE_PATH = 'test.jpg'
SAVE_IMAGE_PATH = 'output.png'
IMAGE_SIZE_RES = 512 #if you want to use 'pspnet_resnet101.onnx', please set 592
IMAGE_SIZE_SQZ = 592

# ======================
# Arguemnt Parser Config
# ======================
parser = get_base_parser(
    'Real-time hair segmentation model', IMAGE_PATH, SAVE_IMAGE_PATH
)
parser.add_argument(
    '-m', '--model', choices=('resnet101', 'squeezenet'),
    default='resnet101', metavar='NAME',
    help='name of neural network'
)
args = update_parser(parser)


# ======================
# Utils
# ======================
def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def img_resize(img, img_size):
    h, w = img.shape[:2]

    if img_size < max(h, w):
        if h < w:
            img = cv2.resize(img, (img_size, int(h * img_size / w + 0.5)), interpolation=cv2.INTER_AREA)
        else:
            img = cv2.resize(img, (int(w * img_size / h + 0.5), img_size), interpolation=cv2.INTER_AREA)

    h, w = img.shape[:2]
    y = (img_size - h) // 2
    x = (img_size - w) // 2
    pad_img = np.ones((img_size, img_size, 3), dtype=np.uint8) * 255

    pad_img[y:y + h, x:x + w, ...] = img

    return img, pad_img


def preprocess(img):
    img = img.astype(np.float32)
    img = normalize_image(img, normalize_type='ImageNet')
    img = img.transpose((2, 0, 1))  # HWC -> CHW
    img = np.expand_dims(img, axis=0)

    return img


def postprocess(src_img, pred):
    pred = sigmoid(pred)[0][0]
    mask = pred >= 0.5

    mask_n = np.zeros(src_img.shape)
    mask_n[:, :, 0] = 255
    mask_n[:, :, 0] *= mask

    image_n = cv2.cvtColor(src_img, cv2.COLOR_RGB2BGR)
    image_n = image_n * 0.5 + mask_n * 0.5
    return image_n


# ======================
# Main functions
# ======================
def recognize_from_image(net, img_size):
    # input image loop
    for image_path in args.input:
        # prepare input data
        logger.info(image_path)

        img = load_image(image_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2RGB)
        img, pad_img = img_resize(img, img_size)
        input_data = preprocess(pad_img)

        # inference
        logger.info('Start inference...')
        if args.benchmark:
            logger.info('BENCHMARK mode')
            for i in range(5):
                start = int(round(time.time() * 1000))
                pred = net.predict(input_data)
                end = int(round(time.time() * 1000))
                logger.info(f'\tailia processing time {end - start} ms')
        else:
            pred = net.predict(input_data)

        # postprocessing
        res_img = postprocess(pad_img, pred)

        h, w = img.shape[:2]
        y = (pad_img.shape[0] - h) // 2
        x = (pad_img.shape[1] - w) // 2
        res_img = res_img[y:y + h, x:x + w, ...]

        savepath = get_savepath(args.savepath, image_path)
        logger.info(f'saved at : {savepath}')
        cv2.imwrite(savepath, res_img)

    logger.info('Script finished successfully.')


def recognize_from_video(net, img_size):
    capture = webcamera_utils.get_capture(args.video)

    # create video writer if savepath is specified as video format
    if args.savepath != SAVE_IMAGE_PATH:
        logger.warning(
            'currently, video results cannot be output correctly...'
        )
        writer = webcamera_utils.get_writer(
            args.savepath, img_size, img_size
        )
    else:
        writer = None

    frame_shown = False
    while (True):
        ret, frame = capture.read()
        if (cv2.waitKey(1) & 0xFF == ord('q')) or not ret:
            break
        if frame_shown and cv2.getWindowProperty('frame', cv2.WND_PROP_VISIBLE) == 0:
            break

        # prepare input data
        img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img, pad_img = img_resize(img, img_size)
        input_data = preprocess(pad_img)

        # inference
        pred = net.predict(input_data)

        # postprocessing
        res_img = postprocess(pad_img, pred)

        h, w = img.shape[:2]
        y = (pad_img.shape[0] - h) // 2
        x = (pad_img.shape[1] - w) // 2
        res_img = res_img[y:y + h, x:x + w, ...]
        cv2.imshow('frame', res_img / 255.0)
        frame_shown = True

        # # save results
        # if writer is not None:
        #     writer.write(res_img)

    capture.release()
    cv2.destroyAllWindows()
    if writer is not None:
        writer.release()

    logger.info('Script finished successfully.')


def main():
    info = {
        'resnet101': (
            WEIGHT_RES_PATH, MODEL_RES_PATH, IMAGE_SIZE_RES),
        'squeezenet': (
            WEIGHT_SQZ_PATH, MODEL_SQZ_PATH, IMAGE_SIZE_SQZ),
    }
    weight_path, model_path, img_size = info[args.model]
    # model files check and download
    check_and_download_models(weight_path, model_path, REMOTE_PATH)

    # net initialize
    net = ailia.Net(model_path, weight_path, env_id=args.env_id, memory_mode=args.memory_mode)

    if args.video is not None:
        # video mode
        recognize_from_video(net, img_size)
    else:
        # image mode
        recognize_from_image(net, img_size)


if __name__ == '__main__':
    main()
