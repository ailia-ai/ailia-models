import sys
import time

import ailia
import cv2
import numpy as np
from PIL import Image

# import original modules
sys.path.append('../../util')

# logger
from logging import getLogger  # noqa: E402

import webcamera_utils  # noqa: E402
from detector_utils import load_image  # noqa: E402
from image_utils import imread  # noqa: E402
from model_utils import check_and_download_models  # noqa: E402
from arg_utils import get_base_parser, get_savepath, update_parser  # noqa: E402

logger = getLogger(__name__)

# ======================
# Parameters
# ======================
WEIGHT_PATH = 'u2net-portrait-matting.onnx'
MODEL_PATH = 'u2net-portrait-matting.onnx.prototxt'
REMOTE_PATH = 'https://storage.googleapis.com/ailia-models/u2net-portrait-matting/'

IMAGE_PATH = 'input.png'
SAVE_IMAGE_PATH = 'output.png'
IMAGE_SIZE = 448

# ======================
# Arguemnt Parser Config
# ======================

parser = get_base_parser('U^2-Net - Portrait matting', IMAGE_PATH, SAVE_IMAGE_PATH)
parser.add_argument(
    '-c', '--composite',
    action='store_true',
    help='Composite input image and predicted alpha value'
)
parser.add_argument(
    '-w', '--width',
    default=IMAGE_SIZE, type=int,
    help='The segmentation width and height for u2net.'
)
parser.add_argument(
    '-h', '--height',
    default=IMAGE_SIZE, type=int,
    help='The segmentation height and height for u2net.'
)
args = update_parser(parser)


# ======================
# Main functions
# ======================

def preprocess(img):
    mean = np.array((0.5, 0.5, 0.5))
    std = np.array((0.5, 0.5, 0.5))

    h, w = img.shape[:2]
    max_wh = max(h, w)
    hp = (max_wh - w) // 2
    vp = (max_wh - h) // 2

    h, w = img.shape[:2]

    img_pad = np.zeros((max_wh, max_wh, 3), dtype=np.uint8)
    img_pad[vp:vp + h, hp:hp + w, ...] = img
    img = img_pad

    img = np.array(Image.fromarray(img).resize(
        (args.width, args.height),
        resample=Image.LANCZOS))

    img = img / 255
    img = (img - mean) / std
    img = img.transpose(2, 0, 1)  # HWC -> CHW
    img = np.expand_dims(img, axis=0)

    return img.astype(np.float32), img_pad


def postprocess(pred, img_0, img_pad):
    h0, w0 = img_0.shape[:2]
    h1, w1 = img_pad.shape[:2]
    pred = cv2.resize(pred, (w1, h1), cv2.INTER_LINEAR)

    vp = (h1 - h0) // 2
    hp = (w1 - w0) // 2
    pred = pred[vp:vp + h0, hp:hp + w0]

    pred = np.clip(pred, 0, 1)

    return pred


def recognize_from_image(net):
    # input image loop
    for image_path in args.input:
        # prepare input data
        logger.info(image_path)

        # prepare input data
        img = img_0 = load_image(image_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2RGB)
        img, img_pad = preprocess(img)

        # inference
        logger.info('Start inference...')
        if args.benchmark:
            logger.info('BENCHMARK mode')
            for i in range(5):
                start = int(round(time.time() * 1000))
                output = net.predict({'img': img})
                end = int(round(time.time() * 1000))
                logger.info(f'\tailia processing time {end - start} ms')
        else:
            output = net.predict({'img': img})

        # postprocessing
        pred = output[0][0]
        pred = postprocess(pred, img_0, img_pad)

        if not args.composite:
            res_img = pred * 255
        else:
            # composite
            h, w = img_0.shape[:2]
            image = imread(image_path)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2BGRA)
            image[:, :, 3] = cv2.resize(pred, (w, h)) * 255
            res_img = image

        savepath = get_savepath(args.savepath, image_path, ext='.png')
        logger.info(f'saved at : {savepath}')
        cv2.imwrite(savepath, res_img)

    logger.info('Script finished successfully.')


def recognize_from_video(net):
    capture = webcamera_utils.get_capture(args.video)

    # create video writer if savepath is specified as video format
    f_h = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    f_w = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    if args.savepath != SAVE_IMAGE_PATH:
        logger.warning(
            'currently, video results cannot be output correctly...'
        )
        writer = webcamera_utils.get_writer(args.savepath, f_h, f_w)
    else:
        writer = None

    frame_shown = False
    while True:
        ret, frame = capture.read()
        if (cv2.waitKey(1) & 0xFF == ord('q')) or not ret:
            break
        if frame_shown and cv2.getWindowProperty('frame', cv2.WND_PROP_VISIBLE) == 0:
            break


        img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img, img_pad = preprocess(img)

        # inference
        output = net.predict({'img': img})

        # postprocessing
        pred = output[0][0]
        pred = postprocess(pred, frame, img_pad)

        # force composite
        frame[:, :, 0] = frame[:, :, 0] * pred + 64 * (1 - pred)
        frame[:, :, 1] = frame[:, :, 1] * pred + 177 * (1 - pred)
        frame[:, :, 2] = frame[:, :, 2] * pred

        cv2.imshow('frame', frame)
        frame_shown = True

        # save results
        if writer is not None:
            writer.write(frame.astype(np.uint8))

    capture.release()
    cv2.destroyAllWindows()
    if writer is not None:
        writer.release()

    logger.info('Script finished successfully.')


def main():
    # model files check and download
    check_and_download_models(WEIGHT_PATH, MODEL_PATH, REMOTE_PATH)

    # net initialize
    net = ailia.Net(MODEL_PATH, WEIGHT_PATH, env_id=args.env_id, memory_mode=args.memory_mode)
    net.set_input_shape((1, 3, args.height, args.width)) # dynamic axis

    if args.video is not None:
        # video mode
        recognize_from_video(net)
    else:
        # image mode
        recognize_from_image(net)


if __name__ == '__main__':
    main()
