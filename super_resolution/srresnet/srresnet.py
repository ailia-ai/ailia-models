import sys
import time

import ailia
import cv2
import numpy as np

# import original modules
sys.path.append('../../util')
# logger
from logging import getLogger  # noqa: E402

import webcamera_utils  # noqa: E402
from image_utils import imread, load_image  # noqa: E402
from model_utils import check_and_download_models  # noqa: E402
from arg_utils import get_base_parser, get_savepath, update_parser  # noqa: E402

logger = getLogger(__name__)


# ======================
# Parameters 1
# ======================
IMAGE_PATH = 'lenna.png'
SAVE_IMAGE_PATH = 'output.png'
IMAGE_HEIGHT = 64    # net.get_input_shape()[3]
IMAGE_WIDTH = 64     # net.get_input_shape()[2]
OUTPUT_HEIGHT = 256  # net.get_output_shape()[3]
OUTPUT_WIDTH = 256   # net.get_output.shape()[2]


# ======================
# Arguemnt Parser Config
# ======================
parser = get_base_parser(
    'Single Image Super-Resolution', IMAGE_PATH, SAVE_IMAGE_PATH,
)
parser.add_argument(
    '-n', '--normal', action='store_true',
    help=('By default, the optimized model is used, but with this option, ' +
          'you can switch to the normal (not optimized) model')
)
parser.add_argument(
    '-p', '--padding', action='store_true',
    help=('Instead of resizing input image when loading it, ' +
          ' padding input and output image')
)
args = update_parser(parser)


# ======================
# Parameters 2
# ======================
if not args.normal:
    WEIGHT_PATH = 'srresnet.opt.onnx'
    MODEL_PATH = 'srresnet.opt.onnx.prototxt'
else:
    WEIGHT_PATH = 'srresnet.onnx'
    MODEL_PATH = 'srresnet.onnx.prototxt'
REMOTE_PATH = 'https://storage.googleapis.com/ailia-models/srresnet/'


# ======================
# Main functions
# ======================
def recognize_from_image():
    # net initialize
    net = ailia.Net(MODEL_PATH, WEIGHT_PATH, env_id=args.env_id, memory_mode=args.memory_mode)

    # input image loop
    for image_path in args.input:
        # prepare input data
        logger.info(image_path)
        input_data = load_image(
            image_path,
            (IMAGE_HEIGHT, IMAGE_WIDTH),
            normalize_type='255',
            gen_input_ailia=True,
        )

        # inference
        logger.info('Start inference...')
        if args.benchmark:
            logger.info('BENCHMARK mode')
            for i in range(5):
                start = int(round(time.time() * 1000))
                preds_ailia = net.predict(input_data)
                end = int(round(time.time() * 1000))
                logger.info(f'\tailia processing time {end - start} ms')
        else:
            preds_ailia = net.predict(input_data)

        # postprocessing
        output_img = preds_ailia[0].transpose((1, 2, 0))
        output_img = cv2.cvtColor(output_img, cv2.COLOR_RGB2BGR)
        savepath = get_savepath(args.savepath, image_path)
        logger.info(f'saved at : {savepath}')
        cv2.imwrite(savepath, output_img * 255)
    logger.info('Script finished successfully.')


def tiling(net, img):
    h, w = img.shape[0], img.shape[1]

    padding_w = int((w + IMAGE_WIDTH - 1) / IMAGE_WIDTH) * IMAGE_WIDTH
    padding_h = int((h+IMAGE_HEIGHT-1) / IMAGE_HEIGHT) * IMAGE_HEIGHT
    scale = int(OUTPUT_HEIGHT / IMAGE_HEIGHT)
    output_padding_w = padding_w * scale
    output_padding_h = padding_h * scale
    output_w = w * scale
    output_h = h * scale

    logger.debug(f'input image : {h}x{w}')
    logger.debug(f'output image : {output_w}x{output_h}')

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img / 255.0
    img = img.transpose(2, 0, 1)
    img = img[np.newaxis, :, :, :]

    pad_img = np.zeros((1, 3, padding_h, padding_w))
    pad_img[:, :, 0:h, 0:w] = img

    output_pad_img = np.zeros((1, 3, output_padding_h, output_padding_w))
    tile_x = int(padding_w / IMAGE_WIDTH)
    tile_y = int(padding_h / IMAGE_HEIGHT)

    # Inference
    start = int(round(time.time() * 1000))
    for y in range(tile_y):
        for x in range(tile_x):
            output_pad_img[
                :,
                :,
                y*OUTPUT_HEIGHT:(y+1)*OUTPUT_HEIGHT,
                x*OUTPUT_WIDTH:(x+1)*OUTPUT_WIDTH
            ] = net.predict(pad_img[
                :,
                :,
                y*IMAGE_HEIGHT:(y+1)*IMAGE_HEIGHT,
                x*IMAGE_WIDTH:(x+1)*IMAGE_WIDTH
            ])
    end = int(round(time.time() * 1000))
    logger.info(f'ailia processing time {end - start} ms')

    # Postprocessing
    output_img = output_pad_img[0, :, :output_h, :output_w]
    output_img = output_img.transpose(1, 2, 0).astype(np.float32)
    output_img = cv2.cvtColor(output_img, cv2.COLOR_RGB2BGR)

    return output_img


def recognize_from_image_tiling():
    # net initialize
    net = ailia.Net(MODEL_PATH, WEIGHT_PATH, env_id=args.env_id, memory_mode=args.memory_mode)

    # processing
    # input image loop
    for image_path in args.input:
        # prepare input data
        # TODO: FIXME: preprocess is different, is it intentionally...?
        logger.info(image_path)
        img = imread(image_path)
        output_img = tiling(net, img)
        savepath = get_savepath(args.savepath, image_path)
        logger.info(f'saved at : {savepath}')
        cv2.imwrite(savepath, output_img * 255)
    logger.info('Script finished successfully.')


def recognize_from_video():
    # net initialize
    net = ailia.Net(MODEL_PATH, WEIGHT_PATH, env_id=args.env_id, memory_mode=args.memory_mode)

    capture = webcamera_utils.get_capture(args.video)

    # create video writer if savepath is specified as video format
    if args.savepath != SAVE_IMAGE_PATH:
        logger.warning(
            'currently, video results cannot be output correctly...'
        )
        f_h = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        f_w = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        writer = webcamera_utils.get_writer(args.savepath, f_h, f_w)
    else:
        writer = None

    frame_shown = False
    while(True):
        ret, frame = capture.read()
        if (cv2.waitKey(1) & 0xFF == ord('q')) or not ret:
            break
        if frame_shown and cv2.getWindowProperty('frame', cv2.WND_PROP_VISIBLE) == 0:
            break

        h, w = frame.shape[0], frame.shape[1]
        frame = frame[h//2:h//2+h//4, w//2:w//2+w//4, :]

        output_img = tiling(net, frame)

        cv2.imshow('frame', output_img)
        frame_shown = True
        # # save results
        # if writer is not None:
        #     writer.write(output_img)

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
        if args.padding:
            recognize_from_image_tiling()
        else:
            recognize_from_image()


if __name__ == '__main__':
    main()
