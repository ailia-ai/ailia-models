import sys
import time

import ailia
import cv2
import numpy as np

sys.path.append('../../util')
# logger
from logging import getLogger  # noqa: E402

from image_utils import imread  # noqa: E402
from model_utils import check_and_download_models  # noqa: E402
from arg_utils import get_base_parser, get_savepath, update_parser  # noqa: E402
from webcamera_utils import get_capture, get_writer  # noqa: E402

from real_esrgan_utils import RealESRGAN
from real_esrgan_utils_v3 import RealESRGANv3

logger = getLogger(__name__)


# ======================
# Parameters
# ======================
INPUT_IMAGE_PATH = 'input.jpg'
SAVE_IMAGE_PATH = 'output.jpg'

REMOTE_PATH = 'https://storage.googleapis.com/ailia-models/real-esrgan/'

# =======================
# Arguments Parser Config
# =======================
parser = get_base_parser(
    'Real-ESRGAN',
    INPUT_IMAGE_PATH,
    SAVE_IMAGE_PATH,
)
parser.add_argument(
    '-m', '--model', metavar='MODEL_NAME',
    default='RealESRGAN',
    help='[RealESRGAN, RealESRGAN_anime, RealESRGAN_anime_v3]'
)


args = update_parser(parser)

MODEL_PATH = args.model + '.opt.onnx.prototxt'
WEIGHT_PATH = args.model + '.opt.onnx'

if args.model == "RealESRGAN_anime_v3":
    RealESRGAN = RealESRGANv3
else:
    RealESRGAN = RealESRGAN


def enhance_image():
    for image_path in args.input:
        # prepare input data
        img = imread(image_path, cv2.IMREAD_UNCHANGED)

        # net initialize
        model = ailia.Net(MODEL_PATH, WEIGHT_PATH, env_id=args.env_id, memory_mode=args.memory_mode)
        model.set_input_shape((3,img.shape[1],img.shape[0]))
        upsampler = RealESRGAN(model)

        # inference
        logger.info('Start inference...')
        if args.benchmark:
            logger.info('BENCHMARK mode')
            for i in range(5):
                start = int(round(time.time() * 1000))
                output = upsampler.enhance(img)
                end = int(round(time.time() * 1000))
                logger.info(f'\tailia processing time {end - start} ms')
        else:
            output = upsampler.enhance(img)

        savepath = get_savepath(args.savepath, image_path)
        logger.info(f'saved at : {savepath}')
        cv2.imwrite(savepath, output)

        logger.info('Script finished successfully.')


def enhance_video():
    # net initialize
    model = ailia.Net(MODEL_PATH, WEIGHT_PATH, env_id=args.env_id, memory_mode=args.memory_mode)
    upsampler = RealESRGAN(model)

    capture = get_capture(args.video)
    # create video writer if savepath is specified as video format
    if args.savepath != SAVE_IMAGE_PATH:
        logger.warning(
            'currently, video results cannot be output correctly...'
        )
        f_h = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        f_w = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        save_h, save_w = calc_adjust_fsize(f_h, f_w, IMAGE_HEIGHT, IMAGE_WIDTH)
        writer = get_writer(args.savepath, save_h, save_w * 2)
    else:
        writer = None
    
    frame_shown = False
    while (True):
        ret, frame = capture.read()
        if (cv2.waitKey(1) & 0xFF == ord('q')) or not ret:
            break
        if frame_shown and cv2.getWindowProperty('frame', cv2.WND_PROP_VISIBLE) == 0:
            break

        h, w = frame.shape[0], frame.shape[1]
        img = frame[h//2:h//2+h//4, w//2:w//2+w//4, :]

        # inference
        output = upsampler.enhance(img)

        #plot result
        cv2.imshow('frame', output)
        frame_shown = True

        if writer is not None:
            writer.release()
        logger.info('Script finished successfully.')


def main():
    # model files check and download
    check_and_download_models(WEIGHT_PATH, MODEL_PATH, REMOTE_PATH)

    if args.video:
        enhance_video()
    else:
        enhance_image()


if __name__ == '__main__':
    main()
