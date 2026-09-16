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
from image_utils import imread  # noqa: E402
from model_utils import check_and_download_models  # noqa: E402
from arg_utils import get_base_parser, get_savepath, update_parser  # noqa: E402

from u2net_utils import load_image, norm, save_result, transform  # noqa: E402

logger = getLogger(__name__)


# ======================
# Parameters
# ======================
IMAGE_PATH = 'input.png'
SAVE_IMAGE_PATH = 'output.png'
IMAGE_SIZE = 320
MODEL_LISTS = ['small', 'large']
OPSET_LISTS = ['10', '11']


# ======================
# Arguemnt Parser Config
# ======================
parser = get_base_parser('U square net', IMAGE_PATH, SAVE_IMAGE_PATH)
parser.add_argument(
    '-a', '--arch', metavar='ARCH',
    default='large', choices=MODEL_LISTS,
    help='model lists: ' + ' | '.join(MODEL_LISTS)
)
parser.add_argument(
    '-c', '--composite',
    action='store_true',
    help='Composite input image and predicted alpha value'
)
parser.add_argument(
    '-o', '--opset', metavar='OPSET',
    default='11', choices=OPSET_LISTS,
    help='opset lists: ' + ' | '.join(OPSET_LISTS)
)
parser.add_argument(
    '-w', '--width',
    default=IMAGE_SIZE, type=int,
    help='The segmentation width and height for u2net. (default: 320)'
)
parser.add_argument(
    '-h', '--height',
    default=IMAGE_SIZE, type=int,
    help='The segmentation height and height for u2net. (default: 320)'
)
parser.add_argument(
    '--rgb',
    action='store_true',
    help='Use rgb color space (default: bgr)'
)
args = update_parser(parser)


# ======================
# Parameters 2
# ======================
if args.opset == "10":
    WEIGHT_PATH = 'u2net.onnx' if args.arch == 'large' else 'u2netp.onnx'
else:
    WEIGHT_PATH = 'u2net_opset11.onnx' \
        if args.arch == 'large' else 'u2netp_opset11.onnx'
MODEL_PATH = WEIGHT_PATH + '.prototxt'
REMOTE_PATH = 'https://storage.googleapis.com/ailia-models/u2net/'


# ======================
# Main functions
# ======================
def recognize_from_image(net):
    # input image loop
    for image_path in args.input:
        # prepare input data
        logger.info(image_path)

        # prepare input data
        input_data, h, w = load_image(
            image_path,
            scaled_size=(args.width,args.height),
            rgb_mode=args.rgb
        )

        # inference
        logger.info('Start inference...')
        if args.benchmark:
            logger.info('BENCHMARK mode')
            for i in range(5):
                start = int(round(time.time() * 1000))
                preds_ailia = net.predict([input_data])
                end = int(round(time.time() * 1000))
                logger.info(f'\tailia processing time {end - start} ms')
        else:
            # dim = [(1, 1, 320, 320), (1, 1, 320, 320),..., ]  len=7
            preds_ailia = net.predict([input_data])

        # postprocessing
        # we only use `d1` (the first output, check the original repository)
        pred = preds_ailia[0][0, 0, :, :]

        savepath = get_savepath(args.savepath, image_path, ext='.png')
        logger.info(f'saved at : {savepath}')
        save_result(pred, savepath, [h, w])

        # composite
        if args.composite:
            image = imread(image_path)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2BGRA)
            image[:, :, 3] = cv2.resize(pred, (w, h)) * 255
            cv2.imwrite(savepath, image)

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
        #writer = webcamera_utils.get_writer(args.savepath, f_h, f_w, rgb=False) # alpha
        writer = webcamera_utils.get_writer(args.savepath, f_h, f_w) # composite
    else:
        writer = None
    
    frame_shown = False
    while(True):
        ret, frame = capture.read()
        if (cv2.waitKey(1) & 0xFF == ord('q')) or not ret:
            break
        if frame_shown and cv2.getWindowProperty('frame', cv2.WND_PROP_VISIBLE) == 0:
            break

        if args.rgb and image.shape[2] == 3:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        input_data = transform(frame, (args.width, args.height))

        # inference
        preds_ailia = net.predict([input_data])

        # postprocessing
        pred = cv2.resize(norm(preds_ailia[0][0, 0, :, :]), (f_w, f_h))

        # force composite
        frame[:, :, 0] = frame[:, :, 0] * pred + 64 * (1 - pred)
        frame[:, :, 1] = frame[:, :, 1] * pred + 177 * (1 - pred)
        frame[:, :, 2] = frame[:, :, 2] * pred
        pred = frame / 255.0

        if args.rgb and image.shape[2] == 3:
            pred = cv2.cvtColor(pred, cv2.COLOR_RGB2BGR)

        cv2.imshow('frame', pred)
        frame_shown = True

        # save results
        if writer is not None:
            writer.write((pred * 255).astype(np.uint8))

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
    if args.width!=IMAGE_SIZE or args.height!=IMAGE_SIZE:
        net.set_input_shape((1,3,args.height,args.width))

    if args.video is not None:
        # video mode
        recognize_from_video(net)
    else:
        # image mode
        recognize_from_image(net)


if __name__ == '__main__':
    main()
