import sys
import cv2
import time
import numpy as np

import ailia

# import original modules
sys.path.append('../../util')
from arg_utils import get_base_parser, update_parser, get_savepath  # noqa: E402
from model_utils import check_and_download_models  # noqa: E402
from image_utils import imread  # noqa: E402
import webcamera_utils  # noqa: E402

# logger
from logging import getLogger   # noqa: E402
logger = getLogger(__name__)

# ======================
# Parameters 1
# ======================
IMAGE_PATH = 'input.png'
SAVE_IMAGE_PATH = 'output.png'

IMAGE_HEIGHT = 512
IMAGE_WIDTH = 512

# ======================
# Argument Parser Config
# ======================
parser = get_base_parser(
    'Swift Parameter-free Attention Network for Efficient Super-Resolution', IMAGE_PATH, SAVE_IMAGE_PATH
)
parser.add_argument(
    '--arch', default="span_ch48", type=str, choices=["span_ch48","span_ch52"],
)
parser.add_argument(
    '--scale', default=2, type=int, choices=[2,4],
    help=('Super-resolution scale. By default 2 (generates an image with twice the resolution).')
)
args = update_parser(parser)


# ======================
# Parameters 2
# ======================
if args.arch == "span_ch48":
    WEIGHT_PATH = "spanx"+ str(args.scale) + "_ch48" + '.onnx'
    MODEL_PATH  = "spanx"+ str(args.scale) + "_ch48" + '.onnx.prototxt'
else:
    WEIGHT_PATH = "spanx" + str(args.scale) + "_ch52" + '.onnx'
    MODEL_PATH  = "spanx" + str(args.scale) + "_ch52" + '.onnx.prototxt'

REMOTE_PATH = 'https://storage.googleapis.com/ailia-models/span/'

# ======================
# Main functions
# ======================

def preprocess(img, bgr2rgb=True, float32=True):
    if img.shape[2] == 3 and bgr2rgb:
        if img.dtype == 'float64':
            img = img.astype('float32')
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.transpose(2, 0, 1).astype(np.float32)
    img = np.expand_dims(img,0)
    return img

def postprocess(img, rgb2bgr=True, out_type=np.uint8, min_max=(0, 1)):

    img = np.squeeze(img,axis=0).astype(np.float32)
    img = np.clip(img, *min_max)

    img = (img - min_max[0]) / (min_max[1] - min_max[0])

    n_dim = img.ndim
    img = img.transpose(1, 2, 0)
    if img.shape[2] == 1:  # gray image
        img = np.squeeze(img, axis=2)
    else:
        if rgb2bgr:
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    if out_type == np.uint8:
        img = (img * 255.0).round()
    result = img.astype(out_type)
    return result


def recognize_from_image(net):

    for image_path in args.input:
        # prepare input data
        logger.info('Input image: ' + image_path)

        img = cv2.imread(image_path).astype(np.float32) / 255.
        # preprocessing
        if img.ndim == 2:
            img = np.expand_dims(img, axis=2)
        c = img.shape[2]
        if c == 1:
            img = np.concatenate([img] * 3, 2)

        img = preprocess(img, bgr2rgb=True, float32=True)

        # inference
        logger.info('Start inference...')
        if args.benchmark:
            logger.info('BENCHMARK mode')
            for i in range(5):
                start = int(round(time.time() * 1000))
                output = net.run(img)[0]
                end = int(round(time.time() * 1000))
                logger.info(f'\tailia processing time {end - start} ms')
        else:
            output = net.run(img)[0]

        # postprocessing
        output = postprocess(output)

        savepath = get_savepath(args.savepath, image_path)
        logger.info(f'saved at : {savepath}')
        cv2.imwrite(savepath, output)
        
    logger.info('Script finished successfully.')

def recognize_from_video(net):
 
    capture = webcamera_utils.get_capture(args.video)

    # create video writer if savepath is specified as video format
    if args.savepath != SAVE_IMAGE_PATH:
        f_h = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT) * int(args.scale))
        f_w = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH) * int(args.scale))
        writer = webcamera_utils.get_writer(args.savepath, f_h, f_w)
    else:
        writer = None

    frame_shown = False

    while (True):
        ret, frame = capture.read()

        if  not ret:
            break
        if (cv2.waitKey(1) & 0xFF == ord('q')) or not ret:
            break
        if frame_shown and cv2.getWindowProperty('output', cv2.WND_PROP_VISIBLE) == 0:
            break

        # preprocessing
        img = frame
        if img.ndim == 2:
            img = np.expand_dims(img, axis=2)
        c = img.shape[2]
        if c == 1:
            img = np.concatenate([img] * 3, 2)
        img = preprocess(img, bgr2rgb=True, float32=True)

        output = net.run(img)[0]
        output = postprocess(output)*255

        cv2.imshow('output', output)
        frame_shown = True

        # save results
        if writer is not None:
            writer.write(output)

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
    logger.info('Model: ' + WEIGHT_PATH[:-5])
    logger.info('Scale: ' + str(args.scale))
    
    if args.video is not None:
        # video mode
        recognize_from_video(net)
    else:
        # image mode
        recognize_from_image(net)


if __name__ == '__main__':
    main()
