import sys
import time

import numpy as np
import cv2

import ailia

# import original modules
sys.path.append('../../util')
from arg_utils import get_base_parser, update_parser, get_savepath  # noqa: E402
from model_utils import check_and_download_models  # noqa: E402
from image_utils import imread

# logger
from logging import getLogger   # noqa: E402
logger = getLogger(__name__)

# ======================
# Parameters
# ======================
MODEL_LIST = ['8s', '16s', '32s']
MODEL_8s_PATH = 'pytorch_fcn8s.onnx.prototxt'
WEIGHT_8s_PATH = 'pytorch_fcn8s.onnx'
MODEL_16s_PATH = 'pytorch_fcn16s.onnx.prototxt'
WEIGHT_16s_PATH = 'pytorch_fcn16s.onnx'
MODEL_32s_PATH = 'pytorch_fcn32s.onnx.prototxt'
WEIGHT_32s_PATH = 'pytorch_fcn32s.onnx'
REMOTE_PATH = 'https://storage.googleapis.com/ailia-models/pytorch_fcn/'

IMAGE_PATH = 'image.jpg'
SAVE_IMAGE_PATH = 'result.jpg'

IMAGE_WIDTH = 640
IMAGE_HEIGHT = 480

MEAN_BGR = np.array([104.00698793, 116.66876762, 122.67891434], dtype=np.float32)

# ======================
# Arguemnt Parser Config
# ======================
parser = get_base_parser(
    'pytorch_fcn image segmetation model', IMAGE_PATH, SAVE_IMAGE_PATH
)
parser.add_argument(
    '-m', '--model_type', metavar='MODEL_TYPE',
    default='32s', choices=MODEL_LIST,
    help='Set model pixel stride size: ' + ' | ' .join(MODEL_LIST)
)
args = update_parser(parser)

# ======================
# Utils
# ======================
def load_image(input_path):
    return imread(input_path)

def preprocess(img):
    img = cv2.resize(
        img, (IMAGE_WIDTH, IMAGE_HEIGHT), interpolation=cv2.INTER_AREA
    )
    img = np.expand_dims(img, 0)
    img = img.astype(np.float32) - MEAN_BGR
    img_trnas = img.transpose((0, 3, 1, 2))
    # if img_trnas.max() > 1:
    #     img_trnas = img_trnas / 255.0

    return img_trnas.astype(np.float32)

def make_palette(num_class):
    palette = np.zeros((num_class, 3), dtype=np.uint8)
    for k in range(0, num_class):
        label = k
        i = 0
        while label:
            palette[k,0] |= (((label >> 0) & 1) << (7 - i))
            palette[k,1] |= (((label >> 1) & 1) << (7 - i))
            palette[k,2] |= (((label >> 2) & 1) << (7 - i))
            label >>= 3
            i += 1
    return palette

def post_process(img, seg):
    alpha = 0.5
    # create palette
    pl = make_palette(num_class=21)

    # create color
    seg = seg[0].transpose(1,2,0)
    img = img[0].transpose(1,2,0)
    img += MEAN_BGR
    # img *= 255
    cl = np.argmax(seg, axis=2)

    # segmentation image
    img *= 1.0 - alpha
    img += alpha * pl[cl]
    return img.astype(np.uint8)

def segment_image(img, net):
    h,w = img.shape[:2]
    img = preprocess(img)
    
    output = net.predict({'input.1': img})[0]

    out = post_process(img, output)
    out = cv2.resize(out, (w,h))

    return out

# ======================
# Main functions
# ======================
def recognize(net):
    image_path = args.input[0]
    logger.info(image_path)
    img = load_image(image_path)
    logger.debug(f'input image shape: {img.shape}')

    # inference
    logger.info('Start inference...')
    out = segment_image(img, net)

    savepath = get_savepath(args.savepath, image_path)
    logger.info(f'save at : {savepath}')
    cv2.imwrite(savepath, out)
    logger.info('Script finished successfully.')

def main():
    info = {
        MODEL_LIST[0]: (WEIGHT_8s_PATH, MODEL_8s_PATH),
        MODEL_LIST[1]: (WEIGHT_16s_PATH, MODEL_16s_PATH),
        MODEL_LIST[2]: (WEIGHT_32s_PATH, MODEL_32s_PATH)
    }
    weight_path, model_path = info[args.model_type]
    check_and_download_models(weight_path, model_path, REMOTE_PATH)
    
    logger.info('model type : ' + args.model_type)
    net = ailia.Net(model_path, weight_path, memory_mode=args.memory_mode)

    recognize(net)

if __name__ == '__main__':
    main()