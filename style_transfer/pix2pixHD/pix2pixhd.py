import sys
import time
import os

import numpy as np
import cv2

import ailia

# import original modules
sys.path.append('../../util')
from arg_utils import get_base_parser, update_parser, get_savepath  # noqa: E402
from model_utils import check_and_download_models  # noqa: E402
from detector_utils import load_image  # noqa: E402C
from webcamera_utils import get_capture, get_writer  # noqa: E402
# logger
from logging import getLogger  # noqa: E402

logger = getLogger(__name__)

# ======================
# Parameters
# ======================

WEIGHT_PATH = 'pix2pixhd.onnx'
MODEL_PATH = 'pix2pixhd.onnx.prototxt'

REMOTE_PATH = \
    'https://storage.googleapis.com/ailia-models/pix2pixhd/'

IMAGE_PATH = 'frankfurt_000000_000576_gtFine_labelIds.png'
INST_PATH = 'frankfurt_000000_000576_gtFine_instanceIds.png'
SAVE_IMAGE_PATH = 'output.png'

IMAGE_HEIGHT = 1024
IMAGE_WIDTH = 2048

# ======================
# Arguemnt Parser Config
# ======================

parser = get_base_parser(
    'pix2pixHD', IMAGE_PATH, SAVE_IMAGE_PATH,
)
parser.add_argument(
    '-im', '--instance_map', type=str, default=INST_PATH,
    help='The instance map to input with label image'
)
parser.add_argument(
    '-k', '--keep',
    action='store_true',
    help='keep aspect when resizing.'
)
parser.add_argument(
    '--onnx',
    action='store_true',
    help='execute onnxruntime version.'
)
args = update_parser(parser)
if os.path.isfile(args.instance_map):
    args.instance_map = [args.instance_map]

# ======================
# Main functions
# ======================

def preprocess(img):
    oh, ow = (IMAGE_HEIGHT, IMAGE_WIDTH)
    img = cv2.resize(img, (ow, oh), interpolation=cv2.INTER_CUBIC)
    if len(img.shape) == 2:
        img = np.expand_dims(img, axis=2)
    im_h, im_w, _ = img.shape
    # resize image to multiple of 16s
    base = float(16)    
    h = int(round(im_h / base) * base)
    w = int(round(im_w / base) * base)
    if not ((h == im_h) and (w == im_w)):
        img = cv2.resize(img, (w, h), interpolation=cv2.INTER_CUBIC)
        if len(img.shape) == 2:
            img = np.expand_dims(img, axis=2)

    img = img / 255.0
    img = img.transpose(2, 0, 1)  # HWC -> CHW
    img = np.expand_dims(img, axis=0)
    img = img.astype(np.float32)

    return img


def post_processing(img, im_hw):
    img = (img.transpose(1, 2, 0) + 1) / 2.0 * 255.0
    img = np.clip(img, 0, 255).astype(np.uint8)
    img = cv2.resize(img, (im_hw[1], im_hw[0]))
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    return img

def get_edges(t):
    edge = np.zeros(t.shape, dtype=np.uint8)
    edge[:,:,:,1:] = edge[:,:,:,1:] | (t[:,:,:,1:] != t[:,:,:,:-1])
    edge[:,:,:,:-1] = edge[:,:,:,:-1] | (t[:,:,:,1:] != t[:,:,:,:-1])
    edge[:,:,1:,:] = edge[:,:,1:,:] | (t[:,:,1:,:] != t[:,:,:-1,:])
    edge[:,:,:-1,:] = edge[:,:,:-1,:] | (t[:,:,1:,:] != t[:,:,:-1,:])
    
    return edge.astype('float32')

def predict(net, img, inst_map):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    im_h, im_w = img.shape[:2]
    inst_map = cv2.cvtColor(inst_map, cv2.COLOR_BGR2GRAY)
    inst_map = np.expand_dims(inst_map, axis=2)

    img = preprocess(img)
    img = img * 255.0
    inst_map = preprocess(inst_map)

    # create one-hot vector for label map 
    size = img.shape
    oneHot_size = (size[0], 35, size[2], size[3])
    input_label = np.zeros(oneHot_size, dtype=np.float32)
    img = img.astype(np.uint8)
    np.put_along_axis(input_label, img, 1.0, axis=1)

    # get edges from instance map
    edge_map = get_edges(inst_map)
    input_label = np.concatenate((input_label, edge_map), axis=1)

    # feedforward
    if not args.onnx:
        output = net.predict([input_label])
    else:
        output = net.run(None, {'input': input_label})

    output = output[0]

    img = post_processing(output[0], (im_h, im_w))

    return img


def recognize_from_image(net):
    # input image loop
    for image_path,instance_map_path in zip(args.input, args.instance_map):
        logger.info(image_path)
        logger.info(instance_map_path)

        # prepare input data
        img = load_image(image_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
        instance_map = load_image(instance_map_path)
        instance_map = cv2.cvtColor(instance_map, cv2.COLOR_BGRA2BGR)

        # inference
        logger.info('Start inference...')
        if args.benchmark:
            logger.info('BENCHMARK mode')
            total_time_estimation = 0
            for i in range(args.benchmark_count):
                start = int(round(time.time() * 1000))
                out_img = predict(net, img, instance_map)
                end = int(round(time.time() * 1000))
                estimation_time = (end - start)

                # Loggin
                logger.info(f'\tailia processing estimation time {estimation_time} ms')
                if i != 0:
                    total_time_estimation = total_time_estimation + estimation_time

            logger.info(f'\taverage time estimation {total_time_estimation / (args.benchmark_count - 1)} ms')
        else:
            out_img = predict(net, img, instance_map)

        # plot result
        savepath = get_savepath(args.savepath, image_path, ext='.png')
        logger.info(f'saved at : {savepath}')
        cv2.imwrite(savepath, out_img)

    logger.info('Script finished successfully.')


def main():
    weight_path, model_path = WEIGHT_PATH, MODEL_PATH
    check_and_download_models(weight_path, model_path, REMOTE_PATH)

    env_id = args.env_id

    # initialize
    if not args.onnx:
        net = ailia.Net(model_path, weight_path, env_id=env_id, memory_mode=args.memory_mode)
    else:
        import onnxruntime
        net = onnxruntime.InferenceSession(weight_path)

    # The exported model need both input label map and instance map, so only image mode is implemented but not video mode
    recognize_from_image(net)


if __name__ == '__main__':
    main()
