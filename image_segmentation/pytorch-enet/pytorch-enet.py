import sys
import time
from collections import OrderedDict
import itertools

from PIL import Image
import numpy as np
import cv2

import ailia

# import original modules
sys.path.append('../../util')
import webcamera_utils  # noqa: E402
from arg_utils import get_base_parser, update_parser, get_savepath  # noqa: E402
from model_utils import check_and_download_models  # noqa: E402
from detector_utils import load_image  # noqa: E402

# logger
from logging import getLogger  # noqa: E402

logger = getLogger(__name__)

# ======================
# Parameters
# ======================

MODEL_LIST = ['camvid', 'cityscapes']
WEIGHT_CAMVID_PATH = './enet_camvid.onnx'
MODEL_CAMVID_PATH = './enet_camvid.onnx.prototxt'
WEIGHT_CITYSCAPES_PATH = './enet_cityscapes.onnx'
MODEL_CITYSCAPES_PATH = './enet_cityscapes.onnx.prototxt'
REMOTE_PATH = \
    'https://storage.googleapis.com/ailia-models/pythorch-enet/'

IMAGE_PATH = 'demo.png'
SAVE_IMAGE_PATH = 'output.png'

CATEGORY_CAMVID = OrderedDict([
    ('sky', (128, 128, 128)),
    ('building', (128, 0, 0)),
    ('pole', (192, 192, 128)),
    ('road_marking', (255, 69, 0)),
    ('road', (128, 64, 128)),
    ('pavement', (60, 40, 222)),
    ('tree', (128, 128, 0)),
    ('sign_symbol', (192, 128, 128)),
    ('fence', (64, 64, 128)),
    ('car', (64, 0, 128)),
    ('pedestrian', (64, 64, 0)),
    ('bicyclist', (0, 128, 192)),
    ('unlabeled', (0, 0, 0)),
])
CATEGORY_CITYSCAPES = OrderedDict([
    ('unlabeled', (0, 0, 0)),
    ('road', (128, 64, 128)),
    ('sidewalk', (244, 35, 232)),
    ('building', (70, 70, 70)),
    ('wall', (102, 102, 156)),
    ('fence', (190, 153, 153)),
    ('pole', (153, 153, 153)),
    ('traffic_light', (250, 170, 30)),
    ('traffic_sign', (220, 220, 0)),
    ('vegetation', (107, 142, 35)),
    ('terrain', (152, 251, 152)),
    ('sky', (70, 130, 180)),
    ('person', (220, 20, 60)),
    ('rider', (255, 0, 0)),
    ('car', (0, 0, 142)),
    ('truck', (0, 0, 70)),
    ('bus', (0, 60, 100)),
    ('train', (0, 80, 100)),
    ('motorcycle', (0, 0, 230)),
    ('bicycle', (119, 11, 32))
])

IMAGE_CAMVID_HEIGTH = 360
IMAGE_CAMVID_WIDTH = 480
IMAGE_CITYSCAPES_HEIGHT = 512
IMAGE_CITYSCAPES_WIDTH = 1024

# ======================
# Arguemnt Parser Config
# ======================
parser = get_base_parser(
    'PyTorch-ENet model', IMAGE_PATH, SAVE_IMAGE_PATH
)
parser.add_argument(
    '-m', '--model_type', metavar='MODEL_TYPE',
    default='camvid', choices=MODEL_LIST,
    help='Set model architecture: ' + ' | '.join(MODEL_LIST)
)
parser.add_argument(
    '--onnx',
    action='store_true',
    help='execute onnxruntime version.'
)
args = update_parser(parser)


# ======================
# Secondaty Functions
# ======================

def preprocess(img, img_size):
    img = cv2.resize(img, (img_size[1], img_size[0]), cv2.INTER_LINEAR)
    img = img.astype(np.float32) / 255.0
    img = img.transpose(2, 0, 1)
    img = np.expand_dims(img, 0)

    return img


def post_processing(output, img_size):
    output = output[0].transpose(1, 2, 0)
    output = np.argmax(output, axis=2)

    output = cv2.resize(
        output.astype(np.uint8),
        (img_size[1], img_size[0]), cv2.INTER_NEAREST)

    return output


def get_palette(num_cls):
    """ Returns the color map for visualizing the segmentation mask.
    Args:
        num_cls: Number of classes
    Returns:
        The color map
    """
    n = num_cls
    palette = [0] * (n * 3)
    for j in range(0, n):
        lab = j
        palette[j * 3 + 0] = 0
        palette[j * 3 + 1] = 0
        palette[j * 3 + 2] = 0
        i = 0
        while lab:
            palette[j * 3 + 0] |= (((lab >> 0) & 1) << (7 - i))
            palette[j * 3 + 1] |= (((lab >> 1) & 1) << (7 - i))
            palette[j * 3 + 2] |= (((lab >> 2) & 1) << (7 - i))
            i += 1
            lab >>= 3
    return palette


# ======================
# Main functions
# ======================

def detect_objects(img, net, img_size):
    h, w = img.shape[:2]

    # initial preprocesses
    img = preprocess(img, img_size)

    logger.debug(f'input image shape: {img.shape}')

    # feedforward
    if not args.onnx:
        output = net.predict([img])
    else:
        output = net.run(
            ["labels"],
            {"img": img})

    output = output[0]

    # post processes
    pixel_labels = post_processing(output, (h, w))

    return pixel_labels


def recognize_from_image(net, params):
    category = params['category']

    # input image loop
    for image_path in args.input:
        logger.info(image_path)

        # prepare input data
        img = load_image(image_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2RGB)

        # inference
        logger.info('Start inference...')
        if args.benchmark:
            logger.info('BENCHMARK mode')
            for i in range(5):
                start = int(round(time.time() * 1000))
                pixel_labels = detect_objects(img, net, params['img_size'])
                end = int(round(time.time() * 1000))
                logger.info(f'\tailia processing time {end - start} ms')
        else:
            pixel_labels = detect_objects(img, net, params['img_size'])

        output_img = Image.fromarray(np.asarray(pixel_labels, dtype=np.uint8))
        # palette = get_palette(len(category))
        palette = list(itertools.chain.from_iterable(category.values()))
        output_img.putpalette(palette)

        savepath = get_savepath(args.savepath, image_path)
        logger.info(f'saved at : {savepath}')
        output_img.save(savepath)


def recognize_from_video(net, params):
    capture = webcamera_utils.get_capture(args.video)

    # create video writer if savepath is specified as video format
    if args.savepath != SAVE_IMAGE_PATH:
        f_h = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        f_w = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        writer = webcamera_utils.get_writer(args.savepath, f_h, f_w)
    else:
        writer = None

    category = params['category']
    # palette = get_palette(len(category))
    palette = list(itertools.chain.from_iterable(category.values()))

    frame_shown = False
    while True:
        ret, frame = capture.read()
        if (cv2.waitKey(1) & 0xFF == ord('q')) or not ret:
            break
        if frame_shown and cv2.getWindowProperty('frame', cv2.WND_PROP_VISIBLE) == 0:
            break

        img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pixel_labels = detect_objects(img, net, params['img_size'])

        # draw segmentation area
        mask = pixel_labels != category['unlabeled']
        im = Image.fromarray(np.asarray(pixel_labels, dtype=np.uint8))
        im.putpalette(palette)

        fill = np.asarray(im.convert("RGB"))
        fill = cv2.cvtColor(fill, cv2.COLOR_RGB2BGR)
        frame[mask] = frame[mask] * 0.6 + fill[mask] * 0.4

        # show
        cv2.imshow('frame', frame)
        frame_shown = True

        # save results
        if writer is not None:
            writer.write(frame)

    capture.release()
    cv2.destroyAllWindows()
    if writer is not None:
        writer.release()


def main():
    # model files check and download
    info = {
        'camvid': (WEIGHT_CAMVID_PATH, MODEL_CAMVID_PATH,
                   (IMAGE_CAMVID_HEIGTH, IMAGE_CAMVID_WIDTH), CATEGORY_CAMVID),
        'cityscapes': (WEIGHT_CITYSCAPES_PATH, MODEL_CITYSCAPES_PATH,
                       (IMAGE_CITYSCAPES_HEIGHT, IMAGE_CITYSCAPES_WIDTH), CATEGORY_CITYSCAPES),
    }
    weight_path, model_path, img_size, category = info[args.model_type]
    check_and_download_models(weight_path, model_path, REMOTE_PATH)

    if args.onnx:
        import onnxruntime
        net = onnxruntime.InferenceSession(weight_path)
    else:
        net = ailia.Net(model_path, weight_path, env_id=args.env_id, memory_mode=args.memory_mode)

    params = {
        'img_size': img_size,
        'category': category
    }
    if args.video is not None:
        # video mode
        recognize_from_video(net, params)
    else:
        # image mode
        # image mode
        recognize_from_image(net, params)


if __name__ == '__main__':
    main()
