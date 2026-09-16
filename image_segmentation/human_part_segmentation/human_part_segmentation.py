import sys
import time

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
from hps_utils import xywh2cs, transform_logits, \
    get_affine_transform  # noqa: E402

# logger
from logging import getLogger  # noqa: E402

logger = getLogger(__name__)

# ======================
# Parameters
# ======================

MODEL_LIST = ['lip', 'atr', 'pascal']
WEIGHT_LIP_PATH = './resnet-lip.onnx'
MODEL_LIP_PATH = './resnet-lip.onnx.prototxt'
WEIGHT_ATR_PATH = './resnet-atr.onnx'
MODEL_ATR_PATH = './resnet-atr.onnx.prototxt'
WEIGHT_PASCAL_PATH = './resnet-pascal.onnx'
MODEL_PASCAL_PATH = './resnet-pascal.onnx.prototxt'
REMOTE_PATH = \
    'https://storage.googleapis.com/ailia-models/human_part_segmentation/'

IMAGE_PATH = 'demo.jpg'
SAVE_IMAGE_PATH = 'output.png'

CATEGORY_LIP = (
    'Background', 'Hat', 'Hair', 'Glove', 'Sunglasses', 'Upper-clothes',
    'Dress', 'Coat', 'Socks', 'Pants', 'Jumpsuits', 'Scarf', 'Skirt', 'Face',
    'Left-arm', 'Right-arm', 'Left-leg', 'Right-leg', 'Left-shoe', 'Right-shoe'
)
CATEGORY_ATR = (
    'Background', 'Hat', 'Hair', 'Sunglasses', 'Upper-clothes', 'Skirt', 'Pants', 'Dress', 'Belt',
    'Left-shoe', 'Right-shoe', 'Face', 'Left-leg', 'Right-leg', 'Left-arm', 'Right-arm', 'Bag', 'Scarf'
)
CATEGORY_PASCAL = (
    'Background', 'Head', 'Torso', 'Upper Arms', 'Lower Arms', 'Upper Legs', 'Lower Legs'
)
IMAGE_LIP_SIZE = 473
IMAGE_ATR_SIZE = 512
IMAGE_PASCAL_SIZE = 512

NORM_MEAN = [0.406, 0.456, 0.485]
NORM_STD = [0.225, 0.224, 0.229]

# ======================
# Arguemnt Parser Config
# ======================
parser = get_base_parser(
    'Human-Part-Segmentation model', IMAGE_PATH, SAVE_IMAGE_PATH
)
parser.add_argument(
    '-a', '--arch', metavar='ARCH',
    default='lip', choices=MODEL_LIST,
    help='Set model architecture: ' + ' | '.join(MODEL_LIST)
)
args = update_parser(parser)


# ======================
# Secondaty Functions
# ======================
def preprocess(img, img_size):
    h, w, _ = img.shape

    # Get person center and scale
    person_center, s = xywh2cs(0, 0, w - 1, h - 1)
    r = 0
    trans = get_affine_transform(
        person_center, s, r, img_size
    )
    img = cv2.warpAffine(
        img,
        trans,
        (img_size[1], img_size[0]),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(0, 0, 0))

    # normalize
    img = ((img / 255.0 - NORM_MEAN) / NORM_STD).astype(np.float32)

    img = img.transpose(2, 0, 1)
    img = np.expand_dims(img, 0)
    data = {
        'img': img,
        'center': person_center,
        'height': h,
        'width': w,
        'scale': s,
        'rotation': r
    }
    return data


def post_processing(data, fusion, img_size):
    fusion = fusion[0].transpose(1, 2, 0)
    upsample_output = cv2.resize(
        fusion, img_size, interpolation=cv2.INTER_LINEAR
    )
    logits_result = transform_logits(
        upsample_output,
        data['center'], data['scale'], data['width'], data['height'],
        input_size=img_size
    )

    pixel_labels = np.argmax(logits_result, axis=2)

    return pixel_labels


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

def detect_objects(img, detector, img_size):
    # initial preprocesses
    data = preprocess(img, img_size)

    # feedforward
    output = detector.predict({
        'img': data['img']
    })
    _, fusion, _ = output

    # post processes
    pixel_labels = post_processing(data, fusion, img_size)

    return pixel_labels


def recognize_from_image(filename, detector, params):
    # prepare input data
    img_0 = load_image(filename)
    logger.debug(f'input image shape: {img_0.shape}')

    img = cv2.cvtColor(img_0, cv2.COLOR_BGRA2BGR)

    # inference
    logger.info('Start inference...')
    if args.benchmark:
        logger.info('BENCHMARK mode')
        for i in range(5):
            start = int(round(time.time() * 1000))
            pixel_labels = detect_objects(img, detector, params['img_size'])
            end = int(round(time.time() * 1000))
            logger.info(f'\tailia processing time {end - start} ms')
    else:
        pixel_labels = detect_objects(img, detector, params['img_size'])

    output_img = Image.fromarray(np.asarray(pixel_labels, dtype=np.uint8))

    category = params['category']
    palette = get_palette(len(category))

    output_img.putpalette(palette)
    savepath = get_savepath(args.savepath, filename)
    logger.info(f'saved at : {savepath}')
    output_img.save(savepath)


def recognize_from_video(video, detector, params):
    capture = webcamera_utils.get_capture(video)

    # create video writer if savepath is specified as video format
    if args.savepath != SAVE_IMAGE_PATH:
        f_h = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        f_w = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        writer = webcamera_utils.get_writer(args.savepath, f_h, f_w)
    else:
        writer = None

    category = params['category']
    palette = get_palette(len(category))

    frame_shown = False
    while True:
        ret, frame = capture.read()
        if (cv2.waitKey(1) & 0xFF == ord('q')) or not ret:
            break
        if frame_shown and cv2.getWindowProperty('frame', cv2.WND_PROP_VISIBLE) == 0:
            break

        pixel_labels = detect_objects(frame, detector, params['img_size'])

        # draw segmentation area
        mask = pixel_labels != 0
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
        'lip': (WEIGHT_LIP_PATH, MODEL_LIP_PATH,
                (IMAGE_LIP_SIZE, IMAGE_LIP_SIZE), CATEGORY_LIP),
        'atr': (WEIGHT_ATR_PATH, MODEL_ATR_PATH,
                (IMAGE_ATR_SIZE, IMAGE_ATR_SIZE), CATEGORY_ATR),
        'pascal': (WEIGHT_PASCAL_PATH, MODEL_PASCAL_PATH,
                   (IMAGE_PASCAL_SIZE, IMAGE_ATR_SIZE), CATEGORY_PASCAL),
    }
    weight_path, model_path, img_size, category = info[args.arch]
    check_and_download_models(weight_path, model_path, REMOTE_PATH)

    # Workaround for accuracy issue on
    # ailia SDK 1.2.4 + opset11 + gpu (metal/vulkan)
    detector = ailia.Net(model_path, weight_path, env_id=args.env_id, memory_mode=args.memory_mode)

    params = {
        'img_size': img_size,
        'category': category
    }
    if args.video is not None:
        # video mode
        recognize_from_video(args.video, detector, params)
    else:
        # image mode
        # input image loop
        for image_path in args.input:
            # prepare input data
            logger.info(image_path)
            recognize_from_image(image_path, detector, params)

    logger.info('Script finished successfully.')


if __name__ == '__main__':
    main()
