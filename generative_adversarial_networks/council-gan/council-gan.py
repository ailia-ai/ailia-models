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

from yolo_face import FaceLocator  # noqa: E402

logger = getLogger(__name__)


# ======================
# PARAMETERS
# ======================
PATH_SUFFIX = [
    'councilGAN-glasses',
    'councilGAN-m2f_256',
    'councilGAN-anime'
]

MODEL = 0

REMOTE_PATH = "https://storage.googleapis.com/ailia-models/council-gan/"

IMAGE_PATH = 'sample.jpg'
SAVE_IMAGE_PATH = 'output.png'


# ======================
# Arguemnt Parser Config
# ======================
parser = get_base_parser(
    'Glasses removal, m2f and anime transformation GAN based on SimGAN',
    IMAGE_PATH,
    SAVE_IMAGE_PATH,
)
parser.add_argument(
    '-f', '--face_recognition',
    action='store_true',
    help='Run face recognition with yolo v3 (only for glasses removal mode)'
)
parser.add_argument(
    '-d', '--dilation', metavar='DILATION',
    default=1,
    help='Dilation value for face recognition image size'
)
parser.add_argument(
    '-g', '--glasses',
    action='store_true',
    help='Run glasses-removal mode'
)
parser.add_argument(
    '-m', '--m2f',
    action='store_true',
    help='Run male-to-female mode'
)
parser.add_argument(
    '-a', '--anime',
    action='store_true',
    help='Run anime mode'
)
parser.add_argument(
    '-r', '--resolution', metavar='RESOLUTION',
    default=128,
    help='Input file resolution for glasses mode'
)
args = update_parser(parser)


# ======================
# Preprocessing functions
# ======================
def preprocess(image):
    """Convert channel-first BGR image as numpy /n
    array to normalized channel-last RGB."""
    image = center_crop(image)
#     size = [128, 256, 256][MODEL]
    image = cv2.resize(image, (args.resolution, args.resolution))
    # BGR to RGB
    image = image[..., ::-1]
    # scale to [0,1]
    image = image/255.
    # swap channel order
    image = np.transpose(image, [2, 0, 1])
    # resize
    # normalize
    image = (image-0.5)/0.5
    return image.astype(np.float32)


def center_crop(image):
    """Crop the image around the center to make square"""
    shape = image.shape[0:2]
    size = min(shape)
    return image[
        (shape[0]-size)//2:(shape[0]+size)//2,
        (shape[1]-size)//2:(shape[1]+size)//2,
        ...
    ]


def square_coords(coords, dilation=1.0):
    """Make coordinates square for the network with /n
    dimension equal to the longer side * dilation, same /n
    center"""
    top, left, bottom, right = coords
    w = right-left
    h = bottom-top

    dim = 1 if w > h else 0

    new_size = int(max(w, h)*dilation)
    change_short = new_size - min(w, h)
    change_long = new_size - max(w, h)

    out = list(coords)
    out[0+dim] -= change_long//2
    out[1-dim] -= change_short//2
    out[2+dim] += change_long//2
    out[3-dim] += change_short//2

    return out


def get_slice(image, coords):
    """Get a subarray of the image using coordinates /n
    that may be outside the bounds of the image. If so, /n
    return a slice as if the image were padded in all /n
    sides with zeros."""
    padded_slice = np.zeros((coords[2]-coords[0], coords[3]-coords[1], 3))
    new_coords = np.zeros((4), dtype=np.int16)
    padded_coords = np.zeros((4), dtype=np.int16)
    #  limit coords to the shape of the image, and get new coordinates relative
    # to new padded shape for later replacement
    for dim in [0, 1]:
        new_coords[0+dim] = 0 if coords[0+dim] < 0 else coords[0+dim]
        new_coords[2+dim] = image.shape[0+dim] \
            if coords[2+dim] > image.shape[0+dim] else coords[2+dim]
        padded_coords[0+dim] = new_coords[0+dim]-coords[0+dim]
        padded_coords[2+dim] = padded_coords[0+dim] + new_coords[2+dim] - \
            new_coords[0+dim]

    # get the new correct slice and put it in padded array
    image_slice = image[sliceify(new_coords)]
    padded_slice[sliceify(padded_coords)] = image_slice

    return padded_slice, new_coords, padded_coords


def sliceify(coords):
    """Turn a list of (top, left, bottom right) into slices for indexing."""
    return slice(coords[0], coords[2]), slice(coords[1], coords[3])


# ======================
# Postprocessing functions
# ======================
def postprocess_image(image):
    """Convert network output to channel-last 8bit unsigned integet array"""
    max_v = np.max(image)
    min_v = np.min(image)
    final_image = np.transpose(
        (image-min_v)/(max_v-min_v)*255+0.5, (1, 2, 0)
    ).round()
    out = np.clip(final_image, 0, 255).astype(np.uint8)
    return out


def replace_face(img, replacement, coords):
    """Replace a face in the input image with a transformed one."""
    img = img.copy()
    img[sliceify(coords)] = cv2.resize(
        replacement, (coords[3]-coords[1], coords[2]-coords[0])
    )
    return img


# ======================
# Main functions
# ======================
def transform_image():
    """Full transormation on a single image loaded from filepath in arguments.
    """
    net = ailia.Net(MODEL_PATH, WEIGHT_PATH, env_id=args.env_id, memory_mode=args.memory_mode)

    # input image loop
    for image_path in args.input:
        # prepare input data
        logger.info(image_path)
        image = imread(image_path)

        if args.face_recognition:
            locator = FaceLocator(args.env_id, args.memory_mode)
        else:
            locator = None

        logger.info('Start inference...')

        if args.benchmark:
            logger.info('BENCHMARK mode')
            for i in range(5):
                start = int(round(time.time() * 1000))

                out_image = process_frame(net, locator, image)

                end = int(round(time.time() * 1000))
                logger.info(f'\tailia processing time {end - start} ms')

        else:
            out_image = process_frame(net, locator, image)

        savepath = get_savepath(args.savepath, image_path)
        logger.info(f'saved at : {savepath}')
        cv2.imwrite(savepath, out_image[..., ::-1])
    return True


def process_frame(net, locator, image):
    """Process a single frame with preloaded network and locator"""
    if args.face_recognition and MODEL == 0:
        # Run with face recognition using yolo
        out_image = image.copy()[..., ::-1]
        # Get face coordinates with yolo
        face_coords = locator.get_faces(image[..., ::-1])
        # Replace each set of coordinates with its glass-less transformation
        for coords in face_coords:
            coords = square_coords(coords, dilation=float(args.dilation))

            image_slice, coords, padded_coords = get_slice(image, coords)

            processed_slice = process_array(net, preprocess(image_slice))
            processed_slice = processed_slice[sliceify(padded_coords)]
            out_image = replace_face(out_image, processed_slice, coords)

    else:
        out_image = process_array(net, preprocess(image))

    return out_image


def process_array(net, img):
    """Apply network to a correctly scaled and centered image """
    preds_ailia = postprocess_image(net.predict(img[None, ...])[0])
    return preds_ailia


def process_video():
    # net initialize
    net = ailia.Net(MODEL_PATH, WEIGHT_PATH, env_id=args.env_id, memory_mode=args.memory_mode)

    if args.face_recognition:
        locator = FaceLocator(args.env_id, args.memory_mode)
    else:
        locator = None

    capture = webcamera_utils.get_capture(args.video)
    if args.savepath != SAVE_IMAGE_PATH:
        writer = webcamera_utils.get_writer(
            args.savepath, args.resolution, args.resolution
        )
    else:
        writer = None

    frame_shown = False
    while(True):
        ret, frame = capture.read()
        if (cv2.waitKey(1) & 0xFF == ord('q')) or not ret:
            break
        if frame_shown and cv2.getWindowProperty('frame', cv2.WND_PROP_VISIBLE) == 0:
            break

        img = process_frame(net, locator, frame)

        img = img[..., ::-1]
        cv2.imshow('frame', img)
        frame_shown = True
        # save results
        if writer is not None:
            writer.write(img)

    capture.release()
    cv2.destroyAllWindows()
    if writer is not None:
        writer.release()
    logger.info('Script finished successfully.')


def main():
    # check model choice, defaults to glasses-removal
    global PATH_SUFFIX
    global MODEL

    model_choice = [args.glasses, args.m2f, args.anime]
    if sum(model_choice) > 1:
        raise ValueError('Please select only one model (-g, -m, or -a)')
    elif sum(model_choice) == 0:
        pass
    else:
        MODEL = np.argmax(model_choice)

    PATH_SUFFIX = PATH_SUFFIX[MODEL]

    # 128 is only available for glasses-mode
    if MODEL != 0:
        args.resolution = 256
    elif args.resolution != 128:
        PATH_SUFFIX += '_' + str(args.resolution)
        args.resolution = int(args.resolution)

    global WEIGHT_PATH
    global MODEL_PATH

    WEIGHT_PATH = PATH_SUFFIX + '.onnx'
    MODEL_PATH = PATH_SUFFIX + '.onnx.prototxt'
    logger.debug(f'weight path : {WEIGHT_PATH}')
    logger.debug(f'model path {MODEL_PATH}')

    # model files check and download
    check_and_download_models(WEIGHT_PATH, MODEL_PATH, REMOTE_PATH)

    if args.video is not None:
        # video mode
        process_video()
    else:
        # image mode
        transform_image()


if __name__ == '__main__':
    main()
