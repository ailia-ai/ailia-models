import sys
import time

import numpy as np
import cv2
from pymatting.alpha.estimate_alpha_cf import estimate_alpha_cf
from pymatting.alpha.estimate_alpha_knn import estimate_alpha_knn
from pymatting.alpha.estimate_alpha_lbdm import estimate_alpha_lbdm
from pymatting.alpha.estimate_alpha_lkm import estimate_alpha_lkm
from pymatting.alpha.estimate_alpha_rw import estimate_alpha_rw
from scipy.ndimage import binary_erosion

import ailia

# import original modules
sys.path.append('../../util')
from arg_utils import get_base_parser, update_parser, get_savepath  # noqa
from model_utils import check_and_download_models  # noqa
from detector_utils import load_image  # noqa
from image_utils import normalize_image  # noqa
from webcamera_utils import get_capture, get_writer  # noqa
# logger
from logging import getLogger  # noqa: E402

logger = getLogger(__name__)

# ======================
# Parameters
# ======================

MASK_WEIGHT_PATH = 'u2net_opset11.onnx'
MASK_MODEL_PATH = MASK_WEIGHT_PATH + '.prototxt'
MASK_REMOTE_PATH = 'https://storage.googleapis.com/ailia-models/u2net/'

IMAGE_PATH = 'animal-1.jpg'
SAVE_IMAGE_PATH = 'output.png'

IMAGE_SIZE = 320

# ======================
# Arguemnt Parser Config
# ======================

parser = get_base_parser('Rembg', IMAGE_PATH, SAVE_IMAGE_PATH)
parser.add_argument(
    '-c', '--composite',
    action='store_true',
    help='Composite input image and predicted alpha value'
)
parser.add_argument(
    '-a', '--algorithm',
    default='cf', type=str,
    help='Algorithm to estimate alpha value.'
        'cf: confidence-based'
        'knn: k-nearest neighbor'
        'lbdm: local binary descriptor matching'
        'lkm: local k-means'
        'rw: random walk'
)
parser.add_argument(
    '-w', '--width',
    default=IMAGE_SIZE, type=int,
    help='The segmentation width for u2net. (default: 320)'
)
parser.add_argument(
    '-h', '--height',
    default=IMAGE_SIZE, type=int,
    help='The segmentation height for u2net. (default: 320)'
)
args = update_parser(parser)
args = update_parser(parser)


# ======================
# Secondaty Functions
# ======================

def norm_pred(d: np.ndarray) -> np.ndarray:
    ma = np.max(d)
    mi = np.min(d)
    return (d - mi) / (ma - mi)


# ======================
# Main functions
# ======================

def preprocess(img):
    use_skimage = False

    img = img[:, :, ::-1]  # BGR -> RGB

    if not use_skimage:
        img = cv2.resize(img, (IMAGE_SIZE, IMAGE_SIZE))
        img = img / 255
    else:
        from skimage import transform
        img = transform.resize(img, (IMAGE_SIZE, IMAGE_SIZE), mode="constant")
        img = img / np.max(img)

    # normalize
    img[:, :, 0] = (img[:, :, 0] - 0.485) / 0.229
    img[:, :, 1] = (img[:, :, 1] - 0.456) / 0.224
    img[:, :, 2] = (img[:, :, 2] - 0.406) / 0.225

    img = img.transpose(2, 0, 1)  # HWC -> CHW
    img = np.expand_dims(img, axis=0)
    img = img.astype(np.float32)

    return img


def estimate_alpha(
        img, mask,
        foreground_threshold=240,
        background_threshold=10,
        erode_structure_size=10):
    # guess likely foreground/background
    is_foreground = mask > foreground_threshold
    is_background = mask < background_threshold

    # erode foreground/background
    structure = None
    if erode_structure_size > 0:
        structure = np.ones(
            (erode_structure_size, erode_structure_size), dtype=np.uint8
        )

    is_foreground = binary_erosion(is_foreground, structure=structure)
    is_background = binary_erosion(is_background, structure=structure, border_value=1)

    # build trimap
    # 0   = background
    # 128 = unknown
    # 255 = foreground
    trimap = np.full(mask.shape, dtype=np.uint8, fill_value=128)
    trimap[is_foreground] = 255
    trimap[is_background] = 0

    # fix trimap did not contain foreground values
    if not(255 in trimap):
        return np.zeros(mask.shape)

    # build the cutout image
    img_normalized = img / 255.0
    trimap_normalized = trimap / 255.0

    estimate_alpha_func = 'estimate_alpha_' + args.algorithm
    alpha = globals()[estimate_alpha_func](img_normalized, trimap_normalized)
    alpha = np.clip(alpha * 255, 0, 255).astype(np.uint8)

    return alpha


def predict(net, img):
    im_h, im_w = img.shape[:2]

    img = preprocess(img)

    # feedforward
    output = net.predict([img])
    d1 = output[0]

    pred = d1[0, 0, :, :]
    pred = norm_pred(pred)

    mask = cv2.resize(
        pred * 255, (im_w, im_h),
        interpolation=cv2.INTER_LANCZOS4)

    return mask


def recognize_from_image(net):
    # input image loop
    for image_path in args.input:
        # prepare input data
        logger.info(image_path)

        # prepare input data
        img = load_image(image_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)

        if args.benchmark:
            logger.info('BENCHMARK mode')
            total_time = 0
            for i in range(args.benchmark_count):
                start = int(round(time.time() * 1000))
                mask = predict(net, img)
                res_img = estimate_alpha(img, mask)
                end = int(round(time.time() * 1000))
                if i != 0:
                    total_time = total_time + (end - start)
                logger.info(f'\tailia processing time {end - start} ms')
            logger.info(f'\taverage time {total_time / (args.benchmark_count-1)} ms')
        else:            
            # inference
            mask = predict(net, img)

            # refine alpha
            res_img = estimate_alpha(img, mask)

        if args.composite:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2BGRA)
            img[:, :, 3] = res_img
            res_img = img

        savepath = get_savepath(args.savepath, image_path, ext='.png')
        logger.info(f'saved at : {savepath}')
        cv2.imwrite(savepath, res_img)

    logger.info('Script finished successfully.')


def recognize_from_video(net):
    video_file = args.video if args.video else args.input[0]
    capture = get_capture(video_file)
    assert capture.isOpened(), 'Cannot capture source'

    # create video writer if savepath is specified as video format
    f_h = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    f_w = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    if args.savepath != SAVE_IMAGE_PATH:
        logger.warning(
            'currently, video results cannot be output correctly...'
        )
        writer = get_writer(args.savepath, f_h, f_w)
    else:
        writer = None

    frame_shown = False
    while True:
        ret, frame = capture.read()
        if (cv2.waitKey(1) & 0xFF == ord('q')) or not ret:
            break
        if frame_shown and cv2.getWindowProperty('frame', cv2.WND_PROP_VISIBLE) == 0:
            break

        # inference
        mask = predict(net, frame)
        alpha = estimate_alpha(frame, mask)

        alpha = alpha[:, :, None].astype(np.float32) / 255
        back = np.ones_like(frame) * 255
        res_img = frame * alpha + (back * (1 - alpha))
        res_img = res_img.astype(np.uint8)

        # show
        cv2.imshow('frame', res_img)
        frame_shown = True

        # save results
        if writer is not None:
            writer.write(res_img)

    capture.release()
    cv2.destroyAllWindows()
    if writer is not None:
        writer.release()

    logger.info('Script finished successfully.')


def main():
    check_and_download_models(MASK_WEIGHT_PATH, MASK_MODEL_PATH, MASK_REMOTE_PATH)

    # load model
    env_id = args.env_id

    # net initialize
    net = ailia.Net(MASK_MODEL_PATH, MASK_WEIGHT_PATH, env_id=env_id, memory_mode=args.memory_mode)
    if args.width!=IMAGE_SIZE or args.height!=IMAGE_SIZE:
        net.set_input_shape((1,3,args.height,args.width))

    if args.video is not None:
        recognize_from_video(net)
    else:
        recognize_from_image(net)


if __name__ == '__main__':
    main()
