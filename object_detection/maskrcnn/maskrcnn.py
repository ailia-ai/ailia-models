import sys
import time
import math

import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
# import pycocotools.mask as mask_util
from PIL import Image

import ailia

# import original modules
sys.path.append('../../util')
from arg_utils import get_base_parser, update_parser, get_savepath  # noqa: E402
from model_utils import check_and_download_models  # noqa: E402
import webcamera_utils  # noqa: E402

# logger
from logging import getLogger   # noqa: E402
logger = getLogger(__name__)


# ======================
# Parameters
# ======================
REMOTE_PATH = 'https://storage.googleapis.com/ailia-models/mask_rcnn/'

IMAGE_PATH = 'demo.jpg'
SAVE_IMAGE_PATH = 'output.png'

CLASSES = [line.rstrip('\n') for line in open('coco_classes.txt')]


# ======================
# Arguemnt Parser Config
# ======================
parser = get_base_parser(
    'Real-time NN for object instance segmentation by Mask R-CNN',
    IMAGE_PATH,
    SAVE_IMAGE_PATH,
    large_model=True
)
parser.add_argument(
    '-n', '--normal',
    action='store_true',
    help='By default, the optimized model is used, but with this option, ' +
    'you can switch to the normal (not optimized) model'
)
args = update_parser(parser)

if args.normal:
    WEIGHT_PATH = 'mask_rcnn_R_50_FPN_1x.onnx'
    MODEL_PATH = 'mask_rcnn_R_50_FPN_1x.onnx.prototxt'
else:
    WEIGHT_PATH = 'mask_rcnn_R_50_FPN_1x.opt.onnx'
    MODEL_PATH = 'mask_rcnn_R_50_FPN_1x.opt.onnx.prototxt'

# ======================
# Utils
# ======================
def preprocess(image):
    # Resize
    ratio = 800.0 / min(image.size[0], image.size[1])
    resize_w = int(ratio * image.size[0])
    resize_h = int(ratio * image.size[1])
    if (max(resize_w, resize_h) > 1280.0):
        ratio = 1280.0 / max(image.size[0], image.size[1])
        resize_w = int(ratio * image.size[0])
        resize_h = int(ratio * image.size[1])
    image = image.resize(
        (resize_w, resize_h),
        Image.BILINEAR
    )

    image = np.array(image)[:, :, [2, 1, 0]].astype('float32')
    image = np.transpose(image, [2, 0, 1])

    # Normalize
    mean_vec = np.array([102.9801, 115.9465, 122.7717])
    for i in range(image.shape[0]):
        image[i, :, :] = image[i, :, :] - mean_vec[i]

    # Pad to be divisible of 32
    padded_h = int(math.ceil(image.shape[1] / 32) * 32)
    padded_w = int(math.ceil(image.shape[2] / 32) * 32)

    padded_image = np.zeros((3, padded_h, padded_w), dtype=np.float32)
    padded_image[:, :image.shape[1], :image.shape[2]] = image
    return padded_image


def create_figure():
    fig, ax = plt.subplots(1, figsize=(12, 9), tight_layout=True)
    return fig, ax


def display_objdetect_image(
        fig, ax, image, boxes, labels, scores, masks,
        score_threshold=0.7, savepath=None
):
    """
    Display or Save result

    Parameters
    ----------
    savepath: str
        When savepath is not None, save output image instead of displaying
    """
    # Resize boxes
    ratio = 800.0 / min(image.size[0], image.size[1])
    resize_w = int(ratio * image.size[0])
    resize_h = int(ratio * image.size[1])
    if (max(resize_w, resize_h) > 1280.0):
        ratio = 1280.0 / max(image.size[0], image.size[1])
    boxes /= ratio

    image = np.array(image)

    for mask, box, label, score in zip(masks, boxes, labels, scores):
        # Showing boxes with score > 0.7
        if score <= score_threshold:
            continue

        # Finding contour based on mask
        mask = mask[0, :, :, None]
        int_box = [int(i) for i in box]
        mask = cv2.resize(
            mask, (int_box[2]-int_box[0]+1, int_box[3]-int_box[1]+1))
        mask = mask > 0.5
        im_mask = np.zeros((image.shape[0], image.shape[1]), dtype=np.uint8)
        x_0 = max(int_box[0], 0)
        x_1 = min(int_box[2] + 1, image.shape[1])
        y_0 = max(int_box[1], 0)
        y_1 = min(int_box[3] + 1, image.shape[0])
        mask_y_0 = max(y_0 - int_box[1], 0)
        mask_y_1 = mask_y_0 + y_1 - y_0
        mask_x_0 = max(x_0 - int_box[0], 0)
        mask_x_1 = mask_x_0 + x_1 - x_0
        im_mask[y_0:y_1, x_0:x_1] = mask[
            mask_y_0: mask_y_1, mask_x_0: mask_x_1
        ]
        im_mask = im_mask[:, :, None]

        # cv2.findContours has changed since OpenCV 3.x,
        # but in OpenCV 4.0 it changes back
        contours, hierarchy = cv2.findContours(
            im_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE
        )[-2:]

        image = cv2.drawContours(image, contours, -1, 25, 3)

        rect = patches.Rectangle(
            (box[0], box[1]),
            box[2] - box[0],
            box[3] - box[1],
            linewidth=1,
            edgecolor='b',
            facecolor='none'
        )
        ax.annotate(
            CLASSES[int(label)] + ':' + str(np.round(score, 2)),
            (box[0], box[1]),
            color='w',
            fontsize=12
        )
        ax.add_patch(rect)

    if savepath is not None:
        ax.imshow(image)
        fig.savefig(savepath, dpi=150)
    else:
        plt.imshow(image)


# ======================
# Main functions
# ======================
def recognize_from_image():
    # net initialize
    net = ailia.Net(MODEL_PATH, WEIGHT_PATH, env_id=args.env_id, memory_mode=args.memory_mode)
    if args.profile:
        net.set_profile_mode(True)

    # input image loop
    for image_path in args.input:
        # prepare input data
        logger.info(image_path)
        image = Image.open(image_path)
        input_data = preprocess(image)
        net.set_input_shape(input_data.shape)

        # inference
        logger.info('Start inference...')
        if args.benchmark:
            logger.info('BENCHMARK mode')
            for i in range(5):
                start = int(round(time.time() * 1000))
                boxes, labels, scores, masks = net.predict([input_data])
                end = int(round(time.time() * 1000))
                logger.info(f'\tailia processing time {end - start} ms')
        else:
            boxes, labels, scores, masks = net.predict([input_data])

        # postprocessing
        fig, ax = create_figure()
        savepath = get_savepath(args.savepath, image_path)
        logger.info(f'saved at : {savepath}')
        display_objdetect_image(
            fig, ax, image, boxes, labels, scores, masks, savepath=savepath
        )

    if args.profile:
        print(net.get_summary())

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

    fig, ax = create_figure()

    while(True):
        ret, frame = capture.read()
        if (cv2.waitKey(1) & 0xFF == ord('q')) or not ret:
            break

        frame = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        input_data = preprocess(frame)
        net.set_input_shape(input_data.shape)

        boxes, labels, scores, masks = net.predict([input_data])

        ax.clear()
        display_objdetect_image(fig, ax, frame, boxes, labels, scores, masks)
        plt.pause(.01)
        if not plt.get_fignums():
            break

        # save results
        # if writer is not None:
        #     writer.write(frame)

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
        recognize_from_image()


if __name__ == '__main__':
    main()
