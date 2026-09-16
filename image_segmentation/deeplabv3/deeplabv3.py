import sys
import time

import ailia
import cv2
import numpy as np
from matplotlib import gridspec
from matplotlib import pyplot as plt

from deeplab_utils import *

# import original modules
sys.path.append('../../util')
# logger
from logging import getLogger  # noqa: E402

import webcamera_utils  # noqa: E402
from image_utils import imread, load_image  # noqa: E402
from model_utils import check_and_download_models  # noqa: E402
from arg_utils import get_base_parser, get_savepath, update_parser  # noqa: E402

logger = getLogger(__name__)


# ======================
# PARAMETERS
# ======================
IMAGE_PATH = 'couple.jpg'
SAVE_IMAGE_PATH = 'output.png'

LABEL_NAMES = np.asarray([
    'background', 'aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus',
    'car', 'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse', 'motorbike',
    'person', 'pottedplant', 'sheep', 'sofa', 'train', 'tv'
])
CLASS_NUM = 21
FULL_LABEL_MAP = np.arange(len(LABEL_NAMES)).reshape(len(LABEL_NAMES), 1)
FULL_COLOR_MAP = label_to_color_image(FULL_LABEL_MAP)
assert CLASS_NUM == len(LABEL_NAMES), 'The number of labels is incorrect.'


# ======================
# Arguemnt Parser Config
# ======================
parser = get_base_parser(
    ('DeepLab is a state-of-art deep learning model '
     'for semantic image segmentation.'),
    IMAGE_PATH,
    SAVE_IMAGE_PATH,
)
parser.add_argument(
    '-n', '--normal',
    action='store_true',
    help='By default, the optimized model is used, but with this option, ' +
    'you can switch to the normal (not optimized) model'
)
args = update_parser(parser)


# ======================
# MODEL PARAMETERS
# ======================
if args.normal:
    MODEL_PATH = 'deeplabv3.opt.onnx.prototxt'
    WEIGHT_PATH = 'deeplabv3.opt.onnx'
else:
    MODEL_PATH = 'deeplabv3.onnx.prototxt'
    WEIGHT_PATH = 'deeplabv3.onnx'
REMOTE_PATH = 'https://storage.googleapis.com/ailia-models/deeplabv3/'


# ======================
# Main functions
# ======================
def segment_from_image():
    # net initialize
    net = ailia.Net(MODEL_PATH, WEIGHT_PATH, env_id=args.env_id, memory_mode=args.memory_mode)
    ailia_input_w = net.get_input_shape()[3]
    ailia_input_h = net.get_input_shape()[2]
    input_shape = [ailia_input_h, ailia_input_w]

    # input image loop
    for image_path in args.input:
        # prepare input data
        logger.info(image_path)
        img = load_image(
            image_path,
            input_shape,
            normalize_type='127.5',
            gen_input_ailia=True,
        )

        # inference
        logger.info('Start inference...')
        if args.benchmark:
            logger.info('BENCHMARK mode')
            total_time = 0
            for i in range(args.benchmark_count):
                start = int(round(time.time() * 1000))
                preds_ailia = net.predict(img)[0]
                end = int(round(time.time() * 1000))
                logger.info(f'ailia processing time {end - start} ms')
                if i != 0:
                    total_time = total_time + (end - start)
            logger.info(f'\taverage time {total_time / (args.benchmark_count-1)} ms')
        else:
            preds_ailia = net.predict(img)[0]

        # postprocessing
        seg_map = np.argmax(preds_ailia.transpose(1, 2, 0), axis=2)
        seg_image = label_to_color_image(seg_map).astype(np.uint8)

        # save just segmented image (simple)
        # seg_image = cv2.cvtColor(seg_image, cv2.COLOR_RGB2BGR)
        # cv2.imwrite('seg_test.png', seg_image)

        # save org_img, segmentation-map, segmentation-overlay
        org_img = cv2.cvtColor(imread(image_path), cv2.COLOR_BGR2RGB)
        org_img = cv2.resize(org_img, (seg_image.shape[1], seg_image.shape[0]))

        plt.figure(figsize=(15, 5))
        grid_spec = gridspec.GridSpec(1, 4, width_ratios=[6, 6, 6, 1])

        plt.subplot(grid_spec[0])
        plt.imshow(org_img)
        plt.axis('off')
        plt.title('input image')

        plt.subplot(grid_spec[1])
        plt.imshow(seg_image)
        plt.axis('off')
        plt.title('segmentation map')

        plt.subplot(grid_spec[2])
        plt.imshow(org_img)
        plt.imshow(seg_image, alpha=0.7)
        plt.axis('off')
        plt.title('segmentation overlay')

        unique_labels = np.unique(seg_map)
        ax = plt.subplot(grid_spec[3])
        plt.imshow(
            FULL_COLOR_MAP[unique_labels].astype(np.uint8),
            interpolation='nearest',
        )
        ax.yaxis.tick_right()
        plt.yticks(range(len(unique_labels)), LABEL_NAMES[unique_labels])
        plt.xticks([], [])
        ax.tick_params(width=0.0)
        plt.grid('off')
        savepath = get_savepath(args.savepath, image_path)
        logger.info(f'saved at : {savepath}')
        plt.savefig(savepath)

    logger.info('Script finished successfully.')


def segment_from_video():
    # net initialize
    net = ailia.Net(MODEL_PATH, WEIGHT_PATH, env_id=args.env_id, memory_mode=args.memory_mode)

    ailia_input_w = net.get_input_shape()[3]
    ailia_input_h = net.get_input_shape()[2]

    capture = webcamera_utils.get_capture(args.video)

    # create video writer if savepath is specified as video format
    if args.savepath != SAVE_IMAGE_PATH:
        f_h = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        f_w = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        save_h, save_w = webcamera_utils.calc_adjust_fsize(
            f_h, f_w, ailia_input_h, ailia_input_w
        )
        writer = webcamera_utils.get_writer(args.savepath, save_h, save_w)
    else:
        writer = None

    frame_shown = False
    while(True):
        ret, frame = capture.read()
        if (cv2.waitKey(1) & 0xFF == ord('q')) or not ret:
            break
        if frame_shown and cv2.getWindowProperty('frame', cv2.WND_PROP_VISIBLE) == 0:
            break

        input_image, input_data = webcamera_utils.preprocess_frame(
            frame, ailia_input_h, ailia_input_w, normalize_type='127.5'
        )

        # inference
        input_blobs = net.get_input_blob_list()
        net.set_input_blob_data(input_data, input_blobs[0])
        net.update()
        preds_ailia = np.array(net.get_results())[0, 0]  # TODO why?

        # postprocessing
        seg_map = np.argmax(preds_ailia.transpose(1, 2, 0), axis=2)
        seg_image = label_to_color_image(seg_map).astype(np.uint8)

        # showing the segmented image (simple)
        seg_image = cv2.cvtColor(seg_image, cv2.COLOR_RGB2BGR)
        seg_image = cv2.resize(
            seg_image, (input_image.shape[1], input_image.shape[0])
        )

        cv2.imshow('frame', seg_image)
        frame_shown = True

        # save results
        if writer is not None:
            writer.write(seg_image)

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
        segment_from_video()
    else:
        # image mode
        segment_from_image()


if __name__ == '__main__':
    main()
