import sys
import time

import ailia
import cv2
import numpy as np

from rotnet_utils import create_figure, generate_rotated_image, visualize

# import original modules
sys.path.append('../../util')
# logger
from logging import getLogger  # noqa: E402

import webcamera_utils  # noqa: E402
from image_utils import imread  # noqa: E402
from model_utils import check_and_download_models  # noqa: E402
from arg_utils import get_base_parser, get_savepath, update_parser  # noqa: E402

logger = getLogger(__name__)


# ======================
# Parameters 1
# ======================
MODEL_NAMES = ['mnist', 'gsv2']
MODEL_DICT = {
    'mnist': "rotnet_mnist",
    'gsv2': "rotnet_gsv_2"
}
IMAGE_PATH = 'test.jpg'
SAVE_IMAGE_PATH = 'output.png'
IMAGE_HEIGHT = 224
IMAGE_WIDTH = 224
IMAGE_CHANNELS = 3


# ======================
# Arguemnt Parser Config
# ======================
parser = get_base_parser(
    'Image Rotation Correction Model', IMAGE_PATH, SAVE_IMAGE_PATH,
)
parser.add_argument(
    '--model', '-m', metavar='model',
    default='gsv2', choices=MODEL_NAMES,
    help=('model architecture: ' + ' | '.join(MODEL_NAMES) +
          ' (default: gsv2)')
)
parser.add_argument(
    '--apply_rotate', action='store_true',
    help='If add this argument, apply random rotation to input image'
)
args = update_parser(parser)


# ======================
# Parameters 2
# ======================
WEIGHT_PATH = MODEL_DICT[args.model] + '.onnx'
MODEL_PATH = MODEL_DICT[args.model] + '.onnx.prototxt'
REMOTE_PATH = 'https://storage.googleapis.com/ailia-models/rotnet/'


# ======================
# Main functions
# ======================
def recognize_from_image():
    # net initialize
    net = ailia.Net(MODEL_PATH, WEIGHT_PATH, env_id=args.env_id, memory_mode=args.memory_mode)

    # input image loop
    for image_path in args.input:
        # prepare input data
        logger.info(image_path)
        org_img = cv2.cvtColor(imread(image_path), cv2.COLOR_BGR2RGB)

        if args.apply_rotate:
            rotation_angle = np.random.randint(360)
            rotated_img = generate_rotated_image(
                org_img,
                rotation_angle,
                size=(IMAGE_HEIGHT, IMAGE_WIDTH),
                crop_center=True,
                crop_largest_rect=True
            )
            input_data = rotated_img.reshape((1, IMAGE_HEIGHT, IMAGE_WIDTH, 3))
        else:
            rotation_angle = 0
            rotated_img = cv2.resize(org_img, (IMAGE_HEIGHT, IMAGE_WIDTH))
            input_data = rotated_img.reshape((1, IMAGE_HEIGHT, IMAGE_WIDTH, 3))
        
        if IMAGE_CHANNELS == 1:
            input_data = input_data[:, :, :, 1:2]

        net.set_input_shape(input_data.shape)

        # inference
        logger.info('Start inference...')
        if args.benchmark:
            logger.info('BENCHMARK mode')
            for i in range(5):
                start = int(round(time.time() * 1000))
                preds_ailia = net.predict(input_data)
                end = int(round(time.time() * 1000))
                logger.info(f'\tailia processing time {end - start} ms')
        else:
            preds_ailia = net.predict(input_data)

        # visualize
        predicted_angle = np.argmax(preds_ailia, axis=1)[0]
        fig = create_figure()
        plt = visualize(fig, rotated_img, rotation_angle, predicted_angle)
        savepath = get_savepath(args.savepath, image_path)
        logger.info(f'saved at : {savepath}')
        plt.savefig(savepath)

    logger.info('Script finished successfully.')


def recognize_from_video():
    # net initialize
    net = ailia.Net(MODEL_PATH, WEIGHT_PATH, env_id=args.env_id, memory_mode=args.memory_mode)
    net.set_input_shape((1, IMAGE_HEIGHT, IMAGE_WIDTH, 3))

    capture = webcamera_utils.get_capture(args.video)

    # create video writer if savepath is specified as video format
    if args.savepath != SAVE_IMAGE_PATH:
        logger.warning(
            'currently, video results cannot be output correctly...'
        )
        # TODO: DEBUG: shape
        f_h = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        f_w = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        writer = webcamera_utils.get_writer(args.savepath, f_h, f_w)
    else:
        writer = None

    fig = create_figure()
    tight_layout = True

    while(True):
        ret, frame = capture.read()
        if (cv2.waitKey(1) & 0xFF == ord('q')) or not ret:
            break

        input_image, resized_img = webcamera_utils.adjust_frame_size(
            frame, IMAGE_HEIGHT, IMAGE_WIDTH
        )
        resized_img = cv2.cvtColor(resized_img, cv2.COLOR_BGR2RGB)

        if args.apply_rotate:
            rotation_angle = np.random.randint(360)
            rotated_img = generate_rotated_image(
                resized_img,
                rotation_angle,
                size=(IMAGE_HEIGHT, IMAGE_WIDTH),
                crop_center=True,
                crop_largest_rect=True
            )
            input_data = rotated_img.reshape((1, IMAGE_HEIGHT, IMAGE_WIDTH, 3))
        else:
            rotation_angle = 0
            rotated_img = resized_img
            input_data = rotated_img.reshape((1, IMAGE_HEIGHT, IMAGE_WIDTH, 3))

        if IMAGE_CHANNELS == 1:
            input_data = input_data[:, :, :, 1:2]

        # inference
        preds_ailia = net.predict(input_data)

        # visualize
        predicted_angle = np.argmax(preds_ailia, axis=1)[0]
        plt = visualize(
            fig, rotated_img, rotation_angle, predicted_angle, tight_layout
        )
        plt.pause(.01)
        if not plt.get_fignums():
            break
        tight_layout = False

        # # save results
        # if writer is not None:
        #     writer.write(res_img)

    capture.release()
    cv2.destroyAllWindows()
    if writer is not None:
        writer.release()
    logger.info('Script finished successfully.')


def main():
    # model files check and download
    check_and_download_models(WEIGHT_PATH, MODEL_PATH, REMOTE_PATH)

    if args.model == "mnist":
        global IMAGE_WIDTH, IMAGE_HEIGHT, IMAGE_CHANNELS
        IMAGE_WIDTH = 28
        IMAGE_HEIGHT = 28
        IMAGE_CHANNELS = 1

    if args.video is not None:
        # video mode
        recognize_from_video()
    else:
        # image mode
        recognize_from_image()


if __name__ == '__main__':
    main()
