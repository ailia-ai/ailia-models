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
from image_utils import imread, load_image  # noqa: E402
from model_utils import check_and_download_models  # noqa: E402
from arg_utils import get_base_parser, get_savepath, update_parser  # noqa: E402

import adain_utils  # noqa: E402

logger = getLogger(__name__)


# ======================
# Parameters
# ======================
VGG_WEIGHT_PATH = 'adain-vgg.onnx'
VGG_MODEL_PATH = 'adain-vgg.onnx.prototxt'
DEC_WEIGHT_PATH = 'adain-decoder.onnx'
DEC_MODEL_PATH = 'adain-decoder.onnx.prototxt'
REMOTE_PATH = 'https://storage.googleapis.com/ailia-models/adain/'

IMAGE_PATH = 'cornell.jpg'
STYLE_PATH = 'woman_with_hat_matisse.jpg'
SAVE_IMAGE_PATH = 'output.png'
IMAGE_HEIGHT = 512
IMAGE_WIDTH = 512


# ======================
# Arguemnt Parser Config
# ======================
parser = get_base_parser(
    'Arbitrary Style Transfer Model', IMAGE_PATH, SAVE_IMAGE_PATH,
)
parser.add_argument(
    '-t', '--style', metavar='STYLE_IMAGE',
    default=STYLE_PATH,
    help='The style image path.'
)
parser.add_argument(
    '-a', '--alpha',
    default=1.0, type=float,
    help='Adjust the degree of stylization. It should be a value between 0.0 and 1.0(default).'
)
args = update_parser(parser)


# ======================
# Utils
# ======================
# TODO multiple style image and weight feature
def style_transfer(vgg, decoder, content, style, alpha=args.alpha):
    assert (0.0 <= alpha <= 1.0)
    content_f = vgg.predict(content.astype(np.float32))
    style_f = vgg.predict(style)
    feat = adain_utils.adaptive_instance_normalization(content_f, style_f)
    feat = feat * alpha + content_f * (1 - alpha)
    return decoder.predict(feat)


# ======================
# Main functions
# ======================
def image_style_transfer():
    # net initialize
    vgg = ailia.Net(VGG_MODEL_PATH, VGG_WEIGHT_PATH, env_id=args.env_id, memory_mode=args.memory_mode)
    decoder = ailia.Net(DEC_MODEL_PATH, DEC_WEIGHT_PATH, env_id=args.env_id, memory_mode=args.memory_mode)

    # input image loop
    for image_path in args.input:
        # prepare input data
        logger.info(image_path)
        input_img = load_image(
            image_path,
            (IMAGE_HEIGHT, IMAGE_WIDTH),
            normalize_type='255',
            gen_input_ailia=True,
        )

        src_h, src_w, _ = imread(image_path).shape
        style_img = load_image(
            args.style,
            (IMAGE_HEIGHT, IMAGE_WIDTH),
            normalize_type='255',
            gen_input_ailia=True,
        )

        # inference
        logger.info('Start inference...')
        if args.benchmark:
            logger.info('BENCHMARK mode')
            for i in range(5):
                start = int(round(time.time() * 1000))
                preds_ailia = style_transfer(
                    vgg, decoder, input_img, style_img
                )
                end = int(round(time.time() * 1000))
                logger.info(f'\tailia processing time {end - start} ms')
        else:
            preds_ailia = style_transfer(vgg, decoder, input_img, style_img)

        res_img = cv2.cvtColor(
            preds_ailia[0].transpose(1, 2, 0),
            cv2.COLOR_RGB2BGR
        )
        res_img = cv2.resize(res_img, (src_w, src_h))
        savepath = get_savepath(args.savepath, image_path)
        logger.info(f'saved at : {savepath}')
        cv2.imwrite(savepath, np.clip(res_img * 255 + 0.5, 0, 255))
    logger.info('Script finished successfully.')


def video_style_transfer():
    # net initialize
    vgg = ailia.Net(VGG_MODEL_PATH, VGG_WEIGHT_PATH, env_id=args.env_id, memory_mode=args.memory_mode)
    decoder = ailia.Net(DEC_MODEL_PATH, DEC_WEIGHT_PATH, env_id=args.env_id, memory_mode=args.memory_mode)

    capture = webcamera_utils.get_capture(args.video)

    # create video writer if savepath is specified as video format
    if args.savepath != SAVE_IMAGE_PATH:
        writer = webcamera_utils.get_writer(
            args.savepath, IMAGE_HEIGHT, IMAGE_WIDTH
        )
    else:
        writer = None

    # Style image
    style_img = load_image(
        args.style,
        (IMAGE_HEIGHT, IMAGE_WIDTH),
        normalize_type='255',
        gen_input_ailia=True
    )

    frame_shown = False
    while(True):
        ret, frame = capture.read()
        if (cv2.waitKey(1) & 0xFF == ord('q')) or not ret:
            break
        if frame_shown and cv2.getWindowProperty('frame', cv2.WND_PROP_VISIBLE) == 0:
            break

        # Resize by padding the perimeter.
        _, input_data = webcamera_utils.preprocess_frame(
            frame, IMAGE_HEIGHT, IMAGE_WIDTH, normalize_type='255'
        )

        # # The image will be distorted by normal resize
        # input_data = (cv2.cvtColor(
        #     cv2.resize(frame, (IMAGE_WIDTH, IMAGE_HEIGHT)), cv2.COLOR_BGR2RGB
        # ) / 255.0).transpose(2, 0, 1)[np.newaxis, :, :, :]

        # inference
        preds_ailia = style_transfer(vgg, decoder, input_data, style_img)

        # post-processing
        res_img = cv2.cvtColor(
            preds_ailia[0].transpose(1, 2, 0), cv2.COLOR_RGB2BGR
        )

        cv2.imshow('frame', res_img)
        frame_shown = True

        # save results
        if writer is not None:
            writer.write(np.clip(res_img * 255 + 0.5, 0, 255).astype(np.uint8))

    capture.release()
    cv2.destroyAllWindows()
    if writer is not None:
        writer.release()

    logger.info('Script finished successfully.')


def main():
    # model files check and download
    check_and_download_models(VGG_WEIGHT_PATH, VGG_MODEL_PATH, REMOTE_PATH)
    check_and_download_models(DEC_WEIGHT_PATH, DEC_MODEL_PATH, REMOTE_PATH)

    if args.video is not None:
        # video mode
        video_style_transfer()
    else:
        # image mode
        image_style_transfer()


if __name__ == '__main__':
    main()
