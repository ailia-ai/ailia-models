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

logger = getLogger(__name__)


# ======================
# PARAMETERS
# ======================
BM_WEIGHT_PATH = "bm_model.onnx"
WC_WEIGHT_PATH = "wc_model.onnx"
BM_MODEL_PATH = "bm_model.onnx.prototxt"
WC_MODEL_PATH = "wc_model.onnx.prototxt"
REMOTE_PATH = "https://storage.googleapis.com/ailia-models/dewarpnet/"

IMAGE_PATH = 'test.png'
SAVE_IMAGE_PATH = 'output.png'

WC_IMG_HEIGHT = 256
WC_IMG_WIDTH = 256
BM_IMG_HEIGHT = 128
BM_IMG_WIDTH = 128


# ======================
# Arguemnt Parser Config
# ======================
parser = get_base_parser(
    'DewarpNet is a model for document image unwarping.',
    IMAGE_PATH,
    SAVE_IMAGE_PATH,
)
args = update_parser(parser)


# ======================
# Utils
# ======================
def grid_sample(img, grid):
    height, width, c = img.shape
    output = np.zeros_like(img)
    grid[:, :, 0] = (grid[:, :, 0] + 1) * (width-1) / 2
    grid[:, :, 1] = (grid[:, :, 1] + 1) * (height-1) / 2
    # TODO speed up here
    for h in range(height):
        for w in range(width):
            h_ = int(grid[h, w, 1])
            w_ = int(grid[h, w, 0])
            output[h, w] = img[h_, w_]
    return output


def unwarp(img, bm):
    w, h = img.shape[0], img.shape[1]
    bm = bm.transpose(1, 2, 0)
    bm0 = cv2.blur(bm[:, :, 0], (3, 3))
    bm1 = cv2.blur(bm[:, :, 1], (3, 3))
    bm0 = cv2.resize(bm0, (h, w))
    bm1 = cv2.resize(bm1, (h, w))
    bm = np.stack([bm0, bm1], axis=-1)
    img = img.astype(float) / 255.0
    res = grid_sample(img, bm)
    return res


# ======================
# Main functions
# ======================
def run_inference(wc_net, bm_net, img, org_img):
    wc_output = wc_net.predict(img)[0]
    pred_wc = np.clip(wc_output, 0, 1.0).transpose(1, 2, 0)
    bm_input = cv2.resize(
        pred_wc, (BM_IMG_WIDTH, BM_IMG_HEIGHT)
    ).transpose(2, 0, 1)
    bm_input = np.expand_dims(bm_input, 0)
    outputs_bm = bm_net.predict(bm_input)[0]
    uwpred = unwarp(org_img, outputs_bm)  # This is not on GPU!
    return uwpred


def unwarp_from_image():
    # net initialize
    bm_net = ailia.Net(BM_MODEL_PATH, BM_WEIGHT_PATH, env_id=args.env_id, memory_mode=args.memory_mode)
    wc_net = ailia.Net(WC_MODEL_PATH, WC_WEIGHT_PATH, env_id=args.env_id, memory_mode=args.memory_mode)

    # input image loop
    for image_path in args.input:
        # prepare input data
        logger.info(image_path)
        org_img = imread(image_path)
        img = load_image(
            image_path,
            (WC_IMG_HEIGHT, WC_IMG_WIDTH),
            normalize_type='255',
            gen_input_ailia=True,
        )

        # inference
        logger.info('Start inference...')
        if args.benchmark:
            logger.info('BENCHMARK mode')
            for i in range(5):
                start = int(round(time.time() * 1000))
                uwpred = run_inference(wc_net, bm_net, img, org_img)
                end = int(round(time.time() * 1000))
                logger.info("\tailia processing time {} ms".format(end-start))
        else:
            uwpred = run_inference(wc_net, bm_net, img, org_img)

        savepath = get_savepath(args.savepath, image_path)
        logger.info(f'saved at : {savepath}')
        cv2.imwrite(savepath, uwpred * 255)
    logger.info('Script finished successfully.')


def unwarp_from_video():
    # net initialize
    bm_net = ailia.Net(BM_MODEL_PATH, BM_WEIGHT_PATH, env_id=args.env_id, memory_mode=args.memory_mode)
    wc_net = ailia.Net(WC_MODEL_PATH, WC_WEIGHT_PATH, env_id=args.env_id, memory_mode=args.memory_mode)

    capture = webcamera_utils.get_capture(args.video)

    # create video writer if savepath is specified as video format
    if args.savepath != SAVE_IMAGE_PATH:
        logger.warning(
            'currently, video results cannot be output correctly...'
        )
        f_h = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        f_w = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        save_h, save_w = webcamera_utils.calc_adjust_fsize(
            f_h, f_w, WC_IMG_HEIGHT, WC_IMG_WIDTH
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

        org_image, input_data = webcamera_utils.preprocess_frame(
            frame, WC_IMG_HEIGHT, WC_IMG_WIDTH, normalize_type='255'
        )

        uwpred = run_inference(wc_net, bm_net, input_data, org_image)

        cv2.imshow('frame', uwpred)
        frame_shown = True
        # TODO: FIXME:
        # >>> error: (-215:Assertion failed)
        # >>> image.depth() == CV_8U in function 'write'
        # # save results
        # if writer is not None:
        #     writer.write(uwpred)

    capture.release()
    cv2.destroyAllWindows()
    if writer is not None:
        writer.release()
    logger.info('Script finished successfully.')


def main():
    # model files check and download
    check_and_download_models(BM_WEIGHT_PATH, BM_MODEL_PATH, REMOTE_PATH)
    check_and_download_models(WC_WEIGHT_PATH, WC_MODEL_PATH, REMOTE_PATH)

    if args.video is not None:
        # video mode
        unwarp_from_video()
    else:
        # image mode
        unwarp_from_image()


if __name__ == '__main__':
    main()
