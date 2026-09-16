import sys
import time

import numpy as np
import cv2
from skimage import img_as_ubyte
from skimage import io

import ailia
from illnet_utils import *

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
WEIGHT_PATH = 'illnet.onnx'
MODEL_PATH = 'illnet.onnx.prototxt'
REMOTE_PATH = 'https://storage.googleapis.com/ailia-models/illnet/'

IMAGE_PATH = 'input.png'
SAVE_IMAGE_PATH = 'output.png'

PATCH_RES = 128


# ======================
# Arguemnt Parser Config
# ======================
parser = get_base_parser(
    'Illumination Correction Model', IMAGE_PATH, SAVE_IMAGE_PATH
)
args = update_parser(parser)


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
        img = io.imread(image_path)
        img = preProcess(img)
        input_data = padCropImg(img)
        input_data = input_data.astype(np.float32) / 255.0

        ynum = input_data.shape[0]
        xnum = input_data.shape[1]

        preds_ailia = np.zeros(
            (ynum, xnum, PATCH_RES, PATCH_RES, 3), dtype=np.float32
        )

        # inference
        logger.info('Start inference...')
        if args.benchmark:
            logger.info('BENCHMARK mode')
            for c in range(5):
                start = int(round(time.time() * 1000))

                for j in range(ynum):
                    for i in range(xnum):
                        patchImg = input_data[j, i]
                        patchImg = (patchImg - 0.5) / 0.5
                        patchImg = patchImg.transpose((2, 0, 1))
                        patchImg = patchImg[np.newaxis, :, :, :]
                        out = net.predict(patchImg)
                        out = out.transpose((0, 2, 3, 1))[0]
                        out = (np.clip(out, 0, 1) * 255).astype(np.uint8)
                        preds_ailia[j, i] = out

                end = int(round(time.time() * 1000))
                logger.info(f'\tailia processing time {end - start} ms')
        else:
            start = int(round(time.time() * 1000))

            for j in range(ynum):
                for i in range(xnum):
                    patchImg = input_data[j, i]
                    patchImg = (patchImg - 0.5) / 0.5
                    patchImg = patchImg.transpose((2, 0, 1))
                    patchImg = patchImg[np.newaxis, :, :, :]
                    out = net.predict(patchImg)
                    out = out.transpose((0, 2, 3, 1))[0]
                    out = (np.clip(out, 0, 1) * 255).astype(np.uint8)
                    preds_ailia[j, i] = out

            end = int(round(time.time() * 1000))

        # postprocessing
        resImg = composePatch(preds_ailia)
        resImg = postProcess(resImg)
        savepath = get_savepath(args.savepath, image_path)
        logger.info(f'saved at : {savepath}')
        resImg.save(savepath)
    logger.info('Script finished successfully.')


def recognize_from_video():
    logger.warning('This is test implementation')
    # net initialize

    net = ailia.Net(MODEL_PATH, WEIGHT_PATH, env_id=args.env_id, memory_mode=args.memory_mode)

    capture = webcamera_utils.get_capture(args.video)

    # create video writer if savepath is specified as video format
    if args.savepath != SAVE_IMAGE_PATH:
        f_h = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        f_w = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        dummy_img = np.zeros((f_h, f_w, 3))
        dummy_img = padCropImg(dummy_img)
        ynum = dummy_img.shape[0]
        xnum = dummy_img.shape[1]
        dummy_img = np.zeros(
            (ynum, xnum, PATCH_RES, PATCH_RES, 3), dtype=np.float32
        )
        dummy_img = composePatch(dummy_img)
        writer = webcamera_utils.get_writer(
            args.savepath, dummy_img.shape[0], dummy_img.shape[1]
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

        img = preProcess(frame)
        input_data = padCropImg(img)
        input_data = input_data.astype(np.float32) / 255.0

        ynum = input_data.shape[0]
        xnum = input_data.shape[1]

        preds_ailia = np.zeros(
            (ynum, xnum, PATCH_RES, PATCH_RES, 3), dtype=np.float32
        )

        for j in range(ynum):
            for i in range(xnum):
                patchImg = input_data[j, i]
                patchImg = (patchImg - 0.5) / 0.5
                patchImg = patchImg.transpose((2, 0, 1))
                patchImg = patchImg[np.newaxis, :, :, :]
                out = net.predict(patchImg)
                out = out.transpose((0, 2, 3, 1))[0]
                out = (np.clip(out, 0, 1) * 255).astype(np.uint8)
                preds_ailia[j, i] = out

                resImg = composePatch(preds_ailia)
        resImg = postProcess(resImg)

        resImg = img_as_ubyte(resImg)
        cv2.imshow('frame', resImg)
        frame_shown = True

        # save results
        if writer is not None:
            writer.write(resImg)

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
