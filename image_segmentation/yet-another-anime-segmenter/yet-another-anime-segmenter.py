import argparse
import sys
import time

import ailia
import cv2
import numpy as np

import yaas_utils as yut

sys.path.append('../../util')
# logger
from logging import getLogger  # noqa: E402

from image_utils import imread, load_image  # noqa: E402
from model_utils import check_and_download_models  # noqa: E402
from arg_utils import get_base_parser, get_savepath, update_parser  # noqa: E402
from webcamera_utils import adjust_frame_size, get_capture  # noqa: E402

logger = getLogger(__name__)

# ======================
# Parameters 1
# ======================
IMAGE_PATH = 'unity_chan.png'
SAVE_IMAGE_PATH = 'output.png'
IMAGE_HEIGHT = 128
IMAGE_WIDTH = 128


# ======================
# Argument Parser Config
# ======================
parser = get_base_parser(
    'Yet-Another-Anime-Segmenter, anime character segmentation.', IMAGE_PATH, SAVE_IMAGE_PATH, large_model=True
)
parser.add_argument(
    '--onnx',
    action='store_true',
    help='By default, the ailia SDK is used, but with this option, ' +
    'you can switch to using ONNX Runtime'
)
args = update_parser(parser)


# ======================
# Parameters 2
# ======================
MODEL_NAME = 'yaas_solov2'
WEIGHT_PATH = f'{MODEL_NAME}.onnx'
MODEL_PATH = f'{MODEL_NAME}.onnx.prototxt'
REMOTE_PATH = f'https://storage.googleapis.com/ailia-models/{MODEL_NAME}/'


# ======================
# Utils
# ======================
def get_seg_img(img, preds):
    """
    docstring
    """
    scores, _, pred_masks, _ = preds
    mask = np.full(img.shape[:2], False, dtype=bool)
    for i in range(scores.shape[0]):
        if scores[i] > 0.5:
            mask = mask | pred_masks[i]
    seg_img = img.copy()
    seg_img[~mask] = 255

    return seg_img

# ======================
# Main functions
# ======================
def recognize_from_image():
    # net initialize
    if args.onnx:
        import onnxruntime
        sess = onnxruntime.InferenceSession(WEIGHT_PATH)
    else:
        segmentor = ailia.Net(MODEL_PATH, WEIGHT_PATH, env_id=args.env_id, memory_mode=args.memory_mode)

    # inference
    logger.info('Start inference...')

    for image_path in args.input:
        # prepare input data
        src_img = imread(image_path)
        input_data = yut.preprocess(src_img)

        if args.benchmark:
            logger.info('BENCHMARK mode')
            for _ in range(5):
                start = int(round(time.time() * 1000))
                if args.onnx:
                    input_name = sess.get_inputs()[0].name
                    preds = sess.run(None, {input_name: input_data.astype(np.float32)})
                    preds = yut.postprocess(preds, input_data.shape[2:], src_img)
                    seg_img = get_seg_img(src_img, preds)
                else:
                    segmentor.set_input_shape(input_data.shape)
                    preds = segmentor.predict([input_data])
                    preds = yut.postprocess(preds, input_data.shape[2:], src_img)
                    seg_img = get_seg_img(src_img, preds)
                end = int(round(time.time() * 1000))
                logger.info(f'\tailia processing time {end - start} ms')
        else:
            if args.onnx:
                input_name = sess.get_inputs()[0].name
                preds = sess.run(None, {input_name: input_data.astype(np.float32)})
                preds = yut.postprocess(preds, input_data.shape[2:], src_img)
                seg_img = get_seg_img(src_img, preds)
            else:
                segmentor.set_input_shape(input_data.shape)
                preds = segmentor.predict([input_data])
                preds = yut.postprocess(preds, input_data.shape[2:], src_img)
                seg_img = get_seg_img(src_img, preds)

        savepath = get_savepath(args.savepath, image_path)
        logger.info(f'saved at : {savepath}')
        cv2.imwrite(savepath, seg_img)
    logger.info('Script finished successfully.')


def recognize_from_video():
    # net initialize
    # This model requires fuge gpu memory so fallback to cpu mode
    segmentor = ailia.Net(MODEL_PATH, WEIGHT_PATH, env_id=args.env_id, memory_mode=args.memory_mode)

    capture = get_capture(args.video)

    # create video writer if savepath is specified as video format
    if args.savepath != SAVE_IMAGE_PATH:
        f_h = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        f_w = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        save_h, save_w = webcamera_utils.calc_adjust_fsize(
            f_h, f_w, IMAGE_HEIGHT, IMAGE_WIDTH
        )
        writer = webcamera_utils.get_writer(args.savepath, save_h, save_w)
    else:
        writer = None

    frame_shown = False
    while(True):
        ret, frame = capture.read()
        if (cv2.waitKey(1) & 0xFF == ord('q')) or not ret:
            break
        if frame_shown and cv2.getWindowProperty('Segmented frame', cv2.WND_PROP_VISIBLE) == 0:
            break


        frame = np.ascontiguousarray(frame[:,::-1,:])

        input_data = yut.preprocess(frame)

        # inference
        if args.onnx:
            input_name = sess.get_inputs()[0].name
            preds = sess.run(None, {input_name: input_data.astype(np.float32)})
            preds = yut.postprocess(preds, input_data.shape[2:], frame)
            seg_img = get_seg_img(frame, preds)
        else:
            segmentor.set_input_shape(input_data.shape)
            preds = segmentor.predict([input_data])
            preds = yut.postprocess(preds, input_data.shape[2:], frame)
            seg_img = get_seg_img(frame, preds)

        cv2.imshow('Segmented frame', seg_img)
        frame_shown = True

        # save results
        if writer is not None:
            writer.write(seg_img)

    capture.release()
    cv2.destroyAllWindows()
    logger.info('Script finished successfully.')
    pass


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
