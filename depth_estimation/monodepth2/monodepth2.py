import sys
import time

import ailia
import cv2
import matplotlib.pyplot as plt
# from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
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
# Parameters
# ======================
MODEL_NAME = 'monodepth2_mono+stereo_640x192'
ENC_WEIGHT_PATH = MODEL_NAME + '_enc.onnx'
ENC_MODEL_PATH = MODEL_NAME + '_enc.onnx.prototxt'
DEC_WEIGHT_PATH = MODEL_NAME + '_dec.onnx'
DEC_MODEL_PATH = MODEL_NAME + '_dec.onnx.prototxt'
REMOTE_PATH = 'https://storage.googleapis.com/ailia-models/monodepth2/'

IMAGE_PATH = 'input.jpg'
SAVE_IMAGE_PATH = 'output.png'
IMAGE_HEIGHT = 192
IMAGE_WIDTH = 640


# ======================
# Arguemnt Parser Config
# ======================
parser = get_base_parser('Depth estimation model', IMAGE_PATH, SAVE_IMAGE_PATH)
args = update_parser(parser)


# ======================
# Utils
# ======================
def result_plot(disp, original_width, original_height):
    disp = disp.squeeze()
    disp_resized = cv2.resize(
        disp,
        (original_width, original_height),
        interpolation=cv2.INTER_LINEAR
    )
    vmax = np.percentile(disp_resized, 95)
    return disp_resized, vmax


# ======================
# Main functions
# ======================
def estimate_from_image():
    # net initialize
    enc_net = ailia.Net(ENC_MODEL_PATH, ENC_WEIGHT_PATH, env_id=args.env_id, memory_mode=args.memory_mode)
    dec_net = ailia.Net(DEC_MODEL_PATH, DEC_WEIGHT_PATH, env_id=args.env_id, memory_mode=args.memory_mode)

    # input image loop
    for image_path in args.input:
        logger.info(image_path)

        # prepare input data
        input_data = load_image(
            image_path,
            (IMAGE_HEIGHT, IMAGE_WIDTH),
            gen_input_ailia=True,
        )
        org_height, org_width, _ = imread(image_path).shape

        # inference
        logger.info('Start inference...')
        if args.benchmark:
            logger.info('BENCHMARK mode')
            for i in range(5):
                start = int(round(time.time() * 1000))
                features = enc_net.predict([input_data])
                preds_ailia = dec_net.predict(features)
                end = int(round(time.time() * 1000))
                logger.info(f'\tailia processing time {end - start} ms')
        else:
            features = enc_net.predict([input_data])
            preds_ailia = dec_net.predict(features)

        # post-processing
        disp = preds_ailia[-1]
        disp_resized, vmax = result_plot(disp, org_width, org_height)
        savepath = get_savepath(args.savepath, image_path, ext='.png')
        logger.info(f'saved at : {savepath}')
        plt.imsave(savepath, disp_resized, cmap='magma', vmax=vmax)
    logger.info('Script finished successfully.')


def estimate_from_video():
    # net initialize
    enc_net = ailia.Net(ENC_MODEL_PATH, ENC_WEIGHT_PATH, env_id=args.env_id, memory_mode=args.memory_mode)
    dec_net = ailia.Net(DEC_MODEL_PATH, DEC_WEIGHT_PATH, env_id=args.env_id, memory_mode=args.memory_mode)

    capture = webcamera_utils.get_capture(args.video)

    # create video writer if savepath is specified as video format
    if args.savepath != SAVE_IMAGE_PATH:
        logger.warning('currently video results output feature '
                       'is not supported in this model!')
        f_h = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        f_w = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        save_h, save_w = webcamera_utils.calc_adjust_fsize(
            f_h, f_w, IMAGE_HEIGHT, IMAGE_WIDTH
        )
        # save_w * 2: we stack source frame and estimated heatmap
        writer = webcamera_utils.get_writer(args.savepath, save_h, save_w * 2)
    else:
        writer = None

    ret, frame = capture.read()
    org_height, org_width, _ = frame.shape

    while(True):
        ret, frame = capture.read()
        if (cv2.waitKey(1) & 0xFF == ord('q')) or not ret:
            break

        _, input_data = webcamera_utils.preprocess_frame(
            frame, IMAGE_HEIGHT, IMAGE_WIDTH
        )

        # encoder
        enc_input_blobs = enc_net.get_input_blob_list()
        enc_net.set_input_blob_data(input_data, enc_input_blobs[0])
        enc_net.update()
        features = enc_net.get_results()

        # decoder
        dec_inputs_blobs = dec_net.get_input_blob_list()
        for f_idx in range(len(features)):
            dec_net.set_input_blob_data(
                features[f_idx], dec_inputs_blobs[f_idx]
            )
        dec_net.update()
        preds_ailia = dec_net.get_results()

        # postprocessing
        disp = preds_ailia[-1]
        disp_resized, vmax = result_plot(disp, org_width, org_height)
        plt.imshow(disp_resized, cmap='magma', vmax=vmax)
        plt.pause(.01)
        if not plt.get_fignums():
            break

        # save results
        # FIXME: How to save plt --> cv2.VideoWriter()
        # if writer is not None:
        #     # put pixel buffer in numpy array
        #     canvas = FigureCanvas(fig)
        #     canvas.draw()
        #     mat = np.array(canvas.renderer._renderer)
        #     res_img = cv2.cvtColor(mat, cv2.COLOR_RGB2BGR)
        #     writer.write(res_img)

    capture.release()
    cv2.destroyAllWindows()
    if writer is not None:
        writer.release()
    logger.info('Script finished successfully.')


def main():
    # model files check and download
    check_and_download_models(ENC_WEIGHT_PATH, ENC_MODEL_PATH, REMOTE_PATH)
    check_and_download_models(DEC_WEIGHT_PATH, DEC_MODEL_PATH, REMOTE_PATH)

    if args.video is not None:
        # video mode
        estimate_from_video()
    else:
        # image mode
        estimate_from_image()


if __name__ == '__main__':
    main()
