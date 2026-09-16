import cv2
import os
import sys
import ailia
import numpy as np
import time
import ssl
ssl._create_default_https_context = ssl._create_unverified_context

# import original modules
sys.path.append('../../util')
from image_utils import load_image, get_image_shape  # noqa: E402
from arg_utils import get_base_parser, update_parser, get_savepath  # noqa: E402
import webcamera_utils  # noqa: E402
from model_utils import check_and_download_models  # noqa: E402

# logger
from logging import getLogger   # noqa: E402
logger = getLogger(__name__)


# ======================
# Parameters 1
# ======================

IMAGE_PATH = 'input.png'
SAVE_IMAGE_PATH = 'output.png'
REMOTE_PATH = 'https://storage.googleapis.com/ailia-models/edsr/'


# ======================
# Arguemnt Parser Config
# ======================

parser = get_base_parser('EDSR model', IMAGE_PATH, SAVE_IMAGE_PATH)
parser.add_argument('--scale', choices=['2', '3', '4'], default='2', help='choose scale')
parser.add_argument(
    '--bilinear',
    action='store_true',
    help='execute bilinear version.'
)
args = update_parser(parser)


# ======================
# Parameters 2
# ======================

WEIGHT_PATH = 'edsr_scale' + args.scale + '.onnx'
MODEL_PATH = 'edsr_scale' + args.scale + '.onnx.prototxt'


# ======================
# Main functions
# ======================

def recognize_from_image():
    if args.bilinear:
        logger.error('bilinear mode only supporting in video input')
        return

    # net initialize
    net = ailia.Net(MODEL_PATH, WEIGHT_PATH, env_id=args.env_id)
    logger.info(IMAGE_PATH)

    for image_path in args.input:

        IMAGE_HEIGHT, IMAGE_WIDTH = get_image_shape(image_path)

        # prepare input data
        logger.info(image_path)
        input_data = load_image(
            image_path,
            (IMAGE_HEIGHT, IMAGE_WIDTH),
            gen_input_ailia=True,
            normalize_type='None'
        )
        net.set_input_shape((1,3,IMAGE_HEIGHT,IMAGE_WIDTH))

        # inference
        logger.info('Start inference...')
        if args.benchmark:
            logger.info('BENCHMARK mode')
            for i in range(5):
                start = int(round(time.time() * 1000))
                preds_ailia = net.predict(input_data)[0]
                end = int(round(time.time() * 1000))
                logger.info(f'\tailia processing time {end - start} ms')
        else:
            preds_ailia = net.predict(input=input_data)[0]

        # postprocessing
        output_img = preds_ailia.transpose(1, 2, 0)
        output_img = np.clip(output_img, 0, 255)
        output_img = cv2.cvtColor(output_img, cv2.COLOR_RGB2BGR)
        savepath = get_savepath(args.savepath, image_path, ext='.png')
        logger.info(f'saved at : {savepath}')
        cv2.imwrite(savepath, output_img)
    logger.info('Script finished successfully.')


def recognize_from_video():
    # net initialize
    net = ailia.Net(MODEL_PATH, WEIGHT_PATH, env_id=args.env_id)

    capture = webcamera_utils.get_capture(args.video)

    # create video writer if savepath is specified as video format
    if args.savepath != SAVE_IMAGE_PATH:
        logger.warning(
            'currently, video results cannot be output correctly...'
        )
        f_h = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT) * int(args.scale))
        f_w = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH) * int(args.scale))
        writer = webcamera_utils.get_writer(args.savepath, f_h, f_w)
    else:
        writer = None

    time.sleep(1)  
    
    frame_shown = False
    while(True):
        ret, frame = capture.read()
        if (cv2.waitKey(1) & 0xFF == ord('q')) or not ret:
            break
        if frame_shown and cv2.getWindowProperty('frame', cv2.WND_PROP_VISIBLE) == 0:
            break
            
        IMAGE_HEIGHT, IMAGE_WIDTH = frame.shape[0], frame.shape[1]

        if args.bilinear:
            output_img = cv2.resize(frame,(int(IMAGE_WIDTH*int(args.scale)),int(IMAGE_HEIGHT*int(args.scale))))
        else:
            # Preprocessing
            input_image, input_data = webcamera_utils.preprocess_frame(
                frame, IMAGE_HEIGHT, IMAGE_WIDTH, normalize_type='None'
            )
            net.set_input_shape((1,3,IMAGE_HEIGHT,IMAGE_WIDTH))

            # Inference
            preds_ailia = net.predict(input_data)[0] 

            # Postprocessing
            output_img = preds_ailia.transpose(1, 2, 0)
            output_img = cv2.cvtColor(output_img, cv2.COLOR_RGB2BGR)
            output_img = np.clip(output_img, 0, 255)
            output_img = output_img.astype(np.uint8)

        cv2.imshow('frame', output_img)
        frame_shown = True

        # save results
        if writer is not None:
            writer.write(output_img)

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
