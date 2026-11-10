import cv2
import os
import sys
import ailia
import numpy as np
import argparse
import time

# import original modules
sys.path.append('../../util')
from image_utils import imread, get_image_shape  # noqa: E402
from arg_utils import get_base_parser, update_parser, get_savepath  # noqa: E402
import webcamera_utils  # noqa: E402
from model_utils import check_and_download_models  # noqa: E402

# logger
from logging import getLogger   # noqa: E402
logger = getLogger(__name__)


# ======================
# Parameters 1
# ======================

IMAGE_PATH = 'input.jpg'
SAVE_IMAGE_PATH = 'output.png'
REMOTE_PATH = 'https://storage.googleapis.com/ailia-models/docshadow/'


# ======================
# Arguemnt Parser Config
# ======================

parser = get_base_parser('DocShadow model', IMAGE_PATH, SAVE_IMAGE_PATH)

parser.add_argument(
    "--arch",
    type=str,
    default='sd7k',
    choices=['sd7k','jung','kligler']
)
 
args = update_parser(parser)


# ======================
# Parameters 2
# ======================

WEIGHT_PATH = 'docshadow_' + args.arch + '.onnx'
MODEL_PATH = 'docshadow_' + args.arch + '.onnx.prototxt'


# ======================
# Main functions
# ====================== 

class DocShadowRunner:
    def __init__(self,onnx_path=None):
        self.model = ailia.Net(None,onnx_path, env_id=args.env_id)

    def run(self, images: np.ndarray) -> np.ndarray:
        result = self.model.run({"image": images})[0]
        return result

    @staticmethod
    def preprocess(image: np.ndarray) -> np.ndarray:
        image = image / 255
        image = image[None].transpose(0, 3, 1, 2)
        image = image.astype(np.float32)
        return image

def recognize_from_image():


    runner = DocShadowRunner(WEIGHT_PATH)

    for image_path in args.input:

        IMAGE_HEIGHT, IMAGE_WIDTH = get_image_shape(image_path)

        input_data = imread(image_path)
        img = cv2.cvtColor(input_data, cv2.COLOR_BGR2RGB)

        H, W,_ = img.shape
        image = DocShadowRunner.preprocess(cv2.resize(img,(W, H)))
 
        # inference
        logger.info('Start inference...')
        if args.benchmark:
            logger.info('BENCHMARK mode')
            for i in range(5):
                start = int(round(time.time() * 1000))
                sr = runner.run(image)
                end = int(round(time.time() * 1000))
                logger.info(f'\tailia processing time {end - start} ms')
        else:
           sr = runner.run(image)

        ## postprocessing
        logger.info(f'saved at : {args.savepath}')
        sr = sr[0].transpose(1, 2, 0)* 255
        sr = cv2.cvtColor(sr, cv2.COLOR_RGB2BGR)
        cv2.imwrite(args.savepath, sr)
    logger.info('Script finished successfully.')


def recognize_from_video():
    # net initialize
    runner = DocShadowRunner(WEIGHT_PATH)

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

        img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        H, W,_ = img.shape
        image = DocShadowRunner.preprocess(cv2.resize(img,(W, H)))

        if (cv2.waitKey(1) & 0xFF == ord('q')) or not ret:
            break
            
        ## Preprocessing

        # Inference
        sr = runner.run(image)

        sr = sr[0].transpose(1, 2, 0)
        sr = cv2.cvtColor(sr, cv2.COLOR_RGB2BGR)

        output_img = (sr)

        # Postprocessing
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

