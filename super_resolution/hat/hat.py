import sys
import cv2
import time
import numpy as np

import ailia

# import original modules
sys.path.append('../../util')
from arg_utils import get_base_parser, update_parser, get_savepath  # noqa: E402
from model_utils import check_and_download_models  # noqa: E402
from image_utils import imread  # noqa: E402
import webcamera_utils  # noqa: E402

# logger
from logging import getLogger   # noqa: E402
logger = getLogger(__name__)


# ======================
# Parameters 1
# ======================
IMAGE_PATH = 'input.png'
SAVE_IMAGE_PATH = 'output.png'

IMAGE_HEIGHT = 256
IMAGE_WIDTH = 256

# ======================
# Argument Parser Config
# ======================
parser = get_base_parser(
    'Single Image Super-Resolution with HAT', IMAGE_PATH, SAVE_IMAGE_PATH, fp16_support=False
)
parser.add_argument(
    '--arch', default="HAT", type=str, choices=["HAT","HAT_S","HAT_GAN_REAL_sharper","HAT_GAN_REAL"],
)
parser.add_argument(
    '--scale', default=2, type=int, choices=[2,3,4],
    help=('Super-resolution scale. By default 2 (generates an image with twice the resolution).')
)
args = update_parser(parser)


# ======================
# Parameters 2
# ======================
if args.arch == "HAT_GAN_REAL_sharper" or args.arch == "HAT_GAN_REAL":
    WEIGHT_PATH = args.arch + '.onnx'
    MODEL_PATH = WEIGHT_PATH + ".prototxt"
    args.scale = 4
else:
    WEIGHT_PATH = args.arch + "_x" + str(args.scale) + '.onnx'
    MODEL_PATH = WEIGHT_PATH + ".prototxt"

REMOTE_PATH = 'https://storage.googleapis.com/ailia-models/hat/'


# ======================
# Main functions
# ======================

class HATModel():
    def __init__(self,net):
        self.net = net

    def pre_process(self,image):
        image = np.expand_dims(image, 0)
        window_size = 16
        self.scale = args.scale
        self.mod_pad_h, self.mod_pad_w = 0, 0
        _, _, h, w = image.shape
        if h % window_size != 0:
            self.mod_pad_h = window_size - h % window_size
        if w % window_size != 0:
            self.mod_pad_w = window_size - w % window_size
        self.img = np.pad(image, (self.mod_pad_w, self.mod_pad_h), mode='reflect')

    def process(self):
        self.output = np.array(self.net.run(self.img)[0])

    def post_process(self):
        _, _, h, w = self.output.shape
        self.output = self.output[:, :, 0:h - self.mod_pad_h * self.scale, 0:w - self.mod_pad_w * self.scale]

    def nondist_validation(self,image):
        self.pre_process(image)

        self.process()
        self.post_process()

        sr_img = tensor2img([self.output])

        return sr_img

def tensor2img(tensor, rgb2bgr=True, out_type=np.uint8, min_max=(0, 1)):

    result = []
    tensor = np.clip(tensor[0],min_max[0],min_max[1])
    tensor = (tensor - min_max[0]) / (min_max[1] - min_max[0])

    img_np = tensor[0]
    img_np = img_np.transpose(1, 2, 0)

    img_np = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
    if out_type == np.uint8:
        img_np = (img_np * 255.0).round()
    img_np = img_np.astype(out_type)
    result.append(img_np)
    if len(result) == 1:
        result = result[0]
    return result


def recognize_from_image(net):
    hat = HATModel(net)

    for image_path in args.input:
        # prepare input data
        logger.info('Input image: ' + image_path)

        # preprocessing
        img = imread(image_path)
        if img.ndim == 2:
            img = np.expand_dims(img, axis=2)
        c = img.shape[2]
        if c == 1:
            img = np.concatenate([img] * 3, 2)
        img =img.transpose((2,0,1)) / 255

        # inference
        logger.info('Start inference...')
        if args.benchmark:
            logger.info('BENCHMARK mode')
            for i in range(5):
                start = int(round(time.time() * 1000))
                preds_ailia = hat.nondist_validation(img)
                end = int(round(time.time() * 1000))
                logger.info(f'\tailia processing time {end - start} ms')
        else:
            preds_ailia = hat.nondist_validation(img)

        # postprocessing
        preds_ailia = cv2.cvtColor(preds_ailia, cv2.COLOR_RGB2BGR)
        output_img = preds_ailia
        savepath = get_savepath(args.savepath, image_path)
        logger.info(f'saved at : {savepath}')
        cv2.imwrite(savepath, output_img)
        
    logger.info('Script finished successfully.')

def recognize_from_video(net):
    hat = HATModel(net)
 
    capture = webcamera_utils.get_capture(args.video)

    # create video writer if savepath is specified as video format
    if args.savepath != SAVE_IMAGE_PATH:
        f_h = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT) * int(args.scale))
        f_w = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH) * int(args.scale))
        writer = webcamera_utils.get_writer(args.savepath, f_h, f_w)
    else:
        writer = None

    frame_shown = False
    while (True):
        ret, frame = capture.read()
        if (cv2.waitKey(1) & 0xFF == ord('q')) or not ret:
            break
        if frame_shown and cv2.getWindowProperty('output', cv2.WND_PROP_VISIBLE) == 0:
            break

        # resize with keep aspect
        frame,resized_img = webcamera_utils.adjust_frame_size(frame, IMAGE_HEIGHT, IMAGE_WIDTH)

        # preprocessing
        img = resized_img
        if img.ndim == 2:
            img = np.expand_dims(img, axis=2)
        c = img.shape[2]
        if c == 1:
            img = np.concatenate([img] * 3, 2)
        img =img.transpose((2,0,1)) / 255

        preds_ailia = hat.nondist_validation(img)
        out_img = cv2.cvtColor(preds_ailia, cv2.COLOR_RGB2BGR)

        cv2.imshow('output', out_img)
        frame_shown = True

        # save results
        if writer is not None:
            writer.write(out_img)

    capture.release()
    cv2.destroyAllWindows()
    if writer is not None:
        writer.release()
    logger.info('Script finished successfully.')


def main():
    # model files check and download
    check_and_download_models(WEIGHT_PATH, MODEL_PATH, REMOTE_PATH)

    # net initialize
    net = ailia.Net(MODEL_PATH, WEIGHT_PATH, env_id=args.env_id, memory_mode=args.memory_mode)
    logger.info('Model: ' + WEIGHT_PATH[:-5])
    logger.info('Scale: ' + str(args.scale))
    
    if args.video is not None:
        # video mode
        recognize_from_video(net)
    else:
        # image mode
        recognize_from_image(net)


if __name__ == '__main__':
    main()
