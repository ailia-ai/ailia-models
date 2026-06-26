import os
import sys
import time

import ailia
import cv2
import numpy as np

from nanodet_utils import NanoDetABC

# import original modules
sys.path.append('../../util')
# logger
from logging import getLogger

import webcamera_utils
from detector_utils import plot_results, reverse_letterbox, write_predictions
from image_utils import imread  # noqa: E402
from model_utils import check_and_download_models
from arg_utils import get_base_parser, get_savepath, update_parser

logger = getLogger(__name__)

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

# ======================
# Parameters
# ======================
MODEL_PARAMS = {'nanodet_m': {'input_shape': [320, 320], 'reg_max': 7},
                'nanodet_m_416': {'input_shape': [416, 416], 'reg_max': 7},
                'nanodet_t': {'input_shape': [320, 320], 'reg_max': 7},
                'nanodet-EfficientNet-Lite0_320': {'input_shape': [320, 320], 'reg_max': 7},
                'nanodet-EfficientNet-Lite1_416': {'input_shape': [416, 416], 'reg_max': 10},
                'nanodet-EfficientNet-Lite2_512': {'input_shape': [512, 512], 'reg_max': 10},
                'nanodet-RepVGG-A0_416': {'input_shape': [416, 416], 'reg_max': 10}}

REMOTE_PATH = 'https://storage.googleapis.com/ailia-models/nanodet/'

IMAGE_PATH = 'input.jpg'
SAVE_IMAGE_PATH = 'output.jpg'

COCO_CATEGORY = [
    'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
    'train', 'truck', 'boat', 'traffic_light', 'fire_hydrant',
    'stop_sign', 'parking_meter', 'bench', 'bird', 'cat', 'dog',
    'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe',
    'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
    'skis', 'snowboard', 'sports_ball', 'kite', 'baseball_bat',
    'baseball_glove', 'skateboard', 'surfboard', 'tennis_racket',
    'bottle', 'wine_glass', 'cup', 'fork', 'knife', 'spoon', 'bowl',
    'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot',
    'hot_dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
    'potted_plant', 'bed', 'dining_table', 'toilet', 'tv', 'laptop',
    'mouse', 'remote', 'keyboard', 'cell_phone', 'microwave',
    'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock',
    'vase', 'scissors', 'teddy_bear', 'hair_drier', 'toothbrush'
]

# ======================
# Arguemnt Parser Config
# ======================
parser = get_base_parser('nanodet model', IMAGE_PATH, SAVE_IMAGE_PATH)
parser.add_argument(
    '--model_name',
    default='nanodet_m',
    help='[nanodet-EfficientNet-Lite0_320, nanodet-EfficientNet-Lite1_416, nanodet-EfficientNet-Lite2_512'
         'nanodet_m, nanodet_m_416, nanodet_t, nanodet-RepVGG-A0_416]'
)
parser.add_argument(
    '-w', '--write_prediction',
    nargs='?',
    const='txt',
    choices=['txt', 'json'],
    type=str,
    help='Output results to txt or json file.'
)
args = update_parser(parser)

MODEL_NAME = args.model_name
if MODEL_NAME == 'nanodet_t':
    WEIGHT_PATH = MODEL_NAME + ".opt2.onnx"
    MODEL_PATH = MODEL_NAME + ".opt2.onnx.prototxt"
else:
    WEIGHT_PATH = MODEL_NAME + ".opt.onnx"
    MODEL_PATH = MODEL_NAME + ".opt.onnx.prototxt"

HEIGHT = MODEL_PARAMS[MODEL_NAME]['input_shape'][0]
WIDTH = MODEL_PARAMS[MODEL_NAME]['input_shape'][1]

REG_MAX = MODEL_PARAMS[MODEL_NAME]['reg_max']


# ======================
# Detection function
# ======================
class NanoDetDetection(NanoDetABC):
    def __init__(self, model, input_shape, reg_max, *args, **kwargs):
        super(NanoDetDetection, self).__init__(*args, **kwargs)
        self.model = model
        self.input_shape = input_shape
        self.input_size = (self.input_shape[1], self.input_shape[0])
        self.reg_max = reg_max

    def infer_image(self, img_input):
        inference_results = self.model.run(img_input)
        scores = [np.squeeze(x) for x in inference_results[:3]]
        raw_boxes = [np.squeeze(x) for x in inference_results[3:]]
        return scores, raw_boxes


# ======================
# Main functions
# ======================
def recognize_from_image():
    env_id = args.env_id
    net = ailia.Net(MODEL_PATH, WEIGHT_PATH, env_id=env_id)
    net.set_input_shape((1, 3, HEIGHT, WIDTH))
    detector = NanoDetDetection(net, input_shape=[HEIGHT, WIDTH], reg_max=REG_MAX)

    # input image loop
    for image_path in args.input:
        # prepare input data
        logger.debug(f'input image: {image_path}')
        raw_img = imread(image_path)
        logger.debug(f'input image shape: {raw_img.shape}')

        # inference
        logger.info('Start inference...')
        if args.benchmark:
            logger.info('BENCHMARK mode')
            for i in range(5):
                start = int(round(time.time() * 1000))
                detect_object = detector.detect(raw_img)
                end = int(round(time.time() * 1000))
                logger.info(f'\tailia processing time {end - start} ms')
        else:
            detect_object = detector.detect(raw_img)

        detect_object = reverse_letterbox(detect_object, raw_img, (raw_img.shape[0], raw_img.shape[1]))
        res_img = plot_results(detect_object, raw_img, COCO_CATEGORY)

        savepath = get_savepath(args.savepath, image_path)
        logger.info(f'saved at : {savepath}')
        cv2.imwrite(savepath, res_img)

        # write prediction
        if args.write_prediction is not None:
            ext = args.write_prediction
            pred_file = "%s.%s" % (savepath.rsplit('.', 1)[0], ext)
            write_predictions(pred_file, detect_object, raw_img, category=COCO_CATEGORY, file_type=ext)

    logger.info('Script finished successfully.')


def recognize_from_video():
    # net initialize
    env_id = args.env_id
    net = ailia.Net(MODEL_PATH, WEIGHT_PATH, env_id=env_id)
    detector = NanoDetDetection(net, input_shape=[HEIGHT, WIDTH], reg_max=REG_MAX)

    capture = webcamera_utils.get_capture(args.video)

    # create video writer if savepath is specified as video format
    if args.savepath != SAVE_IMAGE_PATH:
        logger.warning(
            'currently, video results cannot be output correctly...'
        )
        f_h = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        f_w = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        save_h, save_w = f_h, f_w
        writer = webcamera_utils.get_writer(args.savepath, save_h, save_w)
    else:
        writer = None

    frame_shown = False
    while (True):
        ret, frame = capture.read()
        if (cv2.waitKey(1) & 0xFF == ord('q')) or not ret:
            break
        if frame_shown and cv2.getWindowProperty('frame', cv2.WND_PROP_VISIBLE) == 0:
            break

        raw_img = frame
        detect_object = detector.detect(raw_img)
        detect_object = reverse_letterbox(detect_object, raw_img, (raw_img.shape[0], raw_img.shape[1]))
        res_img = plot_results(detect_object, raw_img, COCO_CATEGORY)
        cv2.imshow('frame', res_img)
        frame_shown = True

        # save results
        if writer is not None:
            writer.write(res_img)

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
