import time
import os
import sys
import cv2
import onnxruntime
import json

from dab_detr_utils import Detect

import ailia

# import original modules
sys.path.append('../../util')
from arg_utils import get_base_parser, update_parser, get_savepath
from model_utils import check_and_download_models
from detector_utils import load_image, reverse_letterbox, plot_results, write_predictions
import webcamera_utils

# logger
from logging import getLogger

logger = getLogger(__name__)

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

# ======================
# Parameters
# ======================
REMOTE_PATH = 'https://storage.googleapis.com/ailia-models/dab-detr/'

IMAGE_PATH = 'input.jpg'
SAVE_IMAGE_PATH = 'output.jpg'

COCO_CATEGORY = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train",
    "truck", "boat", "traffic light", "fire hydrant", "stop sign",
    "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow",
    "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
    "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard",
    "sports ball", "kite", "baseball bat", "baseball glove", "skateboard",
    "surfboard", "tennis racket", "bottle", "wine glass", "cup", "fork",
    "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange",
    "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair",
    "couch", "potted plant", "bed", "dining table", "toilet", "tv",
    "laptop", "mouse", "remote", "keyboard", "cell phone", "microwave",
    "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase",
    "scissors", "teddy bear", "hair drier", "toothbrush"
]


# ======================
# Arguemnt Parser Config
# ======================
parser = get_base_parser('DAB-DETR model', IMAGE_PATH, SAVE_IMAGE_PATH)
parser.add_argument(
    '-o', '--onnx', action='store_true',
    help="Option to use onnxrutime to run or not."
)
parser.add_argument(
    '-w', '--write_json',
    action='store_true',
    help='Flag to output results to json file.'
)
args = update_parser(parser)

WEIGHT_PATH = "dab_detr.onnx"
MODEL_PATH = "dab_detr.onnx.prototxt"

HEIGHT = 800
WIDTH = 1199

# ======================
# Main functions
# ======================
def recognize_from_image():
    '''
    env_id = args.env_id
    net = ailia.Net(MODEL_PATH, WEIGHT_PATH)
    net.set_input_shape((1, 3, HEIGHT, WIDTH))

    ailiaSDKでモデルを読み込んだ際に下記のエラーが発生
    ailia.core.AiliaException: code: -128 (Unknown error.)
    + error detail : (empty)
    '''

    # Onnx runtime
    if args.onnx:
        session = onnxruntime.InferenceSession(WEIGHT_PATH)
    else:
        session = ailia.Net(MODEL_PATH, WEIGHT_PATH, env_id = args.env_id, memory_mode=args.memory_mode)

    # input image loop
    for image_path in args.input:
        # prepare input data
        logger.debug(f'input image: {image_path}')
        raw_img = cv2.imread(image_path)
        logger.debug(f'input image shape: {raw_img.shape}')
        img = cv2.resize(raw_img, dsize=(HEIGHT, WIDTH))

        if args.benchmark:
            logger.info('BENCHMARK mode')
            total_time = 0
            for i in range(args.benchmark_count):
                start = int(round(time.time() * 1000))
                detect = Detect(session, img)
                output = detect.detect(args)
                end = int(round(time.time() * 1000))
                if i != 0:
                    total_time = total_time + (end - start)
                logger.info(f'\tailia processing time {end - start} ms')
            logger.info(f'\taverage time {total_time / (args.benchmark_count-1)} ms')
        else:
            pass

        # inference
        logger.info('Start inference...')
        detect = Detect(session, img, args)
        output = detect.detect(args)

        detect_object = reverse_letterbox(output, raw_img, (raw_img.shape[0], raw_img.shape[1]))
        res_img = plot_results(detect_object, raw_img, COCO_CATEGORY)

        # plot result
        savepath = get_savepath(args.savepath, image_path)
        logger.info(f'saved at : {savepath}')
        cv2.imwrite(savepath, res_img)

        # write prediction
        if args.write_json:
            json_path = '%s.json' % savepath.rsplit('.', 1)[0]
            out_list = []
            for o in output:
                out_list.append({
                    "category": o.category, "category_name": COCO_CATEGORY[o.category],
                    "prob": float(o.prob), "x": float(o.x), "y": float(o.y), "w": float(o.w), "h": float(o.h)
                })
            with open(json_path, "w", encoding="utf-8") as f:
                json.dump(out_list, f, indent=2)

    logger.info('Script finished successfully.')


def recognize_from_video():
    # net initialize
    if args.onnx:
        session = onnxruntime.InferenceSession(WEIGHT_PATH)
    else:
        session = ailia.Net(MODEL_PATH, WEIGHT_PATH, env_id = args.env_id, memory_mode=args.memory_mode)

    capture = webcamera_utils.get_capture(args.video)

    # create video writer if savepath is specified as video format
    if args.savepath != SAVE_IMAGE_PATH:
        f_h = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        f_w = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        writer = webcamera_utils.get_writer(args.savepath, f_h, f_w)
    else:
        writer = None

    frame_shown = False
    while (True):
        ret, frame = capture.read()
        if (cv2.waitKey(1) & 0xFF == ord('q')) or not ret:
            break
        if frame_shown and cv2.getWindowProperty('frame', cv2.WND_PROP_VISIBLE) == 0:
            break

        img = cv2.resize(frame, dsize=(HEIGHT, WIDTH))

        detect = Detect(session, img, args)
        output = detect.detect(args)

        detect_object = reverse_letterbox(output, frame, (frame.shape[0], frame.shape[1]))
        res_img = plot_results(detect_object, frame, COCO_CATEGORY)

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
