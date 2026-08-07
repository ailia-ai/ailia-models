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
from image_utils import imread  # noqa: E402
from model_utils import check_and_download_models  # noqa: E402
from arg_utils import get_base_parser, get_savepath, update_parser  # noqa: E402
from dtype_utils import numpy_type_to_builtin_type  # noqa: E402

logger = getLogger(__name__)

# for EasyOCR
from easyocr_utils import *

# ======================
# PARAMETERS
# ======================
DETECTOR_MODEL_PATH             = 'detector_craft.onnx.prototxt'
DETECTOR_WEIGHT_PATH            = 'detector_craft.onnx'
RECOGNIZER_CHINESE_MODEL_PATH   = 'recognizer_zh_sim_g2.onnx.prototxt'
RECOGNIZER_CHINESE_WEIGHT_PATH  = 'recognizer_zh_sim_g2.onnx'
RECOGNIZER_JAPANESE_MODEL_PATH  = 'recognizer_japanese_g2.onnx.prototxt'
RECOGNIZER_JAPANESE_WEIGHT_PATH = 'recognizer_japanese_g2.onnx'
RECOGNIZER_ENGLISH_MODEL_PATH   = 'recognizer_english_g2.onnx.prototxt'
RECOGNIZER_ENGLISH_WEIGHT_PATH  = 'recognizer_english_g2.onnx'
RECOGNIZER_FRENCH_MODEL_PATH    = 'recognizer_latin_g2.onnx.prototxt'
RECOGNIZER_FRENCH_WEIGHT_PATH   = 'recognizer_latin_g2.onnx'
RECOGNIZER_KOREAN_MODEL_PATH    = 'recognizer_korean_g2.onnx.prototxt'
RECOGNIZER_KOREAN_WEIGHT_PATH   = 'recognizer_korean_g2.onnx'
RECOGNIZER_THAI_MODEL_PATH      = 'recognizer_thai.onnx.prototxt'
RECOGNIZER_THAI_WEIGHT_PATH     = 'recognizer_thai.onnx'

REMOTE_PATH = 'https://storage.googleapis.com/ailia-models/easyocr/'

IMAGE_PATH = 'example/chinese.jpg'
IMAGE_OR_VIDEO_PATH = 'input.jpg'
SAVE_IMAGE_OR_VIDEO_PATH = 'output.png'

IMAGE_WIDTH = 640
IMAGE_HEIGHT = 339



# ======================
# Arguemnt Parser Config
# ======================
parser = get_base_parser(
    'Ready-to-use OCR', IMAGE_PATH, SAVE_IMAGE_OR_VIDEO_PATH
)
parser.add_argument(
    '-l', '--language', type=str, default='chinese',
)
args = update_parser(parser)


# ======================
# Main functions
# ======================
def predict(img, img_grey):
    # detect
    horizontal_list, free_list = detector_predict(detector, img)

    # recognize
    result = recognizer_predict(args.language, lang_list, character, symbol, recognizer, img_grey, horizontal_list[0], free_list[0])

    return result


def recognize_from_image():
    for image_path in args.input:
        # prepare input data
        logger.info(image_path)
        img_gray = imread(image_path, cv2.IMREAD_GRAYSCALE)
        img = imread(image_path)

        # predict
        result = predict(img, img_gray)

        # print result
        result = numpy_type_to_builtin_type(result)
        print('==============================================================')
        for r in result:
            print(f'  word={r[1]} confidence={r[2]} bbox={r[0]}')

        # draw result
        draw_img = draw_ocr_box_txt(img, result)
        cv2.imwrite(args.savepath, draw_img)
    logger.info('Script finished successfully.')


def recognize_from_video():
    capture = webcamera_utils.get_capture(args.video)

    if args.savepath != SAVE_IMAGE_OR_VIDEO_PATH:
        f_h = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        f_w = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)) * 2
        video_writer = webcamera_utils.get_writer(args.savepath, f_h, f_w)
    else:
        video_writer = None

    i = 0
    frame_shown = False
    while(True):
        i += 1
        ret, frame = capture.read()
        if (cv2.waitKey(1) & 0xFF == ord('q')) or not ret:
            break
        print(cv2.getWindowProperty('frame', cv2.WND_PROP_VISIBLE))
        if frame_shown and cv2.getWindowProperty('frame', cv2.WND_PROP_VISIBLE) == 0:
            break

        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # predict
        result = predict(frame, frame_gray)

        # print result
        result = numpy_type_to_builtin_type(result)
        print('==============================================================')
        for r in result:
            print(f'  word={r[1]} confidence={r[2]} bbox={r[0]}')

        # write a frame image to video
        draw_frame = draw_ocr_box_txt(frame, result)
        if video_writer is not None:
            video_writer.write(draw_frame)

        cv2.imshow('frame', draw_frame)
        frame_shown = True

    capture.release()
    cv2.destroyAllWindows()
    if video_writer is not None:
        video_writer.release()
        logger.info('finished process and write result to %s!' % args.savepath)
    logger.info('Script finished successfully.')


if __name__ == '__main__':
    # model files check and download
    check_and_download_models(DETECTOR_WEIGHT_PATH, DETECTOR_MODEL_PATH, REMOTE_PATH)
    check_and_download_models(RECOGNIZER_CHINESE_WEIGHT_PATH, RECOGNIZER_CHINESE_MODEL_PATH, REMOTE_PATH)
    check_and_download_models(RECOGNIZER_JAPANESE_WEIGHT_PATH, RECOGNIZER_JAPANESE_MODEL_PATH, REMOTE_PATH)
    check_and_download_models(RECOGNIZER_ENGLISH_WEIGHT_PATH, RECOGNIZER_ENGLISH_MODEL_PATH, REMOTE_PATH)
    check_and_download_models(RECOGNIZER_FRENCH_WEIGHT_PATH, RECOGNIZER_FRENCH_MODEL_PATH, REMOTE_PATH)
    check_and_download_models(RECOGNIZER_KOREAN_WEIGHT_PATH, RECOGNIZER_KOREAN_MODEL_PATH, REMOTE_PATH)
    check_and_download_models(RECOGNIZER_THAI_WEIGHT_PATH, RECOGNIZER_THAI_MODEL_PATH, REMOTE_PATH)

    # set model
    detector = ailia.Net(DETECTOR_MODEL_PATH, DETECTOR_WEIGHT_PATH, env_id=args.env_id)
    if args.language == 'chinese':
        recognizer = ailia.Net(RECOGNIZER_CHINESE_MODEL_PATH, RECOGNIZER_CHINESE_WEIGHT_PATH, env_id=args.env_id)
        lang_list = ['ch_sim','en']
        character = recognition_models['zh_sim_g2']['characters']
        symbol = recognition_models['zh_sim_g2']['symbols']

    elif args.language == 'japanese':
        recognizer = ailia.Net(RECOGNIZER_JAPANESE_MODEL_PATH, RECOGNIZER_JAPANESE_WEIGHT_PATH, env_id=args.env_id)
        lang_list = ['ja','en']
        character = recognition_models['japanese_g2']['characters']
        symbol = recognition_models['japanese_g2']['symbols']

    elif args.language == 'english':
        recognizer = ailia.Net(RECOGNIZER_ENGLISH_MODEL_PATH, RECOGNIZER_ENGLISH_WEIGHT_PATH, env_id=args.env_id)
        lang_list = ['en']
        character = recognition_models['english_g2']['characters']
        symbol = recognition_models['english_g2']['symbols']

    elif args.language == 'french':
        recognizer = ailia.Net(RECOGNIZER_FRENCH_MODEL_PATH, RECOGNIZER_FRENCH_WEIGHT_PATH, env_id=args.env_id)
        lang_list = ['fr', 'en']
        character = recognition_models['latin_g2']['characters']
        symbol = recognition_models['latin_g2']['symbols']

    elif args.language == 'korean':
        recognizer = ailia.Net(RECOGNIZER_KOREAN_MODEL_PATH, RECOGNIZER_KOREAN_WEIGHT_PATH, env_id=args.env_id)
        lang_list = ['ko', 'en']
        character = recognition_models['korean_g2']['characters']
        symbol = recognition_models['korean_g2']['symbols']

    elif args.language == 'thai':
        recognizer = ailia.Net(RECOGNIZER_THAI_MODEL_PATH, RECOGNIZER_THAI_WEIGHT_PATH, env_id=args.env_id)
        lang_list = ['th']
        character = recognition_models['thai_g1']['characters']
        symbol = recognition_models['thai_g1']['symbols']

    else:
        logger.info('invalid language.')
        exit()

    # predict
    if args.video is not None:
        # video mode
        recognize_from_video()
    else:
        # image mode
        recognize_from_image()
