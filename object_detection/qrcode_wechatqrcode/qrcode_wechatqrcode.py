from http.client import NON_AUTHORITATIVE_INFORMATION
from math import sqrt
import cv2 
import sys
import time
import numpy as np

import ailia

from qrcode_wechatqrcode_utils import preprocess, postprocess

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
MODEL_PARAMS = {
    'detect_2021nov': {'input_shape': [384, 384]},
    'sr_2021nov': {'input_shape': [224, 224], 'threshold': 160},
}

REMOTE_PATH = 'https://storage.googleapis.com/ailia-models/qrcode_wechatqrcode/'

IMAGE_PATH = 'input.jpg'
SAVE_IMAGE_PATH = 'output.jpg'

# ======================
# Arguemnt Parser Config
# ======================
parser = get_base_parser('QR detection model', IMAGE_PATH, SAVE_IMAGE_PATH)
parser.add_argument(
    '--detect_model_name',
    help='The model name of QR detection.',
    default='detect_2021nov',
)
parser.add_argument(
    '--sr_model_name',
    help='The model name of super resolution.',
    default='sr_2021nov',
)
parser.add_argument(
    '-w', '--write_prediction',
    action='store_true',
    help='Flag to output the prediction file.'
)
parser.add_argument(
    '--decode_qrcode',
    action='store_true',
    help='Decode qrcode using zbar.'  
)
args = update_parser(parser)

if args.decode_qrcode:
    import pyzbar.pyzbar as zbar

DETECT_MODEL_NAME = args.detect_model_name
DETECT_WEIGHT_PATH = "detect" + ".caffemodel"
DETECT_MODEL_PATH = "detect" + ".prototxt"

DETECT_HEIGHT = MODEL_PARAMS[DETECT_MODEL_NAME]['input_shape'][0]
DETECT_WIDTH = MODEL_PARAMS[DETECT_MODEL_NAME]['input_shape'][1]

SR_MODEL_NAME = args.sr_model_name
SR_WEIGHT_PATH = "sr" + ".caffemodel"
SR_MODEL_PATH = "sr" + ".prototxt"

SR_HEIGHT = MODEL_PARAMS[SR_MODEL_NAME]['input_shape'][0]
SR_WIDTH = MODEL_PARAMS[SR_MODEL_NAME]['input_shape'][1]
SR_TH = MODEL_PARAMS[SR_MODEL_NAME]['threshold']

def scale_image(img, scale, sr):
    if scale == 1.0:
        return img
    elif scale == 2.0:
        h, w, _ = img.shape
        if sqrt(w * h * 1.0) < SR_TH:
            img = preprocess(img, (SR_HEIGHT, SR_WIDTH))
            res = sr.run(img[None, None, :, :])
            result = np.clip(res[0][0][0] * 255, 0, 255.0).astype(np.int8)
            return result
        else:
            return cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
    elif scale < 1.0:
        return cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    
    return None

def get_scale_list(width, height):
    if width < 320 or height < 320:
        return [1.0, 2.0, 0.5]
    elif width < 640 and height < 640:
        return [1.0, 0.5]
    else:
        return [0.5, 1.0]

def decode(raw_img, detections, sr):
    PADDING = 0.1
    MIN_PADDING = 15.0

    decoded = []

    for d in detections:
        padx = int(max(PADDING * d.w, MIN_PADDING))
        pady = int(max(PADDING * d.h, MIN_PADDING))

        left = max(d.x - padx, 0)
        top = max(d.y - pady, 0)
        right = min(d.x + d.w + padx, raw_img.shape[1])
        bottom = min(d.y + d.h + pady, raw_img.shape[0])
        cropped = raw_img[top:bottom, left:right, :].copy()
        ch, cw, _ = cropped.shape

        scales = get_scale_list(cw, ch)
        for scale in scales:
            scaled_img = scale_image(cropped, scale, sr)
            text = None
            if args.decode_qrcode:
                qr = zbar.decode(scaled_img)
                if len(qr) > 0:
                    text = qr[0].data.decode()
            if not args.decode_qrcode or text:
                obj = {
                    'left': d.x,
                    'top': d.y,
                    'right': d.x + d.w,
                    'bottom': d.y + d.h,
                    'text': text
                }
                decoded.append(obj)
                break
    
    return decoded

def visualize(raw_img, decoded):
    result_img = raw_img.copy()

    for d in decoded:
        cv2.putText(result_img, d['text'], (d['left'], d['bottom']), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), thickness=2)
        cv2.rectangle(result_img, (d['left'], d['top']), (d['right'], d['bottom']), (0, 0, 255), thickness=3)

    return result_img

def recognize_from_image(detection, sr):
    # input image loop
    for image_path in args.input:
        # prepare input data
        logger.info(image_path)
        raw_img = cv2.imread(image_path)
        logger.debug(f'input image shape: {raw_img.shape}')

        img = preprocess(raw_img, (DETECT_HEIGHT, DETECT_WIDTH))

        # inference
        logger.info('Start inference...')
        res = None
        if args.benchmark:
            logger.info('BENCHMARK mode')
            total_time = 0
            for i in range(args.benchmark_count):
                start = int(round(time.time() * 1000))
                res = detection.run(img[None, None, :, :])
                end = int(round(time.time() * 1000))
                if i != 0:
                    total_time = total_time + (end - start)
                logger.info(f'\tailia processing time {end - start} ms')
            logger.info(f'\taverage time {total_time / (args.benchmark_count-1)} ms')
        else:
            res = detection.run(img[None, None, :, :])

        detections = postprocess(img, raw_img.shape, res)
        decoded = decode(raw_img, detections, sr)
        result_img = visualize(raw_img, decoded)

        # cv2.imshow("QR", result_img)
        # cv2.waitKey()

        savepath = get_savepath(args.savepath, image_path)
        cv2.imwrite(savepath, result_img)
        logger.info(f'saved at : {savepath}')

    logger.info('Script finished successfully.')

def recognize_from_video(detection, sr):
    capture = webcamera_utils.get_capture(args.video)

    # create video writer if savepath is specified as video format
    if args.savepath != SAVE_IMAGE_PATH:
        f_h = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        f_w = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        writer = webcamera_utils.get_writer(args.savepath, f_h, f_w)
    else:
        writer = None

    while(True):
        ret, raw_frame = capture.read()
        if (cv2.waitKey(1) & 0xFF == ord('q')) or not ret:
            break

        frame = preprocess(raw_frame, (DETECT_HEIGHT, DETECT_WIDTH))
        
        res = detection.run(frame[None, None, :, :])
        detections = postprocess(frame, raw_frame.shape, res)
        decoded = decode(raw_frame, detections, sr)
        result_frame = visualize(raw_frame, decoded)

        cv2.imshow('frame', result_frame)

        # save results
        if writer is not None:
            writer.write(result_frame)

    capture.release()
    cv2.destroyAllWindows()
    if writer is not None:
        writer.release()
    logger.info('Script finished successfully.')


def main():
    # model files check and download
    check_and_download_models(DETECT_WEIGHT_PATH, DETECT_MODEL_PATH, REMOTE_PATH)
    check_and_download_models(SR_WEIGHT_PATH, SR_MODEL_PATH, REMOTE_PATH)

    env_id = args.env_id
    detection = ailia.Net(DETECT_MODEL_PATH, DETECT_WEIGHT_PATH, env_id=env_id, memory_mode=args.memory_mode)
    detection.set_input_shape((1, 1, DETECT_WIDTH, DETECT_HEIGHT))

    sr = ailia.Net(SR_MODEL_PATH, SR_WEIGHT_PATH, env_id=env_id, memory_mode=args.memory_mode)
    sr.set_input_shape((1, 1, SR_WIDTH, SR_HEIGHT))

    if args.video is not None:
        recognize_from_video(detection, sr)
    else:
        recognize_from_image(detection, sr)

if __name__ == '__main__':
    main()
