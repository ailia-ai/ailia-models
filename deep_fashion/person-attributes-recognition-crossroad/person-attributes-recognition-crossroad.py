import cv2
import sys
import time
import numpy as np

import ailia

# import original modules
sys.path.append('../../util')
# logger
from logging import getLogger  # noqa: E402

import webcamera_utils  # noqa: E402
from image_utils import imread, load_image  # noqa: E402
from model_utils import check_and_download_models  # noqa: E402
from arg_utils import get_base_parser, update_parser, get_savepath  # noqa: E402

logger = getLogger(__name__)

# ======================
# Parameters
# ======================
REMOTE_PATH = "https://storage.googleapis.com/ailia-models/person-attributes-recognition-crossroad/"

IMAGE_PATH = 'input.jpg'
SAVE_IMAGE_PATH = 'output.jpg'

SLEEP_TIME = 0  # for video mode


# ======================
# Arguemnt Parser Config
# ======================
parser = get_base_parser( 'person-attributes-recognition-crossroad model', IMAGE_PATH, SAVE_IMAGE_PATH
)
parser.add_argument(
    '-m', '--model', default='0230', choices=['0230','0234']
)
args = update_parser(parser)

WEIGHT_PATH = "person-attributes-recognition-crossroad-{}.onnx".format(args.model)
MODEL_PATH = "person-attributes-recognition-crossroad-{}.onnx.prototxt".format(args.model)

YOLOX_REMOTE_PATH = 'https://storage.googleapis.com/ailia-models/yolox/'

YOLOX_MODEL_NAME = "yolox_s"
YOLOX_WEIGHT_PATH = YOLOX_MODEL_NAME + ".opt.onnx"
YOLOX_MODEL_PATH  = YOLOX_MODEL_NAME + ".opt.onnx.prototxt"
YOLOX_SCORE_THR = 0.4
YOLOX_NMS_THR = 0.45

MAX_CLASS_COUNT = 7
RECT_WIDTH = 200
RECT_HEIGHT = 20
RECT_MARGIN = 2

def hsv_to_rgb(h, s, v):
    bgr = cv2.cvtColor(
        np.array([[[h, s, v]]], dtype=np.uint8), cv2.COLOR_HSV2BGR)[0][0]
    return (int(bgr[0]), int(bgr[1]), int(bgr[2]), 255)


def plot_results(i, x, y, w, h, input_image, classifier, labels, top_k=MAX_CLASS_COUNT, logging=True):
    color = hsv_to_rgb(256 * i / (len(labels)+1), 128, 255)
    cv2.rectangle(input_image, (x, y), (x + w, y + h), color, thickness=2)

    y = y + (h - (RECT_HEIGHT + RECT_MARGIN) * top_k) //2

    if logging:
        print('==============================================================')
        print(f'class_count={top_k}')
    for idx in range(top_k):
        if logging:
            print(f'+ idx={idx}')
            print(f'  category={[idx]}['
                f'{labels[idx]} ]')
            print(f'  prob={classifier[idx]}')

        text = f'{labels[idx]} {int(classifier[idx]*100)/100}'

        color = hsv_to_rgb(256 * idx / (len(labels)+1), 128, 255)

        cv2.rectangle(input_image, (x, y), (x + RECT_WIDTH, y + RECT_HEIGHT), color, thickness=-1)
        text_position = (x+4, y+int(RECT_HEIGHT/2)+4)

        color = (0,0,0)
        fontScale = 0.5

        cv2.putText(
            input_image,
            text,
            text_position,
            cv2.FONT_HERSHEY_SIMPLEX,
            fontScale,
            color,
            1
        )

        y=y + RECT_HEIGHT + RECT_MARGIN

def crop_and_resize(raw_data, x, y, w, h):
    # keep aspect
    if w * 2 < h:
        nw = h // 2
        x = x + (w - nw) // 2
        w = nw
    else:
        nh = w * 2
        y = y + (h - nh) // 2
        h = nh

    # crop
    input_data = np.zeros((h, w, 3))

    # zero padding
    iw = raw_data.shape[1]
    ih = raw_data.shape[0]
    if x < 0:
        w = w + x
        x = 0
    if y < 0:
        h = h + y
        y = 0
    if x + w > iw:
        w = iw - x
    if y + h > ih:
        h = ih - y
    input_data[0:h, 0:w, :] = raw_data[y:y+h, x:x+w, :]

    # resize
    input_data = cv2.resize(input_data, (80, 160))
    return input_data, x, y, w, h

# ======================
# Main functions
# ======================

def process_frame(raw_data, yolox_model, net):
    # detect person
    yolox_model.compute(raw_data, YOLOX_SCORE_THR, YOLOX_NMS_THR)
    res_img = raw_data.copy()
    count = yolox_model.get_object_count()
    for i in range(count):
        obj = yolox_model.get_object(i)
        if not obj.category == 0:
            continue
            
        # get person
        img_h,img_w,_ = raw_data.shape
        x = int(obj.x * img_w)
        y = int(obj.y * img_h)
        w = int(obj.w * img_w)
        h = int(obj.h * img_h)
        input_data, x, y, w, h = crop_and_resize(raw_data, x, y, w, h)

        # get attribute
        if args.model == '0230':
            pass
        else:
            input_data = input_data.transpose(2,0,1)

        logger.info('Start inference...')
        if args.benchmark:
            logger.info('BENCHMARK mode')
            for j in range(args.benchmark_count):
                start = int(round(time.time() * 1000))
                result = net.run(input_data)
                end = int(round(time.time() * 1000))
                logger.info(f'\tailia processing time {end - start} ms')
        else:
            result = net.run(input_data)

        if args.model == '0230':
            classes = result[0][0][0][0]
        else:
            classes = result[0][0][:8]
        labels = ['is_male','has_bag','has_backpack','has_hat','has_longsleeves','has_longpants','has_longhair','has_coat_jacket']
        
        # show results
        plot_results(i, x, y, w, h, res_img, classes, np.array(labels), 7)

    return res_img


def recognize_from_image(yolox_model, net):
    # input image loop
    for image_path in args.input:
        # prepare input data
        logger.info(image_path)
        raw_data = imread(image_path)

        res_img = process_frame(raw_data, yolox_model, net)

        savepath = get_savepath(args.savepath, image_path)
        logger.info(f'saved at : {savepath}')
        cv2.imwrite(savepath, res_img)

    logger.info('Script finished successfully.')


def recognize_from_video(yolox_model, net):

    capture = webcamera_utils.get_capture(args.video)

    # create video writer if savepath is specified as video format
    if args.savepath != SAVE_IMAGE_PATH:
        f_h = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        f_w = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        writer = webcamera_utils.get_writer(args.savepath, f_h, f_w)
    else:
        writer = None

    frame_shown = False
    while(True):
        ret, frame = capture.read()
        if (cv2.waitKey(1) & 0xFF == ord('q')) or not ret:
            break
        if frame_shown and cv2.getWindowProperty('frame', cv2.WND_PROP_VISIBLE) == 0:
            break

        res_img = process_frame(frame, yolox_model, net)

        cv2.imshow('frame', res_img)
        frame_shown = True
        time.sleep(SLEEP_TIME)

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
    check_and_download_models(YOLOX_WEIGHT_PATH, YOLOX_MODEL_PATH, YOLOX_REMOTE_PATH)

    # net initialize
    net = ailia.Net(MODEL_PATH, WEIGHT_PATH, env_id=args.env_id, memory_mode=args.memory_mode)
    yolox = ailia.Detector(
            YOLOX_MODEL_PATH,
            YOLOX_WEIGHT_PATH,
            80,
            format=ailia.NETWORK_IMAGE_FORMAT_BGR,
            channel=ailia.NETWORK_IMAGE_CHANNEL_FIRST,
            range=ailia.NETWORK_IMAGE_RANGE_U_INT8,
            algorithm=ailia.DETECTOR_ALGORITHM_YOLOX,
            env_id=args.env_id,
            memory_mode=args.memory_mode)

    if args.video is not None:
        # video mode
        recognize_from_video(yolox,net)
    else:
        # image mode
        recognize_from_image(yolox,net)


if __name__ == '__main__':
    main()
