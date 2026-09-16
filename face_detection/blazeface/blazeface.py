import sys
import time

import ailia
import cv2
import numpy as np

import blazeface_utils as but

# import original modules
sys.path.append('../../util')
# logger
from logging import getLogger  # noqa: E402

import webcamera_utils  # noqa: E402
from detector_utils import load_image  # noqa: E402C
from image_utils import imread, normalize_image  # noqa: E402C
from model_utils import check_and_download_models  # noqa: E402
from arg_utils import get_base_parser, get_savepath, update_parser  # noqa: E402

logger = getLogger(__name__)

# ======================
# PARAMETERS
# ======================

WEIGHT_PATH_FRONT = 'blazeface.onnx'
MODEL_PATH_FRONT = 'blazeface.onnx.prototxt'
ANCHOR_PATH_FRONT = 'anchors.npy'
WEIGHT_PATH_BACK = 'blazefaceback.onnx'
MODEL_PATH_BACK = 'blazefaceback.onnx.prototxt'
ANCHOR_PATH_BACK = 'anchorsback.npy'
REMOTE_PATH = "https://storage.googleapis.com/ailia-models/blazeface/"

IMAGE_PATH = 'input.png'
SAVE_IMAGE_PATH = 'result.png'

IMAGE_HEIGHT_FRONT = 128
IMAGE_WIDTH_FRONT = 128
IMAGE_HEIGHT_BACK = 256
IMAGE_WIDTH_BACK = 256

# ======================
# Arguemnt Parser Config
# ======================

parser = get_base_parser(
    'BlazeFace is a fast and light-weight face detector.',
    IMAGE_PATH,
    SAVE_IMAGE_PATH,
)
parser.add_argument('-bk', '--back', action='store_true')
parser.add_argument(
    '-w', '--write_json',
    action='store_true',
    help='Flag to output results to json file.'
)
args = update_parser(parser)


# ======================
# Main functions
# ======================

def preprocess(img, image_shape):
    h, w = image_shape
    im_h, im_w, _ = img.shape

    r = min(h / im_h, w / im_w)
    oh, ow = int(im_h * r), int(im_w * r)

    resized_img = cv2.resize(
        img,
        (ow, oh),
        interpolation=cv2.INTER_LINEAR,
    )

    data = np.zeros((h, w, 3), dtype=np.uint8)
    ph, pw = (h - oh) // 2, (w - ow) // 2
    data[ph: ph + oh, pw: pw + ow] = resized_img

    data = normalize_image(data, '127.5')

    data = data.transpose((2, 0, 1))
    data = np.expand_dims(data, axis=0)
    data = data.astype(np.float32)

    return data, (ph, pw), (oh, ow)


def recognize_from_image(net):
    if args.back == True:
        IMAGE_HEIGHT = IMAGE_HEIGHT_BACK
        IMAGE_WIDTH = IMAGE_WIDTH_BACK
        ANCHOR_PATH = ANCHOR_PATH_BACK
    else:
        IMAGE_HEIGHT = IMAGE_HEIGHT_FRONT
        IMAGE_WIDTH = IMAGE_WIDTH_FRONT
        ANCHOR_PATH = ANCHOR_PATH_FRONT

    # input image loop
    for image_path in args.input:
        # prepare input data
        logger.info(image_path)
        org_img = imread(image_path, cv2.IMREAD_COLOR)

        input_data = load_image(image_path)
        input_data = cv2.cvtColor(input_data, cv2.COLOR_BGRA2RGB)

        input_data, pad_hw, resized_hw = preprocess(input_data, (IMAGE_HEIGHT, IMAGE_WIDTH))

        # inference
        logger.info('Start inference...')
        if args.benchmark:
            logger.info('BENCHMARK mode')
            total_time = 0
            for i in range(args.benchmark_count):
                start = int(round(time.time() * 1000))
                preds_ailia = net.predict([input_data])
                end = int(round(time.time() * 1000))
                logger.info(f'\tailia processing time {end - start} ms')
                if i != 0:
                    total_time = total_time + (end - start)
            logger.info(f'\taverage time {total_time / (args.benchmark_count - 1)} ms')
        else:
            preds_ailia = net.predict([input_data])

        # post-processing
        detections = but.postprocess(preds_ailia, anchor_path=ANCHOR_PATH, back=args.back)

        # remove padding
        pad_x = pad_hw[1] / IMAGE_WIDTH
        pad_y = pad_hw[0] / IMAGE_HEIGHT
        resized_x = resized_hw[1] / IMAGE_WIDTH
        resized_y = resized_hw[0] / IMAGE_HEIGHT
        for d in detections:
            d[:, [1, 3, 4, 6, 8, 10, 12, 14]] = (d[:, [1, 3, 4, 6, 8, 10, 12, 14]] - pad_x) / resized_x
            d[:, [0, 2, 5, 7, 9, 11, 13, 15]] = (d[:, [0, 2, 5, 7, 9, 11, 13, 15]] - pad_y) / resized_y

        # generate detections
        savepath = get_savepath(args.savepath, image_path)
        logger.info(f'saved at : {savepath}')
        for detection in detections:
            but.plot_detections(org_img, detection, save_image_path=savepath)

        if args.write_json:
            json_file = '%s.json' % savepath.rsplit('.', 1)[0]
            but.save_json(json_file, org_img, detections)

    logger.info('Script finished successfully.')


def recognize_from_video(net):
    if args.back == True:
        IMAGE_HEIGHT = IMAGE_HEIGHT_BACK
        IMAGE_WIDTH = IMAGE_WIDTH_BACK
        ANCHOR_PATH = ANCHOR_PATH_BACK
    else:
        IMAGE_HEIGHT = IMAGE_HEIGHT_FRONT
        IMAGE_WIDTH = IMAGE_WIDTH_FRONT
        ANCHOR_PATH = ANCHOR_PATH_FRONT

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

        input_image, input_data = webcamera_utils.preprocess_frame(
            frame, IMAGE_HEIGHT, IMAGE_WIDTH, normalize_type='127.5'
        )

        # inference
        input_blobs = net.get_input_blob_list()
        net.set_input_blob_data(input_data, input_blobs[0])
        net.update()
        preds_ailia = net.get_results()

        # postprocessing
        detections = but.postprocess(preds_ailia, anchor_path=ANCHOR_PATH, back=args.back)
        but.show_result(input_image, detections)

        # remove padding
        dh = input_image.shape[0]
        dw = input_image.shape[1]
        sh = frame.shape[0]
        sw = frame.shape[1]
        input_image = input_image[(dh - sh) // 2:(dh - sh) // 2 + sh, (dw - sw) // 2:(dw - sw) // 2 + sw, :]

        cv2.imshow('frame', input_image)
        frame_shown = True

        # save results
        if writer is not None:
            writer.write(input_image)

    capture.release()
    cv2.destroyAllWindows()
    if writer is not None:
        writer.release()

    logger.info('Script finished successfully.')


def main():
    # model files check and download
    if args.back == True:
        check_and_download_models(WEIGHT_PATH_BACK, MODEL_PATH_BACK, REMOTE_PATH)
    else:
        check_and_download_models(WEIGHT_PATH_FRONT, MODEL_PATH_FRONT, REMOTE_PATH)

    # net initialize
    if args.back == True:
        net = ailia.Net(MODEL_PATH_BACK, WEIGHT_PATH_BACK, env_id=args.env_id, memory_mode=args.memory_mode)
    else:
        net = ailia.Net(MODEL_PATH_FRONT, WEIGHT_PATH_FRONT, env_id=args.env_id, memory_mode=args.memory_mode)

    if args.video is not None:
        # video mode
        recognize_from_video(net)
    else:
        # image mode
        recognize_from_image(net)


if __name__ == '__main__':
    main()
