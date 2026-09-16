import sys
import time

import numpy as np
import cv2

import ailia

# import original modules
sys.path.append('../../util')
from arg_utils import get_base_parser, update_parser, get_savepath  # noqa
from model_utils import check_and_download_models  # noqa
from nms_utils import nms_boxes  # noqa
from detector_utils import load_image, plot_results  # noqa
from detector_utils import letterbox_convert, reverse_letterbox  # noqa: E402
from webcamera_utils import get_capture, get_writer  # noqa
# logger
from logging import getLogger  # noqa

logger = getLogger(__name__)

# ======================
# Parameters
# ======================

WEIGHT_PATH = 'face-detection-adas-0001.onnx'
MODEL_PATH = 'face-detection-adas-0001.onnx.prototxt'
REMOTE_PATH = 'https://storage.googleapis.com/ailia-models/face-detection-adas/'

IMAGE_PATH = 'input.png'
SAVE_IMAGE_PATH = 'output.png'
PRIORBOX_PATH = 'mbox_priorbox.npy'

IMAGE_HEIGHT = 384
IMAGE_WIDTH = 672

THRESHOLD = 0.5
IOU = 0.5

# ======================
# Arguemnt Parser Config
# ======================

parser = get_base_parser('face-detection-adas', IMAGE_PATH, SAVE_IMAGE_PATH)
parser.add_argument(
    '-th', '--threshold',
    default=THRESHOLD, type=float,
    help='object confidence threshold'
)
parser.add_argument(
    '-iou', '--iou',
    default=IOU, type=float,
    help='IOU threshold for NMS'
)
parser.add_argument(
    '--onnx',
    action='store_true',
    help='execute onnxruntime version.'
)
args = update_parser(parser)


# ======================
# Secondaty Functions
# ======================

def convert_to_detector_object(bboxes, scores, im_w, im_h):
    detector_object = []
    for i in range(len(bboxes)):
        (x1, y1, x2, y2) = bboxes[i]
        score = scores[i]

        r = ailia.DetectorObject(
            category="",
            prob=score,
            x=x1 / im_w,
            y=y1 / im_h,
            w=(x2 - x1) / im_w,
            h=(y2 - y1) / im_h,
        )
        detector_object.append(r)

    return detector_object


# ======================
# Main functions
# ======================

def preprocess(img):
    # img = cv2.resize(img, (IMAGE_WIDTH, IMAGE_HEIGHT), interpolation=cv2.INTER_LINEAR)
    img = letterbox_convert(img, (IMAGE_HEIGHT, IMAGE_WIDTH))

    img = img.transpose(2, 0, 1)  # HWC -> CHW
    img = np.expand_dims(img, axis=0)
    img = img.astype(np.float32)

    return img


def decode_bbox(mbox_loc, mbox_priorbox, variances):
    mbox_loc = mbox_loc.reshape(-1, 4)
    mbox_priorbox = mbox_priorbox.reshape(-1, 4)
    variances = variances.reshape(-1, 4)

    prior_width = mbox_priorbox[:, 2] - mbox_priorbox[:, 0]
    prior_height = mbox_priorbox[:, 3] - mbox_priorbox[:, 1]
    prior_center_x = 0.5 * (mbox_priorbox[:, 2] + mbox_priorbox[:, 0])
    prior_center_y = 0.5 * (mbox_priorbox[:, 3] + mbox_priorbox[:, 1])

    decode_bbox_center_x = mbox_loc[:, 0] * prior_width * variances[:, 0]
    decode_bbox_center_x += prior_center_x
    decode_bbox_center_y = mbox_loc[:, 1] * prior_height * variances[:, 1]
    decode_bbox_center_y += prior_center_y
    decode_bbox_width = np.exp(mbox_loc[:, 2] * variances[:, 2])
    decode_bbox_width *= prior_width
    decode_bbox_height = np.exp(mbox_loc[:, 3] * variances[:, 3])
    decode_bbox_height *= prior_height

    decode_bbox_xmin = decode_bbox_center_x - 0.5 * decode_bbox_width
    decode_bbox_ymin = decode_bbox_center_y - 0.5 * decode_bbox_height
    decode_bbox_xmax = decode_bbox_center_x + 0.5 * decode_bbox_width
    decode_bbox_ymax = decode_bbox_center_y + 0.5 * decode_bbox_height

    bboxes = np.concatenate((
        decode_bbox_xmin[:, None],
        decode_bbox_ymin[:, None],
        decode_bbox_xmax[:, None],
        decode_bbox_ymax[:, None]), axis=-1)

    # bboxes = np.minimum(np.maximum(bboxes, 0.0), 1.0)

    return bboxes


def predict(model_info, img):
    score_th = args.threshold
    nms_th = args.iou

    net = model_info['net']
    prior_box = model_info['prior_box']

    preprocess_img = preprocess(img)

    # feedforward
    if not args.onnx:
        output = net.predict([preprocess_img])
    else:
        output = net.run(None, {'data': preprocess_img})
    mbox_loc, mbox_conf = output

    bboxes = decode_bbox(mbox_loc[0], prior_box[0], prior_box[1])

    mbox_conf = mbox_conf[0].reshape(-1, 2)
    cls_idx = 1
    i = mbox_conf[:, cls_idx] >= score_th
    bboxes = bboxes[i]
    scores = mbox_conf[i][:, 1]

    bboxes[:, [0, 2]] = bboxes[:, [0, 2]] * IMAGE_HEIGHT
    bboxes[:, [1, 3]] = bboxes[:, [1, 3]] * IMAGE_WIDTH

    i = nms_boxes(bboxes, scores, nms_th)
    bboxes = bboxes[i].astype(int)
    scores = scores[i]

    detect_object = convert_to_detector_object(bboxes, scores, IMAGE_HEIGHT, IMAGE_WIDTH)
    detect_object = reverse_letterbox(detect_object, img, (IMAGE_HEIGHT, IMAGE_WIDTH))

    return detect_object


def recognize_from_image(model_info):
    # input image loop
    for image_path in args.input:
        logger.info(image_path)

        # prepare input data
        img = load_image(image_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)

        # inference
        logger.info('Start inference...')
        if args.benchmark:
            logger.info('BENCHMARK mode')
            total_time_estimation = 0
            for i in range(args.benchmark_count):
                start = int(round(time.time() * 1000))
                pred = predict(model_info, img)
                end = int(round(time.time() * 1000))
                estimation_time = (end - start)

                # Loggin
                logger.info(f'\tailia processing estimation time {estimation_time} ms')
                if i != 0:
                    total_time_estimation = total_time_estimation + estimation_time

            logger.info(f'\taverage time estimation {total_time_estimation / (args.benchmark_count - 1)} ms')
        else:
            pred = predict(model_info, img)

        # plot result
        detect_object = pred
        res_img = plot_results(detect_object, img)

        # plot result
        savepath = get_savepath(args.savepath, image_path, ext='.png')
        logger.info(f'saved at : {savepath}')
        cv2.imwrite(savepath, res_img)

    logger.info('Script finished successfully.')


def recognize_from_video(model_info):
    video_file = args.video if args.video else args.input[0]
    capture = get_capture(video_file)
    assert capture.isOpened(), 'Cannot capture source'

    # create video writer if savepath is specified as video format
    f_h = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    f_w = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    if args.savepath != SAVE_IMAGE_PATH:
        logger.warning(
            'currently, video results cannot be output correctly...'
        )
        writer = get_writer(args.savepath, f_h, f_w)
    else:
        writer = None

    while True:
        ret, frame = capture.read()
        if (cv2.waitKey(1) & 0xFF == ord('q')) or not ret:
            break

        # inference
        img = frame
        pred = predict(model_info, img)

        # plot result
        detect_object = pred
        res_img = plot_results(detect_object, img)

        # show
        cv2.imshow('frame', res_img)

        # save results
        if writer is not None:
            writer.write(res_img.astype(np.uint8))

    capture.release()
    cv2.destroyAllWindows()
    if writer is not None:
        writer.release()

    logger.info('Script finished successfully.')


def main():
    # model files check and download
    check_and_download_models(WEIGHT_PATH, MODEL_PATH, REMOTE_PATH)

    env_id = args.env_id

    # initialize
    if not args.onnx:
        net = ailia.Net(MODEL_PATH, WEIGHT_PATH, env_id=env_id, memory_mode=args.memory_mode)
    else:
        import onnxruntime
        net = onnxruntime.InferenceSession(WEIGHT_PATH)

    prior_box = np.squeeze(np.load(PRIORBOX_PATH))

    model_info = {
        'net': net,
        'prior_box': prior_box,
    }

    if args.video is not None:
        recognize_from_video(model_info)
    else:
        recognize_from_image(model_info)


if __name__ == '__main__':
    main()
