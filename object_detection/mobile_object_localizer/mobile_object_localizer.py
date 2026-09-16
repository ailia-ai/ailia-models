import sys
import time

import numpy as np
import cv2
import json

import ailia

# import original modules
sys.path.append('../../util')
from arg_utils import get_base_parser, update_parser, get_savepath  # noqa: E402
from model_utils import check_and_download_models  # noqa: E402
from detector_utils import load_image  # noqa: E402C
from webcamera_utils import get_capture, get_writer  # noqa: E402
# logger
from logging import getLogger  # noqa: E402

logger = getLogger(__name__)

# ======================
# Parameters
# ======================

WEIGHT_PATH = 'mobile_object_localizer.onnx'
MODEL_PATH = 'mobile_object_localizer.onnx.prototxt'
REMOTE_PATH = \
    'https://storage.googleapis.com/ailia-models/mobile_object_localizer/'

IMAGE_PATH = 'demo.jpg'
SAVE_IMAGE_PATH = 'output.png'

IMAGE_HEIGHT = 192
IMAGE_WIDTH = 192

THRESHOLD = 0.3

# ======================
# Arguemnt Parser Config
# ======================

parser = get_base_parser(
    'mobile_object_localizer_v1', IMAGE_PATH, SAVE_IMAGE_PATH
)
parser.add_argument(
    '-th', '--threshold',
    default=THRESHOLD, type=float,
    help='The detection threshold for yolo. (default: ' + str(THRESHOLD) + ')'
)
parser.add_argument(
    '-w', '--write_json',
    action='store_true',
    help='Flag to output results to json file.'
)
args = update_parser(parser)


# ======================
# Secondaty Functions
# ======================

def draw_detections(img, bboxes):
    h, w = img.shape[:2]

    color = (0, 255, 0)
    for bbox in bboxes:
        box = bbox[:4].astype(np.int32)
        det_score = int(100 * bbox[4])

        size = min([h, w]) * 0.002
        text_thickness = int(min([h, w]) * 0.004)
        textSize = cv2.getTextSize(
            text=f'det {det_score}%',
            fontFace=cv2.FONT_HERSHEY_SIMPLEX,
            fontScale=size,
            thickness=text_thickness
        )[0][1] * 1.6

        cv2.rectangle(img, (box[0], box[1]), (box[2], box[3]), color, text_thickness)
        cv2.putText(
            img, f'det: {det_score}%', (box[0], box[1] + int(textSize * 0.8)),
            cv2.FONT_HERSHEY_SIMPLEX, size, color, text_thickness, cv2.LINE_AA)

    return img


def save_result_json(json_path, img, bboxes):
    h, w = img.shape[:2]
    res = []
    for bbox in bboxes:
        box = bbox[:4]
        det_score = 100.0 * bbox[4]
        res.append({
            'det': det_score,
            'x1': box[0], 'y1': box[1],
            'x2': box[2], 'y2': box[3]
        })
    with open(json_path, 'w') as f:
        json.dump(res, f, indent=2)


# ======================
# Main functions
# ======================

def preprocess(img, image_shape):
    h, w = image_shape

    img = cv2.resize(img, (w, h))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    img = img.transpose(2, 0, 1)  # HWC -> CHW
    img = np.expand_dims(img, axis=0)
    img = img.astype(np.uint8)

    return img


def post_processing(boxes, scores, num_objects, image_shape):
    score_thr = args.threshold

    im_h, im_w = image_shape

    results = []
    for i in range(int(num_objects)):
        if scores[i] >= score_thr:
            y1 = (im_h * boxes[i][0]).astype(int)
            y2 = (im_h * boxes[i][2]).astype(int)
            x1 = (im_w * boxes[i][1]).astype(int)
            x2 = (im_w * boxes[i][3]).astype(int)

            bbox = np.array([x1, y1, x2, y2, scores[i]])
            results.append(bbox)

    if results:
        results = np.stack(results)
    else:
        results = np.zeros((0, 5))

    return results


def predict(net, img):
    shape = (IMAGE_HEIGHT, IMAGE_WIDTH)
    im_h, im_w = img.shape[:2]
    img = preprocess(img, shape)

    # feedforward
    output = net.predict([img])

    boxes, classes, scores, num_objects = output

    bboxes = post_processing(boxes[0], scores[0], num_objects[0], (im_h, im_w))

    return bboxes


def recognize_from_image(net):
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
                bboxes = predict(net, img)
                end = int(round(time.time() * 1000))
                estimation_time = (end - start)

                # Loggin
                logger.info(f'\tailia processing estimation time {estimation_time} ms')
                if i != 0:
                    total_time_estimation = total_time_estimation + estimation_time

            logger.info(f'\taverage time estimation {total_time_estimation / (args.benchmark_count - 1)} ms')
        else:
            bboxes = predict(net, img)

        res_img = draw_detections(img, bboxes)

        # plot result
        savepath = get_savepath(args.savepath, image_path, ext='.png')
        logger.info(f'saved at : {savepath}')
        cv2.imwrite(savepath, res_img)

        # write prediction
        if args.write_json:
            pred_file = '%s.json' % savepath.rsplit('.', 1)[0]
            save_result_json(pred_file, img, bboxes)

    logger.info('Script finished successfully.')


def recognize_from_video(net):
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

    frame_shown = False 
    while True:
        ret, frame = capture.read()
        if (cv2.waitKey(1) & 0xFF == ord('q')) or not ret:
            break
        if frame_shown and cv2.getWindowProperty('frame', cv2.WND_PROP_VISIBLE) == 0:
            break

        # inference
        bboxes = predict(net, frame)

        # plot result
        res_img = draw_detections(frame, bboxes)

        # show
        cv2.imshow('frame', res_img)
        frame_shown = True

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
    net = ailia.Net(MODEL_PATH, WEIGHT_PATH, env_id=env_id, memory_mode=args.memory_mode)

    if args.video is not None:
        recognize_from_video(net)
    else:
        recognize_from_image(net)


if __name__ == '__main__':
    main()
