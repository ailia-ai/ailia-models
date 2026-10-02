import os
import sys
import time
import argparse
import numpy as np
import cv2
import ailia
from PIL import Image

sys.path.append('../../util')
from arg_utils import get_base_parser, update_parser, get_savepath  # noqa: E402
from model_utils import check_and_download_models  # noqa: E402
from detector_utils import load_image  # noqa: E402C
from webcamera_utils import get_capture  # noqa: E402

from logging import getLogger   # noqa: E402
logger = getLogger(__name__)


# ======================
# Parameters
# ======================

WEIGHT_PATH = 'Autoware_tlr_SSD_300x300_iter_60000.onnx'
MODEL_PATH = 'Autoware_tlr_SSD_300x300_iter_60000.onnx.prototxt'
REMOTE_PATH = 'https://storage.googleapis.com/ailia-models/traffic-light-recognizer/'

IMAGE_PATH = 'demo.jpg'
SAVE_IMAGE_PATH = 'output.png'


# ======================
# Argument Parser Config
# ======================

parser = get_base_parser(
    'Traffic Light Recognizer',
    IMAGE_PATH,
    SAVE_IMAGE_PATH,
)
parser.add_argument(
    '--onnx',
    action='store_true',
    help='execute onnxruntime version.'
)
parser.add_argument(
    '-w', '--write_json',
    action='store_true',
    help='Flag to output results to json file.'
)
args = update_parser(parser)


# ======================
# Helper Functions
# ======================

def draw_bbox(img, bboxes):
    for bbox in bboxes:
        x1, y1, w, h = bbox.x, bbox.y, bbox.w, bbox.h
        x2 = x1 + w
        y2 = y1 + h

        cv2.putText(
            img,
            '{} ({:.2f})'.format(str(class_name[bbox.category - 1]), bbox.prob),
            (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2,
            cv2.LINE_AA)
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)

    return img


def save_result_json(json_path, bboxes):
    res = []
    for bbox in bboxes:
        res.append({
            'category': str(class_name[bbox.category - 1]),
            'prob': float(bbox.prob),
            'x': float(bbox.x), 'y': float(bbox.y),
            'w': float(bbox.w), 'h': float(bbox.h)
        })
    with open(json_path, 'w') as f:
        json.dump(res, f, indent=2)


# ======================
# Main functions
# ======================

def post_processing(img_shape, boxes, scores, classes):
    score_th = 0.3

    h, w = img_shape

    bboxes = []
    for bbox, score, class_id in zip(boxes, scores, classes):
        if score < score_th:
            break

        x1, y1 = int(bbox[1] * w), int(bbox[0] * h)
        x2, y2 = int(bbox[3] * w), int(bbox[2] * h)

        r = ailia.DetectorObject(
            category=int(class_id),
            prob=score,
            x=x1, y=y1,
            w=x2 - x1, h=y2 - y1
        )
        bboxes.append(r)

    return bboxes


def predict(net, img):
    h, w = img.shape[:2]

    img = img[:, :, ::-1]  # BGR -> RGB
    img = np.expand_dims(img, axis=0)
    img = img.astype(np.uint8)

    if not args.onnx:
        output = net.predict([img])
    else:
        output = net.run(None, {'image_tensor:0': img})

    num_detections, boxes, scores, classes = output

    bboxes = post_processing((h, w), boxes[0], scores[0], classes[0])

    return bboxes


def recognize_from_image(net):
    for image_path in args.input:
        logger.info(image_path)

        img = load_image(image_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)

        logger.info('Start inference...')
        if args.benchmark:
            logger.info('BENCHMARK mode')
            total_time_estimation = 0
            for i in range(args.benchmark_count):
                start = int(round(time.time() * 1000))
                bboxes = predict(net, img)
                end = int(round(time.time() * 1000))
                estimation_time = (end - start)

                logger.info(f'\tailia processing estimation time {estimation_time} ms')
                if i != 0:
                    total_time_estimation = total_time_estimation + estimation_time

            logger.info(f'\taverage time estimation {total_time_estimation / (args.benchmark_count - 1)} ms')
        else:
            bboxes = predict(net, img)

        # res_img = draw_bbox(img, bboxes)

        # savepath = get_savepath(args.savepath, image_path, ext='.png')
        # logger.info(f'saved at : {savepath}')
        # cv2.imwrite(savepath, res_img)

        # if args.write_json:
        #     pred_file = '%s.json' % savepath.rsplit('.', 1)[0]
        #     save_result_json(pred_file, bboxes)

    logger.info('Script finished successfully.')


def main():
    check_and_download_models(WEIGHT_PATH, MODEL_PATH, REMOTE_PATH)

    if not args.onnx:
        memory_mode = ailia.get_memory_mode(reduce_constant=True, reduce_interstage=True)
        net = ailia.Net(MODEL_PATH, WEIGHT_PATH, env_id=args.env_id, memory_mode=memory_mode)
    else:
        import onnxruntime
        net = onnxruntime.InferenceSession(WEIGHT_PATH)

    recognize_from_image(args.input, net)


if __name__ == '__main__':
    main()
