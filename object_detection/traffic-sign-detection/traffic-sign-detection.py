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

WEIGHT_RESNET50_PATH = 'faster_rcnn_resnet50.onnx'
MODEL_RESNET50_PATH = 'faster_rcnn_resnet50.onnx.prototxt'
WEIGHT_INCEPTION_RESNET_PATH = 'faster_rcnn_inception_resnet_v2_atrous.onnx'
MODEL_INCEPTION_RESNET_PATH = 'faster_rcnn_inception_resnet_v2_atrous.onnx.prototxt'
REMOTE_PATH = 'https://storage.googleapis.com/ailia-models/traffic-sign-detection/'

IMAGE_PATH = 'demo.jpg'
SAVE_IMAGE_PATH = 'output.png'

class_name = [
    'prohibitory',
    'mandatory',
    'danger',
]

# ======================
# Arguemnt Parser Config
# ======================

parser = get_base_parser(
    'Traffic Sign Detection', IMAGE_PATH, SAVE_IMAGE_PATH
)
parser.add_argument(
    '-m', '--model_type', default='resnet50', choices=('resnet50', 'inception_resnet'),
    help='model type'
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
# Secondaty Functions
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

    # feedforward
    if not args.onnx:
        output = net.predict([img])
    else:
        output = net.run(None, {'image_tensor:0': img})

    num_detections, boxes, scores, classes = output

    bboxes = post_processing((h, w), boxes[0], scores[0], classes[0])

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

        res_img = draw_bbox(img, bboxes)

        # plot result
        savepath = get_savepath(args.savepath, image_path, ext='.png')
        logger.info(f'saved at : {savepath}')
        cv2.imwrite(savepath, res_img)

        # write prediction
        if args.write_json:
            pred_file = '%s.json' % savepath.rsplit('.', 1)[0]
            save_result_json(pred_file, bboxes)

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
        res_img = draw_bbox(frame, bboxes)

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
    dic_model = {
        'resnet50': (WEIGHT_RESNET50_PATH, MODEL_RESNET50_PATH),
        'inception_resnet': (WEIGHT_INCEPTION_RESNET_PATH, MODEL_INCEPTION_RESNET_PATH),
    }
    weight_path, model_path = dic_model[args.model_type]

    # model files check and download
    check_and_download_models(weight_path, model_path, REMOTE_PATH)

    env_id = args.env_id

    # initialize
    if not args.onnx:
        net = ailia.Net(model_path, weight_path, env_id=env_id, memory_mode=args.memory_mode)
    else:
        import onnxruntime
        net = onnxruntime.InferenceSession(weight_path)

    if args.video is not None:
        recognize_from_video(net)
    else:
        recognize_from_image(net)


if __name__ == '__main__':
    main()
