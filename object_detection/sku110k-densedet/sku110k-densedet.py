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
from image_utils import normalize_image  # noqa: E402C
from webcamera_utils import get_capture, get_writer  # noqa: E402
# logger
from logging import getLogger  # noqa: E402

logger = getLogger(__name__)

# ======================
# Parameters
# ======================

WEIGHT_PATH = 'DenseDet.onnx'
MODEL_PATH = 'DenseDet.onnx.prototxt'
REMOTE_PATH = \
    'https://storage.googleapis.com/ailia-models/sku100k-densedet/'

IMAGE_PATH = 'demo.jpg'
SAVE_IMAGE_PATH = 'output.png'

IMAGE_HEIGHT = 2560
IMAGE_WIDTH = 2560

THRESHOLD = 0.3

# ======================
# Arguemnt Parser Config
# ======================

parser = get_base_parser(
    'SKU110K-DenseDet', IMAGE_PATH, SAVE_IMAGE_PATH
)
parser.add_argument(
    '-th', '--threshold',
    default=THRESHOLD, type=float,
    help='The detection threshold.'
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

def draw_bbox(
        img, bboxes, thickness=1, font_scale=0.5):
    color = (0, 255, 0)
    for bbox in bboxes:
        bbox_int = bbox.astype(np.int32)
        left_top = (bbox_int[0], bbox_int[1])
        right_bottom = (bbox_int[2], bbox_int[3])
        cv2.rectangle(
            img, left_top, right_bottom, color, thickness=thickness)
        text = '{:.02f}'.format(bbox[-1])
        cv2.putText(
            img, text, (bbox_int[0], bbox_int[1] - 2),
            cv2.FONT_HERSHEY_COMPLEX, font_scale, color)

    return img


def save_result_json(json_path, img, bboxes):
    res = []
    for bbox in bboxes:
        res.append({
            'val': float(bbox[-1]),
            'x1': float(bbox[0]), 'y1': float(bbox[1]),
            'x2': float(bbox[2]), 'y2': float(bbox[3])
        })
    with open(json_path, 'w') as f:
        json.dump(res, f, indent=2)


# ======================
# Main functions
# ======================

def preprocess(img, image_shape):
    h, w = image_shape
    im_h, im_w, _ = img.shape

    # keep_aspect
    scale = min(h / im_h, w / im_w)
    ow, oh = int(im_w * scale + 0.5), int(im_h * scale + 0.5)
    if ow != im_w or oh != im_h:
        img = cv2.resize(img, (ow, oh), interpolation=cv2.INTER_LINEAR)

    img = normalize_image(img, normalize_type='ImageNet')

    pad_img = np.zeros((h, w, 3), dtype=img.dtype)
    pad_img[:oh, :ow, ...] = img
    img = pad_img

    img = img.transpose(2, 0, 1)  # HWC -> CHW
    img = np.expand_dims(img, axis=0)
    img = img.astype(np.float32)

    return img, scale


def post_processing(bboxes, scale):
    score_thr = args.threshold

    bboxes[:, :4] = bboxes[:, :4] / scale

    scores = bboxes[:, -1]
    inds = scores > score_thr
    bboxes = bboxes[inds, :]

    return bboxes


def predict(net, img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    shape = (IMAGE_HEIGHT, IMAGE_WIDTH)
    img, scale = preprocess(img, shape)

    # feedforward
    if not args.onnx:
        output = net.predict([img])
    else:
        output = net.run(None, {'img': img})

    bboxes, _ = output

    bboxes = post_processing(bboxes, scale)

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
    # model files check and download
    check_and_download_models(WEIGHT_PATH, MODEL_PATH, REMOTE_PATH)

    env_id = args.env_id

    # initialize
    if not args.onnx:
        logger.info("This model requires 10GB or more memory.")
        net = ailia.Net(MODEL_PATH, WEIGHT_PATH, env_id=env_id, memory_mode=args.memory_mode)
    else:
        import onnxruntime
        net = onnxruntime.InferenceSession(WEIGHT_PATH)

    if args.video is not None:
        recognize_from_video(net)
    else:
        recognize_from_image(net)


if __name__ == '__main__':
    main()
