import sys
import time

import numpy as np
import cv2

import ailia

# import original modules
sys.path.append('../../util')
from arg_utils import get_base_parser, update_parser, get_savepath  # noqa: E402
from model_utils import check_and_download_models  # noqa: E402
from classifier_utils import plot_results, print_results  # noqa: E402
from detector_utils import load_image  # noqa: E402C
import webcamera_utils  # noqa: E402
# logger
from logging import getLogger  # noqa: E402

logger = getLogger(__name__)

# ======================
# Parameters
# ======================

WEIGHT_PATH = 'landmarks_classifier_asia_V1_1.onnx'
MODEL_PATH = 'landmarks_classifier_asia_V1_1.onnx.prototxt'
REMOTE_PATH = 'https://storage.googleapis.com/ailia-models/landmarks_classifier_asia/'

IMAGE_PATH = 'image_1.jpg'

IMAGE_SIZE = 321

LABEL_MAP_FILE = 'landmarks_classifier_asia_V1_label_map.csv'

# ======================
# Arguemnt Parser Config
# ======================

parser = get_base_parser(
    'landmarks_classifier_asia', IMAGE_PATH, None
)
parser.add_argument(
    '-n', '--top_k', default=20, type=int,
    help='Use object detection.'
)
parser.add_argument(
    '-k', '--keep_ratio',
    action='store_true',
    help='keep ratio on resize.'
)
args = update_parser(parser)


# ======================
# Secondaty Functions
# ======================

def read_label_map():
    label_map = {}
    label_set = {}
    with open(LABEL_MAP_FILE, encoding='utf-8') as f:
        for i, row in enumerate(f):
            if i == 0:
                continue
            row = row.strip()
            _id, name = row.split(',', 1)
            if 2 < len(name) and name[0] == '"' and name[-1] == '"':
                name = name[1:-1]

            _id = int(_id)
            label_map[_id] = name

            ids = label_set[name] = label_set.get(name, [])
            ids.append(_id)

    return label_map, label_set


# ======================
# Main functions
# ======================

def preprocess(img):
    keep_ratio = args.keep_ratio

    im_h, im_w, _ = img.shape

    if keep_ratio:
        scale = IMAGE_SIZE / max(im_h, im_w)
        ow, oh = int(im_w * scale), int(im_h * scale)
        if ow != im_w or oh != im_h:
            img = cv2.resize(img, (ow, oh), interpolation=cv2.INTER_LINEAR)

        if ow != IMAGE_SIZE or oh != IMAGE_SIZE:
            pad = np.zeros((IMAGE_SIZE, IMAGE_SIZE, 3))
            pad_h, pad_w = pad.shape[:2]
            pad_h = (pad_h - oh) // 2
            pad_w = (pad_w - ow) // 2
            pad[pad_h:pad_h + oh, pad_w:pad_w + ow] = img
            img = pad
    else:
        img = cv2.resize(img, (IMAGE_SIZE, IMAGE_SIZE), interpolation=cv2.INTER_LINEAR)

    img = img / 255
    img = np.expand_dims(img, axis=0)
    img = img.astype(np.float32)

    return img


def post_processing(logits, label_map, label_set):
    top_k = args.top_k

    # The same category is defined more than once. So summarize it.
    upd_logits = np.ones_like(logits) * -1e+3
    for name, ids in label_set.items():
        i = np.argmax(logits[ids])
        _id = ids[i]
        upd_logits[_id] = logits[_id]

    return upd_logits


def predict(net, img):
    img = preprocess(img)

    # feedforward
    output = net.predict([img])

    logits = output[0]

    return logits[0]


def recognize_from_image(net):
    label_map, label_set = read_label_map()

    # input image loop
    for image_path in args.input:
        logger.info(image_path)

        # prepare input data
        img = load_image(image_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # inference
        logger.info('Start inference...')
        if args.benchmark:
            logger.info('BENCHMARK mode')
            total_time_estimation = 0
            for i in range(args.benchmark_count):
                start = int(round(time.time() * 1000))
                logits = predict(net, img)
                end = int(round(time.time() * 1000))
                estimation_time = (end - start)

                # Loggin
                logger.info(f'\tailia processing estimation time {estimation_time} ms')
                if i != 0:
                    total_time_estimation = total_time_estimation + estimation_time

            logger.info(f'\taverage time estimation {total_time_estimation / (args.benchmark_count - 1)} ms')
        else:
            logits = predict(net, img)

        upd_logits = post_processing(logits, label_map, label_set)

        pred = np.expand_dims(upd_logits,axis=0)
        print_results(pred, label_map, top_k=args.top_k)

    logger.info('Script finished successfully.')


def recognize_from_video(net):
    label_map, label_set = read_label_map()

    capture = webcamera_utils.get_capture(args.video)
    # create video writer if savepath is specified as video format
    if args.savepath is not None:
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

        img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        logits = predict(net, img)
        upd_logits = post_processing(logits, label_map, label_set)

        plot_results(frame, np.expand_dims(upd_logits,axis=0), label_map, top_k=args.top_k)

        cv2.imshow('frame', frame)
        frame_shown = True

        # save results
        if writer is not None:
            writer.write(frame)

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
        # video mode
        recognize_from_video(net)
    else:
        # image mode
        recognize_from_image(net)


if __name__ == '__main__':
    main()
