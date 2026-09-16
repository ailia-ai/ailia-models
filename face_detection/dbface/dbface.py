import time
import sys

import numpy as np
import cv2

import ailia
import dbface_utils

# import original modules
sys.path.append('../../util')
from arg_utils import get_base_parser, update_parser, get_savepath  # noqa: E402
from model_utils import check_and_download_models  # noqa: E402
from detector_utils import load_image  # noqa: E402
import webcamera_utils  # noqa: E402

# logger
from logging import getLogger   # noqa: E402
logger = getLogger(__name__)


# ======================
# Parameters
# ======================
WEIGHT_PATH = 'dbface_pytorch.onnx'
MODEL_PATH = 'dbface_pytorch.onnx.prototxt'
REMOTE_PATH = 'https://storage.googleapis.com/ailia-models/dbface/'

IMAGE_PATH = 'selfie.png'
SAVE_IMAGE_PATH = 'selfie_output.png'

THRESHOLD = 0.4
IOU = 0.45

# ======================
# Arguemnt Parser Config
# ======================
parser = get_base_parser('DBFace model', IMAGE_PATH, SAVE_IMAGE_PATH)
parser.add_argument(
    '-w', '--write_json',
    action='store_true',
    help='Flag to output results to json file.'
)
args = update_parser(parser)


# ======================
# Secondaty Functions
# ======================

def nms(objs, iou):
    if objs is None or len(objs) <= 1:
        return objs

    objs = sorted(objs, key=lambda obj: obj.score, reverse=True)
    keep = []
    flags = [0] * len(objs)
    for index, obj in enumerate(objs):

        if flags[index] != 0:
            continue

        keep.append(obj)
        for j in range(index + 1, len(objs)):
            if flags[j] == 0 and obj.iou(objs[j]) > iou:
                flags[j] = 1
    return keep


def preprocess(img):
    img = dbface_utils.pad(img)
    mean = [0.408, 0.447, 0.47]
    std = [0.289, 0.274, 0.278]
    img = ((img / 255.0 - mean) / std).astype(np.float32)
    img = np.transpose(img, [2, 0, 1])
    img = np.expand_dims(img, 0)
    return img


def detect_objects(img, detector):
    img = preprocess(img)

    detector.set_input_shape((1, 3, img.shape[2], img.shape[3]))
    hm, box, landmark = detector.predict({'input.1': img})

    hm_pool = dbface_utils.max_pool2d(
        A=hm[0][0], kernel_size=3, stride=1, padding=1
    )
    hm_pool = np.expand_dims(np.expand_dims(hm_pool, 0), 0)

    scores, indices = dbface_utils.get_topk_score_indices(hm_pool, hm, k=1000)
    hm_height, hm_width = hm.shape[2:]
    scores = scores.squeeze()
    indices = indices.squeeze()
    ys = list((indices // hm_width))
    xs = list((indices % hm_width))
    scores = list(scores)
    box = box.squeeze()
    landmark = landmark.squeeze()

    stride = 4
    objs = []
    for cx, cy, score in zip(xs, ys, scores):
        if score < THRESHOLD:
            break

        x, y, r, b = box[:, cy, cx]
        xyrb = (np.array([cx, cy, cx, cy]) + [-x, -y, r, b]) * stride
        x5y5 = landmark[:, cy, cx]
        x5y5 = (dbface_utils.exp(x5y5 * 4) + ([cx]*5 + [cy]*5)) * stride
        box_landmark = list(zip(x5y5[:5], x5y5[5:]))
        objs.append(
            dbface_utils.BBox(0, xyrb=xyrb, score=score, landmark=box_landmark)
        )

    return nms(objs, iou=IOU)


# ======================
# Main functions
# ======================
def recognize_from_image(filename):
    # load input image
    img = load_image(filename)
    logger.debug(f'input image shape: {img.shape}')
    img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)

    detector = ailia.Net(MODEL_PATH, WEIGHT_PATH, env_id=args.env_id, memory_mode=args.memory_mode)

    logger.info('Start inference...')
    if args.benchmark:
        logger.info('BENCHMARK mode')
        for i in range(5):
            start = int(round(time.time() * 1000))
            objs = detect_objects(img, detector)
            end = int(round(time.time() * 1000))
            logger.info(f'\tailia processing time {end - start} ms')
    else:
        objs = detect_objects(img, detector)

    # show image
    for obj in objs:
        dbface_utils.drawbbox(img, obj)

    savepath = get_savepath(args.savepath, filename)
    logger.info(f'saved at : {savepath}')
    cv2.imwrite(savepath, img)

    if args.write_json:
        json_file = '%s.json' % savepath.rsplit('.', 1)[0]
        dbface_utils.save_json(json_file, objs)

    logger.info('Script finished successfully.')


def recognize_from_video(video):
    detector = ailia.Net(MODEL_PATH, WEIGHT_PATH, env_id=args.env_id, memory_mode=args.memory_mode)

    capture = webcamera_utils.get_capture(video)

    # create video writer if savepath is specified as video format
    if args.savepath != SAVE_IMAGE_PATH:
        f_h = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        f_w = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        writer = webcamera_utils.get_writer(args.savepath, f_h, f_w)
    else:
        writer = None

    frame_shown = False
    while(True):
        ret, img = capture.read()
        # press q to end video capture
        if (cv2.waitKey(1) & 0xFF == ord('q')) or not ret:
            break
        if frame_shown and cv2.getWindowProperty('frame', cv2.WND_PROP_VISIBLE) == 0:
            break

        objs = detect_objects(img, detector)
        for obj in objs:
            dbface_utils.drawbbox(img, obj)
        cv2.imshow('frame', img)
        frame_shown = True

        # save results
        if writer is not None:
            writer.write(img)

    capture.release()
    cv2.destroyAllWindows()
    if writer is not None:
        writer.release()
    logger.info('Script finished successfully.')


def main():
    # model files check and download
    check_and_download_models(WEIGHT_PATH, MODEL_PATH, REMOTE_PATH)

    if args.video is not None:
        # video mode
        recognize_from_video(args.video)
    else:
        # image mode
        # input image loop
        for image_path in args.input:
            # prepare input data
            logger.info(image_path)
            recognize_from_image(image_path)


if __name__ == '__main__':
    main()
