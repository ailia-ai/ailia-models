import sys
import time

import numpy as np
import cv2

import ailia

# import original modules
sys.path.append('../../util')
from arg_utils import get_base_parser, update_parser, get_savepath  # noqa: E402
from model_utils import check_and_download_models  # noqa: E402
from detector_utils import load_image  # noqa: E402C
from math_utils import softmax  # noqa: E402C
from webcamera_utils import get_capture, get_writer  # noqa: E402
# logger
from logging import getLogger  # noqa: E402

sys.path.append('../../face_detection/blazeface')
from blazeface_utils import compute_blazeface, crop_blazeface  # noqa: E402

logger = getLogger(__name__)

# ======================
# Parameters
# ======================

WEIGHT_PATH = 'MN3_large.onnx'
MODEL_PATH = 'MN3_large.onnx.prototxt'
REMOTE_PATH = 'https://storage.googleapis.com/ailia-models/face-anti-spoofing/'

FACE_WEIGHT_PATH = 'blazefaceback.onnx'
FACE_MODEL_PATH = 'blazefaceback.onnx.prototxt'
FACE_REMOTE_PATH = "https://storage.googleapis.com/ailia-models/blazeface/"
FACE_MIN_SCORE_THRESH = 0.5

IMAGE_PATH = 'demo.png'
SAVE_IMAGE_PATH = 'output.png'

IMAGE_HEIGHT = 128
IMAGE_WIDTH = 128

# ======================
# Arguemnt Parser Config
# ======================

parser = get_base_parser(
    'Lightweight Face Anti Spoofing', IMAGE_PATH, SAVE_IMAGE_PATH
)
parser.add_argument(
    '-d', '--detection',
    action='store_true',
    help='Use object detection.'
)
parser.add_argument(
    '--spoof_thresh',
    type=float, default=0.4,
    help='Threshold for predicting spoof/real. The lower the more model oriented on spoofs'
)
args = update_parser(parser)


# ======================
# Secondaty Functions
# ======================

def draw_detections(frame, detections, confidence, thresh):
    """Draws detections and labels"""
    for i, rect in enumerate(detections):
        left, top, right, bottom = rect
        if confidence[i][1] > thresh:
            label = f'spoof: {round(confidence[i][1] * 100, 3)}%'
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), thickness=2)
        else:
            label = f'real: {round(confidence[i][0] * 100, 3)}%'
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), thickness=2)
        label_size, base_line = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 1, 1)
        top = max(top, label_size[1])
        cv2.rectangle(
            frame,
            (left, top - label_size[1]),
            (left + label_size[0], top + base_line),
            (255, 255, 255), cv2.FILLED)
        cv2.putText(frame, label, (left, top), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0))

    return frame


# ======================
# Main functions
# ======================

def preprocess(img):
    h, w = (IMAGE_HEIGHT, IMAGE_WIDTH)

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (w, h), interpolation=cv2.INTER_CUBIC)

    # normalize
    mean = np.array([0.5931, 0.4690, 0.4229])
    std = np.array([0.2471, 0.2214, 0.2157])
    img = img / 255
    img = (img - mean) / std

    img = img.transpose(2, 0, 1)  # HWC -> CHW
    img = np.expand_dims(img, axis=0)
    img = img.astype(np.float32)

    return img


def predict(net, img):
    img = preprocess(img)

    # feedforward
    output = net.predict([img])
    logits = output[0]

    pred = softmax(logits, axis=1)

    return pred


def recognize_from_image(net, detector=None):
    # input image loop
    for image_path in args.input:
        logger.info(image_path)

        # prepare input data
        img = load_image(image_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)

        if detector:
            recognize_from_frame(net, detector, img)
            savepath = get_savepath(args.savepath, image_path)
            logger.info(f'saved at : {savepath}')
            cv2.imwrite(savepath, img)
            continue

        # inference
        logger.info('Start inference...')
        if args.benchmark:
            logger.info('BENCHMARK mode')
            total_time_estimation = 0
            for i in range(args.benchmark_count):
                start = int(round(time.time() * 1000))
                pred = predict(net, img)
                end = int(round(time.time() * 1000))
                estimation_time = (end - start)

                # Loggin
                logger.info(f'\tailia processing estimation time {estimation_time} ms')
                if i != 0:
                    total_time_estimation = total_time_estimation + estimation_time

            logger.info(f'\taverage time estimation {total_time_estimation / (args.benchmark_count - 1)} ms')
        else:
            pred = predict(net, img)

        pred = pred[0]
        i = np.argmax(pred)

        # show result
        logger.info(
            " face is %s: %.3f%%" %
            ('real' if i == 0 else 'spoof', pred[i] * 100))

    logger.info('Script finished successfully.')


def recognize_from_frame(net, detector, frame):
    spoof_thresh = args.spoof_thresh

    # detect face
    detections = compute_blazeface(
        detector,
        frame,
        anchor_path='../../face_detection/blazeface/anchorsback.npy',
        back=True,
        min_score_thresh=FACE_MIN_SCORE_THRESH
    )

    # adjust face rectangle
    new_detections = []
    for detection in detections:
        margin = 1.5
        r = ailia.DetectorObject(
            category=detection.category,
            prob=detection.prob,
            x=detection.x - detection.w * (margin - 1.0) / 2,
            y=detection.y - detection.h * (margin - 1.0) / 2 - detection.h * margin / 8,
            w=detection.w * margin,
            h=detection.h * margin,
        )
        new_detections.append(r)

    # crop, preprocess
    images = []
    detections = []
    for obj in new_detections:
        # get detected face
        margin = 1.0
        crop_img, top_left, bottom_right = crop_blazeface(
            obj, margin, frame
        )
        if crop_img.shape[0] <= 0 or crop_img.shape[1] <= 0:
            continue

        img = preprocess(crop_img)
        images.append(img)
        detections.append(
            (top_left[0], top_left[1], bottom_right[0], bottom_right[1])
        )

    if not images:
        return frame

    images = np.concatenate(images)

    # feedforward
    output = net.predict([images])
    logits = output[0]
    preds = softmax(logits, axis=1)

    frame = draw_detections(frame, detections, preds, spoof_thresh)

    return frame


def recognize_from_video(net, detector):
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

        res_img = recognize_from_frame(net, detector, frame)

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
    logger.info('Checking Anti Spoofing model...')
    check_and_download_models(WEIGHT_PATH, MODEL_PATH, REMOTE_PATH)

    if args.video or args.detection:
        logger.info('Check object detection model...')
        check_and_download_models(
            FACE_WEIGHT_PATH, FACE_MODEL_PATH, FACE_REMOTE_PATH
        )

    env_id = args.env_id

    # initialize
    net = ailia.Net(MODEL_PATH, WEIGHT_PATH, env_id=env_id, memory_mode=args.memory_mode)
    detector = None
    if args.video or args.detection:
        detector = ailia.Net(FACE_MODEL_PATH, FACE_WEIGHT_PATH, env_id=env_id, memory_mode=args.memory_mode)

    if args.video is not None:
        recognize_from_video(net, detector)
    else:
        recognize_from_image(net, detector)


if __name__ == '__main__':
    main()
