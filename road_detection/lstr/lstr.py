import glob
import os
import sys
import time
import json

import ailia
import cv2
import numpy as np

# import original modules
sys.path.append('../../util')
# logger
from logging import getLogger  # noqa: E402

import webcamera_utils
from image_utils import imread  # noqa: E402
from model_utils import check_and_download_models  # noqa: E402
from arg_utils import get_base_parser, get_savepath, update_parser  # noqa: E402

logger = getLogger(__name__)


# ======================
# Parameters
# ======================
WEIGHT_PATH = 'lstr.onnx'
MODEL_PATH = 'lstr.onnx.prototxt'
REMOTE_PATH = 'https://storage.googleapis.com/ailia-models/lstr/'

IMAGE_PATH = 'input.jpg'
SAVE_IMAGE_PATH = 'output.jpg'

HEIGHT = 360
WIDTH = 640
VIDEO_HEIGHT = 1280
VIDEO_WIDTH = 720

DB_MEAN = [0.40789655, 0.44719303, 0.47026116]
DB_STD = [0.2886383, 0.27408165, 0.27809834]


# ======================
# Arguemnt Parser Config
# ======================
parser = get_base_parser(
    'LSTR', IMAGE_PATH, SAVE_IMAGE_PATH,
)
parser.add_argument(
    '-w', '--write_json',
    action='store_true',
    help='Flag to output results to json file.'
)
args = update_parser(parser)


# ======================
# Main functions
# ======================
def _normalize(image, mean, std):
    image -= mean
    image /= std


def _softmax(x, axis=None):
    e_x = np.exp(x - np.max(x, axis=axis, keepdims=True))
    e_x = e_x / np.sum(e_x, axis=axis, keepdims=True)
    return e_x


def predict(net, image):
    height, width = image.shape[0:2]
    images = np.zeros((1, 3, HEIGHT, WIDTH), dtype=np.float32)
    masks = np.ones((1, 1, HEIGHT, WIDTH), dtype=np.float32)
    pad_image = image.copy()
    pad_mask = np.zeros((height, width, 1), dtype=np.float32)
    resized_image = cv2.resize(pad_image, (WIDTH, HEIGHT))
    resized_mask = cv2.resize(pad_mask, (WIDTH, HEIGHT))
    masks[0][0] = resized_mask.squeeze()
    resized_image = resized_image / 255.
    _normalize(resized_image, DB_MEAN, DB_STD)
    resized_image = resized_image.transpose(2, 0, 1)
    images[0] = resized_image
    outputs = net.predict([images, masks])
    return outputs


def postprocess(out_pred_logits, out_pred_curves, orig_target_sizes):
    out_logits = np.expand_dims(out_pred_logits[0], axis=0)
    out_curves = np.expand_dims(out_pred_curves[0], axis=0)

    assert len(out_logits) == len(orig_target_sizes)
    assert orig_target_sizes.shape[1] == 2

    prob = _softmax(out_logits, -1)
    labels = np.argmax(prob, axis=-1)
    labels[labels != 1] = 0
    labels = np.expand_dims(labels, axis=-1).astype(float)

    results = np.concatenate((labels, out_curves), axis=-1)
    return results


def draw_annotation(pred, img):
    img = (img - np.min(img)) / (np.max(img) - np.min(img))
    img = (img * 255).astype(np.uint8)
    img_h, img_w, _ = img.shape

    # Draw predictions
    # pred = pred[pred[:, 0] != 0]  # filter invalid lanes
    pred = pred[pred[:, 0].astype(int) == 1]
    overlay = img.copy()
    cv2.rectangle(img, (5, 10), (5 + 1270, 25 + 30 * pred.shape[0] + 10), (255, 255, 255), thickness=-1)
    cv2.putText(img, 'Predicted curve parameters:', (10, 30), fontFace=cv2.FONT_HERSHEY_PLAIN, fontScale=1.5, color=(0, 0, 0), thickness=2)

    for i, lane in enumerate(pred):
        color = (0, 255, 0)
        lane = lane[1:]  # remove conf
        lower, upper = lane[0], lane[1]
        lane = lane[2:]  # remove upper, lower positions

        # generate points from the polynomial
        ys = np.linspace(lower, upper, num=100)
        points = np.zeros((len(ys), 2), dtype=np.int32)
        points[:, 1] = (ys * img_h).astype(int)
        points[:, 0] = ((lane[0] / (ys - lane[1]) ** 2 + lane[2] / (ys - lane[1]) + lane[3] + lane[4] * ys - lane[5]) * img_w).astype(int)
        points = points[(points[:, 0] > 0) & (points[:, 0] < img_w)]

        # draw lane with a polyline on the overlay
        for current_point, next_point in zip(points[:-1], points[1:]):
            overlay = cv2.line(overlay, tuple(current_point), tuple(next_point), color=color, thickness=7)

        # draw lane ID
        if len(points) > 0:
            cv2.putText(img, str(i), tuple(points[len(points)//3]), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=2, color=(0, 255, 255), thickness=5)
            content = "{}: k''={:.3}, f''={:.3}, m''={:.3}, n'={:.3}, b''={:.3}, b'''={:.3}, alpha={}, beta={}".format(
                str(i), lane[0], lane[1], lane[2], lane[3], lane[4], lane[5], int(lower * img_h),
                int(upper * img_w)
            )
            cv2.putText(img, content, (10, 30 * (i + 2)), fontFace=cv2.FONT_HERSHEY_PLAIN, fontScale=1.5, color=color, thickness=2)

    w = 0.5
    img = ((1. - w) * img + w * overlay).astype(np.uint8) # Add lanes overlay
    return img


def save_result_json(json_path, img, pred):
    img_h, img_w, _ = img.shape
    results = []
    pred = pred[pred[:, 0].astype(int) == 1]
    for lane in pred:
        conf, lower, upper = lane[0], lane[1], lane[2]
        lane = lane[3:]  # remove conf, upper, lower
        results.append({
            "k''": lane[0], "f''": lane[1], "m''": lane[2], "n'": lane[3], "b''": lane[4], "b'''": lane[5],
            "alpha": lower * img_h, "beta": upper * img_w, "conf": conf
        })
    with open(json_path, 'w') as f:
        json.dump(results, f, indent=2)


def recognize_from_image(net, orig_target_sizes):
    # input image loop
    for image_path in args.input:
        # prepare input data
        image = imread(image_path)

        out_pred_logits, out_pred_curves, _, _, weights = predict(net, image)
        results = postprocess(out_pred_logits, out_pred_curves, orig_target_sizes)
        preds = draw_annotation(results[0], image)

        savepath = get_savepath(args.savepath, image_path)
        logger.info(f'saved at : {savepath}')
        cv2.imwrite(savepath, preds)

        if args.write_json:
            json_file = '%s.json' % savepath.rsplit('.', 1)[0]
            save_result_json(json_file, image, results[0])

    logger.info('Script finished successfully.')


def recognize_from_video(net, orig_target_sizes):
    capture = webcamera_utils.get_capture(args.video)

    # create video writer if savepath is specified as video format
    if args.savepath != SAVE_IMAGE_PATH:
        writer = webcamera_utils.get_writer(args.savepath, HEIGHT, WIDTH)
    else:
        writer = None
    
    frame_shown = False
    while True:
        ret, frame = capture.read()
        if (cv2.waitKey(1) & 0xFF == ord('q')) or not ret:
            break
        if frame_shown and cv2.getWindowProperty('frame', cv2.WND_PROP_VISIBLE) == 0:
            break

        frame = cv2.resize(frame, dsize=(VIDEO_HEIGHT, VIDEO_WIDTH))
        out_pred_logits, out_pred_curves, _, _, weights = predict(net, frame)
        results = postprocess(out_pred_logits, out_pred_curves, orig_target_sizes)
        preds = draw_annotation(results[0], frame)
        cv2.imshow('frame', preds)
        frame_shown = True

        # save results
        if writer is not None:
            writer.write(preds)

    capture.release()
    cv2.destroyAllWindows()
    if writer is not None:
        writer.release()
    logger.info('Script finished successfully.')


def main():
    # model files check and download
    check_and_download_models(WEIGHT_PATH, MODEL_PATH, REMOTE_PATH)

    net = ailia.Net(MODEL_PATH, WEIGHT_PATH, env_id=args.env_id, memory_mode=args.memory_mode)
    orig_target_sizes = np.expand_dims(np.array([HEIGHT, WIDTH]), axis=0)

    # predict
    if args.video is not None:
        recognize_from_video(net, orig_target_sizes)
    else:
        recognize_from_image(net, orig_target_sizes)


if __name__ == '__main__':
    main()
