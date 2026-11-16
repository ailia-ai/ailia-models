import sys

import numpy as np
import cv2

sys.path.append('../../face_detection')
from blazeface import blazeface_utils as but  # noqa
from detector_utils import letterbox_convert  # noqa

FACE_DETECTOR_ANCHOR_PATH = '../../face_detection/blazeface/anchors.npy'
FACE_DETECTOR_IMAGE_SIZE = 128
FACE_ALIGNMENT_IMAGE_SIZE = 256
FACE_MARGIN = 1.4


def predict(net, data, onnx=False):
    if not onnx:
        return net.predict([data])
    else:
        return net.run(None, {net.get_inputs()[0].name: data})


def detect_face(net, img, onnx=False):
    """detect face with BlazeFace
    :param img: RGB image
    :return: face box [x1, y1, x2, y2] of the highest score, or None
    """
    h, w = img.shape[:2]
    size = FACE_DETECTOR_IMAGE_SIZE

    data = letterbox_convert(img, (size, size))
    data = data / 127.5 - 1.0
    data = data.transpose(2, 0, 1)  # HWC -> CHW
    data = np.expand_dims(data, axis=0)
    data = data.astype(np.float32)

    output = predict(net, data, onnx)
    detections = but.postprocess(output, anchor_path=FACE_DETECTOR_ANCHOR_PATH)[0]
    if len(detections) == 0:
        return None

    # letterbox -> original image
    s = max(h, w)
    pad_x = (s - w) // 2
    pad_y = (s - h) // 2
    ymin, xmin, ymax, xmax = detections[0, :4] * s

    return [xmin - pad_x, ymin - pad_y, xmax - pad_x, ymax - pad_y]


def get_preds_from_hm(hm):
    """
    Obtain (x,y) coordinates given a set of N heatmaps.
    ref: face_recognition/face_alignment/face_alignment.py
    """
    idx = np.argmax(
        hm.reshape(hm.shape[0], hm.shape[1], hm.shape[2] * hm.shape[3]), axis=2
    )
    idx += 1
    preds = idx.reshape(idx.shape[0], idx.shape[1], 1)
    preds = np.tile(preds, (1, 1, 2)).astype(float)
    preds[..., 0] = (preds[..., 0] - 1) % hm.shape[3] + 1
    preds[..., 1] = np.floor((preds[..., 1] - 1) / (hm.shape[2])) + 1

    for i in range(preds.shape[0]):
        for j in range(preds.shape[1]):
            hm_ = hm[i, j, :]
            pX, pY = int(preds[i, j, 0]) - 1, int(preds[i, j, 1]) - 1
            if pX > 0 and pX < 63 and pY > 0 and pY < 63:
                diff = np.array(
                    [hm_[pY, pX + 1] - hm_[pY, pX - 1],
                     hm_[pY + 1, pX] - hm_[pY - 1, pX]]).astype(float)
                preds[i, j] = preds[i, j] + (np.sign(diff) * 0.25)

    preds += -0.5

    return preds


def get_landmark(img, net_det, net_align, onnx=False):
    """get landmark with BlazeFace and 2DFAN-4
    :param img: RGB image
    :return: np.array shape=(68, 2)
    """
    box = detect_face(net_det, img, onnx)
    if box is None:
        return None

    # crop face region (same as crop_blazeface)
    h, w = img.shape[:2]
    cx = (box[0] + box[2]) / 2
    cy = (box[1] + box[3]) / 2
    size = max(box[2] - box[0], box[3] - box[1]) * FACE_MARGIN
    left = max(cx - size / 2, 0)
    top = max(cy - size / 2, 0)
    right = min(left + size, w)
    bottom = min(top + size, h)
    left, top, right, bottom = int(left), int(top), int(right), int(bottom)
    crop = img[top:bottom, left:right]

    size = FACE_ALIGNMENT_IMAGE_SIZE
    data = cv2.resize(crop, (size, size))
    data = data / 255.0
    data = data.transpose(2, 0, 1)  # HWC -> CHW
    data = np.expand_dims(data, axis=0)
    data = data.astype(np.float32)

    output = predict(net_align, data, onnx)
    hm = output[0]

    # heatmap -> crop -> original image
    lm = get_preds_from_hm(hm)[0] * (size / hm.shape[3])
    lm[:, 0] = lm[:, 0] * (right - left) / size + left
    lm[:, 1] = lm[:, 1] * (bottom - top) / size + top

    return lm
