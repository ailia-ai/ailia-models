"""
reference: ailia-models/face_detection/retinaface
"""

import sys
import numpy as np

sys.path.append('../../face_detection/retinaface')
import retinaface_utils as rut
from retinaface_utils import PriorBox

CONFIDENCE_THRES = 0.02
TOP_K = 5000
NMS_THRES = 0.4
KEEP_TOP_K = 750

def face_detect(image, retinaface_net):
    """
    Args:
        image (numpy.ndarray): Input image (H, W, C) in BGR format.
        retinaface_net (ailia.Net): Ailia RetinaFace model.

    Returns:
        numpy.ndarray: Bounding boxes of detected faces (N, 4) in (x1, y1, x2, y2) format.
    """
    cfg = rut.cfg_re50

    dim = (image.shape[1], image.shape[0])
    image = image - (104, 117, 123)
    image = np.expand_dims(image.transpose(2, 0, 1), axis=0)

    preds = retinaface_net.predict([image])

    detections = postprocessing(preds, image, cfg=cfg, dim=dim)
    bboxes = detections[:, :4].astype(int)
    return bboxes

def postprocessing(preds_ailia, input_data, cfg, dim):
    IMAGE_WIDTH, IMAGE_HEIGHT = dim
    scale = np.array([IMAGE_WIDTH, IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_HEIGHT])
    loc, conf, landms = preds_ailia
    priorbox = PriorBox(cfg, image_size=(IMAGE_HEIGHT, IMAGE_WIDTH))
    priors = priorbox.forward()
    boxes = rut.decode(np.squeeze(loc, axis=0), priors, cfg['variance'])
    boxes = boxes * scale
    scores = np.squeeze(conf, axis=0)[:, 1]
    landms = rut.decode_landm(np.squeeze(landms, axis=0), priors, cfg['variance'])
    scale1 = np.array([input_data.shape[3], input_data.shape[2], input_data.shape[3], input_data.shape[2],
                            input_data.shape[3], input_data.shape[2], input_data.shape[3], input_data.shape[2],
                            input_data.shape[3], input_data.shape[2]])
    landms = landms * scale1

    # ignore low scores
    inds = np.where(scores > CONFIDENCE_THRES)[0]
    boxes = boxes[inds]
    landms = landms[inds]
    scores = scores[inds]

    # keep top-K before NMS
    order = scores.argsort()[::-1][:TOP_K]
    boxes = boxes[order]
    landms = landms[order]
    scores = scores[order]

    # do NMS
    dets = np.hstack((boxes, scores[:, np.newaxis])).astype(np.float32, copy=False)
    keep = rut.py_cpu_nms(dets, NMS_THRES)
    dets = dets[keep, :]
    landms = landms[keep]

    # keep top-K faster NMS
    dets = dets[:KEEP_TOP_K, :]
    landms = landms[:KEEP_TOP_K, :]

    detections = np.concatenate((dets, landms), axis=1)
    
    return detections
