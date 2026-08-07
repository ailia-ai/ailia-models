import cv2
import numpy as np

import os
import sys
sys.path.append('../../util')
from nms_utils import batched_nms

import random

def xyxy2xywh(bbox_xyxy):
    """Transform the bbox format from x1y1x2y2 to xywh.
    Args:
        bbox_xyxy (np.ndarray): Bounding boxes (with scores), shaped (n, 4) or
            (n, 5). (left, top, right, bottom, [score])
    Returns:
        np.ndarray: Bounding boxes (with scores),
          shaped (n, 4) or (n, 5). (left, top, width, height, [score])
    """
    bbox_xywh = bbox_xyxy.copy()
    bbox_xywh[:, 2] = bbox_xywh[:, 2] - bbox_xywh[:, 0] + 1
    bbox_xywh[:, 3] = bbox_xywh[:, 3] - bbox_xywh[:, 1] + 1
    return bbox_xywh

_COLORS = np.array([
    0.000, 0.447, 0.741, 0.850, 0.325, 0.098, 0.929, 0.694, 0.125, 0.494,
    0.184, 0.556, 0.466, 0.674, 0.188, 0.301, 0.745, 0.933, 0.635, 0.078,
    0.184, 0.300, 0.300, 0.300, 0.600, 0.600, 0.600, 1.000, 0.000, 0.000,
    1.000, 0.500, 0.000, 0.749, 0.749, 0.000, 0.000, 1.000, 0.000, 0.000,
    0.000, 1.000, 0.667, 0.000, 1.000, 0.333, 0.333, 0.000, 0.333, 0.667,
    0.000, 0.333, 1.000, 0.000, 0.667, 0.333, 0.000, 0.667, 0.667, 0.000,
    0.667, 1.000, 0.000, 1.000, 0.333, 0.000, 1.000, 0.667, 0.000, 1.000,
    1.000, 0.000, 0.000, 0.333, 0.500, 0.000, 0.667, 0.500, 0.000, 1.000,
    0.500, 0.333, 0.000, 0.500, 0.333, 0.333, 0.500, 0.333, 0.667, 0.500,
    0.333, 1.000, 0.500, 0.667, 0.000, 0.500, 0.667, 0.333, 0.500, 0.667,
    0.667, 0.500, 0.667, 1.000, 0.500, 1.000, 0.000, 0.500, 1.000, 0.333,
    0.500, 1.000, 0.667, 0.500, 1.000, 1.000, 0.500, 0.000, 0.333, 1.000,
    0.000, 0.667, 1.000, 0.000, 1.000, 1.000, 0.333, 0.000, 1.000, 0.333,
    0.333, 1.000, 0.333, 0.667, 1.000, 0.333, 1.000, 1.000, 0.667, 0.000,
    1.000, 0.667, 0.333, 1.000, 0.667, 0.667, 1.000, 0.667, 1.000, 1.000,
    1.000, 0.000, 1.000, 1.000, 0.333, 1.000, 1.000, 0.667, 1.000, 0.333,
    0.000, 0.000, 0.500, 0.000, 0.000, 0.667, 0.000, 0.000, 0.833, 0.000,
    0.000, 1.000, 0.000, 0.000, 0.000, 0.167, 0.000, 0.000, 0.333, 0.000,
    0.000, 0.500, 0.000, 0.000, 0.667, 0.000, 0.000, 0.833, 0.000, 0.000,
    1.000, 0.000, 0.000, 0.000, 0.167, 0.000, 0.000, 0.333, 0.000, 0.000,
    0.500, 0.000, 0.000, 0.667, 0.000, 0.000, 0.833, 0.000, 0.000, 1.000,
    0.000, 0.000, 0.000, 0.143, 0.143, 0.143, 0.286, 0.286, 0.286, 0.429,
    0.429, 0.429, 0.571, 0.571, 0.571, 0.714, 0.714, 0.714, 0.857, 0.857,
    0.857, 0.000, 0.447, 0.741, 0.314, 0.717, 0.741, 0.50, 0.5, 0
]).astype(np.float32).reshape(-1, 3)

def multiclass_nms(multi_bboxes,
                   multi_scores,
                   score_thr,
                   iou_thr,
                   max_num=100,
                   score_factors=None):
    """NMS for multi-class bboxes.

    Args:
        multi_bboxes (Tensor): shape (n, #class*4) or (n, 4)
        multi_scores (Tensor): shape (n, #class), where the last column
            contains scores of the background class, but this will be ignored.
        score_thr (float): bbox threshold, bboxes with scores lower than it
            will not be considered.
        nms_thr (float): NMS IoU threshold
        max_num (int): if there are more than max_num bboxes after NMS,
            only top max_num will be kept.
        score_factors (Tensor): The factors multiplied to scores before
            applying NMS

    Returns:
        tuple: (bboxes, labels), tensors of shape (k, 5) and (k, 1). Labels \
            are 0-based.
    """
    num_classes = multi_scores.shape[1]
    if multi_bboxes.shape[1] > 4:
        bboxes = multi_bboxes.view(multi_scores.size(0), -1, 4)
    else:
        bboxes = np.broadcast_to(multi_bboxes[:,None], (multi_scores.shape[0],num_classes,4))

    scores = multi_scores
    # filter out boxes with low scores
    valid_mask = scores > score_thr  # 1000 * 80 bool

    # We use masked_select for ONNX exporting purpose,
    # which is equivalent to bboxes = bboxes[valid_mask]
    # (TODO): as ONNX does not support repeat now,
    # we have to use this ugly code
    # bboxes -> 1000, 4
    stack = np.stack((valid_mask, valid_mask, valid_mask, valid_mask), -1)
    bboxes = np.ravel(bboxes)
    stack  = np.ravel(stack)
    bboxes = bboxes[stack].reshape(-1,4)


    if score_factors is not None:
        scores = scores * score_factors[:, None]
    scores = np.ravel(scores) 
    scores = scores[np.ravel(valid_mask)]

    labels = np.nonzero(valid_mask)[1]

    if bboxes.size == 0:
        bboxes = np.zeros((0, 5))
        labels = np.zeros((0, ), dtype=np.int64)
        scores = np.zeros((0, ))

        return bboxes, scores, labels

    keep = batched_nms(bboxes, scores, labels, iou_thr)

    if max_num > 0:
        keep = keep[:max_num]

    return bboxes[keep], scores[keep], labels[keep]



def postprocess(cls_scores,
                bbox_preds,
                num_classes,
                conf_thre=0.7,
                nms_thre=0.45,
                imgs=None):
    batch_size = bbox_preds.shape[0]
    output = [None for _ in range(batch_size)]
    for i in range(batch_size):
        # If none are remaining => process next image
        if not bbox_preds[i].shape[0]:
            continue

        bbox_pred =  bbox_preds[i]
        cls_score =  cls_scores[i]
        detections, scores, labels = multiclass_nms(bbox_pred,
                                                    cls_score, conf_thre,
                                                    nms_thre, 500)
        scores = np.expand_dims(scores,axis = 1)
        labels = np.expand_dims(labels,axis = 1)


        detections = np.concatenate((detections, 
                                    np.ones_like(scores),
                                    scores,
                                    labels), axis=1)

        if output[i] is None:
            output[i] = detections
        else:
            output[i] = np.concatenate((output[i], detections))

    # transfer to BoxList
    for i in range(len(output)):
        res = output[i]
        if res is None or imgs is None:
            boxlist = BoxList(torch.zeros(0, 4), (0, 0), mode='xyxy')
            boxlist.add_field('objectness', 0)
            boxlist.add_field('scores', 0)
            boxlist.add_field('labels', -1)

        else:
            img_h, img_w = imgs.image_sizes[i]
            boxlist = BoxList(res[:, :4], (img_w, img_h), mode='xyxy')
            boxlist.add_field('objectness', res[:, 4])
            boxlist.add_field('scores', res[:, 5])
            boxlist.add_field('labels', res[:, 6] + 1)
        output[i] = boxlist

    return output

class BoxList(object):
    """
    This class represents a set of bounding boxes.
    The bounding boxes are represented as a Nx4 Tensor.
    In order to uniquely determine the bounding boxes with respect
    to an image, we also store the corresponding image dimensions.
    They can contain extra information that is specific to each bounding box,
    such as labels.
    """
    def __init__(self, bbox, image_size, mode='xyxy'):
        self.bbox = bbox
        self.extra_fields = {}

    def add_field(self, field, field_data):
        self.extra_fields[field] = field_data

    def get_field(self, field):
        return self.extra_fields[field]


class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image, target=None):
        for t in self.transforms:
            image, target = t(image, target)
        return image, target



class Resize(object):
    def __init__(self, max_range, target_size=None):
        if not isinstance(max_range, (list, tuple)):
            max_range = (max_range, )
        self.max_range = max_range
        self.target_size = target_size

    def get_size_ratio(self, image_size):
        if self.target_size is None:
            target_size = random.choice(self.max_range)
        w, h = image_size
        if self.target_size is None:
            t_w, t_h = target_size, target_size
        else:
            t_w, t_h = self.target_size[1], self.target_size[0]
        r = min(t_w / w, t_h / h)
        o_w, o_h = int(w * r), int(h * r)
        return (o_w, o_h)

    def __call__(self, image, target=None):
        h, w = image.shape[:2]
        size = self.get_size_ratio((w, h))

        image = cv2.resize(image, size,
                           interpolation=cv2.INTER_LINEAR).astype(np.uint8)
        image = image.transpose((2, 0, 1))
        image = np.ascontiguousarray(image, dtype=np.float32)
        if isinstance(target, list):
            target = [t.resize(size) for t in target]
        elif target is None:
            return image, target
        else:
            target = target.resize(size)
        return image, target


class RandomHorizontalFlip(object):
    def __init__(self, prob=0.5):
        self.prob = prob

    def __call__(self, image, target):
        if random.random() < self.prob:
            image = image[:, :, ::-1]
            image = np.ascontiguousarray(image, dtype=np.float32)
            if target is not None:
                target = target.transpose(0)
        return image, target



def transform_img(origin_img, size_divisibility, image_max_range, flip_prob,
                  image_mean, image_std, infer_size=None):

    transform = [
        Resize(image_max_range, target_size=infer_size),
        RandomHorizontalFlip(flip_prob),
    ]
    transform = Compose(transform)

    img, _ = transform(origin_img)

    mean = [0.0, 0.0, 0.0]
    std  = [1.0, 1.0, 1.0]
    for i in range(3):
        img[i,:, :] = (img[i,:, :] - mean[i]) / std[i]

    img = to_image_list(img, size_divisibility)
    return img


class ImageList(object):
    """
    Structure that holds a list of images (of possibly
    varying sizes) as a single tensor.
    This works by padding the images to the same size,
    and storing in a field the original sizes of each image
    """
    def __init__(self, tensors, image_sizes, pad_sizes):
        """
        Arguments:
            tensors (tensor)
            image_sizes (list[tuple[int, int]])
        """
        self.tensors = tensors
        self.image_sizes = image_sizes
        self.pad_sizes = pad_sizes

def to_image_list(tensors, size_divisible=0, max_size=None):
    """
    tensors can be an ImageList, a torch.Tensor or
    an iterable of Tensors. It can't be a numpy array.
    When tensors is an iterable of Tensors, it pads
    the Tensors with zeros so that they have the same
    shape
    """

    if tensors.ndim == 3:
        tensors = tensors[None]
    assert tensors.ndim == 4
    image_sizes = [tensor.shape[-2:] for tensor in tensors]
    return ImageList(tensors, image_sizes, image_sizes)

