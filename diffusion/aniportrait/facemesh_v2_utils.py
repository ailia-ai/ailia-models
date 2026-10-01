from scipy.spatial.transform import Rotation as R
import math
import sys
from collections import namedtuple

import cv2
import numpy as np

sys.path.append("../../util")
from image_utils import normalize_image

from detection_utils import IMAGE_SIZE as IMAGE_DET_SIZE

ROI = namedtuple("ROI", ["x_center", "y_center", "width", "height", "rotation"])
NUM_LANDMARKS = 478
IMAGE_SIZE = 256


def warp_perspective(
        img, roi: ROI,
        dst_width, dst_height,
        keep_aspect_ratio=True):
    im_h, im_w, _ = img.shape

    v_pad = h_pad = 0
    if keep_aspect_ratio:
        dst_aspect_ratio = dst_height / dst_width
        roi_aspect_ratio = roi.height / roi.width

        if dst_aspect_ratio > roi_aspect_ratio:
            new_height = roi.width * dst_aspect_ratio
            new_width = roi.width
            v_pad = (1 - roi_aspect_ratio / dst_aspect_ratio) / 2
        else:
            new_width = roi.height / dst_aspect_ratio
            new_height = roi.height
            h_pad = (1 - dst_aspect_ratio / roi_aspect_ratio) / 2

        roi = ROI(roi.x_center, roi.y_center, new_width, new_height, roi.rotation)

    a = roi.width
    b = roi.height
    c = math.cos(roi.rotation)
    d = math.sin(roi.rotation)
    e = roi.x_center
    f = roi.y_center
    g = 1 / im_w
    h = 1 / im_h

    project_mat = [
        [a * c * g, -b * d * g, 0.0, (-0.5 * a * c + 0.5 * b * d + e) * g],
        [a * d * h, b * c * h, 0.0, (-0.5 * b * c - 0.5 * a * d + f) * h],
        [0.0, 0.0, a * g, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ]

    rotated_rect = (
        (roi.x_center, roi.y_center),
        (roi.width, roi.height),
        roi.rotation * 180. / math.pi
    )
    pts1 = cv2.boxPoints(rotated_rect)

    pts2 = np.float32([[0, dst_height], [0, 0], [dst_width, 0], [dst_width, dst_height]])
    M = cv2.getPerspectiveTransform(pts1, pts2)
    img = cv2.warpPerspective(
        img, M, (dst_width, dst_height), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)

    return img, project_mat, roi, (h_pad, v_pad)



def preprocess_det(img):
    im_h, im_w, _ = img.shape

    """
    resize & padding
    """
    roi = ROI(0.5 * im_w, 0.5 * im_h, im_w, im_h, 0)
    dst_width = dst_height = IMAGE_DET_SIZE
    img, matrix, *_ = warp_perspective(
        img, roi,
        dst_width, dst_height)

    """
    normalize & reshape
    """
    img = normalize_image(img, normalize_type='127.5')
    img = np.expand_dims(img, axis=0)
    img = img.astype(np.float32)

    return img, matrix


def post_processing(input_tensors, roi, pad):
    num_landmarks = NUM_LANDMARKS
    num_dimensions = 3

    # TensorsToFaceLandmarksGraph
    input_tensors = input_tensors.reshape(-1)
    output_landmarks = np.zeros((num_landmarks, num_dimensions))
    for i in range(num_landmarks):
        offset = i * num_dimensions
        output_landmarks[i] = input_tensors[offset:offset + 3]

    norm_landmarks = output_landmarks / 256

    # LandmarkLetterboxRemovalCalculator
    h_pad, v_pad = pad
    left = h_pad
    top = v_pad
    left_and_right = h_pad * 2
    top_and_bottom = v_pad * 2
    for landmark in norm_landmarks:
        new_x = (landmark[0] - left) / (1 - left_and_right)
        new_y = (landmark[1] - top) / (1 - top_and_bottom)
        new_z = landmark[2] / (1 - left_and_right)  # Scale Z coordinate as X.
        landmark[:3] = (new_x, new_y, new_z)

    # LandmarkProjectionCalculator
    width = roi.width
    height = roi.height
    x_center = roi.x_center
    y_center = roi.y_center
    angle = roi.rotation
    for landmark in norm_landmarks:
        x = landmark[0] - 0.5
        y = landmark[1] - 0.5
        z = landmark[2]
        new_x = math.cos(angle) * x - math.sin(angle) * y
        new_y = math.sin(angle) * x + math.cos(angle) * y

        new_x = new_x * width + x_center
        new_y = new_y * height + y_center
        new_z = z * width

        landmark[...] = new_x, new_y, new_z

    return norm_landmarks


def preprocess(img, roi):
    im_h, im_w, _ = img.shape

    """
    resize & padding
    """
    dst_width = dst_height = IMAGE_SIZE
    img, _, roi, pad = warp_perspective(
        img, roi,
        dst_width, dst_height,
        keep_aspect_ratio=False)

    img = normalize_image(img, normalize_type='255')
    img = np.expand_dims(img, axis=0)
    img = img.astype(np.float32)

    return img, roi, pad


def matrix_to_euler_and_translation(matrix):
    rotation_matrix = matrix[:3, :3]
    translation_vector = matrix[:3, 3]
    rotation = R.from_matrix(rotation_matrix)
    euler_angles = rotation.as_euler('xyz', degrees=True)
    return euler_angles, translation_vector


def smooth_pose_seq(pose_seq, window_size=5):
    smoothed_pose_seq = np.zeros_like(pose_seq)

    for i in range(len(pose_seq)):
        start = max(0, i - window_size // 2)
        end = min(len(pose_seq), i + window_size // 2 + 1)
        smoothed_pose_seq[i] = np.mean(pose_seq[start:end], axis=0)

    return smoothed_pose_seq


def crop_face(img, lmk_extractor, expand=1.5):
    result = lmk_extractor(img)  # cv2 BGR

    if result is None:
        return None
    
    H, W, _ = img.shape
    lmks = result[1]
    lmks[:, 0] *= W
    lmks[:, 1] *= H

    x_min = np.min(lmks[:, 0])
    x_max = np.max(lmks[:, 0])
    y_min = np.min(lmks[:, 1])
    y_max = np.max(lmks[:, 1])

    width = x_max - x_min
    height = y_max - y_min
    
    if width*height >= W*H*0.15:
        if W == H:
            return img
        size = min(H, W)
        offset = int((max(H, W) - size)/2)
        if size == H:
            return img[:, offset:-offset]
        else:
            return img[offset:-offset, :]
    else:
        center_x = x_min + width / 2
        center_y = y_min + height / 2

        width *= expand
        height *= expand

        size = max(width, height)

        x_min = int(center_x - size / 2)
        x_max = int(center_x + size / 2)
        y_min = int(center_y - size / 2)
        y_max = int(center_y + size / 2)

        top = max(0, -y_min)
        bottom = max(0, y_max - img.shape[0])
        left = max(0, -x_min)
        right = max(0, x_max - img.shape[1])
        img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=0)

        cropped_img = img[y_min + top:y_max + top, x_min + left:x_max + left]

    return cropped_img

def euler_and_translation_to_matrix(euler_angles, translation_vector):
    rotation = R.from_euler("xyz", euler_angles, degrees=True)
    rotation_matrix = rotation.as_matrix()

    matrix = np.eye(4)
    matrix[:3, :3] = rotation_matrix
    matrix[:3, 3] = translation_vector
    return matrix