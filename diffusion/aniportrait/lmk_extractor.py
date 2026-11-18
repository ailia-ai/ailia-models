import math
import cv2
import numpy as np

import onnxruntime
import ailia
from ani_portrait_utils import load_canonical_model
from facemesh_v2_utils import preprocess_det, ROI, post_processing, NUM_LANDMARKS, preprocess
from detection_utils import face_detection


def normalize_points(points):
    # 重心を計算
    centroid = np.mean(points, axis=0)
    # 原点周りに中心化
    centered_points = points - centroid
    # スケールを計算 (原点からの二乗平均平方根距離)
    scale = np.sqrt(np.sum(centered_points ** 2) / centered_points.shape[0])
    # 正規化
    normalized_points = centered_points / scale
    return normalized_points, centroid, scale


def estimate_pose(src_points, dst_points) -> np.ndarray:
    src_points = np.array(src_points)
    dst_points = np.array(dst_points)

    # 両方のポイントを正規化
    src_points, src_centroid, src_scale = normalize_points(src_points)
    dst_points, dst_centroid, dst_scale = normalize_points(dst_points)

    # 共分散行列を計算
    covariance_matrix = np.dot(dst_points.T, src_points)

    # 特異値分解 (SVD) を実行
    U, S, Vt = np.linalg.svd(covariance_matrix)

    # 回転行列を計算
    rotation_matrix = np.dot(U, Vt)

    # 適切な回転行列にする (det(R) = 1)
    if np.linalg.det(rotation_matrix) < 0:
        Vt[-1, :] *= -1
        rotation_matrix = np.dot(U, Vt)

    # スケールファクターを計算
    scale = dst_scale / src_scale

    # 平行移動を計算
    translation = dst_centroid - scale * np.dot(src_centroid, rotation_matrix)

    # 変換行列を構築
    transformation_matrix = np.eye(4)
    transformation_matrix[:3, :3] = scale * rotation_matrix
    transformation_matrix[:3, 3] = translation
    return transformation_matrix



class LMKExtractor:
    def __init__(
        self,
        face_landmarks_detector: onnxruntime.InferenceSession | ailia.Net,
        face_detector: onnxruntime.InferenceSession | ailia.Net,
        is_onnx: bool,
    ):
        self.face_landmarks_detector = face_landmarks_detector
        self.face_detector = face_detector
        self.CANONICAL_MODEL = load_canonical_model()
        self.IS_ONNX = is_onnx

    def _predict(self, img):
        im_h, im_w, _ = img.shape
        img = img[:, :, ::-1]  # BGR -> RGB

        input, matrix = preprocess_det(img)

        # feedforward
        if not self.IS_ONNX:
            output = self.face_detector.predict([input])
        else:
            output = self.face_detector.run(None, {'input': input})
        detections, scores = output

        boxes, scores = face_detection(detections, scores, matrix)
        if len(boxes) == 0:
            return np.zeros((0, NUM_LANDMARKS, 3))

        landmarks_list = []
        for box in boxes:
            # DetectionsToRectsCalculator
            rect_width = box[2] - box[0]
            rect_height = box[3] - box[1]
            center_x = (box[0] + box[2]) / 2
            center_y = (box[1] + box[3]) / 2

            x0, y0 = box[4] * im_w, box[5] * im_h
            x1, y1 = box[6] * im_w, box[7] * im_h
            angle = 0 - math.atan2(-(y1 - y0), x1 - x0)
            angle = angle - 2 * math.pi * math.floor((angle - (-math.pi)) / (2 * math.pi));

            # RectTransformationCalculator
            scale_x = scale_y = 1.5
            rect_width = rect_width * scale_x
            rect_height = rect_height * scale_y

            roi = ROI(
                center_x * im_w, center_y * im_h,
                rect_width * im_w, rect_height * im_h,
                angle)
            img, roi, pad = preprocess(img, roi)

            # feedforward
            if not self.IS_ONNX:
                output = self.face_landmarks_detector.predict([img])
            else:
                output = self.face_landmarks_detector.run(None, {'input_12': img})
            landmark_tensors, presence_flag_tensors, _ = output

            norm_rect = ROI(
                roi.x_center / im_w, roi.y_center / im_h,
                roi.width / im_w, roi.height / im_h,
                angle)
            landmarks = post_processing(landmark_tensors, norm_rect, pad)
            landmarks_list.append(landmarks)

        landmarks = np.stack(landmarks_list, axis=0)

        return landmarks

    def __call__(self, img: np.ndarray):
        frame = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
        detection_result = self._predict(frame)
        detection_result_crop = detection_result[0, :468, :].reshape(468, 3)
        trans_mat = estimate_pose(self.CANONICAL_MODEL, detection_result_crop)
        return trans_mat, detection_result