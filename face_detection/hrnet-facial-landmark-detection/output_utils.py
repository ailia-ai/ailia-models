import json

import cv2
import numpy as np

# WFLW 98-landmark contour definitions (0-indexed)
# Face jawline: 0-32
JAWLINE = np.arange(0, 33)
# Left eyebrow: 33-38, Right eyebrow: 42-46 (and 38-41 are inner)
LEFT_EYEBROW = np.arange(33, 42)
RIGHT_EYEBROW = np.arange(42, 51)
# Nose: 51-59
NOSE_BRIDGE = np.arange(51, 55)
NOSE_TIP = np.arange(55, 60)
# Left eye: 60-67, Right eye: 68-75
LEFT_EYE = np.arange(60, 68)
RIGHT_EYE = np.arange(68, 76)
# Mouth outer: 76-87, inner: 88-95
MOUTH_OUTER = np.arange(76, 88)
MOUTH_INNER = np.arange(88, 96)
# Pupils: 96, 97
PUPILS = np.array([96, 97])

CONTOURS = [
    (JAWLINE, (200, 200, 200), False, 'jawline'),
    (LEFT_EYEBROW, (0, 200, 255), False, 'left_eyebrow'),
    (RIGHT_EYEBROW, (0, 200, 255), False, 'right_eyebrow'),
    (NOSE_BRIDGE, (255, 100, 0), False, 'nose_bridge'),
    (NOSE_TIP, (255, 100, 0), False, 'nose_tip'),
    (LEFT_EYE, (50, 220, 50), True, 'left_eye'),
    (RIGHT_EYE, (50, 220, 50), True, 'right_eye'),
    (MOUTH_OUTER, (0, 80, 255), True, 'mouth_outer'),
    (MOUTH_INNER, (0, 140, 255), True, 'mouth_inner'),
]

AFLW_CONTOURS = []  # draw only points for AFLW


def visualize(image, keypoints, bboxes, score_threshold):
    out = image.copy()
    for pts, bbox in zip(keypoints, bboxes):
        box = np.round(bbox[:4]).astype(int)
        score = float(bbox[4])
        lt = max(1, int(3 * (box[2:] - box[:2]).max() / 256))

        cv2.rectangle(out, tuple(box[:2]), tuple(box[2:]), (0, 255, 0), lt)
        cv2.putText(out, f'{score:.2f}', (box[0], box[1] - 4), 0,
                    lt * 0.4, (255, 255, 255), max(1, lt - 1), cv2.LINE_AA)

        n_pts = len(pts)
        contours = CONTOURS if n_pts == 98 else AFLW_CONTOURS
        for indices, color, closed, _ in contours:
            subset = pts[indices]
            visible = subset[:, 2] >= score_threshold
            if visible.any():
                poly = np.round(subset[:, :2]).astype(np.int32)
                cv2.polylines(out, [poly], closed, color, lt)

        for *xy, conf in pts:
            pt = tuple(np.round(xy).astype(int))
            color = (0, 0, 255) if conf >= score_threshold else (0, 255, 255)
            cv2.circle(out, pt, max(1, lt - 1), color, cv2.FILLED)

    return out


def save_json(json_file, keypoints, bboxes):
    results = []
    for pts, bbox in zip(keypoints, bboxes):
        results.append({
            'bbox': bbox[:4].tolist(),
            'score': float(bbox[4]),
            'landmarks': [
                {'x': float(x), 'y': float(y), 'score': float(s)}
                for x, y, s in pts
            ],
        })
    with open(json_file, 'w') as f:
        json.dump(results, f, indent=2)
