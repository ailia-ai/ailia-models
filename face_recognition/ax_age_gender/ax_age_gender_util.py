"""Utilities for the ax_age_gender sample.

BlazeFace detection with the 6 face keypoints, canonical 2-point face
alignment and the head-pose gate, ported from the training-side code so the
demo reproduces the training crop geometry exactly.
"""
import cv2
import numpy as np

# BlazeFace keypoint order
KP_RIGHT_EYE, KP_LEFT_EYE, KP_NOSE, KP_MOUTH, KP_RIGHT_EAR, KP_LEFT_EAR = range(6)

NMS_IOU = 0.3

# Canonical layout, expressed in units of the inter-ocular distance D.
NOSE_DY = 0.5715
MOUTH_DY = 1.1543

# Head pose the two-point alignment does not remove. Yaw and pitch proxies are
# measured against the inter-ocular distance; outside this window the model
# accuracy degrades sharply, so such faces are detected but left unscored.
YAW_MAX = 0.80
PITCH_MIN, PITCH_MAX = 0.30, 0.86


# ======================
# BlazeFace detection
# ======================

def letterbox(img, size):
    """Resize keeping the aspect ratio, pad with black. Returns (img, scale, dx, dy)."""
    h, w = img.shape[:2]
    scale = min(size / w, size / h)
    nw, nh = int(round(w * scale)), int(round(h * scale))
    resized = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)
    out = np.zeros((size, size, img.shape[2]), dtype=img.dtype)
    dx, dy = (size - nw) // 2, (size - nh) // 2
    out[dy:dy + nh, dx:dx + nw] = resized
    return out, scale, dx, dy


def _sigmoid(x):
    x = np.asarray(x, np.float32)
    pos = x >= 0
    out = np.empty_like(x)
    out[pos] = 1.0 / (1.0 + np.exp(-x[pos]))
    e = np.exp(x[~pos])
    out[~pos] = e / (1.0 + e)
    return out


def _decode(raw_boxes, anchors, scale):
    boxes = np.zeros_like(raw_boxes)
    cx = raw_boxes[..., 0] / scale * anchors[:, 2] + anchors[:, 0]
    cy = raw_boxes[..., 1] / scale * anchors[:, 3] + anchors[:, 1]
    w = raw_boxes[..., 2] / scale * anchors[:, 2]
    h = raw_boxes[..., 3] / scale * anchors[:, 3]
    boxes[..., 0] = cy - h / 2.0  # ymin
    boxes[..., 1] = cx - w / 2.0  # xmin
    boxes[..., 2] = cy + h / 2.0  # ymax
    boxes[..., 3] = cx + w / 2.0  # xmax
    for k in range(6):
        o = 4 + k * 2
        boxes[..., o] = raw_boxes[..., o] / scale * anchors[:, 2] + anchors[:, 0]
        boxes[..., o + 1] = raw_boxes[..., o + 1] / scale * anchors[:, 3] + anchors[:, 1]
    return boxes


def _iou(box, others):
    ymin = np.maximum(box[0], others[:, 0])
    xmin = np.maximum(box[1], others[:, 1])
    ymax = np.minimum(box[2], others[:, 2])
    xmax = np.minimum(box[3], others[:, 3])
    inter = np.clip(ymax - ymin, 0, None) * np.clip(xmax - xmin, 0, None)
    a = (box[2] - box[0]) * (box[3] - box[1])
    b = (others[:, 2] - others[:, 0]) * (others[:, 3] - others[:, 1])
    return inter / (a + b - inter + 1e-9)


def _weighted_nms(det):
    if len(det) == 0:
        return np.zeros((0, 17), dtype=np.float32)
    out = []
    remaining = np.argsort(-det[:, 16])
    while len(remaining) > 0:
        best = det[remaining[0]].copy()
        ious = _iou(best[:4], det[remaining, :4])
        overlapping = remaining[ious > NMS_IOU]
        remaining = remaining[ious <= NMS_IOU]
        if len(overlapping) > 1:
            coords = det[overlapping, :16]
            scores = det[overlapping, 16:17]
            best[:16] = (coords * scores).sum(axis=0) / scores.sum()
            best[16] = scores.mean()
        out.append(best)
    return np.stack(out)


class BlazeFace:
    """BlazeFace (back camera, 256px) on the ailia SDK.

    Returns (N, 17) rows: ymin,xmin,ymax,xmax, 6*(kp_x,kp_y), score --
    box and keypoint coordinates in pixels of the input image.
    """

    def __init__(self, net, anchor_path, size=256):
        self.net = net
        self.size = size
        self.anchors = np.load(anchor_path).astype(np.float32)

    def __call__(self, image_bgr, min_score=0.5):
        h, w = image_bgr.shape[:2]
        img, scale, dx, dy = letterbox(image_bgr, self.size)
        blob = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).transpose(2, 0, 1)[None]
        blob = blob.astype(np.float32) / 127.5 - 1.0

        preds = self.net.predict([blob])
        # identify the outputs by their last dimension (boxes: 16, scores: 1)
        raw_box = next(p for p in preds if p.shape[-1] == 16)
        raw_score = next(p for p in preds if p.shape[-1] == 1)

        boxes = _decode(raw_box, self.anchors, float(self.size))
        scores = _sigmoid(np.clip(raw_score, -100.0, 100.0)).squeeze(-1)

        keep = scores[0] >= min_score
        det = np.concatenate([boxes[0][keep], scores[0][keep][:, None]], axis=-1)
        det = _weighted_nms(det.astype(np.float32))
        if len(det) == 0:
            return det

        # normalised letterbox coords -> pixels of the original image
        det[:, [0, 2]] = (det[:, [0, 2]] * self.size - dy) / scale
        det[:, [1, 3]] = (det[:, [1, 3]] * self.size - dx) / scale
        for k in range(6):
            o = 4 + k * 2
            det[:, o] = (det[:, o] * self.size - dx) / scale
            det[:, o + 1] = (det[:, o + 1] * self.size - dy) / scale
        np.clip(det[:, [1, 3]], 0, w, out=det[:, [1, 3]])
        np.clip(det[:, [0, 2]], 0, h, out=det[:, [0, 2]])
        return det

    @staticmethod
    def keypoints(det_row):
        """(6, 2) float array of keypoints from one detection row."""
        return det_row[4:16].reshape(6, 2).astype(np.float32)

    @staticmethod
    def box(det_row):
        """(x1, y1, x2, y2)."""
        ymin, xmin, ymax, xmax = det_row[:4]
        return np.array([xmin, ymin, xmax, ymax], dtype=np.float32)


# ======================
# Canonical face alignment
# ======================

class AlignConfig:
    """Geometry of the aligned crop.

    size
        output crop is ``size`` x ``size``.
    interocular
        distance between the eyes, as a fraction of ``size``.
    eye_y
        vertical position of the eye line, as a fraction of ``size``.
    """

    def __init__(self, size=128, interocular=0.30, eye_y=0.40):
        self.size = int(size)
        self.interocular = float(interocular)
        self.eye_y = float(eye_y)

    @property
    def D(self):
        return self.interocular * self.size


def similarity_from_eyes(right_eye, left_eye, cfg):
    """2x3 affine mapping the two eyes onto the canonical eye positions."""
    v = np.asarray(left_eye, np.float64) - np.asarray(right_eye, np.float64)
    d = float(np.hypot(v[0], v[1]))
    if d < 1e-6:
        return None
    theta = np.arctan2(v[1], v[0])
    s = cfg.D / d
    a, b = s * np.cos(theta), -s * np.sin(theta)
    mx, my = (np.asarray(right_eye, np.float64) + np.asarray(left_eye, np.float64)) / 2.0
    cx, ey = cfg.size / 2.0, cfg.eye_y * cfg.size
    tx = cx - (a * mx - b * my)
    ty = ey - (b * mx + a * my)
    return np.array([[a, -b, tx], [b, a, ty]], dtype=np.float32)


def apply_transform(points, M):
    pts = np.asarray(points, np.float32).reshape(-1, 2)
    return (pts @ M[:, :2].T + M[:, 2]).astype(np.float32)


def align_face(image_bgr, keypoints, cfg, border=cv2.BORDER_REPLICATE):
    """Warp ``image_bgr`` into the canonical frame.

    Returns ``(crop, aligned_keypoints)`` or ``(None, None)`` if the keypoints
    are degenerate.
    """
    kp = np.asarray(keypoints, np.float32)
    M = similarity_from_eyes(kp[KP_RIGHT_EYE], kp[KP_LEFT_EYE], cfg)
    if M is None:
        return None, None
    crop = cv2.warpAffine(image_bgr, M, (cfg.size, cfg.size),
                          flags=cv2.INTER_LINEAR, borderMode=border)
    return crop, apply_transform(kp, M)


# ======================
# Head-pose gate
# ======================

def pose_proxies(keypoints):
    """``(pitch, yaw)`` in inter-ocular units, roll-invariant."""
    kp = np.asarray(keypoints, np.float32)
    eye = kp[KP_LEFT_EYE] - kp[KP_RIGHT_EYE]
    d = float(np.linalg.norm(eye))
    if d < 1e-6:
        return float("nan"), float("nan")
    along = eye / d
    down = np.array([-along[1], along[0]], np.float32)
    v = kp[KP_NOSE] - (kp[KP_LEFT_EYE] + kp[KP_RIGHT_EYE]) / 2.0
    return float(np.dot(v, down) / d), float(np.dot(v, along) / d)


def pose_in_range(aligned_keypoints, yaw_max=YAW_MAX,
                  pitch_range=(PITCH_MIN, PITCH_MAX)):
    """``(ok, pitch, yaw, reason)``. ``reason`` is empty when ok."""
    pitch, yaw = pose_proxies(aligned_keypoints)
    if pitch != pitch or yaw != yaw:  # nan
        return False, pitch, yaw, "degenerate keypoints"
    if abs(yaw) > yaw_max:
        return False, pitch, yaw, f"yaw {abs(yaw):.2f} > {yaw_max:.2f}"
    if pitch < pitch_range[0]:
        return False, pitch, yaw, f"pitch {pitch:.2f} < {pitch_range[0]:.2f}"
    if pitch > pitch_range[1]:
        return False, pitch, yaw, f"pitch {pitch:.2f} > {pitch_range[1]:.2f}"
    return True, pitch, yaw, ""
