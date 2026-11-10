import math
import time

import cv2
import numpy as np


DEGLASS_CENTER_LANDMARK = 168
DEGLASS_EYE_SPAN_LANDMARKS = (143, 372)
DEGLASS_LEFT_DIRECTION_LANDMARKS = (143, 127, 116, 234, 123, 93, 147, 132)
DEGLASS_RIGHT_DIRECTION_LANDMARKS = (372, 356, 345, 454, 352, 323, 376, 361)
DEGLASS_FACE_SIZE_FROM_EYE_SPAN = 0.73
DEGLASS_ROI_WIDTH_FROM_FACE_SIZE = 2.2
DEGLASS_ROI_CENTER_X_DIRECTION_SCALE = 0.3
DEGLASS_ROI_CENTER_Y_OFFSET_FROM_WIDTH = 0.05
DEGLASS_ROI_CENTER_Y_DIRECTION_SCALE = 0.4

LANDMARK_TIME_CONSTANT = 0.02
ROI_TIME_CONSTANT = 0.15


def _normalized(vector):
    length = np.linalg.norm(vector)
    if length <= 1e-8:
        return np.zeros(3, dtype=np.float64)
    return vector / length


def _project_on_plane(vector, normal):
    return vector - normal * np.dot(vector, normal)


def _rotate_axis_angle(vector, axis, angle):
    unit = _normalized(axis)
    if np.all(unit == 0) or abs(angle) <= 1e-8:
        return vector
    cosine = math.cos(angle)
    sine = math.sin(angle)
    return (
        vector * cosine
        + np.cross(unit, vector) * sine
        + unit * np.dot(unit, vector) * (1 - cosine)
    )


def landmarks_to_pixels(landmarks, width, height):
    pixels = np.asarray(landmarks, dtype=np.float64).copy()
    pixels[:, 0] *= width
    pixels[:, 1] *= height
    pixels[:, 2] *= width
    return pixels


def deglass_face_directions(landmarks, width, height):
    """Estimate the face-side direction used by the JavaScript VTO demo."""
    pixels = landmarks_to_pixels(landmarks, width, height)
    scale = max(width, 1)

    def coordinate(index):
        point = pixels[index]
        return np.array(
            [point[0] / scale, -point[1] / scale, -point[2] / scale],
            dtype=np.float64,
        )

    support = _normalized(
        coordinate(DEGLASS_EYE_SPAN_LANDMARKS[1])
        - coordinate(DEGLASS_EYE_SPAN_LANDMARKS[0])
    )
    side_indices = (
        DEGLASS_RIGHT_DIRECTION_LANDMARKS
        if support[2] > 0
        else DEGLASS_LEFT_DIRECTION_LANDMARKS
    )
    center = coordinate(DEGLASS_CENTER_LANDMARK)
    direction = np.zeros(3, dtype=np.float64)
    for pair in range(0, len(side_indices), 2):
        first = _project_on_plane(coordinate(side_indices[pair]) - center, support)
        second = _project_on_plane(
            coordinate(side_indices[pair + 1]) - center, support
        )
        direction += _normalized(first - second)

    horizontal_plane = np.array([direction[0], 0, direction[2]])
    horizontal_unit = _normalized(horizontal_plane)
    if np.any(horizontal_unit != 0):
        angle = math.acos(np.clip(np.dot(horizontal_unit, [0, 0, 1]), -1, 1))
        direction = _rotate_axis_angle(
            direction,
            np.cross(horizontal_plane, [0, 0, 1]),
            -0.05 * angle,
        )

    direction = _normalized(direction)
    horizontal = _normalized(np.array([direction[0], 0, direction[2]]))
    vertical = _normalized(np.array([0, direction[1], direction[2]]))
    return horizontal, vertical


def _js_round(value):
    """Match Math.round, including negative half values."""
    return math.floor(value + 0.5)


def deglass_roi_box(landmarks, width, height, target_width=256, target_height=128):
    """Return the landmark-driven XYXY ROI from the current JavaScript demo."""
    if width <= 0 or height <= 0:
        raise ValueError("Image dimensions must be positive")
    if target_width <= 0 or target_height <= 0:
        raise ValueError("Model dimensions must be positive")

    pixels = landmarks_to_pixels(landmarks, width, height)
    first = pixels[DEGLASS_EYE_SPAN_LANDMARKS[0], :2]
    second = pixels[DEGLASS_EYE_SPAN_LANDMARKS[1], :2]
    eye_span = max(np.linalg.norm(second - first), 1)
    face_size = eye_span * DEGLASS_FACE_SIZE_FROM_EYE_SPAN
    roi_width = face_size * DEGLASS_ROI_WIDTH_FROM_FACE_SIZE
    roi_height = roi_width * target_height / target_width
    horizontal, vertical = deglass_face_directions(landmarks, width, height)

    center_x, center_y = pixels[DEGLASS_CENTER_LANDMARK, :2]
    center_x -= DEGLASS_ROI_CENTER_X_DIRECTION_SCALE * roi_width * horizontal[0]
    center_y += DEGLASS_ROI_CENTER_Y_OFFSET_FROM_WIDTH * roi_width
    if vertical[1] < 0:
        center_y += DEGLASS_ROI_CENTER_Y_DIRECTION_SCALE * roi_width * vertical[1]

    roi_height_pixels = max(1, _js_round(roi_height))
    roi_width_pixels = max(
        1,
        _js_round(roi_height_pixels * target_width / target_height),
    )
    x1 = _js_round(center_x - roi_width_pixels * 0.5)
    y1 = _js_round(center_y - roi_height_pixels * 0.5)
    return np.array(
        [x1, y1, x1 + roi_width_pixels, y1 + roi_height_pixels],
        dtype=np.int32,
    )


def _reflect101_indices(start, end, size):
    if end <= start:
        raise ValueError("ROI bounds must have positive area")
    if size <= 0:
        raise ValueError("Image dimensions must be positive")
    if size == 1:
        return np.zeros(end - start, dtype=np.intp)
    period = 2 * size - 2
    values = np.arange(start, end, dtype=np.int64) % period
    values = np.where(values < size, values, period - values)
    return values.astype(np.intp)


def crop_reflect101(image, bounds):
    """Crop an XYXY box with cv2.BORDER_REFLECT_101 semantics."""
    x1, y1, x2, y2 = (int(value) for value in bounds)
    rows = _reflect101_indices(y1, y2, image.shape[0])
    columns = _reflect101_indices(x1, x2, image.shape[1])
    return image[rows[:, None], columns[None, :]].copy()


def paste_visible_roi(image, roi, bounds):
    """Paste only the part of an out-of-frame ROI that overlaps the image."""
    x1, y1, x2, y2 = (int(value) for value in bounds)
    if roi.shape[1] != x2 - x1 or roi.shape[0] != y2 - y1:
        raise ValueError("ROI shape does not match its bounds")

    visible_x1 = max(0, x1)
    visible_y1 = max(0, y1)
    visible_x2 = min(image.shape[1], x2)
    visible_y2 = min(image.shape[0], y2)
    if visible_x1 >= visible_x2 or visible_y1 >= visible_y2:
        return image

    source_x1 = visible_x1 - x1
    source_y1 = visible_y1 - y1
    source_x2 = source_x1 + visible_x2 - visible_x1
    source_y2 = source_y1 + visible_y2 - visible_y1
    image[visible_y1:visible_y2, visible_x1:visible_x2] = roi[
        source_y1:source_y2, source_x1:source_x2
    ]
    return image


def smoothing_weight(elapsed_seconds, time_constant_seconds):
    if not elapsed_seconds > 0:
        return 1.0
    return 1 - math.exp(-elapsed_seconds / time_constant_seconds)


class TemporalSmoother:
    """Time-based landmark and ROI smoothing used by the JavaScript demo."""

    def __init__(self):
        self.reset()

    def reset(self):
        self.last_landmarks = None
        self.smoothed_roi = None
        self.clocks = {}

    def _elapsed(self, name):
        now = time.perf_counter()
        previous = self.clocks.get(name)
        self.clocks[name] = now
        return math.inf if previous is None else now - previous

    def smooth_landmarks(self, landmarks):
        weight = smoothing_weight(
            self._elapsed("landmarks"), LANDMARK_TIME_CONSTANT
        )
        landmarks = np.asarray(landmarks, dtype=np.float32)
        if self.last_landmarks is not None and weight < 1:
            landmarks = self.last_landmarks + (landmarks - self.last_landmarks) * weight
        self.last_landmarks = landmarks.copy()
        return landmarks

    def smooth_roi(self, bounds):
        weight = smoothing_weight(self._elapsed("roi"), ROI_TIME_CONSTANT)
        bounds = np.asarray(bounds, dtype=np.float64)
        if self.smoothed_roi is None:
            self.smoothed_roi = bounds.copy()
        else:
            self.smoothed_roi += (bounds - self.smoothed_roi) * weight
        return np.array([_js_round(value) for value in self.smoothed_roi], dtype=np.int32)


class GlassesRemover:
    def __init__(self, net, input_width=256, input_height=128):
        self.net = net
        self.input_width = input_width
        self.input_height = input_height

    def _preprocess(self, roi):
        resized = cv2.resize(
            roi,
            (self.input_width, self.input_height),
            interpolation=cv2.INTER_LINEAR,
        )
        resized = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        tensor = resized.astype(np.float32) / 127.5 - 1
        return tensor.transpose(2, 0, 1)[None]

    @staticmethod
    def _outputs(outputs):
        clean = next(
            (output for output in outputs if output.ndim == 4 and output.shape[1] == 3),
            None,
        )
        single_channel = [
            output
            for output in outputs
            if output.ndim == 4 and output.shape[1] == 1
        ]
        if clean is None or not single_channel:
            shapes = ", ".join(str(output.shape) for output in outputs)
            raise RuntimeError(f"Unexpected glasses-removal outputs: {shapes}")
        hole = single_channel[0]
        probability = single_channel[1] if len(single_channel) > 1 else None
        return clean, hole, probability

    def remove(self, frame, bounds, show_mask=False):
        roi = crop_reflect101(frame, bounds)
        outputs = self.net.predict([self._preprocess(roi)])
        clean, hole, probability = self._outputs(outputs)

        clean_rgb = np.clip(clean[0].transpose(1, 2, 0), -1, 1)
        clean_rgb = (clean_rgb + 1) * 127.5
        clean_bgr = clean_rgb[:, :, ::-1]
        alpha = np.clip(hole[0, 0], 0, 1)

        roi_size = (roi.shape[1], roi.shape[0])
        scaled_clean = cv2.resize(clean_bgr, roi_size, interpolation=cv2.INTER_LINEAR)
        scaled_alpha = cv2.resize(alpha, roi_size, interpolation=cv2.INTER_LINEAR)
        scaled_alpha = scaled_alpha[:, :, None]
        composite = scaled_clean * scaled_alpha + roi * (1 - scaled_alpha)
        composite = np.clip(np.rint(composite), 0, 255).astype(np.uint8)

        if show_mask:
            mask = probability[0, 0] if probability is not None else hole[0, 0]
            mask = cv2.resize(mask, roi_size, interpolation=cv2.INTER_LINEAR)
            composite = self._draw_mask(composite, mask)

        result = frame.copy()
        return paste_visible_roi(result, composite, bounds)

    @staticmethod
    def _draw_mask(image, probability):
        value = np.clip(probability, 0, 1)
        tint = np.zeros_like(image, dtype=np.float32)
        tint[:, :, 1] = np.where(value >= 0.3, 255, 90)
        tint[:, :, 2] = np.where(value < 0.3, 255, 0)
        opacity = np.where(value <= 0.02, 0, 150 * np.minimum(1, value * 2))
        opacity = (opacity / 255)[:, :, None]
        overlay = image.astype(np.float32) * (1 - opacity) + tint * opacity
        return np.clip(np.rint(overlay), 0, 255).astype(np.uint8)
