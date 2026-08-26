import cv2
import numpy as np
from scipy.special import expit

import sys
sys.path.append('../../util')
from math_utils import softmax

num_coords = 16
x_scale = 128.0
y_scale = 128.0
h_scale = 128.0
w_scale = 128.0
min_score_thresh = 0.75 # 0.75
min_suppression_threshold = 0.3
num_keypoints = 6

# mediapipe/modules/face_landmark/face_detection_front_detection_to_roi.pbtxt
kp1 = 1  # Left eye
kp2 = 0  # Right eye
theta0 = 0
dscale = 1.5
dy = 0.

resolution = 192

EYE_LEFT_CONTOUR = [
    249, 263, 362, 373, 374,
    380, 381, 382, 384, 385,
    386, 387, 388, 390, 398, 466
]
EYE_RIGHT_CONTOUR = [
    7, 33, 133, 144, 145,
    153, 154, 155, 157, 158,
    159, 160, 161, 163, 173, 246
]


def resize_image(img, out_size, keep_aspect_ratio=True, return_scale_padding=False):
    """
    Resizes the input image to the desired size, keeping the original aspect
    ratio or not.

    Parameters
    ----------
    img: NumPy array
        The image to resize.
    out_size: int or (int, int)  (height, width)
        Resizes the image to the desired size.
    keep_aspect_ratio: bool (default: True)
        If true, resizes while keeping the original aspect ratio. Adds zero-
        padding if necessary.
    return_scale_padding: bool (default: False)
        If true, returns the scale and padding for each dimensions.

    Returns
    -------
    resized: NumPy array
        Resized image.
    scale: NumPy array, optional
        Resized / original, (scale_height, scale_width).
    padding: NumPy array, optional
        Zero padding (top, bottom, left, right) added after resizing.
    """
    img_size = img.shape[:2]
    if isinstance(out_size, int):
        out_size = np.array([out_size, out_size], dtype=int)
    else: # Assuming sequence of len 2
        out_size = np.array(out_size, dtype=int)
    scale = img_size / out_size
    padding = np.zeros(4, dtype=int)

    if img_size[0] != img_size[1] and keep_aspect_ratio:
        scale_long_side = np.max(scale)
        size_new = (img_size / scale_long_side).astype(int)
        padding = out_size - size_new
        padding = np.stack((padding // 2, padding - padding // 2), axis=1).flatten()
        scale[:] = scale_long_side
        resized = cv2.resize(img, (size_new[1], size_new[0]))
        resized = cv2.copyMakeBorder(resized, *padding, cv2.BORDER_CONSTANT, 0)
    else:
        resized = cv2.resize(img, (out_size[1], out_size[0]))

    if return_scale_padding:
        return resized, scale, padding
    else:
        return resized

def face_detector_preprocess(img):
    """
    Preprocesses the image for the face detector.

    Parameters
    ----------
    img: NumPy array
        The image to format in BGR channel order.

    Returns
    -------
    input_face_det: NumPy array
        Formatted image.
    scale: NumPy array
        Resized / original, (scale_height, scale_width)
    padding: NumPy array
        Zero padding (top, bottom, left, right) added after resizing
    """
    input_face_det, scale, padding = resize_image(img[..., ::-1], 128, return_scale_padding=True)
    input_face_det = input_face_det.astype(np.float32) / 127.5 - 1.0
    input_face_det = np.moveaxis(input_face_det, -1, 0)[np.newaxis]
    return input_face_det, scale, padding

def decode_boxes(raw_boxes, anchors):
    """Converts the predictions into actual coordinates using
    the anchor boxes. Processes the entire batch at once.
    """
    boxes = np.zeros_like(raw_boxes)

    x_center = raw_boxes[..., 0] / x_scale * anchors[:, 2] + anchors[:, 0]
    y_center = raw_boxes[..., 1] / y_scale * anchors[:, 3] + anchors[:, 1]

    w = raw_boxes[..., 2] / w_scale * anchors[:, 2]
    h = raw_boxes[..., 3] / h_scale * anchors[:, 3]

    boxes[..., 0] = y_center - h / 2.  # ymin
    boxes[..., 1] = x_center - w / 2.  # xmin
    boxes[..., 2] = y_center + h / 2.  # ymax
    boxes[..., 3] = x_center + w / 2.  # xmax

    for k in range(num_keypoints):
        offset = 4 + k*2
        keypoint_x = raw_boxes[..., offset] / x_scale * anchors[:, 2] + anchors[:, 0]
        keypoint_y = raw_boxes[..., offset + 1] / y_scale * anchors[:, 3] + anchors[:, 1]
        boxes[..., offset] = keypoint_x
        boxes[..., offset + 1] = keypoint_y

    return boxes

def raw_output_to_detections(raw_box, raw_score, anchors):
    """The output of the neural network is an array of shape (b, 896, 16)
    containing the bounding box regressor predictions, as well as an array
    of shape (b, 896, 1) with the classification confidences.

    This function converts these two "raw" arrays into proper detections.
    Returns a list of (num_detections, 13) arrays, one for each image in
    the batch.

    This is based on the source code from:
    mediapipe/calculators/tflite/tflite_tensors_to_detections_calculator.cc
    mediapipe/calculators/tflite/tflite_tensors_to_detections_calculator.proto
    """
    detection_boxes = decode_boxes(raw_box, anchors)

    thresh = 100.0
    raw_score = raw_score.clip(-thresh, thresh)
    # (instead of defining our own sigmoid function which yields a warning)
    # expit = sigmoid
    detection_scores = expit(raw_score).squeeze(axis=-1)

    # Note: we stripped off the last dimension from the scores tensor
    # because there is only has one class. Now we can simply use a mask
    # to filter out the boxes with too low confidence.
    mask = detection_scores >= min_score_thresh

    # Because each image from the batch can have a different number of
    # detections, process them one at a time using a loop.
    output_detections = []
    for i in range(raw_box.shape[0]):
        boxes = detection_boxes[i, mask[i]]
        scores = np.expand_dims(detection_scores[i, mask[i]], axis=-1)
        output_detections.append(np.concatenate((boxes, scores), axis=-1))

    return output_detections

def intersect(box_a, box_b):
    """ We resize both tensors to [A,B,2] without new malloc:
    [A,2] -> [A,1,2] -> [A,B,2]
    [B,2] -> [1,B,2] -> [A,B,2]
    Then we compute the area of intersect between box_a and box_b.
    Args:
      box_a: (tensor) bounding boxes, Shape: [A,4].
      box_b: (tensor) bounding boxes, Shape: [B,4].
    Return:
      (tensor) intersection area, Shape: [A,B].
    """
    A = box_a.shape[0]
    B = box_b.shape[0]
    max_xy = np.minimum(
        np.repeat(np.expand_dims(box_a[:, 2:], axis=1), B, axis=1),
        np.repeat(np.expand_dims(box_b[:, 2:], axis=0), A, axis=0),

    )
    min_xy = np.maximum(
        np.repeat(np.expand_dims(box_a[:, :2], axis=1), B, axis=1),
        np.repeat(np.expand_dims(box_b[:, :2], axis=0), A, axis=0),

    )
    inter = np.clip((max_xy - min_xy), 0, None)
    return inter[:, :, 0] * inter[:, :, 1]

def jaccard(box_a, box_b):
    """Compute the jaccard overlap of two sets of boxes.  The jaccard overlap
    is simply the intersection over union of two boxes.  Here we operate on
    ground truth boxes and default boxes.
    E.g.:
        A ∩ B / A ∪ B = A ∩ B / (area(A) + area(B) - A ∩ B)
    Args:
        box_a: (tensor) Ground truth bounding boxes, Shape: [num_objects,4]
        box_b: (tensor) Prior boxes from priorbox layers, Shape: [num_priors,4]
    Return:
        jaccard overlap: (tensor) Shape: [box_a.size(0), box_b.size(0)]
    """
    inter = intersect(box_a, box_b)
    area_a = np.repeat(
        np.expand_dims(
            (box_a[:, 2]-box_a[:, 0]) * (box_a[:, 3]-box_a[:, 1]),
            axis=1,
        ),
        inter.shape[1],
        axis=1,
    )  # [A,B]
    area_b = np.repeat(
        np.expand_dims(
            (box_b[:, 2]-box_b[:, 0]) * (box_b[:, 3]-box_b[:, 1]),
            axis=0,
        ),
        inter.shape[0],
        axis=0,
    )  # [A,B]
    union = area_a + area_b - inter
    return inter / union  # [A,B]

def overlap_similarity(box, other_boxes):
    """Computes the IOU between a bounding box and set of other boxes."""
    return jaccard(np.expand_dims(box, axis=0), other_boxes).squeeze(0)

def weighted_non_max_suppression(detections):
    """The alternative NMS method as mentioned in the BlazeFace paper:

    "We replace the suppression algorithm with a blending strategy that
    estimates the regression parameters of a bounding box as a weighted
    mean between the overlapping predictions."

    The original MediaPipe code assigns the score of the most confident
    detection to the weighted detection, but we take the average score
    of the overlapping detections.

    The input detections should be a Tensor of shape (count, 17).

    Returns a list of PyTorch tensors, one for each detected face.

    This is based on the source code from:
    mediapipe/calculators/util/non_max_suppression_calculator.cc
    mediapipe/calculators/util/non_max_suppression_calculator.proto
    """
    if len(detections) == 0:
        return []

    output_detections = []

    # Sort the detections from highest to lowest score.
    # argsort() returns ascending order, therefore read the array from end
    remaining = np.argsort(detections[:, num_coords])[::-1]

    while len(remaining) > 0:
        detection = detections[remaining[0]]

        # Compute the overlap between the first box and the other
        # remaining boxes. (Note that the other_boxes also include
        # the first_box.)
        first_box = detection[:4]
        other_boxes = detections[remaining, :4]
        ious = overlap_similarity(first_box, other_boxes)

        # If two detections don't overlap enough, they are considered
        # to be from different faces.
        mask = ious > min_suppression_threshold
        overlapping = remaining[mask]
        remaining = remaining[~mask]

        # Take an average of the coordinates from the overlapping
        # detections, weighted by their confidence scores.
        weighted_detection = detection.copy()
        if len(overlapping) > 1:
            coordinates = detections[overlapping, :num_coords]
            scores = detections[overlapping, num_coords:num_coords+1]
            total_score = scores.sum()
            weighted = (coordinates * scores).sum(axis=0) / total_score
            weighted_detection[:num_coords] = weighted
            weighted_detection[num_coords] = total_score / len(overlapping)

        output_detections.append(weighted_detection)

    return output_detections

def face_detector_postprocess(preds, anchor_path='anchors.npy'):
    """
    Process detection predictions and return filtered detections
    """
    raw_box = preds[0]  # (1, 896, 16)
    raw_score = preds[1]  # (1, 896, 1)

    anchors = np.load(anchor_path).astype("float32")

    # Postprocess the raw predictions:
    detections = raw_output_to_detections(raw_box, raw_score, anchors)

    # Non-maximum suppression to remove overlapping detections:
    filtered_detections = []
    for i in range(len(detections)):
        faces = weighted_non_max_suppression(detections[i])
        faces = np.stack(faces) if len(faces) > 0 else np.zeros((0, num_coords+1))
        filtered_detections.append(faces)

    return filtered_detections

def denormalize_detections(detections, resized_size, scale, pad):
    """ maps detection coordinates from [0,1] to image coordinates

    The input image is padded and resized to fit the
    size while maintaing the aspect ratio. This function maps the
    normalized coordinates back to the original image coordinates.

    Inputs:
        detections: nxm tensor. n is the number of detections.
            m is 4+2*k where the first 4 valuse are the bounding
            box coordinates and k is the number of additional
            keypoints output by the detector.
        resized_size: size of the resized image (i.e. input image)
        scale: scalar that was used to resize the image
        pad: padding in the x (left) and y (top) dimensions

    """
    detections[:, 0] = (detections[:, 0] * resized_size - pad[0]) * scale
    detections[:, 1] = (detections[:, 1] * resized_size - pad[1]) * scale
    detections[:, 2] = (detections[:, 2] * resized_size - pad[0]) * scale
    detections[:, 3] = (detections[:, 3] * resized_size - pad[1]) * scale

    detections[:, 4::2] = (detections[:, 4::2] * resized_size - pad[1]) * scale
    detections[:, 5::2] = (detections[:, 5::2] * resized_size - pad[0]) * scale
    return detections

def detection2roi(detection, detection2roi_method='box'):
    """ Convert detections from detector to an oriented bounding box.

    Adapted from:
    mediapipe/modules/face_landmark/face_detection_front_detection_to_roi.pbtxt

    The center and size of the box is calculated from the center
    of the detected box. Rotation is calculated from the vector
    between kp1 and kp2 relative to theta0. The box is scaled
    and shifted by dscale and dy.

    """
    if detection2roi_method == 'box':
        # compute box center and scale
        # use mediapipe/calculators/util/detections_to_rects_calculator.cc
        xc = (detection[:, 1] + detection[:, 3]) / 2
        yc = (detection[:, 0] + detection[:, 2]) / 2
        scale = (detection[:, 3] - detection[:, 1])  # assumes square boxes

    elif detection2roi_method == 'alignment':
        # compute box center and scale
        # use mediapipe/calculators/util/alignment_points_to_rects_calculator.cc
        xc = detection[:, 4+2*kp1]
        yc = detection[:, 4+2*kp1+1]
        x1 = detection[:, 4+2*kp2]
        y1 = detection[:, 4+2*kp2+1]
        scale = np.sqrt(((xc-x1)**2 + (yc-y1)**2)) * 2
    else:
        raise NotImplementedError(
            "detection2roi_method [%s] not supported" % detection2roi_method
        )

    yc += dy * scale
    scale *= dscale

    # compute box rotation
    x0 = detection[:, 4+2*kp1]
    y0 = detection[:, 4+2*kp1+1]
    x1 = detection[:, 4+2*kp2]
    y1 = detection[:, 4+2*kp2+1]
    theta = np.arctan2(y0-y1, x0-x1) - theta0
    return xc, yc, scale, theta

def extract_roi(frame, xc, yc, theta, scale):
    # take points on unit square and transform them according to the roi
    points = np.array([[-1, -1, 1, 1], [-1, 1, -1, 1]]).reshape(1, 2, 4)
    points = points * scale.reshape(-1, 1, 1)/2
    theta = theta.reshape(-1, 1, 1)
    R = np.concatenate((
        np.concatenate((np.cos(theta), -np.sin(theta)), 2),
        np.concatenate((np.sin(theta), np.cos(theta)), 2),
    ), 1)
    center = np.concatenate((xc.reshape(-1, 1, 1), yc.reshape(-1, 1, 1)), 1)
    points = R @ points + center

    # use the points to compute the affine transform that maps
    # these points back to the output square
    res = resolution
    points1 = np.array([[0, 0, res-1], [0, res-1, 0]], dtype='float32').T
    affines = []
    imgs = []
    for i in range(points.shape[0]):
        pts = points[i, :, :3].T.astype('float32')
        M = cv2.getAffineTransform(pts, points1)
        img = cv2.warpAffine(frame, M, (res, res), borderValue=127.5)
        imgs.append(img)
        affine = cv2.invertAffineTransform(M).astype('float32')
        affines.append(affine)
    if imgs:
        imgs = np.moveaxis(np.stack(imgs), 3, 1).astype('float32') / 127.5 - 1.0
        affines = np.stack(affines)
    else:
        imgs = np.zeros((0, 3, res, res))
        affines = np.zeros((0, 2, 3))

    return imgs, affines, points

def face_lm_preprocess(img, detections, scale, padding):
    """
    Preprocesses the image and face detections for the face landmarks estimator.

    Parameters
    ----------
    img: NumPy array
        The image to format in BGR channel order.
    detections: NumPy array
        Face detections.
    scale: NumPy array
        Scale used when preprocessing the image for the face detection.
        Resized / original, (scale_height, scale_width)
    padding: NumPy array
        Padding used when preprocessing the image for the face detection.
        Zero padding (top, bottom, left, right) added after resizing

    Returns
    -------
    input_face_lm: NumPy array
        Formatted image.
    affines: NumPy array
        Affine transform that maps points in the cropped 192x192 image back to
        the original image
    centers: NumPy array
        Center(s) (x, y) of the cropped faces.
    theta: NumPy array
        rotation angle(s) in radians of the cropping bounding boxes.
    """
    # Only handles detections from the 1st image
    detections = denormalize_detections(detections[0], 128, scale[0], padding[[0, 2]])
    xc, yc, roi_scale, theta = detection2roi(detections)
    input_face_lm, affine, _ = extract_roi(img[..., ::-1], xc, yc, theta, roi_scale)
    centers = np.stack((xc, yc), axis=1)

    return input_face_lm, affine, centers, theta

def denormalize_landmarks(landmarks, affines):
    landmarks = landmarks.reshape((landmarks.shape[0], -1, 3))
    landmarks[:, :, :] *= resolution
    for i in range(len(landmarks)):
        landmark, affine = landmarks[i], affines[i]
        landmark = (affine[:, :2] @ landmark[:, :2].T + affine[:, 2:]).T
        landmarks[i, :, :2] = landmark
    return landmarks

def face_lm_postprocess(landmarks, affines):
    """Compute eye centers from eye contour.

    Parameters
    ----------
    landmarks: NumPy array
        Raw face landmark predictions.
    affines: NumPy array
        Affine transform that maps points in the cropped 192x192 image back to
        the original image

    Returns
    -------
    eye_centers: NumPy array
        Estimated eye centers.
    """
    landmarks_ = landmarks.copy()
    landmarks_ = denormalize_landmarks(landmarks_ / resolution, affines)

    eye_left_centers = landmarks_[:, EYE_LEFT_CONTOUR, :2].mean(axis=1)
    eye_right_centers = landmarks_[:, EYE_RIGHT_CONTOUR, :2].mean(axis=1)
    eye_centers = np.concatenate((eye_left_centers, eye_right_centers), axis=1)
    eye_centers = eye_centers.reshape((-1, 2, 2))
    eye_centers = np.round(eye_centers).astype(int)

    return eye_centers

def iris_preprocess(imgs, raw_landmarks):
    """
    Crop (and flip) eye region image.

    Inputs:
        imgs: 192x192 Face Mesh input images
        raw_landmarks: Face Mesh landmarks with shape (1, 1404) and scale [0, 192]
    Outputs:
        imgs_cropped: 64x64 cropped (and flipped for left eye) eye region images
        origins: upper left (upper right for left eye) corner coordinates of
                 the cropped images in the 192x192 images
    """
    landmarks = raw_landmarks.reshape((-1, 3))

    imgs_cropped = []
    origins = []
    for i in range(len(imgs)):
        eye_left_center = landmarks[EYE_LEFT_CONTOUR, :2].mean(axis=0)
        eye_right_center = landmarks[EYE_RIGHT_CONTOUR, :2].mean(axis=0)

        x_left, y_left = map(int, np.round(eye_left_center - 32))
        # Horizontal flip
        imgs_cropped.append(imgs[i, :, y_left:y_left+64, x_left+63:x_left-1:-1])
        origins.append((x_left+63, y_left))

        x_right, y_right = map(int, np.round(eye_right_center - 32))
        imgs_cropped.append(imgs[i, :, y_right:y_right+64, x_right:x_right+64])
        origins.append((x_right, y_right))

    return np.stack(imgs_cropped), np.stack(origins)

def iris_postprocess(eyes, iris, origins, affines):
    """
    Convert local eye region image coordinates to original image coordinates.

    Inputs:
        eyes: raw eye landmarks output from MediaPipe Iris
        iris: raw iris landmarks output from MediaPipe Iris
        origins: upper left (upper right for left eye) corner coordinates of
                 the cropped images in the 192x192 images
        affines: affine transform that maps points in the 192x192 image back to
                 the original image
    Outputs:
        pupil_centers, (eyes, iris)
    """
    eyes = eyes.copy().reshape((-1, 71, 3))
    iris = iris.copy().reshape((-1, 5, 3))

    # Horizontally flipped left eye processing
    eyes[::2, :, 0] = -eyes[::2, :, 0]
    iris[::2, :, 0] = -iris[::2, :, 0]

    eyes[:, :, :2] += origins[:, None]
    iris[:, :, :2] += origins[:, None]

    iris_landmarks = np.concatenate((eyes, iris), axis=1)
    iris_landmarks = iris_landmarks.reshape((eyes.shape[0] // 2, -1, 3))
    iris_landmarks = denormalize_landmarks(iris_landmarks / resolution, affines)

    iris_landmarks = iris_landmarks.reshape((-1, 2, 76, 3)).round().astype(int)
    eyes = iris_landmarks[:, :, :71]
    iris = iris_landmarks[:, :, 71:]
    pupil_centers = iris[:, :, 0].copy()

    return pupil_centers, (eyes, iris)

def head_pose_preprocess(imgs):
    """
    Preprocesses the image(s) and face detections for the head pose estimator.

    Parameters
    ----------
    imgs: NumPy array
        The image(s) to format with values in the range [-1, 1].

    Returns
    -------
    input_hp: NumPy array
        Formatted image(s).
    """
    tmp = (np.moveaxis(imgs, 1, -1) + 1) / 2
    input_hp = np.empty((tmp.shape[0], 224, 224, 3), dtype=tmp.dtype)
    for i in range(len(tmp)):
        input_hp[i] = cv2.resize(tmp[i], (224, 224))
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape((1, 1, 1, 3))
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape((1, 1, 1, 3))
    input_hp = (input_hp - mean) / std
    input_hp = np.moveaxis(input_hp, -1, 1)

    return input_hp

def head_pose_postprocess(preds_hp, theta):
    """
    Postprocesses the raw head pose predictions (scores for yaw, pitch, roll)
    and returns the head poses (roll, yaw, pitch) in radians.

    Parameters
    ----------
    preds_hp: NumPy array
        Raw head pose predictions.
    theta: NumPy array
        rotation angle(s) in radians of the cropping bounding boxes.

    Returns
    -------
    head_pose: NumPy array
        Roll (left+), yaw (right+), pitch (down+) in radians in the input
        image coordinates (of the head pose network).
    """
    head_pose = np.empty((len(preds_hp[0]),3), dtype=np.float32)
    for i_new, i in enumerate([2, 0, 1]):
        score = preds_hp[i]
        pred = softmax(score)
        tmp = (pred * np.arange(66)[np.newaxis]).sum(axis=1)
        head_pose[:, i_new] = (tmp * 3 - 99)
    # At this point, we have roll left+, yaw right+, pitch up+ in degrees
    head_pose *= np.pi / 180
    head_pose[:, 2] *= -1 # pitch down+

    head_pose_orig = head_pose.copy()
    head_pose_orig[:, 0] += theta
    return head_pose, head_pose_orig

def gaze_postprocess(gazes, affines):
    """Get the gaze vector(s) from the raw predictions.

    Parameters
    ----------
    gazes : NumPy array
        Raw gaze predictions (phi, theta).
    affines : NumPy array
        Affine transform(s) to get back to the original image.

    Returns
    -------
    gaze_vec : NumPy array
        Predicted 3D (x, y, z) gaze vector(s). The axes of
        reference correspond to x oriented positively to the right of the
        image, y oriented positively to the bottom of the image and z
        oriented positively to the back of the image (from the POV of
        someone looking at the image).
    """
    g_phi, g_theta = gazes[:, 0], gazes[:, 1]
    gaze_x = np.cos(g_theta) * -np.sin(g_phi)
    gaze_y = np.sin(g_theta)
    gaze_z = np.cos(g_theta) * -np.cos(g_phi)
    gaze_vec = np.stack((gaze_x, gaze_y, gaze_z), axis=1)
    rot_mat = affines[:, :, :2] / np.sqrt(np.linalg.det(affines[:, :, :2]))
    gaze_vec[:, :2] = np.einsum('ijk,ik->ij', rot_mat, gaze_vec[:, :2])    
    return gaze_vec

def draw_roi(img, roi):
    for i in range(roi.shape[0]):
        (x1, x2, x3, x4), (y1, y2, y3, y4) = roi[i]
        cv2.line(img, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 0), 2)
        cv2.line(img, (int(x1), int(y1)), (int(x3), int(y3)), (0, 255, 0), 2)
        cv2.line(img, (int(x2), int(y2)), (int(x4), int(y4)), (0, 0, 0), 2)
        cv2.line(img, (int(x3), int(y3)), (int(x4), int(y4)), (0, 0, 0), 2)

def draw_landmarks(img, points, color=(0, 0, 255), size=2):
    for point in points:
        x, y = point
        x, y = int(x), int(y)
        cv2.circle(img, (x, y), size, color, thickness=cv2.FILLED)

def draw_eye_iris(
    img,
    eyes,
    iris,
    eye_color=(0, 0, 255),
    iris_color=(255, 0, 0),
    iris_pt_color=(0, 255, 0),
    size=1,
):
    """
    TODO: docstring
    """
    EYE_CONTOUR_ORDERED = [
        0, 1, 2, 3, 4, 5, 6, 7, 8, 15, 14, 13, 12, 11, 10, 9
    ]

    for i in range(2):
        pts = eyes[i, EYE_CONTOUR_ORDERED, :2].round().astype(np.int32)
        pts = pts.reshape((-1, 1, 2))
        cv2.polylines(img, [pts], True, eye_color, thickness=size)

        center = tuple(iris[i, 0])
        radius = int(np.linalg.norm(iris[i, 1] - iris[i, 0]).round())
        cv2.circle(img, center, radius, iris_color, thickness=size)
        draw_landmarks(img, iris[i], color=iris_pt_color, size=size)

def get_rot_mat(axis, angle):
    """
    Creates rotation matrix from axis (x, y or z) and angle. The axes of
    reference correspond to x oriented positively to the left of the image,
    y oriented positively to the bottom of the image and z oriented
    positively to the back of the image.

    Parameters
    ----------
    axis: str
        Axis of rotation. Only x, y and z are supported.
    angle: float
        Angle of rotation in radians.

    Returns
    -------
    rot_mat: NumPy array
        Rotation matrix
        Head pose(s) in radians. Roll (left+), yaw (right+), pitch (down+)
        values are given in the detected person's frame of reference.
    """
    rot_mat = np.zeros((3, 3), dtype=np.float32)
    if axis == 'z':
        i = 2
    elif axis == 'y':
        i = 1
    elif axis == 'x':
        i = 0
    else:
        raise ValueError(f'Axis {axis} is not a valid argument.')

    rot_mat[i, i] = 1
    rot_mat[i-1, i-1] = np.cos(angle)
    rot_mat[i-1, i-2] = np.sin(angle)
    rot_mat[i-2, i-1] = -np.sin(angle)
    rot_mat[i-2, i-2] = np.cos(angle)
    return rot_mat

def draw_head_poses(img, head_poses, centers, horizontal_flip=False):
    """
    Draws the head pose(s) on the image. (Person POV) The axes correspond to
    x (blue) oriented positively to the left, y (green) oriented positively
    to the bottom and z (red) oriented positively to the back.

    Parameters
    ----------
    img: NumPy array
        The image to draw on (BGR channels).
    head_poses: NumPy array
        The head pose(s) to draw.
    centers: NumPy array
        The center(s) of origin of the head pose(s).
    horizontal_flip: bool
        Whether to consider a horizontally flipped image for drawing.
    """
    for hp, c in zip(head_poses, centers):
        rot_mat = get_rot_mat('z', hp[0])
        rot_mat = rot_mat @ get_rot_mat('y', hp[1])
        rot_mat = rot_mat @ get_rot_mat('x', hp[2])
        hp_vecs = rot_mat.T # Each row is rotated x, y, z respectively
        
        if horizontal_flip:
            hp_vecs[0, 1] *= -1
            hp_vecs[1:, 0] *= -1
            c[0] = img.shape[1] - c[0]

        for i, vec in enumerate(hp_vecs):
            tip = tuple((c + 100 * vec[:2]).astype(int))
            color = [0, 0, 0]
            color[i] = 255
            cv2.arrowedLine(img, tuple(c.astype(int)), tip, tuple(color), thickness=2)

def draw_gazes(img, gazes, pupil_centers, horizontal_flip=False, base_color='r', num_bins=10, radius=100, thickness=4):
    """Draw the gaze vector(s) on the image.

    Color-coded depth: the vector has a certain color at the origin and changes
    as it goes closer/farther from the image plane. By default, orange
    corresponds to the image plane depth, red is the minimum depth (the
    closest) and yellow is the maximum (the farthest).
    The vector also gets bigger/thicker as it gets closer.

    Parameters
    ----------
    img : NumPy array
        The image to draw on (BGR channels).
    gazes : NumPy array
        The gaze(s) to draw.
    pupil_centers : NumPy array
        The pupil center(s) (origin of the gaze vector(s)).
    horizontal_flip : bool, optional
        Whether to consider a horizontally flipped image for drawing.
    base_color : str, optional
        Base color for the color range.
    num_bins : int, optional
        Number of bins to segment the vector into for the color range, i.e. the
        maximum number of different colors the vector can have.
    radius : int, optional
        Scaling factor for the gaze vector.
    thickness : int, optional
        Thickness of the gaze vector.
    """
    if horizontal_flip:
        gazes_draw = gazes.copy()
        gazes_draw[:, 0] *= -1
        pupil_centers_draw = pupil_centers.copy()
        pupil_centers_draw[..., 0] = img.shape[1] - pupil_centers_draw[..., 0]
    else:
        gazes_draw = gazes
        pupil_centers_draw = pupil_centers

    for gaze, pupils in zip(gazes_draw, pupil_centers_draw):
        for pc in pupils:
            thickness_ = thickness

            # Create depth bins and associated colors
            assert(num_bins % 2 == 0) # Equal bin size from 0 to 1/-1
            depth_bins_edges = np.linspace(-1, 1, num_bins+1)
            if base_color == 'r':
                base_idx = 2
            elif base_color == 'b':
                base_idx = 0
            else:
                raise ValueError
            bins_color = np.empty((num_bins, 3))
            bins_color[:, base_idx] = 255
            bins_color[:, base_idx-1] = [(np.round(i)) for i in np.linspace(0, 255, num_bins)]
            bins_color[:, base_idx-2] = 0

            # Section gaze vector into corresponding bins
            if gaze[2] > 0:
                bins_valid = np.where((depth_bins_edges >= 0) & (depth_bins_edges < gaze[2]))[0] - num_bins
                thickness_step = -(thickness_-1) / (num_bins / 2 - 1)
            elif gaze[2] < 0:
                bins_valid = np.where((depth_bins_edges > gaze[2]) & (depth_bins_edges <= 0))[0] - 1
                bins_valid = bins_valid[::-1]
                thickness_step = (thickness_-1) / (num_bins / 2 - 1)
            else:
                bins_valid = len(depth_bins_edges) // 2 - num_bins
                thickness_step = 0

            # Draw gaze vector with color varying with depth and the closer to the camera, the bigger the arrow
            x0, y0 = pc[:2]
            x1, y1 = x0, y0
            bin_idx = bins_valid[0]
            if len(bins_valid) > 1: # Avoid dividing by small z value
                scale = radius * depth_bins_edges[bin_idx] / gaze[2]
                x2 = int(np.round(x0 + scale * gaze[0]))
                y2 = int(np.round(y0 + scale * gaze[1]))
                for i_bin_next in range(1, len(bins_valid)):
                    bin_idx_next = bins_valid[i_bin_next]
                    if i_bin_next == len(bins_valid) - 1:
                        scale = radius
                    else:
                        scale = radius * depth_bins_edges[bin_idx_next] / gaze[2]
                    x3 = int(np.round(x0 + scale * gaze[0]))
                    y3 = int(np.round(y0 + scale * gaze[1]))

                    if x2 != x3 or y2 != y3: # If next end point is not the same after rounding
                        cv2.line(img, (x1, y1), (x2, y2), bins_color[bin_idx], thickness=int(np.round(thickness_)))
                        x1, y1 = x2, y2
                        x2, y2 = x3, y3
                        bin_idx = bin_idx_next
                        thickness_ += thickness_step
            else:
                x2 = int(np.round(x0 + radius * gaze[0]))
                y2 = int(np.round(y0 + radius * gaze[1]))

            # Adjust the tip size to match the whole length of the gaze vector
            if np.sqrt((x2 - x1)**2 + (y2 - y1)**2) != 0:
                tip_length = np.sqrt((x2 - x0)**2 + (y2 - y0)**2) / np.sqrt((x2 - x1)**2 + (y2 - y1)**2) * 0.2
            else:
                tip_length = 0.2
            tip_length = min(tip_length, 1.0)
            cv2.arrowedLine(img, (x1, y1), (x2, y2), bins_color[bins_valid[-1]],
                            thickness=int(np.round(thickness_)), tipLength=tip_length)
