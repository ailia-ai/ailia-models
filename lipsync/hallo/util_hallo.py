import os

import cv2
import mediapipe as mp
import numpy as np

# fmt: off
silhouette_ids = [
    10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288,
    397, 365, 379, 378, 400, 377, 152, 148, 176, 149, 150, 136,
    172, 58, 132, 93, 234, 127, 162, 21, 54, 103, 67, 109
]

lip_ids = [61, 185, 40, 39, 37, 0, 267, 269, 270, 409, 291,
           146, 91, 181, 84, 17, 314, 405, 321, 375]
# fmt: on


def compute_face_landmarks(detection_result, h, w):
    """
    Compute face landmarks from a detection result.
    """
    face_landmarks_list = detection_result.face_landmarks
    if len(face_landmarks_list) != 1:
        print("#face is invalid:", len(face_landmarks_list))
        return []
    return [[p.x * w, p.y * h] for p in face_landmarks_list[0]]


def expand_region(region, image_w, image_h, expand_ratio=1.0):
    """
    Expand the given region by a specified ratio.
    Args:
        region (tuple): A tuple containing the coordinates (min_x, max_x, min_y, max_y) of the region.
        image_w (int): The width of the image.
        image_h (int): The height of the image.
        expand_ratio (float, optional): The ratio by which the region should be expanded. Defaults to 1.0.

    Returns:
        tuple: A tuple containing the expanded coordinates (min_x, max_x, min_y, max_y) of the region.
    """

    min_x, max_x, min_y, max_y = region
    mid_x = (max_x + min_x) // 2
    side_len_x = (max_x - min_x) * expand_ratio
    mid_y = (max_y + min_y) // 2
    side_len_y = (max_y - min_y) * expand_ratio
    min_x = mid_x - side_len_x // 2
    max_x = mid_x + side_len_x // 2
    min_y = mid_y - side_len_y // 2
    max_y = mid_y + side_len_y // 2
    if min_x < 0:
        max_x -= min_x
        min_x = 0
    if max_x > image_w:
        min_x -= max_x - image_w
        max_x = image_w
    if min_y < 0:
        max_y -= min_y
        min_y = 0
    if max_y > image_h:
        min_y -= max_y - image_h
        max_y = image_h

    return round(min_x), round(max_x), round(min_y), round(max_y)


def get_face_mask(landmarks, height, width, expand_ratio=1.2):
    """
    Generate a face mask based on the given landmarks.
    """
    face_landmarks = np.take(landmarks, silhouette_ids, 0)
    min_xy_face = np.round(np.min(face_landmarks, 0))
    max_xy_face = np.round(np.max(face_landmarks, 0))
    min_xy_face[0], max_xy_face[0], min_xy_face[1], max_xy_face[1] = expand_region(
        [min_xy_face[0], max_xy_face[0], min_xy_face[1], max_xy_face[1]],
        width,
        height,
        expand_ratio,
    )
    face_mask = np.zeros((height, width), dtype=np.uint8)
    face_mask[
        round(min_xy_face[1]) : round(max_xy_face[1]),
        round(min_xy_face[0]) : round(max_xy_face[0]),
    ] = 255

    return face_mask


def get_landmark(image):
    """
    This function takes a file as input and returns the facial landmarks detected in the file.

    Args:
        file (str): The path to the file containing the video or image to be processed.

    Returns:
        Tuple[List[float], List[float]]: A tuple containing two lists of floats representing the x and y coordinates of the facial landmarks.
    """
    model_path = "./face_analysis/models/face_landmarker_v2_with_blendshapes.task"
    BaseOptions = mp.tasks.BaseOptions
    FaceLandmarker = mp.tasks.vision.FaceLandmarker
    FaceLandmarkerOptions = mp.tasks.vision.FaceLandmarkerOptions
    VisionRunningMode = mp.tasks.vision.RunningMode
    # Create a face landmarker instance with the video mode:
    options = FaceLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=model_path),
        running_mode=VisionRunningMode.IMAGE,
    )

    with FaceLandmarker.create_from_options(options) as landmarker:
        image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image)
        height, width = image.height, image.width
        face_landmarker_result = landmarker.detect(image)
        face_landmark = compute_face_landmarks(face_landmarker_result, height, width)

    return np.array(face_landmark), height, width


def get_lip_mask(landmarks, height, width, expand_ratio=2.0):
    """
    Extracts the lip region from the given landmarks and saves it as an image.

    Parameters:
        landmarks (numpy.ndarray): Array of facial landmarks.
        height (int): Height of the output lip mask image.
        width (int): Width of the output lip mask image.
        out_path (pathlib.Path): Path to save the lip mask image.
        expand_ratio (float): Expand ratio of mask.
    """
    lip_landmarks = np.take(landmarks, lip_ids, 0)
    min_xy_lip = np.round(np.min(lip_landmarks, 0))
    max_xy_lip = np.round(np.max(lip_landmarks, 0))
    min_xy_lip[0], max_xy_lip[0], min_xy_lip[1], max_xy_lip[1] = expand_region(
        [min_xy_lip[0], max_xy_lip[0], min_xy_lip[1], max_xy_lip[1]],
        width,
        height,
        expand_ratio,
    )
    lip_mask = np.zeros((height, width), dtype=np.uint8)
    lip_mask[
        round(min_xy_lip[1]) : round(max_xy_lip[1]),
        round(min_xy_lip[0]) : round(max_xy_lip[0]),
    ] = 255

    return lip_mask


def get_mask(image, face_expand_raio):
    """
    Generate a face mask based on the given landmarks and save it to the specified cache directory.
    """
    landmarks, height, width = get_landmark(image)

    lip_mask = get_lip_mask(landmarks, height, width)
    face_mask = get_face_mask(landmarks, height, width, face_expand_raio)
    blur_mask = get_blur_mask(face_mask, kernel_size=(51, 51))
    sep_lip_mask = get_blur_mask(lip_mask, kernel_size=(31, 31))
    sep_background_mask = get_background_mask(blur_mask)
    sep_face_mask = get_sep_face_mask(blur_mask, sep_lip_mask)

    return face_mask, sep_lip_mask, sep_background_mask, sep_face_mask


def get_blur_mask(mask, resize_dim=(64, 64), kernel_size=(101, 101)):
    # Resize the mask image
    resized_mask = cv2.resize(mask, resize_dim)
    # Apply Gaussian blur to the resized mask image
    blurred_mask = cv2.GaussianBlur(resized_mask, kernel_size, 0)
    # Normalize the blurred image
    normalized_mask = cv2.normalize(blurred_mask, None, 0, 255, cv2.NORM_MINMAX)

    return normalized_mask


def get_background_mask(image):
    """
    Read an image, invert its values, and save the result.
    """
    # Invert the image
    inverted_image = 1.0 - (
        image / 255.0
    )  # Assuming the image values are in [0, 255] range
    # Convert back to uint8
    inverted_image = (inverted_image * 255).astype(np.uint8)

    return inverted_image


def get_sep_face_mask(mask1, mask2):
    """
    Read two images, subtract the second one from the first, and save the result.
    """
    # Subtract the second mask from the first
    result_mask = cv2.subtract(mask1, mask2)

    return result_mask
