import sys
import time
import math
from collections import namedtuple

import cv2
import numpy as np

import ailia

sys.path.append('../../util')
from arg_utils import get_base_parser, update_parser, get_savepath
from model_utils import check_and_download_models
from image_utils import normalize_image
from detector_utils import load_image
from math_utils import sigmoid
import webcamera_utils
# logger
from logging import getLogger  # noqa

from detection_utils import pose_detection
from drawing_utils import draw_landmarks, plot_landmarks

logger = getLogger(__name__)

# ======================
# Parameters
# ======================

MODEL_LIST = ['lite', 'full', 'heavy']
WEIGHT_LITE_PATH = 'pose_landmark_lite.onnx'
MODEL_LITE_PATH = 'pose_landmark_lite.onnx.prototxt'
WEIGHT_FULL_PATH = 'pose_landmark_full.onnx'
MODEL_FULL_PATH = 'pose_landmark_full.onnx.prototxt'
WEIGHT_HEAVY_PATH = 'pose_landmark_heavy.onnx'
MODEL_HEAVY_PATH = 'pose_landmark_heavy.onnx.prototxt'
WEIGHT_DETECTOR_PATH = 'pose_detection.onnx'
MODEL_DETECTOR_PATH = 'pose_detection.onnx.prototxt'
REMOTE_PATH = 'https://storage.googleapis.com/ailia-models/mediapipe_pose_world_landmarks/'

IMAGE_PATH = 'demo.png'
SAVE_IMAGE_PATH = 'output.png'

IMAGE_DET_SIZE = 224
IMAGE_LMK_SIZE = 256

# ======================
# Argument Parser Config
# ======================

parser = get_base_parser(
    'MediaPipe Pose',
    IMAGE_PATH,
    SAVE_IMAGE_PATH,
)
parser.add_argument(
    '-m', '--model', metavar='ARCH',
    default='heavy', choices=MODEL_LIST,
    help='Set model architecture: ' + ' | '.join(MODEL_LIST)
)
parser.add_argument(
    '--world_landmark', action='store_true',
    help='Plot the POSE_WORLD_LANDMARKS.'
)
parser.add_argument(
    '--onnx',
    action='store_true',
    help='execute onnxruntime version.'
)
args = update_parser(parser)


# ======================
# Secondaty Functions
# ======================

def preprocess_detection(img):
    im_h, im_w, _ = img.shape

    """
    resize & padding
    """
    warp_resize = True

    if warp_resize:
        box_size = max(im_h, im_w)
        rotated_rect = ((im_w // 2, im_h // 2), (box_size, box_size), 0)
        pts1 = cv2.boxPoints(rotated_rect)

        h = w = IMAGE_DET_SIZE
        pts2 = np.float32([[0, h], [0, 0], [w, 0], [w, h]])
        M = cv2.getPerspectiveTransform(pts1, pts2)
        img = cv2.warpPerspective(
            img, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)

        pad_h = pad_w = 0
        dst_aspect_ratio = h / w
        src_aspect_ratio = im_h / im_w
        if dst_aspect_ratio > src_aspect_ratio:
            pad_h = (1 - src_aspect_ratio / dst_aspect_ratio) / 2
        else:
            pad_w = (1 - dst_aspect_ratio / src_aspect_ratio) / 2
    else:
        scale = IMAGE_DET_SIZE / max(im_h, im_w)
        ow, oh = int(im_w * scale), int(im_h * scale)
        if ow != im_w or oh != im_h:
            img = cv2.resize(img, (ow, oh), interpolation=cv2.INTER_LINEAR)

        pad_h = pad_w = 0
        if ow != IMAGE_DET_SIZE or oh != IMAGE_DET_SIZE:
            pad_img = np.zeros((IMAGE_DET_SIZE, IMAGE_DET_SIZE, 3))
            pad_h = (IMAGE_DET_SIZE - oh) // 2
            pad_w = (IMAGE_DET_SIZE - ow) // 2
            pad_img[pad_h:pad_h + oh, pad_w:pad_w + ow, :] = img
            img = pad_img

        pad_h = pad_h / IMAGE_DET_SIZE
        pad_w = pad_w / IMAGE_DET_SIZE

    """
    normalize & reshape
    """

    img = normalize_image(img, normalize_type='127.5')

    img = img.transpose(2, 0, 1)  # HWC -> CHW
    img = np.expand_dims(img, axis=0)
    img = img.astype(np.float32)

    return img, (pad_h, pad_w)


def preprocess_landmark(img, center, box_size, rotation):
    im_h, im_w, _ = img.shape

    rotated_rect = (center, (box_size, box_size), rotation * 180. / np.pi)
    pts1 = cv2.boxPoints(rotated_rect)

    h = w = IMAGE_LMK_SIZE
    pts2 = np.float32([[0, h], [0, 0], [w, 0], [w, h]])
    M = cv2.getPerspectiveTransform(pts1, pts2)
    transformed = cv2.warpPerspective(
        img, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)

    transformed = transformed / 255
    transformed = transformed.transpose(2, 0, 1)  # HWC -> CHW
    transformed = np.expand_dims(transformed, axis=0)
    input_tensors = transformed.astype(np.float32)

    return input_tensors


def to_landmark(landmarks):
    num_landmarks = 39

    num = len(landmarks)
    num_dimensions = landmarks.shape[1] // num_landmarks
    output_landmarks = np.zeros((num, num_landmarks, num_dimensions))
    for i in range(num):
        xx = landmarks[i]
        for j in range(num_landmarks):
            offset = j * num_dimensions
            x = xx[offset]
            y = xx[offset + 1]
            z = xx[offset + 2]
            if 3 < num_dimensions:
                visibility = xx[offset + 3]
                presence = xx[offset + 4]
                output_landmarks[i, j] = (x, y, z, sigmoid(visibility), sigmoid(presence))
            else:
                output_landmarks[i, j] = (x, y, z)

    return output_landmarks


def refine_landmark_from_heatmap(landmarks, heatmap):
    """
    For each landmark we replace original value with a value calculated from the
    area in heatmap close to original landmark position (in particular are
    covered with kernel of size kernel_size). To calculate new coordinate
    from heatmap we calculate an weighted average inside the kernel. We update
    the landmark iff heatmap is confident in it's prediction i.e. max(heatmap) in
    kernel is at least min_confidence_to_refine big.
    """
    min_confidence_to_refine = 0.5
    kernel_size = 9
    offset = (kernel_size - 1) // 2

    hm_height, hm_width, hm_channels = heatmap.shape

    for i, lm in enumerate(landmarks):
        center_col = int(lm[0] * hm_width)
        center_row = int(lm[1] * hm_height)
        if center_col < 0 or center_col >= hm_width or center_row < 0 or center_col >= hm_height:
            continue

        begin_col = max(0, center_col - offset)
        end_col = min(hm_width, center_col + offset + 1)
        begin_row = max(0, center_row - offset)
        end_row = min(hm_height, center_row + offset + 1)

        val_sum = 0
        weighted_col = 0
        weighted_row = 0
        max_confidence_value = 0
        for row in range(begin_row, end_row):
            for col in range(begin_col, end_col):
                confidence = sigmoid(heatmap[row, col, i])
                val_sum += confidence
                max_confidence_value = max(max_confidence_value, confidence)
                weighted_col += col * confidence
                weighted_row += row * confidence

        if max_confidence_value >= min_confidence_to_refine and val_sum > 0:
            lm[0] = (weighted_col / hm_width / val_sum)
            lm[1] = (weighted_row / hm_height / val_sum)

    return landmarks


# ======================
# Main functions
# ======================

def pose_estimate(net, det_net, img):
    im_h, im_w = img.shape[:2]
    img = img[:, :, ::-1]  # BGR -> RGB

    """
    Detects poses.
    """

    input, pad = preprocess_detection(img)

    # feedforward
    if not args.onnx:
        output = det_net.predict([input])
    else:
        output = det_net.run(None, {'input_1': input})
    detections, scores = output

    box, score = pose_detection(detections, scores, pad)
    if len(box) == 0:
        return [], []

    # Calculates region of interest based on pose detection, so that can be used
    # to detect landmarks.
    x_center, y_center = box[4:6]
    x_scale, y_scale = box[6:8]
    x_center, y_center = x_center * im_w, y_center * im_h
    x_scale, y_scale = x_scale * im_w, y_scale * im_h
    center = (x_center, y_center)

    box_size = (((x_scale - x_center) ** 2 + (y_scale - y_center) ** 2) ** 0.5) * 2
    box_size *= 1.25

    angle = (np.pi * 90 / 180) - math.atan2(-(y_scale - y_center), x_scale - x_center)
    rotation = angle - 2 * np.pi * np.floor((angle - (-np.pi)) / (2 * np.pi))

    """
    Detects pose landmarks within specified region of interest of the image.
    """

    input_tensors = preprocess_landmark(img, center, box_size, rotation)

    # feedforward
    if not args.onnx:
        output = net.predict([input_tensors])
    else:
        output = net.run(None, {'input_1': input_tensors})

    landmark_tensor, pose_flag_tensor, segmentation_tensor, \
    heatmap_tensor, world_landmark_tensor = output

    # Decodes the tensors into the corresponding landmark
    raw_landmarks = to_landmark(landmark_tensor)
    all_world_landmarks = to_landmark(world_landmark_tensor)

    # Output normalized landmarks
    h = w = IMAGE_LMK_SIZE
    raw_landmarks[:, :, 0] = raw_landmarks[:, :, 0] / w
    raw_landmarks[:, :, 1] = raw_landmarks[:, :, 1] / h
    raw_landmarks[:, :, 2] = raw_landmarks[:, :, 2] / w

    # Refines landmarks with the heatmap tensor.
    all_landmarks = refine_landmark_from_heatmap(raw_landmarks[0], heatmap_tensor[0])

    all_world_landmarks = all_world_landmarks[0]

    # the actual pose landmarks
    all_landmarks = all_landmarks[:33, ...]
    all_world_landmarks = all_world_landmarks[:33, ...]

    # Projects the landmarks in the local coordinates of the
    # (potentially letterboxed) ROI back to the global coordinates
    cosa = math.cos(rotation)
    sina = math.sin(rotation)
    for landmark in all_landmarks:
        x = landmark[0] - 0.5
        y = landmark[1] - 0.5
        landmark[0] = ((cosa * x - sina * y) * box_size + x_center) / im_w
        landmark[1] = ((sina * x + cosa * y) * box_size + y_center) / im_h
        landmark[2] = landmark[2] * box_size / im_w

    # Projects the world landmarks from the letterboxed ROI to the full image.
    for landmark in all_world_landmarks:
        x = landmark[0]
        y = landmark[1]
        landmark[0] = cosa * x - sina * y
        landmark[1] = sina * x + cosa * y

    PoseLandmark = namedtuple('PoseLandmark', ['x', 'y', 'z', 'visibility', 'presence'])
    out_landmarks = [PoseLandmark(lm[0], lm[1], lm[2], lm[3], lm[4]) for lm in all_landmarks]
    PoseWorldLandmark = namedtuple('PoseWorldLandmark', ['x', 'y', 'z', 'visibility'])
    out_world_landmarks = [
        PoseWorldLandmark(wld[0], wld[1], wld[2], lm[3])
        for lm, wld in zip(all_landmarks, all_world_landmarks)
    ]

    return out_landmarks, out_world_landmarks


def recognize_from_image(net, det_net):
    # input image loop
    for image_path in args.input:
        # prepare input data
        logger.info(image_path)

        img = load_image(image_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
        image_height, image_width, _ = img.shape

        # inference
        logger.info('Start inference...')
        if args.benchmark:
            logger.info('BENCHMARK mode')
            total_time_estimation = 0
            for i in range(args.benchmark_count):
                # Pose estimation
                start = int(round(time.time() * 1000))
                output = pose_estimate(net, det_net, img)
                end = int(round(time.time() * 1000))
                estimation_time = (end - start)

                # Loggin
                logger.info(f'\tailia processing estimation time {estimation_time} ms')
                if i != 0:
                    total_time_estimation = total_time_estimation + estimation_time

            logger.info(f'\taverage time estimation {total_time_estimation / (args.benchmark_count - 1)} ms')
        else:
            # inference
            output = pose_estimate(net, det_net, img)

        landmarks, world_landmarks = output

        if len(landmarks) == 0:
            logger.info('pose not detected.')
            continue

        logger.info(
            f'Nose coordinates: ('
            f'{landmarks[0].x * image_width}, '
            f'{landmarks[0].y * image_height})'
        )

        # plot result
        draw_landmarks(img, landmarks)

        if args.world_landmark:
            plot_landmarks(world_landmarks)

        # save results
        savepath = get_savepath(args.savepath, image_path)
        logger.info(f'saved at : {savepath}')
        cv2.imwrite(savepath, img)

    logger.info('Script finished successfully.')


def recognize_from_video(net, det_net):
    capture = webcamera_utils.get_capture(args.video)

    # create video writer if savepath is specified as video format
    if args.savepath != SAVE_IMAGE_PATH:
        f_h = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        f_w = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        writer = webcamera_utils.get_writer(args.savepath, f_h, f_w)
    else:
        writer = None

    frame_shown = False
    while True:
        ret, frame = capture.read()
        if (cv2.waitKey(1) & 0xFF == ord('q')) or not ret:
            break
        if frame_shown and cv2.getWindowProperty('frame', cv2.WND_PROP_VISIBLE) == 0:
            break

        # inference
        output = pose_estimate(net, det_net, frame)
        landmarks, world_landmarks = output

        # plot result
        if 0 < len(landmarks):
            draw_landmarks(frame, landmarks)

        cv2.imshow('frame', frame)
        frame_shown = True

        # save results
        if writer is not None:
            writer.write(frame)

    capture.release()
    cv2.destroyAllWindows()
    if writer is not None:
        writer.release()

    logger.info('Script finished successfully.')


def main():
    # model files check and download
    check_and_download_models(WEIGHT_DETECTOR_PATH, MODEL_DETECTOR_PATH, REMOTE_PATH)
    info = {
        'lite': (WEIGHT_LITE_PATH, MODEL_LITE_PATH),
        'full': (WEIGHT_FULL_PATH, MODEL_FULL_PATH),
        'heavy': (WEIGHT_HEAVY_PATH, MODEL_HEAVY_PATH),
    }
    weight_path, model_path = info[args.model]
    check_and_download_models(weight_path, model_path, REMOTE_PATH)

    env_id = args.env_id

    # initialize
    if not args.onnx:
        det_net = ailia.Net(MODEL_DETECTOR_PATH, WEIGHT_DETECTOR_PATH, env_id=env_id, memory_mode=args.memory_mode)
        net = ailia.Net(model_path, weight_path, env_id=env_id, memory_mode=args.memory_mode)
    else:
        import onnxruntime
        det_net = onnxruntime.InferenceSession(WEIGHT_DETECTOR_PATH)
        net = onnxruntime.InferenceSession(weight_path)

    if args.video is not None:
        # video mode
        recognize_from_video(net, det_net)
    else:
        # image mode
        recognize_from_image(net, det_net)


if __name__ == '__main__':
    main()
