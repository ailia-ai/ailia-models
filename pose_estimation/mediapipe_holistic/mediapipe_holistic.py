import sys
import time
import math
import json
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
from dtype_utils import numpy_type_to_builtin_type  # noqa: E402
import webcamera_utils
# logger
from logging import getLogger  # noqa

from detection_utils import pose_detection
from drawing_utils import draw_landmarks, draw_face_landmarks, draw_hand_landmarks, plot_landmarks
import face_detection
import hand_detection
from face_detection import face_estimate
from hand_detection import hands_estimate

logger = getLogger(__name__)

# ======================
# Parameters
# ======================

# pose model
MODEL_LIST = ['lite', 'full', 'heavy']
WEIGHT_LITE_PATH = 'pose_landmark_lite.onnx'
MODEL_LITE_PATH = 'pose_landmark_lite.onnx.prototxt'
WEIGHT_FULL_PATH = 'pose_landmark_full.onnx'
MODEL_FULL_PATH = 'pose_landmark_full.onnx.prototxt'
WEIGHT_HEAVY_PATH = 'pose_landmark_heavy.onnx'
MODEL_HEAVY_PATH = 'pose_landmark_heavy.onnx.prototxt'
WEIGHT_DETECTOR_PATH = 'pose_detection.onnx'
MODEL_DETECTOR_PATH = 'pose_detection.onnx.prototxt'
REMOTE_POSE_PATH = 'https://storage.googleapis.com/ailia-models/mediapipe_pose_world_landmarks/'

# face model
WEIGHT_FACE_DETECTOR_PATH = 'face_detection_short_range.onnx'
MODEL_FACE_DETECTOR_PATH = 'face_detection_short_range.onnx.prototxt'
WEIGHT_FACE_LANDMARK_PATH = 'face_landmark_with_attention.onnx'
MODEL_FACE_LANDMARK_PATH = 'face_landmark_with_attention.onnx.prototxt'
# hand model
WEIGHT_HAND_DETECTOR_PATH = 'hand_recrop.onnx'
MODEL_HAND_DETECTOR_PATH = 'hand_recrop.onnx.prototxt'
WEIGHT_HAND_LANDMARK_PATH = 'hand_landmark_full.onnx'
MODEL_HAND_LANDMARK_PATH = 'hand_landmark_full.onnx.prototxt'
REMOTE_PATH = 'https://storage.googleapis.com/ailia-models/mediapipe_holistic/'

# detection model
WEIGHT_YOLOX_PATH = 'yolox_s.opt.onnx'
MODEL_YOLOX_PATH = 'yolox_s.opt.onnx.prototxt'
REMOTE_YOLOX_PATH = 'https://storage.googleapis.com/ailia-models/yolox/'

IMAGE_PATH = 'demo.jpg'
SAVE_IMAGE_PATH = 'output.png'

POSE_DET_SIZE = 224
POSE_LMK_SIZE = 256

DETECTION_THRESHOLD = 0.4
DETECTION_IOU = 0.45

# ======================
# Argument Parser Config
# ======================

parser = get_base_parser(
    'MediaPipe Holistic',
    IMAGE_PATH,
    SAVE_IMAGE_PATH,
)

# Normal options
parser.add_argument(
    '-m', '--model', metavar='ARCH',
    default='full', choices=MODEL_LIST,
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

# Multi persom options
parser.add_argument(
    '--detector',
    action='store_true',
    help='Perform person detection as preprocessing.'
)
parser.add_argument(
    '--detection_width',
    default=640, type=int,
    help='The detection width and height for yolo. (default: auto)'
)

# Pre and Post processing options
parser.add_argument(
    '--crop',
    action='store_true',
    help='Crop detected person as postprocessing.'
)
parser.add_argument(
    '--scale',
    default=None, type=int,
    help='Enlarge the input image for better viewing of the output.'
)
parser.add_argument(
    '--frame_skip',
    default=None, type=int,
    help='Skip the frames of input video.'
)
parser.add_argument(
    '-w', '--write_json',
    action='store_true',
    help='save result to json'
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

        h = w = POSE_DET_SIZE
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
        scale = POSE_DET_SIZE / max(im_h, im_w)
        ow, oh = int(im_w * scale), int(im_h * scale)
        if ow != im_w or oh != im_h:
            img = cv2.resize(img, (ow, oh), interpolation=cv2.INTER_LINEAR)

        pad_h = pad_w = 0
        if ow != POSE_DET_SIZE or oh != POSE_DET_SIZE:
            pad_img = np.zeros((POSE_DET_SIZE, POSE_DET_SIZE, 3))
            pad_h = (POSE_DET_SIZE - oh) // 2
            pad_w = (POSE_DET_SIZE - ow) // 2
            pad_img[pad_h:pad_h + oh, pad_w:pad_w + ow, :] = img
            img = pad_img

        pad_h = pad_h / POSE_DET_SIZE
        pad_w = pad_w / POSE_DET_SIZE

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

    h = w = POSE_LMK_SIZE
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
# Core functions
# ======================

def pose_estimate(models, img):
    h, w = img.shape[:2]

    # Multi person
    if args.detector:
        det_net = models["det_net"]
        det_net.compute(img, DETECTION_THRESHOLD, DETECTION_IOU)
        count = det_net.get_object_count()

        h, w = img.shape[0], img.shape[1]
        count = det_net.get_object_count()
        pose_detections = []
        for idx in range(count):
            obj = det_net.get_object(idx)
            top_left = (int(w*obj.x), int(h*obj.y))
            bottom_right = (int(w*(obj.x+obj.w)), int(h*(obj.y+obj.h)))
            CATEGORY_PERSON = 0
            if obj.category != CATEGORY_PERSON:
                continue
            px1 = max(0, top_left[0])
            px2 = min(bottom_right[0], w)
            py1 = max(0, top_left[1])
            py2 = min(bottom_right[1], h)
            crop_img = img[py1:py2, px1:px2, :]
            pose_landmarks, pose_world_landmarks, left_hand_landmarks, right_hand_landmarks, face_landmarks = pose_estimate_one_person(models, crop_img)
            detect = (pose_landmarks, pose_world_landmarks, left_hand_landmarks,right_hand_landmarks, face_landmarks, px1, py1, px2, py2)
            pose_detections.append(detect)
        return pose_detections
    
    # Single person
    pose_detections = []
    pose_landmarks, pose_world_landmarks, left_hand_landmarks, right_hand_landmarks, face_landmarks = pose_estimate_one_person(models, img)
    detect = (pose_landmarks, pose_world_landmarks, left_hand_landmarks,right_hand_landmarks, face_landmarks, 0, 0, w, h)
    pose_detections.append(detect)
    return pose_detections

def pose_estimate_one_person(models, img):
    im_h, im_w = img.shape[:2]
    img = img[:, :, ::-1]  # BGR -> RGB

    """
    Detects poses.
    """

    input, pad = preprocess_detection(img)

    # feedforward
    det_net = models['pose_det']
    if not args.onnx:
        output = det_net.predict([input])
    else:
        output = det_net.run(None, {'input_1': input})
    detections, scores = output

    box, score = pose_detection(detections, scores, pad)
    if len(box) == 0:
        return [], [], [], [], []

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
    net = models['pose_lmk']
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
    h = w = POSE_LMK_SIZE
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

    face_landmarks = all_landmarks[:11, ...]
    face_landmarks = face_estimate(img, face_landmarks, models)

    left_hand_landmarks = all_landmarks[[15, 17, 19], ...]
    right_hand_landmarks = all_landmarks[[16, 18, 20], ...]
    left_hand_landmarks, right_hand_landmarks = hands_estimate(
        img, left_hand_landmarks, right_hand_landmarks, models)

    PoseLandmark = namedtuple('PoseLandmark', ['x', 'y', 'z', 'visibility', 'presence'])
    Landmark = namedtuple('Landmark', ['x', 'y', 'z'])
    PoseWorldLandmark = namedtuple('PoseWorldLandmark', ['x', 'y', 'z', 'visibility'])

    pose_landmarks = [PoseLandmark(lm[0], lm[1], lm[2], lm[3], lm[4]) for lm in all_landmarks]
    face_landmarks = [Landmark(lm[0], lm[1], lm[2]) for lm in face_landmarks]
    left_hand_landmarks = [Landmark(lm[0], lm[1], lm[2]) for lm in left_hand_landmarks]
    right_hand_landmarks = [Landmark(lm[0], lm[1], lm[2]) for lm in right_hand_landmarks]
    pose_world_landmarks = [
        PoseWorldLandmark(wld[0], wld[1], wld[2], lm[3])
        for lm, wld in zip(all_landmarks, all_world_landmarks)
    ]

    return \
        pose_landmarks, pose_world_landmarks, \
        left_hand_landmarks, right_hand_landmarks, \
        face_landmarks


def save_json(outputs, json_path):
    result = []
    for output in outputs:
        pose_landmarks, pose_world_landmarks, \
        left_hand_landmarks, right_hand_landmarks, \
        face_landmarks, x1, y1, x2, y2 = output

        if len(pose_landmarks) == 0:
            continue

        result.append({
            # with --detector, the landmark x/y are normalized by this bbox
            'bbox': [x1, y1, x2, y2],
            'pose_landmarks': [lm._asdict() for lm in pose_landmarks],
            'pose_world_landmarks': [lm._asdict() for lm in pose_world_landmarks],
            'left_hand_landmarks': [lm._asdict() for lm in left_hand_landmarks],
            'right_hand_landmarks': [lm._asdict() for lm in right_hand_landmarks],
            'face_landmarks': [lm._asdict() for lm in face_landmarks],
        })

    with open(json_path, 'w') as f:
        json.dump(numpy_type_to_builtin_type(result), f, indent=2)


# ======================
# Main functions
# ======================

def recognize_from_image(models):
    # input image loop
    for image_path in args.input:
        # prepare input data
        logger.info(image_path)

        img = load_image(image_path)
        if args.scale:
            img = cv2.resize(img, (img.shape[1] * args.scale, img.shape[0] * args.scale))
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
                outputs = pose_estimate(models, img)
                end = int(round(time.time() * 1000))
                estimation_time = (end - start)

                # Loggin
                logger.info(f'\tailia processing estimation time {estimation_time} ms')
                if i != 0:
                    total_time_estimation = total_time_estimation + estimation_time

            logger.info(f'\taverage time estimation {total_time_estimation / (args.benchmark_count - 1)} ms')
        else:
            # inference
            outputs = pose_estimate(models, img)

        # display result
        for output in outputs:
            pose_landmarks, pose_world_landmarks, \
            left_hand_landmarks, right_hand_landmarks, \
            face_landmarks, x1, y1, x2, y2 = output
            
            if len(pose_landmarks) == 0:
                logger.info('pose not detected.')
                continue

            logger.info(
                f'Nose coordinates: ('
                f'{pose_landmarks[0].x * image_width}, '
                f'{pose_landmarks[0].y * image_height})'
            )

            # plot result
            ref_img = img[y1:y2, x1:x2, :] # reference
            draw_landmarks(ref_img, pose_landmarks)
            draw_face_landmarks(ref_img, face_landmarks)
            draw_hand_landmarks(ref_img, left_hand_landmarks)
            draw_hand_landmarks(ref_img, right_hand_landmarks)

            if args.world_landmark:
                plot_landmarks(pose_world_landmarks)

        # save results
        savepath = get_savepath(args.savepath, image_path)
        logger.info(f'saved at : {savepath}')
        cv2.imwrite(savepath, img)

        if args.write_json:
            save_json(outputs, (savepath.rsplit('.', 1)[0]) + '.json')

    logger.info('Script finished successfully.')


def recognize_from_video(models):
    capture = webcamera_utils.get_capture(args.video)

    frame_shown = False
    frame_cnt = 0
    writer = None

    while True:
        ret, frame = capture.read()
        if (cv2.waitKey(1) & 0xFF == ord('q')) or not ret:
            break
        if frame_shown and cv2.getWindowProperty('frame', cv2.WND_PROP_VISIBLE) == 0:
            break

        # frame resize
        if args.scale:
            frame = cv2.resize(frame, (frame.shape[1] * args.scale, frame.shape[0] * args.scale))

        # frame skip
        if args.frame_skip:
            if frame_cnt % args.frame_skip != 0:
                frame_cnt = frame_cnt + 1
                continue

        # inference
        outputs = pose_estimate(models, frame)

        # crop region
        if frame_cnt == 0:
            crop_x1, crop_x2 = frame.shape[1], 0
            crop_y1, crop_y2 = frame.shape[0], 0

        # display result
        for output in outputs:
            pose_landmarks, pose_world_landmarks, \
            left_hand_landmarks, right_hand_landmarks, \
            face_landmarks, x1, y1, x2, y2 = output

            # calc crop region
            if frame_cnt == 0:
                margin = int(max(x2 - x1, y2 - y1) / 2)
                crop_x1 = min(crop_x1, max(0, x1 - margin))
                crop_x2 = max(crop_x2, min(frame.shape[1], x2 + margin))
                crop_y1 = min(crop_y1, max(0, y1 - margin))
                crop_y2 = max(crop_y2, min(frame.shape[0], y2 + margin))

            # plot result
            if 0 < len(pose_landmarks):
                ref_img = frame[y1:y2, x1:x2, :] # reference
                draw_landmarks(ref_img, pose_landmarks)
                draw_face_landmarks(ref_img, face_landmarks)
                draw_hand_landmarks(ref_img, left_hand_landmarks)
                draw_hand_landmarks(ref_img, right_hand_landmarks)

        # crop
        if args.crop:
            frame = frame[crop_y1:crop_y2, crop_x1:crop_x2, :]

        # display
        cv2.imshow('frame', frame)

        # create video writer if savepath is specified as video format
        if args.savepath != SAVE_IMAGE_PATH:
            if frame_cnt == 0:
                f_h = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
                f_w = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
                if args.scale:
                    f_h = f_h * args.scale
                    f_w = f_w * args.scale
                if args.crop:
                    f_h = crop_y2 - crop_y1
                    f_w = crop_x2 - crop_x1
                writer = webcamera_utils.get_writer(args.savepath, f_h, f_w)
        if writer is not None:
            writer.write(frame)

        # process
        frame_shown = True
        frame_cnt = frame_cnt + 1

    capture.release()
    cv2.destroyAllWindows()
    if writer is not None:
        writer.release()

    logger.info('Script finished successfully.')


def main():
    # model files check and download
    ## pose model
    check_and_download_models(WEIGHT_DETECTOR_PATH, MODEL_DETECTOR_PATH, REMOTE_POSE_PATH)
    info = {
        'lite': (WEIGHT_LITE_PATH, MODEL_LITE_PATH),
        'full': (WEIGHT_FULL_PATH, MODEL_FULL_PATH),
        'heavy': (WEIGHT_HEAVY_PATH, MODEL_HEAVY_PATH),
    }
    WEIGHT_PATH, MODEL_PATH = info[args.model]
    check_and_download_models(WEIGHT_PATH, MODEL_PATH, REMOTE_POSE_PATH)
    ## face model
    check_and_download_models(WEIGHT_FACE_DETECTOR_PATH, MODEL_FACE_DETECTOR_PATH, REMOTE_PATH)
    check_and_download_models(WEIGHT_FACE_LANDMARK_PATH, MODEL_FACE_LANDMARK_PATH, REMOTE_PATH)
    ## hand model
    check_and_download_models(WEIGHT_HAND_DETECTOR_PATH, MODEL_HAND_DETECTOR_PATH, REMOTE_PATH)
    check_and_download_models(WEIGHT_HAND_LANDMARK_PATH, MODEL_HAND_LANDMARK_PATH, REMOTE_PATH)
    # detection model
    if args.detector:
        check_and_download_models(WEIGHT_YOLOX_PATH, MODEL_YOLOX_PATH, REMOTE_YOLOX_PATH)

    env_id = args.env_id

    # initialize
    if not args.onnx:
        pose_det = ailia.Net(MODEL_DETECTOR_PATH, WEIGHT_DETECTOR_PATH, env_id=env_id)
        pose_lmk = ailia.Net(MODEL_PATH, WEIGHT_PATH, env_id=env_id)
        face_det = ailia.Net(MODEL_FACE_DETECTOR_PATH, WEIGHT_FACE_DETECTOR_PATH, env_id=env_id)
        face_lmk = ailia.Net(MODEL_FACE_LANDMARK_PATH, WEIGHT_FACE_LANDMARK_PATH, env_id=env_id)
        hand_det = ailia.Net(MODEL_HAND_DETECTOR_PATH, WEIGHT_HAND_DETECTOR_PATH, env_id=env_id)
        hand_lmk = ailia.Net(MODEL_HAND_LANDMARK_PATH, WEIGHT_HAND_LANDMARK_PATH, env_id=env_id)
    else:
        import onnxruntime
        pose_det = onnxruntime.InferenceSession(WEIGHT_DETECTOR_PATH)
        pose_lmk = onnxruntime.InferenceSession(WEIGHT_PATH)
        face_det = onnxruntime.InferenceSession(WEIGHT_FACE_DETECTOR_PATH)
        face_lmk = onnxruntime.InferenceSession(WEIGHT_FACE_LANDMARK_PATH)
        hand_det = onnxruntime.InferenceSession(WEIGHT_HAND_DETECTOR_PATH)
        hand_lmk = onnxruntime.InferenceSession(WEIGHT_HAND_LANDMARK_PATH)
        face_detection.onnx = True
        hand_detection.onnx = True

    det_net = None
    if args.detector:
        det_net = ailia.Detector(
            MODEL_YOLOX_PATH,
            WEIGHT_YOLOX_PATH,
            80,
            format=ailia.NETWORK_IMAGE_FORMAT_BGR,
            channel=ailia.NETWORK_IMAGE_CHANNEL_FIRST,
            range=ailia.NETWORK_IMAGE_RANGE_U_INT8,
            algorithm=ailia.DETECTOR_ALGORITHM_YOLOX,
            env_id=env_id,
        )
        det_net.set_input_shape(args.detection_width, args.detection_width)

    models = {
        'pose_det': pose_det,
        'pose_lmk': pose_lmk,
        'face_det': face_det,
        'face_lmk': face_lmk,
        'hand_det': hand_det,
        'hand_lmk': hand_lmk,
        'det_net': det_net
    }

    if args.video is not None:
        # video mode
        recognize_from_video(models)
    else:
        # image mode
        recognize_from_image(models)


if __name__ == '__main__':
    main()
