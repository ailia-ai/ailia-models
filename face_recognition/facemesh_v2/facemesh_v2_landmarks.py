import sys
import math
import json
from collections import namedtuple
from logging import getLogger

import numpy as np
import cv2

import ailia

# import original modules
sys.path.append('../../util')
from arg_utils import get_base_parser, update_parser, get_savepath  # noqa
from model_utils import check_and_download_models  # noqa
from image_utils import normalize_image  # noqa
from detector_utils import load_image  # noqa

import draw_utils
from detection_utils import face_detection
from detection_utils import IMAGE_SIZE as IMAGE_DET_SIZE

logger = getLogger(__name__)

# ======================
# Parameters
# ======================

WEIGHT_PATH = 'face_landmarks_detector.onnx'
MODEL_PATH = 'face_landmarks_detector.onnx.prototxt'
WEIGHT_DET_PATH = 'face_detector.onnx'
MODEL_DET_PATH = 'face_detector.onnx.prototxt'
REMOTE_PATH = 'https://storage.googleapis.com/ailia-models/facemesh_v2/'

IMAGE_PATH = 'demo.jpg'
SAVE_IMAGE_PATH = 'output.png'
SAVE_LANDMARKS_PATH = 'landmarks.json'

IMAGE_SIZE = 256
NUM_LANDMARKS = 478

ROI = namedtuple('ROI', ['x_center', 'y_center', 'width', 'height', 'rotation'])

# ======================
# Argument Parser Config
# ======================

parser = get_base_parser(
    'FaceMesh-V2 Landmarks Exporter', IMAGE_PATH, SAVE_IMAGE_PATH
)
parser.add_argument(
    '--save_landmarks_path', default=SAVE_LANDMARKS_PATH,
    help='Path to save the output landmarks JSON file.'
)
parser.add_argument(
    '--save_image_path', default=None,
    help='If specified, save the result image with landmarks drawn to this path.'
)
parser.add_argument(
    '--onnx',
    action='store_true',
    help='execute onnxruntime version.'
)
args = update_parser(parser)


# ======================
# Secondary Functions
# ======================

def draw_result(img, face_landmarks):
    draw_utils.draw_landmarks(
        image=img,
        landmark_list=face_landmarks,
        connections=draw_utils.FACEMESH_TESSELATION,
        connection_drawing_spec=draw_utils.get_tesselation_style())

    draw_utils.draw_landmarks(
        image=img,
        landmark_list=face_landmarks,
        connections=draw_utils.FACEMESH_CONTOURS,
        connection_drawing_spec=draw_utils.get_contours_style())

    draw_utils.draw_landmarks(
        image=img,
        landmark_list=face_landmarks,
        connections=draw_utils.FACEMESH_IRISES,
        connection_drawing_spec=draw_utils.get_iris_connections_style())

    return img


# ======================
# Main functions
# ======================

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

    roi = ROI(0.5 * im_w, 0.5 * im_h, im_w, im_h, 0)
    dst_width = dst_height = IMAGE_DET_SIZE
    img, matrix, *_ = warp_perspective(
        img, roi,
        dst_width, dst_height)

    img = normalize_image(img, normalize_type='127.5')
    img = np.expand_dims(img, axis=0)
    img = img.astype(np.float32)

    return img, matrix


def preprocess(img, roi):
    dst_width = dst_height = IMAGE_SIZE
    img, _, roi, pad = warp_perspective(
        img, roi,
        dst_width, dst_height,
        keep_aspect_ratio=False)

    img = normalize_image(img, normalize_type='255')
    img = np.expand_dims(img, axis=0)
    img = img.astype(np.float32)

    return img, roi, pad


def post_processing(input_tensors, roi, pad):
    num_landmarks = NUM_LANDMARKS
    num_dimensions = 3

    input_tensors = input_tensors.reshape(-1)
    output_landmarks = np.zeros((num_landmarks, num_dimensions))
    for i in range(num_landmarks):
        offset = i * num_dimensions
        output_landmarks[i] = input_tensors[offset:offset + 3]

    norm_landmarks = output_landmarks / 256

    h_pad, v_pad = pad
    left = h_pad
    top = v_pad
    left_and_right = h_pad * 2
    top_and_bottom = v_pad * 2
    for landmark in norm_landmarks:
        new_x = (landmark[0] - left) / (1 - left_and_right)
        new_y = (landmark[1] - top) / (1 - top_and_bottom)
        new_z = landmark[2] / (1 - left_and_right)
        landmark[:3] = (new_x, new_y, new_z)

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


def predict(models, img):
    im_h, im_w, _ = img.shape
    img = img[:, :, ::-1]  # BGR -> RGB

    input, matrix = preprocess_det(img)

    det_net = models['det_net']
    if not args.onnx:
        output = det_net.predict([input])
    else:
        output = det_net.run(None, {'input': input})
    detections, scores = output

    boxes, scores = face_detection(detections, scores, matrix)
    if len(boxes) == 0:
        return np.zeros((0, NUM_LANDMARKS, 3))

    landmarks_list = []
    for box in boxes:
        rect_width = box[2] - box[0]
        rect_height = box[3] - box[1]
        center_x = (box[0] + box[2]) / 2
        center_y = (box[1] + box[3]) / 2

        x0, y0 = box[4] * im_w, box[5] * im_h
        x1, y1 = box[6] * im_w, box[7] * im_h
        angle = 0 - math.atan2(-(y1 - y0), x1 - x0)
        angle = angle - 2 * math.pi * math.floor((angle - (-math.pi)) / (2 * math.pi));

        scale_x = scale_y = 1.5
        rect_width = rect_width * scale_x
        rect_height = rect_height * scale_y

        roi = ROI(
            center_x * im_w, center_y * im_h,
            rect_width * im_w, rect_height * im_h,
            angle)
        img, roi, pad = preprocess(img, roi)

        net = models['net']
        if not args.onnx:
            output = net.predict([img])
        else:
            output = net.run(None, {'input_12': img})
        landmark_tensors, presence_flag_tensors, _ = output

        norm_rect = ROI(
            roi.x_center / im_w, roi.y_center / im_h,
            roi.width / im_w, roi.height / im_h,
            angle)
        landmarks = post_processing(landmark_tensors, norm_rect, pad)
        landmarks_list.append(landmarks)

    landmarks = np.stack(landmarks_list, axis=0)

    return landmarks


def recognize_from_image(models):
    for image_path in args.input:
        logger.info(image_path)

        img = load_image(image_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)

        logger.info('Start inference...')
        detection_result = predict(models, img)

        # Convert landmarks to JSON-serializable format
        faces = []
        for face_landmarks in detection_result:
            landmarks = [
                {"x": float(lm[0]), "y": float(lm[1]), "z": float(lm[2])}
                for lm in face_landmarks
            ]
            faces.append({"landmarks": landmarks})

        output = {
            "image_path": image_path,
            "num_faces": len(faces),
            "faces": faces,
        }

        save_path = args.save_landmarks_path
        with open(save_path, 'w') as f:
            json.dump(output, f, indent=2)
        logger.info(f'Landmarks saved at : {save_path}')

        if args.save_image_path is not None:
            res_img = img
            for face_landmarks in detection_result:
                res_img = draw_result(res_img, face_landmarks)
            savepath = get_savepath(args.save_image_path, image_path, ext='.png')
            logger.info(f'Image saved at : {savepath}')
            cv2.imwrite(savepath, res_img)

    logger.info('Script finished successfully.')


def main():
    check_and_download_models(WEIGHT_PATH, MODEL_PATH, REMOTE_PATH)
    check_and_download_models(WEIGHT_DET_PATH, MODEL_DET_PATH, REMOTE_PATH)

    env_id = args.env_id

    if not args.onnx:
        net = ailia.Net(MODEL_PATH, WEIGHT_PATH, env_id=env_id)
        det_net = ailia.Net(MODEL_DET_PATH, WEIGHT_DET_PATH, env_id=env_id)
    else:
        import onnxruntime
        cuda = 0 < ailia.get_gpu_environment_id()
        providers = ['CUDAExecutionProvider', 'CPUExecutionProvider'] if cuda else ['CPUExecutionProvider']
        net = onnxruntime.InferenceSession(WEIGHT_PATH, providers=providers)
        det_net = onnxruntime.InferenceSession(WEIGHT_DET_PATH, providers=providers)

    models = {
        "net": net,
        "det_net": det_net,
    }

    recognize_from_image(models)


if __name__ == '__main__':
    main()
