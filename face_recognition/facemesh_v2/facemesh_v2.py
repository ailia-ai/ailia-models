import sys
import time
import math
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
from webcamera_utils import get_capture, get_writer  # noqa

import draw_utils
from detection_utils import face_detection
from detection_utils import IMAGE_SIZE as IMAGE_DET_SIZE
from blendshape import face_blendshapes, plot_face_blendshapes_bar_graph

logger = getLogger(__name__)

# ======================
# Parameters
# ======================

WEIGHT_PATH = 'face_landmarks_detector.onnx'
MODEL_PATH = 'face_landmarks_detector.onnx.prototxt'
WEIGHT_DET_PATH = 'face_detector.onnx'
MODEL_DET_PATH = 'face_detector.onnx.prototxt'
WEIGHT_BLENDSHAPE_PATH = 'face_blendshapes.onnx'
MODEL_BLENDSHAPE_PATH = 'face_blendshapes.onnx.prototxt'
REMOTE_PATH = 'https://storage.googleapis.com/ailia-models/facemesh_v2/'

IMAGE_PATH = 'demo.jpg'
SAVE_IMAGE_PATH = 'output.png'

IMAGE_SIZE = 256
NUM_LANDMARKS = 478

ROI = namedtuple('ROI', ['x_center', 'y_center', 'width', 'height', 'rotation'])

# ======================
# Arguemnt Parser Config
# ======================

parser = get_base_parser(
    'FaceMesh-V2', IMAGE_PATH, SAVE_IMAGE_PATH
)
parser.add_argument(
    '--blendshape', action="store_true",
    help="visualize the face blendshapes categories using a bar graph."
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

def draw_result(img, face_landmarks):
    # Draw the face landmarks.

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


def predict(models, img):
    im_h, im_w, _ = img.shape
    img = img[:, :, ::-1]  # BGR -> RGB

    input, matrix = preprocess_det(img)

    # feedforward
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
        input, roi, pad = preprocess(img, roi)

        # feedforward
        net = models['net']
        if not args.onnx:
            output = net.predict([input])
        else:
            output = net.run(None, {'input_12': input})
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
    # input image loop
    for image_path in args.input:
        logger.info(image_path)

        # prepare input data
        img = load_image(image_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)

        # inference
        logger.info('Start inference...')
        if args.benchmark:
            logger.info('BENCHMARK mode')
            total_time_estimation = 0
            for i in range(args.benchmark_count):
                start = int(round(time.time() * 1000))
                detection_result = predict(models, img)
                end = int(round(time.time() * 1000))
                estimation_time = (end - start)

                # Logging
                logger.info(f'\tailia processing estimation time {estimation_time} ms')
                if i != 0:
                    total_time_estimation = total_time_estimation + estimation_time

            logger.info(f'\taverage time estimation {total_time_estimation / (args.benchmark_count - 1)} ms')
        else:
            detection_result = predict(models, img)

        res_img = img
        for detection in detection_result:
            res_img = draw_result(res_img, detection)

        # plot result
        savepath = get_savepath(args.savepath, image_path, ext='.png')
        logger.info(f'saved at : {savepath}')
        cv2.imwrite(savepath, res_img)

        if args.blendshape and len(detection_result) > 0:
            bls_net = models['blendshape']
            score = face_blendshapes(bls_net, detection_result[0], img.shape[:2], args.onnx)
            img = plot_face_blendshapes_bar_graph(score)

            cv2.imwrite("bar_graph.png", img)

    logger.info('Script finished successfully.')


def recognize_from_video(models):
    capture = get_capture(args.video)

    # create video writer if savepath is specified as video format
    f_h = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    f_w = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))

    if args.savepath != SAVE_IMAGE_PATH:
        writer = get_writer(args.savepath, f_h, f_w + f_h)
    else:
        writer = None

    frame_shown = False
    while True:
        ret, frame = capture.read()
        if (cv2.waitKey(1) & 0xFF == ord('q')) or not ret:
            break
        if frame_shown and cv2.getWindowProperty('frame', cv2.WND_PROP_VISIBLE) == 0:
            break

        detection_result = predict(models, frame)

        visual_img = frame
        for detection in detection_result:
            visual_img = draw_result(visual_img, detection)

        if args.blendshape and len(detection_result) > 0:
            bls_net = models['blendshape']
            score = face_blendshapes(bls_net, detection_result[0], frame.shape[:2], args.onnx)
            bar_img = plot_face_blendshapes_bar_graph(score)
            bar_img = cv2.resize(bar_img, (f_h, f_h))

            packed_img = np.zeros((f_h, f_w + f_h, 3), dtype=np.uint8)
            packed_img[:,0:f_w,:] = visual_img
            packed_img[:,f_w:f_w+f_h,:] = bar_img[:,:,0:3]

            visual_img = packed_img

        cv2.imshow('frame', visual_img)

        frame_shown = True

        # save results
        if writer is not None:
            writer.write(visual_img)

    capture.release()
    if writer is not None:
        writer.release()
    cv2.destroyAllWindows()
    if writer is not None:
        writer.release()

    logger.info('Script finished successfully.')


def main():
    check_and_download_models(WEIGHT_PATH, MODEL_PATH, REMOTE_PATH)
    check_and_download_models(WEIGHT_DET_PATH, MODEL_DET_PATH, REMOTE_PATH)
    if args.blendshape:
        check_and_download_models(WEIGHT_BLENDSHAPE_PATH, MODEL_BLENDSHAPE_PATH, REMOTE_PATH)

    env_id = args.env_id

    # initialize
    bls_net = None
    if not args.onnx:
        net = ailia.Net(MODEL_PATH, WEIGHT_PATH, env_id=env_id)
        det_net = ailia.Net(MODEL_DET_PATH, WEIGHT_DET_PATH, env_id=env_id)
        if args.blendshape:
            bls_net = ailia.Net(MODEL_BLENDSHAPE_PATH, WEIGHT_BLENDSHAPE_PATH, env_id=env_id)
    else:
        import onnxruntime
        cuda = 0 < ailia.get_gpu_environment_id()
        providers = ['CUDAExecutionProvider', 'CPUExecutionProvider'] if cuda else ['CPUExecutionProvider']
        net = onnxruntime.InferenceSession(WEIGHT_PATH, providers=providers)
        det_net = onnxruntime.InferenceSession(WEIGHT_DET_PATH, providers=providers)
        if args.blendshape:
            bls_net = onnxruntime.InferenceSession(WEIGHT_BLENDSHAPE_PATH, providers=providers)

    models = {
        "net": net,
        "det_net": det_net,
        "blendshape": bls_net,
    }

    if args.video is not None:
        recognize_from_video(models)
    else:
        recognize_from_image(models)


if __name__ == '__main__':
    main()
