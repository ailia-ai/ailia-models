import platform
import sys
import time

import cv2
import numpy as np
np.set_printoptions(suppress=True, precision=2)
from scipy.optimize import linear_sum_assignment

import ailia

# import original modules
sys.path.append('../../util')
# logger
from logging import getLogger  # noqa: E402

import webcamera_utils  # noqa: E402
from classifier_utils import plot_results, print_results  # noqa: E402
from image_utils import imread, load_image  # noqa: E402
from model_utils import check_and_download_models  # noqa: E402
from arg_utils import get_base_parser, update_parser  # noqa: E402
from tracker.byte_tracker import BYTETracker

logger = getLogger(__name__)


# ======================
# PARAMETERS 1
# ======================
IMAGE_OR_VIDEO_PATH = 'input.jpg'  # input.mp4
SAVE_IMAGE_OR_VIDEO_PATH = 'output.png'  # output.mp4
IMAGE_HEIGHT_YOLO = 640
IMAGE_WIDTH_YOLO = 640
IMAGE_HEIGHT_MIVOLO = 224
IMAGE_WIDTH_MIVOLO = 224


# ======================
# Argument Parser Config
# ======================
parser = get_base_parser(
    'MiVOLO: Multi-input Transformer for Age and Gender Estimation',
    IMAGE_OR_VIDEO_PATH,
    SAVE_IMAGE_OR_VIDEO_PATH,
    fp16_support=False
)
parser.add_argument(
    '-ng', '--no_gender', action='store_true',
    help="Options to prevent gender prediction."
)
parser.add_argument(
    '-md', '--model_detection', metavar='MODEL',
    default='yolov8x', choices=['yolov8x', 'yoloxs', 'yoloxx'],
    help='The input detection model path.' +
         'you can choose yolov8-x, yolox-s or yolox-x'
)
args = update_parser(parser)


# ==========================
# MODEL AND OTHER PARAMETERS
# ==========================
MODEL_YOLO_PATH = args.model_detection + '_person_face.onnx.prototxt'
WEIGHT_YOLO_PATH = args.model_detection + '_person_face.onnx'
MODEL_MIVOLO_PATH = 'mivolo.onnx.prototxt'
WEIGHT_MIVOLO_PATH = 'mivolo.onnx'
REMOTE_PATH = 'https://storage.googleapis.com/ailia-models/mivolo/'

SLEEP_TIME = 0  # for web cam mode
# for yolo and non-maximum-supression
if WEIGHT_YOLO_PATH[:5] == 'yolox':
    THRESH_YOLO = 0.7
else:
    THRESH_YOLO = 0.4
THRESH_IOU = 0.6
# for gender and age estimation
IMAGENET_DEFAULT_MEAN = (0.485, 0.456, 0.406)
IMAGENET_DEFAULT_STD = (0.229, 0.224, 0.225)
MIN_AGE = 1
MAX_AGE = 95
AVG_AGE = 48.0
# for tracking
TRACK_THRESH=0.2
TRACK_BUFFER=30
MATCH_THRESH=0.8
FRAME_RATE=30
MOT20=False
MIN_BOX_AREA = 10
# for visualization
COLOR_PALETTE = [[ 31, 119, 180], [255, 127,  14],
                 [ 44, 160,  44], [214,  39,  40],
                 [148, 103, 189], [140,  86,  75],
                 [227, 119, 194], [127, 127, 127],
                 [188, 189,  34], [ 23, 190, 207]]


# ======================
# Sub functions
# ======================
def prep_input(image, width, height, mean=None, std=None):
    input_data = image.copy()
    input_data = cv2.resize(input_data, (width, height))
    input_data = input_data.astype(np.float32)
    input_data = input_data / 255.0
    if (mean is not None):
        if (type(mean) is tuple) | (type(mean) is list):
            mean = np.array(mean)
    if (std is not None):
        if (type(std) is tuple) | (type(std) is list):
            std = np.array(std)
    if (mean is not None) & (std is not None):
        input_data = (input_data - mean[None, None]) / std[None, None]
    input_data = np.transpose(input_data, (2, 0, 1))[None]
    return input_data


def prep_input_yolox(image, width, height, mean=None, std=None):
    padded_img = np.ones((height, width, 3), dtype=np.uint8) * 114
    r = min(height / image.shape[0], width / image.shape[1])
    resized_img = cv2.resize(image.copy()[..., ::-1],
                             (int(image.shape[1] * r), int(image.shape[0] * r)),
                             interpolation=cv2.INTER_LINEAR)
    padded_img[:int(resized_img.shape[0]), :int(resized_img.shape[1])] = resized_img
    input_data = padded_img.astype(np.float32)
    if (mean is not None):
        if (type(mean) is tuple) | (type(mean) is list):
            mean = np.array(mean)
    if (std is not None):
        if (type(std) is tuple) | (type(std) is list):
            std = np.array(std)
    if (mean is not None) & (std is not None):
        input_data = input_data / 255.0
        input_data = (input_data - mean[None, None]) / std[None, None]
    input_data = np.transpose(input_data, (2, 0, 1))[None]
    return input_data


def calc_iou_1xN(bbox_a, bbox_b, area_a, area_b):
    x_min = np.maximum(bbox_a[0], bbox_b[:, 0])  # xmin
    y_min = np.maximum(bbox_a[1], bbox_b[:, 1])  # ymin
    x_max = np.minimum(bbox_a[2], bbox_b[:, 2])  # xmax
    y_max = np.minimum(bbox_a[3], bbox_b[:, 3])  # ymax
    w = np.maximum(0, (x_max - x_min + 1))
    h = np.maximum(0, (y_max - y_min + 1))
    overlap = w * h
    iou = overlap / (area_a + area_b - overlap)
    return iou


def nms(bbox, thresh_iou=0.5):  # bbox:xyxys
    area = (bbox[:, 2] - bbox[:, 0]) * (bbox[:, 3] - bbox[:, 1])
    idx_sort = np.argsort(bbox[:, 4])
    i_watch = -1
    while (len(idx_sort) >= (1 - i_watch)):
        i_max_score = idx_sort[i_watch]
        idx_else = idx_sort[:i_watch]
        iou = calc_iou_1xN(bbox[i_max_score], bbox[idx_else], area[i_max_score], area[idx_else])
        idx_del = np.where(iou >= thresh_iou)
        idx_sort = np.delete(idx_sort, idx_del)
        i_watch -= 1
    bbox = bbox[idx_sort]
    return bbox


def xywh2xyxy(bbox, tl=False):  # bbox:xywh
    if not tl:
        bbox[:, 0] = bbox[:, 0] - (bbox[:, 2] / 2)
        bbox[:, 1] = bbox[:, 1] - (bbox[:, 3] / 2)
    bbox[:, 2] = bbox[:, 2] + bbox[:, 0]
    bbox[:, 3] = bbox[:, 3] + bbox[:, 1]
    return bbox


def calc_iou_MxN(box1, box2, over_second=False):

    def box_area(box):
        # box = 4xn
        return (box[2] - box[0]) * (box[3] - box[1])

    area1 = box_area(box1.T)
    area2 = box_area(box2.T)

    # overlap(N,M) = (rb(N,M,2) - lt(N,M,2)).clamp(0).prod(2)
    overlap = (np.minimum(box1[:, None, 2:4], box2[:, 2:4]) - np.maximum(box1[:, None, :2], box2[:, :2]))
    overlap[overlap < 0] = 0
    overlap = np.prod(overlap, axis=-1)

    iou = overlap / (area1[:, None] + area2 - overlap)  # iou = overlap / (area1 + area2 - overlap)
    if over_second:
        return ((overlap / area2) + iou) / 2  # mean(overlap / area2, iou)
    else:
        return iou


def assign_face(bbox_person, bbox_face, thresh_iou=0.0001):
    idx_person_assigned_face = [None for _ in range(len(bbox_face))]
    idx_person_unassigned_face = [i_person for i_person in range(len(bbox_person))]

    if len(bbox_person) == 0 or len(bbox_face) == 0:
        return idx_person_assigned_face, idx_person_unassigned_face

    cost_matrix = calc_iou_MxN(bbox_person, bbox_face, over_second=True)
    idx_person, idx_face = [], []

    if len(cost_matrix) > 0:
        idx_person, idx_face = linear_sum_assignment(cost_matrix, maximize=True)

    idx_person_matched_face = set()
    for i_person, i_face in zip(idx_person, idx_face):
        iou = cost_matrix[i_person][i_face]
        if iou > thresh_iou:
            if i_person in idx_person_matched_face:
                # Person can not be assigned twice, in reality this should not happen
                continue
            idx_person_assigned_face[i_face] = i_person
            idx_person_matched_face.add(i_person)

    idx_person_unassigned_face = [i_person for i_person in range(len(bbox_person))
                                  if i_person not in idx_person_matched_face]

    return idx_person_assigned_face, idx_person_unassigned_face


def organize_info(image, bbox_person, bbox_face, idx_person_assigned_face, idx_person_unassigned_face):
    image_face_list = []
    image_person_list = []
    bbox_face_list = []
    bbox_person_list = []

    for i_face in range(len(bbox_face)):
        image_full = image.copy()
        image_face = image_full[bbox_face[i_face, 1]:bbox_face[i_face, 3],
                                bbox_face[i_face, 0]:bbox_face[i_face, 2]].copy()
        for j_face in range(len(bbox_face)):
            image_full[bbox_face[j_face, 1]:bbox_face[j_face, 3],
                       bbox_face[j_face, 0]:bbox_face[j_face, 2]] = 0

        i_person = idx_person_assigned_face[i_face]
        for j_person in range(len(bbox_person)):
            if i_person != j_person:
                image_full[bbox_person[j_person, 1]:bbox_person[j_person, 3],
                           bbox_person[j_person, 0]:bbox_person[j_person, 2]] = 0

        if i_person is None:
            image_person = np.zeros([IMAGE_HEIGHT_MIVOLO, IMAGE_WIDTH_MIVOLO, 3], dtype=np.uint8)
        else:
            image_person = image_full[bbox_person[i_person, 1]:bbox_person[i_person, 3],
                                      bbox_person[i_person, 0]:bbox_person[i_person, 2]].copy()

        shape_max = np.max(np.array(image_face.shape))
        shape_resize = (int(round(image_face.shape[1] / shape_max * IMAGE_WIDTH_MIVOLO)),
                        int(round(image_face.shape[0] / shape_max * IMAGE_HEIGHT_MIVOLO)))
        image_face = cv2.resize(image_face, shape_resize)
        h_pad = IMAGE_HEIGHT_MIVOLO - image_face.shape[0]
        if h_pad > 0:
            h_pad_top = int(round(h_pad / 2))
            h_pad_bottom = h_pad - h_pad_top
            image_face = np.pad(image_face, [(h_pad_top, h_pad_bottom), (0, 0), (0, 0)])
        w_pad = IMAGE_WIDTH_MIVOLO - image_face.shape[1]
        if w_pad > 0:
            w_pad_left = int(round(w_pad / 2))
            w_pad_right = w_pad - w_pad_left
            image_face = np.pad(image_face, [(0, 0), (w_pad_left, w_pad_right), (0, 0)])

        shape_max = np.max(np.array(image_person.shape))
        shape_resize = (int(round(image_person.shape[1] / shape_max * IMAGE_WIDTH_MIVOLO)),
                        int(round(image_person.shape[0] / shape_max * IMAGE_HEIGHT_MIVOLO)))
        image_person = cv2.resize(image_person, shape_resize)
        h_pad = IMAGE_HEIGHT_MIVOLO - image_person.shape[0]
        if h_pad > 0:
            h_pad_top = int(round(h_pad / 2))
            h_pad_bottom = h_pad - h_pad_top
            image_person = np.pad(image_person, [(h_pad_top, h_pad_bottom), (0, 0), (0, 0)])
        w_pad = IMAGE_WIDTH_MIVOLO - image_person.shape[1]
        if w_pad > 0:
            w_pad_left = int(round(w_pad / 2))
            w_pad_right = w_pad - w_pad_left
            image_person = np.pad(image_person, [(0, 0), (w_pad_left, w_pad_right), (0, 0)])

        image_face_list.append(image_face)
        image_person_list.append(image_person)
        bbox_face_list.append(bbox_face[i_face])
        if i_person is None:
            bbox_person_list.append(None)
        else:
            bbox_person_list.append(bbox_person[i_person])

    for i_person in idx_person_unassigned_face:
        image_full = image.copy()
        image_face = np.zeros([IMAGE_HEIGHT_MIVOLO, IMAGE_WIDTH_MIVOLO, 3], dtype=np.uint8)

        for i_face in range(len(bbox_face)):
            image_full[bbox_face[i_face, 1]:bbox_face[i_face, 3],
                       bbox_face[i_face, 0]:bbox_face[i_face, 2]] = 0

        for j_person in range(len(bbox_person)):
            if i_person != j_person:
                image_full[bbox_person[j_person, 1]:bbox_person[j_person, 3],
                           bbox_person[j_person, 0]:bbox_person[j_person, 2]] = 0

        image_person = image_full[bbox_person[i_person, 1]:bbox_person[i_person, 3],
                                  bbox_person[i_person, 0]:bbox_person[i_person, 2]].copy()

        shape_max = np.max(np.array(image_person.shape))
        shape_resize = (int(round(image_person.shape[1] / shape_max * IMAGE_WIDTH_MIVOLO)),
                        int(round(image_person.shape[0] / shape_max * IMAGE_HEIGHT_MIVOLO)))
        image_person = cv2.resize(image_person, shape_resize)
        h_pad = IMAGE_HEIGHT_MIVOLO - image_person.shape[0]
        if h_pad > 0:
            h_pad_top = int(round(h_pad / 2))
            h_pad_bottom = h_pad - h_pad_top
            image_person = np.pad(image_person, [(h_pad_top, h_pad_bottom), (0, 0), (0, 0)])
        w_pad = IMAGE_WIDTH_MIVOLO - image_person.shape[1]
        if w_pad > 0:
            w_pad_left = int(round(w_pad / 2))
            w_pad_right = w_pad - w_pad_left
            image_person = np.pad(image_person, [(0, 0), (w_pad_left, w_pad_right), (0, 0)])

        image_face_list.append(image_face)
        image_person_list.append(image_person)
        bbox_face_list.append(None)
        bbox_person_list.append(bbox_person[i_person])

    return image_face_list, image_person_list, bbox_face_list, bbox_person_list


def track(bbox, tracker):
    online_targets = tracker.update(bbox)
    online_tlwhs = []
    online_ids = []
    online_scores = []
    for t in online_targets:
        tlwh = t.tlwh
        tid = t.track_id
        vertical = tlwh[2] / tlwh[3] > 1.6
        if tlwh[2] * tlwh[3] > MIN_BOX_AREA and not vertical:
            online_tlwhs.append(tlwh)
            online_ids.append(tid)
            online_scores.append(t.score)
    if len(online_tlwhs) > 0:
        online_tlwhs = np.vstack(online_tlwhs)
        online_tlwhs[:, :4] = xywh2xyxy(online_tlwhs[:, :4], tl=True)
        return np.hstack([online_tlwhs,
                          np.array(online_scores)[:, None],
                          np.array(online_ids)[:, None]])
    else:
        return np.zeros([0, 6])


def bbox_label(image, bbox, label='', color=(128, 128, 128), txt_color=(255, 255, 255)):
    lw = max(round(sum(image.shape) / 2 * 0.003), 2)  # line width
    tf = max(lw - 1, 1)  # font thickness
    sf = lw / 3  # font scale
    p1, p2 = (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3]))
    cv2.rectangle(image, p1, p2, color, thickness=lw, lineType=cv2.LINE_AA)
    w, h = cv2.getTextSize(label, 0, fontScale=sf, thickness=tf)[0]  # text width, height
    outside = (p1[1] - h >= 3)
    p2 = (p1[0] + w), (p1[1] - h - 3 if outside else p1[1] + h + 3)
    cv2.rectangle(image, p1, p2, color, -1, cv2.LINE_AA)  # filled
    cv2.putText(image, label, (p1[0], (p1[1] - 2 if outside else p1[1] + h + 2)),
                0, sf, txt_color, thickness=tf, lineType=cv2.LINE_AA)
    return image


# ======================
# Main functions
# ======================
def recognize_from_image(net_YOLO, net_mivolo):
    # input image loop
    for image_path in args.input:
        # prepare input data for yolo
        logger.info(image_path)
        image = imread(image_path)[:, :, ::-1].copy()
        if WEIGHT_YOLO_PATH[:5] == 'yolox':
            input_data = prep_input_yolox(image, width=IMAGE_WIDTH_YOLO, height=IMAGE_HEIGHT_YOLO)
        else:
            input_data = prep_input(image, width=IMAGE_WIDTH_YOLO, height=IMAGE_HEIGHT_YOLO)
        print('np.max(input_data) =', np.max(input_data))

        # inference
        if args.benchmark:
            logger.info('BENCHMARK mode')
            for i in range(args.benchmark_count):
                start = int(round(time.time() * 1000))
                output_YOLO = net_YOLO.run(input_data)
                end = int(round(time.time() * 1000))
                logger.info(f'\tailia processing time {end - start} ms')
        else:
            output_YOLO = net_YOLO.run(input_data)
        output_YOLO = output_YOLO[0][0]
        if WEIGHT_YOLO_PATH[:5] == 'yolox':
            class_conf = np.max(output_YOLO[:, 5:], axis=1)
            conf_mask = (output_YOLO[:, 4] * class_conf)
            class_pred = np.argmax(output_YOLO[:, 5:], axis=1)
            output_YOLO = output_YOLO[:, :6]
            output_YOLO[:, 4:] = 0
            output_YOLO[(class_pred == 0), 5] = conf_mask[(class_pred == 0)]
            output_YOLO[(class_pred == 1), 4] = conf_mask[(class_pred == 1)]
        else:
            output_YOLO = output_YOLO.T
        print('output_YOLO.shape =\n', output_YOLO.shape)
        print('output_YOLO =\n', output_YOLO[np.argsort(-output_YOLO[:, 4])])

        # apply threshold
        bbox_face = output_YOLO[output_YOLO[:, 5] > THRESH_YOLO][:, [0, 1, 2, 3, 5]]
        bbox_person = output_YOLO[output_YOLO[:, 4] > THRESH_YOLO][:, :5]
        print('bbox_face.shape =\n', bbox_face.shape)
        print('bbox_person.shape =\n', bbox_person.shape)

        # xywh -> xyxy
        bbox_face[:, :4] = xywh2xyxy(bbox_face[:, :4])
        bbox_person[:, :4] = xywh2xyxy(bbox_person[:, :4])

        # apply non-maximum-suppression
        bbox_face = nms(bbox=bbox_face, thresh_iou=THRESH_IOU)
        bbox_person = nms(bbox=bbox_person, thresh_iou=THRESH_IOU)

        # rescale
        if WEIGHT_YOLO_PATH[:5] == 'yolox':
            r = min(IMAGE_HEIGHT_YOLO / image.shape[0], IMAGE_WIDTH_YOLO / image.shape[1])
            bbox_face[:, :4] = bbox_face[:, :4] / r
            bbox_person[:, :4] = bbox_person[:, :4] / r
        else:
            bbox_face[:, [0, 2]] = bbox_face[:, [0, 2]] * (image.shape[1] / IMAGE_WIDTH_YOLO)
            bbox_face[:, [1, 3]] = bbox_face[:, [1, 3]] * (image.shape[0] / IMAGE_HEIGHT_YOLO)
            bbox_person[:, [0, 2]] = bbox_person[:, [0, 2]] * (image.shape[1] / IMAGE_WIDTH_YOLO)
            bbox_person[:, [1, 3]] = bbox_person[:, [1, 3]] * (image.shape[0] / IMAGE_HEIGHT_YOLO)

        # cast float to int
        bbox_face = bbox_face[:, :4]
        bbox_person = bbox_person[:, :4]
        bbox_face = np.round(bbox_face).astype(np.int32)
        bbox_person = np.round(bbox_person).astype(np.int32)

        # clip
        bbox_face[:, [0, 2]] = np.clip(bbox_face[:, [0, 2]], 0, image.shape[1])
        bbox_face[:, [1, 3]] = np.clip(bbox_face[:, [1, 3]], 0, image.shape[0])
        bbox_person[:, [0, 2]] = np.clip(bbox_person[:, [0, 2]], 0, image.shape[1])
        bbox_person[:, [1, 3]] = np.clip(bbox_person[:, [1, 3]], 0, image.shape[0])

        # assigne bbox of person to bbox of face
        idx_person_assigned_face, idx_person_unassigned_face = assign_face(bbox_person, bbox_face)

        # organize
        image_face_list, image_person_list, \
        bbox_face_list, bbox_person_list = organize_info(image, bbox_person, bbox_face, 
                                                         idx_person_assigned_face, idx_person_unassigned_face)

        for i_person in range(len(image_person_list)):
            image_face = image_face_list[i_person]
            image_person = image_person_list[i_person]

            # prepare input data for mivolo
            input_data_face = prep_input(image_face, width=IMAGE_WIDTH_MIVOLO, height=IMAGE_HEIGHT_MIVOLO,
                                         mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD)
            input_data_person = prep_input(image_person, width=IMAGE_WIDTH_MIVOLO, height=IMAGE_HEIGHT_MIVOLO,
                                           mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD)
            input_data = np.concatenate([input_data_face, input_data_person], axis=1)

            # inference
            if args.benchmark:
                logger.info('BENCHMARK mode')
                for i in range(args.benchmark_count):
                    start = int(round(time.time() * 1000))
                    output_mivolo = net_mivolo.run(input_data)
                    end = int(round(time.time() * 1000))
                    logger.info(f'\tailia processing time {end - start} ms')
            else:
                output_mivolo = net_mivolo.run(input_data)
            output_mivolo = output_mivolo[0][0]

            label = ''
            if args.no_gender:
                y_hat_gender = None
            else:
                # get_gender
                y_hat_gender = output_mivolo[:2]
                y_hat_gender = np.exp(y_hat_gender)
                y_hat_gender = y_hat_gender / np.sum(y_hat_gender)
                label = label + ('M' if (y_hat_gender[0] > 0.5) else 'F')

            # get_age
            age = output_mivolo[2]
            age = age * (MAX_AGE - MIN_AGE) + AVG_AGE
            age = round(age, 2)
            label = ('%.2f ' % age) + label

            image_plot = image.copy()
            if bbox_face_list[i_person] is not None:
                image_plot = bbox_label(image=image_plot, bbox=bbox_face_list[i_person],
                                        label=('face ' + label),
                                        color=COLOR_PALETTE[i_person % len(COLOR_PALETTE)])

            if bbox_person_list[i_person] is not None:
                image_plot = bbox_label(image=image_plot, bbox=bbox_person_list[i_person],
                                        label=('person ' + label),
                                        color=COLOR_PALETTE[i_person % len(COLOR_PALETTE)])
            image = (image.astype(np.float32) * 0.4) + (image_plot.astype(np.float32) * 0.6)
            image = image.astype(np.uint8)

        # save visualization
        logger.info(f'saved at : {args.savepath}')
        cv2.imwrite(args.savepath, image[..., ::-1])

    logger.info('Script finished successfully.')


def recognize_from_video(net_YOLO, net_mivolo, tracker_face, tracker_person):
    # capture video
    capture = webcamera_utils.get_capture(args.video)
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    fps = capture.get(cv2.CAP_PROP_FPS)
    # create video writer if savepath is specified as video format
    if (args.savepath is not None) & (args.savepath.split('.')[-1] == 'mp4'):
        writer = webcamera_utils.get_writer(args.savepath, height, width, fps)
    else:
        writer = None

    frame_shown = False
    while True:
        # read frame
        ret, frame = capture.read()
        if (cv2.waitKey(1) & 0xFF == ord('q')) or not ret:
            break
        if frame_shown and cv2.getWindowProperty('frame', cv2.WND_PROP_VISIBLE) == 0:
            break
        frame = frame[..., ::-1].copy()

        # preprocessing
        if WEIGHT_YOLO_PATH[:5] == 'yolox':
            input_data = prep_input_yolox(frame, width=IMAGE_WIDTH_YOLO, height=IMAGE_HEIGHT_YOLO)
        else:
            input_data = prep_input(frame, width=IMAGE_WIDTH_YOLO, height=IMAGE_HEIGHT_YOLO)

        # inference
        if args.benchmark:
            logger.info('BENCHMARK mode')
            for i in range(args.benchmark_count):
                start = int(round(time.time() * 1000))
                output_YOLO = net_YOLO.run(input_data)
                end = int(round(time.time() * 1000))
                logger.info(f'\tailia processing time {end - start} ms')
        else:
            output_YOLO = net_YOLO.run(input_data)
        output_YOLO = output_YOLO[0][0]
        if WEIGHT_YOLO_PATH[:5] == 'yolox':
            class_conf = np.max(output_YOLO[:, 5:], axis=1)
            conf_mask = (output_YOLO[:, 4] * class_conf)
            class_pred = np.argmax(output_YOLO[:, 5:], axis=1)
            output_YOLO = output_YOLO[:, :6]
            output_YOLO[:, 4:] = 0
            output_YOLO[(class_pred == 0), 5] = conf_mask[(class_pred == 0)]
            output_YOLO[(class_pred == 1), 4] = conf_mask[(class_pred == 1)]
        else:
            output_YOLO = output_YOLO.T

        # apply threshold
        bbox_face = output_YOLO[output_YOLO[:, 5] > THRESH_YOLO][:, [0, 1, 2, 3, 5]]
        bbox_person = output_YOLO[output_YOLO[:, 4] > THRESH_YOLO][:, :5]

        # xywh -> xyxy
        bbox_face[:, :4] = xywh2xyxy(bbox_face[:, :4])
        bbox_person[:, :4] = xywh2xyxy(bbox_person[:, :4])

        # apply non-maximum-suppression
        bbox_face = nms(bbox=bbox_face, thresh_iou=THRESH_IOU)
        bbox_person = nms(bbox=bbox_person, thresh_iou=THRESH_IOU)

        # run tracking
        bbox_face = track(bbox_face, tracker_face)
        bbox_person = track(bbox_person, tracker_person)

        # rescale
        if WEIGHT_YOLO_PATH[:5] == 'yolox':
            r = min(IMAGE_HEIGHT_YOLO / frame.shape[0], IMAGE_WIDTH_YOLO / frame.shape[1])
            bbox_face[:, :4] = bbox_face[:, :4] / r
            bbox_person[:, :4] = bbox_person[:, :4] / r
        else:
            bbox_face[:, [0, 2]] = bbox_face[:, [0, 2]] * (frame.shape[1] / IMAGE_WIDTH_YOLO)
            bbox_face[:, [1, 3]] = bbox_face[:, [1, 3]] * (frame.shape[0] / IMAGE_HEIGHT_YOLO)
            bbox_person[:, [0, 2]] = bbox_person[:, [0, 2]] * (frame.shape[1] / IMAGE_WIDTH_YOLO)
            bbox_person[:, [1, 3]] = bbox_person[:, [1, 3]] * (frame.shape[0] / IMAGE_HEIGHT_YOLO)

        # cast float to int
        bbox_face = bbox_face[:, [0, 1, 2, 3, 5]]
        bbox_person = bbox_person[:, [0, 1, 2, 3, 5]]
        bbox_face = np.round(bbox_face).astype(np.int32)
        bbox_person = np.round(bbox_person).astype(np.int32)

        # clip
        bbox_face[:, [0, 2]] = np.clip(bbox_face[:, [0, 2]], 0, frame.shape[1])
        bbox_face[:, [1, 3]] = np.clip(bbox_face[:, [1, 3]], 0, frame.shape[0])
        bbox_person[:, [0, 2]] = np.clip(bbox_person[:, [0, 2]], 0, frame.shape[1])
        bbox_person[:, [1, 3]] = np.clip(bbox_person[:, [1, 3]], 0, frame.shape[0])

        # assigne bbox of person to bbox of face
        idx_person_assigned_face, idx_person_unassigned_face = assign_face(bbox_person, bbox_face)

        # organize
        image_face_list, image_person_list, \
        bbox_face_list, bbox_person_list = organize_info(frame, bbox_person, bbox_face, 
                                                         idx_person_assigned_face, idx_person_unassigned_face)

        for i_person in range(len(image_person_list)):
            image_face = image_face_list[i_person]
            image_person = image_person_list[i_person]

            # prepare input data for mivolo
            input_data_face = prep_input(image_face, width=IMAGE_WIDTH_MIVOLO, height=IMAGE_HEIGHT_MIVOLO,
                                         mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD)
            input_data_person = prep_input(image_person, width=IMAGE_WIDTH_MIVOLO, height=IMAGE_HEIGHT_MIVOLO,
                                           mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD)
            input_data = np.concatenate([input_data_face, input_data_person], axis=1)

            # inference
            if args.benchmark:
                logger.info('BENCHMARK mode')
                for i in range(args.benchmark_count):
                    start = int(round(time.time() * 1000))
                    output_mivolo = net_mivolo.run(input_data)
                    end = int(round(time.time() * 1000))
                    logger.info(f'\tailia processing time {end - start} ms')
            else:
                output_mivolo = net_mivolo.run(input_data)
            output_mivolo = output_mivolo[0][0]

            label = ''
            if args.no_gender:
                y_hat_gender = None
            else:
                # get_gender
                y_hat_gender = output_mivolo[:2]
                y_hat_gender = np.exp(y_hat_gender)
                y_hat_gender = y_hat_gender / np.sum(y_hat_gender)
                label = label + ('M' if (y_hat_gender[0] > 0.5) else 'F')

            # get_age
            age = output_mivolo[2]
            age = age * (MAX_AGE - MIN_AGE) + AVG_AGE
            age = round(age, 2)
            label = ('%.2f ' % age) + label

            frame_plot = frame.copy()
            if bbox_face_list[i_person] is not None:
                bbox = bbox_face_list[i_person]
                if bbox_person_list[i_person] is not None:
                    label = ('id:%d face ' % bbox_person_list[i_person][4]) + label
                    color = COLOR_PALETTE[bbox_person_list[i_person][4] % len(COLOR_PALETTE)]
                else:
                    label = ('id:(%d) face ' % bbox_face_list[i_person][4]) + label
                    color = COLOR_PALETTE[bbox_face_list[i_person][4] % len(COLOR_PALETTE)]
                frame_plot = bbox_label(image=frame_plot, bbox=bbox, label=label, color=color)

            if bbox_person_list[i_person] is not None:
                bbox = bbox_person_list[i_person]
                label = ('id:%d person ' % bbox_person_list[i_person][4]) + label
                color = COLOR_PALETTE[bbox_person_list[i_person][4] % len(COLOR_PALETTE)]
                frame_plot = bbox_label(image=frame_plot, bbox=bbox, label=label, color=color)
            frame = (frame.astype(np.float32) * 0.4) + (frame_plot.astype(np.float32) * 0.6)
            frame = frame.astype(np.uint8)

        # view result figure
        cv2.imshow('frame', frame[..., ::-1])
        frame_shown = True
        time.sleep(SLEEP_TIME)
        # save result
        if writer is not None:
            writer.write(frame[..., ::-1])

    capture.release()
    cv2.destroyAllWindows()
    if writer is not None:
        writer.release()
        # save visualization
        logger.info(f'saved at : {args.savepath}')

    logger.info('Script finished successfully.')


def main():
    # model files check and download
    check_and_download_models(WEIGHT_YOLO_PATH, MODEL_YOLO_PATH, REMOTE_PATH)
    check_and_download_models(WEIGHT_MIVOLO_PATH, MODEL_MIVOLO_PATH, REMOTE_PATH)

    # disable FP16
    if "FP16" in ailia.get_environment(args.env_id).props or sys.platform == 'Darwin':
        logger.warning('This model do not work on FP16. So use CPU mode.')
        args.env_id = 0

    # net initialize
    logger.info(f'env_id: {args.env_id}')
    net_YOLO = ailia.Net(MODEL_YOLO_PATH, WEIGHT_YOLO_PATH, env_id=args.env_id, memory_mode=args.memory_mode)
    net_mivolo = ailia.Net(MODEL_MIVOLO_PATH, WEIGHT_MIVOLO_PATH, env_id=args.env_id, memory_mode=args.memory_mode)

    tracker_face = BYTETracker(track_thresh=TRACK_THRESH, track_buffer=TRACK_BUFFER,
                               match_thresh=MATCH_THRESH, frame_rate=FRAME_RATE, mot20=MOT20)
    tracker_person = BYTETracker(track_thresh=TRACK_THRESH, track_buffer=TRACK_BUFFER,
                                 match_thresh=MATCH_THRESH, frame_rate=FRAME_RATE, mot20=MOT20)

    if args.video is None:
        # image mode
        recognize_from_image(net_YOLO, net_mivolo)
    else:
        # video mode
        recognize_from_video(net_YOLO, net_mivolo, tracker_face, tracker_person)


if __name__ == '__main__':
    main()
