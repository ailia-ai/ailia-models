# ax Action recognition
# (c) 2020 ax Inc.

import sys
import time
import argparse

import cv2
import numpy as np
import math

import ailia

# import tracking modules
sys.path.append('../../object_tracking/deepsort')
from sort.tracker import Tracker
from sort.nn_matching import NearestNeighborDistanceMetric
from deepsort_utils import Detection,xywh_to_xyxy,xywh_to_tlwh,tlwh_to_xyxy,xyxy_to_tlwh,\
    get_detector_result,non_max_suppression

# import original modules
sys.path.append('../../util')
from webcamera_utils import adjust_frame_size  # noqa: E402
from image_utils import load_image  # noqa: E402
from image_utils import normalize_image  # noqa: E402
from model_utils import check_and_download_models  # noqa: E402
from arg_utils import check_file_existance  # noqa: E402
from arg_utils import get_base_parser, update_parser  # noqa: E402

from ax_action_recognition_util import pose_postprocess,TIME_RANGE,get_detector_result_lw_human_pose,draw_boxes,softmax
sys.path.append('../../pose_estimation/pose_resnet')
from pose_resnet_util import compute,keep_aspect



# ======================
# Parameters 1
# ======================

ALGORITHM = ailia.POSE_ALGORITHM_LW_HUMAN_POSE

MODEL_LISTS = ['lw_human_pose', 'pose_resnet']

COCO_CATEGORY = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train",
    "truck", "boat", "traffic light", "fire hydrant", "stop sign",
    "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow",
    "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
    "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard",
    "sports ball", "kite", "baseball bat", "baseball glove", "skateboard",
    "surfboard", "tennis racket", "bottle", "wine glass", "cup", "fork",
    "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange",
    "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair",
    "couch", "potted plant", "bed", "dining table", "toilet", "tv",
    "laptop", "mouse", "remote", "keyboard", "cell phone", "microwave",
    "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase",
    "scissors", "teddy bear", "hair drier", "toothbrush"
]

LABELS=['stand', 'walk', 'run', 'jump', 'sit', 'squat', 'kick', 'punch', 'wave']

FRAME_SKIP = True

# ======================
# Arguemnt Parser Config
# ======================
parser = get_base_parser('Action recognition', None, None)
parser.add_argument(
    '-n', '--normal',
    action='store_true',
    help='By default, the optimized model is used, but with this option, ' +
    'you can switch to the normal (not optimized) model'
)
parser.add_argument(
    '-a', '--arch', metavar='ARCH',
    default='lw_human_pose', choices=MODEL_LISTS,
    help='model lists: ' + ' | '.join(MODEL_LISTS)
)
parser.add_argument(
    '-f', '--fps',
    default=10,
    help='Input fps for the detection model'
)
args = update_parser(parser)

POSE_KEY = [
    ailia.POSE_KEYPOINT_NOSE,
    ailia.POSE_KEYPOINT_SHOULDER_CENTER,
    ailia.POSE_KEYPOINT_SHOULDER_RIGHT,
    ailia.POSE_KEYPOINT_ELBOW_RIGHT,
    ailia.POSE_KEYPOINT_WRIST_RIGHT,
    ailia.POSE_KEYPOINT_SHOULDER_LEFT,
    ailia.POSE_KEYPOINT_ELBOW_LEFT,
    ailia.POSE_KEYPOINT_WRIST_LEFT,
    ailia.POSE_KEYPOINT_HIP_RIGHT,
    ailia.POSE_KEYPOINT_KNEE_RIGHT,
    ailia.POSE_KEYPOINT_ANKLE_RIGHT,
    ailia.POSE_KEYPOINT_HIP_LEFT,
    ailia.POSE_KEYPOINT_KNEE_LEFT,
    ailia.POSE_KEYPOINT_ANKLE_LEFT,
    ailia.POSE_KEYPOINT_EYE_RIGHT,
    ailia.POSE_KEYPOINT_EYE_LEFT,
    ailia.POSE_KEYPOINT_EAR_RIGHT,
    ailia.POSE_KEYPOINT_EAR_LEFT,
]

def ailia_to_openpose(person):
    pose_keypoints = np.zeros((18, 3))
    for i, key in enumerate(POSE_KEY):
        p = person.points[key]
        pose_keypoints[i, :] = [p.x, p.y, float(p.score)]
    return pose_keypoints

# ======================
# Parameters 2
# ======================
MODEL_NAME = 'lightweight-human-pose-estimation'
if args.normal:
    WEIGHT_PATH = f'{MODEL_NAME}.onnx'
    MODEL_PATH = f'{MODEL_NAME}.onnx.prototxt'
else:
    WEIGHT_PATH = f'{MODEL_NAME}.opt.onnx'
    MODEL_PATH = f'{MODEL_NAME}.opt.onnx.prototxt'
REMOTE_PATH = f'https://storage.googleapis.com/ailia-models/{MODEL_NAME}/'

EX_WEIGHT_PATH = 'deep_sort.onnx'
EX_MODEL_PATH = 'deep_sort.onnx.prototxt'
EX_REMOTE_PATH = 'https://storage.googleapis.com/ailia-models/deep_sort/'

POSE_MODEL_NAME = 'pose_resnet_50_256x192'
POSE_WEIGHT_PATH = f'{POSE_MODEL_NAME}.onnx'
POSE_MODEL_PATH = f'{POSE_MODEL_NAME}.onnx.prototxt'
POSE_REMOTE_PATH = f'https://storage.googleapis.com/ailia-models/pose_resnet/'

DETECTOR_WEIGHT_PATH = 'yolov3.opt.onnx'
DETECTOR_MODEL_PATH = 'yolov3.opt.onnx.prototxt'
DETECTOR_REMOTE_PATH = 'https://storage.googleapis.com/ailia-models/yolov3/'

ACTION_WEIGHT_PATH = 'action.onnx'
ACTION_MODEL_PATH = 'action.onnx.prototxt'
ACTION_REMOTE_PATH = 'https://storage.googleapis.com/ailia-models/ax_action_recognition/'

# Deep sort model input
EX_INPUT_HEIGHT = 128
EX_INPUT_WIDTH = 64

# Metric parameters
MAX_COSINE_DISTANCE = 0.2  # threshold of matching object
NN_BUDGET = 100
MIN_CONFIDENCE = 0.3

#YOLO
THRESHOLD = 0.4
MIN_CONFIDENCE = 0.3
IOU = 0.45
POSE_THRESHOLD = 0.3

# ======================
# Utils
# ======================
def hsv_to_rgb(h, s, v):
    bgr = cv2.cvtColor(
        np.array([[[h, s, v]]], dtype=np.uint8), cv2.COLOR_HSV2BGR
    )[0][0]
    return (int(bgr[2]), int(bgr[1]), int(bgr[0]))


def line(input_img, person, point1, point2):
    threshold = POSE_THRESHOLD
    if person.points[point1].score > threshold and\
       person.points[point2].score > threshold:
        color = hsv_to_rgb(255*point1/ailia.POSE_KEYPOINT_CNT, 255, 255)

        x1 = int(input_img.shape[1] * person.points[point1].x)
        y1 = int(input_img.shape[0] * person.points[point1].y)
        x2 = int(input_img.shape[1] * person.points[point2].x)
        y2 = int(input_img.shape[0] * person.points[point2].y)
        cv2.line(input_img, (x1, y1), (x2, y2), color, 5)


def display_result(input_img, person):
    line(input_img, person, ailia.POSE_KEYPOINT_NOSE,
            ailia.POSE_KEYPOINT_SHOULDER_CENTER)
    line(input_img, person, ailia.POSE_KEYPOINT_SHOULDER_LEFT,
            ailia.POSE_KEYPOINT_SHOULDER_CENTER)
    line(input_img, person, ailia.POSE_KEYPOINT_SHOULDER_RIGHT,
            ailia.POSE_KEYPOINT_SHOULDER_CENTER)

    line(input_img, person, ailia.POSE_KEYPOINT_EYE_LEFT,
            ailia.POSE_KEYPOINT_NOSE)
    line(input_img, person, ailia.POSE_KEYPOINT_EYE_RIGHT,
            ailia.POSE_KEYPOINT_NOSE)
    line(input_img, person, ailia.POSE_KEYPOINT_EAR_LEFT,
            ailia.POSE_KEYPOINT_EYE_LEFT)
    line(input_img, person, ailia.POSE_KEYPOINT_EAR_RIGHT,
            ailia.POSE_KEYPOINT_EYE_RIGHT)

    line(input_img, person, ailia.POSE_KEYPOINT_ELBOW_LEFT,
            ailia.POSE_KEYPOINT_SHOULDER_LEFT)
    line(input_img, person, ailia.POSE_KEYPOINT_ELBOW_RIGHT,
            ailia.POSE_KEYPOINT_SHOULDER_RIGHT)
    line(input_img, person, ailia.POSE_KEYPOINT_WRIST_LEFT,
            ailia.POSE_KEYPOINT_ELBOW_LEFT)
    line(input_img, person, ailia.POSE_KEYPOINT_WRIST_RIGHT,
            ailia.POSE_KEYPOINT_ELBOW_RIGHT)

    line(input_img, person, ailia.POSE_KEYPOINT_BODY_CENTER,
            ailia.POSE_KEYPOINT_SHOULDER_CENTER)
    line(input_img, person, ailia.POSE_KEYPOINT_HIP_LEFT,
            ailia.POSE_KEYPOINT_BODY_CENTER)
    line(input_img, person, ailia.POSE_KEYPOINT_HIP_RIGHT,
            ailia.POSE_KEYPOINT_BODY_CENTER)

    line(input_img, person, ailia.POSE_KEYPOINT_KNEE_LEFT,
            ailia.POSE_KEYPOINT_HIP_LEFT)
    line(input_img, person, ailia.POSE_KEYPOINT_ANKLE_LEFT,
            ailia.POSE_KEYPOINT_KNEE_LEFT)
    line(input_img, person, ailia.POSE_KEYPOINT_KNEE_RIGHT,
            ailia.POSE_KEYPOINT_HIP_RIGHT)
    line(input_img, person, ailia.POSE_KEYPOINT_ANKLE_RIGHT,
            ailia.POSE_KEYPOINT_KNEE_RIGHT)


def action_recognition(box,input_image,pose,detector,model,data):
    if args.arch=="lw_human_pose":
        bbox_xywh, cls_conf, cls_ids = get_detector_result_lw_human_pose(pose, input_image.shape[0], input_image.shape[1], get_all=True)

        idx = -1
        min_d = 32768

        for i in range(pose.get_object_count()):
            target=xywh_to_xyxy(bbox_xywh[i], input_image.shape[0], input_image.shape[1])
            d=math.sqrt((target[0]-box[0])**2 + (target[1]-box[1])**2)
            if d < min_d:
                min_d = d
                idx = i

        if idx == -1:
            return "-", None

        person = pose.get_object_pose(idx)
    else:
        bbox_xywh, cls_conf, cls_ids = get_detector_result(detector, input_image.shape[0], input_image.shape[1])
        px1,py1,px2,py2 = keep_aspect((box[0],box[1]),(box[2],box[3]),input_image,pose)
        crop_img = input_image[py1:py2,px1:px2,:]
        offset_x = px1/input_image.shape[1]
        offset_y = py1/input_image.shape[0]
        scale_x = crop_img.shape[1]/input_image.shape[1]
        scale_y = crop_img.shape[0]/input_image.shape[0]
        detections = compute(pose,crop_img,offset_x,offset_y,scale_x,scale_y)
        person=detections
    
    openpose_keypoints=ailia_to_openpose(person)
    frame = np.expand_dims(openpose_keypoints, axis=1)
    frame = pose_postprocess(frame)

    for i in range(TIME_RANGE-1):
        data[:,i,:]=data[:,i+1,:] #data: (ailia.POSE_KEYPOINT_CNT,TIME_RANGE,3)
   
    data[:,TIME_RANGE-1,:] = frame[:,0,:]

    zero_cnt = 0
    for i in range(TIME_RANGE):
        if np.sum(data[:,i,:])==0:
            zero_cnt=zero_cnt+1

    if zero_cnt>=1:
        return "-", person

    data_rgb = data.transpose((2 ,1, 0))
    data_rgb = data_rgb[:2,...]   # May need to be removed if input for action model changed

    data_rgb = np.expand_dims(data_rgb, axis=3)
    data_rgb.shape = (1,) + data_rgb.shape

    model.set_input_shape(data_rgb.shape)
    action = model.predict(data_rgb)

    action = softmax(action)
    max_prob=0
    class_idx=0
    for i in range(len(LABELS)):
        if max_prob<=action[0][i]:
            max_prob = action[0][i]
            class_idx = i

    return LABELS[class_idx]+" "+str(int(max_prob*100)/100),person

def resize(img, size=(EX_INPUT_WIDTH, EX_INPUT_HEIGHT)):
    return cv2.resize(img.astype(np.float32), size)


# ======================
# Main functions
# ======================

def recognize_from_video():
    try:
        print('[INFO] Webcam mode is activated')
        capture = cv2.VideoCapture(int(args.video))
        if not capture.isOpened():
            print("[ERROR] webcamera not found")
            sys.exit(1)
    except ValueError:
        if check_file_existance(args.video):
            capture = cv2.VideoCapture(args.video)

    frame_rate = capture.get(cv2.CAP_PROP_FPS)
    if FRAME_SKIP:
        action_recognize_fps = int(args.fps)
    else:
        action_recognize_fps = frame_rate

    if args.savepath != "":
        size = (int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)),
                int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)))
        fmt = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
        writer = cv2.VideoWriter(args.savepath, fmt, action_recognize_fps, size)
    else:
        writer = None

    # pose estimation
    env_id = args.env_id
    print(f'env_id: {env_id}')
    if args.arch=="lw_human_pose":
        pose = ailia.PoseEstimator(
            MODEL_PATH, WEIGHT_PATH, env_id=env_id, algorithm=ALGORITHM,
            memory_mode=args.memory_mode
        )

        detector = None
    else:
        detector = ailia.Detector(
            DETECTOR_MODEL_PATH,
            DETECTOR_WEIGHT_PATH,
            len(COCO_CATEGORY),
            format=ailia.NETWORK_IMAGE_FORMAT_RGB,
            channel=ailia.NETWORK_IMAGE_CHANNEL_FIRST,
            range=ailia.NETWORK_IMAGE_RANGE_U_FP32,
            algorithm=ailia.DETECTOR_ALGORITHM_YOLOV3,
            env_id=env_id,
            memory_mode=args.memory_mode
        )

        pose = ailia.Net(POSE_MODEL_PATH, POSE_WEIGHT_PATH, env_id=env_id, memory_mode=args.memory_mode)

    # tracker class instance
    extractor = ailia.Net(EX_MODEL_PATH, EX_WEIGHT_PATH, env_id=env_id, memory_mode=args.memory_mode)
    metric = NearestNeighborDistanceMetric(
        "cosine", MAX_COSINE_DISTANCE, NN_BUDGET
    )
    tracker = Tracker(
        metric,
        max_iou_distance=0.7,
        max_age=70,
        n_init=3
    )

    # action recognition
    env_id = args.env_id
    print(f'env_id: {env_id}')
    model = ailia.Net(ACTION_MODEL_PATH, ACTION_WEIGHT_PATH, env_id=env_id, memory_mode=args.memory_mode)

    action_data = {}

    frame_nb = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    idx_frame = 0

    frame_shown = False
    while(True):
        ret, frame = capture.read()

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        if frame_shown and cv2.getWindowProperty('frame', cv2.WND_PROP_VISIBLE) == 0:
            break
        
        if (not ret) or (frame_nb>=1 and idx_frame>=frame_nb):
            break

        if FRAME_SKIP:
            mod = round(frame_rate/action_recognize_fps)
            if mod>=1:
                if idx_frame%mod != 0:
                    idx_frame = idx_frame + 1
                    continue

        input_image, input_data = adjust_frame_size(
            frame, frame.shape[0], frame.shape[1],
        )
        input_data = cv2.cvtColor(input_data, cv2.COLOR_BGR2BGRA)

        # inferece
        if args.arch=="lw_human_pose":
            _ = pose.compute(input_data)
        else:
            detector.compute(input_data, THRESHOLD, IOU)

        # deepsort format
        h, w = input_image.shape[0], input_image.shape[1]
        if args.arch=="lw_human_pose":
            bbox_xywh, cls_conf, cls_ids = get_detector_result_lw_human_pose(pose, h, w)
        else:
            bbox_xywh, cls_conf, cls_ids = get_detector_result(detector, h, w)

        mask = cls_ids == 0
        bbox_xywh = bbox_xywh[mask]

        # bbox dilation just in case bbox too small,
        # delete this line if using a better pedestrian detector
        if args.arch=="pose_resnet":
            # bbox_xywh[:, 3:] *= 1.2   #May need to be removed in the future
            cls_conf = cls_conf[mask]

        # do tracking
        img_crops = []
        for box in bbox_xywh:
            x1, y1, x2, y2 = xywh_to_xyxy(box, h, w)
            img_crops.append(input_image[y1:y2, x1:x2])

        if img_crops:
            # preprocess
            img_batch = np.concatenate([
                normalize_image(resize(img), 'ImageNet')[np.newaxis, :, :, :]
                for img in img_crops
            ], axis=0).transpose(0, 3, 1, 2)

            # TODO better to pass a batch at once
            # features = extractor.predict(img_batch)
            features = []
            for img in img_batch:
                features.append(extractor.predict(img[np.newaxis, :, :, :])[0])
            features = np.array(features)
        else:
            features = np.array([])

        bbox_tlwh = xywh_to_tlwh(bbox_xywh)
        detections = [
            Detection(bbox_tlwh[i], conf, features[i])
            for i, conf in enumerate(cls_conf) if conf > MIN_CONFIDENCE
        ]

        # run on non-maximum supression
        boxes = np.array([d.tlwh for d in detections])
        scores = np.array([d.confidence for d in detections])
        nms_max_overlap = 1.0
        indices = non_max_suppression(boxes, nms_max_overlap, scores)
        detections = [detections[i] for i in indices]

        # update tracker
        tracker.predict()
        tracker.update(detections)

        # update bbox identities
        outputs = []
        for track in tracker.tracks:
            if not track.is_confirmed() or track.time_since_update > 1:
                continue
            box = track.to_tlwh()
            x1, y1, x2, y2 = tlwh_to_xyxy(box, h, w)
            track_id = track.track_id
            outputs.append(np.array([x1, y1, x2, y2, track_id], dtype=int))
        if len(outputs) > 0:
            outputs = np.stack(outputs, axis=0)

        # action detection
        actions = []
        persons = []
        if len(outputs) > 0:
            bbox_xyxy = outputs[:, :4]
            identities = outputs[:, -1]
            for i, box in enumerate(bbox_xyxy):
                id = identities[i]

                if not(id in action_data):
                    action_data[id] = np.zeros((ailia.POSE_KEYPOINT_CNT-1,TIME_RANGE,3))

                # action recognition
                action,person = action_recognition(box, input_image, pose, detector, model, action_data[id])
                actions.append(action)
                persons.append(person)
                
        # draw box for visualization
        if len(outputs) > 0:
            bbox_tlwh = []
            bbox_xyxy = outputs[:, :4]
            identities = outputs[:, -1]
            frame = draw_boxes(input_image, bbox_xyxy, identities, actions, action_data, (0, 0))

            for bb_xyxy in bbox_xyxy:
                bbox_tlwh.append(xyxy_to_tlwh(bb_xyxy))

        # draw skelton
        for person in persons:
            if person!=None:
                display_result(input_image, person)

        if writer is not None:
            writer.write(input_image)

            # show progress
            if idx_frame == "0":
                print()
            print("\r" + str(idx_frame + 1) + " / " + str(frame_nb) ,end="")
            if idx_frame == frame_nb - 1:
                print()

        cv2.imshow('frame', input_image)
        frame_shown = True

        idx_frame = idx_frame + 1

    if writer is not None:
        writer.release()

    capture.release()
    cv2.destroyAllWindows()
    print('Script finished successfully.')


def main():
    # model files check and download
    if args.arch=="lw_human_pose":
        check_and_download_models(WEIGHT_PATH, MODEL_PATH, REMOTE_PATH)
        check_and_download_models(EX_WEIGHT_PATH, EX_MODEL_PATH, EX_REMOTE_PATH)
        check_and_download_models(ACTION_WEIGHT_PATH, ACTION_MODEL_PATH,ACTION_REMOTE_PATH)
    else:
        check_and_download_models(POSE_WEIGHT_PATH, POSE_MODEL_PATH, POSE_REMOTE_PATH)
        check_and_download_models(DETECTOR_WEIGHT_PATH, DETECTOR_MODEL_PATH, DETECTOR_REMOTE_PATH)
        check_and_download_models(EX_WEIGHT_PATH, EX_MODEL_PATH, EX_REMOTE_PATH)
        check_and_download_models(ACTION_WEIGHT_PATH, ACTION_MODEL_PATH,ACTION_REMOTE_PATH)

    if args.video is not None:
        # video mode
        recognize_from_video()
    else:
        print("video not found")


if __name__ == '__main__':
    main()
