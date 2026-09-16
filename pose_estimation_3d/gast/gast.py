import os
import sys
import time
import json

import numpy as np
import cv2
from tqdm import tqdm

import ailia

# import original modules
sys.path.append('../../util')
from arg_utils import get_base_parser, update_parser  # noqa: E402
from model_utils import check_and_download_models  # noqa: E402
from image_utils import normalize_image  # noqa: E402
from webcamera_utils import get_capture  # noqa: E402

# logger
from logging import getLogger  # noqa: E402

from gast_utils import *

logger = getLogger(__name__)

# ======================
# Parameters
# ======================

WEIGHT_YOLOV3_PATH = 'yolov3.opt.onnx'
MODEL_YOLOV3_PATH = 'yolov3.opt.onnx.prototxt'
REMOTE_YOLOV3_PATH = 'https://storage.googleapis.com/ailia-models/yolov3/'
WEIGHT_POSE_PATH = 'pose_hrnet_w48_384x288.onnx'
MODEL_POSE_PATH = 'pose_hrnet_w48_384x288.onnx.prototxt'
WEIGHT_27FRAME_17JOINT_PATH = '27_frame_17_joint_model.onnx'
MODEL_27FRAME_17JOINT_PATH = '27_frame_17_joint_model.onnx.prototxt'
REMOTE_PATH = 'https://storage.googleapis.com/ailia-models/gast/'

VIDEO_PATH = 'baseball.mp4'
SAVE_PATH = 'output.mp4'

ROT = np.array([0.14070565, -0.15007018, -0.7552408, 0.62232804], dtype=np.float32)

# ======================
# Arguemnt Parser Config
# ======================

parser = get_base_parser('GAST model', VIDEO_PATH, SAVE_PATH)
parser.add_argument(
    '-np', '--num_person', type=int, default=1, choices=(1, 2),
    help='number of estimated human poses. [1, 2]'
)
parser.add_argument(
    '--onnx',
    action='store_true',
    help='execute onnxruntime version.'
)
parser.add_argument(
    '--low_memory',
    action='store_true',
    help='execute low memory version.'
)
args = update_parser(parser)


# ======================
# Main functions
# ======================

def revise_kpts(h36m_kpts, h36m_scores, valid_frames):
    new_h36m_kpts = np.zeros_like(h36m_kpts)
    for index, frames in enumerate(valid_frames):
        kpts = h36m_kpts[index, frames]
        score = h36m_scores[index, frames]

        index_frame = np.where(np.sum(score < 0.3, axis=1) > 0)[0]

        for frame in index_frame:
            less_threshold_joints = np.where(score[frame] < 0.3)[0]

            intersect = [i for i in [2, 3, 5, 6] if i in less_threshold_joints]

            if [2, 3, 5, 6] == intersect:
                kpts[frame, [2, 3, 5, 6]] = kpts[frame, [1, 1, 4, 4]]
            elif [2, 3, 6] == intersect:
                kpts[frame, [2, 3, 6]] = kpts[frame, [1, 1, 5]]
            elif [3, 5, 6] == intersect:
                kpts[frame, [3, 5, 6]] = kpts[frame, [2, 4, 4]]
            elif [3, 6] == intersect:
                kpts[frame, [3, 6]] = kpts[frame, [2, 5]]
            elif [3] == intersect:
                kpts[frame, 3] = kpts[frame, 2]
            elif [6] == intersect:
                kpts[frame, 6] = kpts[frame, 5]
            else:
                continue

        new_h36m_kpts[index, frames] = kpts
    return new_h36m_kpts


def revise_skes(prediction, re_kpts, valid_frames):
    ratio_2d_3d = 500.

    new_prediction = np.zeros((*re_kpts.shape[:-1], 3), dtype=np.float32)
    for i, frames in enumerate(valid_frames):
        new_prediction[i, frames] = prediction[i]

        # The origin of (x, y) is in the upper right corner,
        # while the (x,y) coordinates in the image are in the upper left corner.
        distance = re_kpts[i, frames[1:], :, :2] - re_kpts[i, frames[:1], :, :2]
        distance = np.mean(distance[:, [1, 4, 11, 14]], axis=-2, keepdims=True)
        new_prediction[i, frames[1:], :, 0] -= distance[..., 0] / ratio_2d_3d
        new_prediction[i, frames[1:], :, 1] += distance[..., 1] / ratio_2d_3d

    # The origin of (x, y) is in the upper right corner,
    # while the (x,y) coordinates in the image are in the upper left corner.
    # Calculate the relative distance between two people
    if len(valid_frames) == 2:
        intersec_frames = [frame for frame in valid_frames[0] if frame in valid_frames[1]]
        absolute_distance = re_kpts[0, intersec_frames[:1], :, :2] - re_kpts[1, intersec_frames[:1], :, :2]
        absolute_distance = np.mean(absolute_distance[:, [1, 4, 11, 14]], axis=-2, keepdims=True) / 2.

        new_prediction[0, valid_frames[0], :, 0] -= absolute_distance[..., 0] / ratio_2d_3d
        new_prediction[0, valid_frames[0], :, 1] += absolute_distance[..., 1] / ratio_2d_3d

        new_prediction[1, valid_frames[1], :, 0] += absolute_distance[..., 0] / ratio_2d_3d
        new_prediction[1, valid_frames[1], :, 1] -= absolute_distance[..., 1] / ratio_2d_3d

    # Pre-processing the case where the movement of Z axis is relatively large, such as 'sitting down'
    # Remove the absolute distance
    # new_prediction[:, :, 1:] -= new_prediction[:, :, :1]
    # new_prediction[:, :, 0] = 0
    new_prediction[:, :, :, 2] -= np.amin(new_prediction[:, :, :, 2])

    return new_prediction


def gen_kpts(frames, yolo_model, pose_model, num_peroson=1):
    # collect keypoints coordinate
    logger.info('Generating 2D pose ...')

    people_sort = ObjSort()

    kpts_result = []
    scores_result = []
    for i in tqdm(range(len(frames))):
        frame = frames[i]
        img = cv2.cvtColor(frame, cv2.COLOR_BGR2BGRA)
        bboxs, scores = yolo_human_det(img, yolo_model)
        if not bboxs.any():
            continue

        # Using Sort to track people
        people_track = people_sort.update(bboxs)

        # Track the first two people in the video and remove the ID
        if people_track.shape[0] == 1:
            people_track_ = people_track[-1, :-1].reshape(1, 4)
        elif people_track.shape[0] >= 2:
            people_track_ = people_track[-num_peroson:, :-1].reshape(num_peroson, 4)
            people_track_ = people_track_[::-1]
        else:
            continue

        track_bboxs = []
        for bbox in people_track_:
            bbox = [round(i, 2) for i in list(bbox)]
            track_bboxs.append(bbox)

        inputs, origin_img, center, scale = preprocess(frame, track_bboxs, num_peroson)
        inputs = inputs[:, [2, 1, 0]]

        output = pose_model.predict({'input': inputs})[0]

        # compute coordinate
        preds, maxvals = get_final_preds(output, np.asarray(center), np.asarray(scale))

        kpts = np.zeros((num_peroson, 17, 2), dtype=np.float32)
        scores = np.zeros((num_peroson, 17), dtype=np.float32)
        for i, kpt in enumerate(preds):
            kpts[i] = kpt

        for i, score in enumerate(maxvals):
            scores[i] = score.squeeze()

        kpts_result.append(kpts)
        scores_result.append(scores)

    keypoints = np.array(kpts_result)
    scores = np.array(scores_result)

    keypoints = keypoints.transpose(1, 0, 2, 3)  # (T, M, N, 2) --> (M, T, N, 2)
    scores = scores.transpose(1, 0, 2)  # (T, M, N) --> (M, T, N)

    return keypoints, scores


def yolo_human_det(img, detector, confidence=0.70, nms_thresh=0.4):
    detector.compute(img, confidence, nms_thresh)

    h, w = img.shape[:2]
    bboxs = []
    scores = []

    count = detector.get_object_count()
    for idx in range(count):
        obj = detector.get_object(idx)
        if obj.category != 0:
            # not human
            continue

        bboxs.append(np.array([
            obj.x * w,
            obj.y * h,
            (obj.x + obj.w) * w,
            (obj.x + obj.h) * h,
        ]))
        scores.append(obj.prob)

    bboxs = np.asarray(bboxs)
    scores = np.array(scores).reshape(-1, 1)

    return bboxs, scores


def box_to_center_scale(box, model_image_width, model_image_height):
    """convert a box to center,scale information required for pose transformation
    Parameters
    ----------
    box : (x1, y1, x2, y2)
    model_image_width : int
    model_image_height : int

    Returns
    -------
    (numpy array, numpy array)
        Two numpy arrays, coordinates for the center of the box and the scale of the box
    """
    center = np.zeros((2), dtype=np.float32)
    x1, y1, x2, y2 = box[:4]
    box_width, box_height = x2 - x1, y2 - y1

    center[0] = x1 + box_width * 0.5
    center[1] = y1 + box_height * 0.5

    aspect_ratio = model_image_width * 1.0 / model_image_height
    pixel_std = 200

    if box_width > aspect_ratio * box_height:
        box_height = box_width * 1.0 / aspect_ratio
    elif box_width < aspect_ratio * box_height:
        box_width = box_height * aspect_ratio
    scale = np.array(
        [box_width * 1.0 / pixel_std, box_height * 1.0 / pixel_std],
        dtype=np.float32)
    if center[0] != -1:
        scale = scale * 1.25

    return center, scale


def preprocess(img, bboxs, num_pos=2):
    IMAGE_SIZE = (288, 384)

    inputs = []
    centers = []
    scales = []
    for bbox in bboxs[:num_pos]:
        c, s = box_to_center_scale(bbox, img.shape[0], img.shape[1])
        centers.append(c)
        scales.append(s)
        r = 0

        trans = get_affine_transform(c, s, r, IMAGE_SIZE)
        input = cv2.warpAffine(
            img,
            trans,
            (IMAGE_SIZE[0], IMAGE_SIZE[1]),
            flags=cv2.INTER_LINEAR)

        input = normalize_image(input.astype(np.float32), 'ImageNet')
        input = input.transpose(2, 0, 1)  # HWC -> CHW
        input = np.expand_dims(input, axis=0)
        inputs.append(input)

    inputs = np.vstack(inputs)
    return inputs, img, centers, scales


def gen_pose(
        kpts, valid_frames, width, height, net,
        pad=13, causal_shift=0, num_joints=17):
    joints_left, joints_right, h36m_skeleton, keypoints_metadata = get_joints_info(num_joints)
    kps_left, kps_right = joints_left, joints_right

    norm_seqs = []
    for index, frames in enumerate(valid_frames):
        seq_kps = kpts[index, frames]
        norm_seq_kps = normalize_screen_coordinates(seq_kps, w=width, h=height)
        norm_seqs.append(norm_seq_kps)

    generator = DataLoader(
        norm_seqs,
        pad=pad, causal_shift=causal_shift,
        kps_left=kps_left, kps_right=kps_right
    )

    prediction = []
    for batch_2d in generator.next_epoch():
        if args.low_memory:
            frames = batch_2d.shape[1]-pad*2
            predicted_3d_pos = np.zeros((2,frames,17,3))
            for i in range(0,frames):
                input_data = batch_2d[:,i:i+pad*2+1,:,:]
                output = net.predict({'inputs_2d': input_data})
                predicted_3d_pos[:,i:i+1,:,:] = output[0]
        else:
            if not args.onnx:
                output = net.predict({'inputs_2d': batch_2d})
            else:
                in_name = net.get_inputs()[0].name
                out_name = net.get_outputs()[0].name
                output = net.run([out_name],
                                {in_name: batch_2d})
            predicted_3d_pos = output[0]

        predicted_3d_pos[1, :, :, 0] *= -1
        predicted_3d_pos[1, :, joints_left + joints_right] = \
            predicted_3d_pos[1, :, joints_right + joints_left]
        predicted_3d_pos = np.mean(predicted_3d_pos, axis=0, keepdims=True)
        predicted_3d_pos = predicted_3d_pos.squeeze(0)
        prediction.append(predicted_3d_pos)

    prediction_to_world = []
    for i in range(len(prediction)):
        sub_prediction = prediction[i]
        sub_prediction = camera_to_world(sub_prediction, R=ROT, t=0)
        prediction_to_world.append(sub_prediction)

    return prediction_to_world


def recognize_from_video(net, info):
    video_file = args.video if args.video else args.input[0]
    cap = get_capture(video_file)
    assert cap.isOpened(), 'Cannot capture source'

    num_joints = 17

    # Get the width and height of video
    width = int(round(cap.get(cv2.CAP_PROP_FRAME_WIDTH)))
    height = int(round(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))

    # Load video frame
    video_length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frames = []
    for i in range(video_length):
        ret, frame = cap.read()
        if not ret:
            continue
        frames.append(frame)

    num_person = info["num_person"]
    keypoints, scores = gen_kpts(
        frames, info["yolo_model"], info["pose_model"],
        num_peroson=num_person)

    keypoints, scores, valid_frames = h36m_coco_format(keypoints, scores)
    re_kpts = revise_kpts(keypoints, scores, valid_frames)
    num_person = len(re_kpts)

    receptive_fields = 27
    pad = (receptive_fields - 1) // 2  # Padding on each side
    causal_shift = 0

    logger.info('Generating 3D human pose ...')
    prediction = gen_pose(
        re_kpts, valid_frames, width, height, net,
        pad, causal_shift, num_joints)

    # Adding absolute distance to 3D poses and rebase the height
    if num_person == 2:
        prediction = revise_skes(prediction, re_kpts, valid_frames)
    else:
        prediction[0][:, :, 2] -= np.amin(prediction[0][:, :, 2])

    # If output two 3D human poses, put them in the same 3D coordinate system
    same_coord = False
    if num_person == 2:
        same_coord = True

    anim_output = {}
    for i, anim_prediction in enumerate(prediction):
        anim_output.update({'Reconstruction %d' % (i + 1): anim_prediction})

    _, _, h36m_skeleton, keypoints_metadata = get_joints_info(num_joints)

    logger.info('Rendering ...')
    re_kpts = re_kpts.transpose((1, 0, 2, 3))  # (M, T, N, 2) --> (T, M, N, 2)
    frames = [cv2.cvtColor(f, cv2.COLOR_BGR2RGB) for f in frames]
    render_animation(
        re_kpts, keypoints_metadata, anim_output, h36m_skeleton, 25, 3000,
        np.array(70., dtype=np.float32), args.savepath,
        frames, viewport=(width, height),
        downsample=1, size=5,
        same_coord=same_coord)

    logger.info('Script finished successfully.')


def main():
    # model files check and download
    logger.info("=== YOLOv3 model ===")
    check_and_download_models(WEIGHT_YOLOV3_PATH, MODEL_YOLOV3_PATH, REMOTE_YOLOV3_PATH)
    logger.info("=== HRNet model ===")
    check_and_download_models(WEIGHT_POSE_PATH, MODEL_POSE_PATH, REMOTE_PATH)
    logger.info("=== GAST model ===")
    check_and_download_models(WEIGHT_27FRAME_17JOINT_PATH, MODEL_27FRAME_17JOINT_PATH, REMOTE_PATH)

    num_person = args.num_person

    # net initialize
    detector = ailia.Detector(
        MODEL_YOLOV3_PATH,
        WEIGHT_YOLOV3_PATH,
        80,
        format=ailia.NETWORK_IMAGE_FORMAT_RGB,
        channel=ailia.NETWORK_IMAGE_CHANNEL_FIRST,
        range=ailia.NETWORK_IMAGE_RANGE_U_FP32,
        algorithm=ailia.DETECTOR_ALGORITHM_YOLOV3,
        env_id=args.env_id,
        memory_mode=args.memory_mode,
    )
    pose_net = ailia.Net(MODEL_POSE_PATH, WEIGHT_POSE_PATH, env_id=args.env_id, memory_mode=args.memory_mode)

    if not args.onnx:
        net = ailia.Net(MODEL_27FRAME_17JOINT_PATH, WEIGHT_27FRAME_17JOINT_PATH, env_id=args.env_id, memory_mode=args.memory_mode)
    else:
        import onnxruntime
        net = onnxruntime.InferenceSession(WEIGHT_27FRAME_17JOINT_PATH)

    info = {
        "yolo_model": detector,
        "pose_model": pose_net,
        "num_person": num_person,
    }
    recognize_from_video(net, info)


if __name__ == '__main__':
    main()
