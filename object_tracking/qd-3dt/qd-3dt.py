import sys
import os
import time
import random
import json
from collections import defaultdict

import numpy as np
import cv2
from matplotlib import pyplot as plt
from pyquaternion import Quaternion

import ailia

# import original modules
sys.path.append('../../util')
from arg_utils import get_base_parser, update_parser, get_savepath  # noqa
from model_utils import check_and_download_models  # noqa
from detector_utils import load_image  # noqa
from image_utils import normalize_image  # noqa
from webcamera_utils import get_capture, get_writer  # noqa
# logger
from logging import getLogger  # noqa

from video_utils import load_annotations
import tracking_utils as tu
from motion_tracker import MotionTracker
import tracker_model
from plot_tracking import read_coco, Visualizer, RandomColor, fig2data

logger = getLogger(__name__)

# ======================
# Parameters
# ======================

WEIGHT_DETECTOR_PATH = 'nuScenes_3Dtracking.onnx'
MODEL_DETECTOR_PATH = 'nuScenes_3Dtracking.onnx.prototxt'
WEIGHT_MOTION_PRED_PATH = 'nuScenes_LSTM_motion_pred.onnx'
MODEL_MOTION_PRED_PATH = 'nuScenes_LSTM_motion_pred.onnx.prototxt'
WEIGHT_MOTION_REFINE_PATH = 'nuScenes_LSTM_motion_refine.onnx'
MODEL_MOTION_REFINE_PATH = 'nuScenes_LSTM_motion_refine.onnx.prototxt'
REMOTE_PATH = 'https://storage.googleapis.com/ailia-models/qd-3dt/'

INPUT_PATH = 'data/nuscenes/anns/tracking_val_mini_v0.json'
SAVE_IMAGE_PATH = 'output'

IMAGE_MAX = 1600
IMAGE_MIN = 900

CLASSES = {
    'bicycle': 0, 'motorcycle': 1, 'pedestrian': 2,
    'bus': 3, 'car': 4, 'trailer': 5, 'truck': 6,
    'construction_vehicle': 7, 'traffic_cone': 8, 'barrier': 9
}

NUM_BBOX_HEAD_CLASS = 12

nusc_mapping = {
    'bicycle': 1,
    'motorcycle': 2,
    'pedestrian': 3,
    'bus': 4,
    'car': 5,
    'trailer': 6,
    'truck': 7,
    'construction_vehicle': 8,
    'traffic_cone': 9,
    'barrier': 10,
    'ignore': 11
}

# ======================
# Arguemnt Parser Config
# ======================

parser = get_base_parser(
    'Monocular Quasi-Dense 3D Object Tracking', INPUT_PATH, SAVE_IMAGE_PATH
)
parser.add_argument(
    '-i', '--input', metavar='FILE', default=INPUT_PATH,
    help='The default (model-dependent) input data (json) path. '
)
parser.add_argument(
    '-vid', '--video_id', metavar='VIDEO_ID', type=int, default=None,
    help='filter video id.'
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

def bbox2result(bboxes, labels, num_classes):
    """Convert detection results to a list of numpy arrays.

    Args:
        bboxes (Tensor): shape (n, 5)
        labels (Tensor): shape (n, )
        num_classes (int): class number, including background class

    Returns:
        list(ndarray): bbox results of each class
    """
    if bboxes.shape[0] == 0:
        return [
            np.zeros((0, 5), dtype=np.float32) for i in range(num_classes - 1)
        ]
    else:
        return [bboxes[labels == i, :] for i in range(num_classes - 1)]


def track2results(bboxes, labels, ids):
    outputs = defaultdict(list)
    for bbox, label, id in zip(bboxes, labels, ids):
        outputs[id] = dict(bbox=bbox, label=label)
    return outputs


def get_palette(n):
    palette = [0] * (n * 3)
    for j in range(0, n):
        lab = j
        palette[j * 3 + 0] = 0
        palette[j * 3 + 1] = 0
        palette[j * 3 + 2] = 0
        i = 0
        while lab:
            palette[j * 3 + 0] |= (((lab >> 0) & 1) << (7 - i))
            palette[j * 3 + 1] |= (((lab >> 1) & 1) << (7 - i))
            palette[j * 3 + 2] |= (((lab >> 2) & 1) << (7 - i))
            i += 1
            lab >>= 3
    return palette[3:]


def plt_3d_tracklets(
        img,
        track_results,
        track_depths,
        thickness=2,
        font_scale=0.4):
    color_black = (0, 0, 0)
    for indx, (id, items) in enumerate(track_results.items()):
        bbox = items['bbox']
        label = items['label']

        x1, y1, x2, y2, _ = bbox.astype(np.int32)
        random.seed(id)
        bbox_color = random.choice(plt_3d_tracklets.cm)
        bbox_color = bbox_color.tolist()[::-1]

        info_text = f'T{int(id):03d}'
        info_text += f'_{int(track_depths[indx]):03d}m'

        img[y1:y1 + 12, x1:x1 + 80, :] = bbox_color
        cv2.putText(
            img,
            info_text, (x1, y1 + 10),
            cv2.FONT_HERSHEY_COMPLEX,
            font_scale,
            color=color_black)

        cv2.rectangle(img, (x1, y1), (x2, y2), bbox_color, thickness=thickness)

        if bbox[-1] < 0:
            bbox[-1] = np.nan
        label_text = '{:.02f}'.format(bbox[-1])
        img[y1 - 12:y1, x1:x1 + 30, :] = bbox_color
        cv2.putText(
            img,
            label_text, (x1, y1 - 2),
            cv2.FONT_HERSHEY_COMPLEX,
            font_scale,
            color=color_black)

    return img


plt_3d_tracklets.cm = np.array(get_palette(256)).reshape(-1, 3)


def save_txt(
        outputs,
        img_info,
        use_3d_box_center=False,
        adjust_center=False):
    """
    #Values    Name      Description
    ----------------------------------------------------------------------
    1   frame       Frame within the sequence where the object appearers
    1   track id    Unique tracking id of this object within this sequence
    1   type        Describes the type of object: 'Car', 'Van', 'Truck',
                    'Pedestrian', 'Person_sitting', 'Cyclist', 'Tram',
                    'Misc' or 'DontCare'
    1   truncated   Float from 0 (non-truncated) to 1 (truncated), where
                    truncated refers to the object leaving image boundaries.
                    Truncation 2 indicates an ignored object (in particular
                    in the beginning or end of a track) introduced by manual
                    labeling.
    1   occluded    Integer (0,1,2,3) indicating occlusion state:
                    0 = fully visible, 1 = partly occluded
                    2 = largely occluded, 3 = unknown
    1   alpha       Observation angle of object, ranging [-pi..pi]
    4   bbox        2D bounding box of object in the image (0-based index):
                    contains left, top, right, bottom pixel coordinates
    3   dimensions  3D object dimensions: height, width, length (in meters)
    3   location    3D object location x,y,z in camera coordinates (in meters)
    1   rotation_y  Rotation ry around Y-axis in camera coordinates [-pi..pi]
    1   score       Only for results: Float, indicating confidence in
                    detection, needed for p/r curves, higher is better.
    """
    out_folder = get_savepath(args.savepath, 'txts', post_fix='')
    os.makedirs(out_folder, exist_ok=True)
    vid_name = img_info['vid_name']
    txt_file = os.path.join(out_folder, '{}.txt'.format(vid_name))

    # Expand dimension of results
    n_obj_detect = len(outputs['track_results'])
    if outputs.get('depth_results', None) is not None:
        depths = outputs['depth_results'].reshape(-1, 1)
    else:
        depths = np.full((n_obj_detect, 1), -1000)
    if outputs.get('dim_results', None) is not None:
        dims = outputs['dim_results'].reshape(-1, 3)
    else:
        dims = np.full((n_obj_detect, 3), -1000)
    if outputs.get('alpha_results', None) is not None:
        alphas = outputs['alpha_results'].reshape(-1, 1)
    else:
        alphas = np.full((n_obj_detect, 1), -10)

    if outputs.get('cen_2ds_results', None) is not None:
        centers = outputs['cen_2ds_results'].reshape(-1, 2)
    else:
        centers = [None] * n_obj_detect

    lines = []
    for (trackId, bbox), depth, dim, alpha, cen in zip(
            outputs['track_results'].items(),
            depths, dims, alphas, centers):
        loc, label = bbox['bbox'], bbox['label']
        if use_3d_box_center and cen is not None:
            box_cen = cen
        else:
            box_cen = np.array([loc[0] + loc[2], loc[1] + loc[3]]) / 2
        if alpha == -10:
            roty = np.full((1,), -10)
        else:
            roty = tu.alpha2rot_y(
                alpha,
                box_cen[0] - img_info['width'] / 2,
                img_info['cali'][0][0])
        if np.all(depths == -1000):
            trans = np.full((3,), -1000)
        else:
            trans = tu.imagetocamera(
                box_cen[None], depth,
                np.array(img_info['cali'])).flatten()

        if adjust_center:
            # KITTI GT uses the bottom of the car as center (x, 0, z).
            # Prediction uses center of the bbox as center (x, y, z).
            # So we align them to the bottom center as GT does
            trans[1] += dim[0] / 2.0

        cat = ''
        for key in CLASSES:
            if bbox['label'] == CLASSES[key]:
                cat = key.lower()
                break

        if cat == '':
            continue

        # Create lines of results
        line = f"{img_info['index']} {trackId} {cat} {-1} {-1} " \
               f"{alpha.item():.6f} " \
               f"{loc[0]:.6f} {loc[1]:.6f} {loc[2]:.6f} {loc[3]:.6f} " \
               f"{dim[0]:.6f} {dim[1]:.6f} {dim[2]:.6f} " \
               f"{trans[0]:.6f} {trans[1]:.6f} {trans[2]:.6f} " \
               f"{roty.item():.6f} {loc[4]:.6f}\n"
        lines.append(line)

    if txt_file in save_txt.writed:
        mode = 'a'
    else:
        mode = 'w'
        save_txt.writed.append(txt_file)
    if len(lines) > 0:
        with open(txt_file, mode) as f:
            f.writelines(lines)
    else:
        with open(txt_file, mode):
            pass


save_txt.writed = []


def general_output(
        coco_json, outputs, img_info,
        use_3d_box_center, pred_id,
        cats_mapping):
    if not ('categories' in coco_json.keys()):
        for k, v in cats_mapping.items():
            coco_json['categories'].append(dict(id=v, name=k))

    if img_info.get('is_key_frame') is not None and img_info['is_key_frame']:
        img_info['index'] = img_info['key_frame_index']

    img_info['id'] = len(coco_json['images'])
    vid_name = img_info['vid_name']
    if img_info['first_frame']:
        coco_json['videos'].append(
            dict(id=img_info['video_id'], name=vid_name))

    # pruning img_info
    img_info.pop('filename', None)
    img_info.pop('type', None)
    coco_json['images'].append(img_info)

    # Expand dimension of results
    n_obj_detect = len(outputs['track_results'])
    if outputs.get('depth_results', None) is not None:
        depths = outputs['depth_results'].reshape(-1, 1)
    else:
        depths = np.ones([n_obj_detect, 1]) * -1000
    if outputs.get('dim_results', None) is not None:
        dims = outputs['dim_results'].reshape(-1, 3)
    else:
        dims = np.ones([n_obj_detect, 3]) * -1000
    if outputs.get('alpha_results', None) is not None:
        alphas = outputs['alpha_results'].reshape(-1, 1)
    else:
        alphas = np.ones([n_obj_detect, 1]) * -10
    if outputs.get('cen_2ds_results', None) is not None:
        centers = outputs['cen_2ds_results'].reshape(-1, 2)
    else:
        centers = [None] * n_obj_detect
    if outputs.get('depth_uncertainty_results', None) is not None:
        depths_uncertainty = outputs['depth_uncertainty_results'].reshape(-1, 1)
    else:
        depths_uncertainty = [None] * n_obj_detect

    for (trackId, bbox), depth, dim, alpha, cen, depth_uncertainty, \
            in zip(outputs['track_results'].items(),
                   depths, dims, alphas, centers, depths_uncertainty):
        box = bbox['bbox'].astype(float).tolist()
        cat = ''

        for key in CLASSES:
            if bbox['label'] == CLASSES[key]:
                cat = key.lower()
                break

        if cat == '':
            continue

        x1 = box[0]
        y1 = box[1]
        x2 = box[2]
        y2 = box[3]
        score = box[4]
        if use_3d_box_center and cen is not None:
            box_cen = cen
        else:
            box_cen = np.array([x1 + x2, y1 + y2]) / 2
        if alpha == -10:
            rot_y = -10
        else:
            rot_y = tu.alpha2rot_y(alpha, box_cen[0] - img_info['width'] / 2,
                                   img_info['cali'][0][0])
        if np.all(depths == -1000):
            trans = np.ones([1, 3]) * -1000
        else:
            trans = tu.imagetocamera(box_cen[np.newaxis], depth,
                                     np.array(img_info['cali'])).flatten()
        ann = dict(
            id=pred_id,
            image_id=img_info['id'],
            category_id=cats_mapping[cat],
            instance_id=trackId.tolist(),
            alpha=float(alpha),
            roty=float(rot_y),
            dimension=dim.astype(float).tolist(),
            translation=trans.astype(float).tolist(),
            is_occluded=False,
            is_truncated=False,
            bbox=[x1, y1, x2 - x1, y2 - y1],
            area=(x2 - x1) * (y2 - y1),
            center_2d=box_cen.astype(float).tolist(),
            uncertainty=float(depth_uncertainty),
            depth=depth.tolist(),
            iscrowd=False,
            ignore=False,
            segmentation=[[x1, y1, x1, y2, x2, y2, x2, y1]],
            score=score)

        coco_json['annotations'].append(ann)
        pred_id += 1

    return coco_json, pred_id


# ======================
# Main functions
# ======================

def preprocess(img):
    h, w = img.shape[:2]
    divisor = 32

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # rescale
    scale_factor = min(IMAGE_MAX / max(h, w), IMAGE_MIN / min(h, w))
    ow, oh = int(w * scale_factor + 0.5), int(h * scale_factor + 0.5)
    if ow != w or oh != h:
        img = cv2.resize(img, (ow, oh), interpolation=cv2.INTER_LINEAR)

    img_shape = img.shape

    mean = np.array([106.07, 127.705, 128.08])
    std = np.array([73.458, 70.129, 70.463])
    img = (img - mean) / std

    pad_h = int(np.ceil(oh / divisor)) * divisor
    pad_w = int(np.ceil(ow / divisor)) * divisor
    if pad_w != ow or pad_h != oh:
        pad_img = np.zeros((pad_h, pad_w, 3), dtype=img.dtype)
        pad_img[:oh, :ow, ...] = img
        img = pad_img

    img = img.transpose(2, 0, 1)  # HWC -> CHW
    img = np.expand_dims(img, axis=0)
    img = img.astype(np.float32)

    return img, img_shape, scale_factor


def post_processing(
        img_info, det_bboxes, det_labels, embeds, det_depths,
        det_depths_uncertainty, det_dims, det_alphas, det_2dcs):
    bbox_results = bbox2result(det_bboxes, det_labels, NUM_BBOX_HEAD_CLASS)

    # lifting
    projection = np.array(img_info['calib'])
    position = np.array(img_info['pose']['position'])
    r_camera_to_world = tu.angle2rot(np.array(img_info['pose']['rotation']))
    rotation = np.array(r_camera_to_world)
    cam_rot_quat = Quaternion(matrix=r_camera_to_world)

    corners = tu.imagetocamera(det_2dcs, det_depths, projection)
    corners_global = tu.cameratoworld(corners, position, rotation)
    det_yaws = tu.alpha2yaw(
        det_alphas, corners[:, 0:1], corners[:, 2:3])

    quat_det_yaws_world = {'roll_pitch': [], 'yaw_world': []}
    for det_yaw in det_yaws:
        yaw_quat = Quaternion(
            axis=[0, 1, 0], radians=det_yaw)
        rotation_world = cam_rot_quat * yaw_quat
        if rotation_world.z < 0:
            rotation_world *= -1
        roll_world, pitch_world, yaw_world = tu.quaternion_to_euler(
            rotation_world.w, rotation_world.x, rotation_world.y,
            rotation_world.z)
        quat_det_yaws_world['roll_pitch'].append(
            [roll_world, pitch_world])
        quat_det_yaws_world['yaw_world'].append(yaw_world)

    det_yaws_world = np.array(quat_det_yaws_world['yaw_world'])[:, None]
    det_boxes_3d = np.concatenate([
        corners_global, det_yaws_world, det_dims
    ], axis=1)

    # tracking
    frame_ind = img_info.get('index', -1)
    pure_det = False
    match_bboxes, match_labels, match_boxes_3ds, ids, inds, valids = \
        predict.tracker.match(
            bboxes=det_bboxes,
            labels=det_labels,
            boxes_3d=det_boxes_3d,
            depth_uncertainty=det_depths_uncertainty,
            position=position,
            rotation=rotation,
            embeds=embeds,
            cur_frame=frame_ind,
            pure_det=pure_det)

    # reproject
    match_dims = match_boxes_3ds[:, -3:]
    match_corners_cam = tu.worldtocamera(
        match_boxes_3ds[:, :3], position, rotation)
    match_depths = match_corners_cam[:, 2:3]

    match_yaws = []
    for match_order, match_yaw in zip(
            inds[valids], match_boxes_3ds[:, 3]):
        roll_world, pitch_world = quat_det_yaws_world['roll_pitch'][
            match_order]
        rotation_cam = cam_rot_quat.inverse * Quaternion(
            tu.euler_to_quaternion(roll_world, pitch_world, match_yaw))
        vtrans = np.dot(rotation_cam.rotation_matrix,
                        np.array([1, 0, 0]))
        match_yaws.append(-np.arctan2(vtrans[2], vtrans[0]))

    match_yaws = np.expand_dims(np.array(match_yaws), axis=1)
    match_alphas = tu.yaw2alpha(
        match_yaws,
        match_corners_cam[:, 0:1],
        match_corners_cam[:, 2:3])
    match_corners_frm = tu.cameratoimage(
        match_corners_cam, projection)
    match_2dcs = match_corners_frm

    # parse tracking results
    track_inds = ids > -1
    track_bboxes = match_bboxes[track_inds]
    track_labels = match_labels[track_inds]
    if match_depths is not None:
        track_depths = match_depths[track_inds]
    else:
        track_depths = None
    if match_dims is not None:
        track_dims = match_dims[track_inds]
    else:
        track_dims = None
    if match_alphas is not None:
        track_alphas = match_alphas[track_inds]
    else:
        track_alphas = None
    if match_2dcs is not None:
        track_2dcs = match_2dcs[track_inds]
    else:
        track_2dcs = None
    track_ids = ids[track_inds]
    track_results = track2results(track_bboxes, track_labels, track_ids)

    outputs = dict(
        bbox_results=bbox_results,
        depth_results=track_depths,
        depth_uncertainty_results=det_depths_uncertainty,
        dim_results=track_dims,
        alpha_results=track_alphas,
        cen_2ds_results=track_2dcs,
        track_results=track_results)

    return outputs


def predict(net_det, lstm_pred, lstm_refine, img, img_info):
    img, img_shape, scale_factor = preprocess(img)

    track_config = {
        'lstm_pred': lstm_pred, 'lstm_refine': lstm_refine,
        'init_score_thr': 0.8, 'init_track_id': 0, 'obj_score_thr': 0.5, 'match_score_thr': 0.5,
        'memo_tracklet_frames': 10, 'memo_backdrop_frames': 1, 'memo_momentum': 0.8,
        'motion_momentum': 0.9, 'nms_conf_thr': 0.5, 'nms_backdrop_iou_thr': 0.3,
        'nms_class_iou_thr': 0.7, 'loc_dim': 7, 'with_deep_feat': True, 'with_cats': True,
        'with_bbox_iou': True, 'with_depth_ordering': True,
        'track_bbox_iou': 'box3d', 'depth_match_metric': 'motion', 'match_metric': 'cycle_softmax',
        'match_algo': 'greedy', 'with_depth_uncertainty': True
    }
    if predict.tracker is None:
        predict.tracker = MotionTracker(**track_config)
    elif img_info.get('first_frame', False):
        num_tracklets = predict.tracker.num_tracklets
        track_config['init_track_id'] = num_tracklets
        del predict.tracker
        predict.tracker = MotionTracker(**track_config)

    calib = img_info['cali']
    ori_shape = (img_info['height'], img_info['width'], 3)
    if ori_shape != img_shape:
        focal_length = calib[0][0]
        width = img_shape[1]
        height = img_shape[0]
        calib = [[focal_length * scale_factor, 0, width / 2.0, 0],
                 [0, focal_length * scale_factor, height / 2.0, 0],
                 [0, 0, 1, 0]]

    img_info['calib'] = calib
    img_shape = np.array(img_shape, dtype=np.int64)
    scale_factor = np.array(scale_factor, dtype=np.float32)

    # feedforward
    if not args.onnx:
        output = net_det.predict([img, img_shape, scale_factor])
    else:
        output = net_det.run(None, {'img': img, 'img_shape': img_shape, 'scale_factor': scale_factor})

    det_bboxes, det_labels, embeds, det_depths, det_depths_uncertainty, det_dims, det_alphas, det_2dcs = output

    output = post_processing(
        img_info, det_bboxes, det_labels, embeds, det_depths,
        det_depths_uncertainty, det_dims, det_alphas, det_2dcs)

    return output


predict.tracker = None


def recognize_from_image(net_det, lstm_pred, lstm_ref):
    img_infos = load_annotations(args.input[0])
    if args.video_id is not None:
        img_infos = [x for x in img_infos if x['video_id'] == args.video_id]

    logger.info('VIDEO_ID: %s' % ', '.join([
        str(x) for x in sorted(set(x['video_id'] for x in img_infos))
    ]))

    coco_outputs = defaultdict(list)
    pred_id = 0

    # image loop
    for i, img_info in enumerate(img_infos):
        image_path = img_info['filename']
        logger.info('%s (%d/%d)' % (image_path, i + 1, len(img_infos)))

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
                output = predict(net_det, lstm_pred, lstm_ref, img, img_info)
                end = int(round(time.time() * 1000))
                estimation_time = (end - start)

                # Loggin
                logger.info(f'\tailia processing estimation time {estimation_time} ms')
                if i != 0:
                    total_time_estimation = total_time_estimation + estimation_time

            logger.info(f'\taverage time estimation {total_time_estimation / (args.benchmark_count - 1)} ms')
        else:
            output = predict(net_det, lstm_pred, lstm_ref, img, img_info)

        # save image
        img = plt_3d_tracklets(
            img,
            output['track_results'],
            output['depth_results'])
        out_folder = get_savepath(args.savepath, 'shows', post_fix='')
        out_folder = os.path.join(out_folder, img_info['vid_name'])
        os.makedirs(out_folder, exist_ok=True)
        img_name = os.path.basename(img_info['file_name'])
        save_file = os.path.join(out_folder, img_name.split('-')[-1])
        logger.info(f'saved at : {save_file}')
        cv2.imwrite(save_file, img)

        # save text
        use_3d_center = True
        is_kitti = False
        save_txt(
            output,
            img_info,
            use_3d_box_center=use_3d_center,
            adjust_center=is_kitti)

        coco_outputs, pred_id = general_output(
            coco_outputs, output, img_info,
            use_3d_center, pred_id,
            nusc_mapping)

    out_json = get_savepath(args.savepath, 'output.json', post_fix='')
    logger.info(f'saved at : {out_json}')
    with open(out_json, 'w') as fp:
        json.dump(coco_outputs, fp)

    # Visualize
    info_pd = read_coco(
        coco_outputs,
        category=['Bicycle', 'Motorcycle', 'Pedestrian', 'Bus', 'Car', 'Trailer', 'Truck'])
    vis = Visualizer(res_folder=args.savepath)

    logger.info('Ploting 3D boxes and BEV.')
    vis.save_vid(info_pd)

    logger.info('Script finished successfully.')


def recognize_from_video(net_det, lstm_pred, lstm_ref):
    video_file = args.video if args.video else args.input[0]
    capture = get_capture(video_file)
    assert capture.isOpened(), 'Cannot capture source'

    # create video writer if savepath is specified as video format
    if args.savepath != SAVE_IMAGE_PATH:
        f_h = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        f_w = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)) + f_h
        writer = get_writer(args.savepath, f_h, f_w)
    else:
        writer = None

    coco_outputs = defaultdict(list)
    pred_id = 0
    first_frame = True

    # Visualize
    id_to_color = {}
    loc_world_hist_pd = {}
    cmap = RandomColor(100)
    vis = Visualizer(res_folder=None)

    i = 0
    frame_shown = False
    while True:
        ret, frame = capture.read()
        if (cv2.waitKey(1) & 0xFF == ord('q')) or not ret:
            break
        if frame_shown and cv2.getWindowProperty('frame', cv2.WND_PROP_VISIBLE) == 0:
            break

        h, w = frame.shape[:2]
        img_info = {
            'height': h, 'width': w,
            'cali': [
                [1252.8131103515625, 0.0, 826.588134765625, 0.0],
                [0.0, 1252.8131103515625, 469.9846496582031, 0.0],
                [0.0, 0.0, 1.0, 0.0],
            ],
            'pose': {
                'rotation': [0., 0., 0.],
                'position': [0., 0., 0.],
            },
            'file_name': None,
            'vid_name': '',
            'video_id': 0,
            'index': i,
            'first_frame': first_frame,
        }
        output = predict(net_det, lstm_pred, lstm_ref, frame, img_info)

        use_3d_center = True
        coco_outputs, pred_id = general_output(
            coco_outputs, output, img_info,
            use_3d_center, pred_id,
            nusc_mapping)
        info_pd = read_coco(
            coco_outputs,
            category=['Bicycle', 'Motorcycle', 'Pedestrian', 'Bus', 'Car', 'Trailer', 'Truck'])

        pd_objects = info_pd[0]['frames'][i]
        img1 = vis.draw_annos(
            frame, pd_objects,
            cmap=cmap,
            id_to_color=id_to_color,
            loc_world_hist_pd=loc_world_hist_pd)
        vis.draw_bev_canvas()
        img2 = fig2data(vis.fig)
        plt.cla()

        img2 = cv2.resize(img2, (img1.shape[0], img1.shape[0]))
        out_frame = np.hstack([img1, img2])

        # show
        cv2.imshow('frame', out_frame)
        frame_shown = True

        # save results
        if writer is not None:
            writer.write(out_frame)

        coco_outputs["images"].clear()
        coco_outputs["annotations"].clear()

        first_frame = False
        i += 1

    capture.release()
    cv2.destroyAllWindows()
    if writer is not None:
        writer.release()

    logger.info('Script finished successfully.')


def main():
    # model files check and download
    logger.info('Checking DETECTOR model...')
    check_and_download_models(WEIGHT_DETECTOR_PATH, MODEL_DETECTOR_PATH, REMOTE_PATH)
    logger.info('Checking MOTION_PRED model...')
    check_and_download_models(WEIGHT_MOTION_PRED_PATH, MODEL_MOTION_PRED_PATH, REMOTE_PATH)
    logger.info('Checking MOTION_REFINE model...')
    check_and_download_models(WEIGHT_MOTION_REFINE_PATH, MODEL_MOTION_REFINE_PATH, REMOTE_PATH)

    env_id = args.env_id

    # initialize
    if not args.onnx:
        net_det = ailia.Net(MODEL_DETECTOR_PATH, WEIGHT_DETECTOR_PATH, env_id=env_id, memory_mode=args.memory_mode)
        lstm_pred = ailia.Net(MODEL_MOTION_PRED_PATH, WEIGHT_MOTION_PRED_PATH, env_id=env_id, memory_mode=args.memory_mode)
        lstm_ref = ailia.Net(MODEL_MOTION_REFINE_PATH, WEIGHT_MOTION_REFINE_PATH, env_id=env_id, memory_mode=args.memory_mode)
    else:
        import onnxruntime
        net_det = onnxruntime.InferenceSession(WEIGHT_DETECTOR_PATH)
        lstm_pred = onnxruntime.InferenceSession(WEIGHT_MOTION_PRED_PATH)
        lstm_ref = onnxruntime.InferenceSession(WEIGHT_MOTION_REFINE_PATH)
        tracker_model.onnx = True

    if args.video is not None:
        recognize_from_video(net_det, lstm_pred, lstm_ref)
    else:
        recognize_from_image(net_det, lstm_pred, lstm_ref)


if __name__ == '__main__':
    main()
