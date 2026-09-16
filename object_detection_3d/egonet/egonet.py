import sys
import os
import time
from io import BytesIO

import numpy as np
import cv2
from PIL import Image
import matplotlib.pyplot as plt

import ailia

# import original modules
sys.path.append('../../util')
from arg_utils import get_base_parser, update_parser, get_savepath  # noqa
from model_utils import check_and_download_models  # noqa
from detector_utils import load_image, letterbox_convert, reverse_letterbox  # noqa
from image_utils import normalize_image  # noqa
from webcamera_utils import get_capture, get_writer  # noqa
# logger
from logging import getLogger  # noqa

from egonet_utils import kpts2cs, cs2bbox
from egonet_utils import modify_bbox, get_affine_transform, affine_transform_modified
from egonet_utils import get_observation_angle_trans, get_observation_angle_proj
from egonet_utils import get_6d_rep
from egonet_utils import plot_2d_objects, plot_3d_objects, save_result_json
from instance_utils import csv_read_annot, get_2d_3d_pair

logger = getLogger(__name__)

# ======================
# Parameters
# ======================

WEIGHT_HC_PATH = 'HC.onnx'
MODEL_HC_PATH = 'HC.onnx.prototxt'
WEIGHT_L_PATH = 'L.onnx'
MODEL_L_PATH = 'L.onnx.prototxt'
REMOTE_PATH = 'https://storage.googleapis.com/ailia-models/egonet/'

WEIGHT_YOLOX_PATH = 'yolox_s.opt.onnx'
MODEL_YOLOX_PATH = 'yolox_s.opt.onnx.prototxt'
REMOTE_YOLOX_PATH = 'https://storage.googleapis.com/ailia-models/yolox/'

LS_path = 'LS.npy'

IMAGE_PATH = '007161.png'
SAVE_IMAGE_PATH = 'output.png'

IMAGE_SIZE = 256
IMAGE_YOLO_SIZE = 640

DATASET_WIDTH = 1238
DATASET_HEIGHT = 374

THRESHOLD = 0.4
IOU = 0.45

# ======================
# Arguemnt Parser Config
# ======================

parser = get_base_parser(
    'EgoNet', IMAGE_PATH, SAVE_IMAGE_PATH
)
parser.add_argument(
    '--label_path', type=str, default=None,
    help='the label file (object labels for image) or stored directory path.'
)
parser.add_argument(
    '--gt_label_path', type=str, default=None,
    help='the ground truth label file (ground truth object labels for image) or stored directory path.'
)
parser.add_argument(
    '--calib_path', type=str, default=None,
    help='the calibration file (Camera parameters for image) or stored directory path'
)
parser.add_argument(
    '--detector', action='store_true',
    help='Use object detection.'
)
parser.add_argument(
    '-th', '--threshold',
    default=THRESHOLD, type=float,
    help='object confidence threshold'
)
parser.add_argument(
    '-iou', '--iou',
    default=IOU, type=float,
    help='IOU threshold for NMS'
)
parser.add_argument(
    '--plot_3d', action='store_true',
    help='draw a 3d plot.'
)
parser.add_argument(
    '-w', '--write_json',
    action='store_true',
    help='Flag to output results to json file.'
)
args = update_parser(parser)


# ======================
# Secondaty Functions
# ======================

def get_path(path, file_typ, name):
    if path is None:
        file_path = "%s/%s.txt" % (file_typ, name)
        if os.path.exists(file_path):
            logger.info("%s file: %s" % (file_typ, file_path))
            return file_path
        else:
            return None
    elif os.path.isdir(path):
        file_path = "%s/%s.txt" % (path, name)
        if os.path.exists(file_path):
            logger.info("%s file: %s" % (file_typ, file_path))
            return file_path
    elif os.path.exists(path):
        logger.info("%s file: %s" % (file_typ, path))
        return path

    logger.error("%s file is not found. (path: %s)" % (file_typ, path))
    sys.exit(-1)


def read_annot(
        img,
        label_path,
        calib_path,
        pred=False,
        add_gt=False,
        use_raw_bbox=True,
        enlarge=None):
    if calib_path is not None:
        list_2d, list_3d, list_id, pv, K, anns, raw_bboxes = \
            get_2d_3d_pair(
                img,
                label_path, calib_path,
                pred=pred,
                augment=False, add_raw_bbox=True)

        all_keypoints_2d = np.array([])
        all_keypoints_3d = np.array([])
        if len(list_2d) != 0:
            for idx, kpts in enumerate(list_2d):
                list_2d[idx] = kpts.reshape(1, -1, 3)
                list_3d[idx] = list_3d[idx].reshape(1, -1, 3)
            all_keypoints_2d = np.concatenate(list_2d, axis=0)
            all_keypoints_3d = np.concatenate(list_3d, axis=0)

            # compute 2D bounding box based on the projected 3D boxes
            bboxes_kpt = []
            for idx, keypoints in enumerate(all_keypoints_2d):
                # relatively tight bounding box: use enlarge = 1.0
                # delete invisible instances
                center, crop_size, _, _ = kpts2cs(
                    keypoints[:, :2], enlarge=1.01)
                bbox = np.array(cs2bbox(center, crop_size))
                bboxes_kpt.append(np.array(bbox).reshape(1, 4))

        bboxes = np.array([])
        if use_raw_bbox:
            bboxes = np.vstack(raw_bboxes)
        elif len(bboxes_kpt) != 0:
            bboxes = np.vstack(bboxes_kpt)
    else:
        anns = csv_read_annot(label_path, pred=pred)
        bboxes = []
        for i, a in enumerate(anns):
            bboxes.append(np.array(a["bbox"]).reshape(1, 4))
        bboxes = np.vstack(bboxes)

    d = {
        'bbox_2d': bboxes,
        'raw_anns': anns
    }
    if calib_path is not None:
        d['kpts_3d'] = all_keypoints_3d
        d['K'] = K
        if add_gt:
            pvs = np.vstack(pv) if len(pv) != 0 else []
            d['pose_vecs_gt'] = pvs
            d['kpts_2d_gt'] = all_keypoints_2d
            d['kpts_3d_gt'] = all_keypoints_3d

    if enlarge is not None:
        target_ar = 1.
        for i in range(len(bboxes)):
            bboxes[i] = modify_bbox(
                bboxes[i],
                target_ar=target_ar,
                enlarge=enlarge
            )['bbox']

    if pred:
        thres = args.threshold
        indices = [i for i in range(len(anns)) if anns[i]['score'] >= thres]
        d["bbox_2d"] = d["bbox_2d"][indices]
        d["kpts_3d"] = d["kpts_3d"][indices]
        if add_gt:
            d['pose_vecs_gt'] = d['pose_vecs_gt'][indices]
            d['kpts_2d_gt'] = d['kpts_2d_gt'][indices]
            d['kpts_3d_gt'] = d['kpts_3d_gt'][indices]

    return d


def detect_cars(img, enlarge=None):
    thres = args.threshold
    iou = args.threshold

    h, w, _ = img.shape
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    detect_cars.net.compute(img, thres, iou)
    count = detect_cars.net.get_object_count()
    detect_object = []
    for idx in range(count):
        obj = detect_cars.net.get_object(idx)
        detect_object.append(obj)

    bboxes = []
    car_class = [2, 5, 6]  # car bus truck
    for d in detect_object:
        if not (d.category in car_class):
            continue

        xmin = d.x * w
        ymin = d.y * h
        xmax = xmin + d.w * w
        ymax = ymin + d.h * h
        bboxes.append(np.array([xmin, ymin, xmax, ymax]).reshape(1, 4))

    if bboxes:
        bboxes = np.vstack(bboxes)
    else:
        bboxes = np.array([])

    d = {
        'bbox_2d': bboxes,
    }
    if enlarge is not None:
        target_ar = 1.
        for i in range(len(bboxes)):
            bboxes[i] = modify_bbox(
                bboxes[i],
                target_ar=target_ar,
                enlarge=enlarge
            )['bbox']

    return d


def normalize_1d(data, mean, std, individual=False):
    """
    Normalizes 1D data with mean and standard deviation.

    data: dictionary where values are
    mean: np vector with the mean of the data
    std: np vector with the standard deviation of the data
    individual: whether to perform normalization independently for each input

    Returns
    data_out: normalized data
    """
    if individual:
        # this representation has the implicit assumption that the representation
        # is translational and scaling invariant
        num_data = len(data)
        data = data.reshape(num_data, -1, 2)
        mean_x = np.mean(data[:, :, 0], axis=1).reshape(num_data, 1)
        std_x = np.std(data[:, :, 0], axis=1)
        mean_y = np.mean(data[:, :, 1], axis=1).reshape(num_data, 1)
        std_y = np.std(data[:, :, 1], axis=1)
        denominator = (0.5 * (std_x + std_y)).reshape(num_data, 1)
        data[:, :, 0] = (data[:, :, 0] - mean_x) / denominator
        data[:, :, 1] = (data[:, :, 1] - mean_y) / denominator
        data_out = data.reshape(num_data, -1)
    else:
        data_out = (data - mean) / std

    return data_out


def unnormalize_1d(normalized_data, mean, std):
    orig_data = normalized_data * std + mean
    return orig_data


# ======================
# Main functions
# ======================

def crop_single_instance(
        img,
        bbox,
        resolution):
    """
    Crop a single instance given an image and bounding box.
    """
    width, height = resolution
    target_ar = height / width
    ret = modify_bbox(bbox, target_ar)
    c, s, r = ret['c'], ret['s'], 0.

    # xy_dict: parameters for adding xy coordinate maps
    trans = get_affine_transform(c, s, r, (height, width))
    instance = cv2.warpAffine(
        img,
        trans,
        (int(resolution[0]), int(resolution[1])),
        flags=cv2.INTER_LINEAR
    )

    return instance


def crop_instances(img, annot_dict):
    resolution = [IMAGE_SIZE, IMAGE_SIZE]
    target_ar = resolution[1] / resolution[0]

    all_instances = []
    all_records = []
    boxes = annot_dict['bbox_2d']
    for idx, bbox in enumerate(boxes):
        # crop an instance with required aspect ratio
        instance = crop_single_instance(
            img, bbox, resolution)
        instance = normalize_image(instance, normalize_type='ImageNet')
        instance = instance.transpose(2, 0, 1)  # HWC -> CHW
        instance = np.expand_dims(instance, axis=0)
        instance = instance.astype(np.float32)
        all_instances.append(instance)

        ret = modify_bbox(bbox, target_ar)
        c, s, r = ret['c'], ret['s'], 0.
        all_records.append({
            'center': c,
            'scale': s,
            'rotation': r,
            'bbox': bbox,
            'bbox_resize': ret['bbox'],
        })

    all_instances = np.concatenate(all_instances, axis=0)

    return all_instances, all_records


def add_orientation_arrow(record):
    """
    Generate an arrow for each predicted orientation for visualization.
    """
    pred_kpts = record['kpts_3d_pred']
    gt_kpts = record['kpts_3d_gt']
    K = record['K']

    arrow_2d = np.zeros((len(pred_kpts), 2, 2))
    for idx in range(len(pred_kpts)):
        vector_3d = (pred_kpts[idx][1] - pred_kpts[idx][5])
        arrow_3d = np.concatenate([
            gt_kpts[idx][0].reshape(3, 1),
            (gt_kpts[idx][0] + vector_3d).reshape(3, 1)
        ], axis=1)
        projected = K @ arrow_3d
        arrow_2d[idx][0] = projected[0, :] / projected[2, :]
        arrow_2d[idx][1] = projected[1, :] / projected[2, :]

        # fix the arrow length if not fore-shortened
        vector_2d = arrow_2d[idx][:, 1] - arrow_2d[idx][:, 0]
        length = np.linalg.norm(vector_2d)
        if length > 50:
            vector_2d = vector_2d / length * 60
        arrow_2d[idx][:, 1] = arrow_2d[idx][:, 0] + vector_2d

    return arrow_2d


def gather_lifting_results(
        record,
        alpha_mode='trans'):
    """
    Convert network outputs to pose angles.
    """
    # prepare the prediction strings for submission
    # compute the roll, pitch and yaw angle of the predicted bounding box
    record['euler_angles'], record['translation'] = \
        get_6d_rep(record['kpts_3d_pred'])

    if alpha_mode == 'trans':
        record['alphas'] = get_observation_angle_trans(
            record['euler_angles'],
            record['translation']
        )
    elif alpha_mode == 'proj':
        record['alphas'] = get_observation_angle_proj(
            record['euler_angles'],
            record['kpts_2d_pred'],
            record['K']
        )
    else:
        raise NotImplementedError

    return record


def predict(HC, L, LS, img, annot_dict):
    if len(annot_dict['bbox_2d']) == 0:
        records = {
            'kpts_2d_pred': []
        }
        return records

    instances, records = crop_instances(img, annot_dict)

    # feedforward
    output = HC.predict([instances])

    _, local_coord = output

    # local part coordinates
    resolution = [IMAGE_SIZE, IMAGE_SIZE]
    width, height = resolution
    local_coord *= np.array(resolution).reshape(1, 1, 2)

    # transform local part coordinates to screen coordinates
    centers = [x['center'] for x in records]
    scales = [x['scale'] for x in records]
    rots = [x['rotation'] for x in records]
    for i in range(len(local_coord)):
        trans_inv = get_affine_transform(
            centers[i],
            scales[i],
            rots[i],
            (height, width),
            inv=1)
        screen_coord = affine_transform_modified(
            local_coord[i],
            trans_inv)
        records[i]['kpts_2d_pred'] = screen_coord

    # assemble a dictionary where each key
    records = {
        'bbox_resize': [x['bbox_resize'] for x in records],  # resized bounding box
        'kpts_2d_pred': [x['kpts_2d_pred'].reshape(1, -1) for x in records],
    }
    if 'kpts_3d' in annot_dict:
        records['kpts_3d'] = annot_dict['kpts_3d']
    if 'raw_anns' in annot_dict:
        records['raw_anns'] = annot_dict['raw_anns']

    # lift_2d_to_3d
    data = np.concatenate(records['kpts_2d_pred'], axis=0)
    data = normalize_1d(data, LS['mean_in'], LS['std_in'])
    data = data.astype(np.float32)
    output = L.predict([data])
    prediction = output[0]
    prediction = unnormalize_1d(
        prediction,
        LS['mean_out'],
        LS['std_out']
    )
    records['kpts_3d_pred'] = prediction.reshape(len(prediction), -1, 3)

    if 'pose_vecs_gt' in annot_dict:
        records['pose_vecs_gt'] = annot_dict['pose_vecs_gt']
    if 'kpts_2d_gt' in annot_dict:
        records['kpts_2d_gt'] = annot_dict['kpts_2d_gt']
    if 'kpts_3d_gt' in annot_dict and 'K' in annot_dict:
        records['kpts_3d_gt'] = annot_dict['kpts_3d_gt']
        records['K'] = annot_dict['K']
        records['arrow'] = add_orientation_arrow(records)

    # refine and gather the prediction strings
    records = gather_lifting_results(records)

    return records


def crop_center(img):
    scale_x = (DATASET_WIDTH / img.shape[1])
    crop_y = img.shape[0] * scale_x - DATASET_HEIGHT
    crop_y = int(crop_y / scale_x)  # bottom
    crop_y = int(crop_y / 2)  # center
    img = img[crop_y:, :, :]  # keep aspect
    img = cv2.resize(img, (DATASET_WIDTH, DATASET_HEIGHT), interpolation=cv2.INTER_LINEAR)
    return img


def recognize_from_image(HC, LS, L):
    label_path = args.label_path
    calib_path = args.calib_path
    gt_label_path = args.gt_label_path
    plot_3d = args.plot_3d
    detection = args.detector

    # input image loop
    for image_path in args.input:
        logger.info(image_path)

        # prepare input data
        img = load_image(image_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2RGB)

        name = os.path.splitext(os.path.basename(image_path))[0]
        if not detection:
            label_file_path = get_path(label_path, "label", name)
        calib_file_path = get_path(calib_path, "calib", name)
        gt_label_file_path = get_path(gt_label_path, "gt_label", name)

        if gt_label_file_path and not calib_file_path:
            logger.error("calib file not specified or not found.")
            sys.exit(-1)

        enlarge = 1.2
        if detection:
            annot_dict = detect_cars(img, enlarge=enlarge)
        elif label_file_path is not None:
            annot_dict = read_annot(
                img, label_file_path, calib_file_path,
                pred=True,
                enlarge=enlarge)
        else:
            logger.error("should specify the label file or detector.")
            sys.exit(-1)

        if gt_label_file_path:
            gt_annot_dict = read_annot(
                img, gt_label_file_path, calib_file_path,
                add_gt=True)
        else:
            gt_annot_dict = None

        # inference
        logger.info('Start inference...')
        if args.benchmark:
            logger.info('BENCHMARK mode')
            total_time_estimation = 0
            for i in range(args.benchmark_count):
                start = int(round(time.time() * 1000))
                record = predict(HC, L, LS, img, annot_dict)
                end = int(round(time.time() * 1000))
                estimation_time = (end - start)

                # Loggin
                logger.info(f'\tailia processing estimation time {estimation_time} ms')
                if i != 0:
                    total_time_estimation = total_time_estimation + estimation_time

            logger.info(f'\taverage time estimation {total_time_estimation / (args.benchmark_count - 1)} ms')
        else:
            record = predict(HC, L, LS, img, annot_dict)

        # inference with gt
        if gt_annot_dict:
            gt_record = predict(HC, L, LS, img, gt_annot_dict)
        else:
            gt_record = None

        # plot 2D predictions
        color_dict = {
            'bbox_2d': 'r',
            'kpts': 'rx',
        }
        fig, ax = plot_2d_objects(img, record, draw_bbox=(not gt_record), color_dict=color_dict)
        if gt_record:
            color_dict = {
                'bbox_2d': 'y',
                'kpts': 'yx'
            }
            plot_2d_objects(img, gt_record, color_dict=color_dict, ax=ax)

        save_path = get_savepath(args.savepath, image_path, ext='.png')
        logger.info(f'saved at : {save_path}')
        fig.savefig(save_path, dpi=100, bbox_inches='tight', pad_inches=0)
        plt.close()

        if args.write_json:
            json_file = '%s.json' % save_path.rsplit('.', 1)[0]
            save_result_json(json_file, record)

        if plot_3d:
            all_kpts_3d_pred = record['kpts_3d_pred'].reshape(len(record['kpts_3d_pred']), -1)
            fig, ax = plot_3d_objects(
                all_kpts_3d_pred,
                None,
                None,
                record,
                color='r',
            )
            if gt_record:
                all_kpts_3d_pred = gt_record['kpts_3d_pred'].reshape(len(record['kpts_3d_pred']), -1)
                all_kpts_3d_gt = gt_record['kpts_3d_gt']
                all_pose_vecs_gt = gt_record['pose_vecs_gt']
                _, ax = plot_3d_objects(
                    all_kpts_3d_pred,
                    all_kpts_3d_gt,
                    all_pose_vecs_gt,
                    gt_record,
                    color='y',
                    ax=ax
                )

            ex = os.path.splitext(save_path)
            save_path = '%s_3d%s' % ex
            logger.info(f'saved at : {save_path}')
            fig.savefig(save_path, dpi=100, bbox_inches='tight', pad_inches=0)
            plt.show()

    logger.info('Script finished successfully.')


def recognize_from_video(HC, LS, L):
    video_file = args.video if args.video else args.input[0]
    capture = get_capture(video_file)
    assert capture.isOpened(), 'Cannot capture source'

    # create video writer if savepath is specified as video format
    if args.savepath != SAVE_IMAGE_PATH:
        f_h = DATASET_HEIGHT  # int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        f_w = DATASET_WIDTH  # int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))

        # Draw dummy data to get the size of the output
        dummy = np.zeros((f_h, f_w, 3))
        fig, ax = plot_2d_objects(dummy, {'kpts_2d_pred': []}, {})
        buf = BytesIO()
        fig.savefig(buf, format='png', bbox_inches='tight', pad_inches=0)
        plt.close()
        out_w, out_h = Image.open(buf).size

        writer = get_writer(args.savepath, out_h, out_w)
    else:
        writer = None

    color_dict = {
        'bbox_2d': 'r',
        'kpts': 'rx',
    }

    frame_shown = False
    while True:
        ret, frame = capture.read()
        if (cv2.waitKey(1) & 0xFF == ord('q')) or not ret:
            break
        if frame_shown and cv2.getWindowProperty('frame', cv2.WND_PROP_VISIBLE) == 0:
            break

        img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = crop_center(img)

        enlarge = 1.2
        annot_dict = detect_cars(img, enlarge=enlarge)

        # inference
        record = predict(HC, L, LS, img, annot_dict)

        # plot 2D predictions
        fig, ax = plot_2d_objects(img, record, draw_bbox=True, color_dict=color_dict)
        buf = BytesIO()
        fig.savefig(buf, format='png', bbox_inches='tight', pad_inches=0)
        plt.close()
        res_img = np.array(Image.open(buf))
        res_img = cv2.cvtColor(res_img, cv2.COLOR_RGB2BGR)

        # show
        cv2.imshow('frame', res_img)
        frame_shown = True

        # save results
        if writer is not None:
            writer.write(res_img)

    capture.release()
    cv2.destroyAllWindows()
    if writer is not None:
        writer.release()

    logger.info('Script finished successfully.')


def main():
    # model files check and download
    logger.info('Checking HC model...')
    check_and_download_models(WEIGHT_HC_PATH, MODEL_HC_PATH, REMOTE_PATH)
    logger.info('Checking L model...')
    check_and_download_models(WEIGHT_L_PATH, MODEL_L_PATH, REMOTE_PATH)

    if args.video is not None:
        args.detector = True
    if args.detector:
        logger.info('Checking detector model...')
        check_and_download_models(WEIGHT_YOLOX_PATH, MODEL_YOLOX_PATH, REMOTE_YOLOX_PATH)

    env_id = args.env_id

    # initialize
    HC = ailia.Net(MODEL_HC_PATH, WEIGHT_HC_PATH, env_id=env_id, memory_mode=args.memory_mode)
    L = ailia.Net(MODEL_L_PATH, WEIGHT_L_PATH, env_id=env_id, memory_mode=args.memory_mode)

    # the statistics used by the lifter for normalizing inputs
    LS = np.load(LS_path, allow_pickle=True).item()

    if args.detector:
        detect_cars.net = ailia.Detector(
            MODEL_YOLOX_PATH,
            WEIGHT_YOLOX_PATH,
            80,
            format=ailia.NETWORK_IMAGE_FORMAT_BGR,
            channel=ailia.NETWORK_IMAGE_CHANNEL_FIRST,
            range=ailia.NETWORK_IMAGE_RANGE_U_INT8,
            algorithm=ailia.DETECTOR_ALGORITHM_YOLOX,
            env_id=env_id,
            memory_mode=args.memory_mode)
        detect_cars.net.set_input_shape(IMAGE_YOLO_SIZE, IMAGE_YOLO_SIZE)

    if args.video is not None:
        recognize_from_video(HC, LS, L)
    else:
        recognize_from_image(HC, LS, L)


if __name__ == '__main__':
    main()
