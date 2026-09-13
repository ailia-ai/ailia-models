import sys
import time

import cv2
import numpy as np
import json

import ailia

sys.path.append('../../util')
from arg_utils import get_base_parser, update_parser, get_savepath  # noqa: E402
from model_utils import check_and_download_models  # noqa: E402
from image_utils import normalize_image  # noqa: E402C
from detector_utils import load_image  # noqa: E402C
from dtype_utils import numpy_type_to_builtin_type  # noqa: E402
import webcamera_utils  # noqa: E402

# logger
from logging import getLogger  # noqa: E402

from ap_10k_utils import *

logger = getLogger(__name__)


# ======================
# Parameters
# ======================

MODEL_LIST = ['hrnet32', 'hrnet48', 'res50', 'res101']
WEIGHT_HRNET_W32_PATH  = 'hrnet_w32_ap10k_256x256.onnx'
MODEL_HRNET_W32_PATH   = 'hrnet_w32_ap10k_256x256.onnx.prototxt'
WEIGHT_HRNET_W48_PATH  = 'hrnet_w48_ap10k_256x256.onnx'
MODEL_HRNET_W48_PATH   = 'hrnet_w48_ap10k_256x256.onnx.prototxt'
WEIGHT_RESNET_50_PATH  = 'res50_ap10k_256x256.onnx'
MODEL_RESNET_50_PATH   = 'res50_ap10k_256x256.onnx.prototxt'
WEIGHT_RESNET_101_PATH = 'res101_ap10k_256x256.onnx'
MODEL_RESNET_101_PATH  = 'res101_ap10k_256x256.onnx.prototxt'
REMOTE_PATH = 'https://storage.googleapis.com/ailia-models/ap-10k/'

DETECTION_MODEL_LIST = ['yolov3', 'yolox_m']
WEIGHT_YOLOV3_PATH = 'yolov3.opt2.onnx'
MODEL_YOLOV3_PATH = 'yolov3.opt2.onnx.prototxt'
REMOTE_YOLOV3_PATH = 'https://storage.googleapis.com/ailia-models/yolov3/'
WEIGHT_YOLOX_PATH = 'yolox_m.opt.onnx'
MODEL_YOLOX_PATH = 'yolox_m.opt.onnx.prototxt'
REMOTE_YOLOX_PATH = 'https://storage.googleapis.com/ailia-models/yolox/'

IMAGE_PATH = 'input.jpg'
SAVE_IMAGE_PATH = 'output.png'
IMAGE_SIZE = 256

DETECTION_THRESHOLD = 0.4
DETECTION_IOU = 0.45
DETECTION_SIZE = 416


# ======================
# Argument Parser Config
# ======================

parser = get_base_parser(
    'AP-10K: A Benchmark for Animal Pose Estimation in the Wild',
    IMAGE_PATH,
    SAVE_IMAGE_PATH,
)
parser.add_argument(
    '-m', '--model', metavar='ARCH',
    default='hrnet32', choices=MODEL_LIST,
    help='Set model architecture: ' + ' | '.join(MODEL_LIST)
)
parser.add_argument(
    '-d', '--detection_model', metavar='ARCH',
    default='yolov3', choices=DETECTION_MODEL_LIST,
    help='Set model architecture: ' + ' | '.join(DETECTION_MODEL_LIST)
)
parser.add_argument(
    '-n', '--max_num', default=None, type=int,
    help='Maximum number to detect objects. (without setting is for unlimited)'
)
parser.add_argument(
    '-th', '--threshold',
    default=DETECTION_THRESHOLD, type=float,
    help='The detection threshold'
)
parser.add_argument(
    '-iou', '--iou',
    default=DETECTION_IOU, type=float,
    help='The detection iou'
)
parser.add_argument(
    '-w', '--write_json',
    action='store_true',
    help='save result to json'
)
args = update_parser(parser)


# ======================
# Utils
# ======================


def _box2cs(box):
    """This encodes bbox(x,y,w,h) into (center, scale)
    Args:
        x, y, w, h

    Returns:
        tuple: A tuple containing center and scale.

        - np.ndarray[float32](2,): Center of the bbox (x, y).
        - np.ndarray[float32](2,): Scale of the bbox w & h.
    """

    input_size = (IMAGE_SIZE, IMAGE_SIZE)
    x, y, w, h = box[:4]

    aspect_ratio = input_size[0] / input_size[1]
    center = np.array([x + w * 0.5, y + h * 0.5], dtype=np.float32)

    if w > aspect_ratio * h:
        h = w * 1.0 / aspect_ratio
    elif w < aspect_ratio * h:
        w = h * aspect_ratio

    # pixel std is 200.0
    scale = np.array([w / 200.0, h / 200.0], dtype=np.float32)

    scale = scale * 1.25

    return center, scale


def _xywh2xyxy(bbox_xywh):
    """Transform the bbox format from xywh to x1y1x2y2.

    Args:
        bbox_xywh (ndarray): Bounding boxes (with scores),
            shaped (n, 4) or (n, 5). (left, top, width, height, [score])
    Returns:
        np.ndarray: Bounding boxes (with scores), shaped (n, 4) or
          (n, 5). (left, top, right, bottom, [score])
    """
    bbox_xyxy = bbox_xywh.copy()
    bbox_xyxy[2] = bbox_xyxy[2] + bbox_xyxy[0] - 1
    bbox_xyxy[3] = bbox_xyxy[3] + bbox_xyxy[1] - 1

    return bbox_xyxy


def preprocess(img, bbox):
    image_size = (IMAGE_SIZE, IMAGE_SIZE)

    c, s = _box2cs(bbox)
    r = 0

    trans = get_affine_transform(c, s, r, image_size)
    img = cv2.warpAffine(
        img,
        trans, (int(image_size[0]), int(image_size[1])),
        flags=cv2.INTER_LINEAR)

    # normalize
    img = normalize_image(img, normalize_type='ImageNet')

    img = img.transpose(2, 0, 1)  # HWC -> CHW
    img = np.expand_dims(img, axis=0)

    img_metas = [{
        'center': c,
        'scale': s,
    }]

    return img, img_metas


def postprocess(output, img_metas):
    """Decode keypoints from heatmaps.

    Args:
        output (np.ndarray[N, K, H, W]): model predicted heatmaps.
        img_metas (list(dict)): Information about data augmentation
            By default this includes:
            - "image_file: path to the image file
            - "center": center of the bbox
            - "scale": scale of the bbox
            - "rotation": rotation of the bbox
            - "bbox_score": score of bbox
    """
    batch_size = len(img_metas)

    c = np.zeros((batch_size, 2), dtype=np.float32)
    s = np.zeros((batch_size, 2), dtype=np.float32)
    score = np.ones(batch_size)
    for i in range(batch_size):
        c[i, :] = img_metas[i]['center']
        s[i, :] = img_metas[i]['scale']

    preds, maxvals = keypoints_from_heatmaps(output, c, s)

    all_preds = np.zeros((batch_size, preds.shape[1], 3), dtype=np.float32)
    all_boxes = np.zeros((batch_size, 6), dtype=np.float32)
    all_preds[:, :, 0:2] = preds[:, :, 0:2]
    all_preds[:, :, 2:3] = maxvals
    all_boxes[:, 0:2] = c[:, 0:2]
    all_boxes[:, 2:4] = s[:, 0:2]
    all_boxes[:, 4] = np.prod(s * 200.0, axis=1)
    all_boxes[:, 5] = score

    result = {}
    result['preds'] = all_preds
    result['boxes'] = all_boxes

    return result


def pose_estimate(net, det_net, img):
    h, w = img.shape[:2]
    n = args.max_num

    logger.debug(f'input image shape: {img.shape}')

    if det_net:
        det_net.set_input_shape(DETECTION_SIZE, DETECTION_SIZE)
        det_net.compute(img, args.threshold, args.iou)
        count = det_net.get_object_count()

        if 0 < count:
            a = sorted([
                det_net.get_object(i) for i in range(count)
            ], key=lambda x: x.prob, reverse=True)
            a = a[:n] if n else a
            bboxes = np.array([
                (int(w * obj.x), int(h * obj.y), int(w * obj.w), int(h * obj.h))
                for obj in a[:n]
            ])
        else:
            bboxes = np.array([[0, 0, w, h]])
    else:
        bboxes = np.array([[0, 0, w, h]])

    img_0 = cv2.cvtColor(img, cv2.COLOR_BGRA2RGB)

    pose_results = []
    for bbox in bboxes:
        img, img_metas = preprocess(img_0, bbox)

        # inference
        output = net.predict([img])
        heatmap = output[0]

        result = postprocess(heatmap, img_metas)
        pose = result['preds'][0]

        # plot result
        pose_results.append({
            'bbox': _xywh2xyxy(bbox),
            'keypoints': pose,
        })

    return pose_results


def vis_pose_result(img, result):
    skeleton = [[0, 1], [0, 2], [1, 2], [2, 3], [3, 4], [3, 5], [5, 6], 
                [6, 7], [3, 8], [8, 9], [9, 10], [4, 11], [11, 12], [12, 13], 
                [4, 14], [14, 15], [15, 16]]
    pose_kpt_color = np.array([
        [0, 255, 0], [255, 128, 0], [51, 153, 255], 
        [51, 153, 255], [51, 153, 255], [51, 153, 255],
        [51, 153, 255], [0, 255, 0], [0, 255, 0],
        [255, 128, 0], [0, 255, 0], [255, 128, 0],
        [255, 128, 0], [0, 255, 0], [0, 255, 0],
        [0, 255, 0], [0, 255, 0]
    ])
    pose_limb_color = np.array([
        [0, 0, 255], [0, 0, 255], [0, 0, 255],
        [0, 255, 0], [0, 255, 0], [0, 255, 255],
        [0, 255, 255], [0, 255, 255], [6, 156, 250],
        [6, 156, 250], [6, 156, 250], [0, 255, 255],
        [0, 255, 255], [0, 255, 255], [6, 156, 250],
        [6, 156, 250], [6, 156, 250]
    ])
    
    img = show_result(
        img,
        result,
        skeleton,
        pose_kpt_color=pose_kpt_color,
        pose_limb_color=pose_limb_color,
        thickness=2)

    return img


def save_json(pose_results, json_path):
    output = []
    for r in pose_results:
        o = {}
        o['bbox'] = r['bbox']
        o['keypoints'] = [
            {'pos': [k[0], k[1]], 'prob': k[2]} for k in r['keypoints']
        ]
        output.append(o)

    with open(json_path, 'w') as f:
        json.dump(numpy_type_to_builtin_type(output), f, indent=2)


# ======================
# Main functions
# ======================

def recognize_from_image(net, det_net):
    # input image loop
    for image_path in args.input:
        # prepare input data
        logger.info(image_path)

        img = load_image(image_path)

        # inference
        logger.info('Start inference...')
        if args.benchmark:
            logger.info('BENCHMARK mode')
            total_time_estimation = 0
            for i in range(args.benchmark_count):
                # Pose estimation
                start = int(round(time.time() * 1000))
                pose_results = pose_estimate(net, det_net, img)
                end = int(round(time.time() * 1000))
                estimation_time = (end - start)

                # Loggin
                logger.info(f'\tailia processing estimation time {estimation_time} ms')
                if i != 0:
                    total_time_estimation = total_time_estimation + estimation_time

            logger.info(f'\taverage time estimation {total_time_estimation / (args.benchmark_count - 1)} ms')
        else:
            # inference
            pose_results = pose_estimate(net, det_net, img)
        
        # plot result
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
        img = vis_pose_result(img, pose_results) # TODO: この処理がおかしい可能性

        # save results
        savepath = get_savepath(args.savepath, image_path)
        logger.info(f'saved at : {savepath}')
        cv2.imwrite(savepath, img)

        if args.write_json:
            save_json(pose_results, (savepath.rsplit('.', 1)[0]) + '.json')

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
    while (True):
        ret, frame = capture.read()
        if (cv2.waitKey(1) & 0xFF == ord('q')) or not ret:
            break
        if frame_shown and cv2.getWindowProperty('frame', cv2.WND_PROP_VISIBLE) == 0:
            break

        # inference
        img = cv2.cvtColor(frame, cv2.COLOR_BGR2BGRA)
        pose_results = pose_estimate(net, det_net, img)

        # plot result
        frame = vis_pose_result(frame, pose_results)

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
    detector = True

    if detector:
        logger.info('=== detector model ===')
        if args.detection_model=="yolov3":
            check_and_download_models(WEIGHT_YOLOV3_PATH, MODEL_YOLOV3_PATH, REMOTE_YOLOV3_PATH)
        else:
            check_and_download_models(WEIGHT_YOLOX_PATH, MODEL_YOLOX_PATH, REMOTE_YOLOX_PATH)
    
    logger.info('=== AP-10k model ===')
    info = {
        'hrnet32': (WEIGHT_HRNET_W32_PATH, MODEL_HRNET_W32_PATH),
        'hrnet48': (WEIGHT_HRNET_W48_PATH, MODEL_HRNET_W48_PATH),
        'res50': (WEIGHT_RESNET_50_PATH, MODEL_RESNET_50_PATH),
        'res101': (WEIGHT_RESNET_101_PATH, MODEL_RESNET_101_PATH),
    }
    weight_path, model_path = info[args.model]
    check_and_download_models(weight_path, model_path, REMOTE_PATH)

    env_id = args.env_id

    # initialize
    if detector:
        if args.detection_model=="yolov3":
            det_net = ailia.Detector(
                MODEL_YOLOV3_PATH,
                WEIGHT_YOLOV3_PATH,
                80,
                format=ailia.NETWORK_IMAGE_FORMAT_RGB,
                channel=ailia.NETWORK_IMAGE_CHANNEL_FIRST,
                range=ailia.NETWORK_IMAGE_RANGE_U_FP32,
                algorithm=ailia.DETECTOR_ALGORITHM_YOLOV3,
                env_id=env_id,
            )
        else:
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
    else:
        det_net = None
    net = ailia.Net(model_path, weight_path, env_id=env_id)

    if args.video is not None:
        # video mode
        recognize_from_video(net, det_net)
    else:
        # image mode
        recognize_from_image(net, det_net)


if __name__ == '__main__':
    main()
