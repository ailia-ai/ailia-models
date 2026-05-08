import sys
import time

import ailia
import cv2
import numpy as np

sys.path.append('../../util')
from arg_utils import get_base_parser, update_parser, get_savepath  # noqa: E402
from model_utils import check_and_download_models  # noqa: E402
from detector_utils import load_image, letterbox_convert, reverse_letterbox  # noqa: E402
from nms_utils import nms_boxes  # noqa: E402
from webcamera_utils import get_capture, get_writer  # noqa: E402
from logging import getLogger  # noqa: E402

from transforms_utils import get_affine_transform, transform_preds
from output_utils import visualize, save_json

logger = getLogger(__name__)

# ======================
# Parameters
# ======================

WEIGHT_DETECTOR_PATH = 'face-detection-adas-0001.onnx'
MODEL_DETECTOR_PATH = 'face-detection-adas-0001.onnx.prototxt'
REMOTE_DETECTOR_PATH = 'https://storage.googleapis.com/ailia-models/face-detection-adas/'

WEIGHT_WFLW_PATH = 'hrnet_w18_wflw_256x256.onnx'
MODEL_WFLW_PATH = 'hrnet_w18_wflw_256x256.onnx.prototxt'
WEIGHT_AFLW_PATH = 'hrnet_w18_aflw_256x256.onnx'
MODEL_AFLW_PATH = 'hrnet_w18_aflw_256x256.onnx.prototxt'
WEIGHT_COFW_PATH = 'hrnet_w18_cofw_256x256.onnx'
MODEL_COFW_PATH = 'hrnet_w18_cofw_256x256.onnx.prototxt'
WEIGHT_300W_PATH = 'hrnet_w18_300w_256x256.onnx'
MODEL_300W_PATH = 'hrnet_w18_300w_256x256.onnx.prototxt'
REMOTE_LANDMARK_PATH = 'https://storage.googleapis.com/ailia-models/hrnet-facial-landmark-detection/'

IMAGE_PATH = 'input.jpg'
SAVE_IMAGE_PATH = 'output.png'

DETECTOR_HEIGHT = 384
DETECTOR_WIDTH = 672
LANDMARK_SIZE = 256

DETECTION_THRESHOLD = 0.5
LANDMARK_SCORE_THRESHOLD = 0.3

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

# WFLW 98-landmark flip pairs (0-indexed)
WFLW_FLIP_PAIRS = [
    [0, 32], [1, 31], [2, 30], [3, 29], [4, 28], [5, 27], [6, 26],
    [7, 25], [8, 24], [9, 23], [10, 22], [11, 21], [12, 20], [13, 19],
    [14, 18], [15, 17],
    [33, 46], [34, 45], [35, 44], [36, 43], [37, 42],
    [38, 50], [39, 49], [40, 48], [41, 47],
    [60, 72], [61, 71], [62, 70], [63, 69], [64, 68], [65, 75],
    [66, 74], [67, 73],
    [55, 59], [56, 58],
    [76, 82], [77, 81], [78, 80], [87, 83], [86, 84],
    [88, 92], [89, 91], [95, 93], [96, 97],
]

# AFLW 19-landmark flip pairs (0-indexed)
AFLW_FLIP_PAIRS = [
    [0, 5], [1, 4], [2, 3],
    [6, 11], [7, 10], [8, 9],
    [12, 14],
    [15, 17],
]

# COFW 29-landmark flip pairs (0-indexed)
COFW_FLIP_PAIRS = [
    [0, 1], [4, 6], [2, 3], [5, 7],
    [8, 9], [10, 11], [12, 14], [16, 17],
    [13, 15], [18, 19], [22, 23],
]

# 300W 68-landmark flip pairs (0-indexed)
W300_FLIP_PAIRS = [
    [0, 16], [1, 15], [2, 14], [3, 13], [4, 12], [5, 11], [6, 10], [7, 9],
    [17, 26], [18, 25], [19, 24], [20, 23], [21, 22],
    [31, 35], [32, 34],
    [36, 45], [37, 44], [38, 43], [39, 42], [40, 47], [41, 46],
    [48, 54], [49, 53], [50, 52], [61, 63], [60, 64], [67, 65], [58, 56], [59, 55],
]

MODEL_LIST = ['wflw', 'aflw', 'cofw', '300w']

# ======================
# Argument Parser Config
# ======================

parser = get_base_parser(
    'HRNet Facial Landmark Detection',
    IMAGE_PATH,
    SAVE_IMAGE_PATH,
)
parser.add_argument(
    '-m', '--model', default='wflw', choices=MODEL_LIST,
    help='Landmark model: wflw (98 pts), aflw (19 pts), cofw (29 pts), 300w (68 pts). Default: wflw'
)
parser.add_argument(
    '-th', '--threshold',
    default=DETECTION_THRESHOLD, type=float,
    help='Face detection confidence threshold'
)
parser.add_argument(
    '-w', '--write_json',
    action='store_true',
    help='Flag to output results to json file.'
)
args = update_parser(parser)


# ======================
# Secondary Functions
# ======================

PRIORBOX_PATH = 'mbox_priorbox.npy'
_prior_box = None


def get_prior_box():
    global _prior_box
    if _prior_box is None:
        _prior_box = np.squeeze(np.load(PRIORBOX_PATH))
    return _prior_box


def preprocess_detector(img):
    blob = letterbox_convert(img, (DETECTOR_HEIGHT, DETECTOR_WIDTH))
    blob = blob.transpose(2, 0, 1)
    blob = np.expand_dims(blob, axis=0).astype(np.float32)
    return blob


def _decode_bbox(mbox_loc, prior_box):
    mbox_loc = mbox_loc.reshape(-1, 4)
    pb = prior_box[0].reshape(-1, 4)
    var = prior_box[1].reshape(-1, 4)

    pw = pb[:, 2] - pb[:, 0]
    ph = pb[:, 3] - pb[:, 1]
    pcx = 0.5 * (pb[:, 2] + pb[:, 0])
    pcy = 0.5 * (pb[:, 3] + pb[:, 1])

    cx = mbox_loc[:, 0] * pw * var[:, 0] + pcx
    cy = mbox_loc[:, 1] * ph * var[:, 1] + pcy
    w = np.exp(mbox_loc[:, 2] * var[:, 2]) * pw
    h = np.exp(mbox_loc[:, 3] * var[:, 3]) * ph

    bboxes = np.stack([cx - w/2, cy - h/2, cx + w/2, cy + h/2], axis=1)
    return bboxes


def decode_detections(output, img, threshold):
    prior_box = get_prior_box()
    mbox_loc, mbox_conf = output

    bboxes = _decode_bbox(mbox_loc[0], prior_box)
    conf = mbox_conf[0].reshape(-1, 2)

    mask = conf[:, 1] >= threshold
    bboxes = bboxes[mask]
    scores = conf[mask, 1]

    bboxes[:, [0, 2]] *= DETECTOR_HEIGHT
    bboxes[:, [1, 3]] *= DETECTOR_WIDTH

    keep = nms_boxes(bboxes, scores, 0.5)
    bboxes = bboxes[keep].astype(int)
    scores = scores[keep]

    det_objs = []
    for (x1, y1, x2, y2), s in zip(bboxes, scores):
        det_objs.append(ailia.DetectorObject(
            category='', prob=float(s),
            x=x1/DETECTOR_HEIGHT, y=y1/DETECTOR_WIDTH,
            w=(x2-x1)/DETECTOR_HEIGHT, h=(y2-y1)/DETECTOR_WIDTH,
        ))
    det_objs = reverse_letterbox(det_objs, img, (DETECTOR_HEIGHT, DETECTOR_WIDTH))

    im_h, im_w = img.shape[:2]
    boxes = []
    for d in det_objs:
        x1 = max(0, d.x * im_w)
        y1 = max(0, d.y * im_h)
        x2 = min(im_w - 1, (d.x + d.w) * im_w)
        y2 = min(im_h - 1, (d.y + d.h) * im_h)
        boxes.append([x1, y1, x2, y2, d.prob])

    return (np.array(boxes, dtype=np.float32)
            if boxes else np.zeros((0, 5), dtype=np.float32))


def box2cs(box_xyxy):
    """Convert xyxy bbox to (center, scale) matching WFLW dataset convention."""
    x1, y1, x2, y2 = box_xyxy[:4]
    w = x2 - x1
    h = y2 - y1
    center = np.array([x1 + w * 0.5, y1 + h * 0.5], dtype=np.float32)
    # keep square crop, pixel_std=200
    scale = max(w, h) / 200.0 * 1.25
    return center, scale


def get_max_preds(heatmaps):
    N, K, H, W = heatmaps.shape
    flat = heatmaps.reshape(N, K, -1)
    # 1-indexed to match original repo
    idx = np.argmax(flat, axis=2) + 1
    maxvals = np.amax(flat, axis=2)

    preds = np.zeros((N, K, 2), dtype=np.float32)
    preds[:, :, 0] = (idx - 1) % W + 1
    preds[:, :, 1] = np.floor((idx - 1) / W) + 1

    mask = (maxvals > 0)[..., np.newaxis]
    preds = preds * mask
    return preds, maxvals[..., np.newaxis]


def decode_heatmaps(heatmaps, centers, scales):
    """Decode heatmap predictions back to image coordinates.

    Matches original repo's decode_preds: 1-indexed coords, left/up diff, +0.5 offset.
    """
    N, K, H, W = heatmaps.shape
    preds, maxvals = get_max_preds(heatmaps)

    # sub-pixel refinement (original uses left/up neighbours, not right/down)
    for n in range(N):
        for k in range(K):
            px = int(np.floor(preds[n, k, 0]))
            py = int(np.floor(preds[n, k, 1]))
            if 1 < px < W and 1 < py < H:
                dx = heatmaps[n, k, py - 1, px] - heatmaps[n, k, py - 1, px - 2]
                dy = heatmaps[n, k, py, px - 1] - heatmaps[n, k, py - 2, px - 1]
                preds[n, k, 0] += np.sign(dx) * 0.25
                preds[n, k, 1] += np.sign(dy) * 0.25
    preds += 0.5  # offset matching original

    # map back to original image space
    all_preds = np.zeros((N, K, 3), dtype=np.float32)
    for i in range(N):
        coords = transform_preds(preds[i], centers[i], scales[i], (W, H))
        all_preds[i, :, :2] = coords
        all_preds[i, :, 2] = maxvals[i, :, 0]

    return all_preds


def flip_back_heatmaps(flipped, flip_pairs):
    """Flip heatmaps horizontally and swap paired keypoints."""
    out = flipped[:, :, :, ::-1].copy()
    for left, right in flip_pairs:
        out[:, left, ...], out[:, right, ...] = (
            flipped[:, right, :, ::-1].copy(),
            flipped[:, left, :, ::-1].copy(),
        )
    return out


# ======================
# Main Functions
# ======================

def detect_faces(img, face_detector):
    blob = preprocess_detector(img)
    output = face_detector.predict([blob])
    return decode_detections(output, img, args.threshold)


def predict(img, face_detector, landmark_detector, flip_pairs):
    bboxes = detect_faces(img, face_detector)
    if len(bboxes) == 0:
        return [], bboxes

    img_rgb = img[:, :, ::-1].copy()  # BGR -> RGB

    batch_data = []
    centers, scales = [], []
    for bbox in bboxes:
        c, s = box2cs(bbox)
        centers.append(c)
        scales.append(s)

        trans = get_affine_transform(c, s, 0, (LANDMARK_SIZE, LANDMARK_SIZE))
        crop = cv2.warpAffine(
            img_rgb, trans, (LANDMARK_SIZE, LANDMARK_SIZE),
            flags=cv2.INTER_LINEAR)

        crop = (crop.astype(np.float32) / 255.0 - IMAGENET_MEAN) / IMAGENET_STD
        batch_data.append(crop)

    batch = np.asarray(batch_data).transpose(0, 3, 1, 2).astype(np.float32)

    heatmap = landmark_detector.predict([batch])[0]

    # flip augmentation
    flipped_out = landmark_detector.predict([batch[:, :, :, ::-1]])[0]
    # shift by 1 pixel to compensate for heatmap alignment
    flipped_out[:, :, :, 1:] = flipped_out[:, :, :, :-1].copy()
    heatmap = (heatmap + flip_back_heatmaps(flipped_out, flip_pairs)) * 0.5

    centers_arr = np.array(centers, dtype=np.float32)
    scales_arr = np.array(scales, dtype=np.float32)
    keypoints = decode_heatmaps(heatmap, centers_arr, scales_arr)
    return keypoints, bboxes


def recognize_from_image(face_detector, landmark_detector, flip_pairs):
    for image_path in args.input:
        logger.info(image_path)

        img = load_image(image_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)

        logger.info('Start inference...')
        if args.benchmark:
            logger.info('BENCHMARK mode')
            total_time = 0
            for i in range(args.benchmark_count):
                start = int(round(time.time() * 1000))
                keypoints, bboxes = predict(
                    img, face_detector, landmark_detector, flip_pairs)
                end = int(round(time.time() * 1000))
                logger.info(
                    f'\tailia processing estimation time {end - start} ms')
                if i != 0:
                    total_time += end - start
            logger.info(
                f'\taverage time estimation '
                f'{total_time / (args.benchmark_count - 1)} ms')
        else:
            keypoints, bboxes = predict(
                img, face_detector, landmark_detector, flip_pairs)

        res_img = visualize(img, keypoints, bboxes, LANDMARK_SCORE_THRESHOLD)

        savepath = get_savepath(args.savepath, image_path, ext='.png')
        logger.info(f'saved at : {savepath}')
        cv2.imwrite(savepath, res_img)

        if args.write_json:
            json_file = '%s.json' % savepath.rsplit('.', 1)[0]
            save_json(json_file, keypoints, bboxes)

    logger.info('Script finished successfully.')


def recognize_from_video(face_detector, landmark_detector, flip_pairs):
    video_file = args.video if args.video else args.input[0]
    capture = get_capture(video_file)
    assert capture.isOpened(), 'Cannot capture source'

    f_h = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    f_w = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    writer = None
    if args.savepath != SAVE_IMAGE_PATH:
        writer = get_writer(args.savepath, f_h, f_w)

    frame_shown = False
    while True:
        ret, frame = capture.read()
        if (cv2.waitKey(1) & 0xFF == ord('q')) or not ret:
            break
        if (frame_shown
                and cv2.getWindowProperty('frame', cv2.WND_PROP_VISIBLE) == 0):
            break

        keypoints, bboxes = predict(
            frame, face_detector, landmark_detector, flip_pairs)
        res_img = visualize(frame, keypoints, bboxes, LANDMARK_SCORE_THRESHOLD)

        cv2.imshow('frame', res_img)
        frame_shown = True

        if writer is not None:
            writer.write(res_img.astype(np.uint8))

    capture.release()
    cv2.destroyAllWindows()
    if writer is not None:
        writer.release()

    logger.info('Script finished successfully.')


def main():
    logger.info('Checking face detector model...')
    check_and_download_models(
        WEIGHT_DETECTOR_PATH, MODEL_DETECTOR_PATH, REMOTE_DETECTOR_PATH)

    dic_model = {
        'wflw': (WEIGHT_WFLW_PATH, MODEL_WFLW_PATH, WFLW_FLIP_PAIRS),
        'aflw': (WEIGHT_AFLW_PATH, MODEL_AFLW_PATH, AFLW_FLIP_PAIRS),
        'cofw': (WEIGHT_COFW_PATH, MODEL_COFW_PATH, COFW_FLIP_PAIRS),
        '300w': (WEIGHT_300W_PATH, MODEL_300W_PATH, W300_FLIP_PAIRS),
    }
    weight_path, model_path, flip_pairs = dic_model[args.model]

    logger.info('Checking landmark model...')
    check_and_download_models(weight_path, model_path, REMOTE_LANDMARK_PATH)

    env_id = args.env_id

    face_detector = ailia.Net(
        MODEL_DETECTOR_PATH, WEIGHT_DETECTOR_PATH, env_id=env_id)
    landmark_detector = ailia.Net(model_path, weight_path, env_id=env_id)

    if args.video is not None:
        recognize_from_video(face_detector, landmark_detector, flip_pairs)
    else:
        recognize_from_image(face_detector, landmark_detector, flip_pairs)


if __name__ == '__main__':
    main()
