import sys
import os
import time
from logging import getLogger

import ailia
import cv2
import numpy as np

sys.path.append('../../util')
import webcamera_utils  # noqa
from image_utils import imread  # noqa
from model_utils import check_and_download_models  # noqa
from arg_utils import get_base_parser, get_savepath, update_parser  # noqa

from ax_age_gender_util import (BlazeFace, AlignConfig, align_face,  # noqa
                                pose_in_range)

_this = os.path.dirname(os.path.abspath(__file__))
top_path = os.path.dirname(os.path.dirname(_this))

logger = getLogger(__name__)

# ======================
# Parameters
# ======================

WEIGHT_PATH = 'ax_age_gender_fp16.onnx'
MODEL_PATH = None
REMOTE_PATH = 'https://storage.googleapis.com/ailia-models/ax_age_gender/'

BLAZEFACE_WEIGHT_PATH = 'blazefaceback.onnx'
BLAZEFACE_MODEL_PATH = 'blazefaceback.onnx.prototxt'
BLAZEFACE_REMOTE_PATH = 'https://storage.googleapis.com/ailia-models/blazeface/'
BLAZEFACE_ANCHORS_PATH = os.path.join(
    top_path, 'face_detection/blazeface/anchorsback.npy')

FACE_MIN_SCORE_THRESH = 0.5
SMOOTH_ALPHA = 0.1

# preprocessing constants carried in the ONNX metadata of the model
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], np.float32)
ALIGN_CONFIG = AlignConfig(size=128, interocular=0.30, eye_y=0.40)

GENDER_NAMES = ('Male', 'Female')
MASK_NAMES = ('nomask', 'mask')
MASK_TINT = (255, 0, 255)
PERSON_EDGE = (0, 255, 255)

IMAGE_PATH = 'demo.jpg'
SAVE_IMAGE_PATH = 'output.png'

# ======================
# Argument Parser Config
# ======================

parser = get_base_parser(
    'ax_age_gender', IMAGE_PATH, SAVE_IMAGE_PATH,
)
parser.add_argument(
    '--no-pose-gate', action='store_true',
    help='Estimate even for head angles the model is not accurate on.'
)
parser.add_argument(
    '--smooth-alpha', default=SMOOTH_ALPHA, type=float,
    help='Video-only exponential smoothing factor; 0 disables smoothing.'
)
parser.add_argument(
    '--no-segmentation', action='store_true',
    help='Do not draw the person and face-mask segmentation.'
)
args = update_parser(parser)
if not 0.0 <= args.smooth_alpha <= 1.0:
    parser.error('--smooth-alpha must be between 0 and 1.')


# ======================
# Temporal smoothing
# ======================

def _box_iou(a, b):
    x1 = max(a[0], b[0])
    y1 = max(a[1], b[1])
    x2 = min(a[2], b[2])
    y2 = min(a[3], b[3])
    inter = max(x2 - x1, 0) * max(y2 - y1, 0)
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[2] - b[0]) * (b[3] - b[1])
    return inter / (area_a + area_b - inter + 1e-9)


class TemporalSmoother:
    """Per-face exponential average over consecutive video frames."""

    def __init__(self, alpha=SMOOTH_ALPHA, iou_thresh=0.3, max_missing=8):
        self.alpha = alpha
        self.iou_thresh = iou_thresh
        self.max_missing = max_missing
        self.tracks = {}
        self.next_id = 0

    def _match(self, box, unavailable):
        best_id = None
        best_iou = self.iou_thresh
        for track_id, track in self.tracks.items():
            if track_id in unavailable:
                continue
            iou = _box_iou(box, track['box'])
            if iou >= best_iou:
                best_id = track_id
                best_iou = iou
        return best_id

    def __call__(self, results):
        seen = set()
        for result in results:
            if not result.get('pose_ok', True) or 'age_prob' not in result:
                continue

            track_id = self._match(result['box'], seen)
            if track_id is None:
                track_id = self.next_id
                self.next_id += 1
                self.tracks[track_id] = {
                    'age_prob': result['age_prob'].copy(),
                    'gender': result['gender_probs'].copy(),
                    'mask': result['mask_probs'].copy(),
                }
            else:
                track = self.tracks[track_id]
                alpha = self.alpha
                track['age_prob'] = (
                    alpha * result['age_prob']
                    + (1 - alpha) * track['age_prob']
                )
                track['gender'] = (
                    alpha * result['gender_probs']
                    + (1 - alpha) * track['gender']
                )
                track['mask'] = (
                    alpha * result['mask_probs']
                    + (1 - alpha) * track['mask']
                )

            track = self.tracks[track_id]
            track['box'] = result['box'].copy()
            track['missing'] = 0
            seen.add(track_id)

            bins = np.arange(len(track['age_prob']), dtype=np.float32)
            age = np.sum(track['age_prob'] * bins) / np.sum(track['age_prob'])
            gi = int(np.argmax(track['gender']))
            mi = int(np.argmax(track['mask']))
            result.update({
                'track_id': track_id,
                'age': float(age),
                'gender': GENDER_NAMES[gi],
                'gender_prob': float(track['gender'][gi]),
                'mask': MASK_NAMES[mi],
                'mask_prob': float(track['mask'][mi]),
                'smoothed': True,
            })

        for track_id in list(self.tracks):
            if track_id in seen:
                continue
            track = self.tracks[track_id]
            track['missing'] = track.get('missing', 0) + 1
            if track['missing'] > self.max_missing:
                del self.tracks[track_id]

        return results


# ======================
# Main functions
# ======================

def estimate_age_gender(net, crop_img):
    img = cv2.cvtColor(crop_img, cv2.COLOR_BGR2RGB).astype(np.float32)
    img = (img / 255.0 - IMAGENET_MEAN) / IMAGENET_STD
    img = np.ascontiguousarray(img.transpose(2, 0, 1)[np.newaxis])

    output = net.predict([img])
    age, gender, mask, age_prob, seg = output[:5]

    age = float(age[0, 0])
    gi = int(np.argmax(gender[0]))
    mi = int(np.argmax(mask[0]))
    return {
        'age': age,
        'age_prob': age_prob[0].copy(),
        'gender': GENDER_NAMES[gi],
        'gender_prob': float(gender[0][gi]),
        'gender_probs': gender[0].copy(),
        'mask': MASK_NAMES[mi],
        'mask_prob': float(mask[0][mi]),
        'mask_probs': mask[0].copy(),
        'seg': seg[0].copy(),
        'smoothed': False,
    }


def draw_segmentation(image, results, threshold=0.5, alpha=0.45):
    """Draw face-mask fill and person silhouette in the source image."""
    out = image.copy()
    im_h, im_w = image.shape[:2]
    mask_fill = np.zeros((im_h, im_w), dtype=np.uint8)
    person_edge = np.zeros((im_h, im_w), dtype=np.uint8)

    for result in results:
        seg = result.get('seg')
        inverse = result.get('seg_inverse')
        if seg is None or inverse is None:
            continue

        person = (seg[0] > threshold).astype(np.uint8) * 255
        mask = (seg[1] > threshold).astype(np.uint8) * 255
        warped_mask = cv2.warpAffine(
            mask, inverse, (im_w, im_h), flags=cv2.INTER_NEAREST
        )
        mask_fill = np.maximum(mask_fill, warped_mask)

        warped_person = cv2.warpAffine(
            person, inverse, (im_w, im_h), flags=cv2.INTER_NEAREST
        )
        contours = cv2.findContours(
            warped_person, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )[-2]
        cv2.drawContours(person_edge, contours, -1, 255, 2)

    if mask_fill.any():
        tint = np.zeros_like(out)
        tint[:] = MASK_TINT
        selected = mask_fill > 0
        out[selected] = (
            out[selected] * (1 - alpha) + tint[selected] * alpha
        ).astype(np.uint8)
        contours = cv2.findContours(
            mask_fill, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )[-2]
        cv2.drawContours(out, contours, -1, MASK_TINT, 2)

    out[person_edge > 0] = PERSON_EDGE
    return out


def draw_results(image, results):
    if args.no_segmentation:
        output = image.copy()
    else:
        output = draw_segmentation(image, results)

    for result in results:
        x1, y1, x2, y2 = result['box'].astype(int)
        if not result['pose_ok']:
            color = (140, 140, 140)
            cv2.rectangle(output, (x1, y1), (x2, y2), color, thickness=2)
            continue

        gender = result['gender']
        age = result['age']
        mask = result['mask']

        label_width = x2 - x1
        label_height = 20
        if gender == 'Male':
            color = (255, 128, 128)
        else:
            color = (128, 128, 255)
        cv2.rectangle(output, (x1, y1), (x2, y2), color, thickness=2)
        cv2.rectangle(
            output,
            (x1, y1),
            (x1 + label_width, y1 + label_height),
            color,
            thickness=-1,
        )

        label = '{} {}'.format(gender, round(age))
        if mask == 'mask':
            label += ' [mask]'
        text_position = (x1, y1 + label_height // 2)
        cv2.putText(
            output,
            label,
            text_position,
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 0, 0),
            1,
        )

    return output


def recognize_image(net, detector, image, smoother=None):
    # detect face
    detections = detector(image, min_score=FACE_MIN_SCORE_THRESH)

    results = []

    # estimate age and gender
    for det in detections:
        box = detector.box(det)

        # align face from the eye keypoints
        keypoints = detector.keypoints(det)
        crop_img, aligned_kp, transform = align_face(
            image, keypoints, ALIGN_CONFIG
        )
        if crop_img is None:
            continue

        # skip head angles the model is not accurate on
        ok, pitch, yaw, why = pose_in_range(aligned_kp)
        result = {
            'box': box,
            'pose_ok': bool(ok or args.no_pose_gate),
            'pitch': pitch,
            'yaw': yaw,
        }
        results.append(result)
        if not result['pose_ok']:
            logger.info(" skipped: %s (pitch %.2f yaw %+.2f)" % (why, pitch, yaw))
            continue

        # inference
        result.update(estimate_age_gender(net, crop_img))
        result['seg_inverse'] = cv2.invertAffineTransform(transform)

    if smoother is not None:
        smoother(results)

    for result in results:
        if not result['pose_ok']:
            continue
        logger.info(
            " gender is: %s (%.2f)"
            % (result['gender'], result['gender_prob'] * 100)
        )
        logger.info(" age is: %.1f" % result['age'])
        logger.info(
            " mask is: %s (%.2f)"
            % (result['mask'], result['mask_prob'] * 100)
        )

    return draw_results(image, results)


def recognize_from_image(net, detector):
    # input image loop
    for image_path in args.input:
        # prepare input data
        logger.info(image_path)
        image = imread(image_path)

        # inference
        logger.info('Start inference...')
        if args.benchmark:
            logger.info('BENCHMARK mode')
            total_time_estimation = 0
            for i in range(args.benchmark_count):
                start = int(round(time.time() * 1000))
                image_out = recognize_image(net, detector, image.copy())
                end = int(round(time.time() * 1000))
                estimation_time = (end - start)

                # Logging
                logger.info(f'\tailia processing estimation time {estimation_time} ms')
                if i != 0:
                    total_time_estimation = total_time_estimation + estimation_time

            logger.info(f'\taverage time estimation {total_time_estimation / (args.benchmark_count - 1)} ms')
        else:
            image_out = recognize_image(net, detector, image.copy())

        savepath = get_savepath(args.savepath, image_path)
        logger.info(f'saved at : {savepath}')
        cv2.imwrite(savepath, image_out)

    logger.info('Script finished successfully.')


def recognize_from_video(net, detector):
    capture = webcamera_utils.get_capture(args.video)

    # create video writer if savepath is specified as video format
    if args.savepath != SAVE_IMAGE_PATH:
        f_h = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        f_w = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        fps = capture.get(cv2.CAP_PROP_FPS)
        if fps <= 0:
            fps = 20
        writer = webcamera_utils.get_writer(
            args.savepath, f_h, f_w, fps=fps
        )
    else:
        writer = None

    smoother = None
    if args.smooth_alpha > 0:
        smoother = TemporalSmoother(args.smooth_alpha)

    frame_shown = False
    while True:
        ret, frame = capture.read()
        if (cv2.waitKey(1) & 0xFF == ord('q')) or not ret:
            break
        if frame_shown and cv2.getWindowProperty('frame', cv2.WND_PROP_VISIBLE) == 0:
            break

        frame = recognize_image(net, detector, frame, smoother=smoother)

        # show result
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
    logger.info('=== ax_age_gender model ===')
    check_and_download_models(
        WEIGHT_PATH, MODEL_PATH, REMOTE_PATH
    )
    logger.info('=== face detection model ===')
    check_and_download_models(
        BLAZEFACE_WEIGHT_PATH, BLAZEFACE_MODEL_PATH, BLAZEFACE_REMOTE_PATH
    )

    # load model
    env_id = args.env_id

    # net initialize
    net = ailia.Net(
        MODEL_PATH, WEIGHT_PATH, env_id=env_id
    )
    detector_net = ailia.Net(
        BLAZEFACE_MODEL_PATH, BLAZEFACE_WEIGHT_PATH, env_id=env_id
    )
    detector = BlazeFace(detector_net, BLAZEFACE_ANCHORS_PATH)

    if args.video is not None:
        # video mode
        recognize_from_video(net, detector)
    else:
        # image mode
        recognize_from_image(net, detector)


if __name__ == '__main__':
    main()
