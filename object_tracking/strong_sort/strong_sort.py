import sys
import os
import time
from logging import getLogger

import numpy as np
import cv2
from PIL import Image
import matplotlib

import ailia

# import original modules
sys.path.append('../../util')
from arg_utils import get_base_parser, update_parser, get_savepath  # noqa: E402
from model_utils import check_and_download_models  # noqa: E402
from load_model import load_bytetrack
from webcamera_utils import get_capture, get_writer  # noqa: E402

from ecc import ECC
from deep_sort import nn_matching
from deep_sort.detection import Detection
from deep_sort.tracker import Tracker
from AppFreeLink import AFLink
from GSI import GSInterpolation

_this = os.path.dirname(os.path.abspath(__file__))
top_path = os.path.dirname(os.path.dirname(_this))

logger = getLogger(__name__)

# ======================
# Parameters
# ======================

WEIGHT_AFLINK_PATH = 'AFLink_epoch20.onnx'
MODEL_AFLINK_PATH = 'AFLink_epoch20.onnx.prototxt'
WEIGHT_FRID_PATH = 'duke_bot_S50.onnx'
MODEL_FRID_PATH = 'duke_bot_S50.onnx.prototxt'
REMOTE_FRID_PATH = \
    'https://storage.googleapis.com/ailia-models/strong_sort/'

WEIGHT_MOT17_X_PATH = 'bytetrack_x_mot17.onnx'
MODEL_MOT17_X_PATH = 'bytetrack_x_mot17.onnx.prototxt'
WEIGHT_MOT17_S_PATH = 'bytetrack_s_mot17.onnx'
MODEL_MOT17_S_PATH = 'bytetrack_s_mot17.onnx.prototxt'
WEIGHT_MOT17_TINY_PATH = 'bytetrack_tiny_mot17.onnx'
MODEL_MOT17_TINY_PATH = 'bytetrack_tiny_mot17.onnx.prototxt'
WEIGHT_MOT20_X_PATH = 'bytetrack_x_mot20.onnx'
MODEL_MOT20_X_PATH = 'bytetrack_x_mot20.onnx.prototxt'
REMOTE_BYTRK_PATH = \
    'https://storage.googleapis.com/ailia-models/bytetrack/'

WEIGHT_YOLOX_S_PATH = 'yolox_s.opt.onnx'
MODEL_YOLOX_S_PATH = 'yolox_s.opt.onnx.prototxt'
WEIGHT_YOLOX_TINY_PATH = 'yolox_tiny.opt.onnx'
MODEL_YOLOX_TINY_PATH = 'yolox_tiny.opt.onnx.prototxt'
REMOTE_YOLOX_PATH = \
    'https://storage.googleapis.com/ailia-models/yolox/'

VIDEO_PATH = 'demo.mp4'
SAVE_TEXT_PATH = 'output.txt'

# ======================
# Arguemnt Parser Config
# ======================

parser = get_base_parser(
    'StrongSORT', VIDEO_PATH, None
)
parser.add_argument(
    "--score_thre", type=float, default=0.6,
    help="Score threshould to filter the result.",
)
parser.add_argument(
    "--nms_thre", type=float, default=0.7,
    help="NMS threshould.",
)
parser.add_argument(
    '--AFLink', action='store_true',
    help='Appearance-Free Link'
)
parser.add_argument(
    '--GSI', action='store_true',
    help='Gaussian-smoothed Interpolation'
)
parser.add_argument(
    '-m', '--model_type', default='mot17_x',
    choices=('mot17_x', 'mot20_x', 'mot17_s', 'mot17_tiny', 'yolox_s', 'yolox_tiny'),
    help='model type'
)
parser.add_argument(
    '--cui',
    action='store_true',
    help="Don't display preview in GUI."
)
# tracking args
parser.add_argument('--min-box-area', type=float, default=10, help='filter out tiny boxes')
args = update_parser(parser)


# ======================
# Secondaty Functions
# ======================


def setup_detector(net):
    mod = load_bytetrack(args)

    def _detector(img):
        dets = mod.predict(net, img)
        return dets

    detector = _detector

    return detector


def get_colors(n, colormap="gist_ncar"):
    # Get n color samples from the colormap, derived from: https://stackoverflow.com/a/25730396/583620
    # gist_ncar is the default colormap as it appears to have the highest number of color transitions.
    # tab20 also seems like it would be a good option but it can only show a max of 20 distinct colors.
    # For more options see:
    # https://matplotlib.org/examples/color/colormaps_reference.html
    # and https://matplotlib.org/users/colormaps.html

    if hasattr(matplotlib, "colormaps"):
        cm = matplotlib.colormaps[colormap]
    else:
        cm = matplotlib.cm.get_cmap(colormap)
    colors = cm(np.linspace(0, 1, n))
    # Randomly shuffle the colors
    np.random.shuffle(colors)
    # Opencv expects bgr while cm returns rgb, so we swap to match the colormap (though it also works fine without)
    # Also multiply by 255 since cm returns values in the range [0, 1]
    colors = colors[:, (2, 1, 0)] * 255

    return colors


num_colors = 50
vis_colors = get_colors(num_colors)


def rectangle(image, x, y, w, h):
    color = (0, 255, 117)
    thickness = 2

    pt1 = int(x), int(y)
    pt2 = int(x + w), int(y + h)
    cv2.rectangle(image, pt1, pt2, color, thickness)


def frame_vis_generator(frame, bboxes, ids):
    for i, entity_id in enumerate(ids):
        color = vis_colors[int(entity_id) % num_colors]

        x1, y1, w, h = np.round(bboxes[i]).astype(int)
        x2 = x1 + w
        y2 = y1 + h
        cv2.rectangle(frame, (x1, y1), (x2, y2), color=color, thickness=3)
        cv2.putText(frame, str(entity_id), (x1 + 5, y1 + 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.5, color, thickness=3)

    return frame


# ======================
# Main functions
# ======================

def compute_ecc(src_img, dst_img):
    warp_matrix, src_aligned = ECC(
        src_img, dst_img, warp_mode=cv2.MOTION_EUCLIDEAN, eps=1e-5,
        max_iter=100, scale=0.1, align=False)
    if warp_matrix is None:
        return None
    [a, b] = warp_matrix
    warp_matrix = np.array([a, b, [0, 0, 1]])
    return warp_matrix.tolist()


def preprocess(img):
    h, w = (256, 128)

    img = img[:, :, ::-1]  # BGR -> RGB
    img = np.array(Image.fromarray(img).resize((w, h), Image.Resampling.BILINEAR))

    img = img.transpose(2, 0, 1)  # HWC -> CHW
    img = np.expand_dims(img, axis=0)
    img = img.astype(np.float32)

    return img


def predict(mod, img):
    detector = mod["detector"]
    frid_net = mod["frid_net"]

    dets = detector(img)
    dets[:, 2] -= dets[:, 0]
    dets[:, 3] -= dets[:, 1]

    bboxes, confidences = dets[:, :4], dets[:, 4]

    im = Image.fromarray(img)
    crop_imgs = [
        np.array(im.crop((b[0], b[1], b[0] + b[2], b[1] + b[3]))) for b in dets[:, :4]
    ]

    mask = [True] * len(crop_imgs)
    imgs = []
    for i, img in enumerate(crop_imgs):
        if img.shape[0] == 0 or img.shape[1] == 0:
            mask[i] = False
            continue
        img = preprocess(img)
        imgs.append(img)

    bboxes = bboxes[mask]
    confidences = confidences[mask]

    if len(imgs) == 0:
        return np.zeros((0, 0)), np.zeros((0, 0)), np.zeros((0, 0))

    batch = np.concatenate(imgs, axis=0)

    # feedforward
    output = frid_net.predict([batch])
    features = output[0]

    ind = np.argsort(-confidences)
    bboxes = bboxes[ind]
    confidences = confidences[ind]
    features = features[ind]

    return bboxes, confidences, features


def benchmarking(net):
    video_file = args.video if args.video else args.input[0]
    capture = get_capture(video_file)
    assert capture.isOpened(), 'Cannot capture source'

    _, frame = capture.read()

    logger.info('BENCHMARK mode')

    total_time_estimation = 0
    for i in range(args.benchmark_count):
        start = int(round(time.time() * 1000))
        predict(net, frame)
        end = int(round(time.time() * 1000))
        estimation_time = (end - start)

        # Loggin
        logger.info(f'\tailia processing estimation time {estimation_time} ms')
        if i != 0:
            total_time_estimation = total_time_estimation + estimation_time

    logger.info(f'\taverage time estimation {total_time_estimation / (args.benchmark_count - 1)} ms')


def recognize_from_video(mod):
    min_box_area = args.min_box_area

    video_file = args.video if args.video else args.input[0]
    capture = get_capture(video_file)
    assert capture.isOpened(), 'Cannot capture source'

    # create video writer if savepath is specified as video format
    f_h = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    f_w = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    if args.savepath is not None:
        writer = get_writer(args.savepath, f_h, f_w)
    else:
        writer = None

    tracker = mod["tracker"]
    linker = mod["linker"]

    track = []

    frame_idx = 1
    prev_frame = None
    frame_shown = False
    while True:
        ret, frame = capture.read()
        if (cv2.waitKey(1) & 0xFF == ord('q')) or not ret:
            break
        if frame_shown and cv2.getWindowProperty('frame', cv2.WND_PROP_VISIBLE) == 0:
            break

        # inference
        bboxes, confidences, features = predict(mod, frame)

        detections = []
        for bbox, confidence, feature in zip(bboxes, confidences, features):
            detections.append(Detection(bbox, confidence, feature))

        ecc = compute_ecc(prev_frame, frame) if prev_frame is not None else None
        tracker.camera_update(ecc)

        prev_frame = np.copy(frame)

        # run tracking
        tracker.predict()
        tracker.update(detections)
        online_tlwhs = []
        online_ids = []
        for t in tracker.tracks:
            tlwh = t.to_tlwh()
            tid = t.track_id
            if not t.is_confirmed() or t.time_since_update > 1:
                rectangle(frame, tlwh[0], tlwh[1], tlwh[2], tlwh[3])
                continue
            track.append([frame_idx, tid, tlwh[0], tlwh[1], tlwh[2], tlwh[3], 1, -1, -1, -1])

            if t.time_since_update > 0:
                continue
            vertical = tlwh[2] / tlwh[3] > 1.6
            if tlwh[2] * tlwh[3] > min_box_area and not vertical:
                online_tlwhs.append(tlwh)
                online_ids.append(tid)

        frame_idx += 1

        res_img = frame_vis_generator(frame, online_tlwhs, online_ids)

        # show
        if not args.cui or args.video:
            cv2.imshow('frame', res_img)
            frame_shown = True
        else:
            print("Online ids", online_ids)

        # save results
        if writer is not None:
            writer.write(res_img.astype(np.uint8))

    capture.release()
    cv2.destroyAllWindows()
    if writer is not None:
        writer.release()

    if 0 < len(track):
        track = np.array(track)
    else:
        track = np.zeros((0, 10))

    if args.AFLink and 0 < len(track):
        track = linker.link(track)

    if args.GSI and 0 < len(track):
        track = GSInterpolation(
            track,
            interval=20,
            tau=10
        )

    # save result
    savepath = get_savepath(
        args.savepath if args.savepath else SAVE_TEXT_PATH,
        "video-%s" % args.video if args.video else args.input[0], ext='.txt')
    logger.info(f'saved at : {savepath}')
    np.savetxt(savepath, track, fmt='%d,%d,%.2f,%.2f,%.2f,%.2f,%.2f,%d,%d,%d')

    logger.info('Script finished successfully.')


def main():
    dic_model = {
        'mot17_x': (WEIGHT_MOT17_X_PATH, MODEL_MOT17_X_PATH),
        'mot17_s': (WEIGHT_MOT17_S_PATH, MODEL_MOT17_S_PATH),
        'mot17_tiny': (WEIGHT_MOT17_TINY_PATH, MODEL_MOT17_TINY_PATH),
        'mot20_x': (WEIGHT_MOT20_X_PATH, MODEL_MOT20_X_PATH),
        'yolox_s': (WEIGHT_YOLOX_S_PATH, MODEL_YOLOX_S_PATH),
        'yolox_tiny': (WEIGHT_YOLOX_TINY_PATH, MODEL_YOLOX_TINY_PATH),
    }
    model_type = args.model_type
    WEIGHT_PATH, MODEL_PATH = dic_model[model_type]

    # model files check and download
    check_and_download_models(WEIGHT_FRID_PATH, MODEL_FRID_PATH, REMOTE_FRID_PATH)
    if args.AFLink:
        check_and_download_models(WEIGHT_AFLINK_PATH, MODEL_AFLINK_PATH, REMOTE_FRID_PATH)
    check_and_download_models(
        WEIGHT_PATH, MODEL_PATH,
        REMOTE_BYTRK_PATH if model_type.startswith('mot') else REMOTE_YOLOX_PATH)

    env_id = args.env_id

    # initialize
    linker = None
    if args.AFLink:
        linker = ailia.Net(MODEL_AFLINK_PATH, WEIGHT_AFLINK_PATH, env_id=env_id, memory_mode=args.memory_mode)
    frid_net = ailia.Net(MODEL_FRID_PATH, WEIGHT_FRID_PATH, env_id=env_id, memory_mode=args.memory_mode)

    det_net = ailia.Net(MODEL_PATH, WEIGHT_PATH, env_id=env_id, memory_mode=args.memory_mode)
    detector = setup_detector(det_net)

    max_cosine_distance = 0.45
    nn_budget = 1
    metric = nn_matching.NearestNeighborDistanceMetric(
        'cosine',
        max_cosine_distance,
        nn_budget
    )
    tracker = Tracker(metric)

    linker = AFLink(
        model=linker,
        thrT=(0, 30),  # (-10, 30) for CenterTrack, FairMOT, TransTrack.
        thrS=75,
        thrP=0.05  # 0.10 for CenterTrack, FairMOT, TransTrack.
    )

    mod = {
        "detector": detector,
        "frid_net": frid_net,
        "tracker": tracker,
        "linker": linker,
    }

    if args.benchmark:
        benchmarking(mod)
    else:
        recognize_from_video(mod)


if __name__ == '__main__':
    main()
