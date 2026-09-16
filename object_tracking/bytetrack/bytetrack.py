import sys
import time

import numpy as np
import cv2
import matplotlib

import ailia

# import original modules
sys.path.append('../../util')
from arg_utils import get_base_parser, update_parser
from model_utils import check_and_download_models  # noqa: E402
from image_utils import normalize_image  # noqa: E402C
from webcamera_utils import get_capture, get_writer  # noqa: E402
# logger
from logging import getLogger  # noqa: E402

logger = getLogger(__name__)

from bytetrack_utils import multiclass_nms

# ======================
# Parameters
# ======================

WEIGHT_MOT17_X_PATH = 'bytetrack_x_mot17.onnx'
MODEL_MOT17_X_PATH = 'bytetrack_x_mot17.onnx.prototxt'
WEIGHT_MOT17_S_PATH = 'bytetrack_s_mot17.onnx'
MODEL_MOT17_S_PATH = 'bytetrack_s_mot17.onnx.prototxt'
WEIGHT_MOT17_TINY_PATH = 'bytetrack_tiny_mot17.onnx'
MODEL_MOT17_TINY_PATH = 'bytetrack_tiny_mot17.onnx.prototxt'
WEIGHT_MOT20_X_PATH = 'bytetrack_x_mot20.onnx'
MODEL_MOT20_X_PATH = 'bytetrack_x_mot20.onnx.prototxt'
REMOTE_PATH = \
    'https://storage.googleapis.com/ailia-models/bytetrack/'

WEIGHT_YOLOX_S_PATH = 'yolox_s.opt.onnx'
MODEL_YOLOX_S_PATH = 'yolox_s.opt.onnx.prototxt'
WEIGHT_YOLOX_TINY_PATH = 'yolox_tiny.opt.onnx'
MODEL_YOLOX_TINY_PATH = 'yolox_tiny.opt.onnx.prototxt'
REMOTE_YOLOX_PATH = \
    'https://storage.googleapis.com/ailia-models/yolox/'

VIDEO_PATH = 'demo.mp4'

IMAGE_MOT17_X_HEIGHT = 800
IMAGE_MOT17_X_WIDTH = 1440
IMAGE_MOT17_S_HEIGHT = 608
IMAGE_MOT17_S_WIDTH = 1088
IMAGE_MOT17_TINY_HEIGHT = 416
IMAGE_MOT17_TINY_WIDTH = 416
IMAGE_MOT20_X_HEIGHT = 896
IMAGE_MOT20_X_WIDTH = 1600
IMAGE_YOLOX_S_HEIGHT = 640
IMAGE_YOLOX_S_WIDTH = 640
IMAGE_YOLOX_TINY_HEIGHT = 416
IMAGE_YOLOX_TINY_WIDTH = 416

# ======================
# Arguemnt Parser Config
# ======================

parser = get_base_parser(
    'ByteTrack', VIDEO_PATH, None
)
parser.add_argument(
    "--score_thre", type=float, default=0.1,
    help="Score threshould to filter the result.",
)
parser.add_argument(
    "--nms_thre", type=float, default=0.7,
    help="NMS threshould.",
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
parser.add_argument("--track_thresh", type=float, default=0.5, help="tracking confidence threshold")
parser.add_argument("--track_buffer", type=int, default=30, help="the frames for keep lost tracks")
parser.add_argument("--match_thresh", type=float, default=0.8, help="matching threshold for tracking")
parser.add_argument('--min-box-area', type=float, default=10, help='filter out tiny boxes')
args = update_parser(parser)


# ======================
# Secondaty Functions
# ======================

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

def preprocess(img, img_size, normalize=True):
    h, w = img_size
    im_h, im_w, _ = img.shape

    r = min(h / im_h, w / im_w)
    oh, ow = int(im_h * r), int(im_w * r)

    resized_img = cv2.resize(
        img,
        (ow, oh),
        interpolation=cv2.INTER_LINEAR,
    )

    img = np.ones((h, w, 3)) * 114.0
    img[: oh, : ow] = resized_img

    if normalize:
        img = img[:, :, ::-1]  # BGR -> RGB
        img = normalize_image(img, 'ImageNet')

    img = img.transpose((2, 0, 1))
    img = np.expand_dims(img, axis=0)
    img = img.astype(np.float32)

    return img, r


def postprocess(output, ratio, img_size, p6=False, nms_thre=0.7, score_thre=0.1):
    grids = []
    expanded_strides = []

    if not p6:
        strides = [8, 16, 32]
    else:
        strides = [8, 16, 32, 64]

    hsizes = [img_size[0] // stride for stride in strides]
    wsizes = [img_size[1] // stride for stride in strides]

    for hsize, wsize, stride in zip(hsizes, wsizes, strides):
        xv, yv = np.meshgrid(np.arange(wsize), np.arange(hsize))
        grid = np.stack((xv, yv), 2).reshape(1, -1, 2)
        grids.append(grid)
        shape = grid.shape[:2]
        expanded_strides.append(np.full((*shape, 1), stride))

    grids = np.concatenate(grids, 1)
    expanded_strides = np.concatenate(expanded_strides, 1)
    output[..., :2] = (output[..., :2] + grids) * expanded_strides
    output[..., 2:4] = np.exp(output[..., 2:4]) * expanded_strides

    predictions = output[0]

    boxes = predictions[:, :4]
    scores = predictions[:, 4:5] * predictions[:, 5:]

    boxes_xyxy = np.ones_like(boxes)
    boxes_xyxy[:, 0] = boxes[:, 0] - boxes[:, 2] / 2.
    boxes_xyxy[:, 1] = boxes[:, 1] - boxes[:, 3] / 2.
    boxes_xyxy[:, 2] = boxes[:, 0] + boxes[:, 2] / 2.
    boxes_xyxy[:, 3] = boxes[:, 1] + boxes[:, 3] / 2.
    boxes_xyxy /= ratio

    dets = multiclass_nms(boxes_xyxy, scores, nms_thr=nms_thre, score_thr=score_thre)

    return dets[:, :-1] if dets is not None else np.zeros((0, 5))


def predict(net, img):
    dic_model = {
        'mot17_x': (IMAGE_MOT17_X_HEIGHT, IMAGE_MOT17_X_WIDTH),
        'mot17_s': (IMAGE_MOT17_S_HEIGHT, IMAGE_MOT17_S_WIDTH),
        'mot17_tiny': (IMAGE_MOT17_TINY_HEIGHT, IMAGE_MOT17_TINY_WIDTH),
        'mot20_x': (IMAGE_MOT20_X_HEIGHT, IMAGE_MOT20_X_WIDTH),
        'yolox_s': (IMAGE_YOLOX_S_HEIGHT, IMAGE_YOLOX_S_WIDTH),
        'yolox_tiny': (IMAGE_YOLOX_TINY_HEIGHT, IMAGE_YOLOX_TINY_WIDTH),
    }
    model_type = args.model_type
    img_size = dic_model[model_type]

    img, ratio = preprocess(img, img_size, normalize=model_type.startswith('mot'))

    # feedforward
    output = net.predict([img])
    output = output[0]

    # For yolox, retrieve only the person class
    output = output[..., :6]

    score_thre = args.score_thre
    nms_thre = args.nms_thre
    dets = postprocess(output, ratio, img_size, nms_thre=nms_thre, score_thre=score_thre)

    return dets


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


def recognize_from_video(net):
    min_box_area = args.min_box_area
    mot20 = args.model_type == 'mot20'

    video_file = args.video if args.video else args.input[0]
    capture = get_capture(video_file)
    assert capture.isOpened(), 'Cannot capture source'

    # create video writer if savepath is specified as video format
    f_h = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    f_w = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    if args.savepath != None:
        logger.warning(
            'currently, video results cannot be output correctly...'
        )
        writer = get_writer(args.savepath, f_h, f_w)
    else:
        writer = None

    tracker = BYTETracker(
        track_thresh=args.track_thresh, track_buffer=args.track_buffer,
        match_thresh=args.match_thresh, frame_rate=30,
        mot20=mot20)

    frame_shown = False
    while True:
        ret, frame = capture.read()
        if (cv2.waitKey(1) & 0xFF == ord('q')) or not ret:
            break
        if frame_shown and cv2.getWindowProperty('frame', cv2.WND_PROP_VISIBLE) == 0:
            break

        # inference
        output = predict(net, frame)

        # run tracking
        online_targets = tracker.update(output)
        online_tlwhs = []
        online_ids = []
        online_scores = []
        for t in online_targets:
            tlwh = t.tlwh
            tid = t.track_id
            vertical = tlwh[2] / tlwh[3] > 1.6
            if tlwh[2] * tlwh[3] > min_box_area and not vertical:
                online_tlwhs.append(tlwh)
                online_ids.append(tid)
                online_scores.append(t.score)

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
    weight_path, model_path = dic_model[model_type]

    # model files check and download
    check_and_download_models(
        weight_path, model_path,
        REMOTE_PATH if model_type.startswith('mot') else REMOTE_YOLOX_PATH)

    env_id = args.env_id

    # initialize
    net = ailia.Net(model_path, weight_path, env_id=env_id, memory_mode=args.memory_mode)

    if args.benchmark:
        benchmarking(net)
    else:
        recognize_from_video(net)


if __name__ == '__main__':
    from tracker.byte_tracker import BYTETracker

    main()
