import sys
import time

import numpy as np
import cv2
from PIL import Image
import matplotlib

import ailia

# import original modules
sys.path.append('../../util')
from arg_utils import get_base_parser, update_parser
from model_utils import check_and_download_models  # noqa: E402
from image_utils import normalize_image  # noqa: E402C
from math_utils import softmax  # noqa: E402C
from webcamera_utils import get_capture, get_writer  # noqa: E402
# logger
from logging import getLogger  # noqa: E402

logger = getLogger(__name__)

from this_utils import BBox, box_decode
from this_utils import boxes_cat, boxes_filter
from this_utils import filter_results
from track_utils import track_forward, track_head, track_solver

# ======================
# Parameters
# ======================

WEIGHT_PERSON_RPN_PATH = 'person_rpn.onnx'
MODEL_PERSON_RPN_PATH = 'person_rpn.onnx.prototxt'
WEIGHT_PERSON_BOX_PATH = 'person_box.onnx'
MODEL_PERSON_BOX_PATH = 'person_box.onnx.prototxt'
WEIGHT_PERSON_TRACK_PATH = 'person_track.onnx'
MODEL_PERSON_TRACK_PATH = 'person_track.onnx.prototxt'
WEIGHT_PERSON_FEAT_EXT_PATH = 'person_feat_ext.onnx'
MODEL_PERSON_FEAT_EXT_PATH = 'person_feat_ext.onnx.prototxt'
WEIGHT_VEHICLE_RPN_PATH = 'person_vehicle_rpn.onnx'
MODEL_VEHICLE_RPN_PATH = 'person_vehicle_rpn.onnx.prototxt'
WEIGHT_VEHICLE_BOX_PATH = 'person_vehicle_box.onnx'
MODEL_VEHICLE_BOX_PATH = 'person_vehicle_box.onnx.prototxt'
WEIGHT_VEHICLE_TRACK_PATH = 'person_vehicle_track.onnx'
MODEL_VEHICLE_TRACK_PATH = 'person_vehicle_track.onnx.prototxt'
WEIGHT_VEHICLE_FEAT_EXT_PATH = 'person_vehicle_feat_ext.onnx'
MODEL_VEHICLE_FEAT_EXT_PATH = 'person_vehicle_feat_ext.onnx.prototxt'
REMOTE_PATH = \
    'https://storage.googleapis.com/ailia-models/siam-mot/'

VIDEO_PATH = 'demo.mp4'

IMAGE_HEIGHT = 800
IMAGE_WIDTH = 1280

# ======================
# Arguemnt Parser Config
# ======================

parser = get_base_parser(
    'SiamMOT', VIDEO_PATH, None
)
parser.add_argument(
    '-t', '--track-class', default='person', choices=('person', 'person_vehicle'),
    help='Tracking person or person/vehicle jointly'
)
parser.add_argument(
    '--onnx',
    action='store_true',
    help='execute onnxruntime version.'
)
parser.add_argument(
    '--cui',
    action='store_true',
    help="Don't display preview in GUI."
)
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


def frame_vis_generator(frame, results: BBox):
    ids = results.ids
    results = boxes_filter(results, ids >= 0)
    bbox = results.bbox
    ids = results.ids.tolist()
    labels = results.labels.tolist()

    class_names = ['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat']

    for i, entity_id in enumerate(ids):
        color = vis_colors[int(entity_id) % num_colors]
        class_name = class_names[int(labels[i]) - 1]
        text_width = len(class_name) * 20
        x1, y1, x2, y2 = (np.round(bbox[i, :])).astype(int)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color=color, thickness=3)
        cv2.putText(frame, str(entity_id), (x1 + 5, y1 + 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.5, color, thickness=3)

        # Draw black background rectangle for test
        cv2.rectangle(frame, (x1 - 5, y1 - 25), (x1 + text_width, y1), color, -1)
        cv2.putText(frame, '{}'.format(class_name), (x1 + 5, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), thickness=2)

    return frame


# ======================
# Main functions
# ======================

def preprocess(img):
    h, w = (IMAGE_HEIGHT, IMAGE_WIDTH)
    im_h, im_w, _ = img.shape

    max_orig_size = max(im_h, im_w)
    min_orig_size = min(im_h, im_w)
    if max_orig_size / min_orig_size * h > w:
        size = int(round(w * min_orig_size / max_orig_size))
    else:
        size = h

    if im_h > im_w:
        scale = size / im_w
        ow = size
        oh = (size * im_h) // im_w
    else:
        scale = size / im_h
        oh = size
        ow = (size * im_w) // im_h
    if ow != im_w or oh != im_h:
        img = np.array(Image.fromarray(img).resize((ow, oh), Image.BILINEAR))

    img = normalize_image(img, normalize_type='ImageNet')

    # padding
    new_img = np.zeros((h, w, 3))
    x = (w - ow) // 2
    y = (h - oh) // 2
    new_img[y:y + oh, x:x + ow, :] = img
    img = new_img

    img = img.transpose(2, 0, 1)  # HWC -> CHW
    img = img.astype(np.float32)

    return img, (x, y), scale


def post_processing(
        class_logits, box_regression, bbox,
        ids=None, labels=None):
    prob = softmax(class_logits, -1)

    proposals = box_decode(
        box_regression, bbox,
        weights=(10.0, 10.0, 5.0, 5.0)
    )

    num_classes = prob.shape[1]

    # deafult id is -1
    ids = ids if ids is not None else np.zeros(len(bbox), dtype=int) - 1

    # this only happens for tracks
    if labels is not None and 0 < len(labels):
        # tracks
        track_inds = np.nonzero(ids >= 0)[0]

        # avoid track bbs be suppressed during nms
        if 0 < len(track_inds):
            prob_cp = np.array(prob)
            prob[track_inds, :] = 0.
            prob[track_inds, labels] = prob_cp[track_inds, labels] + 1.

    boxes = BBox(
        bbox=proposals.reshape(-1, 4),
        scores=prob.reshape(-1),
        ids=ids
    )
    boxes.bbox[:, 0] = boxes.bbox[:, 0].clip(0, max=IMAGE_WIDTH - 1)
    boxes.bbox[:, 1] = boxes.bbox[:, 1].clip(0, max=IMAGE_HEIGHT - 1)
    boxes.bbox[:, 2] = boxes.bbox[:, 2].clip(0, max=IMAGE_WIDTH - 1)
    boxes.bbox[:, 3] = boxes.bbox[:, 3].clip(0, max=IMAGE_HEIGHT - 1)

    boxes = filter_results(boxes, num_classes)

    return boxes


def refine_tracks(net, features, tracks):
    """
    Use box head to refine the bounding box location
    The final vis score is an average between appearance and matching score
    """
    if len(tracks[0].bbox) == 0:
        return tracks[0]

    track_scores = tracks[0].scores + 1.

    proposals = tracks[0].bbox.astype(np.float32)
    inputs = features[:4] + [proposals]
    if not args.onnx:
        output = net.predict(inputs)
    else:
        output = net.run(
            None, {k: v for k, v in zip((
                "feature_0", "feature_1", "feature_2", "feature_3", "proposals"),
                inputs)})
    class_logits, box_regression = output
    tracks = post_processing(
        class_logits, box_regression, proposals,
        tracks[0].ids, tracks[0].labels)

    det_scores = tracks.scores
    det_boxes = tracks.bbox

    scores = (det_scores + track_scores) / 2.
    boxes = det_boxes
    r_tracks = BBox(
        bbox=boxes,
        scores=scores,
        ids=tracks.ids,
        labels=tracks.labels,
    )
    return [r_tracks]


def predict(rpn, box, tracker, feat_ext, img, cache={}):
    h, w, _ = img.shape
    img, pad, scale = preprocess(img)

    # feedforward
    if not args.onnx:
        output = rpn.predict([img])
    else:
        output = rpn.run(None, {'img': img})

    features = output[:5]
    proposal = output[5]
    score = output[6]
    boxes = BBox(bbox=proposal, scores=score)

    ### roi_heads.box

    proposals = boxes.bbox
    inputs = features[:4] + [proposals]
    if not args.onnx:
        output = box.predict(inputs)
    else:
        output = box.run(
            None, {k: v for k, v in zip((
                "feature_0", "feature_1", "feature_2", "feature_3", "proposals"),
                inputs)})
    class_logits, box_regression = output
    boxes = post_processing(class_logits, box_regression, proposals)

    ### roi_heads.track

    track_memory = cache.get('x', None)
    y, tracks = track_forward(tracker, features, track_memory, args.onnx)

    if tracks is not None:
        tracks = refine_tracks(box, features, tracks)
        boxes = boxes_cat([boxes] + tracks)

    boxes = track_solver.solve(boxes)

    # get the current state for tracking
    x = track_head.get_track_memory(
        feat_ext, features, boxes, args.onnx)

    cache['x'] = x

    boxes.bbox[:, 0] = (boxes.bbox[:, 0] - pad[0]) / scale
    boxes.bbox[:, 1] = (boxes.bbox[:, 1] - pad[1]) / scale
    boxes.bbox[:, 2] = (boxes.bbox[:, 2] - pad[0]) / scale
    boxes.bbox[:, 3] = (boxes.bbox[:, 3] - pad[1]) / scale

    return boxes


def benchmarking(rpn, box, tracker, feat_ext):
    video_file = args.video if args.video else args.input[0]
    capture = get_capture(video_file)
    assert capture.isOpened(), 'Cannot capture source'

    _, frame = capture.read()
    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    logger.info('BENCHMARK mode')

    total_time_estimation = 0
    for i in range(args.benchmark_count):
        start = int(round(time.time() * 1000))
        predict(rpn, box, tracker, feat_ext, img)
        end = int(round(time.time() * 1000))
        estimation_time = (end - start)

        # Loggin
        logger.info(f'\tailia processing estimation time {estimation_time} ms')
        if i != 0:
            total_time_estimation = total_time_estimation + estimation_time

    logger.info(f'\taverage time estimation {total_time_estimation / (args.benchmark_count - 1)} ms')


def recognize_from_video(rpn, box, tracker, feat_ext):
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

    frame_shown = False
    while True:
        ret, frame = capture.read()
        if (cv2.waitKey(1) & 0xFF == ord('q')) or not ret:
            break
        if frame_shown and cv2.getWindowProperty('frame', cv2.WND_PROP_VISIBLE) == 0:
            break

        # inference
        img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        boxes = predict(rpn, box, tracker, feat_ext, img)

        res_img = frame_vis_generator(frame, boxes)

        # show
        if not args.cui or args.video:
            cv2.imshow('frame', res_img)
            frame_shown = True
        else:
            print("Tracking ids",boxes[2])

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
        'person': (
            WEIGHT_PERSON_RPN_PATH, MODEL_PERSON_RPN_PATH,
            WEIGHT_PERSON_BOX_PATH, MODEL_PERSON_BOX_PATH,
            WEIGHT_PERSON_TRACK_PATH, MODEL_PERSON_TRACK_PATH,
            WEIGHT_PERSON_FEAT_EXT_PATH, MODEL_PERSON_FEAT_EXT_PATH),
        'person_vehicle': (
            WEIGHT_VEHICLE_RPN_PATH, MODEL_VEHICLE_RPN_PATH,
            WEIGHT_VEHICLE_BOX_PATH, MODEL_VEHICLE_BOX_PATH,
            WEIGHT_VEHICLE_TRACK_PATH, MODEL_VEHICLE_TRACK_PATH,
            WEIGHT_VEHICLE_FEAT_EXT_PATH, MODEL_VEHICLE_FEAT_EXT_PATH),
    }
    weight_rpn_path, model_rpn_path, weight_box_path, model_box_path, \
    weight_track_path, model_track_path, weight_feat_path, model_feat_path = dic_model[args.track_class]

    # model files check and download
    logger.info('Checking RPN model...')
    check_and_download_models(weight_rpn_path, model_rpn_path, REMOTE_PATH)
    logger.info('Checking BOX model...')
    check_and_download_models(weight_box_path, model_box_path, REMOTE_PATH)
    logger.info('Checking TRACK model...')
    check_and_download_models(weight_track_path, model_track_path, REMOTE_PATH)
    logger.info('Checking FEAT_EXT model...')
    check_and_download_models(weight_feat_path, model_feat_path, REMOTE_PATH)

    env_id = args.env_id

    # initialize
    if not args.onnx:
        rpn = ailia.Net(model_rpn_path, weight_rpn_path, env_id=env_id, memory_mode=args.memory_mode)
        box = ailia.Net(model_box_path, weight_box_path, env_id=env_id, memory_mode=args.memory_mode)
        tracker = ailia.Net(model_track_path, weight_track_path, env_id=env_id, memory_mode=args.memory_mode)
        feat_ext = ailia.Net(model_feat_path, weight_feat_path, env_id=env_id, memory_mode=args.memory_mode)
    else:
        import onnxruntime
        rpn = onnxruntime.InferenceSession(weight_rpn_path)
        box = onnxruntime.InferenceSession(weight_box_path)
        tracker = onnxruntime.InferenceSession(weight_track_path)
        feat_ext = onnxruntime.InferenceSession(weight_feat_path)

    if args.benchmark:
        benchmarking(rpn, box, tracker, feat_ext)
    else:
        recognize_from_video(rpn, box, tracker, feat_ext)


if __name__ == '__main__':
    main()
