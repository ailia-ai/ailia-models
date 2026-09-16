import sys
import time
import yaml

import numpy as np
import cv2
import json

import ailia

# import original modules
sys.path.append('../../util')
from arg_utils import get_base_parser, update_parser, get_savepath  # noqa: E402
from model_utils import check_and_download_models  # noqa: E402
from detector_utils import load_image  # noqa: E402C
from nms_utils import nms_boxes  # noqa: E402C
from webcamera_utils import get_capture, get_writer  # noqa: E402
# logger
from logging import getLogger  # noqa: E402

from post_utils import DmPost

logger = getLogger(__name__)

# ======================
# Parameters
# ======================

WEIGHT_PATH = 'cdnet.onnx'
MODEL_PATH = 'cdnet.onnx.prototxt'
REMOTE_PATH = 'https://storage.googleapis.com/ailia-models/cdnet/'

IMAGE_PATH = 'example/filename_00038.jpg'
SAVE_IMAGE_PATH = 'output.png'

IMAGE_SIZE = 640

THRESHOLD = 0.4
IOU = 0.5

names = ['crosswalk', 'guide_arrows']

# ======================
# Arguemnt Parser Config
# ======================

parser = get_base_parser(
    'CDNet', IMAGE_PATH, SAVE_IMAGE_PATH
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
    '--plot-classes', metavar='NAME', nargs='+',
    default=('crosswalk',),
    help='specifies which classes will be drawn. it is selected from (crosswalk, guide_arrows)'
)
parser.add_argument(
    '--control-line-setting', type=str,
    default='settings/cl_setting.yaml',
    help='control line setting'
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

def init_clrl():
    conf_file = args.control_line_setting
    with open(conf_file, 'r') as f:
        conls = yaml.load(f, Loader=yaml.FullLoader)

    return conls


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def make_grid(nx=20, ny=20):
    xv, yv = np.meshgrid(np.arange(nx), np.arange(ny))
    xy = np.stack((xv, yv), axis=2).reshape((1, 1, ny, nx, 2))
    xy = xy.astype(np.float32)

    return xy


def xywh2xyxy(x):
    y = np.zeros_like(x)
    y[:, 0] = x[:, 0] - x[:, 2] / 2  # top left x
    y[:, 1] = x[:, 1] - x[:, 3] / 2  # top left y
    y[:, 2] = x[:, 0] + x[:, 2] / 2  # bottom right x
    y[:, 3] = x[:, 1] + x[:, 3] / 2  # bottom right y

    return y


def non_max_suppression(
        prediction,
        conf_thres=0.1, iou_thres=0.6,
        classes=None, agnostic=False):
    """Performs Non-Maximum Suppression (NMS) on inference results

    Returns:
         detections with shape: nx6 (x1, y1, x2, y2, conf, cls)
    """

    nc = prediction[0].shape[1] - 5  # number of classes
    xc = prediction[..., 4] > conf_thres  # candidates

    # Settings
    min_wh, max_wh = 2, 4096  # (pixels) minimum and maximum box width and height
    max_det = 300  # maximum number of detections per image

    multi_label = nc > 1  # multiple labels per box (adds 0.5ms/img)

    output = [None] * prediction.shape[0]
    for xi, x in enumerate(prediction):  # image index, image inference
        # Apply constraints
        x = x[xc[xi]]  # confidence

        # If none remain process next image
        if not x.shape[0]:
            continue

        # Compute conf
        x[:, 5:] *= x[:, 4:5]  # conf = obj_conf * cls_conf

        # Box (center x, center y, width, height) to (x1, y1, x2, y2)
        box = xywh2xyxy(x[:, :4])

        # Detections matrix nx6 (xyxy, conf, cls)
        if multi_label:
            i, j = (x[:, 5:] > conf_thres).nonzero()
            x = np.concatenate((box[i], x[i, j + 5, None], j[:, None]), axis=1)
        else:  # best class only
            conf = x[:, 5:].max(1, keepdims=True)
            j = x[:, 5:].argmax(axis=1).reshape(-1, 1)
            x = np.concatenate((box, conf, j), axis=1)
            x = x[conf.reshape(-1) > conf_thres]

        # Filter by class
        if classes:
            idx = (x[:, 5:6] == np.array(classes)).any(axis=1)
            x = x[idx]

        # If none remain process next image
        n = x.shape[0]  # number of boxes
        if not n:
            continue

        # Batched NMS
        c = x[:, 5:6] * (0 if agnostic else max_wh)  # classes
        boxes, scores = x[:, :4] + c, x[:, 4]  # boxes (offset by class), scores
        idx = nms_boxes(boxes, scores, iou_thres)
        idx = idx[:max_det]

        output[xi] = x[idx]

    return output


def plot_one_box(x, img, color, label=None, line_thickness=None):
    # Plots one bounding box on image img
    tl = line_thickness or round(0.002 * (img.shape[0] + img.shape[1]) / 2) + 1  # line/font thickness
    c1, c2 = (int(x[0]), int(x[1])), (int(x[2]), int(x[3]))
    cv2.rectangle(img, c1, c2, color, thickness=tl, lineType=cv2.LINE_AA)
    if label:
        tf = max(tl - 1, 1)  # font thickness
        t_size = cv2.getTextSize(label, 0, fontScale=tl / 3, thickness=tf)[0]
        c2 = c1[0] + t_size[0], c1[1] - t_size[1] - 3
        cv2.rectangle(img, c1, c2, color, -1, cv2.LINE_AA)  # filled
        cv2.putText(img, label, (c1[0], c1[1] - 2), 0, tl / 3, [225, 255, 255], thickness=tf, lineType=cv2.LINE_AA)


def draw_result(img, pred):
    plot_classes = args.plot_classes

    if pred is None:
        return img

    for *xyxy, conf, cls in pred:
        label = '%s %.2f' % (names[int(cls)], conf)
        if names[int(cls)] in plot_classes:
            color = (255, 85, 33)
            plot_one_box(xyxy, img, label=label, color=color, line_thickness=5)

    return img


def save_result_json(json_path, pred):
    plot_classes = args.plot_classes
    results = []
    if pred is not None:
        for *xyxy, conf, class_id in pred:
            class_name = names[int(class_id)]
            if class_name in plot_classes:
                results.append({
                    'box': xyxy, 'class': class_name, 'conf': conf
                })
    with open(json_path, 'w') as f:
        json.dump(results, f, indent=2)


# ======================
# Main functions
# ======================

cache = {
    "grid": [None, None, None],
    "anchor_grid": np.load("anchor_grid.npy")
}

clrl = init_clrl()


def preprocess(img):
    h, w = (IMAGE_SIZE, IMAGE_SIZE)
    im_h, im_w, _ = img.shape

    # resize
    r = min((h / im_h), (w / im_w))
    ow, oh = int(im_w * r), int(im_h * r)
    if ow != im_w or oh != im_h:
        img = cv2.resize(img, (ow, oh), interpolation=cv2.INTER_LINEAR)

    control_line = clrl['control_line']
    x1, x2 = 0, ow
    y1, y2 = control_line[0] / im_h * oh, control_line[1] / im_h * oh
    img = img[int(y1):int(y2), :]

    rest_h = img.shape[0] % 32
    rest_w = img.shape[1] % 32
    dh = 0 if rest_h == 0 else (32 - rest_h) / 2
    dw = 0 if rest_w == 0 else (32 - rest_w) / 2
    ofs_xy = (int(x1) - dw, int(y1) - dh)

    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))

    # add border
    color = (114, 114, 114)
    img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)

    img = img[:, :, ::-1]  # BGR -> RGB
    img = img / 255
    img = img.transpose(2, 0, 1)  # HWC -> CHW
    img = np.expand_dims(img, axis=0)
    img = img.astype(np.float32)

    return img, r, ofs_xy


def post_processing(x):
    stride = [8, 16, 32]
    grid = cache['grid']
    anchor_grid = cache['anchor_grid']

    z = []
    for i, _ in enumerate(x):
        bs, _, ny, nx, _ = x[i].shape
        if grid[i] is None or grid[i].shape[2:4] != x[i].shape[2:4]:
            grid[i] = make_grid(nx, ny)

        y = sigmoid(x[i])
        y[..., 0:2] = (y[..., 0:2] * 2. - 0.5 + grid[i]) * stride[i]  # xy
        y[..., 2:4] = (y[..., 2:4] * 2) ** 2 * anchor_grid[i]  # wh
        z.append(y.reshape((bs, -1, 7)))

    z = np.concatenate(z, axis=1)

    return z


def predict(net, img):
    conf_thres = args.threshold
    iou_thres = args.iou

    img, r, xy = preprocess(img)

    # feedforward
    output = net.predict([img])

    pred = post_processing(output)

    pred = non_max_suppression(
        pred, conf_thres, iou_thres)

    pred = pred[0]
    if pred is not None:
        pred[:, 0], pred[:, 2] = pred[:, 0] + xy[0], pred[:, 2] + xy[0]
        pred[:, 1], pred[:, 3] = pred[:, 1] + xy[1], pred[:, 3] + xy[1]
        pred[:, :4] = (pred[:, :4] / r).round()

    return pred


def recognize_from_image(net):
    dp = DmPost(clrl)

    # input image loop
    for image_path in args.input:
        logger.info(image_path)

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
                pred = predict(net, img)
                end = int(round(time.time() * 1000))
                estimation_time = (end - start)

                # Loggin
                logger.info(f'\tailia processing estimation time {estimation_time} ms')
                if i != 0:
                    total_time_estimation = total_time_estimation + estimation_time

            logger.info(f'\taverage time estimation {total_time_estimation / (args.benchmark_count - 1)} ms')
        else:
            pred = predict(net, img)

        img = draw_result(img, pred)
        if 1 < len(args.input):
            res_img = dp.dmpost(img, pred, names=names)
        else:
            res_img = img

        # plot result
        savepath = get_savepath(args.savepath, image_path, ext='.png')
        logger.info(f'saved at : {savepath}')
        cv2.imwrite(savepath, res_img)

        if args.write_json:
            json_file = '%s.json' % savepath.rsplit('.', 1)[0]
            save_result_json(json_file, pred)

    logger.info('Script finished successfully.')


def recognize_from_video(net):
    video_file = args.video if args.video else args.input[0]
    capture = get_capture(video_file)
    assert capture.isOpened(), 'Cannot capture source'

    # create video writer if savepath is specified as video format
    f_h = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    f_w = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    if args.savepath != SAVE_IMAGE_PATH:
        logger.warning(
            'currently, video results cannot be output correctly...'
        )
        writer = get_writer(args.savepath, f_h, f_w)
    else:
        writer = None

    dp = DmPost(clrl)
    frame_shown = False
    while True:
        ret, frame = capture.read()
        if (cv2.waitKey(1) & 0xFF == ord('q')) or not ret:
            break
        if frame_shown and cv2.getWindowProperty('frame', cv2.WND_PROP_VISIBLE) == 0:
            break

        # inference
        img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pred = predict(net, img)

        img = draw_result(img, pred)
        res_img = dp.dmpost(img, pred, names=names)

        # show
        cv2.imshow('frame', res_img)
        frame_shown = True

        # save results
        if writer is not None:
            writer.write(res_img.astype(np.uint8))

    capture.release()
    cv2.destroyAllWindows()
    if writer is not None:
        writer.release()

    logger.info('Script finished successfully.')


def main():
    # model files check and download
    check_and_download_models(WEIGHT_PATH, MODEL_PATH, REMOTE_PATH)

    env_id = args.env_id

    # initialize
    net = ailia.Net(MODEL_PATH, WEIGHT_PATH, env_id=env_id, memory_mode=args.memory_mode)

    if args.video is not None:
        recognize_from_video(net)
    else:
        recognize_from_image(net)


if __name__ == '__main__':
    main()
