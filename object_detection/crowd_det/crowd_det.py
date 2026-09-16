import sys
import time

import numpy as np
import cv2

import ailia

# import original modules
sys.path.append('../../util')
from arg_utils import get_base_parser, update_parser, get_savepath  # noqa
from model_utils import check_and_download_models  # noqa
from detector_utils import load_image, plot_results  # noqa
from webcamera_utils import get_capture, get_writer  # noqa
from nms_utils import nms_boxes
# logger
from logging import getLogger  # noqa

logger = getLogger(__name__)

# ======================
# Parameters
# ======================

WEIGHT_FPN_BASE_PATH = 'rcnn_fpn_baseline_mge.onnx'
MODEL_FPN_BASE_PATH = 'rcnn_fpn_baseline_mge.onnx.prototxt'
WEIGHT_EMD_SIMPLE_PATH = 'rcnn_emd_simple_mge.onnx'
MODEL_EMD_SIMPLE_PATH = 'rcnn_emd_simple_mge.onnx.prototxt'
WEIGHT_EMD_REFINE_PATH = 'rcnn_emd_refine_mge.onnx'
MODEL_EMD_REFINE_PATH = 'rcnn_emd_refine_mge.onnx.prototxt'
REMOTE_PATH = 'https://storage.googleapis.com/ailia-models/crowd_det/'

IMAGE_PATH = 'demo.jpg'
SAVE_IMAGE_PATH = 'output.png'

# ======================
# Arguemnt Parser Config
# ======================

parser = get_base_parser(
    'Detection in Crowded Scenes', IMAGE_PATH, SAVE_IMAGE_PATH
)
parser.add_argument(
    '-m', '--model_type', default='rcnn_fpn_baseline',
    choices=('rcnn_fpn_baseline', 'rcnn_emd_simple', 'rcnn_emd_refine'),
    help='model type'
)
parser.add_argument(
    '-t', '--threshold', type=float, default=0.3,
    help='threshold'
)
parser.add_argument(
    '-it', '--iou_threshold', type=float, default=0.5,
    help='iou_threshold'
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

def set_cpu_nms(dets, thresh):
    """Pure Python NMS baseline."""

    def _overlap(det_boxes, basement, others):
        eps = 1e-8
        x1_basement, y1_basement, x2_basement, y2_basement \
            = det_boxes[basement, 0], det_boxes[basement, 1], \
              det_boxes[basement, 2], det_boxes[basement, 3]
        x1_others, y1_others, x2_others, y2_others \
            = det_boxes[others, 0], det_boxes[others, 1], \
              det_boxes[others, 2], det_boxes[others, 3]
        areas_basement = (x2_basement - x1_basement) * (y2_basement - y1_basement)
        areas_others = (x2_others - x1_others) * (y2_others - y1_others)
        xx1 = np.maximum(x1_basement, x1_others)
        yy1 = np.maximum(y1_basement, y1_others)
        xx2 = np.minimum(x2_basement, x2_others)
        yy2 = np.minimum(y2_basement, y2_others)
        w = np.maximum(0.0, xx2 - xx1)
        h = np.maximum(0.0, yy2 - yy1)
        inter = w * h
        ovr = inter / (areas_basement + areas_others - inter + eps)
        return ovr

    scores = dets[:, 4]
    order = np.argsort(-scores)
    dets = dets[order]

    numbers = dets[:, -1]
    keep = np.ones(len(dets)) == 1
    ruler = np.arange(len(dets))
    while ruler.size > 0:
        basement = ruler[0]
        ruler = ruler[1:]
        num = numbers[basement]
        # calculate the body overlap
        overlap = _overlap(dets[:, :4], basement, ruler)
        indices = np.where(overlap > thresh)[0]
        loc = np.where(numbers[ruler][indices] == num)[0]
        # the mask won't change in the step
        mask = keep[ruler[indices][loc]]
        keep[ruler[indices]] = False
        keep[ruler[indices][loc][mask]] = True
        ruler[~keep[ruler]] = -1
        ruler = ruler[ruler > 0]

    keep = keep[np.argsort(order)]
    return keep


# ======================
# Main functions
# ======================

def preprocess(img):
    short_size, max_size = 800, 1400

    im_h, im_w, _ = img.shape

    im_min = min(im_h, im_w)
    im_max = max(im_h, im_w)
    scale = short_size / im_min
    if scale * im_max > max_size:
        scale = max_size / im_max
    ow, oh = int(round(im_w * scale)), int(round(im_h * scale))
    if ow != im_w or oh != im_h:
        img = cv2.resize(img, (ow, oh), interpolation=cv2.INTER_LINEAR)

    img = img.transpose(2, 0, 1)  # HWC -> CHW
    img = np.expand_dims(img, axis=0)
    img = img.astype(np.float32)

    return img, scale


def post_processing(pred_boxes):
    pred_cls_threshold = 0.01
    visual_thresh = args.threshold
    nms_thres = args.iou_threshold

    if args.model_type in ('rcnn_emd_simple', 'rcnn_emd_refine'):
        top_k = pred_boxes.shape[-1] // 6
        n = pred_boxes.shape[0]

        pred_boxes = pred_boxes.reshape(-1, 6)
        idents = np.tile(np.arange(n)[:, None], (1, top_k)).reshape(-1, 1)
        pred_boxes = np.hstack((pred_boxes, idents))
        keep = pred_boxes[:, 4] > pred_cls_threshold
        pred_boxes = pred_boxes[keep]
        keep = set_cpu_nms(pred_boxes, nms_thres)
        pred_boxes = pred_boxes[keep]
    else:
        pred_boxes = pred_boxes.reshape(-1, 6)
        keep = pred_boxes[:, 4] > pred_cls_threshold
        pred_boxes = pred_boxes[keep]
        keep = nms_boxes(pred_boxes, pred_boxes[:, 4], nms_thres)
        pred_boxes = pred_boxes[keep]

    keep = pred_boxes[:, 4] > visual_thresh
    pred_boxes = pred_boxes[keep]

    return pred_boxes


def convert_to_detector_object(preds, im_w, im_h):
    detector_object = []
    for i in range(len(preds)):
        (x1, y1, w, h) = preds[i, :4]
        score = float(preds[i, 4])

        r = ailia.DetectorObject(
            category="person",
            prob=score,
            x=x1 / im_w,
            y=y1 / im_h,
            w=w / im_w,
            h=h / im_h,
        )
        detector_object.append(r)

    return detector_object


def predict(net, img):
    im_h, im_w, _ = img.shape
    img, scale = preprocess(img)
    _, _, oh, ow = img.shape
    im_info = np.array([[oh, ow, scale, im_h, im_w, 0]])

    # feedforward
    if not args.onnx:
        output = net.predict([img, im_info])
    else:
        output = net.run(None, {'img': img, 'im_info': im_info})

    pred_boxes = output[0]
    pred_boxes = post_processing(pred_boxes)
    pred_boxes[:, :4] /= scale
    pred_boxes[:, 2:4] -= pred_boxes[:, :2]

    return pred_boxes


def recognize_from_image(net):
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

                # Logging
                logger.info(f'\tailia processing estimation time {estimation_time} ms')
                if i != 0:
                    total_time_estimation = total_time_estimation + estimation_time

            logger.info(f'\taverage time estimation {total_time_estimation / (args.benchmark_count - 1)} ms')
        else:
            pred = predict(net, img)

        # plot result
        detect_object = convert_to_detector_object(pred, img.shape[1], img.shape[0])
        res_img = plot_results(detect_object, img)

        savepath = get_savepath(args.savepath, image_path, ext='.png')
        logger.info(f'saved at : {savepath}')
        cv2.imwrite(savepath, res_img)

    logger.info('Script finished successfully.')


def recognize_from_video(net):
    video_file = args.video if args.video else args.input[0]
    capture = get_capture(video_file)
    assert capture.isOpened(), 'Cannot capture source'

    # create video writer if savepath is specified as video format
    if args.savepath != SAVE_IMAGE_PATH:
        f_h = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        f_w = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
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
        img = frame
        pred = predict(net, img)

        # plot result
        detect_object = convert_to_detector_object(pred, img.shape[1], img.shape[0])
        res_img = plot_results(detect_object, img)

        # show
        cv2.imshow('frame', res_img)
        frame_shown = True

        # save results
        if writer is not None:
            res_img = res_img.astype(np.uint8)
            writer.write(res_img)

    capture.release()
    cv2.destroyAllWindows()
    if writer is not None:
        writer.release()

    logger.info('Script finished successfully.')


def main():
    dic_model = {
        'rcnn_fpn_baseline': (WEIGHT_FPN_BASE_PATH, MODEL_FPN_BASE_PATH),
        'rcnn_emd_simple': (WEIGHT_EMD_SIMPLE_PATH, MODEL_EMD_SIMPLE_PATH),
        'rcnn_emd_refine': (WEIGHT_EMD_REFINE_PATH, MODEL_EMD_REFINE_PATH),
    }
    weight_path, model_path = dic_model[args.model_type]

    # model files check and download
    check_and_download_models(weight_path, model_path, REMOTE_PATH)

    env_id = args.env_id

    # initialize
    if not args.onnx:
        net = ailia.Net(model_path, weight_path, env_id=env_id, memory_mode=args.memory_mode)
    else:
        import onnxruntime
        net = onnxruntime.InferenceSession(weight_path)

    if args.video is not None:
        recognize_from_video(net)
    else:
        recognize_from_image(net)


if __name__ == '__main__':
    main()
