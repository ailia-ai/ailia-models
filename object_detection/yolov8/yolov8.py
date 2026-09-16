import sys
import time

import cv2
import numpy as np

import ailia

# import original modules
sys.path.append('../../util')
from arg_utils import get_base_parser, update_parser, get_savepath  # noqa
from model_utils import check_and_download_models  # noqa
from image_utils import normalize_image  # noqa
from detector_utils import plot_results, write_predictions, load_image  # noqa
from nms_utils import batched_nms
from webcamera_utils import get_capture, get_writer  # noqa

# logger
from logging import getLogger  # noqa

logger = getLogger(__name__)

# ======================
# Parameters
# ======================

WEIGHT_YOLOV8N_PATH = 'yolov8n.onnx'
MODEL_YOLOV8N_PATH = 'yolov8n.onnx.prototxt'
WEIGHT_YOLOV8S_PATH = 'yolov8s.onnx'
MODEL_YOLOV8S_PATH = 'yolov8s.onnx.prototxt'
WEIGHT_YOLOV8M_PATH = 'yolov8m.onnx'
MODEL_YOLOV8M_PATH = 'yolov8m.onnx.prototxt'
WEIGHT_YOLOV8L_PATH = 'yolov8l.onnx'
MODEL_YOLOV8L_PATH = 'yolov8l.onnx.prototxt'
WEIGHT_YOLOV8X_PATH = 'yolov8x.onnx'
MODEL_YOLOV8X_PATH = 'yolov8x.onnx.prototxt'
REMOTE_PATH = 'https://storage.googleapis.com/ailia-models/yolov8/'

IMAGE_PATH = 'demo.jpg'
SAVE_IMAGE_PATH = 'output.png'

COCO_CATEGORY = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train",
    "truck", "boat", "traffic light", "fire hydrant", "stop sign",
    "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow",
    "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
    "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard",
    "sports ball", "kite", "baseball bat", "baseball glove", "skateboard",
    "surfboard", "tennis racket", "bottle", "wine glass", "cup", "fork",
    "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange",
    "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair",
    "couch", "potted plant", "bed", "dining table", "toilet", "tv",
    "laptop", "mouse", "remote", "keyboard", "cell phone", "microwave",
    "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase",
    "scissors", "teddy bear", "hair drier", "toothbrush"
]

THRESHOLD = 0.25
IOU = 0.7
DETECTION_SIZE = 640

# ======================
# Arguemnt Parser Config
# ======================

parser = get_base_parser('Ultralytics YOLOv8', IMAGE_PATH, SAVE_IMAGE_PATH)
parser.add_argument(
    '-th', '--threshold',
    default=THRESHOLD, type=float,
    help='The detection threshold for yolo.'
)
parser.add_argument(
    '-iou', '--iou',
    default=IOU, type=float,
    help='The detection iou for yolo.'
)
parser.add_argument(
    '-w', '--write_prediction',
    nargs='?',
    const='txt',
    choices=['txt', 'json'],
    type=str,
    help='Output results to txt or json file.'
)
parser.add_argument(
    '-ds', '--detection_size',
    default=DETECTION_SIZE, type=int,
    help='The detection width and height for yolo.'
)
parser.add_argument(
    '-m', '--model_type', default='v8n',
    choices=('v8n', 'v8s', 'v8m', 'v8l', 'v8x'),
    help='model type'
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

def xywh2xyxy(x):
    y = np.copy(x)
    y[..., 0] = x[..., 0] - x[..., 2] / 2  # top left x
    y[..., 1] = x[..., 1] - x[..., 3] / 2  # top left y
    y[..., 2] = x[..., 0] + x[..., 2] / 2  # bottom right x
    y[..., 3] = x[..., 1] + x[..., 3] / 2  # bottom right y

    return y


def scale_boxes(img1_shape, boxes, img0_shape):
    """
    Rescales bounding boxes (in the format of xyxy) from the shape of the image they were originally specified in
    (img1_shape) to the shape of a different image (img0_shape).

    Args:
      img1_shape (tuple): The shape of the image that the bounding boxes are for, in the format of (height, width).
      boxes (numpy.ndarray): the bounding boxes of the objects in the image, in the format of (x1, y1, x2, y2)
      img0_shape (tuple): the shape of the target image, in the format of (height, width).
    Returns:
      boxes (numpy.ndarray): The scaled bounding boxes, in the format of (x1, y1, x2, y2)
    """
    gain = min(img1_shape[0] / img0_shape[0], img1_shape[1] / img0_shape[1])  # gain  = old / new
    pad = (img1_shape[1] - img0_shape[1] * gain) / 2, (img1_shape[0] - img0_shape[0] * gain) / 2  # wh padding

    boxes[..., [0, 2]] -= pad[0]  # x padding
    boxes[..., [1, 3]] -= pad[1]  # y padding
    boxes[..., :4] /= gain

    boxes[..., [0, 2]] = boxes[..., [0, 2]].clip(0, img0_shape[1])  # x1, x2
    boxes[..., [1, 3]] = boxes[..., [1, 3]].clip(0, img0_shape[0])  # y1, y2

    return boxes


# ======================
# Main functions
# ======================

def preprocess(img):
    im_h, im_w, _ = img.shape
    size = args.detection_size

    r = min(size / im_h, size / im_w)
    oh, ow = int(round(im_h * r)), int(round(im_w * r))
    if ow != im_w or oh != im_h:
        img = cv2.resize(img, (ow, oh), interpolation=cv2.INTER_LINEAR)

    dh, dw = size - oh, size - ow
    if True:
        stride = 32
        dw, dh = np.mod(dw, stride), np.mod(dh, stride)  # wh padding
    # divide padding into 2 sides
    dw /= 2
    dh /= 2

    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    img = cv2.copyMakeBorder(
        img, top, bottom, left, right, cv2.BORDER_CONSTANT,
        value=(114, 114, 114))  # add border

    img = normalize_image(img, normalize_type='255')

    img = img.transpose((2, 0, 1))  # HWC -> CHW
    img = np.expand_dims(img, axis=0)
    img = img.astype(np.float32)

    return img


def post_processing(preds, img, orig_shape):
    conf_thres = args.threshold
    iou_thres = args.iou

    xc = np.max(preds[:, 4:], axis=1) > conf_thres

    none_out = np.zeros((0, 6))

    x = preds[0].T[xc[0]]  # confidence
    if not x.shape[0]:
        return none_out

    box, cls = np.split(x, [4], axis=1)
    box = xywh2xyxy(box)  # center_x, center_y, width, height) to (x1, y1, x2, y2)

    j = np.argmax(cls, axis=1)
    conf = cls[np.arange(len(cls)), j]

    x = np.concatenate((box, conf.reshape(-1, 1), j.reshape(-1, 1)), axis=1)

    # Check shape
    n = x.shape[0]  # number of boxes
    if not n:  # no boxes
        return none_out

    # sort by confidence and remove excess boxes
    max_nms = 30000
    x = x[np.argsort(-x[:, 4])[:max_nms]]

    c = x[:, 5]
    boxes, scores = x[:, :4], x[:, 4]  # boxes, scores

    # Batched NMS
    i = batched_nms(boxes, scores, c, iou_thres)

    max_det = 300
    i = i[:max_det]  # limit detections
    preds = x[i]

    preds[:, :4] = np.round(scale_boxes(img.shape[2:], preds[:, :4], orig_shape))

    return preds


def convert_to_detector_object(preds, im_w, im_h):
    detector_object = []
    for i in range(len(preds)):
        (x1, y1, x2, y2) = preds[i, :4]
        score = float(preds[i, 4])
        cls = int(preds[i, 5])

        r = ailia.DetectorObject(
            category=COCO_CATEGORY[cls],
            prob=score,
            x=x1 / im_w,
            y=y1 / im_h,
            w=(x2 - x1) / im_w,
            h=(y2 - y1) / im_h,
        )
        detector_object.append(r)

    return detector_object


def predict(net, img):
    img = img[:, :, ::-1]  # BGR -> RGB
    orig_shape = img.shape

    img = preprocess(img)

    # feedforward
    if not args.onnx:
        output = net.predict([img])
    else:
        output = net.run(None, {'images': img})
    preds = output[0]

    preds = post_processing(preds, img, orig_shape)

    return preds


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
            total_time = 0
            for i in range(args.benchmark_count):
                start = int(round(time.time() * 1000))
                preds = predict(net, img)
                end = int(round(time.time() * 1000))
                if i != 0:
                    total_time = total_time + (end - start)
                logger.info(f'\tailia processing time {end - start} ms')
            logger.info(f'\taverage time {total_time / (args.benchmark_count - 1)} ms')
        else:
            preds = predict(net, img)

        # plot result
        detect_object = convert_to_detector_object(preds, img.shape[1], img.shape[0])
        res_img = plot_results(detect_object, img)

        savepath = get_savepath(args.savepath, image_path, ext='.png')
        logger.info(f'saved at : {savepath}')
        cv2.imwrite(savepath, res_img)

        # write prediction
        if args.write_prediction is not None:
            ext = args.write_prediction
            pred_file = "%s.%s" % (savepath.rsplit('.', 1)[0], ext)
            write_predictions(pred_file, detect_object, img, category=COCO_CATEGORY, file_type=ext)

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
        preds = predict(net, frame)

        # plot result
        detect_object = convert_to_detector_object(preds, frame.shape[1], frame.shape[0])
        res_img = plot_results(detect_object, frame)

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
        'v8n': (WEIGHT_YOLOV8N_PATH, MODEL_YOLOV8N_PATH),
        'v8s': (WEIGHT_YOLOV8S_PATH, MODEL_YOLOV8S_PATH),
        'v8m': (WEIGHT_YOLOV8M_PATH, MODEL_YOLOV8M_PATH),
        'v8l': (WEIGHT_YOLOV8L_PATH, MODEL_YOLOV8L_PATH),
        'v8x': (WEIGHT_YOLOV8X_PATH, MODEL_YOLOV8X_PATH),
    }
    WEIGHT_PATH, MODEL_PATH = dic_model[args.model_type]

    # model files check and download
    check_and_download_models(WEIGHT_PATH, MODEL_PATH, REMOTE_PATH)

    # initialize
    if not args.onnx:
        net = ailia.Net(MODEL_PATH, WEIGHT_PATH, env_id=args.env_id, memory_mode=args.memory_mode)
    else:
        import onnxruntime
        cuda = 0 < ailia.get_gpu_environment_id()
        providers = ['CUDAExecutionProvider', 'CPUExecutionProvider'] if cuda else ['CPUExecutionProvider']
        net = onnxruntime.InferenceSession(WEIGHT_PATH, providers=providers)

    if args.video is not None:
        # video mode
        recognize_from_video(net)
    else:
        # image mode
        recognize_from_image(net)


if __name__ == '__main__':
    main()
