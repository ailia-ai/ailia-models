import sys
import time

import numpy as np
import cv2

import ailia

# import original modules
sys.path.append('../../util')
from arg_utils import get_base_parser, update_parser, get_savepath
from model_utils import check_and_download_models
from detector_utils import load_image, plot_results, write_predictions
from math_utils import sigmoid
from nms_utils import batched_nms
from webcamera_utils import get_capture, get_writer
# logger
from logging import getLogger

logger = getLogger(__name__)

# ======================
# Parameters
# ======================

WEIGHT_PATH = 'FastestDet.onnx'
MODEL_PATH = 'FastestDet.onnx.prototxt'
REMOTE_PATH = 'https://storage.googleapis.com/ailia-models/fastest-det/'

IMAGE_PATH = 'demo.jpg'
SAVE_IMAGE_PATH = 'output.png'

IMAGE_HEIGHT = 352
IMAGE_WIDTH = 352

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

THRESHOLD = 0.65
IOU = 0.45

# ======================
# Arguemnt Parser Config
# ======================

parser = get_base_parser(
    'FastestDet', IMAGE_PATH, SAVE_IMAGE_PATH
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
    '-w', '--write_prediction',
    nargs='?',
    const='txt',
    choices=['txt', 'json'],
    type=str,
    help='Output results to txt or json file.'
)
args = update_parser(parser)


# ======================
# Main functions
# ======================

def preprocess(img):
    im_h, im_w, _ = img.shape

    img = cv2.resize(img, (IMAGE_WIDTH, IMAGE_HEIGHT), interpolation=cv2.INTER_LINEAR)

    img = img / 255
    img = img.transpose(2, 0, 1)  # HWC -> CHW
    img = np.expand_dims(img, axis=0)
    img = img.astype(np.float32)

    return img


def post_processing(preds, img_size):
    im_h, im_w = img_size
    conf_thresh = args.threshold
    nms_thresh = args.iou

    total_bboxes = []
    # Convert the feature map to the coordinates of the detection box
    N, C, H, W = preds.shape
    bboxes = np.zeros((N, H, W, 6))
    pred = preds.transpose(0, 2, 3, 1)
    # front-background classification branch
    pobj = np.expand_dims(pred[:, :, :, 0], axis=-1)
    # detection box regression branch
    preg = pred[:, :, :, 1:5]
    # target class classification branch
    pcls = pred[:, :, :, 5:]

    # detection box confidence
    bboxes[..., 4] = (np.squeeze(pobj, axis=-1) ** 0.6) * (np.max(pcls, axis=-1) ** 0.4)
    bboxes[..., 5] = np.argmax(pcls, axis=-1)

    # The coordinates of the detection frame
    gx, gy = np.meshgrid(np.arange(W), np.arange(H))
    bw, bh = sigmoid(preg[..., 2]), sigmoid(preg[..., 3])
    bcx = (np.tanh(preg[..., 0]) + gx) / W
    bcy = (np.tanh(preg[..., 1]) + gy) / H

    # cx,cy,w,h = > x1,y1,x2,y1
    x1, y1 = bcx - 0.5 * bw, bcy - 0.5 * bh
    x2, y2 = bcx + 0.5 * bw, bcy + 0.5 * bh

    bboxes[..., 0], bboxes[..., 1] = x1, y1
    bboxes[..., 2], bboxes[..., 3] = x2, y2
    bboxes = bboxes.reshape(N, H * W, 6)
    total_bboxes.append(bboxes)

    batch_bboxes = np.concatenate(total_bboxes, axis=1)

    # Perform NMS processing on the detection frame
    detections = []
    for p in batch_bboxes:
        output, temp = [], []
        b, s, c = [], [], []
        # Threshold filtering
        t = p[:, 4] > conf_thresh
        pb = p[t]
        for bbox in pb:
            obj_score = bbox[4]
            category = bbox[5]
            x1, y1 = bbox[0], bbox[1]
            x2, y2 = bbox[2], bbox[3]
            s.append([obj_score])
            c.append([category])
            b.append([x1 * im_w, y1 * im_h, x2 * im_w, y2 * im_h])
            temp.append([x1, y1, x2, y2, obj_score, category])
        # Torchvision NMS
        if len(b) > 0:
            b = np.array(b)
            c = np.squeeze(np.array(c))
            s = np.squeeze(np.array(s))
            keep = batched_nms(b, s, c, nms_thresh)
            for i in keep:
                x1, y1, x2, y2, score, category = temp[i]
                r = ailia.DetectorObject(
                    category=int(category),
                    prob=score,
                    x=x1,
                    y=y1,
                    w=(x2 - x1),
                    h=(y2 - y1),
                )
                output.append(r)
        detections.append(output)

    return detections[0]


def predict(net, img):
    im_h, im_w, _ = img.shape
    img = preprocess(img)

    # feedforward
    output = net.predict([img])
    preds = output[0]

    pred = post_processing(preds, (im_h, im_w))

    return pred


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
                detect_objects = predict(net, img)
                end = int(round(time.time() * 1000))
                estimation_time = (end - start)

                # Logging
                logger.info(f'\tailia processing estimation time {estimation_time} ms')
                if i != 0:
                    total_time_estimation = total_time_estimation + estimation_time

            logger.info(f'\taverage time estimation {total_time_estimation / (args.benchmark_count - 1)} ms')
        else:
            detect_objects = predict(net, img)

        res_img = plot_results(detect_objects, img, COCO_CATEGORY)

        # plot result
        savepath = get_savepath(args.savepath, image_path, ext='.png')
        logger.info(f'saved at : {savepath}')
        cv2.imwrite(savepath, res_img)

        # write prediction
        if args.write_prediction is not None:
            ext = args.write_prediction
            pred_file = "%s.%s" % (savepath.rsplit('.', 1)[0], ext)
            write_predictions(pred_file, detect_objects, img, category=COCO_CATEGORY, file_type=ext)

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
        detect_objects = predict(net, frame)

        # plot result
        res_img = plot_results(detect_objects, frame, COCO_CATEGORY)

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
