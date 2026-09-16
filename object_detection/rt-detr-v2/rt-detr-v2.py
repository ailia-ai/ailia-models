import sys
import time

import numpy as np
import cv2

import ailia

# import original modules
import sys
sys.path.append('../../util')
from arg_utils import get_base_parser, update_parser, get_savepath  # noqa
from model_utils import check_and_download_models  # noqa
from image_utils import imread  # noqa
from detector_utils import plot_results, write_predictions # noqa
from webcamera_utils import get_capture, get_writer  # noqa

# logger
from logging import getLogger  # noqa

logger = getLogger(__name__)

# ======================
# Parameters
# ======================

MODEL_LISTS = ["rtdetrv2_r18vd_120e","rtdetrv2_r34vd_120e","rtdetrv2_r50vd_6x","rtdetrv2_r50vd_m_7x","rtdetrv2_r101vd_6x"]

IMAGE_PATH = 'demo.jpg'
SAVE_IMAGE_PATH = 'output.png'

COCO_CATEGORY = (
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
)
# ======================
# Arguemnt Parser Config
# ======================

parser = get_base_parser(
    'RT-DETR: DETRs Beat YOLOs on Real-time Object Detection', IMAGE_PATH, SAVE_IMAGE_PATH
)

parser.add_argument(
    '-t', '--threshold', type=float, default=0.6,
    help='threshold'
)

parser.add_argument(
    '-a', '--arch', metavar='ARCH',
    default='rtdetrv2_r18vd_120e', choices=MODEL_LISTS,
    help='model lists: ' + ' | '.join(MODEL_LISTS) + ' , ' 
)

parser.add_argument(
    '-w', '--write_json',
    action='store_true',
    help='Flag to output results to json file.'
)

args = update_parser(parser)



WEIGHT_PATH = args.arch + '_coco.onnx'
MODEL_PATH  = args.arch + '_coco.onnx.prototxt'
REMOTE_PATH = 'https://storage.googleapis.com/ailia-models/rt-detr-v2/'

# ======================
# Main functions
# ======================

def preprocess(im):
    im = cv2.resize(im, (640, 640))
    im = im.astype(np.float32) / 255.0
    im = np.transpose(im, (2, 0, 1))
    return im[np.newaxis, :]

def post_processing(img,labels,boxes,scores):
    im_h, im_w = img.shape[:2]
    thrh=args.threshold
    detections = []
    for bbox, label ,score in zip(boxes, labels,scores):

        label = label[score > thrh]
        bbox = bbox[score > thrh]

        for i,box in enumerate(bbox):
            x1, y1, x2, y2 = box
            r = ailia.DetectorObject(
                category=label[i],
                prob=score[i],
                x=x1 / im_w,
                y=y1 / im_h,
                w=(x2 - x1) / im_w,
                h=(y2 - y1) / im_h,
            )
            detections.append(r)

    return detections


def recognize_from_image(net):
    # input image loop
    for image_path in args.input:
        logger.info(image_path)

        # prepare input data
        img = imread(image_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        h, w, _ = img.shape
        orig_size = np.array([[w, h]])
        img_data = preprocess(img)

        # inference
        logger.info('Start inference...')
        if args.benchmark:
            logger.info('BENCHMARK mode')
            total_time_estimation = 0
            for i in range(args.benchmark_count):
                start = int(round(time.time() * 1000))
                labels,boxes,scores = net.run((img_data,orig_size))
                end = int(round(time.time() * 1000))
                estimation_time = (end - start)

                # Logging
                logger.info(f'\tailia processing estimation time {estimation_time} ms')
                if i != 0:
                    total_time_estimation = total_time_estimation + estimation_time

            logger.info(f'\taverage time estimation {total_time_estimation / (args.benchmark_count - 1)} ms')
        else:
            labels,boxes,scores = net.run((img_data,orig_size))

        # plot result
        detect_object = post_processing(img,labels,boxes,scores)
        res_img = plot_results(detect_object, img,COCO_CATEGORY)
        res_img = cv2.cvtColor(res_img, cv2.COLOR_RGB2BGR)

        savepath = get_savepath(args.savepath, image_path, ext='.png')
        logger.info(f'saved at : {savepath}')
        cv2.imwrite(savepath, res_img)

        if args.write_json:
            jsonpath = "%s.%s" % (savepath.rsplit('.', 1)[0], 'json')
            write_predictions(jsonpath, detect_object, img, category=COCO_CATEGORY, file_type='json')

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
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        h, w, _ = img.shape
        orig_size = np.array([[w, h]])
        img_data = preprocess(img)

        labels,boxes,scores = net.run((img_data,orig_size))


        # plot result
        detect_object = post_processing(img,labels,boxes,scores)
        res_img = plot_results(detect_object, img,COCO_CATEGORY)
        res_img = cv2.cvtColor(res_img, cv2.COLOR_RGB2BGR)

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
    net = ailia.Net(None, WEIGHT_PATH, env_id=env_id, memory_mode=args.memory_mode)

    if args.video is not None:
        recognize_from_video(net)
    else:
        recognize_from_image(net)


if __name__ == '__main__':
    main()
