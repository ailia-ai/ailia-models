import sys
import time

import ailia
import cv2
import numpy as np

# import original modules
sys.path.append('../../util')
# logger
from logging import getLogger  # noqa: E402

import webcamera_utils  # noqa: E402
from detector_utils import load_image, plot_results, write_predictions  # noqa: E402
from image_utils import imread  # noqa: E402
from model_utils import check_and_download_models  # noqa: E402
from arg_utils import get_base_parser, get_savepath, update_parser  # noqa: E402

logger = getLogger(__name__)

from yolo_utils import *

# ======================
# Parameters
# ======================
REMOTE_PATH = 'https://storage.googleapis.com/ailia-models/yolov2-tiny/'
IMAGE_PATH = 'input.jpg'
SAVE_IMAGE_PATH = 'output.png'
IMAGE_HEIGHT = 416  # for video mode
IMAGE_WIDTH = 416  # for video mode


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

VOC_CATEGORY = [
    "aeroplane","bicycle","bird","boat","bottle",
    "bus","car","cat","chair","cow","diningtable",
    "dog","horse","motorbike","person","pottedplant",
    "sheep","sofa","train","tvmonitor"
]


COCO_ANCHORS = np.array(
    [0.57273, 0.677385, 1.87446, 2.06253, 3.33843,
     5.47434, 7.88282, 3.52778, 9.77052, 9.16828]
)

VOC_ANCHORS = np.array(
        [1.08,1.19,  3.42,4.41,  6.63,
         11.38,  9.42,5.11,  16.62,10.52]
)

CATEGORY = COCO_CATEGORY
ANCHORS = COCO_ANCHORS

THRESHOLD = 0.4
IOU = 0.45



# ======================
# Arguemnt Parser Config
# ======================
parser = get_base_parser('Yolov2 tiny model', IMAGE_PATH, SAVE_IMAGE_PATH)
parser.add_argument(
    '-dt', '--detector',
    action='store_true',
    help='Use detector API (require ailia SDK 1.2.7).'
)
parser.add_argument(
    '-dataset', '--dataset',
    metavar='DATASET', default='coco'
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

if args.dataset=='voc':
    ANCHORS  = VOC_ANCHORS
    CATEGORY = VOC_CATEGORY
    WEIGHT_PATH = 'yolov2-tiny-voc.onnx'
    MODEL_PATH  = 'yolov2-tiny-voc.onnx.prototxt'
elif args.dataset=='coco':
    ANCHORS  = COCO_ANCHORS
    CATEGORY = COCO_CATEGORY
    WEIGHT_PATH = 'yolov2-tiny-coco.onnx'
    MODEL_PATH  = 'yolov2-tiny-coco.onnx.prototxt'

def detect(img,output,savepath='output.png', conf_thresh=0.5, nms_thresh=0.4,video=False):
    num_classes = len(CATEGORY) 
    
    if num_classes == 20:
        namesfile = 'voc.names'
    elif num_classes == 80:
        namesfile = 'coco.names'
    else:
        namesfile = 'data/names'

    for i in range(2):
        boxes = get_region_boxes(output, conf_thresh, num_classes, ANCHORS, 5)[0]
        #boxes = get_region_boxes(output, conf_thresh, num_classes, anchors, num_anchors)[0]
        boxes = nms(boxes, nms_thresh)

    class_names = load_class_names(namesfile)
    if video:
        return plot_boxes(img, boxes, False, class_names)
    else:
        return plot_boxes(img, boxes, savepath, class_names)

# ======================
# Main functions
# ======================
def recognize_from_image():
    # net initialize
    if args.detector == True:
        detector = ailia.Detector(
            MODEL_PATH,
            WEIGHT_PATH,
            len(CATEGORY),
            format=ailia.NETWORK_IMAGE_FORMAT_RGB,
            channel=ailia.NETWORK_IMAGE_CHANNEL_FIRST,
            range=ailia.NETWORK_IMAGE_RANGE_S_FP32,
            algorithm=ailia.DETECTOR_ALGORITHM_YOLOV2,
            env_id=args.env_id,
            memory_mode=args.memory_mode,
        )
        detector.set_anchors(ANCHORS)
        if args.profile:
            detector.set_profile_mode(True)
    else:
        print("path",WEIGHT_PATH)
        net = ailia.Net(None,WEIGHT_PATH,memory_mode=args.memory_mode)
    
    # input image loop
    for image_path in args.input:
        # prepare input data logger.info(image_path)
        img = load_image(image_path)
        logger.debug(f'input image shape: {img.shape}')

        # inference
        logger.info('Start inference...')
        if args.benchmark:
            logger.info('BENCHMARK mode')
            for i in range(5):
                start = int(round(time.time() * 1000))
                if args.detector:
                    detector.compute(img, THRESHOLD, IOU)
                else:
                    pass
                end = int(round(time.time() * 1000))
                logger.info(f'\tailia processing time {end - start} ms')
            # plot result
            res_img = plot_results(detector, img, CATEGORY)
            savepath = get_savepath(args.savepath, image_path)
            logger.info(f'saved at : {savepath}')
            cv2.imwrite(savepath, res_img)

            # write prediction
            if args.write_prediction is not None:
                ext = args.write_prediction
                pred_file = "%s.%s" % (savepath.rsplit('.', 1)[0], ext)
                write_predictions(pred_file, detector, img, category=CATEGORY, file_type=ext)

            if args.profile:
                print(detector.get_summary())

        else:
            if args.detector:
                detector.compute(img, THRESHOLD, IOU)
                # plot result
                res_img = plot_results(detector, img, CATEGORY)
                savepath = get_savepath(args.savepath, image_path)
                logger.info(f'saved at : {savepath}')
                cv2.imwrite(savepath, res_img)

                # write prediction
                if args.write_prediction is not None:
                    ext = args.write_prediction
                    pred_file = "%s.%s" % (savepath.rsplit('.', 1)[0], ext)
                    write_predictions(pred_file, detector, img, category=CATEGORY, file_type=ext)

                if args.profile:
                    print(detector.get_summary())
 
            else:
                savepath = get_savepath(args.savepath, image_path)

                img_PIL = Image.open(image_path).convert('RGB')
                input_data = imread(image_path)
                input_data = cv2.resize(input_data, (416,416))/ 255
                input_data = input_data.transpose((2,0,1))
                input_data = input_data[np.newaxis,:,:,:].astype(np.float32)
                results = net.run([input_data])
                results = torch.FloatTensor(results[0])
                detect(img_PIL,results,savepath,video=False)

    logger.info('Script finished successfully.')


def recognize_from_video():
    # net initialize
    if args.detector == True:
        detector = ailia.Detector(
            MODEL_PATH,
            WEIGHT_PATH,
            len(CATEGORY),
            format=ailia.NETWORK_IMAGE_FORMAT_RGB,
            channel=ailia.NETWORK_IMAGE_CHANNEL_FIRST,
            range=ailia.NETWORK_IMAGE_RANGE_S_FP32,
            algorithm=ailia.DETECTOR_ALGORITHM_YOLOV2,
            env_id=args.env_id,
            memory_mode=args.memory_mode,
        )
        detector.set_anchors(ANCHORS)
    else:
        net = ailia.Net(None,WEIGHT_PATH,memory_mode=args.memory_mode)

    capture = webcamera_utils.get_capture(args.video)

    # create video writer if savepath is specified as video format
    if args.savepath != SAVE_IMAGE_PATH:
        f_h = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        f_w = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        writer = webcamera_utils.get_writer(args.savepath, f_h, f_w)
    else:
        writer = None

    frame_shown = False
    while(True):
        ret, frame = capture.read()
        if (cv2.waitKey(1) & 0xFF == ord('q')) or not ret:
            break 
        if frame_shown and cv2.getWindowProperty('frame', cv2.WND_PROP_VISIBLE) == 0:
            break
        if args.detector:
            img = cv2.cvtColor(frame, cv2.COLOR_BGR2BGRA)
            detector.compute(img, THRESHOLD, IOU)
            res_img = plot_results(detector, frame, CATEGORY, logging = False)
        else:
            img_PIL = Image.fromarray(frame)
            img = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            img = cv2.resize(img, (416,416))
            img = img.transpose((2,0,1))/255
            img = img[np.newaxis,:,:,:].astype(np.float32)
            results = net.run([img])
            results = torch.FloatTensor(results[0])
            output_img = detect(img_PIL,results, video=True)
            res_img = np.array(output_img,dtype=np.uint8)

        cv2.imshow('frame', res_img)
        frame_shown = True

        # save results
        if writer is not None:
            writer.write(res_img)

    capture.release()
    cv2.destroyAllWindows()
    if writer is not None:
        writer.release()
    logger.info('Script finished successfully.')


def main():
    # model files check and download
    check_and_download_models(WEIGHT_PATH, MODEL_PATH, REMOTE_PATH)
    
    if args.video is not None:
        # video mode
        recognize_from_video()
    else:
        # image mode
        recognize_from_image()


if __name__ == '__main__':
    main()
