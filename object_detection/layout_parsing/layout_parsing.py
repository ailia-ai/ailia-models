import os
import sys

import ailia
import cv2
from numpy import fix
import json

sys.path.append("../../util")
from logging import getLogger  # noqa: E402

from arg_utils import get_base_parser, update_parser
from detector_utils import plot_results, reverse_letterbox
from model_utils import check_and_download_models  # noqa: E402

from layout_parsing_utils import pdf_to_images, preprocess
from yolox import YOLOX_LABEL_MAP

REMOTE_PATH = "https://storage.googleapis.com/ailia-models/unstructured-inference/"
IMAGE_PATH = "input.jpg"
PDF_PATH = "sample.pdf"
SAVE_IMAGE_PATH = ""
WEIGHT_PATH = "layout_parsing_yolox.onnx"
MODEL_PATH = WEIGHT_PATH + ".prototxt"
INPUT_SHAPE = (1024, 768)
SCORE_THR = 0.25
NMS_THR = 0.1

logger = getLogger(__name__)


parser = get_base_parser('Layout parsing', IMAGE_PATH, SAVE_IMAGE_PATH)
parser.add_argument(
    "-i", "--input",
    type=str,
    help="input file name",
    default=IMAGE_PATH,
)
parser.add_argument(
    "-d", "--dpi", default=200, type=int, help="dpi"
)
parser.add_argument(
    '-th', '--threshold',
    default=SCORE_THR, type=float,
    help='The detection threshold for yolo. (default: '+str(SCORE_THR)+')'
)
parser.add_argument(
    '-iou', '--iou',
    default=NMS_THR, type=float,
    help='The detection iou for yolo. (default: '+str(NMS_THR)+')'
)
parser.add_argument(
    "-fp", "--from_pdf",
    action="store_true",
    help="set this option when target file is pdf",
)
parser.add_argument(
    '-w', '--write_json',
    action='store_true',
    help='Flag to output results to json file.'
)
args = update_parser(parser)


def infer_from_pdf(detector: ailia.Detector):
    image_names = pdf_to_images(args.input[0], dpi=args.dpi, output_folder=args.savepath)

    for image_name in image_names:
        img_orig = cv2.imread(image_name)
        img_processed, ratio = preprocess(img_orig, INPUT_SHAPE)

        def compute():
            detector.compute(img_orig, args.threshold, args.iou)
            return None

        # inference
        logger.info('Start inference...')
        if args.benchmark:
            logger.info('BENCHMARK mode')
            total_time = 0
            for i in range(args.benchmark_count):
                start = int(round(time.time() * 1000))
                output = compute()
                end = int(round(time.time() * 1000))
                if i != 0:
                    total_time = total_time + (end - start)
                logger.info(f'\tailia processing time {end - start} ms')
            logger.info(f'\taverage time {total_time / (args.benchmark_count-1)} ms')
        else:
            output = compute()

        res_img = plot_results(detector, img_orig, YOLOX_LABEL_MAP)
        detect_object = detector

        cv2.imwrite(image_name.replace(".ppm", "_parsed.jpg"), res_img)


def infer_from_image(detector: ailia.Detector):
    image_filename = args.input[0]
    img_orig = cv2.imread(image_filename)
    img_processed, ratio = preprocess(img_orig, INPUT_SHAPE)

    def compute():
        detector.compute(img_orig, args.threshold, args.iou)
        return None

    logger.info("Start: inference...")
    if args.benchmark:
        logger.info("BENCHMARK mode")
        total_time = 0
        for i in range(args.benchmark_count):
            start = int(rount(time.time() * 1000))
            output = compute()
            end = int(rount(time.time() * 1000))
            if i != 0:
                total_time = total_time + (end - start)
            logger.info(f"\tailia processing time {end - start} ms")
        
        logger.info(f"\taverage time {total_time / (args.benchmark_count - 1)} ms")
    else:
        output = compute()
    
    res_img = plot_results(detector, img_orig, YOLOX_LABEL_MAP)

    extension = os.path.splitext(image_filename)[1]
    output_filename = image_filename.replace(extension, "_parsed.jpg")
    cv2.imwrite(output_filename, res_img)

    if args.write_json:
        json_filename = image_filename.replace(extension, "_parsed.json")
        out_list = []
        for idx in range(detector.get_object_count()):
            d = detector.get_object(idx)
            out_list.append({
                "category": d.category, "category_name": YOLOX_LABEL_MAP[d.category],
                "prob": float(d.prob),
                "x": float(d.x), "y": float(d.y), "w": float(d.w), "h": float(d.h)
            })
        with open(json_filename, "w", encoding="utf-8") as f:
            json.dump(out_list, f, indent=2)

if __name__ == "__main__":
    check_and_download_models(WEIGHT_PATH, MODEL_PATH, REMOTE_PATH)
    env_id = args.env_id
    detector = ailia.Detector(
        MODEL_PATH,
        WEIGHT_PATH,
        len(YOLOX_LABEL_MAP),
        format=ailia.NETWORK_IMAGE_FORMAT_BGR,
        channel=ailia.NETWORK_IMAGE_CHANNEL_FIRST,
        range=ailia.NETWORK_IMAGE_RANGE_U_INT8,
        algorithm=ailia.DETECTOR_ALGORITHM_YOLOX,
        env_id=env_id,
        memory_mode=args.memory_mode,
    )
    if args.from_pdf:
        infer_from_pdf(detector)
    else:
        infer_from_image(detector)
