import os
import ast
import sys
import time
from logging import getLogger

import numpy as np
import cv2

import ailia

from predict import FastSAMPredictor
from prompt import FastSAMPrompt 
from util import convert_box_xywh_to_xyxy

# import original modules
sys.path.append('../../util')
from arg_utils import get_base_parser, update_parser, get_savepath  # noqa
from model_utils import urlretrieve, progress_print, check_and_download_models  # noqa
from detector_utils import imread  # noqa
from webcamera_utils import get_capture, get_writer  # noqa

logger = getLogger(__name__)

# ======================
# Parameters
# ======================

REMOTE_FASTSAM_PATH = 'https://storage.googleapis.com/ailia-models/fast_sam/'

WEIGHT_VIT_TEXT_PATH = 'ViT-B32-encode_text.onnx'
MODEL_VIT_TEXT_PATH  = 'ViT-B32-encode_text.onnx.prototxt'
WEIGHT_VIT_IMAGE_PATH = 'ViT-B32-encode_image.onnx'
MODEL_VIT_IMAGE_PATH  = 'ViT-B32-encode_image.onnx.prototxt'
REMOTE_VIT_PATH = 'https://storage.googleapis.com/ailia-models/clip/'

IMAGE_PATH = 'cat.jpg'
SAVE_IMAGE_PATH = 'output.png'

TARGET_LENGTH = 1024

# ======================
# Arguemnt Parser Config
# ======================

parser = get_base_parser(
    'FastSAM', IMAGE_PATH, SAVE_IMAGE_PATH
)
parser.add_argument(
    '-p', '--pos', action='append', type=int, metavar="X", nargs=2,
    help='Positive coordinate specified by x,y.'
)
parser.add_argument("--box_prompt", type=str, default="[[0,0,0,0]]", help="[[x,y,w,h],[x2,y2,w2,h2]] support multiple boxes")

parser.add_argument(
    "--point_prompt", type=str, default="[[0,0]]", help="[[x1,y1],[x2,y2]]"
)

parser.add_argument(
    '-m', '--model_type', default='FastSAM-x', choices=('FastSAM-s', 'FastSAM-x'),
    help='Select model.'
)

parser.add_argument(
    "--text_prompt", type=str, default=None, help='use text prompt eg: "a dog"'
)

parser.add_argument(
    "--point_label",
    type=str,
    default="[0]",
    help="[1,0] 0:background, 1:foreground",
)
parser.add_argument(
    "--conf", type=float, default=0.4, help="object confidence threshold"
)
parser.add_argument("--imgsz", type=int, default=1024, help="image size")
parser.add_argument(
    "--iou",
    type=float,
    default=0.9,
    help="iou threshold for filtering the annotations",
)

args = update_parser(parser)

# ======================
# Parameters2
# ======================

WEIGHT_FASTSAM_PATH = args.model_type+'.onnx'
MODEL_FASTSAM_PATH  = args.model_type+'.onnx.prototxt'

# ======================
# Secondaty Functions
# ======================
def FastSAM(SAM_model, vit_model,input):

    everything_results = SAM_model(
        source=input,
        conf=args.conf,
        iou=args.iou,
        imgsz=args.imgsz
        )
    bboxes = None
    points = None
    point_label = None

    prompt_process = FastSAMPrompt(input, everything_results,*vit_model)
    if args.box_prompt[0][2] != 0 and args.box_prompt[0][3] != 0:
            ann = prompt_process.box_prompt(bboxes=args.box_prompt)
            bboxes = args.box_prompt
    elif args.text_prompt != None:
        ann = prompt_process.text_prompt(text=args.text_prompt)
    elif args.point_prompt[0] != [0, 0]:
        ann = prompt_process.point_prompt(
            points=args.point_prompt, pointlabel=args.point_label
        )
        points = args.point_prompt
        point_label = args.point_label
    else:
        ann = everything_results[0].masks.data

    result = prompt_process.plot(
                annotations=ann,
                output_path=args.savepath,
                bboxes = bboxes,
                points = points,
                point_label = point_label,
            )
    return result

# ======================
# Main functions
# ======================

def recognize_from_image(net_sam, net_image, net_text):
    # input image loop
    for image_path in args.input:
        logger.info(image_path)

        # inference
        logger.info('Start inference...')
        recognize(image_path, net_sam, net_image, net_text)


def recognize(image_path, net_sam, net_image, net_text):
    model = FastSAMPredictor(net_sam)
    args.point_prompt = ast.literal_eval(args.point_prompt)
    args.box_prompt = convert_box_xywh_to_xyxy(ast.literal_eval(args.box_prompt))
    args.point_label = ast.literal_eval(args.point_label)

    input = imread(image_path)

    if args.benchmark:
            logger.info('BENCHMARK mode')
            total_time_estimation = 0
            for i in range(args.benchmark_count):
                start = int(round(time.time() * 1000))
                result = FastSAM(model,(net_image, net_text), input)
                end = int(round(time.time() * 1000))
                estimation_time = (end - start)

                # Logging
                logger.info(f'\tailia processing estimation time {estimation_time} ms')
                if i != 0:
                    total_time_estimation = total_time_estimation + estimation_time

            logger.info(f'\taverage time estimation {total_time_estimation / (args.benchmark_count - 1)} ms')
    else:
        result = FastSAM(model,(net_image,net_text), input)

    # plot result
    savepath = get_savepath(args.savepath, image_path, ext='.png')
    logger.info(f'saved at : {savepath}')
    cv2.imwrite(savepath, result)


def recognize_from_video(net_sam, net_image, net_text):
    model = FastSAMPredictor(net_sam)
    args.point_prompt = ast.literal_eval(args.point_prompt)
    args.box_prompt = convert_box_xywh_to_xyxy(ast.literal_eval(args.box_prompt))
    args.point_label = ast.literal_eval(args.point_label)


    frame_names = None
    capture = get_capture(args.video)
    video_height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    video_width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))

    if args.savepath != SAVE_IMAGE_PATH:
        writer = get_writer(args.savepath, video_height, video_width)
    else:
        writer = None

    frame_shown = False

    frame_idx = 0
    while (True):
        ret, frame = capture.read()

        if (cv2.waitKey(1) & 0xFF == ord('q')) or not ret:
            break
        if frame_shown and cv2.getWindowProperty('frame', cv2.WND_PROP_VISIBLE) == 0:
            break

        frame = FastSAM(model,(net_image, net_text), frame)

        cv2.imshow('frame', frame)
        if frame_names is not None:
            cv2.imwrite(f'video_{frame_idx}.png', frame)

        if writer is not None:
            writer.write(frame)

        frame_shown = True
        frame_idx = frame_idx + 1

    if writer is not None:
        writer.release()


def main():

    # model files check and download
    check_and_download_models(WEIGHT_FASTSAM_PATH, MODEL_FASTSAM_PATH, REMOTE_FASTSAM_PATH)
    check_and_download_models(WEIGHT_VIT_IMAGE_PATH, MODEL_VIT_IMAGE_PATH, REMOTE_VIT_PATH)
    check_and_download_models(WEIGHT_VIT_TEXT_PATH, MODEL_VIT_TEXT_PATH, REMOTE_VIT_PATH)

    env_id = args.env_id

    # disable FP16
    if "FP16" in ailia.get_environment(env_id).props or sys.platform == 'Darwin':
        logger.warning('This model do not work on FP16. So use CPU mode.')
        env_id = 0
    
    # initialize
    net_sam = ailia.Net(MODEL_FASTSAM_PATH, WEIGHT_FASTSAM_PATH, env_id = env_id, memory_mode = args.memory_mode)
    net_image = ailia.Net(MODEL_VIT_IMAGE_PATH, WEIGHT_VIT_IMAGE_PATH, env_id = env_id, memory_mode = args.memory_mode)
    net_text  = ailia.Net(MODEL_VIT_TEXT_PATH, WEIGHT_VIT_TEXT_PATH, env_id = env_id, memory_mode = args.memory_mode)

    if args.profile:
        net_sam.set_profile_mode(True)
        net_image.set_profile_mode(True)
        net_text.set_profile_mode(True)

    if args.video is not None:
        recognize_from_video(net_sam, net_image, net_text)
    else:
        recognize_from_image(net_sam, net_image, net_text)

    if args.profile:
        print(net_sam.get_summary())
        print(net_image.get_summary())
        print(net_text.get_summary())

if __name__ == '__main__':
    main()
