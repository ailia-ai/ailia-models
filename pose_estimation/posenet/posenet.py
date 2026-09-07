import json
import sys
import cv2
import time

import ailia
import numpy as np

sys.path.append('../../util')
# logger
from logging import getLogger  # noqa: E402

import webcamera_utils  # noqa: E402
from image_utils import imread, load_image  # noqa: E402
from model_utils import check_and_download_models  # noqa: E402
from arg_utils import get_base_parser, get_savepath, update_parser  # noqa: E402
from dtype_utils import numpy_type_to_builtin_type  # noqa: E402

logger = getLogger(__name__)

from posenet_util import *

# OPENPOSE: MULTIPERSON KEYPOINT DETECTION
# SOFTWARE LICENSE AGREEMENT
# ACADEMIC OR NON-PROFIT ORGANIZATION NONCOMMERCIAL RESEARCH USE ONLY

# ======================
# Parameters
# ======================
IMAGE_PATH = 'input.jpg'
SAVE_IMAGE_PATH = 'output.png'

IMAGE_HEIGHT = 720
IMAGE_WIDTH = 1280

REMOTE_PATH = 'https://storage.googleapis.com/ailia-models/posenet/'

THRESHOLD_POSE_DEFAULT = 0.15
THRESHOLD_PART_DEFAULT = 0.15
SCALE_DEFAULT = 0.7125

MODEL_LISTS = ['50','75','100','101']

# ======================
# Arguemnt Parser Config
# ======================
parser = get_base_parser(
    'Fast and accurate human pose 2D-estimation.', IMAGE_PATH, SAVE_IMAGE_PATH,
)
parser.add_argument(
    '-t', '--threshold-pose', type=float, default=THRESHOLD_POSE_DEFAULT,
    help='The detection pose threshold.'
)
parser.add_argument(
    '--threshold-part', type=float, default=THRESHOLD_PART_DEFAULT,
    help='The detection part threshold.'
)


parser.add_argument('-a','--arch', type=str, default='101', choices=MODEL_LISTS,
    help='select model: ' + ' | '.join(MODEL_LISTS)
)

parser.add_argument(
    '--scale-factor', type=float, default=SCALE_DEFAULT,
)

parser.add_argument(
    '-w', '--write_json',
    action='store_true',
    help='save result to json'
)

args = update_parser(parser)

def keypoint_draw(image_path,draw_image, pose_scores, keypoint_scores, keypoint_coords):
    draw_image = draw_skel_and_kp(
        draw_image, pose_scores, keypoint_scores, keypoint_coords,
        min_pose_score=args.threshold_pose, min_part_score=args.threshold_part)

    if False:
        print("\nResults for image: %s" % image_path)
        for pi in range(len(pose_scores)):
            if pose_scores[pi] == 0.:
                break
            print('Pose #%d, score = %f' % (pi, pose_scores[pi]))
            for ki, (s, c) in enumerate(zip(keypoint_scores[pi, :], keypoint_coords[pi, :, :])):
                print('Keypoint %s, score = %f, coord = %s' % (PART_NAMES[ki], s, c))
    return draw_image


def save_json(pose_scores, keypoint_scores, keypoint_coords, json_path):
    output = []
    for pose_score, kpt_scores, kpt_coords in zip(pose_scores, keypoint_scores, keypoint_coords):
        if pose_score < args.threshold_pose:
            continue
        output.append({
            'pose_score': pose_score,
            'keypoints': [
                {'pos': [c[1], c[0]], 'prob': s}  # keypoint_coords is in (y, x) order
                for s, c in zip(kpt_scores, kpt_coords)
            ],
        })

    with open(json_path, 'w') as f:
        json.dump(numpy_type_to_builtin_type(output), f, indent=2)


def detect(model,img):
    output_stride = 16
    

    input_image, draw_image, output_scale = process_input(img,args.scale_factor, output_stride=output_stride)

    heatmaps_result, offsets_result, displacement_fwd_result, displacement_bwd_result = model.run(input_image)

    pose_scores, keypoint_scores, keypoint_coords = decode_multiple_poses(
        heatmaps_result.squeeze(0),
        offsets_result.squeeze(0),
        displacement_fwd_result.squeeze(0),
        displacement_bwd_result.squeeze(0),
        output_stride=output_stride,
        max_pose_detections=10,
        min_pose_score=args.threshold_pose)

    keypoint_coords *= output_scale
    return draw_image, pose_scores, keypoint_scores, keypoint_coords,

# ======================
# Main functions
# ======================
def recognize_from_image(model):

    # input image loop
    for image_path in args.input:
        # prepare input data
        logger.info(image_path)

        img = imread(image_path)
        # inference
        logger.info('Start inference...')
        if args.benchmark:
            logger.info('BENCHMARK mode')
            for i in range(5):
                start = int(round(time.time() * 1000))

                draw_image, pose_scores, keypoint_scores, keypoint_coords= detect(model,img)
                end = int(round(time.time() * 1000))
                logger.info(f'\tailia processing time {end - start} ms')
        else:
            draw_image, pose_scores, keypoint_scores, keypoint_coords,= detect(model,img)
        draw_image = keypoint_draw(image_path,draw_image, pose_scores, keypoint_scores, keypoint_coords)

        # postprocessing
        savepath = get_savepath(args.savepath, image_path)
        cv2.imwrite( savepath, draw_image)
        logger.info(f'saved at : {savepath}')

        if args.write_json:
            save_json(pose_scores, keypoint_scores, keypoint_coords,
                      (savepath.rsplit('.', 1)[0]) + '.json')

    logger.info('Script finished successfully.')


def recognize_from_video(model):
    # net initialize

    capture = webcamera_utils.get_capture(args.video)
    # create video writer if savepath is specified as video format
    if args.savepath != SAVE_IMAGE_PATH:
        writer = webcamera_utils.get_writer(args.savepath, IMAGE_HEIGHT, IMAGE_WIDTH)
    else:
        writer = None
    
    frame_shown = False
    while(True):
        ret, frame = capture.read()
        if (cv2.waitKey(1) & 0xFF == ord('q')) or not ret:
            break
        if frame_shown and cv2.getWindowProperty('frame', cv2.WND_PROP_VISIBLE) == 0:
            break

        input_image, input_data = webcamera_utils.adjust_frame_size(
            frame, IMAGE_HEIGHT, IMAGE_WIDTH,
        )

        # inference
        _, pose_scores, keypoint_scores, keypoint_coords= detect(model,input_data)

        # postprocessing
        draw_image = keypoint_draw(None, input_data, pose_scores, keypoint_scores, keypoint_coords)
        cv2.imshow('frame', draw_image)

        frame_shown = True

        # save results
        if writer is not None:
            writer.write(draw_image)

    capture.release()
    cv2.destroyAllWindows()
    if writer is not None:
        writer.release()
    logger.info('Script finished successfully.')


def main():
    # model files check and download
    MODEL_PATH  = 'posenet_' + str(args.arch) + '.onnx.prototxt'
    WEIGHT_PATH = 'posenet_' + str(args.arch) + '.onnx'
    check_and_download_models(WEIGHT_PATH, MODEL_PATH, REMOTE_PATH)

    # net initialize
    model = ailia.Net(MODEL_PATH, WEIGHT_PATH, env_id=args.env_id)
    if args.video is not None:
        # video mode
        recognize_from_video(model)
    else:
        # image mode
        recognize_from_image(model)


if __name__ == '__main__':
    main()
