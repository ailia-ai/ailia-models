import sys
import time
import json

import ailia
import cv2
import numpy as np
import skimage

import movenet_utils

sys.path.append('../../util')
# logger
from logging import getLogger  # noqa: E402

import webcamera_utils  # noqa: E402
from image_utils import imread  # noqa: E402
from model_utils import check_and_download_models  # noqa: E402
from arg_utils import get_base_parser, get_savepath, update_parser  # noqa: E402
from dtype_utils import numpy_type_to_builtin_type  # noqa: E402

logger = getLogger(__name__)

# ======================
# Parameters
# ======================

IMAGE_PATH = 'input.jpg'

SAVE_IMAGE_PATH = 'output.png'
MODEL_VARIANTS = ['thunder','lightning'] # thunder, lightning
MODEL_VARIANT = 'thunder'

parser = get_base_parser(
    'MOVENET,.', IMAGE_PATH, SAVE_IMAGE_PATH,
)
parser.add_argument(
    '-i', '--input', type=str,
    default=IMAGE_PATH,
    help='The input image for movenet.'
)
parser.add_argument(
    '-v', '--video', type=str,
    help='The input video for movenet.'
)
parser.add_argument(
    '-o', '--onnx', action='store_true',
    help="Option to use onnxrutime to run or not."
)
parser.add_argument(
    '-m', '--model_variant', type=str,
    default=MODEL_VARIANT, choices=MODEL_VARIANTS,
    help="The model variant for movenet, 'thunder','lightning'."
)
parser.add_argument(
    '-w', '--write_json',
    action='store_true',
    help='save result to json'
)

args = update_parser(parser)

MODEL_NAME = 'movenet_{}'.format(args.model_variant)
WEIGHT_PATH = f'{MODEL_NAME}.onnx'
MODEL_PATH = f'{MODEL_NAME}.onnx.prototxt'
REMOTE_PATH = 'https://storage.googleapis.com/ailia-models/movenet/'
RESOLUTION = {'thunder': 256, 'lightning': 192}[args.model_variant]
IMAGE_SIZE = RESOLUTION

# # ======================
# # Main functions
# # ======================

def save_json(keypoint_with_scores, json_path):
    output = [
        {'pos': [k[1], k[0]], 'prob': k[2]}
        for k in keypoint_with_scores[0, 0]
    ]

    with open(json_path, 'w') as f:
        json.dump(numpy_type_to_builtin_type(output), f, indent=2)

def recognize_from_image():

    # net initialize
    if args.onnx:
        import onnxruntime
        model = onnxruntime.InferenceSession(WEIGHT_PATH)
    else:
        logger.info(f'env_id: {args.env_id}')
        model = ailia.Net(MODEL_PATH, WEIGHT_PATH, env_id=args.env_id)
    
    for image_path in args.input:

        image = imread(image_path)
        input_image, padding_ratio = movenet_utils.crop_and_padding(image,IMAGE_SIZE)
        input_image = np.expand_dims(input_image, axis=0)
            
        if args.benchmark:
            logger.info('BENCHMARK mode')
            total_time = 0
            for i in range(args.benchmark_count):
                start = int(round(time.time() * 1000))
                keypoint_with_scores = model.run( input_image.astype(np.float32) )[0]
                end = int(round(time.time() * 1000))
                if i != 0:
                    total_time = total_time + (end - start)
                logger.info(f'\tailia processing time {end - start} ms')
            logger.info(f'\taverage time {total_time / (args.benchmark_count-1)} ms')
        else:
            if args.onnx:
                ort_inputs = { model.get_inputs()[0].name : input_image.astype(np.float32)}
                keypoint_with_scores = model.run(None,ort_inputs)[0]
            else:
                keypoint_with_scores = model.run( input_image.astype(np.float32) )[0]

        # convert xy ratio for original image
        if image.shape[0] > image.shape[1]:
            keypoint_with_scores[0, 0, :, 1] = ( keypoint_with_scores[0, 0, :, 1] - padding_ratio ) / (1-2*padding_ratio)
        else:
            keypoint_with_scores[0, 0, :, 0] = ( keypoint_with_scores[0, 0, :, 0] - padding_ratio ) / (1-2*padding_ratio)
        
        # plot result
        result_image = movenet_utils.draw_prediction_on_image( image, keypoint_with_scores)
        savepath = get_savepath(args.savepath, image_path)
        logger.info(f'saved at : {savepath}')
        print(result_image.shape)
        cv2.imwrite(savepath, result_image)

        if args.write_json:
            save_json(keypoint_with_scores, (savepath.rsplit('.', 1)[0]) + '.json')

    logger.info('Script finished successfully.')

def recognize_from_video():
    
    # net initialize
    if args.onnx:
        import onnxruntime
        model = onnxruntime.InferenceSession(WEIGHT_PATH)
    else:
        logger.info(f'env_id: {args.env_id}')
        model = ailia.Net(MODEL_PATH, WEIGHT_PATH, env_id=args.env_id)

    capture = webcamera_utils.get_capture(args.video)
    first_frame_flag = False # to calculate crop region of first frame

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
        if frame_shown and cv2.getWindowProperty('result', cv2.WND_PROP_VISIBLE) == 0:
            break
        
        image_height, image_width, _ = frame.shape

        # calculate first frame crop region
        if first_frame_flag == False:
            crop_region = movenet_utils.init_crop_region(image_height, image_width)
            first_frame_flag = True

        crop_image = movenet_utils.crop_and_resize(frame,crop_region, IMAGE_SIZE)
        input_image = np.expand_dims(crop_image, axis=0)
                        
        if args.onnx:
            ort_inputs = { model.get_inputs()[0].name : input_image.astype(np.float32)}
            keypoints_with_scores = model.run(None,ort_inputs)[0]        
        else:
            keypoints_with_scores = model.run(input_image)[0]

        # convert xy ratio for original frame
        keypoints_with_scores[0, 0, :, 0] = ( crop_region['y_min'] * image_height + crop_region['height'] * image_height * keypoints_with_scores[0, 0, :, 0]) / image_height
        keypoints_with_scores[0, 0, :, 1] = ( crop_region['x_min'] * image_width + crop_region['width'] * image_width * keypoints_with_scores[0, 0, :, 1]) / image_width
        
        # draw keypoints on original frame
        result_image = movenet_utils.draw_prediction_on_image(frame,keypoints_with_scores)

        # update crop region
        crop_region = movenet_utils.determine_crop_region(keypoints_with_scores, image_height, image_width)
    
        cv2.imshow('result', result_image)
        frame_shown = True

        # save results
        if writer is not None:
            writer.write(frame)

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

