import cv2
import sys
import time
import numpy as np

import ailia
import onnxruntime

from  mobile_sam_util import *
from logging import getLogger

# import original modules
sys.path.append('../../util')
from arg_utils import get_base_parser, update_parser, get_savepath  # noqa
from model_utils import  check_and_download_models  # noqa
from image_utils import imread  # noqa

logger = getLogger(__name__)

# ======================
# Parameters
# ======================

WEIGHT_MOBILE_SAM_PATH = 'mobile_sam.onnx'
MODEL_MOVILE_SAM_PATH =  'mobile_sam.onnx.prototxt'

WEIGHT_PREDICTOR_PATH = 'predictor.onnx'
MODEL_PREDICTOR_PATH =  'predictor.onnx.prototxt'
REMOTE_PATH = 'https://storage.googleapis.com/ailia-models/mobile_sam/'

IMAGE_PATH = 'input.jpg'
SAVE_IMAGE_PATH = 'output.png'


POINT = np.array([[250, 375]])
LABEL = np.array([1])

#POINT= np.array([[250, 375], [490, 380], [375, 360]])
#LABEL= np.array([1, 1,0])

TARGET_LENGTH = 1024
threshold = 0.0


# ======================
# Arguemnt Parser Config
# ======================

parser = get_base_parser(
    'Mobile Sam', IMAGE_PATH, SAVE_IMAGE_PATH, fp16_support = False
)
parser.add_argument(
    '-p', '--pos', action='append', type=int, metavar="X", nargs=2,
    help='Positive coordinate specified by x,y.'
)
parser.add_argument(
    '--neg', action='append', type=int, metavar="X", nargs=2,
    help='Negative coordinate specified by x,y.'
)
parser.add_argument(
    '--onnx', action='store_true',
    help='execute onnxruntime version.'
)
args = update_parser(parser)

pos_points = []
neg_points = []

if args.pos is None:
    pos_points = POINT
else:
    pos_points = args.pos
    if args.neg is None:
        neg_points = []
    else:
        neg_points = args.neg
    POINT = np.vstack( (np.array(pos_points), np.array(neg_points).reshape(-1,2)))
    LABEL = [1] * len(pos_points) + [0] * len(neg_points)

POINT = np.array(POINT).astype(np.int32)
LABEL = np.array(LABEL).astype(np.int32)

# ======================
# Main functions
# ======================

def compute(sam_net,predictor,image_embedding,input_point,input_label,mask_input,has_mask_input,image):
    coord = np.concatenate([input_point, np.array([[0.0, 0.0]])], axis=0)[None, :, :]
    label = np.concatenate([input_label, np.array([-1])], axis=0)[None, :].astype(np.float32)
    coord = predictor.transform.apply_coords(coord, image.shape[:2]).astype(np.float32)

    if args.onnx:
        inputs = {
            "image_embeddings": image_embedding,
            "point_coords": coord,
            "point_labels": label,
            "mask_input": mask_input,
            "has_mask_input": has_mask_input,
            "orig_im_size": np.array(image.shape[:2], dtype=np.float32)
        }
        masks, _, low_res_logits = sam_net.run(None, inputs)
    else:
        inputs = (
            image_embedding,
            coord,
            label,
            mask_input,
            has_mask_input,
            np.array(image.shape[:2], dtype=np.float32)
            )
        masks, _, low_res_logits = sam_net.run(inputs)
    masks = masks > threshold
    return masks, low_res_logits


def recognize_from_image(pos_points, neg_points,env_id):

    lf = '\n'
    logger.info(f"Positive coordinate: {pos_points}")
    logger.info(f"Negative coordinate: {neg_points}")

    # input image loop
    for image_path in args.input:
        # inference
        logger.info('Start inference...')
        recognize(image_path, pos_points, neg_points, env_id)


def recognize(image_path,  pos_points, neg_points=None,env_id=0):


    image = imread(image_path)
    predictor = SamPredictor()
    if args.onnx:
        net = onnxruntime.InferenceSession(WEIGHT_PREDICTOR_PATH)
        sam_net = onnxruntime.InferenceSession(WEIGHT_MOBILE_SAM_PATH)
    else:
        net = ailia.Net(None,WEIGHT_PREDICTOR_PATH,env_id, memory_mode=args.memory_mode)
        sam_net = ailia.Net(None,WEIGHT_MOBILE_SAM_PATH,env_id, memory_mode=args.memory_mode)


    image_embedding = predictor.set_image(net,image,args.onnx)

    

    if args.benchmark:
        logger.info('BENCHMARK mode')
        total_time_estimation = 0
        for i in range(args.benchmark_count):
            start = int(round(time.time() * 1000))
            input_point = np.expand_dims(POINT[0],axis=0)
            input_label = np.expand_dims(LABEL[0],axis=0)
            mask_input = np.zeros((1, 1, 256, 256), dtype=np.float32)
            has_mask_input = np.zeros(1, dtype=np.float32)

            masks,low_res_logits = compute(sam_net,predictor,image_embedding,input_point,input_label,mask_input,has_mask_input,image)

            if len(POINT) > 1:
                input_point = POINT
                input_label = LABEL
                mask_input = low_res_logits
                has_mask_input = np.ones(1, dtype=np.float32)

                masks,low_res_logits = compute(sam_net,predictor,image_embedding,input_point,input_label,mask_input,has_mask_input,image)


            end = int(round(time.time() * 1000))
            estimation_time = (end - start)

            # Logging
            logger.info(f'\tailia processing estimation time {estimation_time} ms')
            if i != 0:
                total_time_estimation = total_time_estimation + estimation_time

        logger.info(f'\taverage time estimation {total_time_estimation / (args.benchmark_count - 1)} ms')
    else:

        input_point = np.expand_dims(POINT[0],axis=0)
        input_label = np.expand_dims(LABEL[0],axis=0)
        mask_input = np.zeros((1, 1, 256, 256), dtype=np.float32)
        has_mask_input = np.zeros(1, dtype=np.float32)

        masks,low_res_logits = compute(sam_net,predictor,image_embedding,input_point,input_label,mask_input,has_mask_input,image)

        if len(POINT) > 1:
            input_point = POINT
            input_label = LABEL
            mask_input = low_res_logits
            has_mask_input = np.ones(1, dtype=np.float32)

            masks,low_res_logits = compute(sam_net,predictor,image_embedding,input_point,input_label,mask_input,has_mask_input,image)

    image = show_mask(masks,image)
    image = show_points(input_point,input_label,image)

    res_img = image

    # plot result
    savepath = get_savepath(args.savepath, image_path, ext='.png')
    logger.info(f'saved at : {savepath}')
    cv2.imwrite(savepath, res_img)



def main():

    # model files check and download
    check_and_download_models(WEIGHT_MOBILE_SAM_PATH, MODEL_MOVILE_SAM_PATH, REMOTE_PATH)
    check_and_download_models(WEIGHT_PREDICTOR_PATH, MODEL_PREDICTOR_PATH, REMOTE_PATH)

    env_id = args.env_id

    # initialize
    recognize_from_image(args.pos, args.neg,env_id)

if __name__ == '__main__':
    main()
