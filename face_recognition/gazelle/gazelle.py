import os
import sys
import time
import numpy as np
import cv2
import ast

import ailia

# import original modules
sys.path.append('../../util')
import webcamera_utils  # noqa: E402
from image_utils import imread, load_image  # noqa: E402
from model_utils import check_and_download_models  # noqa: E402
from arg_utils import get_base_parser, get_savepath, update_parser  # noqa: E402

from gazelle_utils.visualization import visualize_heatmap, visualize_all
from gazelle_utils.model import GazeLLE

# logger
from logging import getLogger  # noqa: E402
logger = getLogger(__name__)

# ======================
# Parameters
# ======================
IMAGE_PATH = 'input.png'
SAVE_IMAGE_PATH = 'output.png'
RESIZE_HEIGHT = 448
RESIZE_WIDTH = 448

MODEL_LIST = ['vitb14', 'vitl14', 'vitb14_inout', 'vitl14_inout']
FACE_MODEL_LIST = ['ailia-retinaface_resnet50', 'ailia-retinaface_mobile0.25', 'retina-face']

# ======================
# Arguemnt Parser Config
# ======================
parser = get_base_parser("gazelle", IMAGE_PATH, SAVE_IMAGE_PATH)
parser.add_argument(
    '-a', '--arch',
    default='vitl14_inout',
    choices=MODEL_LIST,
    help='model lists for backbone: ' + ' | '.join(MODEL_LIST)
)
parser.add_argument(
    '--face_detect_arch',
    default='ailia-retinaface_resnet50',
    choices=FACE_MODEL_LIST,
    help=('model lists for face detect: ' + ' | '.join(FACE_MODEL_LIST) + ' | '
          'if you want to use retina-face package, please install it using `pip3 install retina-face`.')
)
parser.add_argument(
    '--heatmap',
    action='store_true',
    help='Flag to output heatmap. Results will be saved in the "outputs" directory.'
)
parser.add_argument(
    '--input_bboxes',
    default=None,
    type=str,
    help=('Input bounding boxes for face detection. '
          'The format is "[[(xmin, ymin, xmax, ymax)]]" and are in [0,1] normalized image coordinates.'
          'If input_bboxes is [[None]], --heatmap option is required.')
)
args = update_parser(parser)

# ======================
# Parameters 2
# ======================
WEIGHT_BACKBONE_PATH = f'gazelle_backbone_{args.arch.replace("_inout", "")}.onnx'
MODEL_BACKBONE_PATH = WEIGHT_BACKBONE_PATH + '.prototxt'
WEIGHT_DECODER_PATH = f'gazelle_decoder_{args.arch}.onnx'
MODEL_DECODER_PATH = WEIGHT_DECODER_PATH + '.prototxt'
REMOTE_PATH = 'https://storage.googleapis.com/ailia-models/gazelle/'

INOUT = True if 'inout' in args.arch else False

if args.face_detect_arch != 'retina-face':
    FACE_WEIGHT_PATH = f'{args.face_detect_arch.replace("ailia-", "")}.onnx'
    FACE_MODEL_PATH = FACE_WEIGHT_PATH + '.prototxt'
    FACE_REMOTE_PATH = "https://storage.googleapis.com/ailia-models/retinaface/"

# ======================
# Utils
# ======================
def detect_faces(image):
    # use retina-face package
    if args.face_detect_arch == 'retina-face':
        from retinaface import RetinaFace
        resp = RetinaFace.detect_faces(image)
        bboxes = [resp[key]['facial_area'] for key in resp.keys()]
    # use ailia model
    else:
        from gazelle_utils.face_detection import ailia_face_detect
        check_and_download_models(FACE_WEIGHT_PATH, FACE_MODEL_PATH, FACE_REMOTE_PATH)
        net_retinaface = ailia.Net(FACE_MODEL_PATH, FACE_WEIGHT_PATH, env_id=args.env_id, memory_mode=args.memory_mode)
        bboxes = ailia_face_detect(image, net_retinaface, args.face_detect_arch)
    return bboxes

# ======================
# Main functions
# ======================
def recognize_from_image(model):
    # input image loop
    for image_path in args.input:
        # prepare input data
        logger.info(image_path)
        ori_image = imread(image_path)
        width, height = ori_image.shape[1], ori_image.shape[0]
        image = load_image(
            image_path,
            (RESIZE_HEIGHT, RESIZE_WIDTH),
            rgb=True,
            normalize_type="ImageNet",
            gen_input_ailia=True,
        )

        # prepare bounding boxes
        if args.input_bboxes is None:
            bboxes = detect_faces(ori_image)
            norm_bboxes = [[np.array(bbox) / np.array([width, height, width, height]) for bbox in bboxes]]
        else:
            norm_bboxes = ast.literal_eval(args.input_bboxes)

        # inference
        logger.info('Start inference...')
        if args.benchmark:
            logger.info('BENCHMARK mode')
            for i in range(5):
                start = int(round(time.time() * 1000))
                output = model.run(image, norm_bboxes)
                end = int(round(time.time() * 1000))
                logger.info(f'\tailia processing time {end - start} ms')
        else:
            output = model.run(image, norm_bboxes)

        # save results
        if args.heatmap:
            for i in range(len(norm_bboxes[0])):
                result = visualize_heatmap(
                    ori_image, 
                    output['heatmap'][0][i], 
                    norm_bboxes[0][i], 
                    inout_score=output['inout'][0][i] if output['inout'] is not None else None
                )
                os.makedirs('outputs', exist_ok=True)
                cv2.imwrite(f'outputs/heatmap_{i}.png', result)
        else:
            result = visualize_all(
                ori_image, 
                output['heatmap'][0], 
                norm_bboxes[0], 
                output['inout'][0] if output['inout'] is not None else None
            )
            savepath = get_savepath(args.savepath, image_path)
            logger.info(f'saved at : {savepath}')
            cv2.imwrite(savepath, result)
    logger.info('Script finished successfully.')


def recognize_from_video(model):
    capture = webcamera_utils.get_capture(args.video)
    f_h = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    f_w = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))

    # create video writer if savepath is specified as video format
    if args.savepath != SAVE_IMAGE_PATH:
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

        # prepare input data
        _, preprocessed_frame = webcamera_utils.preprocess_frame(
            frame,
            RESIZE_HEIGHT,
            RESIZE_WIDTH,
            data_rgb=True,
            normalize_type="ImageNet",
        )

        # prepare bounding boxes
        if args.input_bboxes is None:
            bboxes = detect_faces(frame)
            norm_bboxes = [[np.array(bbox) / np.array([f_w, f_h, f_w, f_h]) for bbox in bboxes]]
        else:
            norm_bboxes = ast.literal_eval(args.input_bboxes)

        # inference
        output = model.run(preprocessed_frame, norm_bboxes)

        # visualize results
        if args.heatmap:
            result = visualize_heatmap(
                frame, 
                output['heatmap'][0][0], 
                norm_bboxes[0][0], 
                inout_score=output['inout'][0][0] if output['inout'] is not None else None
            )
        else:
            result = visualize_all(
                frame, 
                output['heatmap'][0], 
                norm_bboxes[0], 
                output['inout'][0] if output['inout'] is not None else None
            )
        
        cv2.imshow('frame', result)
        frame_shown = True
        # save results
        if writer is not None:
            writer.write(result)

    capture.release()
    cv2.destroyAllWindows()
    if writer is not None:
        writer.release()
    logger.info('Script finished successfully.')


def main():
    # model files check and download
    check_and_download_models(WEIGHT_BACKBONE_PATH, MODEL_BACKBONE_PATH, REMOTE_PATH)
    check_and_download_models(WEIGHT_DECODER_PATH, MODEL_DECODER_PATH, REMOTE_PATH)

    # model initialize
    net_backbone = ailia.Net(MODEL_BACKBONE_PATH, WEIGHT_BACKBONE_PATH, env_id=args.env_id, memory_mode=args.memory_mode)
    net_decoder = ailia.Net(MODEL_DECODER_PATH, WEIGHT_DECODER_PATH, env_id=args.env_id, memory_mode=args.memory_mode)
    model = GazeLLE(backbone=net_backbone, decoder=net_decoder, inout=INOUT)

    if args.video is not None:
        # video mode
        recognize_from_video(model)
    else:
        # image mode
        recognize_from_image(model)

if __name__ == '__main__':
    main()
