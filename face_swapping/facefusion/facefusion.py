import cv2
import sys
import time
import numpy as np

import ailia

# import original modules
sys.path.append('../../util')
from arg_utils import get_base_parser, update_parser, get_savepath  # noqa
from model_utils import check_and_download_models  # noqa
from webcamera_utils import get_capture, get_writer  # noqa

# logger
from logging import getLogger  # noqa

from facefusion_utils.face_analyser import get_one_face, get_average_face
from facefusion_utils.utils import read_static_images, read_static_image, write_image
from facefusion_utils.face_store import append_reference_face
from facefusion_utils.face_swapper import pre_process, post_process
from facefusion_utils.face_swapper import get_reference_frame as fs_get_reference_frame, process_image as fs_process_image
from facefusion_utils.face_enhancer import get_reference_frame as fe_get_reference_frame, process_image as fe_process_image

logger = getLogger(__name__)

# ======================
# Parameters
# ======================

WEIGHT_FACE_DETECTOR_PATH = 'yoloface_8n.onnx'
MODEL_FACE_DETECTOR_PATH = 'yoloface_8n.onnx.prototxt'
WEIGHT_FACE_LANDMARKER_PATH = '2dfan4.onnx'
MODEL_FACE_LANDMARKER_PATH = '2dfan4.onnx.prototxt'
WEIGHT_FACE_RECOGNIZER_PATH = 'arcface_w600k_r50.onnx'
MODEL_FACE_RECOGNIZER_PATH = 'arcface_w600k_r50.onnx.prototxt'
WEIGHT_FACE_SWAPPER_PATH = 'inswapper_128.onnx'
MODEL_FACE_SWAPPER_PATH = 'inswapper_128.onnx.prototxt'
WEIGHT_FACE_ENHANCER_PATH = 'gfpgan_1.4.onnx'
MODEL_FACE_ENHANCER_PATH = 'gfpgan_1.4.onnx.prototxt'
REMOTE_PATH = 'https://storage.googleapis.com/ailia-models/facefusion/'

MODEL_MATRIX_PATH = 'model_matrix.npy'

TARGET_IMAGE_PATH = 'target.jpg'
SOURCE_IMAGE_PATH = 'source.jpg'
SAVE_IMAGE_PATH = 'output.jpg'

FACE_DETECTOR_SCORE = 0.5
REFERENCE_FACE_DISTANCE = 0.6

# ======================
# Argument Parser Config
# ======================

parser = get_base_parser(
    'Facefusion', TARGET_IMAGE_PATH, SAVE_IMAGE_PATH
)
parser.add_argument(
    '-src', '--source', nargs='+', metavar='IMAGE', default=[SOURCE_IMAGE_PATH,],
    help=('The source image(s) to swap the face from.')
)
parser.add_argument(
    '--skip_enhance', action='store_true',
    help='Whether to skip face enhancement using GFPGAN.'
)
parser.add_argument(
    '-th', '--threshold', type=float, default=FACE_DETECTOR_SCORE,
    help='Face detector score threshold.'
)
parser.add_argument(
    '-dist', '--face_distance', type=float, default=REFERENCE_FACE_DISTANCE,
    help='Face distance similarity score threshold.'
)
parser.add_argument(
    '--onnx',
    action='store_true',
    help='Execute onnxruntime version.'
)
args = update_parser(parser)

# ======================
# Secondary functions
# ======================

def get_model_matrix(model_matrix_path):
    return np.load(model_matrix_path)

def conditional_append_reference_faces(source_img_paths, target_img_path, nets):
    source_frames = read_static_images(source_img_paths)
    source_face = get_average_face(source_frames, nets, FACE_DETECTOR_SCORE)

    reference_frame = read_static_image(target_img_path)
    reference_face = get_one_face(reference_frame, nets, FACE_DETECTOR_SCORE)
    append_reference_face('origin', reference_face)
    if source_face and reference_face:
        abstract_reference_frame = fs_get_reference_frame(source_face, reference_face, reference_frame, nets)
        if np.any(abstract_reference_frame):
            reference_frame = abstract_reference_frame
            reference_face = get_one_face(reference_frame, nets, FACE_DETECTOR_SCORE)
            append_reference_face('face_swapper', reference_face)

        if 'face_enhancer' in nets:
            abstract_reference_frame = fe_get_reference_frame(source_face, reference_face, reference_frame, nets)
            if np.any(abstract_reference_frame):
                reference_frame = abstract_reference_frame
                reference_face = get_one_face(reference_frame, nets, FACE_DETECTOR_SCORE)
                append_reference_face('face_enhancer', reference_face)


def process_image(source_img_paths, target_img_path, nets):
    res_image = fs_process_image(source_img_paths, target_img_path, REFERENCE_FACE_DISTANCE, nets, FACE_DETECTOR_SCORE)

    if 'face_enhancer' in nets:
        res_image = fe_process_image(source_img_paths, res_image, REFERENCE_FACE_DISTANCE, nets, FACE_DETECTOR_SCORE)

    post_process()
    return res_image

# ======================
# Main functions
# ======================

def recognize_from_image(nets):
    # input image loop
    for target_image_path in args.input:
        logger.info(target_image_path)

        # inference
        logger.info('Start inference...')
        if args.benchmark:
            logger.info('BENCHMARK mode')
            total_time = 0
            for i in range(args.benchmark_count):
                start = int(round(time.time() * 1000))
                # prepare input data
                pre_process(args.source, nets, FACE_DETECTOR_SCORE)
                conditional_append_reference_faces(args.source, target_image_path, nets)
                res_image = process_image(args.source, target_image_path, nets)
                end = int(round(time.time() * 1000))
                if i != 0:
                    total_time = total_time + (end - start)
                logger.info(f'\tailia processing time {end - start} ms')
            logger.info(f'\taverage time {total_time / (args.benchmark_count - 1)} ms')
        else:
            # prepare input data
            pre_process(args.source, nets, FACE_DETECTOR_SCORE)
            conditional_append_reference_faces(args.source, target_image_path, nets)
            res_image = process_image(args.source, target_image_path, nets)

        # save result
        savepath = get_savepath(args.savepath, target_image_path, ext='.png')
        logger.info(f'saved at : {savepath}')
        write_image(savepath, res_image)

    logger.info('Script finished successfully.')

def recognize_from_video(nets):
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

        # prepare input data
        pre_process(args.source, nets, FACE_DETECTOR_SCORE)
        conditional_append_reference_faces(args.source, frame, nets)

        # inference
        res_image = process_image(args.source, frame, nets)

        # show
        cv2.imshow('frame', res_image)
        frame_shown = True

        # save results
        if writer is not None:
            res_image = res_image.astype(np.uint8)
            writer.write(res_image)

    capture.release()
    cv2.destroyAllWindows()
    if writer is not None:
        writer.release()
    logger.info('Script finished successfully.')


def main():
    dic_model = {
        'face_detector': (WEIGHT_FACE_DETECTOR_PATH, MODEL_FACE_DETECTOR_PATH),
        'face_landmarker': (WEIGHT_FACE_LANDMARKER_PATH, MODEL_FACE_LANDMARKER_PATH),
        'face_recognizer': (WEIGHT_FACE_RECOGNIZER_PATH, MODEL_FACE_RECOGNIZER_PATH),
        'face_swapper': (WEIGHT_FACE_SWAPPER_PATH, MODEL_FACE_SWAPPER_PATH)
    }

    if not args.skip_enhance:
        dic_model['face_enhancer'] = (WEIGHT_FACE_ENHANCER_PATH, MODEL_FACE_ENHANCER_PATH)

    # model files check and download
    for weight_path, model_path in dic_model.values():
        check_and_download_models(weight_path, model_path, REMOTE_PATH)

    # initialize
    if not args.onnx:
        nets = {k: ailia.Net(v[1], v[0], env_id=0, memory_mode=args.memory_mode) for k, v in dic_model.items()}
    else:
        import onnxruntime
        cuda = 0 < ailia.get_gpu_environment_id()
        providers = ['CUDAExecutionProvider', 'CPUExecutionProvider'] if cuda else ['CPUExecutionProvider']
        nets = {k: onnxruntime.InferenceSession(v[0], providers=providers) for k, v in dic_model.items()}

    nets['is_onnx'] = args.onnx
    nets['model_matrix'] = get_model_matrix(MODEL_MATRIX_PATH)

    if args.video is not None:
        # video mode
        recognize_from_video(nets)
    else:
        # image mode
        recognize_from_image(nets)


if __name__ == '__main__':
    main()
