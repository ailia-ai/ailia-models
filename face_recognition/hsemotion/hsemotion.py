import os
import sys
import math
import time

import ailia
import cv2
import numpy as np

# import original modules
sys.path.append('../../util')
sys.path.append('../../face_detection/blazeface')

# logger
from logging import getLogger  # noqa: E402

import webcamera_utils  # noqa: E402
from blazeface_utils import compute_blazeface, crop_blazeface  # noqa: E402
from image_utils import imread  # noqa: E402
from model_utils import check_and_download_models  # noqa: E402
from arg_utils import get_base_parser, update_parser  # noqa: E402

logger = getLogger(__name__)

# ======================
# PARAMETERS
# ======================
ENET_B0_8_BEST_AFEW_WEIGHT_PATH = 'enet_b0_8_best_afew.onnx'
ENET_B0_8_BEST_AFEW_MODEL_PATH = 'enet_b0_8_best_afew.onnx.prototxt'
ENET_B0_8_BEST_VGAF_WEIGHT_PATH = 'enet_b0_8_best_vgaf.onnx'
ENET_B0_8_BEST_VGAF_MODEL_PATH = 'enet_b0_8_best_vgaf.onnx.prototxt'
ENET_B0_8_VA_MTL_WEIGHT_PATH = 'enet_b0_8_va_mtl.onnx'
ENET_B0_8_VA_MTL_MODEL_PATH = 'enet_b0_8_va_mtl.onnx.prototxt'
ENET_B2_8_WEIGHT_PATH = 'enet_b2_8.onnx'
ENET_B2_8_MODEL_PATH = 'enet_b2_8.onnx.prototxt'

REMOTE_PATH = \
    'https://storage.googleapis.com/ailia-models/hsemotion/'

IMAGE_PATH = 'lenna.png'
EMOTION_MAX_CLASS_COUNT = 4
SLEEP_TIME = 0

EMOTION_CATEGORY = [
    "Anger",
    "Contempt",
    "Disgust",
    "Fear",
    "Happiness",
    "Neutral",
    "Sadness",
    "Surprise"
]

FACE_WEIGHT_PATH = 'blazeface.onnx'
FACE_MODEL_PATH = 'blazeface.onnx.prototxt'
FACE_REMOTE_PATH = "https://storage.googleapis.com/ailia-models/blazeface/"
FACE_MARGIN = 1.0


# ======================
# Arguemnt Parser Config
# ======================
parser = get_base_parser(
    'High-Speed face Emotion recognition Model',
    IMAGE_PATH,
    None,
)
parser.add_argument(
    '-m', '--model_name',
    default='b0_8_best_afew',
    choices=['b0_8_best_afew', 'b0_8_best_vgaf', 'b0_8_va_mtl', 'b2_8']
)
args = update_parser(parser)

MODEL_NAME = args.model_name
WEIGHT_PATH = 'enet_' + MODEL_NAME + '.onnx'
MODEL_PATH = WEIGHT_PATH + '.prototxt'

IMG_SIZE = 224 if '_b0_' in MODEL_PATH else 260
IS_MTL = 'mtl' in MODEL_PATH
# ======================
# Main functions
# ======================
def preprocess(img):
    x = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    x = cv2.resize(x, (IMG_SIZE, IMG_SIZE)) / 255
    mean = np.array([[[0.485, 0.456, 0.406]]])
    std = np.array([[[0.229, 0.224, 0.225]]])
    x = (x - mean) / std
    return x.transpose(2, 0, 1)[np.newaxis, ...]

def postprocess(logits, max_class_count=None):
    logits = logits.squeeze()

    if IS_MTL:
        x = logits[:-2]
    else:
        x = logits

    x = np.exp(x - np.max(x)[np.newaxis])
    scores = x / x.sum()[np.newaxis]

    rank = np.argsort(-scores).squeeze()

    if max_class_count is not None:
        rank = rank[:max_class_count]
    
    return rank, scores

def recognize_from_image():
    hsemotion = ailia.Net(
        MODEL_PATH,
        WEIGHT_PATH,
        env_id=args.env_id,
        memory_mode=args.memory_mode,
    )

    # input image loop
    for image_path in args.input:
        # prepare input data
        logger.info(image_path)
        # load input image and convert to BGRA
        img = imread(image_path, cv2.IMREAD_UNCHANGED)
        if img.shape[2] == 4:
            img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
        elif img.shape[2] == 1:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

        img = preprocess(img)

        # inference emotion
        logger.info('Start inference...')
        if args.benchmark:
            logger.info('BENCHMARK mode')
            for i in range(5):
                start = int(round(time.time() * 1000))
                logits = hsemotion.predict(img)
                end = int(round(time.time() * 1000))
                logger.info(
                    f'\t[EMOTION MODEL] ailia processing time {end - start} ms'
                )
        else:
            logits = hsemotion.predict(img)
            
        rank, scores = postprocess(logits, EMOTION_MAX_CLASS_COUNT)
        logger.info(f'emotion_class_count={len(rank)}')

        # logger.info result
        for idx, category in enumerate(rank):
            logger.info(f'+ idx={idx}')
            logger.info(f'  category={category} '
                        f'[ {EMOTION_CATEGORY[category]} ]')
            logger.info(f'  prob={scores[category]}')
        logger.info('')
    logger.info('Script finished successfully.')


def recognize_from_video():
    # net initialize
    hsemotion = ailia.Net(
        MODEL_PATH,
        WEIGHT_PATH,
        env_id=args.env_id,
        memory_mode=args.memory_mode,
    )
    detector = ailia.Net(FACE_MODEL_PATH, FACE_WEIGHT_PATH, env_id=args.env_id, memory_mode=args.memory_mode)

    capture = webcamera_utils.get_capture(args.video)

    # create video writer if savepath is specified as video format
    if args.savepath is not None:
        logger.warning('[WARNING] currently video results output feature '
                       'is not supported in this model!')
        # TODO: shape should be debugged!
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

        # detect face
        # WIP: FIXME: AiliaInvalidArgumentException error
        detections = compute_blazeface(
            detector,
            frame,
            anchor_path='../../face_detection/blazeface/anchors.npy',
        )

        for obj in detections:
            # get detected face
            crop_img, top_left, bottom_right = crop_blazeface(
                obj, FACE_MARGIN, frame
            )
            if crop_img.shape[0] <= 0 or crop_img.shape[1] <= 0:
                continue
            crop_img = cv2.cvtColor(crop_img, cv2.COLOR_BGR2RGB)

            # emotion inference
            img = preprocess(crop_img)
            logits = hsemotion.predict(img)
            rank, scores = postprocess(logits, EMOTION_MAX_CLASS_COUNT)

            count = len(rank)
            logger.info('=' * 80)
            logger.info(f'emotion_class_count={count}')

            # logger.info result
            emotion_text = ""
            for idx, category in enumerate(rank):
                logger.info(f'+ idx={idx}')
                logger.info(
                    f'  category={category} ' +
                    f'[ {EMOTION_CATEGORY[category]} ]'
                )
                logger.info(f'  prob={scores[category]}')
                if idx == 0:
                    emotion_text = (f'[ {EMOTION_CATEGORY[category]} ] '
                                    f'prob={scores[category]:.3f}')
            logger.info('')

            # display label
            LABEL_WIDTH = 400
            LABEL_HEIGHT = 20
            color = (255, 255, 255)
            cv2.rectangle(frame, top_left, bottom_right, color, thickness=2)
            cv2.rectangle(
                frame,
                top_left,
                (top_left[0]+LABEL_WIDTH, top_left[1]+LABEL_HEIGHT),
                color,
                thickness=-1,
            )

            text_position = (top_left[0], top_left[1]+LABEL_HEIGHT//2)
            color = (0, 0, 0)
            fontScale = 0.5
            cv2.putText(
                frame,
                emotion_text,
                text_position,
                cv2.FONT_HERSHEY_SIMPLEX,
                fontScale,
                color,
                1,
            )

            # show result
            cv2.imshow('frame', frame)
            frame_shown = True
            time.sleep(SLEEP_TIME)

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
    
    check_and_download_models(
        WEIGHT_PATH, MODEL_PATH, REMOTE_PATH
    )
    
    if args.video:
        check_and_download_models(
            FACE_WEIGHT_PATH, FACE_MODEL_PATH, FACE_REMOTE_PATH
        )
    
    if args.video is not None:
        # video mode
        recognize_from_video()
    else:
        # image mode
        recognize_from_image()


if __name__ == '__main__':
    main()
