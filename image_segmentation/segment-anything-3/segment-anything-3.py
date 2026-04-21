import os
import re
import sys
import time
import numpy as np
import cv2
from logging import getLogger

sys.path.append('../../util')
import webcamera_utils  # noqa
from arg_utils import get_base_parser, update_parser, get_savepath  # noqa
from model_utils import check_and_download_models  # noqa
from webcamera_utils import get_capture, get_writer  # noqa

logger = getLogger(__name__)


# ======================
# Parameters
# ======================

IMAGE_PATH = 'demo/truck.jpg'
SAVE_IMAGE_PATH = 'output.png'

TARGET_SIZE = 1008


# ======================
# Argument Parser Config
# ======================

parser = get_base_parser(
    'Segment Anything 3', IMAGE_PATH, SAVE_IMAGE_PATH
)
parser.add_argument(
    '--box', type=int, metavar='X', nargs=4,
    help='Box coordinate specified by x1,y1,x2,y2.'
)
parser.add_argument(
    '--prompt', type=str, default='visual',
    help='Text prompt for segmentation.',
)
parser.add_argument(
    '--onnx', action='store_true',
    help='execute onnxruntime version.'
)

args = update_parser(parser)


# ======================
# Model path
# ======================

REMOTE_PATH = 'https://storage.googleapis.com/ailia-models/segment-anything-3/'


# ======================
# Support
# ======================

from sam3_image_predictor import SAM3ImagePredictor

np.random.seed(3)


def show_mask(mask, image, color=np.array([255, 144, 30]), obj_id=None):
    color = color.reshape(1, 1, -1)

    h, w = mask.shape[-2:]
    mask = mask.reshape(h, w, 1)

    mask_image = mask * color
    image = (image * ~mask) + (image * mask) * 0.6 + mask_image * 0.4

    return image


def show_box(box, image):
    if box is None:
        return image

    cv2.rectangle(
        image, (box[0], box[1]), (box[2], box[3]),
        color=(2, 118, 2),
        thickness=3,
        lineType=cv2.LINE_4,
        shift=0,
    )

    return image


def get_input_point():
    box = args.box
    if box:
        input_box = np.array(box)
    else:
        input_box = None
    return input_box


def preprocess_frame(image, image_size):
    image_mean = (0.485, 0.456, 0.406)
    image_std = (0.229, 0.224, 0.225)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (image_size, image_size))
    image = image / 255.0
    image = image - image_mean
    image = image / image_std
    image = np.transpose(image, (2, 0, 1))
    return image


# ======================
# Main
# ======================

def recognize_from_image(image_encoder, prompt_encoder, mask_decoder):
    image_predictor = SAM3ImagePredictor()

    input_box = get_input_point()
    prompt = args.prompt

    for image_path in args.input:
        image = cv2.imread(image_path)
        orig_hw = [image.shape[0], image.shape[1]]
        image_np = preprocess_frame(image, image_size=TARGET_SIZE)

        if args.benchmark:
            logger.info('BENCHMARK mode')
            total_time_estimation = 0
            for i in range(args.benchmark_count):
                start = int(round(time.time() * 1000))
                features = image_predictor.set_image(image_np, image_encoder, args.onnx)
                scores, boxes, masks = image_predictor.predict(
                    orig_hw=orig_hw,
                    features=features,
                    prompt=prompt,
                    box=input_box,
                    prompt_encoder=prompt_encoder,
                    mask_decoder=mask_decoder,
                    onnx=args.onnx
                )
                end = int(round(time.time() * 1000))
                estimation_time = (end - start)

                logger.info(f'\tailia processing estimation time {estimation_time} ms')
                if i != 0:
                    total_time_estimation = total_time_estimation + estimation_time

            logger.info(f'\taverage time estimation {total_time_estimation / (args.benchmark_count - 1)} ms')
        else:
            features = image_predictor.set_image(image_np, image_encoder, args.onnx)
            scores, masks, boxes = image_predictor.predict(
                orig_hw=orig_hw,
                features=features,
                prompt=prompt,
                box=input_box,
                prompt_encoder=prompt_encoder,
                mask_decoder=mask_decoder,
                onnx=args.onnx
            )

            sorted_ind = np.argsort(scores)[::-1]
            scores = scores[sorted_ind]
            masks = masks[sorted_ind]
            boxes = boxes[sorted_ind]

            savepath = get_savepath(args.savepath, image_path, ext='.png')
            for mask in masks:
                image = show_mask(mask, image)
            image = show_box(input_box, image)
            cv2.imwrite(savepath, image.astype(np.uint8))
            logger.info(f'saved at : {savepath}')


def recognize_from_video(image_encoder, prompt_encoder, mask_decoder):
    if args.video == 'demo':
        frame_names = [
            p for p in os.listdir(args.video)
            if re.match(r'^\d+\.(jpg|jpeg)$', p, re.IGNORECASE) is not None
        ]
        frame_names.sort(key=lambda p: int(os.path.splitext(p)[0]))
        input_point = np.array([[210, 350], [250, 220]], dtype=np.float32)
        input_label = np.array([1, 1], np.int32)
        input_box = None
        video_width = 960
        video_height = 540
    else:
        frame_names = None
        capture = webcamera_utils.get_capture(args.video)
        video_height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        video_width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        input_point, input_label, input_box = get_input_point()

    if args.savepath != SAVE_IMAGE_PATH:
        writer = webcamera_utils.get_writer(args.savepath, video_height, video_width)
    else:
        writer = None

    image_predictor = SAM3ImagePredictor()

    input_box = get_input_point()
    prompt = args.prompt

    if args.benchmark:
        start = int(round(time.time() * 1000))

    frame_shown = False
    frame_idx = 0

    while True:
        if frame_names is None:
            ret, frame = capture.read()
        else:
            ret = True
            if frame_idx >= len(frame_names):
                break
            frame = cv2.imread(os.path.join(args.video, frame_names[frame_idx]))
            video_height = frame.shape[0]
            video_width = frame.shape[1]

        if (cv2.waitKey(1) & 0xFF == ord('q')) or not ret:
            break

        if frame_shown and cv2.getWindowProperty('frame', cv2.WND_PROP_VISIBLE) == 0:
            break

        orig_hw = [frame.shape[0], frame.shape[1]]
        frame_np = preprocess_frame(frame, image_size=TARGET_SIZE)

        features = image_predictor.set_image(frame_np, image_encoder, args.onnx)
        scores, masks, boxes = image_predictor.predict(
            orig_hw=orig_hw,
            features=features,
            prompt=prompt,
            box=input_box,
            prompt_encoder=prompt_encoder,
            mask_decoder=mask_decoder,
            onnx=args.onnx
        )

        if not args.benchmark:
            sorted_ind = np.argsort(scores)[::-1]
            scores = scores[sorted_ind]
            masks = masks[sorted_ind]
            boxes = boxes[sorted_ind]

            for mask in masks:
                frame = show_mask(mask, frame)
            frame = show_box(input_box, frame)

            cv2.imshow('frame', frame)
            frame_shown = True

            if frame_names is not None:
                cv2.imwrite(f'video_{frame_idx}.png', frame)

            if writer is not None:
                writer.write(frame)

        frame_idx = frame_idx + 1

    if args.benchmark:
        end = int(round(time.time() * 1000))
        estimation_time = (end - start)
        logger.info(f'\ttotal processing time {estimation_time} ms')

    if writer is not None:
        writer.release()


def main():
    WEIGHT_IMAGE_ENCODER_L_PATH = 'sam3_image_encoder.onnx'
    MODEL_IMAGE_ENCODER_L_PATH = 'sam3_image_encoder.onnx.prototxt'
    WEIGHT_PROMPT_ENCODER_L_PATH = 'sam3_language_encoder.onnx'
    MODEL_PROMPT_ENCODER_L_PATH = 'sam3_language_encoder.onnx.prototxt'
    WEIGHT_MASK_DECODER_L_PATH = 'sam3_decoder.onnx'
    MODEL_MASK_DECODER_L_PATH = 'sam3_decoder.onnx.prototxt'

    check_and_download_models(WEIGHT_IMAGE_ENCODER_L_PATH, MODEL_IMAGE_ENCODER_L_PATH, REMOTE_PATH)
    check_and_download_models(WEIGHT_PROMPT_ENCODER_L_PATH, MODEL_PROMPT_ENCODER_L_PATH, REMOTE_PATH)
    check_and_download_models(WEIGHT_MASK_DECODER_L_PATH, MODEL_MASK_DECODER_L_PATH, REMOTE_PATH)

    if args.onnx:
        import onnxruntime
        image_encoder = onnxruntime.InferenceSession(WEIGHT_IMAGE_ENCODER_L_PATH)
        prompt_encoder = onnxruntime.InferenceSession(WEIGHT_PROMPT_ENCODER_L_PATH)
        mask_decoder = onnxruntime.InferenceSession(WEIGHT_MASK_DECODER_L_PATH)
    else:
        import ailia
        memory_mode = ailia.get_memory_mode(reduce_constant=True, ignore_input_with_initializer=True, reduce_interstage=False, reuse_interstage=True)
        image_encoder = ailia.Net(weight=WEIGHT_IMAGE_ENCODER_L_PATH, stream=MODEL_IMAGE_ENCODER_L_PATH, memory_mode=memory_mode, env_id=args.env_id)
        prompt_encoder = ailia.Net(weight=WEIGHT_PROMPT_ENCODER_L_PATH, stream=MODEL_PROMPT_ENCODER_L_PATH, memory_mode=memory_mode, env_id=args.env_id)
        mask_decoder = ailia.Net(weight=WEIGHT_MASK_DECODER_L_PATH, stream=MODEL_MASK_DECODER_L_PATH, memory_mode=memory_mode, env_id=args.env_id)

    if args.video is not None:
        recognize_from_video(image_encoder, prompt_encoder, mask_decoder)
    else:
        recognize_from_image(image_encoder, prompt_encoder, mask_decoder)

    logger.info('Script finished successfully.')


if __name__ == '__main__':
    main()
