import os
import sys
import time
import numpy as np
import cv2
from logging import getLogger

sys.path.append('../../util')
import webcamera_utils  # noqa
from arg_utils import get_base_parser, update_parser, get_savepath  # noqa
from model_utils import check_and_download_file  # noqa
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
# from sam2_video_predictor import SAM2VideoPredictor

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


# def annotate_frame(points, labels, box, predictor, inference_state, image_encoder, prompt_encoder, mask_decoder, memory_attention, memory_encoder, mlp, obj_ptr_tpos_proj):
#     ann_frame_idx = 0  # the frame index we interact with
#     ann_obj_id = 1  # give a unique id to each object we interact with (it can be any integers)
# 
#     _, out_obj_ids, out_mask_logits = predictor.add_new_points_or_box(
#         inference_state=inference_state,
#         frame_idx=ann_frame_idx,
#         obj_id=ann_obj_id,
#         points=points,
#         labels=labels,
#         box=box,
#         image_encoder=image_encoder,
#         prompt_encoder=prompt_encoder,
#         mask_decoder=mask_decoder,
#         memory_attention=memory_attention,
#         memory_encoder=memory_encoder,
#         mlp=mlp
#     )
# 
#     predictor.propagate_in_video_preflight(inference_state,
#                                                                             image_encoder = image_encoder,
#                                                                             prompt_encoder = prompt_encoder,
#                                                                             mask_decoder = mask_decoder,
#                                                                             memory_attention = memory_attention,
#                                                                             memory_encoder = memory_encoder,
#                                                                             mlp = mlp,
#                                                                             obj_ptr_tpos_proj = obj_ptr_tpos_proj)


# def process_frame(image, frame_idx, predictor, inference_state, image_encoder, prompt_encoder, mask_decoder, memory_attention, memory_encoder, mlp, obj_ptr_tpos_proj):
#     out_frame_idx, out_obj_ids, out_mask_logits = predictor.propagate_in_video(inference_state,
#                                                                                 image_encoder = image_encoder,
#                                                                                 prompt_encoder = prompt_encoder,
#                                                                                 mask_decoder = mask_decoder,
#                                                                                 memory_attention = memory_attention,
#                                                                                 memory_encoder = memory_encoder,
#                                                                                 mlp = mlp,
#                                                                                 obj_ptr_tpos_proj = obj_ptr_tpos_proj,
#                                                                                 frame_idx = frame_idx)
# 
#     image = show_mask((out_mask_logits[0] > 0.0), image, color = np.array([30, 144, 255]), obj_id = out_obj_ids[0])
# 
#     return image


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

                # Logging
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

        if args.benchmark:
            return

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
    raise NotImplementedError
#     if args.video == 'demo':
#         frame_names = [
#             p for p in os.listdir(args.video)
#             if os.path.splitext(p)[-1] in ['.jpg', '.jpeg', '.JPG', '.JPEG']
#         ]
#         frame_names.sort(key=lambda p: int(os.path.splitext(p)[0]))
#         input_point = np.array([[210, 350], [250, 220]], dtype=np.float32)
#         input_label = np.array([1, 1], np.int32)
#         input_box = None
#         video_width = 960
#         video_height = 540
#     else:
#         frame_names = None
#         capture = webcamera_utils.get_capture(args.video)
#         video_height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
#         video_width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
#         input_point, input_label, input_box = get_input_point()
# 
#     if args.savepath != SAVE_IMAGE_PATH:
#         writer = webcamera_utils.get_writer(args.savepath, video_height, video_width)
#     else:
#         writer = None
# 
#     predictor = SAM2VideoPredictor(args.onnx, args.normal, args.benchmark)
# 
#     inference_state = predictor.init_state(args.num_mask_mem, args.max_obj_ptrs_in_encoder, args.version)
#     predictor.reset_state(inference_state)
# 
#     frame_shown = False
# 
#     if args.benchmark:
#         start = int(round(time.time() * 1000))
# 
#     frame_idx = 0
#     while (True):
#         if frame_names is None:
#             ret, frame = capture.read()
#         else:
#             ret = True
#             if frame_idx >= len(frame_names):
#                 break
#             frame = cv2.imread(os.path.join(args.video, frame_names[frame_idx]))
#             video_height = frame.shape[0]
#             video_width = frame.shape[1]
# 
#         if (cv2.waitKey(1) & 0xFF == ord('q')) or not ret:
#             break
#         if frame_shown and cv2.getWindowProperty('frame', cv2.WND_PROP_VISIBLE) == 0:
#             break
# 
#         image = preprocess_frame(frame, image_size=TARGET_SIZE)
# 
#         predictor.append_image(
#             inference_state,
#             image,
#             video_height,
#             video_width,
#             image_encoder)
# 
#         if frame_idx == 0:
#             annotate_frame(input_point, input_label, input_box, predictor, inference_state, image_encoder, prompt_encoder, mask_decoder, memory_attention, memory_encoder, mlp, obj_ptr_tpos_proj)
# 
#         frame = process_frame(frame, frame_idx, predictor, inference_state, image_encoder, prompt_encoder, mask_decoder, memory_attention, memory_encoder, mlp, obj_ptr_tpos_proj)
#         frame = frame.astype(np.uint8)
# 
#         if frame_idx == 0:
#             frame = show_points(input_point.astype(np.int64), input_label.astype(np.int64), frame)
#             frame = show_box(input_box, frame)
# 
#         cv2.imshow('frame', frame)
#         if frame_names is not None:
#             cv2.imwrite(f'video_{frame_idx}.png', frame)
# 
#         if writer is not None:
#             writer.write(frame)
# 
#         frame_shown = True
#         frame_idx = frame_idx + 1
# 
#     if args.benchmark:
#         end = int(round(time.time() * 1000))
#         estimation_time = (end - start)
#         logger.info(f'\ttotal processing time {estimation_time} ms')
# 
#     if writer is not None:
#         writer.release()


def main():
    WEIGHT_IMAGE_ENCODER_L_PATH = 'sam3_image_encoder.onnx'
    DATA_IMAGE_ENCODER_L_PATH = 'sam3_image_encoder.onnx.data'
    WEIGHT_PROMPT_ENCODER_L_PATH = 'sam3_language_encoder.onnx'
    DATA_PROMPT_ENCODER_L_PATH = 'sam3_language_encoder.onnx.data'
    WEIGHT_MASK_DECODER_L_PATH = 'sam3_decoder.onnx'
    DATA_MASK_DECODER_L_PATH = 'sam3_decoder.onnx.data'

    check_and_download_file(WEIGHT_IMAGE_ENCODER_L_PATH, REMOTE_PATH)
    check_and_download_file(DATA_IMAGE_ENCODER_L_PATH, REMOTE_PATH)
    check_and_download_file(WEIGHT_PROMPT_ENCODER_L_PATH, REMOTE_PATH)
    check_and_download_file(DATA_PROMPT_ENCODER_L_PATH, REMOTE_PATH)
    check_and_download_file(WEIGHT_MASK_DECODER_L_PATH, REMOTE_PATH)
    check_and_download_file(DATA_MASK_DECODER_L_PATH, REMOTE_PATH)

    if args.onnx:
        import onnxruntime
        image_encoder = onnxruntime.InferenceSession(WEIGHT_IMAGE_ENCODER_L_PATH)
        prompt_encoder = onnxruntime.InferenceSession(WEIGHT_PROMPT_ENCODER_L_PATH)
        mask_decoder = onnxruntime.InferenceSession(WEIGHT_MASK_DECODER_L_PATH)
    else:
        import ailia
        memory_mode = ailia.get_memory_mode(reduce_constant=True, ignore_input_with_initializer=True, reduce_interstage=False, reuse_interstage=True)
        image_encoder = ailia.Net(weight=WEIGHT_IMAGE_ENCODER_L_PATH, memory_mode=memory_mode, env_id=args.env_id)
        prompt_encoder = ailia.Net(weight=WEIGHT_PROMPT_ENCODER_L_PATH, memory_mode=memory_mode, env_id=args.env_id)
        mask_decoder = ailia.Net(weight=WEIGHT_MASK_DECODER_L_PATH, memory_mode=memory_mode, env_id=args.env_id)

    if args.video is not None:
        recognize_from_video(image_encoder, prompt_encoder, mask_decoder)
    else:
        recognize_from_image(image_encoder, prompt_encoder, mask_decoder)

    logger.info('Script finished successfully.')


if __name__ == '__main__':
    main()
