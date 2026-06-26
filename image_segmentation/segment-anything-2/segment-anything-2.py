import sys
import os
import time
from logging import getLogger

import numpy as np
import cv2

import os
import numpy as np

# import original modules
sys.path.append('../../util')
import webcamera_utils
from arg_utils import get_base_parser, update_parser, get_savepath  # noqa
from model_utils import check_and_download_models  # noqa
from webcamera_utils import get_capture, get_writer  # noqa

logger = getLogger(__name__)

# ======================
# Parameters
# ======================

IMAGE_PATH = 'truck.jpg'
SAVE_IMAGE_PATH = 'output.png'

POINT1 = (500, 375)
POINT2 = (1125, 625)

TARGET_LENGTH = 1024

# ======================
# Arguemnt Parser Config
# ======================

parser = get_base_parser(
    'Segment Anything 2', IMAGE_PATH, SAVE_IMAGE_PATH
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
    '--box', type=int, metavar="X", nargs=4,
    help='Box coordinate specified by x1,y1,x2,y2.'
)
parser.add_argument(
    '--mask', type=str, default=None,
    help='Mask image path (grayscale, white=foreground) for mask prompt input.'
)
parser.add_argument(
    '--num_mask_mem', type=int, default=7, choices=(0, 1, 2, 3, 4, 5, 6, 7),
    help='Number of mask mem. (default 1 input frame + 6 previous frames)'
)
parser.add_argument(
    '--max_obj_ptrs_in_encoder', type=int, default=16, choices=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15),
    help='Number of obj ptr in encoder.'
)
parser.add_argument(
    '-m', '--model_type', default='hiera_l', choices=('hiera_l', 'hiera_b+', 'hiera_s', 'hiera_t'),
    help='Select model.'
)
parser.add_argument(
    '--onnx', action='store_true',
    help='execute onnxruntime version.'
)
parser.add_argument(
    '--version', default='2', choices=('2', '2.1'),
    help='Select model.'
)
parser.add_argument(
    '--legacy', action='store_true',
    help='Use legacy ONNX model. (4D matmul with batch=1, mask prompt not supported)'
)
parser.add_argument(
    '--auto', action='store_true',
    help='Automatic mask generation mode. Segment all objects in the image.'
)
parser.add_argument(
    '--points_per_side', type=int, default=32,
    help='Grid density for auto mode. (default: 32, generates 32x32=1024 points)'
)
parser.add_argument(
    '--points_per_batch', type=int, default=64,
    help='Batch size for auto mode inference. (default: 64)'
)
parser.add_argument(
    '--pred_iou_thresh', type=float, default=0.8,
    help='Predicted IoU threshold for auto mode. (default: 0.8)'
)
parser.add_argument(
    '--stability_score_thresh', type=float, default=0.95,
    help='Stability score threshold for auto mode. (default: 0.95)'
)
parser.add_argument(
    '--box_nms_thresh', type=float, default=0.7,
    help='Box NMS IoU threshold for auto mode. (default: 0.7)'
)

args = update_parser(parser)


# ======================
# Model path
# ======================

if args.version == "2.1":
    REMOTE_PATH = 'https://storage.googleapis.com/ailia-models/segment-anything-2.1/'
else:
    REMOTE_PATH = 'https://storage.googleapis.com/ailia-models/segment-anything-2/'


# ======================
# Utility
# ======================

np.random.seed(3)


def show_mask(mask, img, color = np.array([255, 144, 30]), obj_id=None):
    color = color.reshape(1, 1, -1)

    h, w = mask.shape[-2:]
    mask = mask.reshape(h, w, 1)

    mask_image = mask * color
    img = (img * ~mask) + (img * mask) * 0.5 + mask_image * 0.5

    return img


def show_points(coords, labels, img):
    pos_points = coords[labels == 1]
    neg_points = coords[labels == 0]

    for p in pos_points:
        cv2.drawMarker(
            img, p, (0, 255, 0), markerType=cv2.MARKER_TILTED_CROSS, line_type=cv2.LINE_AA,
            markerSize=30, thickness=5)
    for p in neg_points:
        cv2.drawMarker(
            img, p, (0, 0, 255), markerType=cv2.MARKER_TILTED_CROSS, line_type=cv2.LINE_AA,
            markerSize=30, thickness=5)

    return img


def show_box(box, img):
    if box is None:
        return img

    cv2.rectangle(
        img, (box[0], box[1]), (box[2], box[3]), color=(2, 118, 2),
        thickness=3,
        lineType=cv2.LINE_4,
        shift=0)

    return img

# ======================
# Logic
# ======================

from sam2_image_predictor import SAM2ImagePredictor
from sam2_video_predictor import SAM2VideoPredictor
from sam2_automatic_mask_generator import SAM2AutomaticMaskGenerator

# ======================
# Main
# ======================


def get_input_point():
    pos_points = args.pos
    neg_points = args.neg
    box = args.box

    if pos_points is None:
        if neg_points is None and box is None:
            pos_points = [POINT1]
        else:
            pos_points = []
    if neg_points is None:
        neg_points = []

    input_point = []
    input_label = []
    if pos_points:
        for i in range(len(pos_points)):
            input_point.append(pos_points[i])
            input_label.append(1)
    if neg_points:
        for i in range(len(neg_points)):
            input_point.append(neg_points[i])
            input_label.append(0)
    input_point = np.array(input_point)
    input_label = np.array(input_label)
    input_box = None
    if box:
        input_box = np.array(box)
    return input_point, input_label, input_box


def load_mask_input(mask_path):
    """Load a mask image and convert to mask logits for prompt encoder input."""
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise FileNotFoundError(f"Mask image not found: {mask_path}")
    # Resize to prompt encoder mask input size (256x256)
    mask = cv2.resize(mask, (256, 256), interpolation=cv2.INTER_LINEAR)
    # Convert to logits: foreground (>128) -> positive, background -> negative
    mask_logits = np.where(mask > 128, 10.0, -10.0).astype(np.float32)
    # Shape: [1, 256, 256]
    mask_logits = mask_logits[None, :, :]
    return mask_logits


def recognize_from_image(image_encoder, prompt_encoder, mask_decoder):
    input_point, input_label, input_box = get_input_point()

    # Load mask input if specified
    mask_input = None
    if args.mask is not None:
        if args.legacy:
            raise RuntimeError("--mask requires new prompt_encoder model. Cannot be used with --legacy.")
        mask_input = load_mask_input(args.mask)
        logger.info(f'Mask prompt loaded from: {args.mask}')

    image_predictor = SAM2ImagePredictor(args.legacy, args.version, args.model_type)

    for image_path in args.input:
        image = cv2.imread(image_path)
        orig_hw = [image.shape[0], image.shape[1]]
        image_size = 1024
        image_np = preprocess_frame(image, image_size=image_size)

        if args.benchmark:
            logger.info('BENCHMARK mode')
            total_time_estimation = 0
            for i in range(args.benchmark_count):
                start = int(round(time.time() * 1000))
                features = image_predictor.set_image(image_np, image_encoder, args.onnx)
                masks, scores, logits = image_predictor.predict(
                    orig_hw=orig_hw,
                    features=features,
                    point_coords=input_point,
                    point_labels=input_label,
                    box=input_box,
                    mask_input=mask_input,
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
            masks, scores, logits = image_predictor.predict(
                orig_hw=orig_hw,
                features=features,
                point_coords=input_point,
                point_labels=input_label,
                box=input_box,
                mask_input=mask_input,
                prompt_encoder=prompt_encoder,
                mask_decoder=mask_decoder,
                onnx=args.onnx
            )

        sorted_ind = np.argsort(scores)[::-1]
        masks = masks[sorted_ind]
        scores = scores[sorted_ind]
        logits = logits[sorted_ind]

        savepath = get_savepath(args.savepath, image_path, ext='.png')
        logger.info(f'saved at : {savepath}')
        image = show_mask(masks[0], image)
        image = show_points(input_point, input_label, image)
        image = show_box(input_box, image)
        cv2.imwrite(savepath, image)


def recognize_from_image_auto(image_encoder, prompt_encoder, mask_decoder):
    mask_generator = SAM2AutomaticMaskGenerator(
        legacy=args.legacy,
        version=args.version,
        model_type=args.model_type,
        points_per_side=args.points_per_side,
        points_per_batch=args.points_per_batch,
        pred_iou_thresh=args.pred_iou_thresh,
        stability_score_thresh=args.stability_score_thresh,
        box_nms_thresh=args.box_nms_thresh,
    )

    for image_path in args.input:
        logger.info(f'Processing: {image_path}')
        image = cv2.imread(image_path)
        orig_hw = [image.shape[0], image.shape[1]]
        image_size = 1024
        image_np = preprocess_frame(image, image_size=image_size)

        start = int(round(time.time() * 1000))
        features = mask_generator.image_predictor.set_image(image_np, image_encoder, args.onnx)
        binary_masks, iou_scores = mask_generator.generate(
            features=features,
            orig_hw=orig_hw,
            prompt_encoder=prompt_encoder,
            mask_decoder=mask_decoder,
            onnx=args.onnx,
        )
        end = int(round(time.time() * 1000))
        logger.info(f'\tprocessing time {end - start} ms')
        logger.info(f'\tdetected {len(binary_masks)} masks')

        # Sort masks by area (largest first, so smaller masks render on top)
        areas = np.sum(binary_masks, axis=(1, 2))
        sorted_idx = np.argsort(-areas)
        binary_masks = binary_masks[sorted_idx]

        # Visualize with random colors and contours (matching official SAM2 style)
        image = image.astype(np.float64)
        np.random.seed(0)
        for mask in binary_masks:
            color = (np.random.random(3) * 255).astype(np.float64)
            image = show_mask(mask, image, color=color)
            # Draw contour
            mask_uint8 = mask.astype(np.uint8)
            contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            cv2.drawContours(image, contours, -1, (0, 0, 255), thickness=1, lineType=cv2.LINE_AA)
        image = np.clip(image, 0, 255).astype(np.uint8)

        savepath = get_savepath(args.savepath, image_path, ext='.png')
        logger.info(f'saved at : {savepath}')
        cv2.imwrite(savepath, image)


def preprocess_frame(img, image_size):
    img_mean=(0.485, 0.456, 0.406)
    img_std=(0.229, 0.224, 0.225)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (image_size, image_size))
    img = img / 255.0
    img = img - img_mean
    img = img / img_std
    img = np.transpose(img, (2, 0, 1))
    return img


def recognize_from_video(image_encoder, prompt_encoder, mask_decoder, memory_attention, memory_encoder, mlp, obj_ptr_tpos_proj):
    image_size = 1024

    if args.video == "demo":
        frame_names = [
            p for p in os.listdir(args.video)
            if os.path.splitext(p)[-1] in [".jpg", ".jpeg", ".JPG", ".JPEG"]
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

    predictor = SAM2VideoPredictor(args.onnx, args.benchmark, args.legacy)

    inference_state = predictor.init_state(args.num_mask_mem, args.max_obj_ptrs_in_encoder, args.version, args.model_type)
    predictor.reset_state(inference_state)

    frame_shown = False

    if args.benchmark:
        start = int(round(time.time() * 1000))

    frame_idx = 0
    while (True):
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

        image = preprocess_frame(frame, image_size)

        predictor.append_image(
            inference_state,
            image,
            video_height,
            video_width,
            image_encoder)

        if frame_idx == 0:
            annotate_frame(input_point, input_label, input_box, predictor, inference_state, image_encoder, prompt_encoder, mask_decoder, memory_attention, memory_encoder, mlp, obj_ptr_tpos_proj)

        frame = process_frame(frame, frame_idx, predictor, inference_state, image_encoder, prompt_encoder, mask_decoder, memory_attention, memory_encoder, mlp, obj_ptr_tpos_proj)
        frame = frame.astype(np.uint8)

        if frame_idx == 0:
            frame = show_points(input_point.astype(np.int64), input_label.astype(np.int64), frame)
            frame = show_box(input_box, frame)

        cv2.imshow('frame', frame)
        if frame_names is not None:
            cv2.imwrite(f'video_{frame_idx}.png', frame)

        if writer is not None:
            writer.write(frame)

        frame_shown = True
        frame_idx = frame_idx + 1

    if args.benchmark:
        end = int(round(time.time() * 1000))
        estimation_time = (end - start)
        logger.info(f'\ttotal processing time {estimation_time} ms')

    if writer is not None:
        writer.release()

def annotate_frame(points, labels, box, predictor, inference_state, image_encoder, prompt_encoder, mask_decoder, memory_attention, memory_encoder, mlp, obj_ptr_tpos_proj):
    ann_frame_idx = 0  # the frame index we interact with
    ann_obj_id = 1  # give a unique id to each object we interact with (it can be any integers)

    _, out_obj_ids, out_mask_logits = predictor.add_new_points_or_box(
        inference_state=inference_state,
        frame_idx=ann_frame_idx,
        obj_id=ann_obj_id,
        points=points,
        labels=labels,
        box=box,
        image_encoder=image_encoder,
        prompt_encoder=prompt_encoder,
        mask_decoder=mask_decoder,
        memory_attention=memory_attention,
        memory_encoder=memory_encoder,
        mlp=mlp
    )

    predictor.propagate_in_video_preflight(inference_state,
                                                                            image_encoder = image_encoder,
                                                                            prompt_encoder = prompt_encoder,
                                                                            mask_decoder = mask_decoder,
                                                                            memory_attention = memory_attention,
                                                                            memory_encoder = memory_encoder,
                                                                            mlp = mlp,
                                                                            obj_ptr_tpos_proj = obj_ptr_tpos_proj)

def process_frame(image, frame_idx, predictor, inference_state, image_encoder, prompt_encoder, mask_decoder, memory_attention, memory_encoder, mlp, obj_ptr_tpos_proj):
    out_frame_idx, out_obj_ids, out_mask_logits = predictor.propagate_in_video(inference_state,
                                                                                image_encoder = image_encoder,
                                                                                prompt_encoder = prompt_encoder,
                                                                                mask_decoder = mask_decoder,
                                                                                memory_attention = memory_attention,
                                                                                memory_encoder = memory_encoder,
                                                                                mlp = mlp,
                                                                                obj_ptr_tpos_proj = obj_ptr_tpos_proj,
                                                                                frame_idx = frame_idx)

    image = show_mask((out_mask_logits[0] > 0.0), image, color = np.array([30, 144, 255]), obj_id = out_obj_ids[0])

    return image


def main():
    # fetch image encoder model
    model_type = args.model_type
    model_type_versioned = model_type
    if args.version == "2.1":
        model_type_versioned = model_type + "_2.1"
    WEIGHT_IMAGE_ENCODER_L_PATH = 'image_encoder_'+model_type_versioned+'.onnx'
    MODEL_IMAGE_ENCODER_L_PATH = 'image_encoder_'+model_type_versioned+'.onnx.prototxt'
    WEIGHT_MASK_DECODER_L_PATH = 'mask_decoder_'+model_type_versioned+'.onnx'
    MODEL_MASK_DECODER_L_PATH = 'mask_decoder_'+model_type_versioned+'.onnx.prototxt'
    WEIGHT_MEMORY_ENCODER_L_PATH = 'memory_encoder_'+model_type_versioned+'.onnx'
    MODEL_MEMORY_ENCODER_L_PATH = 'memory_encoder_'+model_type_versioned+'.onnx.prototxt'
    WEIGHT_MLP_L_PATH = 'mlp_'+model_type_versioned+'.onnx'
    MODEL_MLP_L_PATH = 'mlp_'+model_type_versioned+'.onnx.prototxt'
    if args.version == "2.1":
        WEIGHT_TPOS_L_PATH = 'obj_ptr_tpos_proj_'+model_type_versioned+'.onnx'
        MODEL_TPOS_L_PATH = 'obj_ptr_tpos_proj_'+model_type_versioned+'.onnx.prototxt'
    else:
        WEIGHT_TPOS_L_PATH = None
        MODEL_TPOS_L_PATH = None

    if args.legacy:
        WEIGHT_PROMPT_ENCODER_L_PATH = 'prompt_encoder_'+model_type_versioned+'.onnx'
        MODEL_PROMPT_ENCODER_L_PATH = 'prompt_encoder_'+model_type_versioned+'.onnx.prototxt'
        # 4dim matmul with batch 1
        WEIGHT_MEMORY_ATTENTION_L_PATH = 'memory_attention_'+model_type_versioned+'.opt.onnx'
        MODEL_MEMORY_ATTENTION_L_PATH = 'memory_attention_'+model_type_versioned+'.opt.onnx.prototxt'
    else:
        # New models: 4D mask prompt encoder and 6D matmul memory attention with dynamic batch
        WEIGHT_PROMPT_ENCODER_L_PATH = 'prompt_encoder_with_mask_'+model_type_versioned+'.onnx'
        MODEL_PROMPT_ENCODER_L_PATH = 'prompt_encoder_with_mask_'+model_type_versioned+'.onnx.prototxt'
        WEIGHT_MEMORY_ATTENTION_L_PATH = 'memory_attention_6d_'+model_type_versioned+'.onnx'
        MODEL_MEMORY_ATTENTION_L_PATH = 'memory_attention_6d_'+model_type_versioned+'.onnx.prototxt'

    # model files check and download
    check_and_download_models(WEIGHT_IMAGE_ENCODER_L_PATH, MODEL_IMAGE_ENCODER_L_PATH, REMOTE_PATH)
    check_and_download_models(WEIGHT_PROMPT_ENCODER_L_PATH, MODEL_PROMPT_ENCODER_L_PATH, REMOTE_PATH)
    check_and_download_models(WEIGHT_MASK_DECODER_L_PATH, MODEL_MASK_DECODER_L_PATH, REMOTE_PATH)
    check_and_download_models(WEIGHT_MEMORY_ATTENTION_L_PATH, MODEL_MEMORY_ATTENTION_L_PATH, REMOTE_PATH)
    check_and_download_models(WEIGHT_MEMORY_ENCODER_L_PATH, MODEL_MEMORY_ENCODER_L_PATH, REMOTE_PATH)
    check_and_download_models(WEIGHT_MLP_L_PATH, MODEL_MLP_L_PATH, REMOTE_PATH)
    if args.version == "2.1":
        check_and_download_models(WEIGHT_TPOS_L_PATH, MODEL_TPOS_L_PATH, REMOTE_PATH)

    if args.onnx:
        import onnxruntime
        image_encoder = onnxruntime.InferenceSession(WEIGHT_IMAGE_ENCODER_L_PATH)
        prompt_encoder = onnxruntime.InferenceSession(WEIGHT_PROMPT_ENCODER_L_PATH)
        mask_decoder = onnxruntime.InferenceSession(WEIGHT_MASK_DECODER_L_PATH)
        memory_attention = onnxruntime.InferenceSession(WEIGHT_MEMORY_ATTENTION_L_PATH)
        memory_encoder = onnxruntime.InferenceSession(WEIGHT_MEMORY_ENCODER_L_PATH)
        mlp = onnxruntime.InferenceSession(WEIGHT_MLP_L_PATH)
        if args.version == "2.1":
            obj_ptr_tpos_proj = onnxruntime.InferenceSession(WEIGHT_TPOS_L_PATH)
        else:
            obj_ptr_tpos_proj = None
    else:
        import ailia
        memory_mode = ailia.get_memory_mode(reduce_constant=True, ignore_input_with_initializer=True, reduce_interstage=False, reuse_interstage=True)
        image_encoder = ailia.Net(weight=WEIGHT_IMAGE_ENCODER_L_PATH, stream=MODEL_IMAGE_ENCODER_L_PATH, memory_mode=memory_mode, env_id=args.env_id)
        prompt_encoder = ailia.Net(weight=WEIGHT_PROMPT_ENCODER_L_PATH, stream=MODEL_PROMPT_ENCODER_L_PATH, memory_mode=memory_mode, env_id=args.env_id)
        mask_decoder = ailia.Net(weight=WEIGHT_MASK_DECODER_L_PATH, stream=MODEL_MASK_DECODER_L_PATH, memory_mode=memory_mode, env_id=args.env_id)
        memory_attention = ailia.Net(weight=WEIGHT_MEMORY_ATTENTION_L_PATH, stream=MODEL_MEMORY_ATTENTION_L_PATH, memory_mode=memory_mode, env_id=args.env_id)
        memory_encoder = ailia.Net(weight=WEIGHT_MEMORY_ENCODER_L_PATH, stream=MODEL_MEMORY_ENCODER_L_PATH, memory_mode=memory_mode, env_id=args.env_id)
        mlp = ailia.Net(weight=WEIGHT_MLP_L_PATH, stream=MODEL_MLP_L_PATH, memory_mode=memory_mode, env_id=args.env_id)
        if args.version == "2.1":
            obj_ptr_tpos_proj = ailia.Net(weight=WEIGHT_TPOS_L_PATH, stream=MODEL_TPOS_L_PATH, memory_mode=memory_mode, env_id=args.env_id)
        else:
            obj_ptr_tpos_proj = None

    if args.auto and args.legacy:
        raise RuntimeError("--auto requires dynamic batch support. Cannot be used with --legacy.")

    if args.video is not None:
        recognize_from_video(image_encoder, prompt_encoder, mask_decoder, memory_attention, memory_encoder, mlp, obj_ptr_tpos_proj)
    elif args.auto:
        recognize_from_image_auto(image_encoder, prompt_encoder, mask_decoder)
    else:
        recognize_from_image(image_encoder, prompt_encoder, mask_decoder)

    logger.info('Script finished successfully.')

if __name__ == '__main__':
    main()
