import os
import sys
import time

import cv2
import numpy as np

try:
    import ailia
    HAS_AILIA = True
except ImportError:
    HAS_AILIA = False

# import original modules
sys.path.append('../../util')
from arg_utils import get_base_parser, get_savepath, update_parser  # noqa
from model_utils import check_and_download_models  # noqa
from image_utils import imread  # noqa
import webcamera_utils  # noqa
# logger
from logging import getLogger  # noqa

logger = getLogger(__name__)


# ======================
# Parameters
# ======================

# Model variants: name -> (input_h, input_w, num_classes, dataset)
MODEL_VARIANTS = {
    'cityscapes-1024-1024': (1024, 1024, 19, 'cityscapes'),
    'cityscapes-768-768':   (768,  768,  19, 'cityscapes'),
    'cityscapes-640-1280':  (640,  1280, 19, 'cityscapes'),
    'cityscapes-512-1024':  (512,  1024, 19, 'cityscapes'),
    'ade-512-512':          (512,  512,  150, 'ade'),
}
DEFAULT_VARIANT = 'cityscapes-1024-1024'

REMOTE_PATH = 'https://storage.googleapis.com/ailia-models/segformer/'

IMAGE_PATH = 'input.jpg'
SAVE_IMAGE_PATH = 'output.png'

# ImageNet normalization (used by HuggingFace SegformerImageProcessor)
IMG_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMG_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

# Cityscapes 19-class palette (standard order matching trainId)
CITYSCAPES_PALETTE = np.array([
    [128, 64, 128],   # road
    [244, 35, 232],   # sidewalk
    [70, 70, 70],     # building
    [102, 102, 156],  # wall
    [190, 153, 153],  # fence
    [153, 153, 153],  # pole
    [250, 170, 30],   # traffic light
    [220, 220, 0],    # traffic sign
    [107, 142, 35],   # vegetation
    [152, 251, 152],  # terrain
    [70, 130, 180],   # sky
    [220, 20, 60],    # person
    [255, 0, 0],      # rider
    [0, 0, 142],      # car
    [0, 0, 70],       # truck
    [0, 60, 100],     # bus
    [0, 80, 100],     # train
    [0, 0, 230],      # motorcycle
    [119, 11, 32],    # bicycle
], dtype=np.uint8)

CITYSCAPES_LABELS = [
    'road', 'sidewalk', 'building', 'wall', 'fence',
    'pole', 'traffic light', 'traffic sign', 'vegetation', 'terrain',
    'sky', 'person', 'rider', 'car', 'truck',
    'bus', 'train', 'motorcycle', 'bicycle',
]


# ======================
# Argument Parser Config
# ======================

parser = get_base_parser(
    'SegFormer: Simple and Efficient Design for Semantic Segmentation '
    'with Transformers (B0)',
    IMAGE_PATH, SAVE_IMAGE_PATH, fp16_support=False
)
parser.add_argument(
    '-a', '--arch', metavar='ARCH',
    default=DEFAULT_VARIANT,
    choices=list(MODEL_VARIANTS.keys()),
    help='Model variant: ' + ' | '.join(MODEL_VARIANTS.keys())
)
parser.add_argument(
    '--alpha', type=float, default=0.5,
    help='Blending alpha for segmentation overlay (0..1)'
)
parser.add_argument(
    '--onnx', action='store_true',
    help='Use ONNX Runtime instead of ailia SDK.'
)
args = update_parser(parser)


# ======================
# Variant-derived parameters
# ======================

IMAGE_HEIGHT, IMAGE_WIDTH, NUM_CLASSES, DATASET = MODEL_VARIANTS[args.arch]
WEIGHT_PATH = f'segformer_b0_{args.arch}.onnx'
MODEL_PATH = WEIGHT_PATH + '.prototxt'


# ======================
# Helpers
# ======================

def get_palette(num_classes, dataset):
    """Return an (N, 3) uint8 RGB palette for the dataset."""
    if dataset == 'cityscapes':
        return CITYSCAPES_PALETTE
    # ADE20K (or any unknown): generate a deterministic palette.
    palette = np.zeros((num_classes, 3), dtype=np.uint8)
    for i in range(num_classes):
        lab = i + 1  # avoid all-black for class 0
        r = g = b = 0
        for j in range(8):
            r |= ((lab >> 0) & 1) << (7 - j)
            g |= ((lab >> 1) & 1) << (7 - j)
            b |= ((lab >> 2) & 1) << (7 - j)
            lab >>= 3
        palette[i] = [r, g, b]
    return palette


def preprocess(img):
    """Preprocess BGR image -> NCHW float32 (1, 3, H, W)."""
    img_resized = cv2.resize(
        img, (IMAGE_WIDTH, IMAGE_HEIGHT), interpolation=cv2.INTER_LINEAR)
    img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB).astype(np.float32)
    img_rgb = img_rgb / 255.0
    img_rgb = (img_rgb - IMG_MEAN) / IMG_STD
    img_chw = img_rgb.transpose(2, 0, 1)
    return np.expand_dims(img_chw, axis=0).astype(np.float32)


def postprocess(logits, orig_h, orig_w):
    """Upsample logits to original image size and return argmax label map."""
    # logits: (1, C, h, w) with h = H/4, w = W/4
    logits = logits[0]  # (C, h, w)
    C, h, w = logits.shape

    # Upsample each channel to the original image resolution with bilinear.
    upsampled = np.empty((C, orig_h, orig_w), dtype=np.float32)
    for c in range(C):
        upsampled[c] = cv2.resize(
            logits[c], (orig_w, orig_h), interpolation=cv2.INTER_LINEAR)
    pred = np.argmax(upsampled, axis=0).astype(np.int32)
    return pred


def colorize(pred, palette):
    """Map an HxW label map to an HxWx3 BGR image using the palette (RGB)."""
    h, w = pred.shape
    # Clamp out-of-range labels to 0
    pred_safe = np.clip(pred, 0, palette.shape[0] - 1)
    rgb = palette[pred_safe]  # (H, W, 3) RGB
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    return bgr


def overlay(img_bgr, color_bgr, alpha):
    """Alpha blend the colorized prediction over the original image."""
    blended = (img_bgr.astype(np.float32) * (1.0 - alpha)
               + color_bgr.astype(np.float32) * alpha)
    return np.clip(blended, 0, 255).astype(np.uint8)


# ======================
# Inference wrappers
# ======================

def predict_onnx(session, img):
    blob = preprocess(img)
    input_name = session.get_inputs()[0].name
    outputs = session.run(None, {input_name: blob})
    return outputs[0]


def predict_ailia(net, img):
    blob = preprocess(img)
    outputs = net.predict([blob])
    return outputs[0]


# ======================
# Main functions
# ======================

def recognize_from_image(predictor, predict_fn):
    palette = get_palette(NUM_CLASSES, DATASET)

    for image_path in args.input:
        logger.info(image_path)
        img = imread(image_path)
        if img is None:
            logger.error(f'Could not read image: {image_path}')
            continue

        h, w = img.shape[:2]

        logger.info('Start inference...')
        if args.benchmark:
            logger.info('BENCHMARK mode')
            total_time = 0
            for i in range(args.benchmark_count):
                start = int(round(time.time() * 1000))
                logits = predict_fn(predictor, img)
                end = int(round(time.time() * 1000))
                logger.info(f'\tailia processing time {end - start} ms')
                if i != 0:
                    total_time += (end - start)
            logger.info(
                f'\taverage time '
                f'{total_time / (args.benchmark_count - 1)} ms')
        else:
            logits = predict_fn(predictor, img)

        pred = postprocess(logits, h, w)
        color_bgr = colorize(pred, palette)
        out_img = overlay(img, color_bgr, args.alpha)

        savepath = get_savepath(args.savepath, image_path, ext='.png')
        cv2.imwrite(savepath, out_img)
        logger.info(f'saved at : {savepath}')

        # Log detected class summary
        unique, counts = np.unique(pred, return_counts=True)
        total = pred.size
        class_summary = []
        for cls_id, cnt in zip(unique, counts):
            ratio = 100.0 * cnt / total
            if DATASET == 'cityscapes' and cls_id < len(CITYSCAPES_LABELS):
                name = CITYSCAPES_LABELS[int(cls_id)]
            else:
                name = f'class_{int(cls_id)}'
            class_summary.append((ratio, name))
        class_summary.sort(reverse=True)
        for ratio, name in class_summary[:10]:
            logger.info(f'  {name}: {ratio:.1f}%')

    logger.info('Script finished successfully.')


def recognize_from_video(predictor, predict_fn):
    palette = get_palette(NUM_CLASSES, DATASET)

    capture = webcamera_utils.get_capture(args.video)
    assert capture.isOpened(), 'Cannot capture source'

    if args.savepath != SAVE_IMAGE_PATH:
        f_h = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        f_w = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        writer = webcamera_utils.get_writer(args.savepath, f_h, f_w)
    else:
        writer = None

    frame_shown = False
    while True:
        ret, frame = capture.read()
        if (cv2.waitKey(1) & 0xFF == ord('q')) or not ret:
            break
        if frame_shown and cv2.getWindowProperty(
                'frame', cv2.WND_PROP_VISIBLE) == 0:
            break

        h, w = frame.shape[:2]
        logits = predict_fn(predictor, frame)
        pred = postprocess(logits, h, w)
        color_bgr = colorize(pred, palette)
        vis = overlay(frame, color_bgr, args.alpha)

        cv2.imshow('frame', vis)
        frame_shown = True
        if writer is not None:
            writer.write(vis)

    capture.release()
    cv2.destroyAllWindows()
    if writer is not None:
        writer.release()
    logger.info('Script finished successfully.')


def main():
    use_onnx = args.onnx or not HAS_AILIA

    check_and_download_models(WEIGHT_PATH, MODEL_PATH, REMOTE_PATH)

    if use_onnx:
        import onnxruntime as ort
        logger.info('Using ONNX Runtime')
        session = ort.InferenceSession(
            WEIGHT_PATH, providers=['CPUExecutionProvider'])
        predict_fn = predict_onnx
        predictor = session
    else:
        env_id = args.env_id
        memory_mode = ailia.get_memory_mode(
            reduce_constant=True, ignore_input_with_initializer=True,
            reduce_interstage=True, reuse_interstage=False)
        net = ailia.Net(
            MODEL_PATH, WEIGHT_PATH,
            env_id=env_id, memory_mode=memory_mode)
        predict_fn = predict_ailia
        predictor = net

        if args.profile:
            net.set_profile_mode(True)

    if args.video is not None:
        recognize_from_video(predictor, predict_fn)
    else:
        recognize_from_image(predictor, predict_fn)

    if args.profile and not args.onnx:
        print(net.get_summary())


if __name__ == '__main__':
    main()
