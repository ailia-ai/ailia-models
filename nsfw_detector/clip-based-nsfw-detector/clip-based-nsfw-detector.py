import sys
import time
from logging import getLogger

import numpy as np
import cv2
from PIL import Image

import ailia

# import original modules
sys.path.append('../../util')
from arg_utils import get_base_parser, update_parser
from model_utils import check_and_download_models
from detector_utils import load_image

logger = getLogger(__name__)

# ======================
# Parameters
# ======================

WEIGHT_VITL14_PATH = 'clip_bin_nsfw.onnx'
MODEL_VITL14_PATH = 'clip_bin_nsfw.onnx.prototxt'
WEIGHT_VITB32_PATH = 'clip_nsfw_b32.onnx'
MODEL_VITB32_PATH = 'clip_nsfw_b32.onnx.prototxt'
REMOTE_PATH = 'https://storage.googleapis.com/ailia-models/clip-based-nsfw-detector/'

WEIGHT_CLIP_VITL14_IMAGE_PATH = 'ViT-L14-encode_image.onnx'
MODEL_CLIP_VITL14_IMAGE_PATH = 'ViT-L14-encode_image.onnx.prototxt'
WEIGHT_CLIP_VITB32_IMAGE_PATH = 'ViT-B32-encode_image.onnx'
MODEL_CLIP_VITB32_IMAGE_PATH = 'ViT-B32-encode_image.onnx.prototxt'
REMOTE_CLIP_PATH = 'https://storage.googleapis.com/ailia-models/clip/'

IMAGE_PATH = '_vyr_6097Sexy-Push-Up-Bikini-Brasilianisch-Bunt-2.jpg'

IMAGE_SIZE = 224

# ======================
# Arguemnt Parser Config
# ======================

parser = get_base_parser(
    'CLIP-based-NSFW-Detector', IMAGE_PATH, None
)
parser.add_argument(
    '-m', '--model_type', default='ViTB32', choices=('ViTB32', 'ViTL14'),
    help='model type'
)
parser.add_argument(
    '--onnx',
    action='store_true',
    help='execute onnxruntime version.'
)
args = update_parser(parser)


# ======================
# Secondaty Functions
# ======================

def normalized(a, axis=-1, order=2):
    l2 = np.atleast_1d(np.linalg.norm(a, order, axis))
    l2[l2 == 0] = 1

    return a / np.expand_dims(l2, axis)


# ======================
# Main functions
# ======================

def preprocess(img):
    h, w = (IMAGE_SIZE, IMAGE_SIZE)
    im_h, im_w, _ = img.shape

    # resize
    scale = h / min(im_h, im_w)
    ow, oh = int(im_w * scale), int(im_h * scale)
    if ow != im_w or oh != im_h:
        img = np.array(Image.fromarray(img).resize((ow, oh), Image.BICUBIC))

    # center_crop
    if ow > w:
        x = (ow - w) // 2
        img = img[:, x:x + w, :]
    if oh > h:
        y = (oh - h) // 2
        img = img[y:y + h, :, :]

    img = img[:, :, ::-1]  # BGR -> RBG
    img = img / 255

    mean = np.array((0.48145466, 0.4578275, 0.40821073))
    std = np.array((0.26862954, 0.26130258, 0.27577711))
    img = (img - mean) / std

    img = img.transpose(2, 0, 1)  # HWC -> CHW
    img = np.expand_dims(img, axis=0)
    img = img.astype(np.float32)

    return img


def predict(net_nsfw, net_image, img):
    img = preprocess(img)

    # encode image
    if not args.onnx:
        output = net_image.predict([img])
    else:
        output = net_image.run(None, {'image': img})
    image_feature = output[0]

    emb = np.asarray(normalized(image_feature))

    # feedforward
    emb = emb.astype(np.float64)
    if not args.onnx:
        output = net_nsfw.predict([emb])
    else:
        output = net_nsfw.run(None, {'input_1': emb})
    nsfw_value = output[0]

    return nsfw_value


def recognize_from_image(net_nsfw, net_image):
    # input image loop
    for image_path in args.input:
        logger.info(image_path)

        # prepare input data
        img = load_image(image_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)

        # inference
        logger.info('Start inference...')
        if args.benchmark:
            logger.info('BENCHMARK mode')
            total_time_estimation = 0
            for i in range(args.benchmark_count):
                start = int(round(time.time() * 1000))
                nsfw_value = predict(net_nsfw, net_image, img)
                end = int(round(time.time() * 1000))
                estimation_time = (end - start)

                # Logging
                logger.info(f'\tailia processing estimation time {estimation_time} ms')
                if i != 0:
                    total_time_estimation = total_time_estimation + estimation_time

            logger.info(f'\taverage time estimation {total_time_estimation / (args.benchmark_count - 1)} ms')
        else:
            nsfw_value = predict(net_nsfw, net_image, img)

        logger.info(" NSFW: %.3f" % (nsfw_value[0][0] * 100))

    logger.info('Script finished successfully.')


def main():
    dic_model = {
        'ViTB32': (
            (WEIGHT_CLIP_VITB32_IMAGE_PATH, MODEL_CLIP_VITB32_IMAGE_PATH),
            (WEIGHT_VITB32_PATH, MODEL_VITB32_PATH)),
        'ViTL14': (
            (WEIGHT_CLIP_VITL14_IMAGE_PATH, MODEL_CLIP_VITL14_IMAGE_PATH),
            (WEIGHT_VITL14_PATH, MODEL_VITL14_PATH)),
    }
    (clip_weigth, clip_model), (nsfw_weigth, nsfw_model) = dic_model[args.model_type]

    # model files check and download
    check_and_download_models(nsfw_weigth, nsfw_model, REMOTE_PATH)
    check_and_download_models(clip_weigth, clip_model, REMOTE_CLIP_PATH)

    env_id = args.env_id

    # initialize
    if not args.onnx:
        net_nsfw = ailia.Net(nsfw_model, nsfw_weigth, env_id=env_id)
        net_image = ailia.Net(clip_model, clip_weigth, env_id=env_id)
    else:
        import onnxruntime
        net_nsfw = onnxruntime.InferenceSession(nsfw_weigth)
        net_image = onnxruntime.InferenceSession(clip_weigth)

    recognize_from_image(net_nsfw, net_image)


if __name__ == '__main__':
    main()
