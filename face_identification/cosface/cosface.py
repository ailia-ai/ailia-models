import sys

sys.path.append('../../util')
from arg_utils import get_base_parser, update_parser
from model_utils import check_and_download_models

import numpy as np
import ailia
from PIL import Image


# logger
from logging import getLogger   # noqa: E402
logger = getLogger(__name__)

# ======================
# PARAMETERS
# ======================

REMOTE_PATH = "https://storage.googleapis.com/ailia-models/cosface/"
WEIGHT_PATH = 'cosface_sphere20_opset10.onnx'
MODEL_PATH = 'cosface_sphere20_opset10.onnx.prototxt'

IMG_PATH_1 = 'image_id.jpg'
IMG_PATH_2 = 'image_target.jpg'

# ======================
# Arguemnt Parser Config
# ======================
parser = get_base_parser(
    'Determine if the person is the same from two facial images by CosFace',
    None,
    None,
)

# Overwrite default config
# NOTE: cosface has different usage for `--input` with other models
parser.add_argument(
    '-i', '--inputs', metavar='IMAGE', nargs=2, default='',
    help='Two image paths for calculating the face match.'
)

args = update_parser(parser)


def preprocessing(img_path):
    with open(img_path, 'rb') as f:
        # Image[H, W, C]
        original_img = Image.open(f).convert('RGB').resize((96, 112), Image.Resampling.LANCZOS)

    # Range [0, 255] -> [0.0,1.0]
    img_01 = np.array(original_img) / 255
    img_01 = img_01.transpose(-1, 0, 1)  # to Image[C, H, W]

    # range [0.0, 1.0] -> [-1.0,1.0]
    MEAN = np.array([0.5, 0.5, 0.5])  # RGB
    STD = np.array([0.5, 0.5, 0.5])   # RGB
    img = (img_01 - MEAN[:, None, None]) / STD[:, None, None]  # Normalize
    img_flipped = np.flip(img, axis=2)  # Horizontally flip

    img, img_flipped = np.expand_dims(img, axis=0), np.expand_dims(img_flipped, axis=0)
    return {'original': img, 'flipped': img_flipped}


def extractDeepFeature(images, model):
    # features of original image and the flipped image are concatenated together
    # to compose the final face representation
    ft = np.concatenate((model.predict(images['original']), model.predict(images['flipped'])), 1)[0]
    return ft


def cosFace(id_img_path, query_img_path):
    # Pre-processing
    img1 = preprocessing(id_img_path)
    img2 = preprocessing(query_img_path)

    # Inference ( extract feature )
    model = ailia.Net(MODEL_PATH, WEIGHT_PATH, env_id=args.env_id, memory_mode=args.memory_mode)
    f1 = extractDeepFeature(img1, model)
    f2 = extractDeepFeature(img2, model)

    # Post-processing ( cosine distance )
    cosine_distance = f1.dot(f2) / (np.linalg.norm(f1) * np.linalg.norm(f2) + 1e-5)
    return cosine_distance


def main():
    # Check and download CosFace model
    check_and_download_models(WEIGHT_PATH, MODEL_PATH, REMOTE_PATH)

    # Specify default images when there is no inputs
    if len(args.inputs) == 0:
        args.inputs = [IMG_PATH_1, IMG_PATH_2]

    # Comparing two images
    distance = cosFace(args.inputs[0],
                       args.inputs[1])
    logger.info(
        f'Similarity of ({args.inputs[0]}, {args.inputs[1]}) : {distance:.3f}'
    )
    logger.info('Script finished successfully.')


if __name__ == '__main__':
    main()
