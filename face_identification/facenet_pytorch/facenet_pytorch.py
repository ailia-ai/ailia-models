import os
import sys
import time
from logging import getLogger
import json

import ailia
import cv2
import numpy as np
from glob import glob

# import local modules
sys.path.append('../../util')
from arg_utils import get_base_parser, update_parser # noqa: E402
from image_utils import imread
from model_utils import check_and_download_models # noqa: E402
from mtcnn_utils import MTCNN

logger = getLogger(__name__)

# ======================
# Arguemnt Parser Config
# ======================
parser = get_base_parser(
    'facenet-pytorch', None, None 
)
parser.add_argument(
    '-d', '--dir', default='data',
    help='Directory path of input image files.'
)
parser.add_argument(
    '-w', '--weight', default='vggface2',
    choices=['vggface2', 'casia-webface'],
    help='You can choose the model weights.'
)
parser.add_argument(
    '--write_json',
    action='store_true',
    help='Flag to output results to json file.'
)
args = update_parser(parser)


# ======================
# Parameters
# ======================
PNET_WEIGHT_PATH = 'pnet.onnx'
PNET_MODEL_PATH = PNET_WEIGHT_PATH + '.prototxt'

RNET_WEIGHT_PATH = 'rnet.onnx'
RNET_MODEL_PATH = RNET_WEIGHT_PATH + '.prototxt'

ONET_WEIGHT_PATH = 'onet.onnx'
ONET_MODEL_PATH = ONET_WEIGHT_PATH + '.prototxt'

FACENET_WEIGHT_PATH = args.weight + '.onnx'
FACENET_MODEL_PATH = FACENET_WEIGHT_PATH + '.prototxt'
REMOTE_PATH = 'https://storage.googleapis.com/ailia-models/facenet-pytorch/'

# ======================
# Utils
# ======================
def attach_label(sim_matrix, labels):
    matrix = [[''] + labels]
    matrix.extend(sim_matrix)
    for i in range(1, len(matrix)):
        matrix[i] = [labels[i-1]] + matrix[i]
    return np.array(matrix)

# ======================
# Main Functions
# ======================
def preprocess(mtcnn, img_dir):
    files_path = os.path.join(img_dir, '*.*')
    filenames = []
    imgs = []

    for fp in sorted(glob(files_path)):
        img = imread(fp)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        crop_img = mtcnn(img)
        # (h, w, c) -> (1, c, h, w)
        crop_img = crop_img.transpose(2, 0, 1)[None]

        imgs.append(crop_img)
        filenames.append(fp.split(os.sep)[-1].split('.')[0])

    return imgs, filenames


def predict(model, inputs):
    outputs = [model.run(i) for i in inputs]
    return np.concatenate(outputs)


def postprocess(embeddings):
    return [[np.linalg.norm(e1 - e2) for e2 in embeddings] for e1 in embeddings]


def main():
    # model files check and download
    check_and_download_models(PNET_WEIGHT_PATH, PNET_MODEL_PATH, REMOTE_PATH)
    check_and_download_models(RNET_WEIGHT_PATH, RNET_MODEL_PATH, REMOTE_PATH)
    check_and_download_models(ONET_WEIGHT_PATH, ONET_MODEL_PATH, REMOTE_PATH)
    check_and_download_models(FACENET_WEIGHT_PATH, FACENET_MODEL_PATH, REMOTE_PATH)

    # initialize
    pnet = ailia.Net(PNET_MODEL_PATH, PNET_WEIGHT_PATH, env_id=args.env_id, memory_mode=args.memory_mode)
    rnet = ailia.Net(RNET_MODEL_PATH, RNET_WEIGHT_PATH, env_id=args.env_id, memory_mode=args.memory_mode)
    onet = ailia.Net(ONET_MODEL_PATH, ONET_WEIGHT_PATH, env_id=args.env_id, memory_mode=args.memory_mode)
    mtcnn = MTCNN(pnet, rnet, onet,
                  image_size=160, margin=0, min_face_size=20,
                  thresholds=[0.6, 0.7, 0.7], factor=0.709, post_process=True,
    )
    facenet = ailia.Net(FACENET_MODEL_PATH, FACENET_WEIGHT_PATH, env_id=args.env_id, memory_mode=args.memory_mode)


    # prepare data
    imgs, filenames = preprocess(mtcnn, args.dir)

    # inference
    if args.benchmark:
        logger.info('BENCHMARK mode')
        for i in range(5):
            start = int(round(time.time() * 1000))
            embeddings = predict(facenet, imgs)
            end = int(round(time.time() * 1000))
            logger.info(f'\tailia processing time {end-start} ms')
    else:
        embeddings = predict(facenet, imgs)

    # calucate distances
    sim_matrix = postprocess(embeddings)

    # results
    np.set_printoptions(linewidth=1000)
    logger.info('Similarity: \n' + np.array2string(attach_label(sim_matrix, filenames)))

    if args.write_json:
        result = attach_label(sim_matrix, filenames)
        out_data = []
        for r in result:
            out_data.append(tuple(r))
        with open('output.json', 'w', encoding='utf-8') as f:
            json.dump(out_data, f, indent=2)

if __name__ == '__main__':
    main()

