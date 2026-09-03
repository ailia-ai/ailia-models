import os
import sys
import time
import argparse
import numpy as np
import cv2
import ailia

sys.path.append('../util')
from arg_utils import get_base_parser, update_parser, get_savepath  # noqa: E402
from model_utils import check_and_download_models  # noqa: E402
from detector_utils import load_image  # noqa: E402C
from webcamera_utils import get_capture  # noqa: E402

from logging import getLogger   # noqa: E402
logger = getLogger(__name__)


# ======================
# Parameters
# ======================

WEIGHT_PATH = 'sac+logos+ava1-l14-linearMSE.onnx'
MODEL_PATH = 'sac+logos+ava1-l14-linearMSE.onnx.prototxt'
REMOTE_PATH = 'https://storage.googleapis.com/ailia-models/aesthetic-predictor-v2/'
IMAGE_PATH = 'test.jpg'


# ======================
# Argument Parser Config
# ======================

parser = get_base_parser(
    'Aesthetic Score Predictor',
    IMAGE_PATH,
    None,
)
args = update_parser(parser)


# ======================
# Helper functions
# ======================

def preprocess(image):
    pass
    # device = "cuda" if torch.cuda.is_available() else "cpu"
    # model2, preprocess = clip.load("ViT-L/14", device=device)  #RN50x64
    # image = preprocess(pil_image).unsqueeze(0).to(device)
    # with torch.no_grad():
    #    image_features = model2.encode_image(image)
    # im_emb_arr = normalized(image_features.cpu().detach().numpy() )


# ======================
# Main functions
# ======================

def predict(image, net):
    features = np.zeros((1, 768), dtype=np.float32)
    net.set_input_shape(features.shape)
    output = net.predict({'x': features})
    return output[0][0][0]


def recognize_from_image(filename, net):
    for image_path in args.input:
        logger.info(image_path)

        image = load_image(image_path)
        logger.info(f'input image shape: {image.shape}')

        image = cv2.cvtColor(image, cv2.COLOR_BGRA2RGB)

        logger.info('Start inference...')
        if args.benchmark:
            logger.info('BENCHMARK mode')
            for i in range(5):
                start = int(round(time.time() * 1000))
                score = predict(image, net)
                end = int(round(time.time() * 1000))
                logger.info(f'\tailia processing time {end - start} ms')
        else:
            score = predict(image, net)

    logger.info('### Score ###')
    logger.info(score)

    logger.info('Script finished successfully.')


def main():
    check_and_download_models(WEIGHT_PATH, MODEL_PATH, REMOTE_PATH)

    memory_mode = ailia.get_memory_mode(reduce_constant=True, reduce_interstage=True)
    net = ailia.Net(MODEL_PATH, WEIGHT_PATH, env_id=args.env_id, memory_mode=memory_mode)

    recognize_from_image(args.input, net)


if __name__ == '__main__':
    main()
