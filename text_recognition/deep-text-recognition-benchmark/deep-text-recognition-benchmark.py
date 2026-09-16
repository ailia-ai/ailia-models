import argparse
import codecs
import sys
import time

import ailia
import cv2
import numpy

# import original modules
sys.path.append('../../util')
import string
# logger
from logging import getLogger  # noqa: E402

from image_utils import imread  # noqa: E402
from model_utils import check_and_download_models  # noqa: E402
from arg_utils import get_base_parser, get_savepath, update_parser  # noqa: E402
from webcamera_utils import adjust_frame_size, get_capture  # noqa: E402

logger = getLogger(__name__)

# ======================
# PARAMETERS
# ======================
MODEL_PATH = 'None-ResNet-None-CTC.onnx.prototxt'
WEIGHT_PATH = 'None-ResNet-None-CTC.onnx'
REMOTE_PATH = 'https://storage.googleapis.com/ailia-models/deep-text-recognition-benchmark/'

IMAGE_FOLDER_PATH = 'demo_image/'
IMAGE_WIDTH = 100
IMAGE_HEIGHT = 32


# ======================
# Arguemnt Parser Config
# ======================

parser = get_base_parser(
    'deep text recognition benchmark.', IMAGE_FOLDER_PATH, None
)
args = update_parser(parser)


# ======================
# Utils
# ======================

def ctc_decode(text_index, length, character):
    dict_character = list(character)

    dict = {}
    for i, char in enumerate(dict_character):
        dict[char] = i + 1

    character = ['[CTCblank]'] + dict_character
    CTC_BLANK = 0
    
    texts = []
    for index, l in enumerate(length):
        t = text_index[index, :]

        char_list = []
        for i in range(l):
            if t[i] != CTC_BLANK and (not (i > 0 and t[i - 1] == t[i])):
                char_list.append(character[t[i]])
        text = ''.join(char_list)

        texts.append(text)

    return texts

def preprocess_image(sample):
    sample = cv2.resize(sample,(IMAGE_WIDTH,IMAGE_HEIGHT),interpolation=cv2.INTER_CUBIC)
    sample = cv2.cvtColor(sample, cv2.COLOR_BGR2GRAY)
    sample = sample/127.5 - 1.0
    return sample

def softmax(x):
    u = numpy.sum(numpy.exp(x))
    return numpy.exp(x)/u

dashed_line = '-' * 80

def recognize_from_image():
    head = f'{"image_path":25s}\t{"predicted_labels":25s}\tconfidence score'
    
    logger.info(f'{dashed_line}\n{head}\n{dashed_line}')

    session = ailia.Net(MODEL_PATH,WEIGHT_PATH,env_id=args.env_id,memory_mode=args.memory_mode)

    print(args.input)

    for path in args.input:
        recognize_one_image(path,session)

def recognize_one_image(image_path,session):
    """ model configuration """
    character = '0123456789abcdefghijklmnopqrstuvwxyz'
    imgH = IMAGE_HEIGHT
    imgW = IMAGE_WIDTH
    batch_size = 1

    # load image
    input_img = imread(image_path)
    input_img = preprocess_image(input_img)
    input_img = numpy.expand_dims(input_img, axis=0)
    input_img = numpy.expand_dims(input_img, axis=0)

    # predict
    preds = session.predict(input_img)

    # Select max probabilty (greedy decoding) then decode index to character
    preds_size = [int(preds.shape[1])] * batch_size
    preds_index = numpy.argmax(preds,axis=2)

    preds_str = ctc_decode(preds_index, preds_size, character)
    preds_prob = numpy.zeros(preds.shape)
    
    for b in range(0,preds.shape[0]):
        for t in range(0,preds.shape[1]):
            preds_prob[b,t,:]=softmax(preds[b,t,:])

    preds_max_prob = numpy.max(preds_prob,axis=2)
    for img_name, pred, pred_max_prob in zip([image_path], preds_str, preds_max_prob):
        confidence_score = pred_max_prob.cumprod(axis=0)[-1]

        logger.info(f'{img_name:25s}\t{pred:25s}\t{confidence_score:0.4f}')


if __name__ == '__main__':
    # model files check and download
    check_and_download_models(WEIGHT_PATH, MODEL_PATH, REMOTE_PATH)

    recognize_from_image()
