import argparse
import os
import sys
import time

import ailia
import cv2
import numpy as np
from scipy.special import softmax

sys.path.append('../../util')
# logger
from logging import getLogger  # noqa: E402

from image_utils import imread, load_image  # noqa: E402
from math_utils import softmax
from model_utils import check_and_download_models  # noqa: E402
from arg_utils import get_base_parser, get_savepath, update_parser  # noqa: E402
from webcamera_utils import get_capture, get_writer  # noqa: E402

logger = getLogger(__name__)

from simple_tokenizer import SimpleTokenizer as _Tokenizer

_tokenizer = _Tokenizer()

# ======================
# Parameters 1
# ======================
IMAGE_PATH = None
SAVE_IMAGE_PATH = None
MAX_CLASS_COUNT = 5

# ======================
# Argument Parser Config
# ======================
parser:argparse.ArgumentParser = get_base_parser(
    'ActionCLIP',
    IMAGE_PATH,
    SAVE_IMAGE_PATH,
)
parser.add_argument(
    '-t', '--text', dest='text_inputs', type=str,
    action='append',
    help='Input text. (can be specified multiple times)'
)
parser.add_argument(
    '--desc_file', default=None, metavar='DESC_FILE', type=str,
    help='description file'
)
parser.add_argument(
    '-m', '--model_type', default='vit-32-8f', choices=('vit-32-8f'),
    help='model type'
)
parser.add_argument(
    '--seed', default=None, type=int,
    help='random seed'
)
args = update_parser(parser)


# ======================
# Parameters 2
# ======================
MODELS = {
    'vit-32-8f': {
        'num_segments': 8,
    },
}

# ======================
# Utils
# ======================
def get_model_params(model_type):
    params = {'model_type': model_type}
    params.update(MODELS[model_type])

    for submodel in ['text_clip', 'image_clip', 'fusion']:
        stem = f'{model_type}-{submodel}'
        opt = ''
        if submodel == 'text_clip':
            opt = '.opt'

        if submodel not in params:
            params[submodel] = {}

        if 'weight_path' not in params[submodel]:
            params[submodel]['weight_path'] = f'{stem}{opt}.onnx'
        if 'model_path' not in params[submodel]:
            params[submodel]['model_path'] = f'{stem}{opt}.onnx.prototxt'
        if 'remote_path' not in params[submodel]:
            params[submodel]['remote_path'] = f'https://storage.googleapis.com/ailia-models/action_clip/'

    return params

def init(model_paths):
    """Initialize all ailia models"""
    # net initialize
    models = {}
    for k, v in model_paths.items():
        models[k] = ailia.Net(v['model_path'], v['weight_path'],
                              env_id=args.env_id, memory_mode=args.memory_mode)
    return models

def tokenize(texts, context_length: int = 77):
    """
    Returns the tokenized representation of given input string(s)

    Parameters
    ----------
    texts : Union[str, List[str]]
        An input string or a list of input strings to tokenize

    context_length : int
        The context length to use; all CLIP models use 77 as the context length

    Returns
    -------
    A two-dimensional array containing the resulting tokens, shape = [number of input strings, context_length]
    """
    if isinstance(texts, str):
        texts = [texts]

    sot_token = _tokenizer.encoder["<|startoftext|>"]
    eot_token = _tokenizer.encoder["<|endoftext|>"]
    all_tokens = [[sot_token] + _tokenizer.encode(text) + [eot_token] for text in texts]
    result = np.zeros((len(all_tokens), context_length), dtype=np.int64)

    for i, tokens in enumerate(all_tokens):
        if len(tokens) > context_length:
            raise RuntimeError(f"Input {texts[i]} is too long for context length {context_length}")
        result[i, :len(tokens)] = np.asarray(tokens)

    return result

def text_prompt(classes):
    text_aug = [f"a photo of action {{}}", f"a picture of action {{}}", f"Human action of {{}}", f"{{}}, an action",
                f"{{}} this is an action", f"{{}}, a video of action", f"Playing action of {{}}", f"{{}}",
                f"Playing a kind of action, {{}}", f"Doing a kind of action, {{}}", f"Look, the human is {{}}",
                f"Can you recognize the action of {{}}?", f"Video classification of {{}}", f"A video of {{}}",
                f"The man is {{}}", f"The woman is {{}}"]
    text_dict = {}
    num_text_aug = len(text_aug)

    for ii, txt in enumerate(text_aug):
        text_dict[ii] = np.concatenate([tokenize(txt.format(c)) for c in classes])

    classes_ = np.concatenate([v for k, v in text_dict.items()])

    return classes_, num_text_aug, text_dict

def resize(img, size, interpolation=cv2.INTER_CUBIC):
    size_ = size
    if isinstance(size, int):
        size_ = (size, size)
    size_ = np.asarray(size_, dtype=np.int64)

    img_size = np.asarray(img.shape[:2])
    idx_min = np.argmin(img_size)
    scale = size_[idx_min] / img_size[idx_min]
    size_ = np.round(scale * img_size).astype(np.int64)

    new_img = cv2.resize(img, tuple(map(int, size_[::-1])),
                         interpolation=interpolation)
    return new_img

def preprocess_image(imgs, size=224):
    scale_size = size * 256 // 224
    processed = np.stack([resize(img, scale_size) for img in imgs])
    size_ = np.asarray(processed.shape[1:3])
    y0, x0 = (size_ - size) // 2
    processed = processed[:, y0:y0+size, x0:x0+size]

    processed = np.rollaxis((processed / 255.).astype(np.float32), 3, 1) 

    mean = np.asarray([0.48145466, 0.4578275, 0.40821073]).reshape((1, 3, 1, 1))
    std = np.asarray([0.26862954, 0.26130258, 0.27577711]).reshape((1, 3, 1, 1))
    processed = (processed - mean) / std

    return processed

def predict(nets, text, imgs, num_segments):
    text_net, image_net, fusion_net = [
        nets[e] for e in ['text_clip', 'image_clip', 'fusion']
    ]
    classes, num_text_aug, text_dict = text_prompt(text)
    text_features = text_net.predict([classes])[0]

    image_inputs = preprocess_image(imgs)
    image_inputs = image_inputs.reshape(
        (-1, num_segments, 3) + image_inputs.shape[-2:]
    )
    b, t, c, h, w = image_inputs.shape
    image_inputs = image_inputs.reshape((-1, c, h, w))
    image_features = image_net.predict([image_inputs])[0].reshape((b, t, -1))

    image_features = fusion_net.predict([image_features])[0]
    image_features /= np.linalg.norm(image_features, axis=-1, keepdims=True)
    text_features /= np.linalg.norm(text_features, axis=-1, keepdims=True)
    similarity = (100.0 * image_features @ text_features.T)
    similarity = softmax(similarity.reshape((b, num_text_aug, -1)), axis=-1)
    similarity = similarity.mean(axis=1)
    return similarity

def print_results(scores, labels, logger, max_class_count=MAX_CLASS_COUNT):
    """Print classification results"""
    ids_order = np.argsort(-scores)

    logger.info('==============================================================')
    logger.info(f'class_count = {len(ids_order)}')
    for i in range(max_class_count):
        idx = ids_order[i]
        logger.info(f'+ idx = {i}')
        logger.info(f'  category = {idx} [{labels[idx]}]')
        logger.info(f'  prob = {scores[idx]}')
    logger.info('')


# ======================
# Main functions
# ======================
def recognize_from_video(args, models, model_params):
    capture = get_capture(args.video)

    num_segments = model_params['num_segments']
    frame_cnt = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    intervals = (
        np.linspace(0, frame_cnt, num_segments+1).round().astype(np.int64)
    )
    sample_ids = [
        np.random.randint(intervals[i], intervals[i+1])
        for i in range(num_segments)
    ]

    frames = []
    for idx in sample_ids:
        capture.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = capture.read()
        frames.append(frame)

    logger.info('Start inference...')
    if args.benchmark:
        logger.info('BENCHMARK mode')
        for i in range(args.benchmark_count):
            start = int(round(time.time() * 1000))
            scores = predict(models, args.text_inputs, frames, num_segments)[0]
            end = int(round(time.time() * 1000))
            logger.info(f'\tailia processing time {end - start} ms')
    else:
        scores = predict(models, args.text_inputs, frames, num_segments)[0]
    print_results(scores, args.text_inputs, logger)

    capture.release()
    cv2.destroyAllWindows()
    logger.info('Script finished successfully.')


def main():
    if args.video is None:
        args.video = os.path.join(
            os.path.dirname(__file__), 'action_recognition.gif'
        )
    text_inputs = args.text_inputs
    desc_file = args.desc_file
    if desc_file:
        with open(desc_file) as f:
            text_inputs = [x.strip() for x in f.readlines() if x.strip()]
    elif text_inputs is None:
        text_inputs = [
            'applauding', 'dancing', 'driving', 'driving car', 'driving truck',
            'eating', 'punching', 'reading', 'surfing', 'talking phone',
        ]
        logger.info(f'No text provided, using {text_inputs}...')
    args.text_inputs = text_inputs

    model_params = get_model_params(args.model_type)
    model_paths = {e: model_params[e] for e in ['text_clip', 'image_clip', 'fusion']}
    # model files check and download
    for model in model_paths.values():
        check_and_download_models(
            model['weight_path'], model['model_path'], model['remote_path']
        )
    models = init(model_paths)

    recognize_from_video(args, models, model_params)


if __name__ == '__main__':
    if args.seed != None:
        np.random.seed(args.seed)
    main()
