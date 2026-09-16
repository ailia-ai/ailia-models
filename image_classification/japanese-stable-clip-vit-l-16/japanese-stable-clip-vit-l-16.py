import sys
import time
import re

import numpy as np
import cv2
from PIL import Image

import ailia

# import original modules
sys.path.append('../../util')
from arg_utils import get_base_parser, update_parser, get_savepath  # noqa
from model_utils import check_and_download_models  # noqa
from image_utils import normalize_image  # noqa
from detector_utils import load_image  # noqa
from classifier_utils import plot_results, print_results, write_predictions  # noqa
from math_utils import softmax  # noqa
import webcamera_utils  # noqa
# logger
from logging import getLogger  # noqa

logger = getLogger(__name__)

# ======================
# Parameters
# ======================

WEIGHT_IMAGE_PATH = 'CLIP-ViT-L16-image.onnx'
MODEL_IMAGE_PATH = 'CLIP-ViT-L16-image.onnx.prototxt'
WEIGHT_TEXT_PATH = 'CLIP-ViT-L16-text.onnx'
MODEL_TEXT_PATH = 'CLIP-ViT-L16-text.onnx.prototxt'
REMOTE_PATH = 'https://storage.googleapis.com/ailia-models/japanese-stable-clip-vit-l-16/'

IMAGE_PATH = 'dog.jpeg'

IMAGE_SIZE = 224

# ======================
# Arguemnt Parser Config
# ======================

parser = get_base_parser(
    'Japanese Stable CLIP ViT-L/16', IMAGE_PATH, None
)
parser.add_argument(
    '-t', '--text', dest='text_inputs', type=str,
    action='append',
    help='Input text. (can be specified multiple times)'
)
parser.add_argument(
    '--onnx',
    action='store_true',
    help='execute onnxruntime version.'
)
parser.add_argument(
    '-w', '--write_json',
    action='store_true',
    help='Flag to output results to json file.'
)
parser.add_argument(
    '--disable_ailia_tokenizer',
    action='store_true',
    help='disable ailia tokenizer.'
)
args = update_parser(parser)


# ======================
# Secondaty Functions
# ======================

def whitespace_clean(text):
    text = re.sub(r"\s+", " ", text)
    text = text.strip()
    return text


# ======================
# Main functions
# ======================

def tokenize(tokenizer, texts, max_seq_len=77):
    if isinstance(texts, str):
        texts = [texts]

    texts = [whitespace_clean(text) for text in texts]

    inputs = tokenizer(
        texts,
        max_length=max_seq_len - 1,
        padding="max_length",
        truncation=True,
        add_special_tokens=False,
    )
    # add bos token at first place
    input_ids = [[tokenizer.bos_token_id] + ids for ids in inputs['input_ids']]
    attention_mask = [[1] + am for am in inputs['attention_mask']]
    position_ids = [list(range(0, len(input_ids[0])))] * len(texts)

    input_ids = np.array(input_ids, dtype=np.int64)
    attention_mask = np.array(attention_mask, dtype=np.int64)
    position_ids = np.array(position_ids, dtype=np.int64)
    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "position_ids": position_ids,
    }


def get_text_features(models, text):
    tokenizer = models['tokenizer']
    encodings = tokenize(tokenizer, text)

    input_ids = encodings['input_ids']
    attention_mask = encodings['attention_mask']
    position_ids = encodings['position_ids']

    # feedforward
    net = models['text']
    if not args.onnx:
        output = net.predict([input_ids, attention_mask, position_ids])
    else:
        output = net.run(None, {
            'input_ids': input_ids, 'attention_mask': attention_mask,
            'position_ids': position_ids
        })
    text_features = output[0]

    return text_features


def preprocess(img):
    img = img[:, :, ::-1]  # BGR -> RBG
    im_h, im_w, _ = img.shape

    # resize
    short, long = (im_w, im_h) if im_w <= im_h else (im_h, im_w)
    new_short, new_long = IMAGE_SIZE, (IMAGE_SIZE * long) // short
    ow, oh = (new_short, new_long) if im_w <= im_h else (new_long, new_short)
    if ow != im_w or oh != im_h:
        img = np.array(Image.fromarray(img).resize((ow, oh), Image.Resampling.BICUBIC))

    # center_crop
    # In case size is odd, (image_shape[0] + size[0]) // 2 won't give the proper result.
    top = (oh - IMAGE_SIZE) // 2
    bottom = top + IMAGE_SIZE
    # In case size is odd, (image_shape[1] + size[1]) // 2 won't give the proper result.
    left = (ow - IMAGE_SIZE) // 2
    right = left + IMAGE_SIZE
    if top >= 0 and bottom <= oh and left >= 0 and right <= ow:
        img = img[top:bottom, left:right, :]
    else:
        # If the image is too small, pad it with zeros
        pad_h = max(IMAGE_SIZE, oh)
        pad_w = max(IMAGE_SIZE, ow)
        pad_img = np.zeros((pad_h, pad_w, 3))

        top_pad = (pad_h - oh) // 2
        left_pad = (pad_w - ow) // 2
        pad_img[top_pad:top_pad + oh, left_pad:left_pad + ow, :] = img
        img = pad_img

    img = normalize_image(img, normalize_type='127.5')

    img = img.transpose(2, 0, 1)  # HWC -> CHW
    img = np.expand_dims(img, axis=0)
    img = img.astype(np.float32)

    return img


def predict(net, img, text_feature):
    img = preprocess(img)

    # feedforward
    if not args.onnx:
        output = net.predict([img])
    else:
        output = net.run(None, {'image': img})
    image_features = output[0]

    logit_scale = 100
    logits_per_image = (image_features * logit_scale).dot(text_feature.T)

    text_probs = softmax(logits_per_image, axis=1)

    return text_probs[0]


def recognize_from_image(models):
    text_inputs = args.text_inputs
    if text_inputs is None:
        text_inputs = ["犬", "猫", "象"]

    text_features = get_text_features(models, text_inputs)

    net_image = models['image']

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
                pred = predict(net_image, img, text_features)
                end = int(round(time.time() * 1000))
                estimation_time = (end - start)

                # Loggin
                logger.info(f'\tailia processing estimation time {estimation_time} ms')
                if i != 0:
                    total_time_estimation = total_time_estimation + estimation_time

            logger.info(f'\taverage time estimation {total_time_estimation / (args.benchmark_count - 1)} ms')
        else:
            pred = predict(net_image, img, text_features)

        # show results
        pred = np.expand_dims(pred, axis=0)
        print_results(pred, text_inputs)

        if args.write_json:
            json_path = '%s.json' % image_path.rsplit('.', 1)[0]
            write_predictions(json_path, pred, text_inputs, 'json')

    logger.info('Script finished successfully.')


def recognize_from_video(models):
    text_inputs = args.text_inputs
    if text_inputs is None:
        text_inputs = ["犬", "猫", "象"]

    text_features = get_text_features(models, text_inputs)

    capture = webcamera_utils.get_capture(args.video)
    # create video writer if savepath is specified as video format
    if args.savepath is not None:
        f_h = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        f_w = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        writer = webcamera_utils.get_writer(args.savepath, f_h, f_w)
    else:
        writer = None

    net_image = models['image']

    frame_shown = False
    while (True):
        ret, frame = capture.read()
        if (cv2.waitKey(1) & 0xFF == ord('q')) or not ret:
            break
        if frame_shown and cv2.getWindowProperty('frame', cv2.WND_PROP_VISIBLE) == 0:
            break

        img = frame

        pred = predict(net_image, img, text_features)

        plot_results(frame, np.expand_dims(pred, axis=0), text_inputs)

        cv2.imshow('frame', frame)
        frame_shown = True

        # save results
        if writer is not None:
            writer.write(frame)

    capture.release()
    cv2.destroyAllWindows()
    if writer is not None:
        writer.release()

    logger.info('Script finished successfully.')


def main():
    # model files check and download
    check_and_download_models(WEIGHT_IMAGE_PATH, MODEL_IMAGE_PATH, REMOTE_PATH)
    check_and_download_models(WEIGHT_TEXT_PATH, MODEL_TEXT_PATH, REMOTE_PATH)

    env_id = args.env_id

    # initialize
    if not args.onnx:
        net_image = ailia.Net(MODEL_IMAGE_PATH, WEIGHT_IMAGE_PATH, env_id=env_id, memory_mode=args.memory_mode)
        net_text = ailia.Net(MODEL_TEXT_PATH, WEIGHT_TEXT_PATH, env_id=env_id, memory_mode=args.memory_mode)
    else:
        import onnxruntime
        net_image = onnxruntime.InferenceSession(WEIGHT_IMAGE_PATH)
        net_text = onnxruntime.InferenceSession(WEIGHT_TEXT_PATH)

    if args.disable_ailia_tokenizer:
        from transformers import T5Tokenizer
        tokenizer = T5Tokenizer.from_pretrained("tokenizer")
    else:
        from ailia_tokenizer import T5Tokenizer
        tokenizer = T5Tokenizer.from_pretrained("./tokenizer/")
        tokenizer.add_special_tokens({"pad_token" : "[PAD]"})
        tokenizer.cls_token_id = 4
        tokenizer.bos_token_id = 1

    models = {
        "tokenizer": tokenizer,
        "image": net_image,
        "text": net_text,
    }

    if args.video is not None:
        # video mode
        recognize_from_video(models)
    else:
        # image mode
        recognize_from_image(models)


if __name__ == '__main__':
    main()
