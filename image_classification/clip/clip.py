import sys
import time

import numpy as np
import cv2
from PIL import Image

import ailia

# import original modules
sys.path.append('../../util')
from arg_utils import get_base_parser, update_parser, get_savepath  # noqa: E402
from model_utils import check_and_download_models  # noqa: E402
from detector_utils import load_image  # noqa: E402C
from classifier_utils import plot_results, print_results  # noqa: E402
from math_utils import softmax  # noqa: E402C
import webcamera_utils  # noqa: E402
# logger
from logging import getLogger  # noqa: E402

from simple_tokenizer import SimpleTokenizer as _Tokenizer

logger = getLogger(__name__)

_tokenizer = _Tokenizer()

# ======================
# Parameters
# ======================

WEIGHT_VITB32_IMAGE_PATH = 'ViT-B32-encode_image.onnx'
MODEL_VITB32_IMAGE_PATH = 'ViT-B32-encode_image.onnx.prototxt'
WEIGHT_VITB32_TEXT_PATH = 'ViT-B32-encode_text.onnx'
MODEL_VITB32_TEXT_PATH = 'ViT-B32-encode_text.onnx.prototxt'
WEIGHT_VITL14_IMAGE_PATH = 'ViT-L14-encode_image.onnx'
MODEL_VITL14_IMAGE_PATH = 'ViT-L14-encode_image.onnx.prototxt'
WEIGHT_VITL14_TEXT_PATH = 'ViT-L14-encode_text.onnx'
MODEL_VITL14_TEXT_PATH = 'ViT-L14-encode_text.onnx.prototxt'
WEIGHT_RN50_IMAGE_PATH = 'RN50-encode_image.onnx'
MODEL_RN50_IMAGE_PATH = 'RN50-encode_image.onnx.prototxt'
WEIGHT_RN50_TEXT_PATH = 'RN50-encode_text.onnx'
MODEL_RN50_TEXT_PATH = 'RN50-encode_text.onnx.prototxt'
REMOTE_PATH = 'https://storage.googleapis.com/ailia-models/clip/'

IMAGE_PATH = 'chelsea.png'
SAVE_IMAGE_PATH = 'output.png'

IMAGE_SIZE = 224

# ======================
# Arguemnt Parser Config
# ======================

parser = get_base_parser(
    'CLIP', IMAGE_PATH, SAVE_IMAGE_PATH, fp16_support=False
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
    '-m', '--model_type', default='ViTB32', choices=('ViTB32', 'ViTL14', 'RN50'),
    help='model type'
)
parser.add_argument(
    '--onnx',
    action='store_true',
    help='execute onnxruntime version.'
)
args = update_parser(parser)


# ======================
# Main functions
# ======================

def tokenize(texts, context_length=77, truncate=False):
    if isinstance(texts, str):
        texts = [texts]

    sot_token = _tokenizer.encoder["<|startoftext|>"]
    eot_token = _tokenizer.encoder["<|endoftext|>"]
    all_tokens = [[sot_token] + _tokenizer.encode(text) + [eot_token] for text in texts]
    result = np.zeros((len(all_tokens), context_length), dtype=np.int64)

    for i, tokens in enumerate(all_tokens):
        if len(tokens) > context_length:
            if truncate:
                tokens = tokens[:context_length]
                tokens[-1] = eot_token
            else:
                raise RuntimeError(f"Input {texts[i]} is too long for context length {context_length}")

        result[i, :len(tokens)] = np.array(tokens)

    result = result.astype(np.int64)

    return result


def preprocess(img):
    h, w = (IMAGE_SIZE, IMAGE_SIZE)
    im_h, im_w, _ = img.shape

    # resize
    scale = h / min(im_h, im_w)
    ow, oh = round(im_w * scale), round(im_h * scale)
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


def predict(net, img, text_feature):
    img = preprocess(img)

    # feedforward
    if not args.onnx:
        output = net.predict([img])
    else:
        output = net.run(None, {'image': img})

    image_feature = output[0]

    image_feature = image_feature / np.linalg.norm(image_feature, ord=2, axis=-1, keepdims=True)

    logit_scale = 100
    logits_per_image = (image_feature * logit_scale).dot(text_feature.T)

    pred = softmax(logits_per_image, axis=1)

    return pred[0]


def predict_text_feature(net, text):
    text_tokens = tokenize(text)

    # feedforward
    text_feature = []
    batch_size_limit = 16

    for i in range(0, text_tokens.shape[0], batch_size_limit):
        batch_size = min(batch_size_limit, text_tokens.shape[0] - i)
        logger.info("Embedding " + str(i) + " to " + str(i+batch_size))
        if not args.onnx:
            output = net.predict([text_tokens[i:i+batch_size,:]])
        else:
            output = net.run(None, {'text': text_tokens[i:i+batch_size,:]})
        text_feature.append(output[0])

    text_feature = np.concatenate(text_feature)

    text_feature = text_feature / np.linalg.norm(text_feature, ord=2, axis=-1, keepdims=True)

    return text_feature


def recognize_from_image(net_image, net_text):
    text_inputs = args.text_inputs
    desc_file = args.desc_file
    if desc_file:
        with open(desc_file) as f:
            text_inputs = [x.strip() for x in f.readlines() if x.strip()]
    elif text_inputs is None:
        text_inputs = [f"a {c}" for c in ("human", "dog", "cat")]

    text_feature = predict_text_feature(net_text, text_inputs)

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
                pred = predict(net_image, img, text_feature)
                end = int(round(time.time() * 1000))
                estimation_time = (end - start)

                # Loggin
                logger.info(f'\tailia processing estimation time {estimation_time} ms')
                if i != 0:
                    total_time_estimation = total_time_estimation + estimation_time

            logger.info(f'\taverage time estimation {total_time_estimation / (args.benchmark_count - 1)} ms')
        else:
            pred = predict(net_image, img, text_feature)

        # show results
        pred = np.expand_dims(pred, axis=0)
        print_results(pred, text_inputs)

    logger.info('Script finished successfully.')


def recognize_from_video(net_image, net_text):
    text_inputs = args.text_inputs
    desc_file = args.desc_file
    if desc_file:
        with open(desc_file) as f:
            text_inputs = [x.strip() for x in f.readlines() if x.strip()]
    elif text_inputs is None:
        text_inputs = [f"a {c}" for c in ("human", "dog", "cat")]

    text_feature = predict_text_feature(net_text, text_inputs)

    capture = webcamera_utils.get_capture(args.video)
    # create video writer if savepath is specified as video format
    if args.savepath is not None:
        f_h = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        f_w = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        writer = webcamera_utils.get_writer(args.savepath, f_h, f_w)
    else:
        writer = None

    frame_shown = False
    while (True):
        ret, frame = capture.read()
        if (cv2.waitKey(1) & 0xFF == ord('q')) or not ret:
            break
        if frame_shown and cv2.getWindowProperty('frame', cv2.WND_PROP_VISIBLE) == 0:
            break

        img = frame

        pred = predict(net_image, img, text_feature)

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
    dic_model = {
        'ViTB32': (
            (WEIGHT_VITB32_IMAGE_PATH, MODEL_VITB32_IMAGE_PATH),
            (WEIGHT_VITB32_TEXT_PATH, MODEL_VITB32_TEXT_PATH)),
        'ViTL14': (
            (WEIGHT_VITL14_IMAGE_PATH, MODEL_VITL14_IMAGE_PATH),
            (WEIGHT_VITL14_TEXT_PATH, MODEL_VITL14_TEXT_PATH)),
        'RN50': (
            (WEIGHT_RN50_IMAGE_PATH, MODEL_RN50_IMAGE_PATH),
            (WEIGHT_RN50_TEXT_PATH, MODEL_RN50_TEXT_PATH)),
    }
    (WEIGHT_IMAGE_PATH, MODEL_IMAGE_PATH), (WEIGHT_TEXT_PATH, MODEL_TEXT_PATH) = dic_model[args.model_type]

    # model files check and download
    logger.info('Checking encode_image model...')
    check_and_download_models(WEIGHT_IMAGE_PATH, MODEL_IMAGE_PATH, REMOTE_PATH)
    logger.info('Checking encode_text model...')
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

    if args.video is not None:
        # video mode
        recognize_from_video(net_image, net_text)
    else:
        # image mode
        recognize_from_image(net_image, net_text)


if __name__ == '__main__':
    main()
