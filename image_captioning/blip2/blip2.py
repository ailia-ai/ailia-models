import sys
import time
from typing import List
from logging import getLogger

import numpy as np
import cv2
from PIL import Image

# import original modules
sys.path.append('../../util')
from arg_utils import get_base_parser, update_parser  # noqa
from model_utils import check_and_download_models, check_and_download_file  # noqa
from detector_utils import load_image  # noqa

import ailia

logger = getLogger(__name__)

# ======================
# Parameters
# ======================
WEIGHT_PATH = 'blip2-opt-2.7b.onnx'
WEIGHT_PB_PATH = 'blip2-opt-2.7b_weights.pb'
MODEL_PATH = 'blip2-opt-2.7b.onnx.prototxt'
WEIGHT_VIS_PATH = 'vision_model.onnx'
WEIGHT_VIS_PB_PATH = 'vision_model_weights.pb'
MODEL_VIS_PATH = 'vision_model.onnx.prototxt'
REMOTE_PATH = 'https://storage.googleapis.com/ailia-models/blip2/'

IMAGE_PATH = 'merlion.png'

IMG_SIZE = 224

# ======================
# Arguemnt Parser Config
# ======================

parser = get_base_parser(
    'BLIP-2', IMAGE_PATH, None, fp16_support=False
)
parser.add_argument(
    '--onnx',
    action='store_true',
    help='execute onnxruntime version.'
)
parser.add_argument(
    '--disable_ailia_tokenizer',
    action='store_true',
    help='disable ailia tokenizer.'
)
args = update_parser(parser, check_input_type=False)


# ======================
# Main functions
# ======================

def preprocess(img):
    im_h, im_w, _ = img.shape
    h = w = IMG_SIZE

    img = np.array(Image.fromarray(img).resize((w, h), Image.Resampling.BICUBIC))

    img = img / 255

    image_mean = (0.48145466, 0.4578275, 0.40821073)
    image_std = (0.26862954, 0.26130258, 0.27577711)
    img = (img - image_mean) / image_std

    img = img.transpose(2, 0, 1)  # HWC -> CHW
    img = np.expand_dims(img, axis=0)
    img = img.astype(np.float16)

    return img


def decode(
        net,
        inputs_embeds: np.ndarray,
        input_ids: np.ndarray,
        attention_mask: np.ndarray,
        past_key_values: List[np.ndarray]):
    if not args.onnx:
        decoder_output = net.predict([
            attention_mask,
            input_ids,
            inputs_embeds,
            past_key_values[0],
            past_key_values[1],
            past_key_values[2],
            past_key_values[3],
            past_key_values[4],
            past_key_values[5],
            past_key_values[6],
            past_key_values[7],
            past_key_values[8],
            past_key_values[9],
            past_key_values[10],
            past_key_values[11],
            past_key_values[12],
            past_key_values[13],
            past_key_values[14],
            past_key_values[15],
            past_key_values[16],
            past_key_values[17],
            past_key_values[18],
            past_key_values[19],
            past_key_values[20],
            past_key_values[21],
            past_key_values[22],
            past_key_values[23],
            past_key_values[24],
            past_key_values[25],
            past_key_values[26],
            past_key_values[27],
            past_key_values[28],
            past_key_values[29],
            past_key_values[30],
            past_key_values[31],
            past_key_values[32],
            past_key_values[33],
            past_key_values[34],
            past_key_values[35],
            past_key_values[36],
            past_key_values[37],
            past_key_values[38],
            past_key_values[39],
            past_key_values[40],
            past_key_values[41],
            past_key_values[42],
            past_key_values[43],
            past_key_values[44],
            past_key_values[45],
            past_key_values[46],
            past_key_values[47],
            past_key_values[48],
            past_key_values[49],
            past_key_values[50],
            past_key_values[51],
            past_key_values[52],
            past_key_values[53],
            past_key_values[54],
            past_key_values[55],
            past_key_values[56],
            past_key_values[57],
            past_key_values[58],
            past_key_values[59],
            past_key_values[60],
            past_key_values[61],
            past_key_values[62],
            past_key_values[63],
        ])
    else:
        decoder_output = net.run(
            None, {
                'attention_mask': attention_mask,
                'input_ids': input_ids,
                'inputs_embeds': inputs_embeds,
                'past_key_values_0_key': past_key_values[0],
                'past_key_values_0_value': past_key_values[1],
                'past_key_values_1_key': past_key_values[2],
                'past_key_values_1_value': past_key_values[3],
                'past_key_values_2_key': past_key_values[4],
                'past_key_values_2_value': past_key_values[5],
                'past_key_values_3_key': past_key_values[6],
                'past_key_values_3_value': past_key_values[7],
                'past_key_values_4_key': past_key_values[8],
                'past_key_values_4_value': past_key_values[9],
                'past_key_values_5_key': past_key_values[10],
                'past_key_values_5_value': past_key_values[11],
                'past_key_values_6_key': past_key_values[12],
                'past_key_values_6_value': past_key_values[13],
                'past_key_values_7_key': past_key_values[14],
                'past_key_values_7_value': past_key_values[15],
                'past_key_values_8_key': past_key_values[16],
                'past_key_values_8_value': past_key_values[17],
                'past_key_values_9_key': past_key_values[18],
                'past_key_values_9_value': past_key_values[19],
                'past_key_values_10_key': past_key_values[20],
                'past_key_values_10_value': past_key_values[21],
                'past_key_values_11_key': past_key_values[22],
                'past_key_values_11_value': past_key_values[23],
                'past_key_values_12_key': past_key_values[24],
                'past_key_values_12_value': past_key_values[25],
                'past_key_values_13_key': past_key_values[26],
                'past_key_values_13_value': past_key_values[27],
                'past_key_values_14_key': past_key_values[28],
                'past_key_values_14_value': past_key_values[29],
                'past_key_values_15_key': past_key_values[30],
                'past_key_values_15_value': past_key_values[31],
                'past_key_values_16_key': past_key_values[32],
                'past_key_values_16_value': past_key_values[33],
                'past_key_values_17_key': past_key_values[34],
                'past_key_values_17_value': past_key_values[35],
                'past_key_values_18_key': past_key_values[36],
                'past_key_values_18_value': past_key_values[37],
                'past_key_values_19_key': past_key_values[38],
                'past_key_values_19_value': past_key_values[39],
                'past_key_values_20_key': past_key_values[40],
                'past_key_values_20_value': past_key_values[41],
                'past_key_values_21_key': past_key_values[42],
                'past_key_values_21_value': past_key_values[43],
                'past_key_values_22_key': past_key_values[44],
                'past_key_values_22_value': past_key_values[45],
                'past_key_values_23_key': past_key_values[46],
                'past_key_values_23_value': past_key_values[47],
                'past_key_values_24_key': past_key_values[48],
                'past_key_values_24_value': past_key_values[49],
                'past_key_values_25_key': past_key_values[50],
                'past_key_values_25_value': past_key_values[51],
                'past_key_values_26_key': past_key_values[52],
                'past_key_values_26_value': past_key_values[53],
                'past_key_values_27_key': past_key_values[54],
                'past_key_values_27_value': past_key_values[55],
                'past_key_values_28_key': past_key_values[56],
                'past_key_values_28_value': past_key_values[57],
                'past_key_values_29_key': past_key_values[58],
                'past_key_values_29_value': past_key_values[59],
                'past_key_values_30_key': past_key_values[60],
                'past_key_values_30_value': past_key_values[61],
                'past_key_values_31_key': past_key_values[62],
                'past_key_values_31_value': past_key_values[63],
            }
        )

    logits, new_past_key_values = decoder_output[0], decoder_output[1:]

    return logits, new_past_key_values


def stopping_criteria(
        input_ids: np.array) -> bool:
    max_length = 21

    cur_len = input_ids.shape[-1]
    is_done = cur_len >= max_length
    return is_done


def greedy_search(net, inputs_embeds):
    bos_token_id = 2
    eos_token_id = np.array([50118])

    shape = inputs_embeds.shape[:2]
    batch_size = shape[0]

    input_ids = np.ones((batch_size, 1), dtype=int) * bos_token_id
    attention_mask = np.ones(shape, dtype=int)
    past_key_values = [np.zeros((batch_size, shape[1] - 1, 0, 80), dtype=np.float16)] * 64

    # keep track of which sequences are already finished
    unfinished_sequences = np.ones(input_ids.shape[0], dtype=int)

    this_peer_finished = False  # used by synced_gpus only
    while True:
        logits, past_key_values = decode(
            net, inputs_embeds, input_ids[:, 1:][:, -1:], attention_mask, past_key_values
        )

        next_tokens_scores = logits[:, -1, :]

        # argmax
        next_tokens = np.argmax(next_tokens_scores, axis=-1)

        # update generated ids, model inputs, and length for next step
        input_ids = np.concatenate([input_ids, next_tokens[:, None]], axis=-1)
        attention_mask = np.concatenate(
            [attention_mask, np.ones((attention_mask.shape[0], 1), dtype=int)],
            axis=-1
        )
        inputs_embeds = inputs_embeds[:, :0, :]

        # if eos_token was found in one sentence, set sentence to finished
        unfinished_sequences = unfinished_sequences * np.prod(
            np.tile(next_tokens, (eos_token_id.shape[0], 1)) != eos_token_id[:, None],
            axis=0
        )

        # stop when each sentence is finished
        if np.max(unfinished_sequences) == 0:
            this_peer_finished = True

        # stop if we exceed the maximum length
        if stopping_criteria(input_ids):
            this_peer_finished = True

        if this_peer_finished:
            break

    return input_ids


def predict(models, img):
    img = img[:, :, ::-1]  # BGR -> RGB

    img = preprocess(img)

    net = models['vis']
    if not args.onnx:
        output = net.predict([img])
    else:
        output = net.run(None, {'pixel_values': img})
    embeds = output[0]

    net = models['net']
    generated_ids = greedy_search(net, embeds)

    tokenizer = models['tokenizer']
    generated_text = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
    generated_text = generated_text[0].strip()

    return generated_text


def recognize_from_image(models):
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
                generated_text = predict(models, img)
                end = int(round(time.time() * 1000))
                estimation_time = (end - start)

                # Logging
                logger.info(f'\tailia processing estimation time {estimation_time} ms')
                if i != 0:
                    total_time_estimation = total_time_estimation + estimation_time

            logger.info(f'\taverage time estimation {total_time_estimation / (args.benchmark_count - 1)} ms')
        else:
            generated_text = predict(models, img)

    logger.info('### Caption ### ')
    logger.info(generated_text)

    logger.info('Script finished successfully.')


def main():
    # model files check and download
    check_and_download_models(WEIGHT_PATH, MODEL_PATH, REMOTE_PATH)
    check_and_download_models(WEIGHT_VIS_PATH, MODEL_VIS_PATH, REMOTE_PATH)
    check_and_download_file(WEIGHT_PB_PATH, REMOTE_PATH)
    check_and_download_file(WEIGHT_VIS_PB_PATH, REMOTE_PATH)

    env_id = args.env_id

    # initialize
    if not args.onnx:
        net = ailia.Net(MODEL_PATH, WEIGHT_PATH, env_id=env_id, memory_mode=args.memory_mode)
        vis_net = ailia.Net(MODEL_VIS_PATH, WEIGHT_VIS_PATH, env_id=env_id, memory_mode=args.memory_mode)
        if args.profile:
            net.set_profile_mode(True)
            vis_net.set_profile_mode(True)
    else:
        import onnxruntime
        cuda = 0 < ailia.get_gpu_environment_id()
        providers = ['CUDAExecutionProvider', 'CPUExecutionProvider'] if cuda else ['CPUExecutionProvider']
        net = onnxruntime.InferenceSession(WEIGHT_PATH, providers=providers)
        vis_net = onnxruntime.InferenceSession(WEIGHT_VIS_PATH, providers=providers)

    if args.disable_ailia_tokenizer:
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained("tokenizer")
    else:
        from ailia_tokenizer import GPT2Tokenizer
        tokenizer = GPT2Tokenizer.from_pretrained("./tokenizer/")

    models = {
        'net': net,
        'vis': vis_net,
        'tokenizer': tokenizer,
    }

    recognize_from_image(models)

    if args.profile and not args.onnx:
        print("--- profile net")
        print(net.get_summary())
        print("")
        print("--- profile vis_net")
        print(vis_net.get_summary())


if __name__ == '__main__':
    main()
