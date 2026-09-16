import sys
import time
from typing import List
import random

# logger
from logging import getLogger  # noqa

import numpy as np
import cv2
from PIL import Image, ImageDraw
from scipy.special import log_softmax
import matplotlib.pyplot as plt
import matplotlib.patches as patches

import ailia

# import original modules
sys.path.append("../../util")
from arg_utils import get_base_parser, update_parser, get_savepath  # noqa
from model_utils import check_and_download_models  # noqa
from image_utils import normalize_image  # noqa
from detector_utils import load_image  # noqa

from beam_search import BeamSearchScorer
from logit_process import logits_processor
from processing_florence2 import post_process_generation


logger = getLogger(__name__)

# ======================
# Parameters
# ======================

WEIGHT_EMB_BASE_PATH = "embeddings_base.onnx"
WEIGHT_IMG_BASE_PATH = "feature_base.onnx"
WEIGHT_ENC_BASE_PATH = "encoder_base.onnx"
WEIGHT_DEC_BASE_PATH = "decoder_base.onnx"
WEIGHT_EMB_LARGE_PATH = "embeddings_large.onnx"
WEIGHT_IMG_LARGE_PATH = "feature_large.onnx"
WEIGHT_ENC_LARGE_PATH = "encoder_large.onnx"
WEIGHT_DEC_LARGE_PATH = "decoder_large.onnx"
MODEL_EMB_BASE_PATH = "embeddings_base.onnx.prototxt"
MODEL_IMG_BASE_PATH = "feature_base.onnx.prototxt"
MODEL_ENC_BASE_PATH = "encoder_base.onnx.prototxt"
MODEL_DEC_BASE_PATH = "decoder_base.onnx.prototxt"
MODEL_EMB_LARGE_PATH = "embeddings_large.onnx.prototxt"
MODEL_IMG_LARGE_PATH = "feature_large.onnx.prototxt"
MODEL_ENC_LARGE_PATH = "encoder_large.onnx.prototxt"
MODEL_DEC_LARGE_PATH = "decoder_large.onnx.prototxt"
REMOTE_PATH = "https://storage.googleapis.com/ailia-models/florence2/"

IMAGE_PATH = "car.jpg"
SAVE_IMAGE_PATH = "output.png"

IMG_SIZE = 768

# ======================
# Arguemnt Parser Config
# ======================

parser = get_base_parser("Florence-2", IMAGE_PATH, SAVE_IMAGE_PATH)
parser.add_argument(
    "-p",
    "--prompt",
    choices=[
        "CAPTION",
        "DETAILED_CAPTION",
        "MORE_DETAILED_CAPTION",
        "CAPTION_TO_PHRASE_GROUNDING",
        "OD",
        "DENSE_REGION_CAPTION",
        "REGION_PROPOSAL",
        "OCR",
        "OCR_WITH_REGION",
    ],
    default="CAPTION",
    help="prompt",
)
parser.add_argument(
    "--text_input",
    type=str,
    default=None,
    help="TEXT_INPUT (use by CAPTION_TO_PHRASE_GROUNDING)",
)
parser.add_argument(
    "--model_type", choices=["base", "large"], default="base", help="model type"
)
parser.add_argument(
    "--disable_ailia_tokenizer", action="store_true", help="disable ailia tokenizer."
)
parser.add_argument("--onnx", action="store_true", help="execute onnxruntime version.")
args = update_parser(parser, check_input_type=False)


# ======================
# Secondary Functions
# ======================

colormap = [
    "blue",
    "orange",
    "green",
    "purple",
    "brown",
    "pink",
    "gray",
    "olive",
    "cyan",
    "red",
    "lime",
    "indigo",
    "violet",
    "aqua",
    "magenta",
    "coral",
    "gold",
    "tan",
    "skyblue",
]


def plot_bbox(image, data, savepath):
    # Create a figure and axes
    fig, ax = plt.subplots(
        figsize=(image.shape[1] / 100, image.shape[0] / 100), dpi=100
    )

    # Display the image
    ax.imshow(image)

    # Plot each bounding box
    for bbox, label in zip(data["bboxes"], data["labels"]):
        # Unpack the bounding box coordinates
        x1, y1, x2, y2 = bbox
        # Create a Rectangle patch
        rect = patches.Rectangle(
            (x1, y1), x2 - x1, y2 - y1, linewidth=1, edgecolor="r", facecolor="none"
        )
        # Add the rectangle to the Axes
        ax.add_patch(rect)
        # Annotate the label
        plt.text(
            x1,
            y1,
            label,
            color="white",
            fontsize=8,
            bbox=dict(facecolor="red", alpha=0.5),
        )

    # Remove the axis ticks and labels
    ax.axis("off")

    # Set tight layout to fit the image exactly
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0)

    # Save the figure without changing the original size or adding padding
    plt.savefig(savepath, pad_inches=0, dpi=100)


def draw_ocr_bboxes(image, prediction, savepath, scale=1):
    image = Image.fromarray(image)
    draw = ImageDraw.Draw(image)

    bboxes, labels = prediction["quad_boxes"], prediction["labels"]
    for box, label in zip(bboxes, labels):
        color = random.choice(colormap)
        new_box = (np.array(box) * scale).tolist()
        draw.polygon(new_box, width=3, outline=color)
        draw.text(
            (new_box[0] + 8, new_box[1] + 2),
            "{}".format(label),
            align="right",
            fill=color,
        )

    # Save or display the image
    image.save(savepath)


# ======================
# Main functions
# ======================


def preprocess(img):
    h = w = IMG_SIZE

    img = np.array(Image.fromarray(img).resize((w, h), Image.Resampling.BICUBIC))

    img = normalize_image(img, normalize_type="ImageNet")

    img = img.transpose(2, 0, 1)  # HWC -> CHW
    img = np.expand_dims(img, axis=0)
    img = img.astype(np.float16)

    return img


def construct_prompts(text):
    task_prompts_without_inputs = {
        "<OCR>": "What is the text in the image?",
        "<OCR_WITH_REGION>": "What is the text in the image, with regions?",
        "<CAPTION>": "What does the image describe?",
        "<DETAILED_CAPTION>": "Describe in detail what is shown in the image.",
        "<MORE_DETAILED_CAPTION>": "Describe with a paragraph what is shown in the image.",
        "<OD>": "Locate the objects with category name in the image.",
        "<DENSE_REGION_CAPTION>": "Locate the objects in the image, with their descriptions.",
        "<REGION_PROPOSAL>": "Locate the region proposals in the image.",
    }
    task_prompts_with_input = {
        "<CAPTION_TO_PHRASE_GROUNDING>": "Locate the phrases in the caption: {input}",
        "<REFERRING_EXPRESSION_SEGMENTATION>": "Locate {input} in the image with mask",
        "<REGION_TO_SEGMENTATION>": "What is the polygon mask of region {input}",
        "<OPEN_VOCABULARY_DETECTION>": "Locate {input} in the image.",
        "<REGION_TO_CATEGORY>": "What is the region {input}?",
        "<REGION_TO_DESCRIPTION>": "What does the region {input} describe?",
        "<REGION_TO_OCR>": "What text is in the region {input}?",
    }

    # replace the task tokens with the task prompts if task token is in the text
    prompts = []
    for _text in text:
        # 1. fixed task prompts without additional inputs
        for task_token, task_prompt in task_prompts_without_inputs.items():
            if task_token in _text:
                assert (
                    _text == task_token
                ), f"Task token {task_token} should be the only token in the text."
                _text = task_prompt
                break
        # 2. task prompts with additional inputs
        for task_token, task_prompt in task_prompts_with_input.items():
            if task_token in _text:
                _text = task_prompt.format(input=_text.replace(task_token, ""))
                break
        prompts.append(_text)

    return prompts


def decode(
    net,
    input_ids: np.ndarray,
    encoder_hidden_states: np.ndarray,
    past_key_values: List[np.ndarray],
):
    if not args.onnx:
        decoder_output = net.predict(
            [input_ids, encoder_hidden_states, *past_key_values]
        )
    else:
        decoder_output = net.run(
            None,
            {
                "input_ids": input_ids,
                "encoder_hidden_states": encoder_hidden_states,
                "past_key_values.0.decoder.key": past_key_values[0],
                "past_key_values.0.decoder.value": past_key_values[1],
                "past_key_values.0.encoder.key": past_key_values[2],
                "past_key_values.0.encoder.value": past_key_values[3],
                "past_key_values.1.decoder.key": past_key_values[4],
                "past_key_values.1.decoder.value": past_key_values[5],
                "past_key_values.1.encoder.key": past_key_values[6],
                "past_key_values.1.encoder.value": past_key_values[7],
                "past_key_values.2.decoder.key": past_key_values[8],
                "past_key_values.2.decoder.value": past_key_values[9],
                "past_key_values.2.encoder.key": past_key_values[10],
                "past_key_values.2.encoder.value": past_key_values[11],
                "past_key_values.3.decoder.key": past_key_values[12],
                "past_key_values.3.decoder.value": past_key_values[13],
                "past_key_values.3.encoder.key": past_key_values[14],
                "past_key_values.3.encoder.value": past_key_values[15],
                "past_key_values.4.decoder.key": past_key_values[16],
                "past_key_values.4.decoder.value": past_key_values[17],
                "past_key_values.4.encoder.key": past_key_values[18],
                "past_key_values.4.encoder.value": past_key_values[19],
                "past_key_values.5.decoder.key": past_key_values[20],
                "past_key_values.5.decoder.value": past_key_values[21],
                "past_key_values.5.encoder.key": past_key_values[22],
                "past_key_values.5.encoder.value": past_key_values[23],
            },
        )

    logits, new_past_key_values = decoder_output[0], decoder_output[1:]

    return logits, new_past_key_values


def stopping_criteria(input_ids: np.array) -> bool:
    stopping_criteria.max_length = max_length = 1025

    cur_len = input_ids.shape[-1]
    is_done = cur_len >= max_length
    is_done = np.full(input_ids.shape[0], is_done)

    eos_token_id = 2
    is_done = is_done | np.isin(input_ids[:, -1], eos_token_id)

    return is_done


def greedy_search(net, encoder_hidden_states):
    model_type = args.model_type

    pad_token_id = 1
    bos_token_id = 2
    eos_token_id = 2
    batch_size = 1
    num_beams = 3

    # prepare beam search scorer
    beam_scorer = BeamSearchScorer(
        batch_size=batch_size,
        num_beams=num_beams,
        length_penalty=1.0,
        do_early_stopping=True,
        num_beam_hyps_to_keep=1,
        max_length=1025,
    )

    input_ids = np.ones((num_beams, 1), dtype=int) * bos_token_id
    encoder_hidden_states = np.repeat(encoder_hidden_states, repeats=num_beams, axis=0)
    past_key_values = [
        np.zeros(
            (num_beams, 12 if model_type == "base" else 16, 0, 64), dtype=np.float16
        )
    ] * (24 if model_type == "base" else 48)

    # initialise score of first beam with 0 and the rest with -1e9.
    beam_scores = np.zeros((batch_size, num_beams))
    beam_scores[:, 1:] = -1e9
    beam_scores = beam_scores.flatten()

    this_peer_finished = False  # used by synced_gpus only
    decoder_prompt_len = input_ids.shape[-1]  # record the prompt length of decoder

    while True:
        decoder_input_ids = input_ids
        past_length = past_key_values[0].shape[2]
        decoder_input_ids = decoder_input_ids[:, past_length:]

        logits, past_key_values = decode(
            net,
            decoder_input_ids,
            encoder_hidden_states,
            past_key_values,
        )

        next_token_logits = logits[:, -1, :]
        next_token_scores = log_softmax(next_token_logits, axis=-1)

        next_token_scores_processed = logits_processor(input_ids, next_token_scores)
        next_token_scores = next_token_scores_processed + np.broadcast_to(
            beam_scores[:, None], next_token_scores_processed.shape
        )

        # reshape for beam search
        vocab_size = next_token_scores.shape[-1]
        next_token_scores = next_token_scores.reshape(1, num_beams * vocab_size)

        # Beam token selection
        n_eos_tokens = 1
        n_tokens_to_keep = (1 + n_eos_tokens) * num_beams

        next_tokens = np.argsort(-next_token_scores, axis=1, kind="stable")[
            :, :n_tokens_to_keep
        ]
        next_token_scores = np.take_along_axis(next_token_scores, next_tokens, axis=1)

        next_indices = next_tokens // vocab_size
        next_tokens = next_tokens % vocab_size

        beam_outputs = beam_scorer.process(
            input_ids,
            next_token_scores,
            next_tokens,
            next_indices,
            pad_token_id=pad_token_id,
            eos_token_id=eos_token_id,
            decoder_prompt_len=decoder_prompt_len,
        )
        beam_scores = beam_outputs["next_beam_scores"]
        beam_next_tokens = beam_outputs["next_beam_tokens"].astype(int)
        beam_idx = beam_outputs["next_beam_indices"].astype(int)

        input_ids = np.concatenate(
            [input_ids[beam_idx, :], np.expand_dims(beam_next_tokens, axis=-1)], axis=-1
        )

        # temporary_reorder_cache
        reordered_past = []
        n = 24 if model_type == "base" else 48
        for i in range(0, n, 4):
            layer_past = past_key_values[i : i + 4]
            reordered_past += [
                np.take(past_state, beam_idx, axis=0) for past_state in layer_past[:2]
            ] + layer_past[2:]
        past_key_values = reordered_past

        if beam_scorer.is_done or all(stopping_criteria(input_ids)):
            this_peer_finished = True

        if this_peer_finished:
            break

    sequence_outputs = beam_scorer.finalize(
        input_ids,
        beam_scores,
        next_tokens,
        next_indices,
        pad_token_id=pad_token_id,
        eos_token_id=eos_token_id,
        max_length=stopping_criteria.max_length,
        decoder_prompt_len=decoder_prompt_len,
    )

    return sequence_outputs["sequences"]


def predict(models, img, task_prompt, text_input=None):
    im_h, im_w, _ = img.shape
    img = img[:, :, ::-1]  # BGR -> RGB
    pixel_values = preprocess(img)

    if text_input is None:
        prompt = task_prompt
    else:
        prompt = task_prompt + text_input
    text = construct_prompts([prompt])

    tokenizer = models["tokenizer"]
    inputs = tokenizer(
        text,
        return_tensors="np",
        padding=False,
        return_token_type_ids=False,
    )
    input_ids = inputs["input_ids"]
    attention_mask = inputs["attention_mask"]

    # Extra the input embeddings
    net = models["embedding"]
    if not args.onnx:
        output = net.predict([input_ids])
    else:
        output = net.run(None, {"input_ids": input_ids})
    inputs_embeds = output[0]

    # Merge text and images
    net = models["encode_image"]
    if not args.onnx:
        output = net.predict([pixel_values])
    else:
        output = net.run(None, {"pixel_values": pixel_values})
    image_features = output[0]
    inputs_embeds = np.concatenate([image_features, inputs_embeds], axis=1)

    attention_mask = np.ones(inputs_embeds.shape[:2], dtype=int)

    net = models["encoder"]
    if not args.onnx:
        output = net.predict([inputs_embeds, attention_mask])
    else:
        output = net.run(
            None, {"inputs_embeds": inputs_embeds, "attention_mask": attention_mask}
        )
    last_hidden_state = output[0]

    net = models["decoder"]
    generated_ids = greedy_search(net, last_hidden_state)

    tokenizer = models["tokenizer"]
    generated_text = tokenizer.batch_decode(generated_ids, skip_special_tokens=False)[0]

    answer = post_process_generation(
        generated_text, task=task_prompt, image_size=(im_w, im_h)
    )
    return answer


def recognize_from_image(models):
    prompt = "<%s>" % args.prompt
    logger.info("Prompt: %s" % prompt)

    text_input = None
    if prompt == "<CAPTION_TO_PHRASE_GROUNDING>":
        text_input = args.text_input
        if text_input is None:
            raise ValueError("TEXT_INPUT is required for CAPTION_TO_PHRASE_GROUNDING")

        logger.info("TEXT_INPUT: %s" % text_input)

    # input image loop
    for image_path in args.input:
        logger.info(image_path)

        # prepare input data
        img = load_image(image_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
        img_rgb = img[:, :, ::-1]

        # inference
        logger.info("Start inference...")
        if args.benchmark:
            logger.info("BENCHMARK mode")
            total_time_estimation = 0
            for i in range(args.benchmark_count):
                start = int(round(time.time() * 1000))
                answer = predict(models, img, prompt, text_input=text_input)
                end = int(round(time.time() * 1000))
                estimation_time = end - start

                # Logging
                logger.info(f"\tailia processing estimation time {estimation_time} ms")
                if i != 0:
                    total_time_estimation = total_time_estimation + estimation_time

            logger.info(
                f"\taverage time estimation {total_time_estimation / (args.benchmark_count - 1)} ms"
            )
        else:
            answer = predict(models, img, prompt, text_input=text_input)

        print(answer)

        info = next(iter(answer.values()))
        if "bboxes" in info or "quad_boxes" in info:
            # plot result
            savepath = get_savepath(args.savepath, image_path, ext=".png")
            logger.info(f"saved at : {savepath}")
            if "bboxes" in info:
                plot_bbox(img_rgb, info, savepath)
            if "quad_boxes" in info:
                draw_ocr_bboxes(img_rgb, info, savepath)

    logger.info("Script finished successfully.")


def main():
    if args.model_type == "base":
        WEIGHT_EMB_PATH = WEIGHT_EMB_BASE_PATH
        WEIGHT_IMG_PATH = WEIGHT_IMG_BASE_PATH
        WEIGHT_ENC_PATH = WEIGHT_ENC_BASE_PATH
        WEIGHT_DEC_PATH = WEIGHT_DEC_BASE_PATH
        MODEL_EMB_PATH = MODEL_EMB_BASE_PATH
        MODEL_IMG_PATH = MODEL_IMG_BASE_PATH
        MODEL_ENC_PATH = MODEL_ENC_BASE_PATH
        MODEL_DEC_PATH = MODEL_DEC_BASE_PATH
    else:
        WEIGHT_EMB_PATH = WEIGHT_EMB_LARGE_PATH
        WEIGHT_IMG_PATH = WEIGHT_IMG_LARGE_PATH
        WEIGHT_ENC_PATH = WEIGHT_ENC_LARGE_PATH
        WEIGHT_DEC_PATH = WEIGHT_DEC_LARGE_PATH
        MODEL_EMB_PATH = MODEL_EMB_LARGE_PATH
        MODEL_IMG_PATH = MODEL_IMG_LARGE_PATH
        MODEL_ENC_PATH = MODEL_ENC_LARGE_PATH
        MODEL_DEC_PATH = MODEL_DEC_LARGE_PATH
    check_and_download_models(WEIGHT_EMB_PATH, MODEL_EMB_PATH, REMOTE_PATH)
    check_and_download_models(WEIGHT_IMG_PATH, MODEL_IMG_PATH, REMOTE_PATH)
    check_and_download_models(WEIGHT_ENC_PATH, MODEL_ENC_PATH, REMOTE_PATH)
    check_and_download_models(WEIGHT_DEC_PATH, MODEL_DEC_PATH, REMOTE_PATH)

    env_id = args.env_id

    # initialize
    if not args.onnx:
        embedding = ailia.Net(MODEL_EMB_PATH, WEIGHT_EMB_PATH, env_id=env_id, memory_mode=args.memory_mode)
        encode_image = ailia.Net(MODEL_IMG_PATH, WEIGHT_IMG_PATH, env_id=env_id, memory_mode=args.memory_mode)
        encoder = ailia.Net(MODEL_ENC_PATH, WEIGHT_ENC_PATH, env_id=env_id, memory_mode=args.memory_mode)
        decoder = ailia.Net(MODEL_DEC_PATH, WEIGHT_DEC_PATH, env_id=env_id, memory_mode=args.memory_mode)
    else:
        import onnxruntime

        providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]

        embedding = onnxruntime.InferenceSession(WEIGHT_EMB_PATH, providers=providers)
        encode_image = onnxruntime.InferenceSession(
            WEIGHT_IMG_PATH, providers=providers
        )
        encoder = onnxruntime.InferenceSession(WEIGHT_ENC_PATH, providers=providers)
        decoder = onnxruntime.InferenceSession(WEIGHT_DEC_PATH, providers=providers)

    if args.disable_ailia_tokenizer:
        import transformers

        tokenizer = transformers.BartTokenizerFast.from_pretrained("./tokenizer")

        #tokenizer.additional_special_tokensにすでにSpecialTokensが含まれている
        if len(tokenizer.additional_special_tokens) != 1024:
            raise("special token error")

        #tokens_to_add = {
        #    "additional_special_tokens": tokenizer.additional_special_tokens
        #    + ["<od>", "</od>", "<ocr>", "</ocr>"]
        #    + [f"<loc_{x}>" for x in range(1000)]
        #    + [
        #        # fmt: off
        #        '<cap>', '</cap>', '<ncap>', '</ncap>','<dcap>', '</dcap>', '<grounding>', 
        #        '</grounding>', '<seg>', '</seg>', '<sep>', '<region_cap>', '</region_cap>', 
        #        '<region_to_desciption>', '</region_to_desciption>', '<proposal>', '</proposal>', 
        #        '<poly>', '</poly>', '<and>'
        #        # fmt: on
        #    ]
        #}
        #tokenizer.add_special_tokens(tokens_to_add)
    else:
        import ailia_tokenizer
        tokenizer = ailia_tokenizer.RobertaTokenizer.from_pretrained("./tokenizer")

        tokens_to_add = {
            "additional_special_tokens":
              ["<od>", "</od>", "<ocr>", "</ocr>"]
            + [f"<loc_{x}>" for x in range(1000)]
            + [
                # fmt: off
                '<cap>', '</cap>', '<ncap>', '</ncap>','<dcap>', '</dcap>', '<grounding>', 
                '</grounding>', '<seg>', '</seg>', '<sep>', '<region_cap>', '</region_cap>', 
                '<region_to_desciption>', '</region_to_desciption>', '<proposal>', '</proposal>', 
                '<poly>', '</poly>', '<and>'
                # fmt: on
            ]
        }
        tokenizer.add_special_tokens(tokens_to_add)

    models = {
        "tokenizer": tokenizer,
        "embedding": embedding,
        "encode_image": encode_image,
        "encoder": encoder,
        "decoder": decoder,
    }

    # generate
    recognize_from_image(models)


if __name__ == "__main__":
    main()
