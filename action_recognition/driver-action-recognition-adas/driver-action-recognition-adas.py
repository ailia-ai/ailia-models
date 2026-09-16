import sys

import numpy as np
import cv2
import json

import ailia

# import original modules
sys.path.append('../../util')
from arg_utils import get_base_parser, update_parser  # noqa: E402
from model_utils import check_and_download_models  # noqa: E402
from math_utils import softmax  # noqa: E402
import webcamera_utils  # noqa: E402

# logger
from logging import getLogger  # noqa: E402

logger = getLogger(__name__)

from labels import LABELS

# ======================
# Parameters
# ======================
WEIGHT_ENC_PATH = 'driver-action-recognition-adas-0002-encoder.onnx'
MODEL_ENC_PATH = 'driver-action-recognition-adas-0002-encoder.onnx.prototxt'
WEIGHT_DEC_PATH = 'driver-action-recognition-adas-0002-decoder.onnx'
MODEL_DEC_PATH = 'driver-action-recognition-adas-0002-decoder.onnx.prototxt'
REMOTE_PATH = 'https://storage.googleapis.com/ailia-models/driver-action-recognition-adas/'

VIDEO_PATH = 'action_recognition.gif'

IMAGE_SIZE = 224

# ======================
# Arguemnt Parser Config
# ======================

parser = get_base_parser('driver-action-recognition-adas', VIDEO_PATH, None)
parser.add_argument(
    '--cui',
    action='store_true',
    help="Don't display preview in GUI."
)
parser.add_argument(
    '-w', '--write_json',
    action='store_true',
    help='save result to json'
)
args = update_parser(parser)


# ======================
# Utils
# ======================

def render_frame(frame, display_text):
    xmin, ymax = (0, 70)
    xmax, ymin = (700, 0)
    alpha = 0.6
    color = (0, 0, 0)
    frame[ymin:ymax, xmin:xmax, :] = \
        frame[ymin:ymax, xmin:xmax, :] * (1 - alpha) + np.asarray(color) * alpha

    TEXT_LEFT_MARGIN = 15
    TEXT_VERTICAL_INTERVAL = 45
    text_loc = (TEXT_LEFT_MARGIN, TEXT_VERTICAL_INTERVAL)

    FONT_STYLE = 2
    FONT_SIZE = 1
    FONT_COLOR = (255, 255, 255)
    cv2.putText(frame, display_text, text_loc, FONT_STYLE, FONT_SIZE, FONT_COLOR)

    return frame


def save_json(json_path, results):
    out = []
    for r in results:
        if r is None:
            continue
        else:
            r['prob'] = float(r['prob'])
            out.append(r)
    with open(json_path, 'w') as f:
        json.dump(out, f, indent=2)


# ======================
# Main functions
# ======================

def preprocess(img):
    size = IMAGE_SIZE
    h, w, _ = img.shape

    # adaptive_resize
    scale = size / min(h, w)
    ow, oh = int(w * scale), int(h * scale)
    if ow != w or oh != h:
        img = cv2.resize(img, (ow, oh))

    # center_crop
    if ow > size:
        x = (ow - size) // 2
        img = img[:, x:x + size, :]
    if oh > size:
        y = (oh - size) // 2
        img = img[y:y + size, :, :]

    img = np.expand_dims(img, axis=0)

    return img


def recognize_from_video(enc, dec):
    video_file = args.video if args.video else args.input[0]
    capture = webcamera_utils.get_capture(video_file)
    assert capture.isOpened(), 'Cannot capture source'

    # create video writer if savepath is specified as video format
    if args.savepath != None and not args.cui:
        logger.warning(
            'currently, video results cannot be output correctly...'
        )
        f_h = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        f_w = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        writer = webcamera_utils.get_writer(args.savepath, f_h, f_w)
    else:
        writer = None

    sequence_size = 16

    embeddings = []
    results_per_frame = []
    frame_shown = False
    while True:
        ret, frame = capture.read()
        if (cv2.waitKey(1) & 0xFF == ord('q')) or not ret:
            break
        if frame_shown and cv2.getWindowProperty('frame', cv2.WND_PROP_VISIBLE) == 0:
            break

        img = preprocess(frame)

        # feedforward
        output = enc.predict([img])
        embedding = output[0]
        embedding = embedding.reshape((1, -1))

        embeddings.append(embedding)
        embeddings = embeddings[-sequence_size:]
        result = None
        if len(embeddings) == sequence_size:

            decoder_input = np.concatenate(embeddings, axis=0)
            decoder_input = np.expand_dims(decoder_input, axis=0)

            output = dec.predict([decoder_input])
            logits = output[0]

            probs = softmax(logits)
            probs = probs[0]

            i = np.argmax(probs)
            result = {
                'label': LABELS[i],
                'prob': probs[np.argmax(probs)] * 100
            }
        results_per_frame.append(result)

        if result is not None:
            display_text = '{} - {:.2f}%'.format(
                result['label'],
                result['prob']
            )
        else:
            display_text = 'Preparing...'

        frame = render_frame(frame, display_text)

        if not args.cui or args.video:
            cv2.imshow('frame', frame)
            frame_shown = True
        else:
            print(display_text)

        # save results
        if writer is not None:
            writer.write(frame)

    capture.release()
    cv2.destroyAllWindows()
    if writer is not None:
        writer.release()

    if args.write_json:
        if not args.savepath:
            logger.warning('json file was not saved because savepath is not specified.')
        else:
            json_path = args.savepath.rsplit('.', 1)[0] + '.json'
            save_json(json_path, results_per_frame)
            logger.info(f'result saved to {json_path}')

    logger.info('Script finished successfully.')


def main():
    # model files check and download
    logger.info('Checking encoder model...')
    check_and_download_models(WEIGHT_ENC_PATH, MODEL_ENC_PATH, REMOTE_PATH)
    logger.info('Checking decoder model...')
    check_and_download_models(WEIGHT_DEC_PATH, MODEL_DEC_PATH, REMOTE_PATH)

    # net initialize
    enc = ailia.Net(MODEL_ENC_PATH, WEIGHT_ENC_PATH, env_id=args.env_id, memory_mode=args.memory_mode)
    dec = ailia.Net(MODEL_DEC_PATH, WEIGHT_DEC_PATH, env_id=args.env_id, memory_mode=args.memory_mode)

    # video mode
    recognize_from_video(enc, dec)


if __name__ == '__main__':
    main()
