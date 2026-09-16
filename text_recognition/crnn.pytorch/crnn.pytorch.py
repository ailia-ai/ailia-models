import sys
import time

import cv2
import numpy as np
from PIL import Image

import ailia

# import original modules
sys.path.append('../../util')
from arg_utils import get_base_parser, update_parser  # noqa: E402
from model_utils import check_and_download_models  # noqa: E402
import webcamera_utils  # noqa: E402

# logger
from logging import getLogger   # noqa: E402
logger = getLogger(__name__)

# ======================
# Parameters
# ======================
WEIGHT_PATH = 'crnn_pytorch.onnx'
MODEL_PATH = 'crnn_pytorch.onnx.prototxt'
REMOTE_PATH = 'https://storage.googleapis.com/ailia-models/crnn_pytorch/'

IMAGE_PATH = 'demo.png'

# ======================
# Arguemnt Parser Config
# ======================
parser = get_base_parser(
    'Convolutional Recurrent Neural Network',
    IMAGE_PATH,
    None,
)
parser.add_argument(
    '-o', '--onnx',
    action='store_true',
    default=False,
    help='Use onnx runtime'
)
args = update_parser(parser)

alphabet = '0123456789abcdefghijklmnopqrstuvwxyz'

# ======================
# Utils
# ======================
def pre_process(image):
    image = np.round((np.array(image.resize((100, 32), Image.BILINEAR))/255 - 0.5)/0.5, 4)
    image = np.expand_dims(np.expand_dims(image, 0), 0).astype(np.float32)
    return image


def post_process(preds, length, alphabet):
    preds = np.argmax(preds, axis=2).transpose(1, 0)[0]
    
    alphabet = alphabet + '-'  # for `-1` index
    dict = {}
    for i, char in enumerate(alphabet):
        dict[char] = i + 1

    assert len(preds)== length, "text with length: {} does not match declared length: {}".format(len(preds), length)
    char_list = []
    for i in range(length):
        if preds[i] != 0 and (not (i > 0 and preds[i - 1] == preds[i])):
            char_list.append(alphabet[preds[i] - 1])
    return ''.join(char_list)


def predict(net, image):
    preds = net.predict({'input.1':image})[0]
    return preds


# ======================
# Main functions
# ======================
def recognize_from_image(net):
    for image_path in args.input:
        # prepare input data
        image = Image.open(image_path).convert('L')
        image = pre_process(image)

        # inference
        logger.info('Start inference...')
        if args.benchmark:
            logger.info('BENCHMARK mode')
            for i in range(5):
                start = int(round(time.time() * 1000))
                preds = predict(net, image)
                end = int(round(time.time() * 1000))
                logger.info(f'\tailia processing time {end - start} ms')
        else:
            preds = predict(net, image)

        sim_pred = post_process(preds, len(preds), alphabet)
        logger.info('============================================')
        logger.info('String recognized from image is:'+str(sim_pred))
        logger.info('============================================')

    logger.info('Script finished successfully.')


def recognize_from_video(net):
    capture = webcamera_utils.get_capture(args.video)

    # create video writer if savepath is specified as video format
    # if args.savepath != SAVE_IMAGE_PATH:
    #     f_h = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    #     f_w = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    #     writer = webcamera_utils.get_writer(args.savepath, f_h, f_w)
    # else:
    #     writer = None
    
    frame_shown = False
    while(True):
        ret, image = capture.read()
        # press q to end video capture
        if (cv2.waitKey(1) & 0xFF == ord('q')) or not ret:
            break
        if frame_shown and cv2.getWindowProperty('frame', cv2.WND_PROP_VISIBLE) == 0:
            break

        logger.info('============================================')
        cv2.imshow('frame', image)
        frame_shown = True
        image = Image.fromarray(np.uint8(image)).convert('L')
        image = pre_process(image)
        preds = predict(net, image)
        sim_pred = post_process(preds, len(preds), alphabet)
        logger.info('String recognized from image is:'+str(sim_pred))

        # save results
        # if writer is not None:
        #     writer.write(img)

    capture.release()
    cv2.destroyAllWindows()
    # if writer is not None:
    #     writer.release()
    logger.info('Script finished successfully.')


def main():
    # model files check and download
    check_and_download_models(WEIGHT_PATH, MODEL_PATH, REMOTE_PATH)

    # model initialize
    net = ailia.Net(MODEL_PATH, WEIGHT_PATH, env_id=args.env_id, memory_mode=args.memory_mode)

    if args.video is not None:
        # video mode
        recognize_from_video(net)
    else:
        # image mode
        recognize_from_image(net)


if __name__ == '__main__':
    main()
