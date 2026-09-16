import sys
import time

import ailia
import cv2
import numpy as np

# import original modules
sys.path.append('../../util')
# logger
from logging import getLogger  # noqa: E402

import webcamera_utils  # noqa: E402
from image_utils import normalize_image  # noqa: E402
from detector_utils import load_image  # noqa: E402
from model_utils import check_and_download_models  # noqa: E402
from arg_utils import get_base_parser, get_savepath, update_parser  # noqa: E402

logger = getLogger(__name__)

# ======================
# Parameters 1
# ======================
IMAGE_PATH = 'input.jpg'
SAVE_IMAGE_PATH = 'output.jpg'
IMAGE_SIZE = 1024

# ======================
# Arguemnt Parser Config
# ======================
parser = get_base_parser(
    'DIS segmentation model', IMAGE_PATH, SAVE_IMAGE_PATH
)
parser.add_argument(
    '--img-size', type=int, default=IMAGE_SIZE,
    help='hyperparameter, input image size of the net'
)
parser.add_argument(
    '-n', '--normal',
    action='store_true',
    help='By default, the optimized model is used, but with this option, ' +
    'you can switch to the normal (not optimized) model'
)
args = update_parser(parser)


# ======================
# Parameters 2
# ======================
WEIGHT_PATH = 'isnet-general-use.onnx' if not args.normal else 'dis.onnx'
MODEL_PATH = WEIGHT_PATH + '.prototxt'
REMOTE_PATH = "https://storage.googleapis.com/ailia-models/dis/"


# ======================
# Utils
# ======================

def preprocess(img):
    im_h, im_w, _ = img.shape

    s = args.img_size

    img = cv2.resize(img, (s, s), interpolation=cv2.INTER_LINEAR)
    img = normalize_image(img, normalize_type='127.5') / 2

    img = img.transpose(2, 0, 1)  # HWC -> CHW
    img = np.expand_dims(img, axis=0)
    img = img.astype(np.float32)

    return img

def predict(net, img):
    im_h, im_w = img.shape[:2]
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = preprocess(img)

    # feedforward
    output = net.predict([img])
    pred = output[0][0]

    mask = pred.transpose(1, 2, 0)  # CHW -> HWC
    mask = cv2.resize(mask, (im_w, im_h), interpolation=cv2.INTER_LINEAR)[:, :, np.newaxis]

    ma = np.max(mask)
    mi = np.min(mask)
    mask = (mask-mi)/(ma-mi)

    return mask

# ======================
# Main functions
# ======================
def recognize_from_image():
    # net initialize
    net = ailia.Net(MODEL_PATH, WEIGHT_PATH, env_id=args.env_id, memory_mode=args.memory_mode)

    # input image loop
    for image_path in args.input:
        # prepare input data
        logger.info(image_path)
        img = load_image(image_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)

        # inference
        logger.info('Start inference...')
        if args.benchmark:
            logger.info('BENCHMARK mode')
            for i in range(5):
                start = int(round(time.time() * 1000))
                # preds_ailia = net.predict(input_data)
                mask = predict(net, img)
                end = int(round(time.time() * 1000))
                logger.info(f'\tailia processing time {end - start} ms')
        else:
            # preds_ailia = net.predict(input_data)
            mask = predict(net, img)

        # postprocessing
        res_img = np.concatenate((mask * img, mask * 255), axis=2).astype(np.uint8)

        # plot result
        savepath = get_savepath(args.savepath, image_path, ext='.png')
        logger.info(f'saved at : {savepath}')
        cv2.imwrite(savepath, res_img)

    logger.info('Script finished successfully.')


def recognize_from_video():
    # net initialize
    net = ailia.Net(MODEL_PATH, WEIGHT_PATH, env_id=args.env_id, memory_mode=args.memory_mode)
    flag_set_shape = False

    capture = webcamera_utils.get_capture(args.video)

    # create video writer if savepath is specified as video format
    if args.savepath != SAVE_IMAGE_PATH:
        f_h = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        f_w = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        save_h, save_w = webcamera_utils.calc_adjust_fsize(
            f_h, f_w, IMAGE_HEIGHT, IMAGE_WIDTH
        )
        writer = webcamera_utils.get_writer(args.savepath, save_h, save_w)
    else:
        writer = None

    frame_shown = False
    while(True):
        ret, frame = capture.read()
        if (cv2.waitKey(1) & 0xFF == ord('q')) or not ret:
            break
        if frame_shown and cv2.getWindowProperty('frame', cv2.WND_PROP_VISIBLE) == 0:
            break

        # inference
        mask = predict(net, frame)

        # plot result
        res_img = (mask * frame).astype(np.uint8)

        # show
        cv2.imshow('frame', res_img)
        frame_shown = True

        # save results
        if writer is not None:
            writer.write(res_img)

    capture.release()
    cv2.destroyAllWindows()
    if writer is not None:
        writer.release()
    logger.info('Script finished successfully.')


def main():
    # model files check and download
    check_and_download_models(WEIGHT_PATH, MODEL_PATH, REMOTE_PATH)

    if args.video is not None:
        # video mode
        recognize_from_video()
    else:
        # image mode
        recognize_from_image()


if __name__ == '__main__':
    main()
