import sys
import time
from logging import getLogger

import ailia
import cv2
import numpy as np

# import original modules
sys.path.append('../../util')
from arg_utils import get_base_parser, update_parser, get_savepath  # noqa
from model_utils import check_and_download_models  # noqa
from image_utils import normalize_image  # noqa
from detector_utils import load_image  # noqa
from webcamera_utils import get_capture, get_writer  # noqa

logger = getLogger(__name__)

# ======================
# Parameters
# ======================

WEIGHT_PATH = 'isnetis.onnx'
MODEL_PATH = 'isnetis.onnx.prototxt'
REMOTE_PATH = 'https://storage.googleapis.com/ailia-models/anime-segmentation/'

IMAGE_PATH = 'demo.png'
SAVE_IMAGE_PATH = 'output.png'

IMAGE_SIZE = 1024

# ======================
# Arguemnt Parser Config
# ======================

parser = get_base_parser(
    'Anime Segmentation', IMAGE_PATH, SAVE_IMAGE_PATH
)
parser.add_argument(
    '--img-size', type=int, default=IMAGE_SIZE,
    help='hyperparameter, input image size of the net'
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

def preprocess(img):
    im_h, im_w, _ = img.shape

    s = args.img_size
    if im_h > im_w:
        h, w = s, int(s * im_w / im_h)
    else:
        h, w = int(s * im_h / im_w), s

    img = cv2.resize(img, (w, h), interpolation=cv2.INTER_LINEAR)
    img = normalize_image(img, normalize_type='255')

    ph, pw = s - h, s - w
    pad_img = np.zeros([s, s, 3], dtype=np.float32)
    pad_img[ph // 2:ph // 2 + h, pw // 2:pw // 2 + w] = img
    img = pad_img

    img = img.transpose(2, 0, 1)  # HWC -> CHW
    img = np.expand_dims(img, axis=0)
    img = img.astype(np.float32)

    return img, (h, w)


def predict(net, img):
    im_h, im_w = img.shape[:2]
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img, (h, w) = preprocess(img)

    # feedforward
    if not args.onnx:
        output = net.predict([img])
    else:
        output = net.run(None, {'img': img})
    pred = output[0]

    s = pred.shape[2]
    ph, pw = s - h, s - w
    pred = pred[0, :, ph // 2:ph // 2 + h, pw // 2:pw // 2 + w]

    mask = pred.transpose(1, 2, 0)  # CHW -> HWC
    mask = cv2.resize(mask, (im_w, im_h), interpolation=cv2.INTER_LINEAR)[:, :, np.newaxis]

    return mask


def recognize_from_image(net):
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
                mask = predict(net, img)
                end = int(round(time.time() * 1000))
                estimation_time = (end - start)

                # Logging
                logger.info(f'\tailia processing estimation time {estimation_time} ms')
                if i != 0:
                    total_time_estimation = total_time_estimation + estimation_time

            logger.info(f'\taverage time estimation {total_time_estimation / (args.benchmark_count - 1)} ms')
        else:
            mask = predict(net, img)

        res_img = np.concatenate((mask * img, mask * 255), axis=2).astype(np.uint8)

        # plot result
        savepath = get_savepath(args.savepath, image_path, ext='.png')
        logger.info(f'saved at : {savepath}')
        cv2.imwrite(savepath, res_img)

    logger.info('Script finished successfully.')


def recognize_from_video(net):
    video_file = args.video if args.video else args.input[0]
    capture = get_capture(video_file)
    assert capture.isOpened(), 'Cannot capture source'

    # create video writer if savepath is specified as video format
    if args.savepath != SAVE_IMAGE_PATH:
        f_h = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        f_w = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        writer = get_writer(args.savepath, f_h, f_w)
    else:
        writer = None

    frame_shown = False
    while True:
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
            res_img = res_img.astype(np.uint8)
            writer.write(res_img)

    capture.release()
    cv2.destroyAllWindows()
    if writer is not None:
        writer.release()

    logger.info('Script finished successfully.')


def main():
    # model files check and download
    check_and_download_models(WEIGHT_PATH, MODEL_PATH, REMOTE_PATH)

    env_id = args.env_id

    # initialize
    if not args.onnx:
        net = ailia.Net(MODEL_PATH, WEIGHT_PATH, env_id=env_id, memory_mode=args.memory_mode)
    else:
        import onnxruntime
        net = onnxruntime.InferenceSession(WEIGHT_PATH)

    if args.video is not None:
        recognize_from_video(net)
    else:
        recognize_from_image(net)


if __name__ == '__main__':
    main()
