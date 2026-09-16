import sys
import time

import cv2
import numpy as np
import matplotlib.pyplot as plt

import ailia

sys.path.append('../../util')
from arg_utils import get_base_parser, update_parser, get_savepath  # noqa: E402
from model_utils import check_and_download_models  # noqa: E402
from detector_utils import load_image  # noqa: E402C
import webcamera_utils  # noqa: E402

# logger
from logging import getLogger  # noqa: E402

logger = getLogger(__name__)

# ======================
# Parameters
# ======================

WEIGHT_KITTI_PATH = 'LDRN_KITTI_ResNext101_data.onnx'
MODEL_KITTI_PATH = 'LDRN_KITTI_ResNext101_data.onnx.prototxt'
WEIGHT_KITTI_GRAD_PATH = 'LDRN_KITTI_ResNext101_data_grad.onnx'
MODEL_KITTI_GRAD_PATH = 'LDRN_KITTI_ResNext101_data_grad.onnx.prototxt'
WEIGHT_NYU_PATH = 'LDRN_NYU_ResNext101_data.onnx'
MODEL_NYU_PATH = 'LDRN_NYU_ResNext101_data.onnx.prototxt'
REMOTE_PATH = 'https://storage.googleapis.com/ailia-models/lap-depth/'

IMAGE_PATH = 'kitti_demo.jpg'
SAVE_IMAGE_PATH = 'output.png'
KITTI_HEIGHT = 352
NYU_HEIGHT = 432

# ======================
# Argument Parser Config
# ======================

parser = get_base_parser(
    'LapDepth',
    IMAGE_PATH,
    SAVE_IMAGE_PATH,
)
parser.add_argument(
    '-m', '--model_type', metavar='MODEL_TYPE',
    default='kitti', choices=('kitti', 'kitti-grad', 'nyu'),
    help='model type: ' + ' | '.join(('kitti', 'kitti-grad', 'nyu'))
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

def preprocess(img, size):
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])

    im_h, im_w = img.shape[:2]

    img = img / 255
    img = (img - mean) / std

    new_h = size
    new_w = im_w * (size / im_h)
    new_w = int((new_w // 16) * 16)
    img = cv2.resize(img, (new_w, new_h), cv2.INTER_LINEAR)

    img = img.transpose((2, 0, 1))  # HWC -> CHW
    img = np.expand_dims(img, axis=0)
    img = img.astype(np.float32)

    return img


def predict(net, img, pretrained='kitti'):
    im_h, im_w = img.shape[:2]
    size = KITTI_HEIGHT if pretrained == 'kitti' else NYU_HEIGHT

    # initial preprocesses
    img = preprocess(img, size)
    img_flip = img[:, :, :, ::-1]

    # feedforward
    if not args.onnx:
        output = net.predict([img])
        out_flip = net.predict([img_flip])
    else:
        output = net.run(['depth'],
                         {'img': img})
        out_flip = net.run(['depth'],
                           {'img': img_flip})

    out = output[-1]
    out_flip = out_flip[-1][:, :, :, ::-1]

    out = (out[0, 0] + out_flip[0, 0]) / 2

    if size > im_h:
        out = cv2.resize(out, (im_w, im_h), cv2.INTER_LINEAR)

    if pretrained == 'kitti':
        out = out[int(out.shape[0] * 0.18):, :]
        out = out * 256.0
    elif pretrained == 'nyu':
        out = out * 1000

    out = out.astype(np.uint16)
    out = (out / out.max()) * 255.0

    return out


def recognize_from_image(net, pretrained='kitti'):
    # input image loop
    for image_path in args.input:
        # prepare input data
        logger.info(image_path)

        img = load_image(image_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2RGB)

        # inference
        logger.info('Start inference...')
        if args.benchmark:
            logger.info('BENCHMARK mode')
            total_time_estimation = 0
            for i in range(args.benchmark_count):
                start = int(round(time.time() * 1000))
                out = predict(net, img, pretrained)
                end = int(round(time.time() * 1000))
                estimation_time = (end - start)

                # Loggin
                logger.info(f'\tailia processing estimation time {estimation_time} ms')
                if i != 0:
                    total_time_estimation = total_time_estimation + estimation_time

            logger.info(f'\taverage time estimation {total_time_estimation / (args.benchmark_count - 1)} ms')
        else:
            # inference
            out = predict(net, img, pretrained)

        # save results
        savepath = get_savepath(args.savepath, image_path, ext='.png')
        logger.info(f'saved at : {savepath}')
        plt.imsave(savepath, np.log10(out), cmap='plasma_r')

    logger.info('Script finished successfully.')


def recognize_from_video(net, pretrained):
    capture = webcamera_utils.get_capture(args.video)

    fig = plt.figure(figsize=(4, 2))

    # create video writer if savepath is specified as video format
    if args.savepath != SAVE_IMAGE_PATH:
        fig.canvas.draw()
        im = np.array(fig.canvas.renderer.buffer_rgba())
        f_h, f_w = im.shape[:2]
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

        # inference
        out = predict(net, frame, pretrained)

        plt.axis("off")
        plt.imshow(np.log10(out), cmap='plasma_r')
        fig.canvas.draw()
        im = np.array(fig.canvas.renderer.buffer_rgba())
        im = cv2.cvtColor(im, cv2.COLOR_RGBA2BGR)
        cv2.imshow('frame', im)
        frame_shown = True

        # save results
        if writer is not None:
            writer.write(im)

    capture.release()
    cv2.destroyAllWindows()
    if writer is not None:
        writer.release()

    logger.info('Script finished successfully.')


def main():
    info = {
        'kitti': (
            WEIGHT_KITTI_PATH, MODEL_KITTI_PATH, 'kitti'),
        'kitti-grad': (
            WEIGHT_KITTI_GRAD_PATH, MODEL_KITTI_GRAD_PATH, 'kitti'),
        'nyu': (
            WEIGHT_NYU_PATH, MODEL_NYU_PATH, 'nyu'),
    }
    weight_path, model_path, pretrained = info[args.model_type]

    check_and_download_models(weight_path, model_path, REMOTE_PATH)

    # initialize
    if not args.onnx:
        net = ailia.Net(model_path, weight_path, env_id=args.env_id, memory_mode=args.memory_mode)
    else:
        import onnxruntime
        net = onnxruntime.InferenceSession(weight_path)

    if args.video is not None:
        # video mode
        recognize_from_video(net, pretrained)
    else:
        # image mode
        recognize_from_image(net, pretrained)


if __name__ == '__main__':
    main()
