import sys, os
import time

import numpy as np
import cv2
from matplotlib import pyplot as plt

import ailia

# import original modules
sys.path.append('../../util')
from arg_utils import get_base_parser, update_parser, get_savepath  # noqa: E402
from model_utils import check_and_download_models  # noqa: E402
from detector_utils import load_image  # noqa: E402C
from webcamera_utils import get_capture, get_writer  # noqa: E402

# logger
from logging import getLogger   # noqa: E402
logger = getLogger(__name__)


# ======================
# Parameters
# ======================

WEIGHT_ALEXNET_PATH = 'AlexNet.onnx'
WEIGHT_VGG_PATH = 'VGG.onnx'
WEIGHT_VGG_DECODER_PATH = 'VGG_DECODER.onnx'
WEIGHT_RESNET50_PATH = 'ResNet50.onnx'
WEIGHT_RESNET101_PATH = 'ResNet101.onnx'
WEIGHT_CSRNET_PATH = 'CSRNet.onnx'
WEIGHT_SANET_PATH = 'SANet.onnx'
MODEL_ALEXNET_PATH = 'AlexNet.onnx.prototxt'
MODEL_VGG_PATH = 'VGG.onnx.prototxt'
MODEL_VGG_DECODER_PATH = 'VGG_DECODER.onnx.prototxt'
MODEL_RESNET50_PATH = 'ResNet50.onnx.prototxt'
MODEL_RESNET101_PATH = 'ResNet101.onnx.prototxt'
MODEL_CSRNET_PATH = 'CSRNet.onnx.prototxt'
MODEL_SANET_PATH = 'SANet.onnx.prototxt'
REMOTE_PATH = \
    'https://storage.googleapis.com/ailia-models/c-3-framework/'

IMAGE_PATH = 'demo.jpg'
SAVE_IMAGE_PATH = 'output.png'

MAX_IMAGE_WIDTH = 1024  # you can increase this value if you have an enough vram

# ======================
# Arguemnt Parser Config
# ======================

parser = get_base_parser(
    'C-3-Framework model', IMAGE_PATH, SAVE_IMAGE_PATH
)
parser.add_argument(
    '-m', '--model', type=str, default='resnet50',
    choices=(
        'alexnet', 'vgg', 'vgg_decoder', 'resnet50', 'resnet101', 'csrnet', 'sanet',
    ),
    help='choice model'
)
args = update_parser(parser)


# ======================
# Secondaty Functions
# ======================


def preprocess(img):
    img = img.astype(np.float32) / 255

    # normalize
    mean = np.array([0.452016860247, 0.447249650955, 0.431981861591])
    std = np.array([0.23242045939, 0.224925786257, 0.221840232611])
    mean = np.float64(mean.reshape(1, -1))
    stdinv = 1 / np.float64(std.reshape(1, -1))
    cv2.subtract(img, mean, img)  # inplace
    cv2.multiply(img, stdinv, img)  # inplace

    img = img.transpose(2, 0, 1)
    img = np.expand_dims(img, axis=0)

    return img


# ======================
# Main functions
# ======================


def predict(img, net):
    img = preprocess(img)

    net.set_input_shape(img.shape)
    pred_map = net.predict({'imgs': img})[0]
    pred_map = pred_map[0, 0, :, :]

    return pred_map


def recognize_from_image(filename, net):
    # prepare input data
    img = load_image(filename)
    logger.info(f'input image shape: {img.shape}')

    img = cv2.cvtColor(img, cv2.COLOR_BGRA2RGB)

    # inference
    logger.info('Start inference...')
    if args.benchmark:
        logger.info('BENCHMARK mode')
        for i in range(5):
            start = int(round(time.time() * 1000))
            pred_map = predict(img, net)
            end = int(round(time.time() * 1000))
            logger.info(f'\tailia processing time {end - start} ms')
    else:
        pred_map = predict(img, net)

    pred = np.sum(pred_map) / 100.0
    pred_map = pred_map / np.max(pred_map + 1e-20)

    logger.info("predict:"+str(pred))

    # plot result
    pred_frame = plt.gca()
    plt.imshow(pred_map, 'jet')
    pred_frame.axes.get_yaxis().set_visible(False)
    pred_frame.axes.get_xaxis().set_visible(False)
    pred_frame.spines['top'].set_visible(False)
    pred_frame.spines['bottom'].set_visible(False)
    pred_frame.spines['left'].set_visible(False)
    pred_frame.spines['right'].set_visible(False)
    savepath = get_savepath(args.savepath, filename)
    logger.info(f'saved at : {savepath}')
    plt.savefig(savepath, bbox_inches='tight', pad_inches=0, dpi=150)
    plt.close()

    logger.info('Script finished successfully.')


def recognize_from_video(video, net):
    capture = get_capture(video)

    from threading import Event
    fin = Event()

    def handle_close(evt):
        fin.set()

    def press(event):
        if event.key == 'q':
            fin.set()

    fig = plt.figure()
    fig.canvas.mpl_connect('close_event', handle_close)
    fig.canvas.mpl_connect('key_press_event', press)

    f_h = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    f_w = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))

    if f_w>MAX_IMAGE_WIDTH:
        f_w, f_h = int(MAX_IMAGE_WIDTH), int(MAX_IMAGE_WIDTH*f_h/f_w)

    # create video writer if savepath is specified as video format
    if args.savepath != SAVE_IMAGE_PATH:
        writer = get_writer(args.savepath, f_h, f_w)
    else:
        writer = None

    while not fin.is_set():
        ret, frame = capture.read()
        if not ret:
            continue
            
        frame = cv2.resize(frame, (f_w,f_h))
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        pred_map = predict(frame, net)
        pred = np.sum(pred_map) / 100.0
        pred_map = pred_map / np.max(pred_map + 1e-20)

        logger.info("predict:"+str(pred))

        # show
        pred_frame = plt.gca()
        plt.imshow(pred_map, 'jet')
        pred_frame.axes.get_yaxis().set_visible(False)
        pred_frame.axes.get_xaxis().set_visible(False)
        pred_frame.spines['top'].set_visible(False)
        pred_frame.spines['bottom'].set_visible(False)
        pred_frame.spines['left'].set_visible(False)
        pred_frame.spines['right'].set_visible(False)
        plt.pause(0.001)  # pause a bit so that plots are updated

        # save results
        if writer is not None:
            frame = cv2.cvtColor(pred_map * 255,cv2.COLOR_GRAY2BGR).astype(np.uint8)
            writer.write(frame)

    capture.release()
    if writer is not None:
        writer.release()
    logger.info('Script finished successfully.')


def main():
    dic_model = {
        'alexnet': (WEIGHT_ALEXNET_PATH, MODEL_ALEXNET_PATH),
        'vgg': (WEIGHT_VGG_PATH, MODEL_VGG_PATH),
        'vgg_decoder': (WEIGHT_VGG_DECODER_PATH, MODEL_VGG_DECODER_PATH),
        'resnet50': (WEIGHT_RESNET50_PATH, MODEL_RESNET50_PATH),
        'resnet101': (WEIGHT_RESNET101_PATH, MODEL_RESNET101_PATH),
        'csrnet': (WEIGHT_CSRNET_PATH, MODEL_CSRNET_PATH),
        'sanet': (WEIGHT_SANET_PATH, MODEL_SANET_PATH),
    }
    weight_path, model_path = dic_model[args.model]

    # model files check and download
    check_and_download_models(weight_path, model_path, REMOTE_PATH)

    # initialize
    net = ailia.Net(model_path, weight_path, env_id=args.env_id, memory_mode=args.memory_mode)

    if args.video is not None:
        recognize_from_video(args.video, net)
    else:
        for input_name in args.input:
          recognize_from_image(input_name, net)


if __name__ == '__main__':
    main()
