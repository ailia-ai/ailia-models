import sys
import time

import cv2

import ailia
import numpy as np
import math

from PIL import Image
import tusimple_DUC_util as cityscapes_labels


# import original modules
sys.path.append('../../util')
from arg_utils import get_base_parser, update_parser, get_savepath  # noqa: E402
from model_utils import check_and_download_models  # noqa: E402
from image_utils import imread # noqa: E402
import webcamera_utils  # noqa: E402

# logger
from logging import getLogger   # noqa: E402
logger = getLogger(__name__)


# ======================
# Parameters 1
# ======================
IMAGE_PATH = 'input.png'
SAVE_IMAGE_PATH = 'output.png'

SLEEP_TIME = 0



# ======================
# Arguemnt Parser Config
# ======================
parser = get_base_parser(
    'tusimple-DUC for semantic segmentations.',
    IMAGE_PATH,
    SAVE_IMAGE_PATH,
)

args = update_parser(parser)


# ======================
# Parameters 2
# ======================
MODEL_NAME = 'ResNet101-DUC-12'
WEIGHT_PATH = MODEL_NAME + ".onnx"
MODEL_PATH = WEIGHT_PATH + ".prototxt"
REMOTE_PATH = 'https://storage.googleapis.com/ailia-models/tusimple-DUC/'


def get_palette():
    # get train id to color mappings from file
    trainId2colors = {label.trainId: label.color for label in cityscapes_labels.labels}
    # prepare and return palette
    palette = [0] * 256 * 3
    for trainId in trainId2colors:
        colors = trainId2colors[trainId]
        if trainId == 255:
            colors = (0, 0, 0)
        for i in range(3):
            palette[trainId * 3 + i] = colors[i]
    return palette

def colorize(labels):
    # generate colorized image from output labels and color palette
    result_img = Image.fromarray(labels).convert('P')
    #color pallet setting
    result_img.putpalette(get_palette())
    return np.array(result_img.convert('RGB'))


def preprocess(im,rgb_mean):
    # Convert to float32
    test_img = im.astype(np.float32)
    # Extrapolate image with a small border in order obtain an accurate reshaped image after DUC layer
    test_shape = [im.shape[0],im.shape[1]]
    cell_shapes = [math.ceil(l / 8)*8 for l in test_shape]
    test_img = cv2.copyMakeBorder(test_img, 0, max(0, int(cell_shapes[0]) - im.shape[0]), 0, max(0, int(cell_shapes[1]) - im.shape[1]), cv2.BORDER_CONSTANT, value=rgb_mean)
    test_img = np.transpose(test_img, (2, 0, 1))
    # subtract rbg mean
    for i in range(3):
        test_img[i] -= rgb_mean[i]
    test_img = np.expand_dims(test_img, axis=0)
    return test_img



def predict(net,im,imgs,result_shape):
    # get input and output dimensions
    result_height, result_width = result_shape
    _, _, img_height, img_width = imgs.shape
    # set downsampling rate
    ds_rate = 8
    # set cell width
    cell_width = 2
    # number of output label classes
    label_num = 19
    
    # Perform forward pass
    labels = net.run(imgs)[0].squeeze()


    # re-arrange output
    test_width = int((int(img_width) / ds_rate) * ds_rate)
    test_height = int((int(img_height) / ds_rate) * ds_rate)
    feat_width = int(test_width / ds_rate)
    feat_height = int(test_height / ds_rate)
    labels = labels.reshape((label_num, 4, 4, feat_height, feat_width))
    labels = np.transpose(labels, (0, 3, 1, 4, 2))
    labels = labels.reshape((label_num, int(test_height / cell_width), int(test_width / cell_width)))

    labels = labels[:, :int(img_height / cell_width),:int(img_width / cell_width)]
    labels = np.transpose(labels, [1, 2, 0])
    labels = cv2.resize(labels, (result_width, result_height), interpolation=cv2.INTER_LINEAR)
    labels = np.transpose(labels, [2, 0, 1])
    
    # get softmax output
    softmax = labels
    
    # get classification labels
    results = np.argmax(labels, axis=0).astype(np.uint8)
    raw_labels = results

    # comput confidence score
    confidence = float(np.max(softmax, axis=0).mean())


    # generate segmented image
    result_img = Image.fromarray(colorize(raw_labels)).resize(result_shape[::-1])
    
    # generate blended image
    blended_img = Image.fromarray(cv2.addWeighted(im[:, :, ::-1], 0.5, np.array(result_img), 0.5, 0))

    return confidence, result_img, blended_img, raw_labels



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

        input_data = imread(image_path)
        im = input_data[:, :, ::-1]
        result_shape = [im.shape[0],im.shape[1]]
        rgb_mean = cv2.mean(im)
 
        # inference
        logger.info('Start inference...')
        if args.benchmark:
            logger.info('BENCHMARK mode')
            total_time = 0
            for i in range(args.benchmark_count):
                start = int(round(time.time() * 1000))
                pre = preprocess(im,rgb_mean)
                conf,result_img,blended_img,raw = predict(net,im,pre,result_shape)
                end = int(round(time.time() * 1000))
                logger.info(f'\tailia processing time {end - start} ms')
                if i != 0:
                    total_time = total_time + (end - start)
            logger.info(f'\taverage time {total_time / (args.benchmark_count-1)} ms')
        else:
            pre = preprocess(im,rgb_mean)
            conf,result_img,blended_img,raw = predict(net,im,pre,result_shape)
        print(args.savepath)
        result_img.save(args.savepath)

        savepath = get_savepath(args.savepath, image_path)
        logger.info(f'saved at : {savepath}')
    logger.info('Script finished successfully.')

def recognize_from_video():
    # net initialize
    net = ailia.Net(MODEL_PATH, WEIGHT_PATH, env_id=args.env_id, memory_mode=args.memory_mode)

    capture = webcamera_utils.get_capture(args.video)

    # create video writer if savepath is specified as video format
    if args.savepath != SAVE_IMAGE_PATH:
        logger.warning(
            'currently, video results cannot be output correctly...'
        )
        f_h = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        f_w = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        writer = webcamera_utils.get_writer(args.savepath, f_h, f_w)
    else:
        writer = None

    while(True):
        ret, frame = capture.read()
        if (cv2.waitKey(1) & 0xFF == ord('q')) or not ret:
            break

        im = frame[:, :, ::-1]
        result_shape = [im.shape[0],im.shape[1]]
        rgb_mean = cv2.mean(im)
 
        # inference
        pre = preprocess(im,rgb_mean)
        conf,result_img,blended_img,raw = predict(net,im,pre,result_shape)

        cv2.imshow('frame', np.array(result_img))
        frame_shown = True
        time.sleep(SLEEP_TIME)

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
    check_and_download_models(WEIGHT_PATH, MODEL_PATH, REMOTE_PATH)

    if args.video is not None:
        # video mode
        recognize_from_video()
    else:
        # image mode
        recognize_from_image()


if __name__ == '__main__':
    main()
