import glob
import sys
import time

import ailia
import cv2
import numpy as np
import onnxruntime

sys.path.append('../../util')
# logger
from logging import getLogger  # noqa: E402

import webcamera_utils  # noqa: E402
from image_utils import imread  # noqa: E402
from model_utils import check_and_download_models  # noqa: E402
from arg_utils import get_base_parser, get_savepath, update_parser  # noqa: E402

logger = getLogger(__name__)

# ======================
# Parameters 1
# ======================

SAVE_IMAGE_PATH = 'output.jpg'
INPUT_IMAGE_PATH = 'input.jpg'

MODEL_LISTS = ['culane','tusimple']

# ======================
# Argument Parser Config
# ======================

parser = get_base_parser(
    'lane_detection',
    INPUT_IMAGE_PATH,
    SAVE_IMAGE_PATH,
)
parser.add_argument(
    '-i','--input', type=str,
    default=INPUT_IMAGE_PATH,
    help='The input image for input image.'
)
parser.add_argument(
    '-v', '--video', type=str,
    help='The input video for pole detection.'
)
parser.add_argument(
    '-o', '--onnx', action='store_true',
    help="Option to use onnxrutime to run or not."
)
parser.add_argument(
    '-a', '--arch', metavar='ARCH',
    default='culane', choices=MODEL_LISTS,
    help='model lists: ' + ' | '.join(MODEL_LISTS)
)
args = update_parser(parser)

# # ======================
# # Select model
# # ======================

if args.arch=="culane":
    WEIGHT_PATH = 'culane_18.onnx'
    MODEL_PATH = 'culane_18.onnx.prototxt'
else:
    WEIGHT_PATH = 'tusimple_18.onnx'
    MODEL_PATH = 'tusimple_18.onnx.prototxt'

REMOTE_PATH = 'https://storage.googleapis.com/ailia-models/ultra-fast-lane-detection/'

INPUT_HEIGHT = 288
INPUT_WIDTH = 800

# # ======================
# # Main functions
# # ======================

# mean = np.array([0.485, 0.456, 0.406])
# std = np.array([0.229, 0.224, 0.225])
mean = np.array([0.406, 0.456, 0.485]) # this is correct
std = np.array([0.225, 0.224, 0.229])

def preprocessing(img):
    original_img,img = webcamera_utils.adjust_frame_size(img, INPUT_HEIGHT, INPUT_WIDTH)
    
    input_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    input_img = input_img / 255.0
    input_img = input_img - mean / std    
    input_img = input_img.transpose(2, 0, 1)
    
    return img, np.expand_dims(input_img, 0).astype(np.float32)

culane_row_anchor = [121, 131, 141, 150, 160, 170, 180, 189, 199, 209, 219, 228, 238, 248, 258, 267, 277, 287]
tusimple_row_anchor = [ 64,  68,  72,  76,  80,  84,  88,  92,  96, 100, 104, 108, 112,
            116, 120, 124, 128, 132, 136, 140, 144, 148, 152, 156, 160, 164,
            168, 172, 176, 180, 184, 188, 192, 196, 200, 204, 208, 212, 216,
            220, 224, 228, 232, 236, 240, 244, 248, 252, 256, 260, 264, 268,
            272, 276, 280, 284]

def postprocessing(vis,y):
    if args.arch=="culane":
        row_anchor = culane_row_anchor
        griding_num = 200
        cls_num_per_lane = 18
    else:
        row_anchor = tusimple_row_anchor
        griding_num = 100
        cls_num_per_lane = 56

    col_sample = np.linspace(0, INPUT_WIDTH - 1, griding_num)
    col_sample_w = col_sample[1] - col_sample[0]
    img_w, img_h = INPUT_WIDTH, INPUT_HEIGHT

    out_j = y
    out_j = out_j[:, ::-1, :]
    prob = scipy.special.softmax(out_j[:-1, :, :], axis=0)

    idx = np.arange(griding_num) + 1
    idx = idx.reshape(-1, 1, 1)
    loc = np.sum(prob * idx, axis=0)
    out_j = np.argmax(out_j, axis=0)
    loc[out_j == griding_num] = 0
    out_j = loc

    # import pdb; pdb.set_trace()
    for i in range(out_j.shape[1]):
        if np.sum(out_j[:, i] != 0) > 2:
            for k in range(out_j.shape[0]):
                if out_j[k, i] > 0:
                    ppp = (int(out_j[k, i] * col_sample_w * img_w / INPUT_WIDTH) - 1, int(img_h * (row_anchor[cls_num_per_lane-1-k]/INPUT_HEIGHT)) - 1 )
                    cv2.circle(vis,ppp,5,(0,255,0),-1)

import scipy.special
import tqdm


def recognize_from_image(net):
    # input image loop
    for image_path in args.input:
        img = imread(image_path)

        if args.onnx:
            input_name = 'input.1'
            output_name = '200'

        img, input_tensor = preprocessing(img)

        logger.info('Start inference...')
        if args.benchmark:
            logger.info('BENCHMARK mode')
            for i in range(args.benchmark_count):
                start = int(round(time.time() * 1000))
                if args.onnx:
                    y = net.run( [output_name], { input_name : input_tensor })[0][0]
                else:
                    y = net.run(input_tensor)[0][0]
                end = int(round(time.time() * 1000))
                logger.info(f'\tailia processing time {end - start} ms')
        else:
            if args.onnx:
                y = net.run( [output_name], { input_name : input_tensor })[0][0]
            else:
                y = net.run(input_tensor)[0][0]

        postprocessing(img,y)

        savepath = get_savepath(args.savepath, image_path)
        logger.info(f'saved at : {savepath}')
        cv2.imwrite(savepath, img)

    logger.info('Script finished successfully.')

def recognize_from_video(net):

    capture = webcamera_utils.get_capture(args.video)

    # create video writer if savepath is specified as video format
    if args.savepath != SAVE_IMAGE_PATH:
        f_h = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        f_w = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        writer = webcamera_utils.get_writer(args.savepath, INPUT_HEIGHT, INPUT_WIDTH)
    else:
        writer = None

    if args.onnx:
        input_name = 'input.1'
        output_name = '200'
    
    
    frame_shown = False
    while (True):
        ret, frame = capture.read()
        if (cv2.waitKey(1) & 0xFF == ord('q')) or not ret:
            break
        if frame_shown and cv2.getWindowProperty('output', cv2.WND_PROP_VISIBLE) == 0:
            break

        input_img = frame
        img, input_tensor = preprocessing(input_img)

        if args.onnx:
            y = net.run( [output_name], { input_name : input_tensor })[0][0]
        else:
            y = net.run(input_tensor)[0][0]

        postprocessing(img,y)
    
        cv2.imshow('output', img)
        frame_shown = True
        cv2.waitKey(3)

        # save results
        if writer is not None:
            writer.write(img)

    capture.release()
    cv2.destroyAllWindows()
    if writer is not None:
        writer.release()

    logger.info('Script finished successfully.')
  
def main():

    # model files check and download
    check_and_download_models(WEIGHT_PATH, MODEL_PATH, REMOTE_PATH)

    # net initialize
    if args.onnx:
        import onnxruntime
        net = onnxruntime.InferenceSession(WEIGHT_PATH)
    else:
        logger.info(f'env_id: {args.env_id}')
        net = ailia.Net(MODEL_PATH, WEIGHT_PATH, env_id=args.env_id, memory_mode=args.memory_mode)
    

    if args.video is not None:
        # video mode
        recognize_from_video(net)
    else:
        # image mode
        recognize_from_image(net)

if __name__ == '__main__':
    main()
