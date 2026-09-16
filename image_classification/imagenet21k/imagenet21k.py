import platform
import sys
import time
import numpy as np
import cv2

import ailia

# import original modules
sys.path.append('../../util')
# logger
from logging import getLogger  # noqa: E402

import webcamera_utils  # noqa: E402
from classifier_utils import *  # noqa: E402
from image_utils import imread, load_image, normalize_image  # noqa: E402
from model_utils import check_and_download_models  # noqa: E402
from arg_utils import get_base_parser, update_parser  # noqa: E402

logger = getLogger(__name__)

from imagenet21k_util import ImageNet21kSemanticSoftmax
from scipy.special import softmax

semantic_softmax_processor = ImageNet21kSemanticSoftmax()

# ======================
# Parameters
# ======================
REMOTE_PATH = "https://storage.googleapis.com/ailia-models/imagenet21k/"

IMAGE_PATH = 'input.jpg'
THRESHOLD = 0.3
SLEEP_TIME = 0  # for video mode

MODEL_LISTS = ['resnet50','mixer','vit','mobilenet']

# ======================
# Arguemnt Parser Config
# ======================
parser = get_base_parser(
    'imagenet21k', IMAGE_PATH, None, fp16_support=False
)

parser.add_argument(
    '-th', '--threshold',
    default=THRESHOLD, type=float,
    help='classifier threshold for imagenet21k. (default: '+str(THRESHOLD)+')'
)

parser.add_argument('-a','--arch', type=str, default="vit",
    help='model layer number lists: ' + ' | '.join(MODEL_LISTS)
)

args = update_parser(parser)

WEIGHT_PATH = ""
MODEL_PATH = ""

if args.arch == "vit":
    model_name = "vit_base_patch16_224_miil_in21k"
elif args.arch == "mixer":
    model_name = "mixer_b16_224_miil_in21k"
elif args.arch == "resnet50":
    model_name = "resnet50"
elif args.arch == "mobilenet":
    model_name = "mobilenetv3_large_100"
MODEL_PATH  = model_name + '.onnx.prototxt'
WEIGHT_PATH = model_name + '.onnx'

def post_process(logits):
    labels = []
    probs  = []
    idx = []
    semantic_logit_list = semantic_softmax_processor.split_logits_to_semantic_logits(logits)
    
    # scanning hirarchy_level_list
    for i in range(len(semantic_logit_list)):
        logits_i = semantic_logit_list[i]
    
        # generate probs
        probabilities = softmax(logits_i[0], axis=0)
    
        top1_id   = [np.argmax(probabilities)]
        top1_prob = probabilities[top1_id]
        
    
        if top1_prob > args.threshold:
            top_class_number = semantic_softmax_processor.hierarchy_indices_list[i][top1_id[0]]
            top_class_name = semantic_softmax_processor.class_list[top_class_number]
            top_class_description = semantic_softmax_processor.class_description[top_class_name]
            labels.append(top_class_description)
            probs.append(top1_prob)
            idx.append(top_class_number)
    return [np.array(idx), np.array(labels), np.array(probs)]

def plot_results(input_image,results, top_k=MAX_CLASS_COUNT, logging=True):
    x = RECT_MARGIN
    y = RECT_MARGIN
    w = RECT_WIDTH
    h = RECT_HEIGHT

    sort_idx    = [i[0] for i in results[2]]
    sort_idx = np.argsort(sort_idx)[::-1]
 
    idx    = results[0][sort_idx]
    labels = results[1][sort_idx]
    probs  = results[2][sort_idx] * 100
    top_k = len(idx)



    if logging:
        print('==============================================================')
        print(f'class_count={top_k}')
    #for idx in range(top_k):
    for i in range(top_k):
        if logging:
            print(f'+ idx={i}')
            print(f'  category={idx[i]}['
                  f'{labels[i]} ]')
            print(f'  prob={probs[i][0]}')

        
        text = f'category={idx[i]}[{labels[i]} ] prob={probs[i]}'

        color = hsv_to_rgb(256 * probs[i] / (len(labels)+1), 128, 255)

        cv2.rectangle(input_image, (x, y), (x + w, y + h), color, thickness=-1)
        text_position = (x+4, y+int(RECT_HEIGHT/2)+4)

        color = (0,0,0)
        fontScale = 0.5

        cv2.putText(
            input_image,
            text,
            text_position,
            cv2.FONT_HERSHEY_SIMPLEX,
            fontScale,
            color,
            1
        )

        y=y + h + RECT_MARGIN
    return input_image



def print_results(results, top_k=MAX_CLASS_COUNT):
    sort_idx    = [i[0] for i in results[2]]
    sort_idx = np.argsort(sort_idx)[::-1]
 
    idx    = results[0][sort_idx]
    labels = results[1][sort_idx]
    probs  = results[2][sort_idx] * 100
    top_k = len(idx)

    print('==============================================================')
    print(f'class_count={top_k}')
    for i in range(top_k):
        print(f'+ idx={i}')
        print(f'  category={idx[i]}['
              f'{labels[i]} ]')
        print(f'  prob={probs[i][0]}')


# ======================
# Main functions
# ======================
def recognize_from_image(net):
    # input image loop
    for image_path in args.input:
        # prepare input data
        logger.info(image_path)
        input_data = imread(image_path)
        input_data = cv2.cvtColor(input_data,cv2.COLOR_BGR2RGB)
        input_data = cv2.resize(input_data,(256,256))
        input_data = input_data[16:240,16:240]

        mean = np.array([0,0,0])
        std = np.array([1,1,1])
        input_data = input_data / 255.0
        for i in range(3):
            input_data[:, :, i] = (input_data[:, :, i] - mean[i]) / std[i]

        input_data = input_data.transpose((2,0,1))
        input_data = np.expand_dims(input_data,0)

        # inference
        logger.info('Start inference...')
        if args.benchmark:
            logger.info('BENCHMARK mode')
            for i in range(args.benchmark_count):
                start = int(round(time.time() * 1000))
                logits = net.run(input_data)[0]
                end = int(round(time.time() * 1000))
                logger.info(f'\tailia processing time {end - start} ms')
        else:
            logits = net.run(input_data.astype(np.float32))[0]

        result = post_process(logits)
        print_results(result)
        frame =imread(image_path)

    logger.info('Script finished successfully.')


def recognize_from_video(net):
    capture = webcamera_utils.get_capture(args.video)

    # create video writer if savepath is specified as video format
    if args.savepath is not None:
        f_h = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        f_w = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        writer = webcamera_utils.get_writer(args.savepath, f_h, f_w)
    else:
        writer = None

    frame_shown = False
    while(True):
        ret, frame = capture.read()
        if (cv2.waitKey(1) & 0xFF == ord('q')) or not ret:
            break
        if frame_shown and cv2.getWindowProperty('frame', cv2.WND_PROP_VISIBLE) == 0:
            break

        # prepare input data
        input_data = frame
        input_data = cv2.resize(input_data,(256,256))
        input_data = input_data[15:241,15:241]
        input_data = normalize_image(input_data,normalize_type='ImageNet')
        input_data = input_data.transpose((2,0,1))
        input_data = np.expand_dims(input_data,0)

 
        # inference
        logits = net.run(input_data)[0]

        # get result
        result = post_process(logits)

        plot_results(frame, result)


        cv2.imshow('frame', frame)
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

    # net initialize
    net = ailia.Net(MODEL_PATH, WEIGHT_PATH, env_id=args.env_id, memory_mode=args.memory_mode)

    if args.video is not None:
        # video mode
        recognize_from_video(net)
    else:
        # image mode
        recognize_from_image(net)


if __name__ == '__main__':
    main()
