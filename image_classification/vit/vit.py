import platform
import sys
import time

import cv2
import matplotlib
import numpy as np

matplotlib.use('Agg')
import ailia
import matplotlib.pyplot as plt

import vit_labels

# import original modules
sys.path.append('../../util')
# logger
from logging import getLogger  # noqa: E402

import webcamera_utils  # noqa: E402
from classifier_utils import plot_results, print_results  # noqa: E402
from image_utils import imread, load_image  # noqa: E402
from model_utils import check_and_download_models  # noqa: E402
from arg_utils import get_base_parser, update_parser  # noqa: E402

logger = getLogger(__name__)


# ======================
# PARAMETERS 1
# ======================
IMAGE_OR_VIDEO_PATH = 'input.jpg'  # input.mp4
SAVE_IMAGE_OR_VIDEO_PATH = 'output.png'  # output.mp4
IMAGE_HEIGHT = 224
IMAGE_WIDTH = 224


# ======================
# Argument Parser Config
# ======================
parser = get_base_parser(
    'Vision Transformer',
    IMAGE_OR_VIDEO_PATH,
    SAVE_IMAGE_OR_VIDEO_PATH,
    fp16_support=False
)
parser.add_argument(
    '-m', '--model', metavar='MODEL',
    default='B_16', choices=['B_16'],
    help='The input model path.' +
         'you can set B_16 to select ViT-B_16'
)
args = update_parser(parser)


# ==========================
# MODEL AND OTHER PARAMETERS
# ==========================
MODEL_PATH = 'ViT-' + args.model + '-224.onnx.prototxt'
WEIGHT_PATH = 'ViT-' + args.model + '-224.onnx'
REMOTE_PATH = 'https://storage.googleapis.com/ailia-models/vit/'

MAX_CLASS_COUNT = 5
SLEEP_TIME = 0  # for web cam mode
FIGURE_HEIGHT = 1000
FIGURE_WIDTH = 1500


# ======================
# Sub functions
# ======================
def prep_input(image):
    # preprocessing
    input_data = cv2.resize(image, (224, 224))  # resize to 224x224
    input_data = np.array(input_data).astype(np.float32)  # cast to float
    input_data = (((input_data / 255) - 0.5) / 0.5)  # normalization
    input_data = input_data.transpose(2, 0, 1)  # CHW
    input_data = input_data[np.newaxis, :, :, :]  # BCHW
    # return preprocessed image
    return input_data


def calc_attention_map(att_mat, height_org=224, width_org=224):
    # Average the attention weights across all heads.
    att_mat = np.mean(att_mat, axis=1)
    # To account for residual connections, we add an identity matrix to the
    # attention matrix and re-normalize the weights.
    residual_att = np.eye(np.shape(att_mat)[1])
    aug_att_mat = att_mat + residual_att
    aug_att_mat = aug_att_mat / np.sum(aug_att_mat, axis=-1, keepdims=True)
    # Recursively multiply the weight matrices
    joint_attentions = np.zeros(np.shape(aug_att_mat))
    joint_attentions[0] = aug_att_mat[0]
    for n in range(1, len(aug_att_mat)):
        joint_attentions[n] = aug_att_mat[n] @ joint_attentions[n-1]
    # Attention from the output token to the input space.
    v = joint_attentions[-1]
    grid_size = int(np.sqrt(np.shape(aug_att_mat)[-1]))
    mask = v[0, 1:].reshape(grid_size, grid_size)
    mask = mask / mask.max()
    mask = cv2.resize(mask, (width_org, height_org))[..., np.newaxis]
    # return mask by attention map
    return mask


import warnings  # provisional...

warnings.simplefilter('ignore', DeprecationWarning)  # provisional...
def visualize_result(image, mask, probs, labels):
    # adjust for output
    labels_and_probs = []
    for i in range(len(probs)):
        labels_and_probs.append('%.3f : %s' % (probs[i], labels[i]))
        if (len(labels_and_probs[-1]) > 50):
            labels_and_probs[-1] = labels_and_probs[-1][:50] + '...'
    # plot and write to image
    plt.figure(figsize=(FIGURE_WIDTH/100, FIGURE_HEIGHT/100), dpi=100)
    plt.rcParams['font.size'] = 14
    plt.tight_layout()
    # show input image
    plt.subplot(2, 2, 1)
    plt.imshow(image)
    plt.title('Input Image')
    # show attention map
    plt.subplot(2, 2, 2)
    plt.imshow(mask[:, :, 0], clim=[0, 1])
    plt.title('Attention Map (color scale 0-1)')
    # show prediction by bar graph
    plt.subplot(2, 2, 3)
    plt.barh(np.arange(len(probs)), probs[::-1])
    plt.gca().set_yticks(np.arange(5))
    plt.gca().set_yticklabels(labels_and_probs[::-1], 
                              horizontalalignment='left', fontsize=12)
    plt.gca().tick_params(axis='y', direction='in', pad=-15)
    plt.grid()
    plt.xlim([0, 1.05])
    plt.title('Prediction Label')
    # show masked image by attention map
    plt.subplot(2, 2, 4)
    plt.imshow((mask**2 * image).astype("uint8"))  # emphasize by square
    plt.title('Attention Map and Image')
    # draw and write
    plt.draw()
    # tostring_rgb was removed in matplotlib 3.10, use buffer_rgba instead
    # (keep 3 channels, cv2.VideoWriter silently drops RGBA frames)
    image_figure = np.asarray(plt.gcf().canvas.buffer_rgba())[:, :, :3].copy()
    image_figure = image_figure.reshape(FIGURE_HEIGHT,
                                        FIGURE_WIDTH, -1)
    # close figure
    plt.close()
    # return figure image
    return image_figure


# ======================
# Main functions
# ======================
def recognize_from_image():
    # net initialize
    classifier = ailia.Net(MODEL_PATH, WEIGHT_PATH, env_id=args.env_id)
    # adjust prediction label
    labels = np.array(vit_labels.imagenet_category)

    # input image loop
    for image_path in args.input:
        # prepare input data
        logger.info(image_path)
        image = imread(image_path)[:, :, ::-1]
        input_data = prep_input(image)

        # inference
        logger.info('Start inference...')
        if args.benchmark:
            logger.info('BENCHMARK mode')
            for i in range(args.benchmark_count):
                start = int(round(time.time() * 1000))
                output = classifier.run(input_data)
                end = int(round(time.time() * 1000))
                logger.info(f'\tailia processing time {end - start} ms')
        else:
            output = classifier.run(input_data)

        # pick up logits and attention map
        logits = output[0]
        att_mat = np.array(output[1:]).squeeze(1)
        # get prediction label and its score
        probs = np.exp(logits[0])
        probs = probs / np.sum(probs)
        topN = np.argsort(-probs)[:MAX_CLASS_COUNT]
        print('\n  Prediction Label and Attention Map!')
        for idx in topN:
            print(f'    {probs[idx]:.5f} : {labels[idx]}')
        print()

        # calculate attention map
        mask = calc_attention_map(att_mat, height_org=np.shape(image)[0],
                                           width_org=np.shape(image)[1])
        # visualize result
        image_figure = visualize_result(image, mask, probs[topN], labels[topN])
        # save visualization
        logger.info(f'saved at : {args.savepath}')
        cv2.imwrite(args.savepath, image_figure[..., ::-1])

    logger.info('Script finished successfully.')


def recognize_from_video():
    # net initialize
    classifier = ailia.Net(MODEL_PATH, WEIGHT_PATH, env_id=args.env_id)
    # adjust prediction label
    labels = np.array(vit_labels.imagenet_category)

    # capture video
    capture = webcamera_utils.get_capture(args.video)
    # create video writer if savepath is specified as video format
    if (args.savepath is not None) & (args.savepath.split('.')[-1] == 'mp4'):
        writer = webcamera_utils.get_writer(args.savepath,
                                            FIGURE_HEIGHT, FIGURE_WIDTH)
    else:
        writer = None

    frame_shown = False
    while(True):
        # read frame
        ret, frame = capture.read()
        if (cv2.waitKey(1) & 0xFF == ord('q')) or not ret:
            break
        if frame_shown and cv2.getWindowProperty('frame', cv2.WND_PROP_VISIBLE) == 0:
            break

        # preprocessing
        frame = frame[..., ::-1]  # BGR2RGB
        input_data = prep_input(frame)

        # inference
        output = classifier.run(input_data)

        # pick up logits and attention map
        logits = output[0]
        att_mat = np.array(output[1:]).squeeze(1)
        # get prediction label and its score
        probs = np.exp(logits[0])
        probs = probs / np.sum(probs)
        topN = np.argsort(-probs)[:5]

        # calculate attention map
        mask = calc_attention_map(att_mat, height_org=np.shape(frame)[0], 
                                           width_org=np.shape(frame)[1])
        # visualize result
        frame_figure = visualize_result(frame, mask, probs[topN], labels[topN])

        # view result figure
        cv2.imshow('frame', frame_figure[..., ::-1])
        frame_shown = True
        time.sleep(SLEEP_TIME)
        # save result
        if writer is not None:
            writer.write(frame_figure[..., ::-1])

    capture.release()
    cv2.destroyAllWindows()
    if writer is not None:
        writer.release()
        # save visualization
        logger.info(f'saved at : {args.savepath}')

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
