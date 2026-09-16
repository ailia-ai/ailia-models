import sys
import time

import numpy as np
import cv2
from PIL import Image

import ailia

# import original modules
sys.path.append('../../util')
from arg_utils import get_base_parser, update_parser, get_savepath  # noqa
from model_utils import check_and_download_models  # noqa
from detector_utils import load_image  # noqa
from image_utils import normalize_image  # noqa
from math_utils import softmax  # noqa
from webcamera_utils import get_capture, get_writer  # noqa
# logger
from logging import getLogger  # noqa

logger = getLogger(__name__)

# ======================
# Parameters
# ======================

WEIGHT_PP_LiteSeg_T_CITYSCAPE_PATH = 'pp_liteseg_stdc1_cityscapes_1024x512_scale1.0_model.onnx'
MODEL_PP_LiteSeg_T_CITYSCAPE_PATH = 'pp_liteseg_stdc1_cityscapes_1024x512_scale1.0_model.onnx.prototxt'
WEIGHT_PP_LiteSeg_B_CITYSCAPE_PATH = 'pp_liteseg_stdc2_cityscapes_1024x512_scale1.0_model.onnx'
MODEL_PP_LiteSeg_B_CITYSCAPE_PATH = 'pp_liteseg_stdc2_cityscapes_1024x512_scale1.0_model.onnx.prototxt'
WEIGHT_PP_LiteSeg_T_CAMVID_PATH = 'pp_liteseg_stdc1_camvid_960x720_10k_model.onnx'
MODEL_PP_LiteSeg_T_CAMVID_PATH = 'pp_liteseg_stdc1_camvid_960x720_10k_model.onnx.prototxt'
WEIGHT_PP_LiteSeg_B_CAMVID_PATH = 'pp_liteseg_stdc2_camvid_960x720_10k_model.onnx'
MODEL_PP_LiteSeg_B_CAMVID_PATH = 'pp_liteseg_stdc2_camvid_960x720_10k_model.onnx.prototxt'

REMOTE_PATH = 'https://storage.googleapis.com/ailia-models/pp_liteseg/'

IMAGE_PATH = 'demo.png'
SAVE_IMAGE_PATH = 'output.png'

IMAGE_CITYSCAPE_HEIGHT = 512
IMAGE_CITYSCAPE_WIDTH = 1024
IMAGE_CAMVID_HEIGHT = 768
IMAGE_CAMVID_WIDTH = 1024

# ======================
# Arguemnt Parser Config
# ======================

parser = get_base_parser(
    'PaddleSeg', IMAGE_PATH, SAVE_IMAGE_PATH
)
parser.add_argument(
    '-p', '--pseudo',
    action='store_true',
    help='Save pseudo color prediction.'
)
parser.add_argument(
    '--aug_pred',
    action='store_true',
    help='Whether to use flip augment for prediction.'
)
parser.add_argument(
    '-m', '--model_type',
    default='stdc1', choices=('stdc1', 'stdc2'),
    help='model type'
)
parser.add_argument(
    '-d', '--dataset',
    default='cityscapes', choices=('cityscapes', 'camvid'),
    help='dataset'
)
args = update_parser(parser)


# ======================
# Secondaty Functions
# ======================

def get_color_map(num_classes):
    num_classes += 1
    cm = num_classes * [0, 0, 0]
    for i in range(0, num_classes):
        j = 0
        lab = i
        while lab:
            cm[i * 3] |= (((lab >> 0) & 1) << (7 - j))
            cm[i * 3 + 1] |= (((lab >> 1) & 1) << (7 - j))
            cm[i * 3 + 2] |= (((lab >> 2) & 1) << (7 - j))
            j += 1
            lab >>= 3

    cm = cm[3:]

    return cm


color_map = get_color_map(256)


def visualize(img, result, weight=0.6):
    cm = [
        color_map[i:i + 3] for i in range(0, len(color_map), 3)
    ]
    cm = np.array(cm).astype("uint8")

    # Use OpenCV LUT for color mapping
    c1 = cv2.LUT(result, cm[:, 0])
    c2 = cv2.LUT(result, cm[:, 1])
    c3 = cv2.LUT(result, cm[:, 2])
    pseudo_img = np.dstack((c1, c2, c3))

    vis_result = cv2.addWeighted(img, weight, pseudo_img, 1 - weight, 0)

    return vis_result


def get_pseudo(pred):
    pred_mask = Image.fromarray(pred.astype(np.uint8), mode='P')
    pred_mask.putpalette(color_map)

    pred_mask = np.asarray(pred_mask.convert('RGB'))
    pred_mask = cv2.cvtColor(pred_mask, cv2.COLOR_RGB2BGR)

    return pred_mask


# ======================
# Main functions
# ======================

def preprocess(img, image_shape):
    h, w = image_shape
    im_h, im_w, _ = img.shape

    # adaptive_resize
    scale = min(h / im_h, w / im_w)
    ow, oh = int(im_w * scale), int(im_h * scale)
    if ow != im_w or oh != im_h:
        img = cv2.resize(img, (ow, oh), interpolation=cv2.INTER_LINEAR)

    pad_h = pad_w = 0
    if ow != w or oh != h:
        pad_img = np.zeros((h, w, 3))
        pad_h = (h - oh) // 2
        pad_w = (w - ow) // 2
        pad_img[pad_h:pad_h + oh, pad_w:pad_w + ow] = img
        img = pad_img

    img = normalize_image(img, normalize_type='127.5')

    img = img.transpose(2, 0, 1)  # HWC -> CHW
    img = np.expand_dims(img, axis=0)
    img = img.astype(np.float32)

    return img, (pad_h, pad_w)


def predict(net, img):
    aug_pred = args.aug_pred
    dataset = args.dataset
    info = {
        'cityscapes': (IMAGE_CITYSCAPE_HEIGHT, IMAGE_CITYSCAPE_WIDTH),
        'camvid': (IMAGE_CAMVID_HEIGHT, IMAGE_CAMVID_WIDTH),
    }
    image_shape = info[dataset]

    h, w = img.shape[:2]
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img, pad_hw = preprocess(img, image_shape)

    # feedforward
    output = net.predict([img])
    logit = output[0]

    # augment inference
    if aug_pred:
        final_logit = softmax(logit, axis=1)

        img = img[:, :, :, ::-1]
        output = net.predict([img])
        logit = output[0][:, :, :, ::-1]
        final_logit += softmax(logit, axis=1)

        logit = final_logit

    pred = np.argmax(logit, axis=1).astype(np.uint8)
    pred = pred[0]

    # reverse_transform
    pad_h, pad_w = pad_hw
    pred = pred[pad_h:-pad_h + image_shape[0], pad_w:-pad_w + image_shape[1]]
    pred = cv2.resize(pred, (w, h), interpolation=cv2.INTER_NEAREST)

    return pred


def recognize_from_image(net):
    pseudo = args.pseudo

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
                pred = predict(net, img)
                end = int(round(time.time() * 1000))
                estimation_time = (end - start)

                # Loggin
                logger.info(f'\tailia processing estimation time {estimation_time} ms')
                if i != 0:
                    total_time_estimation = total_time_estimation + estimation_time

            logger.info(f'\taverage time estimation {total_time_estimation / (args.benchmark_count - 1)} ms')
        else:
            pred = predict(net, img)

        res_img = visualize(img, pred, weight=0.6)

        # plot result
        savepath = get_savepath(args.savepath, image_path, ext='.png')
        logger.info(f'saved at : {savepath}')
        cv2.imwrite(savepath, res_img)

        if pseudo:
            pred_mask = get_pseudo(pred)

            savepath = savepath.replace('.png', '_mask.png')
            logger.info(f'saved at : {savepath}')
            cv2.imwrite(savepath, pred_mask)

    logger.info('Script finished successfully.')


def recognize_from_video(net):
    video_file = args.video if args.video else args.input[0]
    capture = get_capture(video_file)
    assert capture.isOpened(), 'Cannot capture source'

    # create video writer if savepath is specified as video format
    f_h = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    f_w = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    if args.savepath != SAVE_IMAGE_PATH:
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
        pred = predict(net, frame)

        # plot result
        res_img = visualize(frame, pred, weight=0.6)

        # show
        cv2.imshow('frame', res_img)
        frame_shown = True

        # save results
        if writer is not None:
            writer.write(res_img.astype(np.uint8))

    capture.release()
    cv2.destroyAllWindows()
    if writer is not None:
        writer.release()

    logger.info('Script finished successfully.')


def main():
    model_type = args.model_type
    dataset = args.dataset
    info = {
        ('stdc1', 'cityscapes'): (WEIGHT_PP_LiteSeg_T_CITYSCAPE_PATH, MODEL_PP_LiteSeg_T_CITYSCAPE_PATH),
        ('stdc2', 'cityscapes'): (WEIGHT_PP_LiteSeg_B_CITYSCAPE_PATH, MODEL_PP_LiteSeg_B_CITYSCAPE_PATH),
        ('stdc1', 'camvid'): (WEIGHT_PP_LiteSeg_T_CAMVID_PATH, MODEL_PP_LiteSeg_T_CAMVID_PATH),
        ('stdc2', 'camvid'): (WEIGHT_PP_LiteSeg_B_CAMVID_PATH, MODEL_PP_LiteSeg_B_CAMVID_PATH),
    }
    WEIGHT_PATH, MODEL_PATH = info[(model_type, dataset)]
    check_and_download_models(WEIGHT_PATH, MODEL_PATH, REMOTE_PATH)

    env_id = args.env_id

    # initialize
    net = ailia.Net(MODEL_PATH, WEIGHT_PATH, env_id=env_id, memory_mode=args.memory_mode)

    if args.video is not None:
        recognize_from_video(net)
    else:
        recognize_from_image(net)


if __name__ == '__main__':
    main()
