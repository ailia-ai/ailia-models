import sys
import time

import numpy as np
import cv2
from skimage.transform import resize

import ailia

# import original modules
sys.path.append('../../util')
from arg_utils import get_base_parser, update_parser, get_savepath  # noqa
from model_utils import check_and_download_models  # noqa
from image_utils import normalize_image  # noqa
from detector_utils import load_image  # noqa
from webcamera_utils import get_capture, get_writer  # noqa
# logger
from logging import getLogger  # noqa

logger = getLogger(__name__)

# ======================
# Parameters
# ======================

WEIGHT_R342B_TT_PATH = 'gfm_r34_2b_tt.onnx'
MODEL_R342B_TT_PATH = 'gfm_r34_2b_tt.onnx.prototxt'
WEIGHT_D121_TT_PATH = 'gfm_d121_tt.onnx'
MODEL_D121_TT_TT_PATH = 'gfm_d121_tt.onnx.prototxt'
WEIGHT_D121_RIM_PATH = 'gfm_d121_rim.onnx'
MODEL_D121_RIM_TT_PATH = 'gfm_d121_rim.onnx.prototxt'
REMOTE_PATH = 'https://storage.googleapis.com/ailia-models/gfm/'

IMAGE_PATH = 'demo.jpg'
SAVE_IMAGE_PATH = 'output.png'

IMAGE_SIZE = 480
IMAGE_RIM_SIZE = 960

# ======================
# Arguemnt Parser Config
# ======================

parser = get_base_parser(
    'GFM', IMAGE_PATH, SAVE_IMAGE_PATH
)
parser.add_argument(
    '-m', '--model_type', default='r34_2b_tt', choices=('r34_2b_tt', 'd121_tt', 'd121_rim'),
    help='model type'
)
args = update_parser(parser)


# ======================
# Secondaty Functions
# ======================

def get_masked_local_from_global_test(global_result, local_result):
    weighted_global = np.ones(global_result.shape)
    weighted_global[global_result == 255] = 0
    weighted_global[global_result == 0] = 0
    fusion_result = global_result * (1. - weighted_global) / 255 + local_result * weighted_global

    return fusion_result


def gen_trimap_from_segmap_e2e(segmap):
    trimap = np.argmax(segmap, axis=1)[0]
    trimap = trimap.astype(np.int64)
    trimap[trimap == 1] = 128
    trimap[trimap == 2] = 255

    return trimap.astype(np.uint8)


def generate_composite_img(img, alpha_channel):
    r_channel, g_channel, b_channel = cv2.split(img)
    b_channel = (b_channel * alpha_channel).astype(np.uint8)
    g_channel = (g_channel * alpha_channel).astype(np.uint8)
    r_channel = (r_channel * alpha_channel).astype(np.uint8)
    alpha_channel = (alpha_channel * 255).astype(np.uint8)
    img_BGRA = cv2.merge((b_channel, g_channel, r_channel, alpha_channel))

    return img_BGRA


# ======================
# Main functions
# ======================


def preprocess(img):
    img = img.transpose(2, 0, 1)  # HWC -> CHW
    img = np.expand_dims(img, axis=0)
    img = img.astype(np.float32)

    return img


def resize_pad(img, ratio):
    h, w, _ = img.shape
    resize_h = int(h * ratio)
    resize_w = int(w * ratio)

    image_size = IMAGE_RIM_SIZE if args.model_type.endswith('_rim') else IMAGE_SIZE

    scale = image_size / max(resize_h, resize_w)
    if scale < 1:
        resize_w, resize_h = int(resize_w * scale), int(resize_h * scale)

    img = resize(img, (resize_h, resize_w)) * 255.0

    if resize_w != image_size or resize_h != image_size:
        pad_img = np.ones((image_size, image_size, 3)) * 255
        pad_img[:resize_h, :resize_w, ...] = img
        img = pad_img

    img = preprocess(img)

    return img, (resize_h, resize_w)


def post_processing(output):
    pred_global, pred_local, pred_fusion = output

    pred_global = gen_trimap_from_segmap_e2e(pred_global)
    pred_local = pred_local[0, 0, :, :]
    pred_fusion = pred_fusion[0, 0, :, :]

    return pred_global, pred_local, pred_fusion


def predict(net, img):
    h, w, _ = img.shape

    img = img[:, :, ::-1]  # BGR -> RGB

    simple_resize = False
    if simple_resize:
        if args.model_type.endswith('_rim'):
            img = resize(img, (IMAGE_RIM_SIZE, IMAGE_RIM_SIZE)) * 255.0
            img = preprocess(img)

            # feedforward
            output = net.predict([img])
            _, _, pred_tt, _, _, pred_ft, _, _, pred_bt, pred_fusion = output

            pred_tt = resize(pred_tt[0, 0, :, :], (h, w))
            pred_ft = resize(pred_ft[0, 0, :, :], (h, w))
            pred_bt = resize(pred_bt[0, 0, :, :], (h, w))
            pred_fusion = resize(pred_fusion[0, 0, :, :], (h, w))

            return pred_tt, pred_ft, pred_bt, pred_fusion
        else:
            img = resize(img, (IMAGE_SIZE, IMAGE_SIZE)) * 255.0
            img = preprocess(img)

            # feedforward
            output = net.predict([img])
            pred_glance, pred_focus, pred_fusion = post_processing(output)
    elif args.model_type.endswith('_rim'):
        scale_img, resize_hw = resize_pad(img, 1)

        # feedforward
        output = net.predict([scale_img])
        _, _, pred_tt, _, _, pred_ft, _, _, pred_bt, pred_fusion = output

        pred_tt = resize(pred_tt[0, 0, :, :][:resize_hw[0], :resize_hw[1]], (h, w))
        pred_ft = resize(pred_ft[0, 0, :, :][:resize_hw[0], :resize_hw[1]], (h, w))
        pred_bt = resize(pred_bt[0, 0, :, :][:resize_hw[0], :resize_hw[1]], (h, w))
        pred_fusion = resize(pred_fusion[0, 0, :, :][:resize_hw[0], :resize_hw[1]], (h, w))

        return pred_tt, pred_ft, pred_bt, pred_fusion
    else:
        # Combine 1/3 glance and 1/2 focus
        global_ratio = 1 / 3
        local_ratio = 1 / 2

        # feedforward
        scale_img, resize_hw = resize_pad(img, global_ratio)
        output = net.predict([scale_img])
        pred_glance_1, pred_focus_1, pred_fusion_1 = post_processing(output)
        pred_glance = pred_glance_1[:resize_hw[0], :resize_hw[1]]

        scale_img, resize_hw = resize_pad(img, local_ratio)
        output = net.predict([scale_img])
        pred_glance_2, pred_focus_2, pred_fusion_2 = post_processing(output)
        pred_focus = pred_focus_2[:resize_hw[0], :resize_hw[1]]

    pred_glance = resize(pred_glance, (h, w)) * 255.0
    pred_focus = resize(pred_focus, (h, w))
    pred_fusion = get_masked_local_from_global_test(pred_glance, pred_focus)

    return pred_glance, pred_focus, pred_fusion


def recognize_from_image(net):
    # input image loop
    for image_path in args.input:
        logger.info(image_path)

        # prepare input data
        img = load_image(image_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2RGB)

        # inference
        logger.info('Start inference...')
        if args.benchmark:
            logger.info('BENCHMARK mode')
            total_time_estimation = 0
            for i in range(args.benchmark_count):
                start = int(round(time.time() * 1000))
                out = predict(net, img)
                end = int(round(time.time() * 1000))
                estimation_time = (end - start)

                # Logging
                logger.info(f'\tailia processing estimation time {estimation_time} ms')
                if i != 0:
                    total_time_estimation = total_time_estimation + estimation_time

            logger.info(f'\taverage time estimation {total_time_estimation / (args.benchmark_count - 1)} ms')
        else:
            out = predict(net, img)

        pred = out[2]
        res_img = generate_composite_img(img, pred)

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
        img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        out = predict(net, img)
        pred = out[2]

        # plot result
        res_img = generate_composite_img(img, pred)

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
    dic_model = {
        'r34_2b_tt': (WEIGHT_R342B_TT_PATH, MODEL_R342B_TT_PATH),
        'd121_tt': (WEIGHT_D121_TT_PATH, MODEL_D121_TT_TT_PATH),
        'd121_rim': (WEIGHT_D121_RIM_PATH, MODEL_D121_RIM_TT_PATH),
    }
    WEIGHT_PATH, MODEL_PATH = dic_model[args.model_type]

    # model files check and download
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
