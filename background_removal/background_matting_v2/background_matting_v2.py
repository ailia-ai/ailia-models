import sys
import time

import numpy as np
import cv2

import ailia

# import original modules
sys.path.append('../../util')
from arg_utils import get_base_parser, update_parser, get_savepath  # noqa: E402
from model_utils import check_and_download_models  # noqa: E402
from detector_utils import load_image  # noqa: E402
import webcamera_utils  # noqa: E402

# logger
from logging import getLogger  # noqa: E402

logger = getLogger(__name__)

# ======================
# Parameters
# ======================
WEIGHT_MOBILENETV2_PATH = 'mobilenetv2.onnx'
MODEL_MOBILENETV2_PATH = 'mobilenetv2.onnx.prototxt'
WEIGHT_RESNET50_PATH = 'resnet50.onnx'
MODEL_RESNET50_PATH = 'resnet50.onnx.prototxt'
REMOTE_PATH = 'https://storage.googleapis.com/ailia-models/background_matting_v2/'

IMAGE_PATH = 'demo.png'
IMAGE_BGR_PATH = 'bgr.png'
SAVE_IMAGE_PATH = 'output.png'

# ======================
# Arguemnt Parser Config
# ======================

parser = get_base_parser('Real-Time High-Resolution Background Matting', IMAGE_PATH, SAVE_IMAGE_PATH)
parser.add_argument(
    '-bg', '--bgr_image', default=IMAGE_BGR_PATH,
    help='background image'
)
parser.add_argument(
    '-m', '--model_type', default='mobilenetv2', choices=('mobilenetv2', 'resnet50'),
    help='model type'
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

def bgr_image(shape=None):
    file_path = args.bgr_image

    img = load_image(file_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGRA2RGB)

    img = preprocess(img, shape)

    return img


def preprocess(img, shape=None):
    if shape:
        h, w = shape
        img = cv2.resize(img, (w, h))

    img = img / 255
    img = img.transpose(2, 0, 1)  # HWC -> CHW
    img = np.expand_dims(img, axis=0)
    img = img.astype(np.float32)

    return img


def post_process(*args, a=False):
    pha, fgr, pha_sm, fgr_sm, err_sm, ref_sm = args

    if a:
        com = np.concatenate([fgr * np.not_equal(pha, 0), pha], axis=1)
    else:
        bg_clr = np.array([
            120 / 255, 255 / 255, 155 / 255
        ]).reshape((1, 3, 1, 1))
        com = pha * fgr + (1 - pha) * bg_clr

    img = com.transpose((0, 2, 3, 1))[0] * 255
    img = img.astype(np.uint8)

    return img


def predict(net, img, bgr_img):
    _, _, h, w = bgr_img.shape
    im_h, im_w = img.shape[:2]

    shape = (h, w) if im_h != h or im_w != w else None
    img = preprocess(img, shape)

    # feedforward
    if not args.onnx:
        output = net.predict([img, bgr_img])
    else:
        output = net.run(None, {'src': img, 'bgr': bgr_img})

    return output


def recognize_from_image(net):
    # prepare background image
    bgr_img = bgr_image()

    # input image loop
    for image_path in args.input:
        # prepare input data
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
                output = predict(net, img, bgr_img)
                end = int(round(time.time() * 1000))
                estimation_time = (end - start)

                # Loggin
                logger.info(f'\tailia processing estimation time {estimation_time} ms')
                if i != 0:
                    total_time_estimation = total_time_estimation + estimation_time

            logger.info(f'\taverage time estimation {total_time_estimation / (args.benchmark_count - 1)} ms')
        else:
            output = predict(net, img, bgr_img)

        # postprocessing
        res_img = post_process(*output, a=True)
        res_img = cv2.cvtColor(res_img, cv2.COLOR_RGBA2BGRA)

        savepath = get_savepath(args.savepath, image_path, ext='.png')
        logger.info(f'saved at : {savepath}')
        cv2.imwrite(savepath, res_img)

    logger.info('Script finished successfully.')


def recognize_from_video(net):
    capture = webcamera_utils.get_capture(args.video)

    # create video writer if savepath is specified as video format
    f_h = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    f_w = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    if args.savepath != SAVE_IMAGE_PATH:
        logger.warning(
            'currently, video results cannot be output correctly...'
        )
        writer = webcamera_utils.get_writer(args.savepath, f_h, f_w)
    else:
        writer = None

    # prepare background image
    bgr_img = bgr_image((f_h, f_w))

    frame_shown = False
    while True:
        ret, frame = capture.read()
        if (cv2.waitKey(1) & 0xFF == ord('q')) or not ret:
            break
        if frame_shown and cv2.getWindowProperty('frame', cv2.WND_PROP_VISIBLE) == 0:
            break

        # inference
        img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        output = predict(net, img, bgr_img)

        # postprocessing
        res_img = post_process(*output, a=False)
        res_img = cv2.cvtColor(res_img, cv2.COLOR_RGB2BGR)

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
    dic_model = {
        'mobilenetv2': (WEIGHT_MOBILENETV2_PATH, MODEL_MOBILENETV2_PATH),
        'resnet50': (WEIGHT_RESNET50_PATH, MODEL_RESNET50_PATH),
    }
    # weight_path, model_path = dic_model[args.model]
    weight_path, model_path = dic_model[args.model_type]

    # model files check and download
    check_and_download_models(weight_path, model_path, REMOTE_PATH)

    # load model
    env_id = args.env_id
    
    # net initialize
    if not args.onnx:
        net = ailia.Net(model_path, weight_path, env_id=env_id, memory_mode=args.memory_mode)
    else:
        import onnxruntime
        net = onnxruntime.InferenceSession(weight_path)

    if args.video is not None:
        # video mode
        recognize_from_video(net)
    else:
        # image mode
        recognize_from_image(net)


if __name__ == '__main__':
    main()
