import sys
import time

import numpy as np
import cv2
import platform

import ailia

# import original modules
sys.path.append('../../util')
from arg_utils import get_base_parser, update_parser, get_savepath  # noqa
from model_utils import check_and_download_models  # noqa
from image_utils import normalize_image  # noqa
from detector_utils import load_image  # noqa
from webcamera_utils import get_capture, get_writer  # noqa
from functional import grid_sample  # noqa
# logger
from logging import getLogger  # noqa

logger = getLogger(__name__)

# ======================
# Parameters
# ======================

WEIGHT_NH_PATH = 'PSNR2066_SSIM06844_NH.onnx'
MODEL_NH_PATH = 'PSNR2066_SSIM06844_NH.onnx.prototxt'
WEIGHT_OUTDOOR_PATH = 'PSNR3518_SSIM09860_outdoor.onnx'
MODEL_OUTDOOR_PATH = 'PSNR3518_SSIM09860_outdoor.onnx.prototxt'
WEIGHT_DENSE_PATH = 'PSNR1662_SSIM05602_dense.onnx'
MODEL_DENSE_PATH = 'PSNR1662_SSIM05602_dense.onnx.prototxt'
WEIGHT_INDOOR_PATH = 'PSNR3663_ssim09881_indoor.onnx'
MODEL_INDOOR_PATH = 'PSNR3663_ssim09881_indoor.onnx.prototxt'
REMOTE_PATH = 'https://storage.googleapis.com/ailia-models/dehamer/'

IMAGE_PATH = 'canyon.png'
SAVE_IMAGE_PATH = 'output.png'

IMAGE_MAX_SIZE = 960

# ======================
# Arguemnt Parser Config
# ======================

parser = get_base_parser(
    'Dehamer', IMAGE_PATH, SAVE_IMAGE_PATH, fp16_support=False
)
parser.add_argument(
    '-m', '--model_type', default='NH', choices=('NH', 'dense', 'indoor', 'outdoor'),
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

def preprocess(img):
    im_h, im_w, _ = img.shape

    img = _img = img[:, :, ::-1]  # BGR -> RGB
    
    simple_resize = False
    pad_tile = False
    
    if simple_resize:
        img = cv2.resize(img, (IMAGE_MAX_SIZE, IMAGE_MAX_SIZE), interpolation=cv2.INTER_LINEAR)
        oh = ow = IMAGE_MAX_SIZE
    else:
        scale = IMAGE_MAX_SIZE / max(im_h, im_w)
        oh = im_h
        ow = im_w
        if scale < 1:
            oh = int(im_h * scale + 0.5)
            ow = int(im_w * scale + 0.5)
            img = cv2.resize(img, (ow, oh), interpolation=cv2.INTER_LINEAR)
        
        if pad_tile:
            pad_img = np.tile(img, (int(np.ceil(IMAGE_MAX_SIZE / oh)), int(np.ceil(IMAGE_MAX_SIZE / ow)), 1))
            img = pad_img[:IMAGE_MAX_SIZE, :IMAGE_MAX_SIZE, ...]
        else:
            pad_img = cv2.resize(_img, (IMAGE_MAX_SIZE, IMAGE_MAX_SIZE), interpolation=cv2.INTER_LINEAR)
            pad_img[:oh, :ow, ...] = img
            img = pad_img

    mean = np.array((0.64, 0.6, 0.58))
    std = np.array((0.14, 0.15, 0.152))

    img = img / 255
    img = (img - mean) / std

    img = img.transpose((2, 0, 1))  # HWC -> CHW
    img = np.expand_dims(img, axis=0)
    img = img.astype(np.float32)

    return img, (oh, ow)


def predict(net, img):
    im_h, im_w = img.shape[:2]
    img, pp_hw = preprocess(img)

    # feedforward
    if not args.onnx:
        output = net.predict([img])
    else:
        output = net.run(None, {'haze': img})

    dehaze = output[0]

    # postprocess
    dehaze = dehaze[0].transpose((1, 2, 0))  # CHW -> HWC
    dehaze *= 255
    dehaze = np.clip(dehaze, 0, 255)
    dehaze = dehaze[:pp_hw[0], :pp_hw[1], ...]
    if pp_hw[0] != im_h or pp_hw[1] != im_w:
        dehaze = cv2.resize(dehaze, (im_w, im_h), interpolation=cv2.INTER_LINEAR)
    dehaze = dehaze[:, :, ::-1]  # RGB -> BGR
    dehaze = dehaze.astype(np.uint8)

    return dehaze


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
                dehaze = predict(net, img)
                end = int(round(time.time() * 1000))
                estimation_time = (end - start)

                # Loggin
                logger.info(f'\tailia processing estimation time {estimation_time} ms')
                if i != 0:
                    total_time_estimation = total_time_estimation + estimation_time

            logger.info(f'\taverage time estimation {total_time_estimation / (args.benchmark_count - 1)} ms')
        else:
            dehaze = predict(net, img)

        # plot result
        savepath = get_savepath(args.savepath, image_path, ext='.png')
        logger.info(f'saved at : {savepath}')
        cv2.imwrite(savepath, dehaze)

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
        dehaze = predict(net, frame)

        # show
        cv2.imshow('frame', dehaze)
        frame_shown = True

        # save results
        if writer is not None:
            res_img = cv2.resize(dehaze, (f_w, f_h))
            writer.write(res_img.astype(np.uint8))

    capture.release()
    cv2.destroyAllWindows()
    if writer is not None:
        writer.release()

    logger.info('Script finished successfully.')


def main():
    dic_model = {
        'NH': (WEIGHT_NH_PATH, MODEL_NH_PATH),
        'dense': (WEIGHT_DENSE_PATH, MODEL_DENSE_PATH),
        'indoor': (WEIGHT_INDOOR_PATH, MODEL_INDOOR_PATH),
        'outdoor': (WEIGHT_OUTDOOR_PATH, MODEL_OUTDOOR_PATH),
    }
    WEIGHT_PATH, MODEL_PATH = dic_model[args.model_type]

    # model files check and download
    check_and_download_models(WEIGHT_PATH, MODEL_PATH, REMOTE_PATH)

    # initialize
    if not args.onnx:
        net = ailia.Net(MODEL_PATH, WEIGHT_PATH, env_id=args.env_id, memory_mode=args.memory_mode)
    else:
        import onnxruntime
        net = onnxruntime.InferenceSession(WEIGHT_PATH)

    if args.video is not None:
        recognize_from_video(net)
    else:
        recognize_from_image(net)


if __name__ == '__main__':
    main()
