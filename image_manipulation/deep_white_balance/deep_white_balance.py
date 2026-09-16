import sys, os
import time
import cv2

import ailia

# import original modules
sys.path.append('../../util')
from arg_utils import get_base_parser, update_parser, get_savepath  # noqa: E402
from model_utils import check_and_download_models  # noqa: E402

import webcamera_utils  # noqa: E402

from deep_white_balance_utils.deepWB import deep_wb
from deep_white_balance_utils.utils import *

# logger
from logging import getLogger   # noqa: E402
logger = getLogger(__name__)


# ======================
# Parameters
# ======================

WEIGHT_AWB_PATH = "net_awb.onnx"
MODEL_AWB_PATH = "net_awb.onnx.prototxt"
WEIGHT_S_PATH = "net_s.onnx"
MODEL_S_PATH = "net_s.onnx.prototxt"
WEIGHT_T_PATH = "net_t.onnx"
MODEL_T_PATH = "net_t.onnx.prototxt"
REMOTE_PATH = 'https://storage.googleapis.com/ailia-models/deep_white_balance/'

IMAGE_PATH = 'input.jpg'
SAVE_IMAGE_PATH = ''

# Default input size
HEIGHT_SIZE = 320
WIDTH_SIZE = 656

# ======================
# Arguemnt Parser Config
# ======================
parser = get_base_parser(
    'Deep White Balance',
    IMAGE_PATH,
    SAVE_IMAGE_PATH,
)
args = update_parser(parser)


def recognize_from_image():
    # net initialize
    env_id = args.env_id
    net_awb = ailia.Net(MODEL_AWB_PATH, WEIGHT_AWB_PATH, env_id=env_id, memory_mode=args.memory_mode)
    net_s = ailia.Net(MODEL_S_PATH, WEIGHT_S_PATH, env_id=env_id, memory_mode=args.memory_mode)
    net_t = ailia.Net(MODEL_T_PATH, WEIGHT_T_PATH, env_id=env_id, memory_mode=args.memory_mode)

    # input image loop
    for image_path in args.input:
        # prepare input data
        logger.info(image_path)

        img = Image.open(image_path)
        org_w = np.array(img).shape[1]
        org_h = np.array(img).shape[0]

        img = img.resize((WIDTH_SIZE, HEIGHT_SIZE))
        logger.info(f'input image shape: {np.array(img).shape}')

        # inference
        logger.info('Start inference...')
        if args.benchmark:
            logger.info('BENCHMARK mode')
            for i in range(5):
                start = int(round(time.time() * 1000))
                result = deep_wb(img, task='all', net_awb=net_awb, net_s=net_s, net_t=net_t,
                                    device='cpu', s=656)
                end = int(round(time.time() * 1000))
                logger.info(f'\tailia processing time {end - start} ms')
        else:
            result = deep_wb(img, task='all', net_awb=net_awb, net_s=net_s, net_t=net_t,
                             device='cpu', s=656)

        # save result
        out_awb, out_t, out_s = result

        result_awb = to_image(out_awb).resize((org_w,org_h))
        result_t = to_image(out_t).resize((org_w,org_h))
        result_s = to_image(out_s).resize((org_w,org_h))

        logger.info(f'saved at : {args.savepath}')

        result_awb.save(os.path.join(args.savepath, 'output_AWB.png'))
        result_s.save(os.path.join(args.savepath,  'output_S.png'))
        result_t.save(os.path.join(args.savepath,  'output_T.png'))

    logger.info('Script finished successfully.')

def recognize_from_video():
    # net initialize
    env_id = args.env_id
    net_awb = ailia.Net(MODEL_AWB_PATH, WEIGHT_AWB_PATH, env_id=env_id, memory_mode=args.memory_mode)
    net_s = ailia.Net(MODEL_S_PATH, WEIGHT_S_PATH, env_id=env_id, memory_mode=args.memory_mode)
    net_t = ailia.Net(MODEL_T_PATH, WEIGHT_T_PATH, env_id=env_id, memory_mode=args.memory_mode)

    capture = webcamera_utils.get_capture(args.video)
    # create video writer if savepath is specified as video format
    if args.savepath != SAVE_IMAGE_PATH:
        logger.warning(
            'currently, video results cannot be output correctly...'
        )
        f_h = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        f_w = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        save_h, save_w = calc_adjust_fsize(f_h, f_w, IMAGE_HEIGHT, IMAGE_WIDTH)
        # save_w * 2: we stack source frame and estimated heatmap
        writer = get_writer(args.savepath, save_h, save_w * 2)
    else:
        writer = None

    while (True):
        ret, frame = capture.read()
        if (cv2.waitKey(1) & 0xFF == ord('q')) or not ret:
            break

        img = Image.fromarray(frame)
        img = img.resize((WIDTH_SIZE, HEIGHT_SIZE))

        # inference
        result = deep_wb(img, task='all', net_awb=net_awb, net_s=net_s, net_t=net_t,
                         device='cpu', s=656)
        # plot result
        out_awb, out_t, out_s = result
        out_f, out_d, out_c = colorTempInterpolate(out_t, out_s)

        result_awb = to_image(out_awb)
        result_t = to_image(out_t)
        result_s = to_image(out_s)
        result_f = to_image(out_f)
        result_d = to_image(out_d)
        result_c = to_image(out_c)

        imshow(img, result_awb, result_t, result_f, result_d, result_c, result_s)
        plt.pause(1)

        if not plt.get_fignums():
            break

        plt.close()

    capture.release()
    cv2.destroyAllWindows()
    if writer is not None:
        writer.release()
    logger.info('Script finished successfully.')


def main():
    # model files check and download
    check_and_download_models(WEIGHT_AWB_PATH, MODEL_AWB_PATH, REMOTE_PATH)
    check_and_download_models(WEIGHT_S_PATH, MODEL_S_PATH, REMOTE_PATH)
    check_and_download_models(WEIGHT_T_PATH, MODEL_T_PATH, REMOTE_PATH)

    if args.video is not None:
        # video mode
        recognize_from_video()
    else:
        # image mode
        recognize_from_image()


if __name__ == '__main__':
    main()
