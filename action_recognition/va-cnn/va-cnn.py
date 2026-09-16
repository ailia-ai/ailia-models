import sys
import time

import numpy as np

import ailia

# import original modules
sys.path.append('../../util')
from arg_utils import get_base_parser, update_parser, get_savepath  # noqa: E402
from params import MODALITIES, EXTENSIONS  # noqa: E402
from model_utils import check_and_download_models  # noqa: E402

# logger
from logging import getLogger  # noqa: E402

from va_cnn_utils import *
from vis_3d_ske import draw_ske_data
from labels import LABELS

MODALITIES.append('skel')
EXTENSIONS['skel'] = ['*.skeleton']

logger = getLogger(__name__)

# ======================
# Parameters
# ======================
WEIGHT_PATH = 'va-cnn.onnx'
MODEL_PATH = 'va-cnn.onnx.prototxt'
REMOTE_PATH = 'https://storage.googleapis.com/ailia-models/va-cnn/'

INPUT_FILE = 'data/ntu/nturgb+d_skeletons/S001C001P001R001A001.skeleton'
SAVE_PATH = 'output.mp4'

# ======================
# Arguemnt Parser Config
# ======================
parser = get_base_parser('View Adaptive Neural Networks', INPUT_FILE, SAVE_PATH, input_ftype='skel')
parser.add_argument(
    '-s', '--save_video', action='store_true',
    help='Save the video file.'
)
args = update_parser(parser)


# ======================
# Main functions
# ======================

def preprocess(x):
    max = 5.18858098984
    min = -5.28981208801

    imgs, maxmin = torgb([x], max, min)
    imgs = np.stack([imgs[i] for i in range(len(imgs))], axis=0)
    maxmin = np.vstack(maxmin).astype(np.float32)

    return imgs, maxmin


def recognize_from_keypoints(net):
    for skel_path in args.input:
        logger.info(skel_path)

        bodies_data = get_raw_bodies_data(skel_path)
        _, ske_joints, _ = get_raw_denoised_data(bodies_data)
        ske_joints = seq_translation(ske_joints)
        x = align_frames(ske_joints)

        imgs, maxmin = preprocess(x)

        logger.debug(f'input image shape: {imgs.shape}')

        # inference
        logger.info('Start inference...')
        if args.benchmark:
            logger.info('BENCHMARK mode')
            total_time = 0
            for i in range(args.benchmark_count):
                start = int(round(time.time() * 1000))
                output = net.predict({'imgs': imgs, 'maxmin': maxmin})
                end = int(round(time.time() * 1000))
                logger.info(f'\tailia processing time {end - start} ms')
                if i != 0:
                    total_time = total_time + (end - start)
            logger.info(f'\taverage time {total_time / (args.benchmark_count - 1)} ms')
        else:
            output = net.predict({'imgs': imgs, 'maxmin': maxmin})

        pred, img, trans = output

        i = np.argmax(pred, axis=-1)[0]
        label = LABELS[i]
        logger.info('Action estimate -> ' + label)

        if args.save_video:
            savepath = "%s.mp4" % bodies_data['name'] if 1 < len(args.input) else SAVE_PATH
            logger.info(f'saved at : {savepath}')
            draw_ske_data(x, savepath)

    logger.info('Script finished successfully.')


def main():
    # model files check and download
    check_and_download_models(WEIGHT_PATH, MODEL_PATH, REMOTE_PATH)

    # net initialize
    net = ailia.Net(MODEL_PATH, WEIGHT_PATH, env_id=args.env_id, memory_mode=args.memory_mode)

    recognize_from_keypoints(net)


if __name__ == '__main__':
    main()
