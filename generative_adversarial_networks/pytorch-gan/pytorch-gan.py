import sys
import time

import numpy as np
import cv2

import ailia

sys.path.append('../../util')
from arg_utils import get_base_parser, update_parser  # noqa: E402
from model_utils import check_and_download_models  # noqa: E402
from webcamera_utils import get_writer  # noqa: E402

# logger
from logging import getLogger   # noqa: E402
logger = getLogger(__name__)


# ======================
# Parameters
# ======================
SAVE_IMAGE_PATH = 'output.png'  # default value
MODEL_NAME = 'celeb'  # default value

REMOTE_PATH = 'https://storage.googleapis.com/ailia-models/pytorch-gan/'

OUTPUT_SIZE = 0  # uninitialized


# =======================
# Arguments Parser Config
# =======================
parser = get_base_parser(
    ('Generation of anime character faces using '
     'a GNet trained from the PytorchZoo GAN.'),
    None,
    SAVE_IMAGE_PATH,
)
parser.add_argument(
    '-m', '--model', metavar='MODEL_NAME',
    default=MODEL_NAME,
    help='Model to use ("anime" or "celeb". Default is "anime").'
)
parser.add_argument(
    '--seed', type=int, default=128,
    help='random seed for input data'
)
args = update_parser(parser)

np.random.seed(args.seed)

if args.model == 'anime':
    logger.info('Generation using model "AnimeFace"')
    MODEL_INFIX = 'animeface'
    OUTPUT_SIZE = 64
elif args.model == 'celeb':
    logger.info('Generation using model "CelebA"')
    MODEL_INFIX = 'celeba'
    OUTPUT_SIZE = 128
else:
    logger.error(
        f'unknown model name "{args.model}" (must be "anime" or "celeb")'
    )
    exit(-1)

MODEL_PATH = 'pytorch-gnet-'+MODEL_INFIX+'.onnx.prototxt'
WEIGHT_PATH = 'pytorch-gnet-'+MODEL_INFIX+'.onnx'


def generate_image():
    # prepare input data
    rand_input = np.random.rand(1, 512).astype(np.float32)

    # net initialize
    gnet = ailia.Net(MODEL_PATH, WEIGHT_PATH, env_id=args.env_id, memory_mode=args.memory_mode)

    # inference
    logger.info('Start inference...')
    if args.benchmark:
        logger.info('BENCHMARK mode')
        for i in range(5):
            start = int(round(time.time() * 1000))
            _ = gnet.predict(rand_input)
            end = int(round(time.time() * 1000))
            logger.info(f'\tailia processing time {end - start} ms')
    else:
        _ = gnet.predict(rand_input)

    # postprocessing

    [output_blob_idx] = gnet.get_output_blob_list()
    output_data = gnet.get_blob_data(output_blob_idx)

    outp = np.clip((0.5 + 255*output_data.transpose(
        (2, 3, 1, 0)
    ).reshape((OUTPUT_SIZE, OUTPUT_SIZE, 3))).astype(np.float32), 0, 255)

    cv2.imwrite(
        args.savepath,
        cv2.cvtColor(outp.astype(np.uint8), cv2.COLOR_RGB2BGR)
    )
    logger.info('Script finished successfully.')


def generate_video():
    # net initialize
    gnet = ailia.Net(MODEL_PATH, WEIGHT_PATH, env_id=args.env_id, memory_mode=args.memory_mode)

    # create video writer if savepath is specified as video format
    if args.savepath != SAVE_IMAGE_PATH:
        writer = get_writer(args.savepath, OUTPUT_SIZE, OUTPUT_SIZE)
    else:
        writer = None

    # inference
    frame_shown = False
    while(True):
        if (cv2.waitKey(1) & 0xFF == ord('q')):
            break
        if frame_shown and cv2.getWindowProperty('frame', cv2.WND_PROP_VISIBLE) == 0:
            break

        # prepare input data
        no_1 = int(np.random.rand(1)*511)
        no_2 = int(np.random.rand(1)*511)
        rand_input = np.zeros((1, 512))
        rand_input[0, no_1] = 1.0
        rand_input[0, no_2] = 1.0

        # inference
        _ = gnet.predict(rand_input)

        # postprocessing
        [output_blob_idx] = gnet.get_output_blob_list()
        output_data = gnet.get_blob_data(output_blob_idx)

        outp = np.clip((0.5 + 255*output_data.transpose(
            (2, 3, 1, 0)
        ).reshape((OUTPUT_SIZE, OUTPUT_SIZE, 3))).astype(np.float32), 0, 255)

        image = cv2.cvtColor(outp.astype(np.uint8), cv2.COLOR_RGB2BGR)
        cv2.imshow("frame", image)
        frame_shown = True

        # save results
        if writer is not None:
            writer.write(image)

    cv2.destroyAllWindows()
    if writer is not None:
        writer.release()
    logger.info('Script finished successfully.')


def main():
    # model files check and download
    check_and_download_models(WEIGHT_PATH, MODEL_PATH, REMOTE_PATH)

    if args.video:
        generate_video()
    else:
        generate_image()


if __name__ == '__main__':
    main()
