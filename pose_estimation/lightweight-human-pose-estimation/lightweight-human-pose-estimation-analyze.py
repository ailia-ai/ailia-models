import sys
import time

import ailia
import cv2
import matplotlib.pyplot as plt
import numpy as np

sys.path.append('../../util')
# logger
from logging import getLogger  # noqa: E402

from image_utils import imread, load_image  # noqa: E402
from model_utils import check_and_download_models  # noqa: E402
from arg_utils import get_base_parser, get_savepath, update_parser  # noqa: E402

logger = getLogger(__name__)


# ======================
# Parameters 1
# ======================
IMAGE_PATH = 'balloon.png'
SAVE_IMAGE_PATH = 'output.png'
IMAGE_HEIGHT = 240
IMAGE_WIDTH = 320

ALGORITHM = ailia.POSE_ALGORITHM_LW_HUMAN_POSE


# ======================
# Arguemnt Parser Config
# ======================
parser = get_base_parser(
    'Fast and accurate human pose 2D-estimation.', IMAGE_PATH, SAVE_IMAGE_PATH,
)
parser.add_argument(
    '-n', '--normal',
    action='store_true',
    help='By default, the optimized model is used, but with this option, ' +
    'you can switch to the normal (not optimized) model'
)
args = update_parser(parser)


# ======================
# Parameters 2
# ======================
MODEL_NAME = 'lightweight-human-pose-estimation'
if args.normal:
    WEIGHT_PATH = f'{MODEL_NAME}.onnx'
    MODEL_PATH = f'{MODEL_NAME}.onnx.prototxt'
else:
    WEIGHT_PATH = f'{MODEL_NAME}.opt.onnx'
    MODEL_PATH = f'{MODEL_NAME}.opt.onnx.prototxt'
REMOTE_PATH = f'https://storage.googleapis.com/ailia-models/{MODEL_NAME}/'


# ======================
# Utils
# ======================
def plot_images(title, images, tile_shape):
    assert images.shape[0] <= (tile_shape[0] * tile_shape[1])
    from mpl_toolkits.axes_grid1 import ImageGrid
    fig = plt.figure()
    plt.title(title)
    grid = ImageGrid(fig, 111,  nrows_ncols=tile_shape)
    for i in range(images.shape[1]):
        grd = grid[i]
        grd.imshow(images[0, i])


# ======================
# Main functions
# ======================
def main():
    # model files check and download
    check_and_download_models(WEIGHT_PATH, MODEL_PATH, REMOTE_PATH)

    # net initialize
    net = ailia.Net(MODEL_PATH, WEIGHT_PATH, env_id=args.env_id, memory_mode=args.memory_mode)

    # input image loop
    for image_path in args.input:
        # prepare input data
        logger.info(image_path)
        src_img = imread(image_path)
        input_image = load_image(
            image_path,
            (IMAGE_HEIGHT, IMAGE_WIDTH),
            normalize_type='None',
        )
        input_data = (input_image - 128) / 255.0
        input_data = input_data[np.newaxis, :, :, :].transpose((0, 3, 1, 2))

        for i in range(3):
            if(i == 1):
                net.set_profile_mode()
            start = int(round(time.time() * 1000))
            _ = net.predict(input_data)
            end = int(round(time.time() * 1000))
            logger.info(f'ailia processing time {end - start} ms')

        confidence = net.get_blob_data(net.find_blob_index_by_name("397"))
        paf = net.get_blob_data(net.find_blob_index_by_name("400"))
        points = []
        threshold = 0.1

        for i in range(confidence.shape[1]):
            probMap = confidence[0, i, :, :]
            minVal, prob, minLoc, point = cv2.minMaxLoc(probMap)

            x = (src_img.shape[1] * point[0]) / confidence.shape[3]
            y = (src_img.shape[0] * point[1]) / confidence.shape[2]

            if prob > threshold:
                circle_size = 4
                cv2.circle(src_img, (int(x), int(y)), circle_size,
                           (0, 255, 255), thickness=-1, lineType=cv2.FILLED)
                cv2.putText(
                    src_img,
                    "{}".format(i),
                    (int(x), int(y)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1.0,
                    (0, 0, 255),
                    1,
                    lineType=cv2.LINE_AA
                )
                cv2.putText(
                    src_img,
                    ""+str(prob),
                    (int(x), int(y+circle_size)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 0, 255),
                    1,
                    lineType=cv2.LINE_AA
                )

                points.append((int(x), int(y)))
            else:
                points.append(None)

        cv2.imshow("Keypoints", src_img)
        savepath = get_savepath(args.savepath, image_path)
        logger.info(f'saved at : {savepath}')
        cv2.imwrite(savepath, src_img)

        channels = max(confidence.shape[1], paf.shape[1])
        cols = 8

        plot_images(
            "paf",
            paf,
            tile_shape=((int)((channels+cols-1)/cols), cols),
        )
        plot_images(
            "confidence",
            confidence,
            tile_shape=((int)((channels+cols-1)/cols), cols),
        )

        plt.show()

        cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
