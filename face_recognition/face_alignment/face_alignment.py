import collections
import math
import os
import sys
import time

import ailia
import cv2
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.axes_grid1 import ImageGrid

# import original modules
sys.path.append('../../util')
import webcamera_utils  # noqa: E402
from image_utils import imread, load_image  # noqa: E402
from model_utils import check_and_download_models  # noqa: E402
from arg_utils import get_base_parser, get_savepath, update_parser  # noqa: E402

sys.path.append('../../face_detection/blazeface')
# logger
from logging import getLogger  # noqa: E402

from blazeface_utils import compute_blazeface, crop_blazeface  # noqa: E402

logger = getLogger(__name__)


# ======================
# PARAMETERS 1
# ======================
REMOTE_PATH = 'https://storage.googleapis.com/ailia-models/face_alignment/'

IMAGE_PATH = 'aflw-test.jpg'
SAVE_IMAGE_PATH = 'output.png'

IMAGE_HEIGHT = 256
IMAGE_WIDTH = 256
THRESHOLD = 0.1

FACE_WEIGHT_PATH = 'blazeface.onnx'
FACE_MODEL_PATH = 'blazeface.onnx.prototxt'
FACE_REMOTE_PATH = "https://storage.googleapis.com/ailia-models/blazeface/"
FACE_MARGIN = 1.2

# ======================
# Argument Parser Config
# ======================
parser = get_base_parser('Face alignment model', IMAGE_PATH, SAVE_IMAGE_PATH)
parser.add_argument(
    '-3', '--active_3d', action='store_true',
    help='Activate 3D face alignment mode'
)
args = update_parser(parser)


# ======================
# PARAMETERS 2
# ======================
WEIGHT_PATH = '3DFAN-4.onnx' if args.active_3d else '2DFAN-4.onnx'
MODEL_PATH = WEIGHT_PATH + '.prototxt'
DEPTH_WEIGHT_PATH = 'depth_estimation.onnx'
DEPTH_MODEL_PATH = DEPTH_WEIGHT_PATH + '.prototxt'

PRED_TYPE = collections.namedtuple('prediction_type', ['slice', 'color'])
PRED_TYPES = {
    'face': PRED_TYPE(slice(0, 17), (0.682, 0.780, 0.909, 0.5)),
    'eyebrow1': PRED_TYPE(slice(17, 22), (1.0, 0.498, 0.055, 0.4)),
    'eyebrow2': PRED_TYPE(slice(22, 27), (1.0, 0.498, 0.055, 0.4)),
    'nose': PRED_TYPE(slice(27, 31), (0.345, 0.239, 0.443, 0.4)),
    'nostril': PRED_TYPE(slice(31, 36), (0.345, 0.239, 0.443, 0.4)),
    'eye1': PRED_TYPE(slice(36, 42), (0.596, 0.875, 0.541, 0.3)),
    'eye2': PRED_TYPE(slice(42, 48), (0.596, 0.875, 0.541, 0.3)),
    'lips': PRED_TYPE(slice(48, 60), (0.596, 0.875, 0.541, 0.3)),
    'teeth': PRED_TYPE(slice(60, 68), (0.596, 0.875, 0.541, 0.4))
}


# ======================
# Utils
# ======================
def create_figure(active_3d):
    fig = plt.figure(figsize=plt.figaspect(0.5), tight_layout=True)
    axs = None  # for 2D mode
    if active_3d:
        # 3D mode configuration
        ax1 = fig.add_subplot(1, 2, 1)
        ax2 = fig.add_subplot(1, 2, 2, projection='3d')
        axs = [ax1, ax2]
    return fig, axs


def plot_images(title, images, tile_shape, savepath):
    fig = plt.figure()
    plt.title(title)
    grid = ImageGrid(fig, 111,  nrows_ncols=tile_shape, share_all=True)

    grid[0].get_yaxis().set_ticks([])
    grid[0].get_xaxis().set_ticks([])

    for i in range(images.shape[0]):
        grd = grid[i]
        grd.imshow(images[i])
    split_fname = os.path.splitext(savepath)
    save_name = split_fname[0] + '_confidence.' + split_fname[1]
    logger.info(f'saved at : {save_name}')
    fig.savefig(save_name)


def visualize_results(axs, image, pts_img, active_3d=False):
    """Visualize results & clear previous output
    """
    # 2D-plot
    if active_3d:
        ax = axs[0]
        ax.clear()
        ax.imshow(image)
        for pred_type in PRED_TYPES.values():
            ax.plot(
                pts_img[pred_type.slice, 0],
                pts_img[pred_type.slice, 1],
                color=pred_type.color,
                marker='o',
                markersize=4,
                linestyle='-',
                lw=2,
            )

        ax.axis('off')

        # 3D-plot
        ax = axs[1]
        ax.clear()
        ax.scatter(
            pts_img[:, 0],
            pts_img[:, 1],
            pts_img[:, 2],
            c='cyan',
            alpha=1.0,
            edgecolor='b',
        )

        for pred_type in PRED_TYPES.values():
            ax.plot3D(
                pts_img[pred_type.slice, 0],
                pts_img[pred_type.slice, 1],
                pts_img[pred_type.slice, 2],
                color='blue'
            )

        ax.view_init(elev=90, azim=90.)
        ax.set_xlim(ax.get_xlim()[::-1])

    else:
        # 2D
        plt.clf()
        plt.imshow(image)
        for pred_type in PRED_TYPES.values():
            plt.plot(
                pts_img[pred_type.slice, 0],
                pts_img[pred_type.slice, 1],
                color=pred_type.color,
                marker='o',
                markersize=4,
                linestyle='-',
                lw=2,
            )
        plt.axis('off')
    return axs


def transform(point, center, scale, resolution, invert=False):
    """Generate and affine transformation matrix.

    Given a set of points, a center, a scale and a targer resolution, the
    function generates and affine transformation matrix. If invert is ``True``
    it will produce the inverse transformation.

    Arguments:
        point {torch.tensor} -- the input 2D point
        center {torch.tensor or numpy.array} -- the center around which to
            perform the transformations
        scale {float} -- the scale of the face/object
        resolution {float} -- the output resolution

    Keyword Arguments:
        invert {bool} -- define wherever the function should produce the direct
            or the inverse transformation matrix (default: {False})
    """
    _pt = np.ones(3)
    _pt[0] = point[0]
    _pt[1] = point[1]

    h = scale  # NOTE: originally, scale * 200
    t = np.eye(3)
    t[0, 0] = resolution / h
    t[1, 1] = resolution / h
    t[0, 2] = resolution * (-center[0] / h + 0.5)
    t[1, 2] = resolution * (-center[1] / h + 0.5)

    if invert:
        t = np.linalg.inv(t)
    new_point = (np.dot(t, _pt))[0:2]
    return new_point.astype(int)


def get_preds_from_hm(hm):
    """
    Obtain (x,y) coordinates given a set of N heatmaps.
    ref: 1adrianb/face-alignment/blob/master/face_alignment/utils.py

    TODO: docstring

    Parameters
    ----------
    hm : np.array

    Returns
    -------
    preds:
    preds_orig:

    """
    idx = np.argmax(
        hm.reshape(hm.shape[0], hm.shape[1], hm.shape[2] * hm.shape[3]), axis=2
    )
    idx += 1
    preds = idx.reshape(idx.shape[0], idx.shape[1], 1)
    preds = np.tile(preds, (1, 1, 2)).astype(float)
    preds[..., 0] = (preds[..., 0] - 1) % hm.shape[3] + 1
    preds[..., 1] = np.floor((preds[..., 1] - 1) / (hm.shape[2])) + 1

    for i in range(preds.shape[0]):
        for j in range(preds.shape[1]):
            hm_ = hm[i, j, :]
            pX, pY = int(preds[i, j, 0]) - 1, int(preds[i, j, 1]) - 1
            if pX > 0 and pX < 63 and pY > 0 and pY < 63:
                diff = np.array(
                    [hm_[pY, pX + 1] - hm_[pY, pX - 1],
                     hm_[pY + 1, pX] - hm_[pY - 1, pX]]).astype(float)
                preds[i, j] = preds[i, j] + (np.sign(diff) * 0.25)

    preds += -0.5
    preds_orig = np.zeros_like(preds)

    for i in range(hm.shape[0]):
        for j in range(hm.shape[1]):
            preds_orig[i, j] = transform(
                preds[i, j],  # point
                np.array([IMAGE_HEIGHT // 2, IMAGE_WIDTH // 2]),  # center
                (IMAGE_HEIGHT + IMAGE_WIDTH) // 2,  # FIXME not sure... # scale
                hm.shape[2],  # resolution
                True,
            )
    return preds, preds_orig


def _gaussian(
        size=3,
        sigma=0.25,
        amplitude=1,
        normalize=False,
        width=None,
        height=None,
        sigma_horz=None,
        sigma_vert=None,
        mean_horz=0.5,
        mean_vert=0.5
):
    # handle some defaults
    if width is None:
        width = size
    if height is None:
        height = size
    if sigma_horz is None:
        sigma_horz = sigma
    if sigma_vert is None:
        sigma_vert = sigma
    center_x = mean_horz * width + 0.5
    center_y = mean_vert * height + 0.5
    gauss = np.empty((height, width), dtype=np.float32)
    # generate kernel
    for i in range(height):
        for j in range(width):
            gauss[i][j] = amplitude * math.exp(-(math.pow(
                (j + 1 - center_x) / (sigma_horz * width), 2
            ) / 2.0 + math.pow(
                (i + 1 - center_y) / (sigma_vert * height), 2
            ) / 2.0))
    if normalize:
        gauss = gauss / np.sum(gauss)
    return gauss


def draw_gaussian(img, point, sigma):
    # Check if the gaussian is inside
    ul = [math.floor(point[0] - 3 * sigma), math.floor(point[1] - 3 * sigma)]
    br = [math.floor(point[0] + 3 * sigma), math.floor(point[1] + 3 * sigma)]
    if ul[0] > img.shape[1] or ul[1] > img.shape[0] or br[0] < 1 or br[1] < 1:
        return img
    size = 6 * sigma + 1
    g = _gaussian(size)
    g_x = [
        int(max(1, -ul[0])),
        int(min(br[0], img.shape[1])) - int(max(1, ul[0])) +
        int(max(1, -ul[0]))
    ]
    g_y = [
        int(max(1, -ul[1])),
        int(min(br[1], img.shape[0])) - int(max(1, ul[1])) +
        int(max(1, -ul[1]))
    ]
    img_x = [int(max(1, ul[0])), int(min(br[0], img.shape[1]))]
    img_y = [int(max(1, ul[1])), int(min(br[1], img.shape[0]))]
    assert (g_x[0] > 0 and g_y[1] > 0)
    img[img_y[0] - 1:img_y[1], img_x[0] - 1:img_x[1]] = img[
        img_y[0] - 1:img_y[1], img_x[0] - 1:img_x[1]
    ] + g[g_y[0] - 1:g_y[1], g_x[0] - 1:g_x[1]]
    img[img > 1] = 1
    return img


# ======================
# Main functions
# ======================
def recognize_from_image():
    # net initialize
    net = ailia.Net(MODEL_PATH, WEIGHT_PATH, env_id=args.env_id, memory_mode=args.memory_mode)
    if args.active_3d:
        logger.info('>>> 3D mode is activated!')
        depth_net = ailia.Net(
            DEPTH_MODEL_PATH, DEPTH_WEIGHT_PATH, env_id=args.env_id,
            memory_mode=args.memory_mode
        )

    # input image loop
    for image_path in args.input:
        # prepare input data
        logger.info(image_path)
        input_img = imread(image_path)
        data = load_image(
            image_path,
            (IMAGE_HEIGHT, IMAGE_WIDTH),
            normalize_type='255',
            gen_input_ailia=True
        )

        # inference
        logger.info('Start inference...')
        if args.benchmark:
            logger.info('BENCHMARK mode')
            for i in range(5):
                start = int(round(time.time() * 1000))
                preds_ailia = net.predict(data)
                end = int(round(time.time() * 1000))
                logger.info(f'\tailia processing time {end - start} ms')
        else:
            preds_ailia = net.predict(data)

        pts, pts_img = get_preds_from_hm(preds_ailia)
        pts, pts_img = pts.reshape(68, 2) * 4, pts_img.reshape(68, 2)

        input_img = cv2.resize(
            cv2.cvtColor(input_img, cv2.COLOR_BGR2RGB),
            (IMAGE_WIDTH, IMAGE_HEIGHT)
        )

        if args.active_3d:
            # 3D mode
            heatmaps = np.zeros(
                (68, IMAGE_HEIGHT, IMAGE_WIDTH), dtype=np.float32
            )
            for i in range(68):
                if pts[i, 0] > 0:
                    heatmaps[i] = draw_gaussian(heatmaps[i], pts[i], 2)
            heatmaps = heatmaps[np.newaxis, :, :, :]
            depth_pred = depth_net.predict(np.concatenate((data, heatmaps), 1))
            depth_pred = depth_pred.reshape(68, 1)
            pts_img = np.concatenate((pts_img, depth_pred * 2), 1)

        fig, axs = create_figure(active_3d=args.active_3d)
        axs = visualize_results(
            axs, input_img, pts_img, active_3d=args.active_3d
        )
        savepath = get_savepath(args.savepath, image_path)
        logger.info(f'saved at : {savepath}')
        fig.savefig(savepath)

        # Confidence Map
        channels = preds_ailia[0].shape[0]
        cols = 8
        plot_images(
            'confidence',
            preds_ailia[0],
            tile_shape=((int)((channels+cols-1)/cols), cols),
            savepath=savepath,
        )
    logger.info('Script finished successfully.')


def recognize_from_video():
    # net initialize
    net = ailia.Net(MODEL_PATH, WEIGHT_PATH, env_id=args.env_id, memory_mode=args.memory_mode)
    if args.active_3d:
        logger.info('>>> 3D mode is activated!')
        depth_net = ailia.Net(
            DEPTH_MODEL_PATH, DEPTH_WEIGHT_PATH, env_id=args.env_id,
            memory_mode=args.memory_mode
        )
    detector = ailia.Net(FACE_MODEL_PATH, FACE_WEIGHT_PATH, env_id=args.env_id, memory_mode=args.memory_mode)

    capture = webcamera_utils.get_capture(args.video)

    # create video writer if savepath is specified as video format
    if args.savepath != SAVE_IMAGE_PATH:
        logger.warning('[WARNING] currently video results output feature '
                       'is not supported in this model!')
        # TODO: shape should be debugged!
        f_h = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        f_w = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        writer = webcamera_utils.get_writer(args.savepath, f_h, f_w)
    else:
        writer = None

    fig, axs = create_figure(active_3d=args.active_3d)

    while(True):
        ret, frame = capture.read()
        if (cv2.waitKey(1) & 0xFF == ord('q')) or not ret:
            break

        # detect face
        detections = compute_blazeface(
            detector,
            frame,
            anchor_path='../../face_detection/blazeface/anchors.npy',
        )

        # get detected face
        if len(detections) == 0:
            crop_img = frame
        else:
            crop_img, top_left, bottom_right = crop_blazeface(
                detections[0], FACE_MARGIN, frame
            )
            if crop_img.shape[0] <= 0 or crop_img.shape[1] <= 0:
                crop_img = frame

        # preprocess
        input_image, input_data = webcamera_utils.preprocess_frame(
            crop_img, IMAGE_HEIGHT, IMAGE_WIDTH, normalize_type='255'
        )

        # inference
        preds_ailia = net.predict(input_data)

        pts, pts_img = get_preds_from_hm(preds_ailia)
        pts, pts_img = pts.reshape(68, 2) * 4, pts_img.reshape(68, 2)

        if args.active_3d:
            # 3D mode
            heatmaps = np.zeros(
                (68, IMAGE_HEIGHT, IMAGE_WIDTH), dtype=np.float32
            )
            for i in range(68):
                if pts[i, 0] > 0:
                    heatmaps[i] = draw_gaussian(heatmaps[i], pts[i], 2)
            heatmaps = heatmaps[np.newaxis, :, :, :]
            depth_pred = depth_net.predict(
                np.concatenate((input_data, heatmaps), 1)
            )
            depth_pred = depth_pred.reshape(68, 1)
            pts_img = np.concatenate((pts_img, depth_pred * 2), 1)

        resized_img = cv2.resize(
            cv2.cvtColor(input_image, cv2.COLOR_BGR2RGB),
            (IMAGE_WIDTH, IMAGE_HEIGHT)
        )

        # visualize results (clear axs at first)
        axs = visualize_results(
            axs, resized_img, pts_img, active_3d=args.active_3d
        )
        plt.pause(0.01)
        if not plt.get_fignums():
            break

        # save results
        # FIXME: How to save plt --> cv2.VideoWriter()
        # if writer is not None:
        #     # put pixel buffer in numpy array
        #     canvas = FigureCanvas(fig)
        #     canvas.draw()
        #     mat = np.array(canvas.renderer._renderer)
        #     res_img = cv2.cvtColor(mat, cv2.COLOR_RGB2BGR)
        #     writer.write(res_img)

    capture.release()
    cv2.destroyAllWindows()
    if writer is not None:
        writer.release()
    logger.info('Script finished successfully.')


def main():
    # model files check and download
    check_and_download_models(WEIGHT_PATH, MODEL_PATH, REMOTE_PATH)

    if args.active_3d:
        check_and_download_models(
            DEPTH_WEIGHT_PATH, DEPTH_MODEL_PATH, REMOTE_PATH
        )

    if args.video:
        check_and_download_models(
            FACE_WEIGHT_PATH, FACE_MODEL_PATH, FACE_REMOTE_PATH
        )

    if args.video is not None:
        # video mode
        recognize_from_video()
    else:
        # image mode
        recognize_from_image()


if __name__ == '__main__':
    main()
