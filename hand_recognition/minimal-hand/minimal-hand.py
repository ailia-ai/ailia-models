import sys
import time
import argparse

import numpy as np
import cv2
import matplotlib.pyplot as plt

import ailia

# import original modules
sys.path.append('../../util')
from arg_utils import get_base_parser, update_parser, get_savepath  # noqa: E402
from model_utils import check_and_download_models  # noqa: E402
from detector_utils import load_image  # noqa: E402
from webcamera_utils import get_capture  # noqa: E402

# logger
from logging import getLogger  # noqa: E402

logger = getLogger(__name__)

# ======================
# Parameters
# ======================

WEIGHT_DET_PATH = './detnet.onnx'
MODEL_DET_PATH = './detnet.onnx.prototxt'
REMOTE_PATH = \
    'https://storage.googleapis.com/ailia-models/minimal-hand/'

IMAGE_PATH = 'demo.jpg'
SAVE_IMAGE_PATH = 'output.png'

IMAGE_SIZE = 256

SLEEP_TIME = 0.01


class MPIIHandJoints:
    n_joints = 21

    labels = [
        'W',  # 0
        'T0', 'T1', 'T2', 'T3',  # 4
        'I0', 'I1', 'I2', 'I3',  # 8
        'M0', 'M1', 'M2', 'M3',  # 12
        'R0', 'R1', 'R2', 'R3',  # 16
        'L0', 'L1', 'L2', 'L3',  # 20
    ]

    parents = [
        None,
        0, 1, 2, 3,
        0, 5, 6, 7,
        0, 9, 10, 11,
        0, 13, 14, 15,
        0, 17, 18, 19
    ]


# ======================
# Arguemnt Parser Config
# ======================

parser = get_base_parser('minimal-hand mode', IMAGE_PATH, SAVE_IMAGE_PATH)
parser.add_argument(
    '--right-hand',
    dest='flip',
    action='store_true',
    help='right hand flag.'
)
parser.add_argument(
    '--onnx',
    action='store_true',
    help='execute onnxruntime version.'
)
args = update_parser(parser)


# ======================
# Secondaty Functions
# ======================


def preprocess(img, img_size):
    if img.shape[0] > img.shape[1]:
        margin = int((img.shape[0] - img.shape[1]) / 2)
        img = img[margin:-margin]
    else:
        if img.shape[0] < img.shape[1]:
            margin = int((img.shape[1] - img.shape[0]) / 2)
            img = img[:, margin:-margin]

    if args.flip:
        img = np.flip(img, axis=1).copy()

    img = cv2.resize(img, img_size, cv2.INTER_LINEAR)

    return img


def visualize_results(axs, img, keypts):
    """Visualize results & clear previous output
    """
    img = preprocess(img, (IMAGE_SIZE, IMAGE_SIZE))

    ax = axs[0]
    ax.clear()
    ax.imshow(img)
    ax.axis('off')

    _len = (IMAGE_SIZE * 0.9) // 2
    keypts *= _len
    keypts += _len

    # 3D-plot
    ax = axs[1]
    ax.clear()
    ax.scatter(
        keypts[:, 0],
        keypts[:, 1],
        keypts[:, 2],
        c='cyan',
        alpha=1.0,
        edgecolor='b',
    )

    lable_idx = {v: i for i, v in enumerate(MPIIHandJoints.labels)}
    for s in 'TIMRL':
        a = filter(lambda x: x.startswith('W') or x.startswith(s), MPIIHandJoints.labels)
        sel = list(map(lambda i: lable_idx[i], a))
        pts = keypts[sel]
        ax.plot3D(
            pts[:, 0], pts[:, 1], pts[:, 2],
            color='blue'
        )

    ax.view_init(elev=90, azim=90.)
    ax.set_xlim(ax.get_xlim()[::-1])

    return axs


# ======================
# Main functions
# ======================


def predict(img, det_model):
    # initial preprocesses
    img = preprocess(img, (128, 128))

    # feedforward
    if not args.onnx:
        output = det_model.predict({
            'import/prior_based_hand/Placeholder:0': img
        })
        xyz, uv = output
    else:
        input_name = det_model.get_inputs()[0].name
        xyz_name = det_model.get_outputs()[0].name
        uv_name = det_model.get_outputs()[1].name
        output = det_model.run([xyz_name, uv_name],
                               {input_name: img})
        xyz, uv = output

    return xyz


def recognize_from_image(filename, det_model):
    # input image loop
    for image_path in args.input:
        logger.info(image_path)

        # prepare input data
        img = load_image(image_path)
        print(f'input image shape: {img.shape}')

        img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)

        # inference
        print('Start inference...')
        if args.benchmark:
            print('BENCHMARK mode')
            for i in range(5):
                start = int(round(time.time() * 1000))
                xyz = predict(img, det_model)
                end = int(round(time.time() * 1000))
                print(f'\tailia processing time {end - start} ms')
        else:
            xyz = predict(img, det_model)

        """
        plot result
        """
        fig = plt.figure(figsize=plt.figaspect(0.5), tight_layout=True)
        ax1 = fig.add_subplot(1, 2, 1)
        ax2 = fig.add_subplot(1, 2, 2, projection='3d')
        axs = [ax1, ax2]

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        visualize_results(axs, img, xyz)

        savepath = get_savepath(args.savepath, image_path, ext='.png')
        fig.savefig(savepath)

    print('Script finished successfully.')


def recognize_from_video(video, det_model):
    capture = get_capture(video)

    fig = plt.figure(figsize=plt.figaspect(0.5), tight_layout=True)
    ax1 = fig.add_subplot(1, 2, 1)
    ax2 = fig.add_subplot(1, 2, 2, projection='3d')
    axs = [ax1, ax2]

    while (True):
        ret, frame = capture.read()
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        if not ret:
            continue

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        xyz = predict(frame, det_model)

        # visualize results (clear axs at first)
        visualize_results(axs, frame, xyz)
        plt.pause(SLEEP_TIME)
        if not plt.get_fignums():
            break

    capture.release()
    cv2.destroyAllWindows()
    print('Script finished successfully.')


def main():
    # model files check and download
    check_and_download_models(WEIGHT_DET_PATH, MODEL_DET_PATH, REMOTE_PATH)

    # initialize
    if not args.onnx:
        det_model = ailia.Net(MODEL_DET_PATH, WEIGHT_DET_PATH, env_id=args.env_id, memory_mode=args.memory_mode)
        # ik_model = ailia.Net(MODEL_IK_PATH, WEIGHT_IK_PATH, env_id=args.env_id)
    else:
        import onnxruntime
        det_model = onnxruntime.InferenceSession(WEIGHT_DET_PATH)
        # ik_model = onnxruntime.InferenceSession(WEIGHT_IK_PATH)

    if args.video is not None:
        recognize_from_video(args.video, det_model)
    else:
        recognize_from_image(args.input, det_model)


if __name__ == '__main__':
    main()
