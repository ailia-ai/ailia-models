from utils_6d_repnet.functions import RetinaFaceOnnx
import os
import sys
import numpy as np
import cv2
from utils_6d_repnet import utils
import matplotlib
from PIL import Image
import time
import onnxruntime as ort
matplotlib.use('TkAgg')

import ailia

# import original modules
sys.path.append('../../util')
# logger
from logging import getLogger

import webcamera_utils
from detector_utils import plot_results, reverse_letterbox
from image_utils import imread  # noqa: E402
from model_utils import check_and_download_models
from arg_utils import get_base_parser, get_savepath, update_parser

logger = getLogger(__name__)

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'


# ======================
# Parameters
# ======================
REMOTE_PATH_FACE = 'https://storage.googleapis.com/ailia-models/6d_repnet/'
REMOTE_PATH_6DRepNet = 'https://storage.googleapis.com/ailia-models/6d_repnet/'

# settings
WEIGHT_PATH_FACE = "RetinaFace.opt.onnx"
MODEL_PATH_FACE = "RetinaFace.opt.onnx.prototxt"

WEIGHT_PATH_6DRepNet = "6DRepNet.opt.onnx"
MODEL_PATH_6DRepNet = "6DRepNet.opt.onnx.prototxt"

IMAGE_PATH = 'input.png'
SAVE_IMAGE_PATH = 'output.png'

MEAN=[0.485, 0.456, 0.406]
STD=[0.229, 0.224, 0.225]

# Default input size
HEIGHT = 224
WIDTH = 224

# ======================
# Arguemnt Parser Config
# ======================
parser = get_base_parser('6DRepNet model', IMAGE_PATH, SAVE_IMAGE_PATH)
parser.add_argument(
    '-w', '--write_json',
    action='store_true',
    help='Flag to output results to json file.'
)
args = update_parser(parser)

# ======================
# Main functions
# ======================
def recognize_from_image():
    env_id = args.env_id
    net = ailia.Net(MODEL_PATH_6DRepNet, WEIGHT_PATH_6DRepNet, env_id=env_id)
    face_detect = ailia.Net(MODEL_PATH_FACE, WEIGHT_PATH_FACE, env_id=env_id)
    detector = RetinaFaceOnnx(face_detect)

    # input image loop
    for image_path in args.input:
        # prepare input data
        logger.debug(f'input image: {image_path}')
        results = []
        raw_img = cv2.imread(image_path)
        resize_img = cv2.resize(raw_img, dsize=(640, 480))
        resize_img = np.array(resize_img)
        logger.debug(f'input image shape: {resize_img.shape}')

        # inference
        logger.info('Start inference...')
        faces = detector(resize_img)
        for box, landmarks, score in faces:
            # Print the location of each face in this image
            if score < .95:
                continue
            x_min = int(box[0])
            y_min = int(box[1])
            x_max = int(box[2])
            y_max = int(box[3])
            bbox_width = abs(x_max - x_min)
            bbox_height = abs(y_max - y_min)

            x_min = max(0, x_min - int(0.2 * bbox_height))
            y_min = max(0, y_min - int(0.2 * bbox_width))
            x_max = x_max + int(0.2 * bbox_height)
            y_max = y_max + int(0.2 * bbox_width)

            img = resize_img[y_min:y_max, x_min:x_max]
            img = cv2.resize(img, dsize=(HEIGHT, WIDTH))
            img = utils.transform(img, MEAN, STD)

            img = np.expand_dims(img, 0)
            img = np.array(img, dtype='float32')

            c = cv2.waitKey(1)
            if c == 27:
                break

            start = time.time()

            R_pred = net.run(img)[0]
            end = time.time()
            print('Head pose estimation: %2f ms' % ((end - start) * 1000.))

            euler = utils.compute_euler_angles_from_rotation_matrices(R_pred) * 180 / np.pi
            p_pred_deg = euler[:, 0][0]
            y_pred_deg = euler[:, 1][0]
            r_pred_deg = euler[:, 2][0]
            results.append({'yaw': y_pred_deg, 'pitch': p_pred_deg, 'roll': r_pred_deg})

            utils.plot_pose_cube(resize_img, y_pred_deg, p_pred_deg, r_pred_deg, x_min + int(.5 * (
                    x_max - x_min)), y_min + int(.5 * (y_max - y_min)), size=bbox_width)

        # plot result
        savepath = get_savepath(args.savepath, image_path)
        logger.info(f'saved at : {savepath}')
        resize_img = cv2.cvtColor(resize_img, cv2.COLOR_BGR2RGB)
        Image.fromarray(resize_img).save(savepath)

        if args.write_json:
            json_file = '%s.json' % savepath.rsplit('.', 1)[0]
            utils.save_json_result(json_file, results)

    logger.info('Script finished successfully.')

def recognize_from_video():
    env_id = args.env_id
    net = ailia.Net(MODEL_PATH_6DRepNet, WEIGHT_PATH_6DRepNet, env_id=env_id)
    face_detect = ailia.Net(MODEL_PATH_FACE, WEIGHT_PATH_FACE, env_id=env_id)
    detector = RetinaFaceOnnx(face_detect)

    capture = webcamera_utils.get_capture(args.video)

    # create video writer if savepath is specified as video format
    if args.savepath != SAVE_IMAGE_PATH:
        logger.warning(
            'currently, video results cannot be output correctly...'
        )
        f_h = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        f_w = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        save_h, save_w = f_h, f_w
        writer = webcamera_utils.get_writer(args.savepath, save_h, save_w)
    else:
        writer = None

    frame_shown = False
    while (True):
        ret, frame = capture.read()
        if (cv2.waitKey(1) & 0xFF == ord('q')) or not ret:
            break
        if frame_shown and cv2.getWindowProperty('frame', cv2.WND_PROP_VISIBLE) == 0:
            break

        frame = cv2.resize(frame, dsize=(640, 480))
        faces = detector(frame)
        for box, landmarks, score in faces:
            # Print the location of each face in this image
            if score < .95:
                continue
            x_min = int(box[0])
            y_min = int(box[1])
            x_max = int(box[2])
            y_max = int(box[3])
            bbox_width = abs(x_max - x_min)
            bbox_height = abs(y_max - y_min)

            x_min = max(0, x_min - int(0.2 * bbox_height))
            y_min = max(0, y_min - int(0.2 * bbox_width))
            x_max = x_max + int(0.2 * bbox_height)
            y_max = y_max + int(0.2 * bbox_width)

            img = frame[y_min:y_max, x_min:x_max]
            img = cv2.resize(img, dsize=(HEIGHT, WIDTH))
            img = utils.transform(img, MEAN, STD)

            img = np.expand_dims(img, 0)
            img = np.array(img, dtype='float32')

            c = cv2.waitKey(1)
            if c == 27:
                break

            start = time.time()

            R_pred = net.run(img)[0]
            end = time.time()
            print('Head pose estimation: %2f ms' % ((end - start) * 1000.))

            euler = utils.compute_euler_angles_from_rotation_matrices(R_pred) * 180 / np.pi
            p_pred_deg = euler[:, 0]
            y_pred_deg = euler[:, 1]
            r_pred_deg = euler[:, 2]

            utils.plot_pose_cube(frame, y_pred_deg, p_pred_deg, r_pred_deg, x_min + int(.5 * (
                    x_max - x_min)), y_min + int(.5 * (y_max - y_min)), size=bbox_width)

        cv2.imshow("Demo", frame)
        cv2.waitKey(5)

        # save results
        if writer is not None:
            writer.write(frame)

    capture.release()
    cv2.destroyAllWindows()
    if writer is not None:
        writer.release()
    logger.info('Script finished successfully.')


def main():
    # model files check and download
    check_and_download_models(WEIGHT_PATH_6DRepNet, MODEL_PATH_6DRepNet, REMOTE_PATH_6DRepNet)
    check_and_download_models(WEIGHT_PATH_FACE, MODEL_PATH_FACE, REMOTE_PATH_FACE)

    if args.video is not None:
        # video mode
        recognize_from_video()
    else:
        # image mode
        recognize_from_image()


if __name__ == '__main__':
    main()
