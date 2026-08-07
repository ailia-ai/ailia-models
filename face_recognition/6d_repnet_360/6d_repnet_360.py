from utils_6d_repnet_360.functions import RetinaFaceOnnx
import os
import sys
import numpy as np
import cv2
from utils_6d_repnet_360 import utils
import matplotlib
from PIL import Image
import time
import onnxruntime as ort
matplotlib.use('TkAgg')

import ailia

sys.path.append('../../util')
from logging import getLogger

import webcamera_utils
from detector_utils import plot_results, reverse_letterbox
from image_utils import imread
from model_utils import check_and_download_models
from arg_utils import get_base_parser, get_savepath, update_parser

logger = getLogger(__name__)

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'


REMOTE_PATH_FACE = 'https://storage.googleapis.com/ailia-models/6d_repnet/'
REMOTE_PATH_6DRepNet360 = 'https://storage.googleapis.com/ailia-models/6d_repnet_360/'

WEIGHT_PATH_FACE = "RetinaFace.opt.onnx"
MODEL_PATH_FACE = "RetinaFace.opt.onnx.prototxt"

WEIGHT_PATH_6DRepNet360 = "6DRepNet360.onnx"
MODEL_PATH_6DRepNet360 = "6DRepNet360.onnx.prototxt"

IMAGE_PATH = 'input.png'
SAVE_IMAGE_PATH = 'output.png'

MEAN=[0.485, 0.456, 0.406]
STD=[0.229, 0.224, 0.225]

HEIGHT = 224
WIDTH = 224

parser = get_base_parser('6DRepNet360 model', IMAGE_PATH, SAVE_IMAGE_PATH)
parser.add_argument(
    '-w', '--write_json',
    action='store_true',
    help='Flag to output results to json file.'
)
args = update_parser(parser)

def resize_and_pad_with_aspect_ratio(raw_img, target_width=640, target_height=480, padding_color=(0, 0, 0)):
    # 元の画像の幅と高さ
    original_height, original_width = raw_img.shape[:2]

    # 幅と高さの比率を計算
    width_ratio = target_width / original_width
    height_ratio = target_height / original_height

    # 小さい方の比率を使ってリサイズを行う
    if width_ratio < height_ratio:
        new_width = target_width
        new_height = int(original_height * width_ratio)
    else:
        new_width = int(original_width * height_ratio)
        new_height = target_height

    # 新しいサイズにリサイズ
    resized_img = cv2.resize(raw_img, (new_width, new_height))
    
    # パディングするための新しい画像を作成
    padded_img = np.full((target_height, target_width, 3), padding_color, dtype=np.uint8)
    
    # パディングされた画像の中央にリサイズされた画像を配置
    x_offset = (target_width - new_width) // 2
    y_offset = (target_height - new_height) // 2
    padded_img[y_offset:y_offset+new_height, x_offset:x_offset+new_width] = resized_img

    return padded_img

def recognize_from_image():
    env_id = args.env_id
    net = ailia.Net(MODEL_PATH_6DRepNet360, WEIGHT_PATH_6DRepNet360, env_id=env_id)
    face_detect = ailia.Net(MODEL_PATH_FACE, WEIGHT_PATH_FACE, env_id=env_id)
    detector = RetinaFaceOnnx(face_detect)

    for image_path in args.input:
        logger.debug(f'input image: {image_path}')
        results = []
        raw_img = cv2.imread(image_path)
        resize_img = resize_and_pad_with_aspect_ratio(raw_img)
        resize_img = np.array(resize_img)
        logger.debug(f'input image shape: {resize_img.shape}')

        logger.info('Start inference...')
        faces = detector(resize_img)
        for box, landmarks, score in faces:
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
    net = ailia.Net(MODEL_PATH_6DRepNet360, WEIGHT_PATH_6DRepNet360, env_id=env_id)
    face_detect = ailia.Net(MODEL_PATH_FACE, WEIGHT_PATH_FACE, env_id=env_id)
    detector = RetinaFaceOnnx(face_detect)

    capture = webcamera_utils.get_capture(args.video)

    if args.savepath != SAVE_IMAGE_PATH:
        logger.warning(
            'currently, video results cannot be output correctly...'
        )
        f_h = 480
        f_w = 640
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

        frame = resize_and_pad_with_aspect_ratio(frame)
        faces = detector(frame)
        for box, landmarks, score in faces:
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

        if writer is not None:
            writer.write(frame)

    capture.release()
    cv2.destroyAllWindows()
    if writer is not None:
        writer.release()
    logger.info('Script finished successfully.')


def main():
    check_and_download_models(WEIGHT_PATH_6DRepNet360, MODEL_PATH_6DRepNet360, REMOTE_PATH_6DRepNet360)
    check_and_download_models(WEIGHT_PATH_FACE, MODEL_PATH_FACE, REMOTE_PATH_FACE)

    if args.video is not None:
        recognize_from_video()
    else:
        recognize_from_image()


if __name__ == '__main__':
    main()
