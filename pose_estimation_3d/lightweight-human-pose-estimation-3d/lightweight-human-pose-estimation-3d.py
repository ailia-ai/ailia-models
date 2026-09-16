import os
import sys
import time
import json

import cv2
import numpy as np

from modules.input_reader import VideoReader, ImageReader
from modules.draw import Plotter3d, draw_poses
from modules.parse_poses import parse_poses

import ailia

# import original modules
sys.path.append('../../util')
from arg_utils import get_base_parser, update_parser, get_savepath  # noqa: E402
from arg_utils import check_file_existance  # noqa: E402
from model_utils import check_and_download_models  # noqa: E402
import webcamera_utils  # noqa: E402

# logger
from logging import getLogger   # noqa: E402
logger = getLogger(__name__)


# ======================
# Parameters
# ======================
WEIGHT_PATH = 'human-pose-estimation-3d.onnx'
MODEL_PATH = 'human-pose-estimation-3d.onnx.prototxt'
REMOTE_PATH = 'https://storage.googleapis.com/ailia-models/lightweight-human-pose-estimation-3d/'

IMAGE_PATH = 'input.jpg'
SAVE_IMAGE_PATH = 'output.png'
FILE_PATH = 'extrinsics.json'
IMAGE_HEIGHT = 256
IMAGE_WIDTH = 448

STRIDE = 8


# ======================
# Arguemnt Parser Config
# ======================
parser = get_base_parser(
    ('Lightweight 3D human pose estimation demo. '
     'Press esc to exit, "p" to (un)pause video or process next image.'),
    IMAGE_PATH,
    SAVE_IMAGE_PATH,
)
parser.add_argument(
    '--rotate3d', action='store_true', default=False,
    help='allowing 3D canvas rotation while on pause',
)
parser.add_argument(
    '--put_fps', action='store_true', default=False,
    help='put FPS on result image',
)
args = update_parser(parser)


# ======================
# Utils
# ======================
def rotate_poses(poses_3d, R, t):
    R_inv = np.linalg.inv(R)
    for pose_id in range(len(poses_3d)):
        pose_3d = poses_3d[pose_id].reshape((-1, 4)).transpose()
        pose_3d[0:3, :] = np.dot(R_inv, pose_3d[0:3, :] - t)
        poses_3d[pose_id] = pose_3d.transpose().reshape(-1)

    return poses_3d


# ======================
# Main functions
# ======================
def main():
    # model files check and download
    check_and_download_models(WEIGHT_PATH, MODEL_PATH, REMOTE_PATH)
    check_file_existance(FILE_PATH)

    # prepare input data
    with open(FILE_PATH, 'r') as f:
        extrinsics = json.load(f)

    R = np.array(extrinsics['R'], dtype=np.float32)
    t = np.array(extrinsics['t'], dtype=np.float32)

    if args.video is None:
        frame_provider = ImageReader(args.input)
        is_video = False
    else:
        frame_provider = VideoReader(args.video)
        is_video = True

    canvas_3d = np.zeros((720, 1280, 3), dtype=np.uint8)
    canvas_3d_window_name = 'Canvas3D'
    plotter = Plotter3d(canvas_3d.shape[:2])
    if is_video or args.rotate3d:
        cv2.namedWindow(canvas_3d_window_name)
        cv2.setMouseCallback(canvas_3d_window_name, Plotter3d.mouse_callback)

    fx = -1
    delay = 1
    esc_code = 27
    p_code = 112
    q_code = 113
    space_code = 32
    mean_time = 0
    img_mean = np.array([128, 128, 128], dtype=np.float32)
    base_width_calculated = False

    # net initialize
    net = ailia.Net(MODEL_PATH, WEIGHT_PATH, env_id=args.env_id, memory_mode=args.memory_mode)

    # inference
    for frame_id, frame in enumerate(frame_provider):
        current_time = cv2.getTickCount()
        if frame is None:
            break

        if frame_id == 0:
            # create video writer if savepath is specified as video format
            if args.savepath != SAVE_IMAGE_PATH and is_video:
                f_h = int(frame_provider.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                f_w = int(frame_provider.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                writer = webcamera_utils.get_writer(args.savepath, f_h, f_w)
            else:
                writer = None

        if not base_width_calculated:
            IMAGE_WIDTH = frame.shape[1]*(IMAGE_HEIGHT/frame.shape[0])
            IMAGE_WIDTH = int(IMAGE_WIDTH/STRIDE)*STRIDE
            net.set_input_shape((1, 3, IMAGE_HEIGHT, IMAGE_WIDTH))
            base_width_calculated = True

        input_scale = IMAGE_HEIGHT / frame.shape[0]
        scaled_img = cv2.resize(
            frame, dsize=None, fx=input_scale, fy=input_scale
        )
        # better to pad, but cut out for demo
        scaled_img = scaled_img[:, 0:scaled_img.shape[1] -
                                (scaled_img.shape[1] % STRIDE)]

        if fx < 0:  # Focal length is unknown
            fx = np.float32(0.8 * frame.shape[1])

        normalized_img = (scaled_img.astype(np.float32) - img_mean) / 255.0
        normalized_img = np.expand_dims(
            normalized_img.transpose(2, 0, 1), axis=0
        )

        # execution
        if is_video:
            input_blobs = net.get_input_blob_list()
            net.set_input_blob_data(normalized_img, input_blobs[0])
            net.update()
            features, heatmaps, pafs = net.get_results()

        else:
            logger.info('Start inference...')
            if args.benchmark:
                logger.info('BENCHMARK mode')
                for i in range(5):
                    start = int(round(time.time() * 1000))
                    features, heatmaps, pafs = net.predict([normalized_img])
                    end = int(round(time.time() * 1000))
                    logger.info(f'\tailia processing time {end - start} ms')
            else:
                features, heatmaps, pafs = net.predict([normalized_img])

        inference_result = (
            features[-1].squeeze(),
            heatmaps[-1].squeeze(),
            pafs[-1].squeeze()
        )

        poses_3d, poses_2d = parse_poses(
            inference_result,
            input_scale,
            STRIDE,
            fx,
            is_video
        )
        edges = []
        if len(poses_3d):
            poses_3d = rotate_poses(poses_3d, R, t)
            poses_3d_copy = poses_3d.copy()
            x = poses_3d_copy[:, 0::4]
            y = poses_3d_copy[:, 1::4]
            z = poses_3d_copy[:, 2::4]
            poses_3d[:, 0::4], poses_3d[:, 1::4], poses_3d[:, 2::4] = -z, x, -y

            poses_3d = poses_3d.reshape(poses_3d.shape[0], 19, -1)[:, :, 0:3]
            edges = (
                Plotter3d.SKELETON_EDGES +
                19 * np.arange(poses_3d.shape[0]).reshape((-1, 1, 1))
            ).reshape((-1, 2))
        plotter.plot(canvas_3d, poses_3d, edges)

        if is_video:
            cv2.imshow(canvas_3d_window_name, canvas_3d)
            # save results
            if writer is not None:
                writer.write(canvas_3d)
        else:
            cv2.imwrite(os.path.join(
                os.path.dirname(args.savepath), f'Canvas3D_{frame_id}.png'
            ), canvas_3d)

        draw_poses(frame, poses_2d)
        current_time = (cv2.getTickCount()-current_time)/cv2.getTickFrequency()
        if mean_time == 0:
            mean_time = current_time
        else:
            mean_time = mean_time * 0.95 + current_time * 0.05
        if args.put_fps:
            cv2.putText(frame, 'FPS: {}'.format(int(1 / mean_time * 10) / 10),
                        (40, 80), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255))

        if is_video:
            cv2.imshow('ICV 3D Human Pose Estimation', frame)
        else:
            savepath = get_savepath(args.savepath, args.input[frame_id])
            logger.info(f'saved at : {savepath}')
            cv2.imwrite(savepath, frame)

        if is_video or args.rotate3d:
            key = cv2.waitKey(delay)
            if key == esc_code or key == q_code:
                break
            if cv2.getWindowProperty('ICV 3D Human Pose Estimation', cv2.WND_PROP_VISIBLE) == 0:
                break
            if cv2.getWindowProperty(canvas_3d_window_name, cv2.WND_PROP_VISIBLE) == 0:
                break
            if key == p_code:
                if delay == 1:
                    delay = 0
                else:
                    delay = 1

            if delay == 0 and args.rotate3d:
                key = 0
                while (key != p_code
                       and key != esc_code
                       and key != q_code
                       and key != space_code):
                    plotter.plot(canvas_3d, poses_3d, edges)
                    cv2.imshow(canvas_3d_window_name, canvas_3d)
                    key = cv2.waitKey(33)
                    if cv2.getWindowProperty(canvas_3d_window_name, cv2.WND_PROP_VISIBLE) == 0:
                        break
                if key == esc_code or key == q_code:
                    break
                elif cv2.getWindowProperty(canvas_3d_window_name, cv2.WND_PROP_VISIBLE) == 0:
                    break
                else:
                    delay = 1

    if writer is not None:
        writer.release()
    logger.info('Script finished successfully.')


if __name__ == '__main__':
    main()
