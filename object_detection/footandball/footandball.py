import sys
import time

import numpy as np
import cv2

import ailia

# import original modules
sys.path.append('../../util')
from arg_utils import get_base_parser, update_parser, get_savepath
from model_utils import check_and_download_models
from detector_utils import load_image
from webcamera_utils import get_capture, get_writer
# logger
from logging import getLogger

logger = getLogger(__name__)

from footandball_utils import *


# ======================
# Parameters
# ======================

WEIGHT_PATH = 'footandball.onnx'
MODEL_PATH = 'footandball.onnx.prototxt'
REMOTE_PATH = 'https://storage.googleapis.com/ailia-models/footandball/'

IMAGE_PATH = 'input.jpg'
SAVE_IMAGE_PATH = 'output.png'


# ======================
# Arguemnt Parser Config
# ======================

parser = get_base_parser(
    'FootAndBall', IMAGE_PATH, SAVE_IMAGE_PATH
)
parser.add_argument(
    '--onnx',
    action='store_true',
    help='execute onnxruntime version.'
)
parser.add_argument(
    '-pth', '--player_threshold',
    default=0.7, type=float,
    help='player confidence threshold'
)
parser.add_argument(
    '-bth', '--ball_threshold',
    default=0.7, type=float,
    help='ball confidence threshold'
)
#parser.add_argument(
#    '-bbs', '--ball_bbox_size',
#    default=40, type=int,
#    help='size of ball binding box'
#)
args = update_parser(parser)
args.ball_bbox_size = 40


# ======================
# Main functions
# ======================

def draw_bboxes(image, detections):
    font = cv2.FONT_HERSHEY_SIMPLEX
    for box, label, score in zip(detections['boxes'], detections['labels'], detections['scores']):
        if label == PLAYER_LABEL:
            x1, y1, x2, y2 = box
            color = (255, 0, 0)
            cv2.rectangle(image, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
            cv2.putText(image, '{:0.2f}'.format(score), (int(x1), max(0, int(y1)-10)), 
                        font, 1, color, 2)

        elif label == BALL_LABEL:
            x1, y1, x2, y2 = box
            x = int((x1 + x2) / 2)
            y = int((y1 + y2) / 2)
            color = (0, 0, 255)
            radius = 25
            cv2.circle(image, (int(x), int(y)), radius, color, 2)
            cv2.putText(image, '{:0.2f}'.format(score), (max(0, int(x - radius)), max(0, (y - radius - 10))), 
                        font, 1, color, 2)

    return image


def predict(net, img):
    img = numpy2tensor(img)
    img = img[np.newaxis, :, :, :]

    # feedforward
    if not args.onnx:
        output = net.run(img)
    else:
        output = net.run([
            net.get_outputs()[0].name,
            net.get_outputs()[1].name,
            net.get_outputs()[2].name
        ], {
            net.get_inputs()[0].name: img, 
        })

    player_feature_map, player_bbox, ball_feature_map = output[0], output[1], output[2]
    output = detect(player_feature_map, player_bbox, ball_feature_map,
                    player_threshold=args.player_threshold, 
                    ball_threshold=args.ball_threshold, 
                    ball_bbox_size=args.ball_bbox_size)
    output = output[0]

    return output


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
                detections = predict(net, img)
                end = int(round(time.time() * 1000))
                estimation_time = (end - start)

                # Logging
                logger.info(f'\tailia processing estimation time {estimation_time} ms')
                if i != 0:
                    total_time_estimation = total_time_estimation + estimation_time

            logger.info(f'\taverage time estimation {total_time_estimation / (args.benchmark_count - 1)} ms')
        else:
            detections = predict(net, img)

        # draw
        img = draw_bboxes(img, detections)

        # plot result
        savepath = get_savepath(args.savepath, image_path, ext='.png')
        logger.info(f'saved at : {savepath}')
        cv2.imwrite(savepath, img)

    logger.info('Script finished successfully.')


def recognize_from_video(net):
    video_file = args.video if args.video else args.input[0]
    capture = get_capture(video_file)
    n_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    assert capture.isOpened(), 'Cannot capture source'

    # create video writer if savepath is specified as video format
    if args.savepath != SAVE_IMAGE_PATH:
        fps = capture.get(cv2.CAP_PROP_FPS)
        f_h = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        f_w = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        writer = get_writer(args.savepath, f_h, f_w, fps=fps)
    else:
        writer = None

    i=0
    frame_shown = False
    while True:
        i += 1
        if i%50==0:
            logger.info('{} frames have been processed.'.format(i))
        ret, frame = capture.read()
        if (cv2.waitKey(1) & 0xFF == ord('q')) or not ret:
            break
        if frame_shown and cv2.getWindowProperty('frame', cv2.WND_PROP_VISIBLE) == 0:
            break

        # inference
        detections = predict(net, frame)

        # draw
        frame = draw_bboxes(frame, detections)

        # show
        cv2.imshow("frame", frame)

        # save results
        if writer is not None:
            res_img = frame
            res_img = res_img.astype(np.uint8)
            writer.write(res_img)

    capture.release()
    cv2.destroyAllWindows()
    if writer is not None:
        writer.release()

    logger.info('Script finished successfully.')


def main():
    # model files check and download
    check_and_download_models(WEIGHT_PATH, MODEL_PATH, REMOTE_PATH)

    env_id = args.env_id

    # initialize
    if not args.onnx:
        net = ailia.Net(MODEL_PATH, WEIGHT_PATH, env_id=env_id, memory_mode=args.memory_mode)
    else:
        import onnxruntime
        net = onnxruntime.InferenceSession(WEIGHT_PATH, providers = ["CPUExecutionProvider", "CUDAExecutionProvider"])

    if args.video is not None:
        recognize_from_video(net)
    else:
        recognize_from_image(net)


if __name__ == '__main__':
    main()
