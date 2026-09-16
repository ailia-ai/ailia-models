import os
import re
import sys
import time
from collections import deque

import ailia
import cv2
import numpy as np

# import original modules
sys.path.append('../../util')
from classifier_utils import plot_results  # noqa: E402
from image_utils import imread, load_image  # noqa: E402
from model_utils import check_and_download_models  # noqa: E402
from arg_utils import get_base_parser, update_parser  # noqa: E402
from webcamera_utils import get_capture  # noqa: E402

# ======================
# Parameters
# ======================
WEIGHT_PATH = 'mars.onnx'
MODEL_PATH = 'mars.onnx.prototxt'
REMOTE_PATH = 'https://storage.googleapis.com/ailia-models/mars/'

IMAGE_PATH = 'inputs'
IMAGE_HEIGHT = 112
IMAGE_WIDTH = 112
DURATION = 16

HMDB51_LABEL = [
    'brush_hair',
    'cartwheel',
    'catch',
    'chew',
    'clap',
    'climb',
    'climb_stairs',
    'dive',
    'draw_sword',
    'dribble',
    'drink',
    'eat',
    'fall_floor',
    'fencing',
    'flic_flac',
    'golf',
    'handstand',
    'hit',
    'hug',
    'jump',
    'kick',
    'kick_ball',
    'kiss',
    'laugh',
    'pick',
    'pour',
    'pullup',
    'punch',
    'push',
    'pushup',
    'ride_bike',
    'ride_horse',
    'run',
    'shake_hands',
    'shoot_ball',
    'shoot_bow',
    'shoot_gun',
    'sit',
    'situp',
    'smile',
    'smoke',
    'somersault',
    'stand',
    'swing_baseball',
    'sword',
    'sword_exercise',
    'talk',
    'throw',
    'turn',
    'walk',
    'wave',
]


# ======================
# Arguemnt Parser Config
# ======================
parser = get_base_parser('MARS model', IMAGE_PATH, None)

parser.add_argument(
    '-d', '--duration', metavar='DURATION', default=DURATION, type=int,
    help='Sampling duration.',
)
parser.add_argument(
    '-t', '--top', metavar='TOP', default=3, type=int,
    help='Number of outputs for category.',
)
parser.add_argument(
    '--cui',
    action='store_true',
    help="Don't display preview in GUI."
)
args = update_parser(parser)


# ======================
# Main functions
# ======================
def print_mars_result(result):
    indexes = result[0, :].argsort()[::-1]
    print('==============================================================')
    print(f'class_count={args.top}')
    for i, idx in enumerate(indexes[0:args.top]):
        print(f'+ idx={i}')
        print(f'  category={idx}[ {HMDB51_LABEL[idx]} ]')
        print(f'  prob={result[0, idx]}')


def recognize_from_image():
    # prepare input data
    num = lambda val: int(re.sub("\\D", "", val))
    sorted_inputs_path = sorted(args.input, key=num)
    input_blob = np.empty((1, 3, args.duration, IMAGE_HEIGHT, IMAGE_WIDTH))
    for i, input_path in enumerate(sorted_inputs_path[0:args.duration]):
        img = load_image(
            input_path,
            (IMAGE_HEIGHT, IMAGE_WIDTH),
            normalize_type='None',
            gen_input_ailia=True
        )
        input_blob[0, :, i, :, :] = img
    next_input_index = args.duration
    input_frame_size = len(sorted_inputs_path)

    # net initialize
    net = ailia.Net(MODEL_PATH, WEIGHT_PATH, env_id=args.env_id, memory_mode=args.memory_mode)
    net.set_input_shape((1, 3, args.duration, IMAGE_HEIGHT, IMAGE_WIDTH))

    # inference
    print('Start inference...')
    if args.benchmark:
        print('BENCHMARK mode')
        for i in range(5):
            start = int(round(time.time() * 1000))
            result = net.predict(input_blob)
            end = int(round(time.time() * 1000))
            print(f'\tailia processing time {end - start} ms')
    else:
        frame_shown = False
        while(next_input_index < input_frame_size):
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            if frame_shown and cv2.getWindowProperty('preview', cv2.WND_PROP_VISIBLE) == 0:
                break
            result = net.predict(input_blob)

            print_mars_result(result)

            preview_img = imread(sorted_inputs_path[
                    next_input_index - args.duration
            ])
            if not args.cui:
                cv2.imshow('preview', preview_img)
                frame_shown = True
            
            for i in range(args.duration - 1):
                input_blob[0, :, i, :, :] = input_blob[0, :, i + 1, :, :]

            img = load_image(
                sorted_inputs_path[next_input_index],
                (IMAGE_HEIGHT, IMAGE_WIDTH),
                normalize_type='None',
                gen_input_ailia=True
            )
            input_blob[0, :, args.duration - 1, :, :] = img
            next_input_index += 1

    print('Script finished successfully.')


def convert_input_frame(frame):
    frame = cv2.resize(frame, (IMAGE_HEIGHT, IMAGE_WIDTH))
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame = frame.transpose((2, 0, 1))
    frame = frame[np.newaxis, :, :, :]
    return frame


def recognize_from_video():
    capture = get_capture(args.video)

    # net initialize
    net = ailia.Net(MODEL_PATH, WEIGHT_PATH, env_id=args.env_id, memory_mode=args.memory_mode)
    net.set_input_shape((1, 3, args.duration, IMAGE_HEIGHT, IMAGE_WIDTH))

    # prepare input data
    original_queue = deque([])
    input_blob = np.empty((1, 3, args.duration, IMAGE_HEIGHT, IMAGE_WIDTH))
    for i in range(args.duration - 1):
        ret, frame = capture.read()
        if not ret:
            continue
        original_queue.append(frame)
        input_blob[0, :, i, :, :] = convert_input_frame(frame)

    next_input_index = args.duration - 1
    input_frame_size = capture.get(cv2.CAP_PROP_FRAME_COUNT)
    
    frame_shown = False
    while(next_input_index <= input_frame_size or input_frame_size == 0):
        ret, frame = capture.read()
        if (cv2.waitKey(1) & 0xFF == ord('q')) or not ret:
            break
        if frame_shown and cv2.getWindowProperty('preview', cv2.WND_PROP_VISIBLE) == 0:
            break
            
        original_queue.append(frame)
        input_blob[0, :, args.duration - 1, :, :] = convert_input_frame(frame)

        result = net.predict(input_blob)
        print_mars_result(result)
        preview_img = original_queue.popleft()

        plot_results(preview_img, result, HMDB51_LABEL)

        cv2.imshow('preview', preview_img)
        frame_shown = True

        for i in range(args.duration - 1):
            input_blob[0, :, i, :, :] = input_blob[0, :, i + 1, :, :]

        next_input_index += 1

    capture.release()
    print('Script finished successfully.')


def main():
    # model files check and download
    check_and_download_models(WEIGHT_PATH, MODEL_PATH, REMOTE_PATH)
    if args.video is not None:
        # video mode
        recognize_from_video()
    else:
        # image mode
        recognize_from_image()


if __name__ == '__main__':
    main()
