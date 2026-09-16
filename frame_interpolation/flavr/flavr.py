import sys
import time
import os
import platform

import numpy as np
import cv2
from tqdm import tqdm

import ailia

# import original modules
sys.path.append('../../util')
from arg_utils import get_base_parser, update_parser, get_savepath  # noqa: E402
from model_utils import check_and_download_models  # noqa: E402
from detector_utils import load_image  # noqa: E402
import webcamera_utils  # noqa: E402

# logger
from logging import getLogger  # noqa: E402

logger = getLogger(__name__)

# ======================
# Parameters
# ======================
WEIGHT_2x_PATH = 'FLAVR_2x.onnx'
MODEL_2x_PATH = 'FLAVR_2x.onnx.prototxt'
WEIGHT_4x_PATH = 'FLAVR_4x.onnx'
MODEL_4x_PATH = 'FLAVR_4x.onnx.prototxt'
WEIGHT_8x_PATH = 'FLAVR_8x.onnx'
MODEL_8x_PATH = 'FLAVR_8x.onnx.prototxt'
REMOTE_PATH = 'https://storage.googleapis.com/ailia-models/flavr/'

IMAGE_PATH = 'sample'
SAVE_IMAGE_PATH = 'output.png'

# ======================
# Arguemnt Parser Config
# ======================
parser = get_base_parser('FLAVR model', IMAGE_PATH, SAVE_IMAGE_PATH, large_model=True)
parser.add_argument(
    '-ip', '--interpolation', type=int, choices=(2, 4, 8), default=2,
    help='2x/4x/8x Interpolation'
)
parser.add_argument(
    '-n', '--num_frame',
    default=None,
    help='select input frame numbers (string of four numbers). ex. "1357"'
)
parser.add_argument(
    '-hw', metavar='HEIGHT,WIDTH',
    default="256,448",
    help='Specify the size to resize.'
)
args = update_parser(parser)


# ======================
# Main functions
# ======================

def preprocess(img):
    h, w = map(int, args.hw.split(','))
    img = cv2.resize(img, (w, h))

    img = img / 255
    img = img.transpose(2, 0, 1)  # HWC -> CHW
    img = np.expand_dims(img, axis=0)
    img = img.astype(np.float32)

    return img


def postprocess(output):
    output = output.clip(0.0, 1.0)
    output = output.transpose((1, 2, 0)) * 255.0
    img = output.astype(np.uint8)
    img = img[:, :, ::-1]  # RGB -> BGR

    return img


def recognize_from_image(net, n_output):
    # Load images
    images = [load_image(pth) for pth in args.input]
    images = [cv2.cvtColor(img, cv2.COLOR_BGRA2RGB) for img in images]
    images = [preprocess(img) for img in images]

    ## Select only relevant inputs
    if args.num_frame:
        inputs = [int(i) - 1 for i in args.num_frame]
        images = [images[i] for i in inputs]

    imgx = ["img%d" % i for i in range(4)]

    # inference
    logger.info('Start inference...')
    if args.benchmark:
        logger.info('BENCHMARK mode')
        total_time = 0
        for i in range(args.benchmark_count):
            start = int(round(time.time() * 1000))
            output = net.predict({k: v for k, v in zip(imgx, images)})
            end = int(round(time.time() * 1000))
            logger.info(f'\tailia processing time {end - start} ms')
            if i != 0:
                total_time = total_time + (end - start)
        logger.info(f'\taverage time {total_time / (args.benchmark_count - 1)} ms')
    else:
        output = net.predict({k: v for k, v in zip(imgx, images)})

    images = [postprocess(x[0]) for x in output]

    savepath = os.path.join(args.savepath, SAVE_IMAGE_PATH)
    if 1 < n_output:
        name, ext = os.path.splitext(savepath)
        for i in range(n_output):
            savepath = "%s_%s%s" % (name, i, ext)
            logger.info(f'saved at : {savepath}')
            cv2.imwrite(savepath, images[i])
    else:
        logger.info(f'saved at : {savepath}')
        cv2.imwrite(savepath, images[0])

    logger.info('Script finished successfully.')


def recognize_from_video(net, n_output):
    cap = webcamera_utils.get_capture(args.video)

    video_length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    logger.info(f"video_length: {video_length}")

    # create video writer if savepath is specified as video format
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    f_h, f_w = map(int, args.hw.split(','))
    writer = None
    if args.savepath!=SAVE_IMAGE_PATH:
        writer = webcamera_utils.get_writer(args.savepath, f_h, f_w, fps=fps)

    # create output buffer
    output_buffer = np.zeros((f_h*(n_output+2),f_w,3))
    output_buffer = output_buffer.astype(np.uint8)

    imgx = ["img%d" % i for i in range(4)]

    images = []
    inputs = []
    if 0 < video_length:
        it = iter(tqdm(range(video_length)))
        next(it)
        
    frame_shown = False
    while True:
        if 0 < video_length:
            try:
                next(it)
            except StopIteration:
                break
        ret, frame = cap.read()
        if (cv2.waitKey(1) & 0xFF == ord('q')) or not ret:
            break
        if frame_shown and cv2.getWindowProperty('frame', cv2.WND_PROP_VISIBLE) == 0:
            break

        images.append(cv2.resize(frame, (f_w, f_h)))
        inputs.append(
            preprocess(cv2.cvtColor(frame, cv2.COLOR_BGRA2RGB))
        )
        if len(inputs) < 4:
            continue
        elif len(inputs) > 4:
            inputs = inputs[1:]
            images = images[1:]

        output = net.predict({k: v for k, v in zip(imgx, inputs)})

        # save results
        if writer is not None:
            writer.write(images[1])
        output_buffer[0:f_h,0:f_w,:]=images[1]
        output_buffer[f_h*(n_output+1):f_h*(n_output+2),0:f_w,:]=images[2]
        for i in range(n_output):
            out_img = postprocess(output[i][0])
            if writer is not None:
                writer.write(out_img)
            output_buffer[f_h*(i+1):f_h*(i+2),0:f_w,:]=out_img
        
        #preview
        cv2.imshow('frame', output_buffer)
        frame_shown = True

    cap.release()
    cv2.destroyAllWindows()
    if writer is not None:
        writer.release()
    logger.info('Script finished successfully.')


def main():
    info = {
        2: (WEIGHT_2x_PATH, MODEL_2x_PATH, 1),
        4: (WEIGHT_4x_PATH, MODEL_4x_PATH, 3),
        8: (WEIGHT_8x_PATH, MODEL_8x_PATH, 7),
    }
    weight_path, model_path, n_output = info[args.interpolation]
    # model files check and download
    check_and_download_models(weight_path, model_path, REMOTE_PATH)

    # net initialize
    net = ailia.Net(model_path, weight_path, env_id=args.env_id, memory_mode=args.memory_mode)

    if args.video is not None:
        # video mode
        recognize_from_video(net, n_output)
    else:
        # image mode
        recognize_from_image(net, n_output)


if __name__ == '__main__':
    main()
