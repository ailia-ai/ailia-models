import sys
import os
import shutil
import time

import numpy as np
import cv2

import ailia

# import original modules
sys.path.append('../../util')
from arg_utils import get_base_parser, update_parser, get_savepath  # noqa
from model_utils import check_and_download_models  # noqa
from detector_utils import load_image  # noqa
from webcamera_utils import get_capture, get_writer  # noqa
# logger
from logging import getLogger  # noqa

logger = getLogger(__name__)

# ======================
# Parameters
# ======================

WEIGHT_PATH = 'RIFE_HDv3.onnx'
MODEL_PATH = 'RIFE_HDv3.onnx.prototxt'
REMOTE_PATH = 'https://storage.googleapis.com/ailia-models/rife/'

IMAGE_PATH = 'imgs'
SAVE_IMAGE_PATH = 'output.png'

NAME_EXT = os.path.splitext(SAVE_IMAGE_PATH)

# ======================
# Arguemnt Parser Config
# ======================

parser = get_base_parser(
    'Real-Time Intermediate Flow Estimation', IMAGE_PATH, SAVE_IMAGE_PATH
)
parser.add_argument(
    '-i2', '--input2', metavar='IMAGE2', default=None,
    help='The second input image path.'
)
parser.add_argument(
    '--exp', type=int, default=1,
    help='exp'
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

def img_save(no, mid_img=None, img_path=None):
    save_file = "%s_%03d%s" % (NAME_EXT[0], no, NAME_EXT[1])
    save_path = get_savepath(args.savepath, save_file, post_fix='', ext='.png')

    if mid_img is not None:
        logger.info(f'saved at : {save_path}')
        cv2.imwrite(save_path, mid_img)

    if img_path is not None:
        logger.info(f'copy {img_path} -> {save_path}')
        shutil.copy(img_path, save_path)

    return no + 1


# ======================
# Main functions
# ======================

def preprocess(img):
    h, w, _ = img.shape

    align = 32
    if h % align != 0 or w % align != 0:
        ph = ((h - 1) // align + 1) * align
        pw = ((w - 1) // align + 1) * align

        pad_img = np.zeros(shape=(ph, pw, 3))
        pad_img[:h, :w, :] = img

        img = pad_img

    img = img / 255
    img = img.transpose(2, 0, 1)  # HWC -> CHW
    img = np.expand_dims(img, axis=0)
    img = img.astype(np.float32)

    return img


def predict(net, img1, img2):
    h, w, _ = img1.shape
    img1 = img1[:, :, ::-1]  # BGR -> RGB
    img2 = img2[:, :, ::-1]  # BGR -> RGB

    x0 = preprocess(img1)
    x1 = preprocess(img2)

    # feedforward
    if not args.onnx:
        output = net.predict([x0, x1])
    else:
        output = net.run(None, {'I0': x0, 'I1': x1})

    mid_img = output[0]

    mid_img = mid_img[0].transpose(1, 2, 0)  # CHW -> HWC
    mid_img = np.clip(mid_img * 255, 0, 255)
    mid_img = (mid_img + 0.5).astype(np.uint8)
    mid_img = mid_img[:, :, ::-1]  # RGB -> BGR
    mid_img = mid_img[:h, :w, ...]

    return mid_img


def make_inference(net, img1, img2, n):
    mid_img = predict(net, img1, img2)

    if n == 1:
        return [mid_img]

    first_half = make_inference(net, img1, mid_img, n=n // 2)
    second_half = make_inference(net, mid_img, img2, n=n // 2)
    if n % 2:
        return [*first_half, mid_img, *second_half]
    else:
        return [*first_half, *second_half]


def recognize_from_image(net):
    inputs = args.input
    exp = args.exp
    copy_img = True

    # Load images
    n_input = len(inputs)
    if n_input == 1 and args.input2:
        inputs.extend([args.input2])
        copy_img = False

    if len(inputs) < 2:
        logger.error("Specified input must be at least two or more images")
        sys.exit(-1)

    no = 0
    for image_paths in zip(inputs, inputs[1:]):
        logger.info(image_paths)

        # prepare input data
        images = [load_image(p) for p in image_paths]
        img1, img2 = [cv2.cvtColor(im, cv2.COLOR_BGRA2BGR) for im in images]

        # inference
        logger.info('Start inference...')
        if args.benchmark:
            logger.info('BENCHMARK mode')
            total_time_estimation = 0
            for i in range(args.benchmark_count):
                start = int(round(time.time() * 1000))
                mid_img = predict(net, img1, img2)
                end = int(round(time.time() * 1000))
                estimation_time = (end - start)

                # Logging
                logger.info(f'\tailia processing estimation time {estimation_time} ms')
                if i != 0:
                    total_time_estimation = total_time_estimation + estimation_time

            logger.info(f'\taverage time estimation {total_time_estimation / (args.benchmark_count - 1)} ms')

            copy_img = False
            no = img_save(no, mid_img=mid_img)
        else:
            output = make_inference(net, img1, img2, 2 ** exp - 1)

            if copy_img:
                no = img_save(no, img_path=image_paths[0])
            for mid in output:
                no = img_save(no, mid_img=mid)

    if copy_img:
        img_save(no, img_path=image_paths[-1])

    logger.info('Script finished successfully.')


def recognize_from_video(net):
    video_file = args.video if args.video else args.input[0]
    capture = get_capture(video_file)
    assert capture.isOpened(), 'Cannot capture source'

    video_length = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    if 0 < video_length:
        logger.info(f"video_length: {video_length}")

    # create video writer if savepath is specified as video format
    fps = int(capture.get(cv2.CAP_PROP_FPS))
    f_h = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    f_w = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    writer = None
    if args.savepath != SAVE_IMAGE_PATH:
        writer = get_writer(args.savepath, f_h, f_w, fps=fps)

    # create output buffer
    n_output = 1
    output_buffer = np.zeros((f_h * (n_output + 2), f_w, 3))
    output_buffer = output_buffer.astype(np.uint8)

    images = []

    it = None
    try:
        import tqdm
        if 0 < video_length:
            it = iter(tqdm.tqdm(range(video_length)))
            next(it)
    except ImportError:
        pass

    frame_shown = False
    while True:
        if it and 0 < video_length:
            try:
                next(it)
            except StopIteration:
                break
        ret, frame = capture.read()
        if (cv2.waitKey(1) & 0xFF == ord('q')) or not ret:
            break
        if frame_shown and cv2.getWindowProperty('frame', cv2.WND_PROP_VISIBLE) == 0:
            break

        # set inputs
        images.append(cv2.resize(frame, (f_w, f_h)))
        if len(images) < 2:
            continue
        elif len(images) > 2:
            images = images[1:]

        # inference
        img1, img2 = images
        out_img = predict(net, img1, img2)

        output_buffer[:f_h, :f_w, :] = images[0]
        output_buffer[f_h * 1:f_h * 2, :f_w, :] = out_img
        output_buffer[f_h * 2:f_h * 3, :f_w, :] = images[1]

        # preview
        cv2.imshow('frame', output_buffer)
        frame_shown = True

        # save results
        if writer is not None:
            writer.write(images[0])
            writer.write(out_img)

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
        net = onnxruntime.InferenceSession(WEIGHT_PATH)

    if args.video is not None:
        recognize_from_video(net)
    else:
        recognize_from_image(net)


if __name__ == '__main__':
    main()
