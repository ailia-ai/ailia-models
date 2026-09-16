import sys
import time

import numpy as np
import cv2
from PIL import Image

import ailia

# import original modules
sys.path.append('../../util')
from arg_utils import get_base_parser, update_parser, get_savepath  # noqa: E402
from model_utils import check_and_download_models  # noqa: E402
from image_utils import normalize_image  # noqa: E402
from detector_utils import load_image  # noqa: E402
import webcamera_utils  # noqa: E402

import blazeface_utils as but

# logger
from logging import getLogger  # noqa: E402

logger = getLogger(__name__)

# ======================
# Parameters
# ======================
WEIGHT_PATH = 'G_ep300.onnx'
MODEL_PATH = 'G_ep300.onnx.prototxt'
REMOTE_PATH = 'https://storage.googleapis.com/ailia-models/beauty_gan/'

WEIGHT_BLAZE_PATH = 'blazeface.onnx'
MODEL_BLAZE_PATH = 'blazeface.onnx.prototxt'
REMOTE_BLAZE_PATH = "https://storage.googleapis.com/ailia-models/blazeface/"

IMAGE_PATH = 'xfsy_0147.png'
IMAGE_MAKEUP_PATH = 'makeup_vFG48.png'
SAVE_IMAGE_PATH = 'output.png'

IMAGE_SIZE = 256

# ======================
# Arguemnt Parser Config
# ======================
parser = get_base_parser('BeautyGAN model', IMAGE_PATH, SAVE_IMAGE_PATH)
parser.add_argument(
    '-im', '--image_makeup',
    default=IMAGE_MAKEUP_PATH, type=str, metavar='IMAGE',
    help='Makeup image.'
)
args = update_parser(parser)


# ======================
# Main functions
# ======================

def preprocess(img):
    mean = np.array((0.5, 0.5, 0.5))
    std = np.array((0.5, 0.5, 0.5))

    h, w = img.shape[:2]
    if h > w:
        scale = IMAGE_SIZE / w
        h = int(h / w * IMAGE_SIZE)
        w = IMAGE_SIZE
    else:
        scale = IMAGE_SIZE / h
        w = int(w / h * IMAGE_SIZE)
        h = IMAGE_SIZE

    img = np.array(Image.fromarray(img).resize(
        (w, h),
        resample=Image.BILINEAR))

    if h > IMAGE_SIZE:
        p = (h - IMAGE_SIZE) // 2
        img = img[p:p + IMAGE_SIZE, :, :]
    elif w > IMAGE_SIZE:
        p = (w - IMAGE_SIZE) // 2
        img = img[:, p:p + IMAGE_SIZE, :]

    img = img / 255
    img = (img - mean) / std
    img = img.transpose(2, 0, 1)  # HWC -> CHW
    img = np.expand_dims(img, axis=0)

    return img, scale


def postprocess(output):
    output = (output.transpose((1, 2, 0)) + 1) / 2.0 * 255.0
    output = np.clip(output, 0, 255)
    img = output.astype(np.uint8)
    img = img[:, :, ::-1]  # RGB -> BGR

    return img


def face_detect(img, face_net):
    IMAGE_BLAZE_SIZE = 128

    img_0 = img

    img = normalize_image(img, normalize_type='127.5')
    img = cv2.resize(img, (IMAGE_BLAZE_SIZE, IMAGE_BLAZE_SIZE))
    img = img.transpose((2, 0, 1))
    img = np.expand_dims(img, axis=0)

    output = face_net.predict([img])
    detections = but.postprocess(output)
    detections = detections[0]

    # sort by confidence
    detections = sorted(detections, key=lambda x: x[16], reverse=True)
    if len(detections) == 0:
        return None, (0, 0)

    detection = detections[0]

    h, w = img_0.shape[:2]
    ymin = int(detection[0] * h)
    xmin = int(detection[1] * w)
    ymax = int(detection[2] * h)
    xmax = int(detection[3] * w)

    h = ymax - ymin
    w = xmax - xmin
    if h > w:
        p = (h - w) // 2
        w = h
        xmin -= p
    else:
        p = (w - h) // 2
        h = w
        ymin -= p

    img = img_0[ymin:ymin + h, xmin:xmin + w]

    h2, w2 = img.shape[:2]
    if h != h2 or w != w2:
        return None, (0, 0)

    return img, (ymin, xmin)


def recognize_from_image(net):
    img_B = load_image(args.image_makeup)
    img_B = cv2.cvtColor(img_B, cv2.COLOR_BGRA2RGB)
    img_B, _ = preprocess(img_B)

    # input image loop
    for image_path in args.input:
        logger.info(image_path)

        # prepare grand truth
        img_A = load_image(image_path)
        img_A = cv2.cvtColor(img_A, cv2.COLOR_BGRA2RGB)
        img_A, _ = preprocess(img_A)

        logger.debug(f'input image shape: {img_A.shape}')

        # inference
        logger.info('Start inference...')
        if args.benchmark:
            logger.info('BENCHMARK mode')
            total_time = 0
            for i in range(args.benchmark_count):
                start = int(round(time.time() * 1000))
                output = net.predict({'img_A': img_A, 'img_B': img_B})
                end = int(round(time.time() * 1000))
                logger.info(f'\tailia processing time {end - start} ms')
                if i != 0:
                    total_time = total_time + (end - start)
            logger.info(f'\taverage time {total_time / (args.benchmark_count - 1)} ms')
        else:
            output = net.predict({'img_A': img_A, 'img_B': img_B})

        fake_A, fake_B = output
        output = np.concatenate([img_A[0], img_B[0], fake_A[0], fake_B[0]], axis=2)
        res_img = postprocess(output)

        savepath = get_savepath(args.savepath, image_path, ext='.png')
        logger.info(f'saved at : {savepath}')
        cv2.imwrite(savepath, res_img)

    logger.info('Script finished successfully.')


def recognize_from_video(net, face_net):
    capture = webcamera_utils.get_capture(args.video)

    # create video writer if savepath is specified as video format
    f_h = f_w = IMAGE_SIZE
    if args.savepath != SAVE_IMAGE_PATH:
        logger.warning(
            'currently, video results cannot be output correctly...'
        )
        writer = webcamera_utils.get_writer(args.savepath, f_h, f_w*2)
    else:
        writer = None

    # create output buffer
    res_img = np.ones((f_h, f_w*2, 3))

    # get style image
    img_B = load_image(args.image_makeup)
    img_B = cv2.cvtColor(img_B, cv2.COLOR_BGRA2RGB)
    img_B_style = cv2.cvtColor(cv2.resize(img_B,(f_w//4,f_h//4)),cv2.COLOR_RGB2BGR) 
    img_B, _ = preprocess(img_B)

    frame_shown = False
    while True:
        ret, frame = capture.read()
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        if frame_shown and cv2.getWindowProperty('frame', cv2.WND_PROP_VISIBLE) == 0:
            break
        if not ret:
            continue

        # face detect
        img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img_A, (y, x) = face_detect(img, face_net)
        
        if not(img_A is None):
            res_img[:,0:f_w,:] = cv2.cvtColor(cv2.resize(img_A,(f_w,f_h)),cv2.COLOR_RGB2BGR)
            res_img[f_h//4*3:f_h,0:f_w//4,:] = img_B_style

            img_A, scale = preprocess(img_A)

            output = net.predict({'img_A': img_A, 'img_B': img_B})
            fake_A, fake_B = output

            res_img[:,f_w:f_w*2,:] = postprocess(fake_A[0])

        # save results
        res_img = res_img.astype(np.uint8)
        if writer is not None:
            writer.write(res_img)

        # show
        cv2.imshow('frame', res_img)
        frame_shown = True

    capture.release()
    cv2.destroyAllWindows()
    if writer is not None:
        writer.release()

    print('Script finished successfully.')


def main():
    # model files check and download
    logger.info('=== BeautyGAN model ===')
    check_and_download_models(WEIGHT_PATH, MODEL_PATH, REMOTE_PATH)
    if args.video:
        logger.info('=== BlazeFace model ===')
        check_and_download_models(WEIGHT_BLAZE_PATH, MODEL_BLAZE_PATH, REMOTE_BLAZE_PATH)

    # net initialize
    net = ailia.Net(MODEL_PATH, WEIGHT_PATH, env_id=args.env_id, memory_mode=args.memory_mode)

    if args.video is not None:
        # video mode
        face_net = ailia.Net(MODEL_BLAZE_PATH, WEIGHT_BLAZE_PATH, env_id=args.env_id, memory_mode=args.memory_mode)
        recognize_from_video(net, face_net)
    else:
        # image mode
        recognize_from_image(net)


if __name__ == '__main__':
    main()
