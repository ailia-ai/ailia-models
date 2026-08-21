import sys
import time

import ailia
import cv2
import numpy as np

# import original modules
sys.path.append('../../util')
# logger
from logging import getLogger  # noqa: E402

import webcamera_utils  # noqa: E402
from image_utils import imread  # noqa: E402
from model_utils import check_and_download_models  # noqa: E402
from arg_utils import get_base_parser, get_savepath, update_parser  # noqa: E402

sys.path.append('../../face_detection/blazeface')
from blazeface_utils import compute_blazeface  # noqa: E402

logger = getLogger(__name__)


# ======================
# Parameters
# ======================
WEIGHT_PATH = 'u2net_portrait.onnx'
MODEL_PATH = 'u2net_portrait.onnx.prototxt'
REMOTE_PATH = 'https://storage.googleapis.com/ailia-models/u2net_portrait/'

IMAGE_PATH = 'your_portrait_im/kid1.jpg'
SAVE_IMAGE_PATH = 'your_portrait_results/kid1.jpg'

FACE_DIR = '../../face_detection/blazeface/'
FACE_WEIGHT_PATH = FACE_DIR + 'blazeface.onnx'
FACE_MODEL_PATH = FACE_DIR + 'blazeface.onnx.prototxt'
FACE_ANCHOR_PATH = FACE_DIR + 'anchors.npy'
FACE_REMOTE_PATH = 'https://storage.googleapis.com/ailia-models/blazeface/'

IMAGE_WIDTH = 512
IMAGE_HEIGHT = 512


# ======================
# Arguemnt Parser Config
# ======================
parser = get_base_parser(
    'U^2-Net: Going Deeper with Nested U-Structure for Salient Object Detection',
    IMAGE_PATH,
    SAVE_IMAGE_PATH,
)
args = update_parser(parser)


# ======================
# Utils
# ======================
def detect_single_face(detector, img):
    height, width = img.shape[0:2]

    # Detect faces
    detections = compute_blazeface(detector, img, anchor_path=FACE_ANCHOR_PATH)
    if len(detections) == 0:
        logger.warning("no face detection, "
                       "the portrait u2net will run on the whole image!")
        return None

    # filter to keep the largest face
    # (the coordinates of the detection are normalized)
    obj = max(detections, key=lambda d: d.w * d.h)
    x = int(obj.x * width)
    y = int(obj.y * height)
    w = int(obj.w * width)
    h = int(obj.h * height)

    return (x, y, w, h)


# crop, pad and resize face region to 512x512 resolution
def crop_face(img, face):

    # no face detected, return the whole image and
    # the inference will run on the whole image
    if face is None:
        return cv2.resize(
            img, (IMAGE_WIDTH, IMAGE_HEIGHT), interpolation=cv2.INTER_AREA
        )
    (x, y, w, h) = face

    height, width = img.shape[0:2]

    # crop the face with a bigger bbox
    l, r, t, b = 0, 0, 0, 0
    lpad = int(float(w)*0.4)
    left = x-lpad
    if(left < 0):
        l = lpad-x
        left = 0

    rpad = int(float(w)*0.4)
    right = x+w+rpad
    if(right > width):
        r = right-width
        right = width

    tpad = int(float(h)*0.6)
    top = y - tpad
    if(top < 0):
        t = tpad-y
        top = 0

    bpad = int(float(h)*0.2)
    bottom = y+h+bpad
    if(bottom > height):
        b = bottom-height
        bottom = height

    im_face = img[top:bottom, left:right]
    if(len(im_face.shape) == 2):
        im_face = np.repeat(im_face[:, :, np.newaxis], (1, 1, 3))

    im_face = np.pad(
        im_face,
        ((t, b), (l, r), (0, 0)),
        mode='constant',
        constant_values=((255, 255), (255, 255), (255, 255))
    )

    # pad to achieve image with square shape for avoding face deformation
    # after resizing
    hf, wf = im_face.shape[0:2]
    if(hf-2 > wf):
        wfp = int((hf-wf)/2)
        im_face = np.pad(
            im_face,
            ((0, 0), (wfp, wfp), (0, 0)),
            mode='constant',
            constant_values=((255, 255), (255, 255), (255, 255))
        )
    elif(wf-2 > hf):
        hfp = int((wf-hf)/2)
        im_face = np.pad(
            im_face,
            ((hfp, hfp), (0, 0), (0, 0)),
            mode='constant',
            constant_values=((255, 255), (255, 255), (255, 255))
        )

    # resize to have 512x512 resolution
    im_face = cv2.resize(
        im_face,
        (IMAGE_WIDTH, IMAGE_HEIGHT),
        interpolation=cv2.INTER_AREA
    )

    return im_face


def preprocess(img, detector):
    if detector is not None:
        face = detect_single_face(detector, img)
    else:
        face = None
    im_face = crop_face(img, face)

    # normalize the input
    input_img = np.zeros((im_face.shape[0], im_face.shape[1], 3))
    im_face = im_face/np.max(im_face)
    input_img[:, :, 0] = (im_face[:, :, 2]-0.406)/0.225
    input_img[:, :, 1] = (im_face[:, :, 1]-0.456)/0.224
    input_img[:, :, 2] = (im_face[:, :, 0]-0.485)/0.229

    # convert BGR to RGB
    input_img = input_img.transpose((2, 0, 1))
    input_img = input_img[np.newaxis, :, :, :]

    return input_img


def post_process(d1):
    pred = 1.0 - d1[:, 0, :, :]

    # normalization
    ma = np.max(pred)
    mi = np.min(pred)
    pred = (pred-mi)/(ma-mi)

    return (np.squeeze(pred)*255).astype(np.uint8)


# ======================
# Main functions
# ======================
def recognize_from_image():
    # net initialize
    net = ailia.Net(MODEL_PATH, WEIGHT_PATH, env_id=args.env_id)
    detector = ailia.Net(
        FACE_MODEL_PATH, FACE_WEIGHT_PATH, env_id=args.env_id
    )

    # input image loop
    for image_path in args.input:
        # prepare input data
        logger.info(image_path)
        img = imread(image_path)
        logger.debug(f'input image shape: {img.shape}')
        input_img = preprocess(img, detector)

        # inference
        logger.info('Start inference...')
        if args.benchmark:
            logger.info('BENCHMARK mode')
            for i in range(5):
                start = int(round(time.time() * 1000))
                d1, d2, d3, d4, d5, d6, d7 = net.predict(
                    {'input.1': input_img}
                )
                end = int(round(time.time() * 1000))
                logger.info(f'\tailia processing time {end - start} ms')
        else:
            d1, d2, d3, d4, d5, d6, d7 = net.predict({'input.1': input_img})

        out_img = post_process(d1)
        savepath = get_savepath(args.savepath, image_path)
        logger.info(f'saved at : {savepath}')
        cv2.imwrite(savepath, out_img)
    logger.info('Script finished successfully.')


def recognize_from_video():
    # net initialize
    net = ailia.Net(MODEL_PATH, WEIGHT_PATH, env_id=args.env_id)

    capture = webcamera_utils.get_capture(args.video)

    # create video writer if savepath is specified as video format
    if args.savepath != SAVE_IMAGE_PATH:
        writer = webcamera_utils.get_writer(
            args.savepath, IMAGE_HEIGHT, IMAGE_WIDTH, rgb=False
        )
    else:
        writer = None

    frame_shown = False
    while(True):
        ret, img = capture.read()

        # press q to end video capture
        if (cv2.waitKey(1) & 0xFF == ord('q')) or not ret:
            break
        if frame_shown and cv2.getWindowProperty('frame', cv2.WND_PROP_VISIBLE) == 0:
            break

        input_img = preprocess(img, None)
        d1, d2, d3, d4, d5, d6, d7 = net.predict({'input.1': input_img})
        out_img = post_process(d1)
        cv2.imshow('frame', out_img)
        frame_shown = True

        if writer is not None:
            writer.write(out_img)

    capture.release()
    cv2.destroyAllWindows()
    if writer is not None:
        writer.release()
    logger.info('Script finished successfully.')


def main():
    # model files check and download
    check_and_download_models(WEIGHT_PATH, MODEL_PATH, REMOTE_PATH)
    if args.video is None:
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
