import sys
import time

import numpy as np
import cv2

import ailia

# import original modules
from fashionai_key_points_detection_utils import decode_np, draw_keypoints, save_json
sys.path.append('../../util')
from arg_utils import get_base_parser, update_parser, get_savepath  # noqa: E402
from model_utils import check_and_download_models  # noqa: E402
from detector_utils import load_image  # noqa: E402
from webcamera_utils import get_capture  # noqa: E402

# logger
from logging import getLogger   # noqa: E402
logger = getLogger(__name__)


# ======================
# Parameters
# ======================
WEIGHT_BLOUSE_PATH = 'blouse_100.onnx'
MODEL_BLOUSE_PATH = 'blouse_100.onnx.prototxt'
WEIGHT_DRESS_PATH = 'dress_100.onnx'
MODEL_DRESS_PATH = 'dress_100.onnx.prototxt'
WEIGHT_OUTWEAR_PATH = 'outwear_100.onnx'
MODEL_OUTWEAR_PATH = 'outwear_100.onnx.prototxt'
WEIGHT_SKIRT_PATH = 'skirt_100.onnx'
MODEL_SKIRT_PATH = 'skirt_100.onnx.prototxt'
WEIGHT_TROUSERS_PATH = 'trousers_100.onnx'
MODEL_TROUSERS_PATH = 'trousers_100.onnx.prototxt'
REMOTE_PATH = \
    'https://storage.googleapis.com/ailia-models/fashionai-key-points-detection/'

IMAGE_BLOUSE_PATH = 'blouse.jpg'
IMAGE_DRESS_PATH = 'dress.jpg'
IMAGE_OUTWEAR_PATH = 'outwear.jpg'
IMAGE_SKIRT_PATH = 'skirt.jpg'
IMAGE_TROUSERS_PATH = 'trousers.jpg'
SAVE_IMAGE_PATH = 'output.png'

IMAGE_SIZE = 512

MU = 0.65
SIGMA = 0.25
HM_STRIDE = 4

# ======================
# Arguemnt Parser Config
# ======================
parser = get_base_parser('FashionAI model', IMAGE_BLOUSE_PATH, SAVE_IMAGE_PATH)
parser.add_argument(
    '-c', '--clothing-type', type=str, default='blouse',
    choices=('blouse', 'dress', 'outwear', 'skirt', 'trousers'),
    help='clothing type'
)
parser.add_argument(
    '-w', '--write_json',
    action='store_true',
    help='Flag to output results to json file.'
)
args = update_parser(parser)


# ======================
# Secondaty Functions
# ======================
def preprocess(img, img_size):
    (img_w, img_h) = img_size
    img = cv2.resize(img, (img_w, img_h), interpolation=cv2.INTER_CUBIC)
    img = np.transpose(img, (2, 0, 1)).astype(np.float32)  # channel, height, width
    img[[0, 2]] = img[[2, 0]]  # BGR -> RGB
    img = img / 255.0
    img = (img - MU) / SIGMA
    pad_imgs = np.zeros([1, 3, IMAGE_SIZE, IMAGE_SIZE], dtype=np.float32)
    pad_imgs[0, :, :img_h, :img_w] = img

    return pad_imgs


def post_processing(hm_pred, hm_pred2, info):
    keypoints = {
        'blouse': ['neckline_left', 'neckline_right', 'center_front', 'shoulder_left', 'shoulder_right',
                   'armpit_left', 'armpit_right', 'cuff_left_in', 'cuff_left_out', 'cuff_right_in',
                   'cuff_right_out', 'top_hem_left', 'top_hem_right'],
        'outwear': ['neckline_left', 'neckline_right', 'shoulder_left', 'shoulder_right', 'armpit_left',
                    'armpit_right', 'waistline_left', 'waistline_right', 'cuff_left_in', 'cuff_left_out',
                    'cuff_right_in', 'cuff_right_out', 'top_hem_left', 'top_hem_right'],
        'trousers': ['waistband_left', 'waistband_right', 'crotch', 'bottom_left_in', 'bottom_left_out',
                     'bottom_right_in', 'bottom_right_out'],
        'skirt': ['waistband_left', 'waistband_right', 'hemline_left', 'hemline_right'],
        'dress': ['neckline_left', 'neckline_right', 'center_front', 'shoulder_left', 'shoulder_right',
                  'armpit_left', 'armpit_right', 'waistline_left', 'waistline_right', 'cuff_left_in',
                  'cuff_left_out', 'cuff_right_in', 'cuff_right_out', 'hemline_left', 'hemline_right']
    }
    keypoint = keypoints[args.clothing_type]
    conjug = []
    for i, key in enumerate(keypoint):
        if 'left' in key:
            j = keypoint.index(key.replace('left', 'right'))
            conjug.append([i, j])

    a = np.zeros_like(hm_pred2)
    img_w2 = info['img_w2']
    a[:, :, :img_w2 // HM_STRIDE] = np.flip(
        hm_pred2[:, :, :img_w2 // HM_STRIDE], 2
    )
    for conj in conjug:
        a[conj] = a[conj[::-1]]
    hm_pred2 = a

    scale = info['scale']
    img_w = info['img_w']
    img_h = info['img_h']
    x, y = decode_np(
        hm_pred + hm_pred2,
        scale,
        HM_STRIDE,
        (img_w / 2, img_h / 2),
        method='maxoffset',
    )
    keypoints = np.stack([x, y, np.ones(x.shape)], axis=1).astype(np.int16)

    return keypoints


# ======================
# Main functions
# ======================
def predict(img, net):
    img_flip = cv2.flip(img, 1)

    img_h, img_w, _ = img.shape
    scale = IMAGE_SIZE / max(img_w, img_h)
    img_h2 = int(img_h * scale)
    img_w2 = int(img_w * scale)

    # initial preprocesses
    img = preprocess(img, (img_w2, img_h2))
    img_flip = preprocess(img_flip, (img_w2, img_h2))

    # feedforward
    output = net.predict({
        'img': img
    })
    output_flip = net.predict({
        'img': img_flip
    })

    _, hm_pred = output
    _, hm_pred2 = output_flip

    hm_pred = np.maximum(hm_pred, 0)
    hm_pred2 = np.maximum(hm_pred2, 0)
    hm_pred = hm_pred[0]
    hm_pred2 = hm_pred2[0]

    info = {
        'img_h': img_h,
        'img_w': img_w,
        'img_h2': img_h2,
        'img_w2': img_w2,
        'scale': scale,
    }
    keypoints = post_processing(hm_pred, hm_pred2, info)

    return keypoints


def recognize_from_image(filename, net):
    # prepare input data
    img = load_image(filename)
    logger.debug(f'input image shape: {img.shape}')

    img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)

    # inference
    logger.info('Start inference...')
    if args.benchmark:
        logger.info('BENCHMARK mode')
        for i in range(5):
            start = int(round(time.time() * 1000))
            keypoints = predict(img, net)
            end = int(round(time.time() * 1000))
            logger.info(f'\tailia processing time {end - start} ms')
    else:
        keypoints = predict(img, net)

    """
    plot result
    """
    res_img = draw_keypoints(img, keypoints)
    savepath = get_savepath(args.savepath, filename)
    logger.info(f'saved at : {savepath}')
    cv2.imwrite(savepath, res_img)

    if args.write_json:
        json_file = '%s.json' % savepath.rsplit('.', 1)[0]
        save_json(json_file, keypoints)


def recognize_from_video(video, net):
    capture = get_capture(video)

    frame_shown = False
    while (True):
        ret, frame = capture.read()
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        if frame_shown and cv2.getWindowProperty('frame', cv2.WND_PROP_VISIBLE) == 0:
            break
        if not ret:
            continue

        keypoints = predict(frame, net)
        res_img = draw_keypoints(frame, keypoints)

        # show
        cv2.imshow('frame', res_img)
        frame_shown = True

    capture.release()
    cv2.destroyAllWindows()


def main():
    dic_model = {
        'blouse': (WEIGHT_BLOUSE_PATH, MODEL_BLOUSE_PATH, IMAGE_BLOUSE_PATH),
        'dress': (WEIGHT_DRESS_PATH, MODEL_DRESS_PATH, IMAGE_DRESS_PATH),
        'outwear': (WEIGHT_OUTWEAR_PATH, MODEL_OUTWEAR_PATH, IMAGE_OUTWEAR_PATH),
        'skirt': (WEIGHT_SKIRT_PATH, MODEL_SKIRT_PATH, IMAGE_SKIRT_PATH),
        'trousers': (WEIGHT_TROUSERS_PATH, MODEL_TROUSERS_PATH, IMAGE_TROUSERS_PATH),
    }
    weight_path, model_path, img_path = dic_model[args.clothing_type]

    # model files check and download
    check_and_download_models(weight_path, model_path, REMOTE_PATH)

    # initialize
    net = ailia.Net(model_path, weight_path, env_id=args.env_id, memory_mode=args.memory_mode)

    if args.video is not None:
        # video mode
        recognize_from_video(args.video, net)
    else:
        # image mode
        # input image loop
        for image_path in args.input:
            logger.info(image_path)
            recognize_from_image(image_path, net)
    logger.info('Script finished successfully.')


if __name__ == '__main__':
    main()
