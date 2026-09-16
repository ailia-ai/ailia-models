import sys
import os
import time
from copy import deepcopy
from collections import OrderedDict
from logging import getLogger

import numpy as np
import cv2
from PIL import Image

import ailia
import copy

# import original modules
sys.path.append('../../util')
from arg_utils import get_base_parser, update_parser, get_savepath  # noqa
from model_utils import urlretrieve, progress_print, check_and_download_models  # noqa
from image_utils import normalize_image  # noqa
from detector_utils import load_image  # noqa
from webcamera_utils import get_capture, get_writer  # noqa

logger = getLogger(__name__)

# ======================
# Parameters
# ======================

WEIGHT_SAM_H_PATH = 'sam_h_4b8939.onnx'
MODEL_SAM_H_PATH = 'sam_h_4b8939.onnx.prototxt'
WEIGHT_SAM_L_PATH = 'sam_l_0b3195.onnx'
MODEL_SAM_L_PATH = 'sam_l_0b3195.onnx.prototxt'
WEIGHT_SAM_B_PATH = 'sam_b_01ec64.onnx'
MODEL_SAM_B_PATH = 'sam_b_01ec64.onnx.prototxt'
WEIGHT_VIT_H_PATH = 'vit_h_4b8939.onnx'
WEIGHT_VIT_H_PB_PATH = 'vit_h_4b8939_weights.pb'
MODEL_VIT_H_PATH = 'vit_h_4b8939.onnx.prototxt'
WEIGHT_VIT_L_PATH = 'vit_l_0b3195.onnx'
MODEL_VIT_L_PATH = 'vit_l_0b3195.onnx.prototxt'
WEIGHT_VIT_B_PATH = 'vit_b_01ec64.onnx'
MODEL_VIT_B_PATH = 'vit_b_01ec64.onnx.prototxt'
REMOTE_PATH = 'https://storage.googleapis.com/ailia-models/segment-anything/'

IMAGE_PATH = 'truck.jpg'
SAVE_IMAGE_PATH = 'output.png'

POINT1 = (500, 375)
POINT2 = (1125, 625)

TARGET_LENGTH = 1024

# ======================
# Arguemnt Parser Config
# ======================

parser = get_base_parser(
    'Segment Anything', IMAGE_PATH, SAVE_IMAGE_PATH, fp16_support=False
)
parser.add_argument(
    '-p', '--pos', action='append', type=int, metavar="X", nargs=2,
    help='Positive coordinate specified by x,y.'
)
parser.add_argument(
    '--neg', action='append', type=int, metavar="X", nargs=2,
    help='Negative coordinate specified by x,y.'
)
parser.add_argument(
    '--box', type=int, metavar="X", nargs=4,
    help='Box coordinate specified by x1,y1,x2,y2.'
)
parser.add_argument(
    '--idx', type=int, choices=(0, 1, 2, 3),
    help='Select mask index.'
)
parser.add_argument(
    '-m', '--model_type', default='sam_h', choices=('sam_h', 'sam_l', 'sam_b'),
    help='Select model.'
)
parser.add_argument(
    '--onnx', action='store_true',
    help='execute onnxruntime version.'
)
parser.add_argument(
    '--gui', action='store_true',
    help='Open mouse click GUI.'
)
args = update_parser(parser)


# ======================
# Secondaty Functions
# ======================

def get_preprocess_shape(h: int, w: int, long_side_length: int):
    """
    Compute the output size given input size and target long side length.
    """
    scale = long_side_length / max(h, w)
    new_h, new_w = int(h * scale + 0.5), int(w * scale + 0.5)

    return (new_h, new_w)


def apply_coords(coords, h, w):
    new_h, new_w = get_preprocess_shape(
        h, w, TARGET_LENGTH
    )
    coords = deepcopy(coords).astype(float)
    coords[..., 0] = coords[..., 0] * (new_w / w)
    coords[..., 1] = coords[..., 1] * (new_h / h)

    return coords


def show_mask(mask, img):
    global area_img
    color = np.array([255, 144, 30])
    color = color.reshape(1, 1, -1)

    h, w = mask.shape[-2:]
    mask = mask.reshape(h, w, 1)

    mask_image = mask * color
    img = (img * ~mask) + (img * mask) * 0.6 + mask_image * 0.4
    area_img = copy.deepcopy(mask_image)

    return img


def show_points(coords, labels, img):
    pos_points = coords[labels == 1]
    neg_points = coords[labels == 0]

    for p in pos_points:
        cv2.drawMarker(
            img, p, (0, 255, 0), markerType=cv2.MARKER_TILTED_CROSS, line_type=cv2.LINE_AA,
            markerSize=30, thickness=5)
    for p in neg_points:
        cv2.drawMarker(
            img, p, (0, 0, 255), markerType=cv2.MARKER_TILTED_CROSS, line_type=cv2.LINE_AA,
            markerSize=30, thickness=5)

    return img


def show_box(box, img):
    cv2.rectangle(
        img, box[0], box[1], color=(2, 118, 2),
        thickness=3,
        lineType=cv2.LINE_4,
        shift=0)

    return img


# ======================
# Main functions
# ======================

def preprocess(img):
    im_h, im_w, _ = img.shape

    oh, ow = get_preprocess_shape(im_h, im_w, TARGET_LENGTH)
    img = np.array(Image.fromarray(img).resize((ow, oh), Image.Resampling.BILINEAR))

    img = normalize_image(img, normalize_type='ImageNet')

    pad_h = TARGET_LENGTH - oh
    pad_w = TARGET_LENGTH - ow
    img = np.pad(img, ((0, pad_h), (0, pad_w), (0, 0)), "constant")

    img = img.transpose((2, 0, 1))  # HWC -> CHW
    img = np.expand_dims(img, axis=0)
    img = img.astype(np.float32)

    return img


def postprocess_score(iou_preds, num_points):
    num_mask_tokens = 4
    score_reweight = np.array([1000] + [0] * (num_mask_tokens - 1))
    score = iou_preds + (num_points - 2.5) * score_reweight

    return score


def predict_img_enc(models, img):
    img = img[:, :, ::-1]  # BGR -> RGB
    im_h, im_w = img.shape[:2]
    img = preprocess(img)

    # feedforward
    if args.profile:
        start = int(round(time.time() * 1000))

    img_enc = models["img_enc"]
    if not args.onnx:
        output = img_enc.predict([img])
    else:
        output = img_enc.run(None, {'img': img})
    image_embedding = output[0]

    if args.profile:
        end = int(round(time.time() * 1000))
        estimation_time = (end - start)
        logger.info(f'img_enc processing estimation time {estimation_time} ms')

    return image_embedding, im_h, im_w


def predict_sam_net(models, image_embedding, im_h, im_w, pos_points, neg_points=None, box=None):   
    coord = []
    label = []
    if pos_points:
        coord.append(np.array(pos_points))
        label.append(np.ones(len(pos_points)))
    if neg_points:
        coord.append(np.array(neg_points))
        label.append(np.zeros(len(neg_points)))
    if box is not None:
        coord.append(box)
        label.append(np.array([2, 3]))

    coord = np.concatenate(coord, axis=0)[None, :, :]
    label = np.concatenate(label, axis=0)[None, :].astype(np.float32)
    coord = apply_coords(coord, im_h, im_w).astype(np.float32)

    mask_input = np.zeros((1, 1, 256, 256), dtype=np.float32)
    has_mask_input = np.zeros(1, dtype=np.float32)

    input = OrderedDict([
        ("image_embeddings", image_embedding),
        ("point_coords", coord),
        ("point_labels", label),
        ("mask_input", mask_input),
        ("has_mask_input", has_mask_input),
        ("orig_im_size", np.array((im_h, im_w), dtype=np.float32))
    ])

    # feedforward
    if args.profile:
        start = int(round(time.time() * 1000))

    sam_net = models["sam_net"]
    if not args.onnx:
        output = sam_net.predict(list(input.values()))
    else:
        output = sam_net.run(None, input)

    if args.profile:
        end = int(round(time.time() * 1000))
        estimation_time = (end - start)
        logger.info(f'sam_net processing estimation time {estimation_time} ms')

    masks, iou_predictions, low_res_logits = output
    masks = masks > 0

    masks = masks[0]
    scores = postprocess_score(iou_predictions[0], coord.shape[1])
    logits = low_res_logits[0]

    return masks, scores


def recognize_from_image(models, pos_points, neg_points):
    global img_path, img_enc_output
    box = args.box

    if pos_points is None:
        if neg_points is None and box is None:
            pos_points = [POINT1]
        else:
            pos_points = []
    if neg_points is None:
        neg_points = []
    if box is not None:
        box = np.array(box).reshape(2, 2)

    lf = '\n'
    logger.info(f"Positive coordinate: {pos_points}")
    logger.info(f"Negative coordinate: {neg_points}")
    logger.info(f"Box coordinate: {lf if box is not None else ''}{box}")

    # input image loop
    for image_path in args.input:
        img_path = copy.deepcopy(image_path)
        logger.info(image_path)

        # prepare input data
        img = load_image(image_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
        img_enc_output = predict_img_enc(models, img)

        # inference
        logger.info('Start inference...')
        recognize(image_path, img, models, img_enc_output, pos_points, neg_points, box)


def recognize(image_path, img, models, img_enc_output, pos_points, neg_points=None, box=None):
    image_embedding, im_h, im_w = img_enc_output
    sel_idx = args.idx
    if args.benchmark:
            logger.info('BENCHMARK mode')
            total_time_estimation = 0
            for i in range(args.benchmark_count):
                start = int(round(time.time() * 1000))
                output = predict_sam_net(models, image_embedding, im_h, im_w, pos_points, neg_points, box)
                end = int(round(time.time() * 1000))
                estimation_time = (end - start)

                # Logging
                logger.info(f'\tailia processing estimation time {estimation_time} ms')
                if i != 0:
                    total_time_estimation = total_time_estimation + estimation_time

            logger.info(f'\taverage time estimation {total_time_estimation / (args.benchmark_count - 1)} ms')
    else:
        output = predict_sam_net(models, image_embedding, im_h, im_w, pos_points, neg_points, box)

    masks, scores = output
    logger.info(f'scores : {", ".join(["(%d) %.2f" % (i, s * 100) for i, s in enumerate(scores)])}')

    if sel_idx:
        i = sel_idx
    else:
        i = np.argmax(scores)
    mask = masks[i, :, :]
    score = scores[i]

    coord = []
    label = []
    if pos_points:
        coord.append(np.array(pos_points))
        label.append(np.ones(len(pos_points)))
    if neg_points:
        coord.append(np.array(neg_points))
        label.append(np.zeros(len(neg_points)))

    res_img = show_mask(mask, img)
    if coord:
        coord = np.concatenate(coord, axis=0)
        label = np.concatenate(label, axis=0)
        res_img = show_points(coord, label, res_img)
    if box is not None:
        res_img = show_box(box, res_img)

    # plot result
    savepath = get_savepath(args.savepath, image_path, ext='.png')
    logger.info(f'saved at : {savepath}')
    cv2.imwrite(savepath, res_img)


def predict_on_click(event,x,y,flags,param):
    global area_img, img, img_path, models, img_enc_output
    if event == cv2.EVENT_LBUTTONDOWN:
        img = cv2.imread(img_path)
        pos_points = [(x - 1, y - 1)]
        recognize(img_path, img, models, img_enc_output, pos_points, args.neg, args.box)
        for idy, yx in enumerate(area_img):
            for idx, xx in enumerate(yx):
                if (xx == np.array([255, 144, 30])).all():
                    cv2.rectangle(img, (idx - 1, idy - 1), (idx, idy), (0, 255, 0), -1)


def show_GUI(imgPath):
    global img, img_path
    img = cv2.imread(imgPath)
    #画像のウインドウに名前をつけ、コールバック関数をセット
    cv2.namedWindow('Mouse click GUI')
    cv2.setMouseCallback('Mouse click GUI', predict_on_click)
    
    while(1):
        cv2.imshow('Mouse click GUI', img)
        #ESCキーでブレーク
        if cv2.waitKey(20) & 0xFF == 27 or not cv2.getWindowProperty('Mouse click GUI', cv2.WND_PROP_VISIBLE):
            break
    cv2.destroyAllWindows()


def main():
    global img_path, models
    model_type = args.model_type
    dic_model = {
        'sam_h': (
            (WEIGHT_SAM_H_PATH, MODEL_SAM_H_PATH),
            (WEIGHT_VIT_H_PATH, MODEL_VIT_H_PATH)),
        'sam_l': (
            (WEIGHT_SAM_L_PATH, MODEL_SAM_L_PATH),
            (WEIGHT_VIT_L_PATH, MODEL_VIT_L_PATH)),
        'sam_b': (
            (WEIGHT_SAM_B_PATH, MODEL_SAM_B_PATH),
            (WEIGHT_VIT_B_PATH, MODEL_VIT_B_PATH)),
    }
    (WEIGHT_SAM_PATH, MODEL_SAM_PATH), (WEIGHT_VIT_PATH, MODEL_VIT_PATH) = dic_model[model_type]

    # model files check and download
    check_and_download_models(WEIGHT_SAM_PATH, MODEL_SAM_PATH, REMOTE_PATH)
    check_and_download_models(WEIGHT_VIT_PATH, MODEL_VIT_PATH, REMOTE_PATH)
    if model_type == "sam_h" and not os.path.exists(WEIGHT_VIT_H_PB_PATH):
        urlretrieve(
            REMOTE_PATH + WEIGHT_VIT_H_PB_PATH,
            WEIGHT_VIT_H_PB_PATH,
            progress_print,
        )

    env_id = args.env_id

    # initialize
    if not args.onnx:
        sam_net = ailia.Net(MODEL_SAM_PATH, WEIGHT_SAM_PATH, env_id=env_id, memory_mode=args.memory_mode)
        img_enc = ailia.Net(MODEL_VIT_PATH, WEIGHT_VIT_PATH, env_id=env_id, memory_mode=args.memory_mode)

        if args.profile:
            sam_net.set_profile_mode(True)
            img_enc.set_profile_mode(True)
    else:
        import onnxruntime

        cuda = 0 < ailia.get_gpu_environment_id()
        providers = ['CUDAExecutionProvider', 'CPUExecutionProvider'] if cuda else ['CPUExecutionProvider']
        sam_net = onnxruntime.InferenceSession(WEIGHT_SAM_PATH, providers=providers)
        img_enc = onnxruntime.InferenceSession(WEIGHT_VIT_PATH, providers=providers)

    models = dict(
        sam_net=sam_net,
        img_enc=img_enc,
    )
    recognize_from_image(models, args.pos, args.neg)
    if args.gui:
        for image_path in args.input:
            img_path = copy.deepcopy(image_path)
            show_GUI(img_path)

    if args.profile and not args.onnx:
        print("--- profile sam_net")
        print(sam_net.get_summary())
        print("")
        print("--- profile img_enc")
        print(img_enc.get_summary())
        print("")

if __name__ == '__main__':
    main()
