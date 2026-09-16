import sys
import time

import numpy as np
import cv2

import ailia

# import original modules
sys.path.append('../../util')
from arg_utils import get_base_parser, update_parser, get_savepath  # noqa: E402
from model_utils import check_and_download_models  # noqa: E402
from detector_utils import plot_results, load_image  # noqa: E402
from image_utils import normalize_image  # noqa: E402
import webcamera_utils  # noqa: E402

# logger
from logging import getLogger  # noqa: E402

logger = getLogger(__name__)

# ======================
# Parameters
# ======================
WEIGHT_PATH = './mask_rcnn_r50_fpn_1x.onnx'
MODEL_PATH = './mask_rcnn_r50_fpn_1x.onnx.prototxt'
REMOTE_PATH = 'https://storage.googleapis.com/ailia-models/mmfashion/'

IMAGE_PATH = '01_4_full.jpg'
SAVE_IMAGE_PATH = 'output.png'

CATEGORY = (
    'top', 'skirt', 'leggings', 'dress', 'outer', 'pants', 'bag',
    'neckwear', 'headwear', 'eyeglass', 'belt', 'footwear', 'hair',
    'skin', 'face'
)
THRESHOLD = 0.6

RESIZE_RANGE = (750, 1101)
NORM_MEAN = [123.675, 116.28, 103.53]
NORM_STD = [58.395, 57.12, 57.375]
RCNN_MASK_THRE = 0.5

U2NET_MODEL_LIST = ['small', 'large']
WEIGHT_U2NET_LARGE_PATH = 'u2net_opset11.onnx'
MODEL_U2NET_LARGE_PATH = 'u2net_opset11.onnx.prototxt'
WEIGHT_U2NET_SMALL_PATH = 'u2netp_opset11.onnx'
MODEL_U2NET_SMALL_PATH = 'u2netp_opset11.onnx.prototxt'
REMOTE_U2NET_PATH = 'https://storage.googleapis.com/ailia-models/u2net/'
U2NET_IMAGE_SIZE = 320

# ======================
# Arguemnt Parser Config
# ======================
parser = get_base_parser('MMFashion model', IMAGE_PATH, SAVE_IMAGE_PATH)
parser.add_argument(
    '-th', '--threshold',
    default=THRESHOLD, type=float,
    help='The detection threshold for yolo. (default: '+str(THRESHOLD)+')'
)
parser.add_argument(
    '-pp', '--preprocess', metavar='ARCH',
    default=None, choices=U2NET_MODEL_LIST,
    help='preprocess model (U square net) architecture: ' + ' | '.join(U2NET_MODEL_LIST)
)
args = update_parser(parser)


# ======================
# Secondaty Functions
# ======================
def transform(img, pp_net):
    img_0 = img

    img = cv2.resize(img, (U2NET_IMAGE_SIZE, U2NET_IMAGE_SIZE))

    # ToTensorLab part in original repo
    img = img / np.max(img) * 255
    img = normalize_image(img, normalize_type='ImageNet')
    input_data = img.transpose((2, 0, 1))[np.newaxis, :, :, :]

    output = pp_net.predict(input_data)
    pred = output[0, 0, :, :]

    h, w = img_0.shape[:2]
    mask = cv2.resize(pred, (w, h))
    mask = np.clip(mask, 0, 1)
    mask = np.expand_dims(mask, axis=2)

    back = np.ones((h, w, 3)) * 255
    img = img_0 * mask + back * (1 - mask)

    return img


def preprocess(img):
    h, w = img.shape[:2]

    # scale
    max_long_edge = max(RESIZE_RANGE)
    max_short_edge = min(RESIZE_RANGE)
    scale_factor = min(max_long_edge / max(h, w), max_short_edge / min(h, w))
    new_w = int(w * float(scale_factor) + 0.5)
    new_h = int(h * float(scale_factor) + 0.5)
    img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    scale_w = new_w / w
    scale_h = new_h / h

    # normalize
    img = img.astype(np.float32)
    mean = np.array(NORM_MEAN)
    std = np.array(NORM_STD)
    mean = np.float64(mean.reshape(1, -1))
    stdinv = 1 / np.float64(std.reshape(1, -1))
    cv2.subtract(img, mean, img)  # inplace
    cv2.multiply(img, stdinv, img)  # inplace

    # padding
    divisor = 32
    pad_h = int(np.ceil(img.shape[0] / divisor)) * divisor
    pad_w = int(np.ceil(img.shape[1] / divisor)) * divisor
    img = cv2.copyMakeBorder(
        img, 0, pad_h - img.shape[0], 0, pad_w - img.shape[1],
        cv2.BORDER_CONSTANT, value=0)

    img = img.transpose(2, 0, 1)
    img = np.expand_dims(img, 0)
    data = {
        'img': img,
        'scale_factor': (scale_h, scale_w),
        'ori_shape': (h, w, 3),
        'img_shape': (new_h, new_w, 3),
        'pad_shape': (pad_h, pad_w, 3),
    }
    return data


def post_processing(data, boxes, labels, masks):
    bbox_list = [boxes[labels == i, :] for i in range(len(CATEGORY))]
    mask_list = [masks[labels == i, :] for i in range(len(CATEGORY))]

    ###########################################
    # remove duplicate
    new_bbox_list = []
    new_mask_list = []
    for idx, (bbox, mask) in enumerate(zip(bbox_list, mask_list)):
        if len(bbox) < 1:
            new_bbox_list.append(None)
            new_mask_list.append(None)
            continue

        i = np.argmax(bbox[:, -1])
        new_bbox_list.append(bbox[i, :])
        new_mask_list.append(mask[i, :])

    bbox_list = new_bbox_list
    mask_list = new_mask_list
    #########################################

    ori_shape = data['ori_shape'][:2]
    img_shape = data['img_shape'][:2]
    scale_factor = data['scale_factor']

    ret_boxes = []
    segm_masks = []
    for cls_ind, (box, mask) in enumerate(zip(bbox_list, mask_list)):
        if box is None:
            continue

        score = box[-1]
        x, y, x2, y2 = box[:4]

        if score < args.threshold:
            continue

        w = (x2 - x)
        h = (y2 - y)
        ori_x = int(x / scale_factor[1])
        ori_y = int(y / scale_factor[0])
        ori_x2 = int(x2 / scale_factor[1])
        ori_y2 = int(y2 / scale_factor[0])
        ori_w = int(w / scale_factor[1])
        ori_h = int(h / scale_factor[0])

        # segment mask
        mask = cv2.resize(mask, (ori_w, ori_h), interpolation=cv2.INTER_LINEAR)
        segm_mask = np.zeros(
            (max(ori_shape[0], ori_y2), max(ori_shape[1], ori_x2))
        )
        segm_mask[ori_y:ori_y + ori_h, ori_x:ori_x + ori_w] = mask
        segm_mask = segm_mask[:ori_shape[0], :ori_shape[1]]
        segm_mask = (segm_mask > RCNN_MASK_THRE).astype(np.uint8)

        # bbox
        w = w / img_shape[1]
        h = h / img_shape[0]
        x = x / img_shape[1]
        y = y / img_shape[0]
        r = ailia.DetectorObject(
            category=cls_ind, prob=score,
            x=x, y=y, w=w, h=h,
        )
        ret_boxes.append(r)
        segm_masks.append(segm_mask)

    return ret_boxes, segm_masks


# ======================
# Main functions
# ======================


def detect_objects(img, detector, pp_net):
    # initial preprocesses
    if pp_net:
        img = transform(img, pp_net)
    data = preprocess(img)

    # feedforward
    detector.set_input_shape(
        (1, 3, data['img'].shape[2], data['img'].shape[3])
    )
    output = detector.predict({
        'image': data['img']
    })
    boxes, labels, masks = output

    # post processes
    detect_object, seg_masks = post_processing(data, boxes, labels, masks)

    return detect_object, seg_masks


def recognize_from_image(filename, detector, pp_net):
    # prepare input data
    img_0 = load_image(filename)
    logger.debug(f'input image shape: {img_0.shape}')

    img = cv2.cvtColor(img_0, cv2.COLOR_BGRA2RGB)

    # inference
    logger.info('Start inference...')
    if args.benchmark:
        logger.info('BENCHMARK mode')
        for i in range(5):
            start = int(round(time.time() * 1000))
            detect_object, seg_masks = detect_objects(img, detector, pp_net)
            end = int(round(time.time() * 1000))
            logger.info(f'\tailia processing time {end - start} ms')
    else:
        detect_object, seg_masks = detect_objects(img, detector, pp_net)

    # plot result
    res_img = plot_results(
        detect_object, img_0, CATEGORY, segm_masks=seg_masks
    )
    savepath = get_savepath(args.savepath, filename)
    logger.info(f'saved at : {savepath}')
    cv2.imwrite(savepath, res_img)


def recognize_from_video(video, detector, pp_net):
    capture = webcamera_utils.get_capture(args.video)

    # create video writer if savepath is specified as video format
    if args.savepath != SAVE_IMAGE_PATH:
        f_h = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        f_w = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        writer = webcamera_utils.get_writer(args.savepath, f_h, f_w)
    else:
        writer = None

    frame_shown = False
    while True:
        ret, frame = capture.read()
        if cv2.waitKey(1) & 0xFF == ord('q') or not ret:
            break
        if frame_shown and cv2.getWindowProperty('frame', cv2.WND_PROP_VISIBLE) == 0:
            break

        x = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        detect_object, seg_masks = detect_objects(x, detector, pp_net)
        res_img = plot_results(
            detect_object, frame, CATEGORY, segm_masks=seg_masks
        )
        cv2.imshow('frame', res_img)
        frame_shown = True

        # save results
        if writer is not None:
            writer.write(res_img)

    capture.release()
    cv2.destroyAllWindows()
    if writer is not None:
        writer.release()
    logger.info('Script finished successfully.')


def main():
    # model files check and download
    logger.info('=== MMFashion model ===')
    check_and_download_models(WEIGHT_PATH, MODEL_PATH, REMOTE_PATH)
    if args.preprocess:
        info = {
            'large': (
                WEIGHT_U2NET_LARGE_PATH, MODEL_U2NET_LARGE_PATH),
            'small': (
                WEIGHT_U2NET_SMALL_PATH, MODEL_U2NET_SMALL_PATH),
        }
        weight_path, model_path = info[args.preprocess]
        logger.info('=== U square net model ===')
        check_and_download_models(weight_path, model_path, REMOTE_U2NET_PATH)
    else:
        weight_path = model_path = None

    # initialize
    detector = ailia.Net(MODEL_PATH, WEIGHT_PATH, env_id=args.env_id, memory_mode=args.memory_mode)
    if weight_path:
        pp_net = ailia.Net(model_path, weight_path, env_id=args.env_id, memory_mode=args.memory_mode)
    else:
        pp_net = None

    if args.video is not None:
        # video mode
        recognize_from_video(args.video, detector, pp_net)
    else:
        # image mode
        # input image loop
        for image_path in args.input:
            recognize_from_image(image_path, detector, pp_net)
    logger.info('Script finished successfully.')


if __name__ == '__main__':
    main()
