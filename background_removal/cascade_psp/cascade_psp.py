import sys
import time

import numpy as np
import cv2

import ailia

# import original modules
sys.path.append('../../util')
from arg_utils import get_base_parser, update_parser, get_savepath  # noqa
from model_utils import check_and_download_models  # noqa
from detector_utils import load_image  # noqa
from image_utils import normalize_image  # noqa
# logger
from logging import getLogger  # noqa: E402

logger = getLogger(__name__)

# ======================
# Parameters
# ======================
WEIGHT_INTER_S8_PATH = 'inter_s8.onnx'
MODEL_INTER_S8_PATH = 'inter_s8.onnx.prototxt'
WEIGHT_PATH = 'refinement.onnx'
MODEL_PATH = 'refinement.onnx.prototxt'
REMOTE_PATH = 'https://storage.googleapis.com/ailia-models/cascade_psp/'

IMAGE_PATH = 'aeroplane.jpg'
IMAGE_MASK_PATH = 'aeroplane.png'
SAVE_IMAGE_PATH = 'output.png'

INPUT_SIZE = 912
L = 900

# ======================
# Arguemnt Parser Config
# ======================

parser = get_base_parser('CascadePSP', IMAGE_PATH, SAVE_IMAGE_PATH)
parser.add_argument(
    '-m', '--mask_image', default=IMAGE_MASK_PATH,
    help='mask image'
)
parser.add_argument(
    '-g', '--generate_mask',
    action='store_true',
    help='Generate mask image using u2net'
)
parser.add_argument(
    '-c', '--composite',
    action='store_true',
    help='Composite input image and predicted alpha value'
)
args = update_parser(parser)


# ======================
# Secondaty Functions
# ======================

def resize(img, size=None, out_shape=None, method='bilinear'):
    if out_shape:
        oh, ow = out_shape
    else:
        h, w = img.shape[-2:]
        max_side = max(h, w)
        ratio = size / max_side
        oh = int(ratio * h)
        ow = int(ratio * w)

    use_pytorch = False
    if not use_pytorch:
        inp = {
            'bilinear': cv2.INTER_LINEAR,
            'bicubic': cv2.INTER_CUBIC,
            'area': cv2.INTER_AREA,
        }[method]
        img = img[0].transpose(1, 2, 0)
        img = cv2.resize(img, (ow, oh), interpolation=inp)
        img = img[:, :, None] if len(img.shape) < 3 else img
        img = img.transpose(2, 0, 1)
        img = img[None, :, :, :]
    else:
        import torch
        import torch.nn.functional as F

        img = torch.from_numpy(img)
        img = F.interpolate(img, (oh, ow), mode=method)
        img = np.asarray(img)

    return img


# ======================
# Generate mask image
# ======================

MASK_WEIGHT_PATH = 'u2net_opset11.onnx'
MASK_MODEL_PATH = MASK_WEIGHT_PATH + '.prototxt'
MASK_REMOTE_PATH = 'https://storage.googleapis.com/ailia-models/u2net/'

def imagenet_preprocess(input_data):
    input_data = input_data / 255.0
    input_data[:, :, 0] = (input_data[:, :, 0]-0.485)/0.229
    input_data[:, :, 1] = (input_data[:, :, 1]-0.456)/0.224
    input_data[:, :, 2] = (input_data[:, :, 2]-0.406)/0.225
    input_data = input_data.transpose((2, 0, 1))[np.newaxis, :, :, :]
    return input_data

def norm(pred):
    ma = np.max(pred)
    mi = np.min(pred)
    return (pred - mi) / (ma - mi)

def generate_mask(net, input_data):
    src_data = input_data.copy()
    
    h = input_data.shape[0]
    w = input_data.shape[1]

    input_shape = net.get_input_shape()
    input_data = cv2.resize(input_data, (input_shape[2], input_shape[3]))
    input_data = imagenet_preprocess(input_data)

    preds_ailia = net.predict([input_data])

    pred = preds_ailia[0][0, 0, :, :]
    pred = norm(pred)

    pred = cv2.resize(pred * 255, (w, h))
    pred = pred.reshape((h, w, 1))

    thre = 128
    pred[pred > thre] = 255
    pred[pred <= thre] = 0

    return pred


# ======================
# Main functions
# ======================

def preprocess(img, gray=False):
    if gray:
        img = img / 255
        img = (img - 0.5) / 0.5
        img = img[:, :, None]
    else:
        img = normalize_image(img, normalize_type='ImageNet')

    img = img.transpose(2, 0, 1)  # HWC -> CHW
    img = np.expand_dims(img, axis=0)
    img = img.astype(np.float32)

    return img


def safe_forward(net, img, seg, inter_s8=None):
    _, _, ph, pw = seg.shape

    oh = ow = INPUT_SIZE

    p_img = np.zeros((1, 3, oh, ow))
    p_seg = np.zeros((1, 1, oh, ow)) - 1
    p_img[:, :, 0:ph, 0:pw] = img
    p_seg[:, :, 0:ph, 0:pw] = seg
    img = p_img
    seg = p_seg

    if inter_s8 is not None:
        p_inter_s8 = np.zeros((1, 1, oh, ow)) - 1
        p_inter_s8[:, :, 0:ph, 0:pw] = inter_s8
        inter_s8 = p_inter_s8

        output = net.predict([img, seg, inter_s8])
    else:
        output = net.predict([img, seg])

    output = [x[:, :, 0:ph, 0:pw] for x in output]

    return output


def predict(net, net_s8, img, seg):
    im_h, im_w = img.shape[:2]
    seg_h, seg_w = seg.shape[:2]

    if im_h != seg_h or im_w != seg_w:
        logger.error('input image size is differ from mask mask image size.')
        sys.exit(-1)

    img = preprocess(img)
    seg = preprocess(seg, gray=True)

    """
    Global Step
    """
    logger.info("Begin global step")
    if max(im_h, im_w) > L:
        im_small = resize(img, size=L, method='area')
        seg_small = resize(seg, size=L, method='area')
    elif max(im_h, im_w) < L:
        im_small = resize(img, size=L, method='bicubic')
        seg_small = resize(seg, size=L, method='bilinear')
    else:
        im_small = img
        seg_small = seg

    output = safe_forward(net_s8, im_small, seg_small)
    inter_s8 = output[0]
    output = safe_forward(net, im_small, seg_small, inter_s8)
    pred_224 = output[0]
    pred_56 = output[2]

    """
    Local step
    """
    logger.info("Begin local step")
    new_size = max(im_h, im_w)
    im_small = resize(img, size=new_size, method='area')
    seg_small = resize(seg, size=new_size, method='area')
    _, _, h, w = seg_small.shape

    combined_224 = np.zeros_like(seg_small)
    combined_weight = np.zeros_like(seg_small)

    r_pred_224 = resize(pred_224, out_shape=(h, w), method='bilinear') > 0.5
    r_pred_224 = r_pred_224.astype(np.float32) * 2 - 1
    r_pred_56 = resize(pred_56, out_shape=(h, w), method='bilinear') * 2 - 1

    stride = L // 2
    padding = 16
    step_size = stride - padding * 2
    step_len = L

    used_start_idx = {}
    for x_idx in range(w // step_size + 1):
        logger.info("Processing "+str(x_idx)+"/"+str((w // step_size + 1)))
        for y_idx in range((h) // step_size + 1):
            start_x = x_idx * step_size
            start_y = y_idx * step_size
            end_x = start_x + step_len
            end_y = start_y + step_len

            # Shift when required
            if end_y > h:
                end_y = h
                start_y = h - step_len
            if end_x > w:
                end_x = w
                start_x = w - step_len

            # Bound x/y range
            start_x = max(0, start_x)
            start_y = max(0, start_y)
            end_x = min(w, end_x)
            end_y = min(h, end_y)

            # The same crop might appear twice due to bounding/shifting
            start_idx = start_y * w + start_x
            if start_idx in used_start_idx:
                continue
            else:
                used_start_idx[start_idx] = True

            # Take crop
            im_part = im_small[:, :, start_y:end_y, start_x:end_x]
            seg_224_part = r_pred_224[:, :, start_y:end_y, start_x:end_x]
            seg_56_part = r_pred_56[:, :, start_y:end_y, start_x:end_x]

            # Skip when it is not an interesting crop anyway
            seg_part_norm = (seg_224_part > 0).astype(np.float32)
            high_thres = 0.9
            low_thres = 0.1
            if (seg_part_norm.mean() > high_thres) or (seg_part_norm.mean() < low_thres):
                continue

            grid_images = safe_forward(net, im_part, seg_224_part, seg_56_part)
            grid_pred_224 = grid_images[0]

            # Padding
            pred_sx = pred_sy = 0
            pred_ex = step_len
            pred_ey = step_len

            if start_x != 0:
                start_x += padding
                pred_sx += padding
            if start_y != 0:
                start_y += padding
                pred_sy += padding
            if end_x != w:
                end_x -= padding
                pred_ex -= padding
            if end_y != h:
                end_y -= padding
                pred_ey -= padding

            combined_224[:, :, start_y:end_y, start_x:end_x] += grid_pred_224[:, :, pred_sy:pred_ey, pred_sx:pred_ex]

            del grid_pred_224

            # Used for averaging
            combined_weight[:, :, start_y:end_y, start_x:end_x] += 1

    # Final full resolution output
    seg_norm = (r_pred_224 / 2 + 0.5)
    pred_224 = np.divide(combined_224, combined_weight, out=seg_norm, where=combined_weight != 0)

    return pred_224[0, 0]


def recognize_from_image(net, net_s8):
    # prepare mask image
    if not args.generate_mask:
        mask_path = args.mask_image
        mask_img = load_image(mask_path)
        mask_img = cv2.cvtColor(mask_img, cv2.COLOR_BGRA2GRAY)

    # input image loop
    for image_path in args.input:
        # prepare input data
        logger.info(image_path)

        # prepare input data
        img = load_image(image_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)

        # generate mask
        if args.generate_mask:
            check_and_download_models(MASK_WEIGHT_PATH, MASK_MODEL_PATH, MASK_REMOTE_PATH)
            mask_net = ailia.Net(MASK_MODEL_PATH, MASK_WEIGHT_PATH, env_id=args.env_id, memory_mode=args.memory_mode)
            mask_output = generate_mask(mask_net, img)
            mask_img = mask_output[:,:,0]
            cv2.imwrite("generated_mask.png",mask_output)

        # inference
        logger.info('Start inference...')
        if args.benchmark:
            logger.info('BENCHMARK mode')
            total_time_estimation = 0
            for i in range(args.benchmark_count):
                start = int(round(time.time() * 1000))
                output = predict(net, net_s8, img, mask_img)
                end = int(round(time.time() * 1000))
                estimation_time = (end - start)

                # Loggin
                logger.info(f'\tailia processing estimation time {estimation_time} ms')
                if i != 0:
                    total_time_estimation = total_time_estimation + estimation_time

            logger.info(f'\taverage time estimation {total_time_estimation / (args.benchmark_count - 1)} ms')
        else:
            output = predict(net, net_s8, img, mask_img)

        # postprocessing
        res_img = (output * 255).astype(np.uint8)

        if args.composite:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2BGRA)
            img[:,:,3] = res_img
            res_img = img

        savepath = get_savepath(args.savepath, image_path, ext='.png')
        logger.info(f'saved at : {savepath}')
        cv2.imwrite(savepath, res_img)

    logger.info('Script finished successfully.')


def main():
    logger.info('Checking refinement model...')
    check_and_download_models(WEIGHT_PATH, MODEL_PATH, REMOTE_PATH)
    logger.info('Checking s8 model...')
    check_and_download_models(WEIGHT_INTER_S8_PATH, MODEL_INTER_S8_PATH, REMOTE_PATH)

    if args.video:
        logger.error("This model does not support video mode.")
        return

    # load model
    env_id = args.env_id

    # net initialize
    logger.info("This model requires 10GB or more memory.")
    net = ailia.Net(MODEL_PATH, WEIGHT_PATH, env_id=env_id, memory_mode=args.memory_mode)
    net_s8 = ailia.Net(MODEL_INTER_S8_PATH, WEIGHT_INTER_S8_PATH, env_id=env_id, memory_mode=args.memory_mode)

    recognize_from_image(net, net_s8)


if __name__ == '__main__':
    main()
