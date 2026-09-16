import sys
import time

import numpy as np
import cv2
from PIL import Image

import ailia

# import original modules
sys.path.append('../../util')
from arg_utils import get_base_parser, update_parser, get_savepath  # noqa
from model_utils import check_and_download_models  # noqa
from detector_utils import load_image  # noqa
from image_utils import normalize_image  # noqa
from webcamera_utils import get_capture, get_writer  # noqa
# logger
from logging import getLogger  # noqa

import face_detect_crop
from face_detect_crop import crop_face, get_kps
import face_align
import image_infer
from image_infer import setup_mxnet, get_landmarks
from masks import face_mask_static

logger = getLogger(__name__)

use_pytorch = False

# ======================
# Parameters
# ======================

WEIGHT_G_PATH = 'G_unet_2blocks.onnx'
MODEL_G_PATH = 'G_unet_2blocks.onnx.prototxt'
WEIGHT_ARCFACE_PATH = 'scrfd_10g_bnkps.onnx'
MODEL_ARCFACE_PATH = 'scrfd_10g_bnkps.onnx.prototxt'
WEIGHT_BACKBONE_PATH = 'arcface_backbone.onnx'
MODEL_BACKBONE_PATH = 'arcface_backbone.onnx.prototxt'
WEIGHT_LANDMARK_PATH = 'face_landmarks.onnx'
MODEL_LANDMARK_PATH = 'face_landmarks.onnx.prototxt'
WEIGHT_PIX2PIX_PATH = '10_net_G.onnx'
MODEL_PIX2PIX_PATH = '10_net_G.onnx.prototxt'
REMOTE_PATH = 'https://storage.googleapis.com/ailia-models/sber-swap/'

IMAGE_PATH = 'beckham.jpg'
SOURCE_PATH = 'elon_musk.jpg'
SAVE_IMAGE_PATH = 'output.png'

CROP_SIZE = 224

IOU = 0.4

# ======================
# Arguemnt Parser Config
# ======================

parser = get_base_parser(
    'SberSwap', IMAGE_PATH, SAVE_IMAGE_PATH
)
parser.add_argument(
    '-src', '--source', default=SOURCE_PATH,
    help='source image'
)
parser.add_argument(
    '-iou', '--iou',
    default=IOU, type=float,
    help='IOU threshold for NMS'
)
parser.add_argument(
    '--use_sr',
    action='store_true',
    help='True for super resolution on swap image'
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

def get_final_img(output, tar_img, net_lmk):
    final_img, crop_img, tfm = output

    h, w = tar_img.shape[:2]
    final = tar_img.copy()

    landmarks = get_landmarks(net_lmk, final_img)
    landmarks_tgt = get_landmarks(net_lmk, crop_img)

    mask, _ = face_mask_static(
        crop_img, landmarks, landmarks_tgt, None)
    mat_rev = cv2.invertAffineTransform(tfm)

    swap_t = cv2.warpAffine(final_img, mat_rev, (w, h), borderMode=cv2.BORDER_REPLICATE)
    mask_t = cv2.warpAffine(mask, mat_rev, (w, h))
    mask_t = np.expand_dims(mask_t, 2)

    final = mask_t * swap_t + (1 - mask_t) * final
    final = final.astype(np.uint8)

    return final


# ======================
# Main functions
# ======================

def preprocess(img, half_scale=True):
    if half_scale and not use_pytorch:
        im_h, im_w, _ = img.shape
        img = np.array(Image.fromarray(img).resize(
            (im_w // 2, im_h // 2), Image.Resampling.BILINEAR))

    img = normalize_image(img, normalize_type='127.5')

    img = img.transpose(2, 0, 1)  # HWC -> CHW
    img = np.expand_dims(img, axis=0)
    img = img.astype(np.float32)

    if half_scale and use_pytorch:
        import torch
        import torch.nn.functional as F
        img = F.interpolate(
            torch.from_numpy(img), scale_factor=0.5, mode='bilinear', align_corners=True
        ).numpy()

    return img


def predict(net_iface, net_G, src_embeds, tar_img):
    kps = get_kps(tar_img, net_iface, nms_threshold=args.iou)

    if kps is None:
        return None

    M, _ = face_align.estimate_norm(kps[0], CROP_SIZE, mode='None')
    crop_img = cv2.warpAffine(tar_img, M, (CROP_SIZE, CROP_SIZE), borderValue=0.0)

    new_size = (256, 256)
    img = cv2.resize(crop_img, new_size)
    img = preprocess(img[:, :, ::-1], half_scale=False)
    img = img.astype(np.float16)

    # feedforward
    if not args.onnx:
        output = net_G.predict([img, src_embeds])
    else:
        output = net_G.run(None, {'target': img, 'source_emb': src_embeds})
    y_st = output[0]

    y_st = y_st[0].transpose(1, 2, 0)
    y_st = y_st * 127.5 + 127.5
    y_st = y_st[:, :, ::-1]  # RGB -> BGR
    y_st = y_st.astype(np.uint8)

    final_img = cv2.resize(y_st, (CROP_SIZE, CROP_SIZE))

    return final_img, crop_img, M


def face_enhancement(net_pix2pix, output):
    final_img, crop_img, M = output
    final_img = cv2.resize(final_img, (256, 256))
    final_img = final_img[:, :, ::-1]
    final_img = final_img.astype(np.float32)
    final_img = final_img / 255.0
    final_img = np.expand_dims(final_img, axis = 0)
    final_img = np.transpose(final_img, (0, 3, 1, 2))
    final_img = net_pix2pix.predict(final_img)
    final_img = final_img * 255
    final_img = np.clip(final_img, 0, 255)
    final_img = final_img.astype(np.uint8)
    final_img = np.transpose(final_img, (0, 2, 3, 1))
    final_img = final_img[0, :, :, ::-1]
    final_img = cv2.resize(final_img, (CROP_SIZE, CROP_SIZE))
    return final_img, crop_img, M 


def recognize_from_image(net_iface, net_back, net_G, net_lmk, net_pix2pix):
    source_path = args.source
    logger.info('Source: {}'.format(source_path))

    src_img = load_image(source_path)
    src_img = cv2.cvtColor(src_img, cv2.COLOR_BGRA2BGR)
    src_img = crop_face(src_img, net_iface, CROP_SIZE, nms_threshold=args.iou)
    if src_img is None:
        logger.info("Source face not recognized.")
        sys.exit(0)
    src_img = src_img[:, :, ::-1]  # BGR -> RGB

    # source embeds
    img = preprocess(src_img)
    if not args.onnx:
        output = net_back.predict([img])
    else:
        output = net_back.run(None, {'img': img})
    src_embeds = output[0]
    src_embeds = src_embeds.astype(np.float16)

    # input image loop
    for image_path in args.input:
        logger.info('Target: {}'.format(image_path))

        # prepare input data
        tar_img = load_image(image_path)
        tar_img = cv2.cvtColor(tar_img, cv2.COLOR_BGRA2BGR)

        # inference
        logger.info('Start inference...')
        if args.benchmark:
            logger.info('BENCHMARK mode')
            total_time_estimation = 0
            for i in range(args.benchmark_count):
                start = int(round(time.time() * 1000))
                output = predict(net_iface, net_G, src_embeds, tar_img)
                end = int(round(time.time() * 1000))
                estimation_time = (end - start)

                # Loggin
                logger.info(f'\tailia processing estimation time {estimation_time} ms')
                if i != 0:
                    total_time_estimation = total_time_estimation + estimation_time

            logger.info(f'\taverage time estimation {total_time_estimation / (args.benchmark_count - 1)} ms')
        else:
            output = predict(net_iface, net_G, src_embeds, tar_img)

        if output is None:
            logger.info("Target face not recognized.")
            continue
            
        if args.use_sr:
            output = face_enhancement(net_pix2pix, output)

        res_img = get_final_img(output, tar_img, net_lmk)

        # plot result
        savepath = get_savepath(args.savepath, image_path, ext='.png')
        logger.info(f'saved at : {savepath}')
        cv2.imwrite(savepath, res_img)

    logger.info('Script finished successfully.')


def recognize_from_video(net_iface, net_back, net_G, net_lmk, net_pix2pix):
    video_file = args.video if args.video else args.input[0]
    capture = get_capture(video_file)
    assert capture.isOpened(), 'Cannot capture source'

    source_path = args.source
    logger.info('Source: {}'.format(source_path))

    src_img = load_image(source_path)
    src_img = cv2.cvtColor(src_img, cv2.COLOR_BGRA2BGR)
    src_img = crop_face(src_img, net_iface, CROP_SIZE)
    if src_img is None:
        logger.info("Source face not recognized.")
        sys.exit(0)
    src_img = src_img[:, :, ::-1]  # BGR -> RGB

    # source embeds
    img = preprocess(src_img)
    if not args.onnx:
        output = net_back.predict([img])
    else:
        output = net_back.run(None, {'img': img})
    src_embeds = output[0]
    src_embeds = src_embeds.astype(np.float16)

    # create video writer if savepath is specified as video format
    if args.savepath != SAVE_IMAGE_PATH:
        f_h = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        f_w = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        writer = get_writer(args.savepath, f_h, f_w)
    else:
        writer = None

    frame_shown = False
    while True:
        ret, frame = capture.read()
        if (cv2.waitKey(1) & 0xFF == ord('q')) or not ret:
            break
        if frame_shown and cv2.getWindowProperty('frame', cv2.WND_PROP_VISIBLE) == 0:
            break

        # inference
        output = predict(net_iface, net_G, src_embeds, frame)

        if args.use_sr:
            output = face_enhancement(net_pix2pix, output)

        if output:
            # plot result
            res_img = get_final_img(output, frame, net_lmk)
        else:
            res_img = frame

        # show
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
    logger.info('Checking G model...')
    check_and_download_models(WEIGHT_G_PATH, MODEL_G_PATH, REMOTE_PATH)
    logger.info('Checking arcface model...')
    check_and_download_models(WEIGHT_ARCFACE_PATH, MODEL_ARCFACE_PATH, REMOTE_PATH)
    logger.info('Checking backbone model...')
    check_and_download_models(WEIGHT_BACKBONE_PATH, MODEL_BACKBONE_PATH, REMOTE_PATH)
    logger.info('Checking landmark model...')
    check_and_download_models(WEIGHT_LANDMARK_PATH, MODEL_LANDMARK_PATH, REMOTE_PATH)
    if args.use_sr:
        logger.info('Checking pix2pix model...')
        check_and_download_models(WEIGHT_PIX2PIX_PATH, MODEL_PIX2PIX_PATH, REMOTE_PATH)

    env_id = args.env_id

    # initialize
    if not args.onnx:
        net_iface = ailia.Net(MODEL_ARCFACE_PATH, WEIGHT_ARCFACE_PATH, env_id=env_id, memory_mode=args.memory_mode)
        net_back = ailia.Net(MODEL_BACKBONE_PATH, WEIGHT_BACKBONE_PATH, env_id=env_id, memory_mode=args.memory_mode)
        net_G = ailia.Net(MODEL_G_PATH, WEIGHT_G_PATH, env_id=env_id, memory_mode=args.memory_mode)
        net_lmk = ailia.Net(MODEL_LANDMARK_PATH, WEIGHT_LANDMARK_PATH, env_id=env_id, memory_mode=args.memory_mode)
        if args.use_sr:
            pix2pix_env_id = env_id
            if "FP16" in ailia.get_environment(args.env_id).props:  # disable FP16
                pix2pix_env_id = 0
            net_pix2pix = ailia.Net(MODEL_PIX2PIX_PATH, WEIGHT_PIX2PIX_PATH, env_id=pix2pix_env_id, memory_mode=args.memory_mode)
        else:
            net_pix2pix = None
    else:
        import onnxruntime
        net_iface = onnxruntime.InferenceSession(WEIGHT_ARCFACE_PATH)
        net_back = onnxruntime.InferenceSession(WEIGHT_BACKBONE_PATH)
        net_G = onnxruntime.InferenceSession(WEIGHT_G_PATH)
        net_lmk = onnxruntime.InferenceSession(WEIGHT_LANDMARK_PATH)
        if args.use_sr:
            net_pix2pix = onnxruntime.InferenceSession(WEIGHT_PIX2PIX_PATH)
        else:
            net_pix2pix = None

        face_detect_crop.onnx = True
        image_infer.onnx = True

    # # use mxnet
    # net_lmk = setup_mxnet()

    if args.video is not None:
        recognize_from_video(net_iface, net_back, net_G, net_lmk, net_pix2pix)
    else:
        recognize_from_image(net_iface, net_back, net_G, net_lmk, net_pix2pix)


if __name__ == '__main__':
    main()
