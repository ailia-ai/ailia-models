import argparse
import colorsys
import sys
import time

import ailia
import cv2
import numpy as np

import ax_facial_features_utils as fut

sys.path.append('../../util')
# logger
from logging import getLogger  # noqa: E402

from image_utils import imread, load_image  # noqa: E402
from math_utils import softmax
from model_utils import check_and_download_models  # noqa: E402
from arg_utils import get_base_parser, get_savepath, update_parser  # noqa: E402
from webcamera_utils import get_capture, get_writer  # noqa: E402

logger = getLogger(__name__)


# ======================
# Parameters 1
# ======================
IMAGE_PATH = 'man-with-beard.jpg'
SAVE_IMAGE_PATH = 'output.png'
EYELIDS_CLASSES = ['double', 'single']
EYELASHES_CLASSES = ['dense', 'moderate', 'sparse']
FACIAL_HAIR_CLASSES = ['moustache', 'beard', 'mouth_side_hair']


# ======================
# Argument Parser Config
# ======================
parser:argparse.ArgumentParser = get_base_parser(
    'ax Facial Features',
    IMAGE_PATH,
    SAVE_IMAGE_PATH,
)
parser.add_argument(
    '-m', '--mode', nargs='+', default=['eyelids', 'eyelashes', 'facial_hair']
)
args = update_parser(parser)


# ======================
# Parameters 2
# ======================
MODELS = {
    'blazeface': {},
    'facemesh': {},
    'ax_eyelids': {},
    'ax_eyelashes': {},
    'ax_facial_hair': {},
}

for k, v in MODELS.items():
    stem = f'{k}.opt'

    if k[:3] == 'ax_':
        stem += '.obf'
        v['remote_path'] = f'https://storage.googleapis.com/ailia-models/ax_facial_features/'

    if 'weight_path' not in v:
        v['weight_path'] = f'{stem}.onnx'
    if 'model_path' not in v:
        v['model_path'] = f'{stem}.onnx.prototxt'
    if 'remote_path' not in v:
        v['remote_path'] = f'https://storage.googleapis.com/ailia-models/{k}/'


# ======================
# Utils
# ======================
def print_results(results):
    """Print all facial features classification results"""
    for res in results:
        tmp = ['top_left', 'top_right', 'bottom_left', 'bottom_right']
        logger.info('ROI')
        logger.info('==============================================================')
        for i, c in enumerate(tmp):
            logger.info(f'{c} = ({res["roi"][i][0]:.2f}, {res["roi"][i][1]:.2f})'
                        f' (x, y)')
        logger.info('')

        if 'eyelids' in res:
            logger.info('Eyelids')
            fut.print_results(res['eyelids'], EYELIDS_CLASSES, logger)
        if 'eyelashes' in res:
            logger.info('Eyelashes')
            fut.print_results(res['eyelashes'], EYELASHES_CLASSES, logger)
        if 'facial_hair' in res:
            logger.info('Facial hair')
            fut.print_results(res['facial_hair'], FACIAL_HAIR_CLASSES, logger,
                          multilabel=True)

def plot_results(img, results, horizontal_flip=False):
    """Plot the facial features classification results on the image"""
    img_w = img.shape[1]

    l = [
        ['eyelids', EYELIDS_CLASSES, False],
        ['eyelashes', EYELASHES_CLASSES, False],
        ['facial_hair', FACIAL_HAIR_CLASSES, True],
    ]

    n = 5
    hsv_tuples = np.asarray([
        (1 * x / n, 1., 1.) for x in range(n)])
    colors = np.apply_along_axis(
        lambda x: colorsys.hsv_to_rgb(*x), 1, hsv_tuples)
    colors = (colors * 255).astype(np.uint8) # 0-255 BGR
    rng = np.random.default_rng(0)
    rng.shuffle(colors)
    
    if horizontal_flip:
        img_draw = np.ascontiguousarray(img[:, ::-1])
    else:
        img_draw = img.copy()

    for res in results:
        roi = np.stack(res['roi'])
        if horizontal_flip:
            roi[:, 0] = img_w - roi[:, 0] # Flip coordinates
            # Flip top left with top right,...
            tmp = roi[::2].copy()
            roi[::2] = roi[1::2]
            roi[1::2] = tmp

        fut.draw_roi(img_draw, roi)
        x, y = tuple(int(ee) for ee in (roi[0] + 2))
        x = max(2, x)
        y = max(2, y)

        for e, c in zip(l, colors):
            scores = res[e[0]]
            labels = e[1]
            ids_order = fut.filter_sort_results(scores, labels, multilabel=e[2])

            color = tuple(int(ee) for ee in c)
            text = ''
            for idx in ids_order:
                text += f'{labels[idx]} ({scores[idx]*100:4.2f}%); '
            text = text[:-2]

            # txt_color = (0, 0, 0) if np.mean(color) > 127.5 else (255, 255, 255)
            txt_color = (0, 0, 0)
            txt_bbox_thick = 1
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = img_w / 1450
            txt_size = cv2.getTextSize(text, font, font_scale, txt_bbox_thick)[0]
            w = txt_size[0] + 4
            h = txt_size[1] + 8
            text_position = (x + 4, y + h//2 + 4)

            cv2.rectangle(img_draw, (x, y), (x + w, y + h), color, thickness=-1)
            cv2.putText(img_draw, text, text_position, font, font_scale, txt_color, 1)

            y = y + h + 2

    return img_draw

def init():
    """Initialize all ailia models"""
    # net initialize
    models = {}
    for k, v in MODELS.items():
        models[k] = ailia.Net(v['model_path'], v['weight_path'],
                                env_id=args.env_id, memory_mode=args.memory_mode)
    return models

def predict(models, raw_img):
    """Compute facial features classification predictions"""
    res = []

    # Face detection
    input_data, scale, padding = fut.face_detector_preprocess(raw_img)
    preds = models['blazeface'].predict([input_data])
    detections = fut.face_detector_postprocess(preds)

    # Face landmark estimation
    if detections[0].size != 0:
        face_imgs, face_affs, _, _ = fut.face_lm_preprocess(
            raw_img[:, :, ::-1], detections, scale, padding
        )
        models['facemesh'].set_input_shape(face_imgs.shape)
        preds = models['facemesh'].predict([face_imgs])
        landmarks, _, _ = fut.face_lm_postprocess(preds, face_affs)

        for i in range(len(landmarks)):
            res_i = {}
            lm = landmarks[i:i+1]
            res_i['roi'] = fut.get_roi(lm)

            if 'eyelids' in args.mode or 'eyelashes' in args.mode:
                in_data = fut.facial_features_preprocess(raw_img, lm, 'eyes')

                if 'eyelids' in args.mode:
                    preds = models['ax_eyelids'].predict(in_data)[0]
                    scores = softmax(preds)
                    res_i['eyelids'] = scores

                if 'eyelashes' in args.mode:
                    preds = models['ax_eyelashes'].predict(in_data)[0]
                    scores = softmax(preds)
                    res_i['eyelashes'] = scores

            if 'facial_hair' in args.mode:
                in_data = fut.facial_features_preprocess(raw_img, lm, 'face')
                preds = models['ax_facial_hair'].predict(in_data)[0]
                res_i['facial_hair'] = preds

            res.append(res_i)

    return res


# ======================
# Main functions
# ======================
def recognize_from_image():
    models = init()

    if args.benchmark:
        logger.info('BENCHMARK mode')
        n = 5
        timings = [None] * n
        for i in range(n):
            start = time.perf_counter()
            for img_path in args.input:
                img = imread(img_path)
                res = predict(models, img)
                print_results(res)
            duration = time.perf_counter() - start
            logger.info(f'\tailia processing time {round(duration * 1e3):.0f} ms')
            timings[i] = duration
        logger.info(
            f'\tmean ailia processing time {round(np.mean(timings) * 1e3):.0f} '
            f'ms ({n} run(s))'
        )
    else:
        for img_path in args.input:
            img = imread(img_path)
            res = predict(models, img)
            print_results(res)
            # plot_results(img, res, horizontal_flip=True)

    logger.info('Script finished successfully.')


def recognize_from_video():
    models = init()

    capture = get_capture(args.video)

    # create video writer if savepath is specified as video format
    if args.savepath != SAVE_IMAGE_PATH:
        f_h = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        f_w = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        writer = get_writer(args.savepath, f_h, f_w)
    else:
        writer = None
    
    frame_shown = False
    while(True):
        ret, frame = capture.read()
        if (cv2.waitKey(1) & 0xFF == ord('q')) or not ret:
            break
        if frame_shown and cv2.getWindowProperty('frame', cv2.WND_PROP_VISIBLE) == 0:
            break
        # inference
        res = predict(models, frame)
        if len(res) > 0:
            frame_draw = plot_results(frame, res)
        else:
            frame_draw = frame.copy()

        if args.video == '0': # Flip horizontally if camera
            if len(res) > 0:
                visual_img = plot_results(frame, res, horizontal_flip=True)
            else:
                visual_img = np.ascontiguousarray(frame[:, ::-1])
        else:
            visual_img = frame_draw

        cv2.imshow('frame', visual_img)
        frame_shown = True

        # save results
        if writer is not None:
            writer.write(frame_draw)

    capture.release()
    if writer is not None:
        writer.release()
    cv2.destroyAllWindows()
    if writer is not None:
        writer.release()
    logger.info('Script finished successfully.')


def main():
    # model files check and download
    for model in MODELS.values():
        check_and_download_models(
            model['weight_path'], model['model_path'], model['remote_path']
        )

    if args.video is not None:
        # video mode
        recognize_from_video()
    else:
        # image mode
        recognize_from_image()


if __name__ == '__main__':
    main()
