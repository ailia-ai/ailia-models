import json
import sys
import time

import numpy as np
import cv2

import ailia

# import original modules
sys.path.append('../../util')
from arg_utils import get_base_parser, update_parser, get_savepath  # noqa
from model_utils import check_and_download_models  # noqa
from image_utils import normalize_image  # noqa
from detector_utils import load_image  # noqa
from webcamera_utils import get_capture, get_writer  # noqa
from dtype_utils import numpy_type_to_builtin_type  # noqa
# logger
from logging import getLogger  # noqa

logger = getLogger(__name__)

# ======================
# Parameters
# ======================

WEIGHT_ResNet50_320_PATH = 'COCO_ResNet50_320x320.onnx'
MODEL_ResNet50_320_PATH = 'COCO_ResNet50_320x320.onnx.prototxt'
WEIGHT_ResNet101_512_PATH = 'COCO_ResNet101_512x512.onnx'
MODEL_ResNet101_512_PATH = 'COCO_ResNet101_512x512.onnx.prototxt'
WEIGHT_ResNet152_448_PATH = 'COCO_ResNet152_448x448.onnx'
MODEL_ResNet152_448_PATH = 'COCO_ResNet152_448x448.onnx.prototxt'
WEIGHT_MobleNetV2_320_PATH = 'COCO_MobileNetV2_320x320.onnx'
MODEL_MobleNetV2_320_PATH = 'COCO_MobileNetV2_320x320.onnx.prototxt'
WEIGHT_MobleNetV2_448_PATH = 'COCO_MobileNetV2_448x512.onnx'
MODEL_MobleNetV2_448_PATH = 'COCO_MobileNetV2_448x512.onnx.prototxt'
REMOTE_PATH = 'https://storage.googleapis.com/ailia-models/e2pose/'

IMAGE_PATH = 'input.jpg'
SAVE_IMAGE_PATH = 'output.png'

THRESHOLD = 0.5

COCO_SKELETON = [
    [15, 13], [13, 11], [16, 14], [14, 12], [11, 12], [5, 11],
    [6, 12], [5, 6], [5, 7], [6, 8], [7, 9], [8, 10], [1, 2],
    [0, 1], [0, 2], [1, 3], [2, 4], [3, 5], [4, 6]]

COLOR_LIMBS = [
    [255, 0, 0], [255, 82, 0], [255, 165, 0], [251, 245, 0], [179, 255, 0], [90, 255, 0],
    [7, 255, 0], [0, 255, 74], [0, 255, 157], [0, 255, 245], [0, 181, 255], [0, 98, 255],
    [0, 15, 255], [66, 0, 255], [155, 0, 255], [238, 0, 255], [255, 0, 189], [255, 0, 106], [255, 0, 23]]

COLOR_JOINT = [
    [255, 0, 0], [255, 94, 0], [255, 189, 0], [226, 255, 0], [131, 255, 0], [37, 255, 0],
    [0, 255, 57], [0, 255, 151], [0, 255, 245], [0, 169, 255], [0, 75, 255], [19, 0, 255],
    [113, 0, 255], [208, 0, 255], [255, 0, 207], [255, 0, 112], [255, 0, 23]]

# ======================
# Arguemnt Parser Config
# ======================

parser = get_base_parser(
    'E2Pose', IMAGE_PATH, SAVE_IMAGE_PATH
)
parser.add_argument(
    '-th', '--threshold',
    default=THRESHOLD, type=float,
    help='object confidence threshold'
)
parser.add_argument(
    '-m', '--model_type', default='resnet101',
    choices=('resnet50', 'resnet101', 'resnet152', 'mobilenet_320', 'mobilenet_448'),
    help='model type'
)
parser.add_argument(
    '--onnx',
    action='store_true',
    help='execute onnxruntime version.'
)
parser.add_argument(
    '-w', '--write_json',
    action='store_true',
    help='save result to json'
)
args = update_parser(parser)

model_type = args.model_type


# ======================
# Secondaty Functions
# ======================


def draw_keypoints(img, anns):
    th = args.threshold
    thickness = max(1, int(np.sqrt(np.sum(np.square(img.shape[:2]))) * 0.003))

    # Draw limbs
    for kpts in anns:
        for ii, (idx1, idx2) in enumerate(COCO_SKELETON):
            color = COLOR_LIMBS[ii]
            j1, j2 = kpts[idx1], kpts[idx2]
            if j1[-1] > th and j2[-1] > th:
                cv2.line(img, tuple(j1[:2].astype(np.int32).tolist()), tuple(j2[:2].astype(np.int32).tolist()),
                         tuple(color), thickness=thickness, lineType=cv2.LINE_AA)
    # Draw Joint
    for kpts in anns:
        for ii, j1 in enumerate(kpts):
            color = COLOR_JOINT[ii]
            if j1[-1] > th:
                cv2.circle(img, tuple(j1[:2].astype(np.int32).tolist()), radius=thickness, color=tuple(color),
                           thickness=-1, lineType=cv2.LINE_AA)

    return img


def save_json(pred, json_path):
    output = []
    for kpts in pred:
        output.append([{'pos': [k[0], k[1]], 'prob': k[2]} for k in kpts])

    with open(json_path, 'w') as f:
        json.dump(numpy_type_to_builtin_type(output), f, indent=2)


# ======================
# Main functions
# ======================

def preprocess(img, image_shape):
    h, w = image_shape
    im_h, im_w, _ = img.shape

    # adaptive_resize
    scale = h / min(im_h, im_w)
    ow, oh = int(im_w * scale), int(im_h * scale)
    if ow != im_w or oh != im_h:
        img = cv2.resize(img, (w, h), interpolation=cv2.INTER_LINEAR)

    img = np.expand_dims(img, axis=0)
    img = img.astype(np.float32)

    return img


def post_processing(pv, kpt, im_hw):
    th = args.threshold

    pv = pv[0].reshape(-1)
    kpt = kpt[0][pv >= th]
    kpt[:, :, -1] *= im_hw[0]
    kpt[:, :, -2] *= im_hw[1]
    kpt[:, :, -3] *= 2

    ret = []
    for human in kpt:
        mask = np.stack([(human[:, 0] >= th).astype(np.float32)], axis=-1)
        human *= mask
        keypoints = np.stack([human[:, _ii] for _ii in [1, 2, 0]], axis=-1)
        ret.append(keypoints)

    return ret


def predict(net, img):
    h, w, _ = img.shape
    img = img[:, :, ::-1]  # BGR -> RGB

    dic_shape = {
        'resnet50': (320, 320),
        'resnet101': (512, 512),
        'resnet152': (448, 448),
        'mobilenet_320': (320, 320),
        'mobilenet_448': (448, 512),
    }
    shape = dic_shape[model_type]
    img = preprocess(img, shape)

    # feedforward
    if not args.onnx:
        output = net.predict([img])
    else:
        output = net.run(None, {'e2pose/inputimg:0': img})
    pv, kpt = output

    pred = post_processing(pv, kpt, (h, w))

    return pred


def recognize_from_image(net):
    # input image loop
    for image_path in args.input:
        logger.info(image_path)

        # prepare input data
        img = load_image(image_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)

        # inference
        logger.info('Start inference...')
        if args.benchmark:
            logger.info('BENCHMARK mode')
            total_time_estimation = 0
            for i in range(args.benchmark_count):
                start = int(round(time.time() * 1000))
                pred = predict(net, img)
                end = int(round(time.time() * 1000))
                estimation_time = (end - start)

                # Logging
                logger.info(f'\tailia processing estimation time {estimation_time} ms')
                if i != 0:
                    total_time_estimation = total_time_estimation + estimation_time

            logger.info(f'\taverage time estimation {total_time_estimation / (args.benchmark_count - 1)} ms')
        else:
            pred = predict(net, img)

        res_img = draw_keypoints(img, pred)

        # plot result
        savepath = get_savepath(args.savepath, image_path, ext='.png')
        logger.info(f'saved at : {savepath}')
        cv2.imwrite(savepath, res_img)

        if args.write_json:
            save_json(pred, (savepath.rsplit('.', 1)[0]) + '.json')

    logger.info('Script finished successfully.')


def recognize_from_video(net):
    video_file = args.video if args.video else args.input[0]
    capture = get_capture(video_file)
    assert capture.isOpened(), 'Cannot capture source'

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
        pred = predict(net, frame)

        # plot result
        res_img = draw_keypoints(frame, pred)

        # show
        cv2.imshow('frame', res_img)
        frame_shown = True

        # save results
        if writer is not None:
            res_img = res_img.astype(np.uint8)
            writer.write(res_img)

    capture.release()
    cv2.destroyAllWindows()
    if writer is not None:
        writer.release()

    logger.info('Script finished successfully.')


def main():
    dic_model = {
        'resnet50': (WEIGHT_ResNet50_320_PATH, MODEL_ResNet50_320_PATH),
        'resnet101': (WEIGHT_ResNet101_512_PATH, MODEL_ResNet101_512_PATH),
        'resnet152': (WEIGHT_ResNet152_448_PATH, MODEL_ResNet152_448_PATH),
        'mobilenet_320': (WEIGHT_MobleNetV2_320_PATH, MODEL_MobleNetV2_320_PATH),
        'mobilenet_448': (WEIGHT_MobleNetV2_448_PATH, MODEL_MobleNetV2_448_PATH),
    }
    weight_path, model_path = dic_model[model_type]

    # model files check and download
    check_and_download_models(weight_path, model_path, REMOTE_PATH)

    env_id = args.env_id

    # initialize
    if not args.onnx:
        net = ailia.Net(model_path, weight_path, env_id=env_id)
    else:
        import onnxruntime
        net = onnxruntime.InferenceSession(weight_path)

    if args.video is not None:
        recognize_from_video(net)
    else:
        recognize_from_image(net)


if __name__ == '__main__':
    main()
