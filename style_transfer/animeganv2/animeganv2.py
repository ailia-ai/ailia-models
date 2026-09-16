import sys
import time

import numpy as np
import cv2

import ailia

# import original modules
sys.path.append('../../util')
from arg_utils import get_base_parser, update_parser, get_savepath  # noqa: E402
from model_utils import check_and_download_models  # noqa: E402
from detector_utils import load_image  # noqa: E402C
from webcamera_utils import get_capture, get_writer  # noqa: E402
# logger
from logging import getLogger  # noqa: E402

logger = getLogger(__name__)

# ======================
# Parameters
# ======================

WEIGHT_PAPRIKA_PATH = 'generator_Paprika.onnx'
MODEL_PAPRIKA_PATH = 'generator_Paprika.onnx.prototxt'
WEIGHT_HAYAO_PATH = 'generator_Hayao.onnx'
MODEL_HAYAO_PATH = 'generator_Hayao.onnx.prototxt'
WEIGHT_SHINKAI_PATH = 'generator_Shinkai.onnx'
MODEL_SHINKAI_PATH = 'generator_Shinkai.onnx.prototxt'
WEIGHT_CELEBA_PATH = 'celeba_distill.onnx'
MODEL_CELEBA_PATH = 'celeba_distill.onnx.prototxt'
WEIGHT_FACE_PAINT_PATH = 'face_paint_512_v2.onnx'
MODEL_FACE_PAINT_PATH = 'face_paint_512_v2.onnx.prototxt'
REMOTE_PATH = \
    'https://storage.googleapis.com/ailia-models/animeganv2/'

IMAGE_PATH = 'sample.jpg'
SAVE_IMAGE_PATH = 'output.png'

IMAGE_HEIGHT = 512
IMAGE_WIDTH = 512

# ======================
# Arguemnt Parser Config
# ======================

parser = get_base_parser(
    'Anime GAN v2', IMAGE_PATH, SAVE_IMAGE_PATH,
)
parser.add_argument(
    '-m', '--model_name', default='paprika',
    choices=('paprika', 'hayao', 'shinkai', 'celeba', 'face_paint'),
    help='model name'
)
parser.add_argument(
    '-k', '--keep',
    action='store_true',
    help='keep aspect when resizing.'
)
parser.add_argument(
    '--x32',
    action="store_true",
    help='resize image to multiple of 32s.'
)
parser.add_argument(
    '--onnx',
    action='store_true',
    help='execute onnxruntime version.'
)
args = update_parser(parser)


# ======================
# Main functions
# ======================

def preprocess(img):
    x32 = args.x32
    keep = args.keep

    h, w = (IMAGE_HEIGHT, IMAGE_WIDTH)
    im_h, im_w, _ = img.shape

    if x32:
        # resize image to multiple of 32s
        def to_32s(x):
            return 256 if x < 256 else x - x % 32

        oh, ow = to_32s(im_h), to_32s(im_w)
        ph = pw = 0
        img = cv2.resize(img, (ow, oh))
    elif keep:
        # adaptive_resize
        r = min(h / im_h, w / im_w)
        oh, ow = int(im_h * r), int(im_w * r)

        resized = cv2.resize(img, (ow, oh))

        img = np.zeros((h, w, 3), dtype=np.uint8)
        ph, pw = (h - oh) // 2, (w - ow) // 2
        img[ph: ph + oh, pw: pw + ow] = resized
    else:
        oh, ow = h, w
        ph = pw = 0
        img = cv2.resize(img, (ow, oh))

    img = img / 127.5 - 1
    img = img.transpose(2, 0, 1)  # HWC -> CHW
    img = np.expand_dims(img, axis=0)
    img = img.astype(np.float32)

    return img, (ph, pw), (oh, ow)


def post_processing(img, im_hw, pad_hw, resized_hw):
    img = img.transpose(1, 2, 0)

    pad_x = pad_hw[1]
    pad_y = pad_hw[0]
    resized_x = resized_hw[1]
    resized_y = resized_hw[0]
    img = img[pad_y:pad_y + resized_y, pad_x:pad_x + resized_x, ...]
    img = cv2.resize(img, (im_hw[1], im_hw[0]))

    img = np.clip(img, -1, 1)
    img = img * 127.5 + 127.5

    return img


def predict(net, img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    im_h, im_w = img.shape[:2]

    img, pad_hw, resized_hw = preprocess(img)

    # feedforward
    if not args.onnx:
        output = net.predict([img])
    else:
        output = net.run(None, {'input_image': img})

    output = output[0]

    img = post_processing(output[0], (im_h, im_w), pad_hw, resized_hw)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    img = img.astype(np.uint8)

    return img


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
                out_img = predict(net, img)
                end = int(round(time.time() * 1000))
                estimation_time = (end - start)

                # Loggin
                logger.info(f'\tailia processing estimation time {estimation_time} ms')
                if i != 0:
                    total_time_estimation = total_time_estimation + estimation_time

            logger.info(f'\taverage time estimation {total_time_estimation / (args.benchmark_count - 1)} ms')
        else:
            out_img = predict(net, img)

        # plot result
        savepath = get_savepath(args.savepath, image_path, ext='.png')
        logger.info(f'saved at : {savepath}')
        cv2.imwrite(savepath, out_img)

    logger.info('Script finished successfully.')


def recognize_from_video(net):
    video_file = args.video if args.video else args.input[0]
    capture = get_capture(video_file)
    assert capture.isOpened(), 'Cannot capture source'

    # create video writer if savepath is specified as video format
    f_h = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    f_w = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    if args.savepath != SAVE_IMAGE_PATH:
        logger.warning(
            'currently, video results cannot be output correctly...'
        )
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
        out_img = predict(net, frame)

        # show
        cv2.imshow('frame', out_img)
        frame_shown = True

        # save results
        if writer is not None:
            writer.write(out_img)

    capture.release()
    cv2.destroyAllWindows()
    if writer is not None:
        writer.release()

    logger.info('Script finished successfully.')


def main():
    dic_model = {
        'paprika': (WEIGHT_PAPRIKA_PATH, MODEL_PAPRIKA_PATH),
        'hayao': (WEIGHT_HAYAO_PATH, MODEL_HAYAO_PATH),
        'shinkai': (WEIGHT_SHINKAI_PATH, MODEL_SHINKAI_PATH),
        'celeba': (WEIGHT_CELEBA_PATH, MODEL_CELEBA_PATH),
        'face_paint': (WEIGHT_FACE_PAINT_PATH, MODEL_FACE_PAINT_PATH),
    }
    weight_path, model_path = dic_model[args.model_name]
    check_and_download_models(weight_path, model_path, REMOTE_PATH)

    env_id = args.env_id

    # initialize
    if not args.onnx:
        net = ailia.Net(model_path, weight_path, env_id=env_id, memory_mode=args.memory_mode)
    else:
        import onnxruntime
        net = onnxruntime.InferenceSession(weight_path)

    if args.video is not None:
        recognize_from_video(net)
    else:
        recognize_from_image(net)


if __name__ == '__main__':
    main()
