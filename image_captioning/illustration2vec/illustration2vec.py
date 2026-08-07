import sys
import time
import json
from pprint import pprint

import numpy as np
import cv2

import ailia

# import original modules
sys.path.append('../../util')
from arg_utils import get_base_parser, update_parser  # noqa: E402
from arg_utils import check_file_existance  # noqa: E402
from model_utils import check_and_download_models  # noqa: E402
from image_utils import load_image  # noqa: E402
import webcamera_utils  # noqa: E402
from dtype_utils import numpy_type_to_builtin_type  # noqa: E402

# logger
from logging import getLogger   # noqa: E402
logger = getLogger(__name__)


# ======================
# Parameters
# ======================
TAG_WEIGHT_PATH = 'illust2vec_tag_ver200.onnx'
TAG_MODEL_PATH = 'illust2vec_tag_ver200.onnx.prototxt'
FE_WEIGHT_PATH = 'illust2vec_ver200.onnx'
FE_MODEL_PATH = 'illust2vec_ver200.onnx.prototxt'
REMOTE_PATH = 'https://storage.googleapis.com/ailia-models/ill2vec/'
TAG_PATH = 'tag_list.json'

IMAGE_PATH = 'input.jpg'
IMAGE_HEIGHT = 224
IMAGE_WIDTH = 224

THRESHOLD = 0.5  # TODO argument ?
SLEEP_TIME = 0


# ======================
# Arguemnt Parser Config
# ======================
parser = get_base_parser('illustration2vec model', IMAGE_PATH, None)
parser.add_argument(
    '-f', '--featurevec',
    action='store_true',
    help='Extracting feature vector instead of estimating tags'
)
args = update_parser(parser)


# ======================
# Utils
# ======================
def prepare_input_data(img, bgr=False):
    mean = np.array([164.76139251,  167.47864617,  181.13838569])
    if bgr:
        img = img.astype(np.float32)
    else:
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR).astype(np.float32)
    img -= mean
    img = img.transpose((2, 0, 1))
    return img[np.newaxis, :, :, :]


def estimate_top_tags(prob, tags, n_tag=10):
    general_prob = prob[:, :512]
    character_prob = prob[:, 512:1024]
    copyright_prob = prob[:, 1024:1536]
    rating_prob = prob[:, 1536:]
    general_arg = np.argsort(-general_prob, axis=1)[:, :n_tag]
    character_arg = np.argsort(-character_prob, axis=1)[:, :n_tag]
    copyright_arg = np.argsort(-copyright_prob, axis=1)[:, :n_tag]
    rating_arg = np.argsort(-rating_prob, axis=1)
    result = []

    for i in range(prob.shape[0]):
        result.append({
            'general': list(zip(
                tags[general_arg[i]],
                general_prob[i, general_arg[i]].tolist())
            ),
            'character': list(zip(
                tags[512 + character_arg[i]],
                character_prob[i, character_arg[i]].tolist())
            ),
            'copyright': list(zip(
                tags[1024 + copyright_arg[i]],
                copyright_prob[i, copyright_arg[i]].tolist())
            ),
            'rating': list(zip(
                tags[1536 + rating_arg[i]],
                rating_prob[i, rating_arg[i]].tolist())
            ),
        })
        return result


def apply_threshold(preds, threshold=0.25, threshold_rule='constant'):
    # [WARNING] for now, only 'constant' can be used as threshold_rule.
    def __apply_threshold(preds, f):
        result = []
        for pred in preds:
            general = [(t, p) for t, p in pred['general'] if f(t, p)]
            character = [(t, p) for t, p in pred['character'] if f(t, p)]
            copyright = [(t, p) for t, p in pred['copyright'] if f(t, p)]
            result.append({
                'general': general,
                'character': character,
                'copyright': copyright,
                'rating': pred['rating'],
            })
        return result

    if threshold_rule == 'constant':
        return __apply_threshold(preds, lambda t, p: p > threshold)
    else:
        raise TypeError('unknown threshold rule specified')


# ======================
# Main functions
# ======================
def recognize_tag_from_image():
    # net initialize
    tag_net = ailia.Net(TAG_MODEL_PATH, TAG_WEIGHT_PATH, env_id=args.env_id)

    # input image loop
    for image_path in args.input:
        # prepare input data
        logger.info(image_path)
        input_img = load_image(
            image_path,
            (IMAGE_HEIGHT, IMAGE_WIDTH),
            normalize_type='None',
        )
        input_data = prepare_input_data(input_img)
        tag_net.set_input_shape(input_data.shape)

        if check_file_existance(TAG_PATH):
            tags = np.array(json.loads(open(TAG_PATH, 'r').read()))
            assert(len(tags) == 1539)

        input_dict = {'data': input_data}

        # inference
        logger.info('Start inference...')
        if args.benchmark:
            logger.info('BENCHMARK mode')
            for i in range(5):
                start = int(round(time.time() * 1000))
                preds_ailia = tag_net.predict(input_dict)[0]
                end = int(round(time.time() * 1000))
                logger.info(f'\tailia processing time {end - start} ms')
        else:
            preds_ailia = tag_net.predict(input_dict)[0]

        prob = preds_ailia.reshape(preds_ailia.shape[0], -1)
        preds = estimate_top_tags(prob, tags, 512)  # TODO how to decide n_tag?
        pprint(numpy_type_to_builtin_type(apply_threshold(preds, THRESHOLD)))
    logger.info('Script finished successfully.')


def extract_feature_vec_from_image():
    # net initialize
    fe_net = ailia.Net(FE_MODEL_PATH, FE_WEIGHT_PATH, env_id=args.env_id)

    # input image loop
    for image_path in args.input:
        # prepare input data
        logger.info(image_path)
        input_img = load_image(
            image_path,
            (IMAGE_HEIGHT, IMAGE_WIDTH),
            normalize_type='None',
        )
        input_data = prepare_input_data(input_img)
        fe_net.set_input_shape(input_data.shape)

        input_dict = {'data': input_data}

        # inference
        logger.info('Start inference...')
        if args.benchmark:
            logger.info('BENCHMARK mode')
            for i in range(5):
                start = int(round(time.time() * 1000))
                _ = fe_net.predict(input_dict)[0]
                end = int(round(time.time() * 1000))
                logger.info(f'\tailia processing time {end - start} ms')
        else:
            _ = fe_net.predict(input_dict)[0]

        # Extracting the output of a specifc layer
        idx = fe_net.find_blob_index_by_name('encode1')
        preds_ailia = fe_net.get_blob_data(idx)
        logger.info(preds_ailia.reshape(preds_ailia.shape[0], -1))
    logger.info('Script finished successfully.')


def recognize_tag_from_video():
    # net initialize
    tag_net = ailia.Net(TAG_MODEL_PATH, TAG_WEIGHT_PATH, env_id=args.env_id)

    capture = webcamera_utils.get_capture(args.video)

    if check_file_existance(TAG_PATH):
        tags = np.array(json.loads(open(TAG_PATH, 'r').read()))
        assert len(tags) == 1539

    # create video writer if savepath is specified as video format
    if args.savepath is not None:
        writer = webcamera_utils.get_writer(
            args.savepath, IMAGE_HEIGHT, IMAGE_WIDTH
        )
    else:
        writer = None
    
    frame_shown = False
    while(True):
        ret, frame = capture.read()
        if (cv2.waitKey(1) & 0xFF == ord('q')) or not ret:
            break
        if frame_shown and cv2.getWindowProperty('frame', cv2.WND_PROP_VISIBLE) == 0:
            break

        _, frame = webcamera_utils.adjust_frame_size(
            frame, IMAGE_HEIGHT, IMAGE_WIDTH
        )

        input_data = prepare_input_data(frame, bgr=True)
        input_dict = {'data': input_data}
        tag_net.set_input_shape(input_data.shape)

        # inference
        preds_ailia = tag_net.predict(input_dict)[0]

        prob = preds_ailia.reshape(preds_ailia.shape[0], -1)
        preds = estimate_top_tags(prob, tags, 512)
        logger.info('=' * 80)
        pprint(numpy_type_to_builtin_type(apply_threshold(preds, THRESHOLD)))
        cv2.imshow('frame', frame)
        frame_shown = True
        time.sleep(SLEEP_TIME)

        # save results
        if writer is not None:
            writer.write(frame)

    capture.release()
    cv2.destroyAllWindows()
    if writer is not None:
        writer.release()
    logger.info('Script finished successfully.')


def extract_feature_vec_from_video():
    # net initialize
    fe_net = ailia.Net(FE_MODEL_PATH, FE_WEIGHT_PATH, env_id=args.env_id)

    capture = webcamera_utils.get_capture(args.video)

    # create video writer if savepath is specified as video format
    if args.savepath is not None:
        writer = webcamera_utils.get_writer(
            args.savepath, IMAGE_HEIGHT, IMAGE_WIDTH
        )
    else:
        writer = None

    frame_shown = False
    while(True):
        ret, frame = capture.read()
        if (cv2.waitKey(1) & 0xFF == ord('q')) or not ret:
            break
        if frame_shown and cv2.getWindowProperty('frame', cv2.WND_PROP_VISIBLE) == 0:
            break

        _, frame = webcamera_utils.adjust_frame_size(
            frame, IMAGE_HEIGHT, IMAGE_WIDTH
        )
        input_data = prepare_input_data(frame, bgr=True)
        input_dict = {'data': input_data}
        fe_net.set_input_shape(input_data.shape)

        # inference
        _ = fe_net.predict(input_dict)[0]
        # Extracting the output of a specifc layer
        idx = fe_net.find_blob_index_by_name('encode1')
        preds_ailia = fe_net.get_blob_data(idx)

        logger.info('=' * 80)
        logger.info(preds_ailia.reshape(preds_ailia.shape[0], -1))
        cv2.imshow('frame', frame)
        frame_shown = True
        time.sleep(SLEEP_TIME)

        # save results
        if writer is not None:
            writer.write(frame)

    capture.release()
    cv2.destroyAllWindows()
    if writer is not None:
        writer.release()
    logger.info('Script finished successfully.')


def main():
    # model files check and download
    check_and_download_models(TAG_WEIGHT_PATH, TAG_MODEL_PATH, REMOTE_PATH)
    check_and_download_models(FE_WEIGHT_PATH, FE_MODEL_PATH, REMOTE_PATH)

    if args.video is not None:
        # video mode
        if args.featurevec:
            # Feature vector extracting mode
            extract_feature_vec_from_video()
        else:
            # Tag estimating mode
            recognize_tag_from_video()
    else:
        # image mode
        if args.featurevec:
            # Feature vector extracting mode
            extract_feature_vec_from_image()
        else:
            # Tag estimating mode
            recognize_tag_from_image()


if __name__ == '__main__':
    main()
