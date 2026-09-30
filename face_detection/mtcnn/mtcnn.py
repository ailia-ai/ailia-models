import sys
import time

import cv2
import json

from mtcnn_util import MTCNN

# import original modules
sys.path.append('../../util')
from arg_utils import get_base_parser, update_parser, get_savepath  # noqa: E402
from model_utils import check_and_download_models  # noqa: E402
import webcamera_utils  # noqa: E402
from detector_utils import plot_results, load_image  # noqa: E402

# logger.info
from logging import getLogger   # noqa: E402
logger = getLogger(__name__)



# ======================
# Parameters
# ======================
WEIGHT_PATH1 = 'pnet.onnx'
MODEL_PATH1 = 'pnet.onnx.prototxt'

WEIGHT_PATH2 = 'rnet.onnx'
MODEL_PATH2 = 'rnet.onnx.prototxt'

WEIGHT_PATH3 = 'onet.onnx'
MODEL_PATH3 = 'onet.onnx.prototxt'

REMOTE_PATH = 'https://storage.googleapis.com/ailia-models/mtcnn/'

IMAGE_PATH = 'input.jpg'
SAVE_IMAGE_PATH = 'output.jpg'


# ======================
# Arguemnt Parser Config
# ======================
parser = get_base_parser(
        'Face Detection using mtcnn', IMAGE_PATH, SAVE_IMAGE_PATH
)
parser.add_argument(
    '-th', '--threshold',
    nargs="*", type=float,
    default=[0.6,0.7,0.7],
    help='object confidence threshold'
)
parser.add_argument(
    '-w', '--write_json',
    action='store_true',
    help='Flag to output results to json file.'
)
args = update_parser(parser)


# ======================
# Main functions
# ======================
def recognize_from_image():

    detector = MTCNN(args, steps_threshold=args.threshold)

    # input image loop
    for image_path in args.input:
        # prepare input data
        logger.info(image_path)
        img = load_image(image_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        logger.debug(f'input image shape: {img.shape}')

        # inference
        logger.info('Start inference...')
        if args.benchmark:
            logger.info('BENCHMARK mode')
            for i in range(5):
                start = int(round(time.time() * 1000))
                result = detector.detect_faces(img)
                end = int(round(time.time() * 1000))
                logger.info(f'\tailia processing time {end - start} ms')
        else:
            result = detector.detect_faces(img)

        for i in range(len(result)):
            bounding_box = result[i]['box']
            keypoints = result[i]['keypoints']

            cv2.rectangle(img,
                          (bounding_box[0], bounding_box[1]),
                          (bounding_box[0]+bounding_box[2], bounding_box[1] + bounding_box[3]),
                          (0,155,255),
                          2)

            cv2.circle(img,(keypoints['left_eye']), 2, (0,155,255), 2)
            cv2.circle(img,(keypoints['right_eye']), 2, (0,155,255), 2)
            cv2.circle(img,(keypoints['nose']), 2, (0,155,255), 2)
            cv2.circle(img,(keypoints['mouth_left']), 2, (0,155,255), 2)
            cv2.circle(img,(keypoints['mouth_right']), 2, (0,155,255), 2)


    # plot result
    res_img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    savepath = get_savepath(args.savepath, image_path)
    logger.info(f'saved at : {savepath}')
    cv2.imwrite(savepath, res_img)

    if args.write_json:
        json_file = '%s.json' % savepath.rsplit('.', 1)[0]
        with open(json_file, 'w') as f:
            json.dump(result, f, indent=2)

    logger.info('Script finished successfully.')


def recognize_from_video():

    capture = webcamera_utils.get_capture(args.video)

    detector = MTCNN(args, steps_threshold=[0.6,0.7,0.7])

    # create video writer if savepath is specified as video format
    if args.savepath != SAVE_IMAGE_PATH:
        f_h = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        f_w = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        writer = webcamera_utils.get_writer(
                args.savepath, f_h, f_w
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

        img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = detector.detect_faces(img)

        for i in range(len(result)):
            bounding_box = result[i]['box']
            keypoints = result[i]['keypoints']

            cv2.rectangle(img,
                          (bounding_box[0], bounding_box[1]),
                          (bounding_box[0]+bounding_box[2], bounding_box[1] + bounding_box[3]),
                          (0,155,255),
                          2)

            cv2.circle(img,(keypoints['left_eye']), 2, (0,155,255), 2)
            cv2.circle(img,(keypoints['right_eye']), 2, (0,155,255), 2)
            cv2.circle(img,(keypoints['nose']), 2, (0,155,255), 2)
            cv2.circle(img,(keypoints['mouth_left']), 2, (0,155,255), 2)
            cv2.circle(img,(keypoints['mouth_right']), 2, (0,155,255), 2)

        res_img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
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
    check_and_download_models(WEIGHT_PATH1, MODEL_PATH1, REMOTE_PATH)
    check_and_download_models(WEIGHT_PATH2, MODEL_PATH2, REMOTE_PATH)
    check_and_download_models(WEIGHT_PATH3, MODEL_PATH3, REMOTE_PATH)

    if args.video is not None:
        # video mode
        recognize_from_video()
    else:
        # image mode
        recognize_from_image()


if __name__ == '__main__':
    main()
