import sys
import time

import ailia
import cv2

# import original modules
sys.path.append('../../util')
sys.path.append('../../face_detection/blazeface')
# logger
from logging import getLogger  # noqa: E402

import webcamera_utils  # noqa: E402
from blazeface_utils import compute_blazeface, crop_blazeface  # noqa: E402
from image_utils import imread  # noqa: E402
from model_utils import check_and_download_models  # noqa: E402
from arg_utils import get_base_parser, update_parser  # noqa: E402

logger = getLogger(__name__)


# ======================
# PARAMETERS
# ======================
EMOTION_WEIGHT_PATH = 'emotion_miniXception.caffemodel'
EMOTION_MODEL_PATH = 'emotion_miniXception.prototxt'
GENDER_WEIGHT_PATH = "gender_miniXception.caffemodel"
GENDER_MODEL_PATH = "gender_miniXception.prototxt"
REMOTE_PATH = \
    'https://storage.googleapis.com/ailia-models/face_classification/'

IMAGE_PATH = 'lenna.png'
EMOTION_MAX_CLASS_COUNT = 3
GENDER_MAX_CLASS_COUNT = 2
SLEEP_TIME = 0

EMOTION_CATEGORY = [
    "angry",
    "disgust",
    "fear",
    "happy",
    "sad",
    "surprise",
    "neutral"
]
GENDER_CATEGORY = ["female", "male"]

FACE_WEIGHT_PATH = 'blazeface.onnx'
FACE_MODEL_PATH = 'blazeface.onnx.prototxt'
FACE_REMOTE_PATH = "https://storage.googleapis.com/ailia-models/blazeface/"
FACE_MARGIN = 1.0


# ======================
# Arguemnt Parser Config
# ======================
parser = get_base_parser(
    'Face Classificaiton Model (emotion & gender)',
    IMAGE_PATH,
    None,
)
args = update_parser(parser)


# ======================
# Main functions
# ======================
def recognize_from_image():
    # net initialize
    emotion_classifier = ailia.Classifier(
        EMOTION_MODEL_PATH,
        EMOTION_WEIGHT_PATH,
        env_id=args.env_id,
        format=ailia.NETWORK_IMAGE_FORMAT_GRAY,
        range=ailia.NETWORK_IMAGE_RANGE_S_FP32,
        channel=ailia.NETWORK_IMAGE_CHANNEL_FIRST,
        memory_mode=args.memory_mode,
    )
    gender_classifier = ailia.Classifier(
        GENDER_MODEL_PATH,
        GENDER_WEIGHT_PATH,
        env_id=args.env_id,
        format=ailia.NETWORK_IMAGE_FORMAT_GRAY,
        range=ailia.NETWORK_IMAGE_RANGE_S_FP32,
        channel=ailia.NETWORK_IMAGE_CHANNEL_FIRST,
        memory_mode=args.memory_mode,
    )

    # input image loop
    for image_path in args.input:
        # prepare input data
        logger.info(image_path)
        # load input image and convert to BGRA
        img = imread(image_path, cv2.IMREAD_UNCHANGED)
        if img.shape[2] == 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2BGRA)
        elif img.shape[2] == 1:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGRA)

        # inference emotion
        logger.info('Start inference...')
        if args.benchmark:
            logger.info('BENCHMARK mode')
            for i in range(5):
                start = int(round(time.time() * 1000))
                emotion_classifier.compute(img, EMOTION_MAX_CLASS_COUNT)
                end = int(round(time.time() * 1000))
                logger.info(
                    f'\t[EMOTION MODEL] ailia processing time {end - start} ms'
                )
        else:
            emotion_classifier.compute(img, EMOTION_MAX_CLASS_COUNT)
        count = emotion_classifier.get_class_count()
        logger.info(f'emotion_class_count={count}')

        # logger.info result
        for idx in range(count):
            logger.info(f'+ idx={idx}')
            info = emotion_classifier.get_class(idx)
            logger.info(f'  category={info.category} '
                        f'[ {EMOTION_CATEGORY[info.category]} ]')
            logger.info(f'  prob={info.prob}')
        logger.info('')

        # inference gender
        if args.benchmark:
            logger.info('BENCHMARK mode')
            for i in range(5):
                start = int(round(time.time() * 1000))
                gender_classifier.compute(img, GENDER_MAX_CLASS_COUNT)
                end = int(round(time.time() * 1000))
                logger.info(
                    f'\t[EMOTION MODEL] ailia processing time {end - start} ms'
                )
        else:
            gender_classifier.compute(img, GENDER_MAX_CLASS_COUNT)
        count = gender_classifier.get_class_count()
        logger.info(f'gender_class_count={count}')

        # logger.info reuslt
        for idx in range(count):
            logger.info(f'+ idx={idx}')
            info = gender_classifier.get_class(idx)
            logger.info(f'  category={info.category} '
                        f'[ {GENDER_CATEGORY[info.category]} ]')
            logger.info(f'  prob={info.prob}')
    logger.info('Script finished successfully.')


def recognize_from_video():
    # net initialize
    emotion_classifier = ailia.Classifier(
        EMOTION_MODEL_PATH,
        EMOTION_WEIGHT_PATH,
        env_id=args.env_id,
        format=ailia.NETWORK_IMAGE_FORMAT_GRAY,
        range=ailia.NETWORK_IMAGE_RANGE_S_FP32,
        channel=ailia.NETWORK_IMAGE_CHANNEL_FIRST,
        memory_mode=args.memory_mode,
    )
    gender_classifier = ailia.Classifier(
        GENDER_MODEL_PATH,
        GENDER_WEIGHT_PATH,
        env_id=args.env_id,
        format=ailia.NETWORK_IMAGE_FORMAT_GRAY,
        range=ailia.NETWORK_IMAGE_RANGE_S_FP32,
        channel=ailia.NETWORK_IMAGE_CHANNEL_FIRST,
        memory_mode=args.memory_mode,
    )
    detector = ailia.Net(FACE_MODEL_PATH, FACE_WEIGHT_PATH, env_id=args.env_id, memory_mode=args.memory_mode)

    capture = webcamera_utils.get_capture(args.video)

    # create video writer if savepath is specified as video format
    if args.savepath is not None:
        logger.warning('[WARNING] currently video results output feature '
                       'is not supported in this model!')
        # TODO: shape should be debugged!
        f_h = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        f_w = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        writer = webcamera_utils.get_writer(args.savepath, f_h, f_w)
    else:
        writer = None

    frame_shown = False
    while(True):
        ret, frame = capture.read()

        if (cv2.waitKey(1) & 0xFF == ord('q')) or not ret:
            break
        if frame_shown and cv2.getWindowProperty('frame', cv2.WND_PROP_VISIBLE) == 0:
            break

        # detect face
        # WIP: FIXME: AiliaInvalidArgumentException error
        detections = compute_blazeface(
            detector,
            frame,
            anchor_path='../../face_detection/blazeface/anchors.npy',
        )

        for obj in detections:
            # get detected face
            crop_img, top_left, bottom_right = crop_blazeface(
                obj, FACE_MARGIN, frame
            )
            if crop_img.shape[0] <= 0 or crop_img.shape[1] <= 0:
                continue
            crop_img = cv2.cvtColor(crop_img, cv2.COLOR_BGR2BGRA)

            # emotion inference
            emotion_classifier.compute(crop_img, EMOTION_MAX_CLASS_COUNT)
            count = emotion_classifier.get_class_count()
            logger.info('=' * 80)
            logger.info(f'emotion_class_count={count}')

            # logger.info result
            emotion_text = ""
            for idx in range(count):
                logger.info(f'+ idx={idx}')
                info = emotion_classifier.get_class(idx)
                logger.info(
                    f'  category={info.category} ' +
                    f'[ {EMOTION_CATEGORY[info.category]} ]'
                )
                logger.info(f'  prob={info.prob}')
                if idx == 0:
                    emotion_text = (f'[ {EMOTION_CATEGORY[info.category]} ] '
                                    f'prob={info.prob:.3f}')
            logger.info('')

            # gender inference
            gender_text = ""
            gender_classifier.compute(crop_img, GENDER_MAX_CLASS_COUNT)
            count = gender_classifier.get_class_count()
            # logger.info reuslt
            for idx in range(count):
                logger.info(f'+ idx={idx}')
                info = gender_classifier.get_class(idx)
                logger.info(
                    f'  category={info.category} ' +
                    f'[ {GENDER_CATEGORY[info.category]} ]'
                )
                logger.info(f'  prob={info.prob}')
                if idx == 0:
                    gender_text = (f'[ {GENDER_CATEGORY[info.category]} ] '
                                   f'prob={info.prob:.3f}')
            logger.info('')

            # display label
            LABEL_WIDTH = 400
            LABEL_HEIGHT = 20
            color = (255, 255, 255)
            cv2.rectangle(frame, top_left, bottom_right, color, thickness=2)
            cv2.rectangle(
                frame,
                top_left,
                (top_left[0]+LABEL_WIDTH, top_left[1]+LABEL_HEIGHT),
                color,
                thickness=-1,
            )

            text_position = (top_left[0], top_left[1]+LABEL_HEIGHT//2)
            color = (0, 0, 0)
            fontScale = 0.5
            cv2.putText(
                frame,
                emotion_text + " " + gender_text,
                text_position,
                cv2.FONT_HERSHEY_SIMPLEX,
                fontScale,
                color,
                1,
            )

            # show result
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
    check_and_download_models(
        EMOTION_WEIGHT_PATH, EMOTION_MODEL_PATH, REMOTE_PATH
    )
    check_and_download_models(
        GENDER_WEIGHT_PATH, GENDER_MODEL_PATH, REMOTE_PATH
    )
    if args.video:
        check_and_download_models(
            FACE_WEIGHT_PATH, FACE_MODEL_PATH, FACE_REMOTE_PATH
        )

    if args.video is not None:
        # video mode
        recognize_from_video()
    else:
        # image mode
        recognize_from_image()


if __name__ == '__main__':
    main()
