import sys
import time

import numpy as np
import cv2
from PIL import Image
from scipy import ndimage

import ailia

# import original modules
sys.path.append('../../util')
from arg_utils import get_base_parser, update_parser, get_savepath  # noqa: E402
from model_utils import check_and_download_models  # noqa: E402
from detector_utils import load_image  # noqa: E402C
import webcamera_utils
# logger
from logging import getLogger  # noqa: E402

sys.path.append('../../face_detection/blazeface')
from blazeface_utils import compute_blazeface, crop_blazeface  # noqa: E402

logger = getLogger(__name__)

# ======================
# Parameters
# ======================

WEIGHT_MAJOR_PATH = 'VGG13_majority.onnx'
MODEL_MAJOR_PATH = 'VGG13_majority.onnx.prototxt'
WEIGHT_PROB_PATH = 'VGG13_probability.onnx'
MODEL_PROB_PATH = 'VGG13_probability.onnx.prototxt'
WEIGHT_CRSE_PATH = 'VGG13_crossentropy.onnx'
MODEL_CRSE_PATH = 'VGG13_crossentropy.onnx.prototxt'
WEIGHT_MLTT_PATH = 'VGG13_multi_target.onnx'
MODEL_MLTT_PATH = 'VGG13_multi_target.onnx.prototxt'
REMOTE_PATH = 'https://storage.googleapis.com/ailia-models/ferplus/'

FACE_WEIGHT_PATH = 'blazefaceback.onnx'
FACE_MODEL_PATH = 'blazefaceback.onnx.prototxt'
FACE_REMOTE_PATH = "https://storage.googleapis.com/ailia-models/blazeface/"
FACE_MIN_SCORE_THRESH = 0.5

IMAGE_PATH = 'img/fer0032227.png'
SAVE_IMAGE_PATH = 'output.png'

IMAGE_HEIGHT = 64
IMAGE_WIDTH = 64

emotion_table = [
    'neutral',
    'happiness',
    'surprise',
    'sadness',
    'anger',
    'disgust',
    'fear',
    'contempt',
]

# ======================
# Arguemnt Parser Config
# ======================

parser = get_base_parser(
    'FER+', IMAGE_PATH, SAVE_IMAGE_PATH
)
parser.add_argument(
    '-d', '--detection',
    action='store_true',
    help='Use object detection.'
)
parser.add_argument(
    '-m', '--model_name', default='majority',
    choices=('majority', 'probability', 'crossentropy', 'multi_target'),
    help='model name'
)
args = update_parser(parser)


# ======================
# Secondaty Functions
# ======================

def compute_norm_mat(base_width, base_height):
    # normalization matrix used in image pre-processing
    x = np.arange(base_width)
    y = np.arange(base_height)
    X, Y = np.meshgrid(x, y)
    X = X.flatten()
    Y = Y.flatten()
    A = np.array([X * 0 + 1, X, Y]).T
    A_pinv = np.linalg.pinv(A)

    return A, A_pinv


A, A_pinv = compute_norm_mat(IMAGE_WIDTH, IMAGE_HEIGHT)


# ======================
# Main functions
# ======================

def crop_img(
        img, roi,
        crop_width, crop_height,
        shift_x, shift_y, scale_x, scale_y,
        angle, skew_x, skew_y):
    # current face center
    ctr_in = np.array(((roi[0] + roi[2]) / 2, (roi[1] + roi[3]) / 2))
    ctr_out = np.array((crop_height / 2.0 + shift_y, crop_width / 2.0 + shift_x))
    out_shape = (crop_height, crop_width)
    s_y = scale_y * ((roi[2] - roi[0]) - 1) * 1.0 / (crop_height - 1)
    s_x = scale_x * ((roi[3] - roi[1]) - 1) * 1.0 / (crop_width - 1)

    # rotation and scale
    ang = angle * np.pi / 180.0
    transform = np.array([[np.cos(ang), -np.sin(ang)], [np.sin(ang), np.cos(ang)]])
    transform = transform.dot(np.array([[1.0, skew_y], [0.0, 1.0]]))
    transform = transform.dot(np.array([[1.0, 0.0], [skew_x, 1.0]]))
    transform = transform.dot(np.diag([s_y, s_x]))
    offset = ctr_in - ctr_out.dot(transform)

    # each point p in the output image is transformed to pT+s, where T is the matrix and s is the offset
    T_im = ndimage.interpolation.affine_transform(
        input=img,
        matrix=np.transpose(transform),
        offset=offset,
        output_shape=out_shape,
        order=1,  # bilinear interpolation
        mode='reflect',
        prefilter=False)

    return T_im


def preprocess(img):
    h, w = img.shape

    img = Image.fromarray(img)
    roi = (0, 0, h, w)
    img = crop_img(img, roi, IMAGE_WIDTH, IMAGE_HEIGHT, 0, 0, 1, 1, 0, 0, 0)

    # compute image histogram
    img_flat = img.flatten()
    img_hist = np.bincount(img_flat, minlength=256)

    # cumulative distribution function
    cdf = img_hist.cumsum()
    cdf = cdf * (2.0 / cdf[-1]) - 1.0  # normalize

    # histogram equalization
    img_eq = cdf[img_flat]

    diff = img_eq - np.dot(A, np.dot(A_pinv, img_eq))

    # after plane fitting, the mean of diff is already 0
    std = np.sqrt(np.dot(diff, diff) / diff.size)
    if std > 1e-6:
        diff = diff / std

    data = diff.reshape((1, IMAGE_HEIGHT, IMAGE_WIDTH))
    data = np.expand_dims(data, axis=0)

    return data


def predict(net, img):
    img = preprocess(img)

    # feedforward
    output = net.predict([img])
    emotion = output[0]

    return emotion


def recognize_from_image(net, detector):
    # input image loop
    for image_path in args.input:
        logger.info(image_path)

        # prepare input data
        img = load_image(image_path)

        if args.detection:
            img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
            recognize_from_frame(net, detector, img)
            savepath = get_savepath(args.savepath, image_path)
            logger.info(f'saved at : {savepath}')
            cv2.imwrite(savepath, img)
            continue

        img = cv2.cvtColor(img, cv2.COLOR_BGRA2GRAY)

        # inference
        logger.info('Start inference...')
        if args.benchmark:
            logger.info('BENCHMARK mode')
            total_time_estimation = 0
            for i in range(args.benchmark_count):
                start = int(round(time.time() * 1000))
                emotion = predict(net, img)
                end = int(round(time.time() * 1000))
                estimation_time = (end - start)

                # Loggin
                logger.info(f'\tailia processing estimation time {estimation_time} ms')
                if i != 0:
                    total_time_estimation = total_time_estimation + estimation_time

            logger.info(f'\taverage time estimation {total_time_estimation / (args.benchmark_count - 1)} ms')
        else:
            emotion = predict(net, img)

        idx = np.argmax(emotion)
        logger.info(" emotion: %s" % emotion_table[idx])

    logger.info('Script finished successfully.')


def recognize_from_frame(net, detector, frame):
    # detect face
    detections = compute_blazeface(
        detector,
        frame,
        anchor_path='../../face_detection/blazeface/anchorsback.npy',
        back=True,
        min_score_thresh=FACE_MIN_SCORE_THRESH
    )

    # adjust face rectangle
    new_detections = []
    for detection in detections:
        margin = 1.5
        r = ailia.DetectorObject(
            category=detection.category,
            prob=detection.prob,
            x=detection.x - detection.w * (margin - 1.0) / 2,
            y=detection.y - detection.h * (margin - 1.0) / 2 - detection.h * margin / 8,
            w=detection.w * margin,
            h=detection.h * margin,
        )
        new_detections.append(r)
    detections = new_detections

    # estimate emotion
    for obj in detections:
        # get detected face
        margin = 1.0
        crop_img, top_left, bottom_right = crop_blazeface(
            obj, margin, frame
        )
        if crop_img.shape[0] <= 0 or crop_img.shape[1] <= 0:
            continue

        crop_img = cv2.cvtColor(crop_img, cv2.COLOR_BGR2GRAY)
        emotion = predict(net, crop_img)
        idx = np.argmax(emotion)
        emotion = emotion_table[idx]

        # display label
        LABEL_WIDTH = bottom_right[1] - top_left[1]
        LABEL_HEIGHT = 20
        color = (255, 128, 128)
        cv2.rectangle(frame, top_left, bottom_right, color, thickness=2)
        cv2.rectangle(
            frame,
            top_left,
            (top_left[0] + LABEL_WIDTH, top_left[1] + LABEL_HEIGHT),
            color,
            thickness=-1,
        )

        text_position = (top_left[0], top_left[1] + LABEL_HEIGHT // 2)
        color = (0, 0, 0)
        fontScale = 0.5
        cv2.putText(
            frame,
            emotion,
            text_position,
            cv2.FONT_HERSHEY_SIMPLEX,
            fontScale,
            color,
            1,
        )


def recognize_from_video(net, detector):
    capture = webcamera_utils.get_capture(args.video)

    # create video writer if savepath is specified as video format
    if args.savepath is not None:
        f_h = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        f_w = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        writer = webcamera_utils.get_writer(args.savepath, f_h, f_w)
    else:
        writer = None

    frame_shown = False
    while (True):
        ret, frame = capture.read()
        if (cv2.waitKey(1) & 0xFF == ord('q')) or not ret:
            break
        if frame_shown and cv2.getWindowProperty('frame', cv2.WND_PROP_VISIBLE) == 0:
            break

        recognize_from_frame(net, detector, frame)

        # show result
        cv2.imshow('frame', frame)
        frame_shown = True

        # save results
        if writer is not None:
            writer.write(frame)

    capture.release()
    cv2.destroyAllWindows()
    if writer is not None:
        writer.release()

    logger.info('Script finished successfully.')


def main():
    logger.info('=== FERPlus model ===')
    dic_model = {
        'majority': (WEIGHT_MAJOR_PATH, MODEL_MAJOR_PATH),
        'probability': (WEIGHT_PROB_PATH, MODEL_PROB_PATH),
        'crossentropy': (WEIGHT_CRSE_PATH, MODEL_CRSE_PATH),
        'multi_target': (WEIGHT_MLTT_PATH, MODEL_MLTT_PATH),
    }
    weight_path, model_path = dic_model[args.model_name]
    check_and_download_models(weight_path, model_path, REMOTE_PATH)

    if args.video or args.detection:
        logger.info('=== face detection model ===')
        check_and_download_models(
            FACE_WEIGHT_PATH, FACE_MODEL_PATH, FACE_REMOTE_PATH
        )

    env_id = args.env_id

    # initialize
    net = ailia.Net(model_path, weight_path, env_id=env_id, memory_mode=args.memory_mode)
    detector = None
    if args.video or args.detection:
        detector = ailia.Net(FACE_MODEL_PATH, FACE_WEIGHT_PATH, env_id=env_id, memory_mode=args.memory_mode)

    if args.video is not None:
        recognize_from_video(net, detector)
    else:
        recognize_from_image(net, detector)


if __name__ == '__main__':
    main()
