import sys
import time

import numpy as np
import cv2

import ailia

# import original modules
sys.path.append('../../util')
from arg_utils import get_base_parser, update_parser  # noqa: E402
import webcamera_utils  # noqa: E402
from image_utils import load_image  # noqa: E402
from model_utils import check_and_download_models  # noqa: E402
from detector_utils import hsv_to_rgb  # noqa: E402
from nms_utils import nms_between_categories  # noqa: E402

# logger
from logging import getLogger   # noqa: E402
logger = getLogger(__name__)


# ======================
# PARAMETERS
# ======================

MODEL_LISTS = [
    'arcface',
    'arcface_mixed_90_82',
    'arcface_mixed_90_99',
    'arcface_mixed_eq_90_89'
]

REMOTE_PATH = "https://storage.googleapis.com/ailia-models/arcface/"

IMG_PATH_1 = 'correct_pair_1.jpg'
IMG_PATH_2 = 'correct_pair_2.jpg'
IMAGE_HEIGHT = 128
IMAGE_WIDTH = 128

# the threshold was calculated by the `test_performance` function in `test.py`
# of the original repository
THRESHOLD = 0.25572845
# THRESHOLD = 0.45 # for mixed model

# face detection
FACE_MODEL_LISTS = ['yolov3', 'blazeface', 'yolov3-mask']
FACE_THRESHOLD = 0.4
FACE_IOU = 0.45


# ======================
# Arguemnt Parser Config
# ======================
parser = get_base_parser(
    'Determine if the person is the same from two facial images.',
    None,
    None,
)
# overwrite default config
# NOTE: arcface has different usage for `--input` with other models
parser.add_argument(
    '-i', '--inputs', metavar='IMAGE', nargs=2, default='',
    help='Two image paths for calculating the face match.'
)
parser.add_argument(
    '-a', '--arch', metavar='ARCH',
    default='arcface', choices=MODEL_LISTS,
    help='model lists: ' + ' | '.join(MODEL_LISTS)
)
parser.add_argument(
    '-t', '--threshold', type=float, default=THRESHOLD,
    help='Similality threshold for identification'
)
parser.add_argument(
    '-f', '--face', metavar='FACE_ARCH',
    default='yolov3', choices=FACE_MODEL_LISTS,
    help='Face detection model lists: ' + ' | '.join(FACE_MODEL_LISTS)
)
parser.add_argument(
    '-ft', '--face_threshold', type=float, default=FACE_THRESHOLD,
    help='Threshold for face detection'
)
args = update_parser(parser)

WEIGHT_PATH = args.arch + '.onnx'
MODEL_PATH = args.arch + '.onnx.prototxt'


# ======================
# Face detection models
# ======================
if args.face == "yolov3":
    FACE_WEIGHT_PATH = 'yolov3-face.opt.onnx'
    FACE_MODEL_PATH = 'yolov3-face.opt.onnx.prototxt'
    FACE_REMOTE_PATH = \
        'https://storage.googleapis.com/ailia-models/yolov3-face/'
    FACE_CATEGORY = ["face"]
    FACE_ALGORITHM = ailia.DETECTOR_ALGORITHM_YOLOV3
    FACE_RANGE = ailia.NETWORK_IMAGE_RANGE_U_FP32
    FACE_MARGIN = 1.2
elif args.face == "yolov3-mask":
    FACE_WEIGHT_PATH = 'face-mask-detection-yolov3-tiny.opt.obf.onnx'
    FACE_MODEL_PATH = 'face-mask-detection-yolov3-tiny.opt.onnx.prototxt'
    FACE_REMOTE_PATH = \
        'https://storage.googleapis.com/ailia-models/face-mask-detection/'
    FACE_CATEGORY = ["no_mask", "mask"]
    FACE_ALGORITHM = ailia.DETECTOR_ALGORITHM_YOLOV3
    FACE_RANGE = ailia.NETWORK_IMAGE_RANGE_U_FP32
    FACE_MARGIN = 1.4
else:
    FACE_WEIGHT_PATH = 'blazeface.onnx'
    FACE_MODEL_PATH = 'blazeface.onnx.prototxt'
    FACE_REMOTE_PATH = "https://storage.googleapis.com/ailia-models/blazeface/"
    FACE_CATEGORY = ["face"]
    FACE_ALGORITHM = None
    FACE_RANGE = None
    FACE_MARGIN = 1.4
    sys.path.append('../../face_detection/blazeface')
    from blazeface_utils import *   # noqa: E402


# ======================
# Utils
# ======================
def preprocess_image(image, input_is_bgr=False):
    # (ref: https://github.com/ronghuaiyang/arcface-pytorch/issues/14)
    # use origin image and fliped image to infer,
    # and concat the feature as the final feature of the origin image.
    if input_is_bgr:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    if "eq_" in args.arch:
        image = cv2.equalizeHist(image)
    image = np.dstack((image, np.fliplr(image)))
    image = image.transpose((2, 0, 1))
    image = image[:, np.newaxis, :, :]
    image = image.astype(np.float32, copy=False)
    return image / 127.5 - 1.0  # normalize


def prepare_input_data(image_path):
    image = load_image(
        image_path,
        image_shape=(IMAGE_HEIGHT, IMAGE_WIDTH),
        rgb=False,
        normalize_type='None'
    )
    return preprocess_image(image)


def cosin_metric(x1, x2):
    return np.dot(x1, x2) / (np.linalg.norm(x1) * np.linalg.norm(x2))


# ======================
# Face Tracking
# ======================
FACE_TRACK_T = 15   # Face buffer size
FACE_REMOVE_T = 80  # Remove track after this frames


class FaceTrack():
    def __init__(self, id, fe, image, frame_no):
        self.id = id
        self.fe = [fe]
        self.image = [image]
        self.frame_no = [frame_no]
        self.score = 0

    def update(self, fe, image, score, frame_no):
        self.fe.append(fe)
        self.image.append(image)
        self.frame_no.append(frame_no)
        self.score = score

    def pop(self, frame_no):
        if len(self.frame_no) > FACE_TRACK_T:
            self.fe.pop(0)
            self.image.pop(0)
            self.frame_no.pop(0)

        if len(self.frame_no) >= 1:
            if frame_no - self.frame_no[0] >= FACE_REMOVE_T:
                self.fe.pop(0)
                self.image.pop(0)
                self.frame_no.pop(0)


def face_identification(tracks, net, detections, frame_no):
    BATCH_SIZE = net.get_input_shape()[0]

    for i in range(len(detections)):
        resized_frame = detections[i]["resized_frame"]

        # prepare target face and input face
        input_frame = preprocess_image(resized_frame, input_is_bgr=True)
        if BATCH_SIZE == 4:
            input_data = np.concatenate([input_frame, input_frame], axis=0)
        else:
            input_data = input_frame

        # inference
        preds_ailia = net.predict(input_data)

        # postprocessing
        if BATCH_SIZE == 4:
            fe_1 = np.concatenate([preds_ailia[0], preds_ailia[1]], axis=0)
            fe_2 = np.concatenate([preds_ailia[2], preds_ailia[3]], axis=0)
        else:
            fe_1 = np.concatenate([preds_ailia[0], preds_ailia[1]], axis=0)
            fe_2 = fe_1

        detections[i]["fe"] = fe_2

    score_matrix = np.zeros((len(detections), len(tracks)))
    for i in range(len(detections)):
        for j in range(len(tracks)):
            max_sim = 0
            avg_sim = 0
            for k in range(len(tracks[j].fe)):
                sim = cosin_metric(detections[i]["fe"], tracks[j].fe[k])
                avg_sim = avg_sim + sim
                if max_sim < sim:
                    max_sim = sim
            score_matrix[i, j] = max_sim
            if len(tracks[j].fe) >= 1:
                avg_sim = avg_sim / len(tracks[j].fe)
            score_matrix[i, j] = avg_sim

    for i in range(len(detections)):
        if len(tracks) == 0:
            continue

        # search max score between detections and tracks
        max_score = np.unravel_index(
            np.argmax(score_matrix), score_matrix.shape
        )

        det_sim = max_score[0]
        id_sim = max_score[1]
        score_sim = score_matrix[det_sim, id_sim]

        detections[det_sim]["id_sim"] = id_sim
        detections[det_sim]["score_sim"] = score_sim

        score_matrix[det_sim, :] = 0
        score_matrix[:, id_sim] = 0

    for i in range(len(tracks)):
        tracks[i].score = 0

    for i in range(len(detections)):
        # identification
        if detections[i]["score_sim"] < args.threshold:
            id_sim = len(tracks)
            fe_obj = FaceTrack(
                id_sim,
                detections[i]["fe"],
                detections[i]["resized_frame"],
                frame_no
            )
            tracks.append(fe_obj)
            detections[i]["score_sim"] = 0
            detections[i]["id_sim"] = id_sim
        else:
            tracks[detections[i]["id_sim"]].update(
                detections[i]["fe"],
                detections[i]["resized_frame"],
                detections[i]["score_sim"],
                frame_no
            )

    for i in range(len(tracks)):
        tracks[i].pop(frame_no)


def get_faces(detector, frame, w, h):
    # detect face
    org_detections = []
    if args.face == "blazeface":
        org_detections = compute_blazeface(detector, frame)
    else:
        img = cv2.cvtColor(frame, cv2.COLOR_BGR2BGRA)
        detector.compute(img, args.face_threshold, FACE_IOU)
        count = detector.get_object_count()
        for idx in range(count):
            obj = detector.get_object(idx)
            org_detections.append(obj)

    # remove over-wrapped detection
    org_detections = nms_between_categories(
        org_detections, w, h, categories=[0, 1], iou_threshold=FACE_IOU
    )

    detections = []
    for idx in range(len(org_detections)):
        # get detected face
        obj = org_detections[idx]
        margin = FACE_MARGIN

        cx = (obj.x + obj.w/2) * w
        cy = (obj.y + obj.h/2) * h
        cw = max(obj.w * w * margin, obj.h * h * margin)
        fx = max(cx - cw/2, 0)
        fy = max(cy - cw/2, 0)
        fw = min(cw, w-fx)
        fh = min(cw, h-fy)
        top_left = (int(fx), int(fy))
        bottom_right = (int((fx+fw)), int(fy+fh))

        # get detected face
        crop_img = frame[
            top_left[1]:bottom_right[1], top_left[0]:bottom_right[0], 0:3
        ]
        if crop_img.shape[0] <= 0 or crop_img.shape[1] <= 0:
            continue
        crop_img, resized_frame = webcamera_utils.adjust_frame_size(
            crop_img, IMAGE_HEIGHT, IMAGE_WIDTH
        )
        detections.append({
            "resized_frame": resized_frame,
            "top_left": top_left,
            "bottom_right": bottom_right,
            "id_sim": 0,
            "score_sim": 0,
            "fe": None
        })
    return detections


def display_detections(ui, w, h, detections):
    for detection in detections:
        # display result
        fontScale = w / 512.0
        thickness = 2
        color = hsv_to_rgb(256 * detection["id_sim"] / 16, 255, 255)
        cv2.rectangle(
            ui, detection["top_left"], detection["bottom_right"], color, 2
        )

        text_position = (
            int(detection["top_left"][0])+4,
            int((detection["bottom_right"][1])-8)
        )

        cv2.putText(
            ui,
            f"{detection['id_sim']}",
            text_position,
            cv2.FONT_HERSHEY_SIMPLEX,
            fontScale,
            color,
            thickness
        )


def display_tracks(ui, w, h, tracks):
    cnt = 0
    for i in range(len(tracks)):
        if len(tracks[i].image) <= 0:
            continue

        y0 = int(IMAGE_HEIGHT/4)*(cnt*2+0)
        y1 = int(IMAGE_HEIGHT/4)*(cnt*2+1)
        y2 = int(IMAGE_HEIGHT/4)*(cnt*2+2)

        for j in range(len(tracks[i].image)):
            x1 = w+int(IMAGE_WIDTH/4)*j
            x2 = w+int(IMAGE_WIDTH/4)*(j+1)
            if x2 > ui.shape[1] or y2 > ui.shape[0]:
                continue
            face = tracks[i].image[j]
            face = cv2.resize(
                face, ((int)(IMAGE_WIDTH/4), (int)(IMAGE_HEIGHT/4))
            )
            ui[y1:y2, x1:x2, :] = face

        fontScale = 0.5
        thickness = 2
        color = hsv_to_rgb(256 * i / 16, 255, 255)
        cv2.rectangle(ui, (w, y0), (ui.shape[1]-2, y2-2), color, 2)

        text_position = (w, y0 + 16)

        cv2.putText(
            ui,
            f"ID {i} : {tracks[i].score:5.3f}",
            text_position,
            cv2.FONT_HERSHEY_SIMPLEX,
            fontScale,
            color,
            thickness
        )

        cnt = cnt + 1


# ======================
# Main functions
# ======================
def compare_images():
    # prepare input data
    imgs_1 = prepare_input_data(args.inputs[0])
    imgs_2 = prepare_input_data(args.inputs[1])
    imgs = np.concatenate([imgs_1, imgs_2], axis=0)

    # net initialize
    net = ailia.Net(MODEL_PATH, WEIGHT_PATH, env_id=args.env_id, memory_mode=args.memory_mode)
    BATCH_SIZE = net.get_input_shape()[0]

    # inference
    logger.info('Start inference...')
    if BATCH_SIZE == 2:
        shape = net.get_output_shape()
        shape = (4, shape[1])
        preds_ailia = np.zeros(shape)
    if args.benchmark:
        logger.info('BENCHMARK mode')
        for i in range(5):
            start = int(round(time.time() * 1000))
            if BATCH_SIZE == 4:
                preds_ailia = net.predict(imgs)
            else:
                preds_ailia[0:2] = net.predict(imgs[0:2])
                preds_ailia[2:4] = net.predict(imgs[2:4])
            end = int(round(time.time() * 1000))
            logger.info(f'\tailia processing time {end - start} ms')
    else:
        if BATCH_SIZE == 4:
            preds_ailia = net.predict(imgs)
        else:
            preds_ailia[0:2] = net.predict(imgs[0:2])
            preds_ailia[2:4] = net.predict(imgs[2:4])

    # postprocessing
    fe_1 = np.concatenate([preds_ailia[0], preds_ailia[1]], axis=0)
    fe_2 = np.concatenate([preds_ailia[2], preds_ailia[3]], axis=0)
    sim = cosin_metric(fe_1, fe_2)

    logger.info(
        f'Similarity of ({args.inputs[0]}, {args.inputs[1]}) : {sim:.3f}'
    )
    if args.threshold > sim:
        logger.info('They are not the same face!')
    else:
        logger.info('They are the same face!')


def compare_video():
    # prepare base image
    tracks = []

    # net initialize
    net = ailia.Net(MODEL_PATH, WEIGHT_PATH, env_id=args.env_id, memory_mode=args.memory_mode)

    # detector initialize
    if args.face == "blazeface":
        detector = ailia.Net(
            FACE_MODEL_PATH, FACE_WEIGHT_PATH, env_id=args.env_id, memory_mode=args.memory_mode
        )
    else:
        detector = ailia.Detector(
            FACE_MODEL_PATH,
            FACE_WEIGHT_PATH,
            len(FACE_CATEGORY),
            format=ailia.NETWORK_IMAGE_FORMAT_RGB,
            channel=ailia.NETWORK_IMAGE_CHANNEL_FIRST,
            range=FACE_RANGE,
            algorithm=FACE_ALGORITHM,
            env_id=args.env_id,
            memory_mode=args.memory_mode
        )

    # web camera
    capture = webcamera_utils.get_capture(args.video)

    # ui buffer
    ui_width = capture.get(cv2.CAP_PROP_FRAME_WIDTH)+IMAGE_WIDTH/4*FACE_TRACK_T
    ui_height = max(
        capture.get(cv2.CAP_PROP_FRAME_HEIGHT), IMAGE_HEIGHT/4*2*8
    )
    ui = np.zeros((int(ui_height), int(ui_width), 3), np.uint8)
    frame_no = 0

    # writer
    writer = None
    if args.savepath is not None:
        writer = webcamera_utils.get_writer(
            args.savepath,
            ui.shape[0],
            ui.shape[1],
            fps=capture.get(cv2.CAP_PROP_FPS),
        )

    # inference loop
    frame_shown = False
    while(True):
        ret, frame = capture.read()
        if (cv2.waitKey(1) & 0xFF == ord('q')) or not ret:
            break
        if frame_shown and cv2.getWindowProperty('arcface', cv2.WND_PROP_VISIBLE) == 0:
            break

        # get frame size
        h, w = frame.shape[0], frame.shape[1]

        # get faces from image
        detections = get_faces(detector, frame, w, h)

        # track face
        face_identification(tracks, net, detections, frame_no)
        frame_no = frame_no+1

        # display result
        ui[:, :, :] = 0
        ui[0:h, 0:w, :] = frame[:, :, :]
        display_detections(ui, w, h, detections)
        display_tracks(ui, w, h, tracks)

        # show
        cv2.imshow('arcface', ui)
        frame_shown = True

        if writer is not None:
            writer.write(ui)

    if writer is not None:
        writer.release()

    capture.release()
    cv2.destroyAllWindows()
    logger.info('Script finished successfully.')


def main():
    # model files check and download
    check_and_download_models(WEIGHT_PATH, MODEL_PATH, REMOTE_PATH)
    if args.video:
        check_and_download_models(
            FACE_WEIGHT_PATH, FACE_MODEL_PATH, FACE_REMOTE_PATH
        )

    if args.video is None:
        # still image mode
        # comparing two images specified args.inputs
        if len(args.inputs)==0:
            args.inputs=[IMG_PATH_1,IMG_PATH_2]
        compare_images()
    else:
        # video mode
        # comparing the specified image and the video
        compare_video()


if __name__ == "__main__":
    main()
