import sys
import time
from contextlib import contextmanager

import ailia
import cv2
import numpy as np
import json

import ax_gaze_estimation_utils as gut

sys.path.append('../../util')
# logger
from logging import getLogger  # noqa: E402

from image_utils import imread  # noqa: E402
from model_utils import check_and_download_models  # noqa: E402
from arg_utils import get_base_parser, get_savepath, update_parser  # noqa: E402
from webcamera_utils import get_capture, get_writer  # noqa: E402

logger = getLogger(__name__)


# ======================
# Parameters 1
# ======================
IMAGE_PATH = 'woman_face.jpg'
SAVE_IMAGE_PATH = 'output.png'
IMAGE_HEIGHT = 128
IMAGE_WIDTH = 128


# ======================
# Argument Parser Config
# ======================
parser = get_base_parser(
    'Gaze estimation.', IMAGE_PATH, SAVE_IMAGE_PATH,
)
parser.add_argument(
    '-n', '--normal',
    action='store_true',
    help='By default, the optimized model is used, but with this option, ' +
    'you can switch to the normal (not optimized) model'
)
parser.add_argument(
    '--include-iris',
    action='store_true',
    help='By default, the model does not estimate iris landmarks and uses a' +
    'rough estimation for the pupil centers. This option allows a more ' +
    'accurate estimation but adds overhead (slower).'
)
parser.add_argument(
    '--draw-iris',
    action='store_true',
    help='Whether to draw the iris landmarks or not.'
)
parser.add_argument(
    '--include-head-pose',
    action='store_true',
    help='By default, the model only uses the face images to predict the' +
    'gaze. This option allows including the head pose for prediction (higher' +
    'accuracy but slower).'
)
parser.add_argument(
    '--draw-head-pose',
    action='store_true',
    help='Whether to draw the head pose(s) or not.'
)
parser.add_argument(
    '-l', '--lite',
    action='store_true',
    help='With this option, a lite version of the head pose model is used ' +
    '(only valid when --include-head-pose is specified).'
)
parser.add_argument(
    '-w', '--write_json',
    action='store_true',
    help='Flag to output results to json file.'
)
args = update_parser(parser)


# ======================
# Parameters 2
# ======================
FACE_DET_MODEL_NAME = 'blazeface'
FACE_LM_MODEL_NAME = 'facemesh'
IRIS_LM_MODEL_NAME = 'iris'
if args.lite:
    HEAD_POSE_MODEL_NAME = 'hopenet_lite'
else:
    HEAD_POSE_MODEL_NAME = 'hopenet_robust_alpha1'
if args.include_head_pose:
    GAZE_MODEL_NAME = 'ax_gaze_estimation_hp'
else:
    GAZE_MODEL_NAME = 'ax_gaze_estimation'
if args.normal:
    FACE_DET_WEIGHT_PATH = f'{FACE_DET_MODEL_NAME}.onnx'
    FACE_DET_MODEL_PATH = f'{FACE_DET_MODEL_NAME}.onnx.prototxt'
    FACE_LM_WEIGHT_PATH = f'{FACE_LM_MODEL_NAME}.onnx'
    FACE_LM_MODEL_PATH = f'{FACE_LM_MODEL_NAME}.onnx.prototxt'
    IRIS_LM_WEIGHT_PATH = f'{IRIS_LM_MODEL_NAME}.onnx'
    IRIS_LM_MODEL_PATH = f'{IRIS_LM_MODEL_NAME}.onnx.prototxt'
    HEAD_POSE_WEIGHT_PATH = f'{HEAD_POSE_MODEL_NAME}.onnx'
    HEAD_POSE_MODEL_PATH = f'{HEAD_POSE_MODEL_NAME}.onnx.prototxt'
    GAZE_WEIGHT_PATH = f'{GAZE_MODEL_NAME}.onnx'
    GAZE_MODEL_PATH = f'{GAZE_MODEL_NAME}.onnx.prototxt'
else:
    FACE_DET_WEIGHT_PATH = f'{FACE_DET_MODEL_NAME}.opt.onnx'
    FACE_DET_MODEL_PATH = f'{FACE_DET_MODEL_NAME}.opt.onnx.prototxt'
    FACE_LM_WEIGHT_PATH = f'{FACE_LM_MODEL_NAME}.opt.onnx'
    FACE_LM_MODEL_PATH = f'{FACE_LM_MODEL_NAME}.opt.onnx.prototxt'
    IRIS_LM_WEIGHT_PATH = f'{IRIS_LM_MODEL_NAME}.opt.onnx'
    IRIS_LM_MODEL_PATH = f'{IRIS_LM_MODEL_NAME}.opt.onnx.prototxt'
    HEAD_POSE_WEIGHT_PATH = f'{HEAD_POSE_MODEL_NAME}.opt.onnx'
    HEAD_POSE_MODEL_PATH = f'{HEAD_POSE_MODEL_NAME}.opt.onnx.prototxt'
    GAZE_WEIGHT_PATH = f'{GAZE_MODEL_NAME}.opt.obf.onnx'
    GAZE_MODEL_PATH = f'{GAZE_MODEL_NAME}.opt.obf.onnx.prototxt'
FACE_DET_REMOTE_PATH = f'https://storage.googleapis.com/ailia-models/{FACE_DET_MODEL_NAME}/'
FACE_LM_REMOTE_PATH = f'https://storage.googleapis.com/ailia-models/{FACE_LM_MODEL_NAME}/'
IRIS_LM_REMOTE_PATH = f'https://storage.googleapis.com/ailia-models/mediapipe_{IRIS_LM_MODEL_NAME}/'
HEAD_POSE_REMOTE_PATH = f'https://storage.googleapis.com/ailia-models/hopenet/'
GAZE_REMOTE_PATH = f'https://storage.googleapis.com/ailia-models/ax_gaze_estimation/'


# ======================
# Utils
# ======================
@contextmanager
def time_execution(msg):
    start = time.perf_counter()
    yield
    logger.debug(f'{msg} {(time.perf_counter() - start) * 1000:.0f} ms')

class GazeEstimator:
    """Class for estimating the gaze direction

    Wrap all neural networks in the pipeline to provide a centralized and
    easy-to-use class for estimating the gaze direction given an image.
    Include convenient draw method.
    """

    def __init__(self, include_iris=False, include_head_pose=False):
        """Initialize a gaze estimator with or without head pose estimation.

        Parameters
        ----------
        include_iris : bool, optional
            Estimate iris landmarks for more accurate centers of origin of the
            gaze vectors.
        include_head_pose : bool, optional
            Estimate the gaze with or without head pose information.
        """
        self.include_iris = include_iris
        self.include_head_pose = include_head_pose
        # net initialize
        self.face_detector = ailia.Net(
            FACE_DET_MODEL_PATH, FACE_DET_WEIGHT_PATH, env_id=args.env_id,
            memory_mode=args.memory_mode
        )
        self.face_estimator = ailia.Net(
            FACE_LM_MODEL_PATH, FACE_LM_WEIGHT_PATH, env_id=args.env_id,
            memory_mode=args.memory_mode
        )
        if self.include_iris:
            self.iris_estimator = ailia.Net(
                IRIS_LM_MODEL_PATH, IRIS_LM_WEIGHT_PATH, env_id=args.env_id,
                memory_mode=args.memory_mode
            )
        if self.include_head_pose:
            self.hp_estimator = ailia.Net(
                HEAD_POSE_MODEL_PATH, HEAD_POSE_WEIGHT_PATH, env_id=args.env_id,
                memory_mode=args.memory_mode
            )
        self.gaze_estimator = ailia.Net(
            GAZE_MODEL_PATH, GAZE_WEIGHT_PATH, env_id=args.env_id,
            memory_mode=args.memory_mode
        )

    def predict(self, img, gazes_only=True):
        """Predict the gaze given an image.

        Parameters
        ----------
        img : NumPy array
            The image in BGR channels.
        gazes_only : bool, optional
            If True, only return the predicted gaze(s).

        Returns
        -------
        gazes_vec : NumPy array
            Predicted 3D (x, y, z) gaze vector(s). The axes of
            reference correspond to x oriented positively to the right of the
            image, y oriented positively to the bottom of the image and z
            oriented positively to the back of the image (from the POV of
            someone looking at the image).
        gaze_centers : NumPy array, optional
            Estimated centers of origin for the gaze vectors.
        eyes_iris : tuple[NumPy array, NumPy array], optional
            Predicted eye-region and iris landmarks.
        hps_orig : NumPy array, optional
            Head pose(s) in radians. Roll (left+), yaw (right+), pitch (down+)
            values are given in the detected person's frame of reference.
        roi_centers : NumPy array, optional
            Centers (x, y) of the cropped face image(s). Used for drawing the
            head pose(s).
        """
        gazes_vec = None
        gaze_centers = None
        eyes_iris = None
        hps_orig = None
        roi_centers = None
        # Face detection
        with time_execution('\t\t\tpreprocessing'):
            input_face_det, scale, padding = gut.face_detector_preprocess(img)
        with time_execution('\t\tBlazeFace'):
            preds_det = self.face_detector.predict([input_face_det])
        with time_execution('\t\t\tpostprocessing'):
            detections = gut.face_detector_postprocess(preds_det)

        # Face landmark estimation
        if detections[0].size != 0:
            with time_execution('\t\t\tpreprocessing'):
                face_imgs, face_affs, roi_centers, theta = gut.face_lm_preprocess(
                    img, detections, scale, padding
                )
                self.face_estimator.set_input_shape(face_imgs.shape)
            with time_execution('\t\tFace Mesh'):
                landmarks, confidences = self.face_estimator.predict([face_imgs])
            if not self.include_iris:
                with time_execution('\t\t\tpostprocessing'):
                    gaze_centers = gut.face_lm_postprocess(landmarks, face_affs)
            else:
                # Iris landmark estimation (optional)
                with time_execution('\t\t\tpreprocessing'):
                    eye_imgs, eye_origins = gut.iris_preprocess(face_imgs, landmarks)
                    self.iris_estimator.set_input_shape(eye_imgs.shape)
                with time_execution('\t\tIris'):
                    eyes_norm, iris_norm = self.iris_estimator.predict([eye_imgs])
                with time_execution('\t\t\tpostprocessing'):
                    gaze_centers, eyes_iris = gut.iris_postprocess(eyes_norm, iris_norm, eye_origins, face_affs)

            # Head pose estimation (optional)
            if self.include_head_pose:
                with time_execution('\t\t\tpreprocessing'):
                    input_hp = gut.head_pose_preprocess(face_imgs)
                    self.hp_estimator.set_input_shape(input_hp.shape)
                with time_execution('\t\tHopenet'):
                    hps = self.hp_estimator.predict([input_hp])
                with time_execution('\t\t\tpostprocessing'):
                    hps, hps_orig = gut.head_pose_postprocess(hps, theta)

            # Gaze estimation
            with time_execution('\t\t\tpreprocessing'):
                gaze_input_blob = self.gaze_estimator.get_input_blob_list()
                gaze_input1 = np.moveaxis(face_imgs, 1, -1)
                self.gaze_estimator.set_input_blob_shape(gaze_input1.shape, gaze_input_blob[0])
                self.gaze_estimator.set_input_blob_data(gaze_input1, gaze_input_blob[0])
                if self.include_head_pose:
                    gaze_input2 = hps
                    self.gaze_estimator.set_input_blob_shape(gaze_input2.shape, gaze_input_blob[1])
                    self.gaze_estimator.set_input_blob_data(gaze_input2, gaze_input_blob[1])
            with time_execution('\t\tGaze estimation'):
                self.gaze_estimator.update()
                gazes = self.gaze_estimator.get_results()[0]
            with time_execution('\t\t\tpostprocessing'):
                gazes_vec = gut.gaze_postprocess(gazes, face_affs)

        if gazes_only:
            return gazes_vec
        else:
            return gazes_vec, gaze_centers, eyes_iris, hps_orig, roi_centers

    def draw(self, img, gazes, gaze_centers, eyes_iris=None, hps=None, roi_centers=None, draw_iris=False,
             draw_head_pose=False, horizontal_flip=False):
        """Draw the gaze(s) and landmarks (and head pose(s)) on the image.

        Regarding the head pose(s), (person POV) the axes correspond to
        x (blue) oriented positively to the left, y (green) oriented positively
        to the bottom and z (red) oriented positively to the back.

        Parameters
        ----------
        img : NumPy array
            The image to draw on (BGR channels).
        gazes : NumPy array
            The gaze(s) to draw.
        gaze_centers : NumPy array
            The centers of origin of the gaze(s).
        eyes_iris : NumPy array, optional
            The eye-region and iris landmarks to draw.
        hps : NumPy array, optional
            The head pose(s) to draw.
        roi_centers : NumPy array, optional
            The center(s) of origin of the head pose(s).
        draw_iris : bool, optional
            Whether to draw the iris landmarks or not.
        draw_head_pose : bool, optional
            Whether to draw the head pose(s) or not.
        horizontal_flip : bool, optional
            Whether to consider a horizontally flipped image for drawing.

        Returns
        -------
        img_draw : NumPy array
            Image with the gaze(s) and landmarks (and head pose(s)) drawn on it.
        """
        with time_execution('\t\tDrawing'):
            img_draw = img.copy()
            if eyes_iris is not None and draw_iris:
                eyes, iris = eyes_iris
                for i in range(len(eyes)):
                    gut.draw_eye_iris(
                        img_draw, eyes[i, :, :16, :2], iris[i, :, :, :2], size=1
                    )
            if horizontal_flip:
                img_draw = np.ascontiguousarray(img_draw[:, ::-1])
            if hps is not None and roi_centers is not None and draw_head_pose:
                gut.draw_head_poses(img_draw, hps, roi_centers, horizontal_flip=horizontal_flip)
            gut.draw_gazes(img_draw, gazes, gaze_centers, horizontal_flip=horizontal_flip)

        return img_draw

    def predict_and_draw(self, img, draw_iris=False, draw_head_pose=False, results=None):
        """Predict and draw the gaze(s) and landmarks (and head pose(s)).

        Convenient method for predicting the gaze(s) and landmarks (and head
        pose(s)) and drawing them at once.

        Parameters
        ----------
        img : NumPy array
            The image in BGR channels.

        Returns
        -------
        img_draw : NumPy array
            Image with the gaze(s) and landmarks (and head pose(s)) drawn on it.
        draw_iris : bool, optional
            Whether to draw the iris landmarks or not.
        draw_head_pose : bool, optional
            Whether to draw the head pose(s) or not.
        results: list, optional
            Result values stored to this list.
        """
        if results is not None:
            results.clear()
        img_draw = img.copy()
        preds = self.predict(img, gazes_only=False)
        if preds[0] is not None:
            img_draw = self.draw(img, *preds, draw_iris=draw_iris,
                                 draw_head_pose=draw_head_pose)
            if results is not None:
                results.append({
                    'gazes': preds[0],
                    'gaze_centers': preds[1],
                    'eyes_iris': preds[2],
                    'head_poses': preds[3],
                    'roi_centers': preds[4]
                })
        return img_draw


def save_result_json(json_path, results):
    output = []
    for r in results:
        output.append({k: v.tolist() for k, v in r.items() if v is not None})
    with open(json_path, 'w') as f:
        json.dump(output, f, indent=2)


# ======================
# Main functions
# ======================
def recognize_from_image():
    estimator = GazeEstimator(args.include_iris, args.include_head_pose)

    # input image loop
    for image_path in args.input:
        results = []
        logger.info(image_path)
        src_img = imread(image_path)

        # inference
        logger.info('Start inference...')
        if args.benchmark:
            logger.info('BENCHMARK mode')
            for _ in range(5):
                start = int(round(time.time() * 1000))
                img_draw = estimator.predict_and_draw(src_img, args.draw_iris, args.draw_head_pose, results)
                end = int(round(time.time() * 1000))
                logger.info(f'\tailia processing time {end - start} ms')
        else:
            img_draw = estimator.predict_and_draw(src_img, args.draw_iris, args.draw_head_pose, results)

        savepath = get_savepath(args.savepath, image_path)
        logger.info(f'saved at : {savepath}')
        cv2.imwrite(savepath, img_draw)

        if args.write_json:
            json_file = '%s.json' % savepath.rsplit('.', 1)[0]
            save_result_json(json_file, results)

    logger.info('Script finished successfully.')


def recognize_from_video():
    estimator = GazeEstimator(args.include_iris, args.include_head_pose)

    capture = get_capture(args.video)

    # create video writer if savepath is specified as video format
    if args.savepath != SAVE_IMAGE_PATH:
        f_h = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        f_w = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        writer = get_writer(args.savepath, f_h, f_w, fps=capture.get(cv2.CAP_PROP_FPS))
    else:
        writer = None

    frame_shown = False
    while(True):
        ret, frame = capture.read()
        if (cv2.waitKey(1) & 0xFF == ord('q')) or not ret:
            break
        if frame_shown and cv2.getWindowProperty('frame', cv2.WND_PROP_VISIBLE) == 0:
            break

        preds = estimator.predict(frame, gazes_only=False)
        if preds[0] is not None:
            frame_draw = estimator.draw(frame, *preds, draw_iris=args.draw_iris, draw_head_pose=args.draw_head_pose)
        else:
            frame_draw = frame.copy()

        if args.video == '0': # Flip horizontally if camera
            visual_img = cv2.flip(frame_draw, 1)
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
    logger.info('Script finished successfully.')
    pass


def main():
    # model files check and download
    check_and_download_models(
        FACE_DET_WEIGHT_PATH, FACE_DET_MODEL_PATH, FACE_DET_REMOTE_PATH
    )
    check_and_download_models(
        FACE_LM_WEIGHT_PATH, FACE_LM_MODEL_PATH, FACE_LM_REMOTE_PATH
    )
    if args.include_iris:
        check_and_download_models(
            IRIS_LM_WEIGHT_PATH, IRIS_LM_MODEL_PATH, IRIS_LM_REMOTE_PATH
        )
    if args.include_head_pose:
        check_and_download_models(
            HEAD_POSE_WEIGHT_PATH, HEAD_POSE_MODEL_PATH, HEAD_POSE_REMOTE_PATH
        )
    check_and_download_models(
        GAZE_WEIGHT_PATH, GAZE_MODEL_PATH, GAZE_REMOTE_PATH
    )

    if args.video is not None:
        # video mode
        recognize_from_video()
    else:
        # image mode
        recognize_from_image()


if __name__ == '__main__':
    main()
