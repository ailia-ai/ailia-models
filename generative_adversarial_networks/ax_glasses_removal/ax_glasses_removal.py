import math
import sys
import time
from logging import getLogger

import ailia
import cv2
import numpy as np

sys.path.append("../../util")
from arg_utils import get_base_parser, get_savepath, update_parser  # noqa: E402
from image_utils import imread  # noqa: E402
from load_model import load_facemesh_v2  # noqa: E402
from model_utils import (  # noqa: E402
    check_and_download_file,
    check_and_download_models,
)
from webcamera_utils import get_capture, get_writer  # noqa: E402

from ax_glasses_removal_utils import (  # noqa: E402
    GlassesRemover,
    TemporalSmoother,
    deglass_roi_box,
)


logger = getLogger(__name__)


WEIGHT_PATH = "ax_glasses_removal_mobile.onnx"
REMOTE_PATH = "https://storage.googleapis.com/ailia-models/ax_glasses_removal/"

FACE_DET_WEIGHT_PATH = "face_detector.onnx"
FACE_DET_MODEL_PATH = "face_detector.onnx.prototxt"
FACE_LMK_WEIGHT_PATH = "face_landmarks_detector.onnx"
FACE_LMK_MODEL_PATH = "face_landmarks_detector.onnx.prototxt"
FACEMESH_REMOTE_PATH = "https://storage.googleapis.com/ailia-models/facemesh_v2/"

IMAGE_PATH = "sample.jpg"
SAVE_IMAGE_PATH = "output.png"

NUM_LANDMARKS = 478
PRESENCE_THRESHOLD = 0.5
TRACK_SIZE_FROM_BBOX = 1.475
TRACK_CENTER_X_FROM_BBOX = 0.038
TRACK_CENTER_Y_FROM_BBOX = 0.028
EYE_OUTER_RIGHT = 33
EYE_OUTER_LEFT = 263


parser = get_base_parser(
    "Remove eyeglasses from a face image.",
    IMAGE_PATH,
    SAVE_IMAGE_PATH,
)
parser.add_argument(
    "--show-mask",
    action="store_true",
    help="Overlay the raw glasses segmentation probability.",
)
args = update_parser(parser)


def _sigmoid(value):
    value = np.clip(value, -100, 100)
    return 1 / (1 + math.exp(-float(value)))


class FaceMeshV2:
    """Single-face FaceMesh-v2 pipeline matching the current JavaScript demo."""

    def __init__(self, detector, landmark, module):
        self.detector = detector
        self.landmark = landmark
        self.module = module
        self.tracked = None

    def reset_tracking(self):
        self.tracked = None

    def _tracking_roi(self, landmarks, width, height):
        x0 = landmarks[EYE_OUTER_RIGHT, 0] * width
        y0 = landmarks[EYE_OUTER_RIGHT, 1] * height
        x1 = landmarks[EYE_OUTER_LEFT, 0] * width
        y1 = landmarks[EYE_OUTER_LEFT, 1] * height
        if not np.all(np.isfinite([x0, y0, x1, y1])):
            return None

        angle = -math.atan2(-(y1 - y0), x1 - x0)
        angle -= 2 * math.pi * math.floor((angle + math.pi) / (2 * math.pi))
        cosine = math.cos(angle)
        sine = math.sin(angle)

        points_x = landmarks[:, 0] * width
        points_y = landmarks[:, 1] * height
        if not np.all(np.isfinite(points_x)) or not np.all(np.isfinite(points_y)):
            return None
        u = points_x * cosine + points_y * sine
        v = -points_x * sine + points_y * cosine
        min_u, max_u = np.min(u), np.max(u)
        min_v, max_v = np.min(v), np.max(v)
        box_width = max_u - min_u
        box_height = max_v - min_v
        if not (box_width > 0 and box_height > 0):
            return None

        size = TRACK_SIZE_FROM_BBOX * max(box_width, box_height)
        center_u = (min_u + max_u) / 2 + TRACK_CENTER_X_FROM_BBOX * box_width
        center_v = (min_v + max_v) / 2 + TRACK_CENTER_Y_FROM_BBOX * box_height
        return self.module.ROI(
            center_u * cosine - center_v * sine,
            center_u * sine + center_v * cosine,
            size,
            size,
            angle,
        )

    def _detect(self, rgb):
        detector_input, matrix = self.module.preprocess_det(rgb)
        outputs = self.detector.predict([detector_input])
        regressors = next(
            (output for output in outputs if output.shape[-1] == 16), None
        )
        scores_tensor = next(
            (output for output in outputs if output.shape[-1] == 1), None
        )
        if regressors is None or scores_tensor is None:
            shapes = ", ".join(str(output.shape) for output in outputs)
            raise RuntimeError(f"Unexpected face-detector outputs: {shapes}")
        boxes, scores = self.module.face_detection(regressors, scores_tensor, matrix)
        if len(boxes) == 0:
            return None
        return boxes[int(np.argmax(scores))]

    def _face_roi_from_box(self, box, width, height):
        rect_width = box[2] - box[0]
        rect_height = box[3] - box[1]
        center_x = (box[0] + box[2]) * 0.5
        center_y = (box[1] + box[3]) * 0.5
        x0, y0 = box[4] * width, box[5] * height
        x1, y1 = box[6] * width, box[7] * height
        angle = -math.atan2(-(y1 - y0), x1 - x0)
        angle -= 2 * math.pi * math.floor((angle + math.pi) / (2 * math.pi))
        return self.module.ROI(
            center_x * width,
            center_y * height,
            rect_width * width * 1.5,
            rect_height * height * 1.5,
            angle,
        )

    def predict(self, frame):
        height, width = frame.shape[:2]
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        roi = (
            self._tracking_roi(self.tracked, width, height)
            if self.tracked is not None
            else None
        )
        if roi is None:
            box = self._detect(rgb)
            if box is None:
                self.tracked = None
                return None
            roi = self._face_roi_from_box(box, width, height)

        landmark_input, roi, pad = self.module.preprocess(rgb, roi)
        outputs = self.landmark.predict([landmark_input])
        raw = next((output for output in outputs if output.size >= NUM_LANDMARKS * 3), None)
        presence = next((output for output in outputs if output.size == 1), None)
        if raw is None or presence is None:
            shapes = ", ".join(str(output.shape) for output in outputs)
            raise RuntimeError(f"Unexpected face-landmark outputs: {shapes}")
        if _sigmoid(presence.reshape(-1)[0]) < PRESENCE_THRESHOLD:
            self.tracked = None
            return None

        normalized_roi = self.module.ROI(
            roi.x_center / width,
            roi.y_center / height,
            roi.width / width,
            roi.height / height,
            roi.rotation,
        )
        landmarks = self.module.post_processing(raw, normalized_roi, pad)
        self.tracked = landmarks.copy()
        return landmarks


def process_frame(face_mesh, remover, frame, smoother, show_mask=False):
    landmarks = face_mesh.predict(frame)
    if landmarks is None:
        smoother.reset()
        return frame.copy(), False

    landmarks = smoother.smooth_landmarks(landmarks)
    # JavaScript keeps the same Float32Array for tracking and smoothing.
    face_mesh.tracked = landmarks.copy()
    raw_bounds = deglass_roi_box(
        landmarks,
        frame.shape[1],
        frame.shape[0],
        remover.input_width,
        remover.input_height,
    )
    bounds = smoother.smooth_roi(raw_bounds)
    return remover.remove(frame, bounds, show_mask=show_mask), True


def recognize_from_image(face_mesh, remover):
    for image_path in args.input:
        logger.info(image_path)
        image = imread(image_path)
        if image is None:
            logger.error("Could not read %s", image_path)
            continue

        logger.info("Start inference...")
        smoother = TemporalSmoother()
        if args.benchmark:
            times = []
            result = image
            found = False
            for _ in range(args.benchmark_count):
                face_mesh.reset_tracking()
                smoother.reset()
                start = time.perf_counter()
                result, found = process_frame(
                    face_mesh, remover, image, smoother, args.show_mask
                )
                elapsed = (time.perf_counter() - start) * 1000
                times.append(elapsed)
                logger.info("\tailia processing time %.1f ms", elapsed)
            measured = times[1:] if len(times) > 1 else times
            logger.info("\taverage time %.1f ms", np.mean(measured))
        else:
            face_mesh.reset_tracking()
            result, found = process_frame(
                face_mesh, remover, image, smoother, args.show_mask
            )

        if not found:
            logger.warning("No face was detected; saving the original image.")
        savepath = get_savepath(args.savepath, image_path)
        logger.info("saved at : %s", savepath)
        cv2.imwrite(savepath, result)
    logger.info("Script finished successfully.")


def recognize_from_video(face_mesh, remover):
    capture = get_capture(args.video)
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    fps = capture.get(cv2.CAP_PROP_FPS)
    if not fps or not np.isfinite(fps):
        fps = 20
    writer = (
        get_writer(args.savepath, height, width, fps=fps)
        if args.savepath != SAVE_IMAGE_PATH
        else None
    )

    smoother = TemporalSmoother()
    frame_shown = False
    while True:
        ret, frame = capture.read()
        if not ret or (cv2.waitKey(1) & 0xFF == ord("q")):
            break
        if frame_shown and cv2.getWindowProperty("frame", cv2.WND_PROP_VISIBLE) == 0:
            break

        result, found = process_frame(
            face_mesh, remover, frame, smoother, args.show_mask
        )
        if not found:
            face_mesh.reset_tracking()
        cv2.imshow("frame", result)
        frame_shown = True
        if writer is not None:
            writer.write(result)

    capture.release()
    if writer is not None:
        writer.release()
    cv2.destroyAllWindows()
    logger.info("Script finished successfully.")


def main():
    check_and_download_file(WEIGHT_PATH, REMOTE_PATH)
    check_and_download_models(
        FACE_DET_WEIGHT_PATH,
        FACE_DET_MODEL_PATH,
        FACEMESH_REMOTE_PATH,
    )
    check_and_download_models(
        FACE_LMK_WEIGHT_PATH,
        FACE_LMK_MODEL_PATH,
        FACEMESH_REMOTE_PATH,
    )

    facemesh_module = load_facemesh_v2(args)
    detector = ailia.Net(
        FACE_DET_MODEL_PATH,
        FACE_DET_WEIGHT_PATH,
        env_id=args.env_id,
    )
    landmark = ailia.Net(
        FACE_LMK_MODEL_PATH,
        FACE_LMK_WEIGHT_PATH,
        env_id=args.env_id,
    )
    removal_net = ailia.Net(None, WEIGHT_PATH, env_id=args.env_id)
    face_mesh = FaceMeshV2(detector, landmark, facemesh_module)
    remover = GlassesRemover(removal_net)

    if args.video is not None:
        recognize_from_video(face_mesh, remover)
    else:
        recognize_from_image(face_mesh, remover)


if __name__ == "__main__":
    main()
