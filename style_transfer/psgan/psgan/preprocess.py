import os.path as osp

pwd = osp.split(osp.realpath(__file__))[0]
import sys

sys.path.append(pwd + "/..")

import ailia
import cv2
import numpy as np
from PIL import Image

import faceutils as futils

sys.path.append("../../util")
from image_utils import load_image  # noqa: E402

sys.path.append("../../face_detection")
from blazeface import blazeface_utils as but

# Logger
from logging import getLogger  # noqa: E402

logger = getLogger(__name__)

FACE_ALIGNMENT_IMAGE_HEIGHT = 256
FACE_ALIGNMENT_IMAGE_WIDTH = 256
FACE_DETECTOR_IMAGE_HEIGHT = 128
FACE_DETECTOR_IMAGE_WIDTH = 128


def resize_multi_channel(img, dsize, interpolation):
    # The maximum number of channels of cv2.resize is 512 until OpenCV 4 and 128 from OpenCV 5.
    CV_CN_MAX = 128

    channels = img.shape[2]
    if channels <= CV_CN_MAX:
        return cv2.resize(img, dsize=dsize, interpolation=interpolation)

    # resize every CV_CN_MAX channels
    resized = np.empty((dsize[1], dsize[0], channels), dtype=img.dtype)
    for i in range(0, channels, CV_CN_MAX):
        part = cv2.resize(
            img[:, :, i:i + CV_CN_MAX], dsize=dsize, interpolation=interpolation
        )
        # cv2.resize drops the last axis when the image has only one channel
        part = part.reshape(dsize[1], dsize[0], -1)
        resized[:, :, i:i + CV_CN_MAX] = part

    return resized


def copy_area(tar, src, lms):
    rect = [
        int(min(lms[:, 1])) - PreProcess.eye_margin,
        int(min(lms[:, 0])) - PreProcess.eye_margin,
        int(max(lms[:, 1])) + PreProcess.eye_margin + 1,
        int(max(lms[:, 0])) + PreProcess.eye_margin + 1,
    ]
    tar[:, :, rect[1] : rect[3], rect[0] : rect[2]] = src[
        :, :, rect[1] : rect[3], rect[0] : rect[2]
    ]
    src[:, :, rect[1] : rect[3], rect[0] : rect[2]] = 0


class FaceAlignment:
    def __init__(self, use_onnx, face_alignment_model, face_alignment_weight, env_id):
        self.face_alignment_model = face_alignment_model
        self.face_alignment_weight = face_alignment_weight
        self.use_onnx = use_onnx
        # initialize net
        if not self.use_onnx:
            self.net = ailia.Net(
                self.face_alignment_model, self.face_alignment_weight, env_id=env_id
            )
        else:
            import onnxruntime

            self.net = onnxruntime.InferenceSession(self.face_alignment_weight)

    def predict(self, data):
        if not self.use_onnx:
            return self.net.predict(data)
        else:
            inputs = {self.net.get_inputs()[0].name: data}
            return np.array(self.net.run(None, inputs))[0]


class FaceDetector:
    def __init__(
        self, use_onnx, face_detector_model, face_detector_weight, input, env_id
    ):
        self.face_detector_model = face_detector_model
        self.face_detector_weight = face_detector_weight
        self.use_onnx = use_onnx
        self.input = input
        # initialize net
        if not self.use_onnx:
            self.net = ailia.Net(
                self.face_detector_model, self.face_detector_weight, env_id=env_id
            )
        else:
            import onnxruntime

            self.net = onnxruntime.InferenceSession(self.face_detector_weight)

    def predict(self, image: Image):
        if self.input == None:
            data = np.array(image)
            data = cv2.resize(
                data, (FACE_DETECTOR_IMAGE_WIDTH, FACE_DETECTOR_IMAGE_HEIGHT)
            )
            data = data / 127.5 - 1.0
            data = data.transpose((2, 0, 1))  # channel first
            data = data[np.newaxis, :, :, :].astype(
                np.float32
            )  # (batch_size, channel, h, w)
        else:
            data = load_image(
                self.input,
                (FACE_DETECTOR_IMAGE_HEIGHT, FACE_DETECTOR_IMAGE_WIDTH),
                normalize_type="127.5",
                gen_input_ailia=True,
            ).astype(np.float32)

        if not self.use_onnx:
            preds_ailia = self.net.predict([data])
        else:
            inputs = {self.net.get_inputs()[0].name: data}
            preds_ailia = self.net.run(None, inputs)
        # postprocessing
        detected = but.postprocess(
            preds_ailia,
            anchor_path=pwd + "/../../../face_detection/blazeface/anchors.npy",
        )[0][0]
        ymin = int(detected[0] * image.size[1])
        xmin = int(detected[1] * image.size[0])
        ymax = int(detected[2] * image.size[1])
        xmax = int(detected[3] * image.size[0])
        return [ymin, xmin, ymax, xmax]


class PreProcess:
    eye_margin = 16
    diff_size = (64, 64)

    def __init__(
        self,
        args=None,
        face_parser_path=None,
        face_alignment_path=None,
        face_detector_path=None,
        need_parser=True,
        return_landmarks=False
    ):

        DATA_IMG_SIZE = 256
        PREPROCESS_UP_RATIO = 0.6 / 0.85  # delta_size / face_size
        PREPROCESS_DOWN_RATIO = 0.2 / 0.85  # delta_size / face_size
        PREPROCESS_WIDTH_RATIO = 0.2 / 0.85  # delta_size / face_size
        PREPROCESS_LIP_CLASS = [7, 9]
        PREPROCESS_FACE_CLASS = [1, 6]
        PREPROCESS_LANDMARK_POINTS = 68

        self.img_size = DATA_IMG_SIZE

        xs, ys = np.meshgrid(
            np.linspace(0, self.img_size - 1, self.img_size),
            np.linspace(0, self.img_size - 1, self.img_size),
        )
        xs = xs[None].repeat(PREPROCESS_LANDMARK_POINTS, axis=0)
        ys = ys[None].repeat(PREPROCESS_LANDMARK_POINTS, axis=0)
        self.fix = np.concatenate([ys, xs], axis=0)
        if need_parser:
            self.face_parse = futils.mask.FaceParser(
                args=args, face_parser_path=face_parser_path
            )
        self.up_ratio = PREPROCESS_UP_RATIO
        self.down_ratio = PREPROCESS_DOWN_RATIO
        self.width_ratio = PREPROCESS_WIDTH_RATIO
        self.lip_class = PREPROCESS_LIP_CLASS
        self.face_class = PREPROCESS_FACE_CLASS
        self.use_onnx = args.onnx
        self.use_dlib = args.use_dlib
        if not self.use_dlib:
            if not args.input:
                self.input = None  # video mode
            else:
                self.input = args.input[0]
            self.face_alignment = FaceAlignment(
                self.use_onnx,
                face_alignment_path[0],
                face_alignment_path[1],
                args.env_id,
            )
            self.face_detector = FaceDetector(
                self.use_onnx,
                face_detector_path[0],
                face_detector_path[1],
                self.input,
                args.env_id,
            )
        self.return_landmarks = return_landmarks

    def relative2absolute(self, lms):
        return lms * self.img_size

    def detect_landmark(self, image, face):
        if self.use_dlib:
            import dlib
            import faceutils.dlibutils as futils_dlib
            predictor = dlib.shape_predictor(
                pwd + "/../faceutils/dlibutils/shape_predictor_68_face_landmarks.dat"
            )
            lms = (
                futils_dlib.landmarks(predictor, image, face)
                * self.img_size
                / image.width
            )
            lms = lms.round()
        else:
            data = np.array(image)
            data = cv2.resize(
                data, (FACE_ALIGNMENT_IMAGE_WIDTH, FACE_ALIGNMENT_IMAGE_HEIGHT)
            )
            data = data / 255.0
            data = data.transpose((2, 0, 1))  # channel first
            data = data[np.newaxis, :, :, :].astype(
                np.float32
            )  # (batch_size, channel, h, w)

            preds_ailia = self.face_alignment.predict(data)
            pts, _ = _get_preds_from_hm(preds_ailia)
            lms = pts.reshape(68, 2) * 4

        return lms

    def process(self, mask, lms):
        lms_eye_left = lms[42:48]
        lms_eye_right = lms[36:42]
        lms = lms.transpose((1, 0)).reshape(-1, 1, 1)  # transpose to (y-x)
        # lms = np.tile(lms, (1, 256, 256))  # (136, h, w)
        diff = np.expand_dims(self.fix.astype("float32") - lms, 0)

        mask_lip = (mask == self.lip_class[0]) + (mask == self.lip_class[1])
        mask_face = (mask == self.face_class[0]) + (mask == self.face_class[1])

        mask_eyes = np.zeros_like(mask)
        copy_area(mask_eyes, mask_face, lms_eye_left)
        copy_area(mask_eyes, mask_face, lms_eye_right)

        mask_list = [mask_lip, mask_face, mask_eyes]
        mask_aug = np.concatenate(mask_list, 0)  # (3, 1, h, w)
        mask_re = cv2.resize(
            mask_aug.squeeze().transpose((1, 2, 0)),
            dsize=self.diff_size,
            interpolation=cv2.INTER_NEAREST,
        ).transpose(2, 0, 1)
        mask_re = np.tile(
            np.expand_dims(mask_re, 1), (1, diff.shape[1], 1, 1)
        )  # (3, 136, 64, 64)
        diff_re = resize_multi_channel(
            diff.squeeze().transpose((1, 2, 0)),
            dsize=self.diff_size,
            interpolation=cv2.INTER_NEAREST,
        ).transpose(2, 0, 1)
        diff_re = np.tile(np.expand_dims(diff_re, 0), (3, 1, 1, 1))  # (3, 136, 64, 64)
        diff_re = diff_re * mask_re  # (3, 136, 32, 32)
        norm = np.tile(
            np.linalg.norm(diff_re, axis=1, keepdims=True), (1, diff_re.shape[1], 1, 1)
        )
        norm = np.where(norm == 0, 1e10, norm)
        diff_re /= norm

        return mask_aug, diff_re

    def __call__(self, image: Image):
        if self.use_dlib:
            import faceutils.dlibutils as futils_dlib
            face = futils_dlib.detect(image)
            if not face:
                return None, None, None
            else:
                face_on_image = face[0]
                image, face, crop_face = futils_dlib.crop(
                    image,
                    face_on_image,
                    self.up_ratio,
                    self.down_ratio,
                    self.width_ratio,
                )
                face_on_image = [
                    face_on_image.left(),
                    face_on_image.top(),
                    face_on_image.right(),
                    face_on_image.bottom(),
                ]
                crop_face = {
                    "left": crop_face.left(),
                    "top": crop_face.top(),
                    "right": crop_face.right(),
                    "bottom": crop_face.bottom(),
                }
        else:
            detected = self.face_detector.predict(image)
            face = [[detected[1], detected[0], detected[3], detected[2]]]

            if not face:
                return None, None, None
            else:
                face_on_image = face[0]
                image, face, crop_face = futils.nondlib.crop(
                    image,
                    face_on_image,
                    self.up_ratio,
                    self.down_ratio,
                    self.width_ratio,
                )
        
        lms = self.detect_landmark(image, face)
        if self.return_landmarks:
            return lms

        np_image = np.array(image).astype(np.float32)
        mask = self.face_parse.parse(cv2.resize(np_image, (512, 512)))
        # obtain face parsing result
        # image = image.resize((512, 512), Image.LANCZOS)
        mask = np.array(
            Image.fromarray(mask).resize(
                (self.img_size, self.img_size), resample=Image.NEAREST
            )
        )
        mask = np.expand_dims(mask, (0, 1))

        mask, diff = self.process(mask, lms)
        image = image.resize((self.img_size, self.img_size), Image.LANCZOS)
        image = np.array(image).transpose((2, 0, 1)) / 255
        means = np.expand_dims([0.5, 0.5, 0.5], (1, 2))
        stds = np.expand_dims([0.5, 0.5, 0.5], (1, 2))
        image = (image - means) / stds
        real = np.expand_dims(image, 0)
        return (
            [real.astype(np.float32), mask.astype(np.float32), diff.astype(np.float32)],
            face_on_image,
            crop_face,
        )


# Copied from face_recognition/face_alignment/face_alignment.py
def _get_preds_from_hm(hm):
    """
    Obtain (x,y) coordinates given a set of N heatmaps.
    ref: 1adrianb/face-alignment/blob/master/face_alignment/utils.py

    Parameters
    ----------
    hm : np.array

    Returns
    -------
    preds:
    preds_orig:

    """
    IMAGE_HEIGHT = 256
    IMAGE_WIDTH = 256

    idx = np.argmax(
        hm.reshape(hm.shape[0], hm.shape[1], hm.shape[2] * hm.shape[3]), axis=2
    )
    idx += 1
    preds = idx.reshape(idx.shape[0], idx.shape[1], 1)
    preds = np.tile(preds, (1, 1, 2)).astype(float)
    preds[..., 0] = (preds[..., 0] - 1) % hm.shape[3] + 1
    preds[..., 1] = np.floor((preds[..., 1] - 1) / (hm.shape[2])) + 1

    for i in range(preds.shape[0]):
        for j in range(preds.shape[1]):
            hm_ = hm[i, j, :]
            pX, pY = int(preds[i, j, 0]) - 1, int(preds[i, j, 1]) - 1
            if pX > 0 and pX < 63 and pY > 0 and pY < 63:
                diff = np.array(
                    [
                        hm_[pY, pX + 1] - hm_[pY, pX - 1],
                        hm_[pY + 1, pX] - hm_[pY - 1, pX],
                    ]
                ).astype(float)
                preds[i, j] = preds[i, j] + (np.sign(diff) * 0.25)

    preds += -0.5
    preds_orig = np.zeros_like(preds)

    for i in range(hm.shape[0]):
        for j in range(hm.shape[1]):
            preds_orig[i, j] = _transform(
                preds[i, j],  # point
                np.array([IMAGE_HEIGHT // 2, IMAGE_WIDTH // 2]),  # center
                (IMAGE_HEIGHT + IMAGE_WIDTH) // 2,  # FIXME not sure... # scale
                hm.shape[2],  # resolution
                True,
            )
    return preds, preds_orig


def _transform(point, center, scale, resolution, invert=False):
    """Generate and affine transformation matrix.

    Given a set of points, a center, a scale and a targer resolution, the
    function generates and affine transformation matrix. If invert is ``True``
    it will produce the inverse transformation.

    Arguments:
        point {torch.tensor} -- the input 2D point
        center {torch.tensor or numpy.array} -- the center around which to
            perform the transformations
        scale {float} -- the scale of the face/object
        resolution {float} -- the output resolution

    Keyword Arguments:
        invert {bool} -- define wherever the function should produce the direct
            or the inverse transformation matrix (default: {False})
    """
    _pt = np.ones(3)
    _pt[0] = point[0]
    _pt[1] = point[1]

    h = scale  # NOTE: originally, scale * 200
    t = np.eye(3)
    t[0, 0] = resolution / h
    t[1, 1] = resolution / h
    t[0, 2] = resolution * (-center[0] / h + 0.5)
    t[1, 2] = resolution * (-center[1] / h + 0.5)

    if invert:
        t = np.linalg.inv(t)
    new_point = (np.dot(t, _pt))[0:2]
    return new_point.astype(int)
