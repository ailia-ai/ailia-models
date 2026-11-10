import os
import sys
import time

import numpy as np
import cv2
from skimage import transform as sk_trans

import ailia

# import original modules
sys.path.append("../../util")
from arg_utils import get_base_parser, update_parser  # noqa
from model_utils import check_and_download_models  # noqa
from detector_utils import load_image  # noqa
from nms_utils import nms_boxes
from math_utils import softmax
from load_model import load_facemesh_v2
from webcamera_utils import get_capture, get_writer  # noqa

from utils_crop import crop_image

# logger
from logging import getLogger  # noqa


PI = np.pi

top_path = os.path.dirname(
    os.path.dirname(
        os.path.dirname(
            os.path.abspath(__file__),
        )
    )
)

logger = getLogger(__name__)


# ======================
# Parameters
# ======================

WEIGHT_F_PATH = "appearance_feature_extractor.onnx"
MODEL_F_PATH = "appearance_feature_extractor.onnx.prototxt"
WEIGHT_M_PATH = "motion_extractor.onnx"
MODEL_M_PATH = "motion_extractor.onnx.prototxt"
WEIGHT_W_PATH = "warping_module.onnx"
MODEL_W_PATH = "warping_module.onnx.prototxt"
WEIGHT_G_PATH = "spade_generator.onnx"
MODEL_G_PATH = "spade_generator.onnx.prototxt"
WEIGHT_S_PATH = "stitching.onnx"
MODEL_S_PATH = "stitching.onnx.prototxt"
WEIGHT_L_PATH = "landmark.onnx"
MODEL_L_PATH = "landmark.onnx.prototxt"
# insightface model
WEIGHT_IF_DET_PATH = "det_10g.onnx"
MODEL_IF_DET_PATH = "det_10g.onnx.prototxt"
WEIGHT_IF_LMK_PATH = "2d106det.onnx"
MODEL_IF_LMK_PATH = "2d106det.onnx.prototxt"
# facemesh v2 model
WEIGHT_FM_DET_PATH = "face_detector.onnx"
MODEL_FM_DET_PATH = "face_detector.onnx.prototxt"
WEIGHT_FM_LMK_PATH = "face_landmarks_detector.onnx"
MODEL_FM_LMK_PATH = "face_landmarks_detector.onnx.prototxt"

REMOTE_PATH = "https://storage.googleapis.com/ailia-models/live_portrait/"
REMOTE_FM_V2_PATH = "https://storage.googleapis.com/ailia-models/facemesh_v2/"

IMAGE_PATH = "s6.jpg"
DRIVING_VIDEO_PATH = "d0.mp4"
MASK_PATH = "mask_template.png"


# ======================
# Arguemnt Parser Config
# ======================

parser = get_base_parser("LivePortrait", IMAGE_PATH, None)
parser.add_argument(
    "--driving", metavar="VIDEO", default=DRIVING_VIDEO_PATH, help="Driving video."
)
parser.add_argument(
    "--det",
    default="insightface",
    choices=["insightface", "facemesh"],
    help="Face analysis module.",
)
parser.add_argument(
    "--composite",
    action="store_true",
    help='Combine "driving frame | source image | generation frame".',
)
parser.add_argument(
    "--cui",
    action="store_true",
    help="Don't display perview in GUI.",
)
parser.add_argument("--onnx", action="store_true", help="execute onnxruntime version.")
args = update_parser(parser)


# ======================
# Secondary Functions
# ======================


def distance2bbox(points, distance, max_shape=None):
    x1 = points[:, 0] - distance[:, 0]
    y1 = points[:, 1] - distance[:, 1]
    x2 = points[:, 0] + distance[:, 2]
    y2 = points[:, 1] + distance[:, 3]
    if max_shape is not None:
        x1 = x1.clamp(min=0, max=max_shape[1])
        y1 = y1.clamp(min=0, max=max_shape[0])
        x2 = x2.clamp(min=0, max=max_shape[1])
        y2 = y2.clamp(min=0, max=max_shape[0])
    return np.stack([x1, y1, x2, y2], axis=-1)


def distance2kps(points, distance, max_shape=None):
    """Decode distance prediction to bounding box.

    Args:
        points (Tensor): Shape (n, 2), [x, y].
        distance (Tensor): Distance from the given point to 4
            boundaries (left, top, right, bottom).
        max_shape (tuple): Shape of the image.

    Returns:
        Tensor: Decoded bboxes.
    """
    preds = []
    for i in range(0, distance.shape[1], 2):
        px = points[:, i % 2] + distance[:, i]
        py = points[:, i % 2 + 1] + distance[:, i + 1]
        if max_shape is not None:
            px = px.clamp(min=0, max=max_shape[1])
            py = py.clamp(min=0, max=max_shape[0])
        preds.append(px)
        preds.append(py)
    return np.stack(preds, axis=-1)


def face_align(data, center, output_size, scale, rotation):
    scale_ratio = scale
    rot = float(rotation) * np.pi / 180.0

    t1 = sk_trans.SimilarityTransform(scale=scale_ratio)
    cx = center[0] * scale_ratio
    cy = center[1] * scale_ratio
    t2 = sk_trans.SimilarityTransform(translation=(-1 * cx, -1 * cy))
    t3 = sk_trans.SimilarityTransform(rotation=rot)
    t4 = sk_trans.SimilarityTransform(translation=(output_size / 2, output_size / 2))
    t = t1 + t2 + t3 + t4
    M = t.params[0:2]
    cropped = cv2.warpAffine(data, M, (output_size, output_size), borderValue=0.0)

    return cropped, M


def trans_points2d(pts, M):
    new_pts = np.zeros(shape=pts.shape, dtype=np.float32)
    for i in range(pts.shape[0]):
        pt = pts[i]
        new_pt = np.array([pt[0], pt[1], 1.0], dtype=np.float32)
        new_pt = np.dot(M, new_pt)
        new_pts[i] = new_pt[0:2]

    return new_pts


def calculate_distance_ratio(
    lmk: np.ndarray, idx1: int, idx2: int, idx3: int, idx4: int, eps: float = 1e-6
) -> np.ndarray:
    return np.linalg.norm(lmk[:, idx1] - lmk[:, idx2], axis=1, keepdims=True) / (
        np.linalg.norm(lmk[:, idx3] - lmk[:, idx4], axis=1, keepdims=True) + eps
    )


def get_rotation_matrix(pitch_, yaw_, roll_):
    """the input is in degree"""
    # transform to radian
    pitch = pitch_ / 180 * PI
    yaw = yaw_ / 180 * PI
    roll = roll_ / 180 * PI

    # calculate the euler matrix
    bs = pitch.shape[0]
    ones = np.ones([bs, 1], dtype=np.float32)
    zeros = np.zeros([bs, 1], dtype=np.float32)
    x, y, z = pitch, yaw, roll

    rot_x = np.concatenate(
        [ones, zeros, zeros, zeros, np.cos(x), -np.sin(x), zeros, np.sin(x), np.cos(x)],
        axis=1,
    ).reshape([bs, 3, 3])

    rot_y = np.concatenate(
        [np.cos(y), zeros, np.sin(y), zeros, ones, zeros, -np.sin(y), zeros, np.cos(y)],
        axis=1,
    ).reshape([bs, 3, 3])

    rot_z = np.concatenate(
        [np.cos(z), -np.sin(z), zeros, np.sin(z), np.cos(z), zeros, zeros, zeros, ones],
        axis=1,
    ).reshape([bs, 3, 3])

    rot = rot_z @ rot_y @ rot_x
    return rot.transpose(0, 2, 1)  # transpose


def transform_keypoint(kp_info: dict):
    """
    transform the implicit keypoints with the pose, shift, and expression deformation
    kp: BxNx3
    """
    kp = kp_info["kp"]  # (bs, k, 3)
    pitch, yaw, roll = kp_info["pitch"], kp_info["yaw"], kp_info["roll"]

    t, exp = kp_info["t"], kp_info["exp"]
    scale = kp_info["scale"]

    bs = kp.shape[0]
    num_kp = kp.shape[1]  # Bxnum_kpx3

    rot_mat = get_rotation_matrix(pitch, yaw, roll)  # (bs, 3, 3)

    # Eqn.2: s * (R * x_c,s + exp) + t
    kp_transformed = kp.reshape(bs, num_kp, 3) @ rot_mat + exp.reshape(bs, num_kp, 3)
    kp_transformed *= scale[..., None]  # (bs, k, 3) * (bs, 1, 1) = (bs, k, 3)
    kp_transformed[:, :, 0:2] += t[:, None, 0:2]  # remove z, only apply tx ty

    return kp_transformed


def prepare_paste_back(mask_crop, crop_M_c2o, dsize):
    """prepare mask for later image paste back"""
    mask_ori = cv2.warpAffine(
        mask_crop, crop_M_c2o[:2, :], dsize=dsize, flags=cv2.INTER_LINEAR
    )
    mask_ori = mask_ori.astype(np.float32) / 255.0
    return mask_ori


def paste_back(img_crop, M_c2o, img_ori, mask_ori):
    """paste back the image"""
    dsize = (img_ori.shape[1], img_ori.shape[0])
    result = cv2.warpAffine(img_crop, M_c2o[:2, :], dsize=dsize, flags=cv2.INTER_LINEAR)
    result = np.clip(mask_ori * result + (1 - mask_ori) * img_ori, 0, 255).astype(
        np.uint8
    )
    return result


def get_fps(video, default_fps=25):
    fps = cv2.VideoCapture(video).get(cv2.CAP_PROP_FPS)
    if fps in (0, None):
        fps = default_fps

    return fps


def concat_frame(driving_img, src_img, I_p):
    h, w, _ = I_p.shape

    src_img = cv2.resize(src_img, (w, h))
    driving_img = cv2.resize(driving_img, (w, h))
    out = np.hstack((driving_img, src_img, I_p))

    return out


# ======================
# Main functions
# ======================


def get_face_analysis(det_face, landmark):

    def get_landmark(img, face):
        input_size = 192

        bbox = face["bbox"]
        w, h = (bbox[2] - bbox[0]), (bbox[3] - bbox[1])
        center = (bbox[2] + bbox[0]) / 2, (bbox[3] + bbox[1]) / 2
        rotate = 0
        _scale = input_size / (max(w, h) * 1.5)
        aimg, M = face_align(img, center, input_size, _scale, rotate)
        input_size = tuple(aimg.shape[0:2][::-1])

        aimg = aimg.transpose(2, 0, 1)  # HWC -> CHW
        aimg = np.expand_dims(aimg, axis=0)
        aimg = aimg.astype(np.float32)

        # feedforward
        if not args.onnx:
            output = landmark.predict([aimg])
        else:
            output = landmark.run(None, {"data": aimg})
        pred = output[0][0]

        pred = pred.reshape((-1, 2))
        pred[:, 0:2] += 1
        pred[:, 0:2] *= input_size[0] // 2

        IM = cv2.invertAffineTransform(M)
        pred = trans_points2d(pred, IM)

        return pred

    def face_analysis(img):
        input_size = 512

        im_ratio = float(img.shape[0]) / img.shape[1]
        if im_ratio > 1:
            new_height = input_size
            new_width = int(new_height / im_ratio)
        else:
            new_width = input_size
            new_height = int(new_width * im_ratio)
        det_scale = float(new_height) / img.shape[0]
        resized_img = cv2.resize(img, (new_width, new_height))
        det_img = np.zeros((input_size, input_size, 3), dtype=np.uint8)
        det_img[:new_height, :new_width, :] = resized_img

        det_img = (det_img - 127.5) / 128
        det_img = det_img.transpose(2, 0, 1)  # HWC -> CHW
        det_img = np.expand_dims(det_img, axis=0)
        det_img = det_img.astype(np.float32)

        # feedforward
        if not args.onnx:
            output = det_face.predict([det_img])
        else:
            output = det_face.run(None, {"input.1": det_img})

        scores_list = []
        bboxes_list = []
        kpss_list = []

        det_thresh = 0.5
        fmc = 3
        feat_stride_fpn = [8, 16, 32]
        center_cache = {}
        for idx, stride in enumerate(feat_stride_fpn):
            scores = output[idx]
            bbox_preds = output[idx + fmc]
            bbox_preds = bbox_preds * stride
            kps_preds = output[idx + fmc * 2] * stride
            height = input_size // stride
            width = input_size // stride
            K = height * width
            key = (height, width, stride)
            if key in center_cache:
                anchor_centers = center_cache[key]
            else:
                anchor_centers = np.stack(
                    np.mgrid[:height, :width][::-1], axis=-1
                ).astype(np.float32)

                anchor_centers = (anchor_centers * stride).reshape((-1, 2))
                num_anchors = 2
                anchor_centers = np.stack(
                    [anchor_centers] * num_anchors, axis=1
                ).reshape((-1, 2))
                if len(center_cache) < 100:
                    center_cache[key] = anchor_centers

            pos_inds = np.where(scores >= det_thresh)[0]
            bboxes = distance2bbox(anchor_centers, bbox_preds)
            pos_scores = scores[pos_inds]
            pos_bboxes = bboxes[pos_inds]
            scores_list.append(pos_scores)
            bboxes_list.append(pos_bboxes)

            kpss = distance2kps(anchor_centers, kps_preds)
            kpss = kpss.reshape((kpss.shape[0], -1, 2))
            pos_kpss = kpss[pos_inds]
            kpss_list.append(pos_kpss)

        scores = np.vstack(scores_list)
        scores_ravel = scores.ravel()
        order = scores_ravel.argsort()[::-1]
        bboxes = np.vstack(bboxes_list) / det_scale
        kpss = np.vstack(kpss_list) / det_scale
        pre_det = np.hstack((bboxes, scores)).astype(np.float32, copy=False)
        pre_det = pre_det[order, :]

        nms_thresh = 0.4
        keep = nms_boxes(pre_det, [1 for s in pre_det], nms_thresh)
        bboxes = pre_det[keep, :]
        kpss = kpss[order, :, :]
        kpss = kpss[keep, :, :]

        if bboxes.shape[0] == 0:
            return []

        ret = []
        for i in range(bboxes.shape[0]):
            bbox = bboxes[i, 0:4]
            det_score = bboxes[i, 4]
            kps = None
            if kpss is not None:
                kps = kpss[i]
            face = dict(bbox=bbox, kps=kps, det_score=det_score)
            lmk = get_landmark(img, face)
            face["landmark_2d_106"] = lmk

            ret.append(face)

        src_face = sorted(
            ret,
            key=lambda face: (face["bbox"][2] - face["bbox"][0])
            * (face["bbox"][3] - face["bbox"][1]),
            reverse=True,
        )

        return src_face

    return face_analysis


def setup_facemesh(det_face, landmark):
    mod = load_facemesh_v2(args)

    def face_analysis(img):
        h, w = img.shape[:2]
        models = {
            "det_net": det_face,
            "net": landmark,
        }
        landmarks = mod.predict(models, img[:, :, ::-1])  # RGB -> BGR

        if len(landmarks) == 0:
            return []
        fm_lmk = landmarks[0]
        fm_lmk[:, 0] = fm_lmk[:, 0] * w
        fm_lmk[:, 1] = fm_lmk[:, 1] * h

        keypoint_map = {
            # lips
            52: (61,),
            64: (39, 40),
            63: (37,),
            71: (0,),
            67: (267,),
            68: (269, 270),
            61: (291,),
            65: (78,),
            66: (81,),
            62: (13,),
            70: (311,),
            69: (308,),
            54: (178,),
            60: (14,),
            57: (402,),
            55: (91,),
            56: (84, 181),
            53: (17,),
            59: (314, 405),
            58: (321,),
            # left eye
            89: (362,),
            95: (384, 385),
            94: (386,),
            96: (387, 388),
            93: (263,),
            88: (473,),
            92: (473,),
            90: (380, 381),
            87: (374,),
            91: (373, 390),
            81: (465,),
            # left eyebrow
            102: (336,),
            103: (296,),
            104: (334,),
            105: (293,),
            97: (285,),
            98: (295,),
            99: (282,),
            100: (283,),
            101: (276,),
            # right eye
            35: (33,),
            41: (160, 161),
            40: (159,),
            42: (157, 158),
            39: (133,),
            34: (468,),
            38: (468,),
            36: (163, 144),
            33: (145,),
            37: (153, 154),
            75: (245,),
            # right eyebrow
            48: (63,),
            49: (105,),
            51: (66,),
            50: (107,),
            43: (46,),
            44: (53,),
            45: (52,),
            47: (65,),
            46: (55,),
            # face oval
            1: (162,),
            9: (162, 127, 127),  # 1:2
            10: (127, 127, 234),  # 2:1
            11: (234,),
            12: (234, 93, 93),  # 1:2
            13: (93, 93, 132),  # 2:1
            14: (132,),
            15: (132, 58, 58),  # 1:2
            16: (58, 58, 172),  # 2:1
            2: (172,),
            3: (172, 136, 136),  # 1:2
            4: (136, 136, 150),  # 2:1
            5: (150,),
            6: (149,),
            7: (176,),
            8: (148,),
            0: (152,),
            24: (377,),
            23: (400,),
            22: (378,),
            21: (379,),
            20: (379, 365, 365),  # 1:2
            19: (365, 365, 397),  # 2:1
            18: (397,),
            32: (397, 288, 288),  # 1:2
            31: (288, 288, 361),  # 2:1
            30: (361,),
            29: (323, 323, 361),  # 2:1
            28: (323, 323, 454),  # 2:1
            27: (454,),
            26: (454, 356, 356),  # 1:2
            25: (356, 356, 389),  # 2:1
            17: (389,),
            # nose
            72: (168,),
            73: (6, 197, 197),  # 1:2
            74: (195, 195, 5),  # 2:1
            86: (4,),
            76: (209,),
            77: (64,),
            78: (64, 60, 60),  # 1:2
            79: (60, 60, 2),  # 2:1
            80: (2,),
            85: (2, 290, 290),  # 1:2
            84: (290, 290, 294),  # 2:1
            83: (294,),
            82: (429,),
        }
        lmk = np.zeros((106, 2))
        for i, kps in keypoint_map.items():
            x = y = 0
            for idx in kps:
                x += fm_lmk[idx][0]
                y += fm_lmk[idx][1]
            lmk[i] = x / len(kps), y / len(kps)

        face = dict(det_score=0)
        face["landmark_2d_106"] = lmk

        return [face]

    return face_analysis


def preprocess(img):
    img = img / 255.0
    img = np.clip(img, 0, 1)  # clip to 0~1
    img = img.transpose(2, 0, 1)  # HxWx3x1 -> 1x3xHxW
    img = np.expand_dims(img, axis=0)
    img = img.astype(np.float32)

    return img


def src_preprocess(img):
    h, w = img.shape[:2]

    # ajust the size of the image according to the maximum dimension
    max_dim = 1280
    if max(h, w) > max_dim:
        if h > w:
            new_h = max_dim
            new_w = int(w * (max_dim / h))
        else:
            new_w = max_dim
            new_h = int(h * (max_dim / w))
        img = cv2.resize(img, (new_w, new_h))

    # ensure that the image dimensions are multiples of n
    division = 2
    new_h = img.shape[0] - (img.shape[0] % division)
    new_w = img.shape[1] - (img.shape[1] % division)

    if new_h == 0 or new_w == 0:
        # when the width or height is less than n, no need to process
        return img

    if new_h != img.shape[0] or new_w != img.shape[1]:
        img = img[:new_h, :new_w]

    return img


def crop_src_image(models, img):
    face_analysis = models["face_analysis"]
    src_face = face_analysis(img)

    if len(src_face) == 0:
        logger.info("No face detected in the source image.")
        return None
    elif len(src_face) > 1:
        logger.info(f"More than one face detected in the image, only pick one face.")

    src_face = src_face[0]
    lmk = src_face["landmark_2d_106"]  # this is the 106 landmarks from insightface

    # crop the face
    crop_info = crop_image(img, lmk, dsize=512, scale=2.3, vy_ratio=-0.125)

    lmk = landmark_runner(models, img, lmk)

    crop_info["lmk_crop"] = lmk
    crop_info["img_crop_256x256"] = cv2.resize(
        crop_info["img_crop"], (256, 256), interpolation=cv2.INTER_AREA
    )
    crop_info["lmk_crop_256x256"] = crop_info["lmk_crop"] * 256 / 512

    return crop_info


def landmark_runner(models, img, lmk):
    crop_dct = crop_image(img, lmk, dsize=224, scale=1.5, vy_ratio=-0.1)
    img_crop = crop_dct["img_crop"]

    img_crop = img_crop / 255
    img_crop = img_crop.transpose(2, 0, 1)  # HWC -> CHW
    img_crop = np.expand_dims(img_crop, axis=0)
    img_crop = img_crop.astype(np.float32)

    # feedforward
    net = models["landmark_runner"]
    if not args.onnx:
        output = net.predict([img_crop])
    else:
        output = net.run(None, {"input": img_crop})
    out_pts = output[2]

    # 2d landmarks 203 points
    lmk = out_pts[0].reshape(-1, 2) * 224  # scale to 0-224
    # _transform_pts
    M = crop_dct["M_c2o"]
    lmk = lmk @ M[:2, :2].T + M[:2, 2]

    return lmk


def extract_feature_3d(models, x):
    net = models["appearance_feature_extractor"]

    # feedforward
    if not args.onnx:
        output = net.predict([x])
    else:
        output = net.run(None, {"x": x})
    f_s = output[0]
    f_s = f_s.astype(np.float32)

    return f_s


def get_kp_info(models, x):
    net = models["motion_extractor"]

    # feedforward
    if not args.onnx:
        output = net.predict([x])
    else:
        output = net.run(None, {"x": x})
    pitch, yaw, roll, t, exp, scale, kp = output

    kp_info = dict(pitch=pitch, yaw=yaw, roll=roll, t=t, exp=exp, scale=scale, kp=kp)

    pred = softmax(kp_info["pitch"], axis=1)
    degree = np.sum(pred * np.arange(66), axis=1) * 3 - 97.5
    kp_info["pitch"] = degree[:, None]  # Bx1
    pred = softmax(kp_info["yaw"], axis=1)
    degree = np.sum(pred * np.arange(66), axis=1) * 3 - 97.5
    kp_info["yaw"] = degree[:, None]  # Bx1
    pred = softmax(kp_info["roll"], axis=1)
    degree = np.sum(pred * np.arange(66), axis=1) * 3 - 97.5
    kp_info["roll"] = degree[:, None]  # Bx1

    kp_info = {k: v.astype(np.float32) for k, v in kp_info.items()}

    bs = kp_info["kp"].shape[0]
    kp_info["kp"] = kp_info["kp"].reshape(bs, -1, 3)  # BxNx3
    kp_info["exp"] = kp_info["exp"].reshape(bs, -1, 3)  # BxNx3

    return kp_info


def stitching(models, kp_source, kp_driving):
    """conduct the stitching
    kp_source: Bxnum_kpx3
    kp_driving: Bxnum_kpx3
    """

    bs, num_kp = kp_source.shape[:2]

    kp_driving_new = kp_driving

    bs_src = kp_source.shape[0]
    bs_dri = kp_driving.shape[0]
    feat = np.concatenate(
        [kp_source.reshape(bs_src, -1), kp_driving.reshape(bs_dri, -1)], axis=1
    )

    # feedforward
    net = models["stitching"]
    if not args.onnx:
        output = net.predict([feat])
    else:
        output = net.run(None, {"x": feat})
    delta = output[0]

    delta_exp = delta[..., : 3 * num_kp].reshape(bs, num_kp, 3)  # 1x20x3
    delta_tx_ty = delta[..., 3 * num_kp : 3 * num_kp + 2].reshape(bs, 1, 2)  # 1x1x2

    kp_driving_new += delta_exp
    kp_driving_new[..., :2] += delta_tx_ty

    return kp_driving_new


def warp_decode(models, feature_3d, kp_source, kp_driving):
    """get the image after the warping of the implicit keypoints
    feature_3d: Bx32x16x64x64, feature volume
    kp_source: BxNx3
    kp_driving: BxNx3
    """

    # feedforward
    net = models["warping_module"]
    if not args.onnx:
        output = net.predict([feature_3d, kp_source, kp_driving])
    else:
        output = net.run(
            None,
            {
                "feature_3d": feature_3d,
                "kp_source": kp_source,
                "kp_driving": kp_driving,
            },
        )
    out, occlusion_map, deformation = output
    out = out.astype(np.float32)

    # decode
    net = models["spade_generator"]
    if not args.onnx:
        output = net.predict([out])
    else:
        output = net.run(
            None,
            {
                "feature": out,
            },
        )
    out = output[0]

    ret_dct = {
        "out": out.astype(np.float32),
        "occlusion_map": occlusion_map.astype(np.float32),
        "deformation": deformation.astype(np.float32),
    }

    return ret_dct


def predict(models, x_s_info, R_s, f_s, x_s, img):
    # calc_lmks_from_cropped_video
    frame_0 = predict.lmk is None
    if frame_0:
        face_analysis = models["face_analysis"]
        src_face = face_analysis(img)
        if len(src_face) == 0:
            logger.info(f"No face detected in the frame")
            return None
        elif len(src_face) > 1:
            logger.info(
                f"More than one face detected in the driving frame, only pick one face."
            )
        src_face = src_face[0]
        lmk = src_face["landmark_2d_106"]
        lmk = landmark_runner(models, img, lmk)
    else:
        lmk = landmark_runner(models, img, predict.lmk)
    predict.lmk = lmk

    # calc_driving_ratio
    lmk = lmk[None]
    c_d_eyes = np.concatenate(
        [
            calculate_distance_ratio(lmk, 6, 18, 0, 12),
            calculate_distance_ratio(lmk, 30, 42, 24, 36),
        ],
        axis=1,
    )
    c_d_lip = calculate_distance_ratio(lmk, 90, 102, 48, 66)
    c_d_eyes = c_d_eyes.astype(np.float32)
    c_d_lip = c_d_lip.astype(np.float32)

    # prepare_driving_videos
    img = cv2.resize(img, (256, 256))
    I_d = preprocess(img)

    # collect s_d, R_d, δ_d and t_d for inference
    x_d_info = get_kp_info(models, I_d)
    R_d = get_rotation_matrix(x_d_info["pitch"], x_d_info["yaw"], x_d_info["roll"])
    x_d_info = {
        "scale": x_d_info["scale"].astype(np.float32),
        "R_d": R_d.astype(np.float32),
        "exp": x_d_info["exp"].astype(np.float32),
        "t": x_d_info["t"].astype(np.float32),
    }

    if frame_0:
        predict.x_d_0_info = x_d_info

    x_d_0_info = predict.x_d_0_info
    R_d_0 = x_d_0_info["R_d"]

    R_new = (R_d @ R_d_0.transpose(0, 2, 1)) @ R_s
    delta_new = x_s_info["exp"] + (x_d_info["exp"] - x_d_0_info["exp"])
    scale_new = x_s_info["scale"] * (x_d_info["scale"] / x_d_0_info["scale"])
    t_new = x_s_info["t"] + (x_d_info["t"] - x_d_0_info["t"])

    t_new[..., 2] = 0  # zero tz
    x_c_s = x_s_info["kp"]
    x_d_new = scale_new * (x_c_s @ R_new + delta_new) + t_new

    # with stitching and without retargeting
    x_d_new = stitching(models, x_s, x_d_new)

    out = warp_decode(models, f_s, x_s, x_d_new)
    out = out["out"]
    out = out.transpose(0, 2, 3, 1)  # 1x3xHxW -> 1xHxWx3
    out = np.clip(out, 0, 1)  # clip to 0~1
    out = (out * 255).astype(np.uint8)  # 0~1 -> 0~255
    I_p = out[0]

    return I_p


predict.lmk = None
predict.x_d_0_info = None


def warmup(models):
    img = np.zeros((512, 512, 3), dtype=np.uint8)
    x_s = np.random.randn(1, 21, 3).astype(np.float32)
    x_d = np.random.randn(1, 21, 3).astype(np.float32)
    f_s = np.random.randn(1, 32, 16, 64, 64).astype(np.float32)
    lmk = np.random.randn(106, 2).astype(np.float32)
    I_d = I_s = np.random.randn(1, 3, 256, 256).astype(np.float32)

    start_time = time.time()

    get_kp_info(models, I_s)
    extract_feature_3d(models, I_s)
    landmark_runner(models, img, lmk)
    get_kp_info(models, I_d)
    stitching(models, x_s, x_d)
    warp_decode(models, f_s, x_s, x_d)

    face_analysis = models["face_analysis"]
    face_analysis(img)

    elapse = time.time() - start_time
    logger.info(f"warmup time: {elapse:.3f}s")


def benchmark(models):
    logger.info("BENCHMARK mode")

    img = np.zeros((512, 512, 3), dtype=np.uint8)
    x_s = np.random.randn(1, 21, 3).astype(np.float32)
    x_d = np.random.randn(1, 21, 3).astype(np.float32)
    f_s = np.random.randn(1, 32, 16, 64, 64).astype(np.float32)
    lmk = np.random.randn(106, 2).astype(np.float32)
    I_d = np.random.randn(1, 3, 256, 256).astype(np.float32)

    total_time_estimation = 0
    for i in range(args.benchmark_count):
        start = int(round(time.time() * 1000))
        try:
            landmark_runner(models, img, lmk)
            get_kp_info(models, I_d)
            stitching(models, x_s, x_d)
            warp_decode(models, f_s, x_s, x_d)
        finally:
            end = int(round(time.time() * 1000))
            estimation_time = end - start

        # Loggin
        logger.info(f"\tailia processing estimation time {estimation_time} ms")
        if i != 0:
            total_time_estimation = total_time_estimation + estimation_time

    logger.info(
        f"\taverage time estimation {total_time_estimation / (args.benchmark_count - 1)} ms"
    )


def recognize_from_video(models):
    source_image = args.input[0]
    driving_video = args.driving
    flg_composite = args.composite

    logger.info("Source image: " + source_image)
    logger.info("Driving video: " + str(driving_video))
    logger.info("Start processing...")

    # prepare input data
    img = load_image(source_image)
    img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)

    mask_crop = load_image(MASK_PATH)
    mask_crop = cv2.cvtColor(mask_crop, cv2.COLOR_BGRA2BGR)

    img = img[:, :, ::-1]  # BGR -> RGB
    src_img = src_preprocess(img)
    crop_info = crop_src_image(models, src_img)

    if crop_info is None:
        raise Exception("No face detected in the source image!")

    # prepare_source
    img_crop_256x256 = crop_info["img_crop_256x256"]
    I_s = preprocess(img_crop_256x256)

    x_s_info = get_kp_info(models, I_s)
    R_s = get_rotation_matrix(x_s_info["pitch"], x_s_info["yaw"], x_s_info["roll"])
    f_s = extract_feature_3d(models, I_s)
    x_s = transform_keypoint(x_s_info)

    capture = get_capture(driving_video)
    assert capture.isOpened(), "Cannot capture source"
    # capture = imageio.get_reader(file_path, "ffmpeg")

    # create video writer if savepath is specified as video format
    if args.savepath:
        f_h, f_w = (512, 512 * 3) if flg_composite else src_img.shape[:2]
        output_fps = int(get_fps(driving_video))
        writer = get_writer(args.savepath, f_h, f_w, fps=output_fps)
    else:
        writer = None

    # prepare for pasteback
    mask_ori = prepare_paste_back(
        mask_crop, crop_info["M_c2o"], dsize=(src_img.shape[1], src_img.shape[0])
    )

    frame_shown = False
    frame_idx = 0
    while True:
        ret, frame = capture.read()
        if (cv2.waitKey(1) & 0xFF == ord("q")) or not ret:
            break
        frame_idx += 1
        if frame_shown and cv2.getWindowProperty("frame", cv2.WND_PROP_VISIBLE) == 0:
            break

        # inference
        img_rgb = frame[:, :, ::-1]  # BGR -> RGB
        I_p = predict(models, x_s_info, R_s, f_s, x_s, img_rgb)
        if I_p is None:
            print(f"No face detected in the driving frame {frame_idx}, skipped.")
            # keep showing / saving by falling back to the unanimated source crop
            I_p = crop_info["img_crop"]

        if flg_composite:
            driving_img = concat_frame(img_rgb, img_crop_256x256, I_p)
        else:
            driving_img = paste_back(I_p, crop_info["M_c2o"], src_img, mask_ori)
        driving_img = driving_img[:, :, ::-1]  # RGB -> BGR

        # show
        if not args.cui or args.video:
            cv2.imshow("frame", driving_img)
            frame_shown = True

        # save results
        if writer is not None:
            writer.write(driving_img)

    capture.release()
    cv2.destroyAllWindows()
    if writer is not None:
        writer.release()
        logger.info(f"Result was saved as {args.savepath}")

    logger.info("Script finished successfully.")


def main():
    # model files check and download
    check_and_download_models(WEIGHT_F_PATH, MODEL_F_PATH, REMOTE_PATH)
    check_and_download_models(WEIGHT_M_PATH, MODEL_M_PATH, REMOTE_PATH)
    check_and_download_models(WEIGHT_W_PATH, MODEL_W_PATH, REMOTE_PATH)
    check_and_download_models(WEIGHT_G_PATH, MODEL_G_PATH, REMOTE_PATH)
    check_and_download_models(WEIGHT_S_PATH, MODEL_S_PATH, REMOTE_PATH)
    check_and_download_models(WEIGHT_L_PATH, MODEL_L_PATH, REMOTE_PATH)
    if args.det == "insightface":
        check_and_download_models(WEIGHT_IF_DET_PATH, MODEL_IF_DET_PATH, REMOTE_PATH)
        check_and_download_models(WEIGHT_IF_LMK_PATH, MODEL_IF_LMK_PATH, REMOTE_PATH)
    else:  # facemesh
        check_and_download_models(
            WEIGHT_FM_DET_PATH, MODEL_FM_DET_PATH, REMOTE_FM_V2_PATH
        )
        check_and_download_models(
            WEIGHT_FM_LMK_PATH, MODEL_FM_LMK_PATH, REMOTE_FM_V2_PATH
        )

    env_id = args.env_id

    # initialize
    if not args.onnx:
        net_f = ailia.Net(MODEL_F_PATH, WEIGHT_F_PATH, env_id=env_id)
        net_m = ailia.Net(MODEL_M_PATH, WEIGHT_M_PATH, env_id=env_id)
        net_w = ailia.Net(MODEL_W_PATH, WEIGHT_W_PATH, env_id=env_id)
        net_g = ailia.Net(MODEL_G_PATH, WEIGHT_G_PATH, env_id=env_id)
        net_s = ailia.Net(MODEL_S_PATH, WEIGHT_S_PATH, env_id=env_id)
        net_l = ailia.Net(MODEL_L_PATH, WEIGHT_L_PATH, env_id=env_id)

        if args.det == "insightface":
            det_face = ailia.Net(MODEL_IF_DET_PATH, WEIGHT_IF_DET_PATH, env_id=env_id)
            landmark = ailia.Net(MODEL_IF_LMK_PATH, WEIGHT_IF_LMK_PATH, env_id=env_id)
        else:
            det_face = ailia.Net(MODEL_FM_DET_PATH, WEIGHT_FM_DET_PATH, env_id=env_id)
            landmark = ailia.Net(MODEL_FM_LMK_PATH, WEIGHT_FM_LMK_PATH, env_id=env_id)
    else:
        import onnxruntime

        onnxruntime.set_default_logger_severity(3)

        providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
        net_f = onnxruntime.InferenceSession(WEIGHT_F_PATH, providers=providers)
        net_m = onnxruntime.InferenceSession(WEIGHT_M_PATH, providers=providers)
        net_w = onnxruntime.InferenceSession(WEIGHT_W_PATH, providers=providers)
        net_g = onnxruntime.InferenceSession(WEIGHT_G_PATH, providers=providers)
        net_s = onnxruntime.InferenceSession(WEIGHT_S_PATH, providers=providers)
        net_l = onnxruntime.InferenceSession(WEIGHT_L_PATH, providers=providers)

        if args.det == "insightface":
            det_face = onnxruntime.InferenceSession(
                WEIGHT_IF_DET_PATH, providers=providers
            )
            landmark = onnxruntime.InferenceSession(
                WEIGHT_IF_LMK_PATH, providers=providers
            )
        else:
            det_face = onnxruntime.InferenceSession(
                WEIGHT_FM_DET_PATH, providers=providers
            )
            landmark = onnxruntime.InferenceSession(
                WEIGHT_FM_LMK_PATH, providers=providers
            )

    if args.det == "facemesh":
        face_analysis = setup_facemesh(det_face, landmark)
    else:
        face_analysis = get_face_analysis(det_face, landmark)

    models = {
        "appearance_feature_extractor": net_f,
        "motion_extractor": net_m,
        "warping_module": net_w,
        "spade_generator": net_g,
        "stitching": net_s,
        "landmark_runner": net_l,
        "face_analysis": face_analysis,
    }

    warmup(models)

    if args.benchmark:
        benchmark(models)
    else:
        recognize_from_video(models)


if __name__ == "__main__":
    main()
