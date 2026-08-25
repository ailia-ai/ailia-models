import os
import sys
import time

# logger
from logging import getLogger  # noqa: E402

import ailia
import cv2
import numpy as np

sys.path.append("../../util")
from arg_utils import get_base_parser, get_savepath, update_parser  # noqa: E402
from model_utils import check_and_download_models  # noqa: E402

logger = getLogger(__name__)

# ======================
# Parameters
# ======================

WEIGHT_VITDET = "vitdet.onnx"
WEIGHT_VITDET_PB = "vitdet_weights.pb"
WEIGHT_MOGE = "moge.onnx"

WEIGHT_BACKBONE_VITH = "backbone_vith.onnx"
WEIGHT_BACKBONE_VITH_PB = "backbone_vith_weights.pb"
WEIGHT_BODY_INIT_VITH = "body_decoder_init_vith.onnx"

WEIGHT_BACKBONE_DINOV3 = "backbone_dinov3.onnx"
WEIGHT_BACKBONE_DINOV3_PB = "backbone_dinov3_weights.pb"
WEIGHT_BODY_INIT_DINOV3 = "body_decoder_init_dinov3.onnx"

MODEL_VITDET = WEIGHT_VITDET + ".prototxt"
MODEL_MOGE = WEIGHT_MOGE + ".prototxt"
MODEL_BACKBONE_VITH = WEIGHT_BACKBONE_VITH + ".prototxt"
MODEL_BODY_INIT_VITH = WEIGHT_BODY_INIT_VITH + ".prototxt"
MODEL_BACKBONE_DINOV3 = WEIGHT_BACKBONE_DINOV3 + ".prototxt"
MODEL_BODY_INIT_DINOV3 = WEIGHT_BODY_INIT_DINOV3 + ".prototxt"
REMOTE_PATH = "https://storage.googleapis.com/ailia-models/sam-3d-body/"

# Model input size (W, H) - cfg.MODEL.IMAGE_SIZE
INPUT_SIZE = (384, 512)
# Detector input: ResizeShortestEdge(short=1024, max=1024)
DET_SIZE = 1024
# MHR mesh topology
NUM_VERTICES = 18439

IMAGE_PATH = "dancing.jpg"
SAVE_IMAGE_PATH = "output.png"

# ======================
# Arguments
# ======================

parser = get_base_parser("SAM 3D Body", IMAGE_PATH, SAVE_IMAGE_PATH)
parser.add_argument(
    "-a",
    "--arch",
    default="vith",
    choices=("vith", "dinov3"),
    help="backbone architecture",
)
parser.add_argument(
    "--bbox_thresh", type=float, default=0.8, help="person detection threshold"
)
parser.add_argument(
    "--no_fov",
    action="store_true",
    help="skip the MoGe FOV estimator and use the default focal length",
)
parser.add_argument(
    "--save_ply", action="store_true", help="save the recovered mesh as a .ply file"
)
parser.add_argument("--onnx", action="store_true", help="execute onnxruntime version.")
args = update_parser(parser)


# ======================
# Visualization
# ======================

# mhr70 skeleton, used only for the 2D overlay.
# Index pairs derived from sam_3d_body/metadata/mhr70.py `pose_info["skeleton_info"]`
# (65 links: body + feet + both hands; keypoints 21-41 are the right hand and
# 42-62 the left hand, which is why the wrists 41/62 fan out into five fingers each).
SKELETON = [
    (13, 11),
    (11, 9),
    (14, 12),
    (12, 10),
    (9, 10),
    (5, 9),
    (6, 10),
    (5, 6),
    (5, 7),
    (6, 8),
    (7, 62),
    (8, 41),
    (1, 2),
    (0, 1),
    (0, 2),
    (1, 3),
    (2, 4),
    (3, 5),
    (4, 6),
    (13, 15),
    (13, 16),
    (13, 17),
    (14, 18),
    (14, 19),
    (14, 20),
    (62, 45),
    (45, 44),
    (44, 43),
    (43, 42),
    (62, 49),
    (49, 48),
    (48, 47),
    (47, 46),
    (62, 53),
    (53, 52),
    (52, 51),
    (51, 50),
    (62, 57),
    (57, 56),
    (56, 55),
    (55, 54),
    (62, 61),
    (61, 60),
    (60, 59),
    (59, 58),
    (41, 24),
    (24, 23),
    (23, 22),
    (22, 21),
    (41, 28),
    (28, 27),
    (27, 26),
    (26, 25),
    (41, 32),
    (32, 31),
    (31, 30),
    (30, 29),
    (41, 36),
    (36, 35),
    (35, 34),
    (34, 33),
    (41, 40),
    (40, 39),
    (39, 38),
    (38, 37),
]


def draw_results(img_bgr, results):
    vis = img_bgr.copy()
    for person in results:
        x1, y1, x2, y2 = person["bbox"].astype(int)
        cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 255, 0), 2)

        kps = person["pred_keypoints_2d"]
        for x, y in kps.astype(int):
            cv2.circle(vis, (int(x), int(y)), 3, (0, 200, 255), -1)
        for a, b in SKELETON:
            if a < len(kps) and b < len(kps):
                pa = tuple(kps[a].astype(int))
                pb = tuple(kps[b].astype(int))
                cv2.line(vis, pa, pb, (255, 128, 0), 2)
    return vis


def save_ply(path, vertices, faces=None):
    """Minimal ASCII PLY writer (vertices only if faces are unavailable)."""
    with open(path, "w") as f:
        f.write("ply\nformat ascii 1.0\n")
        f.write(f"element vertex {len(vertices)}\n")
        f.write("property float x\nproperty float y\nproperty float z\n")
        if faces is not None:
            f.write(f"element face {len(faces)}\n")
            f.write("property list uchar int vertex_index\n")
        f.write("end_header\n")
        for v in vertices:
            f.write(f"{v[0]:.6f} {v[1]:.6f} {v[2]:.6f}\n")
        if faces is not None:
            for tri in faces:
                f.write(f"3 {tri[0]} {tri[1]} {tri[2]}\n")


# ======================
# Secondary Functions
# ======================


def bbox_xyxy2cs(bbox, padding=1.0):
    """(x1, y1, x2, y2) -> (center, scale). Port of
    sam_3d_body/data/transforms/bbox_utils.py:bbox_xyxy2cs
    """
    x1, y1, x2, y2 = np.hsplit(bbox.reshape(-1, 4), [1, 2, 3])
    center = np.hstack([x1 + x2, y1 + y2]) * 0.5
    scale = np.hstack([x2 - x1, y2 - y1]) * padding
    return center, scale


def fix_aspect_ratio(bbox_scale, aspect_ratio):
    """Reshape the bbox to a fixed w/h ratio. Port of bbox_utils.py:fix_aspect_ratio"""
    w, h = np.hsplit(bbox_scale.reshape(-1, 2), [1])
    return np.where(
        w > h * aspect_ratio,
        np.hstack([w, w / aspect_ratio]),
        np.hstack([h * aspect_ratio, h]),
    )


def _rotate_point(pt, angle_rad):
    sn, cs = np.sin(angle_rad), np.cos(angle_rad)
    return np.array([pt[0] * cs - pt[1] * sn, pt[0] * sn + pt[1] * cs])


def _get_3rd_point(a, b):
    direction = a - b
    return b + np.r_[-direction[1], direction[0]]


def get_warp_matrix(center, scale, rot, output_size):
    """Port of bbox_utils.py:get_warp_matrix (shift=(0,0), inv=False)."""
    src_w = scale[0]
    dst_w, dst_h = output_size

    rot_rad = np.deg2rad(rot)
    src_dir = _rotate_point(np.array([0.0, src_w * -0.5]), rot_rad)
    dst_dir = np.array([0.0, dst_w * -0.5])

    src = np.zeros((3, 2), dtype=np.float32)
    src[0, :] = center
    src[1, :] = center + src_dir
    src[2, :] = _get_3rd_point(src[0, :], src[1, :])

    dst = np.zeros((3, 2), dtype=np.float32)
    dst[0, :] = [dst_w * 0.5, dst_h * 0.5]
    dst[1, :] = np.array([dst_w * 0.5, dst_h * 0.5]) + dst_dir
    dst[2, :] = _get_3rd_point(dst[0, :], dst[1, :])

    return cv2.getAffineTransform(np.float32(src), np.float32(dst))


def default_cam_int(height, width):
    """prepare_batch's fallback intrinsics when no FOV estimator is used."""
    f = (height**2 + width**2) ** 0.5
    return np.array(
        [[[f, 0, width / 2.0], [0, f, height / 2.0], [0, 0, 1]]], dtype=np.float32
    )


def get_ray_condition(batch):
    """Per-pixel camera rays for the crop, used by the decoder's CameraEncoder.

    Port of SAM3DBody.get_ray_condition (sam_3d_body/models/meta_arch/sam3d_body.py).
    Returns (N, 2, H, W) already flattened over the person axis and cropped to the
    ViT aspect ratio (the model slices [:, :, :, 32:-32] for 512x384 input).
    """
    n_person = batch["img"].shape[0]
    _, _, crop_h, crop_w = batch["img"].shape

    grid = np.stack(
        np.meshgrid(np.arange(crop_w), np.arange(crop_h), indexing="xy"), axis=2
    )  # (H, W, 2) -- 'xy' gives [x, y] on the last axis
    grid = np.tile(grid[None], (n_person, 1, 1, 1)).astype(np.float32)

    affine = batch["affine_trans"][0]  # (N, 2, 3)
    scale = affine[:, [0, 1], [0, 1]][:, None, None, :]  # (N,1,1,2) diagonal
    offset = affine[:, [0, 1], [2, 2]][:, None, None, :]  # (N,1,1,2) translation
    grid = grid / scale
    grid = grid - offset / scale

    cam_int = batch["cam_int"]  # (1, 3, 3)
    center = cam_int[:, [0, 1], [2, 2]][:, None, None, :]  # (1,1,1,2)
    focal = cam_int[:, [0, 1], [0, 1]][:, None, None, :]
    grid = (grid - center) / focal

    ray = grid.transpose(0, 3, 1, 2)  # (N, 2, H, W)
    return ray  # decoder は (N, 2, H, W) の 4 次元を取る


def get_condition_info(batch):
    """CLIFF-style condition (cx/f, cy/f, b/f).

    Port of SAM3DBody._get_decoder_condition with CONDITION_TYPE='cliff'
    and USE_INTRIN_CENTER=True (the released configs set it).
    """
    center = batch["bbox_center"][0]  # (N, 2)
    b = batch["bbox_scale"][0][:, [0]]  # (N, 1)
    cam_int = batch["cam_int"]  # (1, 3, 3)
    focal = cam_int[0, 0, 0]
    full_cxy = cam_int[0, [0, 1], [2, 2]][None]  # (1, 2)

    cond = np.concatenate(
        [center[:, [0]] - full_cxy[:, [0]], center[:, [1]] - full_cxy[:, [1]], b],
        axis=-1,
    ).astype(np.float32)
    cond[:, :2] /= focal
    cond[:, 2] /= focal
    return cond


def dummy_keypoint_prompt(n_person):
    """The initial pass feeds a single invalid ('label == -2') prompt."""
    kp = np.zeros((n_person, 1, 3), dtype=np.float32)
    kp[:, :, -1] = -2
    return kp


def preprocess(img, boxes):
    """Crop each person and build the tensors the decoder needs.

    Port of the `Compose([GetBBoxCenterScale, TopdownAffine, ToTensor])` pipeline
    (sam_3d_body/data/transforms/common.py) plus `prepare_batch`
    (sam_3d_body/data/utils/prepare_batch.py).

    img   : (H, W, 3) RGB
    boxes : (N, 4) xyxy in original-image coordinates
    """
    height, width = img.shape[:2]
    w, h = INPUT_SIZE

    crops, centers, scales, affines = [], [], [], []
    for box in boxes:
        center, scale = bbox_xyxy2cs(box, padding=1.25)
        center, scale = center[0], scale[0]

        # TopdownAffine: expand to the prior 0.75 ratio, then to the model input ratio
        scale = fix_aspect_ratio(scale, aspect_ratio=0.75)[0]
        scale = fix_aspect_ratio(scale, aspect_ratio=w / h)[0]

        warp_mat = get_warp_matrix(center, scale, 0.0, (w, h))
        crop = cv2.warpAffine(img, warp_mat, (int(w), int(h)), flags=cv2.INTER_LINEAR)

        crops.append(crop)
        centers.append(center)
        scales.append(scale)
        affines.append(warp_mat)

    # ToTensor(): HWC uint8 [0,255] -> CHW float32 [0,1]
    crops = np.stack(crops).astype(np.float32).transpose(0, 3, 1, 2) / 255.0

    batch = {
        "img": crops,  # (N, 3, H, W)
        "bbox_center": np.stack(centers)[None].astype(np.float32),  # (1, N, 2)
        "bbox_scale": np.stack(scales)[None].astype(np.float32),  # (1, N, 2)
        "affine_trans": np.stack(affines)[None].astype(np.float32),  # (1, N, 2, 3)
        "img_size": np.tile(
            np.array([w, h], dtype=np.float32), (1, len(boxes), 1)
        ),  # (1, N, 2) model input size
        "ori_img_size": np.tile(
            np.array([width, height], dtype=np.float32), (1, len(boxes), 1)
        ),  # (1, N, 2)
    }
    return batch


# ======================
# Model-running helpers
# ======================


def run_net(net, feed):
    """Run ailia.Net / onnxruntime.InferenceSession with a name->array dict."""
    if args.onnx:
        names = {i.name for i in net.get_inputs()}
        return net.run(None, {k: v for k, v in feed.items() if k in names})
    return net.run(list(feed.values()))


def detect_persons(det_net, img_bgr):
    """ViTDet person detection.

    Port of tools/build_detector.py:run_detectron2_vitdet. The model covers the
    network; the class filter / score threshold / sort stay here, exactly as in
    the original (they are numpy operations outside the model).
    """
    height, width = img_bgr.shape[:2]

    scale = DET_SIZE / min(height, width)
    if round(max(height, width) * scale) > DET_SIZE:
        scale = DET_SIZE / max(height, width)
    new_h, new_w = round(height * scale), round(width * scale)
    resized = cv2.resize(img_bgr, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    inp = resized.astype(np.float32).transpose(2, 0, 1)

    outputs = run_net(det_net, {"image": inp})
    pred_boxes, pred_classes, _pred_masks, scores, _image_size = outputs

    valid = (pred_classes == 0) & (scores > args.bbox_thresh)
    if valid.sum() == 0:
        return np.zeros((0, 4), dtype=np.float32)

    boxes = pred_boxes[valid]
    # boxes are in the resized frame -> back to the original image
    boxes = boxes / scale

    order = np.lexsort((boxes[:, 3], boxes[:, 2], boxes[:, 1], boxes[:, 0]))
    return boxes[order].astype(np.float32)


def estimate_cam_int(moge_net, img_rgb):
    """Camera intrinsics from MoGe.

    The model stops at `MoGeModel.forward` (points/mask); `infer()`'s
    `recover_focal_shift` is a scipy least-squares fit that cannot be traced, so
    it is reproduced here in numpy - the same split the sam-3d-objects sample uses.
    """
    height, width = img_rgb.shape[:2]
    inp = (img_rgb.astype(np.float32) / 255.0).transpose(2, 0, 1)[None]

    points, mask = run_net(moge_net, {"image": inp})
    points, mask = points[0], mask[0] > 0.5

    focal = recover_focal(points, mask)
    # MoGe returns focal relative to half the image diagonal in normalized space;
    # build_fov_estimator.py:denormalize_f scales it into pixels and overrides
    # the horizontal focal with the vertical one.
    fy = focal * height
    return np.array(
        [[[fy, 0, width / 2.0], [0, fy, height / 2.0], [0, 0, 1]]], dtype=np.float32
    )


def recover_focal(points, mask):
    """Least-squares focal recovery from a camera-space point map.

    Reproduces moge.utils.geometry_numpy.recover_focal_shift for the FOV-only use:
    we need the focal, not the shift, so a direct 1-D search over focal is enough.
    """
    valid = mask & np.isfinite(points).all(axis=-1)
    if valid.sum() < 16:
        logger.warning("MoGe mask is (almost) empty; falling back to a default FOV")
        return 1.0

    h, w = points.shape[:2]
    uv = np.stack(
        np.meshgrid(
            (np.arange(w) + 0.5) / w * 2 - 1,
            (np.arange(h) + 0.5) / h * 2 - 1,
            indexing="xy",
        ),
        axis=-1,
    )
    xy = points[valid][:, :2]
    z = points[valid][:, 2]
    uv = uv[valid]

    # x = u * z / f  ->  f = median(u * z / x) over valid pixels
    with np.errstate(divide="ignore", invalid="ignore"):
        fx = np.nanmedian(uv[:, 0] * z / xy[:, 0])
        fy = np.nanmedian(uv[:, 1] * z / xy[:, 1])
    focal = np.nanmean([fx, fy])
    if not np.isfinite(focal) or focal <= 0:
        return 1.0
    return float(focal) * 0.5  # normalized (half-diagonal) -> per-axis


# ======================
# Main inference
# ======================


def predict(models, img_bgr):
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    height, width = img_bgr.shape[:2]

    logger.info("Running person detector...")
    boxes = detect_persons(models["detector"], img_bgr)
    logger.info(f"  detected {len(boxes)} person(s)")
    if len(boxes) == 0:
        return []

    if models["moge"] is not None:
        logger.info("Running FOV estimator...")
        cam_int = estimate_cam_int(models["moge"], img_rgb)
    else:
        cam_int = default_cam_int(height, width)
    logger.info(f"  focal length: {cam_int[0, 0, 0]:.1f}")

    batch = preprocess(img_rgb, boxes)
    batch["cam_int"] = cam_int

    logger.info("Running backbone...")
    (image_embeddings,) = run_net(models["backbone"], {"crop_image": batch["img"]})

    logger.info("Running body decoder...")
    n_person = len(boxes)
    feed = {
        "image_embeddings": image_embeddings,
        "keypoints": dummy_keypoint_prompt(n_person),
        "condition_info": get_condition_info(batch),
        "ray_cond": get_ray_condition(batch),
        "bbox_center": batch["bbox_center"],
        "bbox_scale": batch["bbox_scale"],
        "ori_img_size": batch["ori_img_size"],
        "cam_int": batch["cam_int"],
        "affine_trans": batch["affine_trans"],
        "img_size": batch["img_size"],
        "mask": np.zeros((1, n_person, 1, *INPUT_SIZE[::-1]), dtype=np.float32),
        "mask_score": np.zeros((1, n_person), dtype=np.float32),
    }
    outputs = run_net(models["body_decoder"], feed)

    (
        pred_vertices,
        pred_keypoints_3d,
        pred_keypoints_2d,
        _pred_keypoints_2d_cropped,
        pred_cam_t,
        _pred_cam,
        focal_length,
        _pred_pose_raw,
        _global_rot,
        _body_pose,
        _shape,
        _scale,
        _hand,
        _face,
        _pred_joint_coords,
        _joint_global_rots,
        _mhr_model_params,
        _hand_coords,
        _hand_logits,
    ) = outputs

    results = []
    for i in range(n_person):
        results.append(
            {
                "bbox": boxes[i],
                "pred_vertices": pred_vertices[i],
                "pred_keypoints_3d": pred_keypoints_3d[i],
                "pred_keypoints_2d": pred_keypoints_2d[i],
                "pred_cam_t": pred_cam_t[i],
                "focal_length": float(focal_length[i]),
            }
        )
    return results


def recognize_from_image(models):
    for image_path in args.input:
        logger.info(image_path)
        img_bgr = cv2.imread(image_path)
        if img_bgr is None:
            logger.error(f"cannot read {image_path}")
            continue

        if args.benchmark:
            logger.info("BENCHMARK mode")
            total_time_estimation = 0
            for i in range(args.benchmark_count):
                start = int(round(time.time() * 1000))
                results = predict(models, img_bgr)
                end = int(round(time.time() * 1000))
                logger.info(f"\tailia processing estimation time {end - start} ms")
                if i != 0:
                    total_time_estimation += end - start
            logger.info(
                f"\taverage time estimation {total_time_estimation / (args.benchmark_count - 1)} ms"
            )
        else:
            results = predict(models, img_bgr)

        if not results:
            logger.info("no person detected")
            continue
        for pid, person in enumerate(results):
            logger.info(
                f"  person {pid}: cam_t={person['pred_cam_t']}, "
                f"focal={person['focal_length']:.1f}, "
                f"vertices={person['pred_vertices'].shape}"
            )

        savepath = get_savepath(args.savepath, image_path, ext=".png")
        cv2.imwrite(savepath, draw_results(img_bgr, results))
        logger.info(f"saved at : {savepath}")

        if args.save_ply:
            for pid, person in enumerate(results):
                ply_path = os.path.splitext(savepath)[0] + f"_mesh_{pid:03d}.ply"
                save_ply(ply_path, person["pred_vertices"] + person["pred_cam_t"])
                logger.info(f"saved at : {ply_path}")

    logger.info("Script finished successfully.")


def main():
    arch = args.arch
    weight_backbone, model_backbone, pb_backbone = {
        "vith": (WEIGHT_BACKBONE_VITH, MODEL_BACKBONE_VITH, WEIGHT_BACKBONE_VITH_PB),
        "dinov3": (
            WEIGHT_BACKBONE_DINOV3,
            MODEL_BACKBONE_DINOV3,
            WEIGHT_BACKBONE_DINOV3_PB,
        ),
    }[arch]
    weight_body, model_body = {
        "vith": (WEIGHT_BODY_INIT_VITH, MODEL_BODY_INIT_VITH),
        "dinov3": (WEIGHT_BODY_INIT_DINOV3, MODEL_BODY_INIT_DINOV3),
    }[arch]

    check_and_download_models(WEIGHT_VITDET, MODEL_VITDET, REMOTE_PATH)
    check_and_download_models(weight_backbone, model_backbone, REMOTE_PATH)
    check_and_download_models(weight_body, model_body, REMOTE_PATH)
    if not args.no_fov:
        check_and_download_models(WEIGHT_MOGE, MODEL_MOGE, REMOTE_PATH)

    env_id = args.env_id

    if args.onnx:
        import onnxruntime

        providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
        detector = onnxruntime.InferenceSession(WEIGHT_VITDET, providers=providers)
        backbone = onnxruntime.InferenceSession(weight_backbone, providers=providers)
        body_decoder = onnxruntime.InferenceSession(weight_body, providers=providers)
        moge = (
            None
            if args.no_fov
            else onnxruntime.InferenceSession(WEIGHT_MOGE, providers=providers)
        )
    else:
        memory_mode = ailia.get_memory_mode(
            reduce_constant=True,
            ignore_input_with_initializer=True,
            reduce_interstage=False,
            reuse_interstage=True,
        )
        detector = ailia.Net(
            MODEL_VITDET, WEIGHT_VITDET, env_id=env_id, memory_mode=memory_mode
        )
        backbone = ailia.Net(
            model_backbone, weight_backbone, env_id=env_id, memory_mode=memory_mode
        )
        body_decoder = ailia.Net(
            model_body, weight_body, env_id=env_id, memory_mode=memory_mode
        )
        moge = (
            None
            if args.no_fov
            else ailia.Net(
                MODEL_MOGE, WEIGHT_MOGE, env_id=env_id, memory_mode=memory_mode
            )
        )

    models = {
        "detector": detector,
        "moge": moge,
        "backbone": backbone,
        "body_decoder": body_decoder,
    }

    recognize_from_image(models)


if __name__ == "__main__":
    main()
