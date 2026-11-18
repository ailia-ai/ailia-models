import os
import sys
import time
from functools import partial

# logger
from logging import getLogger  # noqa: E402

import ailia
import cv2
import numpy as np
from PIL import Image
from scipy.optimize import least_squares

sys.path.append("../../util")
from arg_utils import get_base_parser, get_savepath, update_parser  # noqa: E402
from model_utils import check_and_download_file, check_and_download_models  # noqa: E402

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

# MHR mesh topology (character_torch.mesh.faces), shared by both variants.
MHR_FACES = "mhr_faces.npy"

# prompt_encoder.no_mask_embed.weight, saved out of the checkpoint. It is added to
# the backbone output outside the decoder graph (see predict()), so it cannot live
# inside one of the models.
NO_MASK_EMBED_VITH = "no_mask_embed_vith.npy"
NO_MASK_EMBED_DINOV3 = "no_mask_embed_dinov3.npy"

MODEL_VITDET = WEIGHT_VITDET + ".prototxt"
MODEL_MOGE = WEIGHT_MOGE + ".prototxt"
MODEL_BACKBONE_VITH = WEIGHT_BACKBONE_VITH + ".prototxt"
MODEL_BODY_INIT_VITH = WEIGHT_BODY_INIT_VITH + ".prototxt"
MODEL_BACKBONE_DINOV3 = WEIGHT_BACKBONE_DINOV3 + ".prototxt"
MODEL_BODY_INIT_DINOV3 = WEIGHT_BODY_INIT_DINOV3 + ".prototxt"
REMOTE_PATH = "https://storage.googleapis.com/ailia-models/sam-3d-body/"

# Crop size (W, H) - cfg.MODEL.IMAGE_SIZE = [512, 512]
INPUT_SIZE = (512, 512)
# data_preprocess() / forward_pose_branch slice the width only for the ViT
# backbones. vith ("vit_hmr_512_384") -> 512x384, dinov3 -> full 512x512.
CROP_WIDTH = {"vith": 64, "dinov3": 0}
# cfg.MODEL.IMAGE_MEAN / IMAGE_STD
IMAGE_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(1, 3, 1, 1)
IMAGE_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(1, 3, 1, 1)
# Detector input: ResizeShortestEdge(short=1024, max=1024)
DET_SIZE = 1024

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
    "--skip_ply",
    action="store_true",
    help="do not save the recovered mesh as a .ply file",
)
parser.add_argument("--onnx", action="store_true", help="execute onnxruntime version.")
args = update_parser(parser)


# ======================
# Visualization
# ======================

# mhr70 skeleton for the 2D overlay, from
# sam_3d_body/metadata/mhr70.py `pose_info["skeleton_info"]` (body + feet + hands).
# fmt: off
SKELETON = [
    (13, 11), (11, 9), (14, 12), (12, 10), (9, 10), (5, 9), (6, 10), (5, 6),
    (5, 7), (6, 8), (7, 62), (8, 41), (1, 2), (0, 1), (0, 2), (1, 3),
    (2, 4), (3, 5), (4, 6), (13, 15), (13, 16), (13, 17), (14, 18), (14, 19),
    (14, 20), (62, 45), (45, 44), (44, 43), (43, 42), (62, 49), (49, 48), (48, 47),
    (47, 46), (62, 53), (53, 52), (52, 51), (51, 50), (62, 57), (57, 56), (56, 55),
    (55, 54), (62, 61), (61, 60), (60, 59), (59, 58), (41, 24), (24, 23), (23, 22),
    (22, 21), (41, 28), (28, 27), (27, 26), (26, 25), (41, 32), (32, 31), (31, 30),
    (30, 29), (41, 36), (36, 35), (35, 34), (34, 33), (41, 40), (40, 39), (39, 38),
    (38, 37),
]
# fmt: on


MESH_COLOR = (219, 189, 166)  # LIGHT_BLUE of tools/vis_utils.py, in BGR


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


def project(points_3d, cam_t, focal, height, width):
    """Project camera-space points with the model's own pinhole (cx, cy = image center)."""
    p = points_3d + cam_t
    uv = p[:, :2] * focal / p[:, [2]]
    uv[:, 0] += width / 2.0
    uv[:, 1] += height / 2.0
    return uv


def rasterize_mesh(canvas, uv, verts, faces, color, alpha):
    """Paint the mesh with the painter's algorithm (far faces first).

    cli.py renders this with pyrender; here it is flat-shaded with cv2 so the
    sample needs no OpenGL.
    """
    tri = verts[faces]  # (F, 3, 3)
    tri_uv = uv[faces]  # (F, 3, 2)
    tri_z = tri[:, :, 2].mean(1)  # (F,)
    # Lambert shading from a light behind the camera, using the face normal
    nz = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    nz /= np.linalg.norm(nz, axis=1, keepdims=True) + 1e-8
    shade = np.clip(0.35 + 0.65 * np.abs(nz[:, 2]), 0, 1)

    layer = canvas.copy()
    for i in np.argsort(-tri_z):  # far -> near
        c = tuple(int(v * shade[i]) for v in color)
        cv2.fillConvexPoly(layer, tri_uv[i].astype(np.int32), c, lineType=cv2.LINE_AA)
    return cv2.addWeighted(layer, alpha, canvas, 1 - alpha, 0)


def draw_mesh(img_bgr, results, faces, color=MESH_COLOR, alpha=0.9):
    """Overlay the recovered mesh on the input, one person at a time."""
    height, width = img_bgr.shape[:2]
    vis = img_bgr.copy()
    for person in results:
        uv = project(
            person["pred_vertices"],
            person["pred_cam_t"],
            person["focal_length"],
            height,
            width,
        )
        verts = person["pred_vertices"] + person["pred_cam_t"]
        vis = rasterize_mesh(vis, uv, verts, faces, color, alpha)
    return vis


def draw_mesh_side(img_bgr, results, faces, color=MESH_COLOR, rot_deg=90):
    """The same mesh seen from the side, on a white background. Port of the 4th
    panel of tools/vis_utils.py:visualize_sample_together: everyone is merged
    into a single mesh, recentered on the closest two people and turned around
    the vertical axis before being projected with the same camera.
    """
    height, width = img_bgr.shape[:2]
    order = np.argsort([-p["pred_cam_t"][2] for p in results])  # far -> near
    verts = np.concatenate(
        [results[i]["pred_vertices"] + results[i]["pred_cam_t"] for i in order]
    )
    n_vert = len(results[0]["pred_vertices"])
    faces = np.concatenate([faces + n_vert * pid for pid in range(len(results))])

    near = verts[-2 * n_vert :]
    cam_t = (near.max(axis=0) + near.min(axis=0)) / 2

    rad = np.radians(rot_deg)
    rot = np.array(
        [
            [np.cos(rad), 0, np.sin(rad)],
            [0, 1, 0],
            [-np.sin(rad), 0, np.cos(rad)],
        ]
    )
    verts = (verts - cam_t) @ rot.T

    focal = results[order[-1]]["focal_length"]
    uv = project(verts, cam_t, focal, height, width)
    canvas = np.full_like(img_bgr, 255)
    return rasterize_mesh(canvas, uv, verts + cam_t, faces, color, 1.0)


def save_ply(path, vertices, faces, color=MESH_COLOR):
    """Minimal ASCII PLY writer.

    The vertices are turned 180 deg around X first, as cli.py does in
    sam_3d_body/visualization/renderer.py:vertices_to_trimesh, so the mesh comes
    out upright in a viewer instead of upside down in the camera frame.
    """
    vertices = vertices * (1, -1, -1)
    b, g, r = color
    with open(path, "w") as f:
        f.write("ply\nformat ascii 1.0\n")
        f.write(f"element vertex {len(vertices)}\n")
        f.write("property float x\nproperty float y\nproperty float z\n")
        f.write("property uchar red\nproperty uchar green\nproperty uchar blue\n")
        f.write(f"element face {len(faces)}\n")
        f.write("property list uchar int vertex_index\n")
        f.write("end_header\n")
        for v in vertices:
            f.write(f"{v[0]:.6f} {v[1]:.6f} {v[2]:.6f} {r} {g} {b}\n")
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

    Port of SAM3DBody.get_ray_condition. Width-cropped like the backbone input.
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

    ray = grid.transpose(0, 3, 1, 2)  # (N, 2, 512, 512)
    crop = CROP_WIDTH[args.arch]
    return ray[:, :, :, crop:-crop] if crop else ray


def get_condition_info(batch):
    """CLIFF-style condition (cx/f, cy/f, b/f).

    Port of SAM3DBody._get_decoder_condition (CONDITION_TYPE=cliff,
    USE_INTRIN_CENTER=True).
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

    Port of Compose([GetBBoxCenterScale, TopdownAffine, ToTensor]) + prepare_batch.
    img: (H, W, 3) RGB / boxes: (N, 4) xyxy in original-image coordinates.
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
        "img": crops,  # (N, 3, 512, 512) - the square crop, before data_preprocess
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


def data_preprocess(crops):
    """Port of BaseModel.data_preprocess(): normalize, then width-crop for vith."""
    x = (crops - IMAGE_MEAN) / IMAGE_STD
    crop = CROP_WIDTH[args.arch]
    return x[:, :, :, crop:-crop] if crop else x


# ======================
# Model-running helpers
# ======================


def net_input_names(net):
    """Input names the model actually has, in the model's own order.

    Constant folding drops inputs that never reach an output, and ailia feeds
    positionally, so the names have to be looked up on both backends.
    """
    if args.onnx:
        return [i.name for i in net.get_inputs()]
    return [net.get_blob_name(idx) for idx in net.get_input_blob_list()]


def run_net(net, feed):
    """Run ailia.Net / onnxruntime.InferenceSession with a name->array dict."""
    names = net_input_names(net)
    missing = [n for n in names if n not in feed]
    if missing:
        raise KeyError(f"model expects inputs that were not prepared: {missing}")
    if args.onnx:
        return net.run(None, {n: feed[n] for n in names})
    return net.run([feed[n] for n in names])


def resize_shortest_edge_shape(oldh, oldw, short_edge_length, max_size):
    """Port of detectron2 ResizeShortestEdge.get_output_shape.

    The max_size check uses the unrounded floats.
    """
    h, w = oldh, oldw
    size = short_edge_length * 1.0
    scale = size / min(h, w)
    if h < w:
        newh, neww = size, scale * w
    else:
        newh, neww = scale * h, size
    if max(newh, neww) > max_size:
        scale = max_size * 1.0 / max(newh, neww)
        newh, neww = newh * scale, neww * scale
    return int(newh + 0.5), int(neww + 0.5)


def detect_persons(det_net, img_bgr):
    """ViTDet person detection.

    Port of run_detectron2_vitdet. The class filter / threshold / sort stay here,
    as in the original (they are numpy operations outside the model).
    """
    height, width = img_bgr.shape[:2]
    new_h, new_w = resize_shortest_edge_shape(height, width, DET_SIZE, DET_SIZE)
    # detectron2 resizes uint8 images with PIL, not cv2 (cv2.INTER_LINEAR shifts
    # the box by ~0.9 px at this 2.2x downscale).
    resized = np.asarray(
        Image.fromarray(img_bgr).resize((new_w, new_h), Image.BILINEAR)
    )
    inp = resized.astype(np.float32).transpose(2, 0, 1)

    outputs = run_net(det_net, {"image": inp})
    pred_boxes, pred_classes, _pred_masks, scores, _image_size = outputs

    valid = (pred_classes == 0) & (scores > args.bbox_thresh)
    if valid.sum() == 0:
        return np.zeros((0, 4), dtype=np.float32)

    boxes = pred_boxes[valid].copy()
    # Boxes are in the resized frame (do_postprocess=False). detector_postprocess
    # maps them back with a separate scale per axis, then clips.
    boxes[:, 0::2] *= width / new_w
    boxes[:, 1::2] *= height / new_h
    boxes[:, 0::2] = boxes[:, 0::2].clip(0, width)
    boxes[:, 1::2] = boxes[:, 1::2].clip(0, height)

    order = np.lexsort((boxes[:, 3], boxes[:, 2], boxes[:, 1], boxes[:, 0]))
    return boxes[order].astype(np.float32)


def estimate_cam_int(moge_net, img_rgb):
    """Camera intrinsics from MoGe.

    The model stops at `MoGeModel.forward`; `infer()`'s `recover_focal_shift` is a
    scipy fit that cannot be traced, so it is reproduced below in numpy.

    The export needs MoGe's `onnx_compatible_mode`, which drops the antialias from
    the encoder resize (no opset17 symbolic) - about 1.2% on the focal.
    """
    height, width = img_rgb.shape[:2]
    inp = (img_rgb.astype(np.float32) / 255.0).transpose(2, 0, 1)[None]

    points, mask = run_net(moge_net, {"image": inp})
    points, mask = points[0], mask[0] > 0.5

    focal, _shift = recover_focal_shift(points, mask)
    focal = float(focal[0])

    # focal is relative to the half diagonal; infer() + denormalize_f turn it into
    # pixels, and run_moge uses the vertical focal for both axes.
    aspect = width / height
    fy = focal / 2 * (1 + aspect**2) ** 0.5 * height
    return np.array(
        [[[fy, 0, width / 2.0], [0, fy, height / 2.0], [0, 0, 1]]], dtype=np.float32
    )


def _normalized_view_plane_uv(width, height, dtype=np.float32):
    """Centered UV coordinates normalized by the image half-diagonal."""
    half_w, half_h = width / 2.0, height / 2.0
    half_diag = (height**2 + width**2) ** 0.5 / 2.0
    xs = (np.arange(width, dtype=np.float64) + 0.5 - half_w) / half_diag
    ys = (np.arange(height, dtype=np.float64) + 0.5 - half_h) / half_diag
    uv_x, uv_y = np.meshgrid(xs, ys)
    return np.stack([uv_x, uv_y], axis=-1).astype(dtype)


def _nearest_downsample(arr, size):
    """(1, H, W, C) -> (1, out_h, out_w, C), matching F.interpolate(mode='nearest')."""
    _, in_h, in_w, _ = arr.shape
    out_h, out_w = size
    y_idx = np.floor(np.arange(out_h) * in_h / out_h).astype(int)
    x_idx = np.floor(np.arange(out_w) * in_w / out_w).astype(int)
    return arr[:, y_idx[:, None], x_idx[None, :], :]


def _solve_optimal_focal_shift(uv, xyz):
    """Solve `min |focal * xy / (z + shift) - uv|` over shift and focal."""
    uv, xy, z = uv.reshape(-1, 2), xyz[..., :2].reshape(-1, 2), xyz[..., 2].reshape(-1)

    def fn(uv, xy, z, shift):
        xy_proj = xy / (z + shift)[:, None]
        f = (xy_proj * uv).sum() / np.square(xy_proj).sum()
        return (f * xy_proj - uv).ravel()

    solution = least_squares(partial(fn, uv, xy, z), x0=0, ftol=1e-3, method="lm")
    optim_shift = solution["x"].squeeze().astype(np.float32)

    xy_proj = xy / (z + optim_shift)[:, None]
    optim_focal = (xy_proj * uv).sum() / np.square(xy_proj).sum()
    return optim_shift, optim_focal


def recover_focal_shift(points, mask, downsample_size=(64, 64)):
    """Recover focal (relative to the half diagonal) and z-shift from a point map.

    numpy port of moge.utils.geometry_torch.recover_focal_shift, so the sample
    needs neither `moge` nor `utils3d`. Same port the sam-3d-objects sample uses.
    """
    height, width = points.shape[-3], points.shape[-2]
    points = points.reshape(-1, height, width, 3)
    mask = mask.reshape(-1, height, width)
    uv = _normalized_view_plane_uv(width, height, dtype=points.dtype)

    points_lr = _nearest_downsample(points, downsample_size)
    uv_lr = _nearest_downsample(uv[None], downsample_size)[0]
    mask_lr = (
        _nearest_downsample(mask.astype(np.float32)[..., None], downsample_size)[..., 0]
        > 0.0
    )

    focals, shifts = [], []
    for i in range(points.shape[0]):
        pts_i, uv_i = points_lr[i][mask_lr[i]], uv_lr[mask_lr[i]]
        if uv_i.shape[0] < 2:
            logger.warning("MoGe mask is (almost) empty; falling back to a default FOV")
            focals.append(1.0)
            shifts.append(0.0)
            continue
        shift_i, focal_i = _solve_optimal_focal_shift(uv_i, pts_i)
        focals.append(float(focal_i))
        shifts.append(float(shift_i))
    return np.asarray(focals), np.asarray(shifts)


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
    (image_embeddings,) = run_net(
        models["backbone"], {"crop_image": data_preprocess(batch["img"])}
    )

    # forward_pose_branch adds the mask embedding to the backbone output before the
    # decoder (outside its graph). With no mask, that is the learned no_mask_embed.
    image_embeddings = image_embeddings + models["no_mask_embed"].reshape(1, -1, 1, 1)

    logger.info("Running body decoder...")
    n_person = len(boxes)
    feed = {
        "image_embeddings": image_embeddings,
        "keypoints": dummy_keypoint_prompt(n_person),
        "condition_info": get_condition_info(batch),
        "ray_cond": get_ray_condition(batch),
        "bbox_center": batch["bbox_center"],
        "bbox_scale": batch["bbox_scale"],
        "cam_int": batch["cam_int"],
        "affine_trans": batch["affine_trans"],
        "img_size": batch["img_size"],
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

        # output.png is the mesh overlay (the equivalent of cli.py's
        # dancing_overlay_000.png); the keypoint/bbox view goes next to it.
        savepath = get_savepath(args.savepath, image_path, ext=".png")
        cv2.imwrite(savepath, draw_mesh(img_bgr, results, models["faces"]))
        logger.info(f"saved at : {savepath}")

        kpt_path = os.path.splitext(savepath)[0] + "_keypoints.png"
        cv2.imwrite(kpt_path, draw_results(img_bgr, results))
        logger.info(f"saved at : {kpt_path}")

        side_path = os.path.splitext(savepath)[0] + "_side.png"
        cv2.imwrite(side_path, draw_mesh_side(img_bgr, results, models["faces"]))
        logger.info(f"saved at : {side_path}")

        if not args.skip_ply:
            for pid, person in enumerate(results):
                ply_path = os.path.splitext(savepath)[0] + f"_mesh_{pid:03d}.ply"
                save_ply(
                    ply_path,
                    person["pred_vertices"] + person["pred_cam_t"],
                    models["faces"],
                )
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
    no_mask_embed_file = {
        "vith": NO_MASK_EMBED_VITH,
        "dinov3": NO_MASK_EMBED_DINOV3,
    }[arch]

    check_and_download_models(WEIGHT_VITDET, MODEL_VITDET, REMOTE_PATH)
    check_and_download_file(WEIGHT_VITDET_PB, REMOTE_PATH)
    check_and_download_models(weight_backbone, model_backbone, REMOTE_PATH)
    check_and_download_file(pb_backbone, REMOTE_PATH)
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
        "no_mask_embed": np.load(no_mask_embed_file),
        "faces": np.load(MHR_FACES),
        "detector": detector,
        "moge": moge,
        "backbone": backbone,
        "body_decoder": body_decoder,
    }

    recognize_from_image(models)


if __name__ == "__main__":
    main()
