import copy
import glob
import os
import sys
import warnings
from logging import getLogger

import ailia
import cv2
import h5py
import numpy as np
import tqdm

# import original modules
sys.path.append("../../util")
# isort: off
from arg_utils import get_base_parser, get_savepath, update_parser
from math_utils import sigmoid
from model_utils import check_and_download_file, check_and_download_models

# isort: on

# import local modules
from collision_optimization import CollisionNonlinearOptimizer
from lidar_box3d import LiDARInstance3DBoxes
from nuscenes_dataset import NuScenesDataset
from render import Visualizer
from track_instance import Instances
from tracker import RuntimeTrackerBase

logger = getLogger(__name__)


# ======================
# Parameters
# ======================

WEIGHT_BEV_ENC_PATH = "bev_encoder.msfix.onnx"
MODEL_BEV_ENC_PATH = "bev_encoder.msfix.onnx.prototxt"
WEIGHT_TRACK_HEAD_PATH = "track_head.onnx"
MODEL_TRACK_HEAD_PATH = "track_head.onnx.prototxt"
WEIGHT_MEMORY_BANK_PATH = "memory_bank.onnx"
MODEL_MEMORY_BANK_PATH = "memory_bank.onnx.prototxt"
WEIGHT_MEMORY_BANK_UPD_PATH = "memory_bank_update.onnx"
MODEL_MEMORY_BANK_UPD_PATH = "memory_bank_update.onnx.prototxt"
WEIGHT_QUERY_INTERACTION_PATH = "query_interact.onnx"
MODEL_QUERY_INTERACTION_PATH = "query_interact.onnx.prototxt"
WEIGHT_SEG_HEAD_PATH = "seg_head.onnx"
MODEL_SEG_HEAD_PATH = "seg_head.onnx.prototxt"
WEIGHT_MOTION_HEAD_PATH = "motion_head.onnx"
MODEL_MOTION_HEAD_PATH = "motion_head.onnx.prototxt"
WEIGHT_OCC_HEAD_PATH = "occ_head.onnx"
MODEL_OCC_HEAD_PATH = "occ_head.onnx.prototxt"
WEIGHT_PLANNING_HEAD_PATH = "planning_head.onnx"
MODEL_PLANNING_HEAD_PATH = "planning_head.onnx.prototxt"
QUERY_EMBEDDING_FILE = "resources/query_embedding.npy"
REFERENCE_POINTS_WEIGHT_FILE = "resources/reference_points_weight.npy"
REFERENCE_POINTS_BIAS_FILE = "resources/reference_points_bias.npy"
REMOTE_PATH = "https://storage.googleapis.com/ailia-models/uniad/"

DEFAULT_SCENES = ["scene-0102", "scene-0103"]
DEFAULT_VIS_SCENES = ["scene-0103"]
SAVE_IMAGE_PATH = "output_video.avi"


# ======================
# Arguemnt Parser Config
# ======================

parser = get_base_parser("UniAD", None, SAVE_IMAGE_PATH)
parser.add_argument(
    "--scenes",
    type=str,
    nargs="+",
    action="append",
    default=None,
    help="Test scenes to process. Can be specified as: --scenes scene-0102 scene-0103 OR --scenes scene-0102 --scenes scene-0103. If not specified, all scenes will be processed.",
)
parser.add_argument(
    "--vis_scenes",
    type=str,
    nargs="+",
    action="append",
    default=None,
    help="Scenes to visualize. Can be specified as: --vis_scenes scene-0102 scene-0103 OR --vis_scenes scene-0102 --vis_scenes scene-0103. If not specified, all processed scenes will be visualized.",
)
parser.add_argument(
    "--ann_file",
    type=str,
    default=None,
    help="Path to annotation file. If not specified, data will be prepared from scratch.",
)
parser.add_argument(
    "--data_root",
    type=str,
    default="data/nuscenes/",
    help="Path to NuScenes dataset root directory.",
)
parser.add_argument(
    "--version",
    type=str,
    default=None,
    help="NuScenes dataset version (e.g., v1.0-trainval, v1.0-mini).",
)
parser.add_argument(
    "--results_file",
    type=str,
    default="all_frames.h5",
    help="Path to HDF5 file used to stream per-frame results.",
)
parser.add_argument(
    "--visualize",
    type=str,
    default=None,
    help="Path to HDF5 file to visualize (skip inference).",
)
parser.add_argument("--onnx", action="store_true", help="execute onnxruntime version.")
args = update_parser(parser)


# ======================
# Secondary Functions
# ======================


def flatten_args(arg_list):
    """Flatten nested lists from nargs='*' with action='append'."""
    if arg_list is None:
        return None
    flattened = []
    for item in arg_list:
        if isinstance(item, list):
            flattened.extend(item)
        else:
            flattened.append(item)
    return flattened if flattened else None


def save_frame_to_hdf5(h5_group, result, frame_count):
    """Save a single frame result to HDF5 group.

    Args:
        h5_group: HDF5 file handle
        result: Result dictionary containing frame data
        frame_count: Frame index number
    """
    group = h5_group.create_group(f"frame_{frame_count:06d}")
    group.attrs["token"] = result["token"]
    group.attrs["scene_token"] = result["scene_token"]
    group.attrs["command"] = result["command"]
    group.create_dataset("boxes_3d", data=result["boxes_3d"].tensor)
    group.create_dataset("scores_3d", data=result["scores_3d"])
    group.create_dataset("labels_3d", data=result["labels_3d"])
    group.create_dataset("track_ids", data=result["track_ids"])
    group.create_dataset("traj", data=result["traj"])
    group.create_dataset("traj_scores", data=result["traj_scores"])
    group.create_dataset("sdc_boxes_3d", data=result["sdc_boxes_3d"].tensor)
    group.create_dataset("sdc_scores_3d", data=result["sdc_scores_3d"])
    group.create_dataset("planning_traj", data=result["planning_traj"])
    # Flush to disk immediately
    h5_group.flush()


def load_frames_from_hdf5(hdf5_file):
    """Load all frame results from HDF5 file.

    Args:
        hdf5_file: Path to HDF5 file

    Returns:
        List of result dictionaries
    """
    bbox_results = []
    with h5py.File(hdf5_file, "r") as h5f:
        for frame_key in sorted(h5f.keys()):
            group = h5f[frame_key]
            bbox_results.append(
                {
                    "token": group.attrs["token"],
                    "scene_token": group.attrs["scene_token"],
                    "command": group.attrs["command"],
                    "boxes_3d": LiDARInstance3DBoxes(group["boxes_3d"][:], box_dim=9),
                    "scores_3d": group["scores_3d"][:],
                    "labels_3d": group["labels_3d"][:],
                    "track_ids": group["track_ids"][:],
                    "traj": group["traj"][:],
                    "traj_scores": group["traj_scores"][:],
                    "sdc_boxes_3d": LiDARInstance3DBoxes(
                        group["sdc_boxes_3d"][:], box_dim=9
                    ),
                    "sdc_scores_3d": group["sdc_scores_3d"][:],
                    "planning_traj": group["planning_traj"][:],
                }
            )

    return bbox_results


def inverse_sigmoid(x, eps=1e-5):
    """Numerically stable logit."""
    x = np.clip(x, eps, 1 - eps)
    return np.log(x) - np.log(1 - x)


# Cache for query embedding (load once)
_query_embedding = None


def empty_tracks():
    global _query_embedding

    track_instances = Instances((1, 1))

    if _query_embedding is None:
        _query_embedding = np.load(QUERY_EMBEDDING_FILE)

    query = _query_embedding.copy()
    num_queries, dim = query.shape

    track_instances.query = query
    track_instances.ref_pts = reference_points(query)

    track_instances.obj_idxes = np.full((len(track_instances)), -1, dtype=np.int64)
    track_instances.matched_gt_idxes = np.full(
        (len(track_instances)), -1, dtype=np.int64
    )
    track_instances.disappear_time = np.zeros((len(track_instances)), dtype=np.int64)

    track_instances.iou = np.zeros((len(track_instances),), dtype=np.float32)
    track_instances.scores = np.zeros((len(track_instances),), dtype=np.float32)
    track_instances.track_scores = np.zeros((len(track_instances),), dtype=np.float32)

    # init boxes: xy, wl, z, h, sin, cos, vx, vy, vz
    num_box_dims = 10
    track_instances.pred_boxes = np.zeros(
        (len(track_instances), num_box_dims), dtype=np.float32
    )

    num_classes = 10  # number of classes from the dataset config
    track_instances.pred_logits = np.zeros(
        (len(track_instances), num_classes), dtype=np.float32
    )

    mem_bank_len = 4  # typical memory bank length
    track_instances.mem_bank = np.zeros(
        (len(track_instances), mem_bank_len, dim // 2), dtype=np.float32
    )
    track_instances.mem_padding_mask = np.ones(
        (len(track_instances), mem_bank_len), dtype=bool
    )
    track_instances.save_period = np.zeros((len(track_instances),), dtype=np.float32)

    return track_instances


def velo_update(ref_pts, velocity, l2g_r1, l2g_t1, l2g_r2, l2g_t2, time_delta):
    """
    Args:
        ref_pts (np.ndarray): (num_query, 3) in inverse sigmoid space.
        velocity (np.ndarray): (num_query, 2) m/s in the LiDAR frame.
    Outs:
        np.ndarray: updated reference points in inverse sigmoid space.
    """

    num_query = ref_pts.shape[0]
    velo_pad = np.concatenate(
        [velocity, np.zeros((num_query, 1), dtype=velocity.dtype)], axis=-1
    )

    reference_points = sigmoid(ref_pts)
    pc_range = np.array([-51.2, -51.2, -5.0, 51.2, 51.2, 3.0], dtype=np.float32)
    reference_points[..., 0:1] = (
        reference_points[..., 0:1] * (pc_range[3] - pc_range[0]) + pc_range[0]
    )
    reference_points[..., 1:2] = (
        reference_points[..., 1:2] * (pc_range[4] - pc_range[1]) + pc_range[1]
    )
    reference_points[..., 2:3] = (
        reference_points[..., 2:3] * (pc_range[5] - pc_range[2]) + pc_range[2]
    )
    reference_points = reference_points + velo_pad * time_delta

    ref_pts = reference_points @ l2g_r1 + l2g_t1 - l2g_t2
    g2l_r = np.linalg.inv(l2g_r2).astype(np.float32)
    ref_pts = ref_pts @ g2l_r

    ref_pts[..., 0:1] = (ref_pts[..., 0:1] - pc_range[0]) / (pc_range[3] - pc_range[0])
    ref_pts[..., 1:2] = (ref_pts[..., 1:2] - pc_range[1]) / (pc_range[4] - pc_range[1])
    ref_pts[..., 2:3] = (ref_pts[..., 2:3] - pc_range[2]) / (pc_range[5] - pc_range[2])

    ref_pts = inverse_sigmoid(ref_pts)

    return ref_pts


def denormalize_bbox(normalized_bboxes):
    # rotation
    rot_sine = normalized_bboxes[..., 6:7]

    rot_cosine = normalized_bboxes[..., 7:8]
    rot = np.arctan2(rot_sine, rot_cosine)

    # center in the bev
    cx = normalized_bboxes[..., 0:1]
    cy = normalized_bboxes[..., 1:2]
    cz = normalized_bboxes[..., 4:5]

    # size
    w = normalized_bboxes[..., 2:3]
    l = normalized_bboxes[..., 3:4]
    h = normalized_bboxes[..., 5:6]

    w = np.exp(w)
    l = np.exp(l)
    h = np.exp(h)

    # velocity
    vx = normalized_bboxes[:, 8:9]
    vy = normalized_bboxes[:, 9:10]
    denormalized_bboxes = np.concatenate([cx, cy, cz, w, l, h, rot, vx, vy], axis=-1)

    return denormalized_bboxes


def decode_single(cls_scores, bbox_preds, track_scores, obj_idxes, with_mask=True):
    """Decode bboxes.
    Args:
        cls_scores (Tensor): Outputs from the classification head, \
            shape [num_query, cls_out_channels]. Note \
            cls_out_channels should includes background.
        bbox_preds (Tensor): Outputs from the regression \
            head with normalized coordinate format (cx, cy, w, l, cz, h, rot_sine, rot_cosine, vx, vy). \
            Shape [num_query, 9].

    Returns:
        list[dict]: Decoded boxes.
    """
    max_num = 300
    max_num = min(cls_scores.shape[0], max_num)

    num_classes = 10
    cls_scores = sigmoid(cls_scores)
    indexs = np.argmax(cls_scores, axis=-1)
    labels = indexs % num_classes

    bbox_index = np.argpartition(track_scores, -max_num)[-max_num:]
    bbox_index = bbox_index[np.argsort(track_scores[bbox_index])[::-1]]

    labels = labels[bbox_index]
    bbox_preds = bbox_preds[bbox_index]
    track_scores = track_scores[bbox_index]
    obj_idxes = obj_idxes[bbox_index]

    scores = track_scores

    final_box_preds = denormalize_bbox(bbox_preds)
    final_scores = track_scores
    final_preds = labels

    post_center_range = np.array(
        [-61.2000, -61.2000, -10.0000, 61.2000, 61.2000, 10.0000]
    )
    mask = (final_box_preds[..., :3] >= post_center_range[:3]).all(1)
    mask &= (final_box_preds[..., :3] <= post_center_range[3:]).all(1)
    if not with_mask:
        mask = np.ones_like(mask) > 0

    boxes3d = final_box_preds[mask]
    scores = final_scores[mask]
    labels = final_preds[mask]
    track_scores = track_scores[mask]
    obj_idxes = obj_idxes[mask]
    predictions_dict = {
        "bboxes": boxes3d,
        "scores": scores,
        "labels": labels,
        "track_scores": track_scores,
        "obj_idxes": obj_idxes,
        "bbox_index": bbox_index,
        "mask": mask,
    }

    return predictions_dict


def track_instances2results(track_instances, with_mask=True):
    all_cls_scores = track_instances.pred_logits
    all_bbox_preds = track_instances.pred_boxes
    track_scores = track_instances.scores
    obj_idxes = track_instances.obj_idxes
    bboxes_dict = decode_single(
        all_cls_scores,
        all_bbox_preds,
        track_scores,
        obj_idxes,
        with_mask,
    )

    bboxes = bboxes_dict["bboxes"]
    bboxes = LiDARInstance3DBoxes(bboxes, box_dim=9)
    labels = bboxes_dict["labels"]
    scores = bboxes_dict["scores"]
    bbox_index = bboxes_dict["bbox_index"]
    track_scores = bboxes_dict["track_scores"]
    obj_idxes = bboxes_dict["obj_idxes"]
    result_dict = dict(
        boxes_3d=bboxes,
        scores_3d=scores,
        labels_3d=labels,
        track_scores=track_scores,
        bbox_index=bbox_index,
        track_ids=obj_idxes,
        mask=bboxes_dict["mask"],
        track_bbox_results=[
            [
                bboxes.clone(),
                scores.copy(),
                labels.copy(),
                bbox_index.copy(),
                bboxes_dict["mask"].copy(),
            ]
        ],
    )
    return result_dict


def det_instances2results(instances, result_dict):
    """
    Outs:
    active_instances. keys:
    - 'pred_logits':
    - 'pred_boxes': normalized bboxes
    - 'scores'
    - 'obj_idxes'
    out_dict. keys:
        - boxes_3d (torch.Tensor): 3D boxes.
        - scores (torch.Tensor): Prediction scores.
        - labels_3d (torch.Tensor): Box labels.
        - attrs_3d (torch.Tensor, optional): Box attributes.
        - track_ids
        - tracking_score
    """
    # filter out sleep querys
    if instances.pred_logits.size == 0:
        return [None]

    # decode
    with_mask = True
    all_cls_scores = instances.pred_logits
    all_bbox_preds = instances.pred_boxes
    track_scores = instances.scores
    obj_idxes = instances.obj_idxes
    bboxes_dict = decode_single(
        all_cls_scores,
        all_bbox_preds,
        track_scores,
        obj_idxes,
        with_mask,
    )

    bboxes = bboxes_dict["bboxes"]
    bboxes = LiDARInstance3DBoxes(bboxes, box_dim=9)
    labels = bboxes_dict["labels"]
    scores = bboxes_dict["scores"]

    track_scores = bboxes_dict["track_scores"]
    obj_idxes = bboxes_dict["obj_idxes"]
    result_dict.update(
        dict(
            boxes_3d_det=bboxes,
            scores_3d_det=scores,
            labels_3d_det=labels,
        )
    )

    return result_dict


def get_trajs(preds_dicts, bbox_results):
    """
    Generates trajectories from the prediction results, bounding box results.
    """
    num_samples = len(bbox_results)
    num_layers = preds_dicts["all_traj_preds"].shape[0]
    ret_list = []
    for i in range(num_samples):
        preds = dict()
        for j in range(num_layers):
            subfix = "_" + str(j) if j < (num_layers - 1) else ""
            traj = preds_dicts["all_traj_preds"][j, i]
            traj_scores = preds_dicts["all_traj_scores"][j, i]

            preds["traj" + subfix] = traj
            preds["traj_scores" + subfix] = traj_scores
        ret_list.append(preds)
    return ret_list


def to_video(folder_path, out_path, fps=4, downsample=1):
    imgs_path = glob.glob(os.path.join(folder_path, "*.jpg"))
    imgs_path = sorted(imgs_path)
    img_array = []
    for img_path in imgs_path:
        img = cv2.imread(img_path)
        height, width, channel = img.shape
        img = cv2.resize(
            img,
            (width // downsample, height // downsample),
            interpolation=cv2.INTER_AREA,
        )
        height, width, channel = img.shape
        size = (width, height)
        img_array.append(img)
    out = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*"DIVX"), fps, size)
    for i in range(len(img_array)):
        out.write(img_array[i])
    out.release()


def visualize_results(
    results_file, nusc=None, output_dir="vis_output", fps=4, downsample=2
):
    """Visualize detection and planning results.

    Args:
        results_file: Path to HDF5 file containing frame results
        nusc: NuScenes dataset instance (optional, will be loaded if None)
        output_dir: Directory to save output images
        fps: Frames per second for video
        downsample: Downsample factor for video resolution
    """
    # Determine visualization scenes internally
    defaults_on_none = args.ann_file is None and args.version is None
    vis_scenes = flatten_args(args.vis_scenes) or (
        DEFAULT_VIS_SCENES if defaults_on_none else None
    )

    # Load frame results from HDF5 for visualization
    logger.info("Loading frame results for visualization...")
    bbox_results = load_frames_from_hdf5(results_file)

    if len(bbox_results) == 0:
        logger.info("No results to visualize.")
        sys.exit(1)

    # Load NuScenes dataset if not provided
    if nusc is None:
        from nuscenes import NuScenes

        logger.info("Loading NuScenes dataset...")
        data_root = args.data_root
        version = args.version or "v1.0-trainval"
        nusc = NuScenes(version=version, dataroot=data_root, verbose=True)

    vis = Visualizer(bbox_results=bbox_results, nuscenes=nusc)

    if vis_scenes:
        logger.info(f"Visualizing scenes: {vis_scenes}")
    else:
        logger.info("Visualizing all processed scenes")

    scene_token_to_name = dict()
    for i in range(len(vis.nusc.scene)):
        scene_token_to_name[vis.nusc.scene[i]["token"]] = vis.nusc.scene[i]["name"]

    # Clear and create output directory
    for f in glob.glob(os.path.join(output_dir, "*")):
        if os.path.isfile(f):
            os.remove(f)
    os.makedirs(output_dir, exist_ok=True)

    for i, sample in enumerate(bbox_results):
        sample_token = sample["token"]
        scene_token = sample["scene_token"]

        # Get scene name for filtering
        scene_name = scene_token_to_name[scene_token]
        # Skip if not in vis_scenes list
        if vis_scenes and scene_name not in vis_scenes:
            continue

        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=RuntimeWarning)
            bev_img = vis.visualize_bev(sample_token)
            cam_img = vis.visualize_cam(sample_token)
        out_img = cv2.hconcat([cam_img, bev_img])
        out_img = out_img[:, :, ::-1]  # RGB to BGR

        savepath = f"{output_dir}/{str(i).zfill(3)}.jpg"
        cv2.imwrite(savepath, out_img)
        logger.info(f"saved at : {savepath}")

    savepath = get_savepath(args.savepath, output_dir, ext=".avi")
    jpg_paths = glob.glob(os.path.join(output_dir, "*.jpg"))
    if not jpg_paths:
        logger.info("No frames written; skipping video output.")
        return

    to_video(output_dir, savepath, fps=fps, downsample=downsample)
    logger.info(f"saved at : {savepath}")


# ======================
# Main functions
# ======================


def predict(models, img, img_metas, prev_bev=None, track_instances=None):
    can_bus = img_metas["can_bus"].astype(np.float32)
    lidar2img = img_metas["lidar2img"].astype(np.float32)
    img_shape = img_metas["img_shape"]
    if prev_bev is None:
        prev_bev = np.zeros((40000, 1, 256), dtype=np.float32)

    # feedforward
    bev_encoder = models["bev_encoder"]
    if not args.onnx:
        output = bev_encoder.predict([img, can_bus, lidar2img, img_shape, prev_bev])
    else:
        output = bev_encoder.run(
            None,
            {
                "img": img,
                "can_bus": can_bus,
                "lidar2img": lidar2img,
                "img_shape": img_shape,
                "prev_bev": prev_bev,
            },
        )
    bev_embed, bev_pos = output

    query = track_instances.query
    ref_pts = track_instances.ref_pts

    # feedforward
    track_head = models["track_head"]
    if not args.onnx:
        output = track_head.predict([bev_embed, query, ref_pts])
    else:
        output = track_head.run(
            None,
            {
                "bev_embed": bev_embed,
                "query": query,
                "ref_pts": ref_pts,
            },
        )
    (
        output_classes,
        output_coords,
        last_ref_pts,
        query_feats,
        all_past_traj_preds,
    ) = output

    out = {
        "pred_logits": output_classes,
        "pred_boxes": output_coords,
        "ref_pts": last_ref_pts,
        "bev_embed": bev_embed,
        "query_embeddings": query_feats,
        "all_past_traj_preds": all_past_traj_preds,
        "bev_pos": bev_pos,
    }

    return out


def memory_bank(models, track_instances):
    key_padding_mask = track_instances.mem_padding_mask  # [n_, memory_bank_len]
    valid_idxes = key_padding_mask[:, -1] == 0
    embed = track_instances.output_embedding[valid_idxes]  # (n, 256)

    if 0 < len(embed):
        net = models["memory_bank"]
        if not args.onnx:
            output = net.predict(
                [
                    track_instances.mem_padding_mask,
                    track_instances.output_embedding,
                    track_instances.mem_bank,
                ]
            )
        else:
            output = net.run(
                None,
                {
                    "mem_padding_mask": track_instances.mem_padding_mask,
                    "output_embedding": track_instances.output_embedding,
                    "mem_bank": track_instances.mem_bank,
                },
            )
        track_instances.output_embedding = output[0]

    embed = track_instances.output_embedding[:, None]  # ( N, 1, 256)
    save_period = track_instances.save_period
    saved_idxes = (save_period == 0) & (track_instances.scores > 0)
    saved_embed = embed[saved_idxes]

    if 0 < len(saved_embed):
        net = models["memory_bank_update"]
        if not args.onnx:
            output = net.predict(
                [
                    track_instances.output_embedding,
                    track_instances.scores,
                    track_instances.mem_padding_mask,
                    track_instances.save_period,
                    track_instances.mem_bank,
                ]
            )
        else:
            output = net.run(
                None,
                {
                    "output_embedding": track_instances.output_embedding,
                    "scores": track_instances.scores,
                    "mem_padding_mask": track_instances.mem_padding_mask,
                    "save_period": track_instances.save_period,
                    "mem_bank": track_instances.mem_bank,
                },
            )
        mem_padding_mask, mem_bank, save_period = output
        track_instances.mem_padding_mask = mem_padding_mask
        track_instances.mem_bank = mem_bank
        track_instances.save_period = save_period

    return track_instances


def query_interact(models, track_instances):
    active_track_instances = track_instances[track_instances.obj_idxes >= 0]

    if 0 < len(active_track_instances):
        net = models["query_interact"]
        if not args.onnx:
            output = net.predict(
                [active_track_instances.query, active_track_instances.output_embedding]
            )
        else:
            output = net.run(
                None,
                {
                    "query": active_track_instances.query,
                    "output_embedding": active_track_instances.output_embedding,
                },
            )
        updated_query = output[0]
        active_track_instances.query = updated_query

    merged_track_instances = Instances.cat([empty_tracks(), active_track_instances])

    return merged_track_instances


def seg_head_forward(models, pts_feats):
    net = models["seg_head"]
    if not args.onnx:
        output = net.predict([pts_feats])
    else:
        output = net.run(
            None,
            {"bev_embed": pts_feats},
        )
    (memory, memory_mask, memory_pos, query, query_pos, hw_lvl, hw_lvl2) = output

    out = [
        memory,
        memory_mask,
        memory_pos,
        query,
        query_pos,
        [hw_lvl, hw_lvl2],
    ]
    return out


def motion_head_forward(models, bev_embed, outs_track: dict, outs_seg: dict):
    track_query = outs_track["track_query_embeddings"][None, None, ...]
    track_boxes: LiDARInstance3DBoxes = outs_track["track_bbox_results"]

    track_query = np.concatenate(
        [track_query, outs_track["sdc_embedding"][None, None, None, :]], axis=2
    )
    sdc_track_boxes = outs_track["sdc_track_bbox_results"]

    track_boxes[0][0].tensor = np.concatenate(
        [track_boxes[0][0].tensor, sdc_track_boxes[0][0].tensor], axis=0
    )
    track_boxes[0][1] = np.concatenate(
        [track_boxes[0][1], sdc_track_boxes[0][1]], axis=0
    )
    track_boxes[0][2] = np.concatenate(
        [track_boxes[0][2], sdc_track_boxes[0][2]], axis=0
    )
    track_boxes[0][3] = np.concatenate(
        [track_boxes[0][3], sdc_track_boxes[0][3]], axis=0
    )
    memory, memory_mask, memory_pos, lane_query, lane_query_pos, hw_lvl = outs_seg

    net = models["motion_head"]
    if not args.onnx:
        output = net.predict(
            [
                bev_embed,
                track_query,
                lane_query,
                lane_query_pos,
                track_boxes[0][0].tensor,
                track_boxes[0][2],
            ]
        )
    else:
        output = net.run(
            None,
            {
                "bev_embed": bev_embed,
                "track_query": track_query,
                "lane_query": lane_query,
                "lane_query_pos": lane_query_pos,
                "bboxes": track_boxes[0][0].tensor,
                "bbox_labels": track_boxes[0][2],
            },
        )
    (
        outputs_traj_scores,
        outputs_trajs,
        valid_traj_masks,
        inter_states,
        track_query,
        track_query_pos,
    ) = output

    outs_motion = {
        "all_traj_scores": outputs_traj_scores,
        "all_traj_preds": outputs_trajs,
        "valid_traj_masks": valid_traj_masks,
        "traj_query": inter_states,
        "track_query": track_query,
        "track_query_pos": track_query_pos,
    }

    # get_trajs
    traj_results = get_trajs(outs_motion, track_boxes)
    bboxes, scores, labels, bbox_index, mask = track_boxes[0]
    outs_motion["track_scores"] = scores[None, :]
    labels[-1] = 0

    def filter_vehicle_query(outs_motion, labels, vehicle_id_list):
        if len(labels) < 1:  # No other obj query except sdc query.
            return None

        # select vehicle query according to vehicle_id_list
        vehicle_mask = np.zeros_like(labels)
        for veh_id in vehicle_id_list:
            vehicle_mask |= labels == veh_id
        outs_motion["traj_query"] = outs_motion["traj_query"][:, :, vehicle_mask > 0]
        outs_motion["track_query"] = outs_motion["track_query"][:, vehicle_mask > 0]
        outs_motion["track_query_pos"] = outs_motion["track_query_pos"][
            :, vehicle_mask > 0
        ]
        outs_motion["track_scores"] = outs_motion["track_scores"][:, vehicle_mask > 0]
        return outs_motion

    # NuScenes データセットにおける車両（移動体）カテゴリのクラスID一覧
    # モーション予測の対象となる動的オブジェクトのみを選択
    # 0: car (乗用車), 1: truck (トラック), 2: construction_vehicle (建設車両),
    # 3: bus (バス), 4: trailer (トレーラー), 6: motorcycle (バイク), 7: bicycle (自転車)
    # 除外: 5: barrier (バリア), 8: pedestrian (歩行者), 9: traffic_cone (三角コーン)
    vehicle_id_list = [
        0,  # car
        1,  # truck
        2,  # construction_vehicle
        3,  # bus
        4,  # trailer
        6,  # motorcycle
        7,  # bicycle
    ]
    outs_motion = filter_vehicle_query(outs_motion, labels, vehicle_id_list)

    # filter sdc query
    outs_motion["sdc_traj_query"] = outs_motion["traj_query"][:, :, -1]
    outs_motion["sdc_track_query"] = outs_motion["track_query"][:, -1]
    outs_motion["sdc_track_query_pos"] = outs_motion["track_query_pos"][:, -1]
    outs_motion["traj_query"] = outs_motion["traj_query"][:, :, :-1]
    outs_motion["track_query"] = outs_motion["track_query"][:, :-1]
    outs_motion["track_query_pos"] = outs_motion["track_query_pos"][:, :-1]
    outs_motion["track_scores"] = outs_motion["track_scores"][:, :-1]

    return traj_results, outs_motion


def occ_head_forward(models, bev_feat, outs_dict, no_query=False):
    if no_query:
        b, t, h, w = 1, 5, 200, 200
        out_dict = dict()
        out_dict["seg_out"] = np.zeros(
            (b, t, 1, h, w), dtype=np.int64
        )  # [1, 5, 1, 200, 200]
        out_dict["ins_seg_out"] = np.zeros(
            (b, t, h, w), dtype=np.int64
        )  # [1, 5, 200, 200]
        return out_dict

    traj_query = outs_dict["traj_query"]
    track_query = outs_dict["track_query"]
    track_query_pos = outs_dict["track_query_pos"]

    net = models["occ_head"]
    if not args.onnx:
        output = net.predict([bev_feat, traj_query, track_query, track_query_pos])
    else:
        output = net.run(
            None,
            {
                "bev_feat": bev_feat,
                "traj_query": traj_query,
                "track_query": track_query,
                "track_query_pos": track_query_pos,
            },
        )
    _, pred_ins_logits = output

    out_dict = dict()
    out_dict["pred_ins_logits"] = pred_ins_logits

    n_future = 4
    pred_ins_logits = pred_ins_logits[:, :, : 1 + n_future]  # [b, q, t, h, w]
    pred_ins_sigmoid = sigmoid(pred_ins_logits)  # [b, q, t, h, w]

    # with_track_score
    track_scores = outs_dict["track_scores"]  # [b, q]
    track_scores = track_scores[:, :, None, None, None]
    pred_ins_sigmoid = pred_ins_sigmoid * track_scores  # [b, q, t, h, w]

    out_dict["pred_ins_sigmoid"] = pred_ins_sigmoid
    pred_seg_scores = np.max(pred_ins_sigmoid, axis=1)
    test_seg_thresh = 0.1
    seg_out = (pred_seg_scores > test_seg_thresh).astype(np.int64)
    seg_out = np.expand_dims(seg_out, axis=2)  # [b, t, 1, h, w]
    out_dict["seg_out"] = seg_out

    def update_instance_ids(instance_seg, old_ids, new_ids):
        indices = np.arange(int(old_ids.max()) + 1)
        for old_id, new_id in zip(old_ids, new_ids):
            indices[int(old_id)] = int(new_id)

        return indices[instance_seg].astype(np.int64)

    def make_instance_seg_consecutive(instance_seg):
        # Make the indices of instance_seg consecutive
        unique_ids = np.unique(instance_seg)  # include background
        new_ids = np.arange(len(unique_ids))
        instance_seg = update_instance_ids(instance_seg, unique_ids, new_ids)
        return instance_seg

    def predict_instance_segmentation_and_trajectories(
        foreground_masks,
        ins_sigmoid,
        vehicles_id=1,
    ):
        if foreground_masks.ndim == 5 and foreground_masks.shape[2] == 1:
            foreground_masks = np.squeeze(foreground_masks, axis=2)  # [b, t, h, w]
        foreground_masks = (
            foreground_masks == vehicles_id
        )  # [b, t, h, w]  Only these places have foreground id

        argmax_ins = np.argmax(
            ins_sigmoid, axis=1
        )  # long, [b, t, h, w], ins_id starts from 0
        argmax_ins = argmax_ins + 1  # [b, t, h, w], ins_id starts from 1
        instance_seg = (argmax_ins * foreground_masks.astype(np.float32)).astype(
            np.int64
        )  # bg is 0, fg starts with 1

        # Make the indices of instance_seg consecutive
        instance_seg = make_instance_seg_consecutive(instance_seg).astype(np.int64)

        return instance_seg

    # ins_pred
    pred_consistent_instance_seg = predict_instance_segmentation_and_trajectories(
        seg_out, pred_ins_sigmoid
    )  # bg is 0, fg starts with 1, consecutive
    out_dict["ins_seg_out"] = pred_consistent_instance_seg  # [1, 5, 200, 200]

    return out_dict


def planning_head_forward(models, bev_embed, outs_motion, outs_occflow, command):
    sdc_traj_query = outs_motion["sdc_traj_query"]
    sdc_track_query = outs_motion["sdc_track_query"]
    bev_pos = outs_motion["bev_pos"]
    occ_mask = outs_occflow["seg_out"]
    command = np.array([command], dtype=np.int64)

    net = models["planning_head"]
    if not args.onnx:
        output = net.predict(
            [bev_embed, bev_pos, sdc_traj_query, sdc_track_query, command]
        )
    else:
        output = net.run(
            None,
            {
                "bev_embed": bev_embed,
                "bev_pos": bev_pos,
                "sdc_traj_query": sdc_traj_query,
                "sdc_track_query": sdc_track_query,
                "command": command,
            },
        )
    sdc_traj, sdc_traj_all = output

    def collision_optimization(sdc_traj_all, occ_mask):
        """
        Optimize SDC trajectory with occupancy instance mask.
        """
        pos_xy_t = []
        valid_occupancy_num = 0

        if occ_mask.shape[2] == 1:
            occ_mask = np.squeeze(occ_mask, axis=2)
        occ_horizon = occ_mask.shape[1]
        assert occ_horizon == 5

        bev_h = bev_w = 200
        occ_filter_range = 5.0
        planning_steps = 6
        sigma = 1.0
        alpha_collision = 5.0
        for t in range(planning_steps):
            cur_t = min(t + 1, occ_horizon - 1)
            pos_xy = np.argwhere(occ_mask[0][cur_t])
            pos_xy = pos_xy[:, [1, 0]]
            pos_xy[:, 0] = (pos_xy[:, 0] - bev_h // 2) * 0.5 + 0.25
            pos_xy[:, 1] = (pos_xy[:, 1] - bev_w // 2) * 0.5 + 0.25

            # filter the occupancy in range
            keep_index = (
                np.sum((sdc_traj_all[0, t, :2][None, :] - pos_xy[:, :2]) ** 2, axis=-1)
                < occ_filter_range**2
            )
            pos_xy_t.append(pos_xy[keep_index])
            valid_occupancy_num += np.sum(keep_index > 0)
        if valid_occupancy_num == 0:
            return sdc_traj_all

        col_optimizer = CollisionNonlinearOptimizer(
            planning_steps, 0.5, sigma, alpha_collision, pos_xy_t
        )
        col_optimizer.set_reference_trajectory(sdc_traj_all[0])
        sol = col_optimizer.solve()
        sdc_traj_optim = np.stack(
            [sol.value(col_optimizer.position_x), sol.value(col_optimizer.position_y)],
            axis=-1,
        )
        return sdc_traj_optim[None].astype(sdc_traj_all.dtype)

    sdc_traj_all = collision_optimization(sdc_traj_all, occ_mask)

    result_planning = {
        "sdc_traj": sdc_traj_all,
        "sdc_traj_all": sdc_traj_all,
    }
    return result_planning


# Cache for reference points weights and bias (loaded once)
_reference_points_weight = None
_reference_points_bias = None


def reference_points(query):
    global _reference_points_weight, _reference_points_bias

    if _reference_points_weight is None:
        _reference_points_weight = np.load(REFERENCE_POINTS_WEIGHT_FILE)
        _reference_points_bias = np.load(REFERENCE_POINTS_BIAS_FILE)

    num_queries, dim = query.shape
    ref_pts = (
        query[..., : dim // 2] @ _reference_points_weight.T + _reference_points_bias
    )
    return ref_pts


def forward(
    models,
    track_base: RuntimeTrackerBase,
    img,
    img_metas,
    track_instances,
    prev_bev=None,
    l2g_r1=None,
    l2g_t1=None,
    l2g_r2=None,
    l2g_t2=None,
    time_delta=None,
):
    active_inst = track_instances[track_instances.obj_idxes >= 0]
    other_inst = track_instances[track_instances.obj_idxes < 0]

    if l2g_r2 is not None and len(active_inst) > 0 and l2g_r1 is not None:
        ref_pts = active_inst.ref_pts
        velo = active_inst.pred_boxes[:, -2:]
        ref_pts = velo_update(
            ref_pts, velo, l2g_r1, l2g_t1, l2g_r2, l2g_t2, time_delta=time_delta
        )
        ref_pts = np.squeeze(ref_pts, axis=0)
        active_inst.ref_pts = reference_points(active_inst.query)
        active_inst.ref_pts[..., :2] = ref_pts[..., :2]

    track_instances = Instances.cat([other_inst, active_inst])

    output = predict(
        models, img, img_metas, prev_bev=prev_bev, track_instances=track_instances
    )
    bev_embed = output["bev_embed"]
    output_classes = output["pred_logits"]
    output_coords = output["pred_boxes"]
    last_ref_pts = output["ref_pts"]
    query_feats = output["query_embeddings"]
    all_past_traj_preds = output["all_past_traj_preds"]
    bev_pos = output["bev_pos"]

    item = {
        "pred_logits": output_classes,
        "pred_boxes": output_coords,
        "ref_pts": last_ref_pts,
        "bev_embed": bev_embed,
        "query_embeddings": query_feats,
        "all_past_traj_preds": all_past_traj_preds,
        "bev_pos": bev_pos,
    }

    # Apply sigmoid and get max values along last axis
    logits = output_classes[-1, 0, :]
    sigmoid_logits = sigmoid(logits)
    track_scores = np.max(sigmoid_logits, axis=-1)

    # each track will be assigned an unique global id by the track base.
    track_instances.scores = track_scores
    # track_instances.track_scores = track_scores  # [300]
    track_instances.pred_logits = output_classes[-1, 0]  # [300, num_cls]
    track_instances.pred_boxes = output_coords[-1, 0]  # [300, box_dim]
    track_instances.output_embedding = query_feats[-1][0]  # [300, feat_dim]
    track_instances.ref_pts = last_ref_pts[0]
    # hard_code: assume the 901 query is sdc query
    track_instances.obj_idxes[900] = -2

    """ update track base """
    track_base.update(track_instances)

    filter_score_thresh = 0.35
    active_index = (track_instances.obj_idxes >= 0) & (
        track_instances.scores >= filter_score_thresh
    )  # filter out sleep objects
    # select_active_track_query
    result_dict = track_instances2results(track_instances[active_index], with_mask=True)
    result_dict["track_query_embeddings"] = track_instances.output_embedding[
        active_index
    ][result_dict["bbox_index"]][result_dict["mask"]]
    result_dict["track_query_matched_idxes"] = track_instances.matched_gt_idxes[
        active_index
    ][result_dict["bbox_index"]][result_dict["mask"]]
    item.update(result_dict)
    # select_sdc_track_query
    sdc_instance = track_instances[track_instances.obj_idxes == -2]
    result_dict = track_instances2results(sdc_instance, with_mask=False)
    item.update(
        dict(
            sdc_boxes_3d=result_dict["boxes_3d"],
            sdc_scores_3d=result_dict["scores_3d"],
            sdc_track_scores=result_dict["track_scores"],
            sdc_track_bbox_results=result_dict["track_bbox_results"],
            sdc_embedding=sdc_instance.output_embedding[0],
        )
    )

    """ update with memory_bank """
    track_instances = memory_bank(models, track_instances)

    """  Update track instances using matcher """
    item["track_instances_fordet"] = track_instances
    item["track_instances"] = track_instances = query_interact(models, track_instances)

    return item


def process_all_frames(models, data_loader, track_base, h5f):
    """Process all frames from data_loader and stream results to HDF5.

    Args:
        models: Dict of initialized model sessions
        data_loader: Iterable yielding dataset items
        track_base: Tracker base managing IDs and states
        h5f: Open HDF5 file handle to write per-frame results
    """

    track_instances = None
    tiemestamp = None
    scene_token = None
    prev_bev = None
    prev_frame_info = {
        "prev_bev": None,
        "scene_token": None,
        "prev_pos": 0,
        "prev_angle": 0,
    }

    for i, data in tqdm.tqdm(
        enumerate(data_loader), total=len(data_loader), desc="Processing frames"
    ):
        data = data[0]
        img = data["img"]
        img_metas = data["img_metas"]
        l2g_t = data["l2g_t"]
        l2g_r_mat = data["l2g_r_mat"]
        _timestamp = data["timestamp"]

        img = np.stack(img).transpose(0, 3, 1, 2)  # N, C, H, W
        img = np.expand_dims(img, axis=0)  # B, N, C, H, W

        if img_metas["scene_token"] != prev_frame_info["scene_token"]:
            # the first sample of each scene is truncated
            prev_frame_info["prev_bev"] = None
        # update idx
        prev_frame_info["scene_token"] = img_metas["scene_token"]

        # Get the delta of ego position and angle between two timestamps.
        tmp_pos = copy.deepcopy(img_metas["can_bus"][:3])
        tmp_angle = copy.deepcopy(img_metas["can_bus"][-1])
        # first frame
        if prev_frame_info["scene_token"] is None:
            img_metas["can_bus"][:3] = 0
            img_metas["can_bus"][-1] = 0
        # following frames
        else:
            img_metas["can_bus"][:3] -= prev_frame_info["prev_pos"]
            img_metas["can_bus"][-1] -= prev_frame_info["prev_angle"]
        prev_frame_info["prev_pos"] = tmp_pos
        prev_frame_info["prev_angle"] = tmp_angle

        ### simple_test_track logic ###

        if track_instances is None or img_metas["scene_token"] != scene_token:
            scene_token = img_metas["scene_token"]
            prev_bev = None
            track_instances = empty_tracks()
            time_delta, l2g_r1, l2g_t1, l2g_r2, l2g_t2 = None, None, None, None, None
        else:
            time_delta = _timestamp - tiemestamp
            l2g_r1 = l2g_r2
            l2g_t1 = l2g_t2
            l2g_r2 = l2g_r_mat[None, ...]
            l2g_t2 = l2g_t[None, ...]

        frame_res = forward(
            models,
            track_base,
            img,
            img_metas,
            track_instances,
            prev_bev,
            l2g_r1,
            l2g_t1,
            l2g_r2,
            l2g_t2,
            time_delta,
        )

        """ get time_delta and l2g r/t infos """
        """ update frame info for next frame"""
        tiemestamp = _timestamp
        l2g_r2 = l2g_r_mat[None, ...]
        l2g_t2 = l2g_t[None, ...]

        bev_embed = prev_bev = frame_res["bev_embed"]
        track_instances = frame_res["track_instances"]
        track_instances_fordet = frame_res["track_instances_fordet"]

        result_dict = dict(
            bev_embed=bev_embed,
            bev_pos=frame_res["bev_pos"],
            track_query_embeddings=frame_res["track_query_embeddings"],
            track_bbox_results=frame_res["track_bbox_results"],
            boxes_3d=frame_res["boxes_3d"],
            scores_3d=frame_res["scores_3d"],
            labels_3d=frame_res["labels_3d"],
            track_scores=frame_res["track_scores"],
            track_ids=frame_res["track_ids"],
            sdc_boxes_3d=frame_res["sdc_boxes_3d"],
            sdc_scores_3d=frame_res["sdc_scores_3d"],
            sdc_track_scores=frame_res["sdc_track_scores"],
            sdc_track_bbox_results=frame_res["sdc_track_bbox_results"],
            sdc_embedding=frame_res["sdc_embedding"],
        )

        result_track = det_instances2results(track_instances_fordet, result_dict)

        ### End of simple_test_track ###

        # seg_head
        result_seg = seg_head_forward(models, bev_embed)
        # motion_head
        result_motion, outs_motion = motion_head_forward(
            models, bev_embed, outs_track=result_track, outs_seg=result_seg
        )
        outs_motion["bev_pos"] = result_track["bev_pos"]

        result = dict()
        result["token"] = img_metas["sample_idx"]
        result["scene_token"] = img_metas["scene_token"]

        occ_no_query = outs_motion["track_query"].shape[1] == 0
        outs_occ = occ_head_forward(
            models,
            bev_embed,
            outs_motion,
            no_query=occ_no_query,
        )
        result["occ"] = outs_occ

        command = data["command"]
        planning_gt = dict(command=command)
        result_planning = planning_head_forward(
            models, bev_embed, outs_motion, outs_occ, command
        )
        result["planning"] = dict(
            planning_gt=planning_gt,
            result_planning=result_planning,
        )
        result.update(result_track)
        result.update(
            result_motion[0]
        )  # 'traj_0', 'traj_scores_0', 'traj_1', 'traj_scores_1', 'traj', 'traj_scores'

        ### End of forward_test ###

        result["planning_traj"] = result["planning"]["result_planning"]["sdc_traj"]
        result["command"] = result["planning"]["planning_gt"]["command"]

        # Stream frame results to HDF5 file
        save_frame_to_hdf5(h5f, result, i)

        # Clear result from memory
        del result


def recognize_from_image(models):
    defaults_on_none = args.ann_file is None and args.version is None

    ann_file = args.ann_file
    data_root = args.data_root
    version = args.version or "v1.0-trainval"
    test_scenes = flatten_args(args.scenes) or (
        DEFAULT_SCENES if defaults_on_none else None
    )

    if ann_file:
        logger.info(f"Using annotation file: {ann_file}")
    else:
        logger.info(f"Data root: {data_root}")
        logger.info(f"Dataset version: {version}")
    logger.info(f"Processing test scenes: {test_scenes or 'ALL'}")

    dataset = NuScenesDataset(
        data_root=data_root,
        version=version,
        ann_file=ann_file,
        test_scenes=test_scenes,
    )

    if len(dataset) == 0 and not defaults_on_none:
        logger.info(
            "No data found in the dataset. Please check if the scene name is correct."
        )
        sys.exit(1)

    # Simple DataLoader replacement
    class SimpleDataLoader:
        def __init__(self, dataset):
            self.dataset = dataset

        def __iter__(self):
            for i in range(len(self.dataset)):
                yield [self.dataset[i]]

        def __len__(self):
            return len(self.dataset)

    data_loader = SimpleDataLoader(dataset)

    # Determine results file path
    if args.visualize:
        # Visualize mode: skip inference, use provided HDF5 file
        results_file = args.visualize
    else:
        # Inference mode: run process_all_frames and save to results_file
        results_file = args.results_file
        with h5py.File(results_file, "w") as h5f:
            logger.info(f"All frames streamed to: {results_file}")

            score_thresh = 0.4
            filter_score_thresh = 0.35
            miss_tolerance = 5
            track_base = RuntimeTrackerBase(
                score_thresh=score_thresh,
                filter_score_thresh=filter_score_thresh,
                miss_tolerance=miss_tolerance,
            )
            process_all_frames(models, data_loader, track_base, h5f)

    # Visualize results
    visualize_results(
        results_file=results_file, nusc=dataset.nusc, output_dir="vis_output"
    )

    logger.info("Script finished successfully.")


def main():
    # Skip model initialization if only visualizing
    if args.visualize:
        visualize_results(args.visualize)
        return

    # model files check and download
    check_and_download_models(WEIGHT_BEV_ENC_PATH, MODEL_BEV_ENC_PATH, REMOTE_PATH)
    check_and_download_models(
        WEIGHT_TRACK_HEAD_PATH, MODEL_TRACK_HEAD_PATH, REMOTE_PATH
    )
    check_and_download_models(
        WEIGHT_MEMORY_BANK_PATH, MODEL_MEMORY_BANK_PATH, REMOTE_PATH
    )
    check_and_download_models(
        WEIGHT_MEMORY_BANK_UPD_PATH, MODEL_MEMORY_BANK_UPD_PATH, REMOTE_PATH
    )
    check_and_download_models(
        WEIGHT_QUERY_INTERACTION_PATH, MODEL_QUERY_INTERACTION_PATH, REMOTE_PATH
    )
    check_and_download_models(WEIGHT_SEG_HEAD_PATH, MODEL_SEG_HEAD_PATH, REMOTE_PATH)
    check_and_download_models(
        WEIGHT_MOTION_HEAD_PATH, MODEL_MOTION_HEAD_PATH, REMOTE_PATH
    )
    check_and_download_models(WEIGHT_OCC_HEAD_PATH, MODEL_OCC_HEAD_PATH, REMOTE_PATH)
    check_and_download_models(
        WEIGHT_PLANNING_HEAD_PATH, MODEL_PLANNING_HEAD_PATH, REMOTE_PATH
    )
    check_and_download_file(QUERY_EMBEDDING_FILE, REMOTE_PATH)
    check_and_download_file(REFERENCE_POINTS_WEIGHT_FILE, REMOTE_PATH)
    check_and_download_file(REFERENCE_POINTS_BIAS_FILE, REMOTE_PATH)

    env_id = args.env_id

    # initialize
    if not args.onnx:
        bev_encoder = ailia.Net(MODEL_BEV_ENC_PATH, WEIGHT_BEV_ENC_PATH, env_id=env_id)
        track_head = ailia.Net(
            MODEL_TRACK_HEAD_PATH, WEIGHT_TRACK_HEAD_PATH, env_id=env_id
        )
        memory_bank = ailia.Net(
            MODEL_MEMORY_BANK_PATH, WEIGHT_MEMORY_BANK_PATH, env_id=env_id
        )
        memory_bank_update = ailia.Net(
            MODEL_MEMORY_BANK_UPD_PATH, WEIGHT_MEMORY_BANK_UPD_PATH, env_id=env_id
        )
        query_interact = ailia.Net(
            MODEL_QUERY_INTERACTION_PATH, WEIGHT_QUERY_INTERACTION_PATH, env_id=env_id
        )
        seg_head = ailia.Net(MODEL_SEG_HEAD_PATH, WEIGHT_SEG_HEAD_PATH, env_id=env_id)
        motion_head = ailia.Net(
            MODEL_MOTION_HEAD_PATH, WEIGHT_MOTION_HEAD_PATH, env_id=env_id
        )
        occ_head = ailia.Net(MODEL_OCC_HEAD_PATH, WEIGHT_OCC_HEAD_PATH, env_id=env_id)
        planning_head = ailia.Net(
            MODEL_PLANNING_HEAD_PATH, WEIGHT_PLANNING_HEAD_PATH, env_id=env_id
        )
    else:
        import onnxruntime

        providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
        bev_encoder = onnxruntime.InferenceSession(
            WEIGHT_BEV_ENC_PATH, providers=providers
        )
        track_head = onnxruntime.InferenceSession(
            WEIGHT_TRACK_HEAD_PATH, providers=providers
        )
        memory_bank = onnxruntime.InferenceSession(
            WEIGHT_MEMORY_BANK_PATH, providers=providers
        )
        memory_bank_update = onnxruntime.InferenceSession(
            WEIGHT_MEMORY_BANK_UPD_PATH, providers=providers
        )
        query_interact = onnxruntime.InferenceSession(
            WEIGHT_QUERY_INTERACTION_PATH, providers=providers
        )
        seg_head = onnxruntime.InferenceSession(
            WEIGHT_SEG_HEAD_PATH, providers=providers
        )
        motion_head = onnxruntime.InferenceSession(
            WEIGHT_MOTION_HEAD_PATH, providers=providers
        )
        occ_head = onnxruntime.InferenceSession(
            WEIGHT_OCC_HEAD_PATH, providers=providers
        )
        planning_head = onnxruntime.InferenceSession(
            WEIGHT_PLANNING_HEAD_PATH, providers=providers
        )

    models = dict(
        bev_encoder=bev_encoder,
        track_head=track_head,
        memory_bank=memory_bank,
        memory_bank_update=memory_bank_update,
        query_interact=query_interact,
        seg_head=seg_head,
        motion_head=motion_head,
        occ_head=occ_head,
        planning_head=planning_head,
    )
    recognize_from_image(models)


if __name__ == "__main__":
    main()
