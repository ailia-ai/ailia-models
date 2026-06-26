import os
import sys
from collections import OrderedDict
from logging import getLogger

import cv2
import numpy as np
from tqdm import tqdm
from PIL import Image

import ailia

# import original modules
sys.path.append("../../util")
from arg_utils import get_base_parser, update_parser, get_savepath  # noqa
from image_utils import normalize_image
from model_utils import check_and_download_models  # noqa

from kalman_filter import KalmanFilter


logger = getLogger(__name__)


# ======================
# Parameters
# ======================

WEIGHT_BACKBONE_PATH = "backbone.onnx"
WEIGHT_SAM_PATH = "sam_heads.onnx"
WEIGHT_ENC_PATH = "memory_encoder.onnx"
WEIGHT_ATN_PATH = "memory_attention.onnx"
MODEL_BACKBONE_PATH = "backbone.onnx.prototxt"
MODEL_SAM_PATH = "sam_heads.onnx.prototxt"
MODEL_ENC_PATH = "memory_encoder.onnx.prototxt"
MODEL_ATN_PATH = "memory_attention.onnx.prototxt"
REMOTE_PATH = "https://storage.googleapis.com/ailia-models/samurai/"

VIDEO_PATH = "demo.mp4"
SAVE_VIDEO_PATH = "output.mp4"

IMAGE_SIZE = 1024

# a large negative value as a placeholder score for missing objects
NO_OBJ_SCORE = -1024.0

this_dir = os.path.dirname(os.path.abspath(__file__))
weight_dir = os.path.join(this_dir, "weights")


# ======================
# Arguemnt Parser Config
# ======================

parser = get_base_parser("SAMURAI", VIDEO_PATH, SAVE_VIDEO_PATH)
parser.add_argument(
    "--txt_path",
    type=str,
    default="bbox.txt",
    help="the .txt file contains a single line with the bounding box of the first frame in x,y,w,h format",
)
parser.add_argument("--onnx", action="store_true", help="execute onnxruntime version.")
args = update_parser(parser)


# ======================
# Secondary Functions
# ======================


def load_txt(gt_path):
    with open(gt_path, "r") as f:
        gt = f.readlines()
    prompts = {}
    for fid, line in enumerate(gt):
        x, y, w, h = map(float, line.split(","))
        x, y, w, h = int(x), int(y), int(w), int(h)
        prompts[fid] = ((x, y, x + w, y + h), 0)
    return prompts


def load_video_frames(input_path):
    if isinstance(input_path, list):
        loaded_frames = [cv2.imread(frame_path) for frame_path in input_path]
        height, width = loaded_frames[0].shape[:2]
    else:
        cap = cv2.VideoCapture(input_path)
        frame_rate = cap.get(cv2.CAP_PROP_FPS)
        loaded_frames = []
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            loaded_frames.append(frame)
        cap.release()
        height, width = loaded_frames[0].shape[:2]

        if len(loaded_frames) == 0:
            raise ValueError("No frames were loaded from the video.")

    return loaded_frames, height, width


def preprocess_image(img):
    img = img[:, :, ::-1]  # BGR -> RGB

    img = np.array(Image.fromarray(img).resize((IMAGE_SIZE, IMAGE_SIZE)))

    img = normalize_image(img, normalize_type="ImageNet")
    img = img.transpose(2, 0, 1)  # HWC -> CHW
    img = np.expand_dims(img, axis=0)
    img = img.astype(np.float32)

    return img


def select_closest_cond_frames(frame_idx, cond_frame_outputs, max_cond_frame_num):
    """
    Select up to `max_cond_frame_num` conditioning frames from `cond_frame_outputs`
    that are temporally closest to the current frame at `frame_idx`.
    """
    if max_cond_frame_num == -1 or len(cond_frame_outputs) <= max_cond_frame_num:
        selected_outputs = cond_frame_outputs
        unselected_outputs = {}
    else:
        selected_outputs = {}

        # the closest conditioning frame before `frame_idx` (if any)
        idx_before = max((t for t in cond_frame_outputs if t < frame_idx), default=None)
        if idx_before is not None:
            selected_outputs[idx_before] = cond_frame_outputs[idx_before]

        # the closest conditioning frame after `frame_idx` (if any)
        idx_after = min((t for t in cond_frame_outputs if t >= frame_idx), default=None)
        if idx_after is not None:
            selected_outputs[idx_after] = cond_frame_outputs[idx_after]

        # add other temporally closest conditioning frames until reaching a total
        # of `max_cond_frame_num` conditioning frames.
        num_remain = max_cond_frame_num - len(selected_outputs)
        inds_remain = sorted(
            (t for t in cond_frame_outputs if t not in selected_outputs),
            key=lambda x: abs(x - frame_idx),
        )[:num_remain]
        selected_outputs.update((t, cond_frame_outputs[t]) for t in inds_remain)
        unselected_outputs = {
            t: v for t, v in cond_frame_outputs.items() if t not in selected_outputs
        }

    return selected_outputs, unselected_outputs


def get_1d_sine_pe(pos_inds, dim, temperature=10000):
    """
    Get 1D sine positional embedding as in the original Transformer paper.
    """
    pe_dim = dim // 2
    dim_t = np.arange(pe_dim, dtype=np.float32)
    dim_t = temperature ** (2 * (dim_t // 2) / pe_dim)

    pos_embed = np.expand_dims(pos_inds, axis=-1) / dim_t
    pos_embed = np.concatenate([np.sin(pos_embed), np.cos(pos_embed)], axis=-1)
    return pos_embed


# ======================
# Main functions
# ======================


class SAM2VideoPredictor:
    """The predictor class to handle user interactions and manage inference states."""

    def __init__(
        self, backbone, sam_heads, memory_encoder, memory_attn, flg_onnx=False
    ):
        self.backbone = backbone
        self.sam_heads = sam_heads
        self.memory_encoder = memory_encoder
        self.memory_attn = memory_attn
        self.flg_onnx = flg_onnx

        self.num_feature_levels = 3
        self.hidden_dim = 256
        self.maskmem_tpos_enc = np.load(
            os.path.join(weight_dir, "maskmem_tpos_enc.npy")
        )
        self.no_mem_embed = np.load(os.path.join(weight_dir, "no_mem_embed.npy"))

        self.obj_ptr_tpos_proj_weight = np.load(
            os.path.join(weight_dir, "obj_ptr_tpos_proj_weight.npy")
        )
        self.obj_ptr_tpos_proj_bias = np.load(
            os.path.join(weight_dir, "obj_ptr_tpos_proj_bias.npy")
        )

        # Init Kalman Filter
        self.kf = KalmanFilter()
        self.kf_mean = None
        self.kf_covariance = None
        self.stable_frames = 0

        self.stable_frames_threshold = 15

    ### SAM2Base

    def _forward_sam_heads(
        self,
        backbone_features,
        point_inputs,
        mask_inputs,
        high_res_features,
        multimask_output,
    ):
        B = backbone_features.shape[0]

        if point_inputs is not None:
            sam_point_coords = point_inputs["point_coords"]
            sam_point_labels = point_inputs["point_labels"]
        else:
            sam_point_coords = np.zeros((B, 1, 2), dtype=np.float32)
            sam_point_labels = -np.ones((B, 1), dtype=np.int64)

        if mask_inputs is not None:
            sam_mask_prompt = np.zeros((B, 1, 256, 256), dtype=np.float32)
            for i in len(mask_inputs):
                x = cv2.resize(
                    mask_inputs[i].squeeze(0),
                    dsize=(256, 256),
                    interpolation=cv2.INTER_LINEAR,
                )
                sam_mask_prompt[i] = x[np.newaxis, :, :]
        else:
            sam_mask_prompt = np.zeros((0, 1, 256, 256), dtype=np.float32)

        # feedforward
        if not self.flg_onnx:
            output = self.sam_heads.predict(
                [
                    backbone_features,
                    sam_point_coords,
                    sam_point_labels,
                    sam_mask_prompt,
                    high_res_features[0],
                    high_res_features[1],
                    np.array(multimask_output, dtype=np.int64),
                ]
            )
        else:
            output = self.sam_heads.run(
                None,
                {
                    "backbone_features": backbone_features,
                    "point_coords": sam_point_coords,
                    "point_labels": sam_point_labels,
                    "mask_inputs": sam_mask_prompt,
                    "high_res_features_0": high_res_features[0],
                    "high_res_features_1": high_res_features[1],
                    "multimask_output": np.array(multimask_output, dtype=np.int64),
                },
            )
        (
            low_res_multimasks,
            high_res_multimasks,
            ious,
            obj_ptrs,
            object_score_logits,
        ) = output

        obj_ptr = obj_ptrs[:, 0]
        kf_ious = None
        if multimask_output:
            if (
                self.kf_mean is None
                and self.kf_covariance is None
                or self.stable_frames == 0
            ):
                best_iou_inds = np.argmax(ious, axis=-1)
                batch_inds = np.arange(B)
                low_res_masks = np.expand_dims(
                    low_res_multimasks[batch_inds, best_iou_inds], axis=1
                )
                high_res_masks = np.expand_dims(
                    high_res_multimasks[batch_inds, best_iou_inds], axis=1
                )

                non_zero_indices = np.argwhere(high_res_masks[0, 0] > 0.0)
                if len(non_zero_indices) == 0:
                    high_res_bbox = [0, 0, 0, 0]
                else:
                    y_min, x_min = non_zero_indices.min(axis=0)
                    y_max, x_max = non_zero_indices.max(axis=0)
                    high_res_bbox = [x_min, y_min, x_max, y_max]

                self.kf_mean, self.kf_covariance = self.kf.initiate(
                    self.kf.xyxy_to_xyah(high_res_bbox)
                )
                if obj_ptrs.shape[1] > 1:
                    obj_ptr = obj_ptrs[batch_inds, best_iou_inds]
                self.stable_frames += 1
            elif self.stable_frames < self.stable_frames_threshold:
                self.kf_mean, self.kf_covariance = self.kf.predict(
                    self.kf_mean, self.kf_covariance
                )
                best_iou_inds = np.argmax(ious, axis=-1)
                batch_inds = np.arange(B)
                low_res_masks = np.expand_dims(
                    low_res_multimasks[batch_inds, best_iou_inds], axis=1
                )
                high_res_masks = np.expand_dims(
                    high_res_multimasks[batch_inds, best_iou_inds], axis=1
                )
                non_zero_indices = np.argwhere(high_res_masks[0][0] > 0.0)

                if len(non_zero_indices) == 0:
                    high_res_bbox = [0, 0, 0, 0]
                else:
                    y_min, x_min = non_zero_indices.min(axis=0)
                    y_max, x_max = non_zero_indices.max(axis=0)
                    high_res_bbox = [x_min, y_min, x_max, y_max]

                stable_ious_threshold = 0.3
                if ious[0][best_iou_inds] > stable_ious_threshold:
                    self.kf_mean, self.kf_covariance = self.kf.update(
                        self.kf_mean,
                        self.kf_covariance,
                        self.kf.xyxy_to_xyah(high_res_bbox),
                    )
                    self.stable_frames += 1
                else:
                    self.stable_frames = 0
                if obj_ptrs.shape[1] > 1:
                    obj_ptr = obj_ptrs[batch_inds, best_iou_inds]
            else:
                self.kf_mean, self.kf_covariance = self.kf.predict(
                    self.kf_mean, self.kf_covariance
                )
                high_res_multibboxes = []
                batch_inds = np.arange(B)
                for i in range(ious.shape[1]):
                    non_zero_indices = np.argwhere(
                        np.expand_dims(high_res_multimasks[batch_inds, i], axis=1)[0][0]
                        > 0.0
                    )
                    if len(non_zero_indices) == 0:
                        high_res_multibboxes.append([0, 0, 0, 0])
                    else:
                        y_min, x_min = non_zero_indices.min(axis=0)
                        y_max, x_max = non_zero_indices.max(axis=0)
                        high_res_multibboxes.append([x_min, y_min, x_max, y_max])

                # compute the IoU between the predicted bbox and the high_res_multibboxes
                kf_ious = np.array(
                    self.kf.compute_iou(self.kf_mean[:4], high_res_multibboxes)
                )
                # weighted iou
                kf_score_weight = 0.25
                weighted_ious = kf_score_weight * kf_ious + (1 - kf_score_weight) * ious
                best_iou_inds = np.argmax(weighted_ious, axis=-1)
                batch_inds = np.arange(B)
                low_res_masks = np.expand_dims(
                    low_res_multimasks[batch_inds, best_iou_inds], axis=1
                )
                high_res_masks = np.expand_dims(
                    high_res_multimasks[batch_inds, best_iou_inds], axis=1
                )
                if obj_ptrs.shape[1] > 1:
                    obj_ptr = obj_ptrs[batch_inds, best_iou_inds]

                stable_ious_threshold = 0.3
                if ious[0][best_iou_inds] < stable_ious_threshold:
                    self.stable_frames = 0
                else:
                    self.kf_mean, self.kf_covariance = self.kf.update(
                        self.kf_mean,
                        self.kf_covariance,
                        self.kf.xyxy_to_xyah(
                            high_res_multibboxes[best_iou_inds.item()]
                        ),
                    )
        else:
            best_iou_inds = 0
            low_res_masks, high_res_masks = low_res_multimasks, high_res_multimasks

        return (
            low_res_multimasks,
            high_res_multimasks,
            ious,
            low_res_masks,
            high_res_masks,
            obj_ptr,
            object_score_logits,
            ious[0][best_iou_inds],
            kf_ious[best_iou_inds] if kf_ious is not None else None,
        )

    def _prepare_backbone_features(self, backbone_out):
        """Prepare and flatten visual features."""
        backbone_out = backbone_out.copy()

        feature_maps = backbone_out["backbone_fpn"][-self.num_feature_levels :]
        vision_pos_embeds = backbone_out["vision_pos_enc"][-self.num_feature_levels :]

        feat_sizes = [(x.shape[-2], x.shape[-1]) for x in vision_pos_embeds]
        # flatten NxCxHxW to HWxNxC
        vision_feats = [
            x.reshape(x.shape[0], x.shape[1], -1).transpose(2, 0, 1)
            for x in feature_maps
        ]
        vision_pos_embeds = [
            x.reshape(x.shape[0], x.shape[1], -1).transpose(2, 0, 1)
            for x in vision_pos_embeds
        ]

        return backbone_out, vision_feats, vision_pos_embeds, feat_sizes

    def _prepare_memory_conditioned_features(
        self,
        frame_idx,
        is_init_cond_frame,
        current_vision_feats,
        current_vision_pos_embeds,
        feat_sizes,
        output_dict,
        num_frames,
        track_in_reverse=False,  # tracking in reverse time order (for demo usage)
    ):
        """Fuse the current frame's visual feature map with previous memory."""
        B = current_vision_feats[-1].shape[1]  # batch size on this frame
        C = self.hidden_dim
        H, W = feat_sizes[-1]  # top-level (lowest-resolution) feature size

        num_obj_ptr_tokens = 0
        tpos_sign_mul = -1 if track_in_reverse else 1
        # Step 1: condition the visual features of the current frame on previous memories
        if not is_init_cond_frame:
            # Retrieve the memories encoded with the maskmem backbone
            to_cat_memory, to_cat_memory_pos_embed = [], []
            # Select a maximum number of temporally closest cond frames for cross attention
            cond_outputs = output_dict["cond_frame_outputs"]
            max_cond_frames_in_attn = -1
            selected_cond_outputs, unselected_cond_outputs = select_closest_cond_frames(
                frame_idx, cond_outputs, max_cond_frames_in_attn
            )
            t_pos_and_prevs = [(0, out) for out in selected_cond_outputs.values()]

            valid_indices = []
            if frame_idx > 1:  # Ensure we have previous frames to evaluate
                for i in range(
                    frame_idx - 1, 1, -1
                ):  # Iterate backwards through previous frames
                    iou_score = output_dict["non_cond_frame_outputs"][i][
                        "best_iou_score"
                    ]  # Get mask affinity score
                    obj_score = output_dict["non_cond_frame_outputs"][i][
                        "object_score_logits"
                    ]  # Get object score
                    kf_score = (
                        output_dict["non_cond_frame_outputs"][i]["kf_score"]
                        if "kf_score" in output_dict["non_cond_frame_outputs"][i]
                        else None
                    )  # Get motion score if available

                    # Check if the scores meet the criteria for being a valid index
                    memory_bank_iou_threshold = 0.5
                    memory_bank_obj_score_threshold = 0.0
                    memory_bank_kf_score_threshold = 0.0
                    if (
                        iou_score > memory_bank_iou_threshold
                        and obj_score > memory_bank_obj_score_threshold
                        and (
                            kf_score is None
                            or kf_score > memory_bank_kf_score_threshold
                        )
                    ):
                        valid_indices.insert(0, i)
                    # Check the number of valid indices
                    max_obj_ptrs_in_encoder = 16
                    if len(valid_indices) >= max_obj_ptrs_in_encoder - 1:
                        break
            if frame_idx - 1 not in valid_indices:
                valid_indices.append(frame_idx - 1)

            num_maskmem = 7
            for t_pos in range(
                1, num_maskmem
            ):  # Iterate over the number of mask memories
                idx = t_pos - num_maskmem  # Calculate the index for valid indices
                if idx < -len(valid_indices):  # Skip if index is out of bounds
                    continue
                out = output_dict["non_cond_frame_outputs"].get(
                    valid_indices[idx], None
                )  # Get output for the valid index
                if out is None:  # If not found, check unselected outputs
                    out = unselected_cond_outputs.get(valid_indices[idx], None)
                t_pos_and_prevs.append(
                    (t_pos, out)
                )  # Append the temporal position and output to the list

            for t_pos, prev in t_pos_and_prevs:
                if prev is None:
                    continue  # skip padding frames
                feats = prev["maskmem_features"]
                to_cat_memory.append(
                    feats.reshape(feats.shape[0], feats.shape[1], -1).transpose(2, 0, 1)
                )
                # Spatial positional encoding (it might have been offloaded to CPU in eval)
                maskmem_enc = prev["maskmem_pos_enc"][-1]
                maskmem_enc = maskmem_enc.reshape(
                    feats.shape[0], feats.shape[1], -1
                ).transpose(2, 0, 1)
                # Temporal positional encoding
                maskmem_enc = (
                    maskmem_enc + self.maskmem_tpos_enc[num_maskmem - t_pos - 1]
                )
                to_cat_memory_pos_embed.append(maskmem_enc)

            # Construct the list of past object pointers
            max_obj_ptrs_in_encoder = min(num_frames, 16)
            # First add those object pointers from selected conditioning frames
            ptr_cond_outputs = {
                t: out
                for t, out in selected_cond_outputs.items()
                if (t >= frame_idx if track_in_reverse else t <= frame_idx)
            }
            pos_and_ptrs = [
                # Temporal pos encoding contains how far away each pointer is from current frame
                ((frame_idx - t) * tpos_sign_mul, out["obj_ptr"])
                for t, out in ptr_cond_outputs.items()
            ]
            # Add up to (max_obj_ptrs_in_encoder - 1) non-conditioning frames before current frame
            for t_diff in range(1, max_obj_ptrs_in_encoder):
                t = frame_idx + t_diff if track_in_reverse else frame_idx - t_diff
                if t < 0 or (num_frames is not None and t >= num_frames):
                    break
                out = output_dict["non_cond_frame_outputs"].get(
                    t, unselected_cond_outputs.get(t, None)
                )
                if out is not None:
                    pos_and_ptrs.append((t_diff, out["obj_ptr"]))
            # If we have at least one object pointer, add them to the across attention
            mem_dim = 64
            if 0 < len(pos_and_ptrs):
                pos_list, ptrs_list = zip(*pos_and_ptrs)
                # stack object pointers along dim=0 into [ptr_seq_len, B, C] shape
                obj_ptrs = np.stack(ptrs_list, axis=0)
                # a temporal positional embedding based on how far each object pointer is from
                # the current frame (sine embedding normalized by the max pointer num).
                t_diff_max = max_obj_ptrs_in_encoder - 1
                tpos_dim = C
                obj_pos = np.array(pos_list, dtype=np.float32)
                obj_pos = get_1d_sine_pe(obj_pos / t_diff_max, dim=tpos_dim)
                obj_pos = (
                    obj_pos @ self.obj_ptr_tpos_proj_weight.T
                    + self.obj_ptr_tpos_proj_bias
                )
                obj_pos = np.broadcast_to(
                    np.expand_dims(obj_pos, axis=1), (obj_pos.shape[0], B, mem_dim)
                )
                if mem_dim < C:
                    # split a pointer into (C // mem_dim) tokens for mem_dim < C
                    obj_ptrs = obj_ptrs.reshape(-1, B, C // mem_dim, mem_dim)
                    obj_ptrs = obj_ptrs.transpose(0, 2, 1, 3).reshape(
                        -1, obj_ptrs.shape[1], obj_ptrs.shape[3]
                    )
                    obj_pos = np.repeat(obj_pos, repeats=(C // mem_dim), axis=0)

                to_cat_memory.append(obj_ptrs)
                to_cat_memory_pos_embed.append(obj_pos)
                num_obj_ptr_tokens = obj_ptrs.shape[0]
            else:
                num_obj_ptr_tokens = 0
        else:
            # directly add no-mem embedding (instead of using the transformer encoder)
            pix_feat_with_mem = current_vision_feats[-1] + self.no_mem_embed
            pix_feat_with_mem = pix_feat_with_mem.transpose(1, 2, 0).reshape(B, C, H, W)
            return pix_feat_with_mem

        # Step 2: Concatenate the memories and forward through the transformer encoder
        memory = np.concatenate(to_cat_memory, axis=0)
        memory_pos_embed = np.concatenate(to_cat_memory_pos_embed, axis=0)

        pix_feat_with_mem = self.run_memory_attention(
            curr=current_vision_feats,
            curr_pos=current_vision_pos_embeds,
            memory=memory,
            memory_pos=memory_pos_embed,
            num_obj_ptr_tokens=num_obj_ptr_tokens,
        )

        # reshape the output (HW)BC => BCHW
        pix_feat_with_mem = pix_feat_with_mem.transpose(1, 2, 0).reshape(B, C, H, W)
        return pix_feat_with_mem

    def _encode_new_memory(
        self,
        current_vision_feats,
        feat_sizes,
        pred_masks_high_res,
        object_score_logits,
        is_mask_from_pts,
    ):
        current_vision_feats = current_vision_feats[-1]
        feat_sizes = np.array(feat_sizes[-1])
        is_mask_from_pts = np.array(is_mask_from_pts, dtype=np.int64)

        if not self.flg_onnx:
            output = self.memory_encoder.predict(
                [
                    current_vision_feats,
                    feat_sizes,
                    pred_masks_high_res,
                    object_score_logits,
                    is_mask_from_pts,
                ]
            )
        else:
            output = self.memory_encoder.run(
                None,
                {
                    "current_vision_feats": current_vision_feats,
                    "feat_sizes": feat_sizes,
                    "high_res_masks": pred_masks_high_res,
                    "object_score_logits": object_score_logits,
                    "is_mask_from_pts": is_mask_from_pts,
                },
            )
        maskmem_features, maskmem_pos_enc = output

        return maskmem_features, maskmem_pos_enc

    def _track_step(
        self,
        frame_idx,
        is_init_cond_frame,
        current_vision_feats,
        current_vision_pos_embeds,
        feat_sizes,
        point_inputs,
        output_dict,
        num_frames,
        track_in_reverse,
        prev_sam_mask_logits,
    ):
        current_out = {"point_inputs": point_inputs}
        # High-resolution feature maps for the SAM head, reshape (HW)BC => BCHW
        high_res_features = [
            x.transpose(1, 2, 0).reshape(x.shape[1], x.shape[2], *s)
            for x, s in zip(current_vision_feats[:-1], feat_sizes[:-1])
        ]

        # fused the visual feature with previous memory features in the memory bank
        pix_feat = self._prepare_memory_conditioned_features(
            frame_idx=frame_idx,
            is_init_cond_frame=is_init_cond_frame,
            current_vision_feats=current_vision_feats[-1:],
            current_vision_pos_embeds=current_vision_pos_embeds[-1:],
            feat_sizes=feat_sizes[-1:],
            output_dict=output_dict,
            num_frames=num_frames,
            track_in_reverse=track_in_reverse,
        )

        # apply SAM-style segmentation head
        # here we might feed previously predicted low-res SAM mask logits into the SAM mask decoder
        mask_inputs = None
        if prev_sam_mask_logits is not None:
            mask_inputs = prev_sam_mask_logits
        multimask_output = self._use_multimask(point_inputs)

        sam_outputs = self._forward_sam_heads(
            backbone_features=pix_feat,
            point_inputs=point_inputs,
            mask_inputs=mask_inputs,
            high_res_features=high_res_features,
            multimask_output=multimask_output,
        )

        return current_out, sam_outputs, high_res_features, pix_feat

    def _encode_memory_in_output(
        self,
        current_vision_feats,
        feat_sizes,
        point_inputs,
        run_mem_encoder,
        high_res_masks,
        object_score_logits,
        current_out,
    ):
        if run_mem_encoder:
            high_res_masks_for_mem_enc = high_res_masks
            maskmem_features, maskmem_pos_enc = self._encode_new_memory(
                current_vision_feats=current_vision_feats,
                feat_sizes=feat_sizes,
                pred_masks_high_res=high_res_masks_for_mem_enc,
                object_score_logits=object_score_logits,
                is_mask_from_pts=(point_inputs is not None),
            )
            current_out["maskmem_features"] = maskmem_features
            current_out["maskmem_pos_enc"] = [maskmem_pos_enc]
        else:
            current_out["maskmem_features"] = None
            current_out["maskmem_pos_enc"] = None

        return current_out

    def track_step(
        self,
        frame_idx,
        is_init_cond_frame,
        current_vision_feats,
        current_vision_pos_embeds,
        feat_sizes,
        point_inputs,
        output_dict,
        num_frames,
        track_in_reverse=False,  # tracking in reverse time order (for demo usage)
        run_mem_encoder=True,
        prev_sam_mask_logits=None,
    ):
        current_out, sam_outputs, _, _ = self._track_step(
            frame_idx,
            is_init_cond_frame,
            current_vision_feats,
            current_vision_pos_embeds,
            feat_sizes,
            point_inputs,
            output_dict,
            num_frames,
            track_in_reverse,
            prev_sam_mask_logits,
        )

        (
            _,
            _,
            _,
            low_res_masks,
            high_res_masks,
            obj_ptr,
            object_score_logits,
            best_iou_score,
            kf_ious,
        ) = sam_outputs

        current_out["pred_masks"] = low_res_masks
        current_out["pred_masks_high_res"] = high_res_masks
        current_out["obj_ptr"] = obj_ptr
        current_out["best_iou_score"] = best_iou_score
        current_out["kf_ious"] = kf_ious
        current_out["object_score_logits"] = object_score_logits

        # Finally run the memory encoder on the predicted mask to encode
        # it into a new memory feature (that can be used in future frames)
        current_out = self._encode_memory_in_output(
            current_vision_feats,
            feat_sizes,
            point_inputs,
            run_mem_encoder,
            high_res_masks,
            object_score_logits,
            current_out,
        )

        return current_out

    def _use_multimask(self, point_inputs):
        """Whether to use multimask output in the SAM head."""
        num_pts = 0 if point_inputs is None else point_inputs["point_labels"].shape[1]
        multimask_output = 0 <= num_pts <= 1

        return multimask_output

    ### VideoPredictor

    def init_state(self, images, video_height, video_width):
        self.images = images
        self.num_frames = len(images)
        self.video_height = video_height
        self.video_width = video_width

        # inputs on each frame
        self.point_inputs_per_obj = {}
        self.mask_inputs_per_obj = {}
        # visual features on a small number of recently visited frames for quick interactions
        self.cached_features = {}
        # values that don't change across frames (so we only need to hold one copy of them)
        self.constants = {}
        # mapping between client-side object id and model-side object index
        self.obj_id_to_idx = OrderedDict()
        self.obj_idx_to_id = OrderedDict()
        self.obj_ids = []
        # A storage to hold the model's tracking results and states on each frame
        self.output_dict = {
            "cond_frame_outputs": {},  # dict containing {frame_idx: <out>}
            "non_cond_frame_outputs": {},  # dict containing {frame_idx: <out>}
        }
        # Slice (view) of each object tracking results, sharing the same memory with "output_dict"
        self.output_dict_per_obj = {}
        # A temporary storage to hold new outputs when user interact with a frame
        # to add clicks or mask (it's merged into "output_dict" before propagation starts)
        self.temp_output_dict_per_obj = {}
        # Frames that already holds consolidated outputs from click or mask inputs
        # (we directly use their consolidated outputs during tracking)
        self.consolidated_frame_inds = {
            "cond_frame_outputs": set(),  # set containing frame indices
            "non_cond_frame_outputs": set(),  # set containing frame indices
        }
        # metadata for each tracking frame (e.g. which direction it's tracked)
        self.tracking_has_started = False
        self.frames_already_tracked = {}

    def _obj_id_to_idx(self, obj_id):
        """Map client-side object id to model-side object index."""
        obj_idx = self.obj_id_to_idx.get(obj_id, None)
        if obj_idx is not None:
            return obj_idx

        # This is a new object id not sent to the server before. We only allow adding
        # new objects *before* the tracking starts.
        if not self.tracking_has_started:
            # get the next object slot
            obj_idx = len(self.obj_id_to_idx)
            self.obj_id_to_idx[obj_id] = obj_idx
            self.obj_idx_to_id[obj_idx] = obj_id
            self.obj_ids = list(self.obj_id_to_idx)
            # set up input and output structures for this object
            self.point_inputs_per_obj[obj_idx] = {}
            self.mask_inputs_per_obj[obj_idx] = {}
            self.output_dict_per_obj[obj_idx] = {
                "cond_frame_outputs": {},  # dict containing {frame_idx: <out>}
                "non_cond_frame_outputs": {},  # dict containing {frame_idx: <out>}
            }
            self.temp_output_dict_per_obj[obj_idx] = {
                "cond_frame_outputs": {},  # dict containing {frame_idx: <out>}
                "non_cond_frame_outputs": {},  # dict containing {frame_idx: <out>}
            }
            return obj_idx
        else:
            raise RuntimeError(
                f"Cannot add new object id {obj_id} after tracking starts. "
                f"All existing object ids: {self.obj_ids}. "
                f"Please call 'reset_state' to restart from scratch."
            )

    def get_obj_num(self):
        return len(self.obj_idx_to_id)

    def add_new_points_or_box(
        self,
        frame_idx,
        obj_id,
        points=None,
        labels=None,
        clear_old_points=True,
        box=None,
    ):
        """Add new points to a frame."""
        obj_idx = self._obj_id_to_idx(obj_id)
        point_inputs_per_frame = self.point_inputs_per_obj[obj_idx]
        mask_inputs_per_frame = self.mask_inputs_per_obj[obj_idx]

        if points is None:
            points = np.zeros((0, 2), dtype=np.float32)
        else:
            points = np.array(points, dtype=np.float32)
        if labels is None:
            labels = np.zeros(0, dtype=np.int64)
        else:
            labels = np.array(labels, dtype=np.int32)
        if points.ndim == 2:
            points = np.expand_dims(points, axis=0)  # add batch dimension
        if labels.ndim == 1:
            labels = np.expand_dims(labels, axis=0)  # add batch dimension

        # If `box` is provided, we add it as the first two points with labels 2 and 3
        # along with the user-provided points (consistent with how SAM 2 is trained).
        if box is not None:
            if not clear_old_points:
                raise ValueError(
                    "cannot add box without clearing old points, since "
                    "box prompt must be provided before any point prompt "
                    "(please use clear_old_points=True instead)"
                )
            if not isinstance(box, np.ndarray):
                box = np.array(box, dtype=np.float32)
            box_coords = box.reshape(1, 2, 2)
            box_labels = np.array([2, 3], dtype=np.int64)
            box_labels = box_labels.reshape(1, 2)
            points = np.concatenate([box_coords, points], axis=1)
            labels = np.concatenate([box_labels, labels], axis=1)

        points = points / np.array(
            [self.video_width, self.video_height], dtype=np.float32
        )
        # scale the (normalized) coordinates by the model's internal image size
        points = points * IMAGE_SIZE

        if not clear_old_points:
            point_inputs = point_inputs_per_frame.get(frame_idx, None)
        else:
            point_inputs = None
        if point_inputs is not None:
            points = np.concatenate([point_inputs["point_coords"], points], axis=1)
            labels = np.concatenate([point_inputs["point_labels"], labels], axis=1)
        point_inputs = {"point_coords": points, "point_labels": labels}
        point_inputs_per_frame[frame_idx] = point_inputs
        mask_inputs_per_frame.pop(frame_idx, None)

        is_init_cond_frame = frame_idx not in self.frames_already_tracked
        # whether to track in reverse time order
        if is_init_cond_frame:
            reverse = False
        else:
            reverse = self.frames_already_tracked[frame_idx]["reverse"]

        obj_output_dict = self.output_dict_per_obj[obj_idx]
        obj_temp_output_dict = self.temp_output_dict_per_obj[obj_idx]
        # Add a frame to conditioning output if it's an initial conditioning frame or
        # if the model sees all frames receiving clicks/mask as conditioning frames.
        is_cond = is_init_cond_frame
        storage_key = "cond_frame_outputs" if is_cond else "non_cond_frame_outputs"

        # Get any previously predicted mask logits on this object and feed it along with
        # the new clicks into the SAM mask decoder.
        prev_sam_mask_logits = None
        # lookup temporary output dict first, which contains the most recent output
        # (if not found, then lookup conditioning and non-conditioning frame output)
        prev_out = obj_temp_output_dict[storage_key].get(frame_idx)
        if prev_out is None:
            prev_out = obj_output_dict["cond_frame_outputs"].get(frame_idx)
            if prev_out is None:
                prev_out = obj_output_dict["non_cond_frame_outputs"].get(frame_idx)

        if prev_out is not None and prev_out["pred_masks"] is not None:
            prev_sam_mask_logits = prev_out["pred_masks"]
            # Clamp the scale of prev_sam_mask_logits to avoid rare numerical issues.
            prev_sam_mask_logits = np.clip(prev_sam_mask_logits, -32.0, 32.0)

        current_out, _ = self.single_frame_inference(
            output_dict=obj_output_dict,  # run on the slice of a single object
            frame_idx=frame_idx,
            batch_size=1,  # run on the slice of a single object
            is_init_cond_frame=is_init_cond_frame,
            point_inputs=point_inputs,
            reverse=reverse,
            # Skip the memory encoder when adding clicks or mask.
            run_mem_encoder=False,
            prev_sam_mask_logits=prev_sam_mask_logits,
        )
        # Add the output to the output dict (to be used as future memory)
        obj_temp_output_dict[storage_key][frame_idx] = current_out

        # Resize the output mask to the original video resolution
        obj_ids = self.obj_ids
        consolidated_out = self.consolidate_temp_output_across_obj(
            frame_idx,
            is_cond=is_cond,
            run_mem_encoder=False,
            consolidate_at_video_res=True,
        )
        _, video_res_masks = self.get_orig_video_res_output(
            consolidated_out["pred_masks_video_res"]
        )

        return frame_idx, obj_ids, video_res_masks

    def get_orig_video_res_output(self, any_res_masks):
        """
        Resize the object scores to the original video resolution (video_res_masks)
        and apply non-overlapping constraints for final output.
        """
        video_H = self.video_height
        video_W = self.video_width
        if any_res_masks.shape[-2:] == (video_H, video_W):
            video_res_masks = any_res_masks
        else:
            video_res_masks = cv2.resize(
                any_res_masks.squeeze(0).squeeze(0),
                dsize=(video_W, video_H),
                interpolation=cv2.INTER_LINEAR,
            )
            video_res_masks = video_res_masks[np.newaxis, np.newaxis, :, :]

        return any_res_masks, video_res_masks

    def consolidate_temp_output_across_obj(
        self,
        frame_idx,
        is_cond,
        run_mem_encoder,
        consolidate_at_video_res=False,
    ):
        batch_size = self.get_obj_num()
        storage_key = "cond_frame_outputs" if is_cond else "non_cond_frame_outputs"
        # Optionally, we allow consolidating the temporary outputs at the original
        # video resolution (to provide a better editing experience for mask prompts).
        if consolidate_at_video_res:
            consolidated_H = self.video_height
            consolidated_W = self.video_width
            consolidated_mask_key = "pred_masks_video_res"
        else:
            consolidated_H = consolidated_W = IMAGE_SIZE // 4
            consolidated_mask_key = "pred_masks"

        # Initialize `consolidated_out`.
        consolidated_out = {
            "maskmem_features": None,
            "maskmem_pos_enc": None,
            consolidated_mask_key: np.full(
                shape=(batch_size, 1, consolidated_H, consolidated_W),
                fill_value=NO_OBJ_SCORE,
                dtype=np.float32,
            ),
            "obj_ptr": np.full(
                shape=(batch_size, self.hidden_dim),
                fill_value=NO_OBJ_SCORE,
                dtype=np.float32,
            ),
            "object_score_logits": np.full(
                shape=(batch_size, 1),
                # default to 10.0 for object_score_logits, i.e. assuming the object is
                # present as sigmoid(10)=1, same as in `predict_masks` of `MaskDecoder`
                fill_value=10.0,
                dtype=np.float32,
            ),
        }

        for obj_idx in range(batch_size):
            obj_temp_output_dict = self.temp_output_dict_per_obj[obj_idx]
            obj_output_dict = self.output_dict_per_obj[obj_idx]
            out = obj_temp_output_dict[storage_key].get(frame_idx, None)

            if out is None:
                out = obj_output_dict["cond_frame_outputs"].get(frame_idx, None)
            if out is None:
                out = obj_output_dict["non_cond_frame_outputs"].get(frame_idx, None)
            if out is None:
                continue

            # Add the temporary object output mask to consolidated output mask
            obj_mask = out["pred_masks"]
            consolidated_pred_masks = consolidated_out[consolidated_mask_key]
            if obj_mask.shape[-2:] == consolidated_pred_masks.shape[-2:]:
                consolidated_pred_masks[obj_idx : obj_idx + 1] = obj_mask
            else:
                target_h, target_w = consolidated_pred_masks.shape[-2:]
                resized_obj_mask = cv2.resize(
                    obj_mask.squeeze(0).squeeze(0),
                    dsize=(target_w, target_h),
                    interpolation=cv2.INTER_LINEAR,
                )
                resized_obj_mask = resized_obj_mask[np.newaxis, np.newaxis, :, :]
                consolidated_pred_masks[obj_idx : obj_idx + 1] = resized_obj_mask

            consolidated_out["obj_ptr"][obj_idx : obj_idx + 1] = out["obj_ptr"]
            consolidated_out["object_score_logits"][obj_idx : obj_idx + 1] = out[
                "object_score_logits"
            ]

        # Optionally, apply non-overlapping constraints on the consolidated scores
        # and rerun the memory encoder
        if run_mem_encoder:
            high_res_masks = cv2.resize(
                consolidated_out["pred_masks"].squeeze(0).squeeze(0),
                dsize=(IMAGE_SIZE, IMAGE_SIZE),
                interpolation=cv2.INTER_LINEAR,
            )
            high_res_masks = high_res_masks[np.newaxis, np.newaxis, :, :]

            maskmem_features, maskmem_pos_enc = self.run_memory_encoder(
                frame_idx=frame_idx,
                batch_size=batch_size,
                high_res_masks=high_res_masks,
                object_score_logits=consolidated_out["object_score_logits"],
                is_mask_from_pts=True,  # these frames are what the user interacted with
            )
            consolidated_out["maskmem_features"] = maskmem_features
            consolidated_out["maskmem_pos_enc"] = maskmem_pos_enc

        return consolidated_out

    def preflight(self):
        """Prepare inference_state and consolidate temporary outputs before tracking."""
        # Tracking has started and we don't allow adding new objects until session is reset.
        self.tracking_has_started = True

        temp_output_dict_per_obj = self.temp_output_dict_per_obj
        output_dict = self.output_dict
        consolidated_frame_inds = self.consolidated_frame_inds
        for is_cond in [False, True]:
            # Separately consolidate conditioning and non-conditioning temp outputs
            storage_key = "cond_frame_outputs" if is_cond else "non_cond_frame_outputs"

            temp_frame_inds = set()
            for obj_temp_output_dict in self.temp_output_dict_per_obj.values():
                temp_frame_inds.update(obj_temp_output_dict[storage_key].keys())
            consolidated_frame_inds[storage_key].update(temp_frame_inds)

            # consolidate the temporary output across all objects on this frame
            for frame_idx in temp_frame_inds:
                consolidated_out = self.consolidate_temp_output_across_obj(
                    frame_idx, is_cond=is_cond, run_mem_encoder=True
                )
                # merge them into "output_dict" and also create per-object slices
                output_dict[storage_key][frame_idx] = consolidated_out
                self.add_output_per_object(frame_idx, consolidated_out, storage_key)

            # clear temporary outputs in `temp_output_dict_per_obj`
            for obj_temp_output_dict in temp_output_dict_per_obj.values():
                obj_temp_output_dict[storage_key].clear()

        # edge case: if an output is added to "cond_frame_outputs", we remove any prior
        # output on the same frame in "non_cond_frame_outputs"
        for frame_idx in output_dict["cond_frame_outputs"]:
            output_dict["non_cond_frame_outputs"].pop(frame_idx, None)
        for obj_output_dict in self.output_dict_per_obj.values():
            for frame_idx in obj_output_dict["cond_frame_outputs"]:
                obj_output_dict["non_cond_frame_outputs"].pop(frame_idx, None)
        for frame_idx in consolidated_frame_inds["cond_frame_outputs"]:
            assert frame_idx in output_dict["cond_frame_outputs"]
            consolidated_frame_inds["non_cond_frame_outputs"].discard(frame_idx)

    def propagate_in_video(self):
        self.preflight()

        output_dict = self.output_dict
        consolidated_frame_inds = self.consolidated_frame_inds
        obj_ids = self.obj_ids
        num_frames = self.num_frames
        batch_size = self.get_obj_num()

        if len(output_dict["cond_frame_outputs"]) == 0:
            raise RuntimeError("No points are provided; please add points first")

        start_frame_idx = min(output_dict["cond_frame_outputs"])
        max_frame_num_to_track = num_frames
        end_frame_idx = min(start_frame_idx + max_frame_num_to_track, num_frames - 1)
        processing_order = range(start_frame_idx, end_frame_idx + 1)

        for frame_idx in tqdm(processing_order, desc="propagate in video"):
            if frame_idx in consolidated_frame_inds["cond_frame_outputs"]:
                storage_key = "cond_frame_outputs"
                current_out = output_dict[storage_key][frame_idx]
                pred_masks = current_out["pred_masks"]
            elif frame_idx in consolidated_frame_inds["non_cond_frame_outputs"]:
                storage_key = "non_cond_frame_outputs"
                current_out = output_dict[storage_key][frame_idx]
                pred_masks = current_out["pred_masks"]
            else:
                storage_key = "non_cond_frame_outputs"
                current_out, pred_masks = self.single_frame_inference(
                    output_dict=output_dict,
                    frame_idx=frame_idx,
                    batch_size=batch_size,
                    is_init_cond_frame=False,
                    point_inputs=None,
                    reverse=False,
                    run_mem_encoder=True,
                )
                output_dict[storage_key][frame_idx] = current_out

            # Create slices of per-object outputs for subsequent interaction with each
            # individual object after tracking.
            self.add_output_per_object(frame_idx, current_out, storage_key)
            self.frames_already_tracked[frame_idx] = {"reverse": False}

            _, video_res_masks = self.get_orig_video_res_output(pred_masks)

            yield frame_idx, obj_ids, video_res_masks

    def add_output_per_object(self, frame_idx, current_out, storage_key):
        """
        Split a multi-object output into per-object output slices and add them into
        `output_dict_per_obj`. The resulting slices share the same tensor storage.
        """
        maskmem_features = current_out["maskmem_features"]
        maskmem_pos_enc = current_out["maskmem_pos_enc"]

        for obj_idx, obj_output_dict in self.output_dict_per_obj.items():
            obj_slice = slice(obj_idx, obj_idx + 1)
            obj_out = {
                "maskmem_features": None,
                "maskmem_pos_enc": None,
                "pred_masks": current_out["pred_masks"][obj_slice],
                "obj_ptr": current_out["obj_ptr"][obj_slice],
                "object_score_logits": current_out["object_score_logits"][obj_slice],
            }
            if maskmem_features is not None:
                obj_out["maskmem_features"] = maskmem_features[obj_slice]
            if maskmem_pos_enc is not None:
                obj_out["maskmem_pos_enc"] = [x[obj_slice] for x in maskmem_pos_enc]
            obj_output_dict[storage_key][frame_idx] = obj_out

    def get_image_feature(self, frame_idx, batch_size):
        """Compute the image features on a given frame."""
        # Look up in the cache first
        image, backbone_out = self.cached_features.get(frame_idx, (None, None))
        if backbone_out is None:
            image = preprocess_image(self.images[frame_idx])

            # feedforward
            if not self.flg_onnx:
                output = self.backbone.predict([image])
            else:
                output = self.backbone.run(None, {"image": image})
            (
                vision_features,
                vision_pos_enc_0,
                vision_pos_enc_1,
                vision_pos_enc_2,
                backbone_fpn_0,
                backbone_fpn_1,
                backbone_fpn_2,
            ) = output
            backbone_out = dict(
                vision_features=vision_features,
                vision_pos_enc=[vision_pos_enc_0, vision_pos_enc_1, vision_pos_enc_2],
                backbone_fpn=[backbone_fpn_0, backbone_fpn_1, backbone_fpn_2],
            )

            # Cache the most recent frame's feature (for repeated interactions with
            # a frame; we can use an LRU cache for more frames in the future).
            self.cached_features = {frame_idx: (image, backbone_out)}

        # expand the features to have the same dimension as the number of objects
        expanded_image = np.broadcast_to(image, (batch_size, *image.shape[1:]))
        expanded_backbone_out = {
            "backbone_fpn": backbone_out["backbone_fpn"].copy(),
            "vision_pos_enc": backbone_out["vision_pos_enc"].copy(),
        }
        for i, feat in enumerate(expanded_backbone_out["backbone_fpn"]):
            expanded_backbone_out["backbone_fpn"][i] = np.broadcast_to(
                feat, (batch_size, *feat.shape[1:])
            )
        for i, pos in enumerate(expanded_backbone_out["vision_pos_enc"]):
            pos = np.broadcast_to(pos, (batch_size, *pos.shape[1:]))
            expanded_backbone_out["vision_pos_enc"][i] = pos

        features = self._prepare_backbone_features(expanded_backbone_out)
        features = (expanded_image,) + features
        return features

    def single_frame_inference(
        self,
        output_dict,
        frame_idx,
        batch_size,
        is_init_cond_frame,
        point_inputs,
        reverse,
        run_mem_encoder,
        prev_sam_mask_logits=None,
    ):
        """Run tracking on a single frame based on current inputs and previous memory."""
        # Retrieve correct image features
        (
            _,
            _,
            current_vision_feats,
            current_vision_pos_embeds,
            feat_sizes,
        ) = self.get_image_feature(frame_idx, batch_size)

        # point and mask should not appear as input simultaneously on the same frame
        current_out = self.track_step(
            frame_idx=frame_idx,
            is_init_cond_frame=is_init_cond_frame,
            current_vision_feats=current_vision_feats,
            current_vision_pos_embeds=current_vision_pos_embeds,
            feat_sizes=feat_sizes,
            point_inputs=point_inputs,
            output_dict=output_dict,
            num_frames=self.num_frames,
            track_in_reverse=reverse,
            run_mem_encoder=run_mem_encoder,
            prev_sam_mask_logits=prev_sam_mask_logits,
        )

        maskmem_features = current_out["maskmem_features"]
        pred_masks = current_out["pred_masks"]  # (B, 1, H, W)
        # "maskmem_pos_enc" is the same across frames, so we only need to store one copy of it
        maskmem_pos_enc = self.get_maskmem_pos_enc(current_out)

        # object pointer is a small tensor, so we always keep it on GPU memory for fast access
        obj_ptr = current_out["obj_ptr"]
        object_score_logits = current_out["object_score_logits"]
        best_iou_score = current_out["best_iou_score"]
        best_kf_score = current_out["kf_ious"]
        # make a compact version of this frame's output to reduce the state size
        compact_current_out = {
            "maskmem_features": maskmem_features,  # (B, C, H, W)
            "maskmem_pos_enc": maskmem_pos_enc,
            "pred_masks": pred_masks,
            "obj_ptr": obj_ptr,
            "object_score_logits": object_score_logits,
            "best_iou_score": best_iou_score,
            "kf_score": best_kf_score,
        }

        return compact_current_out, pred_masks

    def run_memory_encoder(
        self,
        frame_idx,
        batch_size,
        high_res_masks,
        object_score_logits,
        is_mask_from_pts,
    ):
        # Retrieve correct image features
        _, _, current_vision_feats, _, feat_sizes = self.get_image_feature(
            frame_idx, batch_size
        )

        maskmem_features, maskmem_pos_enc = self._encode_new_memory(
            current_vision_feats=current_vision_feats,
            feat_sizes=feat_sizes,
            pred_masks_high_res=high_res_masks,
            object_score_logits=object_score_logits,
            is_mask_from_pts=is_mask_from_pts,
        )
        maskmem_pos_enc = [maskmem_pos_enc]

        # "maskmem_pos_enc" is the same across frames, so we only need to store one copy of it
        maskmem_pos_enc = self.get_maskmem_pos_enc({"maskmem_pos_enc": maskmem_pos_enc})

        return maskmem_features, maskmem_pos_enc

    def run_memory_attention(
        self,
        curr,  # self-attention inputs
        curr_pos,  # pos_enc for self-attention inputs
        memory,  # cross-attention inputs
        memory_pos,  # pos_enc for cross-attention inputs
        num_obj_ptr_tokens,  # number of object pointer *tokens*
    ):
        num_obj_ptr_tokens = np.array(num_obj_ptr_tokens, dtype=int)
        if not self.flg_onnx:
            output = self.memory_attn.predict(
                [
                    curr[0],
                    memory,
                    curr_pos[0],
                    memory_pos,
                    num_obj_ptr_tokens,
                ]
            )
        else:
            output = self.memory_attn.run(
                None,
                {
                    "curr": curr[0],
                    "memory": memory,
                    "curr_pos": curr_pos[0],
                    "memory_pos": memory_pos,
                    "num_obj_ptr_tokens": num_obj_ptr_tokens,
                },
            )
        pix_feat_with_mem = output[0]

        return pix_feat_with_mem

    def get_maskmem_pos_enc(self, current_out):
        """
        `maskmem_pos_enc` is the same across frames and objects, so we cache it as
        a constant in the inference session to reduce session storage size.
        """
        model_constants = self.constants

        # "out_maskmem_pos_enc" should be either a list of tensors or None
        out_maskmem_pos_enc = current_out["maskmem_pos_enc"]
        if out_maskmem_pos_enc is not None:
            if "maskmem_pos_enc" not in model_constants:
                # only take the slice for one object, since it's same across objects
                maskmem_pos_enc = [x[0:1].copy() for x in out_maskmem_pos_enc]
                model_constants["maskmem_pos_enc"] = maskmem_pos_enc
            else:
                maskmem_pos_enc = model_constants["maskmem_pos_enc"]
            # expand the cached maskmem_pos_enc to the actual batch size
            batch_size = out_maskmem_pos_enc[0].shape[0]
            expanded_maskmem_pos_enc = [
                np.broadcast_to(x, (batch_size, *x.shape[1:])) for x in maskmem_pos_enc
            ]
        else:
            expanded_maskmem_pos_enc = None

        return expanded_maskmem_pos_enc


def recognize_from_video(predictor: SAM2VideoPredictor):
    input = (
        args.video
        if args.video
        else args.input[0] if len(args.input) <= 1 else args.input
    )
    txt_path = args.txt_path
    prompts = load_txt(txt_path)
    images, video_height, video_width = load_video_frames(input)

    predictor.init_state(images, video_height, video_width)

    bbox, track_label = prompts[0]
    _, _, masks = predictor.add_new_points_or_box(box=bbox, frame_idx=0, obj_id=0)

    # saves the result as a video file.
    video_output_path = get_savepath(args.savepath, SAVE_VIDEO_PATH, ext=".mp4")
    frame_rate = 30
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    video_out = cv2.VideoWriter(
        video_output_path, fourcc, frame_rate, (video_width, video_height)
    )
    color = [(255, 0, 0)]

    # loop through the video frames and propagate the masks
    for frame_idx, object_ids, masks in predictor.propagate_in_video():
        mask_to_vis = {}
        bbox_to_vis = {}

        for obj_id, mask in zip(object_ids, masks):
            mask = mask[0]
            mask = mask > 0.0
            non_zero_indices = np.argwhere(mask)
            if len(non_zero_indices) == 0:
                bbox = [0, 0, 0, 0]
            else:
                y_min, x_min = non_zero_indices.min(axis=0).tolist()
                y_max, x_max = non_zero_indices.max(axis=0).tolist()
                bbox = [x_min, y_min, x_max - x_min, y_max - y_min]
            bbox_to_vis[obj_id] = bbox
            mask_to_vis[obj_id] = mask

        img = images[frame_idx]
        for obj_id, mask in mask_to_vis.items():
            mask_img = np.zeros((video_height, video_width, 3), np.uint8)
            mask_img[mask] = color[(obj_id + 1) % len(color)]
            img = cv2.addWeighted(img, 1, mask_img, 0.2, 0)

        for obj_id, bbox in bbox_to_vis.items():
            cv2.rectangle(
                img,
                (bbox[0], bbox[1]),
                (bbox[0] + bbox[2], bbox[1] + bbox[3]),
                color[obj_id % len(color)],
                2,
            )

        video_out.write(img)

    video_out.release()

    logger.info(f"saved at : {video_output_path}")

    logger.info("Script finished successfully.")


def main():
    # model files check and download
    check_and_download_models(WEIGHT_BACKBONE_PATH, MODEL_BACKBONE_PATH, REMOTE_PATH)
    check_and_download_models(WEIGHT_SAM_PATH, MODEL_SAM_PATH, REMOTE_PATH)
    check_and_download_models(WEIGHT_ENC_PATH, MODEL_ENC_PATH, REMOTE_PATH)
    check_and_download_models(WEIGHT_ATN_PATH, MODEL_ATN_PATH, REMOTE_PATH)

    env_id = args.env_id

    # initialize
    if not args.onnx:
        backbone = ailia.Net(MODEL_BACKBONE_PATH, WEIGHT_BACKBONE_PATH, env_id=env_id)
        sam_heads = ailia.Net(MODEL_SAM_PATH, WEIGHT_SAM_PATH, env_id=env_id)
        memory_encoder = ailia.Net(MODEL_ENC_PATH, WEIGHT_ENC_PATH, env_id=env_id)
        memory_attn = ailia.Net(MODEL_ATN_PATH, WEIGHT_ATN_PATH, env_id=env_id)
    else:
        import onnxruntime

        providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
        # providers = ["CPUExecutionProvider"]
        backbone = onnxruntime.InferenceSession(
            WEIGHT_BACKBONE_PATH, providers=providers
        )
        sam_heads = onnxruntime.InferenceSession(WEIGHT_SAM_PATH, providers=providers)
        memory_encoder = onnxruntime.InferenceSession(
            WEIGHT_ENC_PATH, providers=providers
        )
        memory_attn = onnxruntime.InferenceSession(WEIGHT_ATN_PATH, providers=providers)

    predictor = SAM2VideoPredictor(
        backbone=backbone,
        sam_heads=sam_heads,
        memory_encoder=memory_encoder,
        memory_attn=memory_attn,
        flg_onnx=args.onnx,
    )

    recognize_from_video(predictor)


if __name__ == "__main__":
    main()
