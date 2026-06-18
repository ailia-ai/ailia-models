import sys
import os
import time

import numpy as np
import cv2

import os
import numpy as np
from typing import Optional
from typing import Tuple

class SAM2ImagePredictor:
    def __init__(self, legacy=False, version="2", model_type="hiera_l"):
        self.legacy = legacy
        self.no_mem_embed = self._load_no_mem_embed(version, model_type)

    def _load_no_mem_embed(self, version, model_type):
        """Load pretrained no_mem_embed from npz file."""
        npz_path = os.path.join(os.path.dirname(__file__), "pretrained_weights.npz")
        key = f"v{version}_{model_type}_no_mem_embed".replace("+", "plus")
        data = np.load(npz_path)
        return data[key].astype(np.float32)  # (1, 1, 256)

    def set_image(self, image, image_encoder, onnx):
        image = np.expand_dims(image, axis=0).astype(np.float32)

        if onnx:
            vision_features, vision_pos_enc_0, vision_pos_enc_1, vision_pos_enc_2, backbone_fpn_0, backbone_fpn_1, backbone_fpn_2 = image_encoder.run(None, {"input_image":image})
        else:
            vision_features, vision_pos_enc_0, vision_pos_enc_1, vision_pos_enc_2, backbone_fpn_0, backbone_fpn_1, backbone_fpn_2 = image_encoder.run({"input_image":image})

        backbone_out = {"vision_features":vision_features,
                        "vision_pos_enc":[vision_pos_enc_0, vision_pos_enc_1, vision_pos_enc_2],
                        "backbone_fpn":[backbone_fpn_0,backbone_fpn_1, backbone_fpn_2]}

        _, vision_feats, _, _ = self._prepare_backbone_features(backbone_out)
        # Add no_mem_embed, which is added to the lowest rest feat. map during training on videos
        vision_feats[-1] = vision_feats[-1] + self.no_mem_embed

        bb_feat_sizes = [
            (256, 256),
            (128, 128),
            (64, 64),
        ]

        feats = [
            np.transpose(feat, (1, 2, 0)).reshape(1, -1, *feat_size)
            for feat, feat_size in zip(vision_feats[::-1], bb_feat_sizes[::-1])
        ][::-1]

        features = {"image_embed": feats[-1], "high_res_feats": feats[:-1]}
        return features

    def _prepare_backbone_features(self, backbone_out):
        """Prepare and flatten visual features."""
        backbone_out = backbone_out.copy()
        num_feature_levels = 3

        feature_maps = backbone_out["backbone_fpn"][-num_feature_levels :]
        vision_pos_embeds = backbone_out["vision_pos_enc"][-num_feature_levels :]

        feat_sizes = [(x.shape[-2], x.shape[-1]) for x in vision_pos_embeds]
        # flatten NxCxHxW to HWxNxC
        vision_feats = [np.transpose(x.reshape(x.shape[0], x.shape[1], -1), (2, 0, 1)) for x in feature_maps]
        vision_pos_embeds = [np.transpose(x.reshape(x.shape[0], x.shape[1], -1), (2, 0, 1)) for x in vision_pos_embeds]

        return backbone_out, vision_feats, vision_pos_embeds, feat_sizes

    def predict(
        self,
        features,
        orig_hw,
        point_coords: Optional[np.ndarray] = None,
        point_labels: Optional[np.ndarray] = None,
        box: Optional[np.ndarray] = None,
        mask_input: Optional[np.ndarray] = None,
        prompt_encoder = None,
        mask_decoder = None,
        onnx = False
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        # Transform input prompts
        mask_input, unnorm_coords, labels, unnorm_box = self._prep_prompts(
        point_coords, point_labels, box, mask_input, orig_hw
        )

        masks, iou_predictions, low_res_masks = self._predict(
            features,
            orig_hw,
            unnorm_coords,
            labels,
            unnorm_box,
            mask_input,
            prompt_encoder=prompt_encoder,
            mask_decoder=mask_decoder,
            onnx=onnx
        )

        return masks[0], iou_predictions[0], low_res_masks[0]

    def _prep_prompts(
        self, point_coords, point_labels, box, mask_logits, orig_hw
    ):

        unnorm_coords, labels, unnorm_box, mask_input = None, None, None, None
        if point_coords is not None and len(point_coords) != 0:
            point_coords = point_coords.astype(np.float32)
            unnorm_coords = self.transform_coords(
                point_coords, orig_hw=orig_hw
            )
            labels = point_labels.astype(np.int64)
            if len(unnorm_coords.shape) == 2:
                unnorm_coords, labels = unnorm_coords[None, ...], labels[None, ...]
        if box is not None:
            box = box.astype(np.float32)
            unnorm_box = self.transform_boxes(
                box, orig_hw=orig_hw
            )  # Bx2x2
        if mask_logits is not None:
            mask_input = mask_logits.astype(np.float32)
            if len(mask_input.shape) == 3:
                mask_input = mask_input[None, :, :, :]
        return mask_input, unnorm_coords, labels, unnorm_box

    def _predict(
        self, 
        features,
        orig_hw,
        point_coords: Optional[np.ndarray],
        point_labels: Optional[np.ndarray],
        boxes: Optional[np.ndarray] = None,
        mask_input: Optional[np.ndarray] = None,
        prompt_encoder = None,
        mask_decoder = None,
        onnx = False
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        if point_coords is not None:
            concat_points = (point_coords, point_labels.astype(np.int32))
        else:
            concat_points = None

        # Embed prompts
        if boxes is not None:
            box_coords = boxes.reshape(-1, 2, 2)
            box_labels = np.array([[2, 3]])
            box_labels = box_labels.repeat(boxes.shape[0], 1)
            # we merge "boxes" and "points" into a single "concat_points" input (where
            # boxes are added at the beginning) to sam_prompt_encoder
            if concat_points is not None:
                concat_coords = np.concatenate([box_coords, concat_points[0]], axis=1)
                concat_labels = np.concatenate([box_labels, concat_points[1]], axis=1)
                concat_points = (concat_coords, concat_labels.astype(np.int32))
            else:
                concat_points = (box_coords, box_labels.astype(np.int32))

        if mask_input is None:
            if self.legacy:
                mask_input_dummy = np.zeros((1, 256, 256), dtype=np.float32)
            else:
                mask_input_dummy = np.zeros((1, 1, 256, 256), dtype=np.float32)
            masks_enable = np.array([0], dtype=np.int32)
        else:
            mask_input_dummy = mask_input
            masks_enable = np.array([1], dtype=np.int32)

        if concat_points is None:
            raise("concat_points must be exists")

        if onnx:
            sparse_embeddings, dense_embeddings, dense_pe = prompt_encoder.run(None, {"coords":concat_points[0], "labels":concat_points[1], "masks":mask_input_dummy, "masks_enable":masks_enable})
        else:
            sparse_embeddings, dense_embeddings, dense_pe = prompt_encoder.run({"coords":concat_points[0], "labels":concat_points[1], "masks":mask_input_dummy, "masks_enable":masks_enable})

        # Predict masks
        high_res_features = [
            feat_level
            for feat_level in features["high_res_feats"]
        ]

        image_feature = features["image_embed"]
        if onnx:
            masks, iou_pred, sam_tokens_out, object_score_logits   = mask_decoder.run(None, {
                "image_embeddings":image_feature,
                "image_pe": dense_pe,
                "sparse_prompt_embeddings": sparse_embeddings,
                "dense_prompt_embeddings": dense_embeddings,
                "high_res_features1":high_res_features[0],
                "high_res_features2":high_res_features[1]})
        else:
            masks, iou_pred, sam_tokens_out, object_score_logits  = mask_decoder.run({
                "image_embeddings":image_feature,
                "image_pe": dense_pe,
                "sparse_prompt_embeddings": sparse_embeddings,
                "dense_prompt_embeddings": dense_embeddings,
                "high_res_features1":high_res_features[0],
                "high_res_features2":high_res_features[1]})

        low_res_masks, iou_predictions, _, _  = self.forward_postprocess(masks, iou_pred, sam_tokens_out, object_score_logits)

        # Upscale the masks to the original image resolution
        masks = self.postprocess_masks(
            low_res_masks, orig_hw
        )
        low_res_masks = np.clip(low_res_masks, -32.0, 32.0)
        mask_threshold = 0.0
        masks = masks > mask_threshold

        return masks, iou_predictions, low_res_masks

    def forward_postprocess(
        self,
        masks,
        iou_pred,
        mask_tokens_out,
        object_score_logits,
    ):
        masks = masks[:, 1:, :, :]
        iou_pred = iou_pred[:, 1:]
        sam_tokens_out = mask_tokens_out[:, 1:]  # [b, 3, c] shape # use_multimask_token_for_obj_ptr
        return masks, iou_pred, sam_tokens_out, object_score_logits

    def transform_coords(
        self, coords, orig_hw
    ):
        h, w = orig_hw
        coords = coords.copy()
        coords[..., 0] = coords[..., 0] / w
        coords[..., 1] = coords[..., 1] / h

        resolution = 1024
        coords = coords * resolution  # unnormalize coords
        return coords

    def transform_boxes(
        self, boxes, orig_hw=None
    ):
        boxes = self.transform_coords(boxes.reshape(-1, 2, 2), orig_hw)
        return boxes

    def postprocess_masks(self, masks: np.ndarray, orig_hw) -> np.ndarray:
        interpolated_masks = []
        for mask in masks:
            mask = np.transpose(mask, (1, 2, 0))
            resized_mask = cv2.resize(mask, (orig_hw[1], orig_hw[0]), interpolation=cv2.INTER_LINEAR)
            resized_mask = np.transpose(resized_mask, (2, 0, 1))
            interpolated_masks.append(resized_mask)
        interpolated_masks = np.array(interpolated_masks)

        return interpolated_masks

    def predict_batch(
        self,
        features,
        orig_hw,
        point_coords_batch,
        prompt_encoder=None,
        mask_decoder=None,
        onnx=False
    ):
        """
        Predict masks for a batch of single-point prompts.

        Parameters
        ----------
        point_coords_batch : np.ndarray
            [B, 2] array of point coordinates in pixel space.

        Returns
        -------
        all_masks : np.ndarray [B*3, H, W] binary masks
        all_ious : np.ndarray [B*3] IoU predictions
        all_low_res_masks : np.ndarray [B*3, 256, 256] low-res logits
        all_points : np.ndarray [B*3, 2] repeated point coordinates
        """
        B = point_coords_batch.shape[0]

        # Transform coordinates to model space
        coords = self.transform_coords(point_coords_batch, orig_hw)  # [B, 2]
        coords = coords[:, None, :].astype(np.float32)  # [B, 1, 2]
        labels = np.ones((B, 1), dtype=np.int32)  # all positive points

        # Mask dummy for batch
        if self.legacy:
            mask_input = np.zeros((B, 256, 256), dtype=np.float32)
        else:
            mask_input = np.zeros((B, 1, 256, 256), dtype=np.float32)
        masks_enable = np.array([0], dtype=np.int32)

        # Run prompt encoder (batch)
        if onnx:
            sparse_embeddings, dense_embeddings, dense_pe = prompt_encoder.run(
                None, {"coords": coords, "labels": labels, "masks": mask_input, "masks_enable": masks_enable})
        else:
            sparse_embeddings, dense_embeddings, dense_pe = prompt_encoder.run(
                {"coords": coords, "labels": labels, "masks": mask_input, "masks_enable": masks_enable})

        # Repeat image features for batch
        image_embed = np.repeat(features["image_embed"], B, axis=0)  # [B, C, H, W]
        high_res_feat1 = np.repeat(features["high_res_feats"][0], B, axis=0)
        high_res_feat2 = np.repeat(features["high_res_feats"][1], B, axis=0)

        # Run mask decoder per-item (mask_decoder does not support batch)
        image_embed = features["image_embed"]  # [1, C, H, W]
        high_res_feat1 = features["high_res_feats"][0]  # [1, C, H, W]
        high_res_feat2 = features["high_res_feats"][1]  # [1, C, H, W]

        all_masks = []
        all_ious = []
        # dense_pe is a constant (same for all items), use first element
        dense_pe_single = dense_pe[0:1] if dense_pe.shape[0] > 1 else dense_pe

        for i in range(B):
            if onnx:
                masks_i, iou_i, _, _ = mask_decoder.run(None, {
                    "image_embeddings": image_embed,
                    "image_pe": dense_pe_single,
                    "sparse_prompt_embeddings": sparse_embeddings[i:i+1],
                    "dense_prompt_embeddings": dense_embeddings[i:i+1],
                    "high_res_features1": high_res_feat1,
                    "high_res_features2": high_res_feat2})
            else:
                masks_i, iou_i, _, _ = mask_decoder.run({
                    "image_embeddings": image_embed,
                    "image_pe": dense_pe_single,
                    "sparse_prompt_embeddings": sparse_embeddings[i:i+1],
                    "dense_prompt_embeddings": dense_embeddings[i:i+1],
                    "high_res_features1": high_res_feat1,
                    "high_res_features2": high_res_feat2})
            # Extract multimask outputs (indices 1:4)
            all_masks.append(masks_i[:, 1:, :, :])  # [1, 3, H, W]
            all_ious.append(iou_i[:, 1:])  # [1, 3]

        masks = np.concatenate(all_masks, axis=0)  # [B, 3, H, W]
        iou_pred = np.concatenate(all_ious, axis=0)  # [B, 3]

        # Flatten: [B, 3, H, W] -> [B*3, H, W], [B*3]
        masks_flat = masks.reshape(-1, masks.shape[-2], masks.shape[-1])  # [B*3, H, W]
        iou_flat = iou_pred.reshape(-1)  # [B*3]

        # Repeat points 3x to match
        points_flat = np.repeat(point_coords_batch, 3, axis=0)  # [B*3, 2]

        return masks_flat, iou_flat, points_flat
