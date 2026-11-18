import os
import sys
import time
import numpy as np
import cv2


class SAM3ImagePredictor:
    def set_image(self, image, image_encoder, onnx):
        image = image.astype(np.uint8)

        if onnx:
            output = image_encoder.run(None, {
                'image': image,
            })
        else:
            output = image_encoder.run({
                'image': image,
            })

        vision_pos_enc_0, vision_pos_enc_1, vision_pos_enc_2, backbone_fpn_0, backbone_fpn_1, backbone_fpn_2 = output

        features = {
            'vision_pos_enc': [vision_pos_enc_0, vision_pos_enc_1, vision_pos_enc_2],
            'backbone_fpn': [backbone_fpn_0, backbone_fpn_1, backbone_fpn_2],
        }

        return features

    def predict(self, features, orig_hw, prompt, box=None, prompt_encoder=None, mask_decoder=None, onnx=False):
        tokens = self._tokenize(prompt)
        box_coords, box_labels, box_masks = self._prep_box(box, orig_hw)

        scores, masks, boxes = self._predict(
            features,
            orig_hw,
            tokens=tokens,
            box_coords=box_coords,
            box_labels=box_labels,
            box_masks=box_masks,
            prompt_encoder=prompt_encoder,
            mask_decoder=mask_decoder,
            onnx=onnx
        )

        return scores, masks, boxes

    def _tokenize(self, prompt):
        from osam._models.yoloworld.clip import tokenize
        tokens = tokenize(texts=[prompt], context_length=32)
        return tokens

    def _prep_box(self, box, orig_hw):
        if box is not None:
            box = box.astype(np.float32)
            box_coords = self._transform_boxes(box, orig_hw=orig_hw).reshape(1, 1, 4)
            box_labels = np.array([[1]], dtype=np.int64)
            box_masks = np.array([False], dtype=np.bool_).reshape(1, 1)
        else:
            box_coords = np.array([0, 0, 0, 0], dtype=np.float32).reshape(1, 1, 4)
            box_labels = np.array([[1]], dtype=np.int64)
            box_masks = np.array([True], dtype=np.bool_).reshape(1, 1)
        return box_coords, box_labels, box_masks

    def _predict(self, features, orig_hw, tokens, box_coords, box_labels, box_masks, prompt_encoder=None, mask_decoder=None, onnx=False):
        if onnx:
            language_mask, language_features, _ = prompt_encoder.run(None, {
                'tokens': tokens,
            })
        else:
            language_mask, language_features, _ = prompt_encoder.run({
                'tokens': tokens,
            })

        image_height = np.array(orig_hw[0], dtype=np.int64)
        image_width = np.array(orig_hw[1], dtype=np.int64)

        backbone_fpn = features['backbone_fpn']
        vision_pos_enc = features['vision_pos_enc']

        if onnx:
            boxes, scores, masks = mask_decoder.run(None, {
                'original_height': image_height,
                'original_width': image_width,
                'backbone_fpn_0': backbone_fpn[0],
                'backbone_fpn_1': backbone_fpn[1],
                'backbone_fpn_2': backbone_fpn[2],
                'vision_pos_enc_2': vision_pos_enc[2],
                'language_mask': language_mask,
                'language_features': language_features,
                'box_coords': box_coords,
                'box_labels': box_labels,
                'box_masks': box_masks,
            })
        else:
            boxes, scores, masks = mask_decoder.run({
                'original_height': image_height,
                'original_width': image_width,
                'backbone_fpn_0': backbone_fpn[0],
                'backbone_fpn_1': backbone_fpn[1],
                'backbone_fpn_2': backbone_fpn[2],
                'vision_pos_enc_2': vision_pos_enc[2],
                'language_mask': language_mask,
                'language_features': language_features,
                'box_coords': box_coords,
                'box_labels': box_labels,
                'box_masks': box_masks,
            })

        return scores, masks, boxes

    def _transform_boxes(self, boxes, orig_hw=None):
        boxes = self._transform_coords(boxes.reshape(-1, 2, 2), orig_hw)

    def _transform_coords(self, coords, orig_hw):
        h, w = orig_hw
        coords = coords.copy()
        coords[..., 0] = coords[..., 0] / w
        coords[..., 1] = coords[..., 1] / h

        resolution = 1008
        coords = coords * resolution  # unnormalize coords
        return coords
        return boxes
