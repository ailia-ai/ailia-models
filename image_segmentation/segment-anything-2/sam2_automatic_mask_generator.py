import numpy as np
import cv2

from sam2_image_predictor import SAM2ImagePredictor


def build_point_grid(n_per_side):
    """Generate an evenly spaced grid of points in [0,1]^2."""
    offset = 1 / (2 * n_per_side)
    points_one_side = np.linspace(offset, 1 - offset, n_per_side)
    points_x = np.tile(points_one_side[None, :], (n_per_side, 1))
    points_y = np.tile(points_one_side[:, None], (1, n_per_side))
    points = np.stack([points_x, points_y], axis=-1).reshape(-1, 2)
    return points


def calculate_stability_score(mask_logits, mask_threshold=0.0, threshold_offset=1.0):
    """Compute stability score: IoU between masks at threshold +/- offset."""
    intersections = np.sum(mask_logits > (mask_threshold + threshold_offset), axis=(-2, -1)).astype(np.float32)
    unions = np.sum(mask_logits > (mask_threshold - threshold_offset), axis=(-2, -1)).astype(np.float32)
    return np.where(unions > 0, intersections / unions, 1.0)


def mask_to_box(masks):
    """Compute bounding boxes from binary masks. masks: [N, H, W] -> boxes: [N, 4] (x1,y1,x2,y2)."""
    N = masks.shape[0]
    boxes = np.zeros((N, 4), dtype=np.float32)
    for i in range(N):
        ys, xs = np.where(masks[i])
        if len(xs) == 0:
            continue
        boxes[i] = [xs.min(), ys.min(), xs.max(), ys.max()]
    return boxes


def box_iou(boxes1, boxes2):
    """Compute IoU between two sets of boxes. boxes: [N, 4] (x1,y1,x2,y2)."""
    x1 = np.maximum(boxes1[:, None, 0], boxes2[None, :, 0])
    y1 = np.maximum(boxes1[:, None, 1], boxes2[None, :, 1])
    x2 = np.minimum(boxes1[:, None, 2], boxes2[None, :, 2])
    y2 = np.minimum(boxes1[:, None, 3], boxes2[None, :, 3])
    inter = np.maximum(x2 - x1, 0) * np.maximum(y2 - y1, 0)
    area1 = (boxes1[:, 2] - boxes1[:, 0]) * (boxes1[:, 3] - boxes1[:, 1])
    area2 = (boxes2[:, 2] - boxes2[:, 0]) * (boxes2[:, 3] - boxes2[:, 1])
    union = area1[:, None] + area2[None, :] - inter
    return np.where(union > 0, inter / union, 0.0)


def nms(boxes, scores, iou_threshold):
    """Greedy NMS. Returns indices to keep."""
    order = np.argsort(-scores)
    keep = []
    suppressed = np.zeros(len(scores), dtype=bool)
    for i in order:
        if suppressed[i]:
            continue
        keep.append(i)
        ious = box_iou(boxes[i:i+1], boxes)[0]
        suppressed |= (ious > iou_threshold)
        suppressed[i] = False  # don't suppress self
    return np.array(keep, dtype=np.intp)


class SAM2AutomaticMaskGenerator:
    def __init__(
        self,
        legacy=False,
        version="2",
        model_type="hiera_l",
        points_per_side=32,
        points_per_batch=64,
        pred_iou_thresh=0.8,
        stability_score_thresh=0.95,
        box_nms_thresh=0.7,
    ):
        self.image_predictor = SAM2ImagePredictor(legacy, version, model_type)
        self.points_per_side = points_per_side
        self.points_per_batch = points_per_batch
        self.pred_iou_thresh = pred_iou_thresh
        self.stability_score_thresh = stability_score_thresh
        self.box_nms_thresh = box_nms_thresh

    def generate(
        self,
        features,
        orig_hw,
        prompt_encoder,
        mask_decoder,
        onnx=False,
    ):
        """
        Generate masks for all objects in the image.

        Parameters
        ----------
        features : dict
            Image features from SAM2ImagePredictor.set_image().
        orig_hw : list
            [height, width] of the original image.
        prompt_encoder : model
            Prompt encoder ONNX/ailia model.
        mask_decoder : model
            Mask decoder ONNX/ailia model.
        onnx : bool
            Whether to use ONNX Runtime.

        Returns
        -------
        binary_masks : np.ndarray [N, H, W] bool
            Binary masks at original resolution.
        iou_scores : np.ndarray [N]
            IoU prediction scores for each mask.
        """
        # Build grid of points in pixel coordinates
        grid = build_point_grid(self.points_per_side)  # [N, 2] in [0,1]
        point_coords = grid * np.array([orig_hw[1], orig_hw[0]])  # scale to pixel
        point_coords = point_coords.astype(np.float32)

        all_masks = []
        all_ious = []

        # Process in batches
        n_points = point_coords.shape[0]
        for i in range(0, n_points, self.points_per_batch):
            batch_coords = point_coords[i:i+self.points_per_batch]
            masks_flat, iou_flat, _ = self.image_predictor.predict_batch(
                features=features,
                orig_hw=orig_hw,
                point_coords_batch=batch_coords,
                prompt_encoder=prompt_encoder,
                mask_decoder=mask_decoder,
                onnx=onnx
            )
            all_masks.append(masks_flat)
            all_ious.append(iou_flat)

        all_masks = np.concatenate(all_masks, axis=0)  # [M, H, W] logits
        all_ious = np.concatenate(all_ious, axis=0)     # [M]

        # Filter by predicted IoU
        keep = all_ious > self.pred_iou_thresh
        all_masks = all_masks[keep]
        all_ious = all_ious[keep]

        # Filter by stability score
        stability = calculate_stability_score(all_masks)
        keep = stability >= self.stability_score_thresh
        all_masks = all_masks[keep]
        all_ious = all_ious[keep]

        # Upscale masks to original resolution and binarize
        binary_masks = np.zeros((all_masks.shape[0], orig_hw[0], orig_hw[1]), dtype=bool)
        for i in range(all_masks.shape[0]):
            resized = cv2.resize(all_masks[i], (orig_hw[1], orig_hw[0]), interpolation=cv2.INTER_LINEAR)
            binary_masks[i] = resized > 0.0

        # Compute bounding boxes and apply NMS
        boxes = mask_to_box(binary_masks)
        keep_idx = nms(boxes, all_ious, self.box_nms_thresh)
        binary_masks = binary_masks[keep_idx]
        all_ious = all_ious[keep_idx]

        return binary_masks, all_ious
