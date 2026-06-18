import warnings
from collections import OrderedDict

import numpy as np
import cv2

import time
from logging import getLogger

logger = getLogger(__name__)

# a large negative value as a placeholder score for missing objects
NO_OBJ_SCORE = -1024.0

# sam2_utils.py
def select_closest_cond_frames(frame_idx, cond_frame_outputs, max_cond_frame_num):
    """
    Select up to `max_cond_frame_num` conditioning frames from `cond_frame_outputs`
    that are temporally closest to the current frame at `frame_idx`. Here, we take
    - a) the closest conditioning frame before `frame_idx` (if any);
    - b) the closest conditioning frame after `frame_idx` (if any);
    - c) any other temporally closest conditioning frames until reaching a total
         of `max_cond_frame_num` conditioning frames.

    Outputs:
    - selected_outputs: selected items (keys & values) from `cond_frame_outputs`.
    - unselected_outputs: items (keys & values) not selected in `cond_frame_outputs`.
    """
    if max_cond_frame_num == -1 or len(cond_frame_outputs) <= max_cond_frame_num:
        selected_outputs = cond_frame_outputs
        unselected_outputs = {}
    else:
        assert max_cond_frame_num >= 2, "we should allow using 2+ conditioning frames"
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

    #pos_embed = pos_inds.unsqueeze(-1) / dim_t
    pos_embed = np.expand_dims(pos_inds, axis=-1) / dim_t
    #pos_embed = np.concatenate([np.sin(pos_embed), np.cos(pos_embed)], axis=-1)
    pos_embed = np.concatenate([np.sin(pos_embed), np.cos(pos_embed)], axis=-1)
    return pos_embed

def concat_points(old_point_inputs, new_points, new_labels):
    """Add new points and labels to previous point inputs (add at the end)."""
    if old_point_inputs is None:
        points, labels = new_points, new_labels
    else:
        points = np.concatenate([old_point_inputs["point_coords"], new_points], axis=1)
        labels = np.concatenate([old_point_inputs["point_labels"], new_labels], axis=1)

    return {"point_coords": points, "point_labels": labels}

def trunc_normal(size, std=0.02, a=-2, b=2):
    values = np.random.normal(loc=0., scale=std, size=size)
    values = np.clip(values, a*std, b*std)
    return values.astype(np.float32)

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def interpolate(low_res_multimasks, image_size):
    high_res_multimasks = np.zeros((low_res_multimasks.shape[0], low_res_multimasks.shape[1], image_size[0], image_size[1]), dtype=np.float32)
    for b in range(low_res_multimasks.shape[0]):
        for c in range(low_res_multimasks.shape[1]):
            high_res_multimasks[b][c] = cv2.resize(low_res_multimasks[b][c], (image_size[1], image_size[0]), high_res_multimasks, interpolation=cv2.INTER_LINEAR)
        #high_res_multimasks = F.interpolate(
        #    low_res_multimasks,
        #    size=(self.image_size, self.image_size),
        #    mode="bilinear",
        #    align_corners=False,
        #)
    return high_res_multimasks

# sam2_video_predictor.py
class SAM2VideoPredictor():
    """The predictor class to handle user interactions and manage inference states."""

    def __init__(
        self,
        onnx,
        benchmark,
        legacy=False,
        fill_hole_area=0,
        # whether to apply non-overlapping constraints on the output object masks
        non_overlap_masks=False,
        # whether to clear non-conditioning memory of the surrounding frames (which may contain outdated information) after adding correction clicks;
        # note that this would only apply to *single-object tracking* unless `clear_non_cond_mem_for_multi_obj` is also set to True)
        clear_non_cond_mem_around_input=False,
        # whether to also clear non-conditioning memory of the surrounding frames (only effective when `clear_non_cond_mem_around_input` is True).
        clear_non_cond_mem_for_multi_obj=False
    ):
        self.onnx = onnx
        self.benchmark = benchmark
        self.legacy = legacy
        self.fill_hole_area = fill_hole_area
        self.non_overlap_masks = non_overlap_masks
        self.clear_non_cond_mem_around_input = clear_non_cond_mem_around_input
        self.clear_non_cond_mem_for_multi_obj = clear_non_cond_mem_for_multi_obj

    def init_state(
        self,
        num_maskmem = 7,  # default 1 input frame + 6 previous frames
        max_obj_ptrs_in_encoder = 16,
        version = "2.1",
        model_type = "hiera_l"
    ):
        """default state from yaml"""
        self.image_size = 1024
        self.num_feature_levels = 3
        self.hidden_dim = 256
        self.num_maskmem = num_maskmem
        self.directly_add_no_mem_embed = True
        self.training = False
        self.mem_dim = 64
        self.add_tpos_enc_to_obj_ptrs = False
        self.version = version
        if version == "2.1":
            self.add_tpos_enc_to_obj_ptrs = True
        self.use_obj_ptrs_in_encoder = True
        self.add_all_frames_to_correct_as_cond = False
        self.multimask_output_in_sam = True
        self.multimask_min_pt_num = 0
        self.multimask_max_pt_num = 1
        self.sam_prompt_embed_dim = self.hidden_dim
        self.backbone_stride = 16
        self.sam_image_embedding_size = self.image_size // self.backbone_stride
        self.pred_obj_scores = True
        self.use_obj_ptrs_in_encoder = True
        self.use_mlp_for_obj_ptr_proj = True
        self.proj_tpos_enc_in_obj_ptrs = False
        if version == "2.1":
            self.proj_tpos_enc_in_obj_ptrs = True
        self.soft_no_obj_ptr = False
        self.fixed_no_obj_ptr = True
        self.non_overlap_masks_for_mem_enc = False
        self.binarize_mask_from_pts_for_mem_enc = True
        self.sigmoid_scale_for_mem_enc = 20
        self.sigmoid_bias_for_mem_enc = -10.0
        self.dynamic_multimask_via_stability = True
        self.dynamic_multimask_stability_delta = 0.05
        self.dynamic_multimask_stability_thresh = 0.98
        self.max_cond_frames_in_attn = -1
        self.memory_temporal_stride_for_eval = 1
        self.max_obj_ptrs_in_encoder = max_obj_ptrs_in_encoder
        self.only_obj_ptrs_in_the_past_for_eval = True
        self.multimask_output_for_tracking = True
        self.use_multimask_token_for_obj_ptr = True

        # Load pretrained weights or fallback to random initialization
        self.maskmem_tpos_enc, self.no_mem_embed, self.no_mem_pos_enc, self.no_obj_ptr = \
            self._load_pretrained_weights(version, model_type, self.num_maskmem, self.mem_dim, self.hidden_dim)

        """Initialize an inference state."""
        inference_state = {}
        # the original video height and width, used for resizing final output scores
        # inputs on each frame
        inference_state["point_inputs_per_obj"] = {}
        inference_state["mask_inputs_per_obj"] = {}
        # visual features on a small number of recently visited frames for quick interactions
        inference_state["cached_features"] = {}
        # values that don't change across frames (so we only need to hold one copy of them)
        inference_state["constants"] = {}
        # mapping between client-side object id and model-side object index
        inference_state["obj_id_to_idx"] = OrderedDict()
        inference_state["obj_idx_to_id"] = OrderedDict()
        inference_state["obj_ids"] = []
        # A storage to hold the model's tracking results and states on each frame
        inference_state["output_dict"] = {
            "cond_frame_outputs": {},  # dict containing {frame_idx: <out>}
            "non_cond_frame_outputs": {},  # dict containing {frame_idx: <out>}
        }
        # Slice (view) of each object tracking results, sharing the same memory with "output_dict"
        inference_state["output_dict_per_obj"] = {}
        # A temporary storage to hold new outputs when user interact with a frame
        # to add clicks or mask (it's merged into "output_dict" before propagation starts)
        inference_state["temp_output_dict_per_obj"] = {}
        # Frames that already holds consolidated outputs from click or mask inputs
        # (we directly use their consolidated outputs during tracking)
        inference_state["consolidated_frame_inds"] = {
            "cond_frame_outputs": set(),  # set containing frame indices
            "non_cond_frame_outputs": set(),  # set containing frame indices
        }
        # metadata for each tracking frame (e.g. which direction it's tracked)
        inference_state["tracking_has_started"] = False
        inference_state["frames_already_tracked"] = {}
        # Warm up the visual backbone and cache the image feature on frame 0
        inference_state["images"] = None
        inference_state["num_frames"] = 0
        # Debug
        self.debug = False
        return inference_state

    def _load_pretrained_weights(self, version, model_type, num_maskmem, mem_dim, hidden_dim):
        """Load pretrained weights from npz file."""
        import os
        npz_path = os.path.join(os.path.dirname(__file__), "pretrained_weights.npz")
        prefix = f"v{version}_{model_type}".replace("+", "plus")
        data = np.load(npz_path)
        maskmem = data[f"{prefix}_maskmem_tpos_enc"].astype(np.float32)
        # maskmem_tpos_enc shape is (7, 1, 1, 64) but num_maskmem may differ
        if maskmem.shape[0] >= num_maskmem:
            maskmem = maskmem[:num_maskmem]
        else:
            maskmem = np.concatenate([maskmem, np.zeros((num_maskmem - maskmem.shape[0], 1, 1, mem_dim), dtype=np.float32)], axis=0)
        return (
            maskmem,
            data[f"{prefix}_no_mem_embed"].astype(np.float32),
            data[f"{prefix}_no_mem_pos_enc"].astype(np.float32),
            data[f"{prefix}_no_obj_ptr"].astype(np.float32),
        )

    def append_image(self,
        inference_state,
        image,
        video_height,
        video_width,
        image_encoder):
        inference_state["video_height"] = video_height
        inference_state["video_width"] = video_width
        if inference_state["images"] is None:
            inference_state["images"] = [image]
        else:
            inference_state["images"].append(image)
        inference_state["num_frames"] = len(inference_state["images"])
        if len(inference_state["images"]) == 1:
            self._get_image_feature(inference_state, frame_idx=0, batch_size=1, image_encoder=image_encoder)

    def _obj_id_to_idx(self, inference_state, obj_id):
        """Map client-side object id to model-side object index."""
        obj_idx = inference_state["obj_id_to_idx"].get(obj_id, None)
        if obj_idx is not None:
            return obj_idx

        # This is a new object id not sent to the server before. We only allow adding
        # new objects *before* the tracking starts.
        allow_new_object = not inference_state["tracking_has_started"]
        if allow_new_object:
            # get the next object slot
            obj_idx = len(inference_state["obj_id_to_idx"])
            inference_state["obj_id_to_idx"][obj_id] = obj_idx
            inference_state["obj_idx_to_id"][obj_idx] = obj_id
            inference_state["obj_ids"] = list(inference_state["obj_id_to_idx"])
            # set up input and output structures for this object
            inference_state["point_inputs_per_obj"][obj_idx] = {}
            inference_state["mask_inputs_per_obj"][obj_idx] = {}
            inference_state["output_dict_per_obj"][obj_idx] = {
                "cond_frame_outputs": {},  # dict containing {frame_idx: <out>}
                "non_cond_frame_outputs": {},  # dict containing {frame_idx: <out>}
            }
            inference_state["temp_output_dict_per_obj"][obj_idx] = {
                "cond_frame_outputs": {},  # dict containing {frame_idx: <out>}
                "non_cond_frame_outputs": {},  # dict containing {frame_idx: <out>}
            }
            return obj_idx
        else:
            raise RuntimeError(
                f"Cannot add new object id {obj_id} after tracking starts. "
                f"All existing object ids: {inference_state['obj_ids']}. "
                f"Please call 'reset_state' to restart from scratch."
            )

    def _obj_idx_to_id(self, inference_state, obj_idx):
        """Map model-side object index to client-side object id."""
        return inference_state["obj_idx_to_id"][obj_idx]

    def _get_obj_num(self, inference_state):
        """Get the total number of unique object ids received so far in this session."""
        return len(inference_state["obj_idx_to_id"])

    def add_new_points_or_box(
        self,
        inference_state,
        frame_idx,
        obj_id,
        points=None,
        labels=None,
        clear_old_points=True,
        normalize_coords=True,
        box=None,
        image_encoder=None,
        prompt_encoder=None,
        mask_decoder=None,
        memory_attention=None,
        memory_encoder=None,
        mlp=None,
        obj_ptr_tpos_proj=None
    ):
        """Add new points to a frame."""
        obj_idx = self._obj_id_to_idx(inference_state, obj_id)
        point_inputs_per_frame = inference_state["point_inputs_per_obj"][obj_idx]
        mask_inputs_per_frame = inference_state["mask_inputs_per_obj"][obj_idx]

        if (points is not None) != (labels is not None):
            raise ValueError("points and labels must be provided together")
        if points is None and box is None:
            raise ValueError("at least one of points or box must be provided as input")

        if points is None or len(points) == 0:
            points = np.zeros((0, 2), dtype=np.float32)
        if labels is None or len(labels) == 0:
            labels = np.zeros((0), dtype=np.int32)
        if points.ndim == 2:
            points = np.expand_dims(points, axis = 0)  # add batch dimension
        if labels.ndim == 1:
            labels = np.expand_dims(labels, axis = 0)  # add batch dimension

        # If `box` is provided, we add it as the first two points with labels 2 and 3
        # along with the user-provided points (consistent with how SAM 2 is trained).
        if box is not None:
            if not clear_old_points:
                raise ValueError(
                    "cannot add box without clearing old points, since "
                    "box prompt must be provided before any point prompt "
                    "(please use clear_old_points=True instead)"
                )
            if inference_state["tracking_has_started"]:
                warnings.warn(
                    "You are adding a box after tracking starts. SAM 2 may not always be "
                    "able to incorporate a box prompt for *refinement*. If you intend to "
                    "use box prompt as an *initial* input before tracking, please call "
                    "'reset_state' on the inference state to restart from scratch.",
                    category=UserWarning,
                    stacklevel=2,
                )
            box_coords = box.reshape(1, 2, 2)
            box_labels = np.array([2, 3], dtype=np.int32)
            box_labels = box_labels.reshape(1, 2)
            points = np.concatenate([box_coords, points], axis=1)
            labels = np.concatenate([box_labels, labels], axis=1)

        if normalize_coords:
            video_H = inference_state["video_height"]
            video_W = inference_state["video_width"]
            points = points / np.array([video_W, video_H])
        # scale the (normalized) coordinates by the model's internal image size
        points = points * self.image_size

        if not clear_old_points:
            point_inputs = point_inputs_per_frame.get(frame_idx, None)
        else:
            point_inputs = None
        point_inputs = concat_points(point_inputs, points, labels)

        point_inputs_per_frame[frame_idx] = point_inputs
        mask_inputs_per_frame.pop(frame_idx, None)
        # If this frame hasn't been tracked before, we treat it as an initial conditioning
        # frame, meaning that the inputs points are to generate segments on this frame without
        # using any memory from other frames, like in SAM. Otherwise (if it has been tracked),
        # the input points will be used to correct the already tracked masks.
        is_init_cond_frame = frame_idx not in inference_state["frames_already_tracked"]
        # whether to track in reverse time order
        if is_init_cond_frame:
            reverse = False
        else:
            reverse = inference_state["frames_already_tracked"][frame_idx]["reverse"]
        obj_output_dict = inference_state["output_dict_per_obj"][obj_idx]
        obj_temp_output_dict = inference_state["temp_output_dict_per_obj"][obj_idx]
        # Add a frame to conditioning output if it's an initial conditioning frame or
        # if the model sees all frames receiving clicks/mask as conditioning frames.
        is_cond = is_init_cond_frame or self.add_all_frames_to_correct_as_cond
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
        current_out, _ = self._run_single_frame_inference(
            inference_state=inference_state,
            output_dict=obj_output_dict,  # run on the slice of a single object
            frame_idx=frame_idx,
            batch_size=1,  # run on the slice of a single object
            is_init_cond_frame=is_init_cond_frame,
            point_inputs=point_inputs,
            mask_inputs=None,
            reverse=reverse,
            # Skip the memory encoder when adding clicks or mask. We execute the memory encoder
            # at the beginning of `propagate_in_video` (after user finalize their clicks). This
            # allows us to enforce non-overlapping constraints on all objects before encoding
            # them into memory.
            run_mem_encoder=False,
            prev_sam_mask_logits=prev_sam_mask_logits,
            image_encoder=image_encoder,
            prompt_encoder=prompt_encoder,
            mask_decoder=mask_decoder,
            memory_attention=memory_attention,
            memory_encoder=memory_encoder,
            mlp=mlp,
            obj_ptr_tpos_proj=obj_ptr_tpos_proj
        )
        # Add the output to the output dict (to be used as future memory)
        obj_temp_output_dict[storage_key][frame_idx] = current_out

        # Resize the output mask to the original video resolution
        obj_ids = inference_state["obj_ids"]
        consolidated_out = self._consolidate_temp_output_across_obj(
            inference_state,
            frame_idx,
            is_cond=is_cond,
            run_mem_encoder=False,
            consolidate_at_video_res=True,
            image_encoder=image_encoder, prompt_encoder=prompt_encoder, mask_decoder=mask_decoder, memory_encoder=memory_encoder, mlp=mlp, obj_ptr_tpos_proj=obj_ptr_tpos_proj
        )
        _, video_res_masks = self._get_orig_video_res_output(
            inference_state, consolidated_out["pred_masks_video_res"]
        )
        return frame_idx, obj_ids, video_res_masks

    def add_new_points(self, *args, **kwargs):
        """Deprecated method. Please use `add_new_points_or_box` instead."""
        return self.add_new_points_or_box(*args, **kwargs)

    def add_new_mask(
        self,
        inference_state,
        frame_idx,
        obj_id,
        mask,
        image_encoder,
        prompt_encoder,
        mask_decoder,
        memory_attention,
        memory_encoder,
        mlp
    ):
        """Add new mask to a frame."""
        obj_idx = self._obj_id_to_idx(inference_state, obj_id)
        point_inputs_per_frame = inference_state["point_inputs_per_obj"][obj_idx]
        mask_inputs_per_frame = inference_state["mask_inputs_per_obj"][obj_idx]

        assert mask.dim() == 2
        mask_H, mask_W = mask.shape
        mask_inputs_orig = mask[None, None]  # add batch and channel dimension
        mask_inputs_orig = mask_inputs_orig.astype(np.float32)
        
        # resize the mask if it doesn't match the model's image size
        if mask_H != self.image_size or mask_W != self.image_size:
            mask_inputs = interpolate(mask_inputs_orig, (self.image_size, self.image_size))
            #mask_inputs = torch.nn.functional.interpolate(
            #    mask_inputs_orig,
            #    size=(self.image_size, self.image_size),
            #    align_corners=False,
            #    mode="bilinear",
            #    antialias=True,  # use antialias for downsampling
            #)
            mask_inputs = (mask_inputs >= 0.5).astype(np.float32)
        else:
            mask_inputs = mask_inputs_orig

        mask_inputs_per_frame[frame_idx] = mask_inputs
        point_inputs_per_frame.pop(frame_idx, None)
        # If this frame hasn't been tracked before, we treat it as an initial conditioning
        # frame, meaning that the inputs points are to generate segments on this frame without
        # using any memory from other frames, like in SAM. Otherwise (if it has been tracked),
        # the input points will be used to correct the already tracked masks.
        is_init_cond_frame = frame_idx not in inference_state["frames_already_tracked"]
        # whether to track in reverse time order
        if is_init_cond_frame:
            reverse = False
        else:
            reverse = inference_state["frames_already_tracked"][frame_idx]["reverse"]
        obj_output_dict = inference_state["output_dict_per_obj"][obj_idx]
        obj_temp_output_dict = inference_state["temp_output_dict_per_obj"][obj_idx]
        # Add a frame to conditioning output if it's an initial conditioning frame or
        # if the model sees all frames receiving clicks/mask as conditioning frames.
        is_cond = is_init_cond_frame or self.add_all_frames_to_correct_as_cond
        storage_key = "cond_frame_outputs" if is_cond else "non_cond_frame_outputs"

        current_out, _ = self._run_single_frame_inference(
            inference_state=inference_state,
            output_dict=obj_output_dict,  # run on the slice of a single object
            frame_idx=frame_idx,
            batch_size=1,  # run on the slice of a single object
            is_init_cond_frame=is_init_cond_frame,
            point_inputs=None,
            mask_inputs=mask_inputs,
            reverse=reverse,
            # Skip the memory encoder when adding clicks or mask. We execute the memory encoder
            # at the beginning of `propagate_in_video` (after user finalize their clicks). This
            # allows us to enforce non-overlapping constraints on all objects before encoding
            # them into memory.
            run_mem_encoder=False,
            image_encoder=image_encoder,
            prompt_encoder=prompt_encoder,
            mask_decoder=mask_decoder,
            memory_attention=memory_attention,
            memory_encoder=memory_encoder,
            mlp=mlp
        )
        # Add the output to the output dict (to be used as future memory)
        obj_temp_output_dict[storage_key][frame_idx] = current_out

        # Resize the output mask to the original video resolution
        obj_ids = inference_state["obj_ids"]
        consolidated_out = self._consolidate_temp_output_across_obj(
            inference_state,
            frame_idx,
            is_cond=is_cond,
            run_mem_encoder=False,
            consolidate_at_video_res=True,
            image_encoder=image_encoder, prompt_encoder=prompt_encoder, mask_decoder=mask_decoder, memory_encoder=memory_encoder, mlp=mlp
        )
        _, video_res_masks = self._get_orig_video_res_output(
            inference_state, consolidated_out["pred_masks_video_res"]
        )
        return frame_idx, obj_ids, video_res_masks

    def _get_orig_video_res_output(self, inference_state, any_res_masks):
        """
        Resize the object scores to the original video resolution (video_res_masks)
        and apply non-overlapping constraints for final output.
        """
        video_H = inference_state["video_height"]
        video_W = inference_state["video_width"]
        if any_res_masks.shape[-2:] == (video_H, video_W):
            video_res_masks = any_res_masks
        else:
            video_res_masks = interpolate(any_res_masks, (video_H, video_W))
            #video_res_masks = torch.nn.functional.interpolate(
            #    any_res_masks,
            #    size=(video_H, video_W),
            #    mode="bilinear",
            #    align_corners=False,
            #)
        if self.non_overlap_masks:
            video_res_masks = self._apply_non_overlapping_constraints(video_res_masks)
        return any_res_masks, video_res_masks

    def _consolidate_temp_output_across_obj(
        self,
        inference_state,
        frame_idx,
        is_cond,
        run_mem_encoder,
        consolidate_at_video_res=False,
        image_encoder=None,
        prompt_encoder=None,
        mask_decoder=None,
        memory_attention=None,
        memory_encoder=None,
        mlp=None,
        obj_ptr_tpos_proj=None
    ):
        """
        Consolidate the per-object temporary outputs in `temp_output_dict_per_obj` on
        a frame into a single output for all objects, including
        1) fill any missing objects either from `output_dict_per_obj` (if they exist in
           `output_dict_per_obj` for this frame) or leave them as placeholder values
           (if they don't exist in `output_dict_per_obj` for this frame);
        2) if specified, rerun memory encoder after apply non-overlapping constraints
           on the object scores.
        """
        batch_size = self._get_obj_num(inference_state)
        storage_key = "cond_frame_outputs" if is_cond else "non_cond_frame_outputs"
        # Optionally, we allow consolidating the temporary outputs at the original
        # video resolution (to provide a better editing experience for mask prompts).
        if consolidate_at_video_res:
            assert not run_mem_encoder, "memory encoder cannot run at video resolution"
            consolidated_H = inference_state["video_height"]
            consolidated_W = inference_state["video_width"]
            consolidated_mask_key = "pred_masks_video_res"
        else:
            consolidated_H = consolidated_W = self.image_size // 4
            consolidated_mask_key = "pred_masks"

        # Initialize `consolidated_out`. Its "maskmem_features" and "maskmem_pos_enc"
        # will be added when rerunning the memory encoder after applying non-overlapping
        # constraints to object scores. Its "pred_masks" are prefilled with a large
        # negative value (NO_OBJ_SCORE) to represent missing objects.
        consolidated_out = {
            "maskmem_features": None,
            "maskmem_pos_enc": None,
            consolidated_mask_key: np.full(
                shape=(batch_size, 1, consolidated_H, consolidated_W),
                fill_value=NO_OBJ_SCORE,
                dtype=np.float32
            ),
            "obj_ptr": np.full(
                shape=(batch_size, self.hidden_dim),
                fill_value=NO_OBJ_SCORE,
                dtype=np.float32
            ),
        }
        empty_mask_ptr = None
        for obj_idx in range(batch_size):
            obj_temp_output_dict = inference_state["temp_output_dict_per_obj"][obj_idx]
            obj_output_dict = inference_state["output_dict_per_obj"][obj_idx]
            out = obj_temp_output_dict[storage_key].get(frame_idx, None)
            # If the object doesn't appear in "temp_output_dict_per_obj" on this frame,
            # we fall back and look up its previous output in "output_dict_per_obj".
            # We look up both "cond_frame_outputs" and "non_cond_frame_outputs" in
            # "output_dict_per_obj" to find a previous output for this object.
            if out is None:
                out = obj_output_dict["cond_frame_outputs"].get(frame_idx, None)
            if out is None:
                out = obj_output_dict["non_cond_frame_outputs"].get(frame_idx, None)
            # If the object doesn't appear in "output_dict_per_obj" either, we skip it
            # and leave its mask scores to the default scores (i.e. the NO_OBJ_SCORE
            # placeholder above) and set its object pointer to be a dummy pointer.
            if out is None:
                # Fill in dummy object pointers for those objects without any inputs or
                # tracking outcomes on this frame (only do it under `run_mem_encoder=True`,
                # i.e. when we need to build the memory for tracking).
                if run_mem_encoder:
                    if empty_mask_ptr is None:
                        empty_mask_ptr = self._get_empty_mask_ptr(
                            inference_state, frame_idx, image_encoder, prompt_encoder, mask_decoder, memory_attention, memory_encoder, mlp, obj_ptr_tpos_proj
                        )
                    # fill object pointer with a dummy pointer (based on an empty mask)
                    consolidated_out["obj_ptr"][obj_idx : obj_idx + 1] = empty_mask_ptr
                continue
            # Add the temporary object output mask to consolidated output mask
            obj_mask = out["pred_masks"]
            consolidated_pred_masks = consolidated_out[consolidated_mask_key]
            if obj_mask.shape[-2:] == consolidated_pred_masks.shape[-2:]:
                consolidated_pred_masks[obj_idx : obj_idx + 1] = obj_mask
            else:
                # Resize first if temporary object mask has a different resolution
                resized_obj_mask = interpolate(obj_mask,consolidated_pred_masks.shape[-2:])
                #resized_obj_mask = torch.nn.functional.interpolate(
                #    obj_mask,
                #    size=consolidated_pred_masks.shape[-2:],
                #    mode="bilinear",
                #    align_corners=False,
                #)
                consolidated_pred_masks[obj_idx : obj_idx + 1] = resized_obj_mask
            consolidated_out["obj_ptr"][obj_idx : obj_idx + 1] = out["obj_ptr"]

        # Optionally, apply non-overlapping constraints on the consolidated scores
        # and rerun the memory encoder
        if run_mem_encoder:
            high_res_masks = interpolate(consolidated_out["pred_masks"],(self.image_size, self.image_size))
            #high_res_masks = torch.nn.functional.interpolate(
            #    consolidated_out["pred_masks"],
            #    size=(self.image_size, self.image_size),
            #    mode="bilinear",
            #    align_corners=False,
            #)
            if self.non_overlap_masks_for_mem_enc:
                high_res_masks = self._apply_non_overlapping_constraints(high_res_masks)
            maskmem_features, maskmem_pos_enc = self._run_memory_encoder(
                inference_state=inference_state,
                frame_idx=frame_idx,
                batch_size=batch_size,
                high_res_masks=high_res_masks,
                is_mask_from_pts=True,  # these frames are what the user interacted with
                image_encoder=image_encoder,
                memory_encoder=memory_encoder
            )
            consolidated_out["maskmem_features"] = maskmem_features
            consolidated_out["maskmem_pos_enc"] = maskmem_pos_enc

        return consolidated_out

    def _get_empty_mask_ptr(self, inference_state, frame_idx, image_encoder, prompt_encoder, mask_decoder, memory_attention, memory_encoder, mlp, obj_ptr_tpos_proj):
        """Get a dummy object pointer based on an empty mask on the current frame."""
        # A dummy (empty) mask with a single object
        batch_size = 1
        mask_inputs = np.zeros(
            (batch_size, 1, self.image_size, self.image_size),
            dtype=np.float32
        )

        # Retrieve correct image features
        (
            _,
            _,
            current_vision_feats,
            current_vision_pos_embeds,
            feat_sizes,
        ) = self._get_image_feature(inference_state, frame_idx, batch_size, image_encoder)

        # Feed the empty mask and image feature above to get a dummy object pointer
        current_out = self.track_step(
            frame_idx=frame_idx,
            is_init_cond_frame=True,
            current_vision_feats=current_vision_feats,
            current_vision_pos_embeds=current_vision_pos_embeds,
            feat_sizes=feat_sizes,
            point_inputs=None,
            mask_inputs=mask_inputs,
            output_dict={},
            num_frames=inference_state["num_frames"],
            track_in_reverse=False,
            run_mem_encoder=False,
            prev_sam_mask_logits=None,
            prompt_encoder=prompt_encoder,
            mask_decoder=mask_decoder,
            memory_attention=memory_attention,
            memory_encoder=memory_encoder,
            mlp=mlp,
            obj_ptr_tpos_proj=obj_ptr_tpos_proj
        )
        return current_out["obj_ptr"]

    def propagate_in_video_preflight(self, inference_state, image_encoder, prompt_encoder, mask_decoder, memory_attention, memory_encoder, mlp, obj_ptr_tpos_proj):
        assert(memory_encoder!=None)
        """Prepare inference_state and consolidate temporary outputs before tracking."""
        # Tracking has started and we don't allow adding new objects until session is reset.
        inference_state["tracking_has_started"] = True
        batch_size = self._get_obj_num(inference_state)

        # Consolidate per-object temporary outputs in "temp_output_dict_per_obj" and
        # add them into "output_dict".
        temp_output_dict_per_obj = inference_state["temp_output_dict_per_obj"]
        output_dict = inference_state["output_dict"]
        # "consolidated_frame_inds" contains indices of those frames where consolidated
        # temporary outputs have been added (either in this call or any previous calls
        # to `propagate_in_video_preflight`).
        consolidated_frame_inds = inference_state["consolidated_frame_inds"]
        for is_cond in [False, True]:
            # Separately consolidate conditioning and non-conditioning temp outputs
            storage_key = "cond_frame_outputs" if is_cond else "non_cond_frame_outputs"
            # Find all the frames that contain temporary outputs for any objects
            # (these should be the frames that have just received clicks for mask inputs
            # via `add_new_points_or_box` or `add_new_mask`)
            temp_frame_inds = set()
            for obj_temp_output_dict in temp_output_dict_per_obj.values():
                temp_frame_inds.update(obj_temp_output_dict[storage_key].keys())
            consolidated_frame_inds[storage_key].update(temp_frame_inds)
            # consolidate the temporary output across all objects on this frame
            for frame_idx in temp_frame_inds:
                consolidated_out = self._consolidate_temp_output_across_obj(
                    inference_state, frame_idx, is_cond=is_cond, run_mem_encoder=True,
                    image_encoder=image_encoder, prompt_encoder=prompt_encoder, mask_decoder=mask_decoder, memory_attention=memory_attention,
                    memory_encoder=memory_encoder, mlp=mlp, obj_ptr_tpos_proj=obj_ptr_tpos_proj
                )
                # merge them into "output_dict" and also create per-object slices
                output_dict[storage_key][frame_idx] = consolidated_out
                self._add_output_per_object(
                    inference_state, frame_idx, consolidated_out, storage_key
                )
                clear_non_cond_mem = self.clear_non_cond_mem_around_input and (
                    self.clear_non_cond_mem_for_multi_obj or batch_size <= 1
                )
                if clear_non_cond_mem:
                    # clear non-conditioning memory of the surrounding frames
                    self._clear_non_cond_mem_around_input(inference_state, frame_idx)

            # clear temporary outputs in `temp_output_dict_per_obj`
            for obj_temp_output_dict in temp_output_dict_per_obj.values():
                obj_temp_output_dict[storage_key].clear()

        # edge case: if an output is added to "cond_frame_outputs", we remove any prior
        # output on the same frame in "non_cond_frame_outputs"
        for frame_idx in output_dict["cond_frame_outputs"]:
            output_dict["non_cond_frame_outputs"].pop(frame_idx, None)
        for obj_output_dict in inference_state["output_dict_per_obj"].values():
            for frame_idx in obj_output_dict["cond_frame_outputs"]:
                obj_output_dict["non_cond_frame_outputs"].pop(frame_idx, None)
        for frame_idx in consolidated_frame_inds["cond_frame_outputs"]:
            assert frame_idx in output_dict["cond_frame_outputs"]
            consolidated_frame_inds["non_cond_frame_outputs"].discard(frame_idx)

        # Make sure that the frame indices in "consolidated_frame_inds" are exactly those frames
        # with either points or mask inputs (which should be true under a correct workflow).
        all_consolidated_frame_inds = (
            consolidated_frame_inds["cond_frame_outputs"]
            | consolidated_frame_inds["non_cond_frame_outputs"]
        )
        input_frames_inds = set()
        for point_inputs_per_frame in inference_state["point_inputs_per_obj"].values():
            input_frames_inds.update(point_inputs_per_frame.keys())
        for mask_inputs_per_frame in inference_state["mask_inputs_per_obj"].values():
            input_frames_inds.update(mask_inputs_per_frame.keys())
        assert all_consolidated_frame_inds == input_frames_inds

    def propagate_in_video(
        self,
        inference_state,
        image_encoder=None,
        prompt_encoder=None,
        mask_decoder=None,
        memory_attention=None,
        memory_encoder=None,
        mlp=None,
        obj_ptr_tpos_proj=None,
        frame_idx = 0
    ):
        """Propagate the input points across frames to track in the entire video."""
        assert(memory_encoder!=None)

        output_dict = inference_state["output_dict"]
        consolidated_frame_inds = inference_state["consolidated_frame_inds"]
        obj_ids = inference_state["obj_ids"]
        num_frames = inference_state["num_frames"]
        batch_size = self._get_obj_num(inference_state)
        if len(output_dict["cond_frame_outputs"]) == 0:
            raise RuntimeError("No points are provided; please add points first")
        clear_non_cond_mem = self.clear_non_cond_mem_around_input and (
            self.clear_non_cond_mem_for_multi_obj or batch_size <= 1
        )

        # set start index, end index, and processing order

        # default: start from the earliest frame with input points
        start_frame_idx = min(output_dict["cond_frame_outputs"])

        # default: track all the frames in the video
        max_frame_num_to_track = num_frames

        reverse = False

        end_frame_idx = min(
            start_frame_idx + max_frame_num_to_track, num_frames - 1
        )
        processing_order = range(start_frame_idx, end_frame_idx + 1)

        if True:#for frame_idx in tqdm(processing_order, desc="propagate in video"):
            # We skip those frames already in consolidated outputs (these are frames
            # that received input clicks or mask). Note that we cannot directly run
            # batched forward on them via `_run_single_frame_inference` because the
            # number of clicks on each object might be different.
            if frame_idx in consolidated_frame_inds["cond_frame_outputs"]:
                storage_key = "cond_frame_outputs"
                current_out = output_dict[storage_key][frame_idx]
                pred_masks = current_out["pred_masks"]
                if clear_non_cond_mem:
                    # clear non-conditioning memory of the surrounding frames
                    self._clear_non_cond_mem_around_input(inference_state, frame_idx)
            elif frame_idx in consolidated_frame_inds["non_cond_frame_outputs"]:
                storage_key = "non_cond_frame_outputs"
                current_out = output_dict[storage_key][frame_idx]
                pred_masks = current_out["pred_masks"]
            else:
                storage_key = "non_cond_frame_outputs"
                current_out, pred_masks = self._run_single_frame_inference(
                    inference_state=inference_state,
                    output_dict=output_dict,
                    frame_idx=frame_idx,
                    batch_size=batch_size,
                    is_init_cond_frame=False,
                    point_inputs=None,
                    mask_inputs=None,
                    reverse=reverse,
                    run_mem_encoder=True,
                    image_encoder=image_encoder,
                    prompt_encoder=prompt_encoder,
                    mask_decoder=mask_decoder,
                    memory_attention=memory_attention,
                    memory_encoder=memory_encoder,
                    mlp=mlp,
                    obj_ptr_tpos_proj=obj_ptr_tpos_proj
                )
                output_dict[storage_key][frame_idx] = current_out
            # Create slices of per-object outputs for subsequent interaction with each
            # individual object after tracking.
            self._add_output_per_object(
                inference_state, frame_idx, current_out, storage_key
            )
            inference_state["frames_already_tracked"][frame_idx] = {"reverse": reverse}

            # Resize the output mask to the original video resolution (we directly use
            # the mask scores on GPU for output to avoid any CPU conversion in between)
            _, video_res_masks = self._get_orig_video_res_output(
                inference_state, pred_masks
            )
            return frame_idx, obj_ids, video_res_masks

    def _add_output_per_object(
        self, inference_state, frame_idx, current_out, storage_key
    ):
        """
        Split a multi-object output into per-object output slices and add them into
        `output_dict_per_obj`. The resulting slices share the same tensor storage.
        """
        maskmem_features = current_out["maskmem_features"]
        #assert maskmem_features is None or isinstance(maskmem_features, np.array)

        maskmem_pos_enc = current_out["maskmem_pos_enc"]
        #assert maskmem_pos_enc is None or isinstance(maskmem_pos_enc, list)

        output_dict_per_obj = inference_state["output_dict_per_obj"]
        for obj_idx, obj_output_dict in output_dict_per_obj.items():
            obj_slice = slice(obj_idx, obj_idx + 1)
            obj_out = {
                "maskmem_features": None,
                "maskmem_pos_enc": None,
                "pred_masks": current_out["pred_masks"][obj_slice],
                "obj_ptr": current_out["obj_ptr"][obj_slice],
            }
            if maskmem_features is not None:
                obj_out["maskmem_features"] = maskmem_features[obj_slice]
            if maskmem_pos_enc is not None:
                obj_out["maskmem_pos_enc"] = [x[obj_slice] for x in maskmem_pos_enc]
            obj_output_dict[storage_key][frame_idx] = obj_out

    def reset_state(self, inference_state):
        """Remove all input points or mask in all frames throughout the video."""
        self._reset_tracking_results(inference_state)
        # Remove all object ids
        inference_state["obj_id_to_idx"].clear()
        inference_state["obj_idx_to_id"].clear()
        inference_state["obj_ids"].clear()
        inference_state["point_inputs_per_obj"].clear()
        inference_state["mask_inputs_per_obj"].clear()
        inference_state["output_dict_per_obj"].clear()
        inference_state["temp_output_dict_per_obj"].clear()

    def _reset_tracking_results(self, inference_state):
        """Reset all tracking inputs and results across the videos."""
        for v in inference_state["point_inputs_per_obj"].values():
            v.clear()
        for v in inference_state["mask_inputs_per_obj"].values():
            v.clear()
        for v in inference_state["output_dict_per_obj"].values():
            v["cond_frame_outputs"].clear()
            v["non_cond_frame_outputs"].clear()
        for v in inference_state["temp_output_dict_per_obj"].values():
            v["cond_frame_outputs"].clear()
            v["non_cond_frame_outputs"].clear()
        inference_state["output_dict"]["cond_frame_outputs"].clear()
        inference_state["output_dict"]["non_cond_frame_outputs"].clear()
        inference_state["consolidated_frame_inds"]["cond_frame_outputs"].clear()
        inference_state["consolidated_frame_inds"]["non_cond_frame_outputs"].clear()
        inference_state["tracking_has_started"] = False
        inference_state["frames_already_tracked"].clear()

    def _get_image_feature(self, inference_state, frame_idx, batch_size, image_encoder):
        """Compute the image features on a given frame."""
        # Look up in the cache first
        image, backbone_out = inference_state["cached_features"].get(
            frame_idx, (None, None)
        )
        if backbone_out is None:
            # Cache miss -- we will run inference on a single image
            image = np.expand_dims(inference_state["images"][frame_idx], axis=0).astype(np.float32)

            if self.debug:
                print("begin image encoder onnx")
                print(frame_idx)
                print(image.shape)
            if self.benchmark:
                start = int(round(time.time() * 1000))
            if self.onnx:
                vision_features, vision_pos_enc_0, vision_pos_enc_1, vision_pos_enc_2, backbone_fpn_0, backbone_fpn_1, backbone_fpn_2 = image_encoder.run(None, {"input_image":image})
            else:
                vision_features, vision_pos_enc_0, vision_pos_enc_1, vision_pos_enc_2, backbone_fpn_0, backbone_fpn_1, backbone_fpn_2 = image_encoder.run({"input_image":image})
            if self.benchmark:
                end = int(round(time.time() * 1000))
                estimation_time = (end - start)
                logger.info(f'\timage_encoder processing {estimation_time} ms')

            backbone_out = {"vision_features":vision_features,
                            "vision_pos_enc":[vision_pos_enc_0, vision_pos_enc_1, vision_pos_enc_2],
                            "backbone_fpn":[backbone_fpn_0, backbone_fpn_1, backbone_fpn_2]}

            # Cache the most recent frame's feature (for repeated interactions with
            # a frame; we can use an LRU cache for more frames in the future).
            inference_state["cached_features"] = {frame_idx: (image, backbone_out)}

        # expand the features to have the same dimension as the number of objects
        expanded_image = np.repeat(image[np.newaxis, ...], batch_size, axis=0)
        expanded_backbone_out = {
            "backbone_fpn": backbone_out["backbone_fpn"].copy(),
            "vision_pos_enc": backbone_out["vision_pos_enc"].copy(),
        }
        for i, feat in enumerate(expanded_backbone_out["backbone_fpn"]):
            new_shape = (batch_size, feat.shape[1], feat.shape[2], feat.shape[3])
            expanded_backbone_out["backbone_fpn"][i] = np.broadcast_to(feat, new_shape)
            #expanded_backbone_out["backbone_fpn"][i] = feat.expand(
            #    batch_size, -1, -1, -1
            #)
        for i, pos in enumerate(expanded_backbone_out["vision_pos_enc"]):
            new_shape = (batch_size, pos.shape[1], pos.shape[2], pos.shape[3])
            pos = np.broadcast_to(pos, new_shape)
            #pos = pos.expand(batch_size, -1, -1, -1)
            expanded_backbone_out["vision_pos_enc"][i] = pos

        features = self._prepare_backbone_features(expanded_backbone_out)
        features = (expanded_image,) + features
        return features

    def _run_single_frame_inference(
        self,
        inference_state,
        output_dict,
        frame_idx,
        batch_size,
        is_init_cond_frame,
        point_inputs,
        mask_inputs,
        reverse,
        run_mem_encoder,
        prev_sam_mask_logits=None,
        image_encoder=None,
        prompt_encoder=None,
        mask_decoder=None,
        memory_attention=None,
        memory_encoder=None,
        mlp=None,
        obj_ptr_tpos_proj=None
    ):
        """Run tracking on a single frame based on current inputs and previous memory."""
        # Retrieve correct image features
        (
            _,
            _,
            current_vision_feats,
            current_vision_pos_embeds,
            feat_sizes,
        ) = self._get_image_feature(inference_state, frame_idx, batch_size, image_encoder)

        # point and mask should not appear as input simultaneously on the same frame
        assert point_inputs is None or mask_inputs is None
        current_out = self.track_step(
            frame_idx=frame_idx,
            is_init_cond_frame=is_init_cond_frame,
            current_vision_feats=current_vision_feats,
            current_vision_pos_embeds=current_vision_pos_embeds,
            feat_sizes=feat_sizes,
            point_inputs=point_inputs,
            mask_inputs=mask_inputs,
            output_dict=output_dict,
            num_frames=inference_state["num_frames"],
            track_in_reverse=reverse,
            run_mem_encoder=run_mem_encoder,
            prev_sam_mask_logits=prev_sam_mask_logits,
            prompt_encoder=prompt_encoder,
            mask_decoder=mask_decoder,
            memory_attention=memory_attention,
            memory_encoder=memory_encoder,
            mlp=mlp,
            obj_ptr_tpos_proj=obj_ptr_tpos_proj
        )

        # optionally offload the output to CPU memory to save GPU space
        maskmem_features = current_out["maskmem_features"]
        pred_masks_gpu = current_out["pred_masks"]
        #if self.fill_hole_area > 0:
        #    pred_masks_gpu = fill_holes_in_mask_scores(
        #        pred_masks_gpu, self.fill_hole_area
        #    )
        # potentially fill holes in the predicted masks
        pred_masks = pred_masks_gpu
        # "maskmem_pos_enc" is the same across frames, so we only need to store one copy of it
        maskmem_pos_enc = self._get_maskmem_pos_enc(inference_state, current_out)
        # object pointer is a small tensor, so we always keep it on GPU memory for fast access
        obj_ptr = current_out["obj_ptr"]
        # make a compact version of this frame's output to reduce the state size
        compact_current_out = {
            "maskmem_features": maskmem_features,
            "maskmem_pos_enc": maskmem_pos_enc,
            "pred_masks": pred_masks,
            "obj_ptr": obj_ptr,
        }
        return compact_current_out, pred_masks_gpu

    def _run_memory_encoder(
        self, inference_state, frame_idx, batch_size, high_res_masks, is_mask_from_pts, image_encoder, memory_encoder
    ):
        """
        Run the memory encoder on `high_res_masks`. This is usually after applying
        non-overlapping constraints to object scores. Since their scores changed, their
        memory also need to be computed again with the memory encoder.
        """
        # Retrieve correct image features
        _, _, current_vision_feats, _, feat_sizes = self._get_image_feature(
            inference_state, frame_idx, batch_size, image_encoder=image_encoder
        )
        maskmem_features, maskmem_pos_enc = self._encode_new_memory(
            current_vision_feats=current_vision_feats,
            feat_sizes=feat_sizes,
            pred_masks_high_res=high_res_masks,
            is_mask_from_pts=is_mask_from_pts,
            memory_encoder=memory_encoder
        )

        # optionally offload the output to CPU memory to save GPU space
        # "maskmem_pos_enc" is the same across frames, so we only need to store one copy of it
        maskmem_pos_enc = self._get_maskmem_pos_enc(
            inference_state, {"maskmem_pos_enc": maskmem_pos_enc}
        )
        return maskmem_features, maskmem_pos_enc

    def _get_maskmem_pos_enc(self, inference_state, current_out):
        """
        `maskmem_pos_enc` is the same across frames and objects, so we cache it as
        a constant in the inference session to reduce session storage size.
        """
        model_constants = inference_state["constants"]
        # "out_maskmem_pos_enc" should be either a list of tensors or None
        out_maskmem_pos_enc = current_out["maskmem_pos_enc"]
        if out_maskmem_pos_enc is not None:
            if "maskmem_pos_enc" not in model_constants:
                assert isinstance(out_maskmem_pos_enc, list)
                # only take the slice for one object, since it's same across objects
                maskmem_pos_enc = [x[0:1].copy() for x in out_maskmem_pos_enc]
                model_constants["maskmem_pos_enc"] = maskmem_pos_enc
            else:
                maskmem_pos_enc = model_constants["maskmem_pos_enc"]
            # expand the cached maskmem_pos_enc to the actual batch size
            batch_size = out_maskmem_pos_enc[0].shape[0]

            expanded_maskmem_pos_enc = [
                np.broadcast_to(x, (batch_size, x.shape[1], x.shape[2], x.shape[3])) for x in maskmem_pos_enc
            ]

            #expanded_maskmem_pos_enc = [
            #    x.expand(batch_size, -1, -1, -1) for x in maskmem_pos_enc
            #]
        else:
            expanded_maskmem_pos_enc = None
        return expanded_maskmem_pos_enc

    def _clear_non_cond_mem_around_input(self, inference_state, frame_idx):
        """
        Remove the non-conditioning memory around the input frame. When users provide
        correction clicks, the surrounding frames' non-conditioning memories can still
        contain outdated object appearance information and could confuse the model.

        This method clears those non-conditioning memories surrounding the interacted
        frame to avoid giving the model both old and new information about the object.
        """
        r = self.memory_temporal_stride_for_eval
        frame_idx_begin = frame_idx - r * self.num_maskmem
        frame_idx_end = frame_idx + r * self.num_maskmem
        output_dict = inference_state["output_dict"]
        non_cond_frame_outputs = output_dict["non_cond_frame_outputs"]
        for t in range(frame_idx_begin, frame_idx_end + 1):
            non_cond_frame_outputs.pop(t, None)
            for obj_output_dict in inference_state["output_dict_per_obj"].values():
                obj_output_dict["non_cond_frame_outputs"].pop(t, None)

    # sam2_base.py
    def track_step(
        self,
        frame_idx,
        is_init_cond_frame,
        current_vision_feats,
        current_vision_pos_embeds,
        feat_sizes,
        point_inputs,
        mask_inputs,
        output_dict,
        num_frames,
        track_in_reverse=False,  # tracking in reverse time order (for demo usage)
        # Whether to run the memory encoder on the predicted masks. Sometimes we might want
        # to skip the memory encoder with `run_mem_encoder=False`. For example,
        # in demo we might call `track_step` multiple times for each user click,
        # and only encode the memory when the user finalizes their clicks. And in ablation
        # settings like SAM training on static images, we don't need the memory encoder.
        run_mem_encoder=True,
        # The previously predicted SAM mask logits (which can be fed together with new clicks in demo).
        prev_sam_mask_logits=None,
        # ONNX Inference
        prompt_encoder=None,
        mask_decoder=None,
        memory_attention=None,
        memory_encoder=None,
        mlp=None,
        obj_ptr_tpos_proj=None
    ):
        current_out = {"point_inputs": point_inputs, "mask_inputs": mask_inputs}
        # High-resolution feature maps for the SAM head, reshape (HW)BC => BCHW
        if len(current_vision_feats) > 1:
            #high_res_features = [
            #    x.permute(1, 2, 0).view(x.size(1), x.size(2), *s)
            #    for x, s in zip(current_vision_feats[:-1], feat_sizes[:-1])
            #]]
            high_res_features = [
                np.reshape(np.transpose(x, (1, 2, 0)), (x.shape[1], x.shape[2], *s))
                for x, s in zip(current_vision_feats[:-1], feat_sizes[:-1])
            ]
        else:
            high_res_features = None
        if mask_inputs is not None and self.use_mask_input_as_output_without_sam:
            # When use_mask_input_as_output_without_sam=True, we directly output the mask input
            # (see it as a GT mask) without using a SAM prompt encoder + mask decoder.
            #pix_feat = current_vision_feats[-1].permute(1, 2, 0)
            #pix_feat = pix_feat.view(-1, self.hidden_dim, *feat_sizes[-1])
            pix_feat = np.transpose(current_vision_feats[-1], (1, 2, 0))
            pix_feat = np.reshape(pix_feat, (-1, self.hidden_dim, *feat_sizes[-1]))

            sam_outputs = self._use_mask_as_output(
                pix_feat, high_res_features, mask_inputs, prompt_encoder, mask_decoder, mlp
            )
        else:
            # fused the visual feature with previous memory features in the memory bank
            pix_feat_with_mem = self._prepare_memory_conditioned_features(
                frame_idx=frame_idx,
                is_init_cond_frame=is_init_cond_frame,
                current_vision_feats=current_vision_feats[-1:],
                current_vision_pos_embeds=current_vision_pos_embeds[-1:],
                feat_sizes=feat_sizes[-1:],
                output_dict=output_dict,
                num_frames=num_frames,
                track_in_reverse=track_in_reverse,
                memory_attention=memory_attention,
                obj_ptr_tpos_proj=obj_ptr_tpos_proj,
            )
            # apply SAM-style segmentation head
            # here we might feed previously predicted low-res SAM mask logits into the SAM mask decoder,
            # e.g. in demo where such logits come from earlier interaction instead of correction sampling
            # (in this case, any `mask_inputs` shouldn't reach here as they are sent to _use_mask_as_output instead)
            if prev_sam_mask_logits is not None:
                assert point_inputs is not None and mask_inputs is None
                mask_inputs = prev_sam_mask_logits
            multimask_output = self._use_multimask(is_init_cond_frame, point_inputs)
            sam_outputs = self._forward_sam_heads(
                backbone_features=pix_feat_with_mem,
                point_inputs=point_inputs,
                mask_inputs=mask_inputs,
                high_res_features=high_res_features,
                multimask_output=multimask_output,
                prompt_encoder=prompt_encoder,
                mask_decoder=mask_decoder,
                mlp=mlp
            )
        (
            _,
            _,
            _,
            low_res_masks,
            high_res_masks,
            obj_ptr,
            _,
        ) = sam_outputs

        current_out["pred_masks"] = low_res_masks
        current_out["pred_masks_high_res"] = high_res_masks
        current_out["obj_ptr"] = obj_ptr

        # Finally run the memory encoder on the predicted mask to encode
        # it into a new memory feature (that can be used in future frames)
        if run_mem_encoder and self.num_maskmem > 0:
            high_res_masks_for_mem_enc = high_res_masks
            maskmem_features, maskmem_pos_enc = self._encode_new_memory(
                current_vision_feats=current_vision_feats,
                feat_sizes=feat_sizes,
                pred_masks_high_res=high_res_masks_for_mem_enc,
                is_mask_from_pts=(point_inputs is not None),
                memory_encoder=memory_encoder,
            )
            current_out["maskmem_features"] = maskmem_features
            current_out["maskmem_pos_enc"] = maskmem_pos_enc
        else:
            current_out["maskmem_features"] = None
            current_out["maskmem_pos_enc"] = None

        return current_out

    # sam_base.py
    def _forward_sam_heads(
        self,
        backbone_features,
        point_inputs=None,
        mask_inputs=None,
        high_res_features=None,
        multimask_output=False,
        prompt_encoder=None,
        mask_decoder=None,
        mlp=None
    ):
        """
        Forward SAM prompt encoders and mask heads.

        Inputs:
        - backbone_features: image features of [B, C, H, W] shape
        - point_inputs: a dictionary with "point_coords" and "point_labels", where
          1) "point_coords" has [B, P, 2] shape and float32 dtype and contains the
             absolute pixel-unit coordinate in (x, y) format of the P input points
          2) "point_labels" has shape [B, P] and int32 dtype, where 1 means
             positive clicks, 0 means negative clicks, and -1 means padding
        - mask_inputs: a mask of [B, 1, H*16, W*16] shape, float or bool, with the
          same spatial size as the image.
        - high_res_features: either 1) None or 2) or a list of length 2 containing
          two feature maps of [B, C, 4*H, 4*W] and [B, C, 2*H, 2*W] shapes respectively,
          which will be used as high-resolution feature maps for SAM decoder.
        - multimask_output: if it's True, we output 3 candidate masks and their 3
          corresponding IoU estimates, and if it's False, we output only 1 mask and
          its corresponding IoU estimate.

        Outputs:
        - low_res_multimasks: [B, M, H*4, W*4] shape (where M = 3 if
          `multimask_output=True` and M = 1 if `multimask_output=False`), the SAM
          output mask logits (before sigmoid) for the low-resolution masks, with 4x
          the resolution (1/4 stride) of the input backbone_features.
        - high_res_multimasks: [B, M, H*16, W*16] shape (where M = 3
          if `multimask_output=True` and M = 1 if `multimask_output=False`),
          upsampled from the low-resolution masks, with shape size as the image
          (stride is 1 pixel).
        - ious, [B, M] shape, where (where M = 3 if `multimask_output=True` and M = 1
          if `multimask_output=False`), the estimated IoU of each output mask.
        - low_res_masks: [B, 1, H*4, W*4] shape, the best mask in `low_res_multimasks`.
          If `multimask_output=True`, it's the mask with the highest IoU estimate.
          If `multimask_output=False`, it's the same as `low_res_multimasks`.
        - high_res_masks: [B, 1, H*16, W*16] shape, the best mask in `high_res_multimasks`.
          If `multimask_output=True`, it's the mask with the highest IoU estimate.
          If `multimask_output=False`, it's the same as `high_res_multimasks`.
        - obj_ptr: [B, C] shape, the object pointer vector for the output mask, extracted
          based on the output token from the SAM mask decoder.
        """
        B = backbone_features.shape[0]
        assert backbone_features.shape[1] == self.sam_prompt_embed_dim
        assert backbone_features.shape[2] == self.sam_image_embedding_size
        assert backbone_features.shape[3] == self.sam_image_embedding_size

        # a) Handle point prompts
        if point_inputs is not None:
            sam_point_coords = point_inputs["point_coords"].astype(np.float32)
            sam_point_labels = point_inputs["point_labels"].astype(np.int32)
            assert sam_point_coords.shape[0] == B and sam_point_labels.shape[0] == B
        else:
            # If no points are provide, pad with an empty point (with label -1)
            sam_point_coords = np.zeros((B, 1, 2), dtype=np.float32)
            sam_point_labels = -np.ones((B, 1), dtype=np.int32)
            
        # b) Handle mask prompts
        if mask_inputs is not None:
            # If mask_inputs is provided, downsize it into low-res mask input if needed
            # and feed it as a dense mask prompt into the SAM mask encoder
            assert len(mask_inputs.shape) == 4 and mask_inputs.shape[:2] == (B, 1)
            if mask_inputs.shape[-2:] != self.sam_prompt_encoder.mask_input_size:
                sam_mask_prompt = interpolate(mask_inputs.astype(np.float32), self.sam_prompt_encoder.mask_input_size)
                #sam_mask_prompt = F.interpolate(
                #    mask_inputs.float(),
                #    size=self.sam_prompt_encoder.mask_input_size,
                #    align_corners=False,
                #    mode="bilinear",
                #    antialias=True,  # use antialias for downsampling
                #)
            else:
                sam_mask_prompt = mask_inputs
        else:
            # Otherwise, simply feed None (and SAM's prompt encoder will add
            # a learned `no_mask_embed` to indicate no mask input in this case).
            sam_mask_prompt = None

        if sam_mask_prompt is None:
            if self.legacy:
                mask_input_dummy = np.zeros((1, 256, 256), dtype=np.float32)
            else:
                mask_input_dummy = np.zeros((1, 1, 256, 256), dtype=np.float32)
            masks_enable = np.array([0], dtype=np.int32)
        else:
            mask_input_dummy = sam_mask_prompt.astype(np.float32)
            masks_enable = np.array([1], dtype=np.int32)

        if self.debug:
            print("begin prompt encoder onnx")

        if self.benchmark:
            start = int(round(time.time() * 1000))

        if self.onnx:
            sparse_embeddings, dense_embeddings, dense_pe = prompt_encoder.run(None, {"coords":sam_point_coords, "labels":sam_point_labels, "masks":mask_input_dummy, "masks_enable":masks_enable})
        else:
            sparse_embeddings, dense_embeddings, dense_pe = prompt_encoder.run({"coords":sam_point_coords, "labels":sam_point_labels, "masks":mask_input_dummy, "masks_enable":masks_enable})

        if self.benchmark:
            end = int(round(time.time() * 1000))
            estimation_time = (end - start)
            logger.info(f'\tprompt_encoder processing {estimation_time} ms')

        if self.debug:
            print("begin mask decoder onnx")
            print("backbone_features", np.sum(backbone_features))
            print("image_pe", np.sum(dense_pe))
            print("sparse_embeddings", np.sum(sparse_embeddings))
            print("dense_embeddings", np.sum(dense_embeddings))
            print("high_res_features", np.sum(high_res_features[0]))
            print("high_res_features", np.sum(high_res_features[1]))

        if self.benchmark:
            start = int(round(time.time() * 1000))

        if self.onnx:
            masks, iou_pred, sam_tokens_out, object_score_logits  = mask_decoder.run(None, {
                "image_embeddings":backbone_features,
                "image_pe": dense_pe,
                "sparse_prompt_embeddings": sparse_embeddings,
                "dense_prompt_embeddings": dense_embeddings,
                #repeat_image=False,  # the image is already batched
                "high_res_features1":high_res_features[0],
                "high_res_features2":high_res_features[1]})
        else:
            masks, iou_pred, sam_tokens_out, object_score_logits  = mask_decoder.run({
                "image_embeddings":backbone_features,
                "image_pe": dense_pe,
                "sparse_prompt_embeddings": sparse_embeddings,
                "dense_prompt_embeddings": dense_embeddings,
                #repeat_image=False,  # the image is already batched
                "high_res_features1":high_res_features[0],
                "high_res_features2":high_res_features[1]})

        if self.benchmark:
            end = int(round(time.time() * 1000))
            estimation_time = (end - start)
            logger.info(f'\tmask_decoder processing {estimation_time} ms')

        low_res_multimasks, ious, sam_output_tokens, object_score_logits  = self.forward_postprocess(masks, iou_pred, sam_tokens_out, object_score_logits, multimask_output)

        if self.pred_obj_scores:
            is_obj_appearing = object_score_logits > 0

            # Mask used for spatial memories is always a *hard* choice between obj and no obj,
            # consistent with the actual mask prediction
            low_res_multimasks = np.where(
                is_obj_appearing[:, None, None],
                low_res_multimasks,
                NO_OBJ_SCORE,
            )

        # convert masks from possibly bfloat16 (or float16) to float32
        # (older PyTorch versions before 2.1 don't support `interpolate` on bf16)
        low_res_multimasks = low_res_multimasks.astype(np.float32)
        high_res_multimasks = interpolate(low_res_multimasks, (self.image_size, self.image_size))
        #high_res_multimasks = F.interpolate(
        #    low_res_multimasks,
        #    size=(self.image_size, self.image_size),
        #    mode="bilinear",
        #    align_corners=False,
        #)

        sam_output_token = sam_output_tokens[:, 0]
        if multimask_output:
            # take the best mask prediction (with the highest IoU estimation)
            best_iou_inds = np.argmax(ious, axis=-1)
            batch_inds = np.arange(B)
            low_res_masks = np.expand_dims(low_res_multimasks[batch_inds, best_iou_inds], axis = 1)
            high_res_masks = np.expand_dims(high_res_multimasks[batch_inds, best_iou_inds], axis = 1)
            if sam_output_tokens.shape[1] > 1:
                sam_output_token = sam_output_tokens[batch_inds, best_iou_inds]
        else:
            low_res_masks, high_res_masks = low_res_multimasks, high_res_multimasks

        # Extract object pointer from the SAM output token (with occlusion handling)
        if self.benchmark:
            start = int(round(time.time() * 1000))
        if self.onnx:
            obj_ptr = mlp.run(None, {"x":sam_output_token})[0]
        else:
            obj_ptr = mlp.run({"x":sam_output_token})[0]
        if self.benchmark:
            end = int(round(time.time() * 1000))
            estimation_time = (end - start)
            logger.info(f'\tmlp processing {estimation_time} ms')
        
        if self.pred_obj_scores:
            # Allow *soft* no obj ptr, unlike for masks
            if self.soft_no_obj_ptr:
                # Only hard possible with gt
                assert not self.teacher_force_obj_scores_for_mem
                lambda_is_obj_appearing = object_score_logits.sigmoid()
            else:
                lambda_is_obj_appearing = is_obj_appearing.astype(np.float32)

            if self.fixed_no_obj_ptr:
                obj_ptr = lambda_is_obj_appearing * obj_ptr
            obj_ptr = obj_ptr + (1 - lambda_is_obj_appearing) * self.no_obj_ptr

        return (
            low_res_multimasks,
            high_res_multimasks,
            ious,
            low_res_masks,
            high_res_masks,
            obj_ptr,
            object_score_logits,
        )

    # mask_decoder.py
    def forward_postprocess(
        self,
        masks,
        iou_pred,
        mask_tokens_out,
        object_score_logits,
        multimask_output: bool,
    ):
        # Select the correct mask or masks for output
        if multimask_output:
            masks = masks[:, 1:, :, :]
            iou_pred = iou_pred[:, 1:]
        elif self.dynamic_multimask_via_stability and not self.training:
            masks, iou_pred = self._dynamic_multimask_via_stability(masks, iou_pred)
        else:
            masks = masks[:, 0:1, :, :]
            iou_pred = iou_pred[:, 0:1]

        if multimask_output and self.use_multimask_token_for_obj_ptr:
            sam_tokens_out = mask_tokens_out[:, 1:]  # [b, 3, c] shape
        else:
            # Take the mask output token. Here we *always* use the token for single mask output.
            # At test time, even if we track after 1-click (and using multimask_output=True),
            # we still take the single mask token here. The rationale is that we always track
            # after multiple clicks during training, so the past tokens seen during training
            # are always the single mask token (and we'll let it be the object-memory token).
            sam_tokens_out = mask_tokens_out[:, 0:1]  # [b, 1, c] shape

        # Prepare output
        return masks, iou_pred, sam_tokens_out, object_score_logits

    # sam2_base.py
    def _use_mask_as_output(self, backbone_features, high_res_features, mask_inputs, prompt_encoder, mask_decoder, mlp):
        """
        Directly turn binary `mask_inputs` into a output mask logits without using SAM.
        (same input and output shapes as in _forward_sam_heads above).
        """
        # Use -10/+10 as logits for neg/pos pixels (very close to 0/1 in prob after sigmoid).
        out_scale, out_bias = 20.0, -10.0  # sigmoid(-10.0)=4.5398e-05
        mask_inputs_float = mask_inputs.astype(np.float32)
        high_res_masks = mask_inputs_float * out_scale + out_bias
        low_res_masks = interpolate(high_res_masks, (high_res_masks.size(-2) // 4, high_res_masks.size(-1) // 4))
        #low_res_masks = F.interpolate(
        #    high_res_masks,
        #    size=(high_res_masks.size(-2) // 4, high_res_masks.size(-1) // 4),
        #    align_corners=False,
        #    mode="bilinear",
        #    antialias=True,  # use antialias for downsampling
        #)
        # a dummy IoU prediction of all 1's under mask input
        ious = mask_inputs.new_ones(mask_inputs.size(0), 1).astype(np.float32)
        if not self.use_obj_ptrs_in_encoder:
            # all zeros as a dummy object pointer (of shape [B, C])
            obj_ptr = np.zeros(
                (mask_inputs.size(0), self.hidden_dim), dtype=np.float32
            )
        else:
            # produce an object pointer using the SAM decoder from the mask input
            _, _, _, _, _, obj_ptr, _ = self._forward_sam_heads(
                backbone_features=backbone_features,
                mask_inputs=self.mask_downsample(mask_inputs_float),
                high_res_features=high_res_features,
                prompt_encoder=prompt_encoder,
                mask_decoder=mask_decoder,
                mlp=mlp
            )
        # In this method, we are treating mask_input as output, e.g. using it directly to create spatial mem;
        # Below, we follow the same design axiom to use mask_input to decide if obj appears or not instead of relying
        # on the object_scores from the SAM decoder.
        is_obj_appearing = np.any(mask_inputs.flatten(1).astype(np.float32) > 0.0, dim=1)
        is_obj_appearing = is_obj_appearing[..., None]
        lambda_is_obj_appearing = is_obj_appearing.astype(np.float32)
        object_score_logits = out_scale * lambda_is_obj_appearing + out_bias
        if self.pred_obj_scores:
            if self.fixed_no_obj_ptr:
                obj_ptr = lambda_is_obj_appearing * obj_ptr
            obj_ptr = obj_ptr + (1 - lambda_is_obj_appearing) * self.no_obj_ptr

        return (
            low_res_masks,
            high_res_masks,
            ious,
            low_res_masks,
            high_res_masks,
            obj_ptr,
            object_score_logits,
        )


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
        memory_attention=None,
        obj_ptr_tpos_proj=None
    ):
        """Fuse the current frame's visual feature map with previous memory."""
        B = current_vision_feats[-1].shape[1]  # batch size on this frame
        C = self.hidden_dim
        H, W = feat_sizes[-1]  # top-level (lowest-resolution) feature size
        # The case of `self.num_maskmem == 0` below is primarily used for reproducing SAM on images.
        # In this case, we skip the fusion with any memory.
        if self.num_maskmem == 0:  # Disable memory and skip fusion
            pix_feat = np.transpose(current_vision_feats[-1], (1, 2, 0)).reshape((B, C, H, W))
            return pix_feat

        num_obj_ptr_tokens = 0
        # Step 1: condition the visual features of the current frame on previous memories
        if not is_init_cond_frame:
            # Retrieve the memories encoded with the maskmem backbone
            to_cat_memory, to_cat_memory_pos_embed = [], []
            # Add conditioning frames's output first (all cond frames have t_pos=0 for
            # when getting temporal positional embedding below)
            assert len(output_dict["cond_frame_outputs"]) > 0
            # Select a maximum number of temporally closest cond frames for cross attention
            cond_outputs = output_dict["cond_frame_outputs"]
            selected_cond_outputs, unselected_cond_outputs = select_closest_cond_frames(
                frame_idx, cond_outputs, self.max_cond_frames_in_attn
            )
            t_pos_and_prevs = [(0, out) for out in selected_cond_outputs.values()]
            # Add last (self.num_maskmem - 1) frames before current frame for non-conditioning memory
            # the earliest one has t_pos=1 and the latest one has t_pos=self.num_maskmem-1
            # We also allow taking the memory frame non-consecutively (with r>1), in which case
            # we take (self.num_maskmem - 2) frames among every r-th frames plus the last frame.
            r = self.memory_temporal_stride_for_eval
            for t_pos in range(1, self.num_maskmem):
                t_rel = self.num_maskmem - t_pos  # how many frames before current frame
                if t_rel == 1:
                    # for t_rel == 1, we take the last frame (regardless of r)
                    if not track_in_reverse:
                        # the frame immediately before this frame (i.e. frame_idx - 1)
                        prev_frame_idx = frame_idx - t_rel
                    else:
                        # the frame immediately after this frame (i.e. frame_idx + 1)
                        prev_frame_idx = frame_idx + t_rel
                else:
                    # for t_rel >= 2, we take the memory frame from every r-th frames
                    if not track_in_reverse:
                        # first find the nearest frame among every r-th frames before this frame
                        # for r=1, this would be (frame_idx - 2)
                        prev_frame_idx = ((frame_idx - 2) // r) * r
                        # then seek further among every r-th frames
                        prev_frame_idx = prev_frame_idx - (t_rel - 2) * r
                    else:
                        # first find the nearest frame among every r-th frames after this frame
                        # for r=1, this would be (frame_idx + 2)
                        prev_frame_idx = -(-(frame_idx + 2) // r) * r
                        # then seek further among every r-th frames
                        prev_frame_idx = prev_frame_idx + (t_rel - 2) * r
                out = output_dict["non_cond_frame_outputs"].get(prev_frame_idx, None)
                if out is None:
                    # If an unselected conditioning frame is among the last (self.num_maskmem - 1)
                    # frames, we still attend to it as if it's a non-conditioning frame.
                    out = unselected_cond_outputs.get(prev_frame_idx, None)
                t_pos_and_prevs.append((t_pos, out))

            for t_pos, prev in t_pos_and_prevs:
                if prev is None:
                    continue  # skip padding frames
                # "maskmem_features" might have been offloaded to CPU in demo use cases,
                # so we load it back to GPU (it's a no-op if it's already on GPU).
                feats = prev["maskmem_features"]
                #to_cat_memory.append(feats.flatten(2).permute(2, 0, 1))
                to_cat_memory.append(np.transpose(np.reshape(feats, (feats.shape[0], feats.shape[1], feats.shape[2]*feats.shape[3])), (2, 0, 1)))
                # Spatial positional encoding (it might have been offloaded to CPU in eval)
                maskmem_enc = prev["maskmem_pos_enc"][-1]
                #maskmem_enc = maskmem_enc.flatten(2).permute(2, 0, 1)
                maskmem_enc = np.transpose(np.reshape(maskmem_enc, (maskmem_enc.shape[0], maskmem_enc.shape[1], maskmem_enc.shape[2]*maskmem_enc.shape[3])), (2, 0, 1))
                # Temporal positional encoding
                maskmem_enc = (
                    maskmem_enc + self.maskmem_tpos_enc[self.num_maskmem - t_pos - 1]
                )
                to_cat_memory_pos_embed.append(maskmem_enc)

            # Construct the list of past object pointers
            if self.use_obj_ptrs_in_encoder:
                max_obj_ptrs_in_encoder = min(num_frames, self.max_obj_ptrs_in_encoder)
                # First add those object pointers from selected conditioning frames
                # (optionally, only include object pointers in the past during evaluation)
                if not self.training and self.only_obj_ptrs_in_the_past_for_eval:
                    ptr_cond_outputs = {
                        t: out
                        for t, out in selected_cond_outputs.items()
                        if (t >= frame_idx if track_in_reverse else t <= frame_idx)
                    }
                else:
                    ptr_cond_outputs = selected_cond_outputs
                pos_and_ptrs = [
                    # Temporal pos encoding contains how far away each pointer is from current frame
                    (abs(frame_idx - t), out["obj_ptr"])
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
                if len(pos_and_ptrs) > 0:
                    pos_list, ptrs_list = zip(*pos_and_ptrs)
                    # stack object pointers along dim=0 into [ptr_seq_len, B, C] shape
                    obj_ptrs = np.stack(ptrs_list, axis=0)
                    # a temporal positional embedding based on how far each object pointer is from
                    # the current frame (sine embedding normalized by the max pointer num).
                    if self.add_tpos_enc_to_obj_ptrs:
                        t_diff_max = max_obj_ptrs_in_encoder - 1
                        tpos_dim = C if self.proj_tpos_enc_in_obj_ptrs else self.mem_dim
                        obj_pos = np.array(pos_list)
                        obj_pos = get_1d_sine_pe(obj_pos / t_diff_max, dim=tpos_dim)

                        tpos = np.zeros((obj_pos.shape[0], 64), np.float32)
                        for i in range(obj_pos.shape[0]):
                            if self.onnx:
                                tpos[i:i+1,:] = obj_ptr_tpos_proj.run(None, {"x": obj_pos[i:i+1,:].astype(np.float32)})[0]
                            else:
                                tpos[i:i+1,:] = obj_ptr_tpos_proj.run({"x": obj_pos[i:i+1,:].astype(np.float32)})[0]
                        obj_pos = tpos

                        #obj_pos = obj_pos.unsqueeze(1).expand(-1, B, self.mem_dim)
                        obj_pos_expanded = np.expand_dims(obj_pos, axis=1) # numpy
                        obj_pos = np.tile(obj_pos_expanded, (1, B, 1))
                    else:
                        obj_pos = np.zeros((len(pos_list), B, self.mem_dim), dtype=np.float32)
                    if self.mem_dim < C:
                        # split a pointer into (C // self.mem_dim) tokens for self.mem_dim < C
                        obj_ptrs = obj_ptrs.reshape(
                            -1, B, C // self.mem_dim, self.mem_dim
                        )
                        obj_ptrs = np.transpose(obj_ptrs, (0, 2, 1, 3))
                        obj_ptrs = np.reshape(obj_ptrs, (obj_ptrs.shape[0] * obj_ptrs.shape[1],) + obj_ptrs.shape[2:])
                        obj_pos = np.repeat(obj_pos, C // self.mem_dim, axis=0)
                    to_cat_memory.append(obj_ptrs)
                    to_cat_memory_pos_embed.append(obj_pos)
                    num_obj_ptr_tokens = obj_ptrs.shape[0]
                else:
                    num_obj_ptr_tokens = 0
        else:
            # for initial conditioning frames, encode them without using any previous memory
            if self.directly_add_no_mem_embed:
                # directly add no-mem embedding (instead of using the transformer encoder)
                pix_feat_with_mem = current_vision_feats[-1] + self.no_mem_embed
                pix_feat_with_mem = np.transpose(pix_feat_with_mem, (1, 2, 0)).reshape(B, C, H, W)
                return pix_feat_with_mem

            # Use a dummy token on the first frame (to avoid empty memory input to tranformer encoder)
            to_cat_memory = [self.no_mem_embed.expand(1, B, self.mem_dim)]
            to_cat_memory_pos_embed = [self.no_mem_pos_enc.expand(1, B, self.mem_dim)]

        # Step 2: Concatenate the memories and forward through the transformer encoder
        memory = np.concatenate(to_cat_memory, axis=0)
        memory_pos_embed = np.concatenate(to_cat_memory_pos_embed, axis=0)

        num_obj_ptr_tokens_numpy = np.array((num_obj_ptr_tokens)).astype(np.int64)
        if self.debug:
            print("curr", np.sum(current_vision_feats[0]))
            print("memory", np.sum(memory))
            print("curr_pos", np.sum(current_vision_pos_embeds[0]))
            print("memory_pos", np.sum(memory_pos_embed))
            print("num_obj_ptr_tokens", np.sum(num_obj_ptr_tokens_numpy))
        if self.benchmark:
            start = int(round(time.time() * 1000))

        if not self.legacy or self.version == "2.1":
            memory_1 = memory[:-num_obj_ptr_tokens,:,:]
            memory_2 = memory[-num_obj_ptr_tokens:,:,:]
            memory_pos_embed_1 = memory_pos_embed[:-num_obj_ptr_tokens,:,:]
            memory_pos_embed_2 = memory_pos_embed[-num_obj_ptr_tokens:,:,:]
            attention_mask_1 = np.zeros((memory_1.shape[0], memory_1.shape[1]), dtype=np.bool_)
            attention_mask_2 = np.zeros((memory_2.shape[0], memory_2.shape[1]), dtype=np.bool_)
            attention_mask_1[:memory_1.shape[0],:] = True
            attention_mask_2[:memory_2.shape[0],:] = True
            if self.onnx:
                pix_feat_with_mem = memory_attention.run(None, {"curr":current_vision_feats[0], "memory_1":memory_1, "memory_2":memory_2, "curr_pos":current_vision_pos_embeds[0], "memory_pos_1":memory_pos_embed_1, "memory_pos_2":memory_pos_embed_2, "attention_mask_1":attention_mask_1, "attention_mask_2":attention_mask_2})
            else:
                pix_feat_with_mem = memory_attention.run({"curr":current_vision_feats[0], "memory_1":memory_1, "memory_2":memory_2, "curr_pos":current_vision_pos_embeds[0], "memory_pos_1":memory_pos_embed_1, "memory_pos_2":memory_pos_embed_2, "attention_mask_1":attention_mask_1, "attention_mask_2":attention_mask_2})
        else:
            # Legacy SAM2: 4D matmul with batch=1, split memory without attention_mask
            memory_1 = memory[:-num_obj_ptr_tokens,:,:]
            memory_2 = memory[-num_obj_ptr_tokens:,:,:]
            memory_pos_embed_1 = memory_pos_embed[:-num_obj_ptr_tokens,:,:]
            memory_pos_embed_2 = memory_pos_embed[-num_obj_ptr_tokens:,:,:]
            if self.onnx:
                pix_feat_with_mem = memory_attention.run(None, {"curr":current_vision_feats[0], "memory_1":memory_1, "memory_2":memory_2, "curr_pos":current_vision_pos_embeds[0], "memory_pos_1":memory_pos_embed_1, "memory_pos_2":memory_pos_embed_2})
            else:
                pix_feat_with_mem = memory_attention.run({"curr":current_vision_feats[0], "memory_1":memory_1, "memory_2":memory_2, "curr_pos":current_vision_pos_embeds[0], "memory_pos_1":memory_pos_embed_1, "memory_pos_2":memory_pos_embed_2})

        if self.benchmark:
            end = int(round(time.time() * 1000))
            estimation_time = (end - start)
            logger.info(f'\tmemory_attention processing {estimation_time} ms')

        pix_feat_with_mem = pix_feat_with_mem[0]
        
        # reshape the output (HW)BC => BCHW
        pix_feat_with_mem = np.transpose(pix_feat_with_mem, (1, 2, 0)).reshape(B, C, H, W)
        return pix_feat_with_mem

    def _encode_new_memory(
        self,
        current_vision_feats,
        feat_sizes,
        pred_masks_high_res,
        is_mask_from_pts,
        memory_encoder
    ):
        """Encode the current image and its prediction into a memory feature."""
        B = current_vision_feats[-1].shape[1]  # batch size on this frame
        C = self.hidden_dim
        H, W = feat_sizes[-1]  # top-level (lowest-resolution) feature size
        # top-level feature, (HW)BC => BCHW
        pix_feat = np.transpose(current_vision_feats[-1], (1, 2, 0)).reshape(B, C, H, W)
        if self.non_overlap_masks_for_mem_enc and not self.training:
            # optionally, apply non-overlapping constraints to the masks (it's applied
            # in the batch dimension and should only be used during eval, where all
            # the objects come from the same video under batch size 1).
            pred_masks_high_res = self._apply_non_overlapping_constraints(
                pred_masks_high_res
            )
        # scale the raw mask logits with a temperature before applying sigmoid
        binarize = self.binarize_mask_from_pts_for_mem_enc and is_mask_from_pts
        if binarize and not self.training:
            mask_for_mem = (pred_masks_high_res > 0).astype(np.float32)
        else:
            # apply sigmoid on the raw mask logits to turn them into range (0, 1)
            mask_for_mem = sigmoid(pred_masks_high_res)
        # apply scale and bias terms to the sigmoid probabilities
        if self.sigmoid_scale_for_mem_enc != 1.0:
            mask_for_mem = mask_for_mem * self.sigmoid_scale_for_mem_enc
        if self.sigmoid_bias_for_mem_enc != 0.0:
            mask_for_mem = mask_for_mem + self.sigmoid_bias_for_mem_enc

        if self.benchmark:
            start = int(round(time.time() * 1000))

        if self.onnx:
            vision_features, vision_pos_enc = memory_encoder.run(None, {"pix_feat":pix_feat, "masks":mask_for_mem})
        else:
            vision_features, vision_pos_enc = memory_encoder.run({"pix_feat":pix_feat, "masks":mask_for_mem})

        if self.benchmark:
            end = int(round(time.time() * 1000))
            estimation_time = (end - start)
            logger.info(f'\tmemory_encoder processing {estimation_time} ms')

        maskmem_out = {"vision_features": vision_features, "vision_pos_enc": [vision_pos_enc]}

        maskmem_features = maskmem_out["vision_features"]
        maskmem_pos_enc = maskmem_out["vision_pos_enc"]

        return maskmem_features, maskmem_pos_enc

    # sam2_base.py
    def _prepare_backbone_features(self, backbone_out):
        """Prepare and flatten visual features."""
        backbone_out = backbone_out.copy()
        assert len(backbone_out["backbone_fpn"]) == len(backbone_out["vision_pos_enc"])
        assert len(backbone_out["backbone_fpn"]) >= self.num_feature_levels

        feature_maps = backbone_out["backbone_fpn"][-self.num_feature_levels :]
        vision_pos_embeds = backbone_out["vision_pos_enc"][-self.num_feature_levels :]

        feat_sizes = [(x.shape[-2], x.shape[-1]) for x in vision_pos_embeds]
        # flatten NxCxHxW to HWxNxC
        #vision_feats = [x.flatten(2).permute(2, 0, 1) for x in feature_maps]
        #vision_pos_embeds = [x.flatten(2).permute(2, 0, 1) for x in vision_pos_embeds]

        vision_feats = [np.transpose(np.reshape(x, (x.shape[0], x.shape[1], x.shape[2] * x.shape[3])), (2, 0, 1)) for x in feature_maps]
        vision_pos_embeds = [np.transpose(np.reshape(x, (x.shape[0], x.shape[1], x.shape[2] * x.shape[3])), (2, 0, 1)) for x in vision_pos_embeds]

        return backbone_out, vision_feats, vision_pos_embeds, feat_sizes

    # sam2_base.py
    def _use_multimask(self, is_init_cond_frame, point_inputs):
        """Whether to use multimask output in the SAM head."""
        num_pts = 0 if point_inputs is None else point_inputs["point_labels"].shape[1]
        multimask_output = (
            self.multimask_output_in_sam
            and (is_init_cond_frame or self.multimask_output_for_tracking)
            and (self.multimask_min_pt_num <= num_pts <= self.multimask_max_pt_num)
        )
        return multimask_output

    def _apply_non_overlapping_constraints(self, pred_masks):
        """
        Apply non-overlapping constraints to the object scores in pred_masks. Here we
        keep only the highest scoring object at each spatial location in pred_masks.
        """
        batch_size = pred_masks.size(0)
        if batch_size == 1:
            return pred_masks

        # "max_obj_inds": object index of the object with the highest score at each location
        max_obj_inds = np.argmax(pred_masks, axis=0, keepdim=True)
        # "batch_obj_inds": object index of each object slice (along dim 0) in `pred_masks`
        batch_obj_inds = np.arange(batch_size)[:, None, None, None]
        keep = max_obj_inds == batch_obj_inds
        # suppress overlapping regions' scores below -10.0 so that the foreground regions
        # don't overlap (here sigmoid(-10.0)=4.5398e-05)
        pred_masks = np.where(keep, pred_masks, np.clip(pred_masks, max=-10.0))
        return pred_masks

    # mask_decoder.py
    def _get_stability_scores(self, mask_logits):
        """
        Compute stability scores of the mask logits based on the IoU between upper and
        lower thresholds, similar to https://github.com/fairinternal/onevision/pull/568.
        """
        mask_logits = np.reshape(mask_logits, mask_logits.shape[:-2] + (mask_logits.shape[-2] * mask_logits.shape[-1],))
        stability_delta = self.dynamic_multimask_stability_delta
        area_i = np.sum(mask_logits > stability_delta, axis=-1).astype(np.float32)
        area_u = np.sum(mask_logits > -stability_delta, axis=-1).astype(np.float32)
        stability_scores = np.where(area_u > 0, area_i / area_u, 1.0)
        return stability_scores

    def _dynamic_multimask_via_stability(self, all_mask_logits, all_iou_scores):
        """
        When outputting a single mask, if the stability score from the current single-mask
        output (based on output token 0) falls below a threshold, we instead select from
        multi-mask outputs (based on output token 1~3) the mask with the highest predicted
        IoU score. This is intended to ensure a valid mask for both clicking and tracking.
        """
        # The best mask from multimask output tokens (1~3)
        multimask_logits = all_mask_logits[:, 1:, :, :]
        multimask_iou_scores = all_iou_scores[:, 1:]
        best_scores_inds = np.argmax(multimask_iou_scores, axis=-1)
        batch_inds = np.arange(
            multimask_iou_scores.shape[0]
        )
        best_multimask_logits = multimask_logits[batch_inds, best_scores_inds]
        best_multimask_logits = np.expand_dims(best_multimask_logits, axis=1)
        best_multimask_iou_scores = multimask_iou_scores[batch_inds, best_scores_inds]
        best_multimask_iou_scores = np.expand_dims(best_multimask_iou_scores, axis=1)

        # The mask from singlemask output token 0 and its stability score
        singlemask_logits = all_mask_logits[:, 0:1, :, :]
        singlemask_iou_scores = all_iou_scores[:, 0:1]
        stability_scores = self._get_stability_scores(singlemask_logits)
        is_stable = stability_scores >= self.dynamic_multimask_stability_thresh

        # Dynamically fall back to best multimask output upon low stability scores.

        #broadcasted_is_stable = is_stable[..., None, None].expand_as(singlemask_logits)

        #broadcasted_is_stable = torch.tensor(is_stable)[..., None, None]
        #broadcasted_is_stable = broadcasted_is_stable.expand_as(torch.tensor(singlemask_logits)).numpy()

        expanded_is_stable = np.expand_dims(is_stable, axis=(-1, -2))
        broadcasted_is_stable = np.broadcast_to(expanded_is_stable, singlemask_logits.shape)

        mask_logits_out = np.where(
            broadcasted_is_stable,
            singlemask_logits,
            best_multimask_logits,
        )

        #broadcasted_is_stable = is_stable.expand_as(singlemask_iou_scores)

        #broadcasted_is_stable = torch.tensor(is_stable)
        #broadcasted_is_stable = broadcasted_is_stable.expand_as(torch.tensor(singlemask_iou_scores)).numpy()

        broadcasted_is_stable = np.broadcast_to(is_stable, singlemask_iou_scores.shape)

        iou_scores_out = np.where(
            broadcasted_is_stable,
            singlemask_iou_scores,
            best_multimask_iou_scores,
        )
        return mask_logits_out, iou_scores_out


