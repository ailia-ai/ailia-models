"""video_tracking.py — SAM 3.1 ONNX tracking (bucket mode).

Entry point: Sam3Tracker

    tracker = Sam3Tracker(models, maskmem_tpos_enc, no_obj_params)
    tracker.add_prompt(frame, caption)          # frame 0, text grounding
    for fi, scores, boxes, masks, ids in tracker.propagate_in_video(frame_paths):
        ...                                     # frames 1 → N
    tracker.remove_object(obj_idx)              # drop object mid-stream
"""

import cv2
import numpy as np
from math_utils import sigmoid
from resize_utils import tv_resize
from segment_anything_3_1 import (
    MASK_CHANNELS,
    MAX_OBJ_PTRS,
    MEMORY_MASK_SIZE,
    MULTIPLEX_COUNT,
    NO_OBJ_SCORE,
    NUM_MASKMEM,
    OBJ_SCORE_THRESHOLD,
    SIGMOID_BIAS_FOR_MEM_ENC,
    SIGMOID_SCALE_FOR_MEM_ENC,
    VALID_EMBED_PATH,
    args,
    postprocess,
    preprocess,
    run_encoder,
    run_grounding,
    tokenize,
)

INVALID_EMBED_PATH = "npy/sam3.1_output_invalid_embed.npy"
INTERACTIVE_MASK_DWN_WEIGHT_PATH = "npy/sam3.1_interactive_mask_downsample_weight.npy"
INTERACTIVE_MASK_DWN_BIAS_PATH = "npy/sam3.1_interactive_mask_downsample_bias.npy"

# Set to False when GPU memory is large enough to keep all models loaded simultaneously.
UNLOAD_MODELS_BETWEEN_STEPS = True

# Set to True to enable periodic mask reconditioning (every 16 frames).
# Adds prompt_encoder + mask_decoder calls per matched object; disable for speed.
ENABLE_RECONDITION = False


# ── SAM3.1 tracking parameters ────────────────────────────────────────────────

SCORE_THRESH_DET = 0.4  # score_threshold_detection
DET_NMS_IOM_THRESH = 0.1  # det_nms_thresh  (det_nms_use_iom=True)
ASSOC_IOM_THRESH = 0.1  # assoc_iou_thresh (use_iom_recondition=True → IoM)
NEW_DET_THRESH = 0.65  # new_det_thresh
MASKLET_CONFIRM_N = 3  # masklet_confirmation_consecutive_det_thresh
UNCONFIRMED_STATUS_DELAY = MASKLET_CONFIRM_N - 1  # lookahead frames for confirmed check
HOTSTART_DELAY = 15  # hotstart_delay
HOTSTART_UNMATCH_THRESH = 8  # hotstart_unmatch_thresh
HOTSTART_DUP_THRESH = 8  # hotstart_dup_thresh
# suppress_unmatched_only_within_hotstart=False → keep_alive suppression is always active
TRK_KEEP_ALIVE_MAX = 8  # max_trk_keep_alive  (Sam3MultiplexBase default)
TRK_KEEP_ALIVE_MIN = -4  # min_trk_keep_alive  (Sam3MultiplexBase default)
INIT_TRK_KEEP_ALIVE = 0  # init_trk_keep_alive (Sam3MultiplexBase default)
SUPPRESS_OVERLAP_IOU_THRESH = (
    0.7  # suppress_overlapping_based_on_recent_occlusion_threshold
)
DET_BOUNDARY_MARGIN = 0.025  # suppress_det_close_to_boundary=True, margin=0.025
NEVER_OCCLUDED = -1  # last_occluded sentinel: object has never been occluded
ALWAYS_OCCLUDED = (
    100_000  # last_occluded sentinel: treated as always-occluded (hotstart-removed)
)


# ── Cached weights (loaded once) ──────────────────────────────────────────────

valid_embed_cache = None  # (16, 256) float32
invalid_embed_cache = None  # (16, 256) float32
imd_weight_cache = None  # (1, 1, 4, 4) float32
imd_bias_cache = None  # (1,) float32


def load_embeds():
    global valid_embed_cache, invalid_embed_cache, imd_weight_cache, imd_bias_cache
    if valid_embed_cache is None:
        valid_embed_cache = np.load(VALID_EMBED_PATH).astype(np.float32)
        invalid_embed_cache = np.load(INVALID_EMBED_PATH).astype(np.float32)
        imd_weight_cache = np.load(INTERACTIVE_MASK_DWN_WEIGHT_PATH).astype(np.float32)
        imd_bias_cache = np.load(INTERACTIVE_MASK_DWN_BIAS_PATH).astype(np.float32)


# ── MemoryBank ────────────────────────────────────────────────────────────────


class MemoryBank:
    """Stores per-frame spatial features and object pointers for memory_attention.

    Conditioning frame (frame 0): kept forever, tpos = maskmem_tpos_enc[NUM_MASKMEM-1].
    Non-conditioning frames: rolling window of at most NON_COND_MAX recent frames,
    tpos based on relative age.

    memory_obj layout:
      [ cond_frame (HW) | spatial_0..k (HW each) | ptr_0..j (MULTIPLEX_COUNT each) ]
      T_mem = (1+k)*HW + j*MULTIPLEX_COUNT,  T_img = (1+k)*HW
    """

    NON_COND_MAX = NUM_MASKMEM - 1  # = 6 spatial non-cond frames
    NON_COND_PTR_MAX = (
        MAX_OBJ_PTRS - 1
    )  # = 15 non-cond ptr frames (1 slot reserved for cond)

    def __init__(self):
        self.cond_frame = None  # conditioning frame (frame 0), always kept
        self.spatial_frames = []  # non-conditioning sliding window, max NON_COND_MAX
        self.ptr_frames_nc = []  # non-cond ptr rolling window, max NON_COND_PTR_MAX

    def add(
        self, frame_idx, fpn2, pos2, mem_feat, mem_pos, all_ptrs, is_conditioning=False
    ):
        entry = dict(
            frame_idx=frame_idx,
            fpn2=fpn2,  # (1, 256, 72, 72)
            pos2=pos2,  # (1, 256, 72, 72)
            mem_feat=mem_feat,  # (1, 256, 72, 72)
            mem_pos=mem_pos,  # (1, 256, 72, 72)
            all_ptrs=all_ptrs.reshape(MULTIPLEX_COUNT, 256),  # (16, 256)
        )
        if is_conditioning:
            self.cond_frame = entry
        else:
            self.spatial_frames.append(entry)
            if len(self.spatial_frames) > self.NON_COND_MAX:
                self.spatial_frames.pop(0)
            # Cond ptr is kept separately; only non-cond ptrs roll off.
            self.ptr_frames_nc.append(entry)
            if len(self.ptr_frames_nc) > self.NON_COND_PTR_MAX:
                self.ptr_frames_nc.pop(0)

    def build_memory_inputs(self, current_frame_idx, maskmem_tpos_enc, models):
        """Build the 4 memory tensors required by memory_attention.

        Returns memory_obj, memory_obj_pos, memory_img, memory_img_pos.
        Each is float32 numpy of shape (T, 1, 256).
        """
        if self.cond_frame is None and not self.spatial_frames:
            raise RuntimeError(
                "MemoryBank is empty; call add() before build_memory_inputs()"
            )

        obj_spatial_list = []
        obj_spatial_pos_list = []
        img_spatial_list = []
        img_spatial_pos_list = []

        def _add_spatial_entry(entry, tpos):
            mf_flat = entry["mem_feat"].reshape(1, 256, -1).transpose(2, 0, 1)
            mp_flat = entry["mem_pos"].reshape(1, 256, -1).transpose(2, 0, 1)
            mp_flat = mp_flat + tpos.reshape(1, 1, 256)
            obj_spatial_list.append(mf_flat)
            obj_spatial_pos_list.append(mp_flat)
            f2_flat = entry["fpn2"].reshape(1, 256, -1).transpose(2, 0, 1)
            p2_flat = entry["pos2"].reshape(1, 256, -1).transpose(2, 0, 1)
            p2_flat = p2_flat + tpos.reshape(1, 1, 256)
            img_spatial_list.append(f2_flat)
            img_spatial_pos_list.append(p2_flat)

        # Conditioning frame: use age-based tpos (use_maskmem_tpos_v2=True in SAM3.1)
        # tpos formula: t_pos<=0 or t_pos>=num_maskmem → index[num_maskmem-1]
        if self.cond_frame is not None:
            t_pos = current_frame_idx - self.cond_frame["frame_idx"]
            if t_pos <= 0 or t_pos >= NUM_MASKMEM:
                cond_tpos = maskmem_tpos_enc[NUM_MASKMEM - 1]
            else:
                cond_tpos = maskmem_tpos_enc[NUM_MASKMEM - t_pos - 1]
            _add_spatial_entry(self.cond_frame, cond_tpos)

        # Non-conditioning frames: tpos by actual frame distance.
        # Newest (1 frame ago) → enc[0], oldest (6 frames ago) → enc[5].
        for entry in self.spatial_frames:
            t_pos = current_frame_idx - entry["frame_idx"]
            tpos_idx = min(t_pos - 1, NUM_MASKMEM - 2)
            tpos = maskmem_tpos_enc[tpos_idx]
            _add_spatial_entry(entry, tpos)

        # Ptr tokens: cond_frame ptr (always kept) + non-cond ptrs (rolling window).
        ptr_entries = []
        if self.cond_frame is not None:
            ptr_entries.append(self.cond_frame)
        ptr_entries.extend(self.ptr_frames_nc)

        t_diffs = np.array(
            [current_frame_idx - e["frame_idx"] for e in ptr_entries],
            dtype=np.float32,
        )
        sine_pes = get_1d_sine_pe(t_diffs / (MAX_OBJ_PTRS - 1), dim=256)  # (J, 256)
        tpos_proj = run_obj_ptr_tpos_proj(models, sine_pes)  # (J, 256)

        obj_ptr_list = []
        for e in ptr_entries:
            block = e["all_ptrs"].reshape(MULTIPLEX_COUNT, 1, 256)
            obj_ptr_list.append(block)

        obj_ptr_pos_list = []
        for i in range(len(ptr_entries)):
            pos = tpos_proj[i].reshape(1, 256)  # (1, 256)
            # repeat MULTIPLEX_COUNT times: each slot gets the same tpos
            pos_block = np.repeat(pos[np.newaxis], MULTIPLEX_COUNT, axis=0).reshape(
                MULTIPLEX_COUNT, 1, 256
            )
            obj_ptr_pos_list.append(pos_block)

        memory_obj = np.concatenate(obj_spatial_list + obj_ptr_list, axis=0)
        memory_obj_pos = np.concatenate(obj_spatial_pos_list + obj_ptr_pos_list, axis=0)
        memory_img = np.concatenate(img_spatial_list, axis=0)
        memory_img_pos = np.concatenate(img_spatial_pos_list, axis=0)

        return memory_obj, memory_obj_pos, memory_img, memory_img_pos


def get_1d_sine_pe(positions, dim=256):
    """1D sinusoidal PE.

    positions : (N,) float32, normalized values
    returns   : (N, dim) float32
    """
    assert dim % 2 == 0
    half = dim // 2
    freq = 1.0 / (10000.0 ** (np.arange(half, dtype=np.float32) / half))
    pos = np.array(positions, dtype=np.float32)  # (N,)
    angles = pos[:, None] * freq[None, :]  # (N, half)
    pe = np.concatenate([np.sin(angles), np.cos(angles)], axis=-1)  # (N, dim)
    return pe


def run_obj_ptr_tpos_proj(models, pe_input):
    """pe_input : (N, 256) sine PE → (N, 256) projected tpos"""
    tpos = models["tpos_proj"]
    x = pe_input.astype(np.float32)
    if not args.onnx:
        out = tpos.predict([x])
    else:
        out = tpos.run(None, {"x": x})
    return out[0]  # (N, 256)


# ── Mask preprocessing ────────────────────────────────────────────────────────


def apply_interactive_mask_downsample(binary_mask_hw):
    """Conv2d(1,1,kernel=4,stride=4) at original resolution → (H//4, W//4)."""
    load_embeds()
    kernel = imd_weight_cache[0, 0]
    bias = float(imd_bias_cache[0])
    H, W = binary_mask_hw.shape
    pH = ((H + 3) // 4) * 4
    pW = ((W + 3) // 4) * 4
    if pH != H or pW != W:
        padded = np.zeros((pH, pW), dtype=np.float32)
        padded[:H, :W] = binary_mask_hw
    else:
        padded = binary_mask_hw
    out_H, out_W = pH // 4, pW // 4
    patches = padded.reshape(out_H, 4, out_W, 4)
    out = np.tensordot(patches, kernel, axes=([1, 3], [0, 1])) + bias
    return out.astype(np.float32)


def mask_for_prompt_encoder(binary_mask_hw, mask_input_size=(288, 288)):
    """Conv2d downsample + bilinear resize → (1, 1, H, W)."""
    ds = apply_interactive_mask_downsample(binary_mask_hw)
    resized = tv_resize(ds, (mask_input_size[0], mask_input_size[1]))
    return resized[np.newaxis, np.newaxis].astype(np.float32)


# ── 32-channel mask builders ──────────────────────────────────────────────────


def build_combined_32ch_mask(binary_masks_orig, is_conditioning):
    """
    (1, 32, MEMORY_MASK_SIZE, MEMORY_MASK_SIZE) for N objects.

    Channel layout mandated by the memory encoder:
      ch 0..N-1      : object mask probabilities (sigmoid scale+bias applied)
      ch N..15       : 0  (unused object slots)
      ch 16..16+N-1  : 1 if conditioning frame, else 0
      ch 16+N..31    : 0
    """
    N = len(binary_masks_orig)
    masks_32ch = np.zeros(
        (1, MASK_CHANNELS, MEMORY_MASK_SIZE, MEMORY_MASK_SIZE), dtype=np.float32
    )
    for k, bm in enumerate(binary_masks_orig):
        bm = bm.astype(np.float32)
        # convert binary {0,1} to logit space so sigmoid recovers near 0/1
        logit = bm * 20.0 - 10.0
        prob = sigmoid(logit) * SIGMOID_SCALE_FOR_MEM_ENC + SIGMOID_BIAS_FOR_MEM_ENC
        resized = tv_resize(prob, (MEMORY_MASK_SIZE, MEMORY_MASK_SIZE))
        masks_32ch[0, k] = resized
        masks_32ch[0, 16 + k] = 1.0 if is_conditioning else 0.0
    return masks_32ch


def build_32ch_mask_tracking(logit_masks_best, is_app_flags):
    """(1, 32, MEMORY_MASK_SIZE, MEMORY_MASK_SIZE) for tracking frames.

    ch 16+k is always 0: tracking frames are never conditioning frames.
    Absent objects get NO_OBJ_SCORE logit so their memory contribution is suppressed.
    """
    masks_32ch = np.zeros(
        (1, MASK_CHANNELS, MEMORY_MASK_SIZE, MEMORY_MASK_SIZE), dtype=np.float32
    )
    for k, (logit, is_app) in enumerate(zip(logit_masks_best, is_app_flags)):
        if not is_app:
            logit = np.full_like(logit, NO_OBJ_SCORE)
        prob = sigmoid(logit) * SIGMOID_SCALE_FOR_MEM_ENC + SIGMOID_BIAS_FOR_MEM_ENC
        resized = tv_resize(prob, (MEMORY_MASK_SIZE, MEMORY_MASK_SIZE))
        masks_32ch[0, k] = resized
        masks_32ch[0, 16 + k] = 0.0
    return masks_32ch


def build_32ch_mask(logit_mask, is_conditioning):
    """
    logit_mask     : (1, 1, H, W) float32 — raw logit mask from mask_decoder (best slot)
    is_conditioning: bool — True for initial frame, False for tracking frames
    Returns        : (1, 32, 1152, 1152) float32

    sigmoid must be applied BEFORE resize because sigmoid is non-linear.
    """
    masks_32ch = np.zeros(
        (1, MASK_CHANNELS, MEMORY_MASK_SIZE, MEMORY_MASK_SIZE), np.float32
    )
    prob = sigmoid(logit_mask[0, 0])  # (H, W), values in [0, 1]
    prob = prob * SIGMOID_SCALE_FOR_MEM_ENC + SIGMOID_BIAS_FOR_MEM_ENC  # [-1, 1]
    resized = tv_resize(prob, (MEMORY_MASK_SIZE, MEMORY_MASK_SIZE))
    masks_32ch[0, 0] = resized
    masks_32ch[0, 16] = 1.0 if is_conditioning else 0.0
    return masks_32ch


# ── Memory input assembly ─────────────────────────────────────────────────────


def build_combined_memory_inputs(memory_banks, frame_idx, maskmem_tpos_enc, models):
    """
    Returns (memory_obj, memory_obj_pos, memory_img, memory_img_pos).

    All banks store identical spatial features and full 16-slot ptr history.
    Bank 0 is representative for all banks.
    """
    return memory_banks[0].build_memory_inputs(frame_idx, maskmem_tpos_enc, models)


# ── Prompt / decoder / memory runners ────────────────────────────────────────


def box_to_corner_points(box, orig_h, orig_w):
    """Convert [x1,y1,x2,y2] pixel box → corner point prompt for prompt_encoder.

    SAM2 convention: top-left = label 2, bottom-right = label 3.
    Returns coords(1,2,2) float32 in pixel space, labels(1,2) int32.
    """
    x1, y1, x2, y2 = box
    coords = np.array([[[x1, y1], [x2, y2]]], dtype=np.float32)  # (1, 2, 2)
    labels = np.array([[2, 3]], dtype=np.int32)  # (1, 2)
    return coords, labels


def run_prompt_encoder(models, coords, labels, mask, mask_enable):
    """
    coords       : (B, P, 2) float32 — pixel coords
    labels       : (B, P)    int32
    mask         : (B, 1, 288, 288) float32
    mask_enable  : (1,) int32 — 1 to use mask, 0 to ignore
    Returns      : sparse_emb, dense_emb, dense_pe
    """
    pe = models["prompt_enc"]
    feed = {
        "coords": coords.astype(np.float32),
        "labels": labels.astype(np.int32),
        "masks": mask.astype(np.float32),
        "masks_enable": mask_enable.astype(np.int32),
    }
    if not args.onnx:
        out = pe.predict(list(feed.values()))
    else:
        out = pe.run(None, feed)
    sparse_emb, dense_emb, dense_pe = out
    return sparse_emb, dense_emb, dense_pe


def run_mask_decoder(
    models, image_embeddings, image_pe, sparse_emb, dense_emb, fpn0, fpn1
):
    """
    Returns masks(B,4,288,288), iou_pred(B,4), sam_tokens_out(B,4,256), obj_score(B,1).
    high_res_features1 = fpn0 (288×288), high_res_features2 = fpn1 (144×144).
    """
    dec = models["mask_dec"]
    feed = {
        "image_embeddings": image_embeddings.astype(np.float32),
        "image_pe": image_pe.astype(np.float32),
        "sparse_prompt_embeddings": sparse_emb.astype(np.float32),
        "dense_prompt_embeddings": dense_emb.astype(np.float32),
        "high_res_features1": fpn0.astype(np.float32),
        "high_res_features2": fpn1.astype(np.float32),
    }
    if not args.onnx:
        out = dec.predict(list(feed.values()))
    else:
        out = dec.run(None, feed)
    masks, iou_pred, sam_tokens_out, object_score_logits = out
    return masks, iou_pred, sam_tokens_out, object_score_logits


def run_tracking_mask_decoder(models, image_embeddings, fpn0, fpn1, extra_embed):
    """
    Tracking-specific MultiplexMaskDecoder (bucket mode, no prompt encoder needed).
    image_embeddings : (1, 256, 72, 72) combined memory-enriched features
    fpn0             : (1, 256, 288, 288) raw FPN level 0
    fpn1             : (1, 256, 144, 144) raw FPN level 1
    extra_embed      : (1, 16, 256) per-slot valid/invalid embed (constructed at runtime)
    Returns masks(1,16,3,288,288), iou_pred(1,16,3), sam_tokens(1,16,3,256), obj_score(1,16)
    """
    tdec = models["track_dec"]
    feed = {
        "image_embeddings": image_embeddings.astype(np.float32),
        "high_res_feat0": fpn0.astype(np.float32),
        "high_res_feat1": fpn1.astype(np.float32),
        "extra_embed": extra_embed.astype(np.float32),
    }
    if not args.onnx:
        out = tdec.predict(list(feed.values()))
    else:
        out = tdec.run(None, feed)
    masks, iou_pred, sam_tokens, obj_score = out
    return masks, iou_pred, sam_tokens, obj_score


def run_memory_encoder(models, fpn2, masks_32ch):
    """
    fpn2       : (B, 256, 72, 72)
    masks_32ch : (B, 32, 1152, 1152)
    Returns    : vision_features(B,256,72,72), vision_pos_enc(B,256,72,72)
    """
    menc = models["mem_enc"]
    feed = {
        "pix_feat": fpn2.astype(np.float32),
        "masks": masks_32ch.astype(np.float32),
    }
    if not args.onnx:
        out = menc.predict(list(feed.values()))
    else:
        out = menc.run(None, feed)
    vision_features, vision_pos_enc = out
    return vision_features, vision_pos_enc


def run_memory_attention(
    models,
    curr_obj,
    curr_obj_pos,
    curr_img,
    memory_obj,
    memory_obj_pos,
    memory_img,
    memory_img_pos,
):
    """
    All inputs in (T, B, C) layout.
    curr_obj/curr_img : (HW, B, 256)
    memory_obj        : (k*5185, B, 256)
    memory_img        : (k*5184, B, 256)
    Returns pix_feat_with_mem : (HW, B, 256)
    """
    mattn = models["mem_attn"]
    feed = {
        "curr_obj": curr_obj.astype(np.float32),
        "curr_obj_pos": curr_obj_pos.astype(np.float32),
        "curr_img": curr_img.astype(np.float32),
        "memory_obj": memory_obj.astype(np.float32),
        "memory_obj_pos": memory_obj_pos.astype(np.float32),
        "memory_img": memory_img.astype(np.float32),
        "memory_img_pos": memory_img_pos.astype(np.float32),
    }
    if not args.onnx:
        out = mattn.predict(list(feed.values()))
    else:
        out = mattn.run(None, feed)
    pix_feat_with_mem = out[0]
    return pix_feat_with_mem


def run_obj_ptr_proj(models, sam_tokens_first):
    """Tracking obj_ptr_proj: sam_tokens_first (B, 256) → (B, 256)."""
    proj = models["obj_proj"]
    x = sam_tokens_first.astype(np.float32)
    if not args.onnx:
        out = proj.predict([x])
    else:
        out = proj.run(None, {"x": x})
    return out[0]  # (B, 256)


def run_interactive_obj_ptr_proj(models, sam_tokens_first):
    """Interactive obj_ptr_proj for init frame: sam_tokens_first (B, 256) → (B, 256).

    Uses interactive_obj_ptr_proj (different weights from obj_ptr_proj)
    when is_interactive=True (i.e. the first frame with a prompt).
    """
    proj = models["iobj_proj"]
    x = sam_tokens_first.astype(np.float32)
    if not args.onnx:
        out = proj.predict([x])
    else:
        out = proj.run(None, {"x": x})
    return out[0]  # (B, 256)


def mask_iom_matrix(masks_a, masks_b):
    """
    (N, H, W) × (M, H, W) → (N, M) float32 IoM matrix.
    IoM = intersection / min(area_a, area_b)
    Source: sam3/sam3/train/masks_ops.py::mask_iom
    """
    N = masks_a.shape[0]
    M = masks_b.shape[0]
    if N == 0 or M == 0:
        return np.zeros((N, M), dtype=np.float32)
    fa = (masks_a > 0).reshape(N, -1).astype(np.float32)
    fb = (masks_b > 0).reshape(M, -1).astype(np.float32)
    inter = fa @ fb.T  # (N, M)
    area_a = fa.sum(axis=1, keepdims=True)  # (N, 1)
    area_b = fb.sum(axis=1, keepdims=True).T  # (1, M)
    min_area = np.minimum(area_a, area_b)  # (N, M)
    return inter / (min_area + 1e-8)


def nms_masks_iom(masks_logit, scores, score_thresh, iom_thresh):
    """
    Greedy NMS using IoM.
    Source: sam3/sam3/model/sam3_multiplex_detector_utils.py::nms_masks (nms_use_iom=True)
    Returns bool keep array (K,).
    """
    is_valid = scores > score_thresh
    valid_idx = np.where(is_valid)[0]
    if len(valid_idx) == 0:
        return is_valid
    v_masks = (masks_logit[valid_idx] > 0).astype(np.float32)
    v_scores = scores[valid_idx]
    iom = mask_iom_matrix(v_masks, v_masks)  # (K, K)
    order = np.argsort(-v_scores)
    suppressed = np.zeros(len(valid_idx), dtype=bool)
    kept = []
    for i in order:
        if suppressed[i]:
            continue
        kept.append(i)
        for j in range(len(valid_idx)):
            if not suppressed[j] and j != i and iom[i, j] > iom_thresh:
                suppressed[j] = True
    keep = np.zeros(len(scores), dtype=bool)
    for ki in kept:
        keep[valid_idx[ki]] = True
    return keep


# ── Sam3Tracker ───────────────────────────────────────────────────────────────


class Sam3Tracker:
    """
    SAM 3.1 ONNX tracker.

    Usage:
        tracker = Sam3Tracker(models, maskmem_tpos_enc, no_obj_params, threshold=0.5)

        # frame 0
        scores, boxes, masks = tracker.add_prompt(frame, caption)
        # or
        scores, boxes, masks = tracker.add_prompt_interactive(frame, box=box)

        # frames 1 → N
        for fi, scores, boxes, masks, obj_ids in tracker.propagate_in_video(frame_paths):
            ...

        # optional: drop an object
        tracker.remove_object(obj_idx)

        # start over
        tracker.reset()
    """

    def __init__(self, models, maskmem_tpos_enc, no_obj_params, threshold=0.5):
        self.models = models
        self.maskmem_tpos_enc = maskmem_tpos_enc
        self.no_obj_params = no_obj_params
        self.threshold = threshold
        self.memory_banks = []  # list[MemoryBank], one per tracked object
        self.caption = None
        # Per-object metadata (parallel to memory_banks)
        self.obj_ids = []  # stable 1-indexed IDs (never reused)
        self.next_id = 1
        self.keep_alive = []  # int: +1 matched / -1 unmatched, clamped [MIN,MAX]
        self.consecutive_det_count = (
            []
        )  # int: consecutive frames matched by a detection
        self.confirmed = []  # bool: True after MASKLET_CONFIRM_N consecutive matches
        self.add_frame = []  # int: frame index when first added
        self.unmatch_total = []  # int: total unmatched frames (never resets)
        self.pairwise_overlap = np.zeros(
            (0, 0), dtype=np.int32
        )  # (N, N) overlap counts
        self.last_occluded = (
            []
        )  # int: frame_idx of last occlusion, NEVER_OCCLUDED if never
        self.current_frame = 0

    # ── state management ───────────────────────────────────────────────────────

    def reset(self):
        """Clear all tracked objects."""
        self.memory_banks = []
        self.obj_ids = []
        self.next_id = 1
        self.keep_alive = []
        self.consecutive_det_count = []
        self.confirmed = []
        self.add_frame = []
        self.unmatch_total = []
        self.pairwise_overlap = np.zeros((0, 0), dtype=np.int32)
        self.last_occluded = []
        self.current_frame = 0

    def _append_object_state(self, frame_idx, confirmed, keep_alive_init):
        """Append per-object state for a newly added object."""
        self.obj_ids.append(self.next_id)
        self.next_id += 1
        self.keep_alive.append(keep_alive_init)
        self.consecutive_det_count.append(MASKLET_CONFIRM_N if confirmed else 0)
        self.confirmed.append(confirmed)
        self.add_frame.append(frame_idx)
        self.unmatch_total.append(0)
        old_N = self.pairwise_overlap.shape[0]
        new_N = old_N + 1
        new_mat = np.zeros((new_N, new_N), dtype=np.int32)
        new_mat[:old_N, :old_N] = self.pairwise_overlap
        self.pairwise_overlap = new_mat
        self.last_occluded.append(NEVER_OCCLUDED)

    def _remove_object_state(self, obj_idx):
        """Remove per-object state at slot obj_idx (parallel to memory_banks.pop)."""
        self.obj_ids.pop(obj_idx)
        self.keep_alive.pop(obj_idx)
        self.consecutive_det_count.pop(obj_idx)
        self.confirmed.pop(obj_idx)
        self.add_frame.pop(obj_idx)
        self.unmatch_total.pop(obj_idx)
        self.pairwise_overlap = np.delete(
            np.delete(self.pairwise_overlap, obj_idx, axis=0), obj_idx, axis=1
        )
        self.last_occluded.pop(obj_idx)

    def remove_object(self, obj_idx):
        """
        Remove the object at index obj_idx from tracking.

        The remaining objects are renumbered.  The change takes effect on the
        next frame processed by propagate_in_video.
        """
        if obj_idx < 0 or obj_idx >= len(self.memory_banks):
            raise IndexError(
                f"obj_idx={obj_idx} out of range (n={len(self.memory_banks)})"
            )
        self.memory_banks.pop(obj_idx)
        self._remove_object_state(obj_idx)

    # ── frame 0 initialisation ─────────────────────────────────────────────────

    def add_prompt(self, frame, caption):
        """
        Text grounding on frame 0; detects objects and initialises memory banks.

        Returns (scores, boxes, bin_masks) for the detected objects.
        bank_idxs are implicitly 0..N-1.
        """
        self.caption = caption
        models = self.models
        no_obj_params = self.no_obj_params

        orig_h, orig_w = frame.shape[:2]
        enc_out = run_encoder(models, preprocess(frame))
        (
            fpn0,
            fpn1,
            fpn2,
            pos0,
            pos1,
            pos2,
            prop_fpn0,
            prop_fpn1,
            prop_fpn2,
            prop_pos2,
        ) = enc_out

        text_tokens = tokenize(caption)
        gnd_out = run_grounding(
            models,
            fpn0,
            fpn1,
            fpn2,
            pos2,
            text_tokens,
            np.zeros((0, 1, 4), dtype=np.float32),
            np.zeros((0, 1), dtype=np.int64),
            np.zeros((1, 0), dtype=bool),
        )
        pred_masks_gnd, pred_boxes_gnd, pred_logits_gnd, presence_gnd = gnd_out

        # Suppress detections whose center is within DET_BOUNDARY_MARGIN of image edges.
        boxes_cxcywh_f0 = pred_boxes_gnd[0]  # (200, 4) normalized cx,cy,w,h
        cx_f0 = boxes_cxcywh_f0[:, 0]
        cy_f0 = boxes_cxcywh_f0[:, 1]
        boundary_suppress_f0 = ~(
            (cx_f0 > DET_BOUNDARY_MARGIN)
            & (cx_f0 < 1.0 - DET_BOUNDARY_MARGIN)
            & (cy_f0 > DET_BOUNDARY_MARGIN)
            & (cy_f0 < 1.0 - DET_BOUNDARY_MARGIN)
        )
        pred_logits_gnd = pred_logits_gnd.copy()
        pred_logits_gnd[0, boundary_suppress_f0, 0] = -100.0

        scores_tmp, _, bin_masks_tmp = postprocess(
            pred_masks_gnd,
            pred_boxes_gnd,
            pred_logits_gnd,
            presence_gnd,
            orig_h,
            orig_w,
            self.threshold,
        )
        if len(scores_tmp) == 0:
            return np.zeros(0), np.zeros((0, 4)), np.zeros((0, orig_h, orig_w), bool)

        N = len(scores_tmp)

        # get per-object obj_ptr from the interactive decoder (one call per object)
        obj_ptrs = []
        for i in range(N):
            mfp = mask_for_prompt_encoder(
                bin_masks_tmp[i].astype(np.float32), mask_input_size=(288, 288)
            )
            coords = np.zeros((1, 1, 2), dtype=np.float32)
            labels = np.array([[-1]], dtype=np.int32)
            mask_enable = np.array([1], dtype=np.int32)
            sparse_emb, dense_emb, dense_pe = run_prompt_encoder(
                models, coords, labels, mfp, mask_enable
            )
            masks_dec, iou_pred, sam_tokens_out, _ = run_mask_decoder(
                models, prop_fpn2, dense_pe, sparse_emb, dense_emb, prop_fpn0, prop_fpn1
            )
            best_slot = int(np.argmax(iou_pred[0]))
            obj_ptrs.append(
                run_interactive_obj_ptr_proj(models, sam_tokens_out[:, best_slot, :])
            )

        # encode all N objects in a single combined call; all banks share these features
        combined_masks = build_combined_32ch_mask(
            [bin_masks_tmp[i] for i in range(N)], is_conditioning=True
        )
        mem_feat_combined, mem_pos_combined = run_memory_encoder(
            models, prop_fpn2, combined_masks
        )
        # add no-obj embeddings for unused slots (slots N..15) so the decoder
        # sees zero signal rather than an uninitialised embedding
        if no_obj_params is not None:
            mem_feat_combined = mem_feat_combined + no_obj_params[0][N:].sum(
                axis=0
            ).reshape(1, 256, 1, 1)

        # build combined 16-slot ptr block for the conditioning frame
        init_all_ptrs = np.zeros((MULTIPLEX_COUNT, 256), dtype=np.float32)
        for i in range(N):
            init_all_ptrs[i] = obj_ptrs[i].reshape(256)
        if no_obj_params is not None:
            _, W, b = no_obj_params
            for k in range(N, MULTIPLEX_COUNT):
                init_all_ptrs[k] = (
                    np.zeros((1, 256), dtype=np.float32) @ W.T + b
                ).reshape(256)

        # create MemoryBanks and per-object state
        self.memory_banks = []
        self.obj_ids = []
        self.next_id = 1
        self.keep_alive = []
        self.consecutive_det_count = []
        self.confirmed = []
        self.add_frame = []
        self.unmatch_total = []
        self.pairwise_overlap = np.zeros((0, 0), dtype=np.int32)
        self.current_frame = 0

        all_scores, all_boxes, all_masks = [], [], []
        for i in range(N):
            mb = MemoryBank()
            self.memory_banks.append(mb)
            mb.add(
                0,
                prop_fpn2,
                prop_pos2,
                mem_feat_combined,
                mem_pos_combined,
                init_all_ptrs,
                is_conditioning=True,
            )
            # Frame-0 prompt objects: pre-confirmed (INIT_TRK_KEEP_ALIVE=0,
            # confirmed=True: frame-0 prompt objects skip the confirmation wait
            self._append_object_state(
                frame_idx=0, confirmed=True, keep_alive_init=INIT_TRK_KEEP_ALIVE
            )

            bm = bin_masks_tmp[i]
            yx = np.where(bm)
            if len(yx[0]) == 0:
                continue
            y1, y2 = int(yx[0].min()), int(yx[0].max())
            x1, x2 = int(yx[1].min()), int(yx[1].max())
            all_scores.append(scores_tmp[i])
            all_boxes.append([x1, y1, x2, y2])
            all_masks.append(bm)

        return (
            np.array(all_scores, dtype=np.float32),
            np.array(all_boxes, dtype=np.float32),
            np.array(all_masks, dtype=bool),
        )

    def add_prompt_interactive(self, frame, points=None, point_labels=None, box=None):
        """
        Point / box prompt on frame 0 (single object).

        points      : (N, 2) float32 pixel coords
        point_labels: (N,) int32
        box         : (4,) float32 [x1, y1, x2, y2]

        Returns (scores, boxes, bin_masks).
        """
        models = self.models
        no_obj_params = self.no_obj_params

        orig_h, orig_w = frame.shape[:2]
        enc_out = run_encoder(models, preprocess(frame))
        prop_fpn0, prop_fpn1, prop_fpn2, prop_pos2 = (
            enc_out[6],
            enc_out[7],
            enc_out[8],
            enc_out[9],
        )

        if box is not None:
            coords, labels = box_to_corner_points(box, orig_h, orig_w)
        else:
            pts = np.array(points, dtype=np.float32)
            coords = pts[np.newaxis]
            lbls = (
                np.array(point_labels, dtype=np.int32)
                if point_labels is not None
                else np.ones(len(pts), dtype=np.int32)
            )
            labels = lbls[np.newaxis]

        mask_in = np.zeros((1, 1, 288, 288), dtype=np.float32)
        mask_enable = np.array([0], dtype=np.int32)
        sparse_emb, dense_emb, dense_pe = run_prompt_encoder(
            models, coords, labels, mask_in, mask_enable
        )
        masks_dec, iou_pred, sam_tokens_out, obj_score = run_mask_decoder(
            models, prop_fpn2, dense_pe, sparse_emb, dense_emb, prop_fpn0, prop_fpn1
        )
        best_slot = int(np.argmax(iou_pred[0]))
        obj_ptr = run_interactive_obj_ptr_proj(models, sam_tokens_out[:, best_slot, :])

        masks_32ch = build_32ch_mask(
            masks_dec[0:1, best_slot : best_slot + 1], is_conditioning=True
        )
        mem_feat, mem_pos = run_memory_encoder(models, prop_fpn2, masks_32ch)
        # add no-obj embeddings for unused slots (slots 1..15)
        if no_obj_params is not None:
            mem_feat = mem_feat + no_obj_params[0][1:].sum(axis=0).reshape(1, 256, 1, 1)

        init_all_ptrs = np.zeros((MULTIPLEX_COUNT, 256), dtype=np.float32)
        init_all_ptrs[0] = obj_ptr.reshape(256)
        if no_obj_params is not None:
            _, W, b = no_obj_params
            for k in range(1, MULTIPLEX_COUNT):
                init_all_ptrs[k] = (
                    np.zeros((1, 256), dtype=np.float32) @ W.T + b
                ).reshape(256)

        mb = MemoryBank()
        mb.add(
            0,
            prop_fpn2,
            prop_pos2,
            mem_feat,
            mem_pos,
            init_all_ptrs,
            is_conditioning=True,
        )
        self.memory_banks = [mb]
        self.obj_ids = []
        self.next_id = 1
        self.keep_alive = []
        self.consecutive_det_count = []
        self.confirmed = []
        self.add_frame = []
        self.unmatch_total = []
        self.pairwise_overlap = np.zeros((0, 0), dtype=np.int32)
        self.current_frame = 0
        self._append_object_state(
            frame_idx=0, confirmed=True, keep_alive_init=INIT_TRK_KEEP_ALIVE
        )

        logit_best = masks_dec[0, best_slot]
        binary_mask = (
            sigmoid(tv_resize(logit_best, (orig_h, orig_w), antialias=False)) > 0.5
        )
        score_val = float(sigmoid(float(np.asarray(obj_score).flat[0])))
        yx = np.where(binary_mask)
        if len(yx[0]) == 0:
            return (
                np.zeros(0),
                np.zeros((0, 4), np.float32),
                np.zeros((0, orig_h, orig_w), bool),
            )

        y1, y2 = int(yx[0].min()), int(yx[0].max())
        x1, x2 = int(yx[1].min()), int(yx[1].max())
        return (
            np.array([score_val], dtype=np.float32),
            np.array([[x1, y1, x2, y2]], dtype=np.float32),
            binary_mask[np.newaxis],
        )

    # ── propagation ───────────────────────────────────────────────────────────

    def propagate_in_video(self, frame_paths, start_frame=1):
        """
        Forward pass; yields (frame_idx, scores, boxes, masks, obj_ids) per frame.

        Tracking parameters match Sam3MultiplexTrackingWithInteractivity in model_builder.py:
          assoc_iou_thresh=0.1 (IoM), new_det_thresh=0.65,
          masklet_confirmation_consecutive_det_thresh=3,
          hotstart_delay=15, hotstart_unmatch_thresh=8, hotstart_dup_thresh=8,
          suppress_unmatched_only_within_hotstart=False,
          det_nms_thresh=0.1 (IoM), score_threshold_detection=0.4
        """
        models = self.models
        maskmem_tpos_enc = self.maskmem_tpos_enc
        no_obj_params = self.no_obj_params

        # frame_idx → set of obj_ids still unconfirmed at that frame (for lookahead)
        unconfirmed_ids_per_frame = {}
        # (frame_idx, candidates) waiting until UNCONFIRMED_STATUS_DELAY future frames exist
        pending_outputs = []
        last_frame_idx = len(frame_paths) - 1
        # cumulative set of obj_ids removed by hotstart (for retroactive hiding)
        hotstart_removed_ids = set()

        for frame_idx in range(start_frame, len(frame_paths)):
            N = len(self.memory_banks)
            if N == 0:
                return

            frame = cv2.imread(frame_paths[frame_idx])
            if frame is None:
                break
            orig_h, orig_w = frame.shape[:2]

            # ── Step 1: image encoder ─────────────────────────────────────
            enc_out = run_encoder(models, preprocess(frame))
            fpn0, fpn1, fpn2 = enc_out[0], enc_out[1], enc_out[2]
            pos2 = enc_out[5]
            prop_fpn0, prop_fpn1, prop_fpn2, prop_pos2 = (
                enc_out[6],
                enc_out[7],
                enc_out[8],
                enc_out[9],
            )

            if UNLOAD_MODELS_BETWEEN_STEPS:
                for key in (
                    "encoder",
                    "grounder",
                    "prompt_enc",
                    "mask_dec",
                    "iobj_proj",
                ):
                    m = models.get(key)
                    if m and hasattr(m, "unload"):
                        m.unload()

            # ── Step 2: memory attention ──────────────────────────────────
            memory_obj, memory_obj_pos, memory_img, memory_img_pos = (
                build_combined_memory_inputs(
                    self.memory_banks, frame_idx, maskmem_tpos_enc, models
                )
            )
            curr_flat = prop_fpn2.reshape(1, 256, -1).transpose(2, 0, 1)
            curr_pos_flat = prop_pos2.reshape(1, 256, -1).transpose(2, 0, 1)
            pix_feat = run_memory_attention(
                models,
                curr_obj=curr_flat,
                curr_obj_pos=curr_pos_flat,
                curr_img=curr_flat,
                memory_obj=memory_obj,
                memory_obj_pos=memory_obj_pos,
                memory_img=memory_img,
                memory_img_pos=memory_img_pos,
            )
            if UNLOAD_MODELS_BETWEEN_STEPS:
                for key in ("mem_attn", "tpos_proj"):
                    m = models.get(key)
                    if m and hasattr(m, "unload"):
                        m.unload()

            # ── Step 3: tracking mask decoder ─────────────────────────────
            extra_embed = self.build_extra_embed(N)
            img_emb = pix_feat.transpose(1, 2, 0).reshape(1, 256, 72, 72)
            masks_all, iou_all, sam_tokens_all, obj_scores_all = (
                run_tracking_mask_decoder(
                    models, img_emb, prop_fpn0, prop_fpn1, extra_embed
                )
            )

            # ── Step 4: per-slot results + ptrs ───────────────────────────
            is_app_flags = []
            logit_bests = []  # (N,) of (288,288) tracking logit masks
            all_ptrs = np.zeros((MULTIPLEX_COUNT, 256), dtype=np.float32)

            for k in range(MULTIPLEX_COUNT):
                best_slot_k = int(np.argmax(iou_all[0, k]))
                is_app_k = k < N and float(obj_scores_all[0, k]) > OBJ_SCORE_THRESHOLD
                obj_ptr_k = run_obj_ptr_proj(
                    models,
                    sam_tokens_all[:, k, best_slot_k : best_slot_k + 1, :].reshape(
                        1, 256
                    ),
                )
                if not is_app_k and no_obj_params is not None:
                    _, W, b = no_obj_params
                    obj_ptr_k = (obj_ptr_k @ W.T + b).astype(np.float32)
                all_ptrs[k] = obj_ptr_k.reshape(256)
                if k < N:
                    logit_bests.append(masks_all[0, k, best_slot_k])
                    is_app_flags.append(is_app_k)

            # 非出現オブジェクトも実際のトラッキングマスクを使う。
            # ゼロマスクにすると新規検出との重複判定が正しく機能しない。
            existing_bin_masks = np.array(
                [lb > 0 for lb in logit_bests], dtype=bool
            )  # (N, 288, 288)

            if UNLOAD_MODELS_BETWEEN_STEPS:
                for key in ("track_dec", "obj_proj"):
                    m = models.get(key)
                    if m and hasattr(m, "unload"):
                        m.unload()

            # ── Step 5: grounding (NMS + association) ────────────────────
            new_cand_scores = []  # float, scores of accepted new detections
            new_cand_bin_masks = []  # (orig_h, orig_w) full-res binary, for output
            new_cand_raw_masks = []  # (288, 288) logit, for memory encoder
            new_obj_ptrs = []  # interactive obj_ptr per new object
            det_to_matched_trk = {}  # d → [k] for pairwise overlap tracking
            im_mask = np.zeros((0, N), dtype=bool)
            # Available after grounding runs (for recondition in Step 7c)
            raw_nms = np.zeros((0, 288, 288), dtype=np.float32)
            scores_nms = np.zeros(0, dtype=np.float32)
            iom_mat = np.zeros((0, N), dtype=np.float32)

            if self.caption is not None and N < MULTIPLEX_COUNT:
                text_tokens = tokenize(self.caption)
                gnd_out = run_grounding(
                    models,
                    fpn0,
                    fpn1,
                    fpn2,
                    pos2,
                    text_tokens,
                    np.zeros((0, 1, 4), dtype=np.float32),
                    np.zeros((0, 1), dtype=np.int64),
                    np.zeros((1, 0), dtype=bool),
                )
                pred_masks_gnd, pred_boxes_gnd, pred_logits_gnd, presence_gnd = gnd_out

                # Filter at SCORE_THRESH_DET (= score_threshold_detection)
                out_probs = sigmoid(pred_logits_gnd[0, :, 0]) * sigmoid(
                    presence_gnd[0, 0]
                )
                gnd_keep = out_probs > SCORE_THRESH_DET
                # Suppress detections whose center is within DET_BOUNDARY_MARGIN of image edges.
                boxes_cxcywh = pred_boxes_gnd[0]  # (200, 4) normalized cx,cy,w,h
                cx = boxes_cxcywh[:, 0]
                cy = boxes_cxcywh[:, 1]
                boundary_keep = (
                    (cx > DET_BOUNDARY_MARGIN)
                    & (cx < 1.0 - DET_BOUNDARY_MARGIN)
                    & (cy > DET_BOUNDARY_MARGIN)
                    & (cy < 1.0 - DET_BOUNDARY_MARGIN)
                )
                gnd_keep = gnd_keep & boundary_keep
                gnd_keep_idx = np.where(gnd_keep)[0]

                if len(gnd_keep_idx) > 0:
                    raw_k = pred_masks_gnd[0][gnd_keep]  # (K, 288, 288) logit
                    scores_k = out_probs[gnd_keep]  # (K,)

                    # NMS with IoM (= det_nms_thresh=0.1, det_nms_use_iom=True)
                    nms_keep = nms_masks_iom(raw_k, scores_k, 0.0, DET_NMS_IOM_THRESH)
                    raw_nms = raw_k[nms_keep]  # (K_nms, 288, 288)
                    scores_nms = scores_k[nms_keep]  # (K_nms,)

                    # Association: IoM matrix (K_nms, N)
                    # Source: _associate_det_trk_compilable with use_iom_recondition=True
                    if N > 0:
                        iom_mat = mask_iom_matrix(
                            (raw_nms > 0), existing_bin_masks
                        )  # (K_nms, N)
                    else:
                        iom_mat = np.zeros((len(scores_nms), 0), np.float32)

                    im_mask = iom_mat >= ASSOC_IOM_THRESH  # (K_nms, N)

                    # det_to_matched_trk for pairwise overlap counting
                    for d in range(len(scores_nms)):
                        matched_ks = [k for k in range(N) if im_mask[d, k]]
                        if matched_ks:
                            det_to_matched_trk[d] = matched_ks

                    # is_new_det: score >= NEW_DET_THRESH AND IoM < ASSOC_IOM_THRESH with ALL tracks
                    is_new = (scores_nms >= NEW_DET_THRESH) & ~im_mask.any(axis=1)

                    for d in range(len(scores_nms)):
                        if not is_new[d]:
                            continue
                        if N + len(new_cand_bin_masks) >= MULTIPLEX_COUNT:
                            break
                        # full-res binary for output
                        bin_full = (
                            sigmoid(
                                cv2.resize(
                                    raw_nms[d],
                                    (orig_w, orig_h),
                                    interpolation=cv2.INTER_LINEAR,
                                )
                            )
                            > 0.5
                        )
                        # get obj_ptr via interactive decoder
                        mfp = mask_for_prompt_encoder(
                            bin_full.astype(np.float32), mask_input_size=(288, 288)
                        )
                        coords = np.zeros((1, 1, 2), dtype=np.float32)
                        labels = np.array([[-1]], dtype=np.int32)
                        mask_enable = np.array([1], dtype=np.int32)
                        sparse_emb, dense_emb, dense_pe = run_prompt_encoder(
                            models, coords, labels, mfp, mask_enable
                        )
                        masks_dec, iou_pred, sam_tokens_out, _ = run_mask_decoder(
                            models,
                            prop_fpn2,
                            dense_pe,
                            sparse_emb,
                            dense_emb,
                            prop_fpn0,
                            prop_fpn1,
                        )
                        best = int(np.argmax(iou_pred[0]))
                        new_obj_ptrs.append(
                            run_interactive_obj_ptr_proj(
                                models, sam_tokens_out[:, best, :]
                            )
                        )
                        new_cand_scores.append(float(scores_nms[d]))
                        new_cand_bin_masks.append(bin_full)
                        new_cand_raw_masks.append(raw_nms[d])

            # ── Step 6: update per-object tracking state ──────────────────
            # Source: _process_hotstart (CPU version) + update_masklet_confirmation_status
            trk_is_matched = (
                im_mask.any(axis=0) if im_mask.shape[0] > 0 else np.zeros(N, dtype=bool)
            )
            trk_is_nonempty = (
                existing_bin_masks.any(axis=(1, 2))
                if N > 0
                else np.array([], dtype=bool)
            )
            trk_is_unmatched = trk_is_nonempty & ~trk_is_matched

            for k in range(N):
                if trk_is_matched[k]:
                    self.keep_alive[k] = min(TRK_KEEP_ALIVE_MAX, self.keep_alive[k] + 1)
                    self.consecutive_det_count[k] += 1
                else:
                    self.keep_alive[k] = max(TRK_KEEP_ALIVE_MIN, self.keep_alive[k] - 1)
                    self.consecutive_det_count[k] = 0
                if trk_is_unmatched[k]:
                    self.unmatch_total[k] += 1
                if self.consecutive_det_count[k] >= MASKLET_CONFIRM_N:
                    self.confirmed[k] = True

            # Update pairwise overlap counts for hotstart duplicate detection
            for d, matched_ks in det_to_matched_trk.items():
                if len(matched_ks) >= 2:
                    first_k = min(matched_ks, key=lambda k: self.add_frame[k])
                    for other_k in matched_ks:
                        if other_k != first_k:
                            self.pairwise_overlap[first_k, other_k] += 1

            # ── Step 7: hotstart removal ───────────────────────────────────
            # Source: _process_hotstart in sam3_video_base.py
            # suppress_unmatched_only_within_hotstart=False → keep_alive suppression always
            to_remove = []
            for k in range(N):
                frames_since_add = frame_idx - self.add_frame[k]
                is_within_hotstart = frames_since_add < HOTSTART_DELAY
                if not is_within_hotstart:
                    continue
                if self.unmatch_total[k] >= HOTSTART_UNMATCH_THRESH:
                    to_remove.append(k)
                    continue
                # Remove by pairwise overlap with any earlier object
                for j in range(N):
                    if self.add_frame[j] < self.add_frame[k]:
                        if self.pairwise_overlap[j, k] >= HOTSTART_DUP_THRESH:
                            to_remove.append(k)
                            break

            # suppress_by_keep_alive: keep_alive <= 0
            # (suppress_unmatched_only_within_hotstart=False → always active)
            suppress_set = {k for k in range(N) if self.keep_alive[k] <= 0}

            # ── Step 7b: overlap suppression based on recent occlusion ────────
            # For each pair (i, j) with tracking mask IoU >= 0.7, suppress the
            # more recently occluded object (if the other was occluded at least once).
            # Hotstart-removed objects are treated as ALWAYS_OCCLUDED for this calculation.
            overlap_suppress = set()
            if N >= 2:
                temp_last_occ = list(self.last_occluded)
                for k in set(to_remove):
                    temp_last_occ[k] = ALWAYS_OCCLUDED

                flat = existing_bin_masks.reshape(N, -1).astype(np.float32)
                inter = flat @ flat.T  # (N, N)
                area = flat.sum(axis=1, keepdims=True)  # (N, 1)
                union = area + area.T - inter
                iou_mat = inter / np.maximum(union, 1.0)

                for i in range(N):
                    for j in range(i + 1, N):
                        if iou_mat[i, j] >= SUPPRESS_OVERLAP_IOU_THRESH:
                            lo_i = temp_last_occ[i]
                            lo_j = temp_last_occ[j]
                            if lo_i > lo_j and lo_j > NEVER_OCCLUDED:
                                overlap_suppress.add(i)
                            elif lo_j > lo_i and lo_i > NEVER_OCCLUDED:
                                overlap_suppress.add(j)
            suppress_set |= overlap_suppress

            # Update last_occluded: non-appearing or suppressed objects record this frame
            for k in range(N):
                mask_is_empty = not existing_bin_masks[k].any()
                if mask_is_empty or k in suppress_set:
                    self.last_occluded[k] = frame_idx

            # ── Step 7c: recondition every 16 frames ─────────────────────
            # For tracks with a high-confidence (≥0.8) detection match (IoM ≥ 0.5),
            # blend the detection mask with the tracking mask and refresh the obj_ptr.
            _RECOND_EVERY_N = 16
            _HIGH_CONF_RECOND = 0.8
            _IOM_THRESH_RECOND = 0.5  # iom_thresh_recondition
            if (
                ENABLE_RECONDITION
                and frame_idx % _RECOND_EVERY_N == 0
                and N > 0
                and len(raw_nms) > 0
            ):
                to_remove_set_7c = set(to_remove)
                for k in range(N):
                    if k in to_remove_set_7c or k in suppress_set:
                        continue
                    obj_score_k = float(sigmoid(float(obj_scores_all[0, k])))
                    if obj_score_k < _HIGH_CONF_RECOND:
                        continue
                    best_d = -1
                    best_iom_k = 0.0
                    for d in range(len(scores_nms)):
                        if scores_nms[d] >= _HIGH_CONF_RECOND and iom_mat.shape[1] > k:
                            iom_dk = float(iom_mat[d, k])
                            if iom_dk >= _IOM_THRESH_RECOND and iom_dk > best_iom_k:
                                best_iom_k = iom_dk
                                best_d = d
                    if best_d < 0:
                        continue
                    # Blend: where det and trk agree keep trk logit, else use det logit
                    det_logit = raw_nms[best_d]
                    trk_logit = logit_bests[k]
                    agree = (det_logit > 0) == (trk_logit > 0)
                    logit_bests[k] = np.where(agree, trk_logit, det_logit)
                    # Refresh obj_ptr via interactive decoder
                    bin_full_k = (
                        sigmoid(
                            cv2.resize(
                                logit_bests[k],
                                (orig_w, orig_h),
                                interpolation=cv2.INTER_LINEAR,
                            )
                        )
                        > 0.5
                    )
                    mfp_k = mask_for_prompt_encoder(
                        bin_full_k.astype(np.float32), mask_input_size=(288, 288)
                    )
                    c_k = np.zeros((1, 1, 2), dtype=np.float32)
                    l_k = np.array([[-1]], dtype=np.int32)
                    me_k = np.array([1], dtype=np.int32)
                    sp_k, de_k, dp_k = run_prompt_encoder(models, c_k, l_k, mfp_k, me_k)
                    md_k, ip_k, st_k, _ = run_mask_decoder(
                        models, prop_fpn2, dp_k, sp_k, de_k, prop_fpn0, prop_fpn1
                    )
                    bs_k = int(np.argmax(ip_k[0]))
                    all_ptrs[k] = run_interactive_obj_ptr_proj(
                        models, st_k[:, bs_k, :]
                    ).reshape(256)

            # ── Step 8: memory encoder (runs on ALL N + M_new objects before removal) ─
            M = len(new_cand_bin_masks)
            total_N = N + M
            combined_masks = np.zeros(
                (1, MASK_CHANNELS, MEMORY_MASK_SIZE, MEMORY_MASK_SIZE), dtype=np.float32
            )
            for k, logit in enumerate(logit_bests):
                # 非出現オブジェクトも実際のlogitをそのまま使う。
                # 出現フラグはno_obj_embedで別途伝達されるため、マスク側をゼロにする必要はない。
                # ただし overlap suppression で抑制されたオブジェクトはマスクをゼロにする。
                if k in overlap_suppress:
                    logit = np.full_like(logit, -10.0)
                prob = (
                    sigmoid(logit) * SIGMOID_SCALE_FOR_MEM_ENC
                    + SIGMOID_BIAS_FOR_MEM_ENC
                )
                combined_masks[0, k] = tv_resize(
                    prob, (MEMORY_MASK_SIZE, MEMORY_MASK_SIZE)
                )
                combined_masks[0, 16 + k] = 0.0
            for i, raw_m in enumerate(new_cand_raw_masks):
                logit = raw_m  # already logit space
                prob = (
                    sigmoid(logit) * SIGMOID_SCALE_FOR_MEM_ENC
                    + SIGMOID_BIAS_FOR_MEM_ENC
                )
                combined_masks[0, N + i] = tv_resize(
                    prob, (MEMORY_MASK_SIZE, MEMORY_MASK_SIZE)
                )
                combined_masks[0, 16 + N + i] = 1.0

            mem_feat_combined, mem_pos_combined = run_memory_encoder(
                models, prop_fpn2, combined_masks
            )
            if no_obj_params is not None:
                no_obj_embed_all = no_obj_params[0]
                mem_feat_combined = mem_feat_combined + no_obj_embed_all[total_N:].sum(
                    axis=0
                ).reshape(1, 256, 1, 1)
                for k in range(N):
                    if not is_app_flags[k]:
                        mem_feat_combined = mem_feat_combined + no_obj_embed_all[
                            k
                        ].reshape(1, 256, 1, 1)

            # ── Step 9: remove objects (compact state) ────────────────────
            to_remove_set = set(to_remove)
            # Record removed obj_ids before they're deleted (for retroactive output hiding)
            for k in to_remove_set:
                hotstart_removed_ids.add(self.obj_ids[k])
            remaining_old_slots = [k for k in range(N) if k not in to_remove_set]
            for k in sorted(to_remove_set, reverse=True):
                self.memory_banks.pop(k)
                self._remove_object_state(k)
            N_remaining = len(remaining_old_slots)

            # Build updated all_ptrs for remaining objects
            updated_all_ptrs = np.zeros((MULTIPLEX_COUNT, 256), dtype=np.float32)
            for new_k, old_k in enumerate(remaining_old_slots):
                updated_all_ptrs[new_k] = all_ptrs[old_k]
            if no_obj_params is not None:
                _, W, b = no_obj_params
                for k in range(N_remaining, MULTIPLEX_COUNT):
                    updated_all_ptrs[k] = (
                        np.zeros((1, 256), dtype=np.float32) @ W.T + b
                    ).reshape(256)

            # ── Step 10: update memory banks for remaining objects ─────────
            for mb in self.memory_banks:
                mb.add(
                    frame_idx,
                    prop_fpn2,
                    prop_pos2,
                    mem_feat_combined,
                    mem_pos_combined,
                    updated_all_ptrs,
                )

            # ── Step 11: add new objects ───────────────────────────────────
            for i, new_ptr in enumerate(new_obj_ptrs):
                slot = N_remaining + i
                cond_ptrs = updated_all_ptrs.copy()
                cond_ptrs[slot] = new_ptr.reshape(256)
                mb = MemoryBank()
                mb.add(
                    frame_idx,
                    prop_fpn2,
                    prop_pos2,
                    mem_feat_combined,
                    mem_pos_combined,
                    cond_ptrs,
                    is_conditioning=True,
                )
                self.memory_banks.append(mb)
                # init_trk_keep_alive=0: new objects start suppressed until first match
                self._append_object_state(
                    frame_idx=frame_idx,
                    confirmed=False,
                    keep_alive_init=INIT_TRK_KEEP_ALIVE,
                )

            if UNLOAD_MODELS_BETWEEN_STEPS:
                for key in ("mem_enc",):
                    m = models.get(key)
                    if m and hasattr(m, "unload"):
                        m.unload()

            # ── Step 12: collect output ────────────────────────────────────
            # Candidates: suppress + is_app pass. The confirmed check is deferred
            # by UNCONFIRMED_STATUS_DELAY frames (PyTorch unconfirmed_status_delay
            # lookahead): output frame N only if the object is confirmed at frame
            # N + UNCONFIRMED_STATUS_DELAY.
            candidates = []
            for new_k, old_k in enumerate(remaining_old_slots):
                if old_k in suppress_set:
                    continue
                if not is_app_flags[old_k]:
                    continue
                binary_mask = (
                    sigmoid(
                        tv_resize(logit_bests[old_k], (orig_h, orig_w), antialias=False)
                    )
                    > 0.5
                )
                yx = np.where(binary_mask)
                if len(yx[0]) == 0:
                    continue
                y1, y2 = int(yx[0].min()), int(yx[0].max())
                x1, x2 = int(yx[1].min()), int(yx[1].max())
                candidates.append(
                    (
                        self.obj_ids[new_k],
                        float(sigmoid(float(obj_scores_all[0, old_k]))),
                        [x1, y1, x2, y2],
                        binary_mask,
                    )
                )

            # Record which obj_ids are unconfirmed this frame (for future lookahead)
            unconfirmed_ids_per_frame[frame_idx] = {
                self.obj_ids[new_k]
                for new_k in range(len(remaining_old_slots))
                if not self.confirmed[new_k]
            }

            pending_outputs.append((frame_idx, candidates))
            self.current_frame = frame_idx

            # Yield oldest buffered frame when its lookahead frame has been processed,
            # or flush all remaining entries on the last frame.
            while len(pending_outputs) > UNCONFIRMED_STATUS_DELAY or (
                frame_idx == last_frame_idx and pending_outputs
            ):
                yield_frame_idx, yield_candidates = pending_outputs.pop(0)
                lookup_idx = min(yield_frame_idx + UNCONFIRMED_STATUS_DELAY, frame_idx)
                hidden_ids = unconfirmed_ids_per_frame.get(lookup_idx, set())
                all_scores_out, all_boxes_out, all_masks_out, obj_ids_out = (
                    [],
                    [],
                    [],
                    [],
                )
                for obj_id, score, box, mask in yield_candidates:
                    if obj_id in hidden_ids or obj_id in hotstart_removed_ids:
                        continue
                    all_scores_out.append(score)
                    all_boxes_out.append(box)
                    all_masks_out.append(mask)
                    obj_ids_out.append(obj_id)

                # Apply non-overlapping constraint: for each pixel assign to the
                # highest-score object; objects that lose all pixels are dropped.
                if len(all_masks_out) > 1:
                    masks_np = np.stack(all_masks_out)  # (K, H, W)
                    sc_np = np.array(all_scores_out, dtype=np.float32)
                    score_map = np.where(masks_np, sc_np[:, None, None], -1.0)
                    best_owner = np.argmax(score_map, axis=0)  # (H, W)
                    all_masks_out = [
                        masks_np[i] & (best_owner == i)
                        for i in range(len(all_masks_out))
                    ]
                    valid = [i for i, m in enumerate(all_masks_out) if m.any()]
                    all_scores_out = [all_scores_out[i] for i in valid]
                    all_masks_out = [all_masks_out[i] for i in valid]
                    obj_ids_out = [obj_ids_out[i] for i in valid]
                    all_boxes_out = []
                    for m in all_masks_out:
                        yx = np.where(m)
                        all_boxes_out.append(
                            [
                                int(yx[1].min()),
                                int(yx[0].min()),
                                int(yx[1].max()),
                                int(yx[0].max()),
                            ]
                        )

                yield (
                    yield_frame_idx,
                    np.array(all_scores_out, dtype=np.float32),
                    np.array(all_boxes_out, dtype=np.float32),
                    np.array(all_masks_out, dtype=bool),
                    obj_ids_out,
                )

    def build_extra_embed(self, n_objects):
        """(1, 16, 256) slot validity signal for the tracking decoder.

        Slots 0..n_objects-1 receive valid_embed (real objects);
        remaining slots get invalid_embed so the decoder ignores them.
        Without this distinction, empty slots produce ghost detections.
        """
        load_embeds()
        extra = invalid_embed_cache.copy()
        for k in range(min(n_objects, MULTIPLEX_COUNT)):
            extra[k] = valid_embed_cache[k]
        return extra[np.newaxis]


# ── Convenience functions for sam3p1.py compatibility ─────────────────────────


def add_prompt(models, frame, caption, maskmem_tpos_enc, no_obj_params=None):
    """Functional wrapper: returns (memory_banks, scores, boxes, masks)."""
    tracker = Sam3Tracker(models, maskmem_tpos_enc, no_obj_params)
    scores, boxes, masks = tracker.add_prompt(frame, caption)
    return tracker.memory_banks, scores, boxes, masks


def add_prompt_interactive(
    models,
    frame,
    maskmem_tpos_enc,
    no_obj_params=None,
    points=None,
    point_labels=None,
    box=None,
):
    """Functional wrapper: returns (memory_banks, scores, boxes, masks)."""
    tracker = Sam3Tracker(models, maskmem_tpos_enc, no_obj_params)
    scores, boxes, masks = tracker.add_prompt_interactive(
        frame, points=points, point_labels=point_labels, box=box
    )
    return tracker.memory_banks, scores, boxes, masks


def propagate_in_video(
    models,
    frame_paths,
    memory_banks,
    maskmem_tpos_enc,
    no_obj_params=None,
    start_frame=1,
):
    """Functional wrapper: yields (frame_idx, scores, boxes, masks, obj_ids)."""
    tracker = Sam3Tracker(models, maskmem_tpos_enc, no_obj_params)
    tracker.memory_banks = memory_banks
    yield from tracker.propagate_in_video(frame_paths, start_frame=start_frame)
