import numpy as np
from scipy.optimize import linear_sum_assignment as linear_assignment

from math_utils import softmax

from tracker_model import LSTM3DTracker


class MotionTracker:
    def __init__(
            self,
            lstm_pred,
            lstm_refine,
            init_score_thr: float = 0.8,
            init_track_id: int = 0,
            obj_score_thr: float = 0.5,
            match_score_thr: float = 0.5,
            memo_tracklet_frames: int = 10,
            memo_backdrop_frames: int = 1,
            memo_momentum: float = 0.8,
            motion_momentum: float = 0.8,
            nms_conf_thr: float = 0.5,
            nms_backdrop_iou_thr: float = 0.3,
            nms_class_iou_thr: float = 0.7,
            loc_dim: int = 7,
            with_deep_feat: bool = True,
            with_cats: bool = True,
            with_bbox_iou: bool = True,
            with_depth_ordering: bool = True,
            with_depth_uncertainty: bool = True,
            track_bbox_iou: str = 'bbox',
            depth_match_metric: str = 'centriod',
            match_metric: str = 'cycle_softmax',
            match_algo: str = 'greedy'):
        assert 0 <= memo_momentum <= 1.0
        assert 0 <= motion_momentum <= 1.0
        assert memo_tracklet_frames >= 0
        assert memo_backdrop_frames >= 0
        assert track_bbox_iou in ['box2d', 'box2d_depth_aware', 'bev', 'box3d']
        assert depth_match_metric in [
            'centroid', 'cosine', 'pure_motion', 'motion'
        ]
        assert match_metric in ['cycle_softmax', 'softmax', 'cosine']

        self.init_score_thr = init_score_thr
        self.obj_score_thr = obj_score_thr
        self.match_score_thr = match_score_thr
        self.memo_tracklet_frames = memo_tracklet_frames
        self.memo_backdrop_frames = memo_backdrop_frames
        self.memo_momentum = memo_momentum
        self.motion_momentum = motion_momentum
        self.nms_conf_thr = nms_conf_thr
        self.nms_backdrop_iou_thr = nms_backdrop_iou_thr
        self.nms_class_iou_thr = nms_class_iou_thr
        self.with_deep_feat = with_deep_feat
        self.with_cats = with_cats
        self.with_depth_ordering = with_depth_ordering
        self.with_depth_uncertainty = with_depth_uncertainty
        self.with_bbox_iou = with_bbox_iou
        self.track_bbox_iou = track_bbox_iou
        self.depth_match_metric = depth_match_metric
        self.lstm_pred = lstm_pred
        self.lstm_refine = lstm_refine
        self.loc_dim = loc_dim
        self.match_metric = match_metric
        self.match_algo = match_algo
        self.bbox_affinity_weight = 0.5 if with_deep_feat else 1.0
        self.feat_affinity_weight = 1 - self.bbox_affinity_weight if with_bbox_iou else 1.0
        self.num_tracklets = init_track_id
        self.tracklets = dict()
        self.backdrops = []

        np.set_printoptions(formatter={'float': '{: 0.3f}'.format})

    @property
    def empty(self):
        return False if self.tracklets else True

    def update_memo(
            self, ids, bboxes, boxes_3d, depth_uncertainty, embeds,
            labels, cur_frame):
        tracklet_inds = ids > -1

        # update memo
        for tid, bbox, box_3d, d_uncertainty, embed, label in zip(
                ids[tracklet_inds], bboxes[tracklet_inds],
                boxes_3d[tracklet_inds], depth_uncertainty[tracklet_inds],
                embeds[tracklet_inds], labels[tracklet_inds]):
            tid = int(tid)
            if tid in self.tracklets.keys():
                self.tracklets[tid]['bbox'] = bbox
                self.tracklets[tid]['tracker'].update(box_3d, d_uncertainty)

                tracker_box = self.tracklets[tid]['tracker'].get_state()[:7]
                pd_box_3d = np.array(tracker_box)

                velocity = \
                    (pd_box_3d - self.tracklets[tid]['box_3d']) \
                    / (cur_frame - self.tracklets[tid]['last_frame'])

                self.tracklets[tid]['box_3d'] = pd_box_3d
                self.tracklets[tid]['embed'] += self.memo_momentum * (
                        embed - self.tracklets[tid]['embed'])
                self.tracklets[tid]['label'] = label
                self.tracklets[tid]['velocity'] = \
                    (self.tracklets[tid]['velocity'] * self.tracklets[tid]['acc_frame'] + velocity) \
                    / (self.tracklets[tid]['acc_frame'] + 1)
                self.tracklets[tid]['last_frame'] = cur_frame
                self.tracklets[tid]['acc_frame'] += 1
            else:
                tracker = LSTM3DTracker(
                    self.lstm_pred,
                    self.lstm_refine,
                    self.loc_dim,
                    box_3d,
                    d_uncertainty,
                )
                self.tracklets[tid] = dict(
                    bbox=bbox,
                    box_3d=box_3d,
                    tracker=tracker,
                    embed=embed,
                    label=label,
                    last_frame=cur_frame,
                    velocity=np.zeros_like(box_3d),
                    acc_frame=0)

        # Handle vanished tracklets
        for tid in self.tracklets:
            if cur_frame > self.tracklets[tid]['last_frame'] and tid > -1:
                self.tracklets[tid]['box_3d'][:self.loc_dim] = np.array(
                    self.tracklets[tid]['tracker'].predict()[:self.loc_dim]
                )

        # Add backdrops
        backdrop_inds = np.nonzero(ids == -1)[0]
        ious = bbox_overlaps(bboxes[backdrop_inds, :-1], bboxes[:, :-1])
        for i, ind in enumerate(backdrop_inds):
            if (ious[i, :ind] > self.nms_backdrop_iou_thr).any():
                backdrop_inds[i] = -1
        backdrop_inds = backdrop_inds[backdrop_inds > -1]

        backdrop_tracker = [
            LSTM3DTracker(
                self.lstm_pred,
                self.lstm_refine,
                self.loc_dim,
                boxes_3d[bd_ind],
                depth_uncertainty[bd_ind],
            ) for bd_ind in backdrop_inds
        ]
        self.backdrops.insert(
            0,
            dict(
                bboxes=bboxes[backdrop_inds],
                boxes_3d=boxes_3d[backdrop_inds],
                tracker=backdrop_tracker,
                embeds=embeds[backdrop_inds],
                labels=labels[backdrop_inds]))

        # pop memo
        invalid_ids = []
        for k, v in self.tracklets.items():
            if cur_frame - v['last_frame'] >= self.memo_tracklet_frames:
                invalid_ids.append(k)
        for invalid_id in invalid_ids:
            self.tracklets.pop(invalid_id)

        if len(self.backdrops) > self.memo_backdrop_frames:
            self.backdrops.pop()

    @property
    def memo(self):
        memo_embeds = []
        memo_ids = []
        memo_bboxes = []
        memo_boxes_3d = []
        memo_trackers = []
        memo_labels = []
        memo_vs = []
        for k, v in self.tracklets.items():
            memo_bboxes.append(v['bbox'][None, :])
            memo_boxes_3d.append(v['box_3d'][None, :])
            memo_trackers.append(v['tracker'])
            memo_embeds.append(v['embed'][None, :])
            memo_ids.append(k)
            memo_labels.append(v['label'].reshape(1, 1))
            memo_vs.append(v['velocity'][None, :])
        memo_ids = np.array(memo_ids, dtype=np.int64).reshape(1, -1)

        for backdrop in self.backdrops:
            backdrop_ids = np.full(
                (1, backdrop['embeds'].shape[0]), -1, dtype=np.int64)
            backdrop_vs = np.zeros_like(backdrop['boxes_3d'])
            memo_bboxes.append(backdrop['bboxes'])
            memo_boxes_3d.append(backdrop['boxes_3d'])
            memo_trackers.extend(backdrop['tracker'])
            memo_embeds.append(backdrop['embeds'])
            memo_ids = np.concatenate([memo_ids, backdrop_ids], axis=1)
            memo_labels.append(backdrop['labels'][:, None])
            memo_vs.append(backdrop_vs)

        memo_bboxes = np.concatenate(memo_bboxes, axis=0)
        memo_boxes_3d = np.concatenate(memo_boxes_3d, axis=0)
        memo_embeds = np.concatenate(memo_embeds, axis=0)
        memo_labels = np.concatenate(memo_labels, axis=0).squeeze(1)
        memo_vs = np.concatenate(memo_vs, axis=0)

        return memo_bboxes, memo_labels, memo_boxes_3d, memo_trackers, memo_embeds, \
               memo_ids.squeeze(0), memo_vs

    def match(
            self,
            bboxes,
            labels,
            boxes_3d,
            depth_uncertainty,
            position,
            rotation,
            embeds,
            cur_frame: int,
            pure_det: bool = False):
        """Match incoming detection results with embedding and 3D infos

        Args:
            bboxes: (N, 5), [x1, y1, x2, y2, conf]
            labels: (N,)
            boxes_3d: (N, 7), 3D information stored
                                     in world coordinates with the format
                                     [X, Y, Z, theta, h, w, l]
            depth_uncertainty: (N, ), confidence in depth
                                     estimation
            position: (3, ), camera position
            rotation: (3, 3), camera rotation
            embeds: (N, C), extracted box feature
            cur_frame (int): indicates the frame index
            pure_det (bool): output pure detection. Defaults False.

        Raises:
            NotImplementedError: raise if self.match_metric not found

        Returns:
            list: A list of matched bbox, labels, boxes_3d and embeds
        """
        if depth_uncertainty is None or not self.with_depth_uncertainty:
            depth_uncertainty = np.array((boxes_3d.shape[0], 1))

        inds = np.argsort(
            bboxes[:, -1] * depth_uncertainty.flatten()
        )[::-1]
        bboxes = bboxes[inds, :]
        labels = labels[inds]
        embeds = embeds[inds, :]
        boxes_3d = boxes_3d[inds]
        depth_uncertainty = depth_uncertainty[inds]

        if pure_det:
            valids = np.ones((bboxes.shape[0]), dtype=bool)
            ids = np.arange(
                self.num_tracklets,
                self.num_tracklets + bboxes.shape[0],
                dtype=np.int64)
            self.num_tracklets += bboxes.shape[0]

            return bboxes, labels, boxes_3d, ids, inds, valids

        # duplicate removal for potential backdrops and cross classes
        valids = np.ones(bboxes.shape[0])
        ious = bbox_overlaps(bboxes[:, :-1], bboxes[:, :-1])
        for i in range(1, bboxes.shape[0]):
            thr = self.nms_backdrop_iou_thr \
                if bboxes[i, -1] < self.obj_score_thr \
                else self.nms_class_iou_thr
            if (ious[i, :i] > thr).any():
                valids[i] = 0

        valids = valids == 1
        bboxes = bboxes[valids, :]
        labels = labels[valids]
        embeds = embeds[valids, :]
        boxes_3d = boxes_3d[valids]
        depth_uncertainty = depth_uncertainty[valids]

        # init ids container
        ids = np.full((bboxes.shape[0],), -1, dtype=np.int64)

        # match if buffer is not empty
        if bboxes.shape[0] > 0 and not self.empty:
            memo_bboxes, memo_labels, memo_boxes_3d, \
            memo_trackers, memo_embeds, memo_ids, memo_vs = self.memo

            memo_boxes_3d_predict = memo_boxes_3d.copy()
            for ind, memo_tracker in enumerate(memo_trackers):
                memo_velo = memo_tracker.predict(
                    update_state=memo_tracker.age != 0)
                memo_boxes_3d_predict[ind, :3] += np.array(memo_velo[7:])

            if self.with_bbox_iou:
                def get_xy_box(boxes_3d_world):
                    box_x_cen = boxes_3d_world[:, 0]
                    box_y_cen = boxes_3d_world[:, 1]
                    box_width = boxes_3d_world[:, 5]
                    box_length = boxes_3d_world[:, 6]

                    dets_xy_box = np.stack([
                        box_x_cen - box_width / 2.0,
                        box_y_cen - box_length / 2.0,
                        box_x_cen + box_width / 2.0,
                        box_y_cen + box_length / 2.0
                    ], axis=1)
                    return dets_xy_box

                if self.track_bbox_iou == 'box2d':
                    scores_iou = bbox_overlaps(
                        bboxes[:, :-1], memo_bboxes[:, :-1])
                elif self.track_bbox_iou == 'box3d':
                    a = boxes_3d[..., None]
                    b = memo_boxes_3d_predict[..., None].transpose(2, 1, 0)
                    depth_weight = pairwise_distance(a, b)
                    scores_iou = np.exp(-depth_weight / 10.0)
                else:
                    raise NotImplementedError
            else:
                scores_iou = np.ones(
                    [bboxes.shape[0], memo_bboxes.shape[0]])

            if self.with_deep_feat:
                def compute_quasi_dense_feat_match(embeds, memo_embeds):
                    if self.match_metric == 'cycle_softmax':
                        feats = np.matmul(embeds, memo_embeds.T)
                        d2t_scores = softmax(feats, axis=1)
                        t2d_scores = softmax(feats, axis=0)
                        scores_feat = (d2t_scores + t2d_scores) / 2
                    elif self.match_metric == 'softmax':
                        feats = np.matmul(embeds, memo_embeds.T)
                        scores_feat = softmax(feats, axis=1)
                    else:
                        raise NotImplementedError
                    return scores_feat

                scores_feat = compute_quasi_dense_feat_match(
                    embeds, memo_embeds)
            else:
                scores_feat = np.ones(scores_iou.shape)

            # Match with depth ordering
            if self.with_depth_ordering:
                def compute_boxoverlap_with_depth(
                        obsv_boxes_3d, memo_boxes_3d, memo_vs):
                    # Sum up all the available region of each tracker
                    if self.depth_match_metric == 'centroid':
                        a = obsv_boxes_3d[..., :3, None]
                        b = memo_boxes_3d[..., :3, None].transpose(2, 1, 0)
                        depth_weight = pairwise_distance(a, b)
                        depth_weight = np.exp(-depth_weight / 10.0)
                    elif self.depth_match_metric == 'motion':
                        a = obsv_boxes_3d[..., :3, None]
                        b = memo_boxes_3d_predict[..., :3, None].transpose(2, 1, 0)
                        centroid_weight = pairwise_distance(a, b)
                        centroid_weight = np.exp(-centroid_weight / 10.0)
                        # Moving distance should be aligned
                        # V_observed-tracked vs. V_velocity
                        a = obsv_boxes_3d[..., :3, None] - memo_boxes_3d[..., :3, None].transpose(2, 1, 0)
                        b = memo_vs[..., :3, None].transpose(2, 1, 0)
                        motion_weight = pairwise_distance(a, b)
                        motion_weight = np.exp(-motion_weight / 5.0)
                        # Moving direction should be aligned
                        # Set to 0.5 when two vector not within +-90 degree
                        a = obsv_boxes_3d[..., :2, None] - memo_boxes_3d[..., :2, None].transpose(2, 1, 0)
                        b = memo_vs[..., :2, None].transpose(2, 1, 0)
                        cos_sim = cosine_similarity(a, b)
                        cos_sim += 1.0
                        cos_sim /= 2.0
                        depth_weight = cos_sim * centroid_weight + (1.0 - cos_sim) * motion_weight
                    else:
                        raise NotImplementedError

                    return depth_weight

                if self.depth_match_metric == 'motion':
                    scores_depth = compute_boxoverlap_with_depth(
                        boxes_3d, memo_boxes_3d, memo_vs)
                else:
                    scores_depth = compute_boxoverlap_with_depth(
                        boxes_3d, memo_boxes_3d_predict, memo_vs)
            else:
                scores_depth = np.array(scores_iou.shape)

            if self.with_cats:
                cat_same = labels.reshape(-1, 1) == memo_labels.reshape(1, -1)
                scores_cats = cat_same.astype(np.float32)
            else:
                scores_cats = np.array(scores_iou.shape)

            scores = self.bbox_affinity_weight * scores_iou * scores_depth + \
                     self.feat_affinity_weight * scores_feat
            scores /= (self.bbox_affinity_weight + self.feat_affinity_weight)
            scores *= (scores_iou > 0.0).astype(np.float32)
            scores *= (scores_depth > 0.0).astype(np.float32)
            scores *= scores_cats

            # Assign matching
            if self.match_algo == 'greedy':
                for i in range(bboxes.shape[0]):
                    memo_ind = np.argmax(scores[i, :], axis=0)
                    conf = scores[i, memo_ind]
                    tid = memo_ids[memo_ind]
                    # Matching confidence
                    if conf > self.match_score_thr:
                        # Update existing tracklet
                        if tid > -1:
                            # Keep object with high 3D objectness
                            if bboxes[i, -1] * depth_uncertainty[i] > self.obj_score_thr:
                                ids[i] = tid
                                scores[:i, memo_ind] = 0
                                scores[i + 1:, memo_ind] = 0
                            else:
                                # Reduce FP w/ low objectness but high match conf
                                if conf > self.nms_conf_thr:
                                    ids[i] = -2
            elif self.match_algo == 'hungarian':
                # Hungarian
                matched_indices = linear_assignment(-scores)
                for idx in range(len(matched_indices[0])):
                    i = matched_indices[0][idx]
                    memo_ind = matched_indices[1][idx]
                    conf = scores[i, memo_ind]
                    tid = memo_ids[memo_ind]
                    if conf > self.match_score_thr and tid > -1:
                        # Keep object with high 3D objectness
                        if bboxes[i, -1] * depth_uncertainty[i] > self.obj_score_thr:
                            ids[i] = tid
                            scores[:i, memo_ind] = 0
                            scores[i + 1:, memo_ind] = 0
                        else:
                            # Reduce FP w/ low objectness but high match conf
                            if conf > self.nms_conf_thr:
                                ids[i] = -2
                del matched_indices

        new_inds = (ids == -1) & (bboxes[:, 4] > self.init_score_thr)
        num_news = new_inds.sum()
        ids[new_inds] = np.arange(
            self.num_tracklets,
            self.num_tracklets + num_news,
            dtype=np.int64)
        self.num_tracklets += num_news

        self.update_memo(
            ids, bboxes, boxes_3d, depth_uncertainty, embeds,
            labels, cur_frame)

        update_bboxes = bboxes
        update_labels = labels
        update_boxes_3d = boxes_3d
        for tid in ids[ids > -1]:
            update_boxes_3d[ids == tid] = self.tracklets[int(tid)]['box_3d']
        update_ids = ids

        return update_bboxes, update_labels, update_boxes_3d, update_ids, inds, valids


def pairwise_distance(a, b):
    pdist = np.power(a - b, 2)
    pdist = np.sum(pdist, axis=1)
    pdist = np.sqrt(pdist)
    return pdist


def cosine_similarity(a, b, eps=1e-12):
    cos_sim = np.zeros((a.shape[0], a.shape[-1]))
    for i, x in enumerate(a):
        cos_sim[i, ...] = np.diag(np.dot(x.T, b[0]))
        cos_sim[i, ...] /= np.linalg.norm(x.T, axis=1) * np.linalg.norm(b[0], axis=0) + eps
    return cos_sim


def bbox_overlaps(bboxes1, bboxes2, mode='iou'):
    """Calculate the ious between each bbox of bboxes1 and bboxes2.

    Args:
        bboxes1(ndarray): shape (n, 4)
        bboxes2(ndarray): shape (k, 4)
        mode(str): iou (intersection over union) or iof (intersection
            over foreground)
    Returns:
        ious(ndarray): shape (n, k)
    """
    assert mode in ['iou', 'iof']

    bboxes1 = bboxes1.astype(np.float32)
    bboxes2 = bboxes2.astype(np.float32)
    rows = bboxes1.shape[0]
    cols = bboxes2.shape[0]
    ious = np.zeros((rows, cols), dtype=np.float32)
    if rows * cols == 0:
        return ious

    exchange = False
    if bboxes1.shape[0] > bboxes2.shape[0]:
        bboxes1, bboxes2 = bboxes2, bboxes1
        ious = np.zeros((cols, rows), dtype=np.float32)
        exchange = True

    area1 = (bboxes1[:, 2] - bboxes1[:, 0] + 1) * (
            bboxes1[:, 3] - bboxes1[:, 1] + 1)
    area2 = (bboxes2[:, 2] - bboxes2[:, 0] + 1) * (
            bboxes2[:, 3] - bboxes2[:, 1] + 1)
    for i in range(bboxes1.shape[0]):
        x_start = np.maximum(bboxes1[i, 0], bboxes2[:, 0])
        y_start = np.maximum(bboxes1[i, 1], bboxes2[:, 1])
        x_end = np.minimum(bboxes1[i, 2], bboxes2[:, 2])
        y_end = np.minimum(bboxes1[i, 3], bboxes2[:, 3])
        overlap = np.maximum(x_end - x_start + 1, 0) * np.maximum(
            y_end - y_start + 1, 0)
        if mode == 'iou':
            union = area1[i] + area2 - overlap
        else:
            union = area1[i] if not exchange else area2
        ious[i, :] = overlap / union

    if exchange:
        ious = ious.T

    return ious
