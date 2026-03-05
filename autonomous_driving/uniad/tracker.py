from track_instance import Instances


class RuntimeTrackerBase(object):
    def __init__(self, score_thresh=0.5, filter_score_thresh=0.4, miss_tolerance=5):
        self.score_thresh = score_thresh
        self.filter_score_thresh = filter_score_thresh
        self.miss_tolerance = miss_tolerance
        self.max_obj_id = 0

    def clear(self):
        self.max_obj_id = 0

    def update(self, track_instances: Instances):
        track_instances.disappear_time[track_instances.scores >= self.score_thresh] = 0
        for i in range(len(track_instances)):
            if (
                track_instances.obj_idxes[i] == -1
                and track_instances.scores[i] >= self.score_thresh
            ):
                # new track
                track_instances.obj_idxes[i] = self.max_obj_id
                self.max_obj_id += 1
            elif (
                track_instances.obj_idxes[i] >= 0
                and track_instances.scores[i] < self.filter_score_thresh
            ):
                # sleep time ++
                track_instances.disappear_time[i] += 1
                if track_instances.disappear_time[i] >= self.miss_tolerance:
                    # mark deaded tracklets: Set the obj_id to -1.
                    track_instances.obj_idxes[i] = -1
