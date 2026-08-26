import sys

import cv2
import numpy as np

sys.path.append('../../util')
from image_utils import imread  # noqa: E402

graph_edge = [
    (0, 0), (1, 1), (2, 2), (3, 3), (4, 4), (5, 5), (6, 6), (7, 7), (8, 8), (9, 9), (10, 10), (11, 11), (12, 12),
    (13, 13), (14, 14), (15, 15), (16, 16), (17, 17), (4, 3), (3, 2), (7, 6), (6, 5), (13, 12), (12, 11), (10, 9),
    (9, 8), (11, 5), (8, 2), (5, 1), (2, 1), (0, 1), (15, 0), (14, 0), (17, 15), (16, 14)
]


class naive_pose_tracker():
    """ A simple tracker for recording person poses and generating skeleton sequences.
    For actual occasion, I recommend you to implement a robuster tracker.
    Pull-requests are welcomed.
    """

    def __init__(self, data_frame=128, num_joint=18, max_frame_dis=np.inf):
        self.data_frame = data_frame
        self.num_joint = num_joint
        self.max_frame_dis = max_frame_dis
        self.latest_frame = 0
        self.trace_info = list()

    def update(self, multi_pose, current_frame):
        # multi_pose.shape: (num_person, num_joint, 3)

        if current_frame <= self.latest_frame:
            return

        if len(multi_pose.shape) != 3:
            return

        score_order = (-multi_pose[:, :, 2].sum(axis=1)).argsort(axis=0)
        for p in multi_pose[score_order]:

            # match existing traces
            matching_trace = None
            matching_dis = None
            for trace_index, (trace, latest_frame) in enumerate(self.trace_info):
                # trace.shape: (num_frame, num_joint, 3)
                if current_frame <= latest_frame:
                    continue
                mean_dis, is_close = self.get_dis(trace, p)
                if is_close:
                    if matching_trace is None:
                        matching_trace = trace_index
                        matching_dis = mean_dis
                    elif matching_dis > mean_dis:
                        matching_trace = trace_index
                        matching_dis = mean_dis

            # update trace information
            if matching_trace is not None:
                trace, latest_frame = self.trace_info[matching_trace]

                # padding zero if the trace is fractured
                pad_mode = 'interp' if latest_frame == self.latest_frame else 'zero'
                pad = current_frame - latest_frame - 1
                new_trace = self.cat_pose(trace, p, pad, pad_mode)
                self.trace_info[matching_trace] = (new_trace, current_frame)

            else:
                new_trace = np.array([p])
                self.trace_info.append((new_trace, current_frame))

        self.latest_frame = current_frame

    def get_skeleton_sequence(self):

        # remove old traces
        valid_trace_index = []
        for trace_index, (trace, latest_frame) in enumerate(self.trace_info):
            if self.latest_frame - latest_frame < self.data_frame:
                valid_trace_index.append(trace_index)
        self.trace_info = [self.trace_info[v] for v in valid_trace_index]

        num_trace = len(self.trace_info)
        if num_trace == 0:
            return None

        data = np.zeros((3, self.data_frame, self.num_joint, num_trace))
        for trace_index, (trace, latest_frame) in enumerate(self.trace_info):
            end = self.data_frame - (self.latest_frame - latest_frame)
            d = trace[-end:]
            beg = end - len(d)
            data[:, beg:end, :, trace_index] = d.transpose((2, 0, 1))

        return data

    # concatenate pose to a trace
    def cat_pose(self, trace, pose, pad, pad_mode):
        # trace.shape: (num_frame, num_joint, 3)
        num_joint = pose.shape[0]
        num_channel = pose.shape[1]
        if pad != 0:
            if pad_mode == 'zero':
                trace = np.concatenate(
                    (trace, np.zeros((pad, num_joint, 3))), 0)
            elif pad_mode == 'interp':
                last_pose = trace[-1]
                coeff = [(p + 1) / (pad + 1) for p in range(pad)]
                interp_pose = [(1 - c) * last_pose + c * pose for c in coeff]
                trace = np.concatenate((trace, interp_pose), 0)
        new_trace = np.concatenate((trace, [pose]), 0)
        return new_trace

    # calculate the distance between a existing trace and the input pose

    def get_dis(self, trace, pose):
        last_pose_xy = trace[-1, :, 0:2]
        curr_pose_xy = pose[:, 0:2]

        mean_dis = ((((last_pose_xy - curr_pose_xy) ** 2).sum(1)) ** 0.5).mean()
        wh = last_pose_xy.max(0) - last_pose_xy.min(0)
        scale = (wh[0] * wh[1]) ** 0.5 + 0.0001
        is_close = mean_dis < scale * self.max_frame_dis
        return mean_dis, is_close


def put_text(img, text, position, scale_factor=1):
    t_w, t_h = cv2.getTextSize(
        text, cv2.FONT_HERSHEY_TRIPLEX, scale_factor, thickness=1)[0]
    H, W, _ = img.shape
    position = (int(W * position[1] - t_w * 0.5),
                int(H * position[0] - t_h * 0.5))
    params = (position, cv2.FONT_HERSHEY_TRIPLEX, scale_factor,
              (255, 255, 255))
    cv2.putText(img, text, *params)


def blend(background, foreground, dx=20, dy=10, fy=0.7):
    foreground = cv2.resize(foreground, (0, 0), fx=fy, fy=fy)
    h, w = foreground.shape[:2]
    b, g, r, a = cv2.split(foreground)
    mask = np.dstack((a, a, a))
    rgb = np.dstack((b, g, r))

    canvas = background[-h - dy:-dy, dx:w + dx]
    imask = mask > 0
    canvas[imask] = rgb[imask]


def stgcn_visualize(
        pose, edge, feature,
        video, label=None, label_sequence=None,
        height=1080, fps=None):
    _, T, V, M = pose.shape
    T = len(video)
    pos_track = [None] * M
    for t in range(T):
        frame = video[t]

        # image resize
        H, W, c = frame.shape
        frame = cv2.resize(frame, (height * W // H // 2, height // 2))
        H, W, c = frame.shape
        scale_factor = 2 * height / 1080

        # draw skeleton
        skeleton = frame * 0
        text = frame * 0
        for m in range(M):

            score = pose[2, t, :, m].max()
            if score < 0.3:
                continue

            for i, j in edge:
                xi = pose[0, t, i, m]
                yi = pose[1, t, i, m]
                xj = pose[0, t, j, m]
                yj = pose[1, t, j, m]
                if xi + yi == 0 or xj + yj == 0:
                    continue
                else:
                    xi = int((xi + 0.5) * W)
                    yi = int((yi + 0.5) * H)
                    xj = int((xj + 0.5) * W)
                    yj = int((yj + 0.5) * H)
                cv2.line(skeleton, (xi, yi), (xj, yj), (255, 255, 255),
                         int(np.ceil(2 * scale_factor)))

            if label_sequence is not None:
                body_label = label_sequence[t // 4][m]
            else:
                body_label = ''
            x_nose = int((pose[0, t, 0, m] + 0.5) * W)
            y_nose = int((pose[1, t, 0, m] + 0.5) * H)
            x_neck = int((pose[0, t, 1, m] + 0.5) * W)
            y_neck = int((pose[1, t, 1, m] + 0.5) * H)

            half_head = int(((x_neck - x_nose) ** 2 + (y_neck - y_nose) ** 2) ** 0.5)
            pos = (x_nose + half_head, y_nose - half_head)
            if pos_track[m] is None:
                pos_track[m] = pos
            else:
                new_x = int(pos_track[m][0] + (pos[0] - pos_track[m][0]) * 0.2)
                new_y = int(pos_track[m][1] + (pos[1] - pos_track[m][1]) * 0.2)
                pos_track[m] = (new_x, new_y)
            cv2.putText(text, body_label, pos_track[m],
                        cv2.FONT_HERSHEY_TRIPLEX, 0.5 * scale_factor,
                        (255, 255, 255))

        # generate mask
        mask = frame * 0
        feature = np.abs(feature)
        feature = feature / feature.mean()
        for m in range(M):
            score = pose[2, t, :, m].max()
            if score < 0.3:
                continue

            f = feature[t // 4, :, m] ** 5
            if f.mean() != 0:
                f = f / f.mean()
            for v in range(V):
                x = pose[0, t, v, m]
                y = pose[1, t, v, m]
                if x + y == 0:
                    continue
                else:
                    x = int((x + 0.5) * W)
                    y = int((y + 0.5) * H)
                cv2.circle(mask, (x, y), 0, (255, 255, 255),
                           int(np.ceil(f[v] ** 0.5 * 8 * scale_factor)))
        blurred_mask = cv2.blur(mask, (12, 12))

        skeleton_result = blurred_mask.astype(float) * 0.75
        skeleton_result += skeleton.astype(float) * 0.25
        skeleton_result += text.astype(float)
        skeleton_result[skeleton_result > 255] = 255
        skeleton_result = skeleton_result.astype(np.uint8)

        rgb_result = blurred_mask.astype(float) * 0.75
        rgb_result += frame.astype(float) * 0.5
        rgb_result += skeleton.astype(float) * 0.25
        rgb_result[rgb_result > 255] = 255
        rgb_result = rgb_result.astype(np.uint8)

        put_text(skeleton, 'inputs of st-gcn', (0.15, 0.5))

        text_1 = imread(
            './resource/demo_asset/original_video.png', cv2.IMREAD_UNCHANGED)
        text_2 = imread(
            './resource/demo_asset/pose_estimation.png', cv2.IMREAD_UNCHANGED)
        text_3 = imread(
            './resource/demo_asset/attention+prediction.png', cv2.IMREAD_UNCHANGED)
        text_4 = imread(
            './resource/demo_asset/attention+rgb.png', cv2.IMREAD_UNCHANGED)

        try:
            blend(frame, text_1)
            blend(skeleton, text_2)
            blend(skeleton_result, text_3)
            blend(rgb_result, text_4)
        except:
            pass

        if label is not None:
            label_name = 'voting result: ' + label
            put_text(skeleton_result, label_name, (0.1, 0.5))

        if fps is not None:
            put_text(skeleton, 'fps:{:.2f}'.format(fps), (0.9, 0.5))

        img0 = np.concatenate((frame, skeleton), axis=1)
        img1 = np.concatenate((skeleton_result, rgb_result), axis=1)
        img = np.concatenate((img0, img1), axis=0)

        yield img


def render_video(data, voting_label_name, video_label_name, intensity, frames):
    images = stgcn_visualize(
        data,
        graph_edge,
        intensity, frames,
        voting_label_name,
        video_label_name)
    return images


def render_image(data, voting_label_name, video_label_name, intensity, image, fps=0):
    images = stgcn_visualize(
        data[:, [-1]],
        graph_edge,
        intensity[[-1]], [image],
        voting_label_name,
        [video_label_name[-1]],
        fps=fps)
    image = next(images)
    image = image.astype(np.uint8)
    return image
