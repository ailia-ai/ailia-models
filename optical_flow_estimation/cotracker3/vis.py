import os
import cv2
import numpy as np


from matplotlib import cm
import matplotlib.pyplot as plt

def read_video_from_path(path):
    try:
        cap = cv2.VideoCapture(path)
    except Exception as e:
        print("Error opening video file: ", e)
        return None
    frames = []

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)  # フレームをリストに追加
    cap.release()

    return np.stack(frames)


class Visualizer:
    def __init__(
        self,
        pad_value: int = 0,
        linewidth: int = 2,
        show_first_frame: int = 10,
        tracks_leave_trace: int = 0,  # -1 for infinite
    ):

        self.color_map = cm.get_cmap("gist_rainbow")

        self.show_first_frame = show_first_frame
        self.tracks_leave_trace = tracks_leave_trace
        self.pad_value = pad_value
        self.linewidth = linewidth

    def visualize(
        self,
        video,
        tracks,
        visibility=None,

        filename: str = "video",

        query_frame=0,
        opacity: float = 1.0,
    ):

        video = pad(video,self.pad_value,255)

        color_alpha = int(opacity * 255)
        tracks = tracks + self.pad_value

        res_video = self.draw_tracks_on_video(
            video=video,
            tracks=tracks,
            visibility=visibility,
            query_frame=query_frame,
            color_alpha=color_alpha,
        )

        self.save_video(res_video, filename=filename)
        return res_video

    def save_video(self, video, filename):


        wide_list = [video[:,i,:,:,:] for i in range(video.shape[1])]

        wide_list = [np.transpose(wide[0],(1, 2, 0)) for wide in wide_list]

        # Prepare the video file path
        save_path =  filename

        # Create a writer object
        height, width, channels = wide_list[0].shape
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # MP4コーデック
        out = cv2.VideoWriter(save_path, fourcc, 30, (width, height))
        
        # Write frames to the video file
        for frame in wide_list[2:-1]:
            out.write(frame)
        out.release()

        print(f"Video saved to {save_path}")

    def draw_tracks_on_video(
        self,
        video,
        tracks,
        visibility = None,
        query_frame=0,
        color_alpha: int = 255,
    ):


        B, T, C, H, W = video.shape
        _, _, N, D = tracks.shape

        segm_mask = None

        assert D == 2
        assert C == 3

        video = np.transpose(video[0],(0, 2, 3, 1)).astype(np.uint8)  # S, H, W, C
        tracks = tracks[0].astype(np.int64)


        res_video = []

        # process input video
        for rgb in video:
            res_video.append(rgb.copy())
        vector_colors = np.zeros((T, N, 3))

        #if segm_mask is None:
        y_min, y_max = (
            tracks[query_frame, :, 1].min(),
            tracks[query_frame, :, 1].max(),
        )
        norm = plt.Normalize(y_min, y_max)
        for n in range(N):
            query_frame_ = query_frame

            color = self.color_map(norm(tracks[query_frame_, n, 1]))
            color = np.array(color[:3])[None] * 255
            vector_colors[:, n] = np.repeat(color, T, axis=0)

        #  draw tracks
        if self.tracks_leave_trace != 0:
            for t in range(query_frame + 1, T):
                first_ind = (
                    max(0, t - self.tracks_leave_trace)
                    if self.tracks_leave_trace >= 0
                    else 0
                )
                curr_tracks = tracks[first_ind : t + 1]
                curr_colors = vector_colors[first_ind : t + 1]

                res_video[t] = self._draw_pred_tracks(
                    res_video[t],
                    curr_tracks,
                    curr_colors,
                )

        #  draw points
        for t in range(T):
            img = np.uint8(res_video[t])
            for i in range(N):
                coord = (tracks[t, i, 0], tracks[t, i, 1])
                visibile = True
                if visibility is not None:
                    visibile = visibility[0, t, i]
                if coord[0] != 0 and coord[1] != 0:
                    img = draw_circle(
                        img,
                        coord=coord,
                        radius=int(self.linewidth * 2),
                        color=vector_colors[t, i].astype(int),
                        visible=visibile,
                        color_alpha=color_alpha,
                    )
            res_video[t] = np.array(img)

        #  construct the final rgb sequence
        if self.show_first_frame > 0:
            res_video = [res_video[0]] * self.show_first_frame + res_video[1:]
        return np.transpose(np.stack(res_video),(0, 3, 1, 2))[np.newaxis, ...].astype(np.uint8)

def draw_ellipse(image, left_up_point, right_down_point, color, visible=True):
    center = (
        (left_up_point[0] + right_down_point[0]) // 2,
        (left_up_point[1] + right_down_point[1]) // 2,
    )
    axes = (
        abs(right_down_point[0] - left_up_point[0]) // 2,
        abs(right_down_point[1] - left_up_point[1]) // 2,
    )
    thickness = -1 if visible else 2
    color = tuple(map(int, color))
    cv2.ellipse(image, center, axes, 0, 0, 360, color, thickness)
    return image

def draw_circle(rgb, coord, radius, color=(255, 0, 0), visible=True, color_alpha=None):
    # Create a draw object
    # Calculate the bounding box of the circle
    left_up_point = (coord[0] - radius, coord[1] - radius)
    right_down_point = (coord[0] + radius, coord[1] + radius)
    # Draw the circle
    color = tuple(list(color) + [color_alpha if color_alpha is not None else 255])

    rgb = draw_ellipse(rgb,left_up_point,right_down_point,color,True)

    return rgb



def pad(video, pad_value, constant_value=255):
    padding = ((0, 0),
               (0, 0),
               (0, 0),
               (pad_value, pad_value),
               (pad_value, pad_value))

    padded_video = np.pad(video, pad_width=padding, mode='constant', constant_values=constant_value)
    return padded_video


