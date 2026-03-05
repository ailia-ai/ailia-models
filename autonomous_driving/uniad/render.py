import cv2
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from nuscenes import NuScenes
from nuscenes.prediction import convert_local_coords_to_global
from nuscenes.utils.data_classes import Box
from nuscenes.utils.geometry_utils import box_in_image, view_points
from PIL import Image
from pyquaternion import Quaternion

# isort: off
from lidar_box3d import LiDARInstance3DBoxes


color_mapping = (
    np.asarray(
        [
            [0, 0, 0],
            [255, 179, 0],
            [128, 62, 117],
            [255, 104, 0],
            [166, 189, 215],
            [193, 0, 32],
            [206, 162, 98],
            [129, 112, 102],
            [0, 125, 52],
            [246, 118, 142],
            [0, 83, 138],
            [255, 122, 92],
            [83, 55, 122],
            [255, 142, 0],
            [179, 40, 81],
            [244, 200, 0],
            [127, 24, 13],
            [147, 170, 0],
            [89, 51, 21],
            [241, 58, 19],
            [35, 44, 22],
            [112, 224, 255],
            [70, 184, 160],
            [153, 0, 255],
            [71, 255, 0],
            [255, 0, 163],
            [255, 204, 0],
            [0, 255, 235],
            [255, 0, 235],
            [255, 0, 122],
            [255, 245, 0],
            [10, 190, 212],
            [214, 255, 0],
            [0, 204, 255],
            [20, 0, 255],
            [255, 255, 0],
            [0, 153, 255],
            [0, 255, 204],
            [41, 255, 0],
            [173, 0, 255],
            [0, 245, 255],
            [71, 0, 255],
            [0, 255, 184],
            [0, 92, 255],
            [184, 255, 0],
            [255, 214, 0],
            [25, 194, 194],
            [92, 0, 255],
            [220, 220, 220],
            [255, 9, 92],
            [112, 9, 255],
            [8, 255, 214],
            [255, 184, 6],
            [10, 255, 71],
            [255, 41, 10],
            [7, 255, 255],
            [224, 255, 8],
            [102, 8, 255],
            [255, 61, 6],
            [255, 194, 7],
            [0, 255, 20],
            [255, 8, 41],
            [255, 5, 153],
            [6, 51, 255],
            [235, 12, 255],
            [160, 150, 20],
            [0, 163, 255],
            [140, 140, 140],
            [250, 10, 15],
            [20, 255, 0],
        ]
    )
    / 255
)


class BaseRender:
    """
    BaseRender class
    """

    def __init__(self, figsize=(10, 10)):
        self.figsize = figsize
        self.fig, self.axes = None, None

    def reset_canvas(self, dx=1, dy=1, tight_layout=False):
        plt.close()
        plt.gca().set_axis_off()
        plt.axis("off")
        self.fig, self.axes = plt.subplots(dx, dy, figsize=self.figsize)
        if tight_layout:
            plt.tight_layout()

    def close_canvas(self):
        plt.close()

    def save_fig(self, filename=None):
        plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
        plt.margins(0, 0)
        if filename is not None:
            print(f"saving to {filename}")
            plt.savefig(filename)
        # Return image as numpy array
        self.fig.canvas.draw()
        img = np.frombuffer(self.fig.canvas.tostring_rgb(), dtype=np.uint8)
        img = img.reshape(self.fig.canvas.get_width_height()[::-1] + (3,))
        return img


class BEVRender(BaseRender):
    """
    Render class for BEV
    """

    def __init__(
        self,
        figsize=(20, 20),
        margin: float = 50,
        view: np.ndarray = np.eye(4),
        show_gt_boxes=False,
    ):
        super(BEVRender, self).__init__(figsize)
        self.margin = margin
        self.view = view
        self.show_gt_boxes = show_gt_boxes

    def set_plot_cfg(self):
        self.axes.set_xlim([-self.margin, self.margin])
        self.axes.set_ylim([-self.margin, self.margin])
        self.axes.set_aspect("equal")
        self.axes.grid(False)

    def render_pred_box_data(self, agent_prediction_list):
        for pred_agent in agent_prediction_list:
            c = np.array([0, 1, 0])
            if (
                hasattr(pred_agent, "pred_track_id")
                and pred_agent.pred_track_id is not None
            ):  # this is true
                tr_id = pred_agent.pred_track_id
                c = color_mapping[tr_id % len(color_mapping)]
            pred_agent.nusc_box.render(axis=self.axes, view=self.view, colors=(c, c, c))
            if pred_agent.is_sdc:
                c = np.array([1, 0, 0])
                pred_agent.nusc_box.render(
                    axis=self.axes, view=self.view, colors=(c, c, c)
                )

    def render_pred_traj(self, agent_prediction_list, top_k=3):
        for pred_agent in agent_prediction_list:
            if pred_agent.is_sdc:
                continue
            sorted_ind = np.argsort(pred_agent.pred_traj_score)[
                ::-1
            ]  # from high to low
            num_modes = len(sorted_ind)
            sorted_traj = pred_agent.pred_traj[sorted_ind, :, :2]
            sorted_score = pred_agent.pred_traj_score[sorted_ind]
            norm_score = np.exp(sorted_score[0])

            sorted_traj = np.concatenate(
                [np.zeros((num_modes, 1, 2)), sorted_traj], axis=1
            )
            trans = pred_agent.pred_center
            rot = Quaternion(axis=np.array([0, 0.0, 1.0]), angle=np.pi / 2)
            vehicle_id_list = [0, 1, 2, 3, 4, 6, 7]
            if pred_agent.pred_label in vehicle_id_list:
                dot_size = 150
            else:
                dot_size = 25
            for i in range(top_k - 1, -1, -1):
                viz_traj = sorted_traj[i, :, :2]
                viz_traj = convert_local_coords_to_global(viz_traj, trans, rot)
                traj_score = np.exp(sorted_score[i]) / norm_score
                self._render_traj(
                    viz_traj,
                    traj_score=traj_score,
                    colormap="winter",
                    dot_size=dot_size,
                )

    def render_planning_data(self, predicted_planning, show_command=False):
        planning_traj = predicted_planning.pred_traj
        planning_traj = np.concatenate([np.zeros((1, 2)), planning_traj], axis=0)
        self._render_traj(planning_traj, colormap="autumn", dot_size=50)
        if show_command:
            self._render_command(predicted_planning.command)

    def _render_traj(
        self,
        future_traj,
        traj_score=1,
        colormap="winter",
        points_per_step=20,
        line_color=None,
        dot_color=None,
        dot_size=25,
    ):
        total_steps = (len(future_traj) - 1) * points_per_step + 1
        dot_colors = matplotlib.colormaps[colormap](np.linspace(0, 1, total_steps))[
            :, :3
        ]
        dot_colors = dot_colors * traj_score + (1 - traj_score) * np.ones_like(
            dot_colors
        )
        total_xy = np.zeros((total_steps, 2))
        for i in range(total_steps - 1):
            unit_vec = (
                future_traj[i // points_per_step + 1]
                - future_traj[i // points_per_step]
            )
            total_xy[i] = (
                i / points_per_step - i // points_per_step
            ) * unit_vec + future_traj[i // points_per_step]
        total_xy[-1] = future_traj[-1]
        self.axes.scatter(total_xy[:, 0], total_xy[:, 1], c=dot_colors, s=dot_size)

    def _render_command(self, command):
        command_dict = ["TURN RIGHT", "TURN LEFT", "KEEP FORWARD"]
        self.axes.text(-48, -45, command_dict[int(command)], fontsize=45)

    def render_sdc_car(self):
        sdc_car_png = cv2.imread("resources/sdc_car.png")
        sdc_car_png = cv2.cvtColor(sdc_car_png, cv2.COLOR_BGR2RGB)
        self.axes.imshow(sdc_car_png, extent=(-1, 1, -2, 2))

    def render_legend(self):
        legend = cv2.imread("resources/legend.png")
        legend = cv2.cvtColor(legend, cv2.COLOR_BGR2RGB)
        self.axes.imshow(legend, extent=(23, 51.2, -50, -40))


class CameraRender(BaseRender):
    def __init__(self, figsize=(53.3333, 20), show_gt_boxes=False):
        super().__init__(figsize=figsize)

        self.cams = [
            "CAM_FRONT_LEFT",
            "CAM_FRONT",
            "CAM_FRONT_RIGHT",
            "CAM_BACK_RIGHT",
            "CAM_BACK",
            "CAM_BACK_LEFT",
        ]
        self.show_gt_boxes = show_gt_boxes

    def get_axis(self, index):
        """Retrieve the corresponding axis based on the index."""
        return self.axes[index // 3, index % 3]

    def project_to_cam(
        self,
        agent_prediction_list,
        sample_data_token,
        nusc,
        lidar_cs_record,
        project_traj=False,
        cam=None,
    ):
        """Project predictions to camera view."""
        _, cs_record, pose_record, cam_intrinsic, imsize = self.get_image_info(
            sample_data_token, nusc
        )
        boxes = []
        for agent in agent_prediction_list:
            box = Box(
                agent.pred_center,
                agent.pred_dim,
                Quaternion(axis=(0.0, 0.0, 1.0), radians=agent.pred_yaw),
                name=agent.pred_label,
                token="predicted",
            )
            box.is_sdc = agent.is_sdc
            if project_traj:
                box.pred_traj = np.zeros((agent.pred_traj_max.shape[0] + 1, 3))
                box.pred_traj[:, 0] = agent.pred_center[0]
                box.pred_traj[:, 1] = agent.pred_center[1]
                box.pred_traj[:, 2] = agent.pred_center[2] - agent.pred_dim[2] / 2
                box.pred_traj[1:, :2] += agent.pred_traj_max[:, :2]
                box.pred_traj = (
                    Quaternion(lidar_cs_record["rotation"]).rotation_matrix
                    @ box.pred_traj.T
                ).T
                box.pred_traj += np.array(lidar_cs_record["translation"])[None, :]
            box.rotate(Quaternion(lidar_cs_record["rotation"]))
            box.translate(np.array(lidar_cs_record["translation"]))
            boxes.append(box)
        # Make list of Box objects including coord system transforms.

        box_list = []
        tr_id_list = []
        for i, box in enumerate(boxes):
            #  Move box to sensor coord system.
            box.translate(-np.array(cs_record["translation"]))
            box.rotate(Quaternion(cs_record["rotation"]).inverse)
            if project_traj:
                box.pred_traj += -np.array(cs_record["translation"])[None, :]
                box.pred_traj = (
                    Quaternion(cs_record["rotation"]).inverse.rotation_matrix
                    @ box.pred_traj.T
                ).T

            tr_id = agent_prediction_list[i].pred_track_id
            if box.is_sdc and cam == "CAM_FRONT":
                box_list.append(box)
            if not box_in_image(box, cam_intrinsic, imsize):
                continue
            box_list.append(box)
            tr_id_list.append(tr_id)
        return box_list, tr_id_list, cam_intrinsic, imsize

    def render_image_data(self, sample_token, nusc):
        """Load and annotate image based on the provided path."""
        sample = nusc.get("sample", sample_token)
        for i, cam in enumerate(self.cams):
            sample_data_token = sample["data"][cam]
            data_path, _, _, _, _ = self.get_image_info(sample_data_token, nusc)
            image = self.load_image(data_path, cam)
            self.update_image(image, i, cam)

    def load_image(self, data_path, cam):
        """Update the axis of the plot with the provided image."""
        image = np.array(Image.open(data_path))
        font = cv2.FONT_HERSHEY_SIMPLEX
        org = (50, 60)
        fontScale = 2
        color = (0, 0, 0)
        thickness = 4
        return cv2.putText(
            image, cam, org, font, fontScale, color, thickness, cv2.LINE_AA
        )

    def update_image(self, image, index, cam):
        """Render image data for each camera."""
        ax = self.get_axis(index)
        ax.imshow(image)
        plt.axis("off")
        ax.axis("off")
        ax.grid(False)

    def render_pred_track_bbox(self, predicted_agent_list, sample_token, nusc):
        """Render bounding box for predicted tracks."""
        sample = nusc.get("sample", sample_token)
        lidar_cs_record = nusc.get(
            "calibrated_sensor",
            nusc.get("sample_data", sample["data"]["LIDAR_TOP"])[
                "calibrated_sensor_token"
            ],
        )
        for i, cam in enumerate(self.cams):
            sample_data_token = sample["data"][cam]
            box_list, tr_id_list, camera_intrinsic, imsize = self.project_to_cam(
                predicted_agent_list, sample_data_token, nusc, lidar_cs_record
            )
            for j, box in enumerate(box_list):
                if box.is_sdc:
                    continue
                tr_id = tr_id_list[j]
                if tr_id is None:
                    tr_id = 0
                c = color_mapping[tr_id % len(color_mapping)]
                box.render(
                    self.axes[i // 3, i % 3],
                    view=camera_intrinsic,
                    normalize=True,
                    colors=(c, c, c),
                )
            # plot gt
            if self.show_gt_boxes:
                data_path, boxes, camera_intrinsic = nusc.get_sample_data(
                    sample_data_token, selected_anntokens=sample["anns"]
                )
                for j, box in enumerate(boxes):
                    c = [0, 1, 0]
                    box.render(
                        self.axes[i // 3, i % 3],
                        view=camera_intrinsic,
                        normalize=True,
                        colors=(c, c, c),
                    )
            self.axes[i // 3, i % 3].set_xlim(0, imsize[0])
            self.axes[i // 3, i % 3].set_ylim(imsize[1], 0)

    def render_pred_traj(
        self,
        predicted_agent_list,
        sample_token,
        nusc,
        render_sdc=False,
        points_per_step=10,
    ):
        """Render predicted trajectories."""
        sample = nusc.get("sample", sample_token)
        lidar_cs_record = nusc.get(
            "calibrated_sensor",
            nusc.get("sample_data", sample["data"]["LIDAR_TOP"])[
                "calibrated_sensor_token"
            ],
        )
        for i, cam in enumerate(self.cams):
            sample_data_token = sample["data"][cam]
            box_list, tr_id_list, camera_intrinsic, imsize = self.project_to_cam(
                predicted_agent_list,
                sample_data_token,
                nusc,
                lidar_cs_record,
                project_traj=True,
                cam=cam,
            )
            for j, box in enumerate(box_list):
                traj_points = box.pred_traj[:, :3]

                total_steps = (len(traj_points) - 1) * points_per_step + 1
                total_xy = np.zeros((total_steps, 3))
                for k in range(total_steps - 1):
                    unit_vec = (
                        traj_points[k // points_per_step + 1]
                        - traj_points[k // points_per_step]
                    )
                    total_xy[k] = (
                        k / points_per_step - k // points_per_step
                    ) * unit_vec + traj_points[k // points_per_step]
                in_range_mask = total_xy[:, 2] > 0.1
                traj_points = view_points(total_xy.T, camera_intrinsic, normalize=True)[
                    :2, :
                ]
                traj_points = traj_points[:2, in_range_mask]
                if box.is_sdc:
                    if render_sdc:
                        self.axes[i // 3, i % 3].scatter(
                            traj_points[0], traj_points[1], color=(1, 0.5, 0), s=150
                        )
                    else:
                        continue
                else:
                    tr_id = tr_id_list[j]
                    if tr_id is None:
                        tr_id = 0
                    c = color_mapping[tr_id % len(color_mapping)]
                    self.axes[i // 3, i % 3].scatter(
                        traj_points[0], traj_points[1], color=c, s=15
                    )
            self.axes[i // 3, i % 3].set_xlim(0, imsize[0])
            self.axes[i // 3, i % 3].set_ylim(imsize[1], 0)

    def get_image_info(self, sample_data_token, nusc):
        """Retrieve image information."""
        sd_record = nusc.get("sample_data", sample_data_token)
        cs_record = nusc.get("calibrated_sensor", sd_record["calibrated_sensor_token"])
        sensor_record = nusc.get("sensor", cs_record["sensor_token"])
        pose_record = nusc.get("ego_pose", sd_record["ego_pose_token"])

        data_path = nusc.get_sample_data_path(sample_data_token)

        if sensor_record["modality"] == "camera":
            cam_intrinsic = np.array(cs_record["camera_intrinsic"])
            imsize = (sd_record["width"], sd_record["height"])
        else:
            cam_intrinsic = None
            imsize = None
        return data_path, cs_record, pose_record, cam_intrinsic, imsize


class AgentPredictionData:
    """
    Agent data class, includes bbox, traj, and occflow
    """

    def __init__(
        self,
        pred_score,
        pred_label,
        pred_center,
        pred_dim,
        pred_yaw,
        pred_vel,
        pred_traj,
        pred_traj_score,
        pred_track_id=None,
        pred_occ_map=None,
        is_sdc=False,
        past_pred_traj=None,
        command=None,
        attn_mask=None,
    ):
        self.pred_score = pred_score
        self.pred_label = pred_label
        self.pred_center = pred_center
        self.pred_dim = pred_dim
        # self.pred_yaw = pred_yaw
        self.pred_yaw = -pred_yaw - np.pi / 2
        self.pred_vel = pred_vel
        self.pred_traj = pred_traj
        self.pred_traj_score = pred_traj_score
        self.pred_track_id = pred_track_id
        self.pred_occ_map = pred_occ_map
        if self.pred_traj is not None:
            if isinstance(self.pred_traj_score, int):
                self.pred_traj_max = self.pred_traj
            else:
                self.pred_traj_max = self.pred_traj[self.pred_traj_score.argmax()]
        else:
            self.pred_traj_max = None
        self.nusc_box = Box(
            center=pred_center,
            size=pred_dim,
            orientation=Quaternion(axis=[0, 0, 1], radians=self.pred_yaw),
            label=pred_label,
            score=pred_score,
        )
        if is_sdc:
            self.pred_center = [0, 0, -1.2 + 1.56 / 2]
        self.is_sdc = is_sdc
        self.past_pred_traj = past_pred_traj
        self.command = command
        self.attn_mask = attn_mask


class Visualizer:
    def __init__(
        self, bbox_results, nuscenes=None, dataroot="data/nuscenes", version="v1.0-mini"
    ):
        if nuscenes is None:
            self.nusc = NuScenes(version=version, dataroot=dataroot, verbose=True)
        else:
            self.nusc = nuscenes

        self.with_planning = True
        self.show_command = True
        self.show_sdc_car = True
        self.show_legend = True
        self.with_pred_traj = True
        self.with_pred_box = True
        show_gt_boxes = False

        self.token_set = set()
        self.predictions = self._parse_predictions(bbox_results)
        self.bev_render = BEVRender(show_gt_boxes=show_gt_boxes)
        self.cam_render = CameraRender(show_gt_boxes=show_gt_boxes)

    def _parse_predictions(self, bbox_results):
        outputs = bbox_results
        prediction_dict = dict()
        for k in range(len(outputs)):
            token = outputs[k]["token"]
            self.token_set.add(token)

            # detection
            bboxes: LiDARInstance3DBoxes = outputs[k]["boxes_3d"]
            scores = outputs[k]["scores_3d"]
            labels = outputs[k]["labels_3d"]

            track_scores = scores
            track_labels = labels
            track_boxes = bboxes.tensor

            track_centers = bboxes.gravity_center
            track_dims = bboxes.dims
            track_yaw = bboxes.yaw

            track_ids = outputs[k]["track_ids"]

            # speed
            track_velocity = bboxes.tensor[:, -2:]

            # trajectories
            trajs = outputs[k]["traj"]
            traj_scores = outputs[k]["traj_scores"]

            predicted_agent_list = []
            for i in range(track_scores.shape[0]):
                if track_scores[i] < 0.25:
                    continue

                if i < len(track_ids):
                    track_id = track_ids[i]
                else:
                    track_id = 0

                predicted_agent_list.append(
                    AgentPredictionData(
                        track_scores[i],
                        track_labels[i],
                        track_centers[i],
                        track_dims[i],
                        track_yaw[i],
                        track_velocity[i],
                        trajs[i],
                        traj_scores[i],
                        pred_track_id=track_id,
                        pred_occ_map=None,
                        past_pred_traj=None,
                    )
                )

            # with planning
            if self.with_planning:
                # detection
                bboxes: LiDARInstance3DBoxes = outputs[k]["sdc_boxes_3d"]
                scores = outputs[k]["sdc_scores_3d"]
                labels = 0

                track_scores = scores
                track_labels = labels

                track_centers = bboxes.gravity_center
                track_dims = bboxes.dims
                track_yaw = bboxes.yaw
                track_velocity = bboxes.tensor[:, -2:]

                command = outputs[k]["command"]
                planning_agent = AgentPredictionData(
                    track_scores[0],
                    track_labels,
                    track_centers[0],
                    track_dims[0],
                    track_yaw[0],
                    track_velocity[0],
                    outputs[k]["planning_traj"][0],
                    1,
                    pred_track_id=-1,
                    pred_occ_map=None,
                    past_pred_traj=None,
                    is_sdc=True,
                    command=command,
                )
                predicted_agent_list.append(planning_agent)
            else:
                planning_agent = None
            prediction_dict[token] = dict(
                predicted_agent_list=predicted_agent_list,
                predicted_map_seg=None,
                predicted_planning=planning_agent,
            )

        return prediction_dict

    def visualize_bev(self, sample_token):
        self.bev_render.reset_canvas(dx=1, dy=1)
        self.bev_render.set_plot_cfg()

        if self.with_pred_box:
            self.bev_render.render_pred_box_data(
                self.predictions[sample_token]["predicted_agent_list"]
            )
        if self.with_pred_traj:
            self.bev_render.render_pred_traj(
                self.predictions[sample_token]["predicted_agent_list"]
            )
        if self.with_planning:
            self.bev_render.render_pred_box_data(
                [self.predictions[sample_token]["predicted_planning"]]
            )
            self.bev_render.render_planning_data(
                self.predictions[sample_token]["predicted_planning"],
                show_command=self.show_command,
            )
        if self.show_sdc_car:
            self.bev_render.render_sdc_car()
        if self.show_legend:
            self.bev_render.render_legend()

        img = self.bev_render.save_fig()
        return img

    def visualize_cam(self, sample_token):
        self.cam_render.reset_canvas(dx=2, dy=3, tight_layout=True)
        self.cam_render.render_image_data(sample_token, self.nusc)
        self.cam_render.render_pred_track_bbox(
            self.predictions[sample_token]["predicted_agent_list"],
            sample_token,
            self.nusc,
        )
        self.cam_render.render_pred_traj(
            self.predictions[sample_token]["predicted_agent_list"],
            sample_token,
            self.nusc,
            render_sdc=self.with_planning,
        )

        img = self.cam_render.save_fig()
        return img
