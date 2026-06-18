import os
import pickle

import cv2
import numpy as np
import tqdm
from nuscenes import NuScenes
from nuscenes.can_bus.can_bus_api import NuScenesCanBus
from nuscenes.eval.common.utils import Quaternion, quaternion_yaw
from nuscenes.prediction import PredictHelper
from nuscenes.utils import splits
from torch.utils.data import Dataset

# isort: off
from lidar_box3d import LiDARInstance3DBoxes


def obtain_sensor2top(
    nusc, sensor_token, l2e_t, l2e_r_mat, e2g_t, e2g_r_mat, sensor_type="lidar"
):
    """Obtain the info with RT matric from general sensor to Top LiDAR.

    Args:
        nusc (class): Dataset class in the nuScenes dataset.
        sensor_token (str): Sample data token corresponding to the
            specific sensor type.
        l2e_t (np.ndarray): Translation from lidar to ego in shape (1, 3).
        l2e_r_mat (np.ndarray): Rotation matrix from lidar to ego
            in shape (3, 3).
        e2g_t (np.ndarray): Translation from ego to global in shape (1, 3).
        e2g_r_mat (np.ndarray): Rotation matrix from ego to global
            in shape (3, 3).
        sensor_type (str): Sensor to calibrate. Default: 'lidar'.

    Returns:
        sweep (dict): Sweep information after transformation.
    """
    sd_rec = nusc.get("sample_data", sensor_token)
    cs_record = nusc.get("calibrated_sensor", sd_rec["calibrated_sensor_token"])
    pose_record = nusc.get("ego_pose", sd_rec["ego_pose_token"])
    data_path = str(nusc.get_sample_data_path(sd_rec["token"]))
    if nusc.dataroot in data_path:
        data_path = data_path.split(f"{nusc.dataroot}")[-1]  # relative path
    sweep = {
        "data_path": data_path,
        "type": sensor_type,
        "sample_data_token": sd_rec["token"],
        "sensor2ego_translation": cs_record["translation"],
        "sensor2ego_rotation": cs_record["rotation"],
        "ego2global_translation": pose_record["translation"],
        "ego2global_rotation": pose_record["rotation"],
        "timestamp": sd_rec["timestamp"],
    }

    l2e_r_s = sweep["sensor2ego_rotation"]
    l2e_t_s = sweep["sensor2ego_translation"]
    e2g_r_s = sweep["ego2global_rotation"]
    e2g_t_s = sweep["ego2global_translation"]

    # obtain the RT from sensor to Top LiDAR
    # sweep->ego->global->ego'->lidar
    l2e_r_s_mat = Quaternion(l2e_r_s).rotation_matrix
    e2g_r_s_mat = Quaternion(e2g_r_s).rotation_matrix
    R = (l2e_r_s_mat.T @ e2g_r_s_mat.T) @ (
        np.linalg.inv(e2g_r_mat).T @ np.linalg.inv(l2e_r_mat).T
    )
    T = (l2e_t_s @ e2g_r_s_mat.T + e2g_t_s) @ (
        np.linalg.inv(e2g_r_mat).T @ np.linalg.inv(l2e_r_mat).T
    )
    T -= (
        e2g_t @ (np.linalg.inv(e2g_r_mat).T @ np.linalg.inv(l2e_r_mat).T)
        + l2e_t @ np.linalg.inv(l2e_r_mat).T
    )
    sweep["sensor2lidar_rotation"] = R.T  # points @ R.T + T
    sweep["sensor2lidar_translation"] = T
    return sweep


def get_can_bus_info(nusc, nusc_can_bus, sample):
    scene_name = nusc.get("scene", sample["scene_token"])["name"]
    sample_timestamp = sample["timestamp"]
    try:
        pose_list = nusc_can_bus.get_messages(scene_name, "pose")
    except:
        return np.zeros(18)  # server scenes do not have can bus information.
    can_bus = []
    # during each scene, the first timestamp of can_bus may be large than the first sample's timestamp
    last_pose = pose_list[0]
    for i, pose in enumerate(pose_list):
        if pose["utime"] > sample_timestamp:
            break
        last_pose = pose
    _ = last_pose.pop("utime")  # useless
    pos = last_pose.pop("pos")
    rotation = last_pose.pop("orientation")
    can_bus.extend(pos)
    can_bus.extend(rotation)
    for key in last_pose.keys():
        can_bus.extend(pose[key])  # 16 elements
    can_bus.extend([0.0, 0.0])
    return np.array(can_bus)


def get_future_traj_info(nusc, sample, predict_steps=16):
    sample_token = sample["token"]
    ann_tokens = np.array(sample["anns"])
    sd_rec = nusc.get("sample", sample_token)
    fut_traj_all = []
    fut_traj_valid_mask_all = []
    _, boxes, _ = nusc.get_sample_data(
        sd_rec["data"]["LIDAR_TOP"], selected_anntokens=ann_tokens
    )
    predict_helper = PredictHelper(nusc)
    for i, ann_token in enumerate(ann_tokens):
        box = boxes[i]
        instance_token = nusc.get("sample_annotation", ann_token)["instance_token"]
        fut_traj_local = predict_helper.get_future_for_agent(
            instance_token,
            sample_token,
            seconds=predict_steps // 2,
            in_agent_frame=True,
        )

        fut_traj = np.zeros((predict_steps, 2))
        fut_traj_valid_mask = np.zeros((predict_steps, 2))
        if fut_traj_local.shape[0] > 0:
            fut_traj_scence_centric = fut_traj_local
            fut_traj[: fut_traj_scence_centric.shape[0], :] = fut_traj_scence_centric
            fut_traj_valid_mask[: fut_traj_scence_centric.shape[0], :] = 1
        fut_traj_all.append(fut_traj)
        fut_traj_valid_mask_all.append(fut_traj_valid_mask)
    if len(ann_tokens) > 0:
        fut_traj_all = np.stack(fut_traj_all, axis=0)
        fut_traj_valid_mask_all = np.stack(fut_traj_valid_mask_all, axis=0)
    else:
        fut_traj_all = np.zeros((0, predict_steps, 2))
        fut_traj_valid_mask_all = np.zeros((0, predict_steps, 2))
    return fut_traj_all, fut_traj_valid_mask_all


class NuScenesTraj:
    def __init__(
        self,
        nusc,
    ):
        self.nusc = nusc
        self.prepare_sdc_vel_info()
        self.planning_steps = 6
        self.with_velocity = True

    def get_vel_transform_mats(self, sample):
        sd_rec = self.nusc.get("sample_data", sample["data"]["LIDAR_TOP"])
        cs_record = self.nusc.get(
            "calibrated_sensor", sd_rec["calibrated_sensor_token"]
        )
        pose_record = self.nusc.get("ego_pose", sd_rec["ego_pose_token"])

        l2e_r = cs_record["rotation"]
        l2e_t = cs_record["translation"]
        e2g_r = pose_record["rotation"]
        e2g_t = pose_record["translation"]
        l2e_r_mat = Quaternion(l2e_r).rotation_matrix
        e2g_r_mat = Quaternion(e2g_r).rotation_matrix

        return l2e_r_mat, e2g_r_mat

    def get_vel_and_time(self, sample):
        lidar_token = sample["data"]["LIDAR_TOP"]
        lidar_top = self.nusc.get("sample_data", lidar_token)
        pose = self.nusc.get("ego_pose", lidar_top["ego_pose_token"])
        xyz = pose["translation"]
        timestamp = sample["timestamp"]
        return xyz, timestamp

    def prepare_sdc_vel_info(self):
        # generate sdc velocity info for all samples
        # Note that these velocity values are converted from
        # global frame to lidar frame
        # as aligned with bbox gts

        self.sdc_vel_info = {}
        for scene in self.nusc.scene:
            scene_token = scene["token"]

            # we cannot infer vel for the last sample, therefore we skip it
            last_sample_token = scene["last_sample_token"]
            sample_token = scene["first_sample_token"]
            sample = self.nusc.get("sample", sample_token)
            xyz, time = self.get_vel_and_time(sample)
            while sample["token"] != last_sample_token:
                next_sample_token = sample["next"]
                next_sample = self.nusc.get("sample", next_sample_token)
                next_xyz, next_time = self.get_vel_and_time(next_sample)
                dc = np.array(next_xyz) - np.array(xyz)
                dt = (next_time - time) / 1e6
                vel = dc / dt

                # global frame to lidar frame
                l2e_r_mat, e2g_r_mat = self.get_vel_transform_mats(sample)
                vel = vel @ np.linalg.inv(e2g_r_mat).T @ np.linalg.inv(l2e_r_mat).T
                vel = vel[:2]

                self.sdc_vel_info[sample["token"]] = vel
                xyz, time = next_xyz, next_time
                sample = next_sample

            # set first sample's vel equal to second sample's
            last_sample = self.nusc.get("sample", last_sample_token)
            second_last_sample_token = last_sample["prev"]
            self.sdc_vel_info[last_sample_token] = self.sdc_vel_info[
                second_last_sample_token
            ]

    def generate_sdc_info(self, sdc_vel):
        """Generate SDC (Self-Driving Car) bounding box info.

        Args:
            sdc_vel: SDC velocity [vx, vy]

        Returns:
            LiDARInstance3DBoxes: SDC bounding box
        """

        # sdc dim from https://forum.nuscenes.org/t/dimensions-of-the-ego-vehicle-used-to-gather-data/550
        # TODO(box3d): we have changed yaw to mmdet3d 1.0.0rc6 format, wlh->lwh -pi->0.5pi
        psudo_sdc_bbox = np.array([0.0, 0.0, 0.0, 4.08, 1.73, 1.56, 0.5 * np.pi])
        if self.with_velocity:
            psudo_sdc_bbox = np.concatenate([psudo_sdc_bbox, sdc_vel], axis=-1)
        gt_bboxes_3d = np.array([psudo_sdc_bbox]).astype(np.float32)

        # the nuscenes box center is [0.5, 0.5, 0.5], we change it to be
        # the same as KITTI (0.5, 0.5, 0)
        gt_bboxes_3d = LiDARInstance3DBoxes(
            gt_bboxes_3d, box_dim=gt_bboxes_3d.shape[-1], origin=(0.5, 0.5, 0.5)
        )

        return gt_bboxes_3d

    def get_l2g_transform(self, sample):
        sd_rec = self.nusc.get("sample_data", sample["data"]["LIDAR_TOP"])
        cs_record = self.nusc.get(
            "calibrated_sensor", sd_rec["calibrated_sensor_token"]
        )
        pose_record = self.nusc.get("ego_pose", sd_rec["ego_pose_token"])

        l2e_r = cs_record["rotation"]
        l2e_t = np.array(cs_record["translation"])
        e2g_r = pose_record["rotation"]
        e2g_t = np.array(pose_record["translation"])
        l2e_r_mat = Quaternion(l2e_r).rotation_matrix
        e2g_r_mat = Quaternion(e2g_r).rotation_matrix

        return l2e_r_mat, l2e_t, e2g_r_mat, e2g_t

    def get_sdc_planning_label(self, sample_token):
        sd_rec = self.nusc.get("sample", sample_token)
        l2e_r_mat_init, l2e_t_init, e2g_r_mat_init, e2g_t_init = self.get_l2g_transform(
            sd_rec
        )

        planning = []
        for _ in range(self.planning_steps):
            next_annotation_token = sd_rec["next"]
            if next_annotation_token == "":
                break
            sd_rec = self.nusc.get("sample", next_annotation_token)
            l2e_r_mat_curr, l2e_t_curr, e2g_r_mat_curr, e2g_t_curr = (
                self.get_l2g_transform(sd_rec)
            )  # (lidar to global at current frame)

            # bbox of sdc under current lidar frame
            next_bbox3d = self.generate_sdc_info(
                self.sdc_vel_info[next_annotation_token]
            )

            # to bbox under curr ego frame
            next_bbox3d.rotate(l2e_r_mat_curr.T)
            next_bbox3d.translate(l2e_t_curr)

            # to bbox under world frame
            next_bbox3d.rotate(e2g_r_mat_curr.T)
            next_bbox3d.translate(e2g_t_curr)

            # to bbox under initial ego frame, first inverse translate, then inverse rotate
            next_bbox3d.translate(-e2g_t_init)
            m1 = np.linalg.inv(e2g_r_mat_init)
            next_bbox3d.rotate(m1.T)

            # to bbox under curr ego frame, first inverse translate, then inverse rotate
            next_bbox3d.translate(-l2e_t_init)
            m2 = np.linalg.inv(l2e_r_mat_init)
            next_bbox3d.rotate(m2.T)

            planning.append(next_bbox3d)

        planning_all = np.zeros((1, self.planning_steps, 3))
        planning_mask_all = np.zeros((1, self.planning_steps, 2))
        n_valid_timestep = len(planning)
        if n_valid_timestep > 0:
            planning = [p.tensor.squeeze(0) for p in planning]
            planning = np.stack(planning, axis=0)  # (valid_t, 9)
            planning = planning[:, [0, 1, 6]]  # (x, y, yaw)
            planning_all[:, :n_valid_timestep, :] = planning
            planning_mask_all[:, :n_valid_timestep, :] = 1

        mask = planning_mask_all[0].any(axis=1)
        if mask.sum() == 0:
            command = 2  #'FORWARD'
        elif planning_all[0, mask][-1][0] >= 2:
            command = 0  #'RIGHT'
        elif planning_all[0, mask][-1][0] <= -2:
            command = 1  #'LEFT'
        else:
            command = 2  #'FORWARD'

        return planning_all, planning_mask_all, command


class NuScenesDataset(Dataset):
    NameMapping = {
        "movable_object.barrier": "barrier",
        "vehicle.bicycle": "bicycle",
        "vehicle.bus.bendy": "bus",
        "vehicle.bus.rigid": "bus",
        "vehicle.car": "car",
        "vehicle.construction": "construction_vehicle",
        "vehicle.motorcycle": "motorcycle",
        "human.pedestrian.adult": "pedestrian",
        "human.pedestrian.child": "pedestrian",
        "human.pedestrian.construction_worker": "pedestrian",
        "human.pedestrian.police_officer": "pedestrian",
        "movable_object.trafficcone": "traffic_cone",
        "vehicle.trailer": "trailer",
        "vehicle.truck": "truck",
    }

    CLASSES = (
        "car",
        "truck",
        "construction_vehicle",
        "bus",
        "trailer",
        "barrier",
        "motorcycle",
        "bicycle",
        "pedestrian",
        "traffic_cone",
    )

    def __init__(
        self,
        data_root=None,
        version=None,
        ann_file=None,
        test_scenes=None,
    ):
        ### Custom3DDataset

        if data_root is None:
            self.data_root = "./"
        elif data_root.endswith("/"):
            self.data_root = data_root
        else:
            self.data_root = data_root + "/"

        if ann_file is None:
            self.version = version or "v1.0-trainval"
            self.data_infos = self.nuscenes_data_prep(
                self.version, self.data_root, test_scenes=test_scenes
            )
        else:
            self.data_infos = self.load_annotations(ann_file)
            self.nusc = NuScenes(
                version=self.version, dataroot=self.data_root, verbose=True
            )

            # filter scenes if test_scenes is provided
            if test_scenes is not None:
                val_scenes = set()
                for s in self.nusc.scene:
                    if s["name"] not in test_scenes:
                        continue
                    val_scenes.add(s["token"])

                self.data_infos = [
                    s for s in self.data_infos if s["scene_token"] in val_scenes
                ]

        self.use_camera = True

        ### NuScenesE2EDataset
        self.traj_api = NuScenesTraj(self.nusc)

    def __len__(self):
        return len(self.data_infos)

    def nuscenes_data_prep(self, version, data_root, test_scenes=None, max_sweeps=10):
        self.nusc = NuScenes(version=version, dataroot=data_root, verbose=True)
        nusc_can_bus = NuScenesCanBus(dataroot=data_root)

        val_scenes = set()
        for s in self.nusc.scene:
            if test_scenes and s["name"] not in test_scenes:
                continue
            val_scenes.add(s["token"])

        samples = [x for x in self.nusc.sample if x["scene_token"] in val_scenes]

        val_nusc_infos = []
        frame_idx = 0
        for sample in tqdm.tqdm(samples, desc="Preparing data"):
            lidar_token = sample["data"]["LIDAR_TOP"]
            sd_rec = self.nusc.get("sample_data", sample["data"]["LIDAR_TOP"])
            cs_record = self.nusc.get(
                "calibrated_sensor", sd_rec["calibrated_sensor_token"]
            )
            pose_record = self.nusc.get("ego_pose", sd_rec["ego_pose_token"])
            lidar_path, boxes, _ = self.nusc.get_sample_data(lidar_token)

            can_bus = get_can_bus_info(self.nusc, nusc_can_bus, sample)
            info = {
                "token": sample["token"],
                "can_bus": can_bus,
                "sweeps": [],
                "cams": dict(),
                "scene_token": sample["scene_token"],  # temporal related info
                "lidar2ego_translation": cs_record["translation"],
                "lidar2ego_rotation": cs_record["rotation"],
                "ego2global_translation": pose_record["translation"],
                "ego2global_rotation": pose_record["rotation"],
                "timestamp": sample["timestamp"],
            }

            if sample["next"] == "":
                frame_idx = 0
            else:
                frame_idx += 1

            l2e_r = info["lidar2ego_rotation"]
            l2e_t = info["lidar2ego_translation"]
            e2g_r = info["ego2global_rotation"]
            e2g_t = info["ego2global_translation"]
            l2e_r_mat = Quaternion(l2e_r).rotation_matrix
            e2g_r_mat = Quaternion(e2g_r).rotation_matrix

            # obtain 6 image's information per frame
            camera_types = [
                "CAM_FRONT",
                "CAM_FRONT_RIGHT",
                "CAM_FRONT_LEFT",
                "CAM_BACK",
                "CAM_BACK_LEFT",
                "CAM_BACK_RIGHT",
            ]
            for cam in camera_types:
                cam_token = sample["data"][cam]
                cam_path, _, cam_intrinsic = self.nusc.get_sample_data(cam_token)
                cam_info = obtain_sensor2top(
                    self.nusc, cam_token, l2e_t, l2e_r_mat, e2g_t, e2g_r_mat, cam
                )
                cam_info.update(cam_intrinsic=cam_intrinsic)
                info["cams"].update({cam: cam_info})

            # obtain sweeps for a single key-frame
            sd_rec = self.nusc.get("sample_data", sample["data"]["LIDAR_TOP"])
            sweeps = []
            while len(sweeps) < max_sweeps:
                if not sd_rec["prev"] == "":
                    sweep = obtain_sensor2top(
                        self.nusc,
                        sd_rec["prev"],
                        l2e_t,
                        l2e_r_mat,
                        e2g_t,
                        e2g_r_mat,
                        "lidar",
                    )
                    sweeps.append(sweep)
                    sd_rec = self.nusc.get("sample_data", sd_rec["prev"])
                else:
                    break
            info["sweeps"] = sweeps
            # obtain annotation
            annotations = [
                self.nusc.get("sample_annotation", token) for token in sample["anns"]
            ]
            locs = np.array([b.center for b in boxes]).reshape(-1, 3)
            dims = np.array([b.wlh for b in boxes]).reshape(-1, 3)
            rots = np.array([b.orientation.yaw_pitch_roll[0] for b in boxes]).reshape(
                -1, 1
            )
            velocity = np.array(
                [self.nusc.box_velocity(token)[:2] for token in sample["anns"]]
            )
            valid_flag = np.array(
                [
                    (anno["num_lidar_pts"] + anno["num_radar_pts"]) > 0
                    for anno in annotations
                ],
                dtype=bool,
            ).reshape(-1)
            instance_inds = [
                self.nusc.getind("instance", ann["instance_token"])
                for ann in annotations
            ]
            future_traj_all, future_traj_valid_mask_all = get_future_traj_info(
                self.nusc, sample
            )
            instance_tokens = [
                ann["instance_token"] for ann in annotations
            ]  # dtype('<U[length_of_str]')

            # convert velo from global to lidar
            for i in range(len(boxes)):
                velo = np.array([*velocity[i], 0.0])
                velo = velo @ np.linalg.inv(e2g_r_mat).T @ np.linalg.inv(l2e_r_mat).T
                velocity[i] = velo[:2]

            names = [b.name for b in boxes]
            for i in range(len(names)):
                if names[i] in NuScenesDataset.NameMapping:
                    names[i] = NuScenesDataset.NameMapping[names[i]]
            names = np.array(names)
            # instance_inds = [nusc.getind('instance', ann['instance_token']) for ann in annotations]
            # TODO(box3d): convert gt_boxes to mmdet3d 1.0.0rc6 LiDARInstance3DBoxes format. [DONE]
            gt_boxes = np.concatenate([locs, dims, rots], axis=1)
            assert len(gt_boxes) == len(
                annotations
            ), f"{len(gt_boxes)}, {len(annotations)}"
            info["gt_boxes"] = gt_boxes
            info["gt_names"] = names
            info["gt_velocity"] = velocity.reshape(-1, 2)
            info["num_lidar_pts"] = np.array([a["num_lidar_pts"] for a in annotations])
            info["num_radar_pts"] = np.array([a["num_radar_pts"] for a in annotations])
            info["valid_flag"] = valid_flag
            info["gt_inds"] = np.array(instance_inds)
            info["gt_ins_tokens"] = np.array(instance_tokens)
            info["fut_traj"] = future_traj_all
            info["fut_traj_valid_mask"] = future_traj_valid_mask_all

            # add visibility_tokens
            visibility_tokens = [int(anno["visibility_token"]) for anno in annotations]
            info["visibility_tokens"] = np.array(visibility_tokens)

            val_nusc_infos.append(info)

        return val_nusc_infos

    def load_annotations(self, ann_file):
        """Load annotations from ann_file."""
        with open(ann_file, "rb") as f:
            value_buf = f.read()
            data = pickle.loads(value_buf)
        data_infos = list(sorted(data["infos"], key=lambda e: e["timestamp"]))
        self.version = data["metadata"]["version"]

        return data_infos

    def __getitem__(self, idx):
        input_dict = self.get_data_info(idx)
        data_dict = self.pipeline(input_dict)

        meta_keys = [
            "sample_idx",
            "img_shape",
            "lidar2img",
            "scene_token",
            "can_bus",
        ]
        img_metas = {}
        for key in meta_keys:
            if key in data_dict:
                img_metas[key] = data_dict[key]
        img_metas["lidar2img"] = np.array(img_metas["lidar2img"])
        img_metas["img_shape"] = np.array(img_metas["img_shape"])
        data_dict["img_metas"] = img_metas

        return data_dict

    def get_ann_info(self, index):
        info = self.data_infos[index]

        sdc_planning, sdc_planning_mask, command = self.traj_api.get_sdc_planning_label(
            info["token"]
        )
        anns_results = dict(
            sdc_planning=sdc_planning,
            sdc_planning_mask=sdc_planning_mask,
            command=command,
        )

        return anns_results

    def get_data_info(self, index):
        """Get data info according to the given index."""
        info = self.data_infos[index]

        input_dict = dict(
            sample_idx=info["token"],
            ego2global_translation=info["ego2global_translation"],
            ego2global_rotation=info["ego2global_rotation"],
            scene_token=info["scene_token"],
            can_bus=info["can_bus"],
            timestamp=info["timestamp"] / 1e6,
        )

        l2e_r = info["lidar2ego_rotation"]
        l2e_t = info["lidar2ego_translation"]
        e2g_r = info["ego2global_rotation"]
        e2g_t = info["ego2global_translation"]
        l2e_r_mat = Quaternion(l2e_r).rotation_matrix
        e2g_r_mat = Quaternion(e2g_r).rotation_matrix

        l2g_r_mat = l2e_r_mat.T @ e2g_r_mat.T
        l2g_t = l2e_t @ e2g_r_mat.T + e2g_t

        input_dict.update(
            dict(l2g_r_mat=l2g_r_mat.astype(np.float32), l2g_t=l2g_t.astype(np.float32))
        )

        # use_camera
        if self.use_camera:
            image_paths = []
            lidar2img_rts = []
            for cam_type, cam_info in info["cams"].items():
                image_paths.append(cam_info["data_path"])
                # obtain lidar to image transformation matrix
                lidar2cam_r = np.linalg.inv(cam_info["sensor2lidar_rotation"])
                lidar2cam_t = cam_info["sensor2lidar_translation"] @ lidar2cam_r.T
                lidar2cam_rt = np.eye(4)
                lidar2cam_rt[:3, :3] = lidar2cam_r.T
                lidar2cam_rt[3, :3] = -lidar2cam_t
                intrinsic = cam_info["cam_intrinsic"]
                viewpad = np.eye(4)
                viewpad[: intrinsic.shape[0], : intrinsic.shape[1]] = intrinsic
                lidar2img_rt = viewpad @ lidar2cam_rt.T
                lidar2img_rts.append(lidar2img_rt)
            input_dict["img_filename"] = image_paths
            input_dict["lidar2img"] = lidar2img_rts

        ann_info = self.get_ann_info(index)
        input_dict["sdc_planning"] = ann_info["sdc_planning"]
        input_dict["sdc_planning_mask"] = ann_info["sdc_planning_mask"]
        input_dict["command"] = ann_info["command"]

        rotation = Quaternion(input_dict["ego2global_rotation"])
        translation = input_dict["ego2global_translation"]
        can_bus = input_dict["can_bus"]
        can_bus[:3] = translation
        # NOTE(lty): fix can_bus format, in https://github.com/OpenDriveLab/UniAD/pull/214
        can_bus[3:7] = rotation.elements
        patch_angle = quaternion_yaw(rotation) / np.pi * 180
        if patch_angle < 0:
            patch_angle += 360
        can_bus[-2] = patch_angle / 180 * np.pi
        can_bus[-1] = patch_angle

        return input_dict

    def pipeline(self, data):
        results = data

        ### LoadMultiViewImageFromFilesInCeph

        filename = results["img_filename"]
        images_multiView = []
        for img_path in filename:
            img_path = os.path.join(self.data_root, img_path)
            img = cv2.imread(img_path)
            images_multiView.append(img)
        img = np.stack(
            images_multiView,
            axis=-1,
        )
        img = img.astype(np.float32)

        results["filename"] = filename
        results["img"] = ori_img = [img[..., i] for i in range(img.shape[-1])]

        ### NormalizeMultiviewImage

        def imnormalize(img, mean, std):
            mean = np.array(mean, dtype=np.float32)
            std = np.array(std, dtype=np.float32)
            img = (img - mean) / std
            return img

        mean = np.array([103.53, 116.28, 123.675])
        std = np.array([1, 1, 1])
        results["img"] = norm_img = [imnormalize(img, mean, std) for img in ori_img]

        ### PadMultiViewImage

        def impad_to_multiple(img, divisor, pad_val=0):
            pad_h = int(np.ceil(img.shape[0] / divisor)) * divisor
            pad_w = int(np.ceil(img.shape[1] / divisor)) * divisor
            # shape = (pad_h, pad_w)
            width = max(pad_w - img.shape[1], 0)
            height = max(pad_h - img.shape[0], 0)
            padding = (0, 0, width, height)

            img = cv2.copyMakeBorder(
                img,
                padding[1],
                padding[3],
                padding[0],
                padding[2],
                cv2.BORDER_CONSTANT,
                value=pad_val,
            )
            return img

        size_divisor = 32
        results["img"] = padded_img = [
            impad_to_multiple(img, size_divisor) for img in norm_img
        ]
        results["img_shape"] = [img.shape for img in padded_img]

        return results
