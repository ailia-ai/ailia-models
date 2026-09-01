from math import ceil
import random
import numpy as np
import cv2
import sys
from typing import Any
from tqdm import tqdm

import onnxruntime
from ani_portrait_utils import get_model_file_names
from lmk_extractor import LMKExtractor
from facemesh_v2_utils import matrix_to_euler_and_translation, smooth_pose_seq, crop_face, euler_and_translation_to_matrix
from scipy.interpolate import interp1d

sys.path.append("../../util")
from detector_utils import load_image
from arg_utils import get_base_parser, update_parser  # noqa: E402
from model_utils import check_and_download_models  # noqa: E402
from scheduling_ddim import DDIMScheduler
import ailia
from audio_processor import prepare_audio_feature

REMOTE_PATH = "https://storage.googleapis.com/ailia-models/AniPortrait/"
FACEMESH_REMOTE_PATH = "https://storage.googleapis.com/ailia-models/facemesh_v2"
MODES = ["Audio2Video", "Video2Video"]
INPUT_IMAGE = "lyl.png"
OUTPUT_IMAGE = ""
REF_IMAGE_SAMPLE = "lyl.png"
AUDIO_SAMPLE = "lyl.wav"
HEAD_POSE_SAMPLE = "pose_ref_video.mp4"

parser = get_base_parser("gpt2 text generation", INPUT_IMAGE, OUTPUT_IMAGE)
parser.add_argument(
    "--onnx",
    action="store_true",
    help="By default, the ailia SDK is used, but with this option, you can switch to using ONNX Runtime",
)
parser.add_argument("-r", "--reference_image", type=str, default=REF_IMAGE_SAMPLE)
parser.add_argument("-hp", "--head_pose_reference_video", type=str, default=None)
parser.add_argument("-a", "--audio", default=AUDIO_SAMPLE)
parser.add_argument("-v", "--source_video")
parser.add_argument("-s", "--steps", type=int, default=25)
parser.add_argument("-S", "--seed", type=int, default=42)
parser.add_argument("-vs", "--video_size", type=int, default=512)
parser.add_argument("-l", "--length", type=int, default=0)
parser.add_argument("-m", "--mode", choices=MODES)
# parser.add_argument("-p", "--prompt", help="prompt text", required=True, type=str)
args = update_parser(parser, check_input_type=False)


def get_head_pose(lmk_extractor: LMKExtractor, video_path):
    trans_mat = []
    cap = cv2.VideoCapture(video_path)
    total_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    fps = cap.get(cv2.CAP_PROP_FPS)
    while True:
        ret, img = cap.read()
        if not ret:
            break
        # img = load_image(args.input[0])
        # img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
        _trans_mat, _ = lmk_extractor(img)
        trans_mat.append(_trans_mat)

    cap.release()
    trans_mat = np.array(trans_mat)
    
    # Compute delta pose
    trans_mat_inv_frame_0 = np.linalg.inv(trans_mat[0])
    pose_arr = np.zeros([trans_mat.shape[0], 6])

    for i in range(pose_arr.shape[0]):
        pose_mat = trans_mat_inv_frame_0 @ trans_mat[i]
        euler_angles, translation_vector = matrix_to_euler_and_translation(pose_mat)
        pose_arr[i, :3] = euler_angles
        pose_arr[i, 3:6] = translation_vector

    new_fps = 30
    old_time = np.linspace(0, total_frames / fps, int(total_frames))
    new_time = np.linspace(0, total_frames / fps, int(total_frames * new_fps / fps))

    pose_arr_interp = np.zeros((len(new_time), 6))
    for i in range(6):
        interp_func = interp1d(old_time, pose_arr[:, i])
        pose_arr_interp[:, i] = interp_func(new_time)

    pose_arr_smooth = smooth_pose_seq(pose_arr_interp, window_size=5)
    return pose_arr_smooth


def draw_landmarks(image_size, keypoints, normed=False):
    ini_size = [512, 512]
    image = np.zeros([ini_size[1], ini_size[0], 3], dtype=np.uint8)
    for i in range(keypoints.shape[0]):
        x = int(keypoints[i, 0])
        y = int(keypoints[i, 1])
        cv2.circle(image, (x, y), 1, (0, 255, 0), -1)
    return image


def smooth_pose_seq(pose_seq, window_size):
    smoothed_pose_seq = np.zeros_like(pose_seq)
    
    for i in range(len(pose_seq)):
        start = max(0, i - window_size // 2)
        end = min(len(pose_seq), i + window_size // 2 + 1)
        smoothed_pose_seq[i] = np.mean(pose_seq[start:end], axis=0)

    return smoothed_pose_seq


def create_perspective_matrix(aspect_ratio):
    k_degrees_to_radians = np.pi / 180.0
    near = 1
    far = 10_000
    perspective_matrix = np.zeros(16, dtype=np.float32)

    f = 1.0 / np.tan(k_degrees_to_radians * 63 / 2.0)

    denom = 1.0 / (near - far)
    perspective_matrix[0] = f / aspect_ratio
    perspective_matrix[5] = f
    perspective_matrix[10] = (near + far) * denom
    perspective_matrix[11] = -1.0
    perspective_matrix[14] = 1.0 * far * near * denom

    perspective_matrix[5] *= -1.0
    return perspective_matrix


def project_points(points_3d, trans_mat, pose_vectors, image_shape):
    P = create_perspective_matrix(image_shape[1] / image_shape[0]).reshape(4, 4).T
    L, N, _ = points_3d.shape
    projected_points = np.zeros((L, N, 2))

    for i in range(L):
        points_3d_frame = points_3d[i]
        ones = np.ones((points_3d_frame.shape[0], 1))
        points_3d_homogeneous = np.hstack((points_3d_frame, ones))
        transformed_points = points_3d_homogeneous @ (trans_mat @ euler_and_traslation_to_matrix(pose_vectors[i][:3])).T @ P
        projected_points_frame = transformed_points[:, :2] / transformed_points[:, 3, np.newaxis]
        projected_points_frame[:, 0] = (projected_points_frame[:, 0] + 1) * 0.5 * image_shape[1]
        projected_points_frame[:, 1] = (projected_points_frame[:, 1] + 1) * 0.5 * image_shape[0]
        projected_points[i] = projected_points_frame

    return projected_points



def generate_from_image(nets: dict[str, Any]):
    if args.mode == "Audio2Video":
        lmk_extractor = LMKExtractor(
            nets["face_landmarks_detector"],
            nets["face_detector"],
            args.onnx,
        )
        ref_image = load_image(args.reference_image)
        ref_image = crop_face(ref_image, lmk_extractor)
        fps = 30
        cfg = 3.5
        length = 60
        fi_step = 3
        width = 512
        height = 512

        lmks3d, lmks = lmk_extractor(ref_image)
        # ref_pose = draw_landmarks((ref_image.shape[1], ref_image.shape[0]), lmks, normed=True)

        sample = prepare_audio_feature(args.audio, wav2vec_model_path="./pretrained_model/wav2vec2-base-960h")
        sample["audio_feature"] = sample["audio_feature"].astype(np.float32)
        sample["audio_feature"] = np.expand_dims(sample["audio_feature"], axis=0)

        # inference
        if args.onnx:
            pred = nets["a2m_model"].run(
                ["output"],
                {"input_value": sample["audio_feature"], "seq_len": [sample["seq_len"]]}
            )
            print(f"{pred=}")
        else:
            pred = nets["a2m_model"].predict(sample["audio_feature"])
        
        pred = pred.squeeze()
        pred = pred.reshape(pred.shape[0], -1, 3)
        pred = pred + lmks3d

        if args.head_pose_reference_video is not None:
            pose_seq = get_head_pose(lmk_extractor, args.head_pose_reference_video)
            mirrored_pose_seq = np.concatenate((pose_seq, pose_seq[-2:0:-1]), axis=0)
            pose_seq = np.tile(mirrored_pose_seq, (sample["seq_len"] // len(mirrored_pose_seq) + 1, 1))[:sample["seq_len"]]
        else:
            chunk_duration = 5
            sr = 16_000
            fps = 30
            chunk_size = sr * chunk_duration

            audio_chunks = []

            for i in range(ceil(sample["audio_feature"].shape[1] / chunk_size)):
                audio_chunks.append(
                    sample["audio_feature"][0, i * chunk_size:(i + 1) * chunk_size].reshape(1, -1)
                )

            seq_len_list = [chunk_duration * fps] * (len(audio_chunks) - 1) + [sample["seq_len"] % (chunk_duration * fps)]

            audio_chunks[-2] = np.concatenate((audio_chunks[-2], audio_chunks[-1]), axis=1)
            seq_len_list[-2] = seq_len_list[-2] + seq_len_list[-1]
            del audio_chunks[-1]
            del seq_len_list[-1]

            pose_seq = []

            for audio, seq_len in zip(audio_chunks, seq_len_list):
                print(f"{audio.shape=}")
                input(">>>")
                if args.onnx:
                    pose_seq_chunk = nets["a2p_model"].run(
                        ["output"],
                        {"input_value": audio, "seq_len": [seq_len], "id_seed": [random.randint(0, 99)]}
                    )
                    print(f"{pose_seq_chunk=}")
                else:
                    pose_seq_chunk = nets["a2p_model"].predict(audio)

                pose_seq_chunk = pose_seq_chunk.squeeze()
                pose_seq_chunk[:, :3] *= 0.5
                pose_seq.append(pose_seq_chunk)

            pose_seq = np.concatenate(pose_seq, 0)
            pose_seq = smooth_pose_seq(pose_seq, 7)

        projected_vertices = project_points(pred, trans_mat, pose_seq, (height, width))

        pose_images = []

        for i, verts in enumerate(projected_vertices):
            lmk_img = draw_landmarks(verts)
            pose_images.append(lmk_img)

        pose_list = []
        args_L = len(pose_images) if length == 0 or length > len(pose_images) else length
        for pose_image_np in pose_images[: args_L : fi_step]:
            pose_image_np = cv2.resize(pose_image_np, (width, height))
            pose_list.append(pose_image_np)

        for i, img in enumerate(pose_list, 1):
            cv2.imwrite(f"pose_{i}.png", img)

    else:
        pass


if __name__ == "__main__":
    model_file_names = get_model_file_names()

    nets = {}

    for root_model, _model_file_names in tqdm(model_file_names.items()):
        for model_name, model_files in tqdm(_model_file_names.items(), leave=False):
            # check_and_download_models(
            #     model_files["weight"],
            #     model_files["model"],
            #     REMOTE_PATH if root_model == "aniportrait" else FACEMESH_REMOTE_PATH,
            # )

            if args.onnx:
                net = onnxruntime.InferenceSession(model_files["weight"])
            else:
                net = ailia.Net(model_files["model"], model_files["weight"], env_id=args.env_id)

            nets[model_name] = net


    generate_from_image(nets)