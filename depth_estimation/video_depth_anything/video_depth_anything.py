import sys
from logging import getLogger

import ailia
import cv2
import numpy as np
from tqdm import tqdm

# import original modules
sys.path.append("../../util")
from arg_utils import get_base_parser, update_parser  # noqa
from model_utils import check_and_download_models  # noqa

logger = getLogger(__name__)

# ======================
# Parameters
# ======================

WEIGHT_PATH = "video_depth_anything_vits.onnx"
MODEL_PATH = "video_depth_anything_vits.onnx.prototxt"
REMOTE_PATH = "https://storage.googleapis.com/ailia-models/video_depth_anything/"

VIDEO_PATH = "davis_rollercoaster.mp4"
SAVE_PATH = "output.mp4"

INFER_LEN = 32
OVERLAP = 10
KEYFRAMES = [0, 12, 24, 25, 26, 27, 28, 29, 30, 31]
INTERP_LEN = 8

MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

# ======================
# Argument Parser Config
# ======================

parser = get_base_parser("Video Depth Anything", VIDEO_PATH, SAVE_PATH)
parser.add_argument("--onnx", action="store_true", help="execute onnxruntime version.")
args = update_parser(parser)


# ======================
# Secondary Functions
# ======================


def read_video_frames(video_path, process_length, target_fps=-1, max_res=-1):
    cap = cv2.VideoCapture(video_path)
    original_fps = cap.get(cv2.CAP_PROP_FPS)
    original_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    original_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))

    if max_res > 0 and max(original_height, original_width) > max_res:
        scale = max_res / max(original_height, original_width)
        height = round(original_height * scale)
        width = round(original_width * scale)

    fps = original_fps if target_fps < 0 else target_fps
    stride = max(round(original_fps / fps), 1)

    frames = []
    frame_count = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret or (process_length > 0 and frame_count >= process_length):
            break
        if frame_count % stride == 0:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            if max_res > 0 and max(original_height, original_width) > max_res:
                frame = cv2.resize(frame, (width, height))
            frames.append(frame)
        frame_count += 1
    cap.release()
    frames = np.stack(frames, axis=0)

    return frames, fps


def constrain_to_multiple_of(x, min_val, multiple_of=14):
    y = (np.round(x / multiple_of) * multiple_of).astype(int)
    if y < min_val:
        y = (np.ceil(x / multiple_of) * multiple_of).astype(int)
    return y


def get_resize_shape(height, width, input_size):
    # keep the aspect ratio so that the shorter side becomes at least input_size
    scale_height = input_size / height
    scale_width = input_size / width
    if scale_width > scale_height:
        scale_height = scale_width
    else:
        scale_width = scale_height

    new_height = constrain_to_multiple_of(scale_height * height, min_val=input_size)
    new_width = constrain_to_multiple_of(scale_width * width, min_val=input_size)
    return new_height, new_width


def preprocess(frame, resize_shape):
    new_height, new_width = resize_shape
    img = frame.astype(np.float32) / 255.0
    img = cv2.resize(img, (new_width, new_height), interpolation=cv2.INTER_CUBIC)
    img = (img - MEAN) / STD
    img = img.transpose(2, 0, 1)
    img = np.ascontiguousarray(img).astype(np.float32)
    return img


def resize_bilinear_align_corners(x, height, width):
    """Bilinear resize of (N, h, w) maps whose corner pixels are aligned."""

    def coords(in_size, out_size):
        scale = (in_size - 1) / (out_size - 1) if out_size > 1 else 0.0
        src = np.arange(out_size, dtype=np.float32) * np.float32(scale)
        i0 = np.minimum(np.floor(src).astype(int), in_size - 1)
        i1 = np.minimum(i0 + 1, in_size - 1)
        w1 = (src - i0).astype(np.float32)
        return i0, i1, w1

    y0, y1, wy = coords(x.shape[1], height)
    x0, x1, wx = coords(x.shape[2], width)
    rows = x[:, y0, :] * (1 - wy)[None, :, None] + x[:, y1, :] * wy[None, :, None]
    out = rows[:, :, x0] * (1 - wx)[None, None, :] + rows[:, :, x1] * wx[None, None, :]
    return out.astype(np.float32)


def compute_scale_and_shift(prediction, target, mask):
    prediction = prediction.astype(np.float32)
    target = target.astype(np.float32)
    mask = mask.astype(np.float32)

    a_00 = np.sum(mask * prediction * prediction)
    a_01 = np.sum(mask * prediction)
    a_11 = np.sum(mask)

    b_0 = np.sum(mask * prediction * target)
    b_1 = np.sum(mask * target)

    x_0 = 1
    x_1 = 0

    det = a_00 * a_11 - a_01 * a_01
    if det != 0:
        x_0 = (a_11 * b_0 - a_01 * b_1) / det
        x_1 = (-a_01 * b_0 + a_00 * b_1) / det

    return x_0, x_1


# ======================
# Main functions
# ======================


def predict(net, x):
    if not args.onnx:
        output = net.predict([x])
    else:
        output = net.run(None, {"image": x})
    (depth,) = output
    return depth[0]


def infer_video_depth(net, frames):
    input_size = 518

    frame_height, frame_width = frames[0].shape[:2]
    ratio = max(frame_height, frame_width) / min(frame_height, frame_width)
    # reduce the input size of extremely wide videos to save memory
    if ratio > 1.78:
        input_size = int(input_size * 1.777 / ratio)
        input_size = round(input_size / 14) * 14
    resize_shape = get_resize_shape(frame_height, frame_width, input_size)

    frame_list = [frames[i] for i in range(frames.shape[0])]
    frame_step = INFER_LEN - OVERLAP
    org_video_len = len(frame_list)
    append_frame_len = (frame_step - (org_video_len % frame_step)) % frame_step + (
        INFER_LEN - frame_step
    )
    frame_list = frame_list + [frame_list[-1].copy()] * append_frame_len

    # infer each window of INFER_LEN frames, reusing the keyframes of the
    # previous window in the first OVERLAP slots
    depth_list = []
    pre_input = None
    for frame_id in tqdm(range(0, org_video_len, frame_step)):
        cur_input = np.stack(
            [
                preprocess(frame_list[frame_id + i], resize_shape)
                for i in range(INFER_LEN)
            ]
        )[None]
        if pre_input is not None:
            cur_input[:, :OVERLAP] = pre_input[:, KEYFRAMES]

        depth = predict(net, cur_input)
        depth = resize_bilinear_align_corners(depth, frame_height, frame_width)
        depth_list += [depth[i] for i in range(depth.shape[0])]

        pre_input = cur_input

    # align each window to the previous one on the overlapping keyframes and
    # blend the overlapping frames
    depth_list_aligned = []
    ref_align = []
    align_len = OVERLAP - INTERP_LEN
    kf_align_list = KEYFRAMES[:align_len]

    for frame_id in range(0, len(depth_list), INFER_LEN):
        if len(depth_list_aligned) == 0:
            depth_list_aligned += depth_list[:INFER_LEN]
            for kf_id in kf_align_list:
                ref_align.append(depth_list[frame_id + kf_id])
        else:
            curr_align = []
            for i in range(len(kf_align_list)):
                curr_align.append(depth_list[frame_id + i])

            scale, shift = compute_scale_and_shift(
                np.concatenate(curr_align),
                np.concatenate(ref_align),
                np.concatenate(np.ones_like(ref_align) == 1),
            )


def recognize_from_video(net):
    for video_path in args.input:
        logger.info(video_path)
        frames, fps = read_video_frames(video_path, -1, -1, 1280)
        logger.info(
            f"frames: {frames.shape[0]}, size: {frames.shape[1:3]}, fps: {fps:.2f}"
        )

        logger.info("Start inference...")
        depths = infer_video_depth(net, frames)


def main():
    check_and_download_models(WEIGHT_PATH, MODEL_PATH, REMOTE_PATH)

    if not args.onnx:
        net = ailia.Net(MODEL_PATH, WEIGHT_PATH, env_id=args.env_id)
    else:
        import onnxruntime

        cuda = 0 < ailia.get_gpu_environment_id()
        providers = (
            ["CUDAExecutionProvider", "CPUExecutionProvider"]
            if cuda
            else ["CPUExecutionProvider"]
        )
        net = onnxruntime.InferenceSession(WEIGHT_PATH, providers=providers)

    recognize_from_video(net)


if __name__ == "__main__":
    main()
