import sys
import time

import numpy as np
import cv2

import ailia

# import original modules
from st_gcn_util import naive_pose_tracker, render_video, render_image
from st_gcn_labels import KINETICS_LABEL

sys.path.append('../../util')
from arg_utils import get_base_parser, update_parser  # noqa: E402
from arg_utils import check_file_existance  # noqa: E402
from model_utils import check_and_download_models  # noqa: E402
from webcamera_utils import get_capture, get_writer  # noqa: E402


# ======================
# Parameters
# ======================
WEIGHT_PATH = 'st_gcn.onnx'
MODEL_PATH = 'st_gcn.onnx.prototxt'
REMOTE_PATH = 'https://storage.googleapis.com/ailia-models/st_gcn/'

VIDEO_PATH = 'skateboarding.mp4'

POSE_KEY = [
    ailia.POSE_KEYPOINT_NOSE,
    ailia.POSE_KEYPOINT_SHOULDER_CENTER,
    ailia.POSE_KEYPOINT_SHOULDER_RIGHT,
    ailia.POSE_KEYPOINT_ELBOW_RIGHT,
    ailia.POSE_KEYPOINT_WRIST_RIGHT,
    ailia.POSE_KEYPOINT_SHOULDER_LEFT,
    ailia.POSE_KEYPOINT_ELBOW_LEFT,
    ailia.POSE_KEYPOINT_WRIST_LEFT,
    ailia.POSE_KEYPOINT_HIP_RIGHT,
    ailia.POSE_KEYPOINT_KNEE_RIGHT,
    ailia.POSE_KEYPOINT_ANKLE_RIGHT,
    ailia.POSE_KEYPOINT_HIP_LEFT,
    ailia.POSE_KEYPOINT_KNEE_LEFT,
    ailia.POSE_KEYPOINT_ANKLE_LEFT,
    ailia.POSE_KEYPOINT_EYE_RIGHT,
    ailia.POSE_KEYPOINT_EYE_LEFT,
    ailia.POSE_KEYPOINT_EAR_RIGHT,
    ailia.POSE_KEYPOINT_EAR_LEFT,
]

PYOPENPOSE_PATH = '/usr/local/python'

MODEL_LISTS = [
    'openpose',
    'pyopenpose',
    'lw_human_pose'
]

# ======================
# Arguemnt Parser Config
# ======================
parser = get_base_parser('ST-GCN model', VIDEO_PATH, None)

parser.add_argument(
    '--fps', default=30, type=int,
    help='FPS of video.'
)
parser.add_argument(
    '-a', '--arch', metavar='ARCH', default='openpose', choices=MODEL_LISTS,
    help='model lists: ' + ' | '.join(MODEL_LISTS)
)
parser.add_argument(
    '--img-save', action='store_true',
    help='Instead of show video, save image file.'
)
args = update_parser(parser)

if args.arch == "pyopenpose":
    sys.path.insert(0, PYOPENPOSE_PATH)
    try:
        from openpose import pyopenpose as op
    except ImportError:
        print('Can not find Openpose Python API.')
        sys.exit(-1)
    MODEL_POSE_PATH = 'pose/coco/pose_deploy_linevec.prototxt'
    WEIGHT_POSE_PATH = 'pose/coco/pose_iter_440000.caffemodel'
    REMOTE_POSE_PATH = 'http://posefs1.perception.cs.cmu.edu/OpenPose/models/pose/coco/'
elif args.arch == "lw_human_pose":
    POSE_ALGORITHM = ailia.POSE_ALGORITHM_LW_HUMAN_POSE
    MODEL_POSE_PATH = 'lightweight-human-pose-estimation.opt.onnx.prototxt'
    WEIGHT_POSE_PATH = 'lightweight-human-pose-estimation.opt.onnx'
    REMOTE_POSE_PATH = 'https://storage.googleapis.com/ailia-models/lightweight-human-pose-estimation/'
else:
    POSE_ALGORITHM_OPEN_POSE_SINGLE_SCALE = (12)
    # POSE_ALGORITHM = ailia.POSE_ALGORITHM_OPEN_POSE
    POSE_ALGORITHM = POSE_ALGORITHM_OPEN_POSE_SINGLE_SCALE
    MODEL_POSE_PATH = 'pose_deploy.prototxt'
    WEIGHT_POSE_PATH = 'pose_iter_440000.caffemodel'
    REMOTE_POSE_PATH = 'https://storage.googleapis.com/ailia-models/openpose/'


# ======================
# Secondaty Functions
# ======================
def pose_postprocess(pose_keypoints):
    pose_keypoints[:, :, 0:2] = pose_keypoints[:, :, 0:2] - 0.5
    pose_keypoints[:, :, 0][pose_keypoints[:, :, 2] == 0] = 0
    pose_keypoints[:, :, 1][pose_keypoints[:, :, 2] == 0] = 0
    return pose_keypoints


def postprocess(output, feature, num_person):
    intensity = (feature * feature).sum(axis=0) ** 0.5

    # get result
    # classification result of the full sequence
    voting_label = output.sum(axis=3). \
        sum(axis=2).sum(axis=1).argmax(axis=0)
    voting_label_name = KINETICS_LABEL[voting_label]

    # FIXME: latest_frame_label_name is never used!
    # classification result for each person of the latest frame
    # latest_frame_label = [
    #     output[:, :, :, m].sum(axis=2)[:, -1].argmax(axis=0)
    #     for m in range(num_person)
    # ]
    # latest_frame_label_name = [KINETICS_LABEL[l] for l in latest_frame_label]

    _, num_frame, _, num_person = output.shape
    video_label_name = list()
    for t in range(num_frame):
        frame_label_name = list()
        for m in range(num_person):
            person_label = output[:, t, :, m].sum(axis=1).argmax(axis=0)
            person_label_name = KINETICS_LABEL[person_label]
            frame_label_name.append(person_label_name)
        video_label_name.append(frame_label_name)

    return voting_label_name, video_label_name, output, intensity


# ======================
# Main functions
# ======================
def recognize_offline(input, pose, net):
    input = input[0]
    capture = cv2.VideoCapture(input)
    video_length = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    pose_tracker = naive_pose_tracker(data_frame=video_length)

    # pose estimation
    frame_index = 0
    frames = list()
    while True:
        ret, frame = capture.read()
        if frame is None:
            break
        if video_length <= frame_index:
            break

        source_H, source_W, _ = frame.shape
        img = cv2.resize(
            frame, (256 * source_W // source_H, 256))
        frames.append(img)
        H, W, _ = img.shape

        # pose estimate
        if args.arch == "pyopenpose":
            datum = op.Datum()
            datum.cvInputData = img
            pose.emplaceAndPop([datum])
            pose_keypoints = datum.poseKeypoints  # (num_person, num_joint, 3)
            if len(pose_keypoints.shape) != 3:
                continue

            # normalization
            pose_keypoints[:, :, 0] = pose_keypoints[:, :, 0] / W
            pose_keypoints[:, :, 1] = pose_keypoints[:, :, 1] / H
        else:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2BGRA)
            pose.compute(img)
            count = pose.get_object_count()

            pose_keypoints = np.zeros((count, 18, 3))
            # pose_keypoints.shape : (num_person, num_joint, 3)
            for idx in range(count):
                person = pose.get_object_pose(idx)
                for i, key in enumerate(POSE_KEY):
                    p = person.points[key]
                    pose_keypoints[idx, i, :] = [p.x, p.y, p.score]

        pose_keypoints = pose_postprocess(pose_keypoints)
        pose_tracker.update(pose_keypoints, frame_index)
        frame_index += 1
        print('Pose estimation ({}/{}).'.format(frame_index, video_length))

    # action recognition
    data = pose_tracker.get_skeleton_sequence()
    input_data = np.expand_dims(data, 0)
    net.set_input_shape(input_data.shape)
    outputs = net.predict({
        'data': input_data
    })
    output, feature = outputs
    output = output[0]
    feature = feature[0]

    # classification result for each person of the latest frame
    _, _, _, num_person = data.shape
    out = postprocess(output, feature, num_person)
    voting_label_name, video_label_name, output, intensity = out
    return data, voting_label_name, video_label_name, output, intensity, frames


def recognize_from_file(input, pose, net):
    # inference
    print('Start inference...')
    if args.benchmark:
        print('BENCHMARK mode')
        for i in range(5):
            start = int(round(time.time() * 1000))
            result = recognize_offline(input, pose, net)
            end = int(round(time.time() * 1000))
            print(f'\tailia processing time {end - start} ms')
    else:
        result = recognize_offline(input, pose, net)

    print('Script finished successfully.')

    # render the video
    data, voting_label_name, video_label_name, output, intensity, frames = result
    images = render_video(
        data, voting_label_name,
        video_label_name, intensity, frames)

    # visualize or save
    writer = None
    for i, image in enumerate(images):
        image = image.astype(np.uint8)

        # init writer
        if (writer is None) and (args.savepath is not None):
            shape = image.shape
            writer = get_writer(args.savepath, image.shape[0], image.shape[1])

        if writer is not None:
            writer.write(image)
        elif args.img_save:
            cv2.imwrite("output/ST-GCN-%08d.png" % i, image)
        else:
            cv2.imshow("ST-GCN", image)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            if cv2.getWindowProperty('ST-GCN', cv2.WND_PROP_VISIBLE) == 0:
                break

    if writer is not None:
        writer.release()


def recognize_realtime(video, pose, net):
    capture = get_capture(args.video)

    pose_tracker = naive_pose_tracker()

    # start recognition
    start_time = time.time()
    frame_index = 0
    frame_shown = False
    while True:
        tic = time.time()

        ret, frame = capture.read()
        if cv2.waitKey(1) & 0xFF == ord('q') or not ret:
            break
        if frame_shown and cv2.getWindowProperty('ST-GCN', cv2.WND_PROP_VISIBLE) == 0:
            break

        source_H, source_W, _ = frame.shape
        img = cv2.resize(
            frame, (256 * source_W // source_H, 256))
        H, W, _ = img.shape

        # pose estimate
        if args.arch == "pyopenpose":
            datum = op.Datum()
            datum.cvInputData = img
            pose.emplaceAndPop([datum])
            pose_keypoints = datum.poseKeypoints  # (num_person, num_joint, 3)
            if len(pose_keypoints.shape) != 3:
                continue

            # normalization
            pose_keypoints[:, :, 0] = pose_keypoints[:, :, 0] / W
            pose_keypoints[:, :, 1] = pose_keypoints[:, :, 1] / H
        else:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2BGRA)
            pose.compute(img)
            count = pose.get_object_count()
            if count == 0:
                continue

            pose_keypoints = np.zeros((count, 18, 3))
            # pose_keypoints.shape : (num_person, num_joint, 3)
            for idx in range(count):
                person = pose.get_object_pose(idx)
                for i, key in enumerate(POSE_KEY):
                    p = person.points[key]
                    pose_keypoints[idx, i, :] = [p.x, p.y, p.score]

        # pose tracking
        if video == '0':
            frame_index = int((time.time() - start_time) * args.fps)
        else:
            frame_index += 1
        pose_keypoints = pose_postprocess(pose_keypoints)
        pose_tracker.update(pose_keypoints, frame_index)

        # action recognition
        data = pose_tracker.get_skeleton_sequence()
        input_data = np.expand_dims(data, 0)
        net.set_input_shape(input_data.shape)
        outputs = net.predict({
            'data': input_data
        })
        output, feature = outputs
        output = output[0]
        feature = feature[0]

        # classification result for each person of the latest frame
        _, _, _, num_person = data.shape
        out = postprocess(output, feature, num_person)
        voting_label_name, video_label_name, output, intensity = out

        # visualization
        app_fps = 1 / (time.time() - tic)
        image = render_image(
            data, voting_label_name,
            video_label_name, intensity, frame, app_fps)

        # show
        if args.img_save:
            cv2.imwrite("output/ST-GCN-%08d.png" % frame_index, image)
        else:
            cv2.imshow('ST-GCN', image)
            frame_shown = True

    capture.release()
    cv2.destroyAllWindows()
    print('Script finished successfully.')


def main():
    # model files check and download
    print("=== ST-GCN model ===")
    check_and_download_models(WEIGHT_PATH, MODEL_PATH, REMOTE_PATH)
    print("=== OpenPose model ===")
    check_and_download_models(
        WEIGHT_POSE_PATH, MODEL_POSE_PATH, REMOTE_POSE_PATH
    )

    # net initialize
    net = ailia.Net(MODEL_PATH, WEIGHT_PATH, env_id=args.env_id, memory_mode=args.memory_mode)

    if args.arch == "pyopenpose":
        pose = op.WrapperPython()
        params = dict(model_folder='.', model_pose='COCO')
        pose.configure(params)
        pose.start()
    else:
        pose = ailia.PoseEstimator(
            MODEL_POSE_PATH,
            WEIGHT_POSE_PATH,
            env_id=args.env_id,
            algorithm=POSE_ALGORITHM,
            memory_mode=args.memory_mode
        )
        if args.arch == "openpose":
            pose.set_threshold(0.1)

    if args.video is not None:
        # realtime mode
        recognize_realtime(args.video, pose, net)
    else:
        # offline mode
        recognize_from_file(args.input, pose, net)


if __name__ == '__main__':
    main()
