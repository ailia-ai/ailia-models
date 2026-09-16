import math
import pathlib
import sys
import time

import ailia
import cv2
import h5py
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D

sys.path.append('../../util')
# logger
from logging import getLogger  # noqa: E402

import webcamera_utils  # noqa: E402
from image_utils import imread, load_image  # noqa: E402
from model_utils import check_and_download_models  # noqa: E402
from arg_utils import get_base_parser, get_savepath, update_parser  # noqa: E402

logger = getLogger(__name__)


# ======================
# Parameters 1
# ======================
IMAGE_PATH = 'girl-5204299_640.jpg'
SAVE_IMAGE_PATH = 'output.png'
IMAGE_HEIGHT = 240
IMAGE_WIDTH = 320

ALGORITHM = ailia.POSE_ALGORITHM_LW_HUMAN_POSE


# ======================
# Arguemnt Parser Config
# ======================
parser = get_base_parser(
    'Fast and accurate human pose 2D-estimation.', IMAGE_PATH, SAVE_IMAGE_PATH,
)
parser.add_argument(
    '-n', '--normal',
    action='store_true',
    help='By default, the optimized model is used, but with this option, ' +
    'you can switch to the normal (not optimized) model'
)
parser.add_argument(
    '--skip_plot',
    action='store_true',
    help='Skip showing 3D plot of result (for CUI)'
)
args = update_parser(parser)


# ======================
# Parameters 2
# ======================
MODEL_NAME = 'lightweight-human-pose-estimation'
if args.normal:
    WEIGHT_PATH = f'{MODEL_NAME}.onnx'
    MODEL_PATH = f'{MODEL_NAME}.onnx.prototxt'
else:
    WEIGHT_PATH = f'{MODEL_NAME}.opt.onnx'
    MODEL_PATH = f'{MODEL_NAME}.opt.onnx.prototxt'
REMOTE_PATH = f'https://storage.googleapis.com/ailia-models/{MODEL_NAME}/'

BASELINE_MODEL_NAME = '3d-pose-baseline'
BASELINE_WEIGHT_PATH = f'{BASELINE_MODEL_NAME}.onnx'
BASELINE_MODEL_PATH = f'{BASELINE_MODEL_NAME}.onnx.prototxt'
BASELINE_REMOTE_PATH = \
    f'https://storage.googleapis.com/ailia-models/{BASELINE_MODEL_NAME}/'


# ======================
# 3d-pose Utils
# ======================

with h5py.File('3d-pose-baseline-mean.h5', 'r') as f:
    data_mean_2d = np.array(f['data_mean_2d'])
    data_std_2d = np.array(f['data_std_2d'])
    data_mean_3d = np.array(f['data_mean_3d'])
    data_std_3d = np.array(f['data_std_3d'])

H36M_NAMES = ['']*32
H36M_NAMES[0] = 'Hip'  # ignore when 3d
H36M_NAMES[1] = 'RHip'
H36M_NAMES[2] = 'RKnee'
H36M_NAMES[3] = 'RFoot'
H36M_NAMES[6] = 'LHip'
H36M_NAMES[7] = 'LKnee'
H36M_NAMES[8] = 'LFoot'
H36M_NAMES[12] = 'Spine'
H36M_NAMES[13] = 'Thorax'
H36M_NAMES[14] = 'Neck/Nose'  # ignore when 2d
H36M_NAMES[15] = 'Head'
H36M_NAMES[17] = 'LShoulder'
H36M_NAMES[18] = 'LElbow'
H36M_NAMES[19] = 'LWrist'
H36M_NAMES[25] = 'RShoulder'
H36M_NAMES[26] = 'RElbow'
H36M_NAMES[27] = 'RWrist'

h36m_2d_mean = [0, 1, 2, 3, 6, 7, 8, 12, 13, 15, 17, 18, 19, 25, 26, 27]
h36m_3d_mean = [1, 2, 3, 6, 7, 8, 12, 13, 14, 15, 17, 18, 19, 25, 26, 27]

OPENPOSE_Nose = 0
OPENPOSE_Neck = 1
OPENPOSE_RightShoulder = 2
OPENPOSE_RightElbow = 3
OPENPOSE_RightWrist = 4
OPENPOSE_LeftShoulder = 5
OPENPOSE_LeftElbow = 6
OPENPOSE_LeftWrist = 7
OPENPOSE_RightHip = 8
OPENPOSE_RightKnee = 9
OPENPOSE_RightAnkle = 10
OPENPOSE_LeftHip = 11
OPENPOSE_LeftKnee = 12
OPENPOSE_LAnkle = 13
OPENPOSE_RightEye = 14
OPENPOSE_LeftEye = 15
OPENPOSE_RightEar = 16
OPENPOSE_LeftEar = 17
OPENPOSE_Background = 18

openpose_to_3dposebaseline = [
    -1, 8, 9, 10, 11, 12, 13, -1, 1, 0, 5, 6, 7, 2, 3, 4
]

fig = plt.figure()
ax = Axes3D(fig)
ax.view_init(18, -70)


def search_name(name, IS_3D):
    j = 0
    for i in range(32):
        if(IS_3D):
            if(H36M_NAMES[i] == "Hip"):
                continue
        else:
            if(H36M_NAMES[i] == "Neck/Nose"):
                continue
        if(H36M_NAMES[i] == ""):
            continue
        if(H36M_NAMES[i] == name):
            return j
        j = j+1
    return -1


def draw_connect(from_id, to_id, color, X, Y, Z, IS_3D):
    from_id = search_name(from_id, IS_3D)
    to_id = search_name(to_id, IS_3D)
    if(from_id == -1 or to_id == -1):
        return
    x = [X[from_id], X[to_id]]
    y = [Y[from_id], Y[to_id]]
    z = [Z[from_id], Z[to_id]]

    ax.plot(x, z, y, "o-", color=color, ms=4, mew=0.5)


def plot(outputs, inputs):
    plt.cla()

    ax.set_xlabel('X axis')
    ax.set_ylabel('Z axis')
    ax.set_zlabel('Y axis')
    ax.set_zlim([600, -600])

    for mode in range(2):
        X = []
        Y = []
        Z = []

        if mode == 0:
            IS_3D = True
        else:
            IS_3D = False

        for i in range(16):
            if IS_3D:
                X.append(outputs[i*3+0])
                Y.append(outputs[i*3+1])
                Z.append(outputs[i*3+2])
            else:
                j = h36m_2d_mean[i]
                X.append(inputs[i*2+0]*data_std_2d[j*2+0]+data_mean_2d[j*2+0])
                Y.append(inputs[i*2+1]*data_std_2d[j*2+1]+data_mean_2d[j*2+1])
                Z.append(0)

        if(IS_3D):
            draw_connect("Head", "Thorax", "#0000aa", X, Y, Z, IS_3D)
            draw_connect("Thorax", 'RShoulder', "#00ff00", X, Y, Z, IS_3D)
            draw_connect('RShoulder', 'RElbow', "#00ff00", X, Y, Z, IS_3D)
            draw_connect('RElbow', 'RWrist', "#00ff00", X, Y, Z, IS_3D)
            draw_connect("Thorax", 'LShoulder', "#00ff00", X, Y, Z, IS_3D)
            draw_connect('LShoulder', 'LElbow', "#00ff00", X, Y, Z, IS_3D)
            draw_connect('LElbow', 'LWrist', "#00ff00", X, Y, Z, IS_3D)
            draw_connect('Thorax', 'Spine', "#00ff00", X, Y, Z, IS_3D)
            draw_connect('Spine', 'LHip', "#00ff00", X, Y, Z, IS_3D)
            draw_connect('Spine', 'RHip', "#00ff00", X, Y, Z, IS_3D)
            draw_connect('RHip', 'RKnee', "#ff0000", X, Y, Z, IS_3D)
            draw_connect('RKnee', 'RFoot', "#ff0000", X, Y, Z, IS_3D)
            draw_connect('LHip', 'LKnee', "#ff0000", X, Y, Z, IS_3D)
            draw_connect('LKnee', 'LFoot', "#ff0000", X, Y, Z, IS_3D)
        else:
            draw_connect("Head", "Thorax", "#0000ff", X, Y, Z, IS_3D)
            draw_connect("Thorax", 'RShoulder', "#00ff00", X, Y, Z, IS_3D)
            draw_connect('RShoulder', 'RElbow', "#00ff00", X, Y, Z, IS_3D)
            draw_connect('RElbow', 'RWrist', "#00ff00", X, Y, Z, IS_3D)
            draw_connect("Thorax", 'LShoulder', "#00ff00", X, Y, Z, IS_3D)
            draw_connect('LShoulder', 'LElbow', "#00ff00", X, Y, Z, IS_3D)
            draw_connect('LElbow', 'LWrist', "#00ff00", X, Y, Z, IS_3D)
            draw_connect('Thorax', 'Spine', "#00ff00", X, Y, Z, IS_3D)
            draw_connect('Spine', 'Hip', "#00ff00", X, Y, Z, IS_3D)
            draw_connect('Hip', 'LHip', "#ff0000", X, Y, Z, IS_3D)
            draw_connect('Hip', 'RHip', "#ff0000", X, Y, Z, IS_3D)
            draw_connect('RHip', 'RKnee', "#ff0000", X, Y, Z, IS_3D)
            draw_connect('RKnee', 'RFoot', "#ff0000", X, Y, Z, IS_3D)
            draw_connect('LHip', 'LKnee', "#ff0000", X, Y, Z, IS_3D)
            draw_connect('LKnee', 'LFoot', "#ff0000", X, Y, Z, IS_3D)


def display_3d_pose(points, baseline):
    inputs = np.zeros(32)

    for i in range(16):
        if openpose_to_3dposebaseline[i] == -1:
            continue
        inputs[i*2+0] = points[openpose_to_3dposebaseline[i]*2+0]
        inputs[i*2+1] = points[openpose_to_3dposebaseline[i]*2+1]

    inputs[0*2+0] = (points[11*2+0]+points[8*2+0])/2
    inputs[0*2+1] = (points[11*2+1]+points[8*2+1])/2
    inputs[7*2+0] = (points[5*2+0]+points[2*2+0])/2
    inputs[7*2+1] = (points[5*2+1]+points[2*2+1])/2

    # spine_x = inputs[24]
    # spine_y = inputs[25]

    for i in range(16):
        j = h36m_2d_mean[i]
        inputs[i*2+0] = (inputs[i*2+0]-data_mean_2d[j*2+0])/data_std_2d[j*2+0]
        inputs[i*2+1] = (inputs[i*2+1]-data_mean_2d[j*2+1])/data_std_2d[j*2+1]

    reshape_input = np.reshape(np.array(inputs), (1, 32))

    outputs = baseline.predict(reshape_input)[0]

    for i in range(16):
        j = h36m_3d_mean[i]
        outputs[i*3+0] = outputs[i*3+0]*data_std_3d[j*3+0]+data_mean_3d[j*3+0]
        outputs[i*3+1] = outputs[i*3+1]*data_std_3d[j*3+1]+data_mean_3d[j*3+1]
        outputs[i*3+2] = outputs[i*3+2]*data_std_3d[j*3+2]+data_mean_3d[j*3+2]

    for i in range(16):
        dx = outputs[i*3+0] - data_mean_3d[0*3+0]
        dy = outputs[i*3+1] - data_mean_3d[0*3+1]
        dz = outputs[i*3+2] - data_mean_3d[0*3+2]

        # Camera tilt of teacher data (average 13 degrees downward)
        theta = math.radians(13)
        outputs[i*3+0] = dx
        outputs[i*3+1] = dy*math.cos(theta) + dz*math.sin(theta)
        outputs[i*3+2] = -dy*math.sin(theta) + dz*math.cos(theta)

    plot(outputs, inputs)


# ======================
# 2d-pose Utils
# ======================
def hsv_to_rgb(h, s, v):
    bgr = cv2.cvtColor(
        np.array([[[h, s, v]]], dtype=np.uint8), cv2.COLOR_HSV2BGR
    )[0][0]
    return (int(bgr[2]), int(bgr[1]), int(bgr[0]))


def line(input_img, person, point1, point2):
    threshold = 0.2
    if person.points[point1].score > threshold and\
       person.points[point2].score > threshold:
        color = hsv_to_rgb(255*point1/ailia.POSE_KEYPOINT_CNT, 255, 255)

        x1 = int(input_img.shape[1] * person.points[point1].x)
        y1 = int(input_img.shape[0] * person.points[point1].y)
        x2 = int(input_img.shape[1] * person.points[point2].x)
        y2 = int(input_img.shape[0] * person.points[point2].y)
        cv2.line(input_img, (x1, y1), (x2, y2), color, 5)


def display_result(input_img, pose, baseline):
    count = pose.get_object_count()
    if count >= 1:
        count = 1

    for idx in range(count):
        person = pose.get_object_pose(idx)

        line(input_img, person, ailia.POSE_KEYPOINT_NOSE,
             ailia.POSE_KEYPOINT_SHOULDER_CENTER)
        line(input_img, person, ailia.POSE_KEYPOINT_SHOULDER_LEFT,
             ailia.POSE_KEYPOINT_SHOULDER_CENTER)
        line(input_img, person, ailia.POSE_KEYPOINT_SHOULDER_RIGHT,
             ailia.POSE_KEYPOINT_SHOULDER_CENTER)

        line(input_img, person, ailia.POSE_KEYPOINT_EYE_LEFT,
             ailia.POSE_KEYPOINT_NOSE)
        line(input_img, person, ailia.POSE_KEYPOINT_EYE_RIGHT,
             ailia.POSE_KEYPOINT_NOSE)
        line(input_img, person, ailia.POSE_KEYPOINT_EAR_LEFT,
             ailia.POSE_KEYPOINT_EYE_LEFT)
        line(input_img, person, ailia.POSE_KEYPOINT_EAR_RIGHT,
             ailia.POSE_KEYPOINT_EYE_RIGHT)

        line(input_img, person, ailia.POSE_KEYPOINT_ELBOW_LEFT,
             ailia.POSE_KEYPOINT_SHOULDER_LEFT)
        line(input_img, person, ailia.POSE_KEYPOINT_ELBOW_RIGHT,
             ailia.POSE_KEYPOINT_SHOULDER_RIGHT)
        line(input_img, person, ailia.POSE_KEYPOINT_WRIST_LEFT,
             ailia.POSE_KEYPOINT_ELBOW_LEFT)
        line(input_img, person, ailia.POSE_KEYPOINT_WRIST_RIGHT,
             ailia.POSE_KEYPOINT_ELBOW_RIGHT)

        line(input_img, person, ailia.POSE_KEYPOINT_BODY_CENTER,
             ailia.POSE_KEYPOINT_SHOULDER_CENTER)
        line(input_img, person, ailia.POSE_KEYPOINT_HIP_LEFT,
             ailia.POSE_KEYPOINT_BODY_CENTER)
        line(input_img, person, ailia.POSE_KEYPOINT_HIP_RIGHT,
             ailia.POSE_KEYPOINT_BODY_CENTER)

        line(input_img, person, ailia.POSE_KEYPOINT_KNEE_LEFT,
             ailia.POSE_KEYPOINT_HIP_LEFT)
        line(input_img, person, ailia.POSE_KEYPOINT_ANKLE_LEFT,
             ailia.POSE_KEYPOINT_KNEE_LEFT)
        line(input_img, person, ailia.POSE_KEYPOINT_KNEE_RIGHT,
             ailia.POSE_KEYPOINT_HIP_RIGHT)
        line(input_img, person, ailia.POSE_KEYPOINT_ANKLE_RIGHT,
             ailia.POSE_KEYPOINT_KNEE_RIGHT)

        points = []
        # OPENPOSE_Nose
        points.append(person.points[ailia.POSE_KEYPOINT_NOSE].x)
        points.append(person.points[ailia.POSE_KEYPOINT_NOSE].y)
        # OPENPOSE_Neck
        points.append(person.points[ailia.POSE_KEYPOINT_SHOULDER_CENTER].x)
        points.append(person.points[ailia.POSE_KEYPOINT_SHOULDER_CENTER].y)
        # OPENPOSE_RightShoulder
        points.append(person.points[ailia.POSE_KEYPOINT_SHOULDER_RIGHT].x)
        points.append(person.points[ailia.POSE_KEYPOINT_SHOULDER_RIGHT].y)
        # OPENPOSE_RightElbow
        points.append(person.points[ailia.POSE_KEYPOINT_ELBOW_RIGHT].x)
        points.append(person.points[ailia.POSE_KEYPOINT_ELBOW_RIGHT].y)
        # OPENPOSE_RightWrist
        points.append(person.points[ailia.POSE_KEYPOINT_WRIST_RIGHT].x)
        points.append(person.points[ailia.POSE_KEYPOINT_WRIST_RIGHT].y)
        # OPENPOSE_LeftShoulder
        points.append(person.points[ailia.POSE_KEYPOINT_SHOULDER_LEFT].x)
        points.append(person.points[ailia.POSE_KEYPOINT_SHOULDER_LEFT].y)
        # OPENPOSE_LeftElbow
        points.append(person.points[ailia.POSE_KEYPOINT_ELBOW_LEFT].x)
        points.append(person.points[ailia.POSE_KEYPOINT_ELBOW_LEFT].y)
        # OPENPOSE_LeftWrist
        points.append(person.points[ailia.POSE_KEYPOINT_WRIST_LEFT].x)
        points.append(person.points[ailia.POSE_KEYPOINT_WRIST_LEFT].y)
        # OPENPOSE_RightHip
        points.append(person.points[ailia.POSE_KEYPOINT_HIP_RIGHT].x)
        points.append(person.points[ailia.POSE_KEYPOINT_HIP_RIGHT].y)
        # OPENPOSE_RightKnee
        points.append(person.points[ailia.POSE_KEYPOINT_KNEE_RIGHT].x)
        points.append(person.points[ailia.POSE_KEYPOINT_KNEE_RIGHT].y)
        # OPENPOSE_RightAnkle
        points.append(person.points[ailia.POSE_KEYPOINT_ANKLE_RIGHT].x)
        points.append(person.points[ailia.POSE_KEYPOINT_ANKLE_RIGHT].y)
        # OPENPOSE_LeftHip
        points.append(person.points[ailia.POSE_KEYPOINT_HIP_LEFT].x)
        points.append(person.points[ailia.POSE_KEYPOINT_HIP_LEFT].y)
        # OPENPOSE_LeftKnee
        points.append(person.points[ailia.POSE_KEYPOINT_KNEE_LEFT].x)
        points.append(person.points[ailia.POSE_KEYPOINT_KNEE_LEFT].y)
        # OPENPOSE_LAnkle
        points.append(person.points[ailia.POSE_KEYPOINT_ANKLE_LEFT].x)
        points.append(person.points[ailia.POSE_KEYPOINT_ANKLE_LEFT].y)
        # OPENPOSE_RightEye
        points.append(person.points[ailia.POSE_KEYPOINT_EYE_RIGHT].x)
        points.append(person.points[ailia.POSE_KEYPOINT_EYE_RIGHT].y)
        # OPENPOSE_LeftEye
        points.append(person.points[ailia.POSE_KEYPOINT_EYE_LEFT].x)
        points.append(person.points[ailia.POSE_KEYPOINT_EYE_LEFT].y)
        # OPENPOSE_RightEar
        points.append(person.points[ailia.POSE_KEYPOINT_EAR_RIGHT].x)
        points.append(person.points[ailia.POSE_KEYPOINT_EAR_RIGHT].y)
        # OPENPOSE_LeftEar
        points.append(person.points[ailia.POSE_KEYPOINT_EYE_LEFT].x)
        points.append(person.points[ailia.POSE_KEYPOINT_EYE_LEFT].y)
        # OPENPOSE_Background
        points.append(person.points[ailia.POSE_KEYPOINT_BODY_CENTER].x)
        points.append(person.points[ailia.POSE_KEYPOINT_BODY_CENTER].y)

        neck_x = person.points[ailia.POSE_KEYPOINT_SHOULDER_CENTER].x
        neck_y = person.points[ailia.POSE_KEYPOINT_SHOULDER_CENTER].y

        hip_x = (person.points[ailia.POSE_KEYPOINT_HIP_RIGHT].x +
                 person.points[ailia.POSE_KEYPOINT_HIP_LEFT].x) / 2
        hip_y = (person.points[ailia.POSE_KEYPOINT_HIP_RIGHT].y +
                 person.points[ailia.POSE_KEYPOINT_HIP_LEFT].y) / 2

        d = math.sqrt(
            (neck_x-hip_x)*(neck_x-hip_x)+(neck_y-hip_y)*(neck_y-hip_y)
        )

        for i in range(int(len(points)/2)):
            # From neck to Hip should be about 110 pixels
            target_width = 110 / d
            points[i*2+0] = points[i*2+0]*input_img.shape[1]*(
                target_width/input_img.shape[1]
            )
            points[i*2+1] = points[i*2+1]*input_img.shape[0]*(
                target_width/input_img.shape[1]
            )

        display_3d_pose(points, baseline)


# ======================
# Main functions
# ======================
def recognize_from_image():
    # net initialize
    pose = ailia.PoseEstimator(
        MODEL_PATH, WEIGHT_PATH, env_id=args.env_id, algorithm=ALGORITHM,
        memory_mode=args.memory_mode
    )
    baseline = ailia.Net(
        BASELINE_MODEL_PATH, BASELINE_WEIGHT_PATH, env_id=args.env_id,
        memory_mode=args.memory_mode
    )
    baseline.set_input_shape((1, 32))

    # input image loop
    for image_path in args.input:
        # prepare input data
        logger.info(image_path)
        src_img = imread(image_path)
        input_image = load_image(
            image_path,
            (IMAGE_HEIGHT, IMAGE_WIDTH),
            normalize_type='None',
        )
        input_data = cv2.cvtColor(input_image, cv2.COLOR_RGB2BGRA)

        # inference
        logger.info('Start inference...')
        if args.benchmark:
            logger.info('BENCHMARK mode')
            for i in range(5):
                start = int(round(time.time() * 1000))
                _ = pose.compute(input_data)
                end = int(round(time.time() * 1000))
                logger.info(f'\tailia processing time {end - start} ms')
        else:
            _ = pose.compute(input_data)

        # postprocessing
        count = pose.get_object_count()
        logger.info(f'person_count={count}')
        display_result(src_img, pose, baseline)
        savepath = get_savepath(args.savepath, image_path)
        logger.info(f'saved at : {savepath}')
        cv2.imwrite(savepath, src_img)

        # display 3d pose
        savepath_3d = pathlib.PurePath(savepath)
        savepath_3d = savepath_3d.stem+"_3dpose"+savepath_3d.suffix
        logger.info(f'saved at : {savepath_3d}')
        plt.savefig(savepath_3d, dpi=120)
        if not args.skip_plot:
            plt.show()

    logger.info('Script finished successfully.')


def recognize_from_video():
    # net initialize
    pose = ailia.PoseEstimator(
        MODEL_PATH, WEIGHT_PATH, env_id=args.env_id, algorithm=ALGORITHM,
        memory_mode=args.memory_mode
    )
    baseline = ailia.Net(
        BASELINE_MODEL_PATH, BASELINE_WEIGHT_PATH, env_id=args.env_id,
        memory_mode=args.memory_mode
    )
    baseline.set_input_shape((1, 32))

    capture = webcamera_utils.get_capture(args.video)

    # create video writer if savepath is specified as video format
    if args.savepath != SAVE_IMAGE_PATH:
        logger.warning(
            'currently, video results cannot be output correctly...'
        )
        f_h = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        f_w = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        writer = webcamera_utils.get_writer(args.savepath, f_h, f_w)
    else:
        writer = None
    
    frame_shown = False
    while(True):
        ret, frame = capture.read()
        if (cv2.waitKey(1) & 0xFF == ord('q')) or not ret:
            break
        if frame_shown and cv2.getWindowProperty('frame', cv2.WND_PROP_VISIBLE) == 0:
            break

        input_image, input_data = webcamera_utils.adjust_frame_size(
            frame, IMAGE_HEIGHT, IMAGE_WIDTH,
        )
        input_data = cv2.cvtColor(input_data, cv2.COLOR_BGR2BGRA)

        # inference
        _ = pose.compute(input_data)

        # postprocessing
        display_result(input_image, pose, baseline)
        cv2.imshow('frame', input_image)
        frame_shown = True

        # display 3d pose
        plt.pause(0.01)
        if not plt.get_fignums():
            break
        # # save results
        # if writer is not None:
        #     writer.write(res_img)

    capture.release()
    cv2.destroyAllWindows()
    if writer is not None:
        writer.release()
    logger.info('Script finished successfully.')


def main():
    # model files check and download
    check_and_download_models(WEIGHT_PATH, MODEL_PATH, REMOTE_PATH)
    check_and_download_models(
        BASELINE_WEIGHT_PATH, BASELINE_MODEL_PATH, BASELINE_REMOTE_PATH
    )

    if args.video is not None:
        # video mode
        recognize_from_video()
    else:
        # image mode
        recognize_from_image()


if __name__ == '__main__':
    main()
