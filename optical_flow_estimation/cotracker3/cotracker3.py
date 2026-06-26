import sys
import cv2
import time
import numpy as np

import ailia
import onnxruntime as ort
from vis import Visualizer

# import original modules
sys.path.append('../../util')
from arg_utils import get_base_parser, update_parser  # noqa: E402
from model_utils import check_and_download_models  # noqa: E402

# logger
from logging import getLogger   # noqa: E402
logger = getLogger(__name__)


# ======================
# Parameters
# ======================
VIDEO_PATH = 'input.mp4'
SAVE_PATH = 'output.mp4'

# ======================
# Argument Parser Config
# ======================
parser = get_base_parser(
    'CoTracker3: Simpler and Better Point Tracking by Pseudo-Labelling Real Videos',
    VIDEO_PATH,
    SAVE_PATH,
)

parser.add_argument("--grid_size", type=int, default=10, help="Regular grid size")
parser.add_argument(
    "--grid_query_frame",
    type=int,
    default=0,
    help="Compute dense and grid tracks starting from this frame",
)
parser.add_argument(
    "--backward_tracking",
    action="store_true",
    help="Compute tracks in both directions, not only forward",
)

parser.add_argument('--onnx', action='store_true', help='execute onnxruntime version.')

args = update_parser(parser)

# ==========================
# MODEL AND OTHER PARAMETERS
# ==========================
WEIGHT_PATH = 'cotracker3.onnx'
MODEL_PATH =  'cotracker3.onnx.prototxt'
REMOTE_PATH = 'https://storage.googleapis.com/ailia-models/cotracker3/'

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
        frames.append(frame)
    cap.release()

    return np.stack(frames)


def compute(net,video):
    if not args.onnx:
        result = net.run((video,np.array(args.grid_size      ,dtype=np.int64),
                                np.array(args.grid_query_frame,dtype=np.int64)))
    else:
        input_name1 = net.get_inputs()[0].name
        input_name2 = net.get_inputs()[1].name
        input_name3 = net.get_inputs()[2].name
        result= net.run([],{input_name1:video,
                            input_name2:np.array(args.grid_size ,dtype=np.int64),
                            input_name3:np.array(args.grid_query_frame,dtype=np.int64)})
    return result

# ======================
# Main functions
# ======================
def recognize_from_video():
    # net initialize
    if not args.onnx:
        memory_mode = ailia.get_memory_mode(   
                      reduce_constant=True, ignore_input_with_initializer=True,
                      reduce_interstage=False, reuse_interstage=True)

        net = ailia.Net(MODEL_PATH, WEIGHT_PATH, env_id=args.env_id,memory_mode=memory_mode)
    else:
        net = ort.InferenceSession(WEIGHT_PATH)

    # load video
    vis = Visualizer( pad_value=120, linewidth=3)

    for path in args.input:
        video = read_video_from_path(path)
        np.transpose(video,(0, 3, 1, 2))
        video = np.transpose(video,(0, 3, 1, 2))[np.newaxis, ...].astype(np.float32)


        # calculate feature map
        logger.info('Start calculating feature map...')
        if args.benchmark:
            logger.info('BENCHMARK mode')
            for i in range(args.benchmark_count):
                start = int(round(time.time() * 1000))
                result = compute(net,video)
                end = int(round(time.time() * 1000))
                logger.info(f'\tailia processing time {end - start} ms')
        else:
            result = compute(net,video)

        pred_tracks     = np.array(result[0])
        pred_visibility = np.array(result[1])

        # save a video with predicted tracks
        logger.info(f'saved at : {args.savepath}')
        vis.visualize(
            video,
            pred_tracks,
            pred_visibility,
            args.savepath
        )
        logger.info('Script finished successfully.')


def main():
    # model files check and download
    check_and_download_models(WEIGHT_PATH, MODEL_PATH, REMOTE_PATH)

    recognize_from_video()


if __name__ == '__main__':
    main()
