import glob
import sys
import time

import ailia
import cv2
import numpy as np
import onnxruntime

sys.path.append('../../util')
# logger
from logging import getLogger  # noqa: E402

import webcamera_utils  # noqa: E402
from image_utils import imread  # noqa: E402
from model_utils import check_and_download_models  # noqa: E402
from arg_utils import get_base_parser, get_savepath, update_parser  # noqa: E402

logger = getLogger(__name__)

from utils_hitnet import CameraConfig, draw_depth, draw_disparity

camera_config =  CameraConfig(0.546, 1000)
max_distance = 30

# ======================
# Parameters 1
# ======================

LEFT_IMAGE_PATH = "./cones/im2.ppm"
RIGHT_IMAGE_PATH = "./cones/im6.ppm"
STEREO_DATA_DIR ="./stereo_data"
SAVE_IMAGE_PATH = 'output.png'

WEIGHT_PATH = 'hitnet.onnx' 
MODEL_PATH = 'hitnet.onnx.prototxt'
REMOTE_PATH = 'https://storage.googleapis.com/ailia-models/hitnet/'

# ======================
# Argument Parser Config
# ======================

parser = get_base_parser(
    'hitnet',
    LEFT_IMAGE_PATH,
    RIGHT_IMAGE_PATH,
    SAVE_IMAGE_PATH,
)
parser.add_argument(
    '-l','--left', type=str,
    default=LEFT_IMAGE_PATH,
    help='The input image for left image.'
)
parser.add_argument(
    '-r', '--right', type=str,
    default=RIGHT_IMAGE_PATH,
    help='The input image for right image.'
)
parser.add_argument(
    '-v', '--video', type=str,
    help='The input video for pole detection.'
)
parser.add_argument(
    '-o', '--onnx', action='store_true',
    help="Option to use onnxrutime to run or not."
)
args = update_parser(parser)

# # ======================
# # Main functions
# # ======================

def preprocessing(left_img,right_img,input_shape):

    left_img = cv2.resize(left_img,(input_shape[3],input_shape[2]))
    right_img = cv2.resize(right_img,(input_shape[3],input_shape[2]))
    
    left_img = cv2.cvtColor(left_img, cv2.COLOR_BGR2RGB)
    right_img = cv2.cvtColor(right_img, cv2.COLOR_BGR2RGB)
    
    combined_img = np.concatenate((left_img, right_img), axis=-1) / 255.0
    combined_img = combined_img.transpose(2, 0, 1)

    return np.expand_dims(combined_img, 0).astype(np.float32)

def recognize_from_image(net):
    
    # read left and right images
    left_img = imread(args.left)
    right_img = imread(args.right)

    # get model info
    if args.onnx:
        input_name = net.get_inputs()[0].name
        output_name = net.get_outputs()[0].name
        input_shape = net.get_inputs()[0].shape
    else:
        input_shape = net.get_input_shape()

    # preprocessing
    input_tensor = preprocessing(left_img,right_img,input_shape)

    if args.onnx:
        disparity_map = net.run( [output_name], { input_name : input_tensor })[0][0]
    else:
        disparity_map = net.run( input_tensor )[0][0]

    # estimate depth
    disparity_map = np.array(disparity_map)
    depth_map = camera_config.f*camera_config.baseline / disparity_map
    color_disparity = draw_disparity(disparity_map)
    color_depth = draw_depth(depth_map, max_distance)

    # save output
    color_depth = cv2.resize(color_depth, (left_img.shape[1],left_img.shape[0])) 
    cv2.imwrite("output.png",color_depth)
  

    logger.info('Script finished successfully.')

def recognize_from_video(net):

    # get image list
    left_images = glob.glob( args.video + '/image_L/*.png')
    left_images.sort()
    right_images = glob.glob( args.video + '/image_R/*.png')
    right_images.sort()

    # get model info
    if args.onnx:
        input_name = net.get_inputs()[0].name
        output_name = net.get_outputs()[0].name
        input_shape = net.get_inputs()[0].shape
    else:
        input_shape = net.get_input_shape()
    
    if not left_images or not right_images:
        logger.error("This model requires stereo images")
        return

    cv2.namedWindow("Estimated depth", cv2.WINDOW_NORMAL)
    for left_path, right_path in zip(left_images, right_images):

        # Read frame from the video
        left_img = imread(left_path)
        right_img = imread(right_path)

        # preprocessing
        input_tensor = preprocessing(left_img,right_img,input_shape)

        if args.onnx:
            disparity_map = net.run( [output_name], { input_name : input_tensor })[0][0]
        else:
            disparity_map = net.run( input_tensor )[0][0]
        
        # estimate depth
        disparity_map = np.array(disparity_map)#.astype(np.uint8)
        depth_map = camera_config.f*camera_config.baseline/disparity_map
        color_disparity = draw_disparity(disparity_map)
        color_depth = draw_depth(depth_map, max_distance)

        # show output
        color_depth = cv2.resize(color_depth, (left_img.shape[1],left_img.shape[0]))
        combined_image = np.hstack((left_img, color_depth))
        cv2.imshow("Estimated depth", combined_image)

        # Press key q to stop
        if cv2.waitKey(1) == ord('q'):
            break
        if cv2.getWindowProperty('Estimated depth', cv2.WND_PROP_VISIBLE) == 0:
            break

    
    cv2.destroyAllWindows()
    logger.info('Script finished successfully.')

def main():

    # model files check and download
    check_and_download_models(WEIGHT_PATH, MODEL_PATH, REMOTE_PATH)

    # net initialize
    if args.onnx:
        import onnxruntime
        net = onnxruntime.InferenceSession(WEIGHT_PATH)
    else:
        logger.info(f'env_id: {args.env_id}')
        net = ailia.Net(MODEL_PATH, WEIGHT_PATH, env_id=args.env_id, memory_mode=args.memory_mode)
    

    if args.video is not None:
        # video mode
        recognize_from_video(net)
    else:
        # image mode
        recognize_from_image(net)

if __name__ == '__main__':
    main()
