import sys
import cv2
import yaml
import time
import ailia
import numpy as np
from did_m3d_util import Detect, get_objects_from_label

# import original modules
sys.path.append('../../util')
# logger
from logging import getLogger  # noqa: E402

from model_utils import check_and_download_models  # noqa
from image_utils import imread  # noqa
from arg_utils import get_base_parser, get_savepath, update_parser  # noqa: E402
import webcamera_utils

import time


logger = getLogger(__name__)

# ======================
# Parameters
# ======================
MODEL_NAME = 'did_m3d'
WEIGHT_PATH = MODEL_NAME + '.onnx'
MODEL_PATH  = MODEL_NAME + '.onnx.prototxt'
REMOTE_PATH = 'https://storage.googleapis.com/ailia-models/did_m3d/'

IMAGE_PATH = '000005.png'
SAVE_IMAGE_PATH = 'output.png'

# ======================
# Arguemnt Parser Config
# ======================
parser = get_base_parser('did m3d model', IMAGE_PATH, SAVE_IMAGE_PATH)

parser.add_argument('--config', type=str, default='kitti.yaml')
parser.add_argument('--calib_path', type=str, default='000005.txt')

args = update_parser(parser)

def visualization(img,result_object):
    objects = get_objects_from_label(result_object)

    with open(args.calib_path, 'r') as f:
        lines = f.readlines()
        P2 = np.array(lines[2].strip().split(' ')[1:], dtype=np.float32).reshape(3, 4)

    def line(img,p1, p2):
        return cv2.line(img,(int(p1[0]), int(p1[1])),
                            (int(p2[0]), int(p2[1])),
                             color=(0,255,0))

    for i in range(len(objects)):
        corners_3d = objects[i].generate_corners3d()
        corners_3d_hom = np.concatenate((corners_3d, np.ones((8, 1))), axis=1)
        corners_img = np.matmul(corners_3d_hom , P2.T)
        corners_img = corners_img[:, :2] / corners_img[:, 2][:, None]

        # draw the upper 4 horizontal lines
        img = line(img,corners_img[0], corners_img[1])
        img = line(img,corners_img[1], corners_img[2])
        img = line(img,corners_img[2], corners_img[3])
        img = line(img,corners_img[3], corners_img[0])

        #dras,w the lower 4 horizontal lines
        img = line(img,corners_img[4], corners_img[5])
        img = line(img,corners_img[5], corners_img[6])
        img = line(img,corners_img[6], corners_img[7])
        img = line(img,corners_img[7], corners_img[4])

        #dras,w the 4 vertical lines
        img = line(img,corners_img[4], corners_img[0])
        img = line(img,corners_img[5], corners_img[1])
        img = line(img,corners_img[6], corners_img[2])
        img = line(img,corners_img[7], corners_img[3])
    
    return img
    


# ======================
# Main functions
# ======================
def recognize_from_image():
    # net initialize
    net = ailia.Net(None,WEIGHT_PATH,memory_mode=args.memory_mode)
    cfg = yaml.load(open(args.config, 'r'), Loader=yaml.Loader)
    detect = Detect(net,cfg['dataset'],th=0.3)

    # input image loop
    for image_path in args.input:
        logger.info(image_path)
        img = imread(image_path)

        # inference
        logger.info('Start inference...')
        if args.benchmark:
            logger.info('BENCHMARK mode')
            for i in range(5):
                start = int(round(time.time() * 1000))
                result_object = detect.run(img,args.calib_path)
                end = int(round(time.time() * 1000))
                logger.info(f'\tailia processing time {end - start} ms')
        else:
            result_object = detect.run(img,args.calib_path)

        savepath = get_savepath(args.savepath, image_path, ext='.png')
        res_img  = visualization(img,result_object)
        cv2.imwrite(savepath,res_img)
        logger.info(f'saved at : {savepath}')
    logger.info('Script finished successfully.')


def recognize_from_video():
    # net initialize
    net = ailia.Net(None,WEIGHT_PATH,memory_mode=args.memory_mode)
    cfg = yaml.load(open(args.config, 'r'), Loader=yaml.Loader)
    detect = Detect(net,cfg['dataset'],th=0.3)

    capture = webcamera_utils.get_capture(args.video)

    # create video writer if savepath is specified as video format
    if args.savepath != SAVE_IMAGE_PATH:
        f_h = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        f_w = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        writer = webcamera_utils.get_writer(args.savepath, f_h, f_w)
    else:
        writer = None

    frame_shown = False
    while (True):
        ret, frame = capture.read()
        if (cv2.waitKey(1) & 0xFF == ord('q')) or not ret:
            break
        if frame_shown and cv2.getWindowProperty('frame', cv2.WND_PROP_VISIBLE) == 0:
            break
        
        result_object = detect.run(frame,args.calib_path)
        res_img = visualization(frame,result_object)

        cv2.imshow('frame', res_img)
        frame_shown = True
        time.sleep(0.1)

        # save results
        if writer is not None:
            writer.write(res_img)

    capture.release()
    cv2.destroyAllWindows()
    if writer is not None:
        writer.release()
    logger.info('Script finished successfully.')


def main():
    # model files check and download
    check_and_download_models(WEIGHT_PATH, MODEL_PATH, REMOTE_PATH)


    if args.video is None:
        # image mode
        recognize_from_image()
    else:
        # video mode
        recognize_from_video()

if __name__ == '__main__':
    main()
