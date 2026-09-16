import argparse
import os
import sys
import time
from glob import glob

import ailia
import cv2
import numpy as np

from hybridnets_utils import *

sys.path.append('../../util')
import webcamera_utils  # noqa: E402
from image_utils import imread  # noqa: E402
from model_utils import check_and_download_models  # noqa: E402
from arg_utils import get_base_parser, get_savepath, update_parser

WEIGHT_PATH = 'hybridnets.onnx'
MODEL_PATH  = 'hybridnets.onnx.prototxt'
REMOTE_PATH = 'https://storage.googleapis.com/ailia-models/hybridnets/'

# logger
from logging import getLogger

logger = getLogger(__name__)

IMAGE_PATH = 'input.jpg'
SAVE_IMAGE_PATH = 'output.jpg'

HEIGHT = 384
WIDTH = 640

parser = get_base_parser('HybridNets model', IMAGE_PATH, SAVE_IMAGE_PATH)

parser.add_argument('--nms_thresh',
    type=restricted_float,
    default='0.25')

parser.add_argument(
    '-m', '--model_name',
    default='hybridnets.onnx', type=str,
    help='model path'
)

parser.add_argument('--iou-thres',
    default=0.45, type=float,
    help='IOU threshold for NMS'
)

args = update_parser(parser)

def detect(frame,model,color_list):
    obj_list = ['car']
    
    frame = cv2.resize(frame,(WIDTH,HEIGHT))
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    h0, w0 = frame.shape[:2]  # orig hw
    r = WIDTH / max(h0, w0)  # resize image to img_size
    input_img = cv2.resize(frame, (int(w0 * r), int(h0 * r)), interpolation=cv2.INTER_AREA)
    h, w = input_img.shape[:2]
    
    (input_img, _, _), ratio, pad = letterbox((input_img, input_img.copy(), input_img.copy()), WIDTH, auto=True,scaleup=False)
    
    shapes = ((h0, w0), ((h / h0, w / w0), pad))
    
    #normalized
    img = input_img.copy().astype(np.float32)  # (3,640,640) RGB
    img = img / 255.0
    img[:, :, 0] -= 0.485
    img[:, :, 1] -= 0.456
    img[:, :, 2] -= 0.406
    img[:, :, 0] /= 0.229
    img[:, :, 1] /= 0.224
    img[:, :, 2] /= 0.225
    
    
    x = np.expand_dims(img,0)
    x = x.transpose(0,3,1,2)
    
    
    regression, classification, seg = model.run(x)
    anchors = np.load('anchors.npy')
    
    seg = seg[:, :, 12:372, :]
    
    da_seg_mask = seg

    #interp Nearest neighbor
    da_seg_mask = cv2.resize(seg[0][0],dsize=(w0,h0))
    da_seg_mask = cv2.resize(seg[0].transpose(1,2,0),dsize=(w0,h0))
    da_seg_mask = np.expand_dims(da_seg_mask.transpose(2,0,1),0)
    
    da_seg_mask = np.argmax(da_seg_mask,axis=1)
    
    da_seg_mask_ = da_seg_mask[0]
    
    color_area = np.zeros((da_seg_mask_.shape[0], da_seg_mask_.shape[1], 3), dtype=np.uint8)
    color_area[da_seg_mask_ == 1] = [0, 255, 0]
    color_area[da_seg_mask_ == 2] = [0, 0, 255]
    color_seg = color_area[..., ::-1]
    
    color_mask = np.mean(color_seg, 2)
    frame[color_mask != 0] = frame[color_mask != 0] * 0.5 + color_seg[color_mask != 0] * 0.5
    frame = frame.astype(np.uint8)
    
    regressBoxes = BBoxTransform()
    clipBoxes = ClipBoxes()
    out = postprocess(x, anchors, regression, classification,
                      regressBoxes, clipBoxes,
                      args.nms_thresh, args.iou_thres)
    out = out[0]
    out['rois'] = scale_coords(frame[:2], out['rois'], shapes[0], shapes[1])
    for j in range(len(out['rois'])):
        x1, y1, x2, y2 = out['rois'][j].astype(int)
        obj = obj_list[out['class_ids'][j]]
        score = float(out['scores'][j])
        plot_one_box(frame, [x1, y1, x2, y2], label=obj, score=score,
                     color=color_list[get_index_label(obj, obj_list)])

    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    return frame

def recognize_from_image():

    t0 = time.time()

    model = ailia.Net(None,args.model_name, env_id=args.env_id, memory_mode=args.memory_mode)
    if args.profile:
        model.set_profile_mode(True)
    
    color_list = standard_to_bgr(STANDARD_COLORS)
    
    for image_path in args.input:
        shapes = []
        
        save_path = get_savepath(args.savepath,image_path)

        img_formats = ['.bmp', '.jpg', '.jpeg', '.png', '.tif', '.tiff', '.dng']
        vid_formats = ['.mov', '.avi', '.mp4', '.mpg', '.mpeg', '.m4v', '.wmv', '.mkv']

        files = os.path.splitext(image_path)[-1].lower()

        images = [x for x in files if files in img_formats]
        videos = [x for x in files if files in vid_formats]
        ni, nv = len(images), len(videos)
        video_flag = any([True] * ni + [False] * nv)
        
        if video_flag:
            image = imread(image_path)

            image = detect(image,model,color_list)

            cv2.imwrite(save_path,image)
        else:
            cap = cv2.VideoCapture(image_path)
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out_stream = cv2.VideoWriter(save_path, fourcc, 30.0,
                                    (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))))

            while True:
            
                ret, frame = cap.read()
            
                color_list = standard_to_bgr(STANDARD_COLORS)
                frame = detect(frame,model,color_list)

                out_stream.write(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
             
            cap.release()
            out_stream.release()

    if args.profile:
        print(model.get_summary())

    logger.info('Script finished successfully.')

def recognize_from_video():

    capture = webcamera_utils.get_capture(args.video)

    if args.savepath != SAVE_IMAGE_PATH:
        writer = webcamera_utils.get_writer(args.savepath, HEIGHT, WIDTH)
    else:
        writer = None
    model = ailia.Net(None,args.model_name, env_id=args.env_id, memory_mode=args.memory_mode)
    while (True):
        ret, frame = capture.read()
        if (cv2.waitKey(1) & 0xFF == ord('q')) or not ret:
            break

        res_img = detect(frame,model,color_list)

        cv2.imshow('frame', res_img)

        # save results
        if writer is not None:
            writer.write(res_img)

    capture.release()
    cv2.destroyAllWindows()
    if writer is not None:
        writer.release()
    logger.info('Script finished successfully.')
    pass

if __name__ == '__main__':

    check_and_download_models(WEIGHT_PATH, MODEL_PATH, REMOTE_PATH)
    if args.video is not None:
        # video mode
        recognize_from_video()
    else:
        # image mode
        recognize_from_image()

