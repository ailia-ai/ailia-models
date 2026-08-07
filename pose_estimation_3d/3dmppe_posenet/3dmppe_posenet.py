import os
import sys

os.environ["KMP_DUPLICATE_LIB_OK"] = "True"
import argparse
import math
import os.path as osp

import cv2
import matplotlib
import numpy as np
from PIL import Image

matplotlib.use('Agg')
import time

import ailia
import matplotlib.pyplot as plt
from tqdm import tqdm

# import original modules
sys.path.append('../../util')
# logger
from logging import getLogger  # noqa: E402

import webcamera_utils  # noqa: E402
from image_utils import imread  # noqa: E402
from model_utils import check_and_download_models  # noqa: E402
from arg_utils import get_base_parser, get_savepath, update_parser  # noqa: E402

logger = getLogger(__name__)

import warnings

warnings.simplefilter("ignore", DeprecationWarning)

# ======================
# Parameters
# ======================
WEIGHT_PATH_MASKRCNN = 'mask_rcnn_R_50_FPN_1x.opt.onnx'
MODEL_PATH_MASKRCNN = 'mask_rcnn_R_50_FPN_1x.opt.onnx.prototxt'
REMOTE_PATH_MASKRCNN = 'https://storage.googleapis.com/ailia-models/mask_rcnn/'

WEIGHT_PATH_ROOTNET = 'rootnet_snapshot_18.opt.onnx'
MODEL_PATH_ROOTNET = 'rootnet_snapshot_18.opt.onnx.prototxt'
REMOTE_PATH_ROOTNET = 'https://storage.googleapis.com/ailia-models/3dmppe_posenet/'

WEIGHT_PATH_POSENET = 'posenet_snapshot_24.opt.onnx'
MODEL_PATH_POSENET = 'posenet_snapshot_24.opt.onnx.prototxt'
REMOTE_PATH_POSENET = 'https://storage.googleapis.com/ailia-models/3dmppe_posenet/'

IMAGE_OR_VIDEO_PATH = 'input.jpg' # input.mp4
SAVE_IMAGE_OR_VIDEO_PATH = 'output.png' # output.mp4


# ======================
# Arguemnt Parser Config
# ======================
parser = get_base_parser(
    'Real-time NN for 3D Multi-person Pose Estimation by PoseNet',
    IMAGE_OR_VIDEO_PATH,
    SAVE_IMAGE_OR_VIDEO_PATH,
    large_model=True
)
args = update_parser(parser)


# ======================
# Utils
# ======================
def preprocess_maskrcnn(image):
    # Resize
    ratio = 800.0 / min(image.size[0], image.size[1])
    resize_w = int(ratio * image.size[0])
    resize_h = int(ratio * image.size[1])
    if (max(resize_w, resize_h) > 1280): # minor fix
        ratio = 1280.0 / max(image.size[0], image.size[1])
        resize_w = int(ratio * image.size[0])
        resize_h = int(ratio * image.size[1])
    image = image.resize(
        (resize_w, resize_h),
        Image.BILINEAR
    )

    image = np.array(image)[:, :, [2, 1, 0]].astype('float32')
    image = np.transpose(image, [2, 0, 1])

    # Normalize
    mean_vec = np.array([102.9801, 115.9465, 122.7717])
    for i in range(image.shape[0]):
        image[i, :, :] = image[i, :, :] - mean_vec[i]

    # Pad to be divisible of 32
    padded_h = int(math.ceil(image.shape[1] / 32) * 32)
    padded_w = int(math.ceil(image.shape[2] / 32) * 32)

    padded_image = np.zeros((3, padded_h, padded_w), dtype=np.float32)
    padded_image[:, :image.shape[1], :image.shape[2]] = image
    return padded_image


# --------------------------------------------------------
# Fast R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick
# --------------------------------------------------------
def nms(dets, scores, thresh):
    '''
    dets is a numpy array : num_dets, 4
    scores ia  nump array : num_dets,
    '''
    x1 = dets[:, 0]
    y1 = dets[:, 1]
    x2 = dets[:, 2]
    y2 = dets[:, 3]

    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    order = scores.argsort()[::-1] # get boxes with more ious first

    keep = []
    while order.size > 0:
        i = order[0] # pick maxmum iou box
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w = np.maximum(0.0, xx2 - xx1 + 1) # maximum width
        h = np.maximum(0.0, yy2 - yy1 + 1) # maxiumum height
        inter = w * h
        ovr = inter / (areas[i] + areas[order[1:]] - inter)

        inds = np.where(ovr <= thresh)[0]
        order = order[inds + 1]

    return keep


# ======================
# Main functions
# ======================
def maskrcnn_to_image(image, net_maskrcnn, benchmark=False, det_thresh=0.5, iou_thresh=0.2):
    # prepare input data
    input_data = preprocess_maskrcnn(image)

    # inference
    if benchmark:
        logger.info('BENCHMARK mode')
        for i in range(5):
            start = int(round(time.time() * 1000))
            boxes, labels, scores, masks = net_maskrcnn.predict([input_data])
            end = int(round(time.time() * 1000))
            logger.info(f'\tailia processing time {end - start} ms')
    else:
        boxes, labels, scores, masks = net_maskrcnn.predict([input_data])

    # narrow down the bounding box only person class [ref : https://github.com/onnx/models/blob/master/vision/object_detection_segmentation/mask-rcnn/dependencies/coco_classes.txt]
    boxes = boxes[(labels == 1) & (scores > det_thresh)]
    scores = scores[(labels == 1) & (scores > det_thresh)]

    # nms
    keep = nms(dets=boxes, scores=scores, thresh=iou_thresh)
    boxes = boxes[keep]

    # Resize boxes
    ratio = 800.0 / min(image.size[0], image.size[1])
    resize_w = int(ratio * image.size[0])
    resize_h = int(ratio * image.size[1])
    if (max(resize_w, resize_h) > 1280.0):
        ratio = 1280.0 / max(image.size[0], image.size[1])
    boxes /= ratio

    return boxes


class Config:
    
    ## input, output
    input_shape = (256, 256) 
    output_shape = (input_shape[0]//4, input_shape[1]//4)
    depth_dim = 64
    bbox_3d_shape = (2000, 2000, 2000) # depth, height, width
    pixel_mean = (0.485, 0.456, 0.406)
    pixel_std = (0.229, 0.224, 0.225)

    bbox_real = (2000, 2000) # Human36M, MuCo, MuPoTS: (2000, 2000), PW3D: (2, 2)

cfg = Config()


def process_bbox(bbox, width, height):
    # sanitize bboxes
    x, y, w, h = bbox
    x1 = np.max((0, x))
    y1 = np.max((0, y))
    x2 = np.min((width - 1, x1 + np.max((0, w - 1))))
    y2 = np.min((height - 1, y1 + np.max((0, h - 1))))
    if w*h > 0 and x2 >= x1 and y2 >= y1:
        bbox = np.array([x1, y1, x2-x1, y2-y1])
    else:
        return None

    # aspect ratio preserving bbox
    w = bbox[2]
    h = bbox[3]
    c_x = bbox[0] + w/2.
    c_y = bbox[1] + h/2.
    aspect_ratio = cfg.input_shape[1]/cfg.input_shape[0]
    if w > aspect_ratio * h:
        h = w / aspect_ratio
    elif w < aspect_ratio * h:
        w = h * aspect_ratio
    bbox[2] = w*1.25
    bbox[3] = h*1.25
    bbox[0] = c_x - bbox[2]/2.
    bbox[1] = c_y - bbox[3]/2.
    return bbox


def pixel2cam(pixel_coord, f, c):
    x = (pixel_coord[:, 0] - c[0]) / f[0] * pixel_coord[:, 2]
    y = (pixel_coord[:, 1] - c[1]) / f[1] * pixel_coord[:, 2]
    z = pixel_coord[:, 2]
    cam_coord = np.concatenate((x[:,None], y[:,None], z[:,None]),1)
    return cam_coord


from matplotlib.axes._axes import \
    _log as matplotlib_axes_logger  # provisional...

matplotlib_axes_logger.setLevel('ERROR')                          # provisional...
def vis_keypoints(img, kps, kps_lines, kp_thresh=0.4, alpha=1):

    # Convert from plt 0-1 RGBA colors to 0-255 BGR colors for opencv.
    cmap = plt.get_cmap('rainbow')
    colors = [cmap(i) for i in np.linspace(0, 1, len(kps_lines) + 2)]
    colors = [(c[2] * 255, c[1] * 255, c[0] * 255) for c in colors]

    # Perform the drawing on a copy of the image, to allow for blending.
    kp_mask = np.copy(img)

    # Draw the keypoints.
    for l in range(len(kps_lines)):
        i1 = kps_lines[l][0]
        i2 = kps_lines[l][1]
        p1 = kps[0, i1].astype(np.int32), kps[1, i1].astype(np.int32)
        p2 = kps[0, i2].astype(np.int32), kps[1, i2].astype(np.int32)
        if kps[2, i1] > kp_thresh and kps[2, i2] > kp_thresh:
            cv2.line(
                kp_mask, p1, p2,
                color=colors[l], thickness=2, lineType=cv2.LINE_AA)
        if kps[2, i1] > kp_thresh:
            cv2.circle(
                kp_mask, p1,
                radius=3, color=colors[l], thickness=-1, lineType=cv2.LINE_AA)
        if kps[2, i2] > kp_thresh:
            cv2.circle(
                kp_mask, p2,
                radius=3, color=colors[l], thickness=-1, lineType=cv2.LINE_AA)

    # Blend the keypoints.
    return cv2.addWeighted(img, 1.0 - alpha, kp_mask, alpha, 0)


def vis_3d_multiple_skeleton(kpt_3d, kpt_3d_vis, kps_lines, fig_h, fig_w):

    fig = plt.figure(figsize=(fig_w/100, fig_h/100), dpi=100)
    ax = fig.add_subplot(111, projection='3d')

    # Convert from plt 0-1 RGBA colors to 0-255 BGR colors for opencv.
    cmap = plt.get_cmap('rainbow')
    colors = [cmap(i) for i in np.linspace(0, 1, len(kps_lines) + 2)]
    colors = [np.array((c[2], c[1], c[0])) for c in colors]
    
    for l in range(len(kps_lines)):
        i1 = kps_lines[l][0]
        i2 = kps_lines[l][1]

        person_num = kpt_3d.shape[0]
        for n in range(person_num):
            x = np.array([kpt_3d[n,i1,0], kpt_3d[n,i2,0]])
            y = np.array([kpt_3d[n,i1,1], kpt_3d[n,i2,1]])
            z = np.array([kpt_3d[n,i1,2], kpt_3d[n,i2,2]])

            if kpt_3d_vis[n,i1,0] > 0 and kpt_3d_vis[n,i2,0] > 0:
                ax.plot(x, z, -y, c=colors[l], linewidth=2)
            if kpt_3d_vis[n,i1,0] > 0:
                ax.scatter(kpt_3d[n,i1,0], kpt_3d[n,i1,2], -kpt_3d[n,i1,1], c=colors[l], marker='o')
            if kpt_3d_vis[n,i2,0] > 0:
                ax.scatter(kpt_3d[n,i2,0], kpt_3d[n,i2,2], -kpt_3d[n,i2,1], c=colors[l], marker='o')

    ax.set_title('output_pose_3d (x,y,z: camera-centered. mm.)')
    ax.set_xlabel('X Label')
    ax.set_ylabel('Z Label')
    ax.set_zlabel('Y Label')
    
    elev = 10
    azim = 330
    ax.view_init(elev=elev, azim=azim)
    ax.set_xlim(-2000, 2000)
    ax.set_ylim(5000, 25000)
    ax.set_zlim(-2000, 2000)
    fig.canvas.draw()
    # tostring_rgb was removed in matplotlib 3.10, use buffer_rgba instead
    vis_img = np.asarray(plt.gcf().canvas.buffer_rgba())[:, :, :3].copy()
    vis_img = vis_img.reshape(fig_h, fig_w, -1)
    plt.close()

    return vis_img
        

def generate_patch_image(cvimg, bbox, do_flip, scale, rot, do_occlusion):
    img = cvimg.copy()
    img_height, img_width, img_channels = img.shape

    # synthetic occlusion
    if do_occlusion:
        while True:
            area_min = 0.0
            area_max = 0.7
            synth_area = (random.random() * (area_max - area_min) + area_min) * bbox[2] * bbox[3]

            ratio_min = 0.3
            ratio_max = 1/0.3
            synth_ratio = (random.random() * (ratio_max - ratio_min) + ratio_min)

            synth_h = math.sqrt(synth_area * synth_ratio)
            synth_w = math.sqrt(synth_area / synth_ratio)
            synth_xmin = random.random() * (bbox[2] - synth_w - 1) + bbox[0]
            synth_ymin = random.random() * (bbox[3] - synth_h - 1) + bbox[1]

            if synth_xmin >= 0 and synth_ymin >= 0 and synth_xmin + synth_w < img_width and synth_ymin + synth_h < img_height:
                xmin = int(synth_xmin)
                ymin = int(synth_ymin)
                w = int(synth_w)
                h = int(synth_h)
                img[ymin:ymin+h, xmin:xmin+w, :] = np.random.rand(h, w, 3) * 255
                break

    bb_c_x = float(bbox[0] + 0.5*bbox[2])
    bb_c_y = float(bbox[1] + 0.5*bbox[3])
    bb_width = float(bbox[2])
    bb_height = float(bbox[3])

    if do_flip:
        img = img[:, ::-1, :]
        bb_c_x = img_width - bb_c_x - 1
    
    trans = gen_trans_from_patch_cv(bb_c_x, bb_c_y, bb_width, bb_height, cfg.input_shape[1], cfg.input_shape[0], scale, rot, inv=False)
    img_patch = cv2.warpAffine(img, trans, (int(cfg.input_shape[1]), int(cfg.input_shape[0])), flags=cv2.INTER_LINEAR)

    img_patch = img_patch[:,:,::-1].copy()
    img_patch = img_patch.astype(np.float32)

    return img_patch, trans


def rotate_2d(pt_2d, rot_rad):
    x = pt_2d[0]
    y = pt_2d[1]
    sn, cs = np.sin(rot_rad), np.cos(rot_rad)
    xx = x * cs - y * sn
    yy = x * sn + y * cs
    return np.array([xx, yy], dtype=np.float32)


def gen_trans_from_patch_cv(c_x, c_y, src_width, src_height, dst_width, dst_height, scale, rot, inv=False):
    # augment size with scale
    src_w = src_width * scale
    src_h = src_height * scale
    src_center = np.array([c_x, c_y], dtype=np.float32)

    # augment rotation
    rot_rad = np.pi * rot / 180
    src_downdir = rotate_2d(np.array([0, src_h * 0.5], dtype=np.float32), rot_rad)
    src_rightdir = rotate_2d(np.array([src_w * 0.5, 0], dtype=np.float32), rot_rad)

    dst_w = dst_width
    dst_h = dst_height
    dst_center = np.array([dst_w * 0.5, dst_h * 0.5], dtype=np.float32)
    dst_downdir = np.array([0, dst_h * 0.5], dtype=np.float32)
    dst_rightdir = np.array([dst_w * 0.5, 0], dtype=np.float32)

    src = np.zeros((3, 2), dtype=np.float32)
    src[0, :] = src_center
    src[1, :] = src_center + src_downdir
    src[2, :] = src_center + src_rightdir

    dst = np.zeros((3, 2), dtype=np.float32)
    dst[0, :] = dst_center
    dst[1, :] = dst_center + dst_downdir
    dst[2, :] = dst_center + dst_rightdir

    if inv:
        trans = cv2.getAffineTransform(np.float32(dst), np.float32(src))
    else:
        trans = cv2.getAffineTransform(np.float32(src), np.float32(dst))

    return trans


def trans_point2d(pt_2d, trans):
    src_pt = np.array([pt_2d[0], pt_2d[1], 1.]).T
    dst_pt = np.dot(trans, src_pt)
    return dst_pt[0:2]


def transform(img):
    # transform = transforms.Compose([transforms.ToTensor(), 
    #                                 transforms.Normalize(mean=cfg.pixel_mean, 
    #                                                      std=cfg.pixel_std)])
    img = np.array(img).astype(np.float32)
    img = img.transpose(2, 0, 1)
    for rgb_i in range(3):
        img[rgb_i, :, :] = img[rgb_i, :, :] - cfg.pixel_mean[rgb_i]
    for rgb_i in range(3):
        img[rgb_i, :, :] = img[rgb_i, :, :] / cfg.pixel_std[rgb_i]
    return img


def posenet_to_image(original_img, bbox_list, net_root, net_pose, benchmark=False):
    # refer from [https://github.com/mks0601/3DMPPE_ROOTNET_RELEASE/blob/master/demo/demo.py]
    # refer from [https://github.com/mks0601/3DMPPE_POSENET_RELEASE/blob/master/demo/demo.py]
    # MuCo joint set
    joint_num = 21
    joints_name = ('Head_top', 'Thorax', 'R_Shoulder', 'R_Elbow', 'R_Wrist', 'L_Shoulder', 'L_Elbow', 'L_Wrist', 'R_Hip', 'R_Knee', 'R_Ankle', 'L_Hip', 'L_Knee', 'L_Ankle', 'Pelvis', 'Spine', 'Head', 'R_Hand', 'L_Hand', 'R_Toe', 'L_Toe')
    flip_pairs = ( (2, 5), (3, 6), (4, 7), (8, 11), (9, 12), (10, 13), (17, 18), (19, 20) )
    skeleton = ( (0, 16), (16, 1), (1, 15), (15, 14), (14, 8), (14, 11), (8, 9), (9, 10), (10, 19), (11, 12), (12, 13), (13, 20), (1, 2), (2, 3), (3, 4), (4, 17), (1, 5), (5, 6), (6, 7), (7, 18) )

    # prepare input image
    original_img_height, original_img_width = original_img.shape[:2]

    # prepare bbox
    person_num = len(bbox_list)

    # normalized camera intrinsics
    focal = [1500, 1500] # x-axis, y-axis
    princpt = [original_img_width/2, original_img_height/2] # x-axis, y-axis

    # for each cropped and resized human image, forward it to PoseNet
    output_pose_2d_list = []
    output_pose_3d_list = []
    for n in range(person_num):
        bbox = process_bbox(np.array(bbox_list[n]), original_img_width, original_img_height)
        img, img2bb_trans = generate_patch_image(original_img, bbox, False, 1.0, 0.0, False) 
        img = transform(img)[None,:,:,:]
        k_value = np.array([math.sqrt(cfg.bbox_real[0]*cfg.bbox_real[1]*focal[0]*focal[1]/(bbox[2]*bbox[3]))]).astype(np.float32)
        k_value = k_value[None,:]

        # inference
        if args.benchmark:
            logger.info('BENCHMARK mode')
            for i in range(5):
                start = int(round(time.time() * 1000))
                root_3d = net_root.predict([img, k_value])[0]
                end = int(round(time.time() * 1000))
                logger.info(f'\tailia processing time {end - start} ms')
        else:
            root_3d = net_root.predict([img, k_value])[0]
        root_3d = root_3d[0]

        # inference
        if args.benchmark:
            logger.info('BENCHMARK mode')
            for i in range(5):
                start = int(round(time.time() * 1000))
                pose_3d = net_pose.predict([img])[0]
                end = int(round(time.time() * 1000))
                logger.info(f'\tailia processing time {end - start} ms')
        else:
            pose_3d = net_pose.predict([img])[0]

        # inverse affine transform (restore the crop and resize)
        pose_3d = pose_3d[0]
        pose_3d[:,0] = pose_3d[:,0] / cfg.output_shape[1] * cfg.input_shape[1]
        pose_3d[:,1] = pose_3d[:,1] / cfg.output_shape[0] * cfg.input_shape[0]
        pose_3d_xy1 = np.concatenate((pose_3d[:,:2], np.ones_like(pose_3d[:,:1])),1)
        img2bb_trans_001 = np.concatenate((img2bb_trans, np.array([0,0,1]).reshape(1,3)))
        pose_3d[:,:2] = np.dot(np.linalg.inv(img2bb_trans_001), pose_3d_xy1.transpose(1,0)).transpose(1,0)[:,:2]
        output_pose_2d_list.append(pose_3d[:,:2].copy())
        
        # root-relative discretized depth -> absolute continuous depth
        pose_3d[:,2] = (pose_3d[:,2] / cfg.depth_dim * 2 - 1) * (cfg.bbox_3d_shape[0]/2) + root_3d[2]  # root_depth_list[n]
        pose_3d = pixel2cam(pose_3d, focal, princpt)
        output_pose_3d_list.append(pose_3d.copy())

    # visualize 2d poses
    vis_img_2d = original_img.copy()
    for n in range(person_num):
        x = round(bbox_list[n, 0])
        y = round(bbox_list[n, 1])
        w = round(bbox_list[n, 2])
        h = round(bbox_list[n, 3])
        cv2.rectangle(vis_img_2d, (x, y), (x + w, y + h), (0, 0, 255), 2)
        vis_kps = np.zeros((3,joint_num))
        vis_kps[0,:] = output_pose_2d_list[n][:,0]
        vis_kps[1,:] = output_pose_2d_list[n][:,1]
        vis_kps[2,:] = 1
        vis_img_2d = vis_keypoints(vis_img_2d, vis_kps, skeleton)

    # visualize 3d poses
    vis_kps = np.array(output_pose_3d_list)
    fig_h, fig_w = np.shape(original_img)[:2]
    vis_img_3d = vis_3d_multiple_skeleton(kpt_3d=vis_kps, kpt_3d_vis=np.ones_like(vis_kps), 
                                          kps_lines=skeleton, fig_h=fig_h, fig_w=fig_w)
    
    # summary result
    vis_img = np.concatenate([vis_img_2d, vis_img_3d], axis=1)

    return vis_img


def recognize_from_image(img_path, net_maskrcnn, net_root, net_pose):
    # input image loop
    for image_path in img_path:
        # prepare input data
        logger.info(image_path)
        src_img = imread(image_path)

        # load image for pposenet
        original_img = imread(image_path)

        # cast to pillow for mask r-cnn
        image = Image.fromarray(original_img.copy()[:, :, ::-1])

        # exec mask r-cnn
        bbox_list = maskrcnn_to_image(image=image, net_maskrcnn=net_maskrcnn)
        bbox_list[:, 2] = bbox_list[:, 2] - bbox_list[:, 0]
        bbox_list[:, 3] = bbox_list[:, 3] - bbox_list[:, 1]
        # print('bbox_list =\n', (bbox_list).astype(int))

        # exec posenet
        vis_img = posenet_to_image(original_img=original_img, bbox_list=bbox_list, 
                                net_root=net_root, net_pose=net_pose)

        # output image
        savepath = get_savepath(args.savepath, image_path)
        logger.info(f'saved at : {savepath}')
        cv2.imwrite(args.savepath, vis_img)

    logger.info('finished process and write result to %s!' % args.savepath)


def recognize_from_video(vid_path, net_maskrcnn, net_root, net_pose):
    # make capture
    video_capture = webcamera_utils.get_capture(args.video)

    # create video writer if savepath is specified as video format
    if args.savepath != SAVE_IMAGE_OR_VIDEO_PATH:
        f_h = int(video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        f_w = int(video_capture.get(cv2.CAP_PROP_FRAME_WIDTH)) * 2
        video_writer = webcamera_utils.get_writer(args.savepath, f_h, f_w)
    else:
        video_writer = None

    # frame read and exec segmentation
    frame_shown = False
    while(True):
        # frame read
        ret, original_img = video_capture.read()

        if (cv2.waitKey(1) & 0xFF == ord('q')) or not ret:
            break
        if frame_shown and cv2.getWindowProperty('frame', cv2.WND_PROP_VISIBLE) == 0:
            break

        # cast to pillow for mask r-cnn
        image = Image.fromarray(original_img.copy()[:, :, ::-1])

        # exec mask r-cnn
        bbox_list = maskrcnn_to_image(image=image, net_maskrcnn=net_maskrcnn)
        bbox_list[:, 2] = bbox_list[:, 2] - bbox_list[:, 0]
        bbox_list[:, 3] = bbox_list[:, 3] - bbox_list[:, 1]

        # exec posenet
        vis_img = posenet_to_image(original_img=original_img, bbox_list=bbox_list, 
                                   net_root=net_root, net_pose=net_pose)
        
        # display
        cv2.imshow("frame", vis_img)
        frame_shown = True

        # write a frame image to video
        if video_writer is not None:
            video_writer.write(vis_img)

    video_capture.release()
    cv2.destroyAllWindows()
    if video_writer is not None:
        video_writer.release()

    logger.info('finished process and write result to %s!' % args.savepath)


def main():
    # model files check and download
    check_and_download_models(WEIGHT_PATH_MASKRCNN, MODEL_PATH_MASKRCNN, REMOTE_PATH_MASKRCNN)
    check_and_download_models(WEIGHT_PATH_ROOTNET, MODEL_PATH_ROOTNET, REMOTE_PATH_ROOTNET)
    check_and_download_models(WEIGHT_PATH_POSENET, MODEL_PATH_POSENET, REMOTE_PATH_POSENET)

    # net initialize
    # Mask R-CNN
    net_maskrcnn = ailia.Net(MODEL_PATH_MASKRCNN, WEIGHT_PATH_MASKRCNN, env_id=args.env_id)
    # RootNet
    net_root = ailia.Net(MODEL_PATH_ROOTNET, WEIGHT_PATH_ROOTNET, env_id=args.env_id)
    # PoseNet
    net_pose = ailia.Net(MODEL_PATH_POSENET, WEIGHT_PATH_POSENET, env_id=args.env_id)

    if args.video is None:
        # image mode
        recognize_from_image(img_path=args.input, net_maskrcnn=net_maskrcnn, 
                             net_root=net_root, net_pose=net_pose)
    else:
        # video mode
        recognize_from_video(vid_path=args.video, net_maskrcnn=net_maskrcnn, 
                             net_root=net_root, net_pose=net_pose)


if __name__ == '__main__':
    main()
