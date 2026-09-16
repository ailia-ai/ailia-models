import sys, os
import time
import json

import numpy as np
import cv2
from PIL import Image, ImageDraw

import ailia

# import original modules
sys.path.append('../../util')
from arg_utils import get_base_parser, update_parser, get_savepath  # noqa: E402
from model_utils import check_and_download_models  # noqa: E402
from detector_utils import load_image  # noqa: E402
import webcamera_utils  # noqa: E402

# logger
from logging import getLogger  # noqa: E402

from mmfashion_tryon_utils import *
from pose_utils import *
from hps_utils import *

logger = getLogger(__name__)

# ======================
# Parameters
# ======================
WEIGHT_GMM_PATH = './GMM_epoch_40.onnx'
MODEL_GMM_PATH = './GMM_epoch_40.onnx.prototxt'
WEIGHT_TOM_PATH = './TOM_epoch_40.onnx'
MODEL_TOM_PATH = './TOM_epoch_40.onnx.prototxt'
REMOTE_PATH = 'https://storage.googleapis.com/ailia-models/mmfashion_tryon/'

WEIGHT_YOLOV3_PATH = 'yolov3.opt2.onnx'
MODEL_YOLOV3_PATH = 'yolov3.opt2.onnx.prototxt'
REMOTE_YOLOV3_PATH = 'https://storage.googleapis.com/ailia-models/yolov3/'

WEIGHT_POSE_PATH = 'pose_resnet_50_256x192.onnx'
MODEL_POSE_PATH = 'pose_resnet_50_256x192.onnx.prototxt'
REMOTE_POSE_PATH = 'https://storage.googleapis.com/ailia-models/pose_resnet/'

WEIGHT_SEG_PATH = 'resnet-lip.onnx'
MODEL_SEG_PATH = 'resnet-lip.onnx.prototxt'
REMOTE_SEG_PATH = 'https://storage.googleapis.com/ailia-models/human_part_segmentation/'

IMAGE_CLOTH_PATH = 'cloth/019029_1.jpg'
IMAGE_PERSON_PATH = 'image/000320_0.jpg'
SAVE_IMAGE_PATH = 'output.png'

IMAGE_HEIGHT = 256
IMAGE_WIDTH = 192
IMAGE_POSE_HEIGTH = 256
IMAGE_POSE_WIDTH = 192
IMAGE_LIP_SIZE = 473

LIP_NORM_MEAN = [0.406, 0.456, 0.485]
LIP_NORM_STD = [0.225, 0.224, 0.229]

# ======================
# Arguemnt Parser Config
# ======================

parser = get_base_parser('MMFashion Virtual Try-on model', IMAGE_CLOTH_PATH, SAVE_IMAGE_PATH)
parser.add_argument(
    '-p', '--person', metavar='PERSON_IMAGE', default=IMAGE_PERSON_PATH,
    help='Image of person.'
)
parser.add_argument(
    '-pp', '--parse', metavar='PARSE_IMAGE/DIR', default=None,
    help='Parsed image of person image. If a directory name is specified, '
         'Search for a png file with the same name as the person image file'
)
parser.add_argument(
    '-k', '--keypoints', metavar='KEYPOINT_FILE/DIR', default=None,
    help='Keypoints json file. If a directory name is specified, '
         'find a json file with the same name as the person image file'
)
parser.add_argument(
    '--onnx',
    action='store_true',
    help='execute onnxruntime version.'
)
args_input = parser.parse_args().input
args = update_parser(parser)


# ======================
# Secondaty Functions
# ======================

def preprocess(img):
    mean = np.array((0.5,) * img.shape[2])
    std = np.array((0.5,) * img.shape[2])

    img = img / 255
    img = (img - mean) / std
    img = img.transpose(2, 0, 1)  # HWC -> CHW

    return img.astype(np.float32)


def post_processing(data):
    data = (data + 1) * 0.5 * 255
    data = np.clip(data, 0, 255)
    data = data.transpose(1, 2, 0)  # CHW -> HWC

    img = data.astype(np.uint8)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    return img


def human_detect(det_net, img):
    h, w = img.shape[:2]
    THRESHOLD = 0.4
    IOU = 0.45
    CATEGORY_PERSON = 0

    # detect bbox
    det_net.compute(img, THRESHOLD, IOU)
    count = det_net.get_object_count()
    a = sorted([
        det_net.get_object(i) for i in range(count)
    ], key=lambda x: x.prob, reverse=True)
    bbox = [
        (w * obj.x, h * obj.y, w * obj.w, h * obj.h)
        for obj in a if obj.category == CATEGORY_PERSON
    ]
    if 0 < len(bbox):
        bbox = bbox[0]
    else:
        return img, (0, 0), (1, 1)

    # adjust bbox
    x0 = bbox[0]
    y0 = bbox[1]
    x1 = bbox[0] + bbox[2]
    y1 = bbox[0] + bbox[3]
    x0, y0, x1, y1 = keep_aspect(
        x0, y0, x1, y1, h, w, IMAGE_POSE_HEIGTH / IMAGE_POSE_WIDTH
    )
    img = img[y0:y1, x0:x1, :]

    offset_x = x0 / w
    offset_y = y0 / h
    scale_x = img.shape[1] / w
    scale_y = img.shape[0] / h

    return img, (offset_x, offset_y), (scale_x, scale_y)


def pose_estimation(pose_net, img):
    img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
    img = cv2.resize(img, (IMAGE_POSE_WIDTH, IMAGE_POSE_HEIGTH))

    # BGR format
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    img = (img / 255.0 - mean) / std
    img = img.transpose(2, 0, 1)  # HWC -> CHW
    img = np.expand_dims(img, axis=0)

    output = pose_net.predict(img)

    center = np.array([IMAGE_POSE_WIDTH / 2, IMAGE_POSE_HEIGTH / 2], dtype=np.float32)
    scale = np.array([1, 1], dtype=np.float32)
    preds, maxvals = get_final_preds(output, [center], [scale])

    ailia_to_mpi = [
        0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, -1, -1
    ]
    pose = []
    for j in range(ailia.POSE_KEYPOINT_CNT):
        i = ailia_to_mpi[j]
        if j == ailia.POSE_KEYPOINT_BODY_CENTER:
            x = (preds[0, ailia_to_mpi[ailia.POSE_KEYPOINT_SHOULDER_LEFT], 0] +
                 preds[0, ailia_to_mpi[ailia.POSE_KEYPOINT_SHOULDER_RIGHT], 0] +
                 preds[0, ailia_to_mpi[ailia.POSE_KEYPOINT_HIP_LEFT], 0] +
                 preds[0, ailia_to_mpi[ailia.POSE_KEYPOINT_HIP_RIGHT], 0]) / 4
            y = (preds[0, ailia_to_mpi[ailia.POSE_KEYPOINT_SHOULDER_LEFT], 1] +
                 preds[0, ailia_to_mpi[ailia.POSE_KEYPOINT_SHOULDER_RIGHT], 1] +
                 preds[0, ailia_to_mpi[ailia.POSE_KEYPOINT_HIP_LEFT], 1] +
                 preds[0, ailia_to_mpi[ailia.POSE_KEYPOINT_HIP_RIGHT], 1]) / 4
        elif j == ailia.POSE_KEYPOINT_SHOULDER_CENTER:
            x = (preds[0, ailia_to_mpi[ailia.POSE_KEYPOINT_SHOULDER_LEFT], 0] +
                 preds[0, ailia_to_mpi[ailia.POSE_KEYPOINT_SHOULDER_RIGHT], 0]) / 2
            y = (preds[0, ailia_to_mpi[ailia.POSE_KEYPOINT_SHOULDER_LEFT], 1] +
                 preds[0, ailia_to_mpi[ailia.POSE_KEYPOINT_SHOULDER_RIGHT], 1]) / 2
        else:
            x = preds[0, i, 0]
            y = preds[0, i, 1]

        pose.append([
            x / IMAGE_POSE_WIDTH,
            y / IMAGE_POSE_HEIGTH,
            0
        ])

    pose = np.array([
        pose[0],  # NOSE
        pose[17],  # SHOULDER_CENTER
        pose[6],  # SHOULDER_RIGHT
        pose[8],  # ELBOW_RIGHT
        pose[10],  # WRIST_RIGHT
        pose[5],  # SHOULDER_LEFT
        pose[7],  # ELBOW_LEFT
        pose[9],  # WRIST_LEFT
        pose[12],  # HIP_RIGHT
        # pose[14],   # KNEE_RIGHT
        # pose[16],   # ANKLE_RIGHT
        [0, 0, 0],
        [0, 0, 0],
        pose[11],  # HIP_LEFT
        # pose[13],   # KNEE_LEFT
        # pose[15],   # ANKLE_LEFT
        [0, 0, 0],
        [0, 0, 0],
        pose[2],  # EYE_RIGHT
        pose[1],  # EYE_LEFT
        pose[4],  # EAR_RIGHT
        pose[3],  # EAR_LEFT
    ])

    return pose


def human_seg(seg_net, img):
    h, w, _ = img.shape

    img_size = (IMAGE_LIP_SIZE, IMAGE_LIP_SIZE)

    # Get person center and scale
    center, s = xywh2cs(0, 0, w - 1, h - 1)
    r = 0
    trans = get_affine_transform(
        center, s, r, img_size
    )
    img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
    img = cv2.warpAffine(
        img,
        trans,
        (img_size[1], img_size[0]),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(0, 0, 0))

    # normalize
    img = ((img / 255.0 - LIP_NORM_MEAN) / LIP_NORM_STD).astype(np.float32)

    img = img.transpose(2, 0, 1)
    img = np.expand_dims(img, 0)

    # feedforward
    output = seg_net.predict([img])
    _, fusion, _ = output

    fusion = fusion[0].transpose(1, 2, 0)
    upsample_output = cv2.resize(
        fusion, img_size, interpolation=cv2.INTER_LINEAR
    )
    parse = transform_logits(
        upsample_output,
        center, s, w, h,
        input_size=img_size
    )

    parse = np.argmax(parse, axis=2)

    return parse


# ======================
# Main functions
# ======================

def cloth_agnostic(pose_net, seg_net, img):
    fine_height = IMAGE_HEIGHT
    fine_width = IMAGE_WIDTH
    radius = 5

    person_path = args.person
    name = os.path.splitext(os.path.basename(person_path))[0]

    img = cv2.resize(
        img, (fine_width, fine_height), interpolation=cv2.INTER_LINEAR
    )

    if pose_net:
        pose_data = pose_estimation(pose_net, img)
        pose_data = pose_data * [fine_width, fine_height, 1]
    else:
        pose_path = (os.path.join(args.keypoints, '%s_keypoints.json' % name) \
                         if os.path.isdir(args.keypoints) else args.keypoints) \
            if args.keypoints else '%s_keypoints.json' % name

        # load pose points
        with open(pose_path, 'r') as f:
            pose_label = json.load(f)
            pose_data = pose_label['people'][0]['pose_keypoints']
            pose_data = np.array(pose_data)
            pose_data = pose_data.reshape((-1, 3))

    if seg_net:
        im_parse = human_seg(seg_net, img)
        head_ids = [1, 2, 4, 13]
    else:
        parse_path = (os.path.join(args.parse, '%s.png' % name) \
                          if os.path.isdir(args.parse) else args.parse) \
            if args.parse else '%s_parse.png' % name

        im_parse = np.array(Image.open(parse_path))
        head_ids = [1, 2, 4, 13]

    # person image
    img = cv2.cvtColor(img, cv2.COLOR_BGRA2RGB)
    img = preprocess(img)

    # parsing image
    parse_shape = (im_parse > 0).astype(np.float32)
    phead = sum([
        (im_parse == i).astype(np.float32) for i in head_ids
    ])

    # shape downsample
    parse_shape = (parse_shape * 255).astype(np.uint8)
    parse_shape = Image.fromarray(parse_shape)
    parse_shape = parse_shape.resize(
        (fine_width // 16, fine_height // 16), Image.BILINEAR)
    parse_shape = parse_shape.resize(
        (fine_width, fine_height), Image.BILINEAR)
    parse_shape = np.expand_dims(np.asarray(parse_shape), axis=2)
    shape = preprocess(parse_shape)

    # upper cloth
    im_h = img * phead - (1 - phead)  # [-1,1], fill 0 for other parts

    point_num = pose_data.shape[0]
    pose_map = np.zeros((point_num, fine_height, fine_width), dtype=np.float32)
    r = radius
    im_pose = Image.new('L', (fine_width, fine_height))
    pose_draw = ImageDraw.Draw(im_pose)
    for i in range(point_num):
        one_map = Image.new('L', (fine_width, fine_height))
        draw = ImageDraw.Draw(one_map)
        pointx = pose_data[i, 0]
        pointy = pose_data[i, 1]
        if pointx > 1 and pointy > 1:
            draw.rectangle(
                (pointx - r, pointy - r, pointx + r, pointy + r), 'white',
                'white')
            pose_draw.rectangle(
                (pointx - r, pointy - r, pointx + r, pointy + r), 'white',
                'white')
        one_map = np.expand_dims(np.asarray(one_map), axis=2)
        one_map = preprocess(one_map)
        pose_map[i] = one_map[0]

    agnostic = np.vstack([shape, im_h, pose_map])

    return agnostic


def predict(GMM_net, TOM_net, cloth, agnostic):
    if not args.onnx:
        output = GMM_net.predict({
            'cloth': cloth, 'agnostic': agnostic
        })
    else:
        in0 = GMM_net.get_inputs()[0].name
        in1 = GMM_net.get_inputs()[1].name
        out0 = GMM_net.get_outputs()[0].name
        output = GMM_net.run(
            [out0],
            {in0: cloth, in1: agnostic})

    grid = output[0]
    warped_cloth = grid_sample(cloth, grid, padding_mode='border')

    if not args.onnx:
        output = TOM_net.predict({
            'cloth': warped_cloth, 'agnostic': agnostic
        })
    else:
        in0 = TOM_net.get_inputs()[0].name
        in1 = TOM_net.get_inputs()[1].name
        out0 = TOM_net.get_outputs()[0].name
        output = TOM_net.run(
            [out0],
            {in0: warped_cloth, in1: agnostic})

    tryon = output[0]

    return tryon, warped_cloth


def recognize_from_image(GMM_net, TOM_net, det_net, pose_net, seg_net):
    img = load_image(args.person)

    if det_net:
        img, offset, scale = human_detect(det_net, img)

    agnostic = cloth_agnostic(pose_net, seg_net, img)
    agnostic = np.expand_dims(agnostic, axis=0)

    # input image loop
    for image_path in args.input:
        logger.info(image_path)

        # prepare cloth image
        img = load_image(image_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2RGB)
        img = cv2.resize(
            img, (IMAGE_WIDTH, IMAGE_HEIGHT), interpolation=cv2.INTER_LINEAR
        )
        img = preprocess(img)
        img = np.expand_dims(img, axis=0)

        logger.debug(f'input image shape: {img.shape}')

        # inference
        logger.info('Start inference...')
        if args.benchmark:
            logger.info('BENCHMARK mode')
            total_time = 0
            for i in range(args.benchmark_count):
                start = int(round(time.time() * 1000))
                output = predict(GMM_net, TOM_net, img, agnostic)
                end = int(round(time.time() * 1000))
                logger.info(f'\tailia processing time {end - start} ms')
                if i != 0:
                    total_time = total_time + (end - start)
            logger.info(f'\taverage time {total_time / (args.benchmark_count - 1)} ms')
        else:
            output = predict(GMM_net, TOM_net, img, agnostic)

        tryon, warped_cloth = output
        tryon = post_processing(tryon[0])
        warped_cloth = post_processing(warped_cloth[0])

        savepath = get_savepath(args.savepath, image_path, ext='.png')
        logger.info(f'saved at : {savepath}')
        cv2.imwrite(savepath, tryon)

        savepath_warp = '%s-warp-cloth%s' % os.path.splitext(savepath)
        logger.info(f'saved at : {savepath_warp}')
        cv2.imwrite(savepath_warp, warped_cloth)

    logger.info('Script finished successfully.')


def recognize_from_video(GMM_net, TOM_net, det_net, pose_net, seg_net):
    capture = webcamera_utils.get_capture(args.video)

    # create video writer if savepath is specified as video format
    if args.savepath != SAVE_IMAGE_PATH:
        f_h = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        f_w = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        writer = webcamera_utils.get_writer(args.savepath, f_h, f_w)
    else:
        writer = None

    # prepare cloth image
    if type(args_input) == list:
        image_path = args_input[0]
    else:
        image_path = args_input
    img = load_image(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGRA2RGB)
    img = cv2.resize(
        img, (IMAGE_WIDTH, IMAGE_HEIGHT), interpolation=cv2.INTER_LINEAR
    )
    img = preprocess(img)
    cloth_img = np.expand_dims(img, axis=0)

    dummy = np.zeros(IMAGE_HEIGHT * IMAGE_WIDTH).reshape(IMAGE_HEIGHT, IMAGE_WIDTH)
    
    frame_shown = False
    while (True):
        ret, frame = capture.read()
        if (cv2.waitKey(1) & 0xFF == ord('q')) or not ret:
            break
        if frame_shown and cv2.getWindowProperty('frame', cv2.WND_PROP_VISIBLE) == 0:
            break

        # inference
        img = cv2.cvtColor(frame, cv2.COLOR_BGR2BGRA)
        img, offset, scale = human_detect(det_net, img)
        if offset == (0, 0):
            # human is not detected
            cv2.imshow('frame', dummy)
            frame_shown = True
            continue

        agnostic = cloth_agnostic(pose_net, seg_net, img)
        agnostic = np.expand_dims(agnostic, axis=0)

        output = predict(GMM_net, TOM_net, cloth_img, agnostic)

        tryon, _ = output
        tryon = post_processing(tryon[0])

        cv2.imshow('frame', tryon)
        frame_shown = True

        # save results
        if writer is not None:
            writer.write(frame)

    capture.release()
    cv2.destroyAllWindows()
    if writer is not None:
        writer.release()

    logger.info('Script finished successfully.')


def main():
    # model files check and download
    logger.info('=== GMM model ===')
    check_and_download_models(WEIGHT_GMM_PATH, MODEL_GMM_PATH, REMOTE_PATH)
    logger.info('=== TOM model ===')
    check_and_download_models(WEIGHT_TOM_PATH, MODEL_TOM_PATH, REMOTE_PATH)
    if args.video or not args.keypoints:
        logger.info('=== detector model ===')
        check_and_download_models(WEIGHT_YOLOV3_PATH, MODEL_YOLOV3_PATH, REMOTE_YOLOV3_PATH)
        logger.info('=== pose model ===')
        check_and_download_models(WEIGHT_POSE_PATH, MODEL_POSE_PATH, REMOTE_POSE_PATH)
    if args.video or not args.parse:
        logger.info('=== human segmentation model ===')
        check_and_download_models(WEIGHT_SEG_PATH, MODEL_SEG_PATH, REMOTE_SEG_PATH)

    # initialize
    if args.onnx:
        import onnxruntime
        GMM_net = onnxruntime.InferenceSession(WEIGHT_GMM_PATH)
        TOM_net = onnxruntime.InferenceSession(WEIGHT_TOM_PATH)
    else:
        GMM_net = ailia.Net(MODEL_GMM_PATH, WEIGHT_GMM_PATH, env_id=args.env_id, memory_mode=args.memory_mode)
        TOM_net = ailia.Net(MODEL_TOM_PATH, WEIGHT_TOM_PATH, env_id=args.env_id, memory_mode=args.memory_mode)

    if args.video or not args.keypoints:
        det_net = ailia.Detector(
            MODEL_YOLOV3_PATH,
            WEIGHT_YOLOV3_PATH,
            80,
            format=ailia.NETWORK_IMAGE_FORMAT_RGB,
            channel=ailia.NETWORK_IMAGE_CHANNEL_FIRST,
            range=ailia.NETWORK_IMAGE_RANGE_U_FP32,
            algorithm=ailia.DETECTOR_ALGORITHM_YOLOV3,
            env_id=args.env_id,
            memory_mode=args.memory_mode,
        )
        pose_net = ailia.Net(
            MODEL_POSE_PATH, WEIGHT_POSE_PATH, env_id=args.env_id, memory_mode=args.memory_mode)
    else:
        det_net = pose_net = None
    if args.video or not args.parse:
        seg_net = ailia.Net(MODEL_SEG_PATH, WEIGHT_SEG_PATH, env_id=args.env_id, memory_mode=args.memory_mode)
    else:
        seg_net = None

    if args.video is not None:
        # video mode
        recognize_from_video(GMM_net, TOM_net, det_net, pose_net, seg_net)
    else:
        # image mode
        recognize_from_image(GMM_net, TOM_net, det_net, pose_net, seg_net)


if __name__ == '__main__':
    main()
