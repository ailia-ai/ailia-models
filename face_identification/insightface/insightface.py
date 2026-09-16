import sys
import os
import time
import glob
from collections import namedtuple

import numpy as np
from numpy.linalg import norm
import cv2

import ailia

# import original modules
sys.path.append('../../util')
from arg_utils import get_base_parser, update_parser, get_savepath  # noqa: E402
from model_utils import check_and_download_models  # noqa: E402
from detector_utils import load_image  # noqa: E402
import webcamera_utils  # noqa: E402
from insightface_utils import PriorBox, decode, decode_landm, nms, \
    face_align_norm_crop, draw_detection  # noqa: E402

# logger
from logging import getLogger   # noqa: E402
logger = getLogger(__name__)


# ======================
# Parameters
# ======================
WEIGHT_DET_PATH = 'retinaface_resnet.onnx'
MODEL_DET_PATH = 'retinaface_resnet.onnx.prototxt'
WEIGHT_REC_R100_PATH = 'arcface_r100_v1.onnx'
MODEL_REC_R100_PATH = 'arcface_r100_v1.onnx.prototxt'
WEIGHT_REC_R50_PATH = 'arcface_r50_v1.onnx'
MODEL_REC_R50_PATH = 'arcface_r50_v1.onnx.prototxt'
WEIGHT_REC_R34_PATH = 'arcface_r34_v1.onnx'
MODEL_REC_R34_PATH = 'arcface_r34_v1.onnx.prototxt'
WEIGHT_REC_MF_PATH = 'arcface_mobilefacenet.onnx'
MODEL_REC_MF_PATH = 'arcface_mobilefacenet.onnx.prototxt'
WEIGHT_GA_PATH = 'genderage_v1.onnx'
MODEL_GA_PATH = 'genderage_v1.onnx.prototxt'
REMOTE_PATH = \
    'https://storage.googleapis.com/ailia-models/insightface/'

IMAGE_PATH = 'demo.jpg'
SAVE_IMAGE_PATH = 'output.png'

IMAGE_SIZE = 512

Face = namedtuple('Face', [
    'category', 'prob', 'cosin_metric',
    'landmark', 'x', 'y', 'w', 'h',
    'embedding', 'gender', 'age'
])


# ======================
# Arguemnt Parser Config
# ======================
parser = get_base_parser('InsightFace model', IMAGE_PATH, SAVE_IMAGE_PATH)
parser.add_argument(
    '--det_thresh', type=float, default=0.02,
    help='det_thresh'
)
parser.add_argument(
    '--nms_thresh', type=float, default=0.4,
    help='nms_thresh'
)
parser.add_argument(
    '--ident_thresh', type=float, default=0.25572845,
    help='ident_thresh'
)
parser.add_argument(
    '--top_k', type=int, default=5000,
    help='top_k'
)
parser.add_argument(
    '-r', '--rec_model', type=str, default='resnet100',
    choices=('resnet100', 'resnet50', 'resnet34', 'mobileface'),
    help='recognition model'
)
args = update_parser(parser)


# ======================
# Secondaty Functions
# ======================
def preprocess(img):
    img = np.float32(img)
    img -= (104, 117, 123)
    img = img.transpose(2, 0, 1)
    img = np.expand_dims(img, axis=0)
    return img


def post_processing(im_height, im_width, loc, conf, landms):
    cfg_re50 = {
        'min_sizes': [[16, 32], [64, 128], [256, 512]],
        'steps': [8, 16, 32],
        'variance': [0.1, 0.2],
        'clip': False,
    }

    priorbox = PriorBox(cfg_re50, image_size=(im_height, im_width))
    priors = priorbox.forward()
    boxes = decode(loc[0], priors, cfg_re50['variance'])

    scale = np.array([im_width, im_height, im_width, im_height])
    boxes = boxes * scale
    scores = conf[0][:, 1]

    landms = decode_landm(landms[0], priors, cfg_re50['variance'])
    scale1 = np.array([
        im_width, im_height, im_width, im_height,
        im_width, im_height, im_width, im_height,
        im_width, im_height
    ])
    landms = landms * scale1

    inds = np.where(scores > args.det_thresh)[0]
    boxes = boxes[inds]
    landms = landms[inds]
    scores = scores[inds]

    # keep top-K before NMS
    order = scores.argsort()[::-1][:args.top_k]
    boxes = boxes[order]
    landms = landms[order]
    scores = scores[order]

    dets = np.hstack(
        (boxes, scores[:, np.newaxis])
    ).astype(np.float32, copy=False)
    keep = nms(dets, args.nms_thresh)
    dets = dets[keep, :]
    landms = landms[keep]

    return dets, landms


def face_identification(faces, ident_feats):
    ident_faces = []
    for i in range(len(faces)):
        face = faces[i]
        emb = face.embedding
        metrics = ident_feats.dot(emb)
        category = np.argmax(metrics)
        face = face._replace(cosin_metric=metrics[category])
        if args.ident_thresh <= face.cosin_metric:
            face = face._replace(category=category)

        ident_faces.append(face)

    return ident_faces


def load_identities(rec_model):
    names = []
    feats = []
    for path in glob.glob("identities/*.PNG"):
        name = ".".join(
            path.replace(os.sep, '/').split('/')[-1].split('.')[:-1]
        )
        names.append(name)

        img = load_image(path)
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2RGB)
        img = np.transpose(img, (2, 0, 1))
        img = np.expand_dims(img, axis=0)
        img = img.astype(np.float32)

        output = rec_model.predict({'data': img})[0]

        embedding = output[0]
        embedding_norm = norm(embedding)
        normed_embedding = embedding / embedding_norm
        feats.append(normed_embedding)

    feats = np.vstack(feats)

    return names, feats


# ======================
# Main functions
# ======================
def predict(img, det_model, rec_model, ga_model):
    # initial preprocesses
    im_height, im_width, _ = img.shape
    _img = preprocess(img)

    # feedforward
    output = det_model.predict({'img': _img})

    loc, conf, landms = output

    bboxes, landmarks = post_processing(im_height, im_width, loc, conf, landms)

    faces = []
    for i in range(bboxes.shape[0]):
        bbox = bboxes[i, 0:4]
        prob = bboxes[i, 4]
        landmark = landmarks[i].reshape((5, 2))

        _img = face_align_norm_crop(img, landmark=landmark)
        _img = cv2.cvtColor(_img, cv2.COLOR_BGR2RGB)
        _img = np.transpose(_img, (2, 0, 1))
        _img = np.expand_dims(_img, axis=0)
        _img = _img.astype(np.float32)
        output = rec_model.predict({'data': _img})[0]

        embedding = output[0]
        embedding_norm = norm(embedding)
        normed_embedding = embedding / embedding_norm

        output = ga_model.predict({'data': _img})[0]

        g = output[0, 0:2]
        gender = np.argmax(g)
        a = output[0, 2:202].reshape((100, 2))
        a = np.argmax(a, axis=1)
        age = int(sum(a))

        face = Face(
            category=None,
            prob=prob,
            cosin_metric=1,
            landmark=landmark,
            x=bbox[0] / im_width,
            y=bbox[1] / im_height,
            w=(bbox[2] - bbox[0]) / im_width,
            h=(bbox[3] - bbox[1]) / im_height,
            embedding=normed_embedding,
            gender=gender,
            age=age
        )
        faces.append(face)

    return faces


def recognize_from_image(filename, det_model, rec_model, ga_model):
    # prepare input data
    img = load_image(filename)
    logger.debug(f'input image shape: {img.shape}')

    # load identities
    ident_names, ident_feats = load_identities(rec_model)

    img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)

    # inference
    logger.info('Start inference...')
    if args.benchmark:
        logger.info('BENCHMARK mode')
        for i in range(5):
            start = int(round(time.time() * 1000))
            faces = predict(img, det_model, rec_model, ga_model)
            end = int(round(time.time() * 1000))
            logger.info(f'\tailia processing time {end - start} ms')
    else:
        faces = predict(img, det_model, rec_model, ga_model)

    faces = face_identification(faces, ident_feats)

    # plot and save result
    res_img = draw_detection(img, faces, ident_names)
    savepath = get_savepath(args.savepath, filename)
    logger.info(f'saved at : {savepath}')
    cv2.imwrite(savepath, res_img)

    # print result
    for i, face in enumerate(faces):
        print(f'Detected face: {i}')
        if face.category is not None:
            print(f'- name: {ident_names[face.category]}')
        else:
            print(f'- name: ?')
        print(f'- cosin_metric: {face.cosin_metric}')
        print(f'- prob: {face.prob}')
        print(f'- pos: ({face.x}, {face.y})')
        print(f'- size: ({face.w}, {face.h})')
        print(f'- age: {face.age}')
        print(f'- gender: {face.gender}')


def recognize_from_video(video, det_model, rec_model, ga_model):
    capture = webcamera_utils.get_capture(video)

    # create video writer if savepath is specified as video format
    if args.savepath != SAVE_IMAGE_PATH:
        f_h = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        f_w = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        writer = webcamera_utils.get_writer(args.savepath, f_h, f_w)
    else:
        writer = None

    # load identities
    ident_names, ident_feats = load_identities(rec_model)

    frame_shown = False
    while (True):
        ret, frame = capture.read()
        if (cv2.waitKey(1) & 0xFF == ord('q')) or not ret:
            break
        if frame_shown and cv2.getWindowProperty('frame', cv2.WND_PROP_VISIBLE) == 0:
            break

        faces = predict(frame, det_model, rec_model, ga_model)
        faces = face_identification(faces, ident_feats)

        # plot result
        res_img = draw_detection(frame, faces, ident_names)

        # show
        cv2.imshow('frame', res_img)
        frame_shown = True

        # save results
        if writer is not None:
            writer.write(res_img)

    capture.release()
    cv2.destroyAllWindows()
    if writer is not None:
        writer.release()


def main():
    rec_model = {
        'resnet100': (WEIGHT_REC_R100_PATH, MODEL_REC_R100_PATH),
        'resnet50': (WEIGHT_REC_R50_PATH, MODEL_REC_R50_PATH),
        'resnet34': (WEIGHT_REC_R34_PATH, MODEL_REC_R34_PATH),
        'mobileface': (WEIGHT_REC_MF_PATH, MODEL_REC_MF_PATH),
    }
    WEIGHT_REC_PATH, MODEL_REC_PATH = rec_model[args.rec_model]

    # model files check and download
    logger.info("=== DET model ===")
    check_and_download_models(WEIGHT_DET_PATH, MODEL_DET_PATH, REMOTE_PATH)
    logger.info("=== REC model ===")
    check_and_download_models(WEIGHT_REC_PATH, MODEL_REC_PATH, REMOTE_PATH)
    logger.info("=== GA model ===")
    check_and_download_models(WEIGHT_GA_PATH, MODEL_GA_PATH, REMOTE_PATH)

    # initialize
    det_model = ailia.Net(MODEL_DET_PATH, WEIGHT_DET_PATH, env_id=args.env_id, memory_mode=args.memory_mode)
    rec_model = ailia.Net(MODEL_REC_PATH, WEIGHT_REC_PATH, env_id=args.env_id, memory_mode=args.memory_mode)
    ga_model = ailia.Net(MODEL_GA_PATH, WEIGHT_GA_PATH, env_id=args.env_id, memory_mode=args.memory_mode)

    if args.video is not None:
        recognize_from_video(args.video, det_model, rec_model, ga_model)
    else:
        # input image loop
        for image_path in args.input:
            # prepare input data
            logger.info(image_path)
            recognize_from_image(image_path, det_model, rec_model, ga_model)
    logger.info('Script finished successfully.')


if __name__ == '__main__':
    main()
