import os
import sys
import time
import pickle

import numpy as np
import cv2
from PIL import Image
from sklearn.covariance import LedoitWolf
from scipy.spatial.distance import mahalanobis
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning) 

import ailia

# import original modules
sys.path.append('../../util')
from arg_utils import get_base_parser, update_parser, get_savepath  # noqa: E402
from model_utils import check_and_download_models  # noqa: E402
from detector_utils import load_image  # noqa: E402
from image_utils import normalize_image  # noqa: E402
import webcamera_utils  # noqa: E402
# logger
from logging import getLogger  # noqa: E402

logger = getLogger(__name__)


# ======================
# Parameters
# ======================
IMAGE_PATH = './bottle_test_broken_large_000.png'
SAVE_IMAGE_PATH = './output.png'
IMAGE_RESIZE = 256
IMAGE_SIZE = 224
KEEP_ASPECT = True


# ======================
# Arguemnt Parser Config
# ======================
parser = get_base_parser('MahalanobisAD', IMAGE_PATH, SAVE_IMAGE_PATH)
parser.add_argument(
    '-m', '--model', metavar='MODEL',
    default='b4', choices=['b0', 'b1', 'b2', 'b3', 'b4', 'b5', 'b6', 'b7'],
    help='The input model path.' +
         'you can set b0 ~ b7 to select efficientnet-b0 ~ efficientnet-b7'
)
parser.add_argument(
    '-f', '--feat', metavar="PICKLE_FILE", default=None,
    help='train set feature pkl files.'
)
parser.add_argument(
    '-bs', '--batch_size', default=32,
    help='batch size.'
)
parser.add_argument(
    '-tr', '--train_dir', metavar="DIR", default="./train",
    help='directory of the train files.'
)
parser.add_argument(
    '-th', '--threshold', type=float, default=None,
    help='threshold'
)
parser.add_argument(
    '-a', '--aug', action='store_true',
    help='process with augmentation.'
)
parser.add_argument(
    '-an', '--aug_num', type=int, default=5,
    help='specify the amplification number of augmentation.'
)
args = update_parser(parser)


# ==========================
# Model PARAMETERS
# ==========================
MODEL_PATH = "efficientnet-" + args.model + ".onnx.prototxt"
WEIGHT_PATH = "efficientnet-" + args.model + ".onnx"
REMOTE_PATH = "https://storage.googleapis.com/ailia-models/mahalanobisad/"


# ======================
# Main functions
# ======================
def preprocess(img, keep_aspect=True):
    h, w = img.shape[:2]
    size = IMAGE_RESIZE
    crop_size = IMAGE_SIZE

    # resize
    if keep_aspect:
        if h > w:
            size = (size, int(size * h / w))
        else:
            size = (int(size * w / h), size)
    else:
        size = (size, size)
    img = np.array(Image.fromarray(img).resize(size, resample=Image.LANCZOS))

    # center crop
    h, w = img.shape[:2]
    pad_h = (h - crop_size) // 2
    pad_w = (w - crop_size) // 2
    img = img[pad_h:(pad_h + crop_size), pad_w:(pad_w + crop_size)]

    # normalize
    img = normalize_image(img.astype(np.float32), 'ImageNet')
    img = img.transpose(2, 0, 1)  # HWC -> CHW
    img = img[None]

    return img


def preprocess_aug(img, keep_aspect=True, angle_range=[-10, 10]):
    h, w = img.shape[:2]
    size = IMAGE_RESIZE
    crop_size = IMAGE_SIZE

    # resize
    if keep_aspect:
        if h > w:
            size = (size, int(size * h / w))
        else:
            size = (int(size * w / h), size)
    else:
        size = (size, size)
    img = np.array(Image.fromarray(img).resize(size, resample=Image.LANCZOS))

    # random rotate
    h, w = img.shape[:2]
    angle = np.random.randint(angle_range[0], (angle_range[1] + 1))
    rot_mat = cv2.getRotationMatrix2D(((w / 2), (h / 2)), angle, 1)
    img = cv2.warpAffine(src=img,
                         M=rot_mat,
                         dsize=(w, h),
                         borderMode=cv2.BORDER_REPLICATE,
                         flags=cv2.INTER_LINEAR)

    # random crop
    h, w = img.shape[:2]
    pad_h = np.random.randint(0, (h - crop_size))
    pad_w = np.random.randint(0, (w - crop_size))
    img = img[pad_h:(pad_h + crop_size), pad_w:(pad_w + crop_size)]

    # normalize
    img = normalize_image(img.astype(np.float32), 'ImageNet')
    img = img.transpose(2, 0, 1)  # HWC -> CHW
    img = img[None]

    return img


def plot_fig(test_filename, score, threshold, savepath,
             return_img=False, test_img=None, score_max=None):
    if test_img is None:
        test_img = load_image(test_filename)
        test_img = cv2.cvtColor(test_img, cv2.COLOR_BGRA2RGB)

    plt.figure(figsize=(10, 4), dpi=100, tight_layout=True)
    plt.subplot(1, 2, 1)
    plt.imshow(test_img)
    plt.title(test_filename)
    plt.tick_params(left=False, labelleft=False,
                    bottom=False, labelbottom=False)
    plt.subplot(1, 2, 2)
    plt.bar(0, score, alpha=0.5, label='score')
    plt.title('score : %.3f' % score)
    plt.xlim([-1, 1])
    plt.plot([-1, 1], [threshold, threshold], '--', color='r',
             linewidth=3, alpha=0.5, label='threshold')
    if score_max is not None:
        plt.ylim([0, score_max * 1.1])
    plt.grid(axis='y')
    plt.legend(loc='lower right')
    plt.tick_params(bottom=False, labelbottom=False)

    plt.gcf().canvas.draw()
    # tostring_rgb was removed in matplotlib 3.10, use buffer_rgba instead
    img_figure = np.asarray(plt.gcf().canvas.buffer_rgba())[:, :, :3].copy()
    img_figure = img_figure.reshape(400, -1, 3)
    plt.close()

    if not return_img:
        cv2.imwrite(savepath, img_figure[..., ::-1])
    else:
        return img_figure[..., ::-1]


def recognize_from_image(net):
    batch_size = int(args.batch_size)

    # load train images
    train_filenames = sorted([os.path.join(args.train_dir, f)
                              for f in os.listdir(args.train_dir)
                              if (f.endswith('.png') or f.endswith('.jpg') or
                                  f.endswith('.bmp'))])
    train_imgs = []
    if args.aug:
        for f in train_filenames:
            for _ in range(args.aug_num):
                img = load_image(f)
                img = cv2.cvtColor(img, cv2.COLOR_BGRA2RGB)
                img = preprocess_aug(img, keep_aspect=KEEP_ASPECT)
                train_imgs.append(img)
    else:
        for f in train_filenames:
            img = load_image(f)
            img = cv2.cvtColor(img, cv2.COLOR_BGRA2RGB)
            img = preprocess(img, keep_aspect=KEEP_ASPECT)
            train_imgs.append(img)

    # prep feature list
    if args.model == 'efficientnet-b0':
        train_outputs = [[] for _ in range(8)]
        train_stats = [[] for _ in range(8)]
    else:
        train_outputs = [[] for _ in range(9)]
        train_stats = [[] for _ in range(9)]

    # extract features from train images
    if args.feat is not None:
        train_feat_file = args.feat
    else:
        train_dir = args.train_dir
        train_feat_file = "%s.pkl" % os.path.basename(train_dir)
    if os.path.exists(train_feat_file):
        logger.info('load train set feature distribution from: %s' % train_feat_file)
        with open(train_feat_file, 'rb') as f:
            train_outputs = pickle.load(f)
        logger.info('loaded.')
    else:
        if args.aug:
            logger.info('extract features from train set with augmentation')
        else:
            logger.info('extract features from train set without augmentation')
        x_batch = []
        N = 0
        for x in train_imgs:
            x_batch.append(x)
            N += 1
            if (len(x_batch) == batch_size) | (N == len(train_imgs)):
                logger.info('from (%s ~ %s) ' %
                            ((max((N - batch_size), (N - len(x_batch))) + 1), N))
                x_batch = np.vstack(x_batch)
                feats = net.predict([x_batch])
                for f_idx, feat in enumerate(feats):
                    train_outputs[f_idx].append(feat)
                x_batch = []
        for t_idx, train_output in enumerate(train_outputs):
            train_outputs[t_idx] = np.vstack(train_output)[:, :, 0, 0]  # (B, C)

        # save extracted feature
        logger.info('saving train set feature to: %s ...' % train_feat_file)
        with open(train_feat_file, 'wb') as f:
            pickle.dump(train_outputs, f)
        logger.info('saved.')

    # fitting a multivariate gaussian to features extracted
    # from every level of ImageNet pre-trained model
    for t_idx, train_output in enumerate(train_outputs):
        mean = np.mean(train_output, axis=0)
        # covariance estimation by using the Ledoit. Wolf et al. method
        cov = LedoitWolf().fit(train_output).covariance_
        train_stats[t_idx] = [mean, cov]

    # setting threshold
    if args.threshold is None:
        # calculate Mahalanobis distance per each level of EfficientNet
        dist_list = []
        for t_idx, train_output in enumerate(train_outputs):
            mean = train_stats[t_idx][0]
            cov_inv = np.linalg.inv(train_stats[t_idx][1])
            dist = [mahalanobis(sample, mean, cov_inv) for sample in train_output]
            dist_list.append(np.array(dist))

        # Anomaly score is followed by unweighted summation of the Mahalanobis distances
        scores = np.sum(np.array(dist_list), axis=0)

        threshold = np.mean(scores) * 2
    else:
        threshold = args.threshold

    # extract test set features
    test_imgs = []
    for f in args.input:
        img = load_image(f)
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2RGB)
        img = preprocess(img, keep_aspect=KEEP_ASPECT)
        test_imgs.append(img)

    # prep feature list
    if args.model == 'efficientnet-b0':
        test_outputs = [[] for _ in range(8)]
    else:
        test_outputs = [[] for _ in range(9)]

    # extract features from test images
    logger.info('extract features from test set')
    x_batch = []
    N = 0
    for x in test_imgs:
        x_batch.append(x)
        N += 1
        if (len(x_batch) == batch_size) | (N == len(test_imgs)):
            logger.info('from (%s ~ %s) ' %
                        ((max((N - batch_size), (N - len(x_batch))) + 1), N))
            x_batch = np.vstack(x_batch)
            feats = net.predict([x_batch])
            for f_idx, feat in enumerate(feats):
                test_outputs[f_idx].append(feat)
            x_batch = []
    for t_idx, test_output in enumerate(test_outputs):
        test_outputs[t_idx] = np.vstack(test_output)[:, :, 0, 0]  # (B, C)

    # calculate Mahalanobis distance per each level of EfficientNet
    dist_list = []
    for t_idx, test_output in enumerate(test_outputs):
        mean = train_stats[t_idx][0]
        cov_inv = np.linalg.inv(train_stats[t_idx][1])
        dist = [mahalanobis(sample, mean, cov_inv) for sample in test_output]
        dist_list.append(np.array(dist))

    # Anomaly score is followed by unweighted summation of the Mahalanobis distances
    scores = np.sum(np.array(dist_list), axis=0)

    for i in range(len(args.input)):
        logger.info('================================================')
        logger.info('input image : %s' % args.input[i])
        logger.info('score of input image : %.3f' % scores[i])
        logger.info('anomaly detection threshold : %.3f' % threshold)
        if scores[i] > threshold:
            logger.info('input image is anomaly')
        else:
            logger.info('input image is normal')
        plot_fig(args.input[i], scores[i], threshold, args.savepath)
    logger.info('================================================')

    logger.info('Script finished successfully.')


def recognize_from_video(net):
    batch_size = int(args.batch_size)

    # load train images
    train_filenames = sorted([os.path.join(args.train_dir, f)
                              for f in os.listdir(args.train_dir)
                              if (f.endswith('.png') or f.endswith('.jpg') or
                                  f.endswith('.bmp'))])
    train_imgs = []
    if args.aug:
        for f in train_filenames:
            for _ in range(args.aug_num):
                img = load_image(f)
                img = cv2.cvtColor(img, cv2.COLOR_BGRA2RGB)
                img = preprocess_aug(img, keep_aspect=KEEP_ASPECT)
                train_imgs.append(img)
    else:
        for f in train_filenames:
            img = load_image(f)
            img = cv2.cvtColor(img, cv2.COLOR_BGRA2RGB)
            img = preprocess(img, keep_aspect=KEEP_ASPECT)
            train_imgs.append(img)

    # prep feature list
    if args.model == 'efficientnet-b0':
        train_outputs = [[] for _ in range(8)]
        train_stats = [[] for _ in range(8)]
    else:
        train_outputs = [[] for _ in range(9)]
        train_stats = [[] for _ in range(9)]

    # extract features from train images
    if args.feat is not None:
        train_feat_file = args.feat
    else:
        train_dir = args.train_dir
        train_feat_file = "%s.pkl" % os.path.basename(train_dir)
    if os.path.exists(train_feat_file):
        logger.info('load train set feature distribution from: %s' % train_feat_file)
        with open(train_feat_file, 'rb') as f:
            train_outputs = pickle.load(f)
        logger.info('loaded.')
    else:
        if args.aug:
            logger.info('extract features from train set with augmentation')
        else:
            logger.info('extract features from train set without augmentation')
        x_batch = []
        N = 0
        for x in train_imgs:
            x_batch.append(x)
            N += 1
            if (len(x_batch) == batch_size) | (N == len(train_imgs)):
                logger.info('from (%s ~ %s) ' %
                            ((max((N - batch_size), (N - len(x_batch))) + 1), N))
                x_batch = np.vstack(x_batch)
                feats = net.predict([x_batch])
                for f_idx, feat in enumerate(feats):
                    train_outputs[f_idx].append(feat)
                x_batch = []
        for t_idx, train_output in enumerate(train_outputs):
            train_outputs[t_idx] = np.vstack(train_output)[:, :, 0, 0]  # (B, C)

        # save extracted feature
        logger.info('saving train set feature to: %s ...' % train_feat_file)
        with open(train_feat_file, 'wb') as f:
            pickle.dump(train_outputs, f)
        logger.info('saved.')

    # fitting a multivariate gaussian to features extracted
    # from every level of ImageNet pre-trained model
    for t_idx, train_output in enumerate(train_outputs):
        mean = np.mean(train_output, axis=0)
        # covariance estimation by using the Ledoit. Wolf et al. method
        cov = LedoitWolf().fit(train_output).covariance_
        train_stats[t_idx] = [mean, cov]

    # setting threshold
    if args.threshold is None:
        # calculate Mahalanobis distance per each level of EfficientNet
        dist_list = []
        for t_idx, train_output in enumerate(train_outputs):
            mean = train_stats[t_idx][0]
            cov_inv = np.linalg.inv(train_stats[t_idx][1])
            dist = [mahalanobis(sample, mean, cov_inv) for sample in train_output]
            dist_list.append(np.array(dist))

        # Anomaly score is followed by unweighted summation of the Mahalanobis distances
        scores = np.sum(np.array(dist_list), axis=0)

        threshold = np.mean(scores) * 2
    else:
        threshold = args.threshold

    # extract test set features
    test_frames = []
    capture = webcamera_utils.get_capture(args.video)
    while True:
        ret, frame = capture.read()
        if not ret:
            break
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = preprocess(frame, keep_aspect=KEEP_ASPECT)
        test_frames.append(frame)
    capture.release()

    # prep feature list
    if args.model == 'efficientnet-b0':
        test_outputs = [[] for _ in range(8)]
    else:
        test_outputs = [[] for _ in range(9)]

    # extract features from test images
    logger.info('extract features from test video')
    x_batch = []
    N = 0
    for x in test_frames:
        x_batch.append(x)
        N += 1
        if (len(x_batch) == batch_size) | (N == len(test_frames)):
            logger.info('from (%s ~ %s) ' %
                        ((max((N - batch_size), (N - len(x_batch))) + 1), N))
            x_batch = np.vstack(x_batch)
            feats = net.predict([x_batch])
            for f_idx, feat in enumerate(feats):
                test_outputs[f_idx].append(feat)
            x_batch = []
    for t_idx, test_output in enumerate(test_outputs):
        test_outputs[t_idx] = np.vstack(test_output)[:, :, 0, 0]  # (B, C)

    # calculate Mahalanobis distance per each level of EfficientNet
    dist_list = []
    for t_idx, test_output in enumerate(test_outputs):
        mean = train_stats[t_idx][0]
        cov_inv = np.linalg.inv(train_stats[t_idx][1])
        dist = [mahalanobis(sample, mean, cov_inv) for sample in test_output]
        dist_list.append(np.array(dist))

    # Anomaly score is followed by unweighted summation of the Mahalanobis distances
    scores = np.sum(np.array(dist_list), axis=0)

    capture = webcamera_utils.get_capture(args.video)
    writer = webcamera_utils.get_writer(args.savepath, 400, 1000, fps=4)
    i = 0
    while True:
        ret, frame = capture.read()
        if not ret:
            break
        frame_fig = plot_fig(('frame : %03d' % i), scores[i], threshold, None,
                             True, frame[..., ::-1], np.max(scores))
        writer.write(frame_fig)
        i += 1
    capture.release()
    writer.release()

    logger.info('Script finished successfully.')


def main():
    # model files check and download
    check_and_download_models(WEIGHT_PATH, MODEL_PATH, REMOTE_PATH)

    # load model
    env_id = args.env_id
    net = ailia.Net(MODEL_PATH, WEIGHT_PATH, env_id=env_id)

    if args.video is None:
        # image mode
        recognize_from_image(net)
    else:
        # video mode
        recognize_from_video(net)


if __name__ == '__main__':
    main()
