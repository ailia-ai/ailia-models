import os
import sys
import time
from collections import OrderedDict
import random
import pickle

import numpy as np
import cv2
from PIL import Image
from scipy.ndimage import gaussian_filter
from sklearn.metrics import precision_recall_curve
import matplotlib.pyplot as plt

use_pytorch = False
try:
    import torch

    use_pytorch = True
except ModuleNotFoundError:
    pass

import ailia

# import original modules
sys.path.append('../../util')
from arg_utils import get_base_parser, update_parser, get_savepath  # noqa: E402
from model_utils import check_and_download_models  # noqa: E402
from image_utils import normalize_image  # noqa: E402
from detector_utils import load_image  # noqa: E402
# logger
from logging import getLogger  # noqa: E402

logger = getLogger(__name__)

# ======================
# Parameters
# ======================

WEIGHT_PATH = 'wide_resnet50_2.onnx'
MODEL_PATH = 'wide_resnet50_2.onnx.prototxt'
REMOTE_PATH = 'https://storage.googleapis.com/ailia-models/spade-pytorch/'

IMAGE_PATH = './bottle_000.png'
SAVE_IMAGE_PATH = './output.png'
IMAGE_RESIZE = 256
IMAGE_SIZE = 224

# ======================
# Arguemnt Parser Config
# ======================

parser = get_base_parser('SPADE', IMAGE_PATH, SAVE_IMAGE_PATH)
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
    '-gt', '--gt_dir', metavar="DIR", default="./gt_masks",
    help='directory of the ground truth mask files.'
)
parser.add_argument(
    '-th', '--threshold', type=float, default=None,
    help='threshold'
)
parser.add_argument(
    '--seed', type=int, default=1024,
    help='random seed'
)
parser.add_argument(
    '-ag', '--aug', action='store_true',
    help='process with augmentation.'
)
parser.add_argument(
    '-an', '--aug_num', type=int, default=5,
    help='specify the amplification number of augmentation.'
)
args = update_parser(parser)

# get_blob_data() reads an intermediate blob, so REUSE_INTERSTAGE cannot be used
args.memory_mode = ailia.get_memory_mode(reduce_constant=True)
logger.info(f'memory_mode: {args.memory_mode} (overridden)')


# ======================
# Secondaty Functions
# ======================

def calc_dist_matrix(x, y):
    """Calculate Euclidean distance matrix with torch.tensor"""
    dist_matrix = x[:, None] - y[None, :]
    dist_matrix = np.sum(np.power(dist_matrix, 2), axis=2)
    dist_matrix = np.sqrt(dist_matrix)

    return dist_matrix


def plot_fig(
        file_list, test_imgs,
        score_map,
        gt_imgs, threshold, save_path):
    for t_idx in range(len(file_list)):
        image_path = file_list[t_idx]
        test_img = test_imgs[t_idx]
        test_gt = gt_imgs[t_idx]
        test_pred = score_map[t_idx]
        test_pred[test_pred <= threshold] = 0
        test_pred[test_pred > threshold] = 1
        test_pred_img = test_img.copy()
        test_pred_img[test_pred == 0] = 0

        fig_img, ax_img = plt.subplots(1, 4, figsize=(12, 4)) \
            if test_gt is not None \
            else plt.subplots(1, 3, figsize=(12, 4))
        fig_img.subplots_adjust(left=0, right=1, bottom=0, top=1)
        for ax_i in ax_img:
            ax_i.axes.xaxis.set_visible(False)
            ax_i.axes.yaxis.set_visible(False)

        i = 0
        ax_img[0].imshow(test_img)
        ax_img[0].title.set_text('Image')
        if test_gt is not None:
            i = 1
            ax_img[i].imshow(test_gt, cmap='gray')
            ax_img[i].title.set_text('GroundTruth')
        ax_img[i + 1].imshow(test_pred, cmap='gray')
        ax_img[i + 1].title.set_text('Predicted mask')
        ax_img[i + 2].imshow(test_pred_img)
        ax_img[i + 2].title.set_text('Predicted anomalous image')

        # plot result
        p = get_savepath(save_path, image_path, ext='.png')
        logger.info(f'saved at : {p}')
        fig_img.savefig(p, dpi=100)
        plt.close()


# ======================
# Main functions
# ======================

def preprocess(img, mask=False):
    h, w = img.shape[:2]
    size = IMAGE_RESIZE
    crop_size = IMAGE_SIZE

    # resize
    if h > w:
        size = (size, int(size * h / w))
    else:
        size = (int(size * w / h), size)
    img = np.array(Image.fromarray(img).resize(
        size, resample=Image.ANTIALIAS if not mask else Image.NEAREST))

    # center crop
    h, w = img.shape[:2]
    pad_h = (h - crop_size) // 2
    pad_w = (w - crop_size) // 2
    img = img_cropped = img[pad_h:pad_h + crop_size, pad_w:pad_w + crop_size, ...]

    # normalize
    if not mask:
        img = normalize_image(img.astype(np.float32), 'ImageNet')
        img = img.transpose(2, 0, 1)  # HWC -> CHW
    else:
        img = img / 255

    img = np.expand_dims(img, axis=0)

    return img, img_cropped


def preprocess_aug(img, mask=False, angle_range=[-10, 10], return_refs=False):
    h, w = img.shape[:2]
    size = IMAGE_RESIZE
    crop_size = IMAGE_SIZE

    # resize
    if h > w:
        size = (size, int(size * h / w))
    else:
        size = (int(size * w / h), size)
    img = np.array(Image.fromarray(img).resize(
        size, resample=Image.ANTIALIAS if not mask else Image.NEAREST))

    # for visualize
    img_resized = img.copy()

    if not mask:
        # random rotate
        h, w = img.shape[:2]
        angle = np.random.randint(angle_range[0], angle_range[0] + 1)
        rot_mat = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1)
        img = cv2.warpAffine(src=img,
                             M=rot_mat,
                             dsize=(w, h),
                             borderMode=cv2.BORDER_REPLICATE,
                             flags=cv2.INTER_LINEAR)

        # random crop
        h, w = img.shape[:2]
        pad_h = np.random.randint(0, (h - crop_size))
        pad_w = np.random.randint(0, (w - crop_size))
        img = img[pad_h:pad_h + crop_size, pad_w:pad_w + crop_size, :]

    # normalize
    if not mask:
        img = normalize_image(img.astype(np.float32), 'ImageNet')
        img = img.transpose(2, 0, 1)  # HWC -> CHW
    else:
        img = img / 255

    img = np.expand_dims(img, axis=0)

    if return_refs:
        return img, img_resized, angle, pad_h, pad_w
    else:
        return img, img_resized


def get_train_outputs(net):
    if args.feat:
        logger.info('loading train set feature from: %s' % args.feat)
        with open(args.feat, 'rb') as f:
            train_outputs = pickle.load(f)
        logger.info('loaded.')
        return train_outputs

    batch_size = int(args.batch_size)

    train_dir = args.train_dir
    train_imgs = sorted([
        os.path.join(train_dir, f) for f in os.listdir(train_dir)
        if f.endswith('.png') or f.endswith('.jpg') or f.endswith('.bmp')
    ])
    if len(train_imgs) == 0:
        logger.error("train images not found in '%s'" % train_dir)
        sys.exit(-1)

    if not args.aug:
        logger.info('extract train set features without augmentation')
        aug_num = 1
    else:
        logger.info('extract train set features with augmentation')
        aug_num = args.aug_num

    train_outputs = OrderedDict([
        ('layer1', []), ('layer2', []), ('layer3', []),
        ('avgpool', []),
    ])
    for i_aug in range(aug_num):
        for i_img in range(0, len(train_imgs), batch_size):
            # prepare input data
            imgs = []
            if not args.aug:
                logger.info('from (%s ~ %s) ' %
                            (train_imgs[i_img],
                             train_imgs[min(len(train_imgs) - 1,
                                            i_img + batch_size)]))
            else:
                logger.info('from (%s ~ %s) on augmentation lap %d' %
                            (train_imgs[i_img],
                             train_imgs[min(len(train_imgs) - 1,
                                            i_img + batch_size)], i_aug))

            for image_path in train_imgs[i_img:i_img + batch_size]:
                img = load_image(image_path)
                img = cv2.cvtColor(img, cv2.COLOR_BGRA2RGB)

                if not args.aug:
                    img, _ = preprocess(img)
                else:
                    img, _ = preprocess_aug(img)

                imgs.append(img)

            imgs = np.vstack(imgs)

            # inference
            _ = net.predict(imgs)

            feat_names = ("356", "398", "460", "493")
            for key, name in zip(train_outputs.keys(), feat_names):
                x = net.get_blob_data(name)
                train_outputs[key].append(x)

    for k, v in train_outputs.items():
        train_outputs[k] = np.vstack(v)

    # save learned distribution
    train_feat_file = "%s.pkl" % os.path.basename(train_dir)
    logger.info('saving train set feature to: %s ...' % train_feat_file)
    with open(train_feat_file, 'wb') as f:
        pickle.dump(train_outputs, f)
    logger.info('saved.')

    return train_outputs


def recognize_from_image(net):
    batch_size = int(args.batch_size)

    np.random.seed(args.seed)
    train_outputs = get_train_outputs(net)

    gt_type_dir = args.gt_dir if args.gt_dir else None
    test_imgs = []
    gt_imgs = []
    gt_masks = []

    if not args.aug:
        logger.info('infer without augmentation')
        aug_num = 1
    else:
        logger.info('infer with augmentation')
        aug_num = args.aug_num

    score_map_list = []
    for i_aug in range(aug_num):
        test_outputs = OrderedDict([
            ('layer1', []), ('layer2', []), ('layer3', []),
            ('avgpool', []),
        ])
        aug_list = []

        # batch loop
        for i_img in range(0, len(args.input), batch_size):
            # prepare input data
            imgs = []
            if not args.aug:
                logger.info('from (%s ~ %s) ' %
                            (args.input[i_img],
                             args.input[min(len(args.input) - 1,
                                            i_img + batch_size)]))
            else:
                logger.info('from (%s ~ %s) on augmentation lap %d' %
                            (args.input[i_img],
                             args.input[min(len(args.input) - 1,
                                            i_img + batch_size)], i_aug))

            for image_path in args.input[i_img:i_img + batch_size]:
                img = load_image(image_path)
                img = cv2.cvtColor(img, cv2.COLOR_BGRA2RGB)
                if not args.aug:
                    img, vis_img = preprocess(img)
                else:
                    (img, vis_img, angle,
                     pad_h, pad_w) = preprocess_aug(img, return_refs=True)
                    aug_list.append((angle, pad_h, pad_w))
                imgs.append(img)
                test_imgs.append(vis_img)

                # ground truth
                if i_aug == 0:
                    fname = os.path.splitext(os.path.basename(image_path))[0]
                    gt_fpath = os.path.join(gt_type_dir, fname + '_mask.png')
                    if os.path.exists(gt_fpath):
                        gt_img = load_image(gt_fpath)
                        gt_img = cv2.cvtColor(gt_img, cv2.COLOR_BGRA2GRAY)
                        if not args.aug:
                            gt_mask, gt_img = preprocess(gt_img, mask=True)
                        else:
                            gt_mask, gt_img = preprocess_aug(gt_img, mask=True)
                    else:
                        gt_img = np.zeros(
                            (IMAGE_SIZE, IMAGE_SIZE) if not args.aug
                            else (IMAGE_RESIZE, IMAGE_RESIZE))
                        gt_mask = gt_img[None, :, :]

                    gt_imgs.append(gt_img)
                    gt_masks.append(gt_mask)

            imgs = np.vstack(imgs)

            logger.debug(f'input images shape: {imgs.shape}')

            # inference
            if args.benchmark:
                logger.info('BENCHMARK mode')
                total_time = 0
                for i in range(args.benchmark_count):
                    start = int(round(time.time() * 1000))
                    _ = net.predict(imgs)
                    end = int(round(time.time() * 1000))
                    logger.info(f'\tailia processing time {end - start} ms')
                    if i != 0:
                        total_time = total_time + (end - start)
                logger.info(f'\taverage time {total_time / (args.benchmark_count - 1)} ms')
            else:
                _ = net.predict(imgs)

            feat_names = ("356", "398", "460", "493")
            for key, name in zip(test_outputs.keys(), feat_names):
                x = net.get_blob_data(name)
                test_outputs[key].append(x)

        for k, v in test_outputs.items():
            test_outputs[k] = np.vstack(v)

        # calculate distance matrix
        dist_matrix = calc_dist_matrix(
            test_outputs['avgpool'].reshape(test_outputs['avgpool'].shape[0], -1),
            train_outputs['avgpool'].reshape(train_outputs['avgpool'].shape[0], -1))

        # select K nearest neighbor
        top_k = 5
        topk_indexes = np.argsort(dist_matrix, axis=1)
        topk_indexes = topk_indexes[:, :top_k]

        n = test_outputs['avgpool'].shape[0]
        for t_idx in range(n):
            logger.info("| localization | %s/%s | %s"
                        % (t_idx + 1, n, args.input[t_idx]))

            score_maps = []
            for layer_name in ['layer1', 'layer2', 'layer3']:  # for each layer
                # construct a gallery of features at all pixel locations of the K nearest neighbors
                topk_feat_map = train_outputs[layer_name][topk_indexes[t_idx]]
                test_feat_map = test_outputs[layer_name][t_idx:t_idx + 1]

                feat_gallery = topk_feat_map.transpose(0, 3, 2, 1)
                feat_gallery = feat_gallery.reshape(-1, feat_gallery.shape[-1])
                feat_gallery = feat_gallery[:, :, None, None]

                # calculate distance matrix
                dist_matrix_list = []
                if use_pytorch and torch.cuda.is_available():
                    feat_gallery = torch.tensor(feat_gallery, device='cuda')
                    test_feat_map = torch.tensor(test_feat_map, device='cuda')
                    for d_idx in range(feat_gallery.shape[0] // 100):
                        dist_matrix = torch.pow(torch.mean(torch.pow(
                            feat_gallery[d_idx * 100:d_idx * 100 + 100] - test_feat_map, 2), 1),
                            0.5)
                        dist_matrix_list.append(dist_matrix.cpu().detach().numpy())
                else:
                    for d_idx in range(feat_gallery.shape[0] // 100):
                        dist_matrix = np.power(feat_gallery[d_idx * 100:d_idx * 100 + 100] - test_feat_map, 2)
                        dist_matrix = np.mean(dist_matrix, axis=1)
                        dist_matrix = np.sqrt(dist_matrix)
                        dist_matrix_list.append(dist_matrix)

                dist_matrix = np.vstack(dist_matrix_list)
                score_map = np.min(dist_matrix, axis=0)
                score_map = np.asarray(
                    Image.fromarray(score_map).resize(
                        (IMAGE_SIZE, IMAGE_SIZE), resample=Image.BILINEAR))
                score_map = score_map[None, None, :, :]
                score_maps.append(score_map)

            score_maps = np.vstack(score_maps)
            score_map = np.mean(score_maps, axis=0)
            if args.aug:
                # reverse crop
                angle, pad_top, pad_left = aug_list[t_idx]
                pad_bottom = IMAGE_RESIZE - IMAGE_SIZE - pad_top
                pad_right = IMAGE_RESIZE - IMAGE_SIZE - pad_left
                score_map = np.pad(np.squeeze(score_map),
                                   ((pad_top, pad_bottom), (pad_left, pad_right)))
                # reverse rotate
                rot_mat = cv2.getRotationMatrix2D((IMAGE_RESIZE / 2, IMAGE_RESIZE / 2), -angle, 1)
                score_map = cv2.warpAffine(src=score_map,
                                           M=rot_mat,
                                           dsize=(IMAGE_RESIZE, IMAGE_RESIZE),
                                           borderMode=cv2.BORDER_REPLICATE,
                                           flags=cv2.INTER_LINEAR)
                score_map = score_map[None, :, :]

            score_map_list.append(score_map)

    if args.aug:
        score_map = np.stack(score_map_list)
        score_map = score_map.reshape(args.aug_num, -1, IMAGE_RESIZE, IMAGE_RESIZE)
        score_map = np.mean(score_map, axis=0)
        score_map_list = score_map[:, None, :, :]

    # apply gaussian smoothing on the score map
    for i in range(len(score_map_list)):
        score_map_list[i] = gaussian_filter(score_map_list[i], sigma=4)

    if args.threshold is None:
        # get optimal threshold
        flatten_gt_mask_list = np.concatenate(gt_masks).ravel()
        flatten_score_map_list = np.concatenate(score_map_list).ravel()

        # get optimal threshold
        precision, recall, thresholds = precision_recall_curve(flatten_gt_mask_list, flatten_score_map_list)
        a = 2 * precision * recall
        b = precision + recall
        f1 = np.divide(a, b, out=np.zeros_like(a), where=b != 0)
        threshold = thresholds[np.argmax(f1)]

        logger.info('Optimal threshold: %f' % threshold)
    else:
        threshold = args.threshold

    score_map = np.vstack(score_map_list)
    plot_fig(args.input, test_imgs, score_map, gt_imgs, threshold, args.savepath)

    logger.info('Script finished successfully.')


def main():
    # model files check and download
    check_and_download_models(WEIGHT_PATH, MODEL_PATH, REMOTE_PATH)

    # load model
    env_id = args.env_id

    net = ailia.Net(MODEL_PATH, WEIGHT_PATH, env_id=env_id, memory_mode=args.memory_mode)

    recognize_from_image(net)


if __name__ == '__main__':
    main()
