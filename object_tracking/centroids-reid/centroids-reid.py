import sys, os
import time
import argparse
import math
import glob
from itertools import chain

import numpy as np
import cv2
import matplotlib.pyplot as plt
from typing import Callable, Dict, List
import ailia

# import original modules
sys.path.append('../../util')
from arg_utils import get_base_parser, update_parser, get_savepath  # noqa: E402
from model_utils import check_and_download_models  # noqa: E402
from detector_utils import load_image  # noqa: E402C

# logger
from logging import getLogger   # noqa: E402
logger = getLogger(__name__)

# ======================
# Parameters
# ======================

REMOTE_PATH = 'https://storage.googleapis.com/ailia-models/centroid-reid/'
WEIGHT_MARKET1501_RESNET50_PATH = 'market1501_resnet50_256_128_epoch_120.onnx'
MODEL_MARKET1501_RESNET50_PATH = 'market1501_resnet50_256_128_epoch_120.onnx.prototxt'
WEIGHT_DUKEMTMCREID_RESNET50_PATH = 'dukemtmcreid_resnet50_256_128_epoch_120.onnx'
MODEL_DUKEMTMCREID_RESNET50_PATH = 'dukemtmcreid_resnet50_256_128_epoch_120.onnx.prototxt'


IMAGE_PATH = 'query/0342_c5s1_079123_00.jpg'
GALLERY_DIR = './gallery'
SAVE_IMAGE_PATH = 'output.png'

IMAGE_HEIGHT = 256
IMAGE_WIDTH = 128

# ======================
# Arguemnt Parser Config
# ======================
parser = get_base_parser(
    'Centroids-ReID Resnet50 model',
    IMAGE_PATH,
    SAVE_IMAGE_PATH,
)

parser.add_argument(
    '-g', '--gallery_dir', type=str, default=GALLERY_DIR,
    help='Gallery file directory'
)

parser.add_argument(
    '-m', '--model', type=str, default='market1501_resnet50',
    choices=('market1501_resnet50', 'dukemtmcreid_resnet50'),
    help='Name of the model.'
)

parser.add_argument(
    '-bs', '--batchsize', type=int, default=1,
    help='Batchsize.'
)

parser.add_argument(
    '--reid_metric', type=str, default='cosine',
    choices=('cosine', 'euclidean'),
    help='Name of Metric.'
)
args = update_parser(parser)


# ======================
# Secondary Functions
# ======================

class DataLoader(object):
    def __init__(self, file_list):
        self.file_list = file_list

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, index):
        imgs = []
        file_list = [self.file_list[index]] if isinstance(index, int) else self.file_list[index]
        for filename in file_list:
            img = load_image(filename)
            img = cv2.cvtColor(img, cv2.COLOR_BGRA2RGB)
            img = preprocess(img)
            imgs.append(img)

        imgs = np.stack(imgs)
        imgs = imgs[0] if isinstance(index, int) else imgs
        file_list = file_list[0] if isinstance(index, int) else file_list

        return imgs, file_list


def preprocess(img):
    """
    input: img (H,W,C) and the channel is RGB order
    """
    img = cv2.resize(img, (IMAGE_WIDTH, IMAGE_HEIGHT))
    img = img.astype(np.float64) / 255.0

    # Transpose image from (height, width, channels) to (channels, height, width)
    img = np.transpose(img, (2, 0, 1))

    # Normalize Image
    # Following values are defined by Original CTL Model, even in torchvision and it's from ImageNet by the way,
    # So these are also used here for the purpose of comparison with Original CTL Model
    # However, these values of mean and std are not universal.
    # TODO: User may need to be adjusted for your datasets or tasks.
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    img = (img - mean[:, None, None]) / std[:, None, None]

    return img


def get_euclidean(x, y):
    """
    input x:            e.g. x.shape (1, 2048)
          y:            e.g. y.shape (18, 2048)
    output distmat      e.g. distmat.shape (1, 18)
    """
    logger.info("use euclidean metric")
    x = x.reshape(1, -1)  # e.g. -> (1, 2048 )
    m = x.shape[0]
    n = y.shape[0]

    distmat = (
        np.power(x, 2).sum(axis=1, keepdims=True).repeat(n, axis=1)
        + np.power(y, 2).sum(axis=1, keepdims=True).repeat(m, axis=1).T
    )
    distmat -= 2 * np.dot(x, y.T)
    return distmat


def cosine_similarity(x, y, eps: float = 1e-12):
    """
    Computes cosine similarity between two numpy arrays.
    Value == 1 means the same vector
    Value == 0 means perpendicular vectors
    """
    x_n, y_n = np.linalg.norm(x, axis=1)[:, np.newaxis], np.linalg.norm(y, axis=1)[:, np.newaxis]
    x_norm = x / np.maximum(x_n, eps * np.ones_like(x_n))
    y_norm = y / np.maximum(y_n, eps * np.ones_like(y_n))
    sim_mt = np.dot(x_norm, y_norm.T)
    return sim_mt


def get_cosine(query_feature, gallery_feature, eps: float = 1e-12):
    """
    Computes cosine distance between two tensors.
    The cosine distance is the inverse cosine similarity
    -> cosine_distance = abs(-cosine_distance) to make it
    similar in behaviour to euclidean distance

    input query_feature:              e.g. query_feature.shape (1, 2048)
          gallery_feature:            e.g. gallery_feature.shape (18, 2048)
    output cos                        e.g. cos.shape (1, 18)

    """
    logger.info("use cosine metric")
    query = query_feature.reshape(1, -1)  # e.g. -> (1, 2048 )
    sim_mt = cosine_similarity(query, gallery_feature, eps)
    cos = np.abs(1 - sim_mt).clip(min=eps)
    return cos


def sort_img(query_feature, gallery_feature):
    """
    Sort based on cosine distance
    input: query_feature   e.g. query_feature.shape (2048,)
           gallery_feature e.g. gallery_feature.shape (18, 2048)
    """
    calc_dist_func = get_dist_func(args.reid_metric)
    dist_mat = calc_dist_func(query_feature, gallery_feature)
    score_index = np.argsort(dist_mat, axis=1)  # smaller first
    score_index = score_index.squeeze()
    return score_index


def get_id(img_path):
    filename = os.path.basename(img_path)
    label = filename[:4]
    try:
        a = filename.split('c')
        camera_id = int(a[1][0])
        if label[:2] == '-1':
            label = -1
        else:
            label = int(label)
    except:
        camera_id = None
        label = None

    return camera_id, label


def imshow(path, title=None, wait=False):
    """Imshow for Tensor."""
    im = plt.imread(path)
    plt.imshow(im)
    if title is not None:
        plt.title(title)
    if wait:
        plt.pause(0.001)  # pause a bit so that plots are updated


def create_pid_path_index(paths: List[str]) -> Dict[str, list]:
    """
    List  file indexes per ID
    input: paths            e.g.: ['./gallery/0252_c3s1_057767_03.jpg', ...]
    output: pid2paths_index e.g.: {252: [0, 4, 7, 11, 17],
                                   672: [1, 13],
                                   342: [2, 3, 5, 6, 8, 9, 10, 12, 14, 15, 16]}
    """
    paths_pids = [get_id(item)[1] for item in paths]
    pid2paths_index = {}
    for idx, item in enumerate(paths_pids):
        if item not in pid2paths_index:
            pid2paths_index[item] = [idx]
        else:
            pid2paths_index[item].append(idx)

    return pid2paths_index


def get_dist_func(func_name="cosine"):
    """ Select Metric """
    if func_name == "cosine":
        dist_func = get_cosine
    elif func_name == "euclidean":
        dist_func = get_euclidean
    return dist_func


def calculate_centroids(embeddings, paths):
    """
    input: embeddingsfor images     e.g.: shape:(18, 2048)
           paths  for image         e.g.: len:18
    output: centroid_embeddings     e.g.: shape:(3, 2048)
            pids_centroids_inds     e.g.: shape:(3,) contents: ['252' '672' '342']
            pid_path_index          e.g.: {252: [0, 4, 7, 11, 17],
                                           672: [1, 13],
                                           342: [2, 3, 5, 6, 8, 9, 10, 12, 14, 15, 16]}
    """

    pid_path_index = create_pid_path_index(paths=paths)
    pids_centroids_inds = []
    centroids = []
    for pid, indices in pid_path_index.items():
        inds = np.array(indices)
        pids_vecs = embeddings[inds]
        length = pids_vecs.shape[0]
        centroid = np.sum(pids_vecs, 0) / length
        pids_centroids_inds.append(pid)
        centroids.append(centroid)
    centroid_embeddings = np.vstack(np.array(centroids))
    pids_centroids_inds = np.array(pids_centroids_inds, dtype=np.str_)
    return centroid_embeddings, pids_centroids_inds, pid_path_index


def apply_centroids(embeddings, paths):
    """
    Replace embeddings with centroid_embeddings
    """

    # Calc Centorid
    gallery_centroid_feature, _, pid_path = calculate_centroids(embeddings, paths)

    # Set centroid feature to each image.
    for c, key in zip(gallery_centroid_feature,pid_path):
        for index in pid_path[key]:
            embeddings[int(index), :] = c
    return embeddings


def feature_normalize(x, axis):
    """
    The following torch function is replaced by numpy function
        x = torch.nn.functional.normalize(
            x, dim=axis, p=2
        )

    """
    norms = np.linalg.norm(x, axis=axis, keepdims=True)
    x_normalized = x / norms
    return x_normalized


def create_top10_figure(gallery_files, index, query_path):
    """
    Create top 10 images as output.png
    """
    _, query_label = get_id(query_path)
    logger.info('query_file:' + str(query_path))
    logger.info('Top 10 images are as follow:')
    try:  # Visualize Ranking Result
        # Graphical User Interface is needed
        fig = plt.figure(figsize=(16, 4))
        ax = plt.subplot(1, 11, 1)
        ax.axis('off')

        imshow(query_path, 'query')

        count = 0
        for i in range(len(index)):
            img_path = gallery_files[index[i]]
            logger.info(img_path)

            ax = plt.subplot(1, 11, count + 2)
            ax.axis('off')
            _, label = get_id(img_path)
            ax.set_title(
                '%d' % (count + 1),
                color='green' if label == query_label else 'red')
            imshow(img_path)

            count += 1
            if count >= 10:
                # plt.show()
                break
    except RuntimeError:
        logger.info('If you want to see the visualization of the ranking result, graphical user interface is needed.')
    return fig


# ======================
# Main functions
# ======================


def extract_feature(imgs, net):
    net.set_input_shape(imgs.shape)
    if args.benchmark:
        logger.info('BENCHMARK mode')
        for i in range(5):
            start = int(round(time.time() * 1000))
            outputs = net.predict({"input": imgs})[0]
            end = int(round(time.time() * 1000))
            logger.info(f'\tailia processing time {end - start} ms')
    else:
        outputs = net.predict({"input": imgs})[0]

    return outputs


def recognize_from_image(query_path, net):

    # Prepare Query and Gallery Images
    image_list = [query_path]
    ext_list = ["jpg", "png"]
    image_list.extend(
        chain.from_iterable([
            glob.glob(os.path.join(args.gallery_dir, "*." + ext)) for ext in ext_list
        ]))
    if len(image_list) == 1:
        logger.info("GALLARY FILE (%s/*.jpg,*.png) not found." % args.gallery_dir)
        return

    # Load Model and Extract Image Features
    start = int(round(time.time() * 1000))
    logger.info('Start inference...')
    dataloader = DataLoader(image_list)
    features = []
    for i in range(0, len(dataloader), args.batchsize):
        imgs, paths = dataloader[i:i + args.batchsize]
        outputs = extract_feature(imgs, net)
        features.append(outputs)
    features = np.vstack(features)
    end = int(round(time.time() * 1000))
    logger.info(f'processing time {end - start} ms')

    # Replace gallery feature with centroid for the same ID
    # Note:
    #  - This makes same ID images get the same feature vectors
    #  - This works because this model was trained for minimizing the distance from positive images' centroid,
    #    and maximizing the distance from  negative images' centroid like Triplet Loss
    query_feature = features[0]

    gallery_feature = features[1:]
    gallery_files = image_list[1:]
    gallery_feature = apply_centroids(gallery_feature, gallery_files)

    # (Optional) Save features
    # data = {'gallery_feature': gallery_feature, 'gallery_file': gallery_files}
    # file_name = "result_%s.npy" % args.model
    # np.save(file_name, data)
    # logger.info("'%s' saved" % file_name)

    # Normalize feature
    query_feature = feature_normalize(query_feature, axis=0)     # query_feature.shape (feature)
    gallery_feature = feature_normalize(gallery_feature, axis=1) # gallery_feature.shape (batch, feature)

    # Ranking
    index = sort_img(query_feature, gallery_feature)

    # Create Top10 Result
    fig = create_top10_figure(gallery_files, index, query_path)
    save_path = get_savepath(args.savepath, query_path)
    logger.info(f'saved at : {save_path}')
    fig.savefig(save_path)

    logger.info('Script finished successfully.')


def main():
    dic_model = {
        'market1501_resnet50': (WEIGHT_MARKET1501_RESNET50_PATH, MODEL_MARKET1501_RESNET50_PATH),
        'dukemtmcreid_resnet50': (WEIGHT_DUKEMTMCREID_RESNET50_PATH, MODEL_DUKEMTMCREID_RESNET50_PATH),
    }
    weight_path, model_path = dic_model[args.model]
    logger.info(f'Loading weight: {weight_path}')
    logger.info(f'Loading model: {model_path}')

    # model files check and download
    check_and_download_models(weight_path, model_path, REMOTE_PATH)

    # initialize
    env_id = args.env_id
    logger.info(f'env_id: {env_id}')
    net = ailia.Net(model_path, weight_path, env_id=env_id, memory_mode=args.memory_mode)

    for input_path in args.input:
        recognize_from_image(input_path, net)


if __name__ == '__main__':
    main()
