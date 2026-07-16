import sys, os
import time
import argparse
import math
import glob
from itertools import chain


import numpy as np
import cv2

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

WEIGHT_PATH_MARKET1501 = 'abd_net_market1501.onnx'
MODEL_PATH_MARKET1501 = 'abd_net_market1501.onnx.prototxt'
WEIGHT_PATH_DUKE = 'abd_net_duke.onnx'
MODEL_PATH_DUKE = 'abd_net_duke.onnx.prototxt'
WEIGHT_PATH_MSMT17 = 'abd_net_msmt17.onnx'
MODEL_PATH_MSMT17 = 'abd_net_msmt17.onnx.prototxt'

REMOTE_PATH = \
    'https://storage.googleapis.com/ailia-models/abd_net/'

IMAGE_PATH = './query/0342_c5s1_079123_00.jpg'
GALLERY_DIR = './gallery'
SAVE_IMAGE_PATH = 'output.png'

IMAGE_HEIGHT = 256
IMAGE_WIDTH = 128

MARKET_1501_DROP_SAME_CAMERA_LABEL = True

# ======================
# Arguemnt Parser Config
# ======================
parser = get_base_parser(
    'abdnet model',
    IMAGE_PATH,
    SAVE_IMAGE_PATH,
)
parser.add_argument(
    '-g', '--gallery_dir', type=str, default=GALLERY_DIR,
    help='Gallery file directory'
)
parser.add_argument(
    '-d', '--data', type=str, default=None,
    help='Intermediate result npy file.'
)
parser.add_argument(
    '-m', '--model', type=str, default='market1501',
    choices=('market1501', 'duke','msmt17'),
    help='Name of the model.'
)
parser.add_argument(
    '-bs', '--batchsize', type=int, default=64,
    help='Batchsize.'
)
parser.add_argument(
    '--cui',
    action='store_true',
    help="Don't display preview in GUI."
)

args = update_parser(parser)

if args.cui:
    import matplotlib
    matplotlib.use('Agg')
import matplotlib.pyplot as plt

# ======================
# Secondaty Functions
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
            img = img.transpose(2, 0, 1)
            imgs.append(img)

        imgs = np.stack(imgs)
        imgs = imgs[0] if isinstance(index, int) else imgs
        file_list = file_list[0] if isinstance(index, int) else file_list

        return imgs, file_list

def preprocess(img):
    if args.model == 'pcb':
        img = cv2.resize(img, (IMAGE_PCB_WIDTH, IMAGE_PCB_HEIGHT), interpolation=cv2.INTER_CUBIC)
    else:
        img = cv2.resize(img, (IMAGE_WIDTH, IMAGE_HEIGHT), interpolation=cv2.INTER_CUBIC)
    
    img = img.astype(np.float32) / 255

    # normalize
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    mean = np.float64(mean.reshape(1, -1))
    stdinv = 1 / np.float64(std.reshape(1, -1))
    cv2.subtract(img, mean, img)  # inplace
    cv2.multiply(img, stdinv, img)  # inplace
    
    return img

def sort_img(query_feature, gallery_feature):
    
    query_tile = np.tile(query_feature , ( len(gallery_feature), 1) )
    all_diffs = query_tile - gallery_feature
    distances = np.sqrt(np.sum(np.square(all_diffs), axis=1))
    distances = distances.squeeze()

    # predict index
    index = np.argsort(distances)  # from small to large

    return index


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

def good_img(img_path, query_camera, query_label):
    camera, label = get_id(img_path)
    if label < 0:
        return False
    if camera == query_camera and label == query_label:
        return False
    return True


def imshow(path, title=None):
    """Imshow for Tensor."""
    im = plt.imread(path)
    plt.imshow(im)
    if title is not None:
        plt.title(title)
    if not args.cui:
        plt.pause(0.001)  # pause a bit so that plots are updated

# ======================
# Main functions
# ======================

def recognize_from_image(query_path, net):
    image_list = [query_path]

    if args.data is None:
        ext_list = ["jpg", "png"]
        image_list.extend(
            chain.from_iterable([
                glob.glob(os.path.join(args.gallery_dir, "*." + ext)) for ext in ext_list
            ]))
        if len(image_list) == 1:
            logger.info("GALLARY FILE (%s/*.jpg,*.png) not found." % args.gallery_dir)
            return

    start = int(round(time.time() * 1000))
    logger.info('Start inference...')
    dataloader = DataLoader(image_list)
    features = []
    file_list = []
    count = 0
    for i in range(0, len(dataloader), args.batchsize):

        imgs, files = dataloader[i:i + args.batchsize]
        n, c, h, w = imgs.shape
        count += n
        logger.info("%d/%d" % (count, len(dataloader)))

        outputs = net.run([imgs])[0]

        # 0 : query file , 1-n : gallery files
        features.append(outputs)
        file_list.extend(files)
        
        # extracting 20 sorterd features and files 
        if len( file_list ) > 500:
            
            features = np.vstack(features) # flatten
            
            # separate query and gallery
            query_feature = features[0]
            gallery_feature = features[1:]
            query_file = file_list[0]
            gallery_files = file_list[1:]
            gallery_feature = np.array(gallery_feature)

            # sort and get index
            index = sort_img(query_feature, gallery_feature)

            # save query feature and file
            sorted_features, sorted_file_list = [], []
            sorted_features.append(query_feature)
            sorted_file_list.append(query_file)
            
            # save sorted gallery features and files
            for i in range(20):
                gallery_vec = gallery_feature[index[i]]
                img_path = gallery_files[index[i]]

                sorted_features.append(gallery_vec)
                sorted_file_list.append(img_path)

            # save and next
            features = sorted_features
            file_list = sorted_file_list

    features = np.vstack(features)
    image_list = file_list

    end = int(round(time.time() * 1000))
    logger.info(f'processing time {end - start} ms')

    query_feature = features[0]
    gallery_feature = features[1:]
    gallery_files = image_list[1:]

    if args.data:
        data = np.load(args.data, allow_pickle=True)
        data = data.item()
        gallery_feature = data['gallery_feature']
        gallery_files = data['gallery_file']
    else:
        data = {'gallery_feature': gallery_feature, 'gallery_file': gallery_files}
        file_name = "result_%s.npy" % args.model
        np.save(file_name, data)
        logger.info("'%s' saved" % file_name)

    index = sort_img(query_feature, gallery_feature)

    query_camera = query_label = None
    if MARKET_1501_DROP_SAME_CAMERA_LABEL:
        query_camera, query_label = get_id(query_path)

    logger.info('query_file:'+str(query_path))
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
            if MARKET_1501_DROP_SAME_CAMERA_LABEL \
                    and not good_img(img_path, query_camera, query_label):
                continue
            logger.info(img_path)

            ax = plt.subplot(1, 11, count + 2)
            ax.axis('off')
            _, label = get_id(img_path)
            ax.set_title(
                '%d' % (count + 1),
                color='black' if not MARKET_1501_DROP_SAME_CAMERA_LABEL \
                    else 'green' if label == query_label else 'red')
            imshow(img_path)

            count += 1
            if count >= 10:
                # plt.show()
                break
    except RuntimeError:
        count = 0
        for i in range(10):
            img_path = gallery_files[index[i]]
            if MARKET_1501_DROP_SAME_CAMERA_LABEL \
                    and not good_img(img_path, query_camera, query_label):
                continue
            logger.info(img_path)
            count += 1
            if count >= 10:
                break
        logger.info('If you want to see the visualization of the ranking result, graphical user interface is needed.')

    savepath = get_savepath(args.savepath, query_path)
    logger.info(f'saved at : {savepath}')

    fig.savefig(savepath)

    # plot result
    logger.info('Script finished successfully.')


def main():

    dic_model = {
        'market1501': (WEIGHT_PATH_MARKET1501, MODEL_PATH_MARKET1501),
        'duke': (WEIGHT_PATH_DUKE, MODEL_PATH_DUKE),
        'msmt17' : (WEIGHT_PATH_MSMT17, MODEL_PATH_MSMT17),
    }

    weight_path, model_path = dic_model[args.model]

    check_and_download_models(weight_path, model_path, REMOTE_PATH)

    # initialize
    logger.info(f'env_id: {args.env_id}')
    net = ailia.Net(model_path, weight_path, env_id=args.env_id)

    for input_path in args.input:
        
        recognize_from_image(input_path, net)

if __name__ == '__main__':
    main()
