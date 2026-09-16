import sys
import time
import os

import cv2
import matplotlib.pyplot as plt
from scipy.spatial.distance import cosine
import json

import ailia

# import original modules
sys.path.append('../../util')
from arg_utils import get_base_parser, update_parser, get_savepath  # noqa: E402
from model_utils import check_and_download_models  # noqa: E402
from detector_utils import load_image  # noqa: E402
import webcamera_utils  # noqa: E402
from process_catalogue import preprocess, process_gallery

# logger
from logging import getLogger  # noqa: E402

logger = getLogger(__name__)


# ======================
# Parameters
# ======================
WEIGHT_PATH = 'in-shop_retriever.onnx'
MODEL_PATH = 'in-shop_retriever.onnx.prototxt'
REMOTE_PATH = 'https://storage.googleapis.com/ailia-models/mmfashion_retrieval/'

IMAGE_PATH = './06_1_front.jpg'
SAVE_IMAGE_PATH = 'output.png'


# ======================
# Argument Parser Config
# ======================
parser = get_base_parser('MMFashion model - In-shop Clothes Retriever', IMAGE_PATH, SAVE_IMAGE_PATH)
parser.add_argument(
    '-d', '--debug',
    action='store_true',
    help='Debugger'
)
parser.add_argument(
    '--onnx_runtime',
    action='store_true',
    help='Inference using onnx runtime'
)
parser.add_argument(
    '--gallery',
    default='./gallery', type=str,
    help="Path to the root folder of the images' gallery"
)
parser.add_argument(
    '-k', '--topk',
    default=5, type=int,
    help='Retrieve the top k results'
)
parser.add_argument(
    '-w', '--write_json',
    action='store_true',
    help='Flag to output results to json file.'
)
args = update_parser(parser)

results_to_save = []

# ======================
# Utils
# ======================
def check(onnx_model):
    onnx.checker.check_model(onnx_model)

def search_gallery(net, pred, filename=None):
    gallery_imgs, gallery_embeds = process_gallery(args.gallery, net)
    show_retrieved_images(pred, gallery_embeds, gallery_imgs, args.topk, filename)

def show_retrieved_images(query_feat, gallery_embeds, gallery_imgs, topk, filename):
    query_dist = {}
    for img in gallery_imgs:
        cosine_dist = cosine(
            gallery_embeds[img].flatten(), query_feat.flatten())
        query_dist[img] = cosine_dist

    order = sorted(query_dist.items(), key=lambda x: x[1])

    if args.video is None and args.write_json:
        # copy results to save
        global results_to_save
        results_to_save = order

    logger.info('Retrieved Top%d Results' % topk)
    show_topk_retrieved_images(order[:topk], filename)

def show_topk_retrieved_images(retrieved_idxes, input_image):
    fig = plt.figure(figsize=(15, 2.5))
    k=1
    n=len(retrieved_idxes)+1
    show_img(input_image, fig, 'Input image', n, k)

    for retrieved_img, _ in retrieved_idxes:
        k+=1 
        filename = os.path.join(args.gallery, retrieved_img)
        logger.info(filename)
        show_img(filename, fig, f'Top-{k-1} result', n, k) 

def show_img(filename, fig, title, topk, idx):
    if isinstance(filename, str):
        img = plt.imread(filename)
    else:
        img = filename
    ax = fig.add_subplot(1, topk, idx)
    plt.axis('off')
    imgplot = plt.imshow(img)
    ax.set_title(title)


# ======================
# Main functions
# ======================
def recognize_from_image(filename, net):
    # prepare input data
    img = load_image(filename)
    #logger.info(f'input image shape: {img.shape}')
    img = cv2.cvtColor(img, cv2.COLOR_BGRA2RGB)
    img = preprocess(img)

    # inference
    logger.info('Start inference...')
    if args.benchmark:
        logger.info('BENCHMARK mode')
        for i in range(5):
            start = int(round(time.time() * 1000))
            preds_ailia = net.predict(img)
            end = int(round(time.time() * 1000))
            logger.info(f'\tailia processing time {end - start} ms')
    else:
        preds_ailia = net.predict(img)

    #logger.info(f'output shape: {preds_ailia.shape}')
    search_gallery(net, preds_ailia, filename)

    savepath = get_savepath(args.savepath, filename)
    plt.savefig(savepath, bbox_inches='tight')
    logger.info(f'saved at : {savepath}')

    if args.write_json:
        json_file = '%s.json' % savepath.rsplit('.', 1)[0]
        with open(json_file, 'w') as f:
            json.dump([{
                'filename': r[0],
                'cosine': r[1]
            } for r in results_to_save], f, indent=2)

def recognize_from_video(filename, net):
    capture = webcamera_utils.get_capture(args.video)

    # create video writer if savepath is specified as video format
    if args.savepath != SAVE_IMAGE_PATH:
        f_h = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        f_w = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        writer = webcamera_utils.get_writer(args.savepath, f_h, f_w)
    else:
        writer = None

    while True:
        ret, frame = capture.read()
        if cv2.waitKey(1) & 0xFF == ord('q') or not ret:
            break
            
        x = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        x = preprocess(x)
        preds_ailia = net.predict(x)
        search_gallery(net, preds_ailia, x[0].transpose(1, 2, 0))

        # save results
        if writer is not None:
            savepath = get_savepath(args.savepath, filename)
            plt.savefig(savepath, bbox_inches='tight')
            #writer.write(res_img)
        plt.show()
        #break

    capture.release()
    cv2.destroyAllWindows()
    if writer is not None:
        writer.release()

def main():
    # model files check and download
    check_and_download_models(WEIGHT_PATH, MODEL_PATH, REMOTE_PATH)
    
    # net initialize
    net = ailia.Net(MODEL_PATH, WEIGHT_PATH, env_id=args.env_id, memory_mode=args.memory_mode)

    if args.video is not None:
        # video mode
        recognize_from_video(SAVE_IMAGE_PATH, net)
    else:
        # image mode
        # input image loop
        for image_path in args.input:
            recognize_from_image(image_path, net)
    logger.info('Script finished successfully.')

if __name__ == '__main__':
    main()
