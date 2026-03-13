import sys
import time

import ailia
import cv2

import numpy as np

# import original modules
sys.path.append('../../util')

# logger
from logging import getLogger  # noqa: E402

import webcamera_utils  # noqa: E402
from image_utils import imread  # noqa: E402
from model_utils import check_and_download_models  # noqa: E402
from arg_utils import get_base_parser, get_savepath, update_parser  # noqa: E402

import matplotlib.pyplot as plt

logger = getLogger(__name__)


# ======================
# Parameters
# ======================

MODEL_NAME = "ego2hands"
WEIGHT_PATH = MODEL_NAME + ".onnx"
MODEL_PATH = WEIGHT_PATH + '.prototxt'

REMOTE_PATH = "https://storage.googleapis.com/ailia-models/" + MODEL_NAME + "/"

DEFAULT_INPUT_PATH = 'sample_image.png'
DEFAULT_SAVE_PATH = 'output.png'

# ======================
# Arguemnt Parser Config
# ======================
parser = get_base_parser(
    'Ego2Hands: Egocentric Two-hand Segmentation and Detection',
    DEFAULT_INPUT_PATH, DEFAULT_SAVE_PATH
)

parser.add_argument(
    '--height', type=int, default=None,
    help='height of the image to run inference on '
)

parser.add_argument(
    '--width', type=int, default=None,
    help='width of the image to run inference on'
)

parser.add_argument(
    '--overlay', action='store_true',
    help='Visualize the mask overlayed on the image'
)

args = update_parser(parser)

# ======================
# Helper functions
# ======================

def preprocess(image, h=None, w=None):
    
    if h is not None and w is not None:
        image = cv2.resize(image, (w, h))

    image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    
    img_edge = cv2.Canny(image, 25, 100).astype(np.float32)
    img_real_test = np.stack((image, img_edge), -1)
    img_real_test = (img_real_test - 128.0) / 256.0
    return img_real_test.transpose(2, 0, 1)


def postprocess(org_image, seg_output, energy_output):
    seg_output_final = cv2.resize(seg_output[0].transpose(1,2,0), dsize=(org_image.shape[1], org_image.shape[0]), interpolation=cv2.INTER_LINEAR_EXACT, )
    seg_output_final = np.argmax(seg_output_final, axis=-1)
    energy_l_final = cv2.resize(energy_output[0,1][None].transpose(1,2,0), dsize=(org_image.shape[1], org_image.shape[0]), interpolation=cv2.INTER_LINEAR_EXACT, )
    energy_r_final = cv2.resize(energy_output[0,2][None].transpose(1,2,0), dsize=(org_image.shape[1], org_image.shape[0]), interpolation=cv2.INTER_LINEAR_EXACT, )
    return seg_output_final, energy_l_final, energy_r_final

def get_bounding_box_from_energy(energy, close_kernel_size = 15, close_op = True):
    energy_positives = (energy > 0.5).astype(np.uint8)
    if close_op:
        energy_positives = cv2.erode(energy_positives, np.ones((close_kernel_size, close_kernel_size)))
        energy_positives = cv2.dilate(energy_positives, np.ones((close_kernel_size, close_kernel_size)))
    coords = np.where(energy_positives.astype(bool))
    if coords[0].size != 0:
        row_min, row_max, col_min, col_max = np.min(coords[0]), np.max(coords[0]), np.min(coords[1]), np.max(coords[1])
    else:
        row_min, row_max, col_min, col_max = 0, 0, 0, 0
    return np.array([row_min, row_max, col_min, col_max])

def create_visualization(image, seg, energy_l, energy_r, overlay=False, savepath=None):
    # visualize segmentation mask
    if overlay:
        mask = np.where((seg == 1)[:,:,None], image/2 + np.array([0,0,128])[None,None], image)
        mask = np.where((seg == 2)[:,:,None], image/2 + np.array([128,0,0])[None,None], mask)
    else:
        mask = seg * 100
        mask = np.tile(mask[:,:,None], (1,1,3))
    mask = (mask).astype(np.uint8)

    # vizualize energy map and bounding box
    if overlay:
        energy_vis = np.where((energy_l > 0.5)[:,:,None], image/2 + np.array([0,0,128])[None,None], image)
        energy_vis = np.where((energy_r > 0.5)[:,:,None], energy_vis/2 + np.array([128,0,0])[None,None], energy_vis)
    else:
        energy_vis = np.tile(((energy_l > 0.5) * 100 + (energy_r > 0.5) * 200)[:,:,None], (1,1,3)).astype('uint8')
    
    bbox_l = get_bounding_box_from_energy(energy_l)
    bbox_r = get_bounding_box_from_energy(energy_r)
    energy_vis = cv2.rectangle(energy_vis, (bbox_l[2], bbox_l[0]), (bbox_l[3], bbox_l[1]), (0, 255, 0), 2)
    energy_vis = cv2.rectangle(energy_vis, (bbox_r[2], bbox_r[0]), (bbox_r[3], bbox_r[1]), (0, 255, 0), 2)
    energy_vis = (energy_vis).astype(np.uint8)
    return mask, energy_vis

def visualize_and_save(image, seg, energy_l, energy_r, overlay=False, savepath=None):
    mask, energy_vis = create_visualization(image, seg, energy_l, energy_r, overlay=overlay)
    
    plt.imshow(mask)
    plt.show()

    plt.imshow(energy_vis)
    plt.show()

    if savepath is not None:
        logger.info(f'saving result to {savepath}')
        mask = cv2.cvtColor(mask, cv2.COLOR_RGB2BGR)
        cv2.imwrite(savepath, mask)
        
        energy_vis = cv2.cvtColor(energy_vis, cv2.COLOR_RGB2BGR)
        energy_savepath = savepath.split('.')
        energy_savepath[-2] += '_energy'
        energy_savepath = '.'.join(energy_savepath)
        cv2.imwrite(energy_savepath, energy_vis)

def update_frame(image, mask, energy, frame):
    vis = np.concatenate([mask, energy], axis=1).astype('uint8')
    if frame is None:
        frame = plt.imshow(vis)
    else:
        frame.set_data(vis)
    plt.pause(0.1)
    return frame

# ======================
# Main functions
# ======================

def recognize_from_image(model):
    logger.info('Start inference...')

    image_path = args.input[0]

    # prepare input data
    org_img = cv2.cvtColor(imread(image_path), cv2.COLOR_BGR2RGB).astype(np.uint8)
    if args.height is not None and args.width is not None:
        h = org_img.shape[0]
        w = org_img.shape[1]
    else:
        h = args.height
        w = args.width

    image = preprocess(org_img, h = h, w = h)[None]

    if args.benchmark and not (args.video is not None):
        logger.info('BENCHMARK mode')
        for i in range(5):
            start = int(round(time.time() * 1000))
            _, _, seg_output_final, energy_output_final = model.predict([image])
            end = int(round(time.time() * 1000))
            logger.info(f'\tailia processing time {end - start} ms')
    else:
        _, _, seg_output_final, energy_output_final = model.predict([image])

    seg, energy_l, energy_r = postprocess(org_img, seg_output_final, energy_output_final)

    # visualize
    visualize_and_save(org_img, seg, energy_l, energy_r, args.overlay, args.savepath)

    logger.info('Script finished successfully.')

def recognize_from_video(model):
    # net initialize

    capture = webcamera_utils.get_capture(args.video)

    _, t = capture.read()
    if args.height is not None and args.width is not None:
        h = t.shape[0]
        w = t.shape[1]
    else:
        h = args.height
        w = args.width

    frame_shown = None

    while(True):
        ret, frame = capture.read()
        if (cv2.waitKey(1) & 0xFF == ord("q")) or not ret:
            break
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB).astype(np.uint8)

        
        # inference
        image = preprocess(frame, h = h, w = w)[None]

        _, _, seg_output_final, energy_output_final = model.predict([image])


        seg, energy_l, energy_r = postprocess(frame, seg_output_final, energy_output_final)
        mask_vis, energy_vis = create_visualization(frame, seg, energy_l, energy_r, overlay=args.overlay)

        # visualize
        frame_shown = update_frame(frame, mask_vis, energy_vis, frame_shown)
        if not plt.get_fignums():
            break

    capture.release()
    logger.info('Script finished successfully.')

def main():
    # model files check and download
    check_and_download_models(WEIGHT_PATH, MODEL_PATH, REMOTE_PATH)
    
    # net initialize
    model = ailia.Net(MODEL_PATH, WEIGHT_PATH, env_id = args.env_id)

    if args.video is not None:
        # video mode
        recognize_from_video(model)
    else:
        # image mode
        recognize_from_image(model)


if __name__ == '__main__':
    main()
