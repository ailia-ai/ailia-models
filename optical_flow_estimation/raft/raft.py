import time
import sys
import platform

import numpy as np
import cv2
from PIL import Image

import ailia

# import original modules
sys.path.append('../../util')
from arg_utils import get_base_parser, update_parser  # noqa: E402
from model_utils import check_and_download_models  # noqa: E402
import webcamera_utils  # noqa: E402

# logger
from logging import getLogger   # noqa: E402
logger = getLogger(__name__)


# ======================
# Parameters
# ======================
IMAGE_BEFORE_PATH = 'input_before.png'
IMAGE_AFTER_PATH = 'input_after.png'
VIDEO_PATH = 'input.mp4'
SAVE_IMAGE_OR_VIDEO_PATH = 'output.png'  # 'output.mp4'

MODEL_LISTS = [
    'things', 'small'
]

SLEEP_TIME = 0  # for web cam mode

RESIZE_ENABLE = True
RESIZE_WIDTH = 512
RESIZE_ALIGNMENT = 32

# ======================
# Argument Parser Config
# ======================
parser = get_base_parser(
    'RAFT: Recurrent All Pairs Field Transforms for Optical Flow',
    None,
    SAVE_IMAGE_OR_VIDEO_PATH,
)
# overwrite default config
# NOTE: arcface has different usage for `--input` with other models
parser.add_argument(
    '-i', '--inputs', metavar='IMAGES', nargs=2,
    default=[IMAGE_BEFORE_PATH, IMAGE_AFTER_PATH],
    help='Two image paths for calculating the optical flow.'
)
parser.add_argument(
    '-v', '--video', metavar='VIDEO',
    default=None,
    help='A video path for calculating the optical flow.'
)
parser.add_argument(
    '-m', '--model', metavar='MODEL',
    default='things', choices=MODEL_LISTS,
    help='model lists: ' + ' | '.join(MODEL_LISTS)
)
parser.add_argument(
    '-itr', '--iterations', type=int, default=0,
    help='If the iterations is large, accuracy will increase.' + 
         'If the iterations is small, speed will increase.' + 
         'default value: {\'things\': 12, \'small\': 5}'
)
args = update_parser(parser)


# ==========================
# MODEL AND OTHER PARAMETERS
# ==========================
WEIGHT_PATH_FNET = 'raft-' + args.model + '_fnet.onnx'
MODEL_PATH_FNET = 'raft-' + args.model + '_fnet.onnx.prototxt'
REMOTE_PATH_FNET = 'https://storage.googleapis.com/ailia-models/raft/'

WEIGHT_PATH_CNET = 'raft-' + args.model + '_cnet.onnx'
MODEL_PATH_CNET = 'raft-' + args.model + '_cnet.onnx.prototxt'
REMOTE_PATH_CNET = 'https://storage.googleapis.com/ailia-models/raft/'

WEIGHT_PATH_UB = 'raft-' + args.model + '_update_block.onnx'
MODEL_PATH_UB = 'raft-' + args.model + '_update_block.onnx.prototxt'
REMOTE_PATH_UB = 'https://storage.googleapis.com/ailia-models/raft/'

if (args.iterations > 0):
    ITERS = args.iterations

if (args.model == 'things'):
    ITERS = 12
    HDIM = 128
    CORR_RADIUS = 4
else:
    ITERS = 5
    HDIM = 96
    CORR_RADIUS = 3


# ======================
# Sub functions
# ======================
def load_image(imfile):
    img = np.array(Image.open(imfile)).astype(np.float32)
    img = img.transpose(2, 0, 1)[np.newaxis, ...]
    return img


def prep_input(image):
    image = np.array(image).astype(np.float32)
    image = image.transpose(2, 0, 1)[np.newaxis, ...]
    return image


class InputPadder:
    """ Pads images such that dimensions are divisible by 8 """
    def __init__(self, dims):
        self.ht, self.wd = dims[-2:]
        pad_ht = (((self.ht // 8) + 1) * 8 - self.ht) % 8
        pad_wd = (((self.wd // 8) + 1) * 8 - self.wd) % 8
        self._pad = ((0, 0), (0, 0),
                     (pad_wd//2, pad_wd - pad_wd//2), 
                     (pad_ht//2, pad_ht - pad_ht//2))

    def pad(self, *inputs):
        return [np.pad(x, self._pad, mode='edge') for x in inputs]


class CorrBlock:
    def __init__(self, fmap1, fmap2, num_levels=4, radius=4):
        self.num_levels = num_levels
        self.radius = radius
        self.corr_pyramid = []

        # all pairs correlation
        corr = CorrBlock.corr(fmap1, fmap2)

        batch, h1, w1, dim, h2, w2 = corr.shape
        corr = corr.reshape(batch*h1*w1, dim, h2, w2)

        self.corr_pyramid.append(corr)
        for i in range(self.num_levels-1):
            corr = corr[:, :, :(corr.shape[2] - (corr.shape[2] % 2)), 
                              :(corr.shape[3] - (corr.shape[3] % 2))]
            corr = corr.reshape(corr.shape[0], corr.shape[1], 
                                int(corr.shape[2] / 2), 2, 
                                int(corr.shape[3] / 2), 2)
            corr = np.mean(np.mean(corr, axis=5), axis=3)
            self.corr_pyramid.append(corr)

    def __call__(self, coords):
        r = self.radius
        coords = coords.transpose(0, 2, 3, 1)
        batch, h1, w1, _ = coords.shape

        out_pyramid = []
        for i in range(self.num_levels):
            corr = self.corr_pyramid[i]
            dx = np.linspace(-r, r, 2*r+1)
            dy = np.linspace(-r, r, 2*r+1)
            delta = np.stack(np.meshgrid(dx, dy)[::-1], axis=-1)

            centroid_lvl = coords.reshape(batch*h1*w1, 1, 1, 2) / 2**i
            delta_lvl = delta.reshape(1, 2*r+1, 2*r+1, 2)
            coords_lvl = centroid_lvl + delta_lvl

            corr = bilinear_sampler(corr, coords_lvl)
            corr = corr.reshape(batch, h1, w1, -1)
            out_pyramid.append(corr)

        out = np.concatenate(out_pyramid, axis=-1)
        return out.transpose(0, 3, 1, 2)

    @staticmethod
    def corr(fmap1, fmap2):
        batch, dim, ht, wd = fmap1.shape
        fmap1 = fmap1.reshape(batch, dim, ht*wd)
        fmap2 = fmap2.reshape(batch, dim, ht*wd) 

        corr = np.matmul(fmap1.transpose(0, 2, 1), fmap2)
        corr = corr.reshape(batch, ht, wd, 1, ht, wd)
        return corr / np.sqrt(float(dim))


def bilinear_sampler(img, coords, mode='bilinear', mask=False):
    """ Wrapper for grid_sample, uses pixel coordinates """
    H, W = img.shape[-2:]
    xgrid = coords[..., [0]]
    ygrid = coords[..., [1]]
    xgrid = 2*xgrid/(W-1) - 1
    ygrid = 2*ygrid/(H-1) - 1

    grid = np.concatenate([xgrid, ygrid], axis=-1)
    img = grid_sample(img, grid, align_corners=True)

    if mask:
        mask = (xgrid > -1) & (ygrid > -1) & (xgrid < 1) & (ygrid < 1)
        return img, mask.astype(np.float32)

    return img


def grid_sampler_unnormalize(coord, side, align_corners):
    if align_corners:
        return ((coord + 1) / 2) * (side - 1)
    else:
        return ((coord + 1) * side - 1) / 2


def grid_sampler_compute_source_index(coord, size, align_corners):
    coord = grid_sampler_unnormalize(coord, size, align_corners)
    return coord


def safe_get_border(image, n, c, x, y, H, W):
    x = np.clip(x, 0, (W - 1))
    y = np.clip(y, 0, (H - 1))
    value = image[n, c, y, x]
    return value


def safe_get_zero(image, n, c, x, y, H, W):
    x = x + 1
    y = y + 1
    x = np.clip(x, 0, (W + 1))
    y = np.clip(y, 0, (H + 1))
    value = np.pad(image, ((0, 0), (0, 0), (1, 1), (1, 1)))
    value = value[n, c, y, x]
    return value


def safe_get(image, n, c, x, y, H, W, padding_mode='zeros'):
    if padding_mode == 'border':
        return safe_get_border(image, n, c, x, y, H, W)
    else:
        return safe_get_zero(image, n, c, x, y, H, W)


# Originally default value of padding_mode is 'zeros', but change to 'border' for speeding up.
def grid_sample(image, grid, padding_mode='border', align_corners=False):
    '''
         input shape = [N, C, H, W]
         grid_shape  = [N, H, W, 2]

         output shape = [N, C, H, W]
    '''
    # print(image.shape)
    N, C, H, W = image.shape
    grid_H = grid.shape[1]
    grid_W = grid.shape[2]

    output_tensor = np.zeros([N, C, grid_H, grid_W], 
                             dtype=type(image.reshape(-1)[0]))

    # get corresponding grid x and y
    y = grid[:, :, :, 1]
    x = grid[:, :, :, 0]

    y = y[:, np.newaxis, :, :]
    x = x[:, np.newaxis, :, :]
    y = y.repeat(C, axis=1)
    x = x.repeat(C, axis=1)

    c = np.zeros_like(y) + np.arange(C)[np.newaxis, :, np.newaxis, np.newaxis]
    n = np.zeros_like(y) + np.arange(N)[:, np.newaxis, np.newaxis, np.newaxis]
    c = c.astype(int)
    n = n.astype(int)

    # Unnormalize with align_corners condition
    ix = grid_sampler_compute_source_index(x, W, align_corners)
    iy = grid_sampler_compute_source_index(y, H, align_corners)

    x0 = np.floor(ix)
    x1 = x0 + 1

    y0 = np.floor(iy)
    y1 = y0 + 1

    # Get W matrix before I matrix, as I matrix requires Channel information
    wa = (x1 - ix) * (y1 - iy)
    wb = (x1 - ix) * (iy - y0)
    wc = (ix - x0) * (y1 - iy)
    wd = (ix - x0) * (iy - y0)

    # Get values of the image by provided x0,y0,x1,y1 by channel

    # image, n, c, x, y, H, W
    x0 = x0.astype(int)
    y0 = y0.astype(int)
    x1 = x1.astype(int)
    y1 = y1.astype(int)
    Ia = safe_get(image, n, c, x0, y0, H, W, padding_mode)
    Ib = safe_get(image, n, c, x0, y1, H, W, padding_mode)
    Ic = safe_get(image, n, c, x1, y0, H, W, padding_mode)
    Id = safe_get(image, n, c, x1, y1, H, W, padding_mode)
    out_ch_val = (
            (Ia * wa) + (Ib * wb) + \
            (Ic * wc) + (Id * wd))

    output_tensor[:, :, :, :] = out_ch_val

    return output_tensor


def initialize_flow(img):
    """ Flow is represented as difference between two coordinate grids flow = coords1 - coords0"""
    N, C, H, W = img.shape
    coords0 = coords_grid(N, H//8, W//8)
    coords1 = coords_grid(N, H//8, W//8)

    # optical flow computed as difference: flow = coords1 - coords0
    return coords0, coords1


def coords_grid(batch, ht, wd):
    coords = np.meshgrid(np.arange(wd), np.arange(ht))[::-1]
    coords = np.stack(coords[::-1], axis=0).astype(np.float32)
    return coords[np.newaxis, ...].repeat(batch, axis=0)


def upsample_flow(flow, mask):
    """ Upsample flow field [H/8, W/8, 2] -> [H, W, 2] using convex combination """
    N, _, H, W = flow.shape
    mask = mask.reshape(N, 1, 9, 8, 8, H, W)
    mask = softmax(mask, axis=2)

    up_flow = unfold(8 * flow, 3, 3, 1, 1)
    up_flow = up_flow.reshape(N, 2, 9, 1, 1, H, W)

    up_flow = np.sum(mask * up_flow, axis=2)
    up_flow = up_flow.transpose(0, 1, 4, 2, 5, 3)
    return up_flow.reshape(N, 2, 8*H, 8*W)


def upflow8(flow, mode='bilinear'):
    new_size = (8 * flow.shape[3], 8 * flow.shape[2])
    return 8 * np.array([[cv2.resize(flow[i, j], new_size, 
                                     interpolation=cv2.INTER_LINEAR) 
                          for j in range(flow.shape[1])] 
                         for i in range(flow.shape[0])])


def softmax(x, axis):
    x = np.exp(x - np.max(x, axis=axis))
    x = x / np.sum(x, axis=axis)
    return x


def get_indices(X_shape, HF, WF, stride, pad):
    """
        Returns index matrices in order to transform our input image into a matrix.

        Parameters:
        -X_shape: Input image shape.
        -HF: filter height.
        -WF: filter width.
        -stride: stride value.
        -pad: padding value.

        Returns:
        -i: matrix of index i.
        -j: matrix of index j.
        -d: matrix of index d. 
            (Use to mark delimitation for each channel
            during multi-dimensional arrays indexing).
    """
    # get input size
    m, n_C, n_H, n_W = X_shape

    # get output size
    out_h = int((n_H + 2 * pad - HF) / stride) + 1
    out_w = int((n_W + 2 * pad - WF) / stride) + 1
  
    # ----Compute matrix of index i----

    # Level 1 vector.
    level1 = np.repeat(np.arange(HF), WF)
    # Duplicate for the other channels.
    level1 = np.tile(level1, n_C)
    # Create a vector with an increase by 1 at each level.
    everyLevels = stride * np.repeat(np.arange(out_h), out_w)
    # Create matrix of index i at every levels for each channel.
    i = level1.reshape(-1, 1) + everyLevels.reshape(1, -1)

    # ----Compute matrix of index j----
    
    # Slide 1 vector.
    slide1 = np.tile(np.arange(WF), HF)
    # Duplicate for the other channels.
    slide1 = np.tile(slide1, n_C)
    # Create a vector with an increase by 1 at each slide.
    everySlides = stride * np.tile(np.arange(out_w), out_h)
    # Create matrix of index j at every slides for each channel.
    j = slide1.reshape(-1, 1) + everySlides.reshape(1, -1)

    # ----Compute matrix of index d----

    # This is to mark delimitation for each channel
    # during multi-dimensional arrays indexing.
    d = np.repeat(np.arange(n_C), HF * WF).reshape(-1, 1)

    return i, j, d


def unfold(X, HF, WF, stride, pad):
    """
        Transforms our input image into a matrix.

        Parameters:
        - X: input image.
        - HF: filter height.
        - WF: filter width.
        - stride: stride value.
        - pad: padding value.

        Returns:
        -cols: output matrix.
    """
    # Padding
    X_padded = np.pad(X, ((0,0), (0,0), (pad, pad), (pad, pad)), mode='constant')
    i, j, d = get_indices(X.shape, HF, WF, stride, pad)
    # Multi-dimensional arrays indexing.
    cols = X_padded[:, d, i, j]
    cols = np.concatenate(cols, axis=-1)
    return cols


def viz(img_before, img_after, flo):
    img_before = img_before[0].transpose(1,2,0)
    img_after = img_after[0].transpose(1,2,0)
    flo = flo[0].transpose(1,2,0)

    # map flow to rgb image
    flo = flow_to_image(flo)
    img_flo = np.concatenate([img_before, img_after, flo], axis=0)

    # convert color from BGR to RGB
    img_BGR = img_flo[:, :, [2,1,0]]
    # return image array
    return img_BGR.astype(np.uint8)


def make_colorwheel():
    """
    Generates a color wheel for optical flow visualization as presented in:
        Baker et al. "A Database and Evaluation Methodology for Optical Flow" (ICCV, 2007)
        URL: http://vision.middlebury.edu/flow/flowEval-iccv07.pdf

    Code follows the original C++ source code of Daniel Scharstein.
    Code follows the the Matlab source code of Deqing Sun.

    Returns:
        np.ndarray: Color wheel
    """

    RY = 15
    YG = 6
    GC = 4
    CB = 11
    BM = 13
    MR = 6

    ncols = RY + YG + GC + CB + BM + MR
    colorwheel = np.zeros((ncols, 3))
    col = 0

    # RY
    colorwheel[0:RY, 0] = 255
    colorwheel[0:RY, 1] = np.floor(255*np.arange(0,RY)/RY)
    col = col+RY
    # YG
    colorwheel[col:col+YG, 0] = 255 - np.floor(255*np.arange(0,YG)/YG)
    colorwheel[col:col+YG, 1] = 255
    col = col+YG
    # GC
    colorwheel[col:col+GC, 1] = 255
    colorwheel[col:col+GC, 2] = np.floor(255*np.arange(0,GC)/GC)
    col = col+GC
    # CB
    colorwheel[col:col+CB, 1] = 255 - np.floor(255*np.arange(CB)/CB)
    colorwheel[col:col+CB, 2] = 255
    col = col+CB
    # BM
    colorwheel[col:col+BM, 2] = 255
    colorwheel[col:col+BM, 0] = np.floor(255*np.arange(0,BM)/BM)
    col = col+BM
    # MR
    colorwheel[col:col+MR, 2] = 255 - np.floor(255*np.arange(MR)/MR)
    colorwheel[col:col+MR, 0] = 255
    return colorwheel


def flow_uv_to_colors(u, v, convert_to_bgr=False):
    """
    Applies the flow color wheel to (possibly clipped) flow components u and v.

    According to the C++ source code of Daniel Scharstein
    According to the Matlab source code of Deqing Sun

    Args:
        u (np.ndarray): Input horizontal flow of shape [H,W]
        v (np.ndarray): Input vertical flow of shape [H,W]
        convert_to_bgr (bool, optional): Convert output image to BGR. Defaults to False.

    Returns:
        np.ndarray: Flow visualization image of shape [H,W,3]
    """
    flow_image = np.zeros((u.shape[0], u.shape[1], 3), np.uint8)
    colorwheel = make_colorwheel()  # shape [55x3]
    ncols = colorwheel.shape[0]
    rad = np.sqrt(np.square(u) + np.square(v))
    a = np.arctan2(-v, -u)/np.pi
    fk = (a+1) / 2*(ncols-1)
    k0 = np.floor(fk).astype(np.int32)
    k1 = k0 + 1
    k1[k1 == ncols] = 0
    f = fk - k0
    for i in range(colorwheel.shape[1]):
        tmp = colorwheel[:,i]
        col0 = tmp[k0] / 255.0
        col1 = tmp[k1] / 255.0
        col = (1-f)*col0 + f*col1
        idx = (rad <= 1)
        col[idx]  = 1 - rad[idx] * (1-col[idx])
        col[~idx] = col[~idx] * 0.75   # out of range
        # Note the 2-i => BGR instead of RGB
        ch_idx = 2-i if convert_to_bgr else i
        flow_image[:,:,ch_idx] = np.floor(255 * col)
    return flow_image


def flow_to_image(flow_uv, clip_flow=None, convert_to_bgr=False):
    """
    Expects a two dimensional flow image of shape.

    Args:
        flow_uv (np.ndarray): Flow UV image of shape [H,W,2]
        clip_flow (float, optional): Clip maximum of flow values. Defaults to None.
        convert_to_bgr (bool, optional): Convert output image to BGR. Defaults to False.

    Returns:
        np.ndarray: Flow visualization image of shape [H,W,3]
    """
    assert flow_uv.ndim == 3, 'input flow must have three dimensions'
    assert flow_uv.shape[2] == 2, 'input flow must have shape [H,W,2]'
    if clip_flow is not None:
        flow_uv = np.clip(flow_uv, 0, clip_flow)
    u = flow_uv[:,:,0]
    v = flow_uv[:,:,1]
    rad = np.sqrt(np.square(u) + np.square(v))
    rad_max = np.max(rad)
    epsilon = 0.01  # 1e-5  # adjusted (original is 1e-5)
    u = u / (rad_max + epsilon)
    v = v / (rad_max + epsilon)
    return flow_uv_to_colors(u, v, convert_to_bgr)


# ======================
# Main functions
# ======================
def recognize_from_image():
    # net initialize
    fnet = ailia.Net(MODEL_PATH_FNET, WEIGHT_PATH_FNET, env_id=0, memory_mode=args.memory_mode)
    cnet = ailia.Net(MODEL_PATH_CNET, WEIGHT_PATH_CNET, env_id=0, memory_mode=args.memory_mode)
    update_block = ailia.Net(MODEL_PATH_UB, WEIGHT_PATH_UB, env_id=0, memory_mode=args.memory_mode)

    # set filename of images
    imfile1 = args.inputs[0]
    imfile2 = args.inputs[1]
    logger.info(imfile1)
    logger.info(imfile2)
    # load images
    image1 = load_image(imfile1)
    image2 = load_image(imfile2)
    # backup for output
    image1_org = image1.copy()
    image2_org = image2.copy()
    # padding for adjust
    padder = InputPadder(image1.shape)
    image1, image2 = padder.pad(image1, image2)
    # normalize
    image1 = 2 * (image1 / 255.0) - 1.0
    image2 = 2 * (image2 / 255.0) - 1.0

    # calculate feature map
    logger.info('Start calculating feature map...')
    if args.benchmark:
        logger.info('BENCHMARK mode')
        for i in range(args.benchmark_count):
            start = int(round(time.time() * 1000))
            fmap = fnet.run(np.concatenate([image1, image2], axis=0))
            end = int(round(time.time() * 1000))
            logger.info(f'\tailia processing time {end - start} ms')
    else:
        fmap = fnet.run(np.concatenate([image1, image2], axis=0))
    fmap1 = fmap[0][[0]]
    fmap2 = fmap[0][[1]]
    # calculate correlation of pixel of feature map
    corr_fn = CorrBlock(fmap1, fmap2, radius=CORR_RADIUS)

    # calculate context
    logger.info('Start calculating context...')
    if args.benchmark:
        logger.info('BENCHMARK mode')
        for i in range(args.benchmark_count):
            start = int(round(time.time() * 1000))
            cmap = cnet.run(image1)[0]
            end = int(round(time.time() * 1000))
            logger.info(f'\tailia processing time {end - start} ms')
    else:
        cmap = cnet.run(image1)[0]
    net = cmap[:, :HDIM]
    inp = cmap[:, HDIM:]
    net = np.tanh(net)
    inp = np.clip(inp, 0, None)

    # initialize coordinates
    coords0, coords1 = initialize_flow(image1)

    # predict optical flow
    flow_predictions = []
    for itr in range(ITERS):

        corr = corr_fn(coords1)  # index correlation volume

        flow = coords1 - coords0
        logger.info('[iter:%d] Start predicting optical flow...' % (itr+1))
        if args.benchmark:
            logger.info('BENCHMARK mode')
            for i in range(args.benchmark_count):
                start = int(round(time.time() * 1000))
                net, up_mask, delta_flow = update_block.run([net, inp, corr, flow])
                end = int(round(time.time() * 1000))
                logger.info(f'\tailia processing time {end - start} ms')
        else:
            net, up_mask, delta_flow = update_block.run([net, inp, corr, flow])

        # F(t+1) = F(t) + \Delta(t)
        coords1 = coords1 + delta_flow

        # upsample predictions
        if (args.model == 'small'):
            flow_up = upflow8(coords1 - coords0)
        else:
            flow_up = upsample_flow(coords1 - coords0, up_mask)

        flow_predictions.append(flow_up)

    # visualize
    img_BGR = viz(image1_org, image2_org, flow_up)
    # save visualization
    logger.info(f'saved at : {args.savepath}')
    cv2.imwrite(args.savepath, img_BGR)

    logger.info('Script finished successfully.')


def recognize_from_video():
    # net initialize
    fnet = ailia.Net(MODEL_PATH_FNET, WEIGHT_PATH_FNET, env_id=0, memory_mode=args.memory_mode)
    cnet = ailia.Net(MODEL_PATH_CNET, WEIGHT_PATH_CNET, env_id=0, memory_mode=args.memory_mode)
    update_block = ailia.Net(MODEL_PATH_UB, WEIGHT_PATH_UB, env_id=0, memory_mode=args.memory_mode)

    # capture video
    capture = webcamera_utils.get_capture(args.video)
    H = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    W = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    
    # resize input video for performance
    if RESIZE_ENABLE:
        if W > RESIZE_WIDTH:
            d = W / RESIZE_WIDTH
            W = int((W / d - (RESIZE_ALIGNMENT-1))//RESIZE_ALIGNMENT * RESIZE_ALIGNMENT)
            H = int((H / d - (RESIZE_ALIGNMENT-1))//RESIZE_ALIGNMENT * RESIZE_ALIGNMENT)

    # create video writer if savepath is specified as video format
    if (args.savepath is not None) & (args.savepath.split('.')[-1] == 'mp4'):
        fps = capture.get(cv2.CAP_PROP_FPS)
        writer = webcamera_utils.get_writer(args.savepath, H*3, W, fps=fps)
    else:
        writer = None

    # read frame
    ret, frame_before = capture.read()
    if RESIZE_ENABLE:
        frame_before = cv2.resize(frame_before, (W,H))
    frame_before = frame_before[..., ::-1]  # BGR2RGB
    
    frame_shown = False
    while True:
        # read frame
        ret, frame_after = capture.read()
        if (cv2.waitKey(1) & 0xFF == ord('q')) or not ret:
            break
        if frame_shown and cv2.getWindowProperty('frame', cv2.WND_PROP_VISIBLE) == 0:
            break
        if RESIZE_ENABLE:
            frame_after = cv2.resize(frame_after, (W,H))

        # preprocessing
        frame_after = frame_after[..., ::-1]  # BGR2RGB
        image1 = frame_before.copy()
        image2 = frame_after.copy()
        image1 = prep_input(image1)
        image2 = prep_input(image2)
        image1_org = image1.copy()
        image2_org = image2.copy()

        # padding for adjust
        padder = InputPadder(image1.shape)
        image1, image2 = padder.pad(image1, image2)
        # normalize
        image1 = 2 * (image1 / 255.0) - 1.0
        image2 = 2 * (image2 / 255.0) - 1.0

        # calculate feature map
        logger.info('Start calculating feature map...')
        if args.benchmark:
            logger.info('BENCHMARK mode')
            for i in range(args.benchmark_count):
                start = int(round(time.time() * 1000))
                fmap = fnet.run(np.concatenate([image1, image2], axis=0))
                end = int(round(time.time() * 1000))
                logger.info(f'\tailia processing time {end - start} ms')
        else:
            fmap = fnet.run(np.concatenate([image1, image2], axis=0))
        fmap1 = fmap[0][[0]]
        fmap2 = fmap[0][[1]]
        # calculate correlation of pixel of feature map
        corr_fn = CorrBlock(fmap1, fmap2, radius=CORR_RADIUS)

        # calculate context
        logger.info('Start calculating context...')
        if args.benchmark:
            logger.info('BENCHMARK mode')
            for i in range(args.benchmark_count):
                start = int(round(time.time() * 1000))
                cmap = cnet.run(image1)[0]
                end = int(round(time.time() * 1000))
                logger.info(f'\tailia processing time {end - start} ms')
        else:
            cmap = cnet.run(image1)[0]
        net = cmap[:, :HDIM]
        inp = cmap[:, HDIM:]
        net = np.tanh(net)
        inp = np.clip(inp, 0, None)

        # initialize coordinates
        coords0, coords1 = initialize_flow(image1)

        # predict optical flow
        flow_predictions = []
        for itr in range(ITERS):

            corr = corr_fn(coords1)  # index correlation volume

            flow = coords1 - coords0
            logger.info('[iter:%d] Start predicting optical flow...' % (itr+1))
            if args.benchmark:
                logger.info('BENCHMARK mode')
                for i in range(args.benchmark_count):
                    start = int(round(time.time() * 1000))
                    net, up_mask, delta_flow = update_block.run([net, inp, corr, flow])
                    end = int(round(time.time() * 1000))
                    logger.info(f'\tailia processing time {end - start} ms')
            else:
                net, up_mask, delta_flow = update_block.run([net, inp, corr, flow])

            # F(t+1) = F(t) + \Delta(t)
            coords1 = coords1 + delta_flow

            # upsample predictions
            if (args.model == 'small'):
                flow_up = upflow8(coords1 - coords0)
            else:
                flow_up = upsample_flow(coords1 - coords0, up_mask)

            flow_predictions.append(flow_up)

        # visualize
        img_BGR = viz(image1_org, image2_org, flow_up)

        # view result figure
        cv2.imshow('frame', img_BGR)
        frame_shown = True
        time.sleep(SLEEP_TIME)
        # save result
        if writer is not None:
            writer.write(img_BGR)

        # slide frame
        frame_before = frame_after

    capture.release()
    cv2.destroyAllWindows()
    if writer is not None:
        writer.release()
        # save visualization
        logger.info(f'saved at : {args.savepath}')

    logger.info('Script finished successfully.')


def main():
    # model files check and download
    check_and_download_models(WEIGHT_PATH_FNET, MODEL_PATH_FNET, REMOTE_PATH_FNET)
    check_and_download_models(WEIGHT_PATH_CNET, MODEL_PATH_CNET, REMOTE_PATH_CNET)
    check_and_download_models(WEIGHT_PATH_UB, MODEL_PATH_UB, REMOTE_PATH_UB)

    if args.video is not None:
        # video mode
        recognize_from_video()
    else:
        # image mode
        recognize_from_image()


if __name__ == '__main__':
    main()
