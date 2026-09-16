import sys
import numpy as np
import imageio
import cv2

import utils_nerf
from load_llff import load_llff_data

import ailia

# import original modules
sys.path.append('../../util')
from arg_utils import get_base_parser, update_parser, get_savepath
from model_utils import check_and_download_models


# logger
from logging import getLogger

logger = getLogger(__name__)

# ======================
# Parameters
# ======================
IMAGE_PATH = None
SAVE_IMAGE_PATH = './output.png'

H = 756.0
W = 1008.0
C = 3
focal = 815.1316

WEIGHT_PATH = "nerf.opt.onnx"
MODEL_PATH = "nerf.opt.onnx.prototxt"
REMOTE_PATH = "https://storage.googleapis.com/ailia-models/nerf/"

# ======================
# Arguemnt Parser Config
# ======================
parser = get_base_parser(
    'run nerf',
    IMAGE_PATH,
    SAVE_IMAGE_PATH
)

# ailia options
parser.add_argument('-a', '--angle', default=0, type=int, help='Rendering angle (0 - 120)')
parser.add_argument('--onnx', action='store_true', help='execute onnxruntime version.')

# angle options
parser.add_argument("--datadir", type=str,
                    default='./data/nerf_llff_data/', help='input data directory')

# rendering options
parser.add_argument("--netchunk", type=int, default=1024*64,
                    help='number of pts sent through network in parallel, decrease if running out of memory')

parser.add_argument("--N_samples", type=int, default=64,
                    help='number of coarse samples per ray')
parser.add_argument("--N_importance", type=int, default=128,
                    help='number of additional fine samples per ray')
parser.add_argument("--perturb", type=float, default=1.,
                    help='set to 0. for no jitter, 1. for jitter')
parser.add_argument("--use_viewdirs", default=True,
                    help='use full 5D input instead of 3D')
parser.add_argument("--i_embed", type=int, default=0,
                    help='set 0 for default positional encoding, -1 for none')
parser.add_argument("--multires", type=int, default=10,
                    help='log2 of max freq for positional encoding (3D location)')
parser.add_argument("--multires_views", type=int, default=4,
                    help='log2 of max freq for positional encoding (2D direction)')
parser.add_argument("--raw_noise_std", type=float, default=0.,
                    help='std dev of noise added to regularize sigma_a output, 1e0 recommended')

parser.add_argument("--render_only", action='store_true',
                    help='do not optimize, reload weights and render out render_poses path')
parser.add_argument("--render_test", action='store_true',
                    help='render the test set instead of render_poses path')
parser.add_argument("--render_factor", type=int, default=4,
                    help='downsampling factor to speed up rendering, set 4 or 8 for fast preview')

# dataset options
parser.add_argument("--dataset_type", type=str, default='llff',
                    help='options: llff / blender / deepvoxels')

# blender flags
parser.add_argument("--white_bkgd", action='store_true',
                    help='set to render synthetic data on a white bkgd (always use for dvoxels)')
parser.add_argument("--half_res", action='store_true',
                    help='load blender synthetic data at 400x400 instead of 800x800')

# llff flags
parser.add_argument("--factor", type=int, default=4,
                    help='downsample factor for LLFF images')
parser.add_argument("--no_ndc", action='store_true',
                    help='do not use normalized device coordinates (set for non-forward facing scenes)')
parser.add_argument("--lindisp", action='store_true',
                    help='sampling linearly in disparity rather than depth')
parser.add_argument("--spherify", action='store_true',
                    help='set for spherical 360 scenes')
parser.add_argument("--llffhold", type=int, default=8,
                    help='will take every 1/N images as LLFF test set, paper uses 8')

args = update_parser(parser)

# ======================
# Main functions
# ======================
def main():
    # net initialize
    if args.onnx:
        import onnxruntime
        net = onnxruntime.InferenceSession(WEIGHT_PATH)
    else:
        net = ailia.Net(MODEL_PATH, WEIGHT_PATH, env_id=args.env_id, memory_mode=args.memory_mode)

    # load angles
    render_poses = load_llff_data(args.datadir, args.factor,
                                    recenter=True, bd_factor=.75,
                                    spherify=args.spherify,
                                    image_shape=(H, W, C))

    # create nerf instance
    render_kwargs = utils_nerf.create_nerf(args, net)
    bds_dict = {
        'near': 0.,
        'far': 1.,
    }

    render_kwargs.update(bds_dict)
    render_kwargs_fast = {k : render_kwargs[k] for k in render_kwargs}
    render_kwargs_fast['N_importance'] = 0

    down = args.render_factor

    # display rendering information
    logger.info("Rendering angles "+str(len(render_poses)))
    logger.info("Rendering resolution "+str(int(W) // down)+"x"+str(int(H) // down))

    # rendering
    frames = []
    for i, c2w in enumerate(render_poses):
        if i!=args.angle:
            continue

        logger.info("Rendering angle "+str(i))
        test = utils_nerf.render(int(H) // down, int(W) // down, focal / down, c2w=c2w[:3, :4], **render_kwargs_fast)
        frame = (255 * np.clip(test[0], 0, 1)).astype(np.uint8)
        frames.append(frame)

        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        logger.info(f'saved at : {args.savepath}')
        cv2.imwrite(args.savepath,frame)

    # finish
    logger.info('Script finished successfully.')


if __name__=='__main__':
    # model files check and download
    check_and_download_models(WEIGHT_PATH, MODEL_PATH, REMOTE_PATH)
    main()
