import sys
import time

import numpy as np

import ailia
import cv2
from util import remove_background, resize_foreground , TSR

# import original modules
sys.path.append('../../util')
from arg_utils import get_base_parser, update_parser
from model_utils import check_and_download_models  # noqa: E402

# logger
from logging import getLogger   # noqa: E402
logger = getLogger(__name__)

# ======================
# Parameters
# ======================
WEIGHT_PATH = 'TripoSR.onnx'
MODEL_PATH  = 'TripoSR.onnx.prototxt'
WEIGHT_DECODER_PATH = 'TripoSR_decoder.onnx'
MODEL_DECODER_PATH  = 'TripoSR_decoder.onnx.prototxt'
REMOTE_PATH = 'https://storage.googleapis.com/ailia-models/tripo_sr/'

IMAGE_PATH = "input.png"
SAVE_PATH  = "output.obj"

RADIUS = 0.87
FEATURE_REDUCTION = "concat"
DENSITY_ACTIVATION = "exp"
DENSITY_BIAS = -1.0
NUM_SAMPLES_PER_RAY = 128
CHUNK_SIZE = 8192

#  ======================
# Arguemnt Parser Config
# ======================

parser = get_base_parser('TripoSR model', IMAGE_PATH,SAVE_PATH)

parser.add_argument(
    "--mc-resolution",
    default=256,
    type=int,
    help="Marching cubes grid resolution. Default: 256"
)
parser.add_argument(
    "--no-remove-bg",
    action="store_true",
    help="If specified, the background will NOT be automatically removed from the input image, and the input image should be an RGB image with gray background and properly-sized foreground. Default: false",
)
parser.add_argument(
    "--foreground-ratio",
    default=0.85,
    type=float,
    help="Ratio of the foreground size to the image size. Only used when --no-remove-bg is not specified. Default: 0.85",
)
parser.add_argument(
    "--model-save-format",
    default="obj",
    type=str,
    choices=["obj", "glb"],
    help="Format to save the extracted mesh. Default: 'obj'",
)

args = update_parser(parser)

# ======================
# Main functions
# ======================

def recognize_from_obj(filename,net,net_decoder):
    # prepare input data
    image = cv2.imread(filename, cv2.IMREAD_UNCHANGED)
    if image.shape[2] == 4:
        image = cv2.cvtColor(image,cv2.COLOR_BGRA2RGBA)
    else:
        image = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)

    model = TSR(net_decoder,
                RADIUS,
                FEATURE_REDUCTION,
                DENSITY_ACTIVATION,
                DENSITY_BIAS,
                NUM_SAMPLES_PER_RAY)
    
    model.renderer.set_chunk_size(CHUNK_SIZE)

    if not args.no_remove_bg:
        image = remove_background(image)
        image = resize_foreground(image, args.foreground_ratio) /255.0
        image = image[:, :, :3] * image[:, :, 3:4] + (1 - image[:, :, 3:4]) * 0.5
    
    size=  512
    image = cv2.resize(image, (size,size), interpolation=cv2.INTER_LINEAR)
    image = np.stack([[image]], axis=0)
    
    def compute(net,image):
        results = np.array(net.run(image))
        
        scene_codes = np.array(results[0])
        
        meshes = model.extract_mesh(scene_codes, resolution=args.mc_resolution)
        return meshes[0]

    # inference
    logger.info('Start inference...')
    if args.benchmark:
        logger.info('BENCHMARK mode')
        for i in range(5):
            start = int(round(time.time() * 1000))
            meshes = compute(net,image)
            end = int(round(time.time() * 1000))
            logger.info(f'\tailia processing time {end - start} ms')
    else:
        meshes = compute(net,image)

    # adjust plot scale
    savepath = args.savepath
    logger.info(f'saved at : {savepath}')
    meshes.export(f"{savepath}")

    logger.info('Script finished successfully.')


def main():
    check_and_download_models(WEIGHT_PATH, MODEL_PATH, REMOTE_PATH)
    check_and_download_models(WEIGHT_DECODER_PATH, MODEL_DECODER_PATH, REMOTE_PATH)

    # load model
    env_id = args.env_id
    logger.info(f'env_id: {env_id}')

    # initialize
    memory_mode = ailia.get_memory_mode(reduce_constant=True,reuse_interstage=True)
    net         = ailia.Net(MODEL_PATH,WEIGHT_PATH,env_id=args.env_id,memory_mode=memory_mode)
    net_decoder = ailia.Net(MODEL_DECODER_PATH,WEIGHT_DECODER_PATH,env_id=args.env_id)

    for point_path in args.input:
        recognize_from_obj(point_path,net,net_decoder)


if __name__ == '__main__':
    main()
