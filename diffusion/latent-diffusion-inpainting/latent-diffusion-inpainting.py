import os
import sys
import time

import numpy as np
import cv2
from PIL import Image

import ailia

# import original modules
sys.path.append('../../util')
from arg_utils import get_base_parser, update_parser, get_savepath  # noqa
from model_utils import check_and_download_models  # noqa
from detector_utils import load_image  # noqa
# logger
from logging import getLogger  # noqa

from constants import alphas_cumprod

logger = getLogger(__name__)

# ======================
# Parameters
# ======================

WEIGHT_COND_STAGE_PATH = 'cond_stage_model.onnx'
MODEL_COND_STAGE_PATH = 'cond_stage_model.onnx.prototxt'
WEIGHT_DFSN_PATH = 'diffusion_model.onnx'
MODEL_DFSN_PATH = 'diffusion_model.onnx.prototxt'
WEIGHT_AUTO_ENC_PATH = 'autoencoder.onnx'
MODEL_AUTO_ENC_PATH = 'autoencoder.onnx.prototxt'
REMOTE_PATH = 'https://storage.googleapis.com/ailia-models/latent-diffusion-inpainting/'

IMAGE_PATH = 'demo.png'
SAVE_IMAGE_PATH = 'output.png'

# ======================
# Arguemnt Parser Config
# ======================

parser = get_base_parser(
    'Latent Diffusion', IMAGE_PATH, SAVE_IMAGE_PATH
)
parser.add_argument(
    "--ddim_steps", type=int, default=50,
    help="number of ddim sampling steps",
)
parser.add_argument(
    "--ddim_eta", type=float, default=0.0,
    help="ddim eta (eta=0.0 corresponds to deterministic sampling)",
)
parser.add_argument(
    '--onnx',
    action='store_true',
    help='execute onnxruntime version.'
)
parser.add_argument(
    '--seed', type=int, default=128,
    help='random seed for input data and noise'
)
args = update_parser(parser)

np.random.seed(args.seed)

# ======================
# Secondaty Functions
# ======================

def make_ddim_timesteps(num_ddim_timesteps, num_ddpm_timesteps):
    c = num_ddpm_timesteps // num_ddim_timesteps
    ddim_timesteps = np.asarray(list(range(0, num_ddpm_timesteps, c)))

    # add one to get the final alpha values right (the ones from first scale to data during sampling)
    steps_out = ddim_timesteps + 1

    return steps_out


def make_ddim_sampling_parameters(alphacums, ddim_timesteps, eta):
    # select alphas for computing the variance schedule
    alphas = alphacums[ddim_timesteps]
    alphas_prev = np.asarray([alphacums[0]] + alphacums[ddim_timesteps[:-1]].tolist())

    # according the the formula provided in https://arxiv.org/abs/2010.02502
    sigmas = eta * np.sqrt((1 - alphas_prev) / (1 - alphas) * (1 - alphas / alphas_prev))

    return sigmas, alphas, alphas_prev


# ======================
# Main functions
# ======================

"""
ddim_timesteps
"""
ddim_num_steps = args.ddim_steps
ddpm_num_timesteps = 1000
ddim_timesteps = make_ddim_timesteps(
    ddim_num_steps, ddpm_num_timesteps)

"""
ddim sampling parameters
"""
ddim_eta = args.ddim_eta
ddim_sigmas, ddim_alphas, ddim_alphas_prev = \
    make_ddim_sampling_parameters(
        alphacums=alphas_cumprod,
        ddim_timesteps=ddim_timesteps,
        eta=ddim_eta)

ddim_sqrt_one_minus_alphas = np.sqrt(1. - ddim_alphas)


def preprocess(img, mask):
    im_h, im_w, _ = img.shape

    img = img / 255
    img = img.transpose(2, 0, 1)  # HWC -> CHW
    img = np.expand_dims(img, axis=0)
    img = img.astype(np.float32)

    mask = mask / 255
    mask[mask < 0.5] = 0
    mask[mask >= 0.5] = 1
    mask = mask[None, None]

    masked_image = (1 - mask) * img
    masked_image = masked_image.astype(np.float32)

    img = img * 2.0 - 1.0
    mask = mask * 2.0 - 1.0
    masked_image = masked_image * 2.0 - 1.0

    return img, mask, masked_image


def ddim_sampling(
        models,
        cond, shape):
    img = np.random.randn(shape[0] * shape[1] * shape[2] * shape[3]).reshape(shape)
    img = img.astype(np.float32)

    timesteps = ddim_timesteps
    time_range = np.flip(timesteps)
    total_steps = timesteps.shape[0]

    logger.info(f"Running DDIM Sampling with {total_steps} timesteps")

    try:
        from tqdm import tqdm
        iterator = tqdm(time_range, desc='DDIM Sampler', total=total_steps)
    except ModuleNotFoundError:
        def iter_func(a):
            for i, x in enumerate(a):
                print("DDIM Sampler: %s/%s" % (i + 1, len(a)))
                yield x

        iterator = iter_func(time_range)

    for i, step in enumerate(iterator):
        index = total_steps - i - 1
        ts = np.full((shape[0],), step, dtype=np.int64)

        img, pred_x0 = p_sample_ddim(
            models,
            img, cond, ts,
            index=index,
        )
        img = img.astype(np.float32)

    return img


# ddim
def p_sample_ddim(
        models, x, c, t, index,
        temperature=1):
    e_t = apply_model(models, x, t, c)

    alphas = ddim_alphas
    alphas_prev = ddim_alphas_prev
    sqrt_one_minus_alphas = ddim_sqrt_one_minus_alphas
    sigmas = ddim_sigmas

    # select parameters corresponding to the currently considered timestep
    b, *_ = x.shape
    a_t = np.full((b, 1, 1, 1), alphas[index])
    a_prev = np.full((b, 1, 1, 1), alphas_prev[index])
    sigma_t = np.full((b, 1, 1, 1), sigmas[index])
    sqrt_one_minus_at = np.full((b, 1, 1, 1), sqrt_one_minus_alphas[index])

    # current prediction for x_0
    pred_x0 = (x - sqrt_one_minus_at * e_t) / np.sqrt(a_t)

    # direction pointing to x_t
    dir_xt = np.sqrt(1. - a_prev - sigma_t ** 2) * e_t

    noise = sigma_t * np.random.randn(x.size).reshape(x.shape) * temperature
    x_prev = np.sqrt(a_prev) * pred_x0 + dir_xt + noise

    return x_prev, pred_x0


# ddpm
def apply_model(models, x, t, cond):
    diffusion_model = models["diffusion_model"]

    xc = np.concatenate([x, cond], axis=1)
    if not args.onnx:
        output = diffusion_model.predict([xc, t])
    else:
        output = diffusion_model.run(None, {'xc': xc, 't': t})
    x_recon = output[0]

    return x_recon


# decoder
def decode_first_stage(models, z):
    scale_factor = 1.0
    z = z / scale_factor

    autoencoder = models['autoencoder']
    if not args.onnx:
        output = autoencoder.predict([z])
    else:
        output = autoencoder.run(None, {'z': z})
    dec = output[0]

    return dec


def predict(models, img, mask):
    img = img[:, :, ::-1]  # BGR -> RGB
    _mask = mask
    img, mask, masked_image = preprocess(img, mask)

    cond_stage_model = models['cond_stage_model']
    if not args.onnx:
        output = cond_stage_model.predict([masked_image])
    else:
        output = cond_stage_model.run(None, {'masked_image': masked_image})
    c = output[0]

    cc = cv2.resize(_mask, c.shape[-2:])
    cc = np.where(cc < 128, -1, 1)
    cc = cc[None, None]
    cc = cc.astype(np.float32)
    c = np.concatenate((c, cc), axis=1)

    shape = (1, c.shape[1] - 1,) + c.shape[2:]
    samples = ddim_sampling(models, c, shape)

    x_samples_ddim = decode_first_stage(models, samples)

    img = np.clip((img + 1.0) / 2.0, a_min=0.0, a_max=1.0)
    mask = np.clip((mask + 1.0) / 2.0, a_min=0.0, a_max=1.0)
    predicted_image = np.clip((x_samples_ddim + 1.0) / 2.0, a_min=0.0, a_max=1.0)

    inpainted = (1 - mask) * img + mask * predicted_image
    inpainted = inpainted[0]
    inpainted = inpainted.transpose(1, 2, 0) * 255
    inpainted = inpainted[:, :, ::-1]  # RGB -> BGR

    return inpainted


def recognize_from_image(models):
    # input image loop
    for image_path in args.input:
        f_name, ext = os.path.splitext(os.path.basename(image_path))
        if f_name.endswith('_mask'):
            continue

        logger.info(image_path)
        mask_path = os.path.join(os.path.dirname(image_path), f_name + '_mask' + ext)
        logger.info('mask_file: %s' % mask_path)
        if not os.path.exists(mask_path):
            logger.error('mask_file: %s not found.' % mask_path)
            continue

        # prepare input data
        img = load_image(image_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
        mask = load_image(mask_path)
        mask = cv2.cvtColor(mask, cv2.COLOR_BGRA2GRAY)
        # mask = np.array(Image.open(mask_path).convert("L"))

        # inference
        logger.info('Start inference...')
        if args.benchmark:
            logger.info('BENCHMARK mode')
            total_time_estimation = 0
            for i in range(args.benchmark_count):
                start = int(round(time.time() * 1000))
                inpainted = predict(models, img, mask)
                end = int(round(time.time() * 1000))
                estimation_time = (end - start)

                # Logging
                logger.info(f'\tailia processing estimation time {estimation_time} ms')
                if i != 0:
                    total_time_estimation = total_time_estimation + estimation_time

            logger.info(f'\taverage time estimation {total_time_estimation / (args.benchmark_count - 1)} ms')
        else:
            inpainted = predict(models, img, mask)

        # plot result
        savepath = get_savepath(args.savepath, image_path, ext='.png')
        logger.info(f'saved at : {savepath}')
        cv2.imwrite(savepath, inpainted)

    logger.info('Script finished successfully.')


def main():
    check_and_download_models(WEIGHT_COND_STAGE_PATH, MODEL_COND_STAGE_PATH, REMOTE_PATH)
    check_and_download_models(WEIGHT_DFSN_PATH, MODEL_DFSN_PATH, REMOTE_PATH)
    check_and_download_models(WEIGHT_AUTO_ENC_PATH, MODEL_AUTO_ENC_PATH, REMOTE_PATH)

    env_id = args.env_id

    # initialize
    if not args.onnx:
        logger.info("This model requires 10GB or more memory.")
        cond_stage_model = ailia.Net(
            MODEL_COND_STAGE_PATH, WEIGHT_COND_STAGE_PATH, env_id=env_id, memory_mode=args.memory_mode)
        diffusion_model = ailia.Net(
            MODEL_DFSN_PATH, WEIGHT_DFSN_PATH, env_id=env_id, memory_mode=args.memory_mode)
        autoencoder = ailia.Net(
            MODEL_AUTO_ENC_PATH, WEIGHT_AUTO_ENC_PATH, env_id=env_id, memory_mode=args.memory_mode)
    else:
        import onnxruntime
        cond_stage_model = onnxruntime.InferenceSession(WEIGHT_COND_STAGE_PATH)
        diffusion_model = onnxruntime.InferenceSession(WEIGHT_DFSN_PATH)
        autoencoder = onnxruntime.InferenceSession(WEIGHT_AUTO_ENC_PATH)

    models = dict(
        cond_stage_model=cond_stage_model,
        diffusion_model=diffusion_model,
        autoencoder=autoencoder,
    )
    recognize_from_image(models)


if __name__ == '__main__':
    main()
