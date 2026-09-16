import os
import sys
import time
from functools import partial

import numpy as np
import cv2

import ailia

# import original modules
sys.path.append('../../util')
from arg_utils import get_base_parser, update_parser, get_savepath  # noqa
from model_utils import check_and_download_models  # noqa
# logger
from logging import getLogger  # noqa

from constants import alphas_cumprod
import k_diffusion
from k_diffusion import sample as kdiffusion_sampling, sample_dpmpp_2m

logger = getLogger(__name__)

# ======================
# Parameters
# ======================

# Legacy multi model (v1_4)
WEIGHT_DFSN_EMB_PATH = 'diffusion_emb.onnx'
MODEL_DFSN_EMB_PATH = 'diffusion_emb.onnx.prototxt'
WEIGHT_DFSN_MID_PATH = 'diffusion_mid.onnx'
MODEL_DFSN_MID_PATH = 'diffusion_mid.onnx.prototxt'
WEIGHT_DFSN_OUT_PATH = 'diffusion_out.onnx'
MODEL_DFSN_OUT_PATH = 'diffusion_out.onnx.prototxt'

WEIGHT_BASIL_MIX_EMB_PATH = 'basil_mix_emb.onnx'
MODEL_BASIL_MIX_EMB_PATH = 'basil_mix_emb.onnx.prototxt'
WEIGHT_BASIL_MIX_MID_PATH = 'basil_mix_mid.onnx'
MODEL_BASIL_MIX_MID_PATH = 'basil_mix_mid.onnx.prototxt'
WEIGHT_BASIL_MIX_OUT_PATH = 'basil_mix_out.onnx'
MODEL_BASIL_MIX_OUT_PATH = 'basil_mix_out.onnx.prototxt'

# New one model (v1_4)
WEIGHT_DFSN_PATH = 'diffusion.opt.onnx'
MODEL_DFSN_PATH = 'diffusion.opt.onnx.prototxt'
WEIGHT_BASIL_MIX_PATH = 'basil_mix.opt.onnx'
MODEL_BASIL_MIX_PATH = 'basil_mix.opt.onnx.prototxt'

# Auto encoder
WEIGHT_AUTO_ENC_PATH = 'autoencoder.onnx'
MODEL_AUTO_ENC_PATH = 'autoencoder.onnx.prototxt'

WEIGHT_VAE_FT_MSE_PATH = 'vae-ft-mse-840000.onnx'
MODEL_VAE_FT_MSE_PATH = 'vae-ft-mse-840000.onnx.prototxt'

REMOTE_PATH = 'https://storage.googleapis.com/ailia-models/stable-diffusion-txt2img/'

# Clip model
WEIGHT_VITL14_TEXT_PATH = 'ViT-L14-encode_text.onnx'
MODEL_VITL14_TEXT_PATH = 'ViT-L14-encode_text.onnx.prototxt'
CLIP_REMOTE_PATH = 'https://storage.googleapis.com/ailia-models/clip/'

SAVE_IMAGE_PATH = 'output.png'

# ======================
# Arguemnt Parser Config
# ======================

parser = get_base_parser(
    'Stable Diffusion', None, SAVE_IMAGE_PATH
)
parser.add_argument(
    "-i", "--input", metavar="TEXT", type=str,
    default="a photograph of an astronaut riding a horse",
    help="the prompt to render"
)
parser.add_argument(
    "--n_prompt", metavar="TEXT", type=str,
    default="",
    help="the negative prompt"
)
parser.add_argument(
    "--n_iter", type=int, default=1,
    help="sample this often",
)
parser.add_argument(
    "--n_samples", type=int, default=1,
    help="how many samples to produce for the given prompt",
)
parser.add_argument(
    "--steps", type=int, default=50,
    help="number of ddim sampling steps",
)
parser.add_argument(
    "--ddim_eta", type=float, default=0.0,
    help="ddim eta (eta=0.0 corresponds to deterministic sampling)",
)
parser.add_argument(
    "--H", metavar="height", type=int, default=512,
    help="image height, in pixel space",
)
parser.add_argument(
    "--W", metavar="width", type=int, default=512,
    help="image width, in pixel space",
)
parser.add_argument(
    "--C", type=int, default=4,
    help="latent channels",
)
parser.add_argument(
    "--f", type=int, default=8,
    help="downsampling factor",
)
parser.add_argument(
    "--scale", type=float, default=7.5,
    help="Classifier Free Guidance Scale"
         " - how strongly the image should conform to prompt - lower values produce more creative results",
)
parser.add_argument(
    "--seed", type=int, default=1001,
    help="random seed",
)
parser.add_argument(
    '--sd', default='default', choices=('default', 'basil_mix'),
    help='Stable Diffusion checkpoint'
)
parser.add_argument(
    '--vae', default='default', choices=('default', 'vae-ft-mse'),
    help='SD VAE'
)
parser.add_argument(
    '--sampler', default='PLMS', choices=('PLMS', 'DDIM', 'DPM++ 2M Kerras'),
    help='Which algorithm to use to produce the image.'
)
parser.add_argument(
    '--onnx', action='store_true',
    help='execute onnxruntime version.'
)
parser.add_argument(
    '--transformers_clip',
    action='store_true',
    help='use transformer version of clip.'
)
parser.add_argument(
    '--legacy',
    action='store_true',
    help='execute legacy multi model version.'
)
parser.add_argument(
    '--disable_ailia_tokenizer',
    action='store_true',
    help='disable ailia tokenizer.'
)
args = update_parser(parser, check_input_type=False)

# ======================
# Options
# ======================

FIX_CONSTANT_CONTEXT = True


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
ddim_num_steps = args.steps
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


# encoder
class FrozenCLIPEmbedder:
    """Uses the CLIP transformer encoder for text (from Hugging Face)"""

    def __init__(self, version="openai/clip-vit-large-patch14", max_length=77, onnx=None):
        if args.disable_ailia_tokenizer:
            from transformers import CLIPTokenizer
            self.tokenizer = CLIPTokenizer.from_pretrained(version)
        else:
            from ailia_tokenizer import CLIPTokenizer
            self.tokenizer = CLIPTokenizer.from_pretrained()
        if onnx is None:
            from transformers import CLIPTextModel
            self.onnx = None
            self.transformer = CLIPTextModel.from_pretrained(version)
        else:
            self.onnx = onnx
            self.transformer = None
        self.max_length = max_length

    def encode(self, text):
        batch_encoding = self.tokenizer(
            text, truncation=True, max_length=self.max_length, padding="max_length", return_tensors="np")
        tokens = batch_encoding["input_ids"]
        if self.onnx != None:
            z = self.onnx.predict(tokens)
            z = self.onnx.get_blob_data(self.onnx.find_blob_index_by_name("/ln_final/Add_1_output_0")) # get hidden state
        else:
            import torch
            outputs = self.transformer(input_ids=torch.from_numpy(tokens))
            z = outputs.last_hidden_state
            z = z.detach().numpy()
        return z


# plms
def plms_sampling(
        models,
        x, conditioning,
        unconditional_conditioning=None,
        cfg_scale=1.0,
        **kwargs):
    img = x
    timesteps = ddim_timesteps
    time_range = np.flip(timesteps)
    total_steps = timesteps.shape[0]

    logger.info(f"Running PLMS Sampling with {total_steps} timesteps")

    try:
        from tqdm import tqdm
        iterator = tqdm(time_range, desc='PLMS Sampler', total=total_steps)
    except ModuleNotFoundError:
        def iter_func(a):
            for i, x in enumerate(a):
                print("PLMS Sampler: %s/%s" % (i + 1, len(a)))
                yield x

        iterator = iter_func(time_range)

    b = x.shape[0]
    old_eps = []

    for i, step in enumerate(iterator):
        index = total_steps - i - 1
        ts = np.full((b,), step, dtype=np.int64)
        ts_next = np.full((b,), time_range[min(i + 1, len(time_range) - 1)], dtype=np.int64)

        if args.benchmark:
            start = int(round(time.time() * 1000))

        outs = p_sample_plms(
            models,
            img, conditioning, ts,
            update_context=(i == 0),
            index=index,
            cfg_scale=cfg_scale,
            unconditional_conditioning=unconditional_conditioning,
            old_eps=old_eps, t_next=ts_next,
        )

        if args.benchmark:
            end = int(round(time.time() * 1000))
            estimation_time = (end - start)
            logger.info(f'\tailia processing estimation time {estimation_time} ms')

        img, pred_x0, e_t = outs
        old_eps.append(e_t)
        if len(old_eps) >= 4:
            old_eps.pop(0)

    return img


def p_sample_plms(
        models, x, c, t, update_context, index,
        temperature=1.,
        unconditional_conditioning=None,
        cfg_scale=1.,
        old_eps=None, t_next=None):
    b, *_ = x.shape

    def get_model_output(x, t):
        x_in = np.concatenate([x] * 2)
        t_in = np.concatenate([t] * 2)
        c_in = np.concatenate([unconditional_conditioning, c])
        x_recon = apply_model(models, x_in, t_in, c_in, update_context)
        e_t_uncond, e_t = np.split(x_recon, 2)

        e_t = e_t_uncond + cfg_scale * (e_t - e_t_uncond)
        return e_t

    def get_x_prev_and_pred_x0(e_t, index):
        alphas = ddim_alphas
        alphas_prev = ddim_alphas_prev
        sqrt_one_minus_alphas = ddim_sqrt_one_minus_alphas
        sigmas = ddim_sigmas

        # select parameters corresponding to the currently considered timestep
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

    e_t = get_model_output(x, t)
    if len(old_eps) == 0:
        # Pseudo Improved Euler (2nd order)
        x_prev, pred_x0 = get_x_prev_and_pred_x0(e_t, index)
        e_t_next = get_model_output(x_prev, t_next)
        e_t_prime = (e_t + e_t_next) / 2
    elif len(old_eps) == 1:
        # 2nd order Pseudo Linear Multistep (Adams-Bashforth)
        e_t_prime = (3 * e_t - old_eps[-1]) / 2
    elif len(old_eps) == 2:
        # 3nd order Pseudo Linear Multistep (Adams-Bashforth)
        e_t_prime = (23 * e_t - 16 * old_eps[-1] + 5 * old_eps[-2]) / 12
    elif len(old_eps) >= 3:
        # 4nd order Pseudo Linear Multistep (Adams-Bashforth)
        e_t_prime = (55 * e_t - 59 * old_eps[-1] + 37 * old_eps[-2] - 9 * old_eps[-3]) / 24

    x_prev, pred_x0 = get_x_prev_and_pred_x0(e_t_prime, index)

    return x_prev, pred_x0, e_t


# ddim
def ddim_sampling(
        models,
        x, conditioning,
        unconditional_conditioning=None,
        cfg_scale=1.0,
        **kwargs):
    img = x
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

    b = x.shape[0]

    for i, step in enumerate(iterator):
        if args.benchmark:
            start = int(round(time.time() * 1000))

        index = total_steps - i - 1
        ts = np.full((b,), step, dtype=np.int64)

        img, pred_x0 = p_sample_ddim(
            models,
            img, conditioning, ts,
            update_context=(i == 0),
            index=index,
            cfg_scale=cfg_scale,
            unconditional_conditioning=unconditional_conditioning,
        )

        if args.benchmark:
            end = int(round(time.time() * 1000))
            estimation_time = (end - start)
            logger.info(f'\tailia processing estimation time {estimation_time} ms')

    return img


def p_sample_ddim(
        models, x, c, t, update_context, index,
        temperature=1.,
        unconditional_conditioning=None,
        cfg_scale=1.,
        **kwargs):
    x_in = np.concatenate([x] * 2)
    t_in = np.concatenate([t] * 2)
    c_in = np.concatenate([unconditional_conditioning, c])

    x_recon = apply_model(models, x_in, t_in, c_in, update_context)
    e_t_uncond, e_t = np.split(x_recon, 2)

    e_t = e_t_uncond + cfg_scale * (e_t - e_t_uncond)

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
def apply_model(models, x, t, cc, update_context=True):
    diffusion_emb = models["diffusion_emb"]
    diffusion_mid = models["diffusion_mid"]
    diffusion_out = models["diffusion_out"]

    if not args.legacy and args.onnx:
        x = x.astype(np.float16)
    else:
        x = x.astype(np.float32)

    if not args.legacy:
        if not args.onnx:
            if not FIX_CONSTANT_CONTEXT or update_context:
                output = diffusion_emb.predict([x, t, cc])
            else:
                output = diffusion_emb.run({'x': x, 'timesteps': t})
        else:
            output = diffusion_emb.run(None, {'x': x, 'timesteps': t, 'context': cc})
        return output[0]

    if not args.onnx:
        if not FIX_CONSTANT_CONTEXT or update_context:
            output = diffusion_emb.predict([x, t, cc])
        else:
            output = diffusion_emb.run({'x': x, 'timesteps': t})
    else:
        if "float16" in diffusion_emb.get_inputs()[0].type:
            x = x.astype(np.float16)
        output = diffusion_emb.run(None, {'x': x, 'timesteps': t, 'context': cc})
    h, emb, *hs = output

    if not args.onnx:
        if not FIX_CONSTANT_CONTEXT or update_context:
            output = diffusion_mid.predict([h, emb, cc, *hs[6:]])
        else:
            output = diffusion_mid.run({
                'h': h, 'emb': emb,
                'h6': hs[6], 'h7': hs[7], 'h8': hs[8],
                'h9': hs[9], 'h10': hs[10], 'h11': hs[11],
            })
    else:
        output = diffusion_mid.run(None, {
            'h': h, 'emb': emb, 'context': cc,
            'h6': hs[6], 'h7': hs[7], 'h8': hs[8],
            'h9': hs[9], 'h10': hs[10], 'h11': hs[11],
        })
    h = output[0]

    if not args.onnx:
        if not FIX_CONSTANT_CONTEXT or update_context:
            output = diffusion_out.predict([h, emb, cc, *hs[:6]])
        else:
            output = diffusion_out.run({
                'h': h, 'emb': emb,
                'h0': hs[0], 'h1': hs[1], 'h2': hs[2],
                'h3': hs[3], 'h4': hs[4], 'h5': hs[5],
            })
    else:
        output = diffusion_out.run(None, {
            'h': h, 'emb': emb, 'context': cc,
            'h0': hs[0], 'h1': hs[1], 'h2': hs[2],
            'h3': hs[3], 'h4': hs[4], 'h5': hs[5],
        })
    out = output[0]

    return out


k_diffusion.apply_model = apply_model


# decoder
def decode_first_stage(models, z):
    scale_factor = 0.18215
    z = z / scale_factor
    z = z.astype(np.float32)

    autoencoder = models['autoencoder']
    if not args.onnx:
        output = autoencoder.predict([z])
    else:
        if "float16" in autoencoder.get_inputs()[0].type:
            z = z.astype(np.float16)
        output = autoencoder.run(None, {'input': z})
    dec = output[0]

    return dec


def predict(
        models,
        c, uc):
    n_samples = args.n_samples
    steps = args.steps
    cfg_scale = args.scale
    sampler = args.sampler
    H = args.H
    W = args.W
    C = args.C
    factor = args.f

    shape = [n_samples, C, H // factor, W // factor]
    x = np.random.randn(shape[0] * shape[1] * shape[2] * shape[3]).reshape(shape)

    sampling_info = {
        "PLMS": plms_sampling,
        "DDIM": ddim_sampling,
        "DPM++ 2M Kerras": partial(kdiffusion_sampling, sampler=sample_dpmpp_2m),
    }
    sampling = sampling_info[sampler]

    samples = sampling(
        models, x, c,
        unconditional_conditioning=uc,
        cfg_scale=cfg_scale,
        steps=steps)

    if args.benchmark:
        start = int(round(time.time() * 1000))

    x_samples = decode_first_stage(models, samples)

    if args.benchmark:
        end = int(round(time.time() * 1000))
        estimation_time = (end - start)
        logger.info(f'\tailia processing estimation time {estimation_time} ms')

    x_samples = np.clip((x_samples + 1.0) / 2.0, a_min=0.0, a_max=1.0)

    results = []
    for x_sample in x_samples:
        x_sample = x_sample.transpose(1, 2, 0)  # CHW -> HWC
        x_sample = x_sample * 255
        img = x_sample.astype(np.uint8)
        img = img[:, :, ::-1]  # RGB -> BGR
        results.append(img)

    return results


def recognize_from_text(models):
    n_iter = 1 if args.benchmark else args.n_iter
    n_samples = args.n_samples
    scale = args.scale

    cond_stage_model = FrozenCLIPEmbedder(onnx = models["clip"])

    prompt = args.input if isinstance(args.input, str) else args.input[0]
    n_prompt = args.n_prompt
    logger.info("prompt: %s" % prompt)
    if n_prompt:
        logger.info("negative prompt: %s" % n_prompt)

    sample_path = os.path.join('outputs', prompt.replace(" ", "-"))
    os.makedirs(sample_path, exist_ok=True)
    base_count = len(os.listdir(sample_path))

    logger.info('Start inference...')
    if args.benchmark:
        logger.info('BENCHMARK mode')
        start = int(round(time.time() * 1000))

    c = cond_stage_model.encode([prompt] * n_samples)
    uc = None
    if scale != 1.0:
        uc = cond_stage_model.encode([n_prompt] * n_samples)

    all_samples = []
    for i in range(n_iter):
        logger.info("iteration: %s" % (i + 1))
        x_samples = predict(models, c, uc)

        for img in x_samples:
            sample_file = os.path.join(sample_path, f"{base_count:04}.png")
            cv2.imwrite(sample_file, img)
            base_count += 1

        x_samples = np.concatenate(x_samples, axis=1)
        all_samples.append(x_samples)

    grid_img = np.concatenate(all_samples, axis=0)

    if args.benchmark:
        end = int(round(time.time() * 1000))
        estimation_time = (end - start)
        logger.info(f'\ttotal time estimation {estimation_time} ms')

    # plot result
    savepath = get_savepath(args.savepath, "", ext='.png')
    logger.info(f'saved at : {savepath}')
    cv2.imwrite(savepath, grid_img)

    logger.info('Script finished successfully.')


def main():
    if not args.legacy:
        dic_sd = {
            'default': (
                (WEIGHT_DFSN_PATH, MODEL_DFSN_PATH),
                (None, None),
                (None, None)),
            'basil_mix': (
                (WEIGHT_BASIL_MIX_PATH, MODEL_BASIL_MIX_PATH),
                (None, None),
                (None, None)),
        }
    else:
        dic_sd = {
            'default': (
                (WEIGHT_DFSN_EMB_PATH, MODEL_DFSN_EMB_PATH),
                (WEIGHT_DFSN_MID_PATH, MODEL_DFSN_MID_PATH),
                (WEIGHT_DFSN_OUT_PATH, MODEL_DFSN_OUT_PATH)),
            'basil_mix': (
                (WEIGHT_BASIL_MIX_EMB_PATH, MODEL_BASIL_MIX_EMB_PATH),
                (WEIGHT_BASIL_MIX_MID_PATH, MODEL_BASIL_MIX_MID_PATH),
                (WEIGHT_BASIL_MIX_OUT_PATH, MODEL_BASIL_MIX_OUT_PATH)),
        }
    dic_vae = {
        'default': (WEIGHT_AUTO_ENC_PATH, MODEL_AUTO_ENC_PATH),
        'vae-ft-mse': (WEIGHT_VAE_FT_MSE_PATH, MODEL_VAE_FT_MSE_PATH),
    }
    (WEIGHT_SD_EMB_PATH, MODEL_SD_EMB_PATH), \
    (WEIGHT_SD_MID_PATH, MODEL_SD_MID_PATH), \
    (WEIGHT_SD_OUT_PATH, MODEL_SD_OUT_PATH) = dic_sd[args.sd]
    WEIGHT_VAE_PATH, MODEL_VAE_PATH = dic_vae[args.vae]
    check_and_download_models(WEIGHT_SD_EMB_PATH, MODEL_SD_EMB_PATH, REMOTE_PATH)
    if args.legacy:
        check_and_download_models(WEIGHT_SD_MID_PATH, MODEL_SD_MID_PATH, REMOTE_PATH)
        check_and_download_models(WEIGHT_SD_OUT_PATH, MODEL_SD_OUT_PATH, REMOTE_PATH)
    check_and_download_models(WEIGHT_VAE_PATH, MODEL_VAE_PATH, REMOTE_PATH)
    check_and_download_models(WEIGHT_VITL14_TEXT_PATH, MODEL_VITL14_TEXT_PATH, CLIP_REMOTE_PATH)

    env_id = args.env_id

    # initialize
    diffusion_emb = None
    diffusion_mid = None
    diffusion_out = None
    autoencoder = None

    if not args.onnx:
        logger.info("This model requires 10GB or more memory.")
        diffusion_emb = ailia.Net(
            MODEL_SD_EMB_PATH, WEIGHT_SD_EMB_PATH, env_id=env_id, memory_mode=args.memory_mode)
        if args.legacy:
            diffusion_mid = ailia.Net(
                MODEL_SD_MID_PATH, WEIGHT_SD_MID_PATH, env_id=env_id, memory_mode=args.memory_mode)
            diffusion_out = ailia.Net(
                MODEL_SD_OUT_PATH, WEIGHT_SD_OUT_PATH, env_id=env_id, memory_mode=args.memory_mode)
        autoencoder = ailia.Net(
            MODEL_VAE_PATH, WEIGHT_VAE_PATH, env_id=env_id, memory_mode=args.memory_mode)
        if not args.transformers_clip:
            env_id_cpu = -1 # clip without low memory mode only work on cpu
            # requires the hidden state, an intermediate blob read with
            # get_blob_data(), so REUSE_INTERSTAGE cannot be used
            clip = ailia.Net(
                MODEL_VITL14_TEXT_PATH, WEIGHT_VITL14_TEXT_PATH, env_id=env_id_cpu,
                memory_mode=ailia.get_memory_mode(reduce_constant=True))
        else:
            clip = None
    else:
        import onnxruntime
        diffusion_emb = onnxruntime.InferenceSession(WEIGHT_SD_EMB_PATH)
        if args.legacy:
            diffusion_mid = onnxruntime.InferenceSession(WEIGHT_SD_MID_PATH)
            diffusion_out = onnxruntime.InferenceSession(WEIGHT_SD_OUT_PATH)
        autoencoder = onnxruntime.InferenceSession(WEIGHT_VAE_PATH)
        clip = None

    seed = args.seed
    if seed is not None:
        np.random.seed(seed)

    models = dict(
        diffusion_emb=diffusion_emb,
        diffusion_mid=diffusion_mid,
        diffusion_out=diffusion_out,
        autoencoder=autoencoder,
        clip = clip
    )
    recognize_from_text(models)


if __name__ == '__main__':
    main()
