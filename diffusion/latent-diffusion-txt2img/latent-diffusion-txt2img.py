import os
import sys
import time

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

logger = getLogger(__name__)

# ======================
# Parameters
# ======================

WEIGHT_TRANS_EMB_PATH = 'transformer_emb.onnx'
MODEL_TRANS_EMB_PATH = 'transformer_emb.onnx.prototxt'
WEIGHT_TRANS_ATTN_PATH = 'transformer_attn.onnx'
MODEL_TRANS_ATTN_PATH = 'transformer_attn.onnx.prototxt'
WEIGHT_DFSN_EMB_PATH = 'diffusion_emb.onnx'
MODEL_DFSN_EMB_PATH = 'diffusion_emb.onnx.prototxt'
WEIGHT_DFSN_MID_PATH = 'diffusion_mid.onnx'
MODEL_DFSN_MID_PATH = 'diffusion_mid.onnx.prototxt'
WEIGHT_DFSN_OUT_PATH = 'diffusion_out.onnx'
MODEL_DFSN_OUT_PATH = 'diffusion_out.onnx.prototxt'
WEIGHT_AUTO_ENC_PATH = 'autoencoder.onnx'
MODEL_AUTO_ENC_PATH = 'autoencoder.onnx.prototxt'
REMOTE_PATH = 'https://storage.googleapis.com/ailia-models/latent-diffusion-txt2img/'

SAVE_IMAGE_PATH = 'output.png'

# ======================
# Arguemnt Parser Config
# ======================

parser = get_base_parser(
    'Latent Diffusion', None, SAVE_IMAGE_PATH
)
parser.add_argument(
    "-i", "--input", metavar="TEXT", type=str,
    default="a painting of a virus monster playing guitar",
    help="the prompt to render"
)
parser.add_argument(
    "--n_iter", type=int, default=1,
    help="sample this often",
)
parser.add_argument(
    "--n_samples", type=int, default=4,
    help="how many samples to produce for the given prompt",
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
    "--H", metavar="height", type=int, default=256,
    help="image height, in pixel space",
)
parser.add_argument(
    "--W", metavar="width", type=int, default=256,
    help="image width, in pixel space",
)
parser.add_argument(
    "--scale", type=float, default=5.0,
    help="unconditional guidance scale: eps = eps(x, empty) + scale * (eps(x, cond) - eps(x, empty))",
)
parser.add_argument(
    "--seed", type=int, default=None,
    help="random seed",
)
parser.add_argument(
    '--onnx',
    action='store_true',
    help='execute onnxruntime version.'
)
parser.add_argument(
    '--disable_ailia_tokenizer',
    action='store_true',
    help='disable ailia tokenizer.'
)
args = update_parser(parser, check_input_type=False)


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


# encoder
class BERTEmbedder:
    """ Uses a pretrained BERT tokenizer by huggingface. Vocab size: 30522 (?)"""

    def __init__(self, transformer_emb, transformer_attn, max_length=77):
        if args.disable_ailia_tokenizer:
            from transformers import BertTokenizerFast
            self.tokenizer = BertTokenizerFast.from_pretrained("bert-base-uncased")
        else:
            from ailia_tokenizer import BertTokenizer
            self.tokenizer = BertTokenizer.from_pretrained("./tokenizer/")

        self.max_length = max_length

        self.transformer_emb = transformer_emb
        self.transformer_attn = transformer_attn

    def encode(self, text):
        batch_encoding = self.tokenizer(
            text, truncation=True, max_length=self.max_length, #return_length=True,
            #return_overflowing_tokens=False,
            padding="max_length", return_tensors="np")
        tokens = batch_encoding["input_ids"]

        if not args.onnx:
            output = self.transformer_emb.predict([tokens])
        else:
            output = self.transformer_emb.run(None, {'x': tokens})
        x = output[0]

        if not args.onnx:
            output = self.transformer_attn.predict([x])
        else:
            output = self.transformer_attn.run(None, {'x': x})
        z = output[0]

        return z


def ddim_sampling(
        models,
        cond, shape,
        unconditional_guidance_scale=1.0,
        unconditional_conditioning=None):
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
            unconditional_guidance_scale=unconditional_guidance_scale,
            unconditional_conditioning=unconditional_conditioning,
        )
        img = img.astype(np.float32)

    return img


# ddim
def p_sample_ddim(
        models, x, c, t, index,
        temperature=1.,
        unconditional_guidance_scale=1.,
        unconditional_conditioning=None):
    x_in = np.concatenate([x] * 2)
    t_in = np.concatenate([t] * 2)
    c_in = np.concatenate([unconditional_conditioning, c])

    x_recon = apply_model(models, x_in, t_in, c_in)
    e_t_uncond, e_t = np.split(x_recon, 2)

    e_t = e_t_uncond + unconditional_guidance_scale * (e_t - e_t_uncond)

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
def apply_model(models, x, t, cc):
    diffusion_emb = models["diffusion_emb"]
    diffusion_mid = models["diffusion_mid"]
    diffusion_out = models["diffusion_out"]

    if not args.onnx:
        output = diffusion_emb.predict([x, t, cc])
    else:
        output = diffusion_emb.run(None, {'x': x, 'timesteps': t, 'context': cc})
    h, emb, *hs = output

    if not args.onnx:
        output = diffusion_mid.predict([h, emb, cc, *hs[6:]])
    else:
        output = diffusion_mid.run(None, {
            'h': h, 'emb': emb, 'context': cc,
            'h6': hs[6], 'h7': hs[7], 'h8': hs[8],
            'h9': hs[9], 'h10': hs[10], 'h11': hs[11],
        })
    h = output[0]

    if not args.onnx:
        output = diffusion_out.predict([h, emb, cc, *hs[:6]])
    else:
        output = diffusion_out.run(None, {
            'h': h, 'emb': emb, 'context': cc,
            'h0': hs[0], 'h1': hs[1], 'h2': hs[2],
            'h3': hs[3], 'h4': hs[4], 'h5': hs[5],
        })
    out = output[0]

    return out


# decoder
def decode_first_stage(models, z):
    scale_factor = 0.18215
    z = z / scale_factor

    autoencoder = models['autoencoder']
    if not args.onnx:
        output = autoencoder.predict([z])
    else:
        output = autoencoder.run(None, {'input': z})
    dec = output[0]

    return dec


def predict(
        models, cond_stage_model,
        prompt, uc):
    n_samples = args.n_samples
    scale = args.scale
    H = args.H
    W = args.W

    c = cond_stage_model.encode([prompt] * n_samples)
    shape = [n_samples, 4, H // 8, W // 8]

    samples = ddim_sampling(
        models, c, shape,
        unconditional_guidance_scale=scale,
        unconditional_conditioning=uc)

    x_samples_ddim = decode_first_stage(models, samples)
    x_samples_ddim = np.clip((x_samples_ddim + 1.0) / 2.0, a_min=0.0, a_max=1.0)

    x_samples = []
    for x_sample in x_samples_ddim:
        x_sample = x_sample.transpose(1, 2, 0)  # CHW -> HWC
        x_sample = x_sample * 255
        img = x_sample.astype(np.uint8)
        img = img[:, :, ::-1]  # RGB -> BGR
        x_samples.append(img)

    return x_samples


def recognize_from_text(models):
    n_iter = 1 if args.benchmark else args.n_iter
    n_samples = args.n_samples
    scale = args.scale

    transformer_emb = models['transformer_emb']
    transformer_attn = models['transformer_attn']
    cond_stage_model = BERTEmbedder(transformer_emb, transformer_attn)

    prompt = args.input if isinstance(args.input, str) else args.input[0]
    logger.info("prompt: %s" % prompt)

    sample_path = os.path.join('outputs', prompt.replace(" ", "-"))
    os.makedirs(sample_path, exist_ok=True)
    base_count = len(os.listdir(sample_path))

    logger.info('Start inference...')
    uc = None
    if scale != 1.0:
        uc = cond_stage_model.encode([""] * n_samples)

    all_samples = []
    for i in range(n_iter):
        logger.info("iteration: %s" % (i + 1))

        if args.benchmark:
            logger.info('BENCHMARK mode')
            total_time_estimation = 0
            for i in range(args.benchmark_count):
                start = int(round(time.time() * 1000))
                x_samples = predict(models, cond_stage_model, prompt, uc)
                end = int(round(time.time() * 1000))
                estimation_time = (end - start)

                # Logging
                logger.info(f'\tailia processing estimation time {estimation_time} ms')
                if i != 0:
                    total_time_estimation = total_time_estimation + estimation_time

            logger.info(f'\taverage time estimation {total_time_estimation / (args.benchmark_count - 1)} ms')
        else:
            x_samples = predict(models, cond_stage_model, prompt, uc)

        for img in x_samples:
            sample_file = os.path.join(sample_path, f"{base_count:04}.png")
            cv2.imwrite(sample_file, img)
            base_count += 1

        x_samples = np.concatenate(x_samples, axis=1)
        all_samples.append(x_samples)

    grid_img = np.concatenate(all_samples, axis=0)

    # plot result
    savepath = get_savepath(args.savepath, "", ext='.png')
    logger.info(f'saved at : {savepath}')
    cv2.imwrite(savepath, grid_img)

    logger.info('Script finished successfully.')


def main():
    check_and_download_models(WEIGHT_TRANS_EMB_PATH, MODEL_TRANS_EMB_PATH, REMOTE_PATH)
    check_and_download_models(WEIGHT_TRANS_ATTN_PATH, MODEL_TRANS_ATTN_PATH, REMOTE_PATH)
    check_and_download_models(WEIGHT_DFSN_EMB_PATH, MODEL_DFSN_EMB_PATH, REMOTE_PATH)
    check_and_download_models(WEIGHT_DFSN_MID_PATH, MODEL_DFSN_MID_PATH, REMOTE_PATH)
    check_and_download_models(WEIGHT_DFSN_OUT_PATH, MODEL_DFSN_OUT_PATH, REMOTE_PATH)
    check_and_download_models(WEIGHT_AUTO_ENC_PATH, MODEL_AUTO_ENC_PATH, REMOTE_PATH)

    env_id = args.env_id

    # initialize
    if not args.onnx:
        logger.info("This model requires 10GB or more memory.")
        transformer_emb = ailia.Net(
            MODEL_TRANS_EMB_PATH, WEIGHT_TRANS_EMB_PATH, env_id=env_id, memory_mode=args.memory_mode)
        transformer_attn = ailia.Net(
            MODEL_TRANS_ATTN_PATH, WEIGHT_TRANS_ATTN_PATH, env_id=env_id, memory_mode=args.memory_mode)
        diffusion_emb = ailia.Net \
            (MODEL_DFSN_EMB_PATH, WEIGHT_DFSN_EMB_PATH, env_id=env_id, memory_mode=args.memory_mode)
        diffusion_mid = ailia.Net(
            MODEL_DFSN_MID_PATH, WEIGHT_DFSN_MID_PATH, env_id=env_id, memory_mode=args.memory_mode)
        diffusion_out = ailia.Net(
            MODEL_DFSN_OUT_PATH, WEIGHT_DFSN_OUT_PATH, env_id=env_id, memory_mode=args.memory_mode)
        autoencoder = ailia.Net(
            MODEL_AUTO_ENC_PATH, WEIGHT_AUTO_ENC_PATH, env_id=env_id, memory_mode=args.memory_mode)
    else:
        import onnxruntime
        transformer_emb = onnxruntime.InferenceSession(WEIGHT_TRANS_EMB_PATH)
        transformer_attn = onnxruntime.InferenceSession(WEIGHT_TRANS_ATTN_PATH)
        diffusion_emb = onnxruntime.InferenceSession(WEIGHT_DFSN_EMB_PATH)
        diffusion_mid = onnxruntime.InferenceSession(WEIGHT_DFSN_MID_PATH)
        diffusion_out = onnxruntime.InferenceSession(WEIGHT_DFSN_OUT_PATH)
        autoencoder = onnxruntime.InferenceSession(WEIGHT_AUTO_ENC_PATH)

    seed = args.seed
    if seed is not None:
        np.random.seed(seed)

    models = dict(
        transformer_emb=transformer_emb,
        transformer_attn=transformer_attn,
        diffusion_emb=diffusion_emb,
        diffusion_mid=diffusion_mid,
        diffusion_out=diffusion_out,
        autoencoder=autoencoder,
    )
    recognize_from_text(models)


if __name__ == '__main__':
    main()
