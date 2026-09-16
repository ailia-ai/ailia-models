import sys
import time

import numpy as np
import cv2

import ailia

# import original modules
sys.path.append('../../util')
from arg_utils import get_base_parser, update_parser, get_savepath  # noqa
from model_utils import check_and_download_models  # noqa
from detector_utils import load_image  # noqa
# logger
from logging import getLogger  # noqa

import annotator.common
from annotator.canny import CannyDetector
from annotator.openpose import OpenposeDetector
from annotator.uniformer import UniformerDetector
from constants import alphas_cumprod

logger = getLogger(__name__)

# ======================
# Parameters
# ======================

WEIGHT_CANNY_PATH = 'control_net_canny.onnx'
MODEL_CANNY_PATH = 'control_net_canny.onnx.prototxt'
WEIGHT_POSE_PATH = 'control_net_pose.onnx'
MODEL_POSE_PATH = 'control_net_pose.onnx.prototxt'
WEIGHT_SEG_PATH = 'control_net_seg.onnx'
MODEL_SEG_PATH = 'control_net_seg.onnx.prototxt'
WEIGHT_POSE_BODY_PATH = 'pose_body.onnx'
MODEL_POSE_BODY_PATH = 'pose_body.onnx.prototxt'
WEIGHT_POSE_HAND_PATH = 'pose_hand.onnx'
MODEL_POSE_HAND_PATH = 'pose_hand.onnx.prototxt'
WEIGHT_SEG_UNIF_PATH = 'upernet_global_small.onnx'
MODEL_SEG_UNIF_PATH = 'upernet_global_small.onnx.prototxt'
REMOTE_PATH = 'https://storage.googleapis.com/ailia-models/control_net/'

WEIGHT_DFSN_EMB_PATH = 'diffusion_emb.onnx'
MODEL_DFSN_EMB_PATH = 'diffusion_emb.onnx.prototxt'
WEIGHT_DFSN_MID_PATH = 'diffusion_mid.onnx'
MODEL_DFSN_MID_PATH = 'diffusion_mid.onnx.prototxt'
WEIGHT_DFSN_OUT_PATH = 'diffusion_out.onnx'
MODEL_DFSN_OUT_PATH = 'diffusion_out.onnx.prototxt'
WEIGHT_AUTO_ENC_PATH = 'autoencoder.onnx'
MODEL_AUTO_ENC_PATH = 'autoencoder.onnx.prototxt'
REMOTE_PATH_SD = 'https://storage.googleapis.com/ailia-models/stable-diffusion-txt2img/'

WEIGHT_VITL14_TEXT_PATH = 'ViT-L14-encode_text.onnx'
MODEL_VITL14_TEXT_PATH = 'ViT-L14-encode_text.onnx.prototxt'
CLIP_REMOTE_PATH = 'https://storage.googleapis.com/ailia-models/clip/'

IMAGE_PATH = 'examples/bird.png'
SAVE_IMAGE_PATH = 'output.png'

# ======================
# Arguemnt Parser Config
# ======================

parser = get_base_parser(
    'ControlNet', IMAGE_PATH, SAVE_IMAGE_PATH
)
parser.add_argument(
    "-p", "--prompt", metavar="TEXT", type=str,
    default="bird",
    help="the prompt to render"
)
parser.add_argument(
    "--a_prompt", metavar="TEXT", type=str,
    default="best quality, extremely detailed",
    help="Added Prompt"
)
parser.add_argument(
    "--n_prompt", metavar="TEXT", type=str,
    default="longbody, lowres, bad anatomy, bad hands, missing fingers,"
            " extra digit, fewer digits, cropped, worst quality, low quality",
    help="Negative Prompt"
)
parser.add_argument(
    "--n_samples", type=int, default=1,
    help="how many samples to produce for the given prompt",
)
parser.add_argument(
    "--ddim_steps", type=int, default=20,
    help="number of ddim sampling steps",
)
parser.add_argument(
    "--ddim_eta", type=float, default=0.0,
    help="ddim eta (eta=0.0 corresponds to deterministic sampling)",
)
parser.add_argument(
    "--image_resolution", type=int, default=512,
    help="Image Resolution, in pixel space",
)
parser.add_argument(
    "--scale", type=float, default=9.0,
    help="unconditional guidance scale: eps = eps(x, empty) + scale * (eps(x, cond) - eps(x, empty))",
)
parser.add_argument(
    "--seed", type=int, default=None,
    help="random seed",
)
parser.add_argument(
    '-m', '--model_type', default='canny', choices=('canny', 'pose', 'seg'),
    help='Select annotator model.'
)
parser.add_argument(
    '--hand_detect', action='store_true',
    help='Using hand models in human poses.'
)
parser.add_argument(
    '--onnx',
    action='store_true',
    help='execute onnxruntime version.'
)
parser.add_argument(
    '--transformers_clip',
    action='store_true',
    help='use transformer version of clip.'
)
parser.add_argument(
    '--disable_ailia_tokenizer',
    action='store_true',
    help='disable ailia tokenizer.'
)
args = update_parser(parser)


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


def HWC3(x):
    assert x.dtype == np.uint8
    if x.ndim == 2:
        x = x[:, :, None]
    assert x.ndim == 3
    H, W, C = x.shape
    assert C == 1 or C == 3 or C == 4
    if C == 3:
        return x
    if C == 1:
        return np.concatenate([x, x, x], axis=2)
    if C == 4:
        color = x[:, :, 0:3].astype(np.float32)
        alpha = x[:, :, 3:4].astype(np.float32) / 255.0
        y = color * alpha + 255.0 * (1.0 - alpha)
        y = y.clip(0, 255).astype(np.uint8)
        return y


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

        if args.benchmark:
            start = int(round(time.time() * 1000))

        img, pred_x0 = p_sample_ddim(
            models,
            img, cond, ts,
            index=index,
            unconditional_guidance_scale=unconditional_guidance_scale,
            unconditional_conditioning=unconditional_conditioning,
            update_context=(i == 0)
        )

        if args.benchmark:
            end = int(round(time.time() * 1000))
            estimation_time = (end - start)
            logger.info(f'\tailia processing estimation time {estimation_time} ms')

        img = img.astype(np.float32)

    return img


# ddim
def p_sample_ddim(
        models, x, c, t, index,
        temperature=1.,
        unconditional_guidance_scale=1.,
        unconditional_conditioning=None,
        update_context=True):
    if unconditional_guidance_scale == 1.:
        model_output = apply_model(models, x, t, c, update_context)
    else:
        model_t = apply_model(models, x, t, c, update_context)
        model_uncond = apply_model(models, x, t, unconditional_conditioning, update_context)
        model_output = model_uncond + unconditional_guidance_scale * (model_t - model_uncond)

    e_t = model_output

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


# diffusion_model
def apply_model(models, x_noisy, t, cond, update_context=True):
    control_net = models["control_net"]
    diffusion_emb = models["diffusion_emb"]
    diffusion_mid = models["diffusion_mid"]
    diffusion_out = models["diffusion_out"]

    hint = np.concatenate(cond['c_concat'], axis=1)
    cond_txt = np.concatenate(cond['c_crossattn'], axis=1)

    if not args.onnx:
        if not FIX_CONSTANT_CONTEXT or update_context:
            output = control_net.predict([x_noisy, hint, t, cond_txt])
        else:
            output = control_net.predict([x_noisy, hint, t])
    else:
        output = control_net.run(None, {'x': x_noisy, 'hint': hint, 'timesteps': t, 'context': cond_txt})
    control = output

    x_noisy = x_noisy.astype(np.float32)
    if not args.onnx:
        if not FIX_CONSTANT_CONTEXT or update_context:
            output = diffusion_emb.predict([x_noisy, t, cond_txt])
        else:
            output = diffusion_emb.predict([x_noisy, t])
    else:
        output = diffusion_emb.run(None, {'x': x_noisy, 'timesteps': t, 'context': cond_txt})
    h, emb, *hs = output

    hs = [(x + v).astype(np.float32) for x, v in zip([*hs, h], control)]
    h = hs.pop()

    if not args.onnx:
        if not FIX_CONSTANT_CONTEXT or update_context:
            output = diffusion_mid.predict([h, emb, cond_txt, *hs[6:]])
        else:
            output = diffusion_mid.run({
                'h': h, 'emb': emb,
                'h6': hs[6], 'h7': hs[7], 'h8': hs[8],
                'h9': hs[9], 'h10': hs[10], 'h11': hs[11],
            })
    else:
        output = diffusion_mid.run(None, {
            'h': h, 'emb': emb, 'context': cond_txt,
            'h6': hs[6], 'h7': hs[7], 'h8': hs[8],
            'h9': hs[9], 'h10': hs[10], 'h11': hs[11],
        })
    h = output[0]

    if not args.onnx:
        if not FIX_CONSTANT_CONTEXT or update_context:
            output = diffusion_out.predict([h, emb, cond_txt, *hs[:6]])
        else:
            output = diffusion_out.run({
                'h': h, 'emb': emb,
                'h0': hs[0], 'h1': hs[1], 'h2': hs[2],
                'h3': hs[3], 'h4': hs[4], 'h5': hs[5],
            })
    else:
        output = diffusion_out.run(None, {
            'h': h, 'emb': emb, 'context': cond_txt,
            'h0': hs[0], 'h1': hs[1], 'h2': hs[2],
            'h3': hs[3], 'h4': hs[4], 'h5': hs[5],
        })
    out = output[0]

    return out


def setup_detector(det_model, net, ext_net):
    if det_model == "canny":
        detector = CannyDetector()
    elif det_model == "pose":
        detector = OpenposeDetector(net, ext_net)
    elif det_model == "seg":
        detector = UniformerDetector(net)

    return detector


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


def preprocess(img, image_resolution):
    im_h, im_w, _ = img.shape

    k = image_resolution / min(im_h, im_w)
    ow, oh = im_w * k, im_h * k
    oh = int(np.round(oh / 64.0)) * 64
    ow = int(np.round(ow / 64.0)) * 64
    img = cv2.resize(img, (ow, oh), interpolation=cv2.INTER_LANCZOS4 if k > 1 else cv2.INTER_AREA)

    return img


def predict(
        models, img,
        prompt, a_prompt, n_prompt):
    detect_resolution = 512
    image_resolution = args.image_resolution
    num_samples = args.n_samples
    scale = args.scale

    guess_mode = False

    H, W, _ = preprocess(img, image_resolution).shape

    img = preprocess(img, detect_resolution)  # BGR
    detector = models["detector"]
    detected_map = detector(img)
    detected_map = HWC3(detected_map)
    detected_map = cv2.resize(detected_map, (W, H), interpolation=cv2.INTER_NEAREST)
    control = np.stack([
        detected_map for _ in range(num_samples)
    ], axis=0).astype(np.float32) / 255
    control = control.transpose((0, 3, 1, 2))  # HWC -> CHW

    cond_stage_model = FrozenCLIPEmbedder(onnx = models["clip"])
    cond = {
        "c_concat": [control],
        "c_crossattn": [cond_stage_model.encode([prompt + ', ' + a_prompt] * num_samples)]
    }
    un_cond = {
        "c_concat": None if guess_mode else [control],
        "c_crossattn": [cond_stage_model.encode([n_prompt] * num_samples)]
    }
    shape = (num_samples, 4, H // 8, W // 8)

    samples = ddim_sampling(
        models, cond, shape,
        unconditional_guidance_scale=scale,
        unconditional_conditioning=un_cond)

    if args.benchmark:
        start = int(round(time.time() * 1000))

    x_samples = decode_first_stage(models, samples)

    if args.benchmark:
        end = int(round(time.time() * 1000))
        estimation_time = (end - start)
        logger.info(f'\tailia processing estimation time {estimation_time} ms')

    x_samples = np.clip(x_samples * 127.5 + 127.5, a_min=0, a_max=255)
    x_samples = x_samples.transpose((0, 2, 3, 1)).astype(np.uint8)  # CHW -> HWC
    x_samples = [x[:, :, ::-1] for x in x_samples]  # RGB -> BGR

    return [detector.map2img(detected_map)] + x_samples


def recognize_from_image_text(models):
    prompt = args.prompt
    a_prompt = args.a_prompt
    n_prompt = args.n_prompt
    image_path = args.input[0]

    logger.info("image_path: %s" % image_path)
    logger.info("prompt: %s" % prompt)
    logger.info("a_prompt: %s" % a_prompt)
    logger.info("n_prompt: %s" % n_prompt)

    # prepare input data
    img = load_image(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)

    logger.info('Start inference...')

    if args.benchmark:
        logger.info('BENCHMARK mode')
        start = int(round(time.time() * 1000))

    x_samples = predict(models, img, prompt, a_prompt, n_prompt)
    x_samples = np.concatenate(x_samples, axis=1)

    if args.benchmark:
        end = int(round(time.time() * 1000))
        estimation_time = (end - start)
        logger.info(f'\ttotal time estimation {estimation_time} ms')

    # plot result
    savepath = get_savepath(args.savepath, "", ext='.png')
    logger.info(f'saved at : {savepath}')
    cv2.imwrite(savepath, x_samples)

    logger.info('Script finished successfully.')


def main():
    dic_model = {
        'canny': (WEIGHT_CANNY_PATH, MODEL_CANNY_PATH),
        'pose': (WEIGHT_POSE_PATH, MODEL_POSE_PATH),
        'seg': (WEIGHT_SEG_PATH, MODEL_SEG_PATH),
    }
    WEIGHT_PATH, MODEL_PATH = dic_model[args.model_type]
    check_and_download_models(WEIGHT_PATH, MODEL_PATH, REMOTE_PATH)
    check_and_download_models(WEIGHT_DFSN_EMB_PATH, MODEL_DFSN_EMB_PATH, REMOTE_PATH_SD)
    check_and_download_models(WEIGHT_DFSN_MID_PATH, MODEL_DFSN_MID_PATH, REMOTE_PATH_SD)
    check_and_download_models(WEIGHT_DFSN_OUT_PATH, MODEL_DFSN_OUT_PATH, REMOTE_PATH_SD)
    check_and_download_models(WEIGHT_AUTO_ENC_PATH, MODEL_AUTO_ENC_PATH, REMOTE_PATH_SD)
    check_and_download_models(WEIGHT_VITL14_TEXT_PATH, MODEL_VITL14_TEXT_PATH, CLIP_REMOTE_PATH)

    det_model = args.model_type

    if det_model == "pose":
        check_and_download_models(WEIGHT_POSE_BODY_PATH, MODEL_POSE_BODY_PATH, REMOTE_PATH)
        if args.hand_detect:
            check_and_download_models(WEIGHT_POSE_HAND_PATH, MODEL_POSE_HAND_PATH, REMOTE_PATH)
    elif det_model == "seg":
        check_and_download_models(WEIGHT_SEG_UNIF_PATH, MODEL_SEG_UNIF_PATH, REMOTE_PATH)

    env_id = args.env_id

    # initialize
    if not args.onnx:
        logger.info("This model requires 10GB or more memory.")
        control_net = ailia.Net(
            MODEL_PATH, WEIGHT_PATH, env_id=env_id, memory_mode=args.memory_mode)
        diffusion_emb = ailia.Net(
            MODEL_DFSN_EMB_PATH, WEIGHT_DFSN_EMB_PATH, env_id=env_id, memory_mode=args.memory_mode)
        diffusion_mid = ailia.Net(
            MODEL_DFSN_MID_PATH, WEIGHT_DFSN_MID_PATH, env_id=env_id, memory_mode=args.memory_mode)
        diffusion_out = ailia.Net(
            MODEL_DFSN_OUT_PATH, WEIGHT_DFSN_OUT_PATH, env_id=env_id, memory_mode=args.memory_mode)
        autoencoder = ailia.Net(
            MODEL_AUTO_ENC_PATH, WEIGHT_AUTO_ENC_PATH, env_id=env_id, memory_mode=args.memory_mode)
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
        control_net = onnxruntime.InferenceSession(WEIGHT_PATH)
        diffusion_emb = onnxruntime.InferenceSession(WEIGHT_DFSN_EMB_PATH)
        diffusion_mid = onnxruntime.InferenceSession(WEIGHT_DFSN_MID_PATH)
        diffusion_out = onnxruntime.InferenceSession(WEIGHT_DFSN_OUT_PATH)
        autoencoder = onnxruntime.InferenceSession(WEIGHT_AUTO_ENC_PATH)
        annotator.common.onnx = True
        clip = None

    det_net = None
    ext_net = None
    if det_model == "pose":
        if not args.onnx:
            det_net = ailia.Net(
                MODEL_POSE_BODY_PATH, WEIGHT_POSE_BODY_PATH, env_id=env_id, memory_mode=args.memory_mode)
            if args.hand_detect:
                ext_net = ailia.Net(
                    MODEL_POSE_HAND_PATH, WEIGHT_POSE_HAND_PATH, env_id=env_id, memory_mode=args.memory_mode)
        else:
            det_net = onnxruntime.InferenceSession(WEIGHT_POSE_BODY_PATH)
            if args.hand_detect:
                ext_net = onnxruntime.InferenceSession(WEIGHT_POSE_HAND_PATH)
    elif det_model == "seg":
        if not args.onnx:
            det_net = ailia.Net(
                MODEL_SEG_UNIF_PATH, WEIGHT_SEG_UNIF_PATH, env_id=env_id, memory_mode=args.memory_mode)
        else:
            det_net = onnxruntime.InferenceSession(WEIGHT_SEG_UNIF_PATH)

    detector = setup_detector(det_model, det_net, ext_net)

    seed = args.seed
    if seed is not None:
        np.random.seed(seed)

    models = dict(
        detector=detector,
        control_net=control_net,
        diffusion_emb=diffusion_emb,
        diffusion_mid=diffusion_mid,
        diffusion_out=diffusion_out,
        autoencoder=autoencoder,
        clip=clip
    )
    recognize_from_image_text(models)


if __name__ == '__main__':
    main()
