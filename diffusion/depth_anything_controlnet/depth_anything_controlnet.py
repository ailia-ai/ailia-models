import cv2
import numpy as np

import ailia

import sys
sys.path.append('../../util')
from image_utils import imread  # noqa: E402
from arg_utils import get_base_parser, update_parser, get_savepath  # noqa
from model_utils import check_and_download_models, check_and_download_file # noqa
# logger
from logging import getLogger  # noqa: E402

logger = getLogger(__name__)

parser = get_base_parser(
    'DepthAnything ControlNet', 'depth_flower.png', 'output.png', large_model = True
)

parser.add_argument('--input', '-i', type=str, default='depth_flower.png', help='Input image path')
parser.add_argument('--width', type=int, default=384, help='Width of the input image')
parser.add_argument('--height', type=int, default=384, help='Height of the input image')
parser.add_argument('--crop', action='store_true', help='Crop the input image after resizing')

parser.add_argument('--prompt', type=str, default='A beautiful flower garden full of tulips.', help='Prompt')
parser.add_argument('--negative_prompt', type=str, default='', help='Negative prompt')
parser.add_argument('--n_timesteps', type=int, default=20, help='Number of timesteps')
parser.add_argument('--sampler', type=str, default='ddpm', help='Sampler type')

parser.add_argument('--seed', type=int, default=0, help='Random seed')

parser.add_argument('--ddim_eta', type=float, default=0.0, help='eta used in ddim.')
parser.add_argument('--guidance_scale', type=float, default=7.5, help='Guidance scale')

parser.add_argument(
    '--disable_ailia_tokenizer',
    action='store_true',
    help='disable ailia tokenizer.'
)

args = update_parser(parser)

TOKENIZER_PATH = "./tokenizer"

REMOTE_PATH = 'https://storage.googleapis.com/ailia-models/depth_anything_controlnet/'

TEXT_ENCODER_MODEL_PATH = "text_encoder.onnx.prototxt"
UNET_MODEL_PATH = "unet.onnx.prototxt"
VAE_DECODER_MODEL_PATH = "vae_decoder.onnx.prototxt"

TEXT_ENCODER_WEIGHT_PATH = "text_encoder.onnx"
UNET_WEIGHT_PATH = "unet.onnx"
VAE_DECODER_WEIGHT_PATH = "vae_decoder.onnx"

WEIGHT_PB_PATH = "weights.pb"

# ======================
# Main functions
# ======================

def create_initial_latents(w, h):
    return np.random.randn(1, 4, h // 8, w // 8).astype(np.float32)

def control_preprocessing(image, w, h):
    image = cv2.resize(image, (w, h))
    cv2.imwrite('depth_flower.png', image)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = image / 255.0
    image = image.transpose((2,0,1))[None]
    image = np.tile(image, (2, 1, 1, 1))
    return image

def encode_prompt(text_encoder, tokenizer, prompt, negative_prompt):
    prompt = tokenizer(
        [negative_prompt] + [prompt],
        padding="max_length",
        max_length=77,
        truncation=True,
        return_tensors="np",
    ).input_ids
    encoded_prompt = text_encoder.predict([prompt])[0]
    text_encoder = None
    return encoded_prompt

class noise_scheduler():
    def __init__(self, n_inference_timesteps, beta_1, beta_T, type, n_train_timesteps=1000):
        if type == 'scaled_linear':
            self.betas = np.concatenate(([0],(np.linspace(beta_1**0.5, beta_T**0.5, n_train_timesteps, dtype=np.float32) ** 2)))
        else:
            raise NotImplementedError
        self._generate_alpha_prods()
        
        if n_inference_timesteps != n_train_timesteps:
            step_ratio = n_train_timesteps // n_inference_timesteps
            timesteps = (np.arange(0, n_inference_timesteps+1) * step_ratio).round().astype(np.int64)
            self.betas = self.respace_betas(timesteps)
        else:
            timesteps = list(range(n_inference_timesteps+1))
        self.timesteps = timesteps
        self.timestep_index = 0
        self._generate_alpha_prods()
    def respace_betas(self, timesteps):
        resampled_betas = []
        for i,t in enumerate(timesteps):
            if i == 0:
                resampled_betas.append(1 - np.prod(1 - self.betas[:timesteps[0]+1]))
            else:
                resampled_betas.append(1 - np.prod(1 - self.betas[timesteps[i-1]+1: timesteps[i]+1]))
        return resampled_betas
    def _generate_alpha_prods(self):
        self.alpha_prods = [1] * len(self.betas)
        self.alpha_prods[0] = 1 - self.betas[0]
        for t in range(1, len(self.betas)):
            self.alpha_prods[t] = self.alpha_prods[t-1] * (1 - self.betas[t])
    def beta(self, t):
        return self.betas[t]
    def alpha(self, t):
        return 1 - self.betas[t]
    def alpha_prod(self, t):
        return self.alpha_prods[t]
    def beta_prod(self, t):
        return 1 - self.alpha_prods[t]
    def to_train_timestep(self, t):
        return self.timesteps[t]
    def step(self):
        self.timestep_index += 1

def step_ddpm(t, pred_noise, sample, ns):
    sample = 1 / (1 - ns.beta(t))**0.5 * (sample - ns.beta(t)/(1 - ns.alpha_prod(t))**0.5 * pred_noise)
    std = (((1 - ns.alpha_prod(t-1)) / (1 - ns.alpha_prod(t))) * ns.beta(t)) ** 0.5
    sample = sample + std * np.random.randn(*sample.shape).astype(np.float32)
    return sample

def step_ddim(t, pred_noise, sample, ns, eta=0):
    sample = (
        (ns.alpha_prod(t-1)**0.5*(sample - (1 - ns.alpha_prod(t))**0.5 * pred_noise) / ns.alpha_prod(t)**0.5) +# scale the sample
        (1 - ns.alpha_prod(t-1) - eta**2)**0.5 * pred_noise +# substract the predicted noise
        eta * np.random.randn(*sample.shape).astype(np.float32))# variance
    return sample


def generate(
        models, control, initial_latent,
        prompt, n_prompt, n_inference_timesteps, guidance_scale = 7.5):

    encoded_prompt = encode_prompt(models['text_encoder'], models['tokenizer'], prompt, n_prompt)

    ns = noise_scheduler(n_inference_timesteps, 0.00085, 0.012, 'scaled_linear')
    
    sample = initial_latent
    # t in this loop corresponds to the index of tau, the rescaled version of the timestep
    for t in reversed(range(1, n_inference_timesteps+1)):
        logger.info("iteration: %s" % (n_inference_timesteps - t + 1))
        
        pred_noise = models['unet'].predict([
            np.tile(sample, (2, 1, 1, 1)),
            np.array([ns.to_train_timestep(t)]),
            encoded_prompt,
            control[None],
            np.array([[1.0]]),
        ])[0]

        # classifier free guidance
        noise_pred_uncond, noise_pred_text = pred_noise[0], pred_noise[1]
        pred_noise = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

        if args.sampler == 'ddpm':
            sample = step_ddpm(t, pred_noise, sample, ns)
        elif args.sampler == 'ddim':
            sample = step_ddim(t, pred_noise, sample, ns)
        else:
            raise ValueError(f"Sampler {args.sampler} is not implemented.")
    
    decoder_out = models['vae_decoder'].predict(sample/ 0.18215)[0].astype('float32')
    decoder_out = ((decoder_out / 2 + 0.5).clip(0, 1).astype('float32').transpose((1,2,0)) * 255).astype('uint8')
    decoder_out = cv2.cvtColor(decoder_out, cv2.COLOR_RGB2BGR)
    return decoder_out



def generate_from_image_text(models, w, h, control_image_path, prompt, negative_prompt, output_path, timesteps=1000, guidance_scale=7.5):
    control = imread(control_image_path)
    control = control_preprocessing(control, w, h)
    sample = create_initial_latents(w, h)
    image = generate(models, control, sample, prompt, negative_prompt, timesteps, guidance_scale)
    
    cv2.imwrite(output_path, image)
    logger.info(f'Result saved at {output_path}')
    logger.info('Script finished successfully.')


def main():
    seed = args.seed
    if seed is not None:
        np.random.seed(seed)

    check_and_download_models(TEXT_ENCODER_MODEL_PATH, TEXT_ENCODER_WEIGHT_PATH, REMOTE_PATH)
    check_and_download_models(UNET_MODEL_PATH, UNET_WEIGHT_PATH, REMOTE_PATH)
    check_and_download_models(VAE_DECODER_MODEL_PATH, VAE_DECODER_WEIGHT_PATH, REMOTE_PATH)
    check_and_download_file(WEIGHT_PB_PATH, REMOTE_PATH)
 
    if args.disable_ailia_tokenizer:
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_PATH)
    else:
        from ailia_tokenizer import CLIPTokenizer
        tokenizer = CLIPTokenizer.from_pretrained(TOKENIZER_PATH)

    models = dict(
        text_encoder = ailia.Net(TEXT_ENCODER_MODEL_PATH, TEXT_ENCODER_WEIGHT_PATH, args.env_id, memory_mode = args.memory_mode),
        tokenizer = tokenizer,
        unet = ailia.Net(UNET_MODEL_PATH, UNET_WEIGHT_PATH, args.env_id, memory_mode = args.memory_mode),
        vae_decoder = ailia.Net(VAE_DECODER_MODEL_PATH, VAE_DECODER_WEIGHT_PATH, args.env_id, memory_mode = args.memory_mode)
    )

    generate_from_image_text(
        models,
        args.width,
        args.height,
        args.input[0],
        args.prompt,
        args.negative_prompt,
        args.savepath,
        args.n_timesteps,
        args.guidance_scale,
    )


if __name__ == '__main__':
    main()
