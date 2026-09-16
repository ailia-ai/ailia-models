import sys
import os
import time
import math
import itertools
from logging import getLogger

import numpy as np
import cv2
from PIL import Image
import matplotlib
from tqdm import tqdm
from scipy.optimize import minimize

import ailia

# import original modules
sys.path.append('../../util')
from arg_utils import get_base_parser, update_parser, get_savepath  # noqa
from model_utils import check_and_download_models, check_and_download_file  # noqa
from image_utils import normalize_image  # noqa
from detector_utils import load_image  # noqa
from webcamera_utils import get_capture, get_writer  # noqa

import df

logger = getLogger(__name__)

# ======================
# Parameters
# ======================

WEIGHT_UNET_PATH = 'unet.onnx'
MODEL_UNET_PATH = 'unet.onnx.prototxt'
WEIGHT_TEXT_ENCODER_PATH = 'text_encoder.onnx'
MODEL_TEXT_ENCODER_PATH = 'text_encoder.onnx.prototxt'
WEIGHT_VAE_ENCODER_PATH = 'vae_encoder.onnx'
MODEL_VAE_ENCODER_PATH = 'vae_encoder.onnx.prototxt'
WEIGHT_VAE_DECODER_PATH = 'vae_decoder.onnx'
MODEL_VAE_DECODER_PATH = 'vae_decoder.onnx.prototxt'
WEIGHT_UNET_PB_PATH = 'unet_weights.pb'
WEIGHT_TEXT_ENCODER_PB_PATH = 'text_encoder_weights.pb'
REMOTE_PATH = 'https://storage.googleapis.com/ailia-models/marigold/'

IMAGE_PATH = 'demo.jpg'
SAVE_IMAGE_PATH = 'output.png'

# ======================
# Arguemnt Parser Config
# ======================

parser = get_base_parser(
    'Marigold', IMAGE_PATH, SAVE_IMAGE_PATH
)
parser.add_argument(
    "--denoise_steps", type=int, default=10,
    help="Diffusion denoising steps, more stepts results in higher accuracy but slower inference speed.",
)
parser.add_argument(
    "--ensemble_size", type=int, default=1,
    help="Number of predictions to be ensembled, more inference gives better results but runs slower.",
)
parser.add_argument(
    "--batch_size", type=int, default=0,
    help="Inference batch size. Default: 0 (will be set automatically).",
)
parser.add_argument(
    "--seed", type=int, default=None,
    help="Random seed."
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
args = update_parser(parser)


# ======================
# Secondaty Functions
# ======================

def colorize_depth_maps(
        depth_map, min_depth, max_depth, cmap="Spectral"):
    """
    Colorize depth maps.
    """
    depth = depth_map.copy().squeeze()
    # reshape to [ (B,) H, W ]
    if depth.ndim < 3:
        depth = depth[np.newaxis, :, :]

    # colorize
    cm = matplotlib.colormaps[cmap]
    depth = ((depth - min_depth) / (max_depth - min_depth)).clip(0, 1)
    img_colored_np = cm(depth, bytes=False)[:, :, :, 0:3]  # value from 0 to 1
    img_colored_np = np.rollaxis(img_colored_np, 3, 1)

    img_colored = img_colored_np

    return img_colored


# ======================
# Main functions
# ======================

def preprocess(img):
    im_h, im_w, _ = img.shape

    max_edge_resolution = 768
    downscale_factor = min(
        max_edge_resolution / im_w, max_edge_resolution / im_h
    )
    ow = int(im_w * downscale_factor)
    oh = int(im_h * downscale_factor)
    img = np.array(Image.fromarray(img).resize((ow, oh), Image.Resampling.BICUBIC))

    img = normalize_image(img, normalize_type='255')

    img = img.transpose(2, 0, 1)  # HWC -> CHW
    # img = np.expand_dims(img, axis=0)
    img = img.astype(np.float32)

    return img


def encode_rgb(vae_encoder, rgb_in):
    # encode
    if not args.onnx:
        output = vae_encoder.predict([rgb_in])
    else:
        output = vae_encoder.run(None, {
            'sample': rgb_in
        })
    mean = output[0]

    # scale latent
    rgb_latent_scale_factor = 0.18215
    rgb_latent = mean * rgb_latent_scale_factor

    return rgb_latent


def single_infer(
        models,
        rgb_in,
        num_inference_steps):
    scheduler = models["scheduler"]
    scheduler.set_timesteps(num_inference_steps)
    timesteps = scheduler.timesteps  # [T]

    # Encode image
    vae_encoder = models["vae_encoder"]
    rgb_latent = encode_rgb(vae_encoder, rgb_in)

    # Initial depth map (noise)
    depth_latent = np.random.randn(
        rgb_latent.size,
    ).reshape(rgb_latent.shape).astype(np.float32)

    if not hasattr(single_infer, "empty_text_embed"):
        prompt = ""
        tokenizer = models["tokenizer"]
        text_inputs = tokenizer(
            prompt,
            padding="do_not_pad",
            max_length=77, #tokenizer.model_max_length,
            truncation=True,
            return_tensors="np",
        )
        text_input_ids = text_inputs.input_ids.astype(np.int32)

        text_encoder = models["text_encoder"]
        if not args.onnx:
            output = text_encoder.predict([text_input_ids])
        else:
            output = text_encoder.run(None, {
                'input_ids': text_input_ids
            })
        last_hidden_state = output[0]
        single_infer.empty_text_embed = last_hidden_state

    empty_text_embed = single_infer.empty_text_embed
    batch_empty_text_embed = np.repeat(empty_text_embed, rgb_latent.shape[0], axis=0)

    net = models["unet"]

    pbar = tqdm(
        enumerate(timesteps),
        total=len(timesteps),
        leave=False,
        desc=" " * 4 + "Diffusion denoising",
    )
    for i, t in pbar:
        timestep = np.array([t], dtype=np.float32)
        unet_input = np.concatenate(
            [rgb_latent, depth_latent], axis=1
        )
        unet_input = unet_input.astype(np.float32)

        if not args.onnx:
            output = net.predict([unet_input, timestep, batch_empty_text_embed])
        else:
            output = net.run(None, {
                'sample': unet_input, 'timestep': timestep,
                'encoder_hidden_states': batch_empty_text_embed
            })
        noise_pred = output[0]

        # compute the previous noisy sample x_t -> x_t-1
        depth_latent = scheduler.step(noise_pred, t, depth_latent)

    depth_latent_scale_factor = 0.18215
    depth_latent = depth_latent / depth_latent_scale_factor

    vae_decoder = models["vae_decoder"]
    if not args.onnx:
        output = vae_decoder.predict([depth_latent])
    else:
        output = vae_decoder.run(None, {
            'latent_sample': depth_latent
        })
    stacked = output[0]
    depth = np.mean(stacked, axis=1, keepdims=True)

    # clip prediction
    depth = np.clip(depth, -1.0, 1.0)
    # shift to [0, 1]
    depth = (depth + 1.0) / 2.0

    return depth


def ensemble_depths(
        input_images: np.ndarray,
        regularizer_strength: float = 0.02,
        max_iter: int = 2,
        tol: float = 1e-3,
        reduction: str = "median"):
    """
    To ensemble multiple affine-invariant depth images (up to scale and shift),
        by aligning estimating the scale and shift
    """
    original_input = np.copy(input_images)
    n_img = input_images.shape[0]

    # init guess
    _min = np.min(input_images.reshape((n_img, -1)), axis=1)
    _max = np.max(input_images.reshape((n_img, -1)), axis=1)
    s_init = 1.0 / (_max - _min).reshape((-1, 1, 1))
    t_init = (-1 * s_init.flatten() * _min.flatten()).reshape((-1, 1, 1))
    x = np.concatenate([s_init, t_init]).reshape(-1)

    def inter_distances(tensors: np.ndarray):
        """
        To calculate the distance between each two depth maps.
        """
        distances = []
        for i, j in itertools.combinations(np.arange(tensors.shape[0]), 2):
            arr1 = tensors[i: i + 1]
            arr2 = tensors[j: j + 1]
            distances.append(arr1 - arr2)
        dist = np.concatenate(distances, axis=0)
        return dist

    # objective function
    def closure(x):
        l = len(x)
        s = x[: l // 2]
        t = x[l // 2:]

        transformed_arrays = input_images * s.reshape((-1, 1, 1)) + t.reshape((-1, 1, 1))
        dists = inter_distances(transformed_arrays)
        sqrt_dist = np.sqrt(np.mean(dists ** 2))

        if "mean" == reduction:
            pred = np.mean(transformed_arrays, axis=0)
        elif "median" == reduction:
            if transformed_arrays.shape[0] % 2 == 0:
                pad = np.ones(transformed_arrays.shape[1:]) * -math.inf
                pad = np.expand_dims(pad, axis=0)
                transformed_arrays = np.concatenate([pad, transformed_arrays], axis=0)
            pred = np.median(transformed_arrays, axis=0)
        else:
            raise ValueError

        near_err = np.sqrt((0 - np.min(pred)) ** 2)
        far_err = np.sqrt((1 - np.max(pred)) ** 2)

        err = sqrt_dist + (near_err + far_err) * regularizer_strength
        err = err.astype(np.float32)
        return err

    res = minimize(
        closure, x, method="BFGS", tol=tol, options={"maxiter": max_iter, "disp": False}
    )
    x = res.x
    l = len(x)
    s = x[: l // 2]
    t = x[l // 2:]

    # Prediction
    transformed_arrays = original_input * s.reshape(-1, 1, 1) + t.reshape(-1, 1, 1)
    if "mean" == reduction:
        aligned_images = np.mean(transformed_arrays, axis=0)
        std = np.std(transformed_arrays, axis=0)
        uncertainty = std
    elif "median" == reduction:
        if transformed_arrays.shape[0] % 2 == 0:
            pad = np.ones(transformed_arrays.shape[1:]) * -math.inf
            pad = np.expand_dims(pad, axis=0)
            x = np.concatenate([pad, transformed_arrays], axis=0)
            aligned_images = np.median(x, axis=0)
        else:
            aligned_images = np.median(transformed_arrays, axis=0)
        # MAD (median absolute deviation) as uncertainty indicator
        abs_dev = np.abs(transformed_arrays - aligned_images)
        if abs_dev.shape[0] % 2 == 0:
            pad = np.ones(abs_dev.shape[1:]) * -math.inf
            pad = np.expand_dims(pad, axis=0)
            abs_dev = np.concatenate([pad, abs_dev], axis=0)
        mad = np.median(abs_dev, axis=0)
        uncertainty = mad
    else:
        raise ValueError(f"Unknown reduction method: {reduction}")

    # Scale and shift to [0, 1]
    _min = np.min(aligned_images)
    _max = np.max(aligned_images)
    aligned_images = (aligned_images - _min) / (_max - _min)
    uncertainty /= _max - _min

    return aligned_images, uncertainty


def predict(models, img):
    img = img[..., ::-1]  # BGR -> RGB
    h, w, _ = img.shape

    ensemble_size = args.ensemble_size
    denoising_steps = args.denoise_steps
    batch_size = args.batch_size

    img = preprocess(img)

    imgs = np.stack([img] * ensemble_size)

    if batch_size > 0:
        bs = batch_size
    else:
        bs = math.ceil(ensemble_size / 2)

    cnt = int(math.ceil(ensemble_size / bs))
    bar = tqdm(
        total=cnt, desc=" " * 2 + "Inference batches", leave=False
    )
    depth_pred_ls = []
    for i in range(cnt):
        batched_img = imgs[i * bs:(i + 1) * bs]
        depth_pred_raw = single_infer(
            models=models,
            rgb_in=batched_img,
            num_inference_steps=denoising_steps,
        )
        depth_pred_ls.append(depth_pred_raw)

        bar.update(1)

    depth_preds = np.concatenate(depth_pred_ls, axis=0)
    depth_preds = np.squeeze(depth_preds)

    if ensemble_size > 1:
        depth_pred, _ = ensemble_depths(depth_preds)
    else:
        depth_pred = depth_preds

    # Scale prediction to [0, 1]
    min_d = np.min(depth_pred)
    max_d = np.max(depth_pred)
    depth_pred = (depth_pred - min_d) / (max_d - min_d)

    pred_img = Image.fromarray(depth_pred)
    pred_img = pred_img.resize((w, h))
    depth_pred = np.asarray(pred_img)

    # Clip output range
    depth_pred = np.clip(depth_pred, 0, 1)

    return depth_pred


def recognize_from_image(models):
    # input image loop
    for image_path in args.input:
        logger.info(image_path)

        # prepare input data
        img = load_image(image_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)

        # inference
        logger.info('Start inference...')
        if args.benchmark:
            logger.info('BENCHMARK mode')
            total_time_estimation = 0
            for i in range(args.benchmark_count):
                start = int(round(time.time() * 1000))
                depth_pred = predict(models, img)
                end = int(round(time.time() * 1000))
                estimation_time = (end - start)

                # Logging
                logger.info(f'\tailia processing estimation time {estimation_time} ms')
                if i != 0:
                    total_time_estimation = total_time_estimation + estimation_time

            logger.info(f'\taverage time estimation {total_time_estimation / (args.benchmark_count - 1)} ms')
        else:
            depth_pred = predict(models, img)

        res_img = (depth_pred * 65535.0).astype(np.uint16)

        # plot result
        savepath = get_savepath(args.savepath, image_path, ext='.png')
        logger.info(f'saved at : {savepath}')
        cv2.imwrite(savepath, res_img)

        # Colorize
        depth_colored = colorize_depth_maps(
            depth_pred, 0, 1, cmap="Spectral",
        ).squeeze()  # [3, H, W], value in (0, 1)
        depth_colored = (depth_colored * 255).astype(np.uint8)
        depth_colored_img = depth_colored.transpose(1, 2, 0)  # CHW -> HWC
        depth_colored_img = depth_colored_img[:, :, ::-1]  # RGB -> BGR

        ex = os.path.splitext(savepath)
        savepath = "".join((ex[0] + "_colorize", ex[1]))
        logger.info(f'saved at : {savepath}')
        cv2.imwrite(savepath, depth_colored_img)

    logger.info('Script finished successfully.')


def main():
    # model files check and download
    check_and_download_models(WEIGHT_UNET_PATH, MODEL_UNET_PATH, REMOTE_PATH)
    check_and_download_models(WEIGHT_TEXT_ENCODER_PATH, MODEL_TEXT_ENCODER_PATH, REMOTE_PATH)
    check_and_download_models(WEIGHT_VAE_ENCODER_PATH, MODEL_VAE_ENCODER_PATH, REMOTE_PATH)
    check_and_download_models(WEIGHT_VAE_DECODER_PATH, MODEL_VAE_DECODER_PATH, REMOTE_PATH)
    check_and_download_file(WEIGHT_UNET_PB_PATH, REMOTE_PATH)
    check_and_download_file(WEIGHT_TEXT_ENCODER_PB_PATH, REMOTE_PATH)

    env_id = args.env_id
    seed = args.seed

    # initialize
    if not args.onnx:
        unet = ailia.Net(
            MODEL_UNET_PATH, WEIGHT_UNET_PATH,
            env_id=env_id, memory_mode=args.memory_mode)
        text_encoder = ailia.Net(
            MODEL_TEXT_ENCODER_PATH, WEIGHT_TEXT_ENCODER_PATH,
            env_id=env_id, memory_mode=args.memory_mode)
        vae_encoder = ailia.Net(
            MODEL_VAE_ENCODER_PATH, WEIGHT_VAE_ENCODER_PATH,
            env_id=env_id, memory_mode=args.memory_mode)
        vae_decoder = ailia.Net(
            MODEL_VAE_DECODER_PATH, WEIGHT_VAE_DECODER_PATH,
            env_id=env_id, memory_mode=args.memory_mode)
    else:
        import onnxruntime
        cuda = 0 < ailia.get_gpu_environment_id()
        providers = ['CUDAExecutionProvider', 'CPUExecutionProvider'] if cuda else ['CPUExecutionProvider']
        unet = onnxruntime.InferenceSession(WEIGHT_UNET_PATH, providers=providers)
        text_encoder = onnxruntime.InferenceSession(WEIGHT_TEXT_ENCODER_PATH, providers=providers)
        vae_encoder = onnxruntime.InferenceSession(WEIGHT_VAE_ENCODER_PATH, providers=providers)
        vae_decoder = onnxruntime.InferenceSession(WEIGHT_VAE_DECODER_PATH, providers=providers)

    if args.disable_ailia_tokenizer:
        from transformers import CLIPTokenizer
        tokenizer = CLIPTokenizer.from_pretrained("tokenizer")
    else:
        from ailia_tokenizer import CLIPTokenizer
        tokenizer = CLIPTokenizer.from_pretrained()

    scheduler = df.schedulers.DDIMScheduler.from_config({
        "prediction_type": "v_prediction",
        "clip_sample": False,
        "beta_schedule": "scaled_linear",
        "beta_start": 0.00085,
        "set_alpha_to_one": False,
        "trained_betas": None,
        "steps_offset": 1,
        "beta_end": 0.012,
        "num_train_timesteps": 1000,
    })

    models = {
        "unet": unet,
        "text_encoder": text_encoder,
        "vae_encoder": vae_encoder,
        "vae_decoder": vae_decoder,
        "tokenizer": tokenizer,
        "scheduler": scheduler,
    }

    if seed is None:
        seed = int(time.time())
    np.random.seed(seed)

    recognize_from_image(models)


if __name__ == '__main__':
    main()
