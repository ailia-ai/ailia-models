import sys
import numpy as np
import cv2
import imageio.v2 as imageio
import tqdm
import transformers
import rembg

import ailia

# import original modules
sys.path.append("../../util")
from image_utils import imread  # noqa
from arg_utils import get_base_parser, update_parser, get_savepath  # noqa
from model_utils import check_and_download_models, check_and_download_file  # noqa

# logger
from logging import getLogger  # noqa
logger = getLogger(__name__)

from df.pipelines.pipeline_mvdream import MVDreamPipeline
from df.schedulers.scheduling_ddim import DDIMScheduler
from lgm_utils.data_process import process_mv_image, preprocess_lgm_model
from lgm_utils.gs import GaussianRenderer, save_ply
from lgm_utils.kiui import orbit_camera

# ======================
# Parameters
# ======================
WEIGHT_UNET_IMAGE_PATH = "unet_image.onnx"
MODEL_UNET_IMAGE_PATH = "unet_image.onnx.prototxt"
WEIGHT_UNET_TEXT_PATH = "unet_text.onnx"
MODEL_UNET_TEXT_PATH = "unet_text.onnx.prototxt"
WEIGHT_VAE_ENCODER_PATH = "vae_encoder.onnx"
MODEL_VAE_ENCODER_PATH = "vae_encoder.onnx.prototxt"
WEIGHT_VAE_DECODER_PATH = "vae_decoder.onnx"
MODEL_VAE_DECODER_PATH = "vae_decoder.onnx.prototxt"
WEIGHT_TEXT_ENCODER_PATH = "text_encoder.onnx"
MODEL_TEXT_ENCODER_PATH = "text_encoder.onnx.prototxt"
WEIGHT_IMAGE_ENCODER_PATH = "image_encoder.onnx"
MODEL_IMAGE_ENCODER_PATH = "image_encoder.onnx.prototxt"
WEIGHT_LGM_PATH = "lgm.onnx"
MODEL_LGM_PATH = "lgm.onnx.prototxt"

REMOTE_PATH = "https://storage.googleapis.com/ailia-models/lgm/"

INPUT_SIZE = 256
FOVY = 49.1
CAM_RADIUS = 1.5
ZFAR = 2.5
ZNEAR = 0.5
IMAGE_PATH = 'catstatue_rgba.png'
SAVE_VIDEO_PATH = 'output.gif'

# ======================
# Arguemnt Parser Config
# ======================
parser = get_base_parser(
    'lgm', IMAGE_PATH, SAVE_VIDEO_PATH
)
parser.add_argument(
    "--prompt",
    metavar="TEXT",
    type=str,
    default=None,
    help="the prompt to render",
)
parser.add_argument(
    "--seed",
    type=int,
    default=2,
    help="random seed",
)
parser.add_argument(
    "--output_size",
    type=int,
    default=256,
    help="output size",
)
parser.add_argument(
    "--save_ply",
    action="store_true",
    help="save gaussians as ply file",
)
parser.add_argument(
    '--disable_ailia_tokenizer',
    action='store_true',
    help='disable ailia tokenizer.'
)
parser.add_argument(
    '-o',
    '--onnx',
    action='store_true',
    help="Option to use onnxrutime to run or not."
)
args = update_parser(parser)

# ======================
# Utils
# ======================
class Options:
    input_size: int = INPUT_SIZE
    output_size: int = args.output_size
    fovy: float = FOVY
    cam_radius: float = CAM_RADIUS
    zfar: float = ZFAR
    znear: float = ZNEAR

opt = Options()

def load_image(image_path):
    image = imread(image_path, flags=cv2.IMREAD_UNCHANGED)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    if image.dtype != np.uint8:
        image = (image * 255).astype(np.uint8)
    return image

def load_model(weight_path, model_path, env_id=None, memory_mode=None, use_onnx=False):
    if use_onnx:
        import onnxruntime
        cuda = 0 < ailia.get_gpu_environment_id()
        providers = (
            ["CUDAExecutionProvider", "CPUExecutionProvider"]
            if cuda
            else ["CPUExecutionProvider"]
        )
        return onnxruntime.InferenceSession(weight_path, providers=providers)
    else:
        return ailia.Net(model_path, weight_path, env_id=env_id, memory_mode=memory_mode)

def generate_multi_view_images(
    input_image, 
    prompt, 
    prompt_neg='', 
    input_elevation=0, 
    input_num_steps=30
):
    env_id = args.env_id
    user_onnx = args.onnx
    bg_remover = rembg.new_session()

    memory_mode = None
    if not user_onnx:
        memory_mode = ailia.get_memory_mode(
            reduce_constant=True,
            ignore_input_with_initializer=True,
            reduce_interstage=False,
            reuse_interstage=True,
        )

    check_and_download_models(WEIGHT_VAE_DECODER_PATH, MODEL_VAE_DECODER_PATH, REMOTE_PATH)
    check_and_download_models(WEIGHT_TEXT_ENCODER_PATH, MODEL_TEXT_ENCODER_PATH, REMOTE_PATH)
    vae_decoder = load_model(
        WEIGHT_VAE_DECODER_PATH, MODEL_VAE_DECODER_PATH, env_id, memory_mode, user_onnx
    )
    text_encoder = load_model(
        WEIGHT_TEXT_ENCODER_PATH, MODEL_TEXT_ENCODER_PATH, env_id, memory_mode, user_onnx
    )

    if args.disable_ailia_tokenizer:
        tokenizer = transformers.CLIPTokenizer.from_pretrained("./tokenizer")
    else:
        from ailia_tokenizer import CLIPTokenizer
        tokenizer = CLIPTokenizer.from_pretrained()
        tokenizer.model_max_length = 77
        tokenizer.add_special_tokens({'pad_token': '!'})

    feature_extractor = transformers.CLIPImageProcessor.from_pretrained("./feature_extractor")

    scheduler = DDIMScheduler.from_config({
        "prediction_type": "epsilon",
        "clip_sample": False,
        "beta_schedule": "scaled_linear",
        "beta_start": 0.00085,
        "set_alpha_to_one": False,
        "trained_betas": None,
        "steps_offset": 1,
        "beta_end": 0.012,
        "num_train_timesteps": 1000,
        "clip_sample_range": 1.0,
        "dynamic_thresholding_ratio": 0.995,
        "thresholding": False,
        "timestep_spacing": "leading",
    })

    # text-conditioned
    if input_image is None:
        check_and_download_models(WEIGHT_UNET_TEXT_PATH, MODEL_UNET_TEXT_PATH, REMOTE_PATH)
        unet_text = load_model(
            WEIGHT_UNET_TEXT_PATH, MODEL_UNET_TEXT_PATH, env_id, memory_mode, user_onnx
        )
        pipe_text = MVDreamPipeline(
            unet=unet_text,
            vae_encoder=None,
            vae_decoder=vae_decoder,
            text_encoder=text_encoder,
            image_encoder=None,
            tokenizer=tokenizer,
            feature_extractor=None,
            scheduler=scheduler,
            use_onnx=args.onnx,
        )

        logger.info("Generating multi-view images...")
        mv_image_uint8 = pipe_text(
            image=None,
            prompt=prompt,
            negative_prompt=prompt_neg,
            elevation=input_elevation,
            num_inference_steps=input_num_steps,
            guidance_scale=7.5
        )
        mv_image_uint8 = (mv_image_uint8 * 255).astype(np.uint8)
        mv_image = []
        for i in range(4):
            image = process_mv_image(mv_image_uint8[i], bg_remover)
            mv_image.append(image)

    # image-conditioned (may also input text, but no text usually works too)
    else:
        check_and_download_models(WEIGHT_UNET_IMAGE_PATH, MODEL_UNET_IMAGE_PATH, REMOTE_PATH)
        check_and_download_models(WEIGHT_VAE_ENCODER_PATH, MODEL_VAE_ENCODER_PATH, REMOTE_PATH)
        check_and_download_models(WEIGHT_IMAGE_ENCODER_PATH, MODEL_IMAGE_ENCODER_PATH, REMOTE_PATH)
        unet_image = load_model(
            WEIGHT_UNET_IMAGE_PATH, MODEL_UNET_IMAGE_PATH, env_id, memory_mode, user_onnx
        )
        vae_encoder = load_model(
            WEIGHT_VAE_ENCODER_PATH, MODEL_VAE_ENCODER_PATH, env_id, memory_mode, user_onnx
        )
        image_encoder = load_model(
            WEIGHT_IMAGE_ENCODER_PATH, MODEL_IMAGE_ENCODER_PATH, env_id, memory_mode, user_onnx
        )
        pipe_image = MVDreamPipeline(
            unet=unet_image,
            vae_encoder=vae_encoder,
            vae_decoder=vae_decoder,
            text_encoder=text_encoder,
            image_encoder=image_encoder,
            tokenizer=tokenizer,
            feature_extractor=feature_extractor,
            scheduler=scheduler,
            use_onnx=args.onnx,
        )

        logger.info("Generating multi-view images...")
        image = process_mv_image(input_image, bg_remover)
        mv_image = pipe_image(
            image=image,
            prompt=prompt,
            negative_prompt=prompt_neg,
            elevation=input_elevation,
            num_inference_steps=input_num_steps,
            guidance_scale=5.0,
        )
        
    mv_image = np.stack([mv_image[1], mv_image[2], mv_image[3], mv_image[0]], axis=0) # [4, 256, 256, 3]
    return mv_image

def generate_gaussians(mv_image):
    check_and_download_models(WEIGHT_LGM_PATH, MODEL_LGM_PATH, REMOTE_PATH)
    lgm = load_model(WEIGHT_LGM_PATH, MODEL_LGM_PATH, args.env_id, use_onnx=args.onnx)

    if args.onnx:
        gaussians = lgm.run(None, {lgm.get_inputs()[0].name: mv_image})[0]
    else:
        gaussians = lgm.predict(mv_image)

    return gaussians

def render_video(gaussians, opt):
    gs = GaussianRenderer(opt)

    tan_half_fov = np.tan(0.5 * np.deg2rad(opt.fovy))
    proj_matrix = np.zeros((4, 4), dtype=np.float32)
    proj_matrix[0, 0] = 1 / tan_half_fov
    proj_matrix[1, 1] = 1 / tan_half_fov
    proj_matrix[2, 2] = (opt.zfar + opt.znear) / (opt.zfar - opt.znear)
    proj_matrix[3, 2] = - (opt.zfar * opt.znear) / (opt.zfar - opt.znear)
    proj_matrix[2, 3] = 1

    images = []
    elevation = 0
    
    azimuth = np.arange(0, 360, 6, dtype=np.int32)

    logger.info("Rendering video...")
    for azi in tqdm.tqdm(azimuth):
        cam_poses = np.expand_dims(orbit_camera(elevation, azi, radius=opt.cam_radius, opengl=True), axis=0)
        cam_poses[:, :3, 1:3] *= -1 # invert up & forward direction

        cam_view = np.linalg.inv(cam_poses).transpose(0, 2, 1) # [V, 4, 4]
        cam_view_proj = np.matmul(cam_view, proj_matrix) # [V, 4, 4]
        cam_pos = - cam_poses[:, :3, 3] # [V, 3]

        image = gs.render(
            gaussians, 
            np.expand_dims(cam_view, axis=0), 
            np.expand_dims(cam_view_proj, axis=0), 
            np.expand_dims(cam_pos, axis=0), 
            scale_modifier=1
        )
        images.append((image.squeeze(1).permute(0,2,3,1).contiguous().float().numpy() * 255).astype(np.uint8))

    images = np.concatenate(images, axis=0)
    return images

# ======================
# Main functions
# ======================
def main():
    np.random.seed(args.seed)

    # load input
    if args.prompt is not None:
        prompt = args.prompt
        input_image = None
    else:
        prompt = ""
        input_image = load_image(args.input[0])

    # generate multi-view images
    multi_view_image = generate_multi_view_images(
        input_image=input_image,
        prompt=prompt,
        prompt_neg="",
        input_elevation=0,
        input_num_steps=30
    )

    # generate gaussians
    processed_mv_image = preprocess_lgm_model(multi_view_image, opt)
    gaussians = generate_gaussians(processed_mv_image)

    # save gaussians as ply file
    if args.save_ply:
        save_ply_path = get_savepath(args.savepath, "", ext=".ply")
        save_ply(gaussians, save_ply_path)

    # render video
    images = render_video(gaussians, opt)

    # save result
    save_path = get_savepath(args.savepath, "", ext=".gif")
    imageio.mimsave(save_path, images, fps=30, loop=0)

    logger.info("Script finished successfully.")

if __name__ == '__main__':
    main()
