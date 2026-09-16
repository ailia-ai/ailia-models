import os
import sys
import time
from functools import partial
import numpy as np
import cv2
import ailia

# import original modules
sys.path.append("../../util")
from arg_utils import get_base_parser, update_parser, get_savepath  # noqa
from model_utils import check_and_download_models, check_and_download_file  # noqa
# logger
from logging import getLogger  # noqa
logger = getLogger(__name__)

#import custom scheduler
import schedulerLCM 
import platform



# ======================
# Parameters
# ======================

# text encoder
WEIGHT_TEXT_ENCODER_PATH = "text_encoder.onnx"
MODEL_TEXT_ENCODER_PATH = "text_encoder.onnx.prototxt"
# unet
WEIGHT_UNET_PATH = "unet.onnx"
MODEL_UNET_PATH = "unet.onnx.prototxt"
DATA_UNET_PATH = "model.onnx_data"
# vae encoder
WEIGHT_VAE_DECODER_PATH = "vae_decoder.onnx"
MODEL_VAE_DECODER_PATH =   "vae_decoder.onnx.prototxt" 

local_clip_tokenizer_path = "tokenizer"
REMOTE_PATH = "https://storage.googleapis.com/ailia-models/latent-consistency-models/"
SAVE_IMAGE_PATH="Output/"




# ======================
# Arguemnt Parser Config
# ======================

parser = get_base_parser(
    "Latent Consistency Model", None, SAVE_IMAGE_PATH
)
parser.add_argument(
    "-i", "--input", metavar="TEXT", type=str,
    default="portrait photo of a girl, photograph, highly detailed face, depth of field, moody light, Tokyo, golden hour, style by Dan Winters, Russell James, Steve McCurry, centered, extremely detailed, Nikon D850, award winning photography",
    help="the prompt to render"
)
parser.add_argument(
    "--num_images_per_prompt",  type=int,
    default= 1,
    help="number of images to generate per prompt"
)
parser.add_argument(
    "--num_inference_steps", type=int, default=4,
    help="The number of diffusion steps used when rendering an image. If used, timesteps must be None",
)
parser.add_argument(
    "--H", metavar="height", type=int, default=768,
    help="image height, in pixel space",
)
parser.add_argument(
    "--W", metavar="width", type=int, default=768,
    help="image width, in pixel space",
)
parser.add_argument(
    "--guidance_scale", type=float, default=8,
    help="Classifier Free Guidance Scale"
         " - how strongly the image should conform to prompt - lower values produce more creative results",
)
parser.add_argument(
    "--seed", type=int, default=1001,
    help="random seed",
)
parser.add_argument(
    "--onnx", action="store_true",
    help="execute onnxruntime version."
)
parser.add_argument(
    "--lcm_origin_steps", type=int, default=50,
    help="execute onnxruntime version."
)
parser.add_argument(
    "--img2img", action="store_true",
    help="execute img2img pipeline version."
)
parser.add_argument(
    '--disable_ailia_tokenizer',
    action='store_true',
    help='disable ailia tokenizer.'
)

args = update_parser(parser, check_input_type=False)
scheduler=schedulerLCM.LCMScheduler(beta_start=0.00085, beta_end=0.0120, beta_schedule="scaled_linear", prediction_type="epsilon")
init_noise_sigma = 1.0
decoder_scaling_factor=0.18215
vae_scale_factor = 8
UNET_in_channels=4



# ======================
# Secondaty Functions
# ======================
def image_post_processing(image):
    x_samples = []  
    image = np.clip((image + 1.0) / 2.0, a_min=0.0, a_max=1.0)
    for x_sample in image:
        x_sample = x_sample.transpose(1, 2, 0)  # CHW -> HWC
        x_sample = (x_sample * 255).round().astype(np.uint8)
        img = x_sample
        img = img[:, :, ::-1]  # RGB -> BGR
        x_samples.append(img)
    return x_samples

def save_image(images, save_path):
    if os.path.isdir(save_path):
        base_count = len(os.listdir(save_path))
        for img in images:
            sample_file = os.path.join(save_path, f"{base_count:04}.png")
            cv2.imwrite(sample_file, img)
            base_count += 1
        print("Images saved at ", save_path)
    else:
        if 1 < len(images):
            splited = os.path.splitext(save_path)
            for i in range(len(images)):
                p = splited[0] + f"{i}" + splited[1]
                cv2.imwrite(p, images[i])
            print("Images saved from ", splited[0] + "0" + splited[1], " to ", splited[0] + f"len(images) - 1" + splited[1])
        else:
            cv2.imwrite(save_path, images[0])
            print("Image saved at ", save_path)
    
    
# ======================
# Main functions
# ======================    

def _encode_prompt(prompt, num_images_per_prompt, tokenizer, text_encoder,  prompt_embeds=None):
    if prompt is not None and isinstance(prompt, str):
        batch_size = 1
    elif prompt is not None and isinstance(prompt, list):
        batch_size = len(prompt)
    else:
        batch_size = prompt_embeds.shape[0]

    if prompt_embeds is None:
        text_inputs = tokenizer(prompt, padding="max_length", max_length=tokenizer.model_max_length, truncation=True, return_tensors = "np")
        text_input_ids = text_inputs.input_ids
        untruncated_ids = tokenizer(prompt, padding="longest", return_tensors="np").input_ids

        if untruncated_ids.shape[-1] >= text_input_ids.shape[-1] and not np.array_equal(text_input_ids, untruncated_ids):
            removed_text = tokenizer.batch_decode(untruncated_ids[:, tokenizer.model_max_length - 1 : -1])
            print(
                "The following part of your input was truncated because CLIP can only handle sequences up to"
                f" {tokenizer.model_max_length} tokens: {removed_text}"
            )

        attention_mask = None

        # onnnx
        
        if not args.onnx:
            prompt_embeds = text_encoder.predict([text_input_ids.astype(np.int32)])
        else:
            
            prompt_embeds=text_encoder.run(None, {"input_ids": text_input_ids.astype(np.int32)})
        
        prompt_embeds = prompt_embeds[0]

    prompt_embeds = prompt_embeds.astype(np.float32)
    bs_embed, seq_len, _ = prompt_embeds.shape
    prompt_embeds = np.repeat(prompt_embeds, num_images_per_prompt, axis=0)
    prompt_embeds = prompt_embeds.reshape(bs_embed * num_images_per_prompt, seq_len, -1)
    return prompt_embeds
    
def prepare_latents(batch_size, num_channels_latents, height, width, dtype, latents=None):
    shape = (batch_size, num_channels_latents, height // vae_scale_factor, width // vae_scale_factor)
    
    if latents is None:
        latents = np.random.randn(*shape).astype(dtype)
        latents = latents * init_noise_sigma
    else:
        latents = latents * init_noise_sigma

    return latents
    
    
def get_w_embedding(w, embedding_dim=512, dtype=np.float32):
    assert len(w.shape) == 1
    w = w * 1000.

    half_dim = embedding_dim // 2
    emb = np.log(10000.) / (half_dim - 1)
    emb = np.exp(np.arange(half_dim, dtype=dtype) * -emb)
    emb = w[:, None] * emb[None, :]
    emb = np.concatenate([np.sin(emb), np.cos(emb)], axis=1)
    
    if embedding_dim % 2 == 1:  # zero pad
        emb = np.pad(emb, ((0, 0), (0, 1)))
    
    assert emb.shape == (w.shape[0], embedding_dim)
    return emb
    

    

def generate_image(
    prompt=args.input,
    height= args.H,
    width=args.W,
    guidance_scale= args.guidance_scale,
    num_images_per_prompt= args.num_images_per_prompt ,
    latents=None,
    num_inference_steps= args.num_inference_steps ,
    lcm_origin_steps=args.lcm_origin_steps,
    prompt_embeds=None,
    output_type="pil",
    models=None
    ):
    
    # 0. Default height and width to unet
    height = height or 96 * vae_scale_factor
    width = width or 96 * vae_scale_factor
  
    # 2. Define call parameters
    if prompt is not None and isinstance(prompt, str):
        batch_size = 1
    elif prompt is not None and isinstance(prompt, list):
        batch_size = len(prompt)
    else:
        batch_size = prompt_embeds.shape[0]
    logger.info("prompt: %s" % prompt)
    logger.info("Start inference...")
    
    
    # 3. Encode input prompt
    prompt_embeds  = _encode_prompt(prompt,num_images_per_prompt, models["tokenizer"], models["text_encoder"])
    logger.info("Prompts are encoded.")
    
    # Set scheduler
    scheduler.set_timesteps(num_inference_steps=num_inference_steps, lcm_origin_steps=lcm_origin_steps)
    timesteps = scheduler.timesteps
    
    # 5. Prepare latents
    num_channels_latents =   UNET_in_channels          #self.unet.config.in_channels
    latents = prepare_latents(batch_size * num_images_per_prompt, num_channels_latents,
        height, width, prompt_embeds.dtype, latents)
    bs = batch_size * num_images_per_prompt
    
    # 6. Get Guidance Scale Embedding
    w = np.full((bs,), guidance_scale, dtype = np.float32)
    w_embedding = get_w_embedding(w, embedding_dim=256)
    
    # 7. LCM MultiStep Sampling Loop:
    unet=models["unet"]
    for i, t in enumerate(timesteps):
        ts = np.full((bs,), t, dtype=np.int64)
        if not args.onnx:
            model_pred = unet.predict([latents.astype(np.float32),  ts, prompt_embeds.astype(np.float32), w_embedding.astype(np.float32)])[0]
        else:
            model_pred = unet.run(None, {"sample": latents.astype(np.float32), "timestep": ts, 
                "encoder_hidden_states": prompt_embeds.astype(np.float32), "timestep_cond": w_embedding.astype(np.float32)})[0]
        latents, denoised = scheduler.step(model_pred, i, t, latents, return_dict=False)
    logger.info("Unet is done.")
    
    if not output_type == "latent":
        denoised= denoised / decoder_scaling_factor
        vae_decoder_model=models["vae_decoder_model"]
        if not args.onnx:
            image = vae_decoder_model.predict([denoised.astype(np.float32)])[0]
        else:
            image =  vae_decoder_model.run(None, {"latent_sample": denoised.astype(np.float32)})[0]    
        image= image_post_processing(image)
        logger.info("VAE Decoder is done.")
    else:
        image = denoised
        has_nsfw_concept = None
    
    #save_image
    save_image(image, args.savepath)
    logger.info("Script finished successfully.")

def main():
    
    check_and_download_models(WEIGHT_TEXT_ENCODER_PATH, MODEL_TEXT_ENCODER_PATH, REMOTE_PATH)
    check_and_download_models(WEIGHT_UNET_PATH, MODEL_UNET_PATH, REMOTE_PATH)
    check_and_download_models(WEIGHT_VAE_DECODER_PATH, MODEL_VAE_DECODER_PATH, REMOTE_PATH)
    check_and_download_file(DATA_UNET_PATH, REMOTE_PATH)

    pf = platform.system()
    if pf == "Darwin":
        logger.info("This model not optimized for macOS GPU currently. So we will use BLAS (env_id = 1).")
        args.env_id = 1
    
    #if args.img2img is True:
    #    check_and_download_models(WEIGHT_VAE_ENCODER_PATH, MODEL_VAE_ENCODER_PATH, REMOTE_PATH)
    
    if args.disable_ailia_tokenizer:
        from transformers import CLIPTokenizer, CLIPImageProcessor
        tokenizer = CLIPTokenizer.from_pretrained(local_clip_tokenizer_path)
    else:
        from ailia_tokenizer import CLIPTokenizer
        tokenizer = CLIPTokenizer.from_pretrained()
        tokenizer.model_max_length = 77

    # initialize
    if not args.onnx:
        
        logger.info("This model requires 10GB or more memory.")
        unet = ailia.Net(
            MODEL_UNET_PATH, WEIGHT_UNET_PATH, env_id=args.env_id, memory_mode=args.memory_mode)
        vae_decoder_model = ailia.Net(
            MODEL_VAE_DECODER_PATH, WEIGHT_VAE_DECODER_PATH, env_id=args.env_id, memory_mode=args.memory_mode)
        text_encoder = ailia.Net \
            (MODEL_TEXT_ENCODER_PATH, WEIGHT_TEXT_ENCODER_PATH, env_id=args.env_id, memory_mode=args.memory_mode)
        """
        if args.img2img is True:
            vae_encoder_model = ailia.Net(
                MODEL_VAE_ENCODER_PATH, WEIGHT_VAE_ENCODER_PATH, env_id=args.env_id, memory_mode=args.memory_mode)
        """
    else:
        import onnxruntime
        
        # Create ONNX Runtime session with GPU as the execution provider
        options = onnxruntime.SessionOptions()
        options.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_ENABLE_ALL
        options.execution_mode = onnxruntime.ExecutionMode.ORT_SEQUENTIAL

        # Specify the execution provider (CUDAExecutionProvider for GPU)
        providers = ["CUDAExecutionProvider"] if "CUDAExecutionProvider" in onnxruntime.get_available_providers() else ["CPUExecutionProvider"]
        unet = onnxruntime.InferenceSession(WEIGHT_UNET_PATH , providers=providers, sess_options=options)
        vae_decoder_model = onnxruntime.InferenceSession(WEIGHT_VAE_DECODER_PATH, providers=providers, sess_options=options)
        text_encoder = onnxruntime.InferenceSession(WEIGHT_TEXT_ENCODER_PATH, providers=providers, sess_options=options)
        #if args.img2img is True:
        #    vae_encoder_model = onnxruntime.InferenceSession(WEIGHT_VAE_ENCODER_PATH, providers=providers, sess_options=options)

    if args.profile and not args.onnx:
        unet.set_profile_mode(True)
        vae_decoder_model.set_profile_mode(True)
        text_encoder.set_profile_mode(True)

    seed = args.seed
    if seed is not None:
        np.random.seed(seed)
        
    models = dict(
        unet= unet,
        vae_decoder_model= vae_decoder_model,
        text_encoder= text_encoder,
        tokenizer=tokenizer,
        )
        
    if args.img2img is True: 
        models[vae_encoder_model] = vae_encoder_model
        
    if args.benchmark:
        logger.info("BENCHMARK mode")
        total_time_estimation=0
        for i in range(5):
            start = int(round(time.time() * 1000))
            generate_image(models=models, num_inference_steps = 1)
            end = int(round(time.time() * 1000))
            logger.info("ailia processing time "+str(end - start)+" ms")
            if i != 0:
                total_time_estimation = total_time_estimation + end-start
        logger.info("average time estimation "+ str(total_time_estimation/4)+ " ms")
    else:
        
        generate_image(models=models)
    
    if args.profile and not args.onnx:
        print(text_encoder.get_summary())
        print(vae_decoder_model.get_summary())
        print(unet.get_summary())


if __name__ == "__main__":
    main()  
