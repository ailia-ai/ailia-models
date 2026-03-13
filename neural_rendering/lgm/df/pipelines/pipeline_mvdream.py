import numpy as np
from typing import Callable, List, Optional, Union
from lgm_utils.kiui import orbit_camera

from df.schedulers.scheduling_ddim import DDIMScheduler

from logging import getLogger  # noqa
logger = getLogger(__name__)

def get_camera(
    num_frames, elevation=0, azimuth_start=0, azimuth_span=360, blender_coord=True, extra_view=False,
) -> np.ndarray:
    angle_gap = azimuth_span / num_frames
    cameras = []
    for azimuth in np.arange(azimuth_start, azimuth_span + azimuth_start, angle_gap):
        
        pose = orbit_camera(elevation, azimuth, radius=1) # [4, 4]

        # opengl to blender
        if blender_coord:
            pose[2] *= -1
            pose[[1, 2]] = pose[[2, 1]]

        cameras.append(pose.flatten())

    if extra_view:
        cameras.append(np.zeros_like(cameras[0]))

    return np.stack(cameras, axis=0).astype(np.float16)

class MVDreamPipeline():
    def __init__(
        self,
        unet,
        vae_encoder,
        vae_decoder,
        text_encoder,
        image_encoder,
        tokenizer,
        feature_extractor,
        scheduler: DDIMScheduler,
        use_onnx: bool = False,
    ):
        self.unet = unet
        self.vae_encoder = vae_encoder
        self.vae_decoder = vae_decoder
        self.text_encoder = text_encoder
        self.image_encoder = image_encoder
        self.tokenizer = tokenizer
        self.feature_extractor = feature_extractor
        self.scheduler = scheduler
        self.use_onnx = use_onnx
        self.vae_scale_factor = 8

    def _encode_prompt(
        self,
        prompt: Union[str, List[str]],
        num_images_per_prompt: int,
        do_classifier_free_guidance: bool,
        negative_prompt: Optional[Union[str, List[str]]] = None,
    ):
        """
        Encodes the prompt into text encoder hidden states.

        Args:
            prompt (`str` or `List[str]`, *optional*):
                prompt to be encoded
            num_images_per_prompt (`int`):
                number of images that should be generated per prompt
            do_classifier_free_guidance (`bool`):
                whether to use classifier free guidance or not
            negative_prompt (`str` or `List[str]`, *optional*):
                The prompt or prompts not to guide the image generation. If not defined, one has to pass
                `negative_prompt_embeds`. instead. If not defined, one has to pass `negative_prompt_embeds`. instead.
                Ignored when not using guidance (i.e., ignored if `guidance_scale` is less than `1`).
            prompt_embeds (`np.ndarray`, *optional*):
                Pre-generated text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt weighting. If not
                provided, text embeddings will be generated from `prompt` input argument.
            negative_prompt_embeds (`np.ndarray`, *optional*):
                Pre-generated negative text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt
                weighting. If not provided, negative_prompt_embeds will be generated from `negative_prompt` input
                argument.
        """
        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            raise ValueError(
                f"`prompt` should be either a string or a list of strings, but got {type(prompt)}."
            )

        text_inputs = self.tokenizer(
            prompt,
            padding="max_length",
            max_length=self.tokenizer.model_max_length,
            truncation=True,
            return_tensors="np",
        )
        text_input_ids = text_inputs.input_ids
        untruncated_ids = self.tokenizer(
            prompt, padding="longest", return_tensors="np"
        ).input_ids

        if untruncated_ids.shape[-1] >= text_input_ids.shape[-1] and not np.array_equal(
            text_input_ids, untruncated_ids
        ):
            removed_text = self.tokenizer.batch_decode(
                untruncated_ids[:, self.tokenizer.model_max_length - 1 : -1]
            )
            logger.warning(
                "The following part of your input was truncated because CLIP can only handle sequences up to"
                f" {self.tokenizer.model_max_length} tokens: {removed_text}"
            )

        if self.use_onnx:
            prompt_embeds = self.text_encoder.run(None, {"input": text_input_ids})[0]
        else:
            prompt_embeds = self.text_encoder.predict({"input": text_input_ids})[0]

        prompt_embeds = prompt_embeds.astype(np.float16)
        bs_embed, seq_len, _ = prompt_embeds.shape
        prompt_embeds = np.repeat(prompt_embeds, num_images_per_prompt, axis=0)
        prompt_embeds = prompt_embeds.reshape(bs_embed * num_images_per_prompt, seq_len, -1)

        # get unconditional embeddings for classifier free guidance
        if do_classifier_free_guidance:
            uncond_tokens: List[str]
            if negative_prompt is None:
                uncond_tokens = [""] * batch_size
            elif type(prompt) is not type(negative_prompt):
                raise TypeError(
                    f"`negative_prompt` should be the same type to `prompt`, but got {type(negative_prompt)} !="
                    f" {type(prompt)}."
                )
            elif isinstance(negative_prompt, str):
                uncond_tokens = [negative_prompt]
            elif batch_size != len(negative_prompt):
                raise ValueError(
                    f"`negative_prompt`: {negative_prompt} has batch size {len(negative_prompt)}, but `prompt`:"
                    f" {prompt} has batch size {batch_size}. Please make sure that passed `negative_prompt` matches"
                    " the batch size of `prompt`."
                )
            else:
                uncond_tokens = negative_prompt

            max_length = prompt_embeds.shape[1]
            uncond_input = self.tokenizer(
                uncond_tokens,
                padding="max_length",
                max_length=max_length,
                truncation=True,
                return_tensors="np",
            )

            if self.use_onnx:
                negative_prompt_embeds = self.text_encoder.run(None, {"input": uncond_input.input_ids})[0]
            else:
                negative_prompt_embeds = self.text_encoder.predict({"input": uncond_input.input_ids})[0]

            negative_prompt_embeds = negative_prompt_embeds.astype(np.float16)
            seq_len = negative_prompt_embeds.shape[1]
            negative_prompt_embeds = np.repeat(negative_prompt_embeds, num_images_per_prompt, axis=0)
            negative_prompt_embeds = negative_prompt_embeds.reshape(
                batch_size * num_images_per_prompt, seq_len, -1
            )
            
            prompt_embeds = np.concatenate([negative_prompt_embeds, prompt_embeds], axis=0)

        return prompt_embeds
    
    def decode_latents(self, latents):
        if self.use_onnx:
            image = self.vae_decoder.run(None, {"input": latents})[0]
        else:
            image = self.vae_decoder.predict({"input": latents})[0]
        image = np.clip((image / 2 + 0.5), 0, 1)
        image = image.transpose(0, 2, 3, 1).astype(np.float32)
        return image

    def prepare_latents(
        self,
        batch_size: int,
        num_channels_latents: int,
        height: int,
        width: int,
        dtype: np.dtype,
        latents: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        shape = (
            batch_size,
            num_channels_latents,
            height // self.vae_scale_factor,
            width // self.vae_scale_factor,
        )
        generator = np.random

        if latents is None:
            latents = generator.randn(*shape).astype(dtype)

        # scale the initial noise by the standard deviation required by the scheduler
        latents = latents * np.float64(self.scheduler.init_noise_sigma)
        return latents

    def encode_image(self, image, num_images_per_prompt):
        image = self.feature_extractor(image, return_tensors="np", do_rescale=False).pixel_values.astype(np.float16)
        if self.use_onnx:
            image_embeds = self.image_encoder.run(None, {"input": image,})[0]
        else:
            image_embeds = self.image_encoder.predict({"input": image,})[0]
        image_embeds = np.repeat(image_embeds, num_images_per_prompt, axis=0)
        return np.zeros_like(image_embeds), image_embeds

    def encode_image_latents(self, image, num_images_per_prompt):
        image = np.transpose(np.expand_dims(image, axis=0), (0, 3, 1, 2)).astype(np.float16)
        image = 2 * image - 1
        if self.use_onnx:
            latents = self.vae_encoder.run(None, {"input": image,})[0]
        else:
            latents = self.vae_encoder.predict({"input": image,})[0]
        latents = np.repeat(latents, num_images_per_prompt, axis=0)

        return np.zeros_like(latents), latents
    
    def progress_bar(self, iterable: Optional[iter] = None, total: Optional[float] = None):
        from tqdm.auto import tqdm

        if iterable is not None:
            return tqdm(iterable)
        elif total is not None:
            return tqdm(total=total)

    def __call__(
        self,
        image: Optional[np.ndarray] = None,
        prompt: Union[str, List[str]] = "",
        height: int = 256,
        width: int = 256,
        elevation: float = 0,
        num_inference_steps: int = 50,
        guidance_scale: float = 7.0,
        negative_prompt: str = "",
        num_images_per_prompt: int = 1,
        output_type: Optional[str] = "numpy", # numpy, latents
        callback: Optional[Callable[[int, int, np.ndarray], None]] = None,
        callback_steps: int = 1,
        num_frames: int = 4,
    ):
        do_classifier_free_guidance = guidance_scale > 1.0

        # Prepare timesteps
        self.scheduler.set_timesteps(num_inference_steps)
        timesteps = self.scheduler.timesteps

        # imagedream variant
        if image is not None:
            assert isinstance(image, np.ndarray) and image.dtype == np.float32
            image_embeds_neg, image_embeds_pos = self.encode_image(image, num_images_per_prompt)
            image_latents_neg, image_latents_pos = self.encode_image_latents(image, num_images_per_prompt)

        _prompt_embeds = self._encode_prompt(
            prompt=prompt,
            num_images_per_prompt=num_images_per_prompt,
            do_classifier_free_guidance=do_classifier_free_guidance,
            negative_prompt=negative_prompt,
        )
        prompt_embeds_neg, prompt_embeds_pos = np.split(_prompt_embeds, 2, axis=0)

        # Prepare latent variables
        actual_num_frames = num_frames if image is None else num_frames + 1
        latents = self.prepare_latents(
            actual_num_frames * num_images_per_prompt,
            4,
            height,
            width,
            prompt_embeds_pos.dtype,
            None,
        )

        if image is not None:
            camera = get_camera(num_frames, elevation=elevation, extra_view=True)
        else:
            camera = get_camera(num_frames, elevation=elevation, extra_view=False)
        camera = np.repeat(camera, num_images_per_prompt, axis=0)

        # Denoising loop
        with self.progress_bar(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps):
                # expand the latents if we are doing classifier free guidance
                multiplier = 2 if do_classifier_free_guidance else 1
                latent_model_input = np.concatenate([latents] * multiplier)

                unet_inputs = {
                    'input': latent_model_input,
                    'timesteps': np.array([t] * actual_num_frames * multiplier, dtype=np.int64),
                    'context': np.concatenate([np.tile(prompt_embeds_neg, (actual_num_frames, 1, 1)), 
                               np.tile(prompt_embeds_pos, (actual_num_frames, 1, 1))], axis=0),
                    'num_frames': np.array(actual_num_frames, dtype=np.int64),
                    'camera': np.tile(camera, (multiplier, 1)),
                }

                if image is not None:
                    unet_inputs['ip'] = np.concatenate([image_embeds_neg] * actual_num_frames + [image_embeds_pos] * actual_num_frames)
                    unet_inputs['ip_img'] = np.concatenate([image_latents_neg] + [image_latents_pos]) # no repeat

                if self.use_onnx:
                    noise_pred = self.unet.run(None, unet_inputs)[0]
                else:
                    noise_pred = self.unet.predict(unet_inputs)[0]
                
                # perform guidance
                if do_classifier_free_guidance:
                    noise_pred_uncond, noise_pred_text = np.split(noise_pred, 2)
                    noise_pred = noise_pred_uncond + guidance_scale * (
                        noise_pred_text - noise_pred_uncond
                    )

                # compute the previous noisy sample x_t -> x_t-1
                latents = self.scheduler.step(
                    noise_pred, t, latents
                )

                progress_bar.update()
                if callback is not None and i % callback_steps == 0:
                    callback(i, t, latents)  # type: ignore

        # Post-processing
        if output_type == "latent":
            image = latents
        else: # numpy
            image = self.decode_latents(latents)

        # Offload last model to CPU
        if hasattr(self, "final_offload_hook") and self.final_offload_hook is not None:
            self.final_offload_hook.offload()

        return image