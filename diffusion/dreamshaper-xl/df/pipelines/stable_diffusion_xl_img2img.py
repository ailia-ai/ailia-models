# Copyright 2024 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import List, Optional, Tuple, Union

import numpy as np
from PIL import Image

from .stable_diffusion_xl import SIZE_ALIGNMENT, StableDiffusionXL


class StableDiffusionXLImg2Img(StableDiffusionXL):
    def __init__(
        self,
        vae_encoder,
        vae_decoder,
        text_encoder,
        text_encoder_2,
        tokenizer,
        tokenizer_2,
        unet,
        scheduler,
        use_onnx: bool = False,
    ):
        super().__init__(
            vae_decoder,
            text_encoder,
            text_encoder_2,
            tokenizer,
            tokenizer_2,
            unet,
            scheduler,
            use_onnx,
        )
        self.vae_encoder = vae_encoder

    def preprocess_image(self, image: Image.Image):
        # the resolution is taken from the input image, truncated to a multiple of SIZE_ALIGNMENT
        width, height = (x - x % SIZE_ALIGNMENT for x in image.size)
        image = image.resize((width, height), resample=Image.Resampling.LANCZOS)

        image = np.array(image).astype(np.float32) / 255.0
        image = image.transpose(2, 0, 1)[None]

        return 2.0 * image - 1.0

    def get_timesteps(self, num_inference_steps, strength):
        # get the original timestep using init_timestep
        init_timestep = min(int(num_inference_steps * strength), num_inference_steps)
        t_start = max(num_inference_steps - init_timestep, 0)

        self.scheduler.set_begin_index(t_start * self.scheduler.order)

        return self.scheduler.timesteps[t_start * self.scheduler.order :]

    def prepare_latents(self, image, timestep, batch_size):
        # the scaling by scaling_factor is part of the exported encoder
        mean, std = self.run_net(self.vae_encoder, [image], ["pixel"])
        init_latents = mean + std * np.random.randn(*mean.shape).astype(np.float32)

        if batch_size % init_latents.shape[0] != 0:
            raise ValueError(
                f"Cannot duplicate `image` of batch size {init_latents.shape[0]} to {batch_size} text prompts."
            )
        init_latents = np.concatenate(
            [init_latents] * (batch_size // init_latents.shape[0]), axis=0
        )

        # add noise to latents using the timesteps
        noise = np.random.randn(*init_latents.shape).astype(np.float32)

        return self.scheduler.add_noise(init_latents, noise, timestep)

    def forward(
        self,
        prompt: Union[str, List[str]],
        image: Image.Image,
        strength: float = 0.75,
        num_inference_steps: int = 25,
        guidance_scale: float = 7.0,
        negative_prompt: Optional[Union[str, List[str]]] = None,
        num_images_per_prompt: int = 1,
        original_size: Optional[Tuple[int, int]] = None,
        crops_coords_top_left: Tuple[int, int] = (0, 0),
        target_size: Optional[Tuple[int, int]] = None,
    ):
        batch_size = 1 if isinstance(prompt, str) else len(prompt)

        do_classifier_free_guidance = guidance_scale > 1.0

        # Encode input prompt
        (
            prompt_embeds,
            negative_prompt_embeds,
            pooled_prompt_embeds,
            negative_pooled_prompt_embeds,
        ) = self.encode_prompt(
            prompt,
            num_images_per_prompt,
            do_classifier_free_guidance,
            negative_prompt,
        )

        # Preprocess image
        image = self.preprocess_image(image)

        # Prepare timesteps
        self.scheduler.set_timesteps(num_inference_steps)
        timesteps = self.get_timesteps(num_inference_steps, strength)
        latent_timestep = np.repeat(
            timesteps[:1], batch_size * num_images_per_prompt, axis=0
        )

        # Prepare latent variables
        latents = self.prepare_latents(
            image, latent_timestep, batch_size * num_images_per_prompt
        )

        height = latents.shape[2] * self.vae_scale_factor
        width = latents.shape[3] * self.vae_scale_factor
        original_size = original_size or (height, width)
        target_size = target_size or (height, width)

        # Prepare added time ids & embeddings
        add_text_embeds = pooled_prompt_embeds
        add_time_ids = self.get_add_time_ids(
            original_size, crops_coords_top_left, target_size
        )
        add_time_ids = np.repeat(
            add_time_ids, batch_size * num_images_per_prompt, axis=0
        )

        if do_classifier_free_guidance:
            prompt_embeds = np.concatenate(
                (negative_prompt_embeds, prompt_embeds), axis=0
            )
            add_text_embeds = np.concatenate(
                (negative_pooled_prompt_embeds, add_text_embeds), axis=0
            )
            add_time_ids = np.concatenate((add_time_ids, add_time_ids), axis=0)

        # Denoising loop
        latents = self.denoise(
            latents,
            timesteps,
            prompt_embeds,
            add_text_embeds,
            add_time_ids,
            guidance_scale,
            do_classifier_free_guidance,
        )

        return self.decode_latents(latents)
