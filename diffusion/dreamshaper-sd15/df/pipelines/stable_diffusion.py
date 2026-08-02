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

from logging import getLogger
from typing import List, Optional, Union

import numpy as np
from tqdm import tqdm

logger = getLogger(__name__)

VAE_SCALE_FACTOR = 8
# the UNet skip connections only line up when the latent height and width are multiples of 8
SIZE_ALIGNMENT = 64


class StableDiffusion:
    def __init__(
        self,
        vae_decoder,
        text_encoder,
        tokenizer,
        unet,
        scheduler,
        use_onnx: bool = False,
    ):
        self.vae_decoder = vae_decoder
        self.text_encoder = text_encoder
        self.tokenizer = tokenizer
        self.unet = unet
        self.scheduler = scheduler
        self.use_onnx = use_onnx

        self.vae_scale_factor = VAE_SCALE_FACTOR

    def run_net(self, net, inputs, input_names):
        if not self.use_onnx:
            return net.run(inputs)

        return net.run(None, dict(zip(input_names, inputs)))

    def encode_text(self, prompt):
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

        text_input_ids = text_input_ids.astype(np.int64)

        (hidden_states,) = self.run_net(
            self.text_encoder, [text_input_ids], ["input_ids"]
        )

        return hidden_states

    def encode_prompt(
        self,
        prompt: Union[str, List[str]],
        num_images_per_prompt: int,
        do_classifier_free_guidance: bool,
        negative_prompt: Optional[Union[str, List[str]]],
    ):
        """
        Encodes the prompt into text encoder hidden states.

        `prompt_embeds` is the last hidden state of the CLIP ViT-L/14 text encoder (768 features). Unlike
        SDXL there is no second encoder and no pooled embedding.
        """
        prompt = [prompt] if isinstance(prompt, str) else prompt
        batch_size = len(prompt)

        prompt_embeds = self.encode_text(prompt)

        negative_prompt_embeds = None
        if do_classifier_free_guidance:
            # the unconditional branch is the encoding of an empty prompt, not a zero vector
            negative_prompt = negative_prompt if negative_prompt is not None else ""
            negative_prompt = (
                batch_size * [negative_prompt]
                if isinstance(negative_prompt, str)
                else negative_prompt
            )
            if batch_size != len(negative_prompt):
                raise ValueError(
                    f"`negative_prompt`: {negative_prompt} has batch size {len(negative_prompt)}, but `prompt`:"
                    f" {prompt} has batch size {batch_size}. Please make sure that passed `negative_prompt` matches"
                    " the batch size of `prompt`."
                )

            negative_prompt_embeds = self.encode_text(negative_prompt)

        prompt_embeds = np.repeat(prompt_embeds, num_images_per_prompt, axis=0)
        if do_classifier_free_guidance:
            negative_prompt_embeds = np.repeat(
                negative_prompt_embeds, num_images_per_prompt, axis=0
            )

        return prompt_embeds, negative_prompt_embeds

    def prepare_latents(self, batch_size, height, width):
        shape = (
            batch_size,
            4,
            height // self.vae_scale_factor,
            width // self.vae_scale_factor,
        )
        latents = np.random.randn(*shape).astype(np.float32)

        # scale the initial noise by the standard deviation required by the scheduler
        return latents * self.scheduler.init_noise_sigma

    def denoise(
        self,
        latents,
        timesteps,
        prompt_embeds,
        guidance_scale,
        do_classifier_free_guidance,
    ):
        for t in tqdm(timesteps):
            # expand the latents if we are doing classifier free guidance
            latent_model_input = (
                np.concatenate([latents] * 2)
                if do_classifier_free_guidance
                else latents
            )
            latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)

            # predict the noise residual
            timestep = np.array([t], dtype=np.float32)
            (noise_pred,) = self.run_net(
                self.unet,
                [latent_model_input, timestep, prompt_embeds],
                ["sample", "timestep", "encoder_hidden_states"],
            )

            # perform guidance
            if do_classifier_free_guidance:
                noise_pred_uncond, noise_pred_text = np.split(noise_pred, 2)
                noise_pred = noise_pred_uncond + guidance_scale * (
                    noise_pred_text - noise_pred_uncond
                )

            # compute the previous noisy sample x_t -> x_t-1
            latents = self.scheduler.step(noise_pred, t, latents)

        return latents

    def decode_latents(self, latents):
        # the scaling by 1 / scaling_factor is part of the exported decoder
        images = [
            self.run_net(self.vae_decoder, [latents[i : i + 1]], ["latent"])[0]
            for i in range(latents.shape[0])
        ]
        image = np.concatenate(images)

        image = np.clip(image / 2 + 0.5, 0, 1)
        image = image.transpose((0, 2, 3, 1))

        return image

    def forward(
        self,
        prompt: Union[str, List[str]],
        height: int = 512,
        width: int = 512,
        num_inference_steps: int = 25,
        guidance_scale: float = 7.5,
        negative_prompt: Optional[Union[str, List[str]]] = None,
        num_images_per_prompt: int = 1,
    ):
        if height % SIZE_ALIGNMENT != 0 or width % SIZE_ALIGNMENT != 0:
            raise ValueError(
                f"`height` and `width` have to be multiples of {SIZE_ALIGNMENT} but are {height} and {width}."
            )

        batch_size = 1 if isinstance(prompt, str) else len(prompt)

        # here `guidance_scale` is defined analog to the guidance weight `w` of equation (2)
        # of the Imagen paper: https://arxiv.org/pdf/2205.11487.pdf . `guidance_scale = 1`
        # corresponds to doing no classifier free guidance.
        do_classifier_free_guidance = guidance_scale > 1.0

        # Encode input prompt
        prompt_embeds, negative_prompt_embeds = self.encode_prompt(
            prompt,
            num_images_per_prompt,
            do_classifier_free_guidance,
            negative_prompt,
        )

        # Prepare timesteps
        self.scheduler.set_timesteps(num_inference_steps)
        timesteps = self.scheduler.timesteps

        # Prepare latent variables
        latents = self.prepare_latents(
            batch_size * num_images_per_prompt, height, width
        )

        if do_classifier_free_guidance:
            prompt_embeds = np.concatenate(
                (negative_prompt_embeds, prompt_embeds), axis=0
            )

        # Denoising loop
        latents = self.denoise(
            latents,
            timesteps,
            prompt_embeds,
            guidance_scale,
            do_classifier_free_guidance,
        )

        return self.decode_latents(latents)
