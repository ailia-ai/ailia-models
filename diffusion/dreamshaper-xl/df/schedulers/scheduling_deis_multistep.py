# Copyright 2024 FLAIR Lab and The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import List, Optional, Union

import numpy as np

from ..configuration_utils import ConfigMixin, register_to_config


class DEISMultistepScheduler(ConfigMixin):
    """Diffusion Exponential Integrator Sampler (log-rho multistep DEIS)."""

    order = 1

    @register_to_config
    def __init__(
        self,
        num_train_timesteps: int = 1000,
        beta_start: float = 0.0001,
        beta_end: float = 0.02,
        beta_schedule: str = "linear",
        trained_betas: Optional[Union[np.ndarray, List[float]]] = None,
        solver_order: int = 2,
        prediction_type: str = "epsilon",
        algorithm_type: str = "deis",
        solver_type: str = "logrho",
        lower_order_final: bool = True,
        timestep_spacing: str = "linspace",
        steps_offset: int = 0,
    ):
        if trained_betas is not None:
            self.betas = np.array(trained_betas, dtype=np.float32)
        elif beta_schedule == "linear":
            self.betas = np.linspace(
                beta_start, beta_end, num_train_timesteps, dtype=np.float32
            )
        elif beta_schedule == "scaled_linear":
            # this schedule is very specific to the latent diffusion model.
            self.betas = (
                np.linspace(
                    beta_start**0.5,
                    beta_end**0.5,
                    num_train_timesteps,
                    dtype=np.float32,
                )
                ** 2
            )
        else:
            raise NotImplementedError(
                f"{beta_schedule} is not implemented for {self.__class__}"
            )

        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = np.cumprod(self.alphas, axis=0)
        self.sigmas = ((1 - self.alphas_cumprod) / self.alphas_cumprod) ** 0.5

        if algorithm_type != "deis":
            raise NotImplementedError(
                f"{algorithm_type} is not implemented for {self.__class__}"
            )
        if solver_type != "logrho":
            raise NotImplementedError(
                f"solver type {solver_type} is not implemented for {self.__class__}"
            )

        # standard deviation of the initial noise distribution
        self.init_noise_sigma = 1.0

        # setable values
        self.num_inference_steps = None
        self.timesteps = np.linspace(
            0, num_train_timesteps - 1, num_train_timesteps, dtype=np.float32
        )[::-1].copy()
        self.model_outputs = [None] * solver_order
        self.lower_order_nums = 0
        self.step_index = None
        self.begin_index = None

    def set_begin_index(self, begin_index: int = 0):
        """
        Sets the begin index for the scheduler. This function should be run from pipeline before the inference.
        """
        self.begin_index = begin_index

    def set_timesteps(self, num_inference_steps: int):
        """
        Sets the discrete timesteps used for the diffusion chain (to be run before inference).

        Args:
            num_inference_steps (`int`):
                The number of diffusion steps used when generating samples with a pre-trained model.
        """
        # "linspace", "leading", "trailing" corresponds to annotation of Table 2. of https://arxiv.org/abs/2305.08891
        if self.config.timestep_spacing == "linspace":
            timesteps = (
                np.linspace(
                    0, self.config.num_train_timesteps - 1, num_inference_steps + 1
                )
                .round()[::-1][:-1]
                .copy()
                .astype(np.int64)
            )
        elif self.config.timestep_spacing == "leading":
            step_ratio = self.config.num_train_timesteps // (num_inference_steps + 1)
            # creates integer timesteps by multiplying by ratio
            # casting to int to avoid issues when num_inference_step is power of 3
            timesteps = (
                (np.arange(0, num_inference_steps + 1) * step_ratio)
                .round()[::-1][:-1]
                .copy()
                .astype(np.int64)
            )
            timesteps += self.config.steps_offset
        elif self.config.timestep_spacing == "trailing":
            step_ratio = self.config.num_train_timesteps / num_inference_steps
            # creates integer timesteps by multiplying by ratio
            # casting to int to avoid issues when num_inference_step is power of 3
            timesteps = (
                np.arange(self.config.num_train_timesteps, 0, -step_ratio)
                .round()
                .copy()
                .astype(np.int64)
            )
            timesteps -= 1
        else:
            raise ValueError(
                f"{self.config.timestep_spacing} is not supported. Please make sure to choose one of 'linspace', 'leading' or 'trailing'."
            )

        sigmas = np.array(((1 - self.alphas_cumprod) / self.alphas_cumprod) ** 0.5)
        sigmas = np.interp(timesteps, np.arange(0, len(sigmas)), sigmas)
        sigma_last = ((1 - self.alphas_cumprod[0]) / self.alphas_cumprod[0]) ** 0.5
        sigmas = np.concatenate([sigmas, [sigma_last]]).astype(np.float32)

        self.sigmas = sigmas
        self.timesteps = timesteps
        self.num_inference_steps = len(timesteps)

        self.model_outputs = [None] * self.config.solver_order
        self.lower_order_nums = 0

        # add an index counter for schedulers that allow duplicated timesteps
        self.step_index = None
        self.begin_index = None

    def sigma_to_alpha_sigma_t(self, sigma):
        alpha_t = 1 / ((sigma**2 + 1) ** 0.5)
        sigma_t = sigma * alpha_t

        return alpha_t, sigma_t

    def convert_model_output(
        self, model_output: np.ndarray, sample: np.ndarray
    ) -> np.ndarray:
        """
        Convert the model output to the corresponding type the DEIS algorithm needs.
        """
        sigma = self.sigmas[self.step_index]
        alpha_t, sigma_t = self.sigma_to_alpha_sigma_t(sigma)
        if self.config.prediction_type == "epsilon":
            x0_pred = (sample - sigma_t * model_output) / alpha_t
        elif self.config.prediction_type == "sample":
            x0_pred = model_output
        elif self.config.prediction_type == "v_prediction":
            x0_pred = alpha_t * sample - sigma_t * model_output
        else:
            raise ValueError(
                f"prediction_type given as {self.config.prediction_type} must be one of `epsilon`, `sample`, or"
                " `v_prediction` for the DEISMultistepScheduler."
            )

        return (sample - alpha_t * x0_pred) / sigma_t

    def deis_first_order_update(
        self, model_output: np.ndarray, sample: np.ndarray
    ) -> np.ndarray:
        """
        One step for the first-order DEIS (equivalent to DDIM).
        """
        sigma_t, sigma_s = (
            self.sigmas[self.step_index + 1],
            self.sigmas[self.step_index],
        )
        alpha_t, sigma_t = self.sigma_to_alpha_sigma_t(sigma_t)
        alpha_s, sigma_s = self.sigma_to_alpha_sigma_t(sigma_s)
        lambda_t = np.log(alpha_t) - np.log(sigma_t)
        lambda_s = np.log(alpha_s) - np.log(sigma_s)

        h = lambda_t - lambda_s
        x_t = (alpha_t / alpha_s) * sample - (
            sigma_t * (np.exp(h) - 1.0)
        ) * model_output

        return x_t

    def multistep_deis_second_order_update(
        self, model_output_list: List[np.ndarray], sample: np.ndarray
    ) -> np.ndarray:
        """
        One step for the second-order multistep DEIS.
        """
        sigma_t, sigma_s0, sigma_s1 = (
            self.sigmas[self.step_index + 1],
            self.sigmas[self.step_index],
            self.sigmas[self.step_index - 1],
        )

        alpha_t, sigma_t = self.sigma_to_alpha_sigma_t(sigma_t)
        alpha_s0, sigma_s0 = self.sigma_to_alpha_sigma_t(sigma_s0)
        alpha_s1, sigma_s1 = self.sigma_to_alpha_sigma_t(sigma_s1)

        m0, m1 = model_output_list[-1], model_output_list[-2]

        rho_t, rho_s0, rho_s1 = (
            sigma_t / alpha_t,
            sigma_s0 / alpha_s0,
            sigma_s1 / alpha_s1,
        )

        def ind_fn(t, b, c):
            # Integrate[(log(t) - log(c)) / (log(b) - log(c)), {t}]
            return t * (-np.log(c) + np.log(t) - 1) / (np.log(b) - np.log(c))

        coef1 = ind_fn(rho_t, rho_s0, rho_s1) - ind_fn(rho_s0, rho_s0, rho_s1)
        coef2 = ind_fn(rho_t, rho_s1, rho_s0) - ind_fn(rho_s0, rho_s1, rho_s0)

        x_t = alpha_t * (sample / alpha_s0 + coef1 * m0 + coef2 * m1)

        return x_t

    def index_for_timestep(self, timestep, schedule_timesteps=None):
        if schedule_timesteps is None:
            schedule_timesteps = self.timesteps

        index_candidates = np.flatnonzero(schedule_timesteps == timestep)

        if len(index_candidates) == 0:
            step_index = len(self.timesteps) - 1
        # The sigma index that is taken for the **very** first `step`
        # is always the second index (or the last index if there is only 1)
        # This way we can ensure we don't accidentally skip a sigma in
        # case we start in the middle of the denoising schedule (e.g. for image-to-image)
        elif len(index_candidates) > 1:
            step_index = index_candidates[1].item()
        else:
            step_index = index_candidates[0].item()

        return step_index

    def init_step_index(self, timestep):
        if self.begin_index is None:
            self.step_index = self.index_for_timestep(timestep)
        else:
            self.step_index = self.begin_index

    def step(
        self, model_output: np.ndarray, timestep: int, sample: np.ndarray
    ) -> np.ndarray:
        """
        Predict the sample from the previous timestep by reversing the SDE. This function propagates the sample with
        the multistep DEIS.

        Args:
            model_output:
                The direct output from learned diffusion model.
            timestep:
                The current discrete timestep in the diffusion chain.
            sample:
                A current instance of a sample created by the diffusion process.
        """
        if self.num_inference_steps is None:
            raise ValueError(
                "Number of inference steps is 'None', you need to run 'set_timesteps' after creating the scheduler"
            )

        if self.step_index is None:
            self.init_step_index(timestep)

        lower_order_final = (
            (self.step_index == len(self.timesteps) - 1)
            and self.config.lower_order_final
            and len(self.timesteps) < 15
        )

        model_output = self.convert_model_output(model_output, sample=sample)
        for i in range(self.config.solver_order - 1):
            self.model_outputs[i] = self.model_outputs[i + 1]
        self.model_outputs[-1] = model_output

        if (
            self.config.solver_order == 1
            or self.lower_order_nums < 1
            or lower_order_final
        ):
            prev_sample = self.deis_first_order_update(model_output, sample=sample)
        elif self.config.solver_order == 2:
            prev_sample = self.multistep_deis_second_order_update(
                self.model_outputs, sample=sample
            )
        else:
            raise NotImplementedError(
                f"solver_order {self.config.solver_order} is not implemented for {self.__class__}"
            )

        if self.lower_order_nums < self.config.solver_order:
            self.lower_order_nums += 1

        # upon completion increase step index by one
        self.step_index += 1

        return prev_sample

    def scale_model_input(self, sample: np.ndarray, timestep: int) -> np.ndarray:
        """
        Ensures interchangeability with schedulers that need to scale the denoising model input depending on the
        current timestep.
        """
        return sample

    def add_noise(
        self,
        original_samples: np.ndarray,
        noise: np.ndarray,
        timesteps: np.ndarray,
    ) -> np.ndarray:
        sigmas = self.sigmas
        schedule_timesteps = self.timesteps

        # begin_index is None when the scheduler is used for training or pipeline does not implement set_begin_index
        if self.begin_index is None:
            step_indices = [
                self.index_for_timestep(t, schedule_timesteps) for t in timesteps
            ]
        elif self.step_index is not None:
            # add_noise is called after first denoising step (for inpainting)
            step_indices = [self.step_index] * timesteps.shape[0]
        else:
            # add noise is called before first denoising step to create initial latent(img2img)
            step_indices = [self.begin_index] * timesteps.shape[0]

        sigma = sigmas[step_indices].flatten()
        while len(sigma.shape) < len(original_samples.shape):
            sigma = np.expand_dims(sigma, axis=-1)

        alpha_t, sigma_t = self.sigma_to_alpha_sigma_t(sigma)
        noisy_samples = alpha_t * original_samples + sigma_t * noise

        return noisy_samples
