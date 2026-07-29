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
        raise NotImplementedError

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

    def step(
        self, model_output: np.ndarray, timestep: int, sample: np.ndarray
    ) -> np.ndarray:
        raise NotImplementedError

    def scale_model_input(self, sample: np.ndarray, timestep: int) -> np.ndarray:
        """
        Ensures interchangeability with schedulers that need to scale the denoising model input depending on the
        current timestep.
        """
        return sample
