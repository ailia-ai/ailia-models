# Copyright 2024 Zhejiang University Team and The HuggingFace Team. All rights reserved.
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


class PNDMScheduler(ConfigMixin):
    """Pseudo numerical methods for diffusion models (PNDM).

    Combines the Runge-Kutta method and a linear multi-step method. See
    https://arxiv.org/abs/2202.09778 for the details.
    """

    order = 1

    @register_to_config
    def __init__(
        self,
        num_train_timesteps: int = 1000,
        beta_start: float = 0.0001,
        beta_end: float = 0.02,
        beta_schedule: str = "linear",
        trained_betas: Optional[Union[np.ndarray, List[float]]] = None,
        skip_prk_steps: bool = False,
        set_alpha_to_one: bool = False,
        prediction_type: str = "epsilon",
        timestep_spacing: str = "leading",
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

        self.final_alpha_cumprod = (
            np.array(1.0) if set_alpha_to_one else self.alphas_cumprod[0]
        )

        # standard deviation of the initial noise distribution
        self.init_noise_sigma = 1.0

        # For now we only support F-PNDM, i.e. the runge-kutta method
        # For more information on the algorithm please take a look at the paper:
        # https://arxiv.org/pdf/2202.09778.pdf mainly at formula (9), (12), (13) and the Algorithm 2.
        self.pndm_order = 4

        # running values
        self.cur_model_output = 0
        self.counter = 0
        self.cur_sample = None
        self.ets = []

        # setable values
        self.num_inference_steps = None
        self._timesteps = np.arange(0, num_train_timesteps)[::-1].copy()
        self.prk_timesteps = None
        self.plms_timesteps = None
        self.timesteps = None

    def set_timesteps(self, num_inference_steps: int):
        """
        Sets the discrete timesteps used for the diffusion chain (to be run before inference).

        Args:
            num_inference_steps (`int`):
                The number of diffusion steps used when generating samples with a pre-trained model.
        """
        self.num_inference_steps = num_inference_steps

        # "linspace", "leading", "trailing" corresponds to annotation of Table 2. of https://arxiv.org/abs/2305.08891
        if self.config.timestep_spacing == "linspace":
            self._timesteps = (
                np.linspace(0, self.config.num_train_timesteps - 1, num_inference_steps)
                .round()
                .astype(np.int64)
            )
        elif self.config.timestep_spacing == "leading":
            step_ratio = self.config.num_train_timesteps // num_inference_steps
            # creates integer timesteps by multiplying by ratio
            # casting to int to avoid issues when num_inference_step is power of 3
            self._timesteps = (np.arange(0, num_inference_steps) * step_ratio).round()
            self._timesteps += self.config.steps_offset
        elif self.config.timestep_spacing == "trailing":
            step_ratio = self.config.num_train_timesteps / num_inference_steps
            self._timesteps = np.round(
                np.arange(self.config.num_train_timesteps, 0, -step_ratio)
            )[::-1].astype(np.int64)
            self._timesteps -= 1
        else:
            raise ValueError(
                f"{self.config.timestep_spacing} is not supported. Please make sure to choose one of 'linspace', 'leading' or 'trailing'."
            )

        if self.config.skip_prk_steps:
            # for some models like stable diffusion the prk steps can/should be skipped to produce better
            # results. The schedule then runs in pure PLMS mode, with the second to last timestep
            # repeated so that the multi-step warm up still converges.
            self.prk_timesteps = np.array([])
            self.plms_timesteps = np.concatenate(
                [self._timesteps[:-1], self._timesteps[-2:-1], self._timesteps[-1:]]
            )[::-1].copy()
        else:
            prk_timesteps = np.array(self._timesteps[-self.pndm_order :]).repeat(
                2
            ) + np.tile(
                np.array(
                    [0, self.config.num_train_timesteps // num_inference_steps // 2]
                ),
                self.pndm_order,
            )
            self.prk_timesteps = (prk_timesteps[:-1].repeat(2)[1:-1])[::-1].copy()
            self.plms_timesteps = self._timesteps[:-3][::-1].copy()

        self.timesteps = np.concatenate(
            [self.prk_timesteps, self.plms_timesteps]
        ).astype(np.int64)

        self.ets = []
        self.counter = 0
        self.cur_model_output = 0

    def step(
        self, model_output: np.ndarray, timestep: int, sample: np.ndarray
    ) -> np.ndarray:
        """
        Predict the sample from the previous timestep by reversing the SDE. This function calls `step_prk` or
        `step_plms` depending on the internal variable `counter`.

        Args:
            model_output:
                The direct output from learned diffusion model.
            timestep:
                The current discrete timestep in the diffusion chain.
            sample:
                A current instance of a sample created by the diffusion process.
        """
        if self.counter < len(self.prk_timesteps) and not self.config.skip_prk_steps:
            return self.step_prk(model_output, timestep, sample)
        else:
            return self.step_plms(model_output, timestep, sample)

    def step_prk(
        self, model_output: np.ndarray, timestep: int, sample: np.ndarray
    ) -> np.ndarray:
        """
        Propagate the sample with the Runge-Kutta method. It performs four forward passes to approximate the
        solution to the differential equation.
        """
        if self.num_inference_steps is None:
            raise ValueError(
                "Number of inference steps is 'None', you need to run 'set_timesteps' after creating the scheduler"
            )

        diff_to_prev = (
            0
            if self.counter % 2
            else self.config.num_train_timesteps // self.num_inference_steps // 2
        )
        prev_timestep = timestep - diff_to_prev
        timestep = self.prk_timesteps[self.counter // 4 * 4]

        if self.counter % 4 == 0:
            self.cur_model_output += 1 / 6 * model_output
            self.ets.append(model_output)
            self.cur_sample = sample
        elif (self.counter - 1) % 4 == 0:
            self.cur_model_output += 1 / 3 * model_output
        elif (self.counter - 2) % 4 == 0:
            self.cur_model_output += 1 / 3 * model_output
        elif (self.counter - 3) % 4 == 0:
            model_output = self.cur_model_output + 1 / 6 * model_output
            self.cur_model_output = 0

        # cur_sample should not be `None`
        cur_sample = self.cur_sample if self.cur_sample is not None else sample

        prev_sample = self.get_prev_sample(
            cur_sample, timestep, prev_timestep, model_output
        )
        self.counter += 1

        return prev_sample

    def step_plms(
        self, model_output: np.ndarray, timestep: int, sample: np.ndarray
    ) -> np.ndarray:
        """
        Propagate the sample with the linear multi-step method. It performs one forward pass multiple times to
        approximate the solution.
        """
        if self.num_inference_steps is None:
            raise ValueError(
                "Number of inference steps is 'None', you need to run 'set_timesteps' after creating the scheduler"
            )

        if not self.config.skip_prk_steps and len(self.ets) < 3:
            raise ValueError(
                f"{self.__class__} can only be run AFTER scheduler has been run in 'prk' mode for at least 12 iterations"
            )

        prev_timestep = (
            timestep - self.config.num_train_timesteps // self.num_inference_steps
        )

        if self.counter != 1:
            self.ets = self.ets[-3:]
            self.ets.append(model_output)
        else:
            prev_timestep = timestep
            timestep = (
                timestep + self.config.num_train_timesteps // self.num_inference_steps
            )

        if len(self.ets) == 1 and self.counter == 0:
            # the first step keeps the model output as is and remembers the sample it started from
            self.cur_sample = sample
        elif len(self.ets) == 1 and self.counter == 1:
            model_output = (model_output + self.ets[-1]) / 2
            sample = self.cur_sample
            self.cur_sample = None
        elif len(self.ets) == 2:
            model_output = (3 * self.ets[-1] - self.ets[-2]) / 2
        elif len(self.ets) == 3:
            model_output = (
                23 * self.ets[-1] - 16 * self.ets[-2] + 5 * self.ets[-3]
            ) / 12
        else:
            model_output = (1 / 24) * (
                55 * self.ets[-1]
                - 59 * self.ets[-2]
                + 37 * self.ets[-3]
                - 9 * self.ets[-4]
            )

        prev_sample = self.get_prev_sample(
            sample, timestep, prev_timestep, model_output
        )
        self.counter += 1

        return prev_sample

    def get_prev_sample(self, sample, timestep, prev_timestep, model_output):
        # See formula (9) of PNDM paper https://arxiv.org/pdf/2202.09778.pdf
        # this function computes x_(t−δ) using the formula of (9)
        # Note that x_t needs to be added to both sides of the equation
        #
        # Notation (<variable name> -> <name in paper>
        # alpha_prod_t -> α_t
        # alpha_prod_t_prev -> α_(t−δ)
        # beta_prod_t -> (1 - α_t)
        # beta_prod_t_prev -> (1 - α_(t−δ))
        # sample -> x_t
        # model_output -> e_θ(x_t, t)
        # prev_sample -> x_(t−δ)
        alpha_prod_t = self.alphas_cumprod[timestep]
        alpha_prod_t_prev = (
            self.alphas_cumprod[prev_timestep]
            if prev_timestep >= 0
            else self.final_alpha_cumprod
        )
        beta_prod_t = 1 - alpha_prod_t
        beta_prod_t_prev = 1 - alpha_prod_t_prev

        if self.config.prediction_type == "v_prediction":
            model_output = (alpha_prod_t**0.5) * model_output + (
                beta_prod_t**0.5
            ) * sample
        elif self.config.prediction_type != "epsilon":
            raise ValueError(
                f"prediction_type given as {self.config.prediction_type} must be one of `epsilon` or `v_prediction`"
            )

        # corresponds to (α_(t−δ) - α_t) divided by
        # denominator of x_t in formula (9) and plus 1
        # Note: (α_(t−δ) - α_t) / (sqrt(α_t) * (sqrt(α_(t−δ)) + sqr(α_t))) =
        # sqrt(α_(t−δ)) / sqrt(α_t))
        sample_coeff = (alpha_prod_t_prev / alpha_prod_t) ** (0.5)

        # corresponds to denominator of e_θ(x_t, t) in formula (9)
        model_output_denom_coeff = alpha_prod_t * beta_prod_t_prev ** (0.5) + (
            alpha_prod_t * beta_prod_t * alpha_prod_t_prev
        ) ** (0.5)

        # full formula (9)
        prev_sample = (
            sample_coeff * sample
            - (alpha_prod_t_prev - alpha_prod_t)
            * model_output
            / model_output_denom_coeff
        )

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
        sqrt_alpha_prod = self.alphas_cumprod[timesteps] ** 0.5
        sqrt_one_minus_alpha_prod = (1 - self.alphas_cumprod[timesteps]) ** 0.5
        while len(sqrt_alpha_prod.shape) < len(original_samples.shape):
            sqrt_alpha_prod = np.expand_dims(sqrt_alpha_prod, axis=-1)
            sqrt_one_minus_alpha_prod = np.expand_dims(
                sqrt_one_minus_alpha_prod, axis=-1
            )

        return sqrt_alpha_prod * original_samples + sqrt_one_minus_alpha_prod * noise
