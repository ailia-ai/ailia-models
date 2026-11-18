# Copyright 2024 TSAIL Team and The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""DPM-Solver++ multistep scheduler (numpy port of diffusers.

`DPMSolverMultistepScheduler`)。DreamShaper 推奨の DPM++ 2M Karras を再現する。
diffusers の実装を numpy に移植したもので、本サンプルで使う設定
(algorithm_type="dpmsolver++", solver_order=2, solver_type="midpoint",
prediction_type="epsilon", use_karras_sigmas=True, final_sigmas_type="zero",
lower_order_final=True) に対応する。SDE 系や thresholding 等は未対応。
"""

from typing import List, Optional, Union

import numpy as np

from ..configuration_utils import ConfigMixin, register_to_config


class DPMSolverMultistepScheduler(ConfigMixin):
    """DPM-Solver++ 2M(Karras) スケジューラ(numpy 版)。"""

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
        algorithm_type: str = "dpmsolver++",
        solver_type: str = "midpoint",
        lower_order_final: bool = True,
        euler_at_final: bool = False,
        use_karras_sigmas: bool = False,
        final_sigmas_type: str = "zero",
        lambda_min_clipped: float = -float("inf"),
        variance_type: Optional[str] = None,
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

        # Currently we only support VP-type noise schedule
        self.alpha_t = np.sqrt(self.alphas_cumprod)
        self.sigma_t = np.sqrt(1 - self.alphas_cumprod)
        self.lambda_t = np.log(self.alpha_t) - np.log(self.sigma_t)
        self.sigmas = ((1 - self.alphas_cumprod) / self.alphas_cumprod) ** 0.5

        # standard deviation of the initial noise distribution
        self.init_noise_sigma = 1.0

        # setable values
        self.num_inference_steps = None
        timesteps = np.linspace(
            0, num_train_timesteps - 1, num_train_timesteps, dtype=np.float32
        )[::-1].copy()
        self.timesteps = timesteps
        self.model_outputs = [None] * solver_order
        self.lower_order_nums = 0
        self._step_index = None
        self._begin_index = None

    @property
    def step_index(self):
        return self._step_index

    @property
    def begin_index(self):
        return self._begin_index

    def set_begin_index(self, begin_index: int = 0):
        self._begin_index = begin_index

    def set_timesteps(self, num_inference_steps: int):
        # Clipping the minimum of all lambda(t) for numerical stability.
        clipped_idx = np.searchsorted(
            np.flip(self.lambda_t), self.config.lambda_min_clipped
        )
        last_timestep = self.config.num_train_timesteps - int(clipped_idx)

        # "linspace", "leading", "trailing" corresponds to Table 2 of arXiv:2305.08891
        if self.config.timestep_spacing == "linspace":
            timesteps = (
                np.linspace(0, last_timestep - 1, num_inference_steps + 1)
                .round()[::-1][:-1]
                .copy()
                .astype(np.int64)
            )
        elif self.config.timestep_spacing == "leading":
            step_ratio = last_timestep // (num_inference_steps + 1)
            timesteps = (
                (np.arange(0, num_inference_steps + 1) * step_ratio)
                .round()[::-1][:-1]
                .copy()
                .astype(np.int64)
            )
            timesteps += self.config.steps_offset
        elif self.config.timestep_spacing == "trailing":
            step_ratio = self.config.num_train_timesteps / num_inference_steps
            timesteps = np.arange(last_timestep, 0, -step_ratio).round().copy()
            timesteps = timesteps.astype(np.int64) - 1
        else:
            raise ValueError(
                f"{self.config.timestep_spacing} is not supported. Choose one of "
                "'linspace', 'leading' or 'trailing'."
            )

        sigmas = np.array(((1 - self.alphas_cumprod) / self.alphas_cumprod) ** 0.5)
        log_sigmas = np.log(sigmas)

        if self.config.use_karras_sigmas:
            sigmas = np.flip(sigmas).copy()
            sigmas = self._convert_to_karras(
                in_sigmas=sigmas, num_inference_steps=num_inference_steps
            )
            timesteps = np.array(
                [self._sigma_to_t(sigma, log_sigmas) for sigma in sigmas]
            ).round()
        else:
            sigmas = np.interp(timesteps, np.arange(0, len(sigmas)), sigmas)

        if self.config.final_sigmas_type == "sigma_min":
            sigma_last = ((1 - self.alphas_cumprod[0]) / self.alphas_cumprod[0]) ** 0.5
        elif self.config.final_sigmas_type == "zero":
            sigma_last = 0
        else:
            raise ValueError(
                "`final_sigmas_type` must be one of 'zero' or 'sigma_min', but got "
                f"{self.config.final_sigmas_type}"
            )

        sigmas = np.concatenate([sigmas, [sigma_last]]).astype(np.float32)

        self.sigmas = sigmas
        self.timesteps = timesteps.astype(np.int64)
        self.num_inference_steps = len(timesteps)

        self.model_outputs = [None] * self.config.solver_order
        self.lower_order_nums = 0
        self._step_index = None
        self._begin_index = None

    def _sigma_to_t(self, sigma, log_sigmas):
        # get log sigma
        log_sigma = np.log(np.maximum(sigma, 1e-10))

        # get distribution
        dists = log_sigma - log_sigmas[:, np.newaxis]

        # get sigmas range
        low_idx = (
            np.cumsum((dists >= 0), axis=0)
            .argmax(axis=0)
            .clip(max=log_sigmas.shape[0] - 2)
        )
        high_idx = low_idx + 1

        low = log_sigmas[low_idx]
        high = log_sigmas[high_idx]

        # interpolate sigmas
        w = (low - log_sigma) / (low - high)
        w = np.clip(w, 0, 1)

        # transform interpolation to time range
        t = (1 - w) * low_idx + w * high_idx
        t = t.reshape(sigma.shape)
        return t

    def _sigma_to_alpha_sigma_t(self, sigma):
        alpha_t = 1 / ((sigma**2 + 1) ** 0.5)
        sigma_t = sigma * alpha_t
        return alpha_t, sigma_t

    def _convert_to_karras(self, in_sigmas, num_inference_steps):
        sigma_min = in_sigmas[-1].item()
        sigma_max = in_sigmas[0].item()

        rho = 7.0  # 7.0 is the value used in the paper
        ramp = np.linspace(0, 1, num_inference_steps)
        min_inv_rho = sigma_min ** (1 / rho)
        max_inv_rho = sigma_max ** (1 / rho)
        sigmas = (max_inv_rho + ramp * (min_inv_rho - max_inv_rho)) ** rho
        return sigmas

    def convert_model_output(self, model_output, sample):
        # DPM-Solver++ needs to solve an integral of the data prediction model.
        if self.config.prediction_type == "epsilon":
            if self.config.variance_type in ["learned", "learned_range"]:
                model_output = model_output[:, :3]
            sigma = self.sigmas[self.step_index]
            alpha_t, sigma_t = self._sigma_to_alpha_sigma_t(sigma)
            x0_pred = (sample - sigma_t * model_output) / alpha_t
        elif self.config.prediction_type == "sample":
            x0_pred = model_output
        elif self.config.prediction_type == "v_prediction":
            sigma = self.sigmas[self.step_index]
            alpha_t, sigma_t = self._sigma_to_alpha_sigma_t(sigma)
            x0_pred = alpha_t * sample - sigma_t * model_output
        else:
            raise ValueError(
                f"prediction_type given as {self.config.prediction_type} must be one "
                "of `epsilon`, `sample`, or `v_prediction` for the "
                "DPMSolverMultistepScheduler."
            )
        return x0_pred

    def dpm_solver_first_order_update(self, model_output, sample):
        sigma_t, sigma_s = (
            self.sigmas[self.step_index + 1],
            self.sigmas[self.step_index],
        )
        alpha_t, sigma_t = self._sigma_to_alpha_sigma_t(sigma_t)
        alpha_s, sigma_s = self._sigma_to_alpha_sigma_t(sigma_s)
        # 最終ステップ(final_sigmas_type="zero")では sigma_t=0 → log(0)=-inf に
        # なるが、後段で exp(-h)=exp(-inf)=0 となり x0_pred へ収束する(想定内)。
        with np.errstate(divide="ignore"):
            lambda_t = np.log(alpha_t) - np.log(sigma_t)
            lambda_s = np.log(alpha_s) - np.log(sigma_s)

        h = lambda_t - lambda_s
        # algorithm_type == "dpmsolver++"
        x_t = (sigma_t / sigma_s) * sample - (
            alpha_t * (np.exp(-h) - 1.0)
        ) * model_output
        return x_t

    def multistep_dpm_solver_second_order_update(self, model_output_list, sample):
        sigma_t, sigma_s0, sigma_s1 = (
            self.sigmas[self.step_index + 1],
            self.sigmas[self.step_index],
            self.sigmas[self.step_index - 1],
        )
        alpha_t, sigma_t = self._sigma_to_alpha_sigma_t(sigma_t)
        alpha_s0, sigma_s0 = self._sigma_to_alpha_sigma_t(sigma_s0)
        alpha_s1, sigma_s1 = self._sigma_to_alpha_sigma_t(sigma_s1)

        lambda_t = np.log(alpha_t) - np.log(sigma_t)
        lambda_s0 = np.log(alpha_s0) - np.log(sigma_s0)
        lambda_s1 = np.log(alpha_s1) - np.log(sigma_s1)

        m0, m1 = model_output_list[-1], model_output_list[-2]

        h, h_0 = lambda_t - lambda_s0, lambda_s0 - lambda_s1
        r0 = h_0 / h
        D0, D1 = m0, (1.0 / r0) * (m0 - m1)
        # algorithm_type == "dpmsolver++"
        if self.config.solver_type == "midpoint":
            x_t = (
                (sigma_t / sigma_s0) * sample
                - (alpha_t * (np.exp(-h) - 1.0)) * D0
                - 0.5 * (alpha_t * (np.exp(-h) - 1.0)) * D1
            )
        elif self.config.solver_type == "heun":
            x_t = (
                (sigma_t / sigma_s0) * sample
                - (alpha_t * (np.exp(-h) - 1.0)) * D0
                + (alpha_t * ((np.exp(-h) - 1.0) / h + 1.0)) * D1
            )
        else:
            raise ValueError(f"solver_type {self.config.solver_type} is not supported.")
        return x_t

    def index_for_timestep(self, timestep, schedule_timesteps=None):
        if schedule_timesteps is None:
            schedule_timesteps = self.timesteps

        index_candidates = np.where(schedule_timesteps == timestep)[0]

        if len(index_candidates) == 0:
            step_index = len(self.timesteps) - 1
        # The sigma index that is taken for the **very** first `step` is always the
        # second index (or the last index if there is only 1). This makes sure we
        # don't accidentally skip a sigma when starting in the middle (img2img).
        elif len(index_candidates) > 1:
            step_index = int(index_candidates[1])
        else:
            step_index = int(index_candidates[0])

        return step_index

    def _init_step_index(self, timestep):
        if self.begin_index is None:
            self._step_index = self.index_for_timestep(timestep)
        else:
            self._step_index = self._begin_index

    def step(self, model_output, timestep, sample):
        if self.num_inference_steps is None:
            raise ValueError(
                "Number of inference steps is 'None', you need to run 'set_timesteps' "
                "after creating the scheduler"
            )

        if self.step_index is None:
            self._init_step_index(timestep)

        # Improve numerical stability for small number of steps
        lower_order_final = (self.step_index == len(self.timesteps) - 1) and (
            self.config.euler_at_final
            or (self.config.lower_order_final and len(self.timesteps) < 15)
            or self.config.final_sigmas_type == "zero"
        )
        lower_order_second = (
            (self.step_index == len(self.timesteps) - 2)
            and self.config.lower_order_final
            and len(self.timesteps) < 15
        )

        model_output = self.convert_model_output(model_output, sample=sample)
        for i in range(self.config.solver_order - 1):
            self.model_outputs[i] = self.model_outputs[i + 1]
        self.model_outputs[-1] = model_output

        # Upcast to avoid precision issues when computing prev_sample
        sample = sample.astype(np.float32)

        if (
            self.config.solver_order == 1
            or self.lower_order_nums < 1
            or lower_order_final
        ):
            prev_sample = self.dpm_solver_first_order_update(
                model_output, sample=sample
            )
        else:
            prev_sample = self.multistep_dpm_solver_second_order_update(
                self.model_outputs, sample=sample
            )

        if self.lower_order_nums < self.config.solver_order:
            self.lower_order_nums += 1

        # upon completion increase step index by one
        self._step_index += 1

        return prev_sample.astype(model_output.dtype)

    def scale_model_input(self, sample: np.ndarray, timestep=None) -> np.ndarray:
        return sample

    def add_noise(self, original_samples, noise, timesteps):
        sigmas = self.sigmas
        schedule_timesteps = self.timesteps

        timesteps = np.atleast_1d(np.asarray(timesteps))
        if self.begin_index is None:
            step_indices = [
                self.index_for_timestep(t, schedule_timesteps) for t in timesteps
            ]
        elif self.step_index is not None:
            # add_noise is called after first denoising step (for inpainting)
            step_indices = [self.step_index] * timesteps.shape[0]
        else:
            # add noise is called before first denoising step (img2img)
            step_indices = [self.begin_index] * timesteps.shape[0]

        sigma = sigmas[step_indices].flatten()
        while len(sigma.shape) < len(original_samples.shape):
            sigma = np.expand_dims(sigma, axis=-1)

        alpha_t, sigma_t = self._sigma_to_alpha_sigma_t(sigma)
        noisy_samples = alpha_t * original_samples + sigma_t * noise
        return noisy_samples

    def __len__(self):
        return self.config.num_train_timesteps
