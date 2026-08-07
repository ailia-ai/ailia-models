# Copyright 2023 TSAIL Team and The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import List, Optional, Union

import numpy as np

from ..configuration_utils import ConfigMixin, register_to_config


class DPMSolverMultistepScheduler(ConfigMixin):
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
            thresholding: bool = False,
            dynamic_thresholding_ratio: float = 0.995,
            sample_max_value: float = 1.0,
            algorithm_type: str = "dpmsolver++",
            solver_type: str = "midpoint",
            lower_order_final: bool = True,
            use_karras_sigmas: Optional[bool] = False,
            lambda_min_clipped: float = -float("inf"),
            variance_type: Optional[str] = None,
    ):
        if trained_betas is not None:
            self.betas = np.array(trained_betas, dtype=np.float64)
        elif beta_schedule == "linear":
            self.betas = np.linspace(beta_start, beta_end, num_train_timesteps, dtype=np.float64)
        elif beta_schedule == "scaled_linear":
            # this schedule is very specific to the latent diffusion model.
            self.betas = (
                    np.linspace(beta_start ** 0.5, beta_end ** 0.5, num_train_timesteps, dtype=np.float64) ** 2
            )
        else:
            raise NotImplementedError(f"{beta_schedule} does is not implemented for {self.__class__}")

        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = np.cumprod(self.alphas, axis=0)
        # Currently we only support VP-type noise schedule
        self.alpha_t = np.sqrt(self.alphas_cumprod)
        self.sigma_t = np.sqrt(1 - self.alphas_cumprod)
        self.lambda_t = np.log(self.alpha_t) - np.log(self.sigma_t)

        # standard deviation of the initial noise distribution
        self.init_noise_sigma = 1.0

        # settings for DPM-Solver
        if algorithm_type not in ["dpmsolver", "dpmsolver++", "sde-dpmsolver", "sde-dpmsolver++"]:
            if algorithm_type == "deis":
                self.register_to_config(algorithm_type="dpmsolver++")
            else:
                raise NotImplementedError(f"{algorithm_type} does is not implemented for {self.__class__}")

        if solver_type not in ["midpoint", "heun"]:
            if solver_type in ["logrho", "bh1", "bh2"]:
                self.register_to_config(solver_type="midpoint")
            else:
                raise NotImplementedError(f"{solver_type} does is not implemented for {self.__class__}")

        # setable values
        self.num_inference_steps = None
        timesteps = np.linspace(0, num_train_timesteps - 1, num_train_timesteps, dtype=np.float32)[::-1]
        self.timesteps = timesteps
        self.model_outputs = [None] * solver_order
        self.lower_order_nums = 0
        self.use_karras_sigmas = use_karras_sigmas

    def set_timesteps(self, num_inference_steps: int = None):
        """
        Sets the timesteps used for the diffusion chain. Supporting function to be run before inference.

        Args:
            num_inference_steps (`int`):
                the number of diffusion steps used when generating samples with a pre-trained model.
        """
        # Clipping the minimum of all lambda(t) for numerical stability.
        # This is critical for cosine (squaredcos_cap_v2) noise schedule.
        clipped_idx = np.searchsorted(np.flip(self.lambda_t, axis=0), self.config.lambda_min_clipped)
        timesteps = (
            np.linspace(0, self.config.num_train_timesteps - 1 - clipped_idx, num_inference_steps + 1)
                .round()[::-1][:-1]
                .copy()
                .astype(np.int64)
        )

        if self.use_karras_sigmas:
            sigmas = np.array(((1 - self.alphas_cumprod) / self.alphas_cumprod) ** 0.5)
            log_sigmas = np.log(sigmas)
            sigmas = self._convert_to_karras(in_sigmas=sigmas, num_inference_steps=num_inference_steps)
            timesteps = np.array([self._sigma_to_t(sigma, log_sigmas) for sigma in sigmas]).round()
            timesteps = np.flip(timesteps).copy().astype(np.int64)

        # when num_inference_steps == num_train_timesteps, we can end up with
        # duplicates in timesteps.
        _, unique_indices = np.unique(timesteps, return_index=True)
        timesteps = timesteps[np.sort(unique_indices)]

        self.timesteps = timesteps
        self.num_inference_steps = len(timesteps)
        self.model_outputs = [
                                 None,
                             ] * self.config.solver_order
        self.lower_order_nums = 0

    # Copied from diffusers.schedulers.scheduling_ddpm.DDPMScheduler._threshold_sample
    def _threshold_sample(self, sample: np.ndarray) -> np.ndarray:
        """
        "Dynamic thresholding: At each sampling step we set s to a certain percentile absolute pixel value in xt0 (the
        prediction of x_0 at timestep t), and if s > 1, then we threshold xt0 to the range [-s, s] and then divide by
        s. Dynamic thresholding pushes saturated pixels (those near -1 and 1) inwards, thereby actively preventing
        pixels from saturation at each step. We find that dynamic thresholding results in significantly better
        photorealism as well as better image-text alignment, especially when using very large guidance weights."

        https://arxiv.org/abs/2205.11487
        """
        dtype = sample.dtype
        batch_size, channels, height, width = sample.shape

        if dtype not in (float, np.float64):
            sample = sample.astype(np.float64)  # upcast for quantile calculation, and clamp not implemented for cpu half

        # Flatten sample for doing quantile calculation along each image
        sample = sample.reshape(batch_size, channels * height * width)

        abs_sample = np.abs(sample)  # "a certain percentile absolute pixel value"

        s = np.quantile(abs_sample, self.config.dynamic_thresholding_ratio, axis=1)
        s = np.clip(
            s, a_min=1, a_max=self.config.sample_max_value
        )  # When clamped to min=1, equivalent to standard clipping to [-1, 1]

        s = np.expand_dims(s, axis=1)  # (batch_size, 1) because clamp will broadcast along dim=0
        sample = np.clip(sample, -s, s) / s  # "we threshold xt0 to the range [-s, s] and then divide by s"

        sample = sample.reshape(batch_size, channels, height, width)
        sample = sample.astype(dtype)

        return sample

    # Copied from diffusers.schedulers.scheduling_euler_discrete.EulerDiscreteScheduler._sigma_to_t
    def _sigma_to_t(self, sigma, log_sigmas):
        # get log sigma
        log_sigma = np.log(sigma)

        # get distribution
        dists = log_sigma - log_sigmas[:, np.newaxis]

        # get sigmas range
        low_idx = np.cumsum((dists >= 0), axis=0).argmax(axis=0).clip(max=log_sigmas.shape[0] - 2)
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

    # Copied from diffusers.schedulers.scheduling_euler_discrete.EulerDiscreteScheduler._convert_to_karras
    def _convert_to_karras(self, in_sigmas: np.ndarray, num_inference_steps) -> np.ndarray:
        """Constructs the noise schedule of Karras et al. (2022)."""

        sigma_min: float = in_sigmas[-1].item()
        sigma_max: float = in_sigmas[0].item()

        rho = 7.0  # 7.0 is the value used in the paper
        ramp = np.linspace(0, 1, num_inference_steps)
        min_inv_rho = sigma_min ** (1 / rho)
        max_inv_rho = sigma_max ** (1 / rho)
        sigmas = (max_inv_rho + ramp * (min_inv_rho - max_inv_rho)) ** rho
        return sigmas

    def convert_model_output(
            self, model_output: np.ndarray, timestep: int, sample: np.ndarray
    ) -> np.ndarray:
        """
        Convert the model output to the corresponding type that the algorithm (DPM-Solver / DPM-Solver++) needs.

        DPM-Solver is designed to discretize an integral of the noise prediction model, and DPM-Solver++ is designed to
        discretize an integral of the data prediction model. So we need to first convert the model output to the
        corresponding type to match the algorithm.

        Note that the algorithm type and the model type is decoupled. That is to say, we can use either DPM-Solver or
        DPM-Solver++ for both noise prediction model and data prediction model.

        Args:
            model_output (`np.ndarray`): direct output from learned diffusion model.
            timestep (`int`): current discrete timestep in the diffusion chain.
            sample (`np.ndarray`):
                current instance of sample being created by diffusion process.

        Returns:
            `np.ndarray`: the converted model output.
        """

        # DPM-Solver++ needs to solve an integral of the data prediction model.
        if self.config.algorithm_type in ["dpmsolver++", "sde-dpmsolver++"]:
            if self.config.prediction_type == "epsilon":
                # DPM-Solver and DPM-Solver++ only need the "mean" output.
                if self.config.variance_type in ["learned", "learned_range"]:
                    model_output = model_output[:, :3]
                alpha_t, sigma_t = self.alpha_t[timestep], self.sigma_t[timestep]
                x0_pred = (sample - sigma_t * model_output) / alpha_t
            elif self.config.prediction_type == "sample":
                x0_pred = model_output
            elif self.config.prediction_type == "v_prediction":
                alpha_t, sigma_t = self.alpha_t[timestep], self.sigma_t[timestep]
                x0_pred = alpha_t * sample - sigma_t * model_output
            else:
                raise ValueError(
                    f"prediction_type given as {self.config.prediction_type} must be one of `epsilon`, `sample`, or"
                    " `v_prediction` for the DPMSolverMultistepScheduler."
                )

            if self.config.thresholding:
                x0_pred = self._threshold_sample(x0_pred)

            return x0_pred

        # DPM-Solver needs to solve an integral of the noise prediction model.
        elif self.config.algorithm_type in ["dpmsolver", "sde-dpmsolver"]:
            if self.config.prediction_type == "epsilon":
                # DPM-Solver and DPM-Solver++ only need the "mean" output.
                if self.config.variance_type in ["learned", "learned_range"]:
                    epsilon = model_output[:, :3]
                else:
                    epsilon = model_output
            elif self.config.prediction_type == "sample":
                alpha_t, sigma_t = self.alpha_t[timestep], self.sigma_t[timestep]
                epsilon = (sample - alpha_t * model_output) / sigma_t
            elif self.config.prediction_type == "v_prediction":
                alpha_t, sigma_t = self.alpha_t[timestep], self.sigma_t[timestep]
                epsilon = alpha_t * model_output + sigma_t * sample
            else:
                raise ValueError(
                    f"prediction_type given as {self.config.prediction_type} must be one of `epsilon`, `sample`, or"
                    " `v_prediction` for the DPMSolverMultistepScheduler."
                )

            if self.config.thresholding:
                alpha_t, sigma_t = self.alpha_t[timestep], self.sigma_t[timestep]
                x0_pred = (sample - sigma_t * epsilon) / alpha_t
                x0_pred = self._threshold_sample(x0_pred)
                epsilon = (sample - alpha_t * x0_pred) / sigma_t

            return epsilon

    def dpm_solver_first_order_update(
            self,
            model_output: np.ndarray,
            timestep: int,
            prev_timestep: int,
            sample: np.ndarray,
            noise: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """
        One step for the first-order DPM-Solver (equivalent to DDIM).

        See https://arxiv.org/abs/2206.00927 for the detailed derivation.

        Args:
            model_output (`np.ndarray`): direct output from learned diffusion model.
            timestep (`int`): current discrete timestep in the diffusion chain.
            prev_timestep (`int`): previous discrete timestep in the diffusion chain.
            sample (`np.ndarray`):
                current instance of sample being created by diffusion process.

        Returns:
            `np.ndarray`: the sample tensor at the previous timestep.
        """
        lambda_t, lambda_s = self.lambda_t[prev_timestep], self.lambda_t[timestep]
        alpha_t, alpha_s = self.alpha_t[prev_timestep], self.alpha_t[timestep]
        sigma_t, sigma_s = self.sigma_t[prev_timestep], self.sigma_t[timestep]
        h = lambda_t - lambda_s
        if self.config.algorithm_type == "dpmsolver++":
            x_t = (sigma_t / sigma_s) * sample - (alpha_t * (np.exp(-h) - 1.0)) * model_output
        elif self.config.algorithm_type == "dpmsolver":
            x_t = (alpha_t / alpha_s) * sample - (sigma_t * (np.exp(h) - 1.0)) * model_output
        elif self.config.algorithm_type == "sde-dpmsolver++":
            assert noise is not None
            x_t = (
                    (sigma_t / sigma_s * np.exp(-h)) * sample
                    + (alpha_t * (1 - np.exp(-2.0 * h))) * model_output
                    + sigma_t * np.sqrt(1.0 - np.exp(-2 * h)) * noise
            )
        elif self.config.algorithm_type == "sde-dpmsolver":
            assert noise is not None
            x_t = (
                    (alpha_t / alpha_s) * sample
                    - 2.0 * (sigma_t * (np.exp(h) - 1.0)) * model_output
                    + sigma_t * np.sqrt(np.exp(2 * h) - 1.0) * noise
            )
        return x_t

    def multistep_dpm_solver_second_order_update(
            self,
            model_output_list: List[np.ndarray],
            timestep_list: List[int],
            prev_timestep: int,
            sample: np.ndarray,
            noise: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """
        One step for the second-order multistep DPM-Solver.

        Args:
            model_output_list (`List[np.ndarray]`):
                direct outputs from learned diffusion model at current and latter timesteps.
            timestep (`int`): current and latter discrete timestep in the diffusion chain.
            prev_timestep (`int`): previous discrete timestep in the diffusion chain.
            sample (`np.ndarray`):
                current instance of sample being created by diffusion process.

        Returns:
            `np.ndarray`: the sample tensor at the previous timestep.
        """
        t, s0, s1 = prev_timestep, timestep_list[-1], timestep_list[-2]
        m0, m1 = model_output_list[-1], model_output_list[-2]
        lambda_t, lambda_s0, lambda_s1 = self.lambda_t[t], self.lambda_t[s0], self.lambda_t[s1]
        alpha_t, alpha_s0 = self.alpha_t[t], self.alpha_t[s0]
        sigma_t, sigma_s0 = self.sigma_t[t], self.sigma_t[s0]
        h, h_0 = lambda_t - lambda_s0, lambda_s0 - lambda_s1
        r0 = h_0 / h
        D0, D1 = m0, (1.0 / r0) * (m0 - m1)
        if self.config.algorithm_type == "dpmsolver++":
            # See https://arxiv.org/abs/2211.01095 for detailed derivations
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
        elif self.config.algorithm_type == "dpmsolver":
            # See https://arxiv.org/abs/2206.00927 for detailed derivations
            if self.config.solver_type == "midpoint":
                x_t = (
                        (alpha_t / alpha_s0) * sample
                        - (sigma_t * (np.exp(h) - 1.0)) * D0
                        - 0.5 * (sigma_t * (np.exp(h) - 1.0)) * D1
                )
            elif self.config.solver_type == "heun":
                x_t = (
                        (alpha_t / alpha_s0) * sample
                        - (sigma_t * (np.exp(h) - 1.0)) * D0
                        - (sigma_t * ((np.exp(h) - 1.0) / h - 1.0)) * D1
                )
        elif self.config.algorithm_type == "sde-dpmsolver++":
            assert noise is not None
            if self.config.solver_type == "midpoint":
                x_t = (
                        (sigma_t / sigma_s0 * np.exp(-h)) * sample
                        + (alpha_t * (1 - np.exp(-2.0 * h))) * D0
                        + 0.5 * (alpha_t * (1 - np.exp(-2.0 * h))) * D1
                        + sigma_t * np.sqrt(1.0 - np.exp(-2 * h)) * noise
                )
            elif self.config.solver_type == "heun":
                x_t = (
                        (sigma_t / sigma_s0 * np.exp(-h)) * sample
                        + (alpha_t * (1 - np.exp(-2.0 * h))) * D0
                        + (alpha_t * ((1.0 - np.exp(-2.0 * h)) / (-2.0 * h) + 1.0)) * D1
                        + sigma_t * np.sqrt(1.0 - np.exp(-2 * h)) * noise
                )
        elif self.config.algorithm_type == "sde-dpmsolver":
            assert noise is not None
            if self.config.solver_type == "midpoint":
                x_t = (
                        (alpha_t / alpha_s0) * sample
                        - 2.0 * (sigma_t * (np.exp(h) - 1.0)) * D0
                        - (sigma_t * (np.exp(h) - 1.0)) * D1
                        + sigma_t * np.sqrt(np.exp(2 * h) - 1.0) * noise
                )
            elif self.config.solver_type == "heun":
                x_t = (
                        (alpha_t / alpha_s0) * sample
                        - 2.0 * (sigma_t * (np.exp(h) - 1.0)) * D0
                        - 2.0 * (sigma_t * ((np.exp(h) - 1.0) / h - 1.0)) * D1
                        + sigma_t * np.sqrt(np.exp(2 * h) - 1.0) * noise
                )
        return x_t

    def multistep_dpm_solver_third_order_update(
            self,
            model_output_list: List[np.ndarray],
            timestep_list: List[int],
            prev_timestep: int,
            sample: np.ndarray,
    ) -> np.ndarray:
        """
        One step for the third-order multistep DPM-Solver.

        Args:
            model_output_list (`List[np.ndarray]`):
                direct outputs from learned diffusion model at current and latter timesteps.
            timestep (`int`): current and latter discrete timestep in the diffusion chain.
            prev_timestep (`int`): previous discrete timestep in the diffusion chain.
            sample (`np.ndarray`):
                current instance of sample being created by diffusion process.

        Returns:
            `np.ndarray`: the sample tensor at the previous timestep.
        """
        t, s0, s1, s2 = prev_timestep, timestep_list[-1], timestep_list[-2], timestep_list[-3]
        m0, m1, m2 = model_output_list[-1], model_output_list[-2], model_output_list[-3]
        lambda_t, lambda_s0, lambda_s1, lambda_s2 = (
            self.lambda_t[t],
            self.lambda_t[s0],
            self.lambda_t[s1],
            self.lambda_t[s2],
        )
        alpha_t, alpha_s0 = self.alpha_t[t], self.alpha_t[s0]
        sigma_t, sigma_s0 = self.sigma_t[t], self.sigma_t[s0]
        h, h_0, h_1 = lambda_t - lambda_s0, lambda_s0 - lambda_s1, lambda_s1 - lambda_s2
        r0, r1 = h_0 / h, h_1 / h
        D0 = m0
        D1_0, D1_1 = (1.0 / r0) * (m0 - m1), (1.0 / r1) * (m1 - m2)
        D1 = D1_0 + (r0 / (r0 + r1)) * (D1_0 - D1_1)
        D2 = (1.0 / (r0 + r1)) * (D1_0 - D1_1)
        if self.config.algorithm_type == "dpmsolver++":
            # See https://arxiv.org/abs/2206.00927 for detailed derivations
            x_t = (
                    (sigma_t / sigma_s0) * sample
                    - (alpha_t * (np.exp(-h) - 1.0)) * D0
                    + (alpha_t * ((np.exp(-h) - 1.0) / h + 1.0)) * D1
                    - (alpha_t * ((np.exp(-h) - 1.0 + h) / h ** 2 - 0.5)) * D2
            )
        elif self.config.algorithm_type == "dpmsolver":
            # See https://arxiv.org/abs/2206.00927 for detailed derivations
            x_t = (
                    (alpha_t / alpha_s0) * sample
                    - (sigma_t * (np.exp(h) - 1.0)) * D0
                    - (sigma_t * ((np.exp(h) - 1.0) / h - 1.0)) * D1
                    - (sigma_t * ((np.exp(h) - 1.0 - h) / h ** 2 - 0.5)) * D2
            )
        return x_t

    def step(
            self,
            model_output: np.ndarray,
            timestep: int,
            sample: np.ndarray,
    ):
        """
        Step function propagating the sample with the multistep DPM-Solver.

        Args:
            model_output (`np.ndarray`): direct output from learned diffusion model.
            timestep (`int`): current discrete timestep in the diffusion chain.
            sample (`np.ndarray`):
                current instance of sample being created by diffusion process.
        """
        if self.num_inference_steps is None:
            raise ValueError(
                "Number of inference steps is 'None', you need to run 'set_timesteps' after creating the scheduler"
            )

        step_index = (self.timesteps == timestep).nonzero()[0]
        if len(step_index) == 0:
            step_index = len(self.timesteps) - 1
        else:
            step_index = step_index.item()
        prev_timestep = 0 if step_index == len(self.timesteps) - 1 else self.timesteps[step_index + 1]
        lower_order_final = (
                (step_index == len(self.timesteps) - 1)
                and self.config.lower_order_final and len(self.timesteps) < 15
        )
        lower_order_second = (
                (step_index == len(self.timesteps) - 2)
                and self.config.lower_order_final and len(self.timesteps) < 15
        )

        model_output = self.convert_model_output(model_output, timestep, sample)
        for i in range(self.config.solver_order - 1):
            self.model_outputs[i] = self.model_outputs[i + 1]
        self.model_outputs[-1] = model_output

        if self.config.algorithm_type in ["sde-dpmsolver", "sde-dpmsolver++"]:
            n = np.prod(model_output.shape)
            noise = np.random.randn(n, dtype=model_output.dtype).reshape(model_output.shape)
        else:
            noise = None

        if self.config.solver_order == 1 or self.lower_order_nums < 1 or lower_order_final:
            prev_sample = self.dpm_solver_first_order_update(
                model_output, timestep, prev_timestep, sample, noise=noise
            )
        elif self.config.solver_order == 2 or self.lower_order_nums < 2 or lower_order_second:
            timestep_list = [self.timesteps[step_index - 1], timestep]
            prev_sample = self.multistep_dpm_solver_second_order_update(
                self.model_outputs, timestep_list, prev_timestep, sample, noise=noise
            )
        else:
            timestep_list = [self.timesteps[step_index - 2], self.timesteps[step_index - 1], timestep]
            prev_sample = self.multistep_dpm_solver_third_order_update(
                self.model_outputs, timestep_list, prev_timestep, sample
            )

        if self.lower_order_nums < self.config.solver_order:
            self.lower_order_nums += 1

        return prev_sample

    def scale_model_input(self, sample: np.ndarray, *args, **kwargs) -> np.ndarray:
        """
        Ensures interchangeability with schedulers that need to scale the denoising model input depending on the
        current timestep.

        Args:
            sample (`np.ndarray`): input sample

        Returns:
            `np.ndarray`: scaled input sample
        """
        return sample

    def __len__(self):
        return self.config.num_train_timesteps
