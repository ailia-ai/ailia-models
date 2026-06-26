# -*- encoding: utf-8 -*-

from pathlib import Path
from typing import Dict, List,  Union

import numpy as np
import yaml

import warnings

class OrtInferSession:
    def __init__(self, model_file, device_id=-1, intra_op_num_threads=4, disable_optimization=False):
        from onnxruntime import (
            GraphOptimizationLevel,
            InferenceSession,
            SessionOptions,
            get_available_providers,
            get_device,
        )

        device_id = str(device_id)
        sess_opt = SessionOptions()
        sess_opt.intra_op_num_threads = intra_op_num_threads
        sess_opt.log_severity_level = 4
        sess_opt.enable_cpu_mem_arena = False
        if disable_optimization:
            sess_opt.graph_optimization_level = GraphOptimizationLevel.ORT_DISABLE_ALL
        else:
            sess_opt.graph_optimization_level = GraphOptimizationLevel.ORT_ENABLE_ALL

        cuda_ep = "CUDAExecutionProvider"
        cuda_provider_options = {
            "device_id": device_id,
            "arena_extend_strategy": "kNextPowerOfTwo",
            "cudnn_conv_algo_search": "EXHAUSTIVE",
            "do_copy_in_default_stream": "true",
        }
        cpu_ep = "CPUExecutionProvider"
        cpu_provider_options = {
            "arena_extend_strategy": "kSameAsRequested",
        }

        EP_list = []
        if device_id != "-1" and get_device() == "GPU" and cuda_ep in get_available_providers():
            EP_list = [(cuda_ep, cuda_provider_options)]
        EP_list.append((cpu_ep, cpu_provider_options))

        self.session = InferenceSession(model_file, sess_options=sess_opt, providers=EP_list)

        if device_id != "-1" and cuda_ep not in self.session.get_providers():
            warnings.warn(
                f"{cuda_ep} is not avaiable for current env, the inference part is automatically shifted to be executed under {cpu_ep}.\n"
                "Please ensure the installed onnxruntime-gpu version matches your cuda and cudnn version, "
                "you can check their relations from the offical web site: "
                "https://onnxruntime.ai/docs/execution-providers/CUDA-ExecutionProvider.html",
                RuntimeWarning,
            )

    def __call__(self, input_content: List[Union[np.ndarray, np.ndarray]], run_options = None) -> np.ndarray:
        input_dict = dict(zip(self.get_input_names(), input_content))
        try:
            return self.session.run(self.get_output_names(), input_dict, run_options)
        except Exception as e:
            raise Exception("ONNXRuntime inferece failed.")

    def get_input_names(
        self,
    ):
        return [v.name for v in self.session.get_inputs()]

    def get_output_names(
        self,
    ):
        return [v.name for v in self.session.get_outputs()]

class AiliaInferSession:
    session = None

    def __init__(self, model_file, env_id = -1, profile = False):
        import ailia
        self.session = ailia.Net(weight=model_file, env_id=env_id, memory_mode=11)
        self.profile = profile
        if self.profile:
            self.session.set_profile_mode(ailia.PROFILE_AVERAGE)

    def __call__(self, input_content: List[Union[np.ndarray, np.ndarray]], run_options = None) -> np.ndarray:
        return self.session.run(input_content)

    def __del__(self):
        if self.session is not None:
            if self.profile is True:
                print(self.session.get_summary())

def read_yaml(yaml_path: Union[str, Path]) -> Dict:
    if not Path(yaml_path).exists():
        raise FileExistsError(f"The {yaml_path} does not exist.")

    with open(str(yaml_path), "rb") as f:
        data = yaml.load(f, Loader=yaml.Loader)
    return data
