# -*- encoding: utf-8 -*-
# Copyright FunASR (https://github.com/alibaba-damo-academy/FunASR). All Rights Reserved.
#  MIT License  (https://opensource.org/licenses/MIT)

from pathlib import Path
from typing import List, Union, Tuple, Dict

import librosa
import numpy as np

from .utils.utils import AiliaInferSession, OrtInferSession, read_yaml
from .utils.frontend import WavFrontendOnline
from .utils.e2e_vad import E2EVadModel

class Fsmn_vad_online:
    """
    Author: Speech Lab of DAMO Academy, Alibaba Group
    Deep-FSMN for Large Vocabulary Continuous Speech Recognition
    https://arxiv.org/abs/1803.05030
    """

    def __init__(
        self,
        batch_size: int = 1,
        device_id: Union[str, int] = "-1",
        intra_op_num_threads: int = 4,
        max_end_sil: int = None,
        env_id: int = -1,
        onnx: bool = False,
        ailia_audio: bool = False,
        profile: bool = False,
        model_file: str = "./speech_fsmn_vad_zh-cn-16k-common.onnx",
    ):
        config_file = "./vad_config/config.yaml"
        self.cmvn_file = "./vad_config/am.mvn"

        self.config = read_yaml(config_file)

        if onnx:
            self.ort_infer = OrtInferSession(
                model_file, device_id, intra_op_num_threads=intra_op_num_threads
            )
        else:
            self.ort_infer = AiliaInferSession(
                model_file, env_id = env_id, profile = profile
            )

        self.batch_size = batch_size
        self.max_end_sil = (
            max_end_sil if max_end_sil is not None else self.config["model_conf"]["max_end_silence_time"]
        )
        self.encoder_conf = self.config["encoder_conf"]
        self.ailia_audio = ailia_audio

    def prepare_cache(self, in_cache: list = []):
        if len(in_cache) > 0:
            return in_cache
        fsmn_layers = self.encoder_conf["fsmn_layers"]
        proj_dim = self.encoder_conf["proj_dim"]
        lorder = self.encoder_conf["lorder"]
        for i in range(fsmn_layers):
            cache = np.zeros((1, proj_dim, lorder - 1, 1)).astype(np.float32)
            in_cache.append(cache)
        return in_cache

    def __call__(self, audio_in: np.ndarray, **kwargs) -> List:
        waveforms = np.expand_dims(audio_in, axis=0)

        param_dict: Dict = kwargs.get("param_dict", dict())
        is_final = param_dict.get("is_final", False)
        frontend: WavFrontendOnline = param_dict.get("frontend", WavFrontendOnline(cmvn_file=self.cmvn_file, ailia_audio=self.ailia_audio, **self.config["frontend_conf"]))
        feats, feats_len = self.extract_feat(frontend=frontend, waveforms=waveforms, is_final=is_final)
        segments = []
        if feats.size != 0:
            in_cache = param_dict.get("in_cache", list())
            in_cache = self.prepare_cache(in_cache)
            vad_scorer = param_dict.get("vad_scorer", E2EVadModel(self.config["model_conf"]))
            try:
                inputs = [feats]
                inputs.extend(in_cache)
                scores, out_caches = self.infer(inputs)
                param_dict["in_cache"] = out_caches
                waveforms = frontend.get_waveforms()
                segments = vad_scorer(
                    scores, waveforms, is_final=is_final, max_end_sil=self.max_end_sil, online=True
                )

            except Exception:
                #logging.warning("input wav is silence or noise")
                segments = []
        param_dict.update({"frontend": frontend, "vad_scorer": vad_scorer})
        return segments

    def extract_feat(
        self, frontend: WavFrontendOnline, waveforms: np.ndarray, is_final: bool = False
    ) -> Tuple[np.ndarray, np.ndarray]:
        waveforms_lens = np.zeros(waveforms.shape[0]).astype(np.int32)
        for idx, waveform in enumerate(waveforms):
            waveforms_lens[idx] = waveform.shape[-1]

        feats, feats_len = frontend.extract_fbank(waveforms, waveforms_lens, is_final)
        return feats.astype(np.float32), feats_len.astype(np.int32)

    @staticmethod
    def pad_feats(feats: List[np.ndarray], max_feat_len: int) -> np.ndarray:
        def pad_feat(feat: np.ndarray, cur_len: int) -> np.ndarray:
            pad_width = ((0, max_feat_len - cur_len), (0, 0))
            return np.pad(feat, pad_width, "constant", constant_values=0)

        feat_res = [pad_feat(feat, feat.shape[0]) for feat in feats]
        feats = np.array(feat_res).astype(np.float32)
        return feats

    def infer(self, feats: List) -> Tuple[np.ndarray, np.ndarray]:

        outputs = self.ort_infer(feats)
        scores, out_caches = outputs[0], outputs[1:]
        return scores, out_caches
