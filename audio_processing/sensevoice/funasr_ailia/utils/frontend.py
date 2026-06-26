# -*- encoding: utf-8 -*-
from pathlib import Path
from typing import List, Tuple, Union
import copy

import numpy as np

import librosa
import ailia.audio

USE_AILIA_AUDIO = True

# -*- coding: utf-8 -*-

class AiliaFrameOptions:
    """Options related to frame extraction (frame_opts)."""
    def __init__(
        self,
        samp_freq: float = 16000.0,
        dither: float = 1.0,
        window_type: str = "hamming",
        frame_shift_ms: float = 10.0,
        frame_length_ms: float = 25.0,
        snip_edges: bool = True,
    ):
        self.samp_freq = samp_freq
        self.dither = dither
        self.window_type = window_type
        self.frame_shift_ms = frame_shift_ms
        self.frame_length_ms = frame_length_ms
        self.snip_edges = snip_edges

class AiliaMelOptions:
    """Options for mel filterbank computation (mel_opts)."""
    def __init__(self, num_bins: int = 80, debug_mel: bool = False):
        self.num_bins = num_bins
        self.debug_mel = debug_mel


class AiliaFbankOptions:
    """Wrapper to mimic kaldi_native_fbank.FbankOptions."""
    def __init__(self):
        self.frame_opts = AiliaFrameOptions()
        self.mel_opts = AiliaMelOptions()
        self.energy_floor = 0.0

class AiliaOnlineFbank:
    """Librosa-based replacement for kaldi_native_fbank.OnlineFbank."""

    def __init__(self, opts):
        self.opts = opts
        self.reset()

    def reset(self):
        """Reset internal states."""
        self.sr = int(self.opts.frame_opts.samp_freq)
        self.dither = float(self.opts.frame_opts.dither)
        self.window_type = self.opts.frame_opts.window_type
        self.frame_length_ms = float(self.opts.frame_opts.frame_length_ms)
        self.frame_shift_ms = float(self.opts.frame_opts.frame_shift_ms)
        self.n_mels = int(self.opts.mel_opts.num_bins)
        self.frame_length = int(self.sr * self.frame_length_ms / 1000)
        self.frame_shift = int(self.sr * self.frame_shift_ms / 1000)
        self.snip_edges = getattr(self.opts.frame_opts, "snip_edges", True)
        self._waveform_buffer = np.array([], dtype=np.float32)
        self._frames_cache = None
        self.preemphasis_coefficient = 0.97
        
    # ――― Kaldi オンライン互換インターフェース ―――
    def accept_waveform(self, sampling_rate: float, waveform):
        """Receive waveform and append it to buffer."""
        waveform = np.asarray(waveform, dtype=np.float32)
        if self.dither > 0:
            waveform += np.random.normal(scale=self.dither, size=len(waveform)).astype(np.float32)
        if self.preemphasis_coefficient > 0: # Kaldiの内部処理をシミュレート
            alpha = self.preemphasis_coefficient
            waveform = np.append(waveform[0], waveform[1:] - alpha * waveform[:-1])
        self._waveform_buffer = np.concatenate([self._waveform_buffer, waveform])
        self._compute_fbanks()

    def _compute_fbanks(self):
        """Compute mel filterbank features for all frames available."""
        if len(self._waveform_buffer) < self.frame_length:
            self._frames_cache = np.empty((0, self.n_mels), dtype=np.float32)
            return

        hop_length = self.frame_shift
        win_length = self.frame_length

        # librosa uses float[-1, 1] range, assume waveform already scaled appropriately
        if USE_AILIA_AUDIO:
            mel_spec = ailia.audio.mel_spectrogram(
                wav=self._waveform_buffer,
                sample_rate=self.sr,
                fft_n= win_length,
                hop_n=hop_length,
                win_n=win_length,
                win_type=self.window_type,
                mel_n=self.n_mels,
                center_mode=not self.snip_edges,
                power=2.0,
            )
        else:
            mel_spec = librosa.feature.melspectrogram(
                y=self._waveform_buffer,
                sr=self.sr,
                n_fft=win_length,
                hop_length=hop_length,
                win_length=win_length,
                window=self.window_type,
                n_mels=self.n_mels,
                center=not self.snip_edges,
                power=2.0,
            )
        mel_spec = np.log(np.maximum(mel_spec, 1e-10)).T.astype(np.float32)
        self._frames_cache = mel_spec

    @property
    def num_frames_ready(self):
        return 0 if self._frames_cache is None else self._frames_cache.shape[0]

    def get_frame(self, i):
        """Get i-th fbank frame as numpy array."""
        if self._frames_cache is None or i >= self._frames_cache.shape[0]:
            raise IndexError("Frame index out of range or cache empty.")
        return self._frames_cache[i]

    def get_all_frames(self):
        """Return all computed frames."""
        return self._frames_cache

    # Optional: if you want to flush buffer or reset
    def flush(self):
        self._waveform_buffer = np.array([], dtype=np.float32)
        self._frames_cache = None

class WavFrontend:
    """Conventional frontend structure for ASR."""

    def __init__(
        self,
        cmvn_file: str = None,
        fs: int = 16000,
        window: str = "hamming",
        n_mels: int = 80,
        frame_length: int = 25,
        frame_shift: int = 10,
        lfr_m: int = 1,
        lfr_n: int = 1,
        dither: float = 1.0,
        ailia_audio: bool = False,
        **kwargs,
    ) -> None:
        self.ailia_audio = ailia_audio

        if not self.ailia_audio:
            import kaldi_native_fbank as knf
            opts = knf.FbankOptions()
        else:
            opts = AiliaFbankOptions()
        opts.frame_opts.samp_freq = fs
        opts.frame_opts.dither = dither
        opts.frame_opts.window_type = window
        opts.frame_opts.frame_shift_ms = float(frame_shift)
        opts.frame_opts.frame_length_ms = float(frame_length)
        opts.mel_opts.num_bins = n_mels
        opts.energy_floor = 0
        opts.frame_opts.snip_edges = True
        opts.mel_opts.debug_mel = False
        self.opts = opts

        self.lfr_m = lfr_m
        self.lfr_n = lfr_n
        self.cmvn_file = cmvn_file

        if self.cmvn_file:
            self.cmvn = load_cmvn(self.cmvn_file)
        self.fbank_fn = None
        self.fbank_beg_idx = 0
        self.reset_status()

    def fbank(self, waveform: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        waveform = waveform * (1 << 15)
        if not self.ailia_audio:
            import kaldi_native_fbank as knf
            fbank_fn = knf.OnlineFbank(self.opts)
        else:
            fbank_fn = AiliaOnlineFbank(self.opts)
        fbank_fn.accept_waveform(self.opts.frame_opts.samp_freq, waveform.tolist())
        frames = fbank_fn.num_frames_ready
        mat = np.empty([frames, self.opts.mel_opts.num_bins])
        for i in range(frames):
            mat[i, :] = fbank_fn.get_frame(i)
        feat = mat.astype(np.float32)
        feat_len = np.array(mat.shape[0]).astype(np.int32)
        return feat, feat_len

    def fbank_online(self, waveform: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        waveform = waveform * (1 << 15)
        self.fbank_fn.accept_waveform(self.opts.frame_opts.samp_freq, waveform.tolist())
        frames = self.fbank_fn.num_frames_ready
        mat = np.empty([frames, self.opts.mel_opts.num_bins])
        for i in range(self.fbank_beg_idx, frames):
            mat[i, :] = self.fbank_fn.get_frame(i)
        feat = mat.astype(np.float32)
        feat_len = np.array(mat.shape[0]).astype(np.int32)
        return feat, feat_len

    def reset_status(self):
        if not self.ailia_audio:
            import kaldi_native_fbank as knf
            self.fbank_fn = knf.OnlineFbank(self.opts)
        else:
            self.fbank_fn = AiliaOnlineFbank(self.opts)
        self.fbank_beg_idx = 0

    def lfr_cmvn(self, feat: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        if self.lfr_m != 1 or self.lfr_n != 1:
            feat = self.apply_lfr(feat, self.lfr_m, self.lfr_n)

        if self.cmvn_file:
            feat = self.apply_cmvn(feat)

        feat_len = np.array(feat.shape[0]).astype(np.int32)
        return feat, feat_len

    @staticmethod
    def apply_lfr(inputs: np.ndarray, lfr_m: int, lfr_n: int) -> np.ndarray:
        LFR_inputs = []

        T = inputs.shape[0]
        T_lfr = int(np.ceil(T / lfr_n))
        left_padding = np.tile(inputs[0], ((lfr_m - 1) // 2, 1))
        inputs = np.vstack((left_padding, inputs))
        T = T + (lfr_m - 1) // 2
        for i in range(T_lfr):
            if lfr_m <= T - i * lfr_n:
                LFR_inputs.append((inputs[i * lfr_n : i * lfr_n + lfr_m]).reshape(1, -1))
            else:
                # process last LFR frame
                num_padding = lfr_m - (T - i * lfr_n)
                frame = inputs[i * lfr_n :].reshape(-1)
                for _ in range(num_padding):
                    frame = np.hstack((frame, inputs[-1]))

                LFR_inputs.append(frame)
        LFR_outputs = np.vstack(LFR_inputs).astype(np.float32)
        return LFR_outputs

    def apply_cmvn(self, inputs: np.ndarray) -> np.ndarray:
        """
        Apply CMVN with mvn data
        """
        frame, dim = inputs.shape
        means = np.tile(self.cmvn[0:1, :dim], (frame, 1))
        vars = np.tile(self.cmvn[1:2, :dim], (frame, 1))
        inputs = (inputs + means) * vars
        return inputs

def load_cmvn(cmvn_file: Union[str, Path]) -> np.ndarray:
    """load cmvn file to numpy array. 

    Args:
        cmvn_file (Union[str, Path]): cmvn file path.

    Raises:
        FileNotFoundError: cmvn file not exits.

    Returns:
        np.ndarray: cmvn array. shape is (2, dim).The first row is means, the second row is vars.
    """

    cmvn_file = Path(cmvn_file)
    if not cmvn_file.exists():
        raise FileNotFoundError("cmvn file not exits")
    
    with open(cmvn_file, "r", encoding="utf-8") as f:
        lines = f.readlines()
    means_list = []
    vars_list = []
    for i in range(len(lines)):
        line_item = lines[i].split()
        if line_item[0] == "<AddShift>":
            line_item = lines[i + 1].split()
            if line_item[0] == "<LearnRateCoef>":
                add_shift_line = line_item[3 : (len(line_item) - 1)]
                means_list = list(add_shift_line)
                continue
        elif line_item[0] == "<Rescale>":
            line_item = lines[i + 1].split()
            if line_item[0] == "<LearnRateCoef>":
                rescale_line = line_item[3 : (len(line_item) - 1)]
                vars_list = list(rescale_line)
                continue

    means = np.array(means_list).astype(np.float64)
    vars = np.array(vars_list).astype(np.float64)
    cmvn = np.array([means, vars])
    return cmvn


class WavFrontendOnline(WavFrontend):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # add variables
        self.frame_sample_length = int(
            self.opts.frame_opts.frame_length_ms * self.opts.frame_opts.samp_freq / 1000
        )
        self.frame_shift_sample_length = int(
            self.opts.frame_opts.frame_shift_ms * self.opts.frame_opts.samp_freq / 1000
        )
        self.waveform = None
        self.reserve_waveforms = None
        self.input_cache = None
        self.lfr_splice_cache = []

    @staticmethod
    # inputs has catted the cache
    def apply_lfr(
        inputs: np.ndarray, lfr_m: int, lfr_n: int, is_final: bool = False
    ) -> Tuple[np.ndarray, np.ndarray, int]:
        """
        Apply lfr with data
        """

        LFR_inputs = []
        T = inputs.shape[0]  # include the right context
        T_lfr = int(
            np.ceil((T - (lfr_m - 1) // 2) / lfr_n)
        )  # minus the right context: (lfr_m - 1) // 2
        splice_idx = T_lfr
        for i in range(T_lfr):
            if lfr_m <= T - i * lfr_n:
                LFR_inputs.append((inputs[i * lfr_n : i * lfr_n + lfr_m]).reshape(1, -1))
            else:  # process last LFR frame
                if is_final:
                    num_padding = lfr_m - (T - i * lfr_n)
                    frame = (inputs[i * lfr_n :]).reshape(-1)
                    for _ in range(num_padding):
                        frame = np.hstack((frame, inputs[-1]))
                    LFR_inputs.append(frame)
                else:
                    # update splice_idx and break the circle
                    splice_idx = i
                    break
        splice_idx = min(T - 1, splice_idx * lfr_n)
        lfr_splice_cache = inputs[splice_idx:, :]
        LFR_outputs = np.vstack(LFR_inputs)
        return LFR_outputs.astype(np.float32), lfr_splice_cache, splice_idx

    @staticmethod
    def compute_frame_num(
        sample_length: int, frame_sample_length: int, frame_shift_sample_length: int
    ) -> int:
        frame_num = int((sample_length - frame_sample_length) / frame_shift_sample_length + 1)
        return frame_num if frame_num >= 1 and sample_length >= frame_sample_length else 0

    def fbank(
        self, input: np.ndarray, input_lengths: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        if not self.ailia_audio:
            import kaldi_native_fbank as knf
            self.fbank_fn = knf.OnlineFbank(self.opts)
        else:
            self.fbank_fn = AiliaOnlineFbank(self.opts)
        batch_size = input.shape[0]
        if self.input_cache is None:
            self.input_cache = np.empty((batch_size, 0), dtype=np.float32)
        input = np.concatenate((self.input_cache, input), axis=1)
        frame_num = self.compute_frame_num(
            input.shape[-1], self.frame_sample_length, self.frame_shift_sample_length
        )
        # update self.in_cache
        self.input_cache = input[
            :, -(input.shape[-1] - frame_num * self.frame_shift_sample_length) :
        ]
        waveforms = np.empty(0, dtype=np.float32)
        feats_pad = np.empty(0, dtype=np.float32)
        feats_lens = np.empty(0, dtype=np.int32)
        if frame_num:
            waveforms = []
            feats = []
            feats_lens = []
            for i in range(batch_size):
                waveform = input[i]
                waveforms.append(
                    waveform[
                        : (
                            (frame_num - 1) * self.frame_shift_sample_length
                            + self.frame_sample_length
                        )
                    ]
                )
                waveform = waveform * (1 << 15)

                self.fbank_fn.accept_waveform(self.opts.frame_opts.samp_freq, waveform.tolist())
                frames = self.fbank_fn.num_frames_ready
                mat = np.empty([frames, self.opts.mel_opts.num_bins])
                for i in range(frames):
                    mat[i, :] = self.fbank_fn.get_frame(i)
                feat = mat.astype(np.float32)
                feat_len = np.array(mat.shape[0]).astype(np.int32)
                feats.append(feat)
                feats_lens.append(feat_len)

            waveforms = np.stack(waveforms)
            feats_lens = np.array(feats_lens)
            feats_pad = np.array(feats)
        self.fbanks = feats_pad
        self.fbanks_lens = copy.deepcopy(feats_lens)
        return waveforms, feats_pad, feats_lens

    def get_fbank(self) -> Tuple[np.ndarray, np.ndarray]:
        return self.fbanks, self.fbanks_lens

    def lfr_cmvn(
        self, input: np.ndarray, input_lengths: np.ndarray, is_final: bool = False
    ) -> Tuple[np.ndarray, np.ndarray, List[int]]:
        batch_size = input.shape[0]
        feats = []
        feats_lens = []
        lfr_splice_frame_idxs = []
        for i in range(batch_size):
            mat = input[i, : input_lengths[i], :]
            lfr_splice_frame_idx = -1
            if self.lfr_m != 1 or self.lfr_n != 1:
                # update self.lfr_splice_cache in self.apply_lfr
                mat, self.lfr_splice_cache[i], lfr_splice_frame_idx = self.apply_lfr(
                    mat, self.lfr_m, self.lfr_n, is_final
                )
            if self.cmvn_file is not None:
                mat = self.apply_cmvn(mat)
            feat_length = mat.shape[0]
            feats.append(mat)
            feats_lens.append(feat_length)
            lfr_splice_frame_idxs.append(lfr_splice_frame_idx)

        feats_lens = np.array(feats_lens)
        feats_pad = np.array(feats)
        return feats_pad, feats_lens, lfr_splice_frame_idxs

    def extract_fbank(
        self, input: np.ndarray, input_lengths: np.ndarray, is_final: bool = False
    ) -> Tuple[np.ndarray, np.ndarray]:
        batch_size = input.shape[0]
        assert (
            batch_size == 1
        ), "we support to extract feature online only when the batch size is equal to 1 now"
        waveforms, feats, feats_lengths = self.fbank(input, input_lengths)  # input shape: B T D
        if feats.shape[0]:
            self.waveforms = (
                waveforms
                if self.reserve_waveforms is None
                else np.concatenate((self.reserve_waveforms, waveforms), axis=1)
            )
            if not self.lfr_splice_cache:
                for i in range(batch_size):
                    self.lfr_splice_cache.append(
                        np.expand_dims(feats[i][0, :], axis=0).repeat((self.lfr_m - 1) // 2, axis=0)
                    )

            if feats_lengths[0] + self.lfr_splice_cache[0].shape[0] >= self.lfr_m:
                lfr_splice_cache_np = np.stack(self.lfr_splice_cache)  # B T D
                feats = np.concatenate((lfr_splice_cache_np, feats), axis=1)
                feats_lengths += lfr_splice_cache_np[0].shape[0]
                frame_from_waveforms = int(
                    (self.waveforms.shape[1] - self.frame_sample_length)
                    / self.frame_shift_sample_length
                    + 1
                )
                minus_frame = (self.lfr_m - 1) // 2 if self.reserve_waveforms is None else 0
                feats, feats_lengths, lfr_splice_frame_idxs = self.lfr_cmvn(
                    feats, feats_lengths, is_final
                )
                if self.lfr_m == 1:
                    self.reserve_waveforms = None
                else:
                    reserve_frame_idx = lfr_splice_frame_idxs[0] - minus_frame
                    # print('reserve_frame_idx:  ' + str(reserve_frame_idx))
                    # print('frame_frame:  ' + str(frame_from_waveforms))
                    self.reserve_waveforms = self.waveforms[
                        :,
                        reserve_frame_idx
                        * self.frame_shift_sample_length : frame_from_waveforms
                        * self.frame_shift_sample_length,
                    ]
                    sample_length = (
                        frame_from_waveforms - 1
                    ) * self.frame_shift_sample_length + self.frame_sample_length
                    self.waveforms = self.waveforms[:, :sample_length]
            else:
                # update self.reserve_waveforms and self.lfr_splice_cache
                self.reserve_waveforms = self.waveforms[
                    :, : -(self.frame_sample_length - self.frame_shift_sample_length)
                ]
                for i in range(batch_size):
                    self.lfr_splice_cache[i] = np.concatenate(
                        (self.lfr_splice_cache[i], feats[i]), axis=0
                    )
                return np.empty(0, dtype=np.float32), feats_lengths
        else:
            if is_final:
                self.waveforms = (
                    waveforms if self.reserve_waveforms is None else self.reserve_waveforms
                )
                feats = np.stack(self.lfr_splice_cache)
                feats_lengths = np.zeros(batch_size, dtype=np.int32) + feats.shape[1]
                feats, feats_lengths, _ = self.lfr_cmvn(feats, feats_lengths, is_final)
        if is_final:
            self.cache_reset()
        return feats, feats_lengths

    def get_waveforms(self):
        return self.waveforms

    def cache_reset(self):
        if not self.ailia_audio:
            import kaldi_native_fbank as knf
            self.fbank_fn = knf.OnlineFbank(self.opts)
        else:
            self.fbank_fn = AiliaOnlineFbank(self.opts)
        self.reserve_waveforms = None
        self.input_cache = None
        self.lfr_splice_cache = []
