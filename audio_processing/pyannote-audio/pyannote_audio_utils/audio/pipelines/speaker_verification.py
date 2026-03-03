# MIT License
#
# Copyright (c) 2021 CNRS
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.


from functools import cached_property
from typing import Optional, Text, Union, Mapping

import numpy as np
import ailia

from pyannote_audio_utils.audio.pipelines.utils.kaldifeat import compute_fbank_feats
from pyannote_audio_utils.audio.core.inference import BaseInference

PipelineModel = Union[Text, Mapping]

class ONNXWeSpeakerPretrainedSpeakerEmbedding(BaseInference):
    """Pretrained WeSpeaker speaker embedding

    Parameters
    ----------
    embedding : str
        Path to WeSpeaker pretrained speaker embedding
    device : torch.device, optional
        Device

    Usage
    -----
    >>> get_embedding = ONNXWeSpeakerPretrainedSpeakerEmbedding("hbredin/wespeaker-voxceleb-resnet34-LM")
    >>> assert waveforms.ndim == 3
    >>> batch_size, num_channels, num_samples = waveforms.shape
    >>> assert num_channels == 1
    >>> embeddings = get_embedding(waveforms)
    >>> assert embeddings.ndim == 2
    >>> assert embeddings.shape[0] == batch_size

    >>> assert binary_masks.ndim == 1
    >>> assert binary_masks.shape[0] == batch_size
    >>> embeddings = get_embedding(waveforms, masks=binary_masks)
    """

    def __init__(
        self,
        embedding: Text = "hbredin/wespeaker-voxceleb-resnet34-LM",
        # device: Optional[torch.device] = None,
        args = None,
        emb_path = None
    ):
        # if not ONNX_IS_AVAILABLE:
            # raise ImportError(
                # f"'onnxruntime' must be installed to use '{embedding}' embeddings."
            # )

        super().__init__()

        # if not Path(embedding).exists():
            # try:
                # embedding = hf_hub_download(
                    # repo_id=embedding,
                    # filename="speaker-embedding.onnx",
                # )
            # except RepositoryNotFoundError:
                # raise ValueError(
                    # f"Could not find '{embedding}' on huggingface.co nor on local disk."
                # )

        self.embedding = embedding
        
        if args.onnx:
            import onnxruntime as ort
            #print("use onnx runtime")
            providers = ["CPUExecutionProvider", ("CUDAExecutionProvider",{"cudnn_conv_algo_search": "DEFAULT"})]

            sess_options = ort.SessionOptions()
            sess_options.inter_op_num_threads = 1
            sess_options.intra_op_num_threads = 1
            self.session_ = ort.InferenceSession(
                embedding, sess_options=sess_options, providers=providers
            )
        else:
            #print("use ailia")

            self.session_ = ailia.Net(emb_path, weight=embedding, env_id=args.env_id)
        
        self.args = args
 
    @cached_property
    def sample_rate(self) -> int:
        return 16000

    @cached_property
    def dimension(self) -> int:
        dummy_waveforms = np.random.rand(1, 1, 16000)
        features = self.compute_fbank(dummy_waveforms)

        if self.args.onnx:
            embeddings = self.session_.run(output_names=["embs"], input_feed={"feats": features}
            )[0]
        else:
            embeddings = self.session_.predict([features])[0]

        _, dimension = embeddings.shape
        return dimension

    @cached_property
    def metric(self) -> str:
        return "cosine"

    @cached_property
    def min_num_samples(self) -> int:
        lower, upper = 2, round(0.5 * self.sample_rate)
        middle = (lower + upper) // 2
        while lower + 1 < upper:
            try:
                features = self.compute_fbank(np.random.randn(1, 1, middle))

            except AssertionError:
                lower = middle
                middle = (lower + upper) // 2
                continue

            if self.args.onnx:
                embeddings = self.session_.run(output_names=["embs"], input_feed={"feats": features})[0]
            else:
                embeddings = self.session_.predict([features])[0]

            if np.any(np.isnan(embeddings)):
                lower = middle
            else:
                upper = middle
            middle = (lower + upper) // 2

        return upper

    @cached_property
    def min_num_frames(self) -> int:
        return self.compute_fbank(np.random.randn(1, 1, self.min_num_samples)).shape[1]

    def compute_fbank(
        self,
        waveforms: np.ndarray,
        num_mel_bins: int = 80,
        frame_length: int = 25,
        frame_shift: int = 10,
        dither: float = 0.0,
    ) -> np.ndarray:
        """Extract fbank features

        Parameters
        ----------
        waveforms : (batch_size, num_channels, num_samples)

        Returns
        -------
        fbank : (batch_size, num_frames, num_mel_bins)

        Source: https://github.com/wenet-e2e/wespeaker/blob/45941e7cba2c3ea99e232d02bedf617fc71b0dad/wespeaker/bin/infer_onnx.py#L30C1-L50
        """
        
        waveforms = waveforms * (1 << 15)

        ### ここで少しずれる ###
        features_numpy = np.stack([compute_fbank_feats(
            waveform=waveform[0],
            num_mel_bins=num_mel_bins,
            frame_length=frame_length,
            frame_shift=frame_shift,
            dither=dither,
            sample_frequency=self.sample_rate,
            window_type="hamming",
            use_energy=False,
        )for waveform in waveforms])
        ### ここで少しずれる ###

        features = features_numpy.astype(np.float32)
 
        return features - np.mean(features, axis=1, keepdims=True)

    def __call__(
        self, waveforms: np.ndarray, masks: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """

        Parameters
        ----------
        waveforms : (batch_size, num_channels, num_samples)
            Only num_channels == 1 is supported.
        masks : (batch_size, num_samples), optional

        Returns
        -------
        embeddings : (batch_size, dimension)

        """

        batch_size, num_channels, num_samples = waveforms.shape
        assert num_channels == 1
        
        features = self.compute_fbank(waveforms)
        _, num_frames, _ = features.shape

        batch_size_masks, _ = masks.shape
        assert batch_size == batch_size_masks
        
        def interpolate_numpy(input_array, size):
            output_array = np.zeros((input_array.shape[0],size))
            
            for i in range(output_array.shape[0]):
                for j in range(output_array.shape[1]):
                    ii = int(np.floor(i * input_array.shape[0] / output_array.shape[0]))
                    jj = int(np.floor(j * input_array.shape[1] / output_array.shape[1]))
                    output_array[i, j] = input_array[ii, jj]
            return output_array

        imasks = interpolate_numpy(masks,size=num_frames)
        imasks = imasks > 0.5
        
        embeddings = np.nan * np.zeros((batch_size, self.dimension))

        for f, (feature, imask) in enumerate(zip(features, imasks)):
            masked_feature = feature[imask]
            if masked_feature.shape[0] < self.min_num_frames:
                continue
            
            if self.args.onnx:
                embeddings[f] = self.session_.run(output_names=["embs"],input_feed={"feats": masked_feature[None]},)[0][0]
            else:
                embeddings[f] = self.session_.predict([masked_feature[None]])[0][0]

        return embeddings

