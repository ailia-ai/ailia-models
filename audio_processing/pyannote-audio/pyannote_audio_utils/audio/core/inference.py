# MIT License
#
# Copyright (c) 2020- CNRS
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


import ailia
import warnings
from pathlib import Path
from typing import Callable, List, Optional, Text, Tuple, Union
from functools import cached_property
from dataclasses import dataclass
import numpy as np

from pyannote_audio_utils.core import Segment, SlidingWindow, SlidingWindowFeature
from pyannote_audio_utils.audio.core.io import AudioFile, Audio
from pyannote_audio_utils.audio.core.task import Resolution, Specifications, Problem
from pyannote_audio_utils.audio.utils.multi_task import map_with_specifications
from pyannote_audio_utils.audio.utils.powerset import Powerset


class BaseInference:
    pass

@dataclass
class Output:
    num_frames: int
    dimension: int
    frames: SlidingWindow

class Inference(BaseInference):
    """Inference

    Parameters
    ----------
    model : Model
        Model. Will be automatically set to eval() mode and moved to `device` when provided.
    window : {"sliding", "whole"}, optional
        Use a "sliding" window and aggregate the corresponding outputs (default)
        or just one (potentially long) window covering the "whole" file or chunk.
    duration : float, optional
        Chunk duration, in seconds. Defaults to duration used for training the model.
        Has no effect when `window` is "whole".
    step : float, optional
        Step between consecutive chunks, in seconds. Defaults to warm-up duration when
        greater than 0s, otherwise 10% of duration. Has no effect when `window` is "whole".
    pre_aggregation_hook : callable, optional
        When a callable is provided, it is applied to the model output, just before aggregation.
        Takes a (num_chunks, num_frames, dimension) numpy array as input and returns a modified
        (num_chunks, num_frames, other_dimension) numpy array passed to overlap-add aggregation.
    skip_aggregation : bool, optional
        Do not aggregate outputs when using "sliding" window. Defaults to False.
    skip_conversion: bool, optional
        In case a task has been trained with `powerset` mode, output is automatically
        converted to `multi-label`, unless `skip_conversion` is set to True.
    batch_size : int, optional
        Batch size. Larger values (should) make inference faster. Defaults to 32.
    device : torch.device, optional
        Device used for inference. Defaults to `model.device`.
        In case `device` and `model.device` are different, model is sent to device.
    use_auth_token : str, optional
        When loading a private huggingface.co model, set `use_auth_token`
        to True or to a string containing your hugginface.co authentication
        token that can be obtained by running `huggingface-cli login`
    """

    def __init__(
        self,
        model: Union[Text, Path],
        window: Text = "sliding",
        duration: float = None,
        step: float = None,
        pre_aggregation_hook: Callable[[np.ndarray], np.ndarray] = None,
        skip_aggregation: bool = False,
        skip_conversion: bool = False,
        batch_size: int = 32,
        use_auth_token: Union[Text, None] = None,
        args = None,
        seg_path = None,
    ):
        # ~~~~ model ~~~~~
        

        if args.onnx:
            #print("use onnx runtime")
            import onnxruntime
            model = onnxruntime.InferenceSession(model, providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
        else:
            #print("use ailia")
            model = ailia.Net(seg_path, weight=model, env_id=args.env_id)

        self.model = model
        self.args = args
    
        specifications = Specifications(problem=Problem.MONO_LABEL_CLASSIFICATION, resolution = Resolution.FRAME, classes=['speaker#1', 'speaker#2', 'speaker#3'])
        
        self.specifications = specifications
        self.audio = Audio(sample_rate=16000, mono="downmix")
        # ~~~~ sliding window ~~~~~

        if window not in ["sliding", "whole"]:
            raise ValueError('`window` must be "sliding" or "whole".')
        
        if window == "whole" and any(
            s.resolution == Resolution.FRAME for s in specifications
        ):
            warnings.warn(
                'Using "whole" `window` inference with a frame-based model might lead to bad results '
                'and huge memory consumption: it is recommended to set `window` to "sliding".'
            )
        self.window = window

        training_duration = next(iter(specifications)).duration
        duration = duration or training_duration
        if training_duration != duration:
            warnings.warn(
                f"Model was trained with {training_duration:g}s chunks, and you requested "
                f"{duration:g}s chunks for inference: this might lead to suboptimal results."
            )
        self.duration = duration

        # ~~~~ powerset to multilabel conversion ~~~~

        self.skip_conversion = skip_conversion

        conversion = list()
        for s in specifications:
            if s.powerset and not skip_conversion:
                c = Powerset(len(s.classes), s.powerset_max_classes)
            conversion.append(c)
        
        
        if isinstance(specifications, Specifications):
            self.conversion = conversion[0]

        # ~~~~ overlap-add aggregation ~~~~~

        self.skip_aggregation = skip_aggregation
        self.pre_aggregation_hook = pre_aggregation_hook

        
        self.warm_up = next(iter(specifications)).warm_up
        # Use that many seconds on the left- and rightmost parts of each chunk
        # to warm up the model. While the model does process those left- and right-most
        # parts, only the remaining central part of each chunk is used for aggregating
        # scores during inference.

        # step between consecutive chunks
        step = step or (
            0.1 * self.duration if self.warm_up[0] == 0.0 else self.warm_up[0]
        )
        if step > self.duration:
            raise ValueError(
                f"Step between consecutive chunks is set to {step:g}s, while chunks are "
                f"only {self.duration:g}s long, leading to gaps between consecutive chunks. "
                f"Either decrease step or increase duration."
            )
        self.step = step

        self.batch_size = batch_size

    def infer(self, chunks: np.ndarray) -> Union[np.ndarray, Tuple[np.ndarray]]:
        """Forward pass

        Takes care of sending chunks to right device and outputs back to CPU

        Parameters
        ----------
        chunks : (batch_size, num_channels, num_samples) torch.Tensor
            Batch of audio chunks.

        Returns
        -------
        outputs : (tuple of) (batch_size, ...) np.ndarray
            Model output.
        """
        chunks = chunks.astype(np.float32)
        if self.args.onnx:
            outputs = self.model.run(None, {"input": chunks})[0]
        else:
            outputs = self.model.predict([chunks])[0]
        
        
        def __convert(output: np.ndarray, conversion, **kwargs):
            return conversion(output)
        
        return map_with_specifications(self.specifications, __convert, outputs, self.conversion)
    
    @cached_property
    def example_output(self) -> Union[Output, Tuple[Output]]:
        """Example output"""
        example_input_array = np.random.randn(1, 1, self.audio.get_num_samples(self.specifications.duration)).astype(np.float32)
        
        example_outputs = self.infer(example_input_array)
            
        def __example_output(
            example_output: np.ndarray,
            specifications: Specifications = None,
        ) -> Output:
            _, num_frames, dimension = example_output.shape
            
            if specifications.resolution == Resolution.FRAME:
                frame_duration = specifications.duration / num_frames
                frames = SlidingWindow(step=frame_duration, duration=frame_duration)
            else:
                frames = None

            return Output(
                num_frames=num_frames,
                dimension=dimension,
                frames=frames,
            )

        return map_with_specifications(
            self.specifications, __example_output, example_outputs
        )
    
    def slide(
        self,
        waveform: np.ndarray,
        sample_rate: int,
        hook: Optional[Callable],
    ) -> Union[SlidingWindowFeature, Tuple[SlidingWindowFeature]]:
        """Slide model on a waveform

        Parameters
        ----------
        waveform: (num_channels, num_samples) torch.Tensor
            Waveform.
        sample_rate : int
            Sample rate.
        hook: Optional[Callable]
            When a callable is provided, it is called everytime a batch is
            processed with two keyword arguments:
            - `completed`: the number of chunks that have been processed so far
            - `total`: the total number of chunks

        Returns
        -------
        output : (tuple of) SlidingWindowFeature
            Model output. Shape is (num_chunks, dimension) for chunk-level tasks,
            and (num_frames, dimension) for frame-level tasks.
        """

        window_size: int = self.audio.get_num_samples(self.duration)
        step_size: int = round(self.step * sample_rate)
        
        _, num_samples = waveform.shape
        
        def __frames(
            example_output, specifications: Optional[Specifications] = None
        ) -> SlidingWindow:
            if specifications.resolution == Resolution.CHUNK:
                return SlidingWindow(start=0.0, duration=self.duration, step=self.step)
            
            return example_output.frames

        frames: Union[SlidingWindow, Tuple[SlidingWindow]] = map_with_specifications(
            self.specifications,
            __frames,
            self.example_output,
        )

        
        # prepare complete chunks
        def unfold_numpy(waveform, window_size, step_size):
            batch_size, waveform_size = waveform.shape
            num_windows = (waveform_size - window_size) // step_size + 1
            shape = (batch_size, num_windows, window_size)
            strides = (
                waveform.strides[0],
                step_size * waveform.strides[1],
                waveform.strides[1],
            )
            
            return np.lib.stride_tricks.as_strided(waveform, shape=shape, strides=strides)
        
        if num_samples >= window_size:
            chunks: np.ndarray = (unfold_numpy(waveform, window_size, step_size)).transpose(1, 0, 2)
            num_chunks = chunks.shape[0]
            
        else:
            num_chunks = 0
        
        # prepare last incomplete chunk
        
        has_last_chunk = (num_samples < window_size) or (num_samples - window_size) % step_size > 0
        
        if has_last_chunk:
            # pad last chunk with zeros
            last_chunk: np.ndarray = waveform[:, num_chunks * step_size :]
            _, last_window_size = last_chunk.shape
            last_pad = window_size - last_window_size
            last_chunk = np.pad(last_chunk, ((0, 0), (0, last_pad)))

        def __empty_list(**kwargs):
            return list()

        outputs: Union[
            List[np.ndarray], Tuple[List[np.ndarray]]
        ] = map_with_specifications(self.specifications, __empty_list)

        if hook is not None:
            hook(completed=0, total=num_chunks + has_last_chunk)

        def __append_batch(output, batch_output, **kwargs) -> None:
            output.append(batch_output)
            return

        
        # slide over audio chunks in batch
        for c in np.arange(0, num_chunks, self.batch_size):
            batch: np.ndarray = chunks[c : c + self.batch_size]
            batch_outputs: Union[np.ndarray, Tuple[np.ndarray]] = self.infer(batch)

            _ = map_with_specifications(
                self.specifications, __append_batch, outputs, batch_outputs
            )

            if hook is not None:
                hook(completed=c + self.batch_size, total=num_chunks + has_last_chunk)

        
        # process orphan last chunk
        if has_last_chunk:
            last_outputs = self.infer(last_chunk[None])

            _ = map_with_specifications(
                self.specifications, __append_batch, outputs, last_outputs
            )

            if hook is not None:
                hook(
                    completed=num_chunks + has_last_chunk,
                    total=num_chunks + has_last_chunk,
                )

        def __vstack(output: List[np.ndarray], **kwargs) -> np.ndarray:
            return np.vstack(output)

        outputs: Union[np.ndarray, Tuple[np.ndarray]] = map_with_specifications(
            self.specifications, __vstack, outputs
        )
        
        def __aggregate(
            outputs: np.ndarray,
            frames: SlidingWindow,
            specifications: Optional[Specifications] = None,
        ) -> SlidingWindowFeature:
            # skip aggregation when requested,
            # or when model outputs just one vector per chunk
            # or when model is permutation-invariant (and not post-processed)
            
            if (
                self.skip_aggregation
                or specifications.resolution == Resolution.CHUNK
                or (
                    specifications.permutation_invariant
                    and self.pre_aggregation_hook is None
                )
            ):
                frames = SlidingWindow(
                    start=0.0, duration=self.duration, step=self.step
                )
                
                return SlidingWindowFeature(outputs, frames)

        return map_with_specifications(
            self.specifications, __aggregate, outputs, frames
        )

    def __call__(
        self, file: AudioFile, hook: Optional[Callable] = None
    ) -> Union[
        Tuple[Union[SlidingWindowFeature, np.ndarray]],
        Union[SlidingWindowFeature, np.ndarray],
    ]:
        """Run inference on a whole file

        Parameters
        ----------
        file : AudioFile
            Audio file.
        hook : callable, optional
            When a callable is provided, it is called everytime a batch is processed
            with two keyword arguments:
            - `completed`: the number of chunks that have been processed so far
            - `total`: the total number of chunks

        Returns
        -------
        output : (tuple of) SlidingWindowFeature or np.ndarray
            Model output, as `SlidingWindowFeature` if `window` is set to "sliding"
            and `np.ndarray` if is set to "whole".

        """

        waveform, sample_rate = self.audio(file)
        
        if self.window == "sliding":
            return self.slide(waveform, sample_rate, hook=hook)
 

    @staticmethod
    def aggregate(
        scores: SlidingWindowFeature,
        frames: SlidingWindow = None,
        warm_up: Tuple[float, float] = (0.0, 0.0),
        epsilon: float = 1e-12,
        hamming: bool = False,
        missing: float = np.nan,
        skip_average: bool = False,
    ) -> SlidingWindowFeature:
        """Aggregation

        Parameters
        ----------
        scores : SlidingWindowFeature
            Raw (unaggregated) scores. Shape is (num_chunks, num_frames_per_chunk, num_classes).
        frames : SlidingWindow, optional
            Frames resolution. Defaults to estimate it automatically based on `scores` shape
            and chunk size. Providing the exact frame resolution (when known) leads to better
            temporal precision.
        warm_up : (float, float) tuple, optional
            Left/right warm up duration (in seconds).
        missing : float, optional
            Value used to replace missing (ie all NaNs) values.
        skip_average : bool, optional
            Skip final averaging step.

        Returns
        -------
        aggregated_scores : SlidingWindowFeature
            Aggregated scores. Shape is (num_frames, num_classes)
        """
        
        num_chunks, num_frames_per_chunk, num_classes = scores.data.shape

        chunks = scores.sliding_window
        if frames is None:
            duration = step = chunks.duration / num_frames_per_chunk
            frames = SlidingWindow(start=chunks.start, duration=duration, step=step)
        else:
            frames = SlidingWindow(
                start=chunks.start,
                duration=frames.duration,
                step=frames.step,
            )

        masks = 1 - np.isnan(scores)
        scores.data = np.nan_to_num(scores.data, copy=True, nan=0.0)

        # Hamming window used for overlap-add aggregation
        hamming_window = (
            np.hamming(num_frames_per_chunk).reshape(-1, 1)
            if hamming
            else np.ones((num_frames_per_chunk, 1))
        )

        # anything before warm_up_left (and after num_frames_per_chunk - warm_up_right)
        # will not be used in the final aggregation

        # warm-up windows used for overlap-add aggregation
        warm_up_window = np.ones((num_frames_per_chunk, 1))
        # anything before warm_up_left will not contribute to aggregation
        warm_up_left = round(
            warm_up[0] / scores.sliding_window.duration * num_frames_per_chunk
        )
        warm_up_window[:warm_up_left] = epsilon
        # anything after num_frames_per_chunk - warm_up_right either
        warm_up_right = round(
            warm_up[1] / scores.sliding_window.duration * num_frames_per_chunk
        )
        warm_up_window[num_frames_per_chunk - warm_up_right :] = epsilon

        # aggregated_output[i] will be used to store the sum of all predictions
        # for frame #i
        num_frames = (
            frames.closest_frame(
                scores.sliding_window.start
                + scores.sliding_window.duration
                + (num_chunks - 1) * scores.sliding_window.step
            )
            + 1
        )
        aggregated_output: np.ndarray = np.zeros(
            (num_frames, num_classes), dtype=np.float32
        )

        # overlapping_chunk_count[i] will be used to store the number of chunks
        # that contributed to frame #i
        overlapping_chunk_count: np.ndarray = np.zeros(
            (num_frames, num_classes), dtype=np.float32
        )

        # aggregated_mask[i] will be used to indicate whether
        # at least one non-NAN frame contributed to frame #i
        aggregated_mask: np.ndarray = np.zeros(
            (num_frames, num_classes), dtype=np.float32
        )
        
        # loop on the scores of sliding chunks
        for (chunk, score), (_, mask) in zip(scores, masks):
            # chunk ~ Segment
            # score ~ (num_frames_per_chunk, num_classes)-shaped np.ndarray
            # mask ~ (num_frames_per_chunk, num_classes)-shaped np.ndarray
            
            start_frame = frames.closest_frame(chunk.start)
            aggregated_output[start_frame : start_frame + num_frames_per_chunk] += (
                score * mask * hamming_window * warm_up_window
            )

            overlapping_chunk_count[
                start_frame : start_frame + num_frames_per_chunk
            ] += (mask * hamming_window * warm_up_window)

            aggregated_mask[
                start_frame : start_frame + num_frames_per_chunk
            ] = np.maximum(
                aggregated_mask[start_frame : start_frame + num_frames_per_chunk],
                mask,
            )
        
        if skip_average:
            average = aggregated_output
        else:
            average = aggregated_output / np.maximum(overlapping_chunk_count, epsilon)

        average[aggregated_mask == 0.0] = missing

        return SlidingWindowFeature(average, frames)

    @staticmethod
    def trim(
        scores: SlidingWindowFeature,
        warm_up: Tuple[float, float] = (0.1, 0.1),
    ) -> SlidingWindowFeature:
        """Trim left and right warm-up regions

        Parameters
        ----------
        scores : SlidingWindowFeature
            (num_chunks, num_frames, num_classes)-shaped scores.
        warm_up : (float, float) tuple
            Left/right warm up ratio of chunk duration.
            Defaults to (0.1, 0.1), i.e. 10% on both sides.

        Returns
        -------
        trimmed : SlidingWindowFeature
            (num_chunks, trimmed_num_frames, num_speakers)-shaped scores
        """
        

        assert (
            scores.data.ndim == 3
        ), "Inference.trim expects (num_chunks, num_frames, num_classes)-shaped `scores`"
        _, num_frames, _ = scores.data.shape

        chunks = scores.sliding_window

        num_frames_left = round(num_frames * warm_up[0])
        num_frames_right = round(num_frames * warm_up[1])

        num_frames_step = round(num_frames * chunks.step / chunks.duration)
        if num_frames - num_frames_left - num_frames_right < num_frames_step:
            warnings.warn(
                f"Total `warm_up` is so large ({sum(warm_up) * 100:g}% of each chunk) "
                f"that resulting trimmed scores does not cover a whole step ({chunks.step:g}s)"
            )
        new_data = scores.data[:, num_frames_left : num_frames - num_frames_right]

        new_chunks = SlidingWindow(
            start=chunks.start + warm_up[0] * chunks.duration,
            step=chunks.step,
            duration=(1 - warm_up[0] - warm_up[1]) * chunks.duration,
        )

        return SlidingWindowFeature(new_data, new_chunks)
