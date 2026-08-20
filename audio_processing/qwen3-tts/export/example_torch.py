"""Run the PyTorch reference on the same inputs as ../qwen3-tts.py and time it.

Tested versions:
    torch 2.10.0
    transformers 4.57.3
    qwen-tts 0.1.1

Usage:
    python3 example_torch.py --parameter_num 0.6B
    python3 example_torch.py --parameter_num 0.6B --device cuda
    python3 example_torch.py --parameter_num 0.6B --device both

This is the reference implementation rather than the exported graphs: it goes
through qwen-tts' own Qwen3TTSModel.generate_voice_clone. The reference audio,
reference text, prompt text and every sampling default are the ones
../qwen3-tts.py uses, and the timing table is printed in the same shape as that
sample's -b output, so the two can be read side by side.

The rows are the parts the export splits the model into, so they mean the same
thing in both tables: the two decode loop modules are timed with forward hooks, and
the speech tokenizer's encode and decode with wrappers around those methods, since
the reference calls them as methods rather than as modules.

On CUDA every measurement synchronises, because the launches are asynchronous and
the numbers would otherwise be meaningless. That costs a little, so the total is
slightly pessimistic there.
"""

import argparse
import os
import sys
import time

import numpy as np
import soundfile as sf
import torch

# export_onnx stubs pysox out before anything imports qwen_tts, see its comment
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import export_onnx as ex  # noqa: E402

from qwen_tts.inference.qwen3_tts_model import Qwen3TTSModel  # noqa: E402

SAMPLE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")

# the defaults of ../qwen3-tts.py, so both samples synthesize the same thing
TEXT_STR = "Good one. Okay, fine, I'm just gonna leave this sock monkey here. Goodbye."
INPUT_WAV_PATH = os.path.join(SAMPLE_DIR, "clone_2.wav")
INPUT_TEXT_STR = ("Okay. Yeah. I resent you. I love you. I respect you. But you know "
                  "what? You blew it! And thanks to you.")


class Timer:
    """Wall time and call counts per label, printed like the sample's -b table.

    A label can cover more than one callable: the talker's output head is a module
    of its own here but part of the talker's ONNX, so it adds its time to the same
    row without counting as another call.
    """

    def __init__(self, sync):
        self.sync = sync
        self.records = {}
        self.undo = []

    def now(self):
        if self.sync:
            torch.cuda.synchronize()
        return time.perf_counter()

    def add(self, label, elapsed, counts):
        calls, total = self.records.get(label, (0, 0.0))
        self.records[label] = (calls + int(counts), total + elapsed)

    def hook(self, label, module, counts=True):
        """Time a module's forward."""
        starts = []
        handles = [
            module.register_forward_pre_hook(
                lambda mod, args: starts.append(self.now())),
            module.register_forward_hook(
                lambda mod, args, out: self.add(label, self.now() - starts.pop(),
                                                counts)),
        ]
        self.undo.append(lambda: [handle.remove() for handle in handles])

    def wrap(self, label, owner, name, counts=True):
        """Time a method the reference calls directly rather than through __call__."""
        original = getattr(owner, name)

        def timed(*args, **kwargs):
            start = self.now()
            try:
                return original(*args, **kwargs)
            finally:
                self.add(label, self.now() - start, counts)

        setattr(owner, name, timed)
        self.undo.append(lambda: setattr(owner, name, original))

    def remove(self):
        for undo in reversed(self.undo):
            undo()
        self.undo = []

    def report(self, order):
        for label in order:
            if label not in self.records:
                continue
            calls, total = self.records[label]
            total_ms = total * 1000
            if calls <= 1:
                print(f"\t{label} processing time {total_ms:.0f} ms")
            else:
                print(f"\t{label} processing time {total_ms:.0f} ms "
                      f"({calls} calls, {total_ms / calls:.1f} ms/call)")


ROWS = ["encoder", "prompt", "talker", "code_predictor", "decoder"]


def install(timer, model):
    """Time the same five parts the export splits the model into."""
    talker = model.talker
    # the reference audio: codec tokens from the speech tokenizer and the speaker
    # embedding from the ECAPA-TDNN, one ONNX in the export
    timer.wrap("encoder", model.speech_tokenizer, "encode")
    timer.hook("encoder", model.speaker_encoder, counts=False)
    timer.hook("prompt", talker.text_projection)
    timer.hook("talker", talker.model)
    timer.hook("talker", talker.codec_head, counts=False)
    timer.hook("code_predictor", talker.code_predictor)
    timer.wrap("decoder", model.speech_tokenizer, "decode")


def run(args, device):
    print(f"[{device}] loading {args.model_dir} as {args.dtype} ...")
    load = time.perf_counter()
    tts = Qwen3TTSModel.from_pretrained(
        args.model_dir,
        dtype=getattr(torch, args.dtype),
        attn_implementation=args.attn_implementation,
        # the speech tokenizer is loaded separately and is not a submodule, so it
        # has to be placed by the loader rather than by a .to() afterwards
        device_map=device,
    )
    tts.model.eval()
    print(f"[{device}] loaded in {time.perf_counter() - load:.1f} s")

    if args.seed is not None:
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)

    timer = Timer(sync=device.startswith("cuda"))
    install(timer, tts.model)
    start = timer.now()
    waveforms, sample_rate = tts.generate_voice_clone(
        text=args.input,
        language=args.language,
        ref_audio=args.ref_audio,
        ref_text=args.ref_text,
        temperature=args.temperature,
        top_k=args.top_k,
        repetition_penalty=args.repetition_penalty,
        subtalker_temperature=args.subtalker_temperature,
        subtalker_top_k=args.subtalker_top_k,
    )
    total = (timer.now() - start) * 1000
    timer.remove()

    waveform = np.asarray(waveforms[0], dtype=np.float32).squeeze()
    timer.report(ROWS)
    print(f"\ttotal processing time {total:.0f} ms")
    # the talker runs once for the prompt and once per generated token, which is
    # what the rest of the table is per, so say how many that was
    talker_calls = timer.records.get("talker", (0, 0))[0]
    seconds = len(waveform) / sample_rate
    print(f"\t{max(talker_calls - 1, 0)} tokens, {seconds:.2f} s of audio at "
          f"{sample_rate} Hz, {total / 1000 / seconds:.2f}x realtime")

    path = args.savepath
    if args.device == "both":
        stem, extension = os.path.splitext(path)
        path = f"{stem}_{device}{extension}"
    sf.write(path, waveform, sample_rate)
    print(f"[{device}] saved as {path}")


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("-p", "--parameter_num", default="0.6B",
                        choices=sorted(ex.MODEL_ID.keys()))
    parser.add_argument("--model_dir", default=None,
                        help="local snapshot of the checkpoint (downloaded from the "
                             "Hub if omitted)")
    parser.add_argument("-d", "--device", default="cpu",
                        choices=["cpu", "cuda", "both"],
                        help="both runs the same synthesis on each in turn")
    parser.add_argument("--dtype", default="float32",
                        choices=["float32", "bfloat16", "float16"],
                        help="the exported ONNX are float32, so that is the "
                             "comparable one")
    parser.add_argument("--attn_implementation", default="eager",
                        choices=["eager", "sdpa"],
                        help="eager is what the export traces, sdpa is faster")
    parser.add_argument("--threads", type=int, default=None,
                        help="torch CPU threads (default: every core)")
    parser.add_argument("-i", "--input", default=TEXT_STR, help="text to synthesize")
    parser.add_argument("--ref_audio", default=INPUT_WAV_PATH)
    parser.add_argument("--ref_text", default=INPUT_TEXT_STR)
    parser.add_argument("--language", default="Auto")
    parser.add_argument("--temperature", type=float, default=0.9)
    parser.add_argument("--top_k", type=int, default=50)
    parser.add_argument("--repetition_penalty", type=float, default=1.05)
    parser.add_argument("--subtalker_temperature", type=float, default=0.9)
    parser.add_argument("--subtalker_top_k", type=int, default=50)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("-s", "--savepath", default="output_torch.wav")
    args = parser.parse_args()

    if args.model_dir is None:
        from huggingface_hub import snapshot_download

        args.model_dir = snapshot_download(ex.MODEL_ID[args.parameter_num])
    if args.threads is not None:
        torch.set_num_threads(args.threads)

    devices = ["cpu", "cuda"] if args.device == "both" else [args.device]
    for device in devices:
        if device == "cuda" and not torch.cuda.is_available():
            print("cuda is not available to this torch build, skipping")
            continue
        run(args, device)


if __name__ == "__main__":
    main()
