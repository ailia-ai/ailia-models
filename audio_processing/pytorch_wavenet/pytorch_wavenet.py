import time
import sys
import numpy as np

import soundfile as sf

import ailia

# import original moduls
sys.path.append("../../util")
from arg_utils import get_base_parser, update_parser, get_savepath  # noqa: E402
from math_utils import softmax  # noqa: E402
from model_utils import check_and_download_models  # noqa: E402

# logger
from logging import getLogger  # noqa: E402

logger = getLogger(__name__)


# ======================
# PARAMETERS
# ======================
WEIGHT_PATH = "wavenet_pytorch_op_17.onnx"
MODEL_PATH = "wavenet_pytorch_op_17.onnx.prototxt"
REMOTE_PATH = "https://storage.googleapis.com/ailia-models/pytorch-wavenet/"

SAMPLE_WAVE_PATH = "first_sample.wav"
SAVE_WAVE_PATH = "output.wav"

# ======================
# Arguemnt Parser Config
# ======================
parser = get_base_parser(
    "pytorch_wavenet", SAMPLE_WAVE_PATH, SAVE_WAVE_PATH, input_ftype="audio"
)
parser.add_argument("--num_samples", type=float, default=16000, help="num_samples")
parser.add_argument("--temperature", type=float, default=1.0, help="temperature")
parser.add_argument(
    "--no_input", action="store_true", default=False, help="not sample input"
)
parser.add_argument(
    "--onnx", action="store_true", default=False, help="Use onnxruntime"
)
parser.add_argument(
    '--seed', default=1000, type=int,
    help='random seed'
)
args = update_parser(parser)


# ======================
# Main function
# ======================
def get_model():
    if not args.onnx:
        logger.info("Use ailia")
        net = ailia.Net(MODEL_PATH, WEIGHT_PATH, env_id=args.env_id, memory_mode=args.memory_mode)
    else:
        logger.info("Use onnxruntime")
        import onnxruntime

        net = onnxruntime.InferenceSession(WEIGHT_PATH)
    return net


def output_wav(filename, rate, data):
    sf.write(filename, data, rate)


def generate_wave(net, num_samples, first_samples=None, temperature=1.0):
    tic = time.time()
    progress_interval = 1000

    classes = 256

    if first_samples is None:
        first_samples = np.zeros(1, dtype=int) + (classes // 2)

    num_given_samples = first_samples.shape[0]
    total_samples = num_given_samples + num_samples

    input = np.zeros((1, classes, 1), dtype=np.float32)
    input[:, first_samples[0], :] = 1

    # prepare queues
    kernel_size = 2
    blocks = 3
    layers = 10
    shapes = []
    for _ in range(blocks):
        new_dilation = 1
        for _ in range(layers):
            shapes.append((32, (kernel_size - 1) * new_dilation + 1))
            new_dilation *= 2
    dilated_queues = [np.zeros(shape, dtype=np.float32) for shape in shapes]

    # fill queues with given samples
    for i in range(num_given_samples - 1):
        _, dilated_queues = _inference(net, input, dilated_queues)

        input.fill(0)
        input[:, first_samples[i + 1], :] = 1

        if i % progress_interval == 0:
            print(str(100 * i // total_samples) + "% generated")

    # generate new samples
    generated = np.array([])
    for i in range(int(num_samples)):
        x, dilated_queues = _inference(net, input, dilated_queues)
        x = x.squeeze()

        if temperature > 0:
            x /= temperature
            prob = softmax(x, axis=0)
            x = np.random.choice(classes, p=prob)
        else:
            x = np.argmax(x)

        # set new input
        input.fill(0)
        input[:, x, :] = 1

        x = np.array([x])
        generated = np.append(generated, x)

        # progress feedback
        if i % progress_interval == 0:
            print(str(100 * (i + num_given_samples) // total_samples) + "% generated")

    generated = (generated / classes) * 2.0 - 1
    mu_gen = _mu_law_expansion(generated, classes)

    toc = time.time()
    print("ailia processing does take {} seconds".format(str(toc - tic)))

    return mu_gen


def _inference(net, input, dilated_queues):
    if not args.onnx:
        output = net.run([input, *dilated_queues])
    else:
        output = net.run(
            None,
            {
                "input": input,
                **{
                    "dilated_queues_%d" % (i + 1): q
                    for i, q in enumerate(dilated_queues)
                },
            },
        )
    x = output[0]
    dilated_queues = output[1:]

    return x, dilated_queues


def _mu_law_expansion(data, mu):
    s = np.sign(data) * (np.exp(np.abs(data) * np.log(mu + 1)) - 1) / mu
    return s


def main():
    # model files check and download
    check_and_download_models(WEIGHT_PATH, MODEL_PATH, REMOTE_PATH)

    # create instance
    net = get_model()

    # generate sample
    if args.no_input:
        sample = None
    else:
        input_file = args.input[0]
        wav, _ = sf.read(input_file)
        sample = (wav * 255).astype(int)

    num_samples = args.num_samples
    temperature = args.temperature

    # generate wave
    generated = generate_wave(
        net, num_samples, first_samples=sample, temperature=temperature
    )

    # output wav
    savepath = get_savepath(
        args.savepath, args.input[0] if not args.no_input else "dummy.wav", ext=".wav"
    )
    logger.info(f"saved at : {savepath}")
    output_wav(savepath, 16000, generated)

    logger.info("Script finished successfully.")


if __name__ == "__main__":
    np.random.seed(args.seed)
    main()
