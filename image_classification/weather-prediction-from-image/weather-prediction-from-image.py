import sys
import time

import cv2
import numpy as np
from PIL import Image

import ailia
import weather_prediction_from_image_utils

# Import original modules.
sys.path.append("../../util")
from arg_utils import get_base_parser, update_parser, get_savepath  # noqa: E402
from model_utils import check_and_download_models  # NOQA: E402
from webcamera_utils import get_capture  # NOQA: E402

# Logger
from logging import getLogger  # noqa: E402

logger = getLogger(__name__)

# ======================
# Parameters
# ======================
WEIGHT_PATH = "weather-prediction-from-image_trainedModelE20.onnx"
MODEL_PATH = "weather-prediction-from-image_trainedModelE20.onnx.prototxt"
REMOTE_PATH = (
    "https://storage.googleapis.com/ailia-models/weather-prediction-from-image/"
)
IMAGE_PATH = "data/img/3020580824.jpg"
SAVE_IMAGE_PATH = "output.png"
CROPPING_SIZE = 100
WEATHER_CLASSES = ["Cloudy", "Sunny", "Rainy", "Snowy", "Foggy"]

# ======================
# Argument Parser Config
# ======================
parser = get_base_parser(
    "Weather Prediction From Image - (Warmth Of Image)", IMAGE_PATH, SAVE_IMAGE_PATH
)
parser.add_argument(
    "--savepath",
    default="img",
    type=str,
    metavar="PATH",
    help="Path to output directory",
)

args = update_parser(parser)


# ======================
# Main functions
# ======================
def _make_dataset(img):
    input_np = weather_prediction_from_image_utils.prepare_data_set(img, CROPPING_SIZE)
    return np.expand_dims(input_np, axis=0)


def _prepare_data(args, image_path=None, frame=None):
    if args.video is not None:
        return _make_dataset(Image.fromarray(frame[:, :, ::-1]))
    else:
        image = Image.open(image_path)
        return _make_dataset(image), image


def _initialize_net(args):
    return ailia.Net(MODEL_PATH, WEIGHT_PATH, env_id=args.env_id, memory_mode=args.memory_mode)


def _infer(img, net):
    return net.predict(img)


def _estimate(img, model):
    pred = _infer(img, model)
    return WEATHER_CLASSES[np.argmax(pred)], np.max(pred)


def _output_text(weather, prob):
    return f"{weather} {round(prob*100, 1)}%"


def recognize_from_image():
    # Input image loop
    for image_path in args.input:
        logger.info(image_path)

        # Prepare input data.
        dataset, image = _prepare_data(args, image_path=image_path)

        # Initialize net.
        net = _initialize_net(args)

        # Inference
        logger.info("Start inference...")
        if args.benchmark:
            logger.info("BENCHMARK mode")
            for i in range(5):
                start = int(round(time.time() * 1000))
                weather, prob = _estimate(dataset, net)
                end = int(round(time.time() * 1000))
                logger.info(f"\tailia processing time {end - start} ms")

        # show result
        weather, prob = _estimate(dataset, net)
        logger.info(f"result : {weather} {prob}")
        filepath = get_savepath(args.savepath, image_path, ext=".png")
        weather_prediction_from_image_utils.save_image(
            _output_text(weather, prob), image, filepath
        )
        logger.info(f"saved at : {filepath}")

    logger.info("Script finished successfully.")


def recognize_from_video():
    # Initialize net.
    net = _initialize_net(args)

    capture = get_capture(args.video)
    
    frame_shown = False
    while True:
        ret, frame = capture.read()
        if (cv2.waitKey(1) & 0xFF == ord("q")) or not ret:
            break
        if frame_shown and cv2.getWindowProperty('frame', cv2.WND_PROP_VISIBLE) == 0:
            break

        # Prepare input data.
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        dataset = _prepare_data(args, frame=frame)

        # Inference
        weather, prob = _estimate(dataset, net)

        # Postprocessing
        cv2.imshow(
            "frame",
            weather_prediction_from_image_utils.annotate_video(
                frame,
                _output_text(weather, prob),
            ),
        )
        frame_shown = True

    capture.release()
    cv2.destroyAllWindows()
    logger.info("Script finished successfully.")


def main():
    # Check model files and download.
    check_and_download_models(
        WEIGHT_PATH,
        MODEL_PATH,
        REMOTE_PATH,
    )

    if args.video is not None:
        # Video mode
        recognize_from_video()
    else:
        # Image mode
        recognize_from_image()


if __name__ == "__main__":
    main()
