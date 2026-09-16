import argparse
from pathlib import Path
import sys
import time

import cv2
import numpy as np
from PIL import Image

import ailia
from psgan.postprocess import PostProcess
from psgan.preprocess import PreProcess

# Import original modules
sys.path.append("../../util")
from arg_utils import get_base_parser, update_parser, get_savepath  # noqa: E402
from model_utils import check_and_download_models  # NOQA: E402
from webcamera_utils import adjust_frame_size, get_capture, cut_max_square  # NOQA: E402

# Logger
from logging import getLogger  # noqa: E402

logger = getLogger(__name__)

# ======================
# Parameters
# ======================
WEIGHT_PATH = "psgan.onnx"
MODEL_PATH = "psgan.onnx.prototxt"
FACE_PARSER_WEIGHT_PATH = "face_parser.onnx"
FACE_PARSER_MODEL_PATH = "face_parser.onnx.prototxt"
FACE_ALIGNMENT_WEIGHT_PATH = "../../face_recognition/face_alignment/2DFAN-4.onnx"
FACE_ALIGNMENT_MODEL_PATH = (
    "../../face_recognition/face_alignment/2DFAN-4.onnx.prototxt"
)
FACE_DETECTOR_WEIGHT_PATH = "../../face_detection/blazeface/blazeface.onnx"
FACE_DETECTOR_MODEL_PATH = "../../face_detection/blazeface/blazeface.onnx.prototxt"
REMOTE_PATH = "https://storage.googleapis.com/ailia-models/psgan/"
FACE_ALIGNMENT_REMOTE_PATH = (
    "https://storage.googleapis.com/ailia-models/face_alignment/"
)
FACE_DETECTOR_REMOTE_PATH = "https://storage.googleapis.com/ailia-models/blazeface/"
face_parser_path = [FACE_PARSER_MODEL_PATH, FACE_PARSER_WEIGHT_PATH]
face_alignment_path = [FACE_ALIGNMENT_MODEL_PATH, FACE_ALIGNMENT_WEIGHT_PATH]
face_detector_path = [FACE_DETECTOR_MODEL_PATH, FACE_DETECTOR_WEIGHT_PATH]

SOURCE_IMAGE_PATH = "images/non-makeup/xfsy_0106.png"
REFERENCE_IMAGE_PATH = "images/makeup"
SAVE_IMAGE_PATH = "output.png"
IMAGE_HEIGHT = 361
IMAGE_WIDTH = 361

# ======================
# Argument Parser Config
# ======================
parser = get_base_parser(
    "PSGAN: Pose and Expression Robust Spatial-Aware GAN for Customizable Makeup Transfer",
    SOURCE_IMAGE_PATH,
    SAVE_IMAGE_PATH,
)
parser.add_argument(
    "opts",
    help="Modify config options using the command-line.",
    default=None,
    nargs=argparse.REMAINDER,
)
parser.add_argument(
    "--reference_dir", default=REFERENCE_IMAGE_PATH, help="Path to reference images"
)
parser.add_argument(
    "--onnx",
    action="store_true",
    help="Execute Onnx Runtime mode.",
)
parser.add_argument(
    "--use_dlib",
    action="store_true",
    help="Use dlib models for inference.",
)

args = update_parser(parser)

# ======================
# Main functions
# ======================


def _prepare_data(args, preprocess, image_type, frame=None):
    real = None
    mask = None
    diff = None
    crop_face = None
    if image_type == "source":
        image = Image.open(args.input[0]).convert("RGB")
        image_input, _, crop_face = preprocess(image)
    elif image_type == "reference":
        paths = list(Path(args.reference_dir).glob("*"))
        image = Image.open(paths[0]).convert("RGB")
        image_input, _, _ = preprocess(image)
    elif image_type == "frame":
        image = frame
        image_input, _, crop_face = preprocess(image)
    else:
        raise ValueError

    if image_input:
        real, mask, diff = image_input

    return image, real, mask, diff, crop_face


def _initialize_net(args):
    if not args.onnx:
        net = ailia.Net(MODEL_PATH, WEIGHT_PATH, env_id=args.env_id, memory_mode=args.memory_mode)
    else:
        import onnxruntime

        net = onnxruntime.InferenceSession(WEIGHT_PATH)

    return net


def _transfer(real_A, real_B, mask_A, mask_B, diff_A, diff_B, net):
    if not args.onnx:
        return net.predict([real_A, real_B, mask_A, mask_B, diff_A, diff_B])
    else:
        inputs = {
            net.get_inputs()[0].name: real_A,
            net.get_inputs()[1].name: real_B,
            net.get_inputs()[2].name: mask_A,
            net.get_inputs()[3].name: mask_B,
            net.get_inputs()[4].name: diff_A,
            net.get_inputs()[5].name: diff_B,
        }
        return net.run(None, inputs)


def _postprocessing(out, source, crop_face, postprocess):
    out_sqz0 = out.squeeze(0)
    min_, max_ = out_sqz0.min(), out_sqz0.max()
    out_sqz0_nrm = (out_sqz0 - min_) / (max_ - min_ + 1e-5) * 255
    pil_img = Image.fromarray(np.transpose(out_sqz0_nrm, (1, 2, 0)).astype(np.uint8))
    source_crop = source.crop(
        (crop_face["left"], crop_face["top"], crop_face["right"], crop_face["bottom"])
    )
    return postprocess(source_crop, pil_img)


def transfer_to_image():
    # Prepare input data
    preprocess = PreProcess(
        args, face_parser_path, face_alignment_path, face_detector_path
    )
    source, real_A, mask_A, diff_A, crop_face = _prepare_data(
        args, preprocess, "source"
    )
    _, real_B, mask_B, diff_B, _ = _prepare_data(args, preprocess, "reference")

    # Net initialize
    net = _initialize_net(args)

    # Inference
    logger.info("Start inference...")
    if args.benchmark:
        logger.info("BENCHMARK mode")
        for i in range(5):
            start = int(round(time.time() * 1000))
            out = _transfer(real_A, real_B, mask_A, mask_B, diff_A, diff_B, net)
            end = int(round(time.time() * 1000))
            logger.info(f"\tailia processing time {end - start} ms")
    else:
        out = _transfer(real_A, real_B, mask_A, mask_B, diff_A, diff_B, net)

    # Postprocessing
    postprocess = PostProcess()
    image = _postprocessing(out[0], source, crop_face, postprocess)
    savepath = get_savepath(args.savepath, args.input[0])
    image.save(savepath)
    logger.info(f"saved at : {savepath}")
    logger.info("Script finished successfully.")


def transfer_to_video():
    # Net initialize
    net = _initialize_net(args)

    preprocess = PreProcess(
        args, face_parser_path, face_alignment_path, face_detector_path
    )
    _, real_B, mask_B, diff_B, _ = _prepare_data(args, preprocess, "reference")
    postprocess = PostProcess()

    capture = get_capture(args.video)
    
    frame_shown = False
    while True:
        ret, frame = capture.read()
        out = None
        if (cv2.waitKey(1) & 0xFF == ord("q")) or not ret:
            break
        if frame_shown and cv2.getWindowProperty('frame', cv2.WND_PROP_VISIBLE) == 0:
            break

        # Prepare input data
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = cut_max_square(frame)
        _, frame = adjust_frame_size(frame, IMAGE_HEIGHT, IMAGE_WIDTH)
        frame = Image.fromarray(frame)
        source, real_A, mask_A, diff_A, crop_face = _prepare_data(
            args, preprocess, "frame", frame
        )

        # Inference
        if real_A is not None:
            out = _transfer(real_A, real_B, mask_A, mask_B, diff_A, diff_B, net)

        # Postprocessing
        if out:
            image = _postprocessing(out[0], source, crop_face, postprocess)
            cv2.imshow("frame", cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR))
            frame_shown = True

    capture.release()
    cv2.destroyAllWindows()
    logger.info("Script finished successfully.")


def main():
    # Check model files and download
    check_and_download_models(
        WEIGHT_PATH,
        MODEL_PATH,
        REMOTE_PATH,
    )
    check_and_download_models(
        FACE_PARSER_WEIGHT_PATH,
        FACE_PARSER_MODEL_PATH,
        REMOTE_PATH,
    )
    if not args.use_dlib:
        check_and_download_models(
            FACE_ALIGNMENT_WEIGHT_PATH,
            FACE_ALIGNMENT_MODEL_PATH,
            FACE_ALIGNMENT_REMOTE_PATH,
        )
        check_and_download_models(
            FACE_DETECTOR_WEIGHT_PATH,
            FACE_DETECTOR_MODEL_PATH,
            FACE_DETECTOR_REMOTE_PATH,
        )

    if args.video is not None:
        # Video mode
        transfer_to_video()
    else:
        # Image mode
        transfer_to_image()


if __name__ == "__main__":
    main()
