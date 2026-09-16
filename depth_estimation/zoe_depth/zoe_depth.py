import sys

import ailia
import cv2
from zoe_depth_util import get_params, postprocess, preprocess, save

# import original modules
sys.path.append("../../util")
# logger
from logging import getLogger  # noqa: E402

import webcamera_utils  # noqa: E401
from arg_utils import get_base_parser, update_parser  # noqa: E402
from model_utils import check_and_download_models

logger = getLogger(__name__)

# ======================
# Parameters
# ======================

REMOTE_PATH = "https://storage.googleapis.com/ailia-models/zoe_depth/"

IMAGE_PATH = "./input.jpg"
SAVE_IMAGE_PATH = "./output.png"

MODEL_ARCHS = ("ZoeD_M12_K", "ZoeD_M12_N", "ZoeD_M12_NK")
INPUT_SIZE = {
    "ZoeD_M12_K": (768, 384),
    "ZoeD_M12_N": (512, 384),
    "ZoeD_M12_NK": (512, 384),
}

# ======================
# Arguemnt Parser Config
# ======================

parser = get_base_parser("ZoeDepth model", IMAGE_PATH, SAVE_IMAGE_PATH)
parser.add_argument(
    "-a", "--arch", default="ZoeD_M12_K", choices=MODEL_ARCHS, help="arch model."
)
parser.add_argument("--onnx", action="store_true", help="execute onnxruntime version.")
args = update_parser(parser)


def infer_onnx(net, image, image_reversed):
    logger.info("Inference with ONNX.")
    input_name = net.get_inputs()[0].name
    if args.arch == "ZoeD_M12_NK":
        output_name = net.get_outputs()[1].name
    else:
        output_name = net.get_outputs()[0].name

    pred_not_reversed = net.run([output_name], {input_name: image})[0]
    pred_reversed = net.run([output_name], {input_name: image_reversed})[0]
    pred = 0.5 * (pred_not_reversed + pred_reversed)
    pred = pred.squeeze()
    return pred


def infer_ailia(net, image, image_reversed):
    logger.info("Inference with Ailia.")
    if args.arch == "ZoeD_M12_NK":
        pred_not_reversed = net.run(image)[1]
        pred_reversed = net.run(image_reversed)[1]
    else:
        pred_not_reversed = net.run(image)[0]
        pred_reversed = net.run(image_reversed)[0]

    pred = 0.5 * (pred_not_reversed + pred_reversed)
    pred = pred.squeeze()
    return pred


# ======================
# Main functions
# ======================
def infer_from_image(net):
    arch = args.arch
    input_image_file = args.input[0]
    img = cv2.imread(input_image_file)
    H, W = img.shape[:2]
    resized_img, resized_img_reversed = preprocess(img, INPUT_SIZE[arch])

    logger.info("Start inference.")
    if args.onnx:
        pred = infer_onnx(net, resized_img, resized_img_reversed)
    else:
        pred = infer_ailia(net, resized_img, resized_img_reversed)

    pred_postprocessed = postprocess(
        pred=pred,
        original_width=W,
        original_height=H,
    )
    logger.info(f"Saving depth image to {SAVE_IMAGE_PATH}.")
    save(pred=pred_postprocessed, output_filename=args.savepath)
    logger.info("Finish inference.")


def infer_from_video(net):
    arch = args.arch
    capture = webcamera_utils.get_capture(args.video)
    W = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))

    if args.savepath != SAVE_IMAGE_PATH:
        writer = webcamera_utils.get_writer(args.savepath, H, W)
    else:
        writer = None

    logger.info("Start inference.")
    logger.info("Finish inference.")

    frame_shown = False
    while True:
        ret, frame = capture.read()
        if (cv2.waitKey(1) & 0xFF == ord("q")) or not ret:
            break
        if frame_shown and cv2.getWindowProperty("frame", cv2.WND_PROP_VISIBLE) == 0:
            break

        resized_img, resized_img_reversed = preprocess(frame, INPUT_SIZE[arch])

        if args.onnx:
            pred = infer_onnx(net, resized_img, resized_img_reversed)
        else:
            pred = infer_ailia(net, resized_img, resized_img_reversed)

        pred_postprocessed = postprocess(
            pred=pred,
            original_width=W,
            original_height=H,
        )

        cv2.imshow("frame", pred_postprocessed[..., ::-1])
        frame_shown = True

        if writer is not None:
            writer.write(pred)

    capture.release()
    cv2.destroyAllWindows()
    if writer is not None:
        writer.release()


def main():
    # model files check and download
    weight_path, model_path = get_params(args.arch)
    check_and_download_models(weight_path, model_path, REMOTE_PATH)

    # create net instance
    if args.onnx:
        import onnxruntime

        net = onnxruntime.InferenceSession(weight_path)
    else:
        net = ailia.Net(model_path, weight_path, env_id=args.env_id, memory_mode=args.memory_mode)

    # check input
    if args.video:
        infer_from_video(net)
    else:
        infer_from_image(net)


if __name__ == "__main__":
    main()
