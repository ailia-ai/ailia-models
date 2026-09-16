import sys
import time

import ailia
import cv2
import numpy as np

import blazepalm_utils as but

sys.path.append('../../util')
# logger
from logging import getLogger  # noqa: E402

import webcamera_utils  # noqa: E402
from image_utils import imread, load_image  # noqa: E402
from model_utils import check_and_download_models  # noqa: E402
from arg_utils import get_base_parser, get_savepath, update_parser  # noqa: E402
from webcamera_utils import get_capture, get_writer  # noqa: E402

logger = getLogger(__name__)


# ======================
# Parameters 1
# ======================
IMAGE_PATH = 'person_with_hands.jpg'
SAVE_IMAGE_PATH = 'output.png'

# ======================
# Argument Parser Config
# ======================
parser = get_base_parser(
    'BlazePalm, on-device real-time palm detection.',
    IMAGE_PATH,
    SAVE_IMAGE_PATH,
)
parser.add_argument(
    '-m', '--model_name',
    default='blazepalm',
    help='[blazepalm, palm_detection, palm_detection_full]'
)
parser.add_argument(
    '--onnx',
    action='store_true',
    help='By default, the ailia SDK is used, but with this option, you can switch to using ONNX Runtime'
)
args = update_parser(parser)


# ======================
# Parameters 2
# ======================
MODEL_NAME = 'blazepalm'

if args.model_name == "blazepalm":
    #MediaPipePyTorch (https://github.com/zmurez/MediaPipePyTorch)
    WEIGHT_PATH = f'blazepalm.onnx'
    MODEL_PATH = WEIGHT_PATH+".prototxt"
    IMAGE_HEIGHT = 256
    IMAGE_WIDTH = 256
    ANCHOR_PATH = 'anchors.npy'
    CHANNEL_FIRST = True
elif args.model_name == "palm_detection":
    #Download palm_detection.tflite
    #https://github.com/google/mediapipe/tree/350fbb2100ad531bc110b93aaea23d96af5a5064/mediapipe/modules/palm_detection
    #python3 -m tf2onnx.convert --opset 11 --tflite palm_detection.tflite --output palm_detection.onnx
    WEIGHT_PATH = f'palm_detection.onnx'
    MODEL_PATH = WEIGHT_PATH+".prototxt"
    IMAGE_HEIGHT = 128
    IMAGE_WIDTH = 128
    ANCHOR_PATH = 'anchors_128.npy'
    CHANNEL_FIRST = False
elif args.model_name == "palm_detection_full":
    #Download palm_detection_full.tflite
    #https://github.com/google/mediapipe/tree/master/mediapipe/modules/palm_detection
    #python3 -m tf2onnx.convert --opset 11 --tflite palm_detection_full.tflite --output palm_detection.onnx
    WEIGHT_PATH = f'palm_detection_full.onnx'
    MODEL_PATH = WEIGHT_PATH+".prototxt"
    IMAGE_HEIGHT = 192
    IMAGE_WIDTH = 192
    ANCHOR_PATH = 'anchors_192.npy'
    CHANNEL_FIRST = False
else:
    raise "unknown model"

REMOTE_PATH = f'https://storage.googleapis.com/ailia-models/{MODEL_NAME}/'

# ======================
# Utils
# ======================
def display_result(img, detections, with_keypoints=True):
    if detections.ndim == 1:
        detections = np.expand_dims(detections, axis=0)

    n_keypoints = detections.shape[1] // 2 - 2

    for i in range(detections.shape[0]):
        ymin = detections[i, 0]
        xmin = detections[i, 1]
        ymax = detections[i, 2]
        xmax = detections[i, 3]

        start_point = (int(xmin), int(ymin))
        end_point = (int(xmax), int(ymax))
        img = cv2.rectangle(img, start_point, end_point, (255, 0, 0), 1)

        if with_keypoints:
            for k in range(n_keypoints):
                kp_x = int(detections[i, 4 + k*2])
                kp_y = int(detections[i, 4 + k*2 + 1])
                cv2.circle(img, (kp_x, kp_y), 2, (0, 0, 255), thickness=2)
    return img


# ======================
# Main functions
# ======================
def recognize_from_image():
    # net initialize
    if args.onnx:
        import onnxruntime
        net = onnxruntime.InferenceSession(WEIGHT_PATH)
    else:
        net = ailia.Net(MODEL_PATH, WEIGHT_PATH, env_id=args.env_id, memory_mode=args.memory_mode)

    # input image loop
    for image_path in args.input:
        # prepare input data
        logger.info(image_path)
        src_img = imread(image_path)
        img256, _, scale, pad = but.resize_pad(src_img[:, :, ::-1],IMAGE_WIDTH)
        input_data = img256.astype('float32') / 255.
        input_data = np.expand_dims(np.moveaxis(input_data, -1, 0), 0)
        if not CHANNEL_FIRST:
            input_data = input_data.transpose((0,2,3,1))

        # inference
        logger.info('Start inference...')
        if args.benchmark:
            logger.info('BENCHMARK mode')
            for _ in range(5):
                start = int(round(time.time() * 1000))
                preds = net.predict([input_data])
                normalized_detections = but.postprocess(preds,anchor_path=ANCHOR_PATH,resolution=IMAGE_WIDTH)[0]
                detections = but.denormalize_detections(
                    normalized_detections, scale, pad, resolution=IMAGE_WIDTH
                )
                end = int(round(time.time() * 1000))
                logger.info(f'\tailia processing time {end - start} ms')
        else:
            if args.onnx:
                input_name = net.get_inputs()[0].name
                preds = net.run(None, {input_name: input_data.astype(np.float32)})
            else:
                preds = net.predict([input_data])
            normalized_detections = but.postprocess(preds, anchor_path=ANCHOR_PATH,resolution=IMAGE_WIDTH)[0]
            detections = but.denormalize_detections(
                normalized_detections, scale, pad, resolution=IMAGE_WIDTH
            )

        # postprocessing
        display_result(src_img, detections)
        savepath = get_savepath(args.savepath, image_path)
        logger.info(f'saved at : {savepath}')
        cv2.imwrite(savepath, src_img)
    logger.info('Script finished successfully.')


def recognize_from_video():
    # net initialize
    net = ailia.Net(MODEL_PATH, WEIGHT_PATH, env_id=args.env_id, memory_mode=args.memory_mode)

    capture = webcamera_utils.get_capture(args.video)

    # create video writer if savepath is specified as video format
    if args.savepath != SAVE_IMAGE_PATH:
        f_h = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        f_w = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        writer = get_writer(args.savepath, f_h, f_w)
    else:
        writer = None

    frame_shown = False
    while(True):
        ret, frame = capture.read()
        if (cv2.waitKey(1) & 0xFF == ord('q')) or not ret:
            break
        if frame_shown and cv2.getWindowProperty('frame', cv2.WND_PROP_VISIBLE) == 0:
            break

        img256, _, scale, pad = but.resize_pad(frame[:, :, ::-1], resolution=IMAGE_WIDTH)
        input_data = img256.astype('float32') / 255.
        input_data = np.expand_dims(np.moveaxis(input_data, -1, 0), 0)
        if not CHANNEL_FIRST:
            input_data = input_data.transpose((0,2,3,1))

        # inference
        preds = net.predict([input_data])
        normalized_detections = but.postprocess(preds, anchor_path=ANCHOR_PATH, resolution=IMAGE_WIDTH)[0]
        detections = but.denormalize_detections(
            normalized_detections, scale, pad, resolution=IMAGE_WIDTH
        )

        # postprocessing
        display_result(frame, detections)
        
        visual_img = frame
        if args.video == '0': # Flip horizontally if camera
            visual_img = np.ascontiguousarray(frame[:,::-1,:])

        cv2.imshow('frame', visual_img)
        frame_shown = True

        # save results
        if writer is not None:
            writer.write(frame)

    capture.release()
    if writer is not None:
        writer.release()
    cv2.destroyAllWindows()
    if writer is not None:
        writer.release()
    logger.info('Script finished successfully.')


def main():
    # model files check and download
    check_and_download_models(WEIGHT_PATH, MODEL_PATH, REMOTE_PATH)

    if args.video is not None:
        # video mode
        recognize_from_video()
    else:
        # image mode
        recognize_from_image()


if __name__ == '__main__':
    main()
