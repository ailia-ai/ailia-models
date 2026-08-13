import sys
import time

import numpy as np
import cv2

import ailia

# import original modules
sys.path.append('../../util')
from arg_utils import get_base_parser, update_parser, get_savepath  # noqa
from model_utils import check_and_download_models  # noqa
from image_utils import normalize_image  # noqa
from detector_utils import load_image  # noqa
from webcamera_utils import get_capture, get_writer  # noqa
# logger
from logging import getLogger  # noqa

from face_restoration import get_face_landmarks_5
from face_restoration import align_warp_face, get_inverse_affine
from face_restoration import paste_faces_to_image

logger = getLogger(__name__)

# ======================
# Parameters
# ======================

WEIGHT_DET_PATH = "retinaface_resnet50.onnx"
MODEL_DET_PATH = "retinaface_resnet50.onnx.prototxt"
REMOTE_PATH = 'https://storage.googleapis.com/ailia-models/gfpgan/'

REALESRGAN_MODEL = 'https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.1/RealESRGAN_x2plus.pth'

IMAGE_PATH = 'demo.png'
SAVE_IMAGE_PATH = 'output.png'

IMAGE_SIZE = 512

# ======================
# Arguemnt Parser Config
# ======================

parser = get_base_parser(
    'GFPGAN', IMAGE_PATH, SAVE_IMAGE_PATH, fp16_support=False
)
parser.add_argument(
    '-u', '--upscale', type=int, default=1,
    help='The final upsampling scale of the image.'
)
parser.add_argument(
    '--aligned', action='store_true',
    help='Input is aligned faces.'
)
parser.add_argument(
    '--facexlib', action='store_true',
    help='Use facexlib module.'
)
parser.add_argument(
    '--realesrgan', action='store_true',
    help='Use realesrgan module.'
)
parser.add_argument(
    '-m', '--model_name',
    default='v1.3',
    choices=['v1.3', 'v1.4'],
    help="Select model version"
)
parser.add_argument(
    '--onnx',
    action='store_true',
    help='execute onnxruntime version.'
)
args = update_parser(parser)


# ======================
# Model selection
# ======================

WEIGHT_PATH = 'GFPGAN'+args.model_name+'.onnx'
MODEL_PATH = 'GFPGAN'+args.model_name+'.onnx.prototxt'


# ======================
# Main functions
# ======================

def preprocess(img):
    img = img[:, :, ::-1]  # BGR -> RGB
    img = normalize_image(img, normalize_type='127.5')

    img = img.transpose(2, 0, 1)  # HWC -> CHW
    img = np.expand_dims(img, axis=0)
    img = img.astype(np.float32)

    return img


def post_processing(pred):
    img = pred[0]
    img = img.transpose(1, 2, 0)  # CHW -> HWC
    img = img[:, :, ::-1]  # RGB -> BGR

    img = np.clip(img, -1, 1)
    img = (img + 1) * 127.5
    img = img.astype(np.uint8)

    return img


def predict(models, img):
    face_det = models.get("face_det")
    face_restor = models.get("face_restor")
    realesrgan = models.get("upsampler")

    upscale = args.upscale

    if face_det:
        det_faces, all_landmarks_5 = get_face_landmarks_5(
            face_det, img, eye_dist_threshold=5)
        cropped_faces, affine_matrices = align_warp_face(img, all_landmarks_5)
    elif face_restor:
        face_restor.clean_all()
        face_restor.read_image(img)
        # get face landmarks for each face
        face_restor.get_face_landmarks_5(only_center_face=False, eye_dist_threshold=5)
        # eye_dist_threshold=5: skip faces whose eye distance is smaller than 5 pixels
        # align and warp each face
        face_restor.align_warp_face()
        cropped_faces = face_restor.cropped_faces
    else:
        img = cv2.resize(img, (IMAGE_SIZE, IMAGE_SIZE))
        cropped_faces = [img]

    gfpgan = models["gfpgan"]

    # face restoration
    restored_faces = []
    for cropped_face in cropped_faces:
        x = preprocess(cropped_face)

        # feedforward
        if not args.onnx:
            output = gfpgan.predict([x])
        else:
            output = gfpgan.run(None, {'x': x})
        pred = output[0]

        restored_face = post_processing(pred)
        restored_faces.append(restored_face)

        if face_restor:
            face_restor.add_restored_face(restored_face)

    if upscale > 1:
        if realesrgan:
            img = realesrgan.enhance(img, outscale=upscale)[0]
        else:
            h, w = img.shape[:2]
            h_up, w_up = int(h * upscale), int(w * upscale)
            img = cv2.resize(img, (w_up, h_up), interpolation=cv2.INTER_LANCZOS4)

    if face_det:
        inverse_affine_matrices = get_inverse_affine(affine_matrices, upscale_factor=upscale)
        restored_img = paste_faces_to_image(
            img, restored_faces, inverse_affine_matrices,
            upscale_factor=upscale)
    elif face_restor:
        face_restor.get_inverse_affine(None)
        restored_img = face_restor.paste_faces_to_input_image(upsample_img=img)
    else:
        restored_img = restored_faces[0]

    return restored_img


def recognize_from_image(models):
    # input image loop
    for image_path in args.input:
        logger.info(image_path)

        # prepare input data
        img = load_image(image_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)

        # inference
        logger.info('Start inference...')
        if args.benchmark:
            logger.info('BENCHMARK mode')
            total_time_estimation = 0
            for i in range(args.benchmark_count):
                start = int(round(time.time() * 1000))
                restored_img = predict(models, img)
                end = int(round(time.time() * 1000))
                estimation_time = (end - start)

                # Logging
                logger.info(f'\tailia processing estimation time {estimation_time} ms')
                if i != 0:
                    total_time_estimation = total_time_estimation + estimation_time

            logger.info(f'\taverage time estimation {total_time_estimation / (args.benchmark_count - 1)} ms')
        else:
            restored_img = predict(models, img)

        # plot result
        savepath = get_savepath(args.savepath, image_path, ext='.png')
        logger.info(f'saved at : {savepath}')
        cv2.imwrite(savepath, restored_img)

    logger.info('Script finished successfully.')


def recognize_from_video(models):
    video_file = args.video if args.video else args.input[0]
    capture = get_capture(video_file)
    assert capture.isOpened(), 'Cannot capture source'

    # create video writer if savepath is specified as video format
    if args.savepath != SAVE_IMAGE_PATH:
        f_h = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)) * args.upscale
        f_w = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)) * args.upscale
        writer = get_writer(args.savepath, f_h, f_w)
    else:
        writer = None

    frame_shown = False
    while True:
        ret, frame = capture.read()
        if (cv2.waitKey(1) & 0xFF == ord('q')) or not ret:
            break
        if frame_shown and cv2.getWindowProperty('frame', cv2.WND_PROP_VISIBLE) == 0:
            break

        # inference
        img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        restored_img = predict(models, img)
        restored_img = cv2.cvtColor(restored_img, cv2.COLOR_RGB2BGR)

        # show
        cv2.imshow('frame', restored_img)
        frame_shown = True

        # save results
        if writer is not None:
            writer.write(restored_img)

    capture.release()
    cv2.destroyAllWindows()
    if writer is not None:
        writer.release()

    logger.info('Script finished successfully.')


def main():
    # model files check and download
    check_and_download_models(WEIGHT_PATH, MODEL_PATH, REMOTE_PATH)
    if not args.aligned and not args.facexlib:
        check_and_download_models(WEIGHT_DET_PATH, MODEL_DET_PATH, REMOTE_PATH)

    env_id = args.env_id

    # initialize
    if not args.onnx:
        gfpgan = ailia.Net(MODEL_PATH, WEIGHT_PATH, env_id=env_id)
    else:
        import onnxruntime
        available_providers = onnxruntime.get_available_providers()
        if 'CUDAExecutionProvider' in available_providers:
            providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
        elif 'DmlExecutionProvider' in available_providers:
            providers = ['DmlExecutionProvider', 'CPUExecutionProvider']
        else:
            providers = ['CPUExecutionProvider']
        gfpgan = onnxruntime.InferenceSession(WEIGHT_PATH, providers=providers)

    models = {
        "gfpgan": gfpgan,
    }

    upscale = args.upscale
    if not args.aligned:
        if args.facexlib:
            import torch
            from facexlib.utils.face_restoration_helper import FaceRestoreHelper

            device = 'cuda' if torch.cuda.is_available() else 'cpu'
            face_restor = FaceRestoreHelper(
                upscale,
                face_size=IMAGE_SIZE,
                crop_ratio=(1, 1),
                det_model='retinaface_resnet50',
                use_parse=True,
                device=device)
            models["face_restor"] = face_restor
        else:
            assert not args.onnx, "onnx option not supported for face_det"

            face_det = ailia.Net(MODEL_DET_PATH, WEIGHT_DET_PATH, env_id=env_id)
            models["face_det"] = face_det

    if upscale > 1 and args.realesrgan:
        from basicsr.archs.rrdbnet_arch import RRDBNet
        from realesrgan import RealESRGANer

        model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=2)
        bg_tile = 400
        upsampler = RealESRGANer(
            scale=upscale,
            model_path=REALESRGAN_MODEL,
            model=model,
            tile=bg_tile,
            tile_pad=10,
            pre_pad=0,
            half=True)  # need to set False in CPU mode
        models["upsampler"] = upsampler

    if args.video is not None:
        recognize_from_video(models)
    else:
        recognize_from_image(models)


if __name__ == '__main__':
    main()
