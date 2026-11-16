import sys
import cv2
import time
import numpy as np

import ailia

# import original modules
sys.path.append('../../util')
from arg_utils import get_base_parser, update_parser, get_savepath  # noqa: E402
from model_utils import check_and_download_models  # noqa: E402
from image_utils import imread  # noqa: E402
import webcamera_utils  # noqa: E402

from codeformer_util import FaceRestoreHelper,img2tensor, tensor2img, is_gray

# logger
from logging import getLogger   # noqa: E402
logger = getLogger(__name__)


# ======================
# Parameters 1
# ======================
IMAGE_PATH = 'input.png'
SAVE_IMAGE_PATH = 'output.png'

# ======================
# Argument Parser Config
# ======================
parser = get_base_parser(
    'Single Image Super-Resolution with HAN', IMAGE_PATH, SAVE_IMAGE_PATH,
)

parser.add_argument('-w', '--fidelity_weight', type=float, default=0.5, 
        help='Balance the quality and fidelity. Default: 0.5')

#parser.add_argument('--detection_model', type=str, default='retinaface_resnet50', 
parser.add_argument('--arch', type=str, default='retinaface_resnet50', 
        help='Face detector. Optional: retinaface_resnet50, retinaface_mobile0.25 \
            Default: retinaface_resnet50')

parser.add_argument('--has_aligned', action='store_true', help='Input are cropped and aligned faces. Default: False')

parser.add_argument('--only_center_face', action='store_true', help='Only restore the center face. Default: False')

parser.add_argument('--draw_box', action='store_true', help='Draw the bounding box for the detected faces. Default: False')
args = update_parser(parser)


# ======================
# Parameters 2
# ======================
WEIGHT_PATH = 'codeformer.onnx'
MODEL_PATH = 'codeformer.onnx.prototxt'

WEIGHT_PATH_FASE_PARSE = 'face_parse.onnx'
MODEL_PATH_FACE_PARSE = 'face_parse.onnx.prototxt'

WEIGHT_PATH_RETINALFACE_RESNET = 'retinaface_resnet50.onnx'
MODEL_PATH_RETINALFACE_RESNET = 'retinaface_resnet50.onnx.prototxt'

WEIGHT_PATH_RETINALFACE_MOBILE = 'retinaface_mobile0.25.onnx'
MODEL_PATH_RETINALFACE_MOBILE = 'retinaface_mobile0.25.onnx.prototxt'

CODEFORMER_REMOTE_PATH = 'https://storage.googleapis.com/ailia-models/codeformer/'
RETINAFACE_REMOTE_PATH = 'https://storage.googleapis.com/ailia-models/retinaface/'


# ======================
# Main functions
# ======================

def compute(net,face_helper,img):

    w = args.fidelity_weight
    if args.has_aligned: 
        # the input faces are already cropped and aligned
        img = cv2.resize(img, (512, 512), interpolation=cv2.INTER_LINEAR)
        face_helper.is_gray = is_gray(img, threshold=10)
        if face_helper.is_gray:
            print('Grayscale input: True')
        face_helper.cropped_faces = [img]
    else:
        img = cv2.resize(img, (512, 512), interpolation=cv2.INTER_LINEAR)
        face_helper.read_image(img)
        # get face landmarks for each face
        num_det_faces = face_helper.get_face_landmarks_5(
            only_center_face=args.only_center_face, resize=640, eye_dist_threshold=5)
        print(f'\tdetect {num_det_faces} faces')
        # align and warp each face
        face_helper.align_warp_face()

    # prepare data
    cropped_face_t = img2tensor(face_helper.cropped_faces[0] / 255., bgr2rgb=True, float32=True)

    mean = np.array([0.5,0.5,0.5])
    std  = np.array([0.5,0.5,0.5])
    cropped_face_t = cropped_face_t
    for i in range(3):
        cropped_face_t[i,:, :] = (cropped_face_t[i,:, :] - mean[i]) / std[i]

    cropped_face_t = np.expand_dims(cropped_face_t,0)

    output = net.run((cropped_face_t, np.array(w) ) )[0]

    restored_face = tensor2img(output, rgb2bgr=True, min_max=(-1, 1))
     
    restored_face = restored_face.astype('uint8')
    face_helper.add_restored_face(restored_face, face_helper.cropped_faces[0])

    # paste_back
    if not args.has_aligned:
        # upsample the background
        face_helper.get_inverse_affine(None)
        # paste each restored face to the input image
        restored_img = face_helper.paste_faces_to_input_image(upsample_img=None, draw_box=args.draw_box)
    if not args.has_aligned and restored_img is not None:
        return restored_img
 
    # save faces
    for restored_face in face_helper.restored_faces:
        return restored_face

def recognize_from_image(net):
    # input image loop

    face_helper = FaceRestoreHelper(
        #args.upscale,
        1,
        face_size=512,
        crop_ratio=(1, 1),
        det_model = args.arch,
        save_ext='png',
        use_parse=True,
        args=args)
 
    for image_path in args.input:
        # prepare input data
        logger.info('Input image: ' + image_path)

        # preprocessing
        img = imread(image_path)
        #541,542
        #img = cv2.resize(img,(541,542))

        # inference
        logger.info('Start inference...')
        if args.benchmark:
            logger.info('BENCHMARK mode')
            for i in range(5):
                start = int(round(time.time() * 1000))
                output_img = compute(net,face_helper,img)
                end = int(round(time.time() * 1000))
                logger.info(f'\tailia processing time {end - start} ms')
        else:
            output_img = compute(net,face_helper,img)

        # postprocessing

        savepath = get_savepath(args.savepath, image_path)
        logger.info(f'saved at : {savepath}')
        cv2.imwrite(savepath, output_img)
        
    logger.info('Script finished successfully.')

def recognize_from_video(net):
    capture = webcamera_utils.get_capture(args.video)

    face_helper = FaceRestoreHelper(
        #args.upscale,
        1,
        face_size=512,
        crop_ratio=(1, 1),
        det_model = args.arch,
        save_ext='png',
        use_parse=True,
        args=args)
 
    # create video writer if savepath is specified as video format
    if args.savepath != SAVE_IMAGE_PATH:
        f_h = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT) * int(args.scale))
        f_w = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH) * int(args.scale))
        writer = webcamera_utils.get_writer(args.savepath, f_h, f_w)
    else:
        writer = None

    frame_shown = False
    while (True):
        ret, frame = capture.read()
        face_helper.parameter_reinit()
        if (cv2.waitKey(1) & 0xFF == ord('q')) or not ret:
            break
        if frame_shown and cv2.getWindowProperty('output', cv2.WND_PROP_VISIBLE) == 0:
            break

        IMAGE_HEIGHT, IMAGE_WIDTH = frame.shape[0], frame.shape[1]
        SQUARE_SIZE = max(IMAGE_WIDTH, IMAGE_HEIGHT)

        # resize with keep aspect
        frame,resized_img = webcamera_utils.adjust_frame_size(frame, SQUARE_SIZE, SQUARE_SIZE)

        out_img = compute(net,face_helper,frame)
        cv2.imshow('output', out_img)
        frame_shown = True

        # save results
        if writer is not None:
            writer.write(out_img)

    capture.release()
    cv2.destroyAllWindows()
    if writer is not None:
        writer.release()
    logger.info('Script finished successfully.')


def main():
    # model files check and download
    check_and_download_models(WEIGHT_PATH, MODEL_PATH, CODEFORMER_REMOTE_PATH)
    check_and_download_models(WEIGHT_PATH_FASE_PARSE, MODEL_PATH_FACE_PARSE, CODEFORMER_REMOTE_PATH)
    check_and_download_models(WEIGHT_PATH_RETINALFACE_RESNET, MODEL_PATH_RETINALFACE_RESNET, RETINAFACE_REMOTE_PATH)
    check_and_download_models(WEIGHT_PATH_RETINALFACE_MOBILE, MODEL_PATH_RETINALFACE_MOBILE, RETINAFACE_REMOTE_PATH)

    # net initialize
    mem_mode = ailia.get_memory_mode(reduce_constant=True, reuse_interstage=True)
    net = ailia.Net(MODEL_PATH, WEIGHT_PATH, env_id=args.env_id, memory_mode=mem_mode)
    if args.video is not None:
        # video mode
        recognize_from_video(net)
    else:
        # image mode
        recognize_from_image(net)


if __name__ == '__main__':
    main()
