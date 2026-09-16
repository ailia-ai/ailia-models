import os
import sys
import subprocess
import time
import argparse

import numpy as np
import cv2

import ailia

# import original modules
sys.path.append('../../util')
sys.path.append('../../style_transfer') # import setup for face alignement (psgan)
sys.path.append('../../style_transfer/psgan') # import preprocess for face alignement (psgan)
from arg_utils import get_base_parser, update_parser, get_savepath  # noqa: E402
from model_utils import check_and_download_models  # noqa: E402
from image_utils import load_image  # noqa: E402
import webcamera_utils  # noqa: E402
from align_crop import align_face # noqa: E402

# logger
from logging import getLogger   # noqa: E402
logger = getLogger(__name__)

# ======================
# Parameters
# ======================
WEIGHT_PATH = 'restyle-encoder-fp16.onnx'
MODEL_PATH = 'restyle-encoder.onnx.prototxt'

FACE_POOL_WEIGHT_PATH = 'face-pool-fp16.onnx'
FACE_POOL_MODEL_PATH = 'face-pool.onnx.prototxt'

TOONIFY_WEIGHT_PATH = 'toonify-fp16.onnx'
TOONIFY_MODEL_PATH = 'toonify.onnx.prototxt'

FACE_ALIGNMENT_WEIGHT_PATH = "../../face_recognition/face_alignment/2DFAN-4.onnx"
FACE_ALIGNMENT_MODEL_PATH = "../../face_recognition/face_alignment/2DFAN-4.onnx.prototxt"

FACE_DETECTOR_WEIGHT_PATH = "../../face_detection/blazeface/blazeface.onnx"
FACE_DETECTOR_MODEL_PATH = "../../face_detection/blazeface/blazeface.onnx.prototxt"

REMOTE_PATH = 'https://storage.googleapis.com/ailia-models/restyle_encoder/'
FACE_ALIGNMENT_REMOTE_PATH = "https://storage.googleapis.com/ailia-models/face_alignment/"
FACE_DETECTOR_REMOTE_PATH = "https://storage.googleapis.com/ailia-models/blazeface/"

face_alignment_path = [FACE_ALIGNMENT_MODEL_PATH, FACE_ALIGNMENT_WEIGHT_PATH]
face_detector_path = [FACE_DETECTOR_MODEL_PATH, FACE_DETECTOR_WEIGHT_PATH]

IMAGE_PATH = 'img/face_img.jpg'
SAVE_IMAGE_PATH = 'img/output.png'
ALIGNED_PATH = 'img/aligned/'

IMAGE_HEIGHT = 1024
IMAGE_WIDTH = 1024

RESIZE_HEIGHT = 256
RESIZE_WIDTH = 256


# ======================
# Arguemnt Parser Config
# ======================
parser = get_base_parser(
    'ReStyle', IMAGE_PATH, SAVE_IMAGE_PATH,
)
parser.add_argument(
    '-iter', '--iteration',
    default=5, type=int,
    help='Number of iterations per batch (default 5)'
)
parser.add_argument(
    '-toon', '--toonify',
    action='store_true',
    help='Run the toonification task'
)
parser.add_argument(
    '-d', '--debug',
    action='store_true',
    help='Debugger'
)
parser.add_argument(
    '--onnx',
    action='store_true',
    help='Inference using onnx runtime'
)
parser.add_argument(
    "--use_dlib",
    action="store_true",
    help="Use dlib models for face alignment",
)
parser.add_argument(
    "opts",
    help="Modify config options using the command-line (for psgan)",
    default=None,
    nargs=argparse.REMAINDER,
)
args = update_parser(parser)


# ======================
# Utils
# ======================
def check(path):
    import onnx
    onnx_model = onnx.load(path)
    onnx.checker.check_model(onnx_model)

def run_on_batch(inputs, net, face_pool_net, iters, avg_image, toonify=None, onnx=False):
    y_hat, latent = None, np.load('average/latent_avg.npy')
    results_batch = {idx: [] for idx in range(inputs.shape[0])}

    for iter in range(iters):
        # First iteration uses the restyle encoder model and the average image as latent
        if iter == 0:
            avg_image_for_batch = np.tile(avg_image, (inputs.shape[0], 1, 1, 1))
            x_input = np.concatenate((inputs, avg_image_for_batch), axis=1)

            logger.info(f"Iteration {iter+1}/{iters}")
            # ailia prediction
            if not onnx:
                y_hat, latent = net.predict({'x_input': x_input, 'latent_input': latent})
            # onnx runtime prediction
            else: 
                ort_inputs = {net.get_inputs()[0].name: x_input.astype(np.float32), net.get_inputs()[1].name: latent.astype(np.float32)}
                ort_outs = net.run(None, ort_inputs)
                y_hat, latent = ort_outs[0], ort_outs[1]
        # Following iterations use the restyle encoder or the toonify model and the previous latent output as new latent
        else:
            x_input = np.concatenate((inputs, y_hat), axis=1)

            logger.info(f"Iteration {iter+1}/{iters}")
            # ailia prediction
            if not onnx:
                if toonify is None:
                    y_hat, latent = net.predict({'x_input': x_input, 'latent_input': latent})
                # toonification task
                else: 
                    y_hat, latent = toonify.predict({'x_input': x_input, 'latent_input': latent})
            # onnx runtime prediction
            else: 
                ort_inputs = {net.get_inputs()[0].name: x_input.astype(np.float32), net.get_inputs()[1].name: latent.astype(np.float32)}
                if toonify is None:
                    ort_outs = net.run(None, ort_inputs)
                    y_hat, latent = ort_outs[0], ort_outs[1]
                # toonification task
                else:
                    ort_outs = toonify.run(None, ort_inputs)
                    y_hat, latent = ort_outs[0], ort_outs[1]

        # store intermediate outputs
        for idx in range(inputs.shape[0]):
            results_batch[idx].append(y_hat[idx])

        # face pooling
        if not onnx:
            y_hat = face_pool_net.predict(y_hat)
        else:
            ort_inputs = {face_pool_net.get_inputs()[0].name: y_hat.astype(np.float32)}
            ort_outs = face_pool_net.run(None, ort_inputs)
            y_hat = ort_outs[0]

    return results_batch

def np2im(var, input=False):
    var = var.astype('float32')
    var = cv2.cvtColor(
        var.transpose(1, 2, 0),
        cv2.COLOR_RGB2BGR
    )
    if not input:
        var = ((var + 1) / 2)
    var[var < 0] = 0
    var[var > 1] = 1
    var = var * 255
    return var.astype('uint8')

def post_processing(result_batch, input_img):
    for i in range(input_img.shape[0]):
        results = [np2im(result_batch[i][iter_idx]) for iter_idx in range(args.iteration)]
        # save step-by-step results side-by-side
        input_im = np2im(input_img[i], input=True)
        res = np.array(results[0])
        for idx, result in enumerate(results[1:]):
            res = np.concatenate([res, result], axis=1)
        res = np.concatenate([res, input_im], axis=1)
    
    return res

# ======================
# Main functions
# ======================
def recognize_from_image(filename, net, face_pool_net, toonify=None, onnx=False): 

    # face alignment
    aligned = align_face(filename, args, face_alignment_path, face_detector_path)
    if aligned is not None:
        path = os.path.join(ALIGNED_PATH, filename.split('/')[-1])
        aligned.save(path)
    else: 
        path = filename

    input_img = load_image(
        path,
        (IMAGE_HEIGHT, IMAGE_WIDTH),
        normalize_type='255',
        gen_input_ailia=True,
    )

    input_img_resized = load_image(
        path,
        (RESIZE_HEIGHT, RESIZE_WIDTH),
        normalize_type='255',
        gen_input_ailia=True,
    )
    input_img_resized = (input_img_resized * 2) - 1

    avg_img = np.load('average/avg_image.npy')
    
    # inference
    logger.info('Start inference...')
    if args.benchmark:
        logger.info('BENCHMARK mode')
        for i in range(5):
            # ailia prediction
            if not onnx:
                if toonify is None:
                    start = int(round(time.time() * 1000))
                    result_batch = run_on_batch(input_img_resized, net, face_pool_net, args.iteration, avg_img)
                    end = int(round(time.time() * 1000))
                    logger.info(f'\tailia processing time {end - start} ms')
                # toonification task
                else:
                    start = int(round(time.time() * 1000))
                    result_batch = run_on_batch(input_img_resized, net, face_pool_net, args.iteration, avg_img, toonify=toonify)
                    end = int(round(time.time() * 1000))
                    logger.info(f'\tailia processing time {end - start} ms')
            # onnx runtime prediction
            else:
                if toonify is None:
                    start = int(round(time.time() * 1000))
                    result_batch = run_on_batch(input_img_resized, net, face_pool_net, args.iteration, avg_img, onnx=True)
                    end = int(round(time.time() * 1000))
                    logger.info(f'\tonnx runtime processing time {end - start} ms')
                # toonification task
                else:
                    start = int(round(time.time() * 1000))
                    result_batch = run_on_batch(input_img_resized, net, face_pool_net, args.iteration, avg_img, toonify=toonify, onnx=True)
                    end = int(round(time.time() * 1000))
                    logger.info(f'\tailia processing time {end - start} ms')
    else:
        # ailia prediction
        if not onnx:
            if toonify is None:
                result_batch = run_on_batch(input_img_resized, net, face_pool_net, args.iteration, avg_img)
            # toonification task
            else:
                result_batch= run_on_batch(input_img_resized, net, face_pool_net, args.iteration, avg_img, toonify=toonify)
        # onnx runtime prediction
        else:
            if toonify is None:
                result_batch = run_on_batch(input_img_resized, net, face_pool_net, args.iteration, avg_img, onnx=True)
            # toonification task
            else:
                result_batch = run_on_batch(input_img_resized, net, face_pool_net, args.iteration, avg_img, toonify=toonify, onnx=True)

    # post processing
    res = post_processing(result_batch, input_img)

    # save results       
    savepath = get_savepath(args.savepath, filename)
    logger.info(f'saved at : {savepath}')
    cv2.imwrite(savepath, res)


def recognize_from_video(filename, net, face_pool_net, toonify=None):

    capture = webcamera_utils.get_capture(args.video)

    # create video writer if savepath is specified as video format
    if args.savepath != SAVE_IMAGE_PATH:
        writer = webcamera_utils.get_writer(
            args.savepath, IMAGE_HEIGHT, IMAGE_WIDTH
        )
    else:
        writer = None

    # average image
    avg_img = np.load('average/avg_image.npy')
    
    frame_shown = False
    while(True):
        ret, frame = capture.read()
        if (cv2.waitKey(1) & 0xFF == ord('q')) or not ret:
            break
        if frame_shown and cv2.getWindowProperty('frame', cv2.WND_PROP_VISIBLE) == 0:
            break

        # Resize by padding the perimeter.
        _, input_data = webcamera_utils.preprocess_frame(
            frame, IMAGE_HEIGHT, IMAGE_WIDTH, normalize_type='255'
        )

        resized_input = cv2.resize(input_data[0].transpose(1,2,0), (RESIZE_HEIGHT, RESIZE_WIDTH))
        resized_input = np.expand_dims(resized_input.transpose(2,0,1), axis=0)
        resized_input = (resized_input * 2) - 1

        # inference
        if toonify is None:
            result_batch = run_on_batch(resized_input, net, face_pool_net, args.iteration, avg_img)
        # toonification task
        else:
            result_batch = run_on_batch(resized_input, net, face_pool_net, args.iteration, avg_img, toonify=toonify)
            
        # post-processing
        res_img = post_processing(result_batch, input_data)
        cv2.imshow('frame', res_img)
        frame_shown = True

        # save results
        if writer is not None:
            writer.write(res_img)

    capture.release()
    cv2.destroyAllWindows()
    if writer is not None:
        writer.release()


def main():
    # model files check and download
    check_and_download_models(WEIGHT_PATH, MODEL_PATH, REMOTE_PATH)
    check_and_download_models(FACE_POOL_WEIGHT_PATH, FACE_POOL_MODEL_PATH, REMOTE_PATH)
    # toonification task
    if args.toonify:
        check_and_download_models(TOONIFY_WEIGHT_PATH, TOONIFY_MODEL_PATH, REMOTE_PATH)
    if not args.use_dlib:
        check_and_download_models(
            FACE_ALIGNMENT_WEIGHT_PATH,
            FACE_ALIGNMENT_MODEL_PATH,
            FACE_ALIGNMENT_REMOTE_PATH
        )
        check_and_download_models(
            FACE_DETECTOR_WEIGHT_PATH,
            FACE_DETECTOR_MODEL_PATH,
            FACE_DETECTOR_REMOTE_PATH
        )

    # debug
    if args.debug:
        check(WEIGHT_PATH)
        check(FACE_POOL_WEIGHT_PATH)
        # toonification task
        if args.toonify:
            check(TOONIFY_WEIGHT_PATH)
        if args.use_dlib:
            check(FACE_ALIGNMENT_WEIGHT_PATH)
            check(FACE_DETECTOR_WEIGHT_PATH)
        logger.info('Debug OK.')
    else:
        if args.video is not None:
            # net initialize
            net = ailia.Net(MODEL_PATH, WEIGHT_PATH, env_id=args.env_id, memory_mode=args.memory_mode)
            face_pool_net = ailia.Net(FACE_POOL_MODEL_PATH, FACE_POOL_WEIGHT_PATH, env_id=args.env_id, memory_mode=args.memory_mode)
            if args.toonify:
                toonify_net = ailia.Net(TOONIFY_MODEL_PATH, TOONIFY_WEIGHT_PATH, env_id=args.env_id, memory_mode=args.memory_mode)
            else: 
                toonify_net = None
            # video mode
            recognize_from_video(SAVE_IMAGE_PATH, net, face_pool_net, toonify_net)
        else:
            # image mode
            if args.onnx:
                import onnxruntime

                # onnx runtime
                if args.benchmark:
                    start = int(round(time.time() * 1000))
                    ort_session = onnxruntime.InferenceSession(WEIGHT_PATH)
                    face_pool_ort_session = onnxruntime.InferenceSession(FACE_POOL_WEIGHT_PATH)
                    # toonification task
                    if args.toonify:
                        toonify_ort_session = onnxruntime.InferenceSession(TOONIFY_WEIGHT_PATH)
                    else: 
                        toonify_ort_session = None
                    end = int(round(time.time() * 1000))
                    logger.info(f'\tonnx runtime initializing time {end - start} ms')
                else:
                    ort_session = onnxruntime.InferenceSession(WEIGHT_PATH)
                    face_pool_ort_session = onnxruntime.InferenceSession(FACE_POOL_WEIGHT_PATH)
                    # toonification task
                    if args.toonify:
                        toonify_ort_session = onnxruntime.InferenceSession(TOONIFY_WEIGHT_PATH)
                    else: 
                        toonify_ort_session = None
                # input image loop
                for image_path in args.input:
                    recognize_from_image(image_path, ort_session, face_pool_ort_session, toonify=toonify_ort_session, onnx=True)
            else:
                # net initialize
                if args.benchmark:
                    start = int(round(time.time() * 1000))
                    net = ailia.Net(MODEL_PATH, WEIGHT_PATH, env_id=args.env_id, memory_mode=args.memory_mode)
                    face_pool_net = ailia.Net(FACE_POOL_MODEL_PATH, FACE_POOL_WEIGHT_PATH, env_id=args.env_id, memory_mode=args.memory_mode)
                    # toonification task
                    if args.toonify:
                        toonify_net = ailia.Net(TOONIFY_MODEL_PATH, TOONIFY_WEIGHT_PATH, env_id=args.env_id, memory_mode=args.memory_mode)
                    else: 
                        toonify_net = None
                    end = int(round(time.time() * 1000))
                    logger.info(f'\tailia initializing time {end - start} ms')
                else:
                    net = ailia.Net(MODEL_PATH, WEIGHT_PATH, env_id=args.env_id, memory_mode=args.memory_mode)
                    face_pool_net = ailia.Net(FACE_POOL_MODEL_PATH, FACE_POOL_WEIGHT_PATH, env_id=args.env_id, memory_mode=args.memory_mode)
                    # toonification task
                    if args.toonify:
                        toonify_net = ailia.Net(TOONIFY_MODEL_PATH, TOONIFY_WEIGHT_PATH, env_id=args.env_id, memory_mode=args.memory_mode)
                    else: 
                        toonify_net = None
                # input image loop
                for image_path in args.input:
                    recognize_from_image(image_path, net, face_pool_net, toonify=toonify_net)
    logger.info('Script finished successfully.')


if __name__ == '__main__':
    main()
