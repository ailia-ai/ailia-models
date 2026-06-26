import os, sys, time
from time import strftime
import shutil
import numpy as np
import random
from argparse import ArgumentParser

import ailia

# import original modules
sys.path.append('../../util')
import webcamera_utils  # noqa: E402
from image_utils import imread, load_image  # noqa: E402
from model_utils import check_and_download_models  # noqa: E402
from arg_utils import get_base_parser, get_savepath, update_parser  # noqa: E402

from preprocess import CropAndExtract
from audio2coeff import Audio2Coeff
from animation import AnimateFromCoeff
from batch_generation import get_data, get_facerender_data

# logger
from logging import getLogger  # noqa: E402
logger = getLogger(__name__)

# ======================
# Parameters
# ======================
IMAGE_PATH = 'input.png'
INPUT_AUDIO_PATH = "input.wav"
SAVE_IMAGE_PATH = 'output.mp4'
LM3D_PATH = "./preprocess/similarity_Lm3D_all.mat"
PREPROCESS_LIST = ['crop', 'extcrop', 'resize', 'full', 'extfull']

# ======================
# Arguemnt Parser Config
# ======================
parser = get_base_parser("sadtalker", IMAGE_PATH, SAVE_IMAGE_PATH)
parser.add_argument("-a", "--audio", default=INPUT_AUDIO_PATH, help="Path to input audio")
parser.add_argument("--result_dir", default='./results', help="path to output")
parser.add_argument("--pose_style", type=int, default=0, help="input pose style from [0, 46)")
parser.add_argument("--expression_scale", type=float, default=1.0, help="the value of expression intensity")
parser.add_argument("--batch_size", type=int, default=2, help="the batch size of facerender")
parser.add_argument("--size", type=int, default=256, help="the image size of the facerender")
parser.add_argument('--enhancer', action="store_true", help="Face enhancer with gfpgan")
parser.add_argument("--still", action="store_true", help="can crop back to the original videos for the full body aniamtion")
parser.add_argument("--preprocess", default='crop', choices=PREPROCESS_LIST, help="how to preprocess the images")
parser.add_argument("--ref_eyeblink", default=None, help="path to reference video providing eye blinking")
parser.add_argument("--ref_pose", default=None, help="path to reference video providing pose")
parser.add_argument('--input_yaw', nargs='+', type=int, default=None, help="the input yaw degree of the user")
parser.add_argument('--input_pitch', nargs='+', type=int, default=None, help="the input pitch degree of the user")
parser.add_argument('--input_roll', nargs='+', type=int, default=None, help="the input roll degree of the user")
parser.add_argument("--verbose", action="store_true", help="saving the intermedia output or not")
parser.add_argument("--seed", type=int, default=42, help="ramdom seed")
parser.add_argument('-o', '--onnx', action='store_true', help="Option to use onnxrutime to run or not.")
args = update_parser(parser)

# ======================
# Parameters 2
# ======================
WEIGHT_FACE3D_RECON_PATH = "face3d_recon.onnx"
MODEL_FACE3D_RECON_PATH = "face3d_recon.onnx.prototxt"
WEIGHT_FACE_ALIGN_PATH = "face_align.onnx"
MODEL_FACE_ALIGN_PATH = "face_align.onnx.prototxt"
WEIGHT_AUDIO2EXP_PATH = "audio2exp.onnx"
MODEL_AUDIO2EXP_PATH = "audio2exp.onnx.prototxt"
WEIGHT_AUDIO2POSE_PATH = "audio2pose.onnx"
MODEL_AUDIO2POSE_PATH = "audio2pose.onnx.prototxt"
WEIGHT_ANIMATION_GENERATOR_PATH = "animation_generator.onnx"
MODEL_ANIMATION_GENERATOR_PATH = "animation_generator.onnx.prototxt"
WEIGHT_KP_DETECTOR_PATH = "kp_detector.onnx"
MODEL_KP_DETECTOR_PATH = "kp_detector.onnx.prototxt"
WEIGHT_MAPPING_NET = "mappingnet_full.onnx" if "full" in args.preprocess else "mappingnet_not_full.onnx"
MODEL_MAPPING_NET = WEIGHT_MAPPING_NET + ".prototxt"
REMOTE_PATH = "https://storage.googleapis.com/ailia-models/sadtalker/"

WEIGHT_FACE_DET_PATH = "retinaface_resnet50.onnx"
MODEL_FACE_DET_PATH = "retinaface_resnet50.onnx.prototxt"
REMOTE_FACE_DET_PATH = "https://storage.googleapis.com/ailia-models/retinaface/"

WEIGHT_GFPGAN_PATH = "GFPGANv1.4.onnx"
MODEL_GFPGAN_PATH = "GFPGANv1.4.onnx.prototxt"
REMOTE_GFPGAN_PATH = "https://storage.googleapis.com/ailia-models/gfpgan/"

# ======================
# Utils
# ======================
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)

def load_model(model_path, weight_path, env_id=args.env_id, use_onnx=args.onnx):
    if use_onnx:
        import onnxruntime
        cuda = 0 < ailia.get_gpu_environment_id()
        providers = (
            ["CUDAExecutionProvider", "CPUExecutionProvider"]
            if cuda
            else ["CPUExecutionProvider"]
        )
        return onnxruntime.InferenceSession(weight_path, providers=providers)
    else:
        return ailia.Net(model_path, weight_path, env_id=env_id)

def generate_ref_coeff(preprocess_model, video_path, save_dir):
    if not video_path:
        return None

    videoname = os.path.splitext(os.path.split(video_path)[-1])[0]
    frame_dir = os.path.join(save_dir, videoname)
    os.makedirs(frame_dir, exist_ok=True)
    
    print(f'3DMM Extraction for reference video: {videoname}')
    coeff_path, _, _ = preprocess_model.generate(
        video_path, 
        frame_dir, 
        args.preprocess, 
        source_image_flag=False
    )
    return coeff_path

# ======================
# Main functions
# ======================
def download_and_load_models():
    check_and_download_models(WEIGHT_FACE3D_RECON_PATH, MODEL_FACE3D_RECON_PATH, REMOTE_PATH)
    check_and_download_models(WEIGHT_FACE_ALIGN_PATH, MODEL_FACE_ALIGN_PATH, REMOTE_PATH)
    check_and_download_models(WEIGHT_AUDIO2EXP_PATH, MODEL_AUDIO2EXP_PATH, REMOTE_PATH)
    check_and_download_models(WEIGHT_AUDIO2POSE_PATH, MODEL_AUDIO2POSE_PATH, REMOTE_PATH)
    check_and_download_models(WEIGHT_ANIMATION_GENERATOR_PATH, MODEL_ANIMATION_GENERATOR_PATH, REMOTE_PATH)
    check_and_download_models(WEIGHT_KP_DETECTOR_PATH, MODEL_KP_DETECTOR_PATH, REMOTE_PATH)
    check_and_download_models(WEIGHT_MAPPING_NET, MODEL_MAPPING_NET, REMOTE_PATH)
    check_and_download_models(WEIGHT_FACE_DET_PATH, MODEL_FACE_DET_PATH, REMOTE_FACE_DET_PATH)
    if args.enhancer:
        check_and_download_models(WEIGHT_GFPGAN_PATH, MODEL_GFPGAN_PATH, REMOTE_GFPGAN_PATH)

    models = {
        "face3d_recon_net": load_model(MODEL_FACE3D_RECON_PATH, WEIGHT_FACE3D_RECON_PATH),
        "face_align_net": load_model(MODEL_FACE_ALIGN_PATH, WEIGHT_FACE_ALIGN_PATH),
        "audio2exp_net": load_model(MODEL_AUDIO2EXP_PATH, WEIGHT_AUDIO2EXP_PATH),
        "audio2pose_net": load_model(MODEL_AUDIO2POSE_PATH, WEIGHT_AUDIO2POSE_PATH),
        "generator_net": load_model(MODEL_ANIMATION_GENERATOR_PATH, WEIGHT_ANIMATION_GENERATOR_PATH),
        "kp_detector_net": load_model(MODEL_KP_DETECTOR_PATH, WEIGHT_KP_DETECTOR_PATH),
        "mapping_net": load_model(MODEL_MAPPING_NET, WEIGHT_MAPPING_NET),
        "retinaface_net": ailia.Net(MODEL_FACE_DET_PATH, WEIGHT_FACE_DET_PATH, env_id=args.env_id),
        "gfpgan_net": ailia.Net(MODEL_GFPGAN_PATH, WEIGHT_GFPGAN_PATH, env_id=args.env_id) if args.enhancer else None
    }
    return models

def preprocess_image(preprocess_model, pic_path, save_dir):
    first_frame_dir = os.path.join(save_dir, 'first_frame_dir')
    os.makedirs(first_frame_dir, exist_ok=True)
    
    first_coeff_path, crop_pic_path, crop_info = preprocess_model.generate(
        pic_path, 
        first_frame_dir, 
        args.preprocess, 
        source_image_flag=True, 
        pic_size=args.size
    )
    if first_coeff_path is None:
        raise ValueError("Error: Can't get the coeffs of the input.")

    return first_coeff_path, crop_pic_path, crop_info

def extract_reference_coeffs(preprocess_model, ref_eyeblink, ref_pose, save_dir):
    ref_eyeblink_coeff_path = generate_ref_coeff(preprocess_model, ref_eyeblink, save_dir)

    if ref_pose == ref_eyeblink:
        ref_pose_coeff_path = ref_eyeblink_coeff_path
    else:
        ref_pose_coeff_path = generate_ref_coeff(preprocess_model, ref_pose, save_dir)

    return ref_eyeblink_coeff_path, ref_pose_coeff_path

def generate_audio_to_coeff(
    audio_to_coeff, 
    first_coeff_path, 
    audio_path, 
    ref_eyeblink_coeff_path, 
    save_dir, 
    ref_pose_coeff_path
):
    batch = get_data(first_coeff_path, audio_path, ref_eyeblink_coeff_path, still=args.still)
    coeff_path = audio_to_coeff.generate(batch, save_dir, args.pose_style, ref_pose_coeff_path)
    return coeff_path

def generate_animation(
    animate_from_coeff, 
    coeff_path, 
    crop_pic_path, 
    first_coeff_path, 
    crop_info, 
    save_dir, 
    pic_path
):
    data = get_facerender_data(
        coeff_path,
        crop_pic_path,
        first_coeff_path,
        args.audio,
        args.batch_size,
        args.input_yaw,
        args.input_pitch,
        args.input_roll,
        expression_scale=args.expression_scale,
        still_mode=args.still,
        preprocess=args.preprocess,
        size=args.size,
    )

    result = animate_from_coeff.generate(
        data,
        save_dir,
        pic_path,
        crop_info,
        enhancer=args.enhancer,
        background_enhancer=None,
        preprocess=args.preprocess,
        img_size=args.size,
    )

    return result

def main():
    set_seed(args.seed)
    save_dir = os.path.join(args.result_dir, strftime("%Y_%m_%d_%H.%M.%S"))
    os.makedirs(save_dir, exist_ok=True)

    models = download_and_load_models()

    # init model
    preprocess_model = CropAndExtract(
        models["face3d_recon_net"], 
        models["face_align_net"], 
        models["retinaface_net"], 
        LM3D_PATH,
        use_onnx=args.onnx
    )
    audio_to_coeff = Audio2Coeff(
        models["audio2exp_net"], 
        models["audio2pose_net"],
        use_onnx=args.onnx
    )
    animate_from_coeff = AnimateFromCoeff(
        models["generator_net"], 
        models["kp_detector_net"], 
        models["mapping_net"],
        models["retinaface_net"],
        models["gfpgan_net"],
        use_onnx=args.onnx
    )

    # crop image and extract 3dmm coefficients
    print('3DMM Extraction for source image')
    first_coeff_path, crop_pic_path, crop_info = preprocess_image(
        preprocess_model, 
        args.input[0], 
        save_dir
    )

    # extract 3dmm coefficients of the reference video (for eye-blink and pose).
    ref_eyeblink_coeff_path, ref_pose_coeff_path = extract_reference_coeffs(
        preprocess_model, 
        args.ref_eyeblink, 
        args.ref_pose, 
        save_dir
    )

    # Generate coefficients for animation from audio data
    coeff_path = generate_audio_to_coeff(
        audio_to_coeff, 
        first_coeff_path, 
        args.audio,
        ref_eyeblink_coeff_path, 
        save_dir, 
        ref_pose_coeff_path
    )

    # generate animation
    result = generate_animation(
        animate_from_coeff, 
        coeff_path, 
        crop_pic_path, 
        first_coeff_path, 
        crop_info, 
        save_dir, 
        args.input[0]
    )
    
    shutil.move(result, args.savepath)
    print('The generated video is named:', args.savepath)

    if not args.verbose:
        shutil.rmtree(args.result_dir)

if __name__ == '__main__':
    main()
