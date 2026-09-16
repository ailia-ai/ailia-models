import time
import sys
import argparse

import numpy as np

import ailia  # noqa: E402

import soundfile as sf

# import original modules
sys.path.append('../../util')
from arg_utils import get_base_parser, update_parser, get_savepath  # noqa: E402
from model_utils import check_and_download_models  # noqa: E402

# logger
from logging import getLogger   # noqa: E402
logger = getLogger(__name__)


# ======================
# Parameters 1
# ======================
WAV_PATH = 'doublenoble_k7rain_part.wav' # noisy speech sample
#WAVE_PATH = '049 - Young Griffo - Facade.wav' # music sample
SAVE_WAV_PATH = 'separated_voice.wav'  
MODEL_LISTS = ['base', 'large']


# ======================
# Arguemnt Parser Config
# ======================
parser = get_base_parser(
    'RSource separation.',
    WAV_PATH,
    SAVE_WAV_PATH,
)
parser.add_argument(
    '-n', '--onnx', 
    action='store_true',
    default=False,
    help='Use onnxruntime'
)
parser.add_argument(
    '-st', '--stereo', 
    action='store_true',
    default=False,
    help='Use stereo mode'
)
parser.add_argument(
    '-a', '--arch', 
    default='base', choices=MODEL_LISTS,
    help='model lists: ' + ' | '.join(MODEL_LISTS)
)
parser.add_argument(
    '--ailia_audio', action='store_true',
    help='use ailia audio library'
)
args = update_parser(parser)

if args.ailia_audio:
    import ailia.audio as ailia_audio
    from unet_source_separation_utils_ailia import preemphasis, inv_preemphasis, lowpass, tfconvert, zero_pad, calc_time  # noqa: E402
else:
    from scipy import signal
    from unet_source_separation_utils import preemphasis, inv_preemphasis, lowpass, tfconvert, zero_pad, calc_time  # noqa: E402


# ======================
# Parameters 2
# ======================

if args.arch == 'base' : # for general voice separation
    WEIGHT_PATH = "second_voice_bank.best.opt2.onnx"
else :  # for singing voice separation
    WEIGHT_PATH = "RefineSpectrogramUnet.best.opt.onnx"
MODEL_PATH = WEIGHT_PATH + ".prototxt"
REMOTE_PATH = "https://storage.googleapis.com/ailia-models/unet_source_separation/"

# fixed parameters for each model
if args.arch == 'base' :
    DESIRED_SR = 22050
    MULT = 2 ** 5
    WINDOW_LEN = 512
    HOP_LEN = 64
else :
    DESIRED_SR = 44100
    MULT = 2 ** 6
    WINDOW_LEN = 1024
    HOP_LEN = 128 

# adjustable parameters 
if args.arch == 'base' :
    LPF_CUTOFF = 10000
else :
    LPF_CUTOFF = 20000


# ======================
# Main function
# ======================
def src_sep(data, session) :
    # inference
    if not args.onnx :
        sep = session.run(data)[0]

    else :
        first_input_name = session.get_inputs()[0].name
        second_input_name = session.get_inputs()[1].name
        first_output_name = session.get_outputs()[0].name
        sep = session.run(
            [first_output_name], 
            {first_input_name: data[0], second_input_name: data[1]})[0]

    return sep


def recognize_one_audio(input_path):
    # load audio
    logger.info('Loading wavfile...')
    wav, sr = sf.read(input_path)
    
    if wav.dtype != np.float32:
        wav = wav.astype(np.float32)

    if wav.ndim == 2 :
        if args.stereo:
            wav = np.transpose(wav,(1,0))   # stereo to batch
        else:
            wav = (wav[:,0][np.newaxis,:] + wav[:,1][np.newaxis,:])/2   # convert to mono
    else:
        wav = wav[np.newaxis,:]

    calc_time(wav.shape[1], sr)

    # convert sample rate
    logger.info('Converting sample rate...')
    if not sr == DESIRED_SR :
        if args.ailia_audio:
            wav = ailia.audio.resample(wav,sr,DESIRED_SR)
        else:
            wav = signal.resample_poly(wav, DESIRED_SR, sr, axis=1)

    # apply preenphasis filter
    logger.info('Generating input feature...')
    wav = preemphasis(wav)

    input_feature = tfconvert(wav, WINDOW_LEN, HOP_LEN, MULT)

    # create instance
    if not args.onnx :
        logger.info('Use ailia')
        env_id = args.env_id
        logger.info(f'env_id: {env_id}')
        session = ailia.Net(MODEL_PATH, WEIGHT_PATH, env_id=env_id, memory_mode=args.memory_mode)
    else :
        logger.info('Use onnxruntime')
        import onnxruntime
        session = onnxruntime.InferenceSession(WEIGHT_PATH)

    # inference
    logger.info('Start inference...')
    if args.benchmark:
        logger.info('BENCHMARK mode')
        for c in range(5) :
            start = int(round(time.time() * 1000))
            sep = src_sep(input_feature, session)
            end = int(round(time.time() * 1000))
            logger.info("\tprocessing time {} ms".format(end-start))
    else:
        sep = src_sep(input_feature, session)

    # postprocessing
    logger.info('Start postprocessing...')
    if LPF_CUTOFF > 0 :
        sep = lowpass(sep, LPF_CUTOFF, DESIRED_SR)

    out_wav = inv_preemphasis(sep).clip(-1.,1.)
    out_wav = out_wav.swapaxes(0,1)
    
    # save sapareted signal
    savepath = get_savepath(args.savepath, input_path)
    logger.info(f'saved at : {savepath}')

    sf.write(savepath, out_wav, DESIRED_SR)
    
    logger.info('Saved separated signal. ')
    logger.info('Script finished successfully.')


def main():
    # model files check and download
    check_and_download_models(WEIGHT_PATH, MODEL_PATH, REMOTE_PATH)

    for input_file in args.input:
        recognize_one_audio(input_file)

if __name__ == "__main__":
     main()
