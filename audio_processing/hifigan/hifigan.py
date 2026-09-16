import time
import sys
import argparse
import re

import numpy as np
import soundfile as sf
import librosa
import librosa.filters
from scipy.io.wavfile import write, read
from librosa.util import normalize
from librosa.filters import mel as librosa_mel_fn



import ailia  # noqa: E402

# import original modules
sys.path.append('../../util')
from arg_utils import get_base_parser, update_parser, get_savepath  # noqa: E402
from model_utils import check_and_download_models  # noqa: E402

# logger
from logging import getLogger   # noqa: E402
logger = getLogger(__name__)

# ======================
# PARAMETERS
# ======================

SAVE_WAV_PATH = 'output.wav'
INPUT_WAV_PATH="tests/test2.wav"
INPUT_NP_PATH="tests/test.npy"
REMOTE_PATH = "https://storage.googleapis.com/ailia-models/hifigan/"


# ======================
# Arguemnt Parser Config
# ======================

parser = get_base_parser( 'HIFI GAN', INPUT_WAV_PATH, SAVE_WAV_PATH)
# overwrite
parser.add_argument(
    '--input', '-i', metavar='TEXT', default=None,
    help='input text'
)
parser.add_argument(
    '--onnx', action='store_true',
    help='use onnx runtime'
)
parser.add_argument(
    '--inputType', default="numpy",
    help='[nupmy, wav]'
)
parser.add_argument(
    '-m', '--model',
    default='hifi',
    help='[hifi]'
)
parser.add_argument(
    '--profile', action='store_true',
    help='use profile model'
)
args = update_parser(parser, check_input_type=False)

if  args.model == "hifi":
    
    WEIGHT_PATH_hifi = 'generator_dynamic.onnx'
else:
    logger.error("unknown model")
    sys.exit()
    
    

MODEL_PATH_hifi =  WEIGHT_PATH_hifi+'.prototxt'   







# ======================
# Parameters
# ======================

if args.onnx:
    import onnxruntime
else:
    import ailia

if args.input:

    text = args.input[0]
    
            
else:
    
    if args.inputType != "numpy":
        text = INPUT_WAV_PATH
    else:
        text = INPUT_NP_PATH

#Parameters reqired to create mell spectograms
sampling_rate = 22050
segment_size = 8192
num_mels = 80
num_freq = 1025
n_fft = 1024
hop_size = 256
win_size = 1024

fmin = 0
fmax = 8000
MAX_WAV_VALUE = 32768.0



# ======================
# Functions
# ======================


def load_wav(full_path):
    sr, data = read(full_path)
    
    #Convertion of stereo to mono
    if data.ndim==2:
        data = np.mean(data, axis=1, dtype=data.dtype)
    return data, sr
    
def dynamic_range_compression(x, C=1, clip_val=1e-5):
    return np.log(np.clip(x, clip_val, None) * C)
    
    
def mel_spectrogram(y, n_fft, num_mels, sr, hop_size, win_size, fmin, fmax, center=False):
  
    if np.min(y) < -1.:
        print('min value is ', np.min(y))
    if np.max(y) > 1.:
        print('max value is ', np.max(y))
    mel_basis = {}
    hann_window = {}
    if fmax not in mel_basis:
        mel_X = librosa_mel_fn(sr=sr, n_fft=n_fft, n_mels=num_mels, fmin=fmin, fmax=fmax)
       
        mel_basis[str(fmax)] = mel_X
        hann_window[str(1)] = np.hanning(win_size)
    pad_size = (int((n_fft - hop_size) / 2), int((n_fft - hop_size) / 2))
    y = np.pad(y, ((0, 0), pad_size), mode='reflect')
    
    y=np.squeeze(y, axis=0)
    spec = librosa.stft(y, n_fft=n_fft, hop_length=hop_size, win_length=win_size, window=hann_window["1"],
                       center=False, pad_mode='reflect')
    spec = np.abs(spec)
    spec = np.dot(  mel_basis[str(fmax)], spec,)
    spec = dynamic_range_compression(spec)
    return np.expand_dims(spec, 0)
    

def get_mel(PATH_TO_MELL):
    return  np.load(PATH_TO_MELL)
    
def check_input_type():
    
    if args.input:
   
        if args.input[0].split(".")[1] in ["npy", "wav"]:
            if args.input[0].split(".")[1] == "npy":
                return "numpy"    
            else:
                return"wav"   
        else:
            return print("Unsupported input")
    else:
        return args.inputType
    



    
def generate_voice(hifi):
    # onnx
    sampling_Rate=sampling_rate
    inputTypes=check_input_type()
        
    
    if inputTypes != "numpy":
        wav, sr =load_wav(text)
        wav = wav / MAX_WAV_VALUE
        
        wav=np.expand_dims(wav, axis=0)
        mel_outputs = mel_spectrogram(wav, n_fft, num_mels, sampling_rate, hop_size, win_size, fmin, fmax, center=False)
        sampling_Rate = sr
    else:
        
        mel_outputs=get_mel(text)
    
    

    if args.benchmark:
        start = int(round(time.time() * 1000))

    if args.onnx:
        hifi_inputs = {hifi.get_inputs()[0].name: mel_outputs.astype(np.float32)}
        audio = hifi.run(None, hifi_inputs)[0]
    else:
        hifi.set_input_shape((1,80,mel_outputs.shape[2]))
        hifi_inputs = [mel_outputs]
        audio = hifi.run( hifi_inputs)[0]
        
    if args.benchmark:
        end = int(round(time.time() * 1000))
        estimation_time = (end - start)
        logger.info(f'\t Hifi processing time {estimation_time} ms')
    
    
    savepath = args.savepath
    logger.info(f'saved at : {savepath}')
    audio = audio.squeeze()
    
    audio = audio * MAX_WAV_VALUE
    audio = audio.astype('int16')
    sf.write(savepath, audio, sampling_Rate)
    logger.info('Script finished successfully.')



def main():
    # model files check and download
    
    check_and_download_models(WEIGHT_PATH_hifi, MODEL_PATH_hifi, REMOTE_PATH)

    #env_id = args.env_id

    if args.onnx:
        
        hifi = onnxruntime.InferenceSession(WEIGHT_PATH_hifi)
    else:
        hifi = ailia.Net(stream = MODEL_PATH_hifi, weight = WEIGHT_PATH_hifi, memory_mode = args.memory_mode, env_id = args.env_id)
        if args.profile:
            hifi.set_profile_mode(True)

    generate_voice( hifi)

    if args.profile:
       
        print("HIFI GAN : ")
        print(hifi.get_summary())

if __name__ == '__main__':
    main()
