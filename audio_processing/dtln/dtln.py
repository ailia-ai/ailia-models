import sys
import time
from logging import getLogger
import onnxruntime

import numpy as np
import soundfile as sf

import ailia

# import original modules
sys.path.append('../../util')
from model_utils import check_and_download_models  # noqa
from arg_utils import get_base_parser, get_savepath, update_parser  # noqa

logger = getLogger(__name__)

# ======================
# Parameters
# ======================

WEIGHT1_PATH = "dtln1.onnx"
MODEL1_PATH = "dtln1.onnx.prototxt"
WEIGHT2_PATH = "dtln2.onnx"
MODEL2_PATH = "dtln2.onnx.prototxt"
REMOTE_PATH = 'https://storage.googleapis.com/ailia-models/dtln/'

SAMPLE_RATE = 16000

WAV_PATH = '1221-135766-0000.wav'
SAVE_WAV_PATH = 'output.wav'

# ======================
# Arguemnt Parser Config
# ======================

parser = get_base_parser(
    'Dual-signal Transformation LSTM Network', WAV_PATH, SAVE_WAV_PATH, input_ftype='audio'
)
parser.add_argument(
    '--onnx',
    action='store_true',
    help='By default, the ailia SDK is used, but with this option, you can switch to using ONNX Runtime'
)
parser.add_argument(
    '--shift',
    default=128, type=int,
)
args = update_parser(parser)

block_shift = args.shift

# ======================
# Main functions
# ======================

def predict(audio,models):
    block_len = 512
    out_file = np.zeros((len(audio)))
    # create buffer
    in_buffer = np.zeros((block_len)).astype('float32')
    out_buffer = np.zeros((block_len)).astype('float32')
    # calculate number of blocks
    num_blocks = (audio.shape[0] - (block_len-block_shift)) // block_shift
    # iterate over the number of blcoks  
    time_array = []      

    inp_shape = [(1, 1, 257) ,(1, 2, 128, 2)]
    model_input_names_1 = ['input_2', 'input_3']
    interpreter_1 = models[0]
    interpreter_2 = models[1]
    model_inputs_1 = {}
    model_inputs_1[model_input_names_1[0]] = np.zeros(
                    [dim if isinstance(dim, int) else 1 for dim in inp_shape[0]],
                    dtype=np.float32)
    model_inputs_1[model_input_names_1[1]] = np.zeros(
                    [dim if isinstance(dim, int) else 1 for dim in inp_shape[1]],
                    dtype=np.float32)

    model_input_names_2 = ['input_4','input_5']
    model_inputs_2 = {}
    inp_shape = [(1, 1, 512) ,(1, 2, 128, 2)]
    model_inputs_2[model_input_names_2[0]] = np.zeros(
                [dim if isinstance(dim, int) else 1 for dim in inp_shape[0]],
                dtype=np.float32)
    model_inputs_2[model_input_names_2[1]] = np.zeros(
                    [dim if isinstance(dim, int) else 1 for dim in inp_shape[1]],
                    dtype=np.float32)


    for idx in range(num_blocks):
        start_time = time.time()
        # shift values and write to buffer
        in_buffer[:-block_shift] = in_buffer[block_shift:]
        in_buffer[-block_shift:] = audio[idx*block_shift:(idx*block_shift)+block_shift]
        # calculate fft of input block
        in_block_fft = np.fft.rfft(in_buffer)
        in_mag = np.abs(in_block_fft)
        in_phase = np.angle(in_block_fft)
        # reshape magnitude to input dimensions
        in_mag = np.reshape(in_mag, (1,1,-1)).astype('float32')
        # set block to input
        model_inputs_1[model_input_names_1[0]] = in_mag
        # run calculation 
        if args.onnx:
            model_outputs_1 = interpreter_1.run([],{'input_2':model_inputs_1['input_2'],'input_3':model_inputs_1['input_3']})
        else:
            inputs = [model_inputs_1['input_2'],model_inputs_1['input_3']]
            model_outputs_1 = interpreter_1.run(inputs)
        # get the output of the first block
        out_mask = model_outputs_1[0]
        # set out states back to input
        model_inputs_1[model_input_names_1[1]] = model_outputs_1[1]  
        # calculate the ifft
        estimated_complex = in_mag * out_mask * np.exp(1j * in_phase)
        estimated_block = np.fft.irfft(estimated_complex)
        # reshape the time domain block
        estimated_block = np.reshape(estimated_block, (1,1,-1)).astype('float32')
        # set tensors to the second block
        model_inputs_2[model_input_names_2[0]] = estimated_block
        # run calculation
        if args.onnx:
            model_outputs_2 = interpreter_2.run([],{'input_4':model_inputs_2['input_4'],'input_5':model_inputs_2['input_5']})
        else:
            inputs = [model_inputs_2['input_4'],model_inputs_2['input_5']]
            model_outputs_2 = interpreter_2.run(inputs)
        # get output
        out_block = model_outputs_2[0]
        # set out states back to input
        model_inputs_2[model_input_names_2[1]] = model_outputs_2[1]
        # shift values and write to buffer
        out_buffer[:-block_shift] = out_buffer[block_shift:]
        out_buffer[-block_shift:] = np.zeros((block_shift))
        out_buffer  += np.squeeze(out_block)
        # write block to output file
        out_file[idx*block_shift:(idx*block_shift)+block_shift] = out_buffer[:block_shift]
        time_array.append(time.time()-start_time)
    return out_file

def recognize_from_audio(models):

    inp_shape = [(1, 1, 257) ,(1, 2, 128, 2)]
    model_input_names_1 = ['input_2', 'input_3']

    model_inputs_1 = {}
    model_inputs_1[model_input_names_1[0]] = np.zeros(
                    [dim if isinstance(dim, int) else 1 for dim in inp_shape[0]],
                    dtype=np.float32)
    model_inputs_1[model_input_names_1[1]] = np.zeros(
                    [dim if isinstance(dim, int) else 1 for dim in inp_shape[1]],
                    dtype=np.float32)

    model_input_names_2 = ['input_4','input_5']
    model_inputs_2 = {}
    inp_shape = [(1, 1, 512) ,(1, 2, 128, 2)]
    model_inputs_2[model_input_names_2[0]] = np.zeros(
                [dim if isinstance(dim, int) else 1 for dim in inp_shape[0]],
                dtype=np.float32)
    model_inputs_2[model_input_names_2[1]] = np.zeros(
                    [dim if isinstance(dim, int) else 1 for dim in inp_shape[1]],
                    dtype=np.float32)


    # input audio loop
    for audio_path in args.input:
        logger.info(audio_path)

        # load audio file
        audio,fs = sf.read(audio_path)
        # check for sampling rate
        if fs != 16000:
            print('This model only supports 16k sampling rate.')
            continue

        # inference
        logger.info('Start inference...')
        if args.benchmark:
            logger.info('BENCHMARK mode')
            start = int(round(time.time() * 1000))
            output,sr = predict(audio, models)
            end = int(round(time.time() * 1000))
            estimation_time = (end - start)
            logger.info(f'\ttotal processing time {estimation_time} ms')
        else:
            output = predict(audio, models)

        # save result
        savepath = get_savepath(args.savepath, audio_path, ext='.wav')
        logger.info(f'saved at : {savepath}')
        sf.write(savepath, output, fs)

    logger.info('Script finished successfully.')


def main():
    check_and_download_models(WEIGHT1_PATH, MODEL1_PATH, REMOTE_PATH)
    check_and_download_models(WEIGHT2_PATH, MODEL2_PATH, REMOTE_PATH)

    env_id = args.env_id

    if args.onnx:
        models = [onnxruntime.InferenceSession(WEIGHT1_PATH),
                  onnxruntime.InferenceSession(WEIGHT2_PATH)]
    else:
        models = [ailia.Net(MODEL1_PATH,WEIGHT1_PATH, env_id = env_id, memory_mode = args.memory_mode),
                  ailia.Net(MODEL2_PATH,WEIGHT2_PATH, env_id = env_id, memory_mode = args.memory_mode)]

    # initialize
    recognize_from_audio(models)


if __name__ == '__main__':
    main()
