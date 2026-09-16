import sys
import time
from logging import getLogger

import numpy as np
import scipy.signal as signal
from PIL import Image
import librosa
import soundfile as sf

import ailia

# import original modules
sys.path.append('../../util')
sys.path.append('../crepe')
from microphone_utils import start_microphone_input  # noqa
from model_utils import check_and_download_models  # noqa
from arg_utils import get_base_parser, get_savepath, update_parser  # noqa

flg_ffmpeg = False

if flg_ffmpeg:
    import ffmpeg

logger = getLogger(__name__)

# ======================
# Parameters
# ======================

WEIGHT_HUBERT_PATH = "hubert_base.onnx"
MODEL_HUBERT_PATH = "hubert_base.onnx.prototxt"
WEIGHT_VC_PATH = "AISO-HOWATTO.onnx"
MODEL_VC_PATH = "AISO-HOWATTO.onnx.prototxt"
REMOTE_PATH = 'https://storage.googleapis.com/ailia-models/rvc/'

SAMPLE_RATE = 16000

WAV_PATH = 'booth.wav'
SAVE_WAV_PATH = 'output.wav'

# ======================
# Arguemnt Parser Config
# ======================

parser = get_base_parser(
    'Retrieval-based-Voice-Conversion', WAV_PATH, SAVE_WAV_PATH, input_ftype='audio'
)
parser.add_argument(
    '--tgt_sr', metavar="SR", type=int, default=40000,
    help='VC model sampling rate.',
)
parser.add_argument(
    '--f0', type=int, default=0, choices=(0, 1),
    help='f0 flag of VC model.',
)
parser.add_argument(
    '--sid', type=int, default=0,
    help='Select Speaker/Singer ID',
)
parser.add_argument(
    '--f0_up_key', metavar="N", type=int, default=0,
    help='Transpose (number of semitones, raise by an octave: 12, lower by an octave: -12)',
)
parser.add_argument(
    '--f0_method', default="pm", choices=("pm", "harvest", "crepe", "crepe_tiny"),
    help='Select the pitch extraction algorithm',
)
parser.add_argument(
    '--file_index', metavar="FILE", type=str, default=None,
    help='Path to the feature index file.',
)
parser.add_argument(
    '--index_rate', metavar="RATIO", type=float, default=0.75,
    help='Search feature ratio. (controls accent strength, too high has artifacting)',
)
parser.add_argument(
    '--filter_radius', metavar="N", type=int, default=3,
    help='If >=3: apply median filtering to the harvested pitch results. The value can reduce breathiness.',
)
parser.add_argument(
    '--resample_sr', metavar="SR", type=int, default=0,
    help='Resample the output audio. Set to 0 for no resampling.',
)
parser.add_argument(
    '--rms_mix_rate', metavar="RATE", type=float, default=0.25,
    help='Adjust the volume envelope scaling.',
)
parser.add_argument(
    '--protect', metavar="N", type=float, default=0.33,
    help='Protect voiceless consonants and breath sounds'
         ' to prevent artifacts such as tearing in electronic music.'
         ' Set to 0.5 to disable',
)
parser.add_argument(
    '-m', '--model_file', default=WEIGHT_VC_PATH,
    help='specify .onnx file'
)
parser.add_argument(
    '--version', default=1, choices=[1, 2], type=int,
    help='specify rvc version'
)
parser.add_argument(
    '--onnx',
    action='store_true',
    help='execute onnxruntime version.'
)
args = update_parser(parser)


class VCParam(object):
    def __init__(self, tgt_sr):
        self.x_pad, self.x_query, self.x_center, self.x_max = (
            3, 10, 60, 65
        )
        self.sr = 16000  # hubert输入采样率
        self.window = 160  # 每帧点数
        self.t_pad = self.sr * self.x_pad  # 每条前后pad时间
        self.t_pad_tgt = tgt_sr * self.x_pad
        self.t_pad2 = self.t_pad * 2
        self.t_query = self.sr * self.x_query  # 查询切点前后查询时间
        self.t_center = self.sr * self.x_center  # 查询切点位置
        self.t_max = self.sr * self.x_max  # 免查询时长阈值


# ======================
# Secondaty Functions
# ======================

def load_audio(file: str, sr: int = SAMPLE_RATE):
    if flg_ffmpeg:
        # https://github.com/openai/whisper/blob/main/whisper/audio.py#L26
        # This launches a subprocess to decode audio while down-mixing and resampling as necessary.
        # Requires the ffmpeg CLI and `ffmpeg-python` package to be installed.
        out, _ = ffmpeg.input(file, threads=0) \
            .output("-", format="f32le", acodec="pcm_f32le", ac=1, ar=sr) \
            .run(cmd=["ffmpeg", "-nostdin"], capture_stdout=True, capture_stderr=True)

        audio = np.frombuffer(out, np.float32).flatten()
    else:
        # prepare input data
        audio, source_sr = librosa.load(file, sr=None)
        # Resample the wav if needed
        if source_sr is not None and source_sr != sr:
            audio = librosa.resample(audio, orig_sr=source_sr, target_sr=sr)

    return audio


def change_rms(data1, sr1, data2, sr2, rate):  # 1是输入音频，2是输出音频,rate是2的占比
    rms1 = librosa.feature.rms(
        y=data1, frame_length=sr1 // 2 * 2, hop_length=sr1 // 2
    )  # 每半秒一个点
    rms2 = librosa.feature.rms(y=data2, frame_length=sr2 // 2 * 2, hop_length=sr2 // 2)

    rms1 = np.array(Image.fromarray(rms1).resize((data2.shape[0], 1), Image.Resampling.BILINEAR))
    rms1 = rms1.flatten()
    rms2 = np.array(Image.fromarray(rms2).resize((data2.shape[0], 1), Image.Resampling.BILINEAR))
    rms2 = rms2.flatten()

    r = np.zeros(rms2.shape) + 1e-6
    rms2 = np.where(rms2 > r, rms2, r)

    data2 *= np.power(rms1, 1 - rate) * np.power(rms2, rate - 1)

    return data2


# ======================
# Main functions
# ======================

def get_f0(
        vc_param,
        x,
        p_len,
        f0_up_key,
        f0_method,
        filter_radius,
        inp_f0=None):
    time_step = vc_param.window / vc_param.sr * 1000
    f0_min = 50
    f0_max = 1100
    f0_mel_min = 1127 * np.log(1 + f0_min / 700)
    f0_mel_max = 1127 * np.log(1 + f0_max / 700)

    if f0_method == "pm":
        import parselmouth

        f0 = (
            parselmouth.Sound(x, vc_param.sr).to_pitch_ac(
                time_step=time_step / 1000,
                voicing_threshold=0.6,
                pitch_floor=f0_min,
                pitch_ceiling=f0_max,
            ).selected_array["frequency"]
        )
        pad_size = (p_len - len(f0) + 1) // 2
        if pad_size > 0 or p_len - len(f0) - pad_size > 0:
            f0 = np.pad(
                f0, [[pad_size, p_len - len(f0) - pad_size]], mode="constant"
            )
    elif f0_method == "harvest":
        import pyworld

        audio = x.astype(np.double)
        fs = vc_param.sr
        frame_period = 10
        f0, t = pyworld.harvest(
            audio,
            fs=fs,
            f0_ceil=f0_max,
            f0_floor=f0_min,
            frame_period=frame_period,
        )
        f0 = pyworld.stonemask(audio, f0, t, fs)

        if filter_radius > 2:
            f0 = signal.medfilt(f0, 3)
    elif f0_method == "crepe" or f0_method == "crepe_tiny":
        import mod_crepe

        # Pick a batch size that doesn't cause memory errors on your gpu
        batch_size = 512
        audio = np.copy(x)[None]
        f0, pd = mod_crepe.predict(
            audio,
            vc_param.sr,
            vc_param.window,
            f0_min,
            f0_max,
            batch_size=batch_size,
            return_periodicity=True,
        )
        pd = mod_crepe.median(pd, 3)
        f0 = mod_crepe.mean(f0, 3)
        f0[pd < 0.1] = 0
        f0 = f0[0]
    else:
        raise ValueError("f0_method: %s" % f0_method)

    f0 *= pow(2, f0_up_key / 12)

    tf0 = vc_param.sr // vc_param.window  # 每秒f0点数
    if inp_f0 is not None:
        delta_t = np.round(
            (inp_f0[:, 0].max() - inp_f0[:, 0].min()) * tf0 + 1
        ).astype("int16")
        replace_f0 = np.interp(
            list(range(delta_t)), inp_f0[:, 0] * 100, inp_f0[:, 1]
        )
        shape = f0[vc_param.x_pad * tf0: vc_param.x_pad * tf0 + len(replace_f0)].shape[0]
        f0[vc_param.x_pad * tf0: vc_param.x_pad * tf0 + len(replace_f0)] = \
            replace_f0[:shape]

    f0bak = f0.copy()
    f0_mel = 1127 * np.log(1 + f0 / 700)
    f0_mel[f0_mel > 0] = \
        (f0_mel[f0_mel > 0] - f0_mel_min) * 254 \
        / (f0_mel_max - f0_mel_min) + 1
    f0_mel[f0_mel <= 1] = 1
    f0_mel[f0_mel > 255] = 255
    f0_coarse = np.rint(f0_mel).astype(int)

    return f0_coarse, f0bak  # 1-0


def vc(
        hubert,
        net_g,
        sid,
        audio0,
        pitch,
        pitchf,
        vc_param,
        index,
        big_npy,
        index_rate,
        protect):
    feats = audio0.reshape(1, -1).astype(np.float32)
    padding_mask = np.zeros(feats.shape, dtype=bool)

    # feedforward
    if not args.onnx:
        output = hubert.predict([feats, padding_mask])
    else:
        output = hubert.run(None, {'source': feats, 'padding_mask': padding_mask})

    if args.version == 1:
        feats = output[0] # v1 : 256
    elif args.version == 2:
        feats = hubert.get_blob_data(hubert.find_blob_index_by_name("/encoder/Slice_5_output_0")) # v2 : 768

    if protect < 0.5 and pitch is not None and pitchf is not None:
        feats0 = np.copy(feats)

    if isinstance(index, type(None)) is False \
            and isinstance(big_npy, type(None)) is False \
            and index_rate > 0:
        x = feats[0]

        score, ix = index.search(x, k=8)
        weight = np.square(1 / score)
        weight /= weight.sum(axis=1, keepdims=True)
        x = np.sum(big_npy[ix] * np.expand_dims(weight, axis=2), axis=1)

        feats = (
                np.expand_dims(x, axis=0) * index_rate
                + (1 - index_rate) * feats
        )

    # interpolate
    new_feats = np.zeros((feats.shape[0], feats.shape[1] * 2, feats.shape[2]), dtype=np.float32)
    for i in range(feats.shape[1]):
        new_feats[:, i * 2 + 0, :] = feats[:, i, :]
        new_feats[:, i * 2 + 1, :] = feats[:, i, :]
    feats = new_feats

    if protect < 0.5 and pitch is not None and pitchf is not None:
        # interpolate
        new_feats = np.zeros((feats0.shape[0], feats0.shape[1] * 2, feats0.shape[2]), dtype=np.float32)
        for i in range(feats0.shape[1]):
            new_feats[:, i * 2 + 0, :] = feats0[:, i, :]
            new_feats[:, i * 2 + 1, :] = feats0[:, i, :]
        feats0 = new_feats

    p_len = audio0.shape[0] // vc_param.window
    if feats.shape[1] < p_len:
        p_len = feats.shape[1]
        if pitch is not None and pitchf is not None:
            pitch = pitch[:, :p_len]
            pitchf = pitchf[:, :p_len]

    if protect < 0.5 and pitch is not None and pitchf is not None:
        pitchff = np.copy(pitchf)
        pitchff[pitchf > 0] = 1
        pitchff[pitchf < 1] = protect
        pitchff = np.expand_dims(pitchff, axis=-1)
        feats = feats * pitchff + feats0 * (1 - pitchff)

    p_len = np.array([p_len], dtype=int)

    # feedforward
    rnd = np.random.randn(1, 192, p_len[0]).astype(np.float32) * 0.66666  # 噪声（加入随机因子）
    if pitch is not None and pitchf is not None:
        if not args.onnx:
            output = net_g.predict([feats, p_len, pitch, pitchf, sid, rnd])
        else:
            output = net_g.run(None, {
                'phone': feats, 'phone_lengths': p_len,
                'pitch': pitch, 'pitchf': pitchf,
                'ds': sid, 'rnd': rnd
            })
    else:
        if not args.onnx:
            output = net_g.predict([feats, p_len, sid, rnd])
        else:
            output = net_g.run(None, {
                'phone': feats, 'phone_lengths': p_len, 'ds': sid, 'rnd': rnd
            })
    audio1 = output[0][0, 0]

    return audio1


bh, ah = signal.butter(N=5, Wn=48, btype="high", fs=16000)


def predict(audio, models, tgt_sr=40000, if_f0=0):
    audio_max = np.abs(audio).max() / 0.95
    if audio_max > 1:
        audio /= audio_max

    sid = args.sid
    file_index = args.file_index
    index_rate = args.index_rate
    resample_sr = args.resample_sr
    rms_mix_rate = args.rms_mix_rate
    protect = args.protect
    f0_up_key = args.f0_up_key
    f0_method = args.f0_method
    filter_radius = args.filter_radius
    inp_f0 = None

    vc_param = VCParam(tgt_sr)

    index = big_npy = None
    if file_index and index_rate > 0:
        import faiss
        try:
            index = faiss.read_index(file_index)
            big_npy = index.reconstruct_n(0, index.ntotal)
        except Exception as e:
            logger.exception(e)

    audio = signal.filtfilt(bh, ah, audio)
    audio_pad = np.pad(audio, (vc_param.window // 2, vc_param.window // 2), mode="reflect")

    opt_ts = []
    if audio_pad.shape[0] > vc_param.t_max:
        audio_sum = np.zeros_like(audio)
        for i in range(vc_param.window):
            audio_sum += audio_pad[i: i - vc_param.window]
        for t in range(vc_param.t_center, audio.shape[0], vc_param.t_center):
            opt_ts.append(
                t - vc_param.t_query
                + np.where(
                    np.abs(audio_sum[t - vc_param.t_query: t + vc_param.t_query])
                    == np.abs(audio_sum[t - vc_param.t_query: t + vc_param.t_query]).min()
                )[0][0]
            )

    s = 0
    audio_opt = []
    t = None
    audio_pad = np.pad(audio, (vc_param.t_pad, vc_param.t_pad), mode="reflect")
    p_len = audio_pad.shape[0] // vc_param.window

    pitch, pitchf = None, None
    if if_f0 == 1:
        pitch, pitchf = get_f0(
            vc_param,
            audio_pad,
            p_len,
            f0_up_key,
            f0_method,
            filter_radius,
            inp_f0,
        )
        pitch = pitch[:p_len]
        pitchf = pitchf[:p_len]
        pitch = np.expand_dims(pitch, axis=0)
        pitchf = np.expand_dims(pitchf, axis=0)
        pitchf = pitchf.astype(np.float32)

    sid = np.array([sid], dtype=int)
    for t in opt_ts:
        t = t // vc_param.window * vc_param.window
        audio1 = vc(
            models["hubert"],
            models["net_g"],
            sid,
            audio_pad[s: t + vc_param.t_pad2 + vc_param.window],
            pitch[:, s // vc_param.window: (t + vc_param.t_pad2) // vc_param.window]
            if if_f0 == 1 else None,
            pitchf[:, s // vc_param.window: (t + vc_param.t_pad2) // vc_param.window]
            if if_f0 == 1 else None,
            vc_param,
            index,
            big_npy,
            index_rate,
            protect,
        )
        audio_opt.append(audio1[vc_param.t_pad_tgt: -vc_param.t_pad_tgt])
        s = t
    audio1 = vc(
        models["hubert"],
        models["net_g"],
        sid,
        audio_pad[t:],
        (pitch[:, t // vc_param.window:] if t is not None else pitch)
        if if_f0 == 1 else None,
        (pitchf[:, t // vc_param.window:] if t is not None else pitchf)
        if if_f0 == 1 else None,
        vc_param,
        index,
        big_npy,
        index_rate,
        protect,
    )
    audio_opt.append(audio1[vc_param.t_pad_tgt: -vc_param.t_pad_tgt])
    audio_opt = np.concatenate(audio_opt)
    audio_opt = audio_opt.astype(np.float32)

    if rms_mix_rate < 1:
        audio_opt = change_rms(audio, 16000, audio_opt, tgt_sr, rms_mix_rate)
    if 16000 <= resample_sr != tgt_sr:
        audio_opt = librosa.resample(
            audio_opt, orig_sr=tgt_sr, target_sr=resample_sr
        )
        tgt_sr = resample_sr

    audio_max = np.abs(audio_opt).max() / 0.99
    max_int16 = 32768
    if audio_max > 1:
        max_int16 /= audio_max
    audio_opt = (audio_opt * max_int16).astype(np.int16)

    return audio_opt, tgt_sr


def recognize_from_audio(models):
    # Depend on voice model
    tgt_sr = args.tgt_sr
    if_f0 = args.f0

    # input audio loop
    for audio_path in args.input:
        logger.info(audio_path)

        # prepare input data
        audio = load_audio(audio_path, SAMPLE_RATE)

        # inference
        logger.info('Start inference...')
        if args.benchmark:
            logger.info('BENCHMARK mode')
            start = int(round(time.time() * 1000))
            output, sr = predict(audio, models, tgt_sr, if_f0)
            end = int(round(time.time() * 1000))
            estimation_time = (end - start)
            logger.info(f'\ttotal processing time {estimation_time} ms')
        else:
            output, sr = predict(audio, models, tgt_sr, if_f0)

        # save result
        savepath = get_savepath(args.savepath, audio_path, ext='.wav')
        logger.info(f'saved at : {savepath}')
        sf.write(savepath, output, sr)

    logger.info('Script finished successfully.')


def main():
    WEIGHT_VC_PATH = args.model_file
    MODEL_VC_PATH = WEIGHT_VC_PATH.replace(".onnx", ".onnx.prototxt")
    check_and_download_models(WEIGHT_HUBERT_PATH, MODEL_HUBERT_PATH, REMOTE_PATH)
    check_and_download_models(WEIGHT_VC_PATH, MODEL_VC_PATH, REMOTE_PATH)

    if args.f0 == 1 and (args.f0_method == "crepe" or  args.f0_method == "crepe_tiny"):
        from mod_crepe import WEIGHT_CREPE_PATH, MODEL_CREPE_PATH, WEIGHT_CREPE_TINY_PATH, MODEL_CREPE_TINY_PATH
        if args.f0_method == "crepe_tiny":
            check_and_download_models(WEIGHT_CREPE_TINY_PATH, MODEL_CREPE_TINY_PATH, REMOTE_PATH)
        else:
            check_and_download_models(WEIGHT_CREPE_PATH, MODEL_CREPE_PATH, REMOTE_PATH)

    env_id = args.env_id

    # initialize
    if not args.onnx:
        # hubert reads an intermediate blob (/encoder/Slice_5_output_0) with
        # get_blob_data(), so REUSE_INTERSTAGE cannot be used
        hubert = ailia.Net(MODEL_HUBERT_PATH, WEIGHT_HUBERT_PATH, env_id=env_id,
                           memory_mode=ailia.get_memory_mode(reduce_constant=True))
        net_g = ailia.Net(MODEL_VC_PATH, WEIGHT_VC_PATH, env_id=env_id, memory_mode=args.memory_mode)
        if args.profile:
            hubert.set_profile_mode(True)
            net_g.set_profile_mode(True)
    else:
        import onnxruntime
        providers = ["CPUExecutionProvider", "CUDAExecutionProvider"]
        hubert = onnxruntime.InferenceSession(WEIGHT_HUBERT_PATH, providers=providers)
        net_g = onnxruntime.InferenceSession(WEIGHT_VC_PATH, providers=providers)

    if args.f0 == 1 and (args.f0_method == "crepe" or args.f0_method == "crepe_tiny"):
        import mod_crepe
        f0_model = mod_crepe.load_model(env_id, args.onnx, args.f0_method == "crepe_tiny",
                                        memory_mode=args.memory_mode)
        if args.profile:
            f0_model.set_profile_mode(True)
    else:
        f0_model = None

    models = {
        "hubert": hubert,
        "net_g": net_g,
    }

    recognize_from_audio(models)

    if args.profile and not args.onnx:
        print("--- profile hubert")
        print(hubert.get_summary())
        print("")
        print("--- profile net_g")
        print(net_g.get_summary())
        print("")
        if f0_model != None:
            print("--- profile f0_model")
            print(f0_model.get_summary())
            print("")


if __name__ == '__main__':
    main()
