import sys
import math
import wave
import struct
from logging import getLogger

import numpy as np
from tqdm import tqdm

import ailia

# import original modules
sys.path.append('../../util')
from arg_utils import get_base_parser, update_parser, get_savepath  # noqa
from model_utils import check_and_download_models  # noqa

from kiss_fft import Complex, opus_fft_alloc_twiddles, opus_fft
from pitch import pitch_downsample, pitch_search, remove_doubling

logger = getLogger(__name__)

# ======================
# Parameters
# ======================

WEIGHT_PATH = 'rnn_model.onnx'
MODEL_PATH = 'rnn_model.onnx.prototxt'
REMOTE_PATH = 'https://storage.googleapis.com/ailia-models/rnnoise/'

AUDIO_PATH = 'babble_15dB.wav'
OUTPUT_PATH = 'denoised.wav'

PITCH_MIN_PERIOD = 60
PITCH_MAX_PERIOD = 768
PITCH_FRAME_SIZE = 960
PITCH_BUF_SIZE = PITCH_MAX_PERIOD + PITCH_FRAME_SIZE

NB_BANDS = 22
CEPS_MEM = 8
NB_DELTA_CEPS = 6
NB_FEATURES = NB_BANDS + 3 * NB_DELTA_CEPS + 2

FRAME_SIZE_SHIFT = 2
FRAME_SIZE = 120 << FRAME_SIZE_SHIFT
WINDOW_SIZE = 2 * FRAME_SIZE
FREQ_SIZE = FRAME_SIZE + 1

# ======================
# Arguemnt Parser Config
# ======================

parser = get_base_parser(
    'rnnoise', AUDIO_PATH, OUTPUT_PATH
)
parser.add_argument(
    '--onnx',
    action='store_true',
    help='execute onnxruntime version.'
)
args = update_parser(parser)


# ======================
# Secondaty Functions
# ======================

class CommonState:
    init = False
    kfft = None
    half_window = np.zeros(FRAME_SIZE)
    dct_table = np.zeros(NB_BANDS * NB_BANDS)


common = CommonState()


class DenoiseState:
    analysis_mem = np.zeros(FRAME_SIZE)
    cepstral_mem = np.zeros((CEPS_MEM, NB_BANDS))
    memid = 0
    synthesis_mem = np.zeros(FRAME_SIZE)
    pitch_buf = np.zeros(PITCH_BUF_SIZE)
    pitch_enh_buf = np.zeros(PITCH_BUF_SIZE)
    last_gain = 0.0
    last_period = 0
    mem_hp_x = np.zeros(2)


def compute_band_energy(bandE, X):
    eband5ms = [
        0, 1, 2, 3, 4, 5, 6, 7, 8, 10, 12, 14, 16, 20, 24, 28, 34, 40, 48, 60, 78, 100
    ]

    _sum = [0] * NB_BANDS
    for i in range(NB_BANDS - 1):
        band_size = (eband5ms[i + 1] - eband5ms[i]) << FRAME_SIZE_SHIFT
        for j in range(band_size):
            frac = j / band_size
            tmp = X[(eband5ms[i] << FRAME_SIZE_SHIFT) + j].r ** 2
            tmp += X[(eband5ms[i] << FRAME_SIZE_SHIFT) + j].i ** 2
            _sum[i] += (1 - frac) * tmp
            _sum[i + 1] += frac * tmp

    _sum[0] *= 2
    _sum[NB_BANDS - 1] *= 2
    for i in range(NB_BANDS):
        bandE[i] = _sum[i]


eband5ms = [
    0, 1, 2, 3, 4, 5, 6, 7, 8, 10, 12, 14, 16, 20, 24, 28, 34, 40, 48, 60, 78, 100
]


def compute_band_corr(bandE, X, P):
    _sum = [0] * NB_BANDS

    for i in range(NB_BANDS - 1):
        band_size = (eband5ms[i + 1] - eband5ms[i]) << FRAME_SIZE_SHIFT
        for j in range(band_size):
            frac = j / band_size
            tmp = X[(eband5ms[i] << FRAME_SIZE_SHIFT) + j].r * P[(eband5ms[i] << FRAME_SIZE_SHIFT) + j].r
            tmp += X[(eband5ms[i] << FRAME_SIZE_SHIFT) + j].i * P[(eband5ms[i] << FRAME_SIZE_SHIFT) + j].i
            _sum[i] += (1 - frac) * tmp
            _sum[i + 1] += frac * tmp

    _sum[0] *= 2
    _sum[NB_BANDS - 1] *= 2
    for i in range(NB_BANDS):
        bandE[i] = _sum[i]


def interp_band_gain(g, bandE):
    g[...] = 0
    for i in range(NB_BANDS - 1):
        band_size = (eband5ms[i + 1] - eband5ms[i]) << FRAME_SIZE_SHIFT
        for j in range(band_size):
            frac = j / band_size
            g[(eband5ms[i] << FRAME_SIZE_SHIFT) + j] = (1 - frac) * bandE[i] + frac * bandE[i + 1]


def check_init():
    if common.init:
        return

    common.kfft = opus_fft_alloc_twiddles(2 * FRAME_SIZE)

    for i in range(FRAME_SIZE):
        common.half_window[i] = math.sin(
            .5 * math.pi * math.sin(.5 * math.pi * (i + .5) / FRAME_SIZE)
            * math.sin(.5 * math.pi * (i + .5) / FRAME_SIZE)
        )

    for i in range(NB_BANDS):
        for j in range(NB_BANDS):
            common.dct_table[i * NB_BANDS + j] = math.cos((i + .5) * j * math.pi / NB_BANDS)
            if j == 0:
                common.dct_table[i * NB_BANDS + j] *= math.sqrt(.5)

    common.init = True


def dct(out, in_data):
    check_init()

    for i in range(NB_BANDS):
        _sum = 0
        for j in range(NB_BANDS):
            _sum += in_data[j] * common.dct_table[j * NB_BANDS + i]
        out[i] = _sum * math.sqrt(2. / 22)

    return out


def forward_transform(out, in_data):
    check_init()

    x = [Complex() for _ in range(WINDOW_SIZE)]
    y = [Complex() for _ in range(WINDOW_SIZE)]

    for i in range(WINDOW_SIZE):
        x[i].r = in_data[i]
        x[i].i = 0

    opus_fft(common.kfft, x, y)
    for i in range(FREQ_SIZE):
        out[i] = y[i]


def inverse_transform(out, in_data):
    check_init()

    x = [Complex() for _ in range(WINDOW_SIZE)]
    y = [Complex() for _ in range(WINDOW_SIZE)]

    for i in range(FREQ_SIZE):
        x[i] = in_data[i]
    for i in range(i + 1, WINDOW_SIZE):
        x[i].r = x[WINDOW_SIZE - i].r
        x[i].i = -x[WINDOW_SIZE - i].i

    opus_fft(common.kfft, x, y)

    # output in reverse order for IFFT.
    out[0] = WINDOW_SIZE * y[0].r
    for i in range(1, WINDOW_SIZE):
        out[i] = WINDOW_SIZE * y[WINDOW_SIZE - i].r


def apply_window(x):
    check_init()

    for i in range(FRAME_SIZE):
        x[i] *= common.half_window[i]
        x[WINDOW_SIZE - 1 - i] *= common.half_window[i]


def frame_analysis(st, X, Ex, in_data):
    x = np.zeros(WINDOW_SIZE)
    x[:FRAME_SIZE] = st.analysis_mem
    x[FRAME_SIZE:] = in_data
    st.analysis_mem[...] = in_data

    apply_window(x)
    forward_transform(X, x)

    compute_band_energy(Ex, X)


def compute_frame_features(st, X, P, Ex, Ep, Exp, features, x):
    E = 0
    spec_variability = 0
    Ly = np.zeros(NB_BANDS)
    p = np.zeros(WINDOW_SIZE)
    pitch_buf = np.zeros(PITCH_BUF_SIZE >> 1)
    tmp = np.zeros(NB_BANDS)

    frame_analysis(st, X, Ex, x)

    st.pitch_buf[:PITCH_BUF_SIZE - FRAME_SIZE] = st.pitch_buf[FRAME_SIZE:]
    st.pitch_buf[PITCH_BUF_SIZE - FRAME_SIZE:] = x
    pre = [st.pitch_buf]
    pitch_downsample(pre, pitch_buf, PITCH_BUF_SIZE, 1)
    pitch_index = pitch_search(
        pitch_buf[PITCH_MAX_PERIOD >> 1:], pitch_buf, PITCH_FRAME_SIZE,
        PITCH_MAX_PERIOD - 3 * PITCH_MIN_PERIOD)
    pitch_index = PITCH_MAX_PERIOD - pitch_index

    p_pitch_index = [pitch_index]
    gain = remove_doubling(
        pitch_buf, PITCH_MAX_PERIOD, PITCH_MIN_PERIOD,
        PITCH_FRAME_SIZE, p_pitch_index, st.last_period, st.last_gain)
    st.last_period = pitch_index = p_pitch_index[0]
    st.last_gain = gain

    for i in range(WINDOW_SIZE):
        p[i] = st.pitch_buf[PITCH_BUF_SIZE - WINDOW_SIZE - pitch_index + i]
    apply_window(p)
    forward_transform(P, p)
    compute_band_energy(Ep, P)
    compute_band_corr(Exp, X, P)
    for i in range(NB_BANDS):
        Exp[i] = Exp[i] / math.sqrt(.001 + Ex[i] * Ep[i])
    dct(tmp, Exp)

    for i in range(NB_DELTA_CEPS):
        features[NB_BANDS + 2 * NB_DELTA_CEPS + i] = tmp[i]
    features[NB_BANDS + 2 * NB_DELTA_CEPS] -= 1.3
    features[NB_BANDS + 2 * NB_DELTA_CEPS + 1] -= 0.9
    features[NB_BANDS + 3 * NB_DELTA_CEPS] = .01 * (pitch_index - 300)

    logMax = -2
    follow = -2
    for i in range(NB_BANDS):
        Ly[i] = math.log10(1e-2 + Ex[i])
        Ly[i] = max(logMax - 7, max(follow - 1.5, Ly[i]))
        logMax = max(logMax, Ly[i])
        follow = max(follow - 1.5, Ly[i])
        E += Ex[i]

    if E < 0.04:
        # If there's no audio, avoid messing up the state.
        features[...] = 0
        return 1

    dct(features, Ly)

    features[0] -= 12
    features[1] -= 4
    ceps_0 = st.cepstral_mem[st.memid]
    ceps_1 = st.cepstral_mem[CEPS_MEM + st.memid - 1] \
        if st.memid < 1 else st.cepstral_mem[st.memid - 1]
    ceps_2 = st.cepstral_mem[CEPS_MEM + st.memid - 2] \
        if st.memid < 2 else st.cepstral_mem[st.memid - 2]
    for i in range(NB_BANDS):
        ceps_0[i] = features[i]
    st.memid += 1

    for i in range(NB_DELTA_CEPS):
        features[i] = ceps_0[i] + ceps_1[i] + ceps_2[i]
        features[NB_BANDS + i] = ceps_0[i] - ceps_2[i]
        features[NB_BANDS + NB_DELTA_CEPS + i] = ceps_0[i] - 2 * ceps_1[i] + ceps_2[i]

    # Spectral variability features.
    if st.memid == CEPS_MEM:
        st.memid = 0

    for i in range(CEPS_MEM):
        mindist = 1e15
        for j in range(CEPS_MEM):
            dist = 0.
            for k in range(NB_BANDS):
                tmp = st.cepstral_mem[i][k] - st.cepstral_mem[j][k]
                dist += tmp * tmp
            if j != i:
                mindist = min(mindist, dist)

        spec_variability += mindist

    features[NB_BANDS + 3 * NB_DELTA_CEPS + 1] = spec_variability / CEPS_MEM - 2.1

    return E < 0.1


def frame_synthesis(st, out, y):
    x = np.zeros(WINDOW_SIZE)
    inverse_transform(x, y)
    apply_window(x)
    for i in range(FRAME_SIZE):
        out[i] = x[i] + st.synthesis_mem[i]
    st.synthesis_mem[...] = x[FRAME_SIZE:]


def biquad(y, mem, x, b, a, N):
    for i in range(N):
        xi = x[i]
        yi = x[i] + mem[0]
        mem[0] = mem[1] + (b[0] * xi - a[0] * yi)
        mem[1] = b[1] * xi - a[1] * yi
        y[i] = yi


def pitch_filter(X, P, Ex, Ep, Exp, g):
    r = np.zeros(NB_BANDS)
    rf = np.zeros(FREQ_SIZE)

    for i in range(NB_BANDS):
        if Exp[i] > g[i]:
            r[i] = 1
        else:
            r[i] = Exp[i] ** 2 * (1 - g[i] ** 2) / (.001 + (g[i] ** 2) * (1 - Exp[i] ** 2))
        r[i] = math.sqrt(min(1, max(0, r[i])))
        r[i] *= math.sqrt(Ex[i] / (1e-8 + Ep[i]))

    interp_band_gain(rf, r)
    for i in range(FREQ_SIZE):
        X[i].r += rf[i] * P[i].r
        X[i].i += rf[i] * P[i].i

    newE = np.zeros(NB_BANDS)
    compute_band_energy(newE, X)
    norm = np.zeros(NB_BANDS)
    normf = np.zeros(FREQ_SIZE)
    for i in range(NB_BANDS):
        norm[i] = math.sqrt(Ex[i] / (1e-8 + newE[i]))

    interp_band_gain(normf, norm)
    for i in range(FREQ_SIZE):
        X[i].r *= normf[i]
        X[i].i *= normf[i]


# ======================
# Main functions
# ======================

def preprocess(st, data):
    X = [Complex() for _ in range(FREQ_SIZE)]
    P = [Complex() for _ in range(WINDOW_SIZE)]
    x = np.zeros(FRAME_SIZE)
    Ex = np.zeros(NB_BANDS)
    Ep = np.zeros(NB_BANDS)
    Exp = np.zeros(NB_BANDS)
    features = np.zeros(NB_FEATURES)

    a_hp = (-1.99599, 0.99600)
    b_hp = (-2., 1.)
    biquad(x, st.mem_hp_x, data, b_hp, a_hp, FRAME_SIZE)
    compute_frame_features(st, X, P, Ex, Ep, Exp, features, x)

    return X, P, Ex, Ep, Exp, features


def postprocess(st, pp, gains, vad_prob):
    outputs = []
    for p, g, prob in zip(pp, gains, vad_prob):
        X = p["X"]
        P = p["P"]
        Ex = p["Ex"]
        Ep = p["Ep"]
        Exp = p["Exp"]
        gf = np.ones(FREQ_SIZE)
        pitch_filter(X, P, Ex, Ep, Exp, g)
        interp_band_gain(gf, g)

        for i in range(FREQ_SIZE):
            X[i].r *= gf[i]
            X[i].i *= gf[i]

        out = np.zeros(FRAME_SIZE)
        frame_synthesis(st, out, X)

        outputs.append(out)

    return outputs


def rnnoise_process_frame(net, x):
    x = np.array(x, dtype=np.float32)
    if x.shape[0] < 100:
        x = np.concatenate([
            x,
            np.zeros((100 - x.shape[0], NB_FEATURES), dtype=np.float32)
        ])

    x = np.expand_dims(x, axis=0)

    # feedforward
    if not args.onnx:
        output = net.predict([x])
    else:
        output = net.run(None, {'main_input:0': x})
    gains, vad_prob = output

    return gains[0], vad_prob[0]


def recognize_from_audio(net):
    wav_path = args.input[0]
    logger.info(wav_path)

    logger.info('Start inference...')
    wf = wave.open(wav_path, "rb")

    save_path = get_savepath(args.savepath, wav_path, ext='.wav')
    wf_out = wave.open(save_path, "wb")
    wf_out.setnchannels(1)
    wf_out.setsampwidth(16 // 8)
    wf_out.setframerate(48000)

    pp = []
    st = DenoiseState()
    bar = tqdm(total=wf.getnframes())
    while True:
        buf = wf.readframes(FRAME_SIZE)
        if not buf:
            break
        data = np.frombuffer(buf, dtype=np.int16)

        X, P, Ex, Ep, Exp, feat = preprocess(st, data)
        pp.append(dict(
            X=X,
            P=P,
            Ex=Ex,
            Ep=Ep,
            Exp=Exp,
            feat=feat
        ))

        if len(pp) == 100:
            x = [p["feat"] for p in pp]
            gains, vad_prob = rnnoise_process_frame(net, x)
            outputs = postprocess(st, pp, gains, vad_prob)
            pp.clear()

            for out in outputs:
                out = np.array(out, dtype=int)
                out = np.clip(out, (-0x7fff - 1), 0x7fff)
                out = struct.pack("h" * len(out), *out)
                wf_out.writeframes(out)

        bar.update(len(data))

    if 0 < len(pp):
        x = [p["feat"] for p in pp]
        gains, vad_prob = rnnoise_process_frame(net, x)
        outputs = postprocess(st, pp, gains, vad_prob)

        for out in outputs:
            out = np.array(out, dtype=int)
            out = np.clip(out, (-0x7fff - 1), 0x7fff)
            out = struct.pack("h" * len(out), *out)
            wf_out.writeframes(out)

    bar.close()
    wf_out.close()
    logger.info(f'saved at : {save_path}')

    logger.info('Script finished successfully.')


def main():
    # model files check and download
    check_and_download_models(WEIGHT_PATH, MODEL_PATH, REMOTE_PATH)

    env_id = args.env_id

    # initialize
    if not args.onnx:
        net = ailia.Net(MODEL_PATH, WEIGHT_PATH, env_id=env_id, memory_mode=args.memory_mode)
    else:
        import onnxruntime
        cuda = 0 < ailia.get_gpu_environment_id()
        providers = ['CUDAExecutionProvider', 'CPUExecutionProvider'] if cuda else ['CPUExecutionProvider']
        net = onnxruntime.InferenceSession(WEIGHT_PATH, providers=providers)

    recognize_from_audio(net)


if __name__ == '__main__':
    main()
