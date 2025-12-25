import ailia
import time
from logging import getLogger
import numpy as np
from typing import Tuple
import wave
import librosa
import sys
sys.path.append('../../util')
from arg_utils import get_base_parser, update_parser  # noqa: E402

logger = getLogger(__name__)

# ======================
# Parameters
# ======================

WAV_PATH = "ja.wav"
SAVE_TEXT_PATH = "output.txt"

WEIGHT_ENC_PATH = "encoder-epoch-75-avg-11-chunk-16-left-128.int8.onnx"
WEIGHT_DEC_PATH = "decoder-epoch-75-avg-11-chunk-16-left-128.onnx"
WEIGHT_JOI_PATH = "joiner-epoch-75-avg-11-chunk-16-left-128.int8.onnx"
TOKEN_PATH = "tokens.txt"

# ======================
# Arguemnt Parser Config
# ======================
parser = get_base_parser("sherpa-onnx", WAV_PATH, SAVE_TEXT_PATH, input_ftype="audio")
parser.add_argument("--memory_mode", default=-1, type=int, help="memory mode")
parser.add_argument("--ailia_audio", action="store_true", help="use ailia audio.")
parser.add_argument("--disable_ailia_tokenizer", action="store_true", help="disable ailia tokenizer.")
parser.add_argument("--onnx", action="store_true", help="execute onnxruntime version.")

args = update_parser(parser)

def read_wave(wave_filename: str) -> Tuple[np.ndarray, int]:
    with wave.open(wave_filename) as f:
        assert f.getnchannels() == 1, f.getnchannels()
        assert f.getsampwidth() == 2, f.getsampwidth()  # it is in bytes
        num_samples = f.getnframes()
        samples = f.readframes(num_samples)
        samples_int16 = np.frombuffer(samples, dtype=np.int16)
        samples_float32 = samples_int16.astype(np.float32)

        samples_float32 = samples_float32 / 32768
        return samples_float32, f.getframerate()
    
def _get_input_infos(net):
    """
    {"name":..., "shape":..., "type":...} の辞書のリストを返す。
    onnxruntime.InferenceSession と ailia.Net の両方に対応
    """
    infos = []
    if not args.onnx:
        for idx in net.get_input_blob_list():
            name = net.get_blob_name(idx)
            shape = net.get_blob_shape(idx)
            infos.append({"name": name, "shape": tuple(shape), "type": None})
    else:
        for inp in net.get_inputs():
            infos.append({"name": inp.name, "shape": tuple(inp.shape), "type": getattr(inp, "type", "")})
    return infos

def init_states(enc_net):
    input_info = _get_input_infos(enc_net)
    states = []
    for i in input_info[1:]:
        shape = [1 if (s is None or isinstance(s, str)) else s for s in i["shape"]]
        dtype_str = i.get("type", "") or ""
        dtype = np.int64 if "int64" in dtype_str else np.float32
        
        states.append(np.zeros(shape, dtype=dtype))
    return states


def build_enc_inputs(enc_net, x_input, current_states):
    input_info = _get_input_infos(enc_net)
    inputs_dict = {input_info[0]["name"]: x_input.astype(np.float32)}
    for i, info in enumerate(input_info[1:]):
        inputs_dict[info["name"]] = current_states[i]
        
    return inputs_dict

def get_features(samples, n_mels, sr=16000):
    """
    音声波形(samples)からLog-Melスペクトログラム(80次元)をすべて抽出する
    """
    # 1. メルスペクトログラムの抽出
    # Zipformer設定: 25ms窓(400), 10ms歩進(160)
    S = librosa.feature.melspectrogram(
        y=samples, 
        sr=sr, 
        n_fft=512, 
        hop_length=160, 
        win_length=400, 
        n_mels=n_mels,
        center=False  # ストリーミングでは未来の音を見ないためFalse
    )
    
    # 2. 対数（Log）に変換
    log_S = np.log(S + 1e-10).astype(np.float32)
    
    # 3. 転置して [時間, 80次元] の形にする
    return log_S.T

def recognize_from_audio(enc_net, dec_net, joi_net, samples, sr=16000, segment_length=45, offset=32, n_mels=80):
    
    context_size = 2  # デコーダの文脈サイズ

    # --- STEP 1. 初期準備 ---
    features = get_features(samples, n_mels, sr=sr) # (Total_Frames, 80)
    
    # --- STEP 2: 状態（キャッシュ）の初期化 ---
    states = init_states(enc_net) # Encoder の記憶
    hyp = [0] * context_size # デコーダの履歴
    
    # 最初の文脈ベクトルを生成
    decoder_input = np.array([hyp], dtype=np.int64)
    decoder_out = dec_net.run(None, { _get_input_infos(dec_net)[0]["name"]: decoder_input})[0]

    num_processed_frames = 0
    final_hyp = []

    # --- STEP 3: ストリーミング・推論ループ ---
    while (len(features) - num_processed_frames) >= segment_length:
        x_chunk = features[num_processed_frames : num_processed_frames + segment_length][np.newaxis, :, :]
        
        # A. Encoder 実行
        enc_inputs = build_enc_inputs(enc_net, x_chunk, states)
        outputs = enc_net.run(None, enc_inputs)
        
        encoder_out = outputs[0]  # (1, T', 512)
        states = outputs[1:]     # 記憶を更新して次のループへ

        # B. Greedy Search
        encoder_out = encoder_out[0] # バッチ次元を消す
        for t in range(encoder_out.shape[0]):
            cur_enc = encoder_out[t:t+1] # 今の瞬間の音ベクトル
            
            # Joiner で確率計算
            logits = joi_net.run(None, {
                _get_input_infos(joi_net)[0]["name"]: cur_enc,
                _get_input_infos(joi_net)[1]["name"]: decoder_out
            })[0]
            
            y = np.argmax(logits) # 最も可能性の高い文字ID
            
            # 文字が確定（Blank以外）したらデコーダを更新
            if y != 0:
                final_hyp.append(y)
                hyp.append(y)
                
                # 直近 context_size文字の履歴で新しい文脈ベクトルを作成
                decoder_input = np.array([hyp[-context_size:]], dtype=np.int64)
                decoder_out = dec_net.run(None, { _get_input_infos(dec_net)[0]["name"]: decoder_input})[0]

        # C. offsetフレーム進める
        num_processed_frames += offset

    return final_hyp # 確定したトークンIDのリスト

class RealtimeEstimator:
    def __init__(self, enc_net, dec_net, joi_net, token_table, sr=16000, segment_length=45, offset=32, n_mels=80):
        self.enc_net = enc_net
        self.dec_net = dec_net
        self.joi_net = joi_net
        self.token_table = token_table
        
        # 内部状態の初期化
        self.states = init_states(enc_net) 
        self.hyp = [0, 0] # context_size=2 を想定
        self.decoder_out = self._init_decoder()
        
        # サンプル蓄積用バッファ
        self.sample_buffer = np.array([], dtype=np.float32)
        
        # 定数 (16kHz想定)
        self.OFFSET_SAMPLES = int(offset * sr *0.01) # 10ms歩進
        self.SEGMENT_SAMPLES = int(segment_length * sr *0.01) # segment_lengthフレーム分

    def _init_decoder(self):
        """初期文脈ベクトルの生成"""
        decoder_input = np.array([self.hyp], dtype=np.int64)
        return self.dec_net.run(None, {self.dec_net.get_inputs()[0].name: decoder_input})[0]

    def decode_chunk(self):
        """バッファ内の音声を特徴量に変換し、推論を実行"""
        # 1. 特徴量抽出 (45フレーム分)
        chunk_samples = self.sample_buffer[:self.SEGMENT_SAMPLES]
        features = get_features(chunk_samples)
        x_input = features[np.newaxis, :, :] 
        
        # 2. Encoder 実行
        enc_inputs = build_enc_inputs(self.enc_net, x_input, self.states)
        outputs = self.enc_net.run(None, enc_inputs)
        
        encoder_out = outputs[0][0] 
        self.states = outputs[1:]
        
        # 3. Greedy Search (Joiner + Decoder)
        new_text = ""
        for t in range(encoder_out.shape[0]):
            cur_enc = encoder_out[t:t+1]
            logits = self.joi_net.run(None, {
                self.joi_net.get_inputs()[0].name: cur_enc,
                self.joi_net.get_inputs()[1].name: self.decoder_out
            })[0]
            
            y = np.argmax(logits)
            if y != 0: # blank_id 以外
                token = self.token_table.get(y, "")
                new_text += token.replace("▁", " ") # ▁をスペースに置換
                
                # デコーダ更新
                self.hyp.append(y)
                decoder_input = np.array([self.hyp[-2:]], dtype=np.int64)
                self.decoder_out = self.dec_net.run(None, {self.dec_net.get_inputs()[0].name: decoder_input})[0]
        
        return new_text
    
def main_inference_loop(mic_info, estimator):
    que = mic_info["que"]
    fin = mic_info["fin"]
    pause = mic_info["pause"]

    print("--- リアルタイム音声認識開始 (終了は CTRL+C またはアプリ操作) ---")
    try:
        while not fin.is_set():
            if pause.is_set():
                time.sleep(0.1)
                continue

            try:
                # 1. マイクから届いたデータを取得 (Timeout 0.1s)
                new_samples = que.get(timeout=0.1)
                estimator.sample_buffer = np.concatenate([estimator.sample_buffer, new_samples])
                
                # 2. 十分なデータが溜まっていれば推論実行
                while len(estimator.sample_buffer) >= estimator.SEGMENT_SAMPLES:
                    # 推論実行
                    recognized_text = estimator.decode_chunk()
                    
                    if recognized_text:
                        print(recognized_text, end="", flush=True)
                    estimator.sample_buffer = estimator.sample_buffer[estimator.OFFSET_SAMPLES:]

            except que.Empty:
                # キューが空のときは次の入力を待つ
                continue
    except KeyboardInterrupt:
        print("\n認識を終了します...")
    finally:
        print("\nDone.")
    
def load_tokens(tokens_path):
    """トークンファイル を読み込んで ID から文字を引く辞書を返す"""
    token_table = {}
    with open(tokens_path, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 2:
                # 文字 ID の順で並んでいると想定
                token = parts[0]
                token_id = int(parts[1])
                token_table[token_id] = token
            elif len(parts) == 1:
                # 空白文字(ID: 0 等)が単独で存在する場合の考慮
                token_table[int(parts[0])] = " "
    return token_table

def tokens_to_text(token_ids, token_table):
    """トークンIDのリストを人間が読める文字列に変換する"""
    text = ""
    for tid in token_ids:
        # 辞書から文字を取得（ID 0 などは無視されることが多い）
        token = token_table.get(tid, "")
        
        # 特殊な空白記号 '▁' を半角スペースに置換
        if token == "▁":
            text += " "
        else:
            text += token
            
    # 文頭・文末の余計な空白を削除して返す
    return text.strip()


def main():
    samples, sr = read_wave(WAV_PATH)
    tail_paddings = np.zeros(int(0.66 * sr), dtype=np.float32) # 0.66s相当のゼロパディングを末尾に追加
    samples = np.concatenate([samples, tail_paddings])

    if not args.onnx:
        enc_net = ailia.Net(None, WEIGHT_ENC_PATH, env_id=args.env_id, memory_mode=args.memory_mode)
        dec_net = ailia.Net(None, WEIGHT_DEC_PATH, env_id=args.env_id, memory_mode=args.memory_mode)
        joi_net = ailia.Net(None, WEIGHT_JOI_PATH, env_id=args.env_id, memory_mode=args.memory_mode)
    else:
        import onnxruntime
        providers = ["CPUExecutionProvider"]
        # providers = ["CUDAExecutionProvider"]
        enc_net = onnxruntime.InferenceSession(WEIGHT_ENC_PATH, providers=providers)
        dec_net = onnxruntime.InferenceSession(WEIGHT_DEC_PATH, providers=providers)
        joi_net = onnxruntime.InferenceSession(WEIGHT_JOI_PATH, providers=providers)

    # モデル入力から期待フレーム数を推定する
    enc_input_shape = _get_input_infos(enc_net)[0]["shape"]
    shape = enc_input_shape  # 例: [1, 45, 80] または [45, 80]
    if len(shape) == 3:
        expected_frames = shape[1]
        n_mels = shape[2]
    elif len(shape) == 2:
        expected_frames = shape[0] 
        n_mels = shape[1]
            
    token_list = recognize_from_audio(enc_net, dec_net, joi_net, samples=samples, sr=sr, segment_length=expected_frames, offset=32, n_mels=n_mels)

    token_table = load_tokens(TOKEN_PATH)
    result_text = tokens_to_text(token_list, token_table)
    print(f"認識結果: {result_text}")


if __name__ == "__main__":
    main()
