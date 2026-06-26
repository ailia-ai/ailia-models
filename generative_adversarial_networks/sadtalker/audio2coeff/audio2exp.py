from tqdm import tqdm
import numpy as np

class Audio2Exp:
    def __init__(self, audio2exp_net, use_onnx):
        self.audio2exp_net = audio2exp_net
        self.use_onnx = use_onnx

    def test(self, batch):
        mel_input = batch['indiv_mels']  # (bs, T, 1, 80, 16)
        ref = batch['ref']              # (bs, T, 70)
        ratio = batch['ratio_gt']       # (bs, T, 1)

        bs, T, _, _, _ = mel_input.shape
        exp_coeff_pred = []

        for i in tqdm(range(0, T, 10), 'audio2exp'):  # 10フレームごとに処理
            current_mel_input = mel_input[:, i:i+10]  # (bs, 10, 1, 80, 16)
            current_ref = ref[:, i:i+10, :64]         # (bs, 10, 64)
            current_ratio = ratio[:, i:i+10]         # (bs, 10, 1)

            audiox = current_mel_input.reshape(-1, 1, 80, 16)  # (bs*T, 1, 80, 16)

            if self.use_onnx:
                curr_exp_coeff_pred = self.audio2exp_net.run(None, {
                    "audio": audiox,
                    "ref": current_ref,
                    "ratio": current_ratio,
                })[0]
            else:
                curr_exp_coeff_pred = self.audio2exp_net.run([audiox, current_ref, current_ratio])[0]

            exp_coeff_pred += [curr_exp_coeff_pred]

        # BS x T x 64
        results_dict = {
            'exp_coeff_pred': np.concatenate(exp_coeff_pred, axis=1)
        }

        return results_dict
