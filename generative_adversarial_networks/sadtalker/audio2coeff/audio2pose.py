import numpy as np

class Audio2Pose:
    def __init__(self, audio2pose_net, use_onnx):
        self.audio2pose_net = audio2pose_net
        self.seq_len = 32
        self.latent_dim = 64
        self.use_onnx = use_onnx

    def test(self, x):
        batch = {}

        ref = x['ref']  # [BS, 1, 70]
        class_id = x['class']  # [BS]
        bs = ref.shape[0]
        pose_ref = ref[:, 0, -6:]  # [BS, 6]

        indiv_mels = x['indiv_mels']  # [BS, T, 1, 80, 16]
        T_total = int(x['num_frames']) - 1  

        chunk_count = (T_total + self.seq_len - 1) // self.seq_len

        pose_motion_pred_list = [
            np.zeros((bs, 1, pose_ref.shape[1]), dtype=np.float32)
        ]

        start_idx = 0
        for _ in range(chunk_count):
            end_idx = min(start_idx + self.seq_len, T_total)
            chunk_len = end_idx - start_idx

            chunk_mels = indiv_mels[:, 1 + start_idx : 1 + end_idx]  

            if chunk_len < self.seq_len:
                pad_len = self.seq_len - chunk_len
                pad_chunk = np.repeat(chunk_mels[:, :1], pad_len, axis=1)
                chunk_mels = np.concatenate([pad_chunk, chunk_mels], axis=1)  

            z = np.random.randn(bs, self.latent_dim).astype(np.float32)

            # Inference using a single model for AudioEncoder and netG.
            if self.use_onnx:
                motion_pred = self.audio2pose_net.run(None, {
                    "chunk_mels": chunk_mels,
                    "z": z,
                    "pose_ref": pose_ref,
                    "class": class_id,
                })[0]
            else:
                motion_pred = self.audio2pose_net.run([chunk_mels, z, pose_ref, class_id])[0]

            if chunk_len < self.seq_len:
                motion_pred = motion_pred[:, -chunk_len:, :]

            pose_motion_pred_list.append(motion_pred)
            start_idx += chunk_len

        pose_motion_pred = np.concatenate(pose_motion_pred_list, axis=1)  # [BS, T_total, 6]
        pose_pred = ref[:, :1, -6:] + pose_motion_pred  # [BS, T_total+1, 6]

        batch['pose_motion_pred'] = pose_motion_pred
        batch['pose_pred'] = pose_pred
        return batch
