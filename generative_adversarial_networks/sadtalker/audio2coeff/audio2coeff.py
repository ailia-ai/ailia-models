import os 
import numpy as np
from scipy.io import savemat, loadmat
from scipy.signal import savgol_filter

from audio2coeff.audio2exp import Audio2Exp
from audio2coeff.audio2pose import Audio2Pose

class Audio2Coeff:
    def __init__(self, audio2exp_net, audio2pose_net, use_onnx):
        self.audio2exp_model = Audio2Exp(audio2exp_net, use_onnx)
        self.audio2pose_model = Audio2Pose(audio2pose_net, use_onnx)

    def generate(self, batch, coeff_save_dir, pose_style, ref_pose_coeff_path=None):
        results_dict_exp= self.audio2exp_model.test(batch)
        exp_pred = results_dict_exp['exp_coeff_pred']  # bs T 64

        batch['class'] = np.array([pose_style], dtype=np.int64)
        results_dict_pose = self.audio2pose_model.test(batch) 
        pose_pred = results_dict_pose['pose_pred']  # bs T 6

        pose_len = pose_pred.shape[1]
        if pose_len < 13:
            pose_len = int((pose_len - 1) / 2) * 2 + 1
            pose_pred = savgol_filter(pose_pred, pose_len, 2, axis=1)
        else:
            pose_pred = savgol_filter(pose_pred, 13, 2, axis=1)
        
        coeffs_pred_numpy = np.concatenate((exp_pred, pose_pred), axis=-1)[0]

        if ref_pose_coeff_path is not None: 
                coeffs_pred_numpy = self.using_refpose(coeffs_pred_numpy, ref_pose_coeff_path)
    
        savemat(os.path.join(coeff_save_dir, '%s##%s.mat'%(batch['pic_name'], batch['audio_name'])),  
                {'coeff_3dmm': coeffs_pred_numpy})

        return os.path.join(coeff_save_dir, '%s##%s.mat'%(batch['pic_name'], batch['audio_name']))
    
    def using_refpose(self, coeffs_pred_numpy, ref_pose_coeff_path):
        num_frames = coeffs_pred_numpy.shape[0]
        refpose_coeff_dict = loadmat(ref_pose_coeff_path)
        refpose_coeff = refpose_coeff_dict['coeff_3dmm'][:, 64:70]

        refpose_num_frames = refpose_coeff.shape[0]
        if refpose_num_frames < num_frames:
            div = num_frames // refpose_num_frames
            re = num_frames % refpose_num_frames

            refpose_coeff_list = [refpose_coeff for _ in range(div)]
            refpose_coeff_list.append(refpose_coeff[:re, :])
            refpose_coeff = np.concatenate(refpose_coeff_list, axis=0)

        # Adjust relative head pose
        coeffs_pred_numpy[:, 64:70] += refpose_coeff[:num_frames, :] - refpose_coeff[0:1, :]

        return coeffs_pred_numpy


