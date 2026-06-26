import numpy as np
from scipy.special import softmax
from tqdm import tqdm

def headpose_pred_to_degree(pred):
    pred = softmax(pred, axis=1)
    idx_tensor = np.arange(66, dtype=np.float32)
    degree = np.sum(pred * idx_tensor, axis=1) * 3 - 99
    return degree

def get_rotation_matrix(yaw, pitch, roll):
    yaw = yaw / 180 * np.pi
    pitch = pitch / 180 * np.pi
    roll = roll / 180 * np.pi

    pitch_mat = np.stack([
        np.ones_like(pitch), np.zeros_like(pitch), np.zeros_like(pitch),
        np.zeros_like(pitch), np.cos(pitch), -np.sin(pitch),
        np.zeros_like(pitch), np.sin(pitch), np.cos(pitch)
    ], axis=1).reshape(-1, 3, 3)

    yaw_mat = np.stack([
        np.cos(yaw), np.zeros_like(yaw), np.sin(yaw),
        np.zeros_like(yaw), np.ones_like(yaw), np.zeros_like(yaw),
        -np.sin(yaw), np.zeros_like(yaw), np.cos(yaw)
    ], axis=1).reshape(-1, 3, 3)

    roll_mat = np.stack([
        np.cos(roll), -np.sin(roll), np.zeros_like(roll),
        np.sin(roll), np.cos(roll), np.zeros_like(roll),
        np.zeros_like(roll), np.zeros_like(roll), np.ones_like(roll)
    ], axis=1).reshape(-1, 3, 3)

    rot_mat = np.einsum('bij,bjk,bkm->bim', pitch_mat, yaw_mat, roll_mat)
    return rot_mat

def keypoint_transformation(kp_canonical, he, wo_exp=False):
    kp = kp_canonical    # (bs, k, 3) 
    yaw, pitch, roll= he['yaw'], he['pitch'], he['roll']      
    yaw = headpose_pred_to_degree(yaw) 
    pitch = headpose_pred_to_degree(pitch)
    roll = headpose_pred_to_degree(roll)

    if 'yaw_in' in he:
        yaw = he['yaw_in']
    if 'pitch_in' in he:
        pitch = he['pitch_in']
    if 'roll_in' in he:
        roll = he['roll_in']

    rot_mat = get_rotation_matrix(yaw, pitch, roll)    # (bs, 3, 3)

    t, exp = he['t'], he['exp']
    if wo_exp:
        exp =  exp*0  
    
    # keypoint rotation
    kp_rotated = np.einsum('bmp,bkp->bkm', rot_mat, kp)

    # keypoint translation
    t[:, 0] = t[:, 0]*0
    t[:, 2] = t[:, 2]*0
    t = np.repeat(t[:, np.newaxis, :], kp.shape[1], axis=1)
    kp_t = kp_rotated + t

    # add expression deviation 
    exp = exp.reshape(exp.shape[0], -1, 3)
    kp_transformed = kp_t + exp

    return kp_transformed

def make_animation(
    source_image, source_semantics, target_semantics,
    generator_net, kp_detector_net, he_estimator_net, mapping_net, 
    yaw_c_seq=None, pitch_c_seq=None, roll_c_seq=None, use_exp=True, use_half=False, 
    use_onnx=False
):
    predictions = []

    if use_onnx:
        kp_canonical = kp_detector_net.run(None, {"input_image": source_image})[0]
        he_source_tmp = mapping_net.run(None, {"input_3dmm": source_semantics})
    else:
        kp_canonical = kp_detector_net.run([source_image])[0]
        he_source_tmp = mapping_net.run([source_semantics])
    he_source = {
        "yaw": he_source_tmp[0],
        "pitch": he_source_tmp[1],
        "roll": he_source_tmp[2],
        "t": he_source_tmp[3],
        "exp": he_source_tmp[4],
    }

    kp_source = keypoint_transformation(kp_canonical, he_source)

    for frame_idx in tqdm(range(target_semantics.shape[1]), 'Face Renderer'):
        target_semantics_frame = target_semantics[:, frame_idx]
        if use_onnx:
            he_driving_tmp = mapping_net.run(None, {"input_3dmm": target_semantics_frame})
        else:
            he_driving_tmp = mapping_net.run([target_semantics_frame])
        he_driving = {
            "yaw": he_driving_tmp[0],
            "pitch": he_driving_tmp[1],
            "roll": he_driving_tmp[2],
            "t": he_driving_tmp[3],
            "exp": he_driving_tmp[4],
        }

        if yaw_c_seq is not None:
            he_driving['yaw_in'] = yaw_c_seq[:, frame_idx]
        if pitch_c_seq is not None:
            he_driving['pitch_in'] = pitch_c_seq[:, frame_idx] 
        if roll_c_seq is not None:
            he_driving['roll_in'] = roll_c_seq[:, frame_idx] 
        
        kp_driving = keypoint_transformation(kp_canonical, he_driving)
        
        if use_onnx:
            out = generator_net.run(None, {
                "source_image": source_image,
                "kp_driving": kp_driving,
                "kp_source": kp_source,
            })[0]
        else:
            out = generator_net.run([source_image, kp_driving, kp_source])[0]
        predictions.append(out)
    return np.stack(predictions, axis=1)
