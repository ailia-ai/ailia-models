import os, sys
import numpy as np
import cv2
from tqdm import tqdm
from PIL import Image
from scipy.io import loadmat, savemat

from preprocess.face3d_utils import align_img
from preprocess.croper import Preprocesser

sys.path.append('../../util')
from image_utils import imread  # noqa: E402

def split_coeff(coeffs):
    return {
        'id': coeffs[:, :80],
        'exp': coeffs[:, 80:144],
        'tex': coeffs[:, 144:224],
        'angle': coeffs[:, 224:227],
        'gamma': coeffs[:, 227:254],
        'trans': coeffs[:, 254:]
    }

def load_lm3d(lm3d_path):
    Lm3D = loadmat(lm3d_path)['lm']

    lm_idx = np.array([31, 37, 40, 43, 46, 49, 55]) - 1
    selected_lms = [
        Lm3D[lm_idx[0], :],
        np.mean(Lm3D[lm_idx[[1, 2]], :], axis=0),
        np.mean(Lm3D[lm_idx[[3, 4]], :], axis=0),
        Lm3D[lm_idx[5], :],
        Lm3D[lm_idx[6], :]
    ]
    
    Lm3D = np.stack(selected_lms, axis=0)
    return Lm3D[[1, 2, 0, 3, 4], :]

class CropAndExtract():
    def __init__(self, face3d_recon_net, face_align_net, face_det_net, lm3d_path, use_onnx):
        self.propress = Preprocesser(face_align_net, face_det_net, use_onnx)
        self.face3d_recon_net = face3d_recon_net
        self.lm3d_std = load_lm3d(lm3d_path)
        self.use_onnx = use_onnx
    
    def generate(self, input_path, save_dir, crop_or_resize='crop', source_image_flag=False, pic_size=256):
        pic_name = os.path.splitext(os.path.split(input_path)[-1])[0]  
        landmarks_path =  os.path.join(save_dir, pic_name+'_landmarks.txt') 
        coeff_path =  os.path.join(save_dir, pic_name+'.mat')  
        png_path =  os.path.join(save_dir, pic_name+'.png')  

        #load input
        if not os.path.isfile(input_path):
            raise ValueError('input_path must be a valid path to video/image file')
        elif input_path.split('.')[-1] in ['jpg', 'png', 'jpeg']:
            # loader for first frame
            full_frames = [imread(input_path)]
            fps = 25
        else:
            # loader for videos
            video_stream = cv2.VideoCapture(input_path)
            fps = video_stream.get(cv2.CAP_PROP_FPS)
            full_frames = [] 
            while 1:
                still_reading, frame = video_stream.read()
                if not still_reading:
                    video_stream.release()
                    break 
                full_frames.append(frame) 
                if source_image_flag:
                    break

        x_full_frames= [cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  for frame in full_frames] 

        #### crop images as the 
        if 'crop' in crop_or_resize.lower(): # default crop
            x_full_frames, crop, quad = self.propress.crop(x_full_frames, still=True if 'ext' in crop_or_resize.lower() else False, xsize=512)
            clx, cly, crx, cry = crop
            lx, ly, rx, ry = quad
            lx, ly, rx, ry = int(lx), int(ly), int(rx), int(ry)
            oy1, oy2, ox1, ox2 = cly+ly, cly+ry, clx+lx, clx+rx
            crop_info = ((ox2 - ox1, oy2 - oy1), crop, quad)
        elif 'full' in crop_or_resize.lower():
            x_full_frames, crop, quad = self.propress.crop(x_full_frames, still=True if 'ext' in crop_or_resize.lower() else False, xsize=512)
            clx, cly, crx, cry = crop
            lx, ly, rx, ry = quad
            lx, ly, rx, ry = int(lx), int(ly), int(rx), int(ry)
            oy1, oy2, ox1, ox2 = cly+ly, cly+ry, clx+lx, clx+rx
            crop_info = ((ox2 - ox1, oy2 - oy1), crop, quad)
        else: # resize mode
            oy1, oy2, ox1, ox2 = 0, x_full_frames[0].shape[0], 0, x_full_frames[0].shape[1] 
            crop_info = ((ox2 - ox1, oy2 - oy1), None, None)

        frames_pil = [Image.fromarray(cv2.resize(frame,(pic_size, pic_size))) for frame in x_full_frames]
        if len(frames_pil) == 0:
            print('No face is detected in the input file')
            return None, None

        # save crop info
        for frame in frames_pil:
            cv2.imwrite(png_path, cv2.cvtColor(np.array(frame), cv2.COLOR_RGB2BGR))

        # 2. get the landmark according to the detected face. 
        if not os.path.isfile(landmarks_path): 
            lm = self.propress.predictor.extract_keypoint(frames_pil, landmarks_path)
        else:
            print(' Using saved landmarks.')
            lm = np.loadtxt(landmarks_path).astype(np.float32)
            lm = lm.reshape([len(x_full_frames), -1, 2])

        if not os.path.isfile(coeff_path):
            # load 3dmm paramter generator from Deep3DFaceRecon_pytorch 
            video_coeffs, full_coeffs = [],  []
            for idx in tqdm(range(len(frames_pil)), desc='3DMM Extraction In Video'):
                frame = frames_pil[idx]
                W,H = frame.size
                lm1 = lm[idx].reshape([-1, 2])
            
                if np.mean(lm1) == -1:
                    lm1 = (self.lm3d_std[:, :2]+1)/2.
                    lm1 = np.concatenate(
                        [lm1[:, :1]*W, lm1[:, 1:2]*H], 1
                    )
                else:
                    lm1[:, -1] = H - 1 - lm1[:, -1]

                trans_params, im1, lm1, _ = align_img(frame, lm1, self.lm3d_std)
 
                trans_params = np.array([float(item) for item in np.hsplit(trans_params, 5)]).astype(np.float32)
                im_np = np.transpose(np.array(im1, dtype=np.float32) / 255.0, (2, 0, 1))[np.newaxis, ...]
                
                if self.use_onnx:
                    full_coeff = self.face3d_recon_net.run(None, {"input_image": im_np})[0]
                else:
                    full_coeff = self.face3d_recon_net.run([im_np])[0]
                coeffs = split_coeff(full_coeff)

                pred_coeff = {key:coeffs[key] for key in coeffs}
 
                pred_coeff = np.concatenate([
                    pred_coeff['exp'], 
                    pred_coeff['angle'],
                    pred_coeff['trans'],
                    trans_params[2:][None],
                    ], 1)
                video_coeffs.append(pred_coeff)
                full_coeffs.append(full_coeff)

            semantic_npy = np.array(video_coeffs)[:,0] 

            savemat(coeff_path, {'coeff_3dmm': semantic_npy, 'full_3dmm': np.array(full_coeffs)[0]})

        return coeff_path, png_path, crop_info
