import os
import cv2
import time
import numpy as np
from PIL import Image
from tqdm import tqdm

from preprocess.face_detection import face_detect
from preprocess.fan import FAN
from preprocess.facexlib_alignment import landmark_98_to_68

class KeypointExtractor():
    def __init__(self, face_align_net, face_det_net, use_onnx):
        self.detector = FAN(face_align_net, use_onnx)
        self.face_det_net = face_det_net

    def extract_keypoint(self, images, name=None):
        if isinstance(images, list):
            keypoints = []
            for image in tqdm(images, desc='landmark Det'):
                current_kp = self.extract_keypoint(image)
                if np.mean(current_kp) == -1 and keypoints:
                    keypoints.append(keypoints[-1])
                else:
                    keypoints.append(current_kp[None])

            keypoints = np.concatenate(keypoints, 0)
            np.savetxt(os.path.splitext(name)[0]+'.txt', keypoints.reshape(-1))
            return keypoints
        else:
            while True:
                try:
                    # face detection -> face alignment.
                    img = np.array(images)
                    bboxes = face_detect(img, self.face_det_net)[0]
                    img = img[int(bboxes[1]):int(bboxes[3]), int(bboxes[0]):int(bboxes[2]), :]

                    keypoints = landmark_98_to_68(self.detector.get_landmarks(img))

                    # keypoints to the original location
                    keypoints[:,0] += int(bboxes[0])
                    keypoints[:,1] += int(bboxes[1])

                    break
                except RuntimeError as e:
                    if str(e).startswith('CUDA'):
                        print("Warning: out of memory, sleep for 1s")
                        time.sleep(1)
                    else:
                        print(e)
                        break    
                except TypeError:
                    print('No face detected in this image')
                    shape = [68, 2]
                    keypoints = -1. * np.ones(shape)                    
                    break
            if name is not None:
                np.savetxt(os.path.splitext(name)[0]+'.txt', keypoints.reshape(-1))
            return keypoints
