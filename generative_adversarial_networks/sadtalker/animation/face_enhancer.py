import os
import sys
import cv2
import numpy as np
from tqdm import tqdm

from animation.videoio import load_video_to_cv2

sys.path.append('../../util')
sys.path.append('../../face_restoration/gfpgan')
from face_restoration import (
    get_face_landmarks_5, align_warp_face, 
    get_inverse_affine, paste_faces_to_image
)

UPSCALE = 2

class GeneratorWithLen(object):
    """ From https://stackoverflow.com/a/7460929 """
    def __init__(self, gen, length):
        self.gen = gen
        self.length = length

    def __len__(self):
        return self.length

    def __iter__(self):
        return self.gen

def enhancer_list(images, method='gfpgan', bg_upsampler='realesrgan'):
    gen = enhancer_generator_no_len(images, method=method, bg_upsampler=bg_upsampler)
    return list(gen)

def enhancer_generator_with_len(
        images, method='gfpgan', bg_upsampler='realesrgan', 
        retinaface_net=None, gfpgan_net=None
    ):
    """Provide a generator with a __len__ method."""

    if os.path.isfile(images): # handle video to images
        # TODO: Create a generator version of load_video_to_cv2
        images = load_video_to_cv2(images)

    gen = enhancer_generator_no_len(
        images, method=method, bg_upsampler=bg_upsampler, 
        retinaface_net=retinaface_net, gfpgan_net=gfpgan_net
    )
    return GeneratorWithLen(gen, len(images))

def enhancer_generator_no_len(
        images, method='gfpgan', bg_upsampler='realesrgan', 
        retinaface_net=None, gfpgan_net=None
    ):
    for idx in tqdm(range(len(images)), desc='Face Enhancer'):
        img = cv2.cvtColor(images[idx], cv2.COLOR_RGB2BGR)

        det_faces, all_landmarks_5 = get_face_landmarks_5(retinaface_net, img, eye_dist_threshold=5)
        cropped_faces, affine_matrices = align_warp_face(img, all_landmarks_5)

        restored_faces = []
        for cropped_face in cropped_faces:
            x = preprocess(cropped_face)
            output = gfpgan_net.predict([x])[0] # feedforward
            restored_face = post_processing(output)
            restored_faces.append(restored_face)

        h, w = img.shape[:2]
        h_up, w_up = int(h * UPSCALE), int(w * UPSCALE)
        img = cv2.resize(img, (w_up, h_up), interpolation=cv2.INTER_LANCZOS4)
        
        inverse_affine_matrices = get_inverse_affine(affine_matrices, upscale_factor=UPSCALE)
        r_img = paste_faces_to_image(img, restored_faces, inverse_affine_matrices, upscale_factor=UPSCALE)

        r_img = cv2.cvtColor(r_img, cv2.COLOR_BGR2RGB)
        yield r_img

def preprocess(img):
    img = img[:, :, ::-1]  # BGR -> RGB
    img = img / 127.5 - 1.0
    img = img.transpose(2, 0, 1)  # HWC -> CHW
    img = np.expand_dims(img, axis=0)
    img = img.astype(np.float32)

    return img

def post_processing(pred):
    img = pred[0]
    img = img.transpose(1, 2, 0)  # CHW -> HWC
    img = img[:, :, ::-1]  # RGB -> BGR

    img = np.clip(img, -1, 1)
    img = (img + 1) * 127.5
    img = img.astype(np.uint8)

    return img
