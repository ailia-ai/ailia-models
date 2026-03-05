import os
import cv2
from functools import lru_cache

import face_alignment
import numpy as np
from PIL import Image

from affine_transform import AlignRestore, laplacianSmooth


MASK_IMAGE = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "mask.png",
)


@lru_cache(maxsize=None)
def load_fixed_mask(resolution):
    mask_image = cv2.imread(MASK_IMAGE)
    mask_image = cv2.cvtColor(mask_image, cv2.COLOR_BGR2RGB)
    mask_image = (
        cv2.resize(
            mask_image, (resolution, resolution), interpolation=cv2.INTER_LANCZOS4
        )
        / 255.0
    )
    mask_image = mask_image.transpose(2, 0, 1)  # HWC -> CHW
    return mask_image


def preprocess_fixed_mask_image(image, size=256):
    image = image.transpose(1, 2, 0)  # CHW -> HWC
    image = np.array(
        Image.fromarray(image).resize((size, size), Image.Resampling.BILINEAR)
    )
    image = image.transpose(2, 0, 1)  # HWC -> CHW

    mask_image = load_fixed_mask(size)

    pixel_values = (image / 255.0 - 0.5) / 0.5
    masked_pixel_values = pixel_values * mask_image
    return pixel_values, masked_pixel_values, mask_image[0:1]


class ImageProcessor:
    def __init__(self, size=256):
        self.resolution = size

        device = "cpu"
        import torch # face_alignment depend to torch
        if torch.cuda.is_available():
            device = "cuda"

        self.restorer = AlignRestore()
        self.smoother = laplacianSmooth()
        self.fa = face_alignment.FaceAlignment(
            face_alignment.LandmarksType.TWO_D,
            flip_input=False,
            device=device
        )

    def affine_transform(self, image: np.ndarray) -> np.ndarray:
        detected_faces = self.fa.get_landmarks(image)
        if detected_faces is None:
            raise RuntimeError("Face not detected")
        lm68 = detected_faces[0]

        points = self.smoother.smooth(lm68)
        lmk3_ = np.zeros((3, 2))
        lmk3_[0] = points[17:22].mean(0)
        lmk3_[1] = points[22:27].mean(0)
        lmk3_[2] = points[27:36].mean(0)

        face, affine_matrix = self.restorer.align_warp_face(
            image.copy(), lmks3=lmk3_, smooth=True
        )
        box = [0, 0, face.shape[1], face.shape[0]]  # x1, y1, x2, y2
        face = cv2.resize(
            face, (self.resolution, self.resolution), interpolation=cv2.INTER_LANCZOS4
        )
        face = face.transpose(2, 0, 1)  # HWC -> CHW

        return face, box, affine_matrix
