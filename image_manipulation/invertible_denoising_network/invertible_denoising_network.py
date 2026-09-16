import copy
import glob
import json
import os
import sys
import time
from logging import getLogger

import ailia
import cv2
import numpy as np
import torch

sys.path.append('../../util')
import webcamera_utils  # noqa: E402
from image_utils import imread  # noqa: E402
from model_utils import check_and_download_models  # noqa: E402
from arg_utils import get_base_parser, get_savepath, update_parser  # noqa: E402

logger = getLogger(__name__)



# ======================
# Parameters
# ======================
WEIGHT_PATH = 'InvDN.onnx'
MODEL_PATH = 'InvDN.onnx.prototxt'
REMOTE_PATH = 'https://storage.googleapis.com/ailia-models/invertible_denoising_network/'
IMAGE_PATH = 'input.png'
SAVE_IMAGE_PATH = 'output.png'



# ======================
# Arguemnt Parser Config
# ======================
parser = get_base_parser(
    'Invertible Denoising Network', IMAGE_PATH, SAVE_IMAGE_PATH
)
args = update_parser(parser)



# ======================
# Main functions
# ======================
class InvNet():
    def __init__(self):
        self.net = ailia.Net(MODEL_PATH, WEIGHT_PATH, env_id=args.env_id, memory_mode=args.memory_mode)

    def predict(self, input):
        input = input.astype(np.float32) / 255. # image array to Numpy float32, HWC, BGR, [0,1]
        input = np.expand_dims(input, 0)
        input = np.transpose(input, (0, 3, 1, 2))

        preds = self.net.run({
            'input': input.astype(np.float32),
            'gaussian_scale': np.array([1])
        })

        output = preds[1]
        output = output[:, :3, :, :]
        output = self.output2img_real(output)
        output = np.transpose(output, (1, 2, 0))

        return output

    def output2img_real(self, output, out_type=np.uint8, min_max=(0, 1)):
        darr = np.clip(np.squeeze(output), *min_max)
        darr = (darr - min_max[0]) / (min_max[1] - min_max[0])  # to range [0,1]
        if out_type == np.uint8:
            darr = (darr * 255.0).round()
        return darr.astype(out_type)

def add_noise(img, noise_param=50):
    height, width = img.shape[0], img.shape[1]
    std = np.random.uniform(0, noise_param)
    noise = np.random.normal(0, std, (height, width, 3))
    noise_img = np.array(img) + noise
    noise_img = np.clip(noise_img, 0, 255).astype(np.uint8)
    return noise_img

# ======================
# Main functions
# ======================

def recognize_from_image():
    net = InvNet()

    # input image loop
    for image_path in args.input:
        savepath = get_savepath(args.savepath, image_path)
        logger.info(f'saved at : {savepath}')
        image = imread(image_path)
        image = net.predict(image)
        cv2.imwrite(
            savepath,
            image
        )
    logger.info('Script finished successfully.')


def recognize_from_video():
    net = InvNet()

    cap = webcamera_utils.get_capture(args.video)
    if not cap.isOpened():
        exit()

    if args.savepath != SAVE_IMAGE_PATH:
        f_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        f_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        writer = webcamera_utils.get_writer(
            args.savepath, f_w, f_h
        )
    else:
        writer = None

    frame_shown = False
    while(True):
        ret, frame = cap.read()
        if (cv2.waitKey(1) & 0xFF == ord('q')) or not ret:
            break
        if frame_shown and cv2.getWindowProperty('frame', cv2.WND_PROP_VISIBLE) == 0:
            break

        _, resized_image = webcamera_utils.adjust_frame_size(frame, 256, 256)
        noised_frame = add_noise(resized_image)
        denoised_frame = net.predict(noised_frame)

        # half and half
        noised_frame[:,128:256,:] = denoised_frame[:,128:256,:]
        cv2.imshow('frame', noised_frame)
        frame_shown = True

        # save results
        if writer is not None:
            writer.write(noised_frame)

    cap.release()
    raw_video.release()
    noised_video.release()
    denoised_video.release()
    cv2.destroyAllWindows()


def main():
    check_and_download_models(WEIGHT_PATH, MODEL_PATH, REMOTE_PATH) # model files check and download

    if args.video is not None:
        # video mode
        recognize_from_video()
    else:
        # image mode
        recognize_from_image()


if __name__ == '__main__':
    main()
