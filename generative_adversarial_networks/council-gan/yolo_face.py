import numpy as np

import ailia 
from arg_utils import check_file_existance  # noqa: E402
from model_utils import check_and_download_models  # noqa: E402


WEIGHT_PATH_YOLO= 'yolov3-face.opt.onnx'
MODEL_PATH_YOLO = 'yolov3-face.opt.onnx.prototxt'
REMOTE_PATH_YOLO = 'https://storage.googleapis.com/ailia-models/yolov3-face/'

FACE_CATEGORY = ['face']
THRESHOLD = 0.2
IOU = 0.45


class FaceLocator():
    """Face detector for use with coucil_gan, in order to improve performace at various distances"""
    def __init__(self, env_id=None, memory_mode=None):
        check_and_download_models(WEIGHT_PATH_YOLO, MODEL_PATH_YOLO, REMOTE_PATH_YOLO)

        # net initialize
        self.detector = detector = ailia.Detector(
                                    MODEL_PATH_YOLO,
                                    WEIGHT_PATH_YOLO,
                                    len(FACE_CATEGORY),
                                    format=ailia.NETWORK_IMAGE_FORMAT_RGB,
                                    channel=ailia.NETWORK_IMAGE_CHANNEL_FIRST,
                                    range=ailia.NETWORK_IMAGE_RANGE_U_FP32,
                                    algorithm=ailia.DETECTOR_ALGORITHM_YOLOV3,
                                    env_id=env_id,
                                    memory_mode=memory_mode
                                )
       
    def get_faces(self, img):
        # prepare input data
        img = np.concatenate((img, np.ones((img.shape[0], img.shape[1], 1))*255), axis =-1)

        print('Running YOLOv3 face recognition... ', end='')
    
        self.detector.compute(img, THRESHOLD, IOU)
        objects = []
        h, w, c = img.shape
        for i in range(self.detector.get_object_count()):
            obj = self.detector.get_object(i)
            # get top left, bottom right
            objects.append((int(h*obj.y), int(w*obj.x), int(h*(obj.y+obj.h)), int(w*(obj.x+obj.w))))

        print('finished!')
        return objects
