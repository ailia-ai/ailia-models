import cv2
import sys
import numpy as np
from PIL import Image

import ailia
sys.path.append('../psgan')
import faceutils as futils

sys.path.append("../../face_detection")
from blazeface import blazeface_utils as but



class Inference:
    """
    An inference wrapper for makeup transfer.
    It takes two image `source` and `reference` in,
    and transfers the makeup of reference to source.
    """
    def __init__(self, args,face_parser_path,detector_parser_path=None,face_aligment_path=None):

        self.preprocess = PreProcess(args,face_parser_path,detector_parser_path,face_aligment_path)
        self.denoise = False
        self.img_size = 256
        # TODO: can be a hyper-parameter
        self.eyeblur = {'margin': 12, 'blur_size':7}

        self.session1 = ailia.Net(None,"elegant1.onnx",env_id=args.env_id)
        self.session2 = ailia.Net(None,"elegant2.onnx",env_id=args.env_id)

    def prepare_input(self, *data_inputs):
        """
        data_inputs: List[image, mask, diff, lms]
        """
        inputs = []
        for i in range(len(data_inputs)):
            inputs.append(np.expand_dims(data_inputs[i],0) .astype(np.float32))
        # prepare mask
        tmp = np.sum(inputs[1][:,1:], axis=1, keepdims=True)
        inputs[1] = np.concatenate(( (inputs[1][:,0:1]) ,tmp ), axis=1)

        return inputs

    def postprocess(self, source, crop_face, result):
        if crop_face is not None:
            source = source[crop_face['top']:crop_face['bottom'],
                            crop_face['left']:crop_face['right']]

        height, width = source.shape[:2]
        small_source = cv2.resize(source, (self.img_size, self.img_size))
        laplacian_diff = source.astype(
            float) - cv2.resize(small_source, (width, height)).astype(float)
        result = (cv2.resize(result, (width, height)) +
                  laplacian_diff).round().clip(0, 255)

        result = result.astype(np.uint8)

        if self.denoise:
            result = cv2.fastNlMeansDenoisingColored(result)
        return result
    
    def transfer(self, source, reference, postprocess=True):
        """
        Args:
            source (Image): The image where makeup will be transfered to.
            reference (Image): Image containing targeted makeup.
        Return:
            Image: Transfered image.
        """

        source_input, face, crop_face = self.preprocess(source)
        reference_input, _, _ = self.preprocess(reference)
        if not (source_input and reference_input):
            return None

        source_input = self.prepare_input(*source_input)
        reference_input = self.prepare_input(*reference_input)

        elegant1_output= self.session1.run((np.array(source_input[0]),
                          np.array(reference_input[0]),
                          np.array(source_input[1]),
                          np.array(reference_input[1]),
                          np.array(source_input[2]),
                          np.array(reference_input[2]),
                          np.array(source_input[3]),
                          np.array(reference_input[3]))
        )

        kernel = elegant1_output[17]
 
        kernel_inv = np.linalg.inv(kernel)

        results = self.session2.run((*elegant1_output[0:6],
                                     *elegant1_output[8:15],
                                     elegant1_output[16],
                                     kernel_inv))

        result = np.array(results)
 
        result = (result +1)/2
        result = np.clip(result,0,1)
        result = result[0][0]*255
        result = np.transpose(result,(1,2,0)).astype(np.uint8)

        if not postprocess:
            return result
        else:
            return self.postprocess(source, crop_face, result)

def _transform(point, center, scale, resolution, invert=False):
    """Generate and affine transformation matrix.  
    Given a set of points, a center, a scale and a targer resolution, the
    function generates and affine transformation matrix. If invert is ``True``
    it will produce the inverse transformation.

    Arguments:
        point {torch.tensor} -- the input 2D point
        center {torch.tensor or numpy.array} -- the center around which to
            perform the transformations
        scale {float} -- the scale of the face/object
        resolution {float} -- the output resolution

    Keyword Arguments:
        invert {bool} -- define wherever the function should produce the direct
            or the inverse transformation matrix (default: {False})
    """
    _pt = np.ones(3)
    _pt[0] = point[0]
    _pt[1] = point[1]

    h = scale  # NOTE: originally, scale * 200
    t = np.eye(3)
    t[0, 0] = resolution / h
    t[1, 1] = resolution / h
    t[0, 2] = resolution * (-center[0] / h + 0.5)
    t[1, 2] = resolution * (-center[1] / h + 0.5)

    if invert:
        t = np.linalg.inv(t)
    new_point = (np.dot(t, _pt))[0:2]
    return new_point.astype(int)


# Copied from face_recognition/face_alignment/face_alignment.py
def _get_preds_from_hm(hm):
    """
    Obtain (x,y) coordinates given a set of N heatmaps.
    ref: 1adrianb/face-alignment/blob/master/face_alignment/utils.py

    Parameters
    ----------
    hm : np.array

    Returns
    -------
    preds:
    preds_orig:
    """

    IMAGE_HEIGHT = 256
    IMAGE_WIDTH = 256

    idx = np.argmax(
        hm.reshape(hm.shape[0], hm.shape[1], hm.shape[2] * hm.shape[3]), axis=2
    )
    idx += 1
    preds = idx.reshape(idx.shape[0], idx.shape[1], 1)
    preds = np.tile(preds, (1, 1, 2)).astype(float)
    preds[..., 0] = (preds[..., 0] - 1) % hm.shape[3] + 1
    preds[..., 1] = np.floor((preds[..., 1] - 1) / (hm.shape[2])) + 1

    for i in range(preds.shape[0]):
        for j in range(preds.shape[1]):
            hm_ = hm[i, j, :]
            pX, pY = int(preds[i, j, 0]) - 1, int(preds[i, j, 1]) - 1
            if pX > 0 and pX < 63 and pY > 0 and pY < 63:
                diff = np.array(
                [
                    hm_[pY, pX + 1] - hm_[pY, pX - 1],
                    hm_[pY + 1, pX] - hm_[pY - 1, pX],
                ]
                ).astype(float)
                preds[i, j] = preds[i, j] + (np.sign(diff) * 0.25)

    preds += -0.5
    preds_orig = np.zeros_like(preds)

    for i in range(hm.shape[0]):
        for j in range(hm.shape[1]):
            preds_orig[i, j] = _transform(
                preds[i, j],  # point
                np.array([IMAGE_HEIGHT // 2, IMAGE_WIDTH // 2]),  # center
                (IMAGE_HEIGHT + IMAGE_WIDTH) // 2,  # FIXME not sure... # scale
                hm.shape[2],  # resolution
                True,
            )
    
    return preds, preds_orig


def resize_by_max(image, max_side=512, force=False):
    h, w = image.shape[:2]
    if max(h, w) < max_side and not force:
        return image
    ratio = max(h, w) / max_side

    w = int(w / ratio + 0.5)
    h = int(h / ratio + 0.5)
    return cv2.resize(image, (w, h))

class PreProcess:

    def detect(self,image: Image) -> "faces":
        #import dlib
        #detector = dlib.get_frontal_face_detector()
        image = np.asarray(image)
        h, w = image.shape[:2]
        #image = resize_by_max(image, 361)
        image = resize_by_max(image, 128)
        image = image / 127.5 - 1.0
        image = np.transpose(image,(2,0,1))
        actual_h, actual_w = image.shape[:2]
        faces_on_small = self.detector.run(np.expand_dims(image,0))
        # postprocessing
        detected = but.postprocess(faces_on_small,
                                   anchor_path= "../../face_detection/blazeface/anchors.npy",)[0][0]
        ymin = int(detected[0] * h)
        xmin = int(detected[1] * w)
        ymax = int(detected[2] * h)
        xmax = int(detected[3] * w)
 

        faces = []
        for face in faces_on_small:
            faces.append(
                [
                    int(xmin),
                    int(ymin),
                    int(xmax),
                    int(ymax),
                ]
            )
        return faces



    def detect_landmark(self, image, face,use_dlib):
       if use_dlib:
           import dlib
           import faceutils.dlibutils as futils_dlib
           predictor = dlib.shape_predictor(
               "../psgan/faceutils/dlibutils/shape_predictor_68_face_landmarks.dat"
           )

           face = dlib.rectangle(*face)
           lms = (
               futils_dlib.landmarks(predictor, image, face)
               * self.img_size
               / image.width
           )
           lms = lms.round()
       else:
           data = np.array(image)
           data = cv2.resize(
               data, (256,256)
           )
           data = data / 255.0
           data = data.transpose((2, 0, 1))  # channel first
           data = data[np.newaxis, :, :, :].astype(
               np.float32
           )  # (batch_size, channel, h, w)
    
           preds_ailia = self.face_alignment.predict(data)
           pts, _ = _get_preds_from_hm(preds_ailia)
           lms = pts.reshape(68, 2) * 4
    
       return lms



    def __init__(self,args, face_parser_path,non_dlib_detector_model=None,non_dlib_face_alegnment_model=None,need_parser=True):
        self.img_size = 256

        xs, ys = np.meshgrid(
            np.linspace(
                0, self.img_size - 1,
                self.img_size
            ),
            np.linspace(
                0, self.img_size - 1,
                self.img_size
            )
        )
        LANDMARK_POINTS = 68
        xs = xs[None].repeat(LANDMARK_POINTS, axis=0)
        ys = ys[None].repeat(LANDMARK_POINTS, axis=0)
        fix = np.concatenate([ys, xs], axis=0) 
        self.fix = fix #(136, h, w)
        if need_parser:
            self.face_parse = futils.mask.FaceParser(args=args,face_parser_path=face_parser_path)

        self.up_ratio    = 0.6 /0.85
        self.down_ratio  = 0.2/ 0.85
        self.width_ratio = 0.2/0.85
        self.lip_class   = [7,9]
        self.face_class  = [1,6]
        self.eyebrow_class  = [2,3]
        self.eye_class  = [4,5]

        self.config_size = 256
        self.use_dlib = args.use_dlib

        self.detector = ailia.Net(*non_dlib_detector_model,env_id=args.env_id)
        if not self.use_dlib:
            self.face_alignment = ailia.Net(*non_dlib_face_alegnment_model,env_id=args.env_id)
    ############################## Mask Process ##############################
    # mask attribute: 0:background 1:face 2:left-eyebrow 3:right-eyebrow 4:left-eye 5: right-eye 6: nose
    # 7: upper-lip 8: teeth 9: under-lip 10:hair 11: left-ear 12: right-ear 13: neck
    def mask_process(self, mask):
        '''
        mask: (1, h, w)
        '''        
        mask_lip = (mask == self.lip_class[0]) + (mask == self.lip_class[1])
        mask_face = (mask == self.face_class[0]) + (mask == self.face_class[1])

        mask_face += (mask == self.eyebrow_class[0])
        mask_face += (mask == self.eyebrow_class[1])

        mask_eye_left = (mask == self.eye_class[0])
        mask_eye_right = (mask == self.eye_class[1])

        #mask_list = [mask_lip, mask_face, mask_eyebrow_left, mask_eyebrow_right, mask_eye_left, mask_eye_right]
        mask_list = [mask_lip, mask_face, mask_eye_left, mask_eye_right]
        mask_aug = np.concatenate(mask_list, axis=0)
        return mask_aug      

    ############################## Landmarks Process ##############################
   
    def diff_process(self, lms):
        '''
        lms:(68, 2)
        '''
        lms = np.transpose(lms,(1, 0)).reshape(-1, 1, 1) # (136, 1, 1)
        diff = self.fix - lms # (136, h, w) 
        return diff


    ############################## Compose Process ##############################
    def preprocess(self, image, is_crop=True):
        '''
        return: image: Image, (H, W), mask: tensor, (1, H, W)
        '''
        image = Image.fromarray(image)
        if self.use_dlib:
            import faceutils.dlibutils  as dlibutils
            face = dlibutils.detect(image)
            face_on_image = face[0]
            face_on_image = [face_on_image.left(),face_on_image.top(),face_on_image.right(),face_on_image.bottom()]
        else:
            face = self.detect(image)
            face_on_image = face[0]
        # face: rectangles, List of rectangles of face region: [(left, top), (right, bottom)]
        if not face:
            return None, None, None

        #is_crop = False
        if is_crop:
            #image, face, crop_face = faceutils.dlibutils.crop(
            image, face, crop_face = futils.nondlibutils.crop(
                image, face_on_image, self.up_ratio, self.down_ratio, self.width_ratio)
        else:
            face = face[0]; crop_face = None
        # image: Image, cropped face
        # face: the same as above
        # crop face: rectangle, face region in cropped face
        np_image = np.array(image) # (h', w', 3)

        mask = self.face_parse.parse(cv2.resize(np_image, (512, 512)))
        # obtain face parsing result
        # mask: Tensor, (512, 512)

        mask = cv2.resize(mask, (self.img_size, self.img_size), interpolation=cv2.INTER_NEAREST)
        mask = mask.astype(int)
        mask = np.expand_dims(mask,0)

        lms = self.detect_landmark(image,face,self.use_dlib)

        lms = np.round(lms).astype(int)

        lms = np.clip(lms,a_min=None,a_max=self.img_size - 1)
        # distinguish upper and lower lips 
        lms[61:64,0] -= 1; lms[65:68,0] += 1
        for i in range(3):
            if np.sum(np.abs(lms[61+i] - lms[67-i])) == 0:
                lms[61+i,0] -= 1;  lms[67-i,0] += 1

        image = image.resize((self.img_size, self.img_size), Image.LANCZOS)
        return [image, mask, lms], face_on_image, crop_face
    
    def process(self, image: Image, mask, lms):
        image = np.array(image)

        image = cv2.resize(image,(self.config_size,self.config_size))
        mean = np.array([0.5,0.5,0.5])
        std = np.array([0.5,0.5,0.5])
        image = image / 255.0
        for i in range(3):
            image[:, :, i] = (image[:, :, i] - mean[i]) / std[i]
        image = np.transpose(image,(2,0,1))

        mask = self.mask_process(mask)
        diff = self.diff_process(lms)
        return [image, mask, diff, lms]
    
    def __call__(self, image, is_crop=True):
        source, face_on_image, crop_face = self.preprocess(image, is_crop)

        if source is None:
            return None, None, None
        return self.process(*source), face_on_image, crop_face


