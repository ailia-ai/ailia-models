import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "True"
import copy
import math
import sys
import time
import unicodedata
# logger
from logging import getLogger

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import json

import ailia

# import original modules
sys.path.append('../../util')
import webcamera_utils  # noqa: E402
from image_utils import imread  # noqa: E402
from model_utils import check_and_download_models  # noqa: E402
from arg_utils import get_base_parser, update_parser, get_savepath  # noqa: E402

logger = getLogger(__name__)

import warnings

warnings.simplefilter("ignore", DeprecationWarning)

# ======================
# Parameters
# ======================
REMOTE_PATH = 'https://storage.googleapis.com/ailia-models/paddle_ocr/'

WEIGHT_PATH_DET_CHN = 'chi_eng_num_sym_server_det_org.onnx'
WEIGHT_PATH_DET_R50_ICDAR15 = 'det_r50_db++_icdar15_infer.onnx'
WEIGHT_PATH_DET_R50_TDTR = 'det_r50_db++_td_tr_infer.onnx'

WEIGHT_PATH_CLS_CHN = 'chi_eng_num_sym_mobile_cls_org.onnx'

WEIGHT_PATH_REC_JPN_MBL = 'jpn_eng_num_sym_mobile_rec_org.onnx'
DICT_PATH_REC_JPN_MBL = './dict/jpn_eng_num_sym_org.txt'

WEIGHT_PATH_REC_JPN_SVR = 'jpn_eng_num_sym_server_rec_add.onnx'
DICT_PATH_REC_JPN_SVR = './dict/jpn_eng_num_sym_add.txt'

WEIGHT_PATH_REC_ENG_MBL = 'eng_num_sym_mobile_rec_org.onnx'
DICT_PATH_REC_ENG_MBL = './dict/eng_num_sym_org.txt'

WEIGHT_PATH_REC_CHN_MBL = 'chi_eng_num_sym_mobile_rec_org.onnx'
DICT_PATH_REC_CHN_MBL = './dict/chi_eng_num_sym_org.txt'

WEIGHT_PATH_REC_CHN_SVR = 'chi_eng_num_sym_server_rec_org.onnx'
DICT_PATH_REC_CHN_SVR = './dict/chi_eng_num_sym_org.txt'

WEIGHT_PATH_REC_GER_MBL = 'ger_eng_num_sym_mobile_rec_org.onnx'
DICT_PATH_REC_GER_MBL = './dict/ger_eng_num_sym_org.txt'

WEIGHT_PATH_REC_FRE_MBL = 'fre_eng_num_sym_mobile_rec_org.onnx'
DICT_PATH_REC_FRE_MBL = './dict/fre_eng_num_sym_org.txt'

WEIGHT_PATH_REC_KOR_MBL = 'kor_eng_num_sym_mobile_rec_org.onnx'
DICT_PATH_REC_KOR_MBL = './dict/kor_eng_num_sym_org.txt'

IMAGE_OR_VIDEO_PATH = 'input.jpg'
SAVE_IMAGE_OR_VIDEO_PATH = 'output.png'

version = ailia.get_version().split(".")
AILIA_VERSION_MAJOR = int(version[0])
AILIA_VERSION_MINOR = int(version[1])
AILIA_VERSION_REVISION = int(version[2])

# Require for ailia SDK <= 1.2.16
REOPEN_REQUIRE_IF_SHAPE_CHANED = (
    (AILIA_VERSION_MAJOR, AILIA_VERSION_MINOR, AILIA_VERSION_REVISION)
    <= (1, 2, 16)
)

# ======================
# Arguemnt Parser Config
# ======================
parser = get_base_parser(
    'PP-OCR: A Practical Ultra Lightweight OCR System',
    IMAGE_OR_VIDEO_PATH,
    SAVE_IMAGE_OR_VIDEO_PATH,
)
parser.add_argument(
    '-c', '--case', default='mobile', choices=('mobile', 'server'),
    help=('You can choose the following model size.'
          '  - mobile : fast and light but low accuracy'
          '  - server : high accuracy but slow and heavy')
)
parser.add_argument(
    '-l', '--language', type=str, default='japanese',
    help=('You can specify OCR for the following languages.'
          '  - japanese, jpn, jp'
          '  - english, eng, en'
          '  - chinese, chi, ch'
          '  - german, ger, ge'
          '  - french, fre, fr'
          '  - korean, kor, ko')
)
parser.add_argument(
    '-lt', '--det_limit_type', type=str, default='max',
    help=('You can limit the size of the input image for text detection.'
          '  - max : Resize based on long side'
          '  - min : Resize based on short side')
)
parser.add_argument(
    '-ll', '--det_limit_side_len', type=int, default=1920,
    help=('You can limit the size of the input image for text detection.'
          'Please set a positive integer.'
          'Generally set to a multiple of 32, such as 960.')
)
parser.add_argument(
    '-d', '--det_model', default='db_res18', choices=('db_res18', 'r50_icdar15', 'r50_trtd'),
    help='det model type'
)
parser.add_argument(
    '-w', '--write_results',
    action='store_true',
    help='Flag to output results to file.'
)
args = update_parser(parser)

det_model = args.det_model


# ======================
# Utils
# ======================

def get_default_config():
    dc = {}

    # params for text detector
    dc['det_algorithm'] = 'DB'
    dc['det_model_path'] = WEIGHT_PATH_DET_CHN
    dc['det_limit_side_len'] = 0  # set by args, defalt 1920
    dc['det_limit_type'] = ''  # set by args, defalt max

    # DB params
    dc['det_db_thresh'] = 0.3
    dc['det_db_box_thresh'] = 0.5
    dc['det_db_unclip_ratio'] = 1.6

    # params for text recognizer
    dc['rec_algorithm'] = 'CRNN'
    dc['rec_model_path'] = WEIGHT_PATH_REC_JPN_SVR
    dc['rec_image_shape'] = '3, 32, 320'
    dc['rec_char_type'] = 'ch'
    dc['rec_batch_num'] = 6
    dc['max_text_length'] = 25
    dc['rec_char_dict_path'] = DICT_PATH_REC_JPN_SVR
    dc['use_space_char'] = True
    if sys.platform == "win32":
        # Windows
        dc['vis_font_path'] = 'C:/windows/Fonts/meiryo.ttc'
    elif sys.platform == "darwin":
        # Mac OS
        dc['vis_font_path'] = '/System/Library/Fonts/ヒラギノ丸ゴ ProN W4.ttc'
    else:
        # Linux
        dc['vis_font_path'] = '/usr/share/fonts/opentype/ipaexfont-gothic/ipaexg.ttf'
    dc['drop_score'] = 0.5  # this is threshold of rec
    dc['rec_bbox_padding'] = 0.1
    dc['limited_max_width'] = 1280
    dc['limited_min_width'] = 16

    # params for text classifier
    dc['use_angle_cls'] = True
    dc['cls_model_path'] = WEIGHT_PATH_CLS_CHN
    dc['cls_image_shape'] = '3, 48, 192'
    dc['label_list'] = ['0', '180']
    dc['cls_batch_num'] = 30
    dc['cls_thresh'] = 0.9

    return dc


def set_config(dc, weight_path_det, det_algorithm,
               weight_path_rec, dict_path_rec, weight_path_cls):
    # params for text detector
    dc['det_model_path'] = weight_path_det
    dc['det_algorithm'] = det_algorithm

    # params for text recognizer
    dc['rec_model_path'] = weight_path_rec
    dc['rec_char_dict_path'] = dict_path_rec

    # params for text classifier
    dc['cls_model_path'] = weight_path_cls

    return dc


def transform(data, ops=None):
    """ transform """
    if ops is None:
        ops = []
    for op in ops:
        data = op(data)
        if data is None:
            return None
    return data


def create_operators(op_param_list):
    """
    create operators based on the config
    Args:
        params(list): a dict list, used to create some operators
    """
    assert isinstance(op_param_list, list), ('operator config should be a list')
    ops = []
    for operator in op_param_list:
        assert isinstance(operator, dict) \
               and len(operator) == 1, "yaml format error"
        op_name = list(operator)[0]
        param = {} if operator[op_name] is None else operator[op_name]
        op = eval(op_name)(**param)
        ops.append(op)

    return ops


def build_post_process(config):
    support_dict = [
        'DBPostProcess', 'CTCLabelDecode', 'ClsPostProcess'
    ]

    config = copy.deepcopy(config)
    module_name = config.pop('name')
    assert module_name in support_dict, Exception(
        'post process only support {}'.format(support_dict))
    module_class = eval(module_name)(**config)

    return module_class


# ======================
# Operators
# ======================

class DetResizeForTest(object):
    def __init__(self, **kwargs):
        super(DetResizeForTest, self).__init__()
        self.resize_type = 0
        if 'image_shape' in kwargs:
            self.image_shape = kwargs['image_shape']
            self.resize_type = 1
        elif 'limit_side_len' in kwargs:
            self.limit_side_len = kwargs['limit_side_len']
            self.limit_type = kwargs.get('limit_type', 'min')
        elif 'resize_long' in kwargs:
            self.resize_type = 2
            self.resize_long = kwargs.get('resize_long', 960)
        else:
            self.limit_side_len = 736
            self.limit_type = 'min'

    def __call__(self, data):
        img = data['image']
        src_h, src_w, _ = img.shape

        if self.resize_type == 0:
            img, [ratio_h, ratio_w] = self.resize_image_type0(img)
        elif self.resize_type == 2:
            img, [ratio_h, ratio_w] = self.resize_image_type2(img)
        else:
            img, [ratio_h, ratio_w] = self.resize_image_type1(img)
        data['image'] = img
        data['shape'] = np.array([src_h, src_w, ratio_h, ratio_w])
        return data

    def resize_image_type1(self, img):
        resize_h, resize_w = self.image_shape
        ori_h, ori_w = img.shape[:2]  # (h, w, c)
        ratio_h = float(resize_h) / ori_h
        ratio_w = float(resize_w) / ori_w
        img = cv2.resize(img, (int(resize_w), int(resize_h)))

        return img, [ratio_h, ratio_w]

    def resize_image_type0(self, img):
        """
        resize image to a size multiple of 32 which is required by the network
        args:
            img(array): array with shape [h, w, c]
        return(tuple):
            img, (ratio_h, ratio_w)
        """
        limit_side_len = self.limit_side_len
        h, w, _ = img.shape

        # limit the max side
        if self.limit_type == 'max':
            if max(h, w) > limit_side_len:
                if h > w:
                    ratio = float(limit_side_len) / h
                else:
                    ratio = float(limit_side_len) / w
            else:
                ratio = 1.
        else:
            if min(h, w) < limit_side_len:
                if h < w:
                    ratio = float(limit_side_len) / h
                else:
                    ratio = float(limit_side_len) / w
            else:
                ratio = 1.
        resize_h = int(h * ratio)
        resize_w = int(w * ratio)

        resize_h = int(round(resize_h / 32) * 32)
        resize_w = int(round(resize_w / 32) * 32)

        try:
            if int(resize_w) <= 0 or int(resize_h) <= 0:
                return None, (None, None)
            img = cv2.resize(img, (int(resize_w), int(resize_h)))
        except:
            logger.error(img.shape, resize_w, resize_h)
            sys.exit(0)
        ratio_h = resize_h / float(h)
        ratio_w = resize_w / float(w)

        return img, [ratio_h, ratio_w]

    def resize_image_type2(self, img):
        h, w, _ = img.shape

        resize_w = w
        resize_h = h

        # Fix the longer side
        if resize_h > resize_w:
            ratio = float(self.resize_long) / resize_h
        else:
            ratio = float(self.resize_long) / resize_w

        resize_h = int(resize_h * ratio)
        resize_w = int(resize_w * ratio)

        max_stride = 128
        resize_h = (resize_h + max_stride - 1) // max_stride * max_stride
        resize_w = (resize_w + max_stride - 1) // max_stride * max_stride
        img = cv2.resize(img, (int(resize_w), int(resize_h)))
        ratio_h = resize_h / float(h)
        ratio_w = resize_w / float(w)

        return img, [ratio_h, ratio_w]


class NormalizeImage(object):
    """ normalize image such as substract mean, divide std
    """

    def __init__(self, scale=None, mean=None, std=None, order='chw', **kwargs):
        if isinstance(scale, str):
            scale = eval(scale)
        self.scale = np.float32(scale if scale is not None else 1.0 / 255.0)
        mean = mean if mean is not None else [0.485, 0.456, 0.406]
        std = std if std is not None else [0.229, 0.224, 0.225]

        shape = (3, 1, 1) if order == 'chw' else (1, 1, 3)
        self.mean = np.array(mean).reshape(shape).astype('float32')
        self.std = np.array(std).reshape(shape).astype('float32')

    def __call__(self, data):
        img = data['image']
        data['image'] = (img.astype('float32') * self.scale - self.mean) / self.std
        return data


class ToCHWImage(object):
    """ convert hwc image to chw image
    """

    def __init__(self, **kwargs):
        pass

    def __call__(self, data):
        img = data['image']
        data['image'] = img.transpose((2, 0, 1))
        return data


class KeepKeys(object):
    def __init__(self, keep_keys, **kwargs):
        self.keep_keys = keep_keys

    def __call__(self, data):
        data_list = []
        for key in self.keep_keys:
            data_list.append(data[key])
        return data_list


class DBPostProcess(object):
    """
    The post process for Differentiable Binarization (DB).
    """

    def __init__(self,
                 thresh=0.3,
                 box_thresh=0.7,
                 max_candidates=1000,
                 unclip_ratio=2.0,
                 use_dilation=False,
                 **kwargs):
        self.thresh = thresh
        self.box_thresh = box_thresh
        self.max_candidates = max_candidates
        self.unclip_ratio = unclip_ratio
        self.min_size = 3
        self.dilation_kernel = None if not use_dilation \
            else np.array([[1, 1], [1, 1]])

    def boxes_from_bitmap(self, pred, _bitmap, dest_width, dest_height):
        '''
        _bitmap: single map with shape (1, H, W),
                whose values are binarized as {0, 1}
        '''

        bitmap = _bitmap
        height, width = bitmap.shape

        outs = cv2.findContours((bitmap * 255).astype(np.uint8), cv2.RETR_LIST,
                                cv2.CHAIN_APPROX_SIMPLE)
        if len(outs) == 3:
            img, contours, _ = outs[0], outs[1], outs[2]
        elif len(outs) == 2:
            contours, _ = outs[0], outs[1]

        num_contours = min(len(contours), self.max_candidates)

        boxes = []
        scores = []
        for index in range(num_contours):
            contour = contours[index]
            points, sside = self.get_mini_boxes(contour)
            if sside < self.min_size:
                continue
            points = np.array(points)
            score = self.box_score_fast(pred, points.reshape(-1, 2))
            if self.box_thresh > score:
                continue

            box = self.unclip(points).reshape(-1, 1, 2)
            box, sside = self.get_mini_boxes(box)
            if sside < self.min_size + 2:
                continue
            box = np.array(box)

            box[:, 0] = np.clip(
                np.round(box[:, 0] / width * dest_width), 0, dest_width)
            box[:, 1] = np.clip(
                np.round(box[:, 1] / height * dest_height), 0, dest_height)
            boxes.append(box.astype(int))
            scores.append(score)

        return np.array(boxes, dtype=np.int16), scores

    def xyrotate(self, coord_xy, angle, center_xy):
        # exec rotate
        rotation_matrix = cv2.getRotationMatrix2D((center_xy[0], center_xy[1]), angle, 1)
        # make variable for output
        coord_xy_rotated = np.zeros(np.shape(coord_xy))
        # loop of coordinate
        for coord_i in range(len(coord_xy)):
            # set x, y
            coord_x_tmp = coord_xy[coord_i, 0]
            coord_y_tmp = coord_xy[coord_i, 1]
            # slide to suit center of rotation
            coord_x_tmp -= center_xy[0]
            coord_y_tmp -= center_xy[1]
            # exec rotation
            coord_xy_tmp = np.array([coord_x_tmp, coord_y_tmp])[:, np.newaxis]
            rotation_matrix_tmp = np.array([
                [np.cos(-angle / 180 * np.pi), -np.sin(-angle / 180 * np.pi)],
                [np.sin(-angle / 180 * np.pi), np.cos(-angle / 180 * np.pi)]
            ])
            coord_xy_tmp = rotation_matrix_tmp @ coord_xy_tmp
            # re-slide to suit center of rotation
            coord_xy_tmp = coord_xy_tmp.reshape(-1)
            coord_xy_tmp[0] += center_xy[0]
            coord_xy_tmp[1] += center_xy[1]
            # stock
            coord_xy_rotated[coord_i, :] = coord_xy_tmp

        return coord_xy_rotated

    def unclip(self, box):
        unclip_ratio = self.unclip_ratio
        poly_area = (np.sqrt(np.sum((box[0, :] - box[1, :]) ** 2)) *
                     np.sqrt(np.sum((box[0, :] - box[3, :]) ** 2)))
        poly_length = (np.sqrt(np.sum((box[0, :] - box[1, :]) ** 2)) +
                       np.sqrt(np.sum((box[0, :] - box[3, :]) ** 2))) * 2
        distance = poly_area * unclip_ratio / poly_length
        # calc angle between upper side of bbox with x axis
        u = box[1] - box[0]
        v = box[1] - box[0]
        v[1] = 0
        i = np.inner(u, v)
        n = np.linalg.norm(u) * np.linalg.norm(v)
        c = i / n
        angle = np.rad2deg(np.arccos(np.clip(c, -1.0, 1.0)))
        # exec coordinates rotation 
        box_ = self.xyrotate(coord_xy=box, angle=angle,
                             center_xy=np.mean(box, axis=0))
        # calculate circle coordinates
        pitch = 10
        x_upper = np.cos(np.arange(1, 0, (-1 / pitch)) * np.pi) * distance
        y_upper = -np.sqrt(np.maximum(distance ** 2 - x_upper ** 2, 0.0))
        x_lower = np.cos(np.arange(0, 1, (1 / pitch)) * np.pi) * distance
        y_lower = np.sqrt(np.maximum(distance ** 2 - x_lower ** 2, 0.0))
        x = np.concatenate([x_upper, x_lower])
        y = np.concatenate([y_upper, y_lower])
        circle = np.concatenate([x[:, np.newaxis], y[:, np.newaxis]], axis=1)
        # calculate circle coordinates around four corners
        expanded = []
        for box_tmp in box_:
            expanded.append(circle + box_tmp)
        expanded = np.array(expanded).reshape(-1, 2)
        # narrow down circle coordinates to outside 
        expanded = expanded[[25, 26, 27, 28, 29, 30, 50, 51, 52, 53, 54, 55,
                             75, 76, 77, 78, 79, 60, 0, 1, 2, 3, 4, 5]]
        # exec coordinates re-rotation 
        expanded = self.xyrotate(coord_xy=expanded, angle=-angle,
                                 center_xy=np.mean(box_, axis=0))
        expanded = np.round(expanded).astype(np.int64)
        return expanded

    def get_mini_boxes(self, contour):
        bounding_box = cv2.minAreaRect(contour)
        points = sorted(list(cv2.boxPoints(bounding_box)), key=lambda x: x[0])

        index_1, index_2, index_3, index_4 = 0, 1, 2, 3
        if points[1][1] > points[0][1]:
            index_1 = 0
            index_4 = 1
        else:
            index_1 = 1
            index_4 = 0
        if points[3][1] > points[2][1]:
            index_2 = 2
            index_3 = 3
        else:
            index_2 = 3
            index_3 = 2

        box = [
            points[index_1], points[index_2], points[index_3], points[index_4]
        ]
        return box, min(bounding_box[1])

    def box_score_fast(self, bitmap, _box):
        h, w = bitmap.shape[:2]
        box = _box.copy()
        xmin = np.clip(np.floor(box[:, 0].min()).astype(int), 0, w - 1)
        xmax = np.clip(np.ceil(box[:, 0].max()).astype(int), 0, w - 1)
        ymin = np.clip(np.floor(box[:, 1].min()).astype(int), 0, h - 1)
        ymax = np.clip(np.ceil(box[:, 1].max()).astype(int), 0, h - 1)

        mask = np.zeros((ymax - ymin + 1, xmax - xmin + 1), dtype=np.uint8)
        box[:, 0] = box[:, 0] - xmin
        box[:, 1] = box[:, 1] - ymin
        cv2.fillPoly(mask, box.reshape(1, -1, 2).astype(np.int32), 1)
        return cv2.mean(bitmap[ymin:ymax + 1, xmin:xmax + 1], mask)[0]

    def __call__(self, outs_dict, shape_list):
        pred = outs_dict['maps']
        pred = pred[:, 0, :, :]
        segmentation = pred > self.thresh

        boxes_batch = []
        for batch_index in range(pred.shape[0]):
            src_h, src_w, ratio_h, ratio_w = shape_list[batch_index]
            if self.dilation_kernel is not None:
                mask = cv2.dilate(
                    np.array(segmentation[batch_index]).astype(np.uint8),
                    self.dilation_kernel)
            else:
                mask = segmentation[batch_index]
            boxes, scores = self.boxes_from_bitmap(
                pred[batch_index], mask, src_w, src_h)

            boxes_batch.append({'points': boxes})
        return boxes_batch


class BaseRecLabelDecode(object):
    """ Convert between text-label and text-index """

    def __init__(self,
                 character_dict_path=None,
                 character_type='ch',
                 use_space_char=False):
        support_character_type = [
            'ch', 'en', 'en_sensitive', 'french', 'german', 'japan', 'korean'
        ]
        assert character_type in support_character_type, "Only {} are supported now but get {}".format(
            support_character_type, character_type)

        if character_type == "en":
            self.character_str = "0123456789abcdefghijklmnopqrstuvwxyz"
            dict_character = list(self.character_str)
        elif character_type in ["ch", "french", "german", "japan", "korean"]:
            self.character_str = ""
            assert character_dict_path is not None, "character_dict_path should not be None when character_type is ch"
            with open(character_dict_path, "rb") as fin:
                lines = fin.readlines()
                for line in lines:
                    line = line.decode('utf-8').strip("\n").strip("\r\n")
                    self.character_str += line
            if use_space_char:
                self.character_str += " "
            dict_character = list(self.character_str)
        elif character_type == "en_sensitive":
            # same with ASTER setting (use 94 char).
            import string
            self.character_str = string.printable[:-6]
            dict_character = list(self.character_str)
        else:
            raise NotImplementedError
        self.character_type = character_type
        dict_character = self.add_special_char(dict_character)
        self.dict = {}
        for i, char in enumerate(dict_character):
            self.dict[char] = i
        self.character = dict_character

    def add_special_char(self, dict_character):
        return dict_character

    def decode(self, text_index, text_prob=None, is_remove_duplicate=True):
        """ convert text-index into text-label. """
        result_list = []
        ignored_tokens = self.get_ignored_tokens()
        batch_size = len(text_index)
        for batch_idx in range(batch_size):
            char_list = []
            conf_list = []
            for idx in range(len(text_index[batch_idx])):
                if text_index[batch_idx][idx] in ignored_tokens:
                    continue
                if is_remove_duplicate:
                    # only for predict
                    if idx > 0 and text_index[batch_idx][idx - 1] == \
                            text_index[batch_idx][idx]:
                        continue
                # print('int(text_index[batch_idx][idx]) =', 
                #        int(text_index[batch_idx][idx]))
                # print('self.character[int(text_index[batch_idx][idx])] =', 
                #        self.character[int(text_index[batch_idx][idx])])
                char_list.append(
                    self.character[int(text_index[batch_idx][idx])])
                if text_prob is not None:
                    conf_list.append(text_prob[batch_idx][idx])
                else:
                    conf_list.append(1)
            text = ''.join(char_list)
            if (conf_list == []):
                result_list.append((text, np.array(conf_list)))
            else:
                result_list.append((text, np.mean(conf_list)))
        return result_list

    def get_ignored_tokens(self):
        return [0]  # for ctc blank


class CTCLabelDecode(BaseRecLabelDecode):
    """ Convert between text-label and text-index """

    def __init__(self,
                 character_dict_path=None,
                 character_type='ch',
                 use_space_char=False,
                 **kwargs):
        super(CTCLabelDecode, self).__init__(
            character_dict_path, character_type, use_space_char)

    def __call__(self, preds, label=None, *args, **kwargs):
        preds_idx = preds.argmax(axis=2)
        preds_prob = preds.max(axis=2)
        text = self.decode(preds_idx, preds_prob)
        if label is None:
            return text
        label = self.decode(label, is_remove_duplicate=False)
        return text, label

    def add_special_char(self, dict_character):
        dict_character = ['blank'] + dict_character
        return dict_character


class ClsPostProcess(object):
    """ Convert between text-label and text-index """

    def __init__(self, label_list, **kwargs):
        super(ClsPostProcess, self).__init__()
        self.label_list = label_list

    def __call__(self, preds, label=None, *args, **kwargs):
        pred_idxs = preds.argmax(axis=1)
        decode_out = [(self.label_list[idx], preds[i, idx])
                      for i, idx in enumerate(pred_idxs)]
        if label is None:
            return decode_out
        label = [(self.label_list[idx], 1.0) for idx in label]
        return decode_out, label


# ======================
# Main functions
# ======================

class TextDetector(object):
    def __init__(self, config, env_id):
        self.config = copy.deepcopy(config)
        self.env_id = env_id
        self.det_algorithm = config['det_algorithm']

        pre_process_list = [
            {
                'DetResizeForTest': {
                    'limit_side_len': config['det_limit_side_len'],
                    'limit_type': config['det_limit_type']
                }
            },
            {
                'NormalizeImage': {
                    'std': [0.229, 0.224, 0.225],
                    'mean': [0.485, 0.456, 0.406],
                    'scale': '1./255.',
                    'order': 'hwc'
                }
            },
            {
                'ToCHWImage': None
            },
            {
                'KeepKeys': {
                    'keep_keys': ['image', 'shape']
                }
            }
        ]

        postprocess_params = {
            'name': 'DBPostProcess',
            "thresh": config['det_db_thresh'],
            "box_thresh": config['det_db_box_thresh'],
            "max_candidates": 1000,
            "unclip_ratio": config['det_db_unclip_ratio'],
            "use_dilation": True
        }
        if self.det_algorithm == "DB++":
            pre_process_list[1] = {
                'NormalizeImage': {
                    'std': [1.0, 1.0, 1.0],
                    'mean': [0.48109378172549, 0.45752457890196, 0.40787054090196],
                    'scale': '1./255.',
                    'order': 'hwc'
                }
            }
            postprocess_params["use_dilation"] = False

        self.preprocess_op = create_operators(pre_process_list)
        self.postprocess_op = build_post_process(postprocess_params)
        self.net = ailia.Net(self.config['det_model_path'] + '.prototxt',
                             self.config['det_model_path'], env_id=self.env_id)
        self.called = False

    def order_points_clockwise(self, pts):
        """
        reference from: https://github.com/jrosebr1/imutils/blob/master/imutils/perspective.py
        # sort the points based on their x-coordinates
        """
        xSorted = pts[np.argsort(pts[:, 0]), :]

        # grab the left-most and right-most points from the sorted
        # x-roodinate points
        leftMost = xSorted[:2, :]
        rightMost = xSorted[2:, :]

        # now, sort the left-most coordinates according to their
        # y-coordinates so we can grab the top-left and bottom-left
        # points, respectively
        leftMost = leftMost[np.argsort(leftMost[:, 1]), :]
        (tl, bl) = leftMost

        rightMost = rightMost[np.argsort(rightMost[:, 1]), :]
        (tr, br) = rightMost

        rect = np.array([tl, tr, br, bl], dtype="float32")
        return rect

    def clip_det_res(self, points, img_height, img_width):
        for pno in range(points.shape[0]):
            points[pno, 0] = int(min(max(points[pno, 0], 0), img_width - 1))
            points[pno, 1] = int(min(max(points[pno, 1], 0), img_height - 1))
        return points

    def filter_tag_det_res(self, dt_boxes, image_shape):
        img_height, img_width = image_shape[0:2]
        dt_boxes_new = []
        for box in dt_boxes:
            box = self.order_points_clockwise(box)
            box = self.clip_det_res(box, img_height, img_width)
            rect_width = int(np.linalg.norm(box[0] - box[1]))
            rect_height = int(np.linalg.norm(box[0] - box[3]))
            if rect_width <= 3 or rect_height <= 3:
                continue
            dt_boxes_new.append(box)
        dt_boxes = np.array(dt_boxes_new)
        return dt_boxes

    def __call__(self, img):
        ori_im = img.copy()
        data = {'image': img}

        data = transform(data, self.preprocess_op)
        img, shape_list = data
        img = np.expand_dims(img, axis=0)
        shape_list = np.expand_dims(shape_list, axis=0)
        img = img.copy()
        starttime = time.time()

        # net initialize, Text Detection
        if self.called and self.net.get_input_shape() != img.shape and REOPEN_REQUIRE_IF_SHAPE_CHANED:
            self.net = ailia.Net(self.config['det_model_path'] + '.prototxt',
                                 self.config['det_model_path'], env_id=self.env_id)
        outputs = self.net.predict(img)
        self.called = True

        preds = {'maps': outputs}

        post_result = self.postprocess_op(preds, shape_list)
        dt_boxes = post_result[0]['points']
        dt_boxes = self.filter_tag_det_res(dt_boxes, ori_im.shape)
        elapse = time.time() - starttime

        return dt_boxes, elapse


class TextClassifier(object):
    def __init__(self, config, env_id):
        OCR_CFG = config
        self.cfg = OCR_CFG
        self.env_id = env_id

        self.cls_image_shape = [int(v) for v in OCR_CFG['cls_image_shape'].split(",")]
        self.cls_batch_num = OCR_CFG['cls_batch_num']
        self.cls_thresh = OCR_CFG['cls_thresh']
        postprocess_params = {
            'name': 'ClsPostProcess',
            "label_list": OCR_CFG['label_list'],
        }
        self.postprocess_op = build_post_process(postprocess_params)
        self.net = ailia.Net(self.cfg['cls_model_path'] + '.prototxt',
                             self.cfg['cls_model_path'], env_id=self.env_id)
        self.called = False

    def resize_norm_img(self, img):
        imgC, imgH, imgW = self.cls_image_shape
        h = img.shape[0]
        w = img.shape[1]
        ratio = w / float(h)
        if math.ceil(imgH * ratio) > imgW:
            resized_w = imgW
        else:
            resized_w = int(math.ceil(imgH * ratio))
        resized_image = cv2.resize(img, (resized_w, imgH))
        resized_image = resized_image.astype('float32')
        if self.cls_image_shape[0] == 1:
            resized_image = resized_image / 255
            resized_image = resized_image[np.newaxis, :]
        else:
            resized_image = resized_image.transpose((2, 0, 1)) / 255
        resized_image -= 0.5
        resized_image /= 0.5
        padding_im = np.zeros((imgC, imgH, imgW), dtype=np.float32)
        padding_im[:, :, 0:resized_w] = resized_image
        return padding_im

    def __call__(self, img_list):
        img_list = copy.deepcopy(img_list)
        img_num = len(img_list)
        # Calculate the aspect ratio of all text bars
        width_list = []
        for img in img_list:
            width_list.append(img.shape[1] / float(img.shape[0]))
        # Sorting can speed up the cls process
        indices = np.argsort(np.array(width_list))

        cls_res = [['', 0.0]] * img_num
        batch_num = self.cls_batch_num
        elapse = 0
        for beg_img_no in range(0, img_num, batch_num):
            end_img_no = min(img_num, beg_img_no + batch_num)
            norm_img_batch = []
            max_wh_ratio = 0
            for ino in range(beg_img_no, end_img_no):
                h, w = img_list[indices[ino]].shape[0:2]
                wh_ratio = w * 1.0 / h
                max_wh_ratio = max(max_wh_ratio, wh_ratio)
            for ino in range(beg_img_no, end_img_no):
                norm_img = self.resize_norm_img(img_list[indices[ino]])
                norm_img = norm_img[np.newaxis, :]
                norm_img_batch.append(norm_img)
            norm_img_batch = np.concatenate(norm_img_batch)
            norm_img_batch = norm_img_batch.copy()
            starttime = time.time()

            # net initialize, Detection Boxes Rectify
            if self.called and self.net.get_input_shape() != norm_img_batch.shape and REOPEN_REQUIRE_IF_SHAPE_CHANED:
                self.net = ailia.Net(self.cfg['cls_model_path'] + '.prototxt',
                                     self.cfg['cls_model_path'], env_id=self.env_id)
            self.net.set_input_shape(norm_img_batch.shape)
            prob_out = self.net.predict(norm_img_batch)
            self.called = True

            cls_result = self.postprocess_op(prob_out)
            elapse += time.time() - starttime
            for rno in range(len(cls_result)):
                label, score = cls_result[rno]
                cls_res[indices[beg_img_no + rno]] = [label, score]
                if '180' in label and score > self.cls_thresh:
                    img_list[indices[beg_img_no + rno]] = cv2.rotate(
                        img_list[indices[beg_img_no + rno]], 1)

        return img_list, cls_res, elapse


class TextRecognizer(object):
    def __init__(self, config, env_id):
        OCR_CFG = config
        self.config = OCR_CFG
        self.env_id = env_id

        self.limited_max_width = OCR_CFG['limited_max_width']
        self.limited_min_width = OCR_CFG['limited_min_width']

        self.rec_image_shape = [int(v) for v in OCR_CFG['rec_image_shape'].split(",")]
        self.character_type = OCR_CFG['rec_char_type']
        self.rec_batch_num = OCR_CFG['rec_batch_num']
        self.rec_algorithm = OCR_CFG['rec_algorithm']
        postprocess_params = {
            'name': 'CTCLabelDecode',
            "character_type": OCR_CFG['rec_char_type'],
            "character_dict_path": OCR_CFG['rec_char_dict_path'],
            "use_space_char": OCR_CFG['use_space_char']
        }
        self.postprocess_op = build_post_process(postprocess_params)
        self.net = ailia.Net(self.config['rec_model_path'] + '.prototxt',
                             self.config['rec_model_path'], env_id=self.env_id)
        self.called = False

    def resize_norm_img(self, img, max_wh_ratio):
        imgC, imgH, imgW = self.rec_image_shape
        assert imgC == img.shape[2]
        if self.character_type == "ch":
            imgW = int((32 * max_wh_ratio))
        imgW = max(min(imgW, self.limited_max_width), self.limited_min_width)
        h, w = img.shape[:2]
        ratio = w / float(h)
        ratio_imgH = math.ceil(imgH * ratio)
        ratio_imgH = max(ratio_imgH, self.limited_min_width)
        if ratio_imgH > imgW:
            resized_w = imgW
        else:
            resized_w = int(math.ceil(imgH * ratio))
        resized_image = cv2.resize(img, (resized_w, imgH))
        resized_image = resized_image.astype('float32')
        resized_image = resized_image.transpose((2, 0, 1)) / 255
        resized_image -= 0.5
        resized_image /= 0.5
        padding_im = np.zeros((imgC, imgH, imgW), dtype=np.float32)
        padding_im[:, :, 0:resized_w] = resized_image
        return padding_im

    def __call__(self, img_list):
        img_num = len(img_list)
        # Calculate the aspect ratio of all text bars
        width_list = []
        for img in img_list:
            width_list.append(img.shape[1] / float(img.shape[0]))
        # Sorting can speed up the recognition process
        indices = np.argsort(np.array(width_list))

        # rec_res = []
        rec_res = [['', 0.0]] * img_num
        batch_num = self.rec_batch_num
        elapse = 0
        for beg_img_no in range(0, img_num, batch_num):
            end_img_no = min(img_num, beg_img_no + batch_num)
            norm_img_batch = []
            max_wh_ratio = 0
            for ino in range(beg_img_no, end_img_no):
                # h, w = img_list[ino].shape[0:2]
                h, w = img_list[indices[ino]].shape[0:2]
                wh_ratio = w * 1.0 / h
                max_wh_ratio = max(max_wh_ratio, wh_ratio)
            for ino in range(beg_img_no, end_img_no):
                # norm_img = self.resize_norm_img(img_list[ino], max_wh_ratio)
                norm_img = self.resize_norm_img(img_list[indices[ino]],
                                                max_wh_ratio)
                norm_img = norm_img[np.newaxis, :]
                norm_img_batch.append(norm_img)
            norm_img_batch = np.concatenate(norm_img_batch)
            norm_img_batch = norm_img_batch.copy()
            starttime = time.time()

            # net initialize, Text Recognition
            if self.called and self.net.get_input_shape() != norm_img_batch.shape and REOPEN_REQUIRE_IF_SHAPE_CHANED:
                self.net = ailia.Net(self.config['rec_model_path'] + '.prototxt',
                                     self.config['rec_model_path'], env_id=self.env_id)
            preds = self.net.predict(norm_img_batch)
            self.called = True

            rec_result = self.postprocess_op(preds)
            for rno in range(len(rec_result)):
                rec_res[indices[beg_img_no + rno]] = rec_result[rno]
            elapse += time.time() - starttime
        return rec_res, elapse


class TextSystem(object):
    def __init__(self, config, env_id):
        self.cfg = config

        self.text_detector = TextDetector(config, env_id)
        self.text_recognizer = TextRecognizer(config, env_id)
        self.use_angle_cls = config['use_angle_cls']
        self.drop_score = config['drop_score']
        if self.use_angle_cls:
            self.text_classifier = TextClassifier(config, env_id)

    def get_rotate_crop_image(self, img, points):
        '''
        img_height, img_width = img.shape[0:2]
        left = int(np.min(points[:, 0]))
        right = int(np.max(points[:, 0]))
        top = int(np.min(points[:, 1]))
        bottom = int(np.max(points[:, 1]))
        img_crop = img[top:bottom, left:right, :].copy()
        points[:, 0] = points[:, 0] - left
        points[:, 1] = points[:, 1] - top
        '''
        img_crop_width = int(
            max(
                np.linalg.norm(points[0] - points[1]),
                np.linalg.norm(points[2] - points[3])))
        img_crop_height = int(
            max(
                np.linalg.norm(points[0] - points[3]),
                np.linalg.norm(points[1] - points[2])))
        pts_std = np.float32([[0, 0], [img_crop_width, 0],
                              [img_crop_width, img_crop_height],
                              [0, img_crop_height]])
        M = cv2.getPerspectiveTransform(points, pts_std)
        dst_img = cv2.warpPerspective(
            img,
            M, (img_crop_width, img_crop_height),
            borderMode=cv2.BORDER_REPLICATE,
            flags=cv2.INTER_CUBIC)
        dst_img_height, dst_img_width = dst_img.shape[0:2]
        if dst_img_height * 1.0 / dst_img_width >= 1.5:
            dst_img = np.rot90(dst_img)
        return dst_img

    def __call__(self, img):
        ori_im = img.copy()
        dt_boxes, elapse = self.text_detector(img)
        logger.info("dt_boxes num : {}, elapse : {}".format(
            len(dt_boxes), elapse))
        if dt_boxes is None:
            return None, None
        img_crop_list = []

        dt_boxes = sorted_boxes(dt_boxes)

        if (len(dt_boxes) > 0):
            ratio_padding = self.cfg['rec_bbox_padding']
            dt_boxes = np.array(dt_boxes)
            height_vec = dt_boxes[:, 2, :] - dt_boxes[:, 1, :]
            width_vec = dt_boxes[:, 3, :] - dt_boxes[:, 0, :]
            if (np.sum(height_vec ** 2) > np.sum(width_vec ** 2)):
                height_vec = width_vec
            padding_vec_tmp = height_vec * ratio_padding
            padding_vec = padding_vec_tmp.copy()
            padding_vec[:, 0] += padding_vec_tmp[:, 1]
            padding_vec[:, 1] -= padding_vec_tmp[:, 0]
            padding_vec = np.round(padding_vec)
            dt_boxes[:, 0, :] -= padding_vec
            dt_boxes[:, 2, :] += padding_vec
            padding_vec[:, 0] *= -1
            dt_boxes[:, 1, :] -= padding_vec
            dt_boxes[:, 3, :] += padding_vec

        for bno in range(len(dt_boxes)):
            tmp_box = copy.deepcopy(dt_boxes[bno])
            img_crop = self.get_rotate_crop_image(ori_im, tmp_box)
            img_crop_list.append(img_crop)
        if self.use_angle_cls:
            img_crop_list, angle_list, elapse = self.text_classifier(
                img_crop_list)
            logger.info("cls num  : {}, elapse : {}".format(
                len(img_crop_list), elapse))

        rec_res, elapse = self.text_recognizer(img_crop_list)
        logger.info("rec_res num  : {}, elapse : {}".format(
            len(rec_res), elapse))
        filter_boxes, filter_rec_res = [], []
        for box, rec_reuslt in zip(dt_boxes, rec_res):
            text, score = rec_reuslt
            if np.isscalar(score) and score >= self.drop_score: # ignore empty ndarray
                filter_boxes.append(box)
                filter_rec_res.append(rec_reuslt)

        return filter_boxes, filter_rec_res


def sorted_boxes(dt_boxes):
    """
    Sort text boxes in order from top to bottom, left to right
    args:
        dt_boxes(array):detected text boxes with shape [4, 2]
    return:
        sorted boxes(array) with shape [4, 2]
    """
    num_boxes = dt_boxes.shape[0]
    sorted_boxes = sorted(dt_boxes, key=lambda x: (x[0][1], x[0][0]))
    _boxes = list(sorted_boxes)

    for i in range(num_boxes - 1):
        if abs(_boxes[i + 1][0][1] - _boxes[i][0][1]) < 10 and \
                (_boxes[i + 1][0][0] < _boxes[i][0][0]):
            tmp = _boxes[i]
            _boxes[i] = _boxes[i + 1]
            _boxes[i + 1] = tmp
    return _boxes


def draw_ocr_box_txt(image,
                     boxes,
                     txts,
                     scores=None,
                     drop_score=0.5,
                     font_path='',
                     bbox_padding=0.):
    h, w = image.height, image.width
    img_left = image.copy()
    img_right = Image.new('RGB', (w, h), (255, 255, 255))

    import random

    random.seed(0)
    draw_left = ImageDraw.Draw(img_left)
    draw_right = ImageDraw.Draw(img_right)
    for idx, (box, txt) in enumerate(zip(boxes, txts)):
        if scores is not None and scores[idx] < drop_score:
            continue
        color = (random.randint(0, 255), random.randint(0, 255),
                 random.randint(0, 255))
        draw_left.polygon(box, fill=color)
        draw_right.polygon(
            [
                box[0][0], box[0][1], box[1][0], box[1][1], box[2][0],
                box[2][1], box[3][0], box[3][1]
            ],
            outline=color)
        box_height = math.sqrt((box[0][0] - box[3][0]) ** 2 + (box[0][1] - box[3][1]) ** 2)
        box_width = math.sqrt((box[0][0] - box[1][0]) ** 2 + (box[0][1] - box[1][1]) ** 2)
        if box_height > 2 * box_width:
            font_size = max(int(box_width * 0.9 * (1 - bbox_padding)), 10)
            font = ImageFont.truetype(font_path, font_size, encoding="utf-8")
            cur_y = box[0][1]
            for c in txt:
                char_size = font.getsize(c)
                draw_right.text(
                    (box[0][0] + 3, cur_y), c, fill=(0, 0, 0), font=font)
                cur_y += char_size[1]
        else:
            font_size = max(int(box_height * 0.8 * (1 - bbox_padding)), 10)
            font = ImageFont.truetype(font_path, font_size, encoding="utf-8")
            draw_right.text(
                [box[0][0], box[0][1]], txt, fill=(0, 0, 0), font=font)
    img_left = Image.blend(image, img_left, 0.5)
    img_show = Image.new('RGB', (w * 2, h), (255, 255, 255))
    img_show.paste(img_left, (0, 0, w, h))
    img_show.paste(img_right, (w, 0, w * 2, h))
    return np.array(img_show)


def output_result_json(file_name, image, boxes, txts, scores=None):
    h, w = image.height, image.width
    results = []
    for idx, (box, txt) in enumerate(zip(boxes, txts)):
        score = scores[idx] if scores is not None else None
        box_corner_xy = [
            float(box[0][0]), float(box[0][1]), float(box[1][0]), float(box[1][1]),
            float(box[2][0]), float(box[2][1]), float(box[3][0]), float(box[3][1])
        ]
        if score is not None:
            results.append({'box': box_corner_xy, 'txt': txt, 'score': score})
        else:
            results.append({'box': box_corner_xy, 'txt': txt})
    with open(file_name, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)


def adjust_half_and_full(txts):
    # loop of txts
    for i_txt in range(len(txts)):
        txt_tmp = txts[i_txt]
        flg_replace = False
        for i_char in range(1, len(txt_tmp)):
            char_tmp = txt_tmp[i_char]
            if (char_tmp == '-'):
                res = unicodedata.east_asian_width(txt_tmp[i_char - 1])
                if (res == 'W'):
                    txt_tmp = txt_tmp.replace(
                        '%s-' % txt_tmp[i_char - 1], '%sー' % txt_tmp[i_char - 1])
                    flg_replace = True
        if flg_replace:
            txts[i_txt] = txt_tmp
    return txts


def recognize_from_image(config, text_sys):
    for img_path in args.input:
        # read image
        img = imread(img_path)

        # exec ocr
        dt_boxes, rec_res = text_sys(img)

        image = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        boxes = dt_boxes
        txts = [rec_res[i][0] for i in range(len(rec_res))]
        scores = [rec_res[i][1] for i in range(len(rec_res))]

        # adjust halfwidth and fullwidth forms
        txts = adjust_half_and_full(txts)

        draw_img = draw_ocr_box_txt(
            image, boxes, txts, scores,
            drop_score=config['drop_score'],
            font_path=config['vis_font_path'],
            bbox_padding=config['rec_bbox_padding'])
        savepath = get_savepath(args.savepath, img_path)
        cv2.imwrite(savepath, draw_img[:, :, ::-1])

        # write results
        if args.write_results:
            json_file = '%s.json' % savepath.rsplit('.', 1)[0]
            output_result_json(json_file, image, boxes, txts, scores=None)

    logger.info('finished process and write result to %s!' % args.savepath)


def recognize_from_video(config, text_sys):
    # make capture
    video_capture = webcamera_utils.get_capture(args.video)

    # create video writer if savepath is specified as video format
    if args.savepath != SAVE_IMAGE_OR_VIDEO_PATH:
        f_h = int(video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        f_w = int(video_capture.get(cv2.CAP_PROP_FRAME_WIDTH)) * 2
        video_writer = webcamera_utils.get_writer(args.savepath, f_h, f_w)
    else:
        video_writer = None

    # frame read and exec segmentation
    frame_shown = False
    while (True):
        # frame read
        ret, img = video_capture.read()

        if (cv2.waitKey(1) & 0xFF == ord('q')) or not ret:
            break
        if frame_shown and cv2.getWindowProperty('exec PaddleOCR', cv2.WND_PROP_VISIBLE) == 0:
            break

        # exec ocr
        dt_boxes, rec_res = text_sys(img)

        image = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        boxes = dt_boxes
        txts = [rec_res[i][0] for i in range(len(rec_res))]
        scores = [rec_res[i][1] for i in range(len(rec_res))]

        draw_img = draw_ocr_box_txt(
            image, boxes, txts, scores,
            drop_score=config['drop_score'],
            font_path=config['vis_font_path'])
        # display
        cv2.imshow('exec PaddleOCR', draw_img[:, :, ::-1])
        frame_shown = True

        # write a frame image to video
        if video_writer is not None:
            video_writer.write(draw_img[:, :, ::-1])

    video_capture.release()
    cv2.destroyAllWindows()
    if video_writer is not None:
        video_writer.release()
        logger.info('finished process and write result to %s!' % args.savepath)


def main():
    # This model requires fuge gpu memory so fallback to cpu mode
    env_id = args.env_id
    if env_id != -1 and ailia.get_environment(env_id).props == "LOWPOWER":
        env_id = -1

    # get default config value and merge args
    config = get_default_config()
    config['det_limit_side_len'] = args.det_limit_side_len
    config['det_limit_type'] = args.det_limit_type

    det_info = {
        'db_res18': (WEIGHT_PATH_DET_CHN, 'DB'),
        'r50_icdar15': (WEIGHT_PATH_DET_R50_ICDAR15, 'DB++'),
        'r50_trtd': (WEIGHT_PATH_DET_R50_TDTR, 'DB++'),
    }
    weight_path_det, det_algorithm = det_info[det_model]
    weight_path_cls = WEIGHT_PATH_CLS_CHN
    weight_path_rec = dict_path_rec = None

    lang_tmp = args.language.lower()
    if lang_tmp in ('japanese', 'jpn', 'jp'):
        if args.case == 'mobile':
            weight_path_rec = WEIGHT_PATH_REC_JPN_MBL
            dict_path_rec = DICT_PATH_REC_JPN_MBL
        elif args.case == 'server':
            weight_path_rec = WEIGHT_PATH_REC_JPN_SVR
            dict_path_rec = DICT_PATH_REC_JPN_SVR
    elif lang_tmp in ('english', 'eng', 'en'):
        if args.case == 'mobile':
            weight_path_rec = WEIGHT_PATH_REC_ENG_MBL
            dict_path_rec = DICT_PATH_REC_ENG_MBL
    elif lang_tmp in ('chinese', 'chi', 'ch'):
        if args.case == 'mobile':
            weight_path_rec = WEIGHT_PATH_REC_CHN_MBL
            dict_path_rec = DICT_PATH_REC_CHN_MBL
        elif args.case == 'server':
            weight_path_rec = WEIGHT_PATH_REC_CHN_SVR
            dict_path_rec = DICT_PATH_REC_CHN_SVR
    elif lang_tmp in ('german', 'ger', 'ge'):
        weight_path_rec = WEIGHT_PATH_REC_GER_MBL
        dict_path_rec = DICT_PATH_REC_GER_MBL
    elif lang_tmp in ('french', 'fre', 'fr'):
        weight_path_rec = WEIGHT_PATH_REC_FRE_MBL
        dict_path_rec = DICT_PATH_REC_FRE_MBL
    elif lang_tmp in ('korean', 'kor', 'ko'):
        weight_path_rec = WEIGHT_PATH_REC_KOR_MBL
        dict_path_rec = DICT_PATH_REC_KOR_MBL
    else:
        logger.error('Unknown language: %s' % args.language)
        sys.exit(-1)

    config = set_config(config,
                        weight_path_det, det_algorithm,
                        weight_path_rec, dict_path_rec,
                        weight_path_cls)

    # model files check and download
    model_path_det = weight_path_det + '.prototxt'
    check_and_download_models(weight_path_det, model_path_det, REMOTE_PATH)
    model_path_cls = weight_path_cls + '.prototxt'
    check_and_download_models(weight_path_cls, model_path_cls, REMOTE_PATH)
    model_path_rec = weight_path_rec + '.prototxt'
    check_and_download_models(weight_path_rec, model_path_rec, REMOTE_PATH)

    # build ocr class
    text_sys = TextSystem(config, env_id)

    if args.video is not None:
        recognize_from_video(config, text_sys)
    else:
        if args.benchmark:
            logger.info('BENCHMARK mode')
            for i in range(args.benchmark_count):
                start = int(round(time.time() * 1000))
                recognize_from_image(config, text_sys)
                end = int(round(time.time() * 1000))
                logger.info(f'\tailia processing time {end - start} ms')
        else:
            recognize_from_image(config, text_sys)


if __name__ == '__main__':
    main()
