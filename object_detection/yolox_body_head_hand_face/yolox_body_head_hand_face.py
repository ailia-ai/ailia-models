#!/usr/bin/env python

import sys
import copy
import cv2
import time
import numpy as np
from enum import Enum
from dataclasses import dataclass
from typing import Tuple, Optional, List
from abc import ABC, abstractmethod
import ailia

# import original modules
sys.path.append('../../util')
# logger
from logging import getLogger

import webcamera_utils
from image_utils import imread  # noqa: E402
from model_utils import check_and_download_models
from arg_utils import get_base_parser, get_savepath, update_parser

logger = getLogger(__name__)

# ======================
# Parameters
# ======================
IMAGE_PATH = 'input.jpg'
SAVE_IMAGE_PATH = 'output.jpg'
CLASS_SCORE_THRETHOLD = 0.35
REMOTE_PATH = 'https://storage.googleapis.com/ailia-models/yolox_body_head_hand_face/'
MODEL_YOLOX_L_NAME = 'yolox_l_body_head_hand_face_0086_0.5143_post_1x3x480x640'
# WEIGHT_YOLOX_M_PATH = 'yolox_m_body_head_hand_face_0111_0.5016_post_1x3x480x640.onnx'
# WEIGHT_YOLOX_N_PATH = 'yolox_n_body_head_hand_face_0299_0.3803_post_1x3x480x640.onnx'
# WEIGHT_YOLOX_S_PATH = 'yolox_s_body_head_hand_face_0299_0.4668_post_1x3x480x640.onnx'
# WEIGHT_YOLOX_T_PATH = 'yolox_t_body_head_hand_face_0299_0.4265_post_1x3x480x640.onnx'
# WEIGHT_YOLOX_X_PATH = 'yolox_x_body_head_hand_face_0076_0.5228_post_1x3x480x640.onnx'

# ======================
# Arguemnt Parser Config
# ======================
parser = get_base_parser('yolox body head hand face model', IMAGE_PATH, SAVE_IMAGE_PATH)
parser.add_argument(
    '-w', '--write_json',
    action='store_true',
    help='save result to json'
)
parser.add_argument(
    '-m', '--model_name', default='l',
    choices=('l'),#, 'm', 'n', 's', 't', 'x'),
    help='model type'
)
parser.add_argument(
    '--onnx',
    action='store_true',
    help='execute onnxruntime version.'
)
args = update_parser(parser)
# ======================
# Main functions
# ======================
class Color(Enum):
    BLACK          = '\033[30m'
    RED            = '\033[31m'
    GREEN          = '\033[32m'
    YELLOW         = '\033[33m'
    BLUE           = '\033[34m'
    MAGENTA        = '\033[35m'
    CYAN           = '\033[36m'
    WHITE          = '\033[37m'
    COLOR_DEFAULT  = '\033[39m'
    BOLD           = '\033[1m'
    UNDERLINE      = '\033[4m'
    INVISIBLE      = '\033[08m'
    REVERSE        = '\033[07m'
    BG_BLACK       = '\033[40m'
    BG_RED         = '\033[41m'
    BG_GREEN       = '\033[42m'
    BG_YELLOW      = '\033[43m'
    BG_BLUE        = '\033[44m'
    BG_MAGENTA     = '\033[45m'
    BG_CYAN        = '\033[46m'
    BG_WHITE       = '\033[47m'
    BG_DEFAULT     = '\033[49m'
    RESET          = '\033[0m'

    def __str__(self):
        return self.value

    def __call__(self, s):
        return str(self) + str(s) + str(Color.RESET)

@dataclass(frozen=False)
class Box():
    classid: int
    score: float
    x1: int
    y1: int
    x2: int
    y2: int

class AbstractModel(ABC):
    """AbstractModel
    Base class of the model.
    """
    _runtime: str = 'onnx'
    _model_path: str = ''
    _class_score_th: float = 0.35
    _input_shapes: List[List[int]] = []
    _input_names: List[str] = []
    _output_shapes: List[List[int]] = []
    _output_names: List[str] = []

    # onnx/tflite
    _interpreter = None
    _inference_model = None
    _providers = None
    _swap = (2, 0, 1)
    _h_index = 2
    _w_index = 3

    # onnx
    _onnx_dtypes_to_np_dtypes = {
        "tensor(float)": np.float32,
        "tensor(uint8)": np.uint8,
        "tensor(int8)": np.int8,
    }

    # tflite
    _input_details = None
    _output_details = None

    @abstractmethod
    def __init__(
        self,
        *,
        runtime: Optional[str] = 'onnx',
        model_path: Optional[str] = '',
        weight_path: Optional[str] = '',
        class_score_th: Optional[float] = 0.35,
        env_id: Optional[int] = 0
    ):
        self._env_id = env_id

        self._runtime = runtime
        self._model_path = model_path
        self._weight_path = weight_path
        self._class_score_th = class_score_th
        
        # Model loading
        if self._runtime == 'onnx':
            import onnxruntime # type: ignore

            match self._env_id:
                case 0:
                    providers = ['CPUExecutionProvider']
                case 2:
                    providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
                case _:
                    providers = None

            session_option = onnxruntime.SessionOptions()
            session_option.log_severity_level = 3
            self._interpreter = \
                onnxruntime.InferenceSession(
                    weight_path,
                    sess_options=session_option,
                    providers=providers,
                )
            self._providers = self._interpreter.get_providers()
            self._input_shapes = [
                input.shape for input in self._interpreter.get_inputs()
            ]
            self._input_names = [
                input.name for input in self._interpreter.get_inputs()
            ]
            self._input_dtypes = [
                self._onnx_dtypes_to_np_dtypes[input.type] for input in self._interpreter.get_inputs()
            ]
            self._output_shapes = [
                output.shape for output in self._interpreter.get_outputs()
            ]
            self._output_names = [
                output.name for output in self._interpreter.get_outputs()
            ]
            self._model = self._interpreter.run
            self._swap = (2, 0, 1)
            self._h_index = 2
            self._w_index = 3
        elif self._runtime == 'ailia':
            self._interpreter = ailia.Net(model_path, weight_path, env_id=env_id, memory_mode=args.memory_mode)
            self._input_shapes = [list(self._interpreter.get_input_shape())]
            self._input_names = [self._interpreter.get_blob_name(0)]
            self._model = self._interpreter.predict
        

    @abstractmethod
    def __call__(
        self,
        *,
        input_datas: List[np.ndarray],
    ) -> List[np.ndarray]:
        datas = {
            f'{input_name}': input_data \
                for input_name, input_data in zip(self._input_names, input_datas)
        }
        if self._runtime == 'onnx':
            outputs = [
                output for output in \
                    self._model(
                        output_names=self._output_names,
                        input_feed=datas,
                    )
            ]
        elif self._runtime == 'ailia':
            reshaped_array = [input_datas[0].tolist()]
            outputs = [
                self._model(np.array(input_datas))
            ]
        return outputs


    @abstractmethod
    def _preprocess(
        self,
        *,
        image: np.ndarray,
        swap: Optional[Tuple[int,int,int]] = (2, 0, 1),
    ) -> np.ndarray:
        pass

    @abstractmethod
    def _postprocess(
        self,
        *,
        image: np.ndarray,
        boxes: np.ndarray,
    ) -> List[Box]:
        pass

class YOLOX(AbstractModel):
    def __init__(
        self,
        *,
        runtime: Optional[str] = 'onnx',
        weight_path: Optional[str] = 'yolox_l_body_head_hand_face_0086_0.5143_post_1x3x480x640.onnx',
        model_path: Optional[str] = 'yolox_l_body_head_hand_face_0086_0.5143_post_1x3x480x640.onnx.prototxt',
        class_score_th: Optional[float] = 0.35,
        # providers: Optional[List] = None,
        env_id: Optional[int] = 0,
    ):
        """YOLOX

        Parameters
        ----------
        runtime: Optional[str]
            Runtime for YOLOX. Default: onnx

        model_path: Optional[str]
            ONNX/TFLite file path for YOLOX

        class_score_th: Optional[float]
            Score threshold. Default: 0.35

        providers: Optional[List]
            Providers for ONNXRuntime.
        """
        super().__init__(
            runtime=runtime,
            weight_path=weight_path,
            model_path=model_path,
            class_score_th=class_score_th,
            # providers=providers,
            env_id=env_id,
        )

    def __call__(
        self,
        image: np.ndarray,
    ) -> List[Box]:
        """YOLOX

        Parameters
        ----------
        image: np.ndarray
            Entire image

        Returns
        -------
        boxes: np.ndarray
            Predicted boxes: [N, x1, y1, x2, y2]

        scores: np.ndarray
            Predicted box scores: [N, score]
        """
        temp_image = copy.deepcopy(image)

        # PreProcess
        resized_image = \
            self._preprocess(
                temp_image,
            )

        # Inference
        if self._runtime == 'onnx':
            inferece_image = np.asarray([resized_image], dtype=self._input_dtypes[0])
        else:
            inferece_image = resized_image
        outputs = super().__call__(input_datas=[inferece_image])
        boxes = outputs[0]

        # PostProcess
        result_boxes = \
            self._postprocess(
                image=temp_image,
                boxes=boxes,
            )

        return result_boxes

    def _preprocess(
        self,
        image: np.ndarray,
    ) -> np.ndarray:
        """_preprocess

        Parameters
        ----------
        image: np.ndarray
            Entire image

        swap: tuple
            HWC to CHW: (2,0,1)
            CHW to HWC: (1,2,0)
            HWC to HWC: (0,1,2)
            CHW to CHW: (0,1,2)

        Returns
        -------
        resized_image: np.ndarray
            Resized and normalized image.
        """
        # Resize + Transpose
        resized_image = cv2.resize(
            image,
            (
                int(self._input_shapes[0][self._w_index]),
                int(self._input_shapes[0][self._h_index]),
            )
        )
        resized_image = resized_image.transpose(self._swap)
        resized_image = \
            np.ascontiguousarray(
                resized_image,
                dtype=np.float32,
            )

        return resized_image

    def _postprocess(
        self,
        image: np.ndarray,
        boxes: np.ndarray,
    ) -> List[Box]:
        """_postprocess

        Parameters
        ----------
        image: np.ndarray
            Entire image.

        boxes: np.ndarray
            float32[N, 7]

        Returns
        -------
        result_boxes: List[Box]
            Predicted boxes: [classid, score, x1, y1, x2, y2]
        """

        """
        Detector is
            N -> Number of boxes detected
            batchno -> always 0: BatchNo.0

        batchno_classid_score_x1y1x2y2: float32[N,7]
        """
        image_height = image.shape[0]
        image_width = image.shape[1]

        result_boxes: List[Box] = []

        if len(boxes) > 0:
            scores = boxes[:, 2:3]
            keep_idxs = scores[:, 0] > self._class_score_th
            scores_keep = scores[keep_idxs, :]
            boxes_keep = boxes[keep_idxs, :]

            if len(boxes_keep) > 0:
                for box, score in zip(boxes_keep, scores_keep):
                    x_min = int(max(0, box[3]) * image_width / self._input_shapes[0][self._w_index])
                    y_min = int(max(0, box[4]) * image_height / self._input_shapes[0][self._h_index])
                    x_max = int(min(box[5], self._input_shapes[0][self._w_index]) * image_width / self._input_shapes[0][self._w_index])
                    y_max = int(min(box[6], self._input_shapes[0][self._h_index]) * image_height / self._input_shapes[0][self._h_index])
                    result_boxes.append(
                        Box(
                            classid=int(box[1]),
                            score=float(score[0]),
                            x1=x_min,
                            y1=y_min,
                            x2=x_max,
                            y2=y_max,
                        )
                    )

        return result_boxes


def is_parsable_to_int(s):
    try:
        int(s)
        return True
    except ValueError:
        return False

def is_package_installed(package_name: str):
    """Checks if the specified package is installed.

    Parameters
    ----------
    package_name: str
        Name of the package to be checked.

    Returns
    -------
    result: bool
        True if the package is installed, false otherwise.
    """
    return importlib.util.find_spec(package_name) is not None

def draw_dashed_line(
    image: np.ndarray,
    pt1: Tuple[int, int],
    pt2: Tuple[int, int],
    color: Tuple[int, int, int],
    thickness: int = 1,
    dash_length: int = 10,
):
    """Function to draw a dashed line"""
    dist = ((pt1[0] - pt2[0]) ** 2 + (pt1[1] - pt2[1]) ** 2) ** 0.5
    dashes = int(dist / dash_length)
    for i in range(dashes):
        start = [int(pt1[0] + (pt2[0] - pt1[0]) * i / dashes), int(pt1[1] + (pt2[1] - pt1[1]) * i / dashes)]
        end = [int(pt1[0] + (pt2[0] - pt1[0]) * (i + 0.5) / dashes), int(pt1[1] + (pt2[1] - pt1[1]) * (i + 0.5) / dashes)]
        cv2.line(image, tuple(start), tuple(end), color, thickness)

def draw_dashed_rectangle(
    image: np.ndarray,
    top_left: Tuple[int, int],
    bottom_right: Tuple[int, int],
    color: Tuple[int, int, int],
    thickness: int = 1,
    dash_length: int = 10
):
    """Function to draw a dashed rectangle"""
    tl_tr = (bottom_right[0], top_left[1])
    bl_br = (top_left[0], bottom_right[1])
    draw_dashed_line(image, top_left, tl_tr, color, thickness, dash_length)
    draw_dashed_line(image, tl_tr, bottom_right, color, thickness, dash_length)
    draw_dashed_line(image, bottom_right, bl_br, color, thickness, dash_length)
    draw_dashed_line(image, bl_br, top_left, color, thickness, dash_length)


def write_image_texts(debug_image, boxes, elapsed_time):
    debug_image_w = debug_image.shape[1]
    cv2.putText(
        debug_image,
        f'{elapsed_time*1000:.2f} ms',
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        debug_image,
        f'{elapsed_time*1000:.2f} ms',
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 0, 255),
        1,
        cv2.LINE_AA,
    )

    for box in boxes:
        classid: int = box.classid
        color = (255,255,255)
        if classid == 0:
            color = (255,0,0)
        elif classid == 1:
            color = (0,0,255)
        elif classid == 2:
            color = (0,255,0)
        elif classid == 3:
            color = (0,200,255)

        if classid != 3:
            cv2.rectangle(
                debug_image,
                (box.x1, box.y1),
                (box.x2, box.y2),
                (255,255,255),
                2,
            )
            cv2.rectangle(
                debug_image,
                (box.x1, box.y1),
                (box.x2, box.y2),
                color,
                1,
            )
        else:
            draw_dashed_rectangle(
                image=debug_image,
                top_left=(box.x1, box.y1),
                bottom_right=(box.x2, box.y2),
                color=color,
                thickness=2,
                dash_length=10
            )
        cv2.putText(
            debug_image,
            f'{box.score:.2f}',
            (
                box.x1 if box.x1+50 < debug_image_w else debug_image_w-50,
                box.y1-10 if box.y1-25 > 0 else 20
            ),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
        cv2.putText(
            debug_image,
            f'{box.score:.2f}',
            (
                box.x1 if box.x1+50 < debug_image_w else debug_image_w-50,
                box.y1-10 if box.y1-25 > 0 else 20
            ),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            color,
            1,
            cv2.LINE_AA,
        )
    return debug_image


def save_json(json_path, output):
    import json
    result = []
    for box in output:
        result.append({
            "classid": box.classid,
            "score": box.score,
            "x1": box.x1,
            "y1": box.y1,
            "x2": box.x2,
            "y2": box.y2
        })
    with open(json_path, 'w') as f:
        json.dump(result, f, indent=2)


def recognize_from_image(model):
    for image_path in args.input:
        logger.debug(f'input image: {image_path}')
        raw_img = imread(image_path, cv2.IMREAD_COLOR)
        logger.debug(f'input image shape: {raw_img.shape}')

        # inference
        logger.info('Start inference...')
        if args.benchmark:
            logger.info('BENCHMARK mode')
            total_time = 0
            for i in range(args.benchmark_count):
                start = int(round(time.time() * 1000))
                start_time = time.perf_counter()
                output = model(raw_img)
                elapsed_time = time.perf_counter() - start_time
                end = int(round(time.time() * 1000))
                if i != 0:
                    total_time = total_time + (end - start)
                logger.info(f'\tailia processing time {end - start} ms')
            logger.info(f'\taverage time {total_time / (args.benchmark_count-1)} ms')
        else:
            start_time = time.perf_counter()
            output = model(raw_img)
            elapsed_time = time.perf_counter() - start_time
        
        res_img = write_image_texts(raw_img, output, elapsed_time)

        # plot result
        savepath = get_savepath(args.savepath, image_path)
        logger.info(f'saved at : {savepath}')
        cv2.imwrite(savepath, res_img)

        if args.write_json:
            json_path = savepath.rsplit('.', 1)[0] + '.json'
            save_json(json_path, output)

    logger.info('Script finished successfully.')


def recognize_from_video(model):
    video_capture = webcamera_utils.get_capture(args.video)

    if args.savepath != SAVE_IMAGE_PATH:
        f_h = int(video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        f_w = int(video_capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        video_writer = webcamera_utils.get_writer(args.savepath, f_h, f_w)
    else:
        video_writer = None

    # frame read and exec segmentation
    frame_shown = False

    while (True):
        ret, frame = video_capture.read()
        if (cv2.waitKey(1) & 0xFF == ord('q')) or not ret:
            break
        if frame_shown and cv2.getWindowProperty('frame', cv2.WND_PROP_VISIBLE) == 0:
            break

        raw_img = frame

        debug_image = copy.deepcopy(raw_img)
        start_time = time.perf_counter()
        boxes = model(debug_image)
        elapsed_time = time.perf_counter() - start_time
        
        res_img = write_image_texts(debug_image, boxes, elapsed_time)

        cv2.imshow('frame', res_img)
        frame_shown = True

        # save results
        if video_writer is not None:
            video_writer.write(res_img)

    capture.release()
    cv2.destroyAllWindows()
    if writer is not None:
        writer.release()
    logger.info('Script finished successfully.')


def main():
    logger.info('Checking encode_image model...')
    dic_model = {
        'l': (MODEL_YOLOX_L_NAME),
        # 'm': (WEIGHT_YOLOX_M_PATH),
        # 'n': (WEIGHT_YOLOX_N_PATH),
        # 's': (WEIGHT_YOLOX_S_PATH),
        # 't': (WEIGHT_YOLOX_T_PATH),
        # 'x': (WEIGHT_YOLOX_X_PATH)
    }
    model = dic_model[args.model_name]
    WEIGHT_PATH = model + ".onnx"
    MODEL_PATH = model + ".onnx.prototxt"
    print(WEIGHT_PATH)
    check_and_download_models(WEIGHT_PATH, MODEL_PATH, REMOTE_PATH)

    env_id = args.env_id
    if not args.onnx:
        # ailia
        model = YOLOX(
            runtime='ailia',
            weight_path=WEIGHT_PATH,
            model_path=MODEL_PATH
            ,
            class_score_th=CLASS_SCORE_THRETHOLD,
            env_id=env_id
        )
    else:
        model = YOLOX(
            runtime='onnx',
            weight_path=WEIGHT_PATH,
            model_path=MODEL_PATH,
            class_score_th=CLASS_SCORE_THRETHOLD,
            env_id=env_id
        )

    if args.video is not None:
        # video mode
        recognize_from_video(model)
    else:
        # image mode
        recognize_from_image(model)

if __name__ == "__main__":
    main()
