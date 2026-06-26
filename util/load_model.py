import os
import sys
from importlib import import_module

import numpy as np

import arg_utils


top_path = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__),
    )
)


def load_model(mod_name):
    mod = None

    def dummy_parser(parser, *args):
        args = parser.parse_args()
        return args

    argv = sys.argv
    update_parser = arg_utils.update_parser
    try:
        sys.argv = sys.argv[:1]
        arg_utils.update_parser = dummy_parser

        mod = import_module(mod_name)
    finally:
        sys.argv = argv
        arg_utils.update_parser = update_parser

    return mod


def load_groudingdino(args=None):
    sys.path.insert(0, os.path.join(top_path, "object_detection/groundingdino"))
    mod = load_model("groundingdino")
    sys.path.pop(0)

    if args:
        for name in vars(args):
            setattr(mod.args, name, getattr(args, name))
    return mod


def load_face_detection_adas(args=None):
    sys.path.insert(0, os.path.join(top_path, "face_detection/face-detection-adas"))
    mod = load_model("face-detection-adas")
    sys.path.pop(0)

    if args:
        for name in vars(args):
            setattr(mod.args, name, getattr(args, name))
    return mod


def load_facemesh_v2(args=None):
    sys.path.insert(0, os.path.join(top_path, "face_recognition/facemesh_v2"))
    mod = load_model("facemesh_v2")
    sys.path.pop(0)

    if args:
        for name in vars(args):
            setattr(mod.args, name, getattr(args, name))
    return mod


def load_bytetrack(args=None):
    sys.path.insert(0, os.path.join(top_path, "object_tracking/bytetrack"))
    mod = load_model("bytetrack")
    sys.path.pop(0)

    if args:
        for name in vars(args):
            setattr(mod.args, name, getattr(args, name))
    return mod


def load_clip(args=None):
    sys.path.insert(0, os.path.join(top_path, "image_classification/clip"))
    mod = load_model("clip")
    sys.path.pop(0)

    if args:
        for name in vars(args):
            setattr(mod.args, name, getattr(args, name))

    def predict_image_feature(net, img):
        img = mod.preprocess(img)
        if not mod.args.onnx:
            output = net.predict([img])
        else:
            output = net.run(None, {"image": img})
        image_feature = output[0]
        image_feature = image_feature / np.linalg.norm(
            image_feature, ord=2, axis=-1, keepdims=True
        )
        return image_feature

    mod.predict_image_feature = predict_image_feature
    return mod
