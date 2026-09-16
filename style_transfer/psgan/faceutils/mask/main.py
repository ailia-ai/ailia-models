import ailia
from PIL import Image
import numpy as np

# Logger
from logging import getLogger  # noqa: E402

logger = getLogger(__name__)


class FaceParser:
    def __init__(self, device="cpu", args=None, face_parser_path=None):
        mapper = [0, 1, 2, 3, 4, 5, 0, 11, 12, 0, 6, 8, 7, 9, 13, 0, 0, 10, 0]
        self.device = device
        self.dic = mapper
        self.net = _initialize_net(args, face_parser_path)
        self.args = args

    def parse(self, image: Image):
        assert image.shape[:2] == (512, 512)
        # Reshaping and normalisation
        image = image.transpose((2, 0, 1)) / 255
        # Standardisation
        means = np.expand_dims([0.485, 0.456, 0.406], (1, 2))
        stds = np.expand_dims([0.229, 0.224, 0.225], (1, 2))
        image = (image - means) / stds
        image = np.expand_dims(image, 0).astype("float32")
        # Parsing
        out = _parse_face(image, self.net, self.args)[0]
        parsing = out.squeeze(0).argmax(0)
        parsing = np.take(self.dic, parsing)

        return parsing.astype("float32")


def _initialize_net(args, face_parser_path):
    env_id = args.env_id
    logger.info(f"env_id (face parser): {env_id}")
    if not args.onnx:
        net = ailia.Net(face_parser_path[0], face_parser_path[1], env_id=env_id, memory_mode=args.memory_mode)
    else:
        import onnxruntime

        net = onnxruntime.InferenceSession(face_parser_path[1])

    return net


def _parse_face(input_image, net, args):
    if not args.onnx:
        return net.predict([input_image])
    else:
        inputs = {net.get_inputs()[0].name: input_image}
        return net.run(None, inputs)
