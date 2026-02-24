import sys
import os
import ailia
from g2pw.api import G2PWConverter

sys.path.append("../../util")
from arg_utils import get_base_parser, update_parser  # noqa: E402

from logging import getLogger  # noqa: E402
logger = getLogger(__name__)

WEIGHT_PATH = "G2PWModel/g2pw.onnx"

parser = get_base_parser(
    "G2PW",
    None,
    None,
)
parser.add_argument(
    "-i",
    "--input",
    type=str,
    default="你好世界",
    help="Input text.",
)
parser.add_argument(
    "--style",
    type=str,
    default="bopomofo",
    choices=["bopomofo", "pinyin"],
    help="Output style. (bopomofo or pinyin)"
)
args = update_parser(parser, check_input_type=False)


class AiliaG2P(G2PWConverter):

    def __init__(self, weight_path, env_id, style='bopomofo', **kwargs):
        model_dir = os.path.dirname(weight_path) or '.'
        self.net = ailia.Net(None, weight_path, env_id=env_id)

        class AiliaSession:
            def __init__(self, net):
                self.net = net

            def run(self, _outputs, inputs):
                return self.net.predict(inputs)

        # 親クラスの初期化時に ailia セッションを渡す
        super().__init__(
            model_dir=model_dir,
            style=style,
            onnx_session=AiliaSession(self.net),
            **kwargs
        )
def main():
    converter = AiliaG2P(
        weight_path=WEIGHT_PATH,
        env_id=args.env_id,
        style=args.style
    )

    results = converter([args.input])

    logger.info("--- Input ----")
    logger.info(args.input)
    logger.info(f"--- Output : {args.style} ---")
    logger.info(str(results))


if __name__ == '__main__':
    main()