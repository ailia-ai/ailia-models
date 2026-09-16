import sys
import time

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import json

import ailia

# import original modules
sys.path.append('../../util')
from arg_utils import get_base_parser, update_parser, get_savepath  # noqa: E402
from model_utils import check_and_download_models  # noqa: E402

# logger
from logging import getLogger   # noqa: E402
logger = getLogger(__name__)

# ======================
# Parameters
# ======================

WEIGHT_PATH_AIRPLANE = 'airplane_100.onnx'
MODEL_PATH_AIRPLANE = 'airplane_100.onnx.prototxt'
WEIGHT_PATH_BAG = 'bag_100.onnx'
MODEL_PATH_BAG = 'bag_100.onnx.prototxt'
WEIGHT_PATH_CAP = 'cap_100.onnx'
MODEL_PATH_CAP = 'cap_100.onnx.prototxt'
WEIGHT_PATH_CAR = 'car_100.onnx'
MODEL_PATH_CAR = 'car_100.onnx.prototxt'
WEIGHT_PATH_CHAIR = 'chair_100.onnx'
MODEL_PATH_CHAIR = 'chair_100.onnx.prototxt'
WEIGHT_PATH_EARPHONE = 'earphone_100.onnx'
MODEL_PATH_EARPHONE = 'earphone_100.onnx.prototxt'
WEIGHT_PATH_GUITAR = 'guitar_100.onnx'
MODEL_PATH_GUITAR = 'guitar_100.onnx.prototxt'
WEIGHT_PATH_KNIFE = 'knife_100.onnx'
MODEL_PATH_KNIFE = 'knife_100.onnx.prototxt'
WEIGHT_PATH_LAMP = 'lamp_100.onnx'
MODEL_PATH_LAMP = 'lamp_100.onnx.prototxt'
WEIGHT_PATH_LAPTOP = 'laptop_100.onnx'
MODEL_PATH_LAPTOP = 'laptop_100.onnx.prototxt'
WEIGHT_PATH_MOTORBIKE = 'motorbike_100.onnx'
MODEL_PATH_MOTORBIKE = 'motorbike_100.onnx.prototxt'
WEIGHT_PATH_MUG = 'mug_100.onnx'
MODEL_PATH_MUG = 'mug_100.onnx.prototxt'
WEIGHT_PATH_PISTOL = 'pistol_100.onnx'
MODEL_PATH_PISTOL = 'pistol_100.onnx.prototxt'
WEIGHT_PATH_ROCKET = 'rocket_100.onnx'
MODEL_PATH_ROCKET = 'rocket_100.onnx.prototxt'
WEIGHT_PATH_SKATEBOARD = 'skateboard_100.onnx'
MODEL_PATH_SKATEBOARD = 'skateboard_100.onnx.prototxt'
WEIGHT_PATH_TABLE = 'table_100.onnx'
MODEL_PATH_TABLE = 'table_100.onnx.prototxt'
WEIGHT_PATH_CLASSIFIER = 'cls_model_100.onnx'
MODEL_PATH_CLASSIFIER = 'cls_model_100.onnx.prototxt'
REMOTE_PATH = \
    'https://storage.googleapis.com/ailia-models/pointnet_pytorch/'

POINT_PATH_AIRPLANE = 'shapenet_dataset/Airplane/a1708ad923f3b51abbf3143b1cb6076a.pts'
POINT_PATH_BAG = 'shapenet_dataset/Bag/4e4fcfffec161ecaed13f430b2941481.pts'
POINT_PATH_CAP = 'shapenet_dataset/Cap/c7122c44495a5ac6aceb0fa31f18f016.pts'
POINT_PATH_CAR = 'shapenet_dataset/Car/1f1b5c7c01557c484354740e038a7994.pts'
POINT_PATH_CHAIR = 'shapenet_dataset/Chair/355fa0f35b61fdd7aa74a6b5ee13e775.pts'
POINT_PATH_EARPHONE = 'shapenet_dataset/Earphone/e33d6e8e39a75268957b6a4f3924d982.pts'
POINT_PATH_GUITAR = 'shapenet_dataset/Guitar/d546e034a6c659a425cd348738a8052a.pts'
POINT_PATH_KNIFE = 'shapenet_dataset/Knife/9d424831d05d363d870906b5178d97bd.pts'
POINT_PATH_LAMP = 'shapenet_dataset/Lamp/b8c87ad9d4930983a8d82fc8a3e54728.pts'
POINT_PATH_LAPTOP = 'shapenet_dataset/Laptop/4d3dde22f529195bc887d5d9a11f3155.pts'
POINT_PATH_MOTORBIKE = 'shapenet_dataset/Motorbike/9d3b07f4475d501e8249f134aca4c817.pts'
POINT_PATH_MUG = 'shapenet_dataset/Mug/10f6e09036350e92b3f21f1137c3c347.pts'
POINT_PATH_PISTOL = 'shapenet_dataset/Pistol/b1bbe535a833635d91f9af3df5b0c8fc.pts'
POINT_PATH_ROCKET = 'shapenet_dataset/Rocket/15474cf9caa757a528eba1f0b7744e9.pts'
POINT_PATH_SKATEBOARD = 'shapenet_dataset/Skateboard/f5d7698b5a57d61226e0640b67de606.pts'
POINT_PATH_TABLE = 'shapenet_dataset/Table/408c3db9b4ee6be2e9f3e9c758fef992.pts'
SAVE_IMAGE_PATH = 'output.png'

# ======================
# Arguemnt Parser Config
# ======================

parser = get_base_parser('PointNet.pytorch model', None, SAVE_IMAGE_PATH)
parser.add_argument(
    '-c', '--choice_class', type=str, default='chair',
    choices=(
        'airplane', 'bag', 'cap', 'car', 'chair', 'earphone', 'guitar',
        'knife', 'lamp', 'laptop', 'motorbike', 'mug', 'pistol',
        'rocket', 'skateboard', 'table'
    ),
    help='choice class'
)
parser.add_argument(
    '--skip_plot',
    action='store_true',
    help='Skip showing 3D plot of result (for CUI)'
)
parser.add_argument(
    '-w', '--write_json',
    action='store_true',
    help='Flag to output results to json file'
)
parser.add_argument(
    '--seed',
    type=int,
    help='Speciry seed value to randomize order of input points'
)
args = update_parser(parser)


# ======================
# Main functions
# ======================

def load_data(point_file, npoints=2500):
    point_set = np.loadtxt(point_file).astype(np.float32)
    if args.seed is not None:
        np.random.seed(args.seed)
    choice = np.random.choice(len(point_set), npoints, replace=True)
    point_set = point_set[choice, :]

    point_set = point_set - np.expand_dims(np.mean(point_set, axis=0), 0)  # center
    dist = np.max(np.sqrt(np.sum(point_set ** 2, axis=1)), 0)
    point_set = point_set / dist  # scale

    return point_set


def predict_seg(points, net):
    points = points.transpose(1, 0)
    points = np.expand_dims(points, axis=0)

    # feedforward
    net.set_input_shape(points.shape)
    output = net.predict({'point': points})

    pred, _ = output
    pred_choice = np.argmax(pred[0], axis=1)

    return pred_choice


def predict_cls(points, net):
    points = points.transpose(1, 0)
    points = np.expand_dims(points, axis=0)

    # feedforward
    net.set_input_shape(points.shape)
    output = net.predict({'points': points})

    pred, _ = output
    pred_choice = np.argmax(pred[0], axis=0)
    pred_choice = [
        'airplane', 'bag', 'cap', 'car', 'chair', 'earphone', 'guitar', 'knife',
        'lamp', 'laptop', 'motorbike', 'mug', 'pistol', 'rocket', 'skateboard', 'table',
    ][pred_choice]

    return pred_choice


def recognize_from_points(filename, net_seg, net_cls):
    # prepare input data
    point = load_data(filename)

    # inference
    logger.info('Start inference...')
    if args.benchmark:
        logger.info('BENCHMARK mode')
        for i in range(5):
            start = int(round(time.time() * 1000))
            pred_seg = predict_seg(point, net_seg)
            pred_cls = predict_cls(point, net_cls)
            end = int(round(time.time() * 1000))
            logger.info(f'\tailia processing time {end - start} ms')
    else:
        pred_seg = predict_seg(point, net_seg)
        pred_cls = predict_cls(point, net_cls)

    if hasattr(matplotlib, "colormaps"):
        cmap = matplotlib.colormaps["hsv"].resampled(10)
    else:
        cmap = plt.cm.get_cmap("hsv", 10)
    cmap = np.array([cmap(i) for i in range(10)])[:, :3]
    pred_color = cmap[pred_seg, :]

    showsz = 800
    point = point - point.mean(axis=0)
    radius = ((point ** 2).sum(axis=-1) ** 0.5).max()
    point /= (radius * 2.2) / showsz

    # plot result
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    X = point[:, 0]
    Y = point[:, 1]
    Z = point[:, 2]
    ax.scatter(X, Y, Z, c=pred_color)

    # adjust plot scale
    max_range = np.array([X.max() - X.min(), Y.max() - Y.min(), Z.max() - Z.min()]).max() * 0.5
    mid_x = (X.max() + X.min()) * 0.5
    mid_y = (Y.max() + Y.min()) * 0.5
    mid_z = (Z.max() + Z.min()) * 0.5
    ax.set_xlim(mid_x - max_range, mid_x + max_range)
    ax.set_ylim(mid_y - max_range, mid_y + max_range)
    ax.set_zlim(mid_z - max_range, mid_z + max_range)

    logger.info('class --'+str(pred_cls))

    savepath = get_savepath(args.savepath, filename)
    logger.info(f'saved at : {savepath}')

    plt.savefig(savepath, dpi=120)
    if not args.skip_plot:
        plt.show()
        
    if args.write_json:
        results = {'class': str(pred_cls), 'points': []}
        for i in range(point.shape[0]):
            results['points'].append({'pos': point[i].tolist(), 'seg': int(pred_seg[i])})
        json_file = '%s.json' % savepath.rsplit('.', 1)[0]
        with open(json_file, 'w') as f:
            json.dump(results, f, indent=2)

    logger.info('Script finished successfully.')


def main():
    rec_model = {
        'airplane': (WEIGHT_PATH_AIRPLANE, MODEL_PATH_AIRPLANE, POINT_PATH_AIRPLANE),
        'bag': (WEIGHT_PATH_BAG, MODEL_PATH_BAG, POINT_PATH_BAG),
        'cap': (WEIGHT_PATH_CAP, MODEL_PATH_CAP, POINT_PATH_CAP),
        'car': (WEIGHT_PATH_CAR, MODEL_PATH_CAR, POINT_PATH_CAR),
        'chair': (WEIGHT_PATH_CHAIR, MODEL_PATH_CHAIR, POINT_PATH_CHAIR),
        'earphone': (WEIGHT_PATH_EARPHONE, MODEL_PATH_EARPHONE, POINT_PATH_EARPHONE),
        'guitar': (WEIGHT_PATH_GUITAR, MODEL_PATH_GUITAR, POINT_PATH_GUITAR),
        'knife': (WEIGHT_PATH_KNIFE, MODEL_PATH_KNIFE, POINT_PATH_KNIFE),
        'lamp': (WEIGHT_PATH_LAMP, MODEL_PATH_LAMP, POINT_PATH_LAMP),
        'laptop': (WEIGHT_PATH_LAPTOP, MODEL_PATH_LAPTOP, POINT_PATH_LAPTOP),
        'motorbike': (WEIGHT_PATH_MOTORBIKE, MODEL_PATH_MOTORBIKE, POINT_PATH_MOTORBIKE),
        'mug': (WEIGHT_PATH_MUG, MODEL_PATH_MUG, POINT_PATH_MUG),
        'pistol': (WEIGHT_PATH_PISTOL, MODEL_PATH_PISTOL, POINT_PATH_PISTOL),
        'rocket': (WEIGHT_PATH_ROCKET, MODEL_PATH_ROCKET, POINT_PATH_ROCKET),
        'skateboard': (WEIGHT_PATH_SKATEBOARD, MODEL_PATH_SKATEBOARD, POINT_PATH_SKATEBOARD),
        'table': (WEIGHT_PATH_TABLE, MODEL_PATH_TABLE, POINT_PATH_TABLE),
    }
    WEIGHT_PATH, MODEL_PATH, POINT_PATH = rec_model[args.choice_class]

    # model files check and download
    logger.info("=== Segmentation model ===")
    check_and_download_models(WEIGHT_PATH, MODEL_PATH, REMOTE_PATH)
    logger.info("=== Classifier model ===")
    check_and_download_models(WEIGHT_PATH_CLASSIFIER, MODEL_PATH_CLASSIFIER, REMOTE_PATH)

    # load model
    env_id = args.env_id
    logger.info(f'env_id: {env_id}')

    # initialize
    net_seg = ailia.Net(MODEL_PATH, WEIGHT_PATH, env_id=env_id, memory_mode=args.memory_mode)
    net_cls = ailia.Net(MODEL_PATH_CLASSIFIER, WEIGHT_PATH_CLASSIFIER, env_id=env_id, memory_mode=args.memory_mode)

    if args.input:
        for point_path in args.input:
            recognize_from_points(point_path, net_seg, net_cls)
    else:
        recognize_from_points(POINT_PATH, net_seg, net_cls)


if __name__ == '__main__':
    main()
