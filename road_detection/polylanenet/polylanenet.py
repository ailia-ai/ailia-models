import sys
import numpy as np
import cv2
import ailia
# import original modules
sys.path.append('../../util')
from arg_utils import get_base_parser, update_parser, get_savepath  # noqa: E402
from model_utils import check_and_download_models  # noqa: E402
from image_utils import imread  # noqa: E402
import webcamera_utils
# logger
from logging import getLogger   # noqa: E402
logger = getLogger(__name__)


# ======================
# Parameters
# ======================
WEIGHT_PATH = 'polylanenet.onnx'
MODEL_PATH = 'polylanenet.onnx.prototxt'
REMOTE_PATH = 'https://storage.googleapis.com/ailia-models/polylanenet/'

IMAGE_PATH = 'input.jpg'
SAVE_IMAGE_PATH = 'output.jpg'

HEIGHT = 360
WIDTH = 640


# ======================
# Arguemnt Parser Config
# ======================
parser = get_base_parser(
    'PolyLaneNet', IMAGE_PATH, SAVE_IMAGE_PATH,
)
args = update_parser(parser)


# ======================
# Main functions
# ======================
def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def decode(output, conf_threshold=0.5, share_top_y=True):
    output = output.reshape(len(output), -1, 7)  # score + upper + lower + 4 coeffs = 7
    output[:, :, 0] = sigmoid(output[:, :, 0])
    output[output[:, :, 0] < conf_threshold] = 0

    if False and share_top_y:
        output[:, :, 0] = output[:, 0, 0].expand(output.shape[0], output.shape[1])

    return output


def draw_annotation(img, pred):
    img = (img/255.).astype(np.float32)
    # Unnormalize
    if False:
        IMAGENET_MEAN = np.array([0.485, 0.456, 0.406])
        IMAGENET_STD = np.array([0.229, 0.224, 0.225])
        img = img * np.array(IMAGENET_STD) + np.array(IMAGENET_MEAN)
    img = (img * 255).astype(np.uint8)

    img_h, img_w, _ = img.shape
    original = img.copy()

    # Draw predictions
    pred = pred[0]
    pred = pred[pred[:, 0] != 0]  # filter invalid lanes
    overlay = img.copy()
    for i, lane in enumerate(pred):
        lane = lane[1:]  # remove conf
        lower, upper = lane[0], lane[1]
        lane = lane[2:]  # remove upper, lower positions

        # generate points from the polynomial
        ys = np.linspace(lower, upper, num=100)
        points = np.zeros((len(ys), 2), dtype=np.int32)
        points[:, 1] = (ys * img_h).astype(int)
        points[:, 0] = (np.polyval(lane, ys) * img_w).astype(int)
        points = points[(points[:, 0] > 0) & (points[:, 0] < img_w)]

        # draw lane with a polyline on the overlay
        for current_point, next_point in zip(points[:-1], points[1:]):
            overlay = cv2.line(overlay, tuple(current_point), tuple(next_point), color=(0, 255, 0), thickness=2)

        # draw lane ID
        if len(points) > 0:
            cv2.putText(img, str(i), tuple(points[0]), fontFace=cv2.FONT_HERSHEY_COMPLEX, fontScale=1, color=(0, 255, 0))

    predicted = img.copy()

    # Add lanes overlay
    w = 0.6
    img = ((1. - w) * img + w * overlay).astype(np.uint8)

    overlayed = img.copy()

    return original, predicted, overlayed


def predict(net, input):
    # resize input image
    height = input.shape[0]
    width = input.shape[1]
    rate = HEIGHT/height
    rs_height = int(height*rate)
    rs_width = int(width*rate)
    if (rs_height!=HEIGHT or rs_width!=WIDTH):
        print('Invalid image shape')
        exit()
    input = cv2.resize(input, (rs_width, rs_height))
    image = input
    input = (input/255.).astype(np.float32)
    input = input.transpose(2, 0, 1)
    input = input[np.newaxis, :, :, :]
    # predict
    output = net.predict(input)
    # postprocess
    output = decode(output)

    return output, image


def recognize_from_image(net):
    for image_path in args.input:
        input = imread(image_path)
        output, image = predict(net, input)
        _, _, overlayed = draw_annotation(image, output)

        args.input[0] = args.input[0].replace('./input/image/', '')
        filename = '_'.join(args.input[0].split('/'))

        savepath = get_savepath(args.savepath, image_path)
        logger.info(f'saved at : {savepath}')
        cv2.imwrite(savepath, overlayed)

    logger.info('Script finished successfully.')

def recognize_from_video(net):
    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        print('Video is not opened.')
        exit()

    if args.savepath != SAVE_IMAGE_PATH:
        writer = webcamera_utils.get_writer(args.savepath, HEIGHT, WIDTH)
    else:
        writer = None

    frame_shown = False

    while True:
        ret, frame = cap.read()
        if (cv2.waitKey(1) & 0xFF == ord('q')) or not ret:
            break
        if frame_shown and cv2.getWindowProperty('frame', cv2.WND_PROP_VISIBLE) == 0:
            break

        output, image = predict(net, frame)
        _, _, overlayed = draw_annotation(image, output)
        cv2.imshow('frame', overlayed)
        frame_shown = True

        # save results
        if writer is not None:
            writer.write(overlayed)

    cap.release()
    cv2.destroyAllWindows()
    if writer is not None:
        writer.release()
    logger.info('Script finished successfully.')

def main():
    # model files check and download
    check_and_download_models(WEIGHT_PATH, MODEL_PATH, REMOTE_PATH)

    net = ailia.Net(MODEL_PATH, WEIGHT_PATH, env_id=args.env_id, memory_mode=args.memory_mode)

    if args.input:
        recognize_from_image(net)

    elif args.video:
        recognize_from_video(net)


if __name__ == '__main__':
    main()
