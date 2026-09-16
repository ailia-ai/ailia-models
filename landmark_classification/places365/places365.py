import sys
import time

import numpy as np
import cv2
from PIL import Image
import ailia


# import original modules
sys.path.append('../../util')
# logger
from logging import getLogger  # noqa: E402

import webcamera_utils  # noqa: E402
from image_utils import imread  # noqa: E402
from model_utils import check_and_download_models  # noqa: E402
from arg_utils import get_base_parser, get_savepath, update_parser  # noqa: E402

logger = getLogger(__name__)


# ======================
# Parameters 1
# ======================
IMAGE_PATH = 'input.jpg'
SAVE_IMAGE_PATH = 'output.png'
IMAGE_HEIGHT = 224
IMAGE_WIDTH = 224


# ======================
# Arguemnt Parser Config
# ======================
parser = get_base_parser(
    'Resnet18 ImageNet classification model', IMAGE_PATH, None
)
parser.add_argument(
    '--model', default='resnet18',
    choices=['resnet18', 'alexnet', 'resnet50', 'densenet161', 'wideresnet18']
)
args = update_parser(parser)


# ======================
# Parameters 2
# ======================
ALEXNET_WEIGHT_PATH      = 'alexnet_places365.onnx'
ALEXNET_MODEL_PATH       = 'alexnet_places365.onnx.prototxt'
RESNET18_WEIGHT_PATH     = 'resnet18_places365.onnx'
RESNET18_MODEL_PATH      = 'resnet18_places365.onnx.prototxt'
RESNET50_WEIGHT_PATH     = 'resnet50_places365.onnx'
RESNET50_MODEL_PATH      = 'resnet50_places365.onnx.prototxt'
WIDERESNET18_WEIGHT_PATH = 'wideresnet18_places365.onnx'
WIDERESNET18_MODEL_PATH  = 'wideresnet18_places365.onnx.prototxt'
REMOTE_PATH = 'https://storage.googleapis.com/ailia-models/places365/'


# ======================
# Utils
# ======================
def softmax(x):
    u = np.sum(np.exp(x))
    return np.exp(x)/u

def get_model():
    if args.model == 'resnet18':
        model_path, weight_path = RESNET18_MODEL_PATH, RESNET18_WEIGHT_PATH
    elif args.model == 'resnet50':
        model_path, weight_path = RESNET50_MODEL_PATH, RESNET50_WEIGHT_PATH
    elif args.model == 'alexnet':
        model_path, weight_path = ALEXNET_MODEL_PATH, ALEXNET_WEIGHT_PATH
    elif args.model == 'wideresnet18':
        model_path, weight_path = WIDERESNET18_MODEL_PATH, WIDERESNET18_WEIGHT_PATH
    else:
        logger.info('Invalid model name.')
        exit()
    # model files check and download
    check_and_download_models(weight_path, model_path, REMOTE_PATH)
    # load model
    model = ailia.Net(model_path, weight_path, env_id=args.env_id, memory_mode=args.memory_mode)
    # get weight
    weight = None
    if args.model in ['wideresnet18']:
        import onnx
        from onnx import numpy_helper
        weight = onnx.load(weight_path)
        weight = weight.graph.initializer
        weight = onnx.numpy_helper.to_array(weight[0])
    return model, weight

def get_label_scene_category():
    file_name = 'categories_places365.txt'
    classes = list()
    with open(file_name) as class_file:
        for line in class_file:
            classes.append(line.strip().split(' ')[0][3:])
    classes = tuple(classes)
    return classes

def get_label_indoor_and_outdoor():
    labels_IO = None
    if args.model in ['wideresnet18']:
        file_name_IO = 'IO_places365.txt'
        with open(file_name_IO) as f:
            lines = f.readlines()
            labels_IO = []
            for line in lines:
                items = line.rstrip().split()
                labels_IO.append(int(items[-1]) -1) # 0 is indoor, 1 is outdoor
        labels_IO = np.array(labels_IO)
    return labels_IO

def get_label_scene_attribute():
    labels_attribute, W_attribute = None, None
    if args.model in ['wideresnet18']:
        file_name_attribute = 'labels_sunattribute.txt'
        with open(file_name_attribute) as f:
            lines = f.readlines()
            labels_attribute = [item.rstrip() for item in lines]
        file_name_W = 'W_sceneattribute_wideresnet18.npy'
        W_attribute = np.load(file_name_W)
    return labels_attribute, W_attribute

def apply_centre_crop(img):
    if args.model in ['wideresnet18']:
        img = cv2.resize(img, (224, 224))
    else:
        img = cv2.resize(img, (256, 256))
        pad = int((256-224)/2)
        img = img[pad:-pad, pad:-pad, :]
    img = img / 255
    mean, std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
    for i in range(3):
        img[:, :, i] = (img[:, :, i]-mean[i])/std[i]
    img = img.transpose(2, 0, 1)
    img = img[np.newaxis, :, :, :]
    return img
    
def returnCAM(feature_conv, weight_softmax, class_idx):
    # generate the class activation maps upsample to 256x256
    size_upsample = (256, 256)
    nc, h, w = feature_conv.shape
    output_cam = []
    for idx in class_idx:
        cam = weight_softmax[class_idx].dot(feature_conv.reshape((nc, h*w)))
        cam = cam.reshape(h, w)
        cam = cam - np.min(cam)
        cam_img = cam / np.max(cam)
        cam_img = np.uint8(255 * cam_img)
        output_cam.append(cv2.resize(cam_img, size_upsample))
    return output_cam


# ======================
# Main functions
# ======================
def recognize_from_image(net, weight, classes, labels_IO, labels_attribute, W_attribute):
    # input image loop
    for image_path in args.input:
        # prepare input data
        logger.info(image_path)
        img = Image.open(image_path)
        img = apply_centre_crop(np.array(img))

        output = net.predict({'input_img': img})
        logit = output[0].astype(np.float32)

        if args.model in ['wideresnet18']:
            out_layer4  = np.squeeze(output[1].astype(np.float32))
            out_avgpool = np.squeeze(output[2].astype(np.float32))

        h_x = softmax(logit[0])
        idx = np.argsort(-h_x)
        probs = h_x[idx]

        logger.info('prediction on {}'.format(image_path))

        # output the IO prediction
        if args.model in ['wideresnet18']:
            io_image = np.mean(labels_IO[idx[:10]]) # vote for the indoor or outdoor
            print('--TYPE OF ENVIRONMENT:')
            if io_image < 0.5:
                print('\tindoor')
            else:
                print('\toutdoor')

        # output the prediction of scene category
        print('--SCENE CATEGORIES:')
        for i in range(0, 5):
            print('\t{:.3f} -> {}'.format(probs[i], classes[idx[i]]))

        if args.model in ['wideresnet18']:
            # output the scene attributes
            responses_attribute = W_attribute.dot(out_avgpool)
            idx_a = np.argsort(responses_attribute)
            print('--SCENE ATTRIBUTES:')
            print('\t', ', '.join([labels_attribute[idx_a[i]] for i in range(-1,-10,-1)]))

            if args.savepath is not None:
                # generate class activation mapping
                logger.info('Class activation map is saved as {}'.format(args.savepath))
                CAMs = returnCAM(out_layer4, weight, [idx[0]])

                # render the CAM and output
                img = cv2.imread(image_path)
                height, width, _ = img.shape
                heatmap = cv2.applyColorMap(cv2.resize(CAMs[0],(width, height)), cv2.COLORMAP_JET)
                result = heatmap * 0.4 + img * 0.5
                cv2.imwrite(args.savepath, result)

    logger.info('Script finished successfully.')

def recognize_from_video(net, weight, classes, labels_IO, labels_attribute, W_attribute):
    capture = webcamera_utils.get_capture(args.video)

    # create video writer if savepath is specified as video format
    writer = None
    if args.model in ['wideresnet18']:
        if args.savepath != SAVE_IMAGE_PATH and args.savepath is not None:
            f_h = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
            f_w = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
            save_h, save_w = f_h, f_w
            writer = webcamera_utils.get_writer(args.savepath, save_h, save_w)

    frame_shown = False
    frame_i = 0
    while (True):
        ret, frame = capture.read()
        if (cv2.waitKey(1) & 0xFF == ord('q')) or not ret:
            break
        if frame_shown and cv2.getWindowProperty('frame', cv2.WND_PROP_VISIBLE) == 0:
            break

        # prepare input data
        org = frame
        frame = Image.fromarray(frame)
        img = apply_centre_crop(np.array(frame))

        output = net.predict({'input_img': img})
        logit = output[0].astype(np.float32)

        if args.model in ['wideresnet18']:
            out_layer4  = np.squeeze(output[1].astype(np.float32))
            out_avgpool = np.squeeze(output[2].astype(np.float32))

        h_x = softmax(logit[0])
        idx = np.argsort(-h_x)
        probs = h_x[idx]

        logger.info('prediction on index_{}'.format(frame_i))
        frame_i += 1

        # output the IO prediction
        if args.model in ['wideresnet18']:
            io_image = np.mean(labels_IO[idx[:10]]) # vote for the indoor or outdoor
            print('--TYPE OF ENVIRONMENT:')
            if io_image < 0.5:
                print('\tindoor')
            else:
                print('\toutdoor')

        # output the prediction of scene category
        print('--SCENE CATEGORIES:')
        for i in range(0, 5):
            print('\t{:.3f} -> {}'.format(probs[i], classes[idx[i]]))

        if args.model in ['wideresnet18']:
            # output the scene attributes
            responses_attribute = W_attribute.dot(out_avgpool)
            idx_a = np.argsort(responses_attribute)
            print('--SCENE ATTRIBUTES:')
            print('\t', ', '.join([labels_attribute[idx_a[i]] for i in range(-1,-10,-1)]))

            if writer is not None:
                # generate class activation mapping
                logger.info('Class activation map is saved.')
                CAMs = returnCAM(out_layer4, weight, [idx[0]])

                # render the CAM and output
                height, width, _ = org.shape
                heatmap = cv2.applyColorMap(cv2.resize(CAMs[0],(width, height)), cv2.COLORMAP_JET)
                result = heatmap * 0.4 + org * 0.5
                result = result.astype(np.uint8)
                cv2.imshow('Class activation map', result)
                frame_shown = True
                writer.write(result)

    capture.release()
    cv2.destroyAllWindows()
    if writer is not None:
        writer.release()

    logger.info('Script finished successfully.')


def main():
    # net initialize
    net, weight = get_model()

    # get label
    classes = get_label_scene_category()
    labels_IO = get_label_indoor_and_outdoor()
    labels_attribute, W_attribute = get_label_scene_attribute()

    if args.video is not None:
        # video mode
        recognize_from_video(net, weight, classes, labels_IO, labels_attribute, W_attribute)
    else:
        # image mode
        recognize_from_image(net, weight, classes, labels_IO, labels_attribute, W_attribute)


if __name__ == '__main__':
    main()
