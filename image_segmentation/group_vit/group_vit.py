import sys
import time
import os.path as osp

import numpy as np
import cv2
import matplotlib.pyplot as plt

import ailia

# import original modules
sys.path.append('../../util')
from arg_utils import get_base_parser, update_parser, get_savepath  # noqa
from model_utils import check_and_download_models  # noqa
from image_utils import normalize_image  # noqa
from detector_utils import load_image  # noqa
from math_utils import softmax
from webcamera_utils import get_capture, get_writer  # noqa
# logger
from logging import getLogger  # noqa

from tokenizer import Tokenize, SimpleTokenizer

logger = getLogger(__name__)

# ======================
# Parameters
# ======================

WEIGHT_YFCC_PATH = 'group_vit_gcc_yfcc_30e-74d335e6.onnx'
MODEL_YFCC_PATH = 'group_vit_gcc_yfcc_30e-74d335e6.onnx.prototxt'
WEIGHT_YFCC_MLC_PATH = 'group_vit_gcc_yfcc_mlc.onnx'
MODEL_YFCC_MLC_PATH = 'group_vit_gcc_yfcc_mlc.onnx.prototxt'
WEIGHT_REDCAP_PATH = 'group_vit_gcc_redcap_30e-3dd09a76.onnx'
MODEL_REDCAP_PATH = 'group_vit_gcc_redcap_30e-3dd09a76.onnx.prototxt'
WEIGHT_REDCAP_MLC_PATH = 'group_vit_gcc_redcap_mlc.onnx'
MODEL_REDCAP_MLC_PATH = 'group_vit_gcc_redcap_mlc.onnx.prototxt'
REMOTE_PATH = 'https://storage.googleapis.com/ailia-models/group_vit/'

CLASSES = [
    'background', 'aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus',
    'car', 'cat', 'chair', 'cow', 'table', 'dog', 'horse',
    'motorbike', 'person', 'plant', 'sheep', 'sofa', 'train', 'monitor'
]

IMAGE_PATH = 'voc.jpg'
SAVE_IMAGE_PATH = 'output.png'
PALLET_TEXT = 'group_palette.txt'

MAX_SEQ_LEN = 77

PALETTE = [
    [0, 0, 0], [128, 0, 0], [0, 128, 0], [128, 128, 0], [0, 0, 128], [128, 0, 128], [0, 128, 128],
    [128, 128, 128], [64, 0, 0], [192, 0, 0], [64, 128, 0], [192, 128, 0], [64, 0, 128], [192, 0, 128],
    [64, 128, 128], [192, 128, 128], [0, 64, 0], [128, 64, 0], [0, 192, 0], [128, 192, 0], [0, 64, 128]
]

LOGIT_SCALE = {
    'yfcc': 4.2057,
    'redcap': 4.0383,
}

# ======================
# Arguemnt Parser Config
# ======================

parser = get_base_parser(
    'GroupViT', IMAGE_PATH, SAVE_IMAGE_PATH
)
parser.add_argument(
    '-ac', '--additional-class', default=None, nargs='+',
    help='Specify the additional classes, could be a list.',
)
parser.add_argument(
    '-m', '--model_type', default='yfcc', choices=('yfcc', 'redcap'),
    help='model type'
)
args = update_parser(parser)


# ======================
# Secondaty Functions
# ======================

def build_class_tokens(text_transform, classnames):
    tokens = []
    templates = ['a photo of a {}.']
    for classname in classnames:
        # format with class
        tokens.append(np.stack([text_transform(template.format(classname)) for template in templates]))

    # [N, T, L], N: number of instance, T: number of captions (including ensembled), L: sequence length
    tokens = np.stack(tokens)

    return tokens


def seg2coord(seg_map):
    """
    Args:
        seg_map (np.ndarray): (H, W)

    Return:
        dict(group_id -> (x, y))
    """
    h, w = seg_map.shape
    # [h ,w, 2]
    coords = np.stack(np.meshgrid(np.arange(h), np.arange(w), indexing='ij'), axis=-1)
    labels = np.unique(seg_map)
    coord_map = {}
    for label in labels:
        coord_map[label] = coords[seg_map == label].mean(axis=0)
    return coord_map


def blend_result(img, seg, opacity=0.5):
    color_seg = np.zeros((seg.shape[0], seg.shape[1], 3), dtype=np.uint8)
    for label, color in enumerate(PALETTE):
        color_seg[seg == label, :] = color
    # convert to BGR
    color_seg = color_seg[..., ::-1]

    fg_mask = seg != 0
    img[fg_mask] = img[fg_mask] * (1 - opacity) + color_seg[fg_mask] * opacity

    img = img.astype(np.uint8)

    return img


def show_result(img, pred):
    labels = np.unique(pred)
    coord_map = seg2coord(pred)

    blended_img = blend_result(img, pred, opacity=0.5)
    blended_img = blended_img[:, :, ::-1]  # BGR -> RGB

    width, height = img.shape[1], img.shape[0]
    EPS = 1e-2
    fig = plt.figure(frameon=False)
    canvas = fig.canvas
    dpi = fig.get_dpi()
    fig.set_size_inches((width + EPS) / dpi, (height + EPS) / dpi)

    # remove white edges by set subplot margin
    plt.subplots_adjust(left=0, right=1, bottom=0, top=1)
    ax = plt.gca()
    ax.axis('off')
    for i, label in enumerate(labels):
        if label == 0:
            continue
        center = coord_map[label].astype(np.int32)
        label_text = CLASSES[label]
        ax.text(
            center[1],
            center[0],
            f'{label_text}',
            bbox={
                'facecolor': 'black',
                'alpha': 0.5,
                'pad': 0.7,
                'edgecolor': 'none'
            },
            color='orangered',
            fontsize=16,
            verticalalignment='top',
            horizontalalignment='left')

    plt.imshow(blended_img)
    stream, _ = canvas.print_to_buffer()
    buffer = np.frombuffer(stream, dtype='uint8')
    plt.close()

    img_rgba = buffer.reshape(height, width, 4)
    rgb, alpha = np.split(img_rgba, [3], axis=2)
    img = rgb.astype('uint8')
    img = img[:, :, ::-1]

    return img


# ======================
# Main functions
# ======================


def preprocess(img):
    h, w, _ = img.shape

    max_long_edge = 2048
    max_short_edge = 448
    scale_factor = min(
        max_long_edge / max(h, w), max_short_edge / min(h, w))

    oh, ow = int(h * scale_factor + 0.5), int(w * scale_factor + 0.5)
    if oh != h or ow != w:
        img = cv2.resize(img, (ow, oh), interpolation=cv2.INTER_LINEAR)

    img = normalize_image(img, normalize_type='ImageNet')

    img = img.transpose(2, 0, 1)  # HWC -> CHW
    img = np.expand_dims(img, axis=0)
    img = img.astype(np.float32)

    return img


def post_processing(text_embedding, attn_map, grouped_img_tokens, img_avg_feat):
    # [H, W, G]
    onehot_attn_map = np.eye(attn_map.shape[-1])[np.argmax(attn_map, axis=-1)]

    num_fg_classes = text_embedding.shape[0]
    class_offset = 1
    text_tokens = text_embedding
    num_classes = num_fg_classes + 1

    logit_scale = LOGIT_SCALE[args.model_type]
    logit_scale = np.clip(np.exp(logit_scale), None, 100)
    # [G, N]
    group_affinity_mat = (grouped_img_tokens @ text_tokens.T) * logit_scale
    pre_group_affinity_mat = softmax(group_affinity_mat, axis=-1)

    avg_affinity_mat = (img_avg_feat @ text_tokens.T) * logit_scale
    avg_affinity_mat = softmax(avg_affinity_mat, axis=-1)

    k = min(5, num_fg_classes)
    indices = np.argsort(-avg_affinity_mat, axis=-1)[..., :k]
    affinity_mask = np.sum(np.eye(avg_affinity_mat.shape[-1])[indices[0]], axis=0, keepdims=True)
    affinity_mask = ~affinity_mask.astype(bool)
    affinity_mask = np.repeat(affinity_mask, group_affinity_mat.shape[0], axis=0)
    group_affinity_mat[affinity_mask] = float('-inf')
    group_affinity_mat = softmax(group_affinity_mat, axis=-1)

    group_affinity_mat *= pre_group_affinity_mat

    pred_logits = np.zeros((num_classes,) + attn_map.shape[:2])

    pred_logits[class_offset:] = (onehot_attn_map @ group_affinity_mat).transpose(2, 0, 1)
    bg_thresh = 0.95
    bg_thresh = min(bg_thresh, np.max(group_affinity_mat))
    pred_logits[0, (onehot_attn_map @ group_affinity_mat).max(axis=-1) < bg_thresh] = 1

    seg_logit = np.expand_dims(pred_logits, axis=0)

    return seg_logit


def predict(net, img, text_embedding):
    h, w, _ = img.shape

    img = img[:, :, ::-1]  # BGR -> RGB
    img = preprocess(img)

    # feedforward
    output = net.predict([img])
    attn_map, grouped_img_tokens, img_avg_feat = output

    seg_logit = post_processing(text_embedding, attn_map, grouped_img_tokens, img_avg_feat)
    seg_logit = (seg_logit[0]).transpose(1, 2, 0)

    oh, ow, _ = seg_logit.shape
    if oh != oh or ow != w:
        seg_logit = cv2.resize(seg_logit, (w, h), interpolation=cv2.INTER_LINEAR)

    seg_pred = np.argmax(seg_logit, axis=-1)

    return seg_pred


def recognize_from_image(net, text_embedding):
    # input image loop
    for image_path in args.input:
        logger.info(image_path)

        # prepare input data
        img = load_image(image_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)

        # inference
        logger.info('Start inference...')
        if args.benchmark:
            logger.info('BENCHMARK mode')
            total_time_estimation = 0
            for i in range(args.benchmark_count):
                start = int(round(time.time() * 1000))
                pred = predict(net, img, text_embedding)
                end = int(round(time.time() * 1000))
                estimation_time = (end - start)

                # Logging
                logger.info(f'\tailia processing estimation time {estimation_time} ms')
                if i != 0:
                    total_time_estimation = total_time_estimation + estimation_time

            logger.info(f'\taverage time estimation {total_time_estimation / (args.benchmark_count - 1)} ms')
        else:
            pred = predict(net, img, text_embedding)

        res_img = show_result(img, pred)

        # plot result
        savepath = get_savepath(args.savepath, image_path, ext='.png')
        logger.info(f'saved at : {savepath}')
        cv2.imwrite(savepath, res_img)

    logger.info('Script finished successfully.')


def recognize_from_video(net, text_embedding):
    video_file = args.video if args.video else args.input[0]
    capture = get_capture(video_file)
    assert capture.isOpened(), 'Cannot capture source'

    # create video writer if savepath is specified as video format
    if args.savepath != SAVE_IMAGE_PATH:
        f_h = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        f_w = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        writer = get_writer(args.savepath, f_h, f_w)
    else:
        writer = None

    frame_shown = False
    while True:
        ret, frame = capture.read()
        if (cv2.waitKey(1) & 0xFF == ord('q')) or not ret:
            break
        if frame_shown and cv2.getWindowProperty('frame', cv2.WND_PROP_VISIBLE) == 0:
            break

        # inference
        img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pred = predict(net, img, text_embedding)

        # plot result
        res_img = show_result(frame, pred)

        # show
        cv2.imshow('frame', res_img)
        frame_shown = True

        # save results
        if writer is not None:
            res_img = res_img.astype(np.uint8)
            writer.write(res_img)

    capture.release()
    cv2.destroyAllWindows()
    if writer is not None:
        writer.release()

    logger.info('Script finished successfully.')


def main():
    dic_model = {
        'yfcc': {
            'enc_dec': (WEIGHT_YFCC_PATH, MODEL_YFCC_PATH),
            'txt_enc': (WEIGHT_YFCC_MLC_PATH, MODEL_YFCC_MLC_PATH)
        },
        'redcap': {
            'enc_dec': (WEIGHT_REDCAP_PATH, MODEL_REDCAP_PATH),
            'txt_enc': (WEIGHT_REDCAP_MLC_PATH, MODEL_REDCAP_MLC_PATH)
        },
    }
    info = dic_model[args.model_type]
    WEIGHT_PATH, MODEL_PATH = info['enc_dec']
    WEIGHT_TXTENC_PATH, MODEL_TXTENC_PATH = info['txt_enc']

    # model files check and download
    check_and_download_models(WEIGHT_TXTENC_PATH, MODEL_TXTENC_PATH, REMOTE_PATH)
    check_and_download_models(WEIGHT_PATH, MODEL_PATH, REMOTE_PATH)

    env_id = args.env_id

    # initialize
    net = ailia.Net(MODEL_PATH, WEIGHT_PATH, env_id=env_id, memory_mode=args.memory_mode)
    text_enc = ailia.Net(MODEL_TXTENC_PATH, WEIGHT_TXTENC_PATH, env_id=env_id, memory_mode=args.memory_mode)

    # additional classes
    additional_classes = args.additional_class
    if additional_classes:
        CLASSES.extend(additional_classes)
        EXT_PALETTE = np.loadtxt(
            osp.join(osp.dirname(osp.abspath(__file__)), PALLET_TEXT),
            dtype=np.uint8)[:, ::-1]
        PALETTE.extend(
            EXT_PALETTE[
                np.random.choice(
                    range(EXT_PALETTE.shape[0]), len(additional_classes))
            ])

    # text_embedding
    transform = Tokenize(SimpleTokenizer(), max_seq_len=MAX_SEQ_LEN)
    text_tokens = build_class_tokens(transform, CLASSES[1:])
    output = text_enc.predict([text_tokens])
    text_embedding = output[0]

    if args.video is not None:
        recognize_from_video(net, text_embedding)
    else:
        recognize_from_image(net, text_embedding)


if __name__ == '__main__':
    main()
