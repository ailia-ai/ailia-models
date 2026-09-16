import sys
import time
import math

import numpy as np
import cv2
from PIL import Image

import ailia

# import original modules
sys.path.append('../../util')
from arg_utils import get_base_parser, update_parser, get_savepath  # noqa
from model_utils import check_and_download_models  # noqa
from image_utils import normalize_image  # noqa
from detector_utils import load_image, plot_results  # noqa
from webcamera_utils import get_capture, get_writer  # noqa
from math_utils import sigmoid
from nms_utils import batched_nms
# logger
from logging import getLogger  # noqa

from glip_utils import run_ner, create_positive_map
from glip_utils import create_positive_map_label_to_token_from_positive_map
import bert_model
from bert_model import bert_encoder
from anchor import anchor_generator

logger = getLogger(__name__)

# ======================
# Parameters
# ======================

WEIGHT_BKBN_PATH = "swin_tiny_backbone.onnx"
MODEL_BKBN_PATH = "swin_tiny_backbone.onnx.prototxt"
WEIGHT_BERT_PATH = "swin_tiny_bert.onnx"
MODEL_BERT_PATH = "swin_tiny_bert.onnx.prototxt"
WEIGHT_RPN_PATH = "swin_tiny_rpn.onnx"
MODEL_RPN_PATH = "swin_tiny_rpn.onnx.prototxt"
REMOTE_PATH = 'https://storage.googleapis.com/ailia-models/glip/'

IMAGE_PATH = 'demo.jpg'
SAVE_IMAGE_PATH = 'output.png'

IMG_MIN_SIZE = 800

THRESHOLD = 0.5

# ======================
# Arguemnt Parser Config
# ======================

parser = get_base_parser(
    'GLIP', IMAGE_PATH, SAVE_IMAGE_PATH
)
parser.add_argument(
    '-c', '--caption', default='sofa . remote . dog . person . car . sky . plane',
    help='The caption for detect.'
)
parser.add_argument(
    '-th', '--threshold',
    default=THRESHOLD, type=float,
    help='The detection threshold.'
)
parser.add_argument(
    '--onnx',
    action='store_true',
    help='execute onnxruntime version.'
)
parser.add_argument(
    '--disable_ailia_tokenizer',
    action='store_true',
    help='disable ailia tokenizer.'
)
args = update_parser(parser)


# ======================
# Secondaty Functions
# ======================


def permute_and_flatten(layer, N, A, C, H, W):
    layer = layer.reshape(N, -1, C, H, W)
    layer = layer.transpose(0, 3, 4, 1, 2)
    layer = layer.reshape(N, -1, C)

    return layer


def box_decode(preds, anchors):
    TO_REMOVE = 1
    widths = anchors[:, 2] - anchors[:, 0] + TO_REMOVE
    heights = anchors[:, 3] - anchors[:, 1] + TO_REMOVE
    ctr_x = (anchors[:, 2] + anchors[:, 0]) / 2
    ctr_y = (anchors[:, 3] + anchors[:, 1]) / 2

    wx, wy, ww, wh = (10., 10., 5., 5.)
    dx = preds[:, 0::4] / wx
    dy = preds[:, 1::4] / wy
    dw = preds[:, 2::4] / ww
    dh = preds[:, 3::4] / wh

    # Prevent sending too large values into exp()
    dw = np.clip(dw, None, math.log(1000. / 16))
    dh = np.clip(dh, None, math.log(1000. / 16))

    pred_ctr_x = dx * widths[:, None] + ctr_x[:, None]
    pred_ctr_y = dy * heights[:, None] + ctr_y[:, None]
    pred_w = np.exp(dw) * widths[:, None]
    pred_h = np.exp(dh) * heights[:, None]

    pred_boxes = np.zeros_like(preds)
    pred_boxes[:, 0::4] = pred_ctr_x - 0.5 * (pred_w - 1)
    pred_boxes[:, 1::4] = pred_ctr_y - 0.5 * (pred_h - 1)
    pred_boxes[:, 2::4] = pred_ctr_x + 0.5 * (pred_w - 1)
    pred_boxes[:, 3::4] = pred_ctr_y + 0.5 * (pred_h - 1)

    return pred_boxes


def clip_to_image(bbox, image_size):
    TO_REMOVE = 1
    x1s = np.clip(bbox[:, 0], 0, image_size[1] - TO_REMOVE)
    y1s = np.clip(bbox[:, 1], 0, image_size[0] - TO_REMOVE)
    x2s = np.clip(bbox[:, 2], 0, image_size[1] - TO_REMOVE)
    y2s = np.clip(bbox[:, 3], 0, image_size[0] - TO_REMOVE)
    bbox = np.stack((x1s, y1s, x2s, y2s), axis=-1)

    return bbox


def remove_small_boxes(bbox, min_size):
    xmin, ymin, xmax, ymax = np.split(bbox, 4, axis=-1)
    TO_REMOVE = 1
    ws = xmax - xmin + TO_REMOVE,
    hs = ymax - ymin + TO_REMOVE

    ws = np.squeeze(ws)
    hs = np.squeeze(hs)
    keep = ((ws >= min_size) & (hs >= min_size)).nonzero()[0]

    return keep


def cat_boxlist(bboxes):
    bbox = np.concatenate([x[0] for x in bboxes], axis=0)
    labels = np.concatenate([x[1] for x in bboxes], axis=0)
    scores = np.concatenate([x[2] for x in bboxes], axis=0)

    return (bbox, labels, scores)


def select_over_all_levels(boxlists):
    nms_thresh = 0.6
    fpn_post_nms_top_n = 100

    num_images = len(boxlists)
    results = []
    for i in range(num_images):
        (bbox, labels, scores) = boxlists[i]

        # multiclass nms
        keep = batched_nms(bbox, scores, labels, nms_thresh)
        bbox = bbox[keep]
        scores = scores[keep]
        labels = labels[keep]

        number_of_detections = len(keep)

        # Limit to max_per_image detections **over all classes**
        if number_of_detections > fpn_post_nms_top_n:
            kth = number_of_detections - fpn_post_nms_top_n + 1
            image_thresh = np.partition(scores, kth)[kth]
            keep = scores >= image_thresh
            keep = np.nonzero(keep)[0]
            bbox = bbox[keep]
            scores = scores[keep]
            labels = labels[keep]

        results.append((bbox, labels, scores))

    return results


# ======================
# Main functions
# ======================

def preprocess(img):
    im_h, im_w, _ = img.shape

    img = img[:, :, ::-1]  # BGR -> RGB

    # adaptive_resize
    scale = IMG_MIN_SIZE / min(im_h, im_w)
    ow, oh = int(im_w * scale), int(im_h * scale)
    if ow != im_w or oh != im_h:
        img = np.array(Image.fromarray(img).resize((ow, oh), Image.Resampling.BILINEAR))

    img = normalize_image(img, 'ImageNet')

    img = img[:, :, ::-1]  # RGB -> BGR
    img = img.transpose(2, 0, 1)  # HWC -> CHW
    img = img.astype(np.float32)

    return img


def post_processing(
        box_regression, centerness, anchors,
        box_cls,
        dot_product_logits,
        positive_map):
    sampled_boxes = []
    anchors = [anchors]
    anchors = list(zip(*anchors))

    for idx, (b, c, a) in enumerate(zip(box_regression, centerness, anchors)):
        if box_cls is not None:
            o = box_cls[idx]
        if dot_product_logits is not None:
            d = dot_product_logits[idx]

        sampled_boxes.append(
            forward_for_single_feature_map(b, c, a, o, d, positive_map)
        )

    boxlists = list(zip(*sampled_boxes))
    boxlists = [cat_boxlist(boxlist) for boxlist in boxlists]

    boxlists = select_over_all_levels(boxlists)

    return boxlists


def forward_for_single_feature_map(
        box_regression, centerness, anchors,
        box_cls=None,
        dot_product_logits=None,
        positive_map=None):
    N, A, H, W = box_regression.shape
    A = A // 4

    # put in the same format as anchors
    if box_cls is not None:
        C = box_cls.shape[1] // A
        box_cls = permute_and_flatten(box_cls, N, A, C, H, W)
        box_cls = sigmoid(box_cls)

    # binary dot product focal version
    if dot_product_logits is not None:
        dot_product_logits = sigmoid(dot_product_logits)
        scores = convert_grounding_to_od_logits(
            logits=dot_product_logits, box_cls=box_cls,
            positive_map=positive_map)
        box_cls = scores

    box_regression = permute_and_flatten(box_regression, N, A, 4, H, W)
    box_regression = box_regression.reshape(N, -1, 4)

    pre_nms_thresh = 0.05
    candidate_inds = box_cls > pre_nms_thresh
    pre_nms_top_n = np.sum(candidate_inds.reshape(N, -1), axis=1)
    pre_nms_top_n = np.clip(pre_nms_top_n, None, pre_nms_top_n)

    centerness = permute_and_flatten(centerness, N, A, 1, H, W)
    centerness = sigmoid(centerness.reshape(N, -1))

    # multiply the classification scores with centerness scores
    box_cls = box_cls * centerness[:, :, None]

    results = []
    for per_box_cls, per_box_regression, per_pre_nms_top_n, per_candidate_inds, per_anchors \
            in zip(box_cls, box_regression, pre_nms_top_n, candidate_inds, anchors):
        per_box_cls = per_box_cls[per_candidate_inds]

        top_k_indices = np.argsort(-per_box_cls)[:per_pre_nms_top_n]
        per_box_cls = per_box_cls[top_k_indices]

        per_candidate_nonzeros = np.stack(per_candidate_inds.nonzero())
        per_candidate_nonzeros = per_candidate_nonzeros.T
        per_candidate_nonzeros = per_candidate_nonzeros[top_k_indices, :]

        per_box_loc = per_candidate_nonzeros[:, 0]
        per_class = per_candidate_nonzeros[:, 1] + 1

        detections = box_decode(
            per_box_regression[per_box_loc, :].reshape(-1, 4),
            per_anchors.bbox[per_box_loc, :].reshape(-1, 4)
        )

        labels = per_class
        scores = np.sqrt(per_box_cls)
        bbox = clip_to_image(detections, per_anchors.image_size)

        # remove_empty
        keep = (bbox[:, 3] > bbox[:, 1]) & (bbox[:, 2] > bbox[:, 0])
        bbox = bbox[keep]
        labels = labels[keep]
        scores = scores[keep]

        keep = remove_small_boxes(bbox, 0)
        bbox = bbox[keep]
        labels = labels[keep]
        scores = scores[keep]

        results.append((bbox, labels, scores))

    return results


def convert_grounding_to_od_logits(
        logits, box_cls, positive_map):
    scores = np.zeros((logits.shape[0], logits.shape[1], box_cls.shape[2]))
    # 256 -> 80, average for each class

    # score aggregation method
    for label_j in positive_map:
        scores[:, :, label_j - 1] = np.mean(logits[:, :, positive_map[label_j]], axis=-1)

    return scores


def predict(models, img, caption):
    im_h, im_w = img.shape[:2]

    img = preprocess(img)
    pp_h, pp_w = img.shape[1:]

    # padding
    size_divisible = 32
    pad_h = int(math.ceil(pp_h / size_divisible) * size_divisible)
    pad_w = int(math.ceil(pp_w / size_divisible) * size_divisible)
    pad_img = np.zeros((1, 3, pad_h, pad_w), dtype=np.float32)
    pad_img[0, :, :pp_h, :pp_w] = img
    img = pad_img

    tokenizer = models["tokenizer"]
    max_length = 256

    tokenized = tokenizer.batch_encode_plus(
        [caption],
        max_length=max_length,
        padding='max_length',
        #return_special_tokens_mask=True,
        return_tensors='np',
        truncation=True)

    tokens_positive, entity_names = run_ner(caption)
    positive_map = create_positive_map(tokenized, tokens_positive)
    positive_map = create_positive_map_label_to_token_from_positive_map(positive_map, plus=1)

    # language embedding
    net = models['bert_encoder']
    language_dict_features = bert_encoder(
        net,
        tokenized.input_ids,
        tokenized.attention_mask,
        tokenized.token_type_ids)

    net = models['backbone']
    if not args.onnx:
        output = net.predict([img])
    else:
        output = net.run(None, {'images': img})

    features = output
    hidden = language_dict_features["hidden"]
    masks = language_dict_features["masks"]

    net = models['rpn']
    if not args.onnx:
        output = net.predict(features + [hidden, masks])
    else:
        output = net.run(None, {
            "feat0": features[0], "feat1": features[1], "feat2": features[2],
            "feat3": features[3], "feat4": features[4],
            "hidden": hidden, "masks": masks,
        })

    box_cls = output[:5]
    box_regression = output[5:10]
    centerness = output[10:15]
    dot_product_logits = output[15:]

    anchors = anchor_generator((pp_h, pp_w), features)

    proposals = post_processing(
        box_regression, centerness, anchors, box_cls, dot_product_logits, positive_map)
    proposals = proposals[0]

    bboxes, labels, scores = proposals

    # reshape prediction into the original image size
    ratio_height, ratio_width = im_h / pp_h, im_w / pp_w
    xmin, ymin, xmax, ymax = np.split(bboxes, 4, axis=-1)
    scaled_xmin = xmin * ratio_width
    scaled_xmax = xmax * ratio_width
    scaled_ymin = ymin * ratio_height
    scaled_ymax = ymax * ratio_height
    bboxes = np.concatenate(
        (scaled_xmin, scaled_ymin, scaled_xmax, scaled_ymax), axis=-1
    )

    thresh = args.threshold
    keep = np.nonzero(scores > thresh)[0]
    bboxes = bboxes[keep]
    labels = labels[keep]
    scores = scores[keep]

    # sort
    ind = np.argsort(-scores)
    bboxes = bboxes[ind]
    labels = labels[ind]
    scores = scores[ind]

    new_labels = []
    for i in labels:
        if i <= len(entity_names):
            new_labels.append(entity_names[i - 1])
    labels = new_labels

    detections = []
    for bbox, score, label in zip(bboxes, scores, labels):
        x1, y1, x2, y2 = bbox
        r = ailia.DetectorObject(
            category=label,
            prob=score,
            x=x1 / im_w,
            y=y1 / im_h,
            w=(x2 - x1) / im_w,
            h=(y2 - y1) / im_h,
        )
        detections.append(r)

    return detections


def recognize_from_image(models):
    caption = args.caption

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
                detect_objects = predict(models, img, caption)
                end = int(round(time.time() * 1000))
                estimation_time = (end - start)

                # Logging
                logger.info(f'\tailia processing estimation time {estimation_time} ms')
                if i != 0:
                    total_time_estimation = total_time_estimation + estimation_time

            logger.info(f'\taverage time estimation {total_time_estimation / (args.benchmark_count - 1)} ms')
        else:
            detect_objects = predict(models, img, caption)

        res_img = plot_results(detect_objects, img)

        # plot result
        savepath = get_savepath(args.savepath, image_path, ext='.png')
        logger.info(f'saved at : {savepath}')
        cv2.imwrite(savepath, res_img)

    logger.info('Script finished successfully.')


def recognize_from_video(models):
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

    caption = args.caption

    frame_shown = False
    while True:
        ret, frame = capture.read()
        if (cv2.waitKey(1) & 0xFF == ord('q')) or not ret:
            break
        if frame_shown and cv2.getWindowProperty('frame', cv2.WND_PROP_VISIBLE) == 0:
            break

        # inference
        detect_objects = predict(models, frame, caption)

        # plot result
        res_img = plot_results(detect_objects, frame)

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
    # model files check and download
    check_and_download_models(WEIGHT_BKBN_PATH, MODEL_BKBN_PATH, REMOTE_PATH)
    check_and_download_models(WEIGHT_BERT_PATH, MODEL_BERT_PATH, REMOTE_PATH)
    check_and_download_models(WEIGHT_RPN_PATH, MODEL_RPN_PATH, REMOTE_PATH)

    env_id = args.env_id

    # initialize
    if not args.onnx:
        logger.info("This model requires 30GB or more memory. If you don't have enough vram, please use CPU mode with -e 0 option.")
        backbone = ailia.Net(MODEL_BKBN_PATH, WEIGHT_BKBN_PATH, env_id=env_id, memory_mode=args.memory_mode)
        bert_encoder = ailia.Net(MODEL_BERT_PATH, WEIGHT_BERT_PATH, env_id=env_id, memory_mode=args.memory_mode)
        rpn = ailia.Net(MODEL_RPN_PATH, WEIGHT_RPN_PATH, env_id=env_id, memory_mode=args.memory_mode)
    else:
        import onnxruntime
        backbone = onnxruntime.InferenceSession(WEIGHT_BKBN_PATH)
        bert_encoder = onnxruntime.InferenceSession(WEIGHT_BERT_PATH)
        rpn = onnxruntime.InferenceSession(WEIGHT_RPN_PATH)
        bert_model.onnx = True

    if args.disable_ailia_tokenizer:
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    else:
        from ailia_tokenizer import BertTokenizer
        tokenizer = BertTokenizer.from_pretrained("./tokenizer/")

    models = dict(
        tokenizer=tokenizer,
        backbone=backbone,
        bert_encoder=bert_encoder,
        rpn=rpn,
    )

    if args.video is not None:
        recognize_from_video(models)
    else:
        recognize_from_image(models)


if __name__ == '__main__':
    main()
