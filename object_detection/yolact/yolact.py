import numpy as np
import argparse
import sys
import os
from glob import glob
import time

import cv2

from yolact_util import COLORS,FastBaseTransform ,cfg, postprocess,Detect

import ailia
sys.path.append('../../util')
from arg_utils import get_base_parser, update_parser, get_savepath
from model_utils import check_and_download_models  # noqa: E402
import webcamera_utils  # noqa: E402
from image_utils import imread  # noqa: E402

# logger
from logging import getLogger

logger = getLogger(__name__)

WEIGHT_PATH = './yolact.onnx'
MODEL_PATH = './yolact.onnx.prototxt'
REMOTE_PATH = 'https://storage.googleapis.com/ailia-models/yolact/'

IMAGE_PATH = 'demo.jpg'
SAVE_IMAGE_PATH = 'output.png'

def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

parser = get_base_parser(
    'Yolact model', IMAGE_PATH, SAVE_IMAGE_PATH
)
parser.add_argument('--top_k', default=5, type=int,
                    help='Further restrict the number of predictions to parse')
parser.add_argument('--fast_nms', default=True, type=str2bool,
                    help='Whether to use a faster, but not entirely correct version of NMS.')
parser.add_argument('--display_masks', default=True, type=str2bool,
                    help='Whether or not to display masks over bounding boxes')
parser.add_argument('--display_bboxes', default=True, type=str2bool,
                    help='Whether or not to display bboxes around masks')
parser.add_argument('--display_text', default=True, type=str2bool,
                    help='Whether or not to display text (class [score])') 
parser.add_argument('--display_scores', default=True, type=str2bool,
                    help='Whether or not to display scores in addition to classes')
parser.add_argument('--config', default=None,
                    help='The config object to use.')
parser.add_argument('--image', default=None, type=str,
                    help='A path to an image to use for display.')
parser.add_argument('--score_threshold', default=0, type=float,
                    help='Detections with a score under this threshold will not be considered. This currently only works in display mode.')
parser.add_argument('--dataset', default=None, type=str,
                    help='If specified, override the dataset specified in the config with this one (example: coco2017_dataset).')
parser.add_argument('--no_crop', default=False, dest='crop', action='store_false',
                    help='Do not crop output masks with the predicted bounding box.')

parser.set_defaults(display=False,
                         crop=True)
 
args = update_parser(parser)


def prep_display(dets_out, img, h, w, class_color=False, mask_alpha=0.45):
    """
    Note: If undo_transform=False then im_h and im_w are allowed to be None.
    """
    
    img_gpu = img / 255.0
    h, w, _ = img.shape

    t = postprocess(dets_out, w, h, crop_masks        = args.crop,
                                    score_threshold   = args.score_threshold)
    if t is None:
        return img

    if len(t) <= 1 :
        return img
    
    if cfg.eval_mask_branch:
       # Masks are drawn on the GPU, so don't copy
       masks = t[3][:args.top_k]
    classes, scores, boxes = [x[:args.top_k] for x in t[:3]]

    num_dets_to_consider = min(args.top_k, classes.shape[0])
    for j in range(num_dets_to_consider):
        if scores[j] < args.score_threshold:
            num_dets_to_consider = j
            break
    
    if num_dets_to_consider == 0:
        # No detections found so just output the original image
        return (img_gpu * 255).byte().cpu().numpy()

    # Quick and dirty lambda for selecting the color for a particular index
    # Also keeps track of a per-gpu color cache for maximum speed
    def get_color(j, on_gpu=None):
        global color_cache
        color_idx = (classes[j] * 5 if class_color else j * 5) % len(COLORS)
        
        if on_gpu is not None and color_idx in color_cache[on_gpu]:
            return color_cache[on_gpu][color_idx]
        else:
            color = COLORS[color_idx]
            color = (color[2], color[1], color[0])

            return color

    # First, draw the masks on the GPU where we can do it really fast
    # Beware: very fast but possibly unintelligible mask-drawing code ahead
    # I wish I had access to OpenGL or Vulkan but alas, I guess Pytorch tensor operations will have to suffice
    if args.display_masks and cfg.eval_mask_branch:
        # After this, mask is of size [num_dets, h, w, 1]
        masks = masks[:num_dets_to_consider, :, :, None]
        
        # Prepare the RGB images for each mask given their color (size [num_dets, h, w, 1])
        x = [(np.array(get_color(j)) / 255 ).reshape(1, 1, 1, 3) for j in range(num_dets_to_consider)]
        colors = np.concatenate(x, axis=0)
        masks_color = np.tile(masks,(1, 1, 1, 3)) * colors * mask_alpha

        # This is 1 everywhere except for 1-mask_alpha where the mask is
        inv_alph_masks = masks * (-mask_alpha) + 1
        
        # I did the math for this on pen and paper. This whole block should be equivalent to:
        masks_color_summand = masks_color[0]
        if num_dets_to_consider > 1:
            inv_alph_cumul = inv_alph_masks[:(num_dets_to_consider-1)].cumprod(axis=0)
            masks_color_cumul = masks_color[1:] * inv_alph_cumul
            masks_color_summand += masks_color_cumul.sum(axis=0)

        img_gpu = img_gpu * inv_alph_masks.prod(axis=0) + masks_color_summand
        
    # Then draw the stuff that needs to be done on the cpu
    # Note, make sure this is a uint8 tensor or opencv will not anti alias text for whatever reason
    img_numpy = (img_gpu * 255).astype(np.uint8)

    classes = classes.astype(np.int8)

    if args.display_text or args.display_bboxes:
        for j in reversed(range(num_dets_to_consider)):
            x1, y1, x2, y2 = boxes[j, :]
            color = get_color(j)
            score = scores[j]

            if args.display_bboxes:
                cv2.rectangle(img_numpy, (x1, y1), (x2, y2), color, 1)

            if args.display_text:
                _class = cfg.dataset.class_names[classes[j]]
                text_str = '%s: %.2f' % (_class, score) if args.display_scores else _class

                font_face = cv2.FONT_HERSHEY_DUPLEX
                font_scale = 0.6
                font_thickness = 1

                text_w, text_h = cv2.getTextSize(text_str, font_face, font_scale, font_thickness)[0]

                text_pt = (x1, y1 - 3)
                text_color = [255, 255, 255]

                cv2.rectangle(img_numpy, (x1, y1), (x1 + text_w, y1 - text_h - 4), color, -1)
                cv2.putText(img_numpy, text_str, text_pt, font_face, font_scale, text_color, font_thickness, cv2.LINE_AA)
    
    return img_numpy



def evalimage(net,frame):

    batch = FastBaseTransform()
    batch = batch.forward(frame)

    pred_onx = net.run(batch)

    detect = Detect(cfg.num_classes, bkg_label=0,
                    top_k=200, conf_thresh=0.05, nms_thresh=0.5)
    preds = detect({'loc': pred_onx[0], 'conf': pred_onx[1], 'mask': pred_onx[2], 'priors': pred_onx[3], 'proto': pred_onx[4]})
    frame = prep_display(preds, frame, None, None).astype(np.uint8)

    return frame


def recognize_from_image(net):

    for image_path in args.input:
        # prepare input data
        logger.debug(f'input image: {image_path}')
        raw_img = imread(image_path)
        print(image_path)
        logger.debug(f'input image shape: {raw_img.shape}')

        # inference
        logger.info('Start inference...')
        if args.benchmark:
            logger.info('BENCHMARK mode')
            for i in range(5):
                start = int(round(time.time() * 1000))

                frame = imread(image_path)
                frame = evalimage(net,frame)
    
                end = int(round(time.time() * 1000))
                logger.info(f'\tailia processing time {end - start} ms')
        else:
            frame = imread(image_path)
            frame = evalimage(net,frame)

        savepath = get_savepath(args.savepath, image_path)
        logger.info(f'saved at : {savepath}')
        cv2.imwrite(savepath,frame)

    logger.info('Script finished successfully.')



def recognize_from_video(net):
    capture = webcamera_utils.get_capture(args.video)

    # create video writer if savepath is specified as video format
    if args.savepath != SAVE_IMAGE_PATH:
        logger.warning(
            'currently, video results cannot be output correctly...'
        )
        writer = webcamera_utils.get_writer(args.savepath, HEIGHT*2, WIDTH)
    else:
        writer = None

    frame_shown = False
    while (True):
        ret, frame = capture.read()
        if (cv2.waitKey(1) & 0xFF == ord('q')) or not ret:
            break
        if frame_shown and cv2.getWindowProperty('output', cv2.WND_PROP_VISIBLE) == 0:
            break

        frame = evalimage(net,frame)

        cv2.imshow('output', frame)
        frame_shown = True

        # save results
        if writer is not None:
            writer.write(output_buffer)

    capture.release()
    cv2.destroyAllWindows()
    if writer is not None:
        writer.release()
    logger.info('Script finished successfully.')

if __name__ == '__main__':
    check_and_download_models(WEIGHT_PATH, MODEL_PATH, REMOTE_PATH)
    net = ailia.Net(None,WEIGHT_PATH)
    if args.video is not None:
        # video mode
        recognize_from_video(net)
    else:
        # image mode
        recognize_from_image(net)

