import os
import sys
import time

import numpy as np
import cv2
import scipy.io as sio
from skimage.io import imread, imsave
from skimage.transform import resize

import ailia

# import original modules
sys.path.append('../../util')
from arg_utils import get_base_parser, update_parser  # noqa: E402
from model_utils import check_and_download_models  # noqa: E402
from image_utils import load_image  # noqa: E402
import webcamera_utils  # noqa: E402

from prnet_utils.net_utils import *  # noqa: E402
from prnet_utils.estimate_pose import estimate_pose  # noqa: E402
from prnet_utils.rotate_vertices import frontalize  # noqa: E402
from prnet_utils.render_app import get_visibility, get_uv_mask, get_depth_image  # noqa: E402
from prnet_utils.write import write_obj_with_colors, write_obj_with_texture  # noqa: E402
from prnet_utils.cv_plot import plot_kpt, plot_vertices, plot_pose_box  # noqa: E402
from prnet_utils.render import render_texture  # noqa: E402

# logger
from logging import getLogger   # noqa: E402
logger = getLogger(__name__)


# ======================
# Parameters
# ======================
WEIGHT_PATH = 'prnet.onnx'
MODEL_PATH = 'prnet.onnx.prototxt'
REMOTE_PATH = 'https://storage.googleapis.com/ailia-models/prnet/'

IMAGE_PATH = 'image00430-cropped.jpg'
SAVE_FOLDER = 'results'

# NOTE: used only for texture editing mode
REF_IMAGE_PATH = 'uv-data/trump_cropped.png'
UV_FACE_PATH = 'uv-data/uv_face.png'
UV_FACE_EYES_PATH = 'uv-data/uv_face_eyes.png'


# NOTE: In the original repository, "resolution of input and output image size"
#       can be specified separately (though the both size are fixed 256)
IMAGE_SIZE = 256

# ntri x 3
TRIANGLES = np.loadtxt('uv-data/triangles.txt').astype(np.int32)

UV_COORDS = generate_uv_coords(IMAGE_SIZE)


# ======================
# Arguemnt Parser Config
# ======================
parser = get_base_parser('PR-Net', IMAGE_PATH, SAVE_FOLDER)

# texture editing mode configuration
parser.add_argument(
    '-t', '--texture', metavar='MODE', type=int, default=-1,
    help='Ways to edit texture. 0 for modifying parts (eyes in this ex.), ' +
         '1 for changing whole, -1 for normal recognition mode'
)
parser.add_argument(
    '-r', '--refpath', metavar='IMAGE',
    default=REF_IMAGE_PATH,
    help='The path to the texture reference image. ' +
         'This image will be used only for texture editing mode.'
)


# original repository argument
parser.add_argument(
    '--is3d', action='store_false',
    help='whether to output 3D face(.obj). default save colors.'
)
parser.add_argument(
    '--isMat', action='store_true',
    help='whether to save vertices,color,triangles as mat for matlab showing'
)
parser.add_argument(
    '--isKpt', action='store_true',
    help='whether to output key points(.txt)'
)
parser.add_argument(
    '--isPose', action='store_true',
    help='whether to output estimated pose(.txt)'
)
parser.add_argument(
    '--isShow', action='store_true',
    help=('whether to show the results with opencv(need opencv) instead of '
          'saving them')
)
parser.add_argument(
    '--isFront', action='store_true',
    help='whether to frontalize vertices(mesh)'
)
parser.add_argument(
    '--isDepth', action='store_true',
    help='whether to output depth image'
)
parser.add_argument(
    '--isTexture', action='store_true',
    help='whether to save texture in obj file'
)
parser.add_argument(
    '--isMask', action='store_true',
    help=('whether to set invisible pixels(due to self-occlusion) in texture '
          'as 0')
)
parser.add_argument(
    '--texture_size', default=256, type=int,
    help='size of texture map, default is 256. need isTexture is True'
)

args = update_parser(parser)


# ======================
# Main functions
# ======================
def recognize_from_image():
    # net initialize
    net = ailia.Net(MODEL_PATH, WEIGHT_PATH, env_id=args.env_id, memory_mode=args.memory_mode)
    net.set_input_shape((1, 256, 256, 3))

    # input image loop
    for image_path in args.input:
        # prepare input data
        logger.info(image_path)
        savepath = os.path.join(
            args.savepath, os.path.splitext(os.path.basename(image_path))[0]
        )
        image = load_image(
            image_path,
            (IMAGE_SIZE, IMAGE_SIZE),
            normalize_type='255',
            gen_input_ailia=False,
        )

        # for now, h = w = IMAGE_SIZE (as we resized the input when loading it)
        h, w = image.shape[0], image.shape[1]
        input_data = image[np.newaxis, :, :, :]

        # inference
        logger.info('Start inference...')
        if args.benchmark:
            logger.info('BENCHMARK mode')
            for i in range(5):
                start = int(round(time.time() * 1000))
                preds_ailia = net.predict(input_data)
                end = int(round(time.time() * 1000))
                logger.info(f'\tailia processing time {end - start} ms')
        else:
            preds_ailia = net.predict(input_data)

        # post-processing
        # INFO self.MaxPos
        pos = preds_ailia[0] * IMAGE_SIZE * 1.1

        if args.is3d or args.isMat or args.isPose:
            # 3D vertices
            vertices = get_vertices(pos, IMAGE_SIZE)
            if args.isFront:
                save_vertices = frontalize(vertices)
            else:
                save_vertices = vertices.copy()
            save_vertices[:, 1] = h - 1 - save_vertices[:, 1]

        if args.is3d:
            # corresponding colors
            colors = get_colors(image, vertices)

            if args.isTexture:
                if args.texture_size != 256:
                    pos_interpolated = resize(
                        pos,
                        (args.texture_size, args.texture_size),
                        preserve_range=True
                    )
                else:
                    pos_interpolated = pos.copy()

                texture = cv2.remap(
                    image,
                    pos_interpolated[:, :, :2].astype(np.float32),
                    None,
                    interpolation=cv2.INTER_LINEAR,
                    borderMode=cv2.BORDER_CONSTANT,
                    borderValue=(0)
                )
                if args.isMask:
                    vertices_vis = get_visibility(vertices, TRIANGLES, h, w)
                    uv_mask = get_uv_mask(
                        vertices_vis,
                        TRIANGLES,
                        UV_COORDS,
                        h,
                        w,
                        IMAGE_SIZE
                    )
                    uv_mask = resize(
                        uv_mask,
                        (args.texture_size, args.texture_size),
                        preserve_range=True
                    )
                    texture = texture * uv_mask[:, :, np.newaxis]
                    # save 3d face with texture(can open with meshlab)
                    write_obj_with_texture(
                        savepath + '.obj',
                        save_vertices,
                        TRIANGLES,
                        texture,
                        UV_COORDS/IMAGE_SIZE
                    )
            else:
                # save 3d face(can open with meshlab)
                write_obj_with_colors(
                    savepath + '.obj',
                    save_vertices,
                    TRIANGLES,
                    colors
                )

        if args.isDepth:
            depth_image = get_depth_image(vertices, TRIANGLES, h, w, True)
            depth = get_depth_image(vertices, TRIANGLES, h, w)
            imsave(savepath + '_depth.jpg', depth_image)
            sio.savemat(savepath + '_depth.mat', {'depth': depth})

        if args.isMat:
            sio.savemat(
                savepath + '_mesh.mat',
                {
                    'vertices': vertices,
                    'colors': colors,
                    'triangles': TRIANGLES,
                }
            )

        if args.isKpt:
            # get landmarks
            kpt = get_landmarks(pos)
            np.savetxt(savepath + '_kpt.txt', kpt)

        if args.isPose:
            # estimate pose
            camera_matrix, pose = estimate_pose(vertices)
            np.savetxt(savepath + '_pose.txt', pose)
            np.savetxt(savepath + '_camera_matrix.txt', camera_matrix)
            np.savetxt(savepath + '_pose.txt', pose)

        image = cv2.cvtColor(image.astype(np.float32), cv2.COLOR_RGB2BGR)
        if args.isShow:
            if args.isKpt:
                cv2.imshow('sparse alignment', plot_kpt(image, kpt))
            if args.is3d or args.isMat or args.isPose:
                cv2.imshow('dense alignment', plot_vertices(image, vertices))
            if args.isPose:
                cv2.imshow('pose', plot_pose_box(image, camera_matrix, kpt))
            cv2.waitKey(0)
        else:
            image = np.clip((image * 255), 0, 255)
            if args.isKpt:
                cv2.imwrite(
                    savepath + '_sparse_alignment.png',
                    plot_kpt(image, kpt).astype(np.uint8)
                )
            if args.is3d or args.isMat or args.isPose:
                cv2.imwrite(
                    savepath + '_dense_alignment.png',
                    plot_vertices(image, vertices).astype(np.uint8)
                )
            if args.isPose:
                cv2.imwrite(
                    savepath + '_pose.png',
                    plot_pose_box(image, camera_matrix, kpt).astype(np.uint8)
                )

    logger.info('Script finished successfully.')


def texture_editing_from_images():
    # net initialize
    net = ailia.Net(MODEL_PATH, WEIGHT_PATH, env_id=args.env_id, memory_mode=args.memory_mode)
    net.set_input_shape((1, 256, 256, 3))

    # input image loop
    for image_path in args.input:
        # prepare input data
        logger.info(image_path)
        savepath = os.path.join(
            args.savepath, os.path.splitext(os.path.basename(image_path))[0]
        )
        image = load_image(
            image_path,
            (IMAGE_SIZE, IMAGE_SIZE),
            normalize_type='255',
            gen_input_ailia=False,
        )

        # for now, h = w = IMAGE_SIZE (as we resized the input when loading it)
        h, w = image.shape[0], image.shape[1]
        input_data = image[np.newaxis, :, :, :]

        # inference
        # 1. 3d reconstruction --> get texture
        pos = net.predict(input_data)[0] * IMAGE_SIZE * 1.1
        vertices = get_vertices(pos, IMAGE_SIZE)
        texture = cv2.remap(
            image,
            pos[:, :, :2].astype(np.float32),
            None,
            interpolation=cv2.INTER_NEAREST,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=(0)
        )

        # 2. texture editing
        MODE = args.texture

        ref_image = load_image(
            args.refpath,
            (IMAGE_SIZE, IMAGE_SIZE),
            normalize_type='255',
            gen_input_ailia=False
        )
        input_data = ref_image[np.newaxis, :, :, :]
        ref_pos = net.predict(input_data)[0] * IMAGE_SIZE * 1.1

        # texture from another image or a processed texture
        ref_texture = cv2.remap(
            ref_image,
            ref_pos[:, :, :2].astype(np.float32),
            None,
            interpolation=cv2.INTER_NEAREST,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=(0)
        )

        # change part of texture (here, modify eyes as example)
        if MODE == 0:
            # load eye mask
            uv_face_eye = imread(UV_FACE_EYES_PATH, as_grey=True) / 255.
            uv_face = imread(UV_FACE_PATH, as_grey=True) / 255.
            eye_mask = (abs(uv_face_eye - uv_face) > 0).astype(np.float32)

            # modify texture
            new_texture = texture * \
                (1 - eye_mask[:, :, np.newaxis]) + \
                ref_texture*eye_mask[:, :, np.newaxis]

        # change whole face(face swap)
        elif MODE == 1:
            # ref_vertices = get_vertices(ref_pos, IMAGE_SIZE)
            new_texture = ref_texture  # (texture + ref_texture)/2.

        else:
            logger.error('Wrong Mode! Mode should be 0 or 1.')
            exit()

        # 3. remap to input image (render).
        vis_colors = np.ones((vertices.shape[0], 1))
        face_mask = render_texture(
            vertices.T, vis_colors.T, TRIANGLES.T, h, w, c=1
        )
        face_mask = np.squeeze(face_mask > 0).astype(np.float32)

        new_colors = get_colors_from_texture(new_texture, IMAGE_SIZE)
        new_image = render_texture(
            vertices.T, new_colors.T, TRIANGLES.T, h, w, c=3
        )
        new_image = image * (1 - face_mask[:, :, np.newaxis]) + \
            new_image * face_mask[:, :, np.newaxis]

        # Possion Editing for blending image
        vis_ind = np.argwhere(face_mask > 0)
        vis_min = np.min(vis_ind, 0)
        vis_max = np.max(vis_ind, 0)
        center = (
            int((vis_min[1] + vis_max[1])/2+0.5),
            int((vis_min[0] + vis_max[0])/2+0.5)
        )
        output = cv2.seamlessClone(
            (new_image*255).astype(np.uint8),
            (image*255).astype(np.uint8),
            (face_mask*255).astype(np.uint8),
            center,
            cv2.NORMAL_CLONE
        )

        # save output
        imsave(savepath + '_texture_edited.png', output)
    logger.info('Script finished successfully.')


def recognize_from_video():
    raise NotImplementedError
    """
    # net initialize
    net = ailia.Net(MODEL_PATH, WEIGHT_PATH, env_id=args.env_id, memory_mode=args.memory_mode)

    capture = get_capture(args.video)

    # create video writer if savepath is specified as video format
    if args.savepath != SAVE_IMAGE_PATH:
        f_h = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        f_w = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        save_h, save_w = webcamera_utils.calc_adjust_fsize(
            f_h, f_w, ailia_input_h, ailia_input_w
        )
        writer = webcamera_utils.get_writer(args.savepath, save_h, save_w)
    else:
        writer = None

    frame_shown = False
    while(True):
        ret, frame = capture.read()
        if (cv2.waitKey(1) & 0xFF == ord('q')) or not ret:
            break
        if frame_shown and cv2.getWindowProperty('frame', cv2.WND_PROP_VISIBLE) == 0:
            break
        
        input_image, input_data = webcamera_utils.preprocess_frame(
            frame, IMAGE_SIZE, IMAGE_SIZE, normalize_type='127.5'
        )
        # ???

        # inference
        # 1.
        preds_ailia = net.predict(input_data)

        # 2.
        input_blobs = net.get_input_blob_list()
        net.set_input_blob_data(input_data, input_blobs[0])
        net.update()
        preds_ailia = net.get_results()

        # postprocessing
        # ???
        cv2.imshow('frame', input_image)
        frame_shown = True

        # save results
        if writer is not None:
            writer.write(seg_image)

    capture.release()
    cv2.destroyAllWindows()
    if writer is not None:
        writer.release()
    logger.info('Script finished successfully.')
    """


def main():
    # model files check and download
    check_and_download_models(WEIGHT_PATH, MODEL_PATH, REMOTE_PATH)

    # make saved data directory
    logger.info(f'Make ./{args.savepath} directory if it does not exist')
    os.makedirs(args.savepath, exist_ok=True)

    if args.video is not None:
        # video mode
        recognize_from_video()
    else:
        # image mode
        if args.texture == -1:
            recognize_from_image()
        else:
            texture_editing_from_images()


if __name__ == '__main__':
    main()
