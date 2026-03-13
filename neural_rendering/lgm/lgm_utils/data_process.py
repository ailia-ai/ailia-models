import numpy as np
import cv2
import rembg
from lgm_utils.kiui import orbit_camera, safe_normalize, recenter

IMAGENET_DEFAULT_MEAN = (0.485, 0.456, 0.406)
IMAGENET_DEFAULT_STD = (0.229, 0.224, 0.225)

def get_rays(pose, h, w, fovy, opengl=True):
    x, y = np.meshgrid(
        np.arange(h),
        np.arange(w),
        indexing="xy"
    )
    x = x.flatten()
    y = y.flatten()

    cx = w * 0.5
    cy = h * 0.5

    focal = h * 0.5 / np.tan(0.5 * np.deg2rad(fovy))

    camera_dirs = np.pad(
        np.stack([
            (x - cx + 0.5) / focal,
            (y - cy + 0.5) / focal * (-1.0 if opengl else 1.0),
        ], axis=-1),
        ((0, 0), (0, 1)),
        constant_values=(-1.0 if opengl else 1.0)
    )  # Shape: [hw, 3]
    
    rays_d = camera_dirs @ pose[:3, :3].T  # [hw, 3]
    rays_o = np.broadcast_to(pose[:3, 3], rays_d.shape)  # [hw, 3]

    rays_o = rays_o.reshape(h, w, 3)
    rays_d = safe_normalize(rays_d).reshape(h, w, 3)

    return rays_o, rays_d

def prepare_default_rays(fovy, cam_radius, input_size, elevation=0):
    # Compute camera poses
    cam_poses = np.stack([
        orbit_camera(elevation, 0, radius=cam_radius),
        orbit_camera(elevation, 90, radius=cam_radius),
        orbit_camera(elevation, 180, radius=cam_radius),
        orbit_camera(elevation, 270, radius=cam_radius),
    ], axis=0)  # Shape: [4, 4, 4]

    rays_embeddings = []
    for i in range(cam_poses.shape[0]):
        rays_o, rays_d = get_rays(
            cam_poses[i], input_size, input_size, fovy
        )  # Shape: [h, w, 3]
        
        # Compute Plücker coordinates for the rays
        rays_plucker = np.concatenate([np.cross(rays_o, rays_d, axis=-1), rays_d], axis=-1)  # Shape: [h, w, 6]
        rays_embeddings.append(rays_plucker)

    # Stack and reshape ray embeddings to match the desired format
    rays_embeddings = np.stack(rays_embeddings, axis=0)  # Shape: [V, h, w, 6]
    rays_embeddings = np.transpose(rays_embeddings, (0, 3, 1, 2))  # Shape: [V, 6, h, w]
    
    return rays_embeddings

def process_mv_image(image, bg_remover):
    # bg removal
    removed_image = rembg.remove(image, session=bg_remover) # [H, W, 4]
    mask = removed_image[..., -1] > 0

    # recenter
    image = recenter(removed_image, mask, border_ratio=0.2)
    image = image.astype(np.float32) / 255.0

    # rgba to rgb white bg
    if image.shape[-1] == 4:
        image = image[..., :3] * image[..., 3:4] + (1 - image[..., 3:4])

    return image

def resize_images(input_image, target_size):
    resized = np.zeros((input_image.shape[0], input_image.shape[1], target_size, target_size), dtype=np.float32)
    for i in range(input_image.shape[0]):
        for c in range(input_image.shape[1]):
            resized[i, c] = cv2.resize(input_image[i, c], (target_size, target_size), interpolation=cv2.INTER_LINEAR)
    return resized

def normalize_images(input_image, mean, std):
    mean = np.array(mean, dtype=np.float32)[:, None, None]
    std = np.array(std, dtype=np.float32)[:, None, None]
    input_image = (input_image - mean[None, :, :, :]) / std[None, :, :, :]
    return input_image

def preprocess_lgm_model(mv_image, opt):
    rays_embeddings = prepare_default_rays(opt.fovy, opt.cam_radius, opt.input_size).astype(np.float32)

    image = mv_image.transpose((0, 3, 1, 2)).astype(np.float32)
    image = resize_images(image, opt.input_size)
    image = normalize_images(image, IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD)

    image = np.concatenate([image, rays_embeddings], axis=1).astype(np.float16) # [1, 4, 9, H, W]

    return image
