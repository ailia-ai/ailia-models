import numpy as np
import torch
from plyfile import PlyData, PlyElement
from c_diff_gaussian_rasterization import GaussianRasterizationSettings, GaussianRasterizer

class GaussianRenderer:
    def __init__(self, opt):
        self.opt = opt
        self.bg_color = torch.tensor([1, 1, 1], dtype=torch.float32)
        self.tan_half_fov = np.tan(0.5 * np.deg2rad(self.opt.fovy))

    def render(self, gaussians, cam_view, cam_view_proj, cam_pos, bg_color=None, scale_modifier=1):
        gaussians = torch.from_numpy(gaussians).to(dtype=torch.float32)
        cam_view = torch.from_numpy(cam_view).to(dtype=torch.float32)
        cam_view_proj = torch.from_numpy(cam_view_proj).to(dtype=torch.float32)
        cam_pos = torch.from_numpy(cam_pos).to(dtype=torch.float32)
        
        B, V = cam_view.shape[:2]
        images = []

        for b in range(B):
            means3D = gaussians[b, :, 0:3].contiguous().float()
            opacity = gaussians[b, :, 3:4].contiguous().float()
            scales = gaussians[b, :, 4:7].contiguous().float()
            rotations = gaussians[b, :, 7:11].contiguous().float()
            rgbs = gaussians[b, :, 11:].contiguous().float()

            for v in range(V):
                view_matrix = cam_view[b, v].float()
                view_proj_matrix = cam_view_proj[b, v].float()
                campos = cam_pos[b, v].float()

                raster_settings = GaussianRasterizationSettings(
                    image_height=self.opt.output_size,
                    image_width=self.opt.output_size,
                    tanfovx=self.tan_half_fov,
                    tanfovy=self.tan_half_fov,
                    bg=self.bg_color if bg_color is None else bg_color,
                    scale_modifier=scale_modifier,
                    viewmatrix=view_matrix,
                    projmatrix=view_proj_matrix,
                    sh_degree=0,
                    campos=campos,
                    prefiltered=False,
                    debug=False,
                )

                rasterizer = GaussianRasterizer(raster_settings=raster_settings)

                rendered_image, radii = rasterizer(
                    means3D=means3D,
                    means2D=torch.zeros_like(means3D, dtype=torch.float32),
                    shs=None,
                    colors_precomp=rgbs,
                    opacities=opacity,
                    scales=scales,
                    rotations=rotations,
                    cov3D_precomp=None,
                )

                rendered_image = rendered_image.clamp(0, 1)
                images.append(rendered_image)

        images = torch.stack(images, dim=0).view(B, V, 3, self.opt.output_size, self.opt.output_size)
        return images

def save_ply(gaussians, path, compatible=True):
    # gaussians: [B, N, 14]
    # compatible: save pre-activated gaussians as in the original paper
    assert gaussians.shape[0] == 1, 'only support batch size 1'

    gaussians = torch.from_numpy(gaussians).to(dtype=torch.float32)

    means3D = gaussians[0, :, 0:3].contiguous().float()
    opacity = gaussians[0, :, 3:4].contiguous().float()
    scales = gaussians[0, :, 4:7].contiguous().float()
    rotations = gaussians[0, :, 7:11].contiguous().float()
    shs = gaussians[0, :, 11:].unsqueeze(1).contiguous().float() # [N, 1, 3]

    # prune by opacity
    mask = opacity.squeeze(-1) >= 0.005
    means3D = means3D[mask]
    opacity = opacity[mask]
    scales = scales[mask]
    rotations = rotations[mask]
    shs = shs[mask]

    # invert activation to make it compatible with the original ply format
    if compatible:
        opacity = inverse_sigmoid(opacity)
        scales = torch.log(scales + 1e-8)
        shs = (shs - 0.5) / 0.28209479177387814

    xyzs = means3D.detach().cpu().numpy()
    f_dc = shs.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
    opacities = opacity.detach().cpu().numpy()
    scales = scales.detach().cpu().numpy()
    rotations = rotations.detach().cpu().numpy()

    l = ['x', 'y', 'z']
    # All channels except the 3 DC
    for i in range(f_dc.shape[1]):
        l.append('f_dc_{}'.format(i))
    l.append('opacity')
    for i in range(scales.shape[1]):
        l.append('scale_{}'.format(i))
    for i in range(rotations.shape[1]):
        l.append('rot_{}'.format(i))

    dtype_full = [(attribute, 'f4') for attribute in l]

    elements = np.empty(xyzs.shape[0], dtype=dtype_full)
    attributes = np.concatenate((xyzs, f_dc, opacities, scales, rotations), axis=1)
    elements[:] = list(map(tuple, attributes))
    el = PlyElement.describe(elements, 'vertex')

    PlyData([el]).write(path)

def inverse_sigmoid(x, eps=1e-6):
    x = x.clamp(eps, 1 - eps)
    return torch.log(x / (1 - x))
