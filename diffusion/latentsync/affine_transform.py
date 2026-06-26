import cv2
import numpy as np


def transformation_from_points(points1, points0, smooth=True, p_bias=None):
    points2 = np.array(points0)
    points2 = points2.astype(np.float64)
    points1 = points1.astype(np.float64)
    c1 = np.mean(points1, axis=0)
    c2 = np.mean(points2, axis=0)
    points1 -= c1
    points2 -= c2
    s1 = np.std(points1)
    s2 = np.std(points2)
    points1 /= s1
    points2 /= s2
    U, S, Vt = np.linalg.svd(np.matmul(points1.T, points2))
    R = (np.matmul(U, Vt)).T
    sR = (s2 / s1) * R
    T = c2.reshape(2, 1) - (s2 / s1) * np.matmul(R, c1.reshape(2, 1))
    M = np.concatenate((sR, T), axis=1)
    if smooth:
        bias = points2[2] - points1[2]
        if p_bias is None:
            p_bias = bias
        else:
            bias = p_bias * 0.2 + bias * 0.8
        p_bias = bias
        M[:, 2] = M[:, 2] + bias
    return M, p_bias


class AlignRestore(object):
    def __init__(self):
        self.upscale_factor = 1
        ratio = 2.8
        self.crop_ratio = (ratio, ratio)
        self.face_template = np.array(
            [[19 - 2, 30 - 10], [56 + 2, 30 - 10], [37.5, 45 - 5]]
        )
        self.face_template = self.face_template * ratio
        self.face_size = (
            int(75 * self.crop_ratio[0]),
            int(100 * self.crop_ratio[1]),
        )
        self.p_bias = None

    def align_warp_face(self, img, lmks3, smooth=True):
        affine_matrix, self.p_bias = transformation_from_points(
            lmks3, self.face_template, smooth, self.p_bias
        )

        cropped_face = cv2.warpAffine(
            img,
            affine_matrix,
            self.face_size,
            flags=cv2.INTER_LANCZOS4,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=[127, 127, 127],
        )
        return cropped_face, affine_matrix

    def restore_img(self, input_img, face, affine_matrix):
        h, w, _ = input_img.shape
        h_up, w_up = int(h * self.upscale_factor), int(w * self.upscale_factor)
        upsample_img = cv2.resize(
            input_img, (w_up, h_up), interpolation=cv2.INTER_LANCZOS4
        )
        inverse_affine = cv2.invertAffineTransform(affine_matrix)
        inverse_affine *= self.upscale_factor
        if self.upscale_factor > 1:
            extra_offset = 0.5 * self.upscale_factor
        else:
            extra_offset = 0
        inverse_affine[:, 2] += extra_offset
        inv_restored = cv2.warpAffine(
            face, inverse_affine, (w_up, h_up), flags=cv2.INTER_LANCZOS4
        )
        mask = np.ones((self.face_size[1], self.face_size[0]), dtype=np.float32)
        inv_mask = cv2.warpAffine(mask, inverse_affine, (w_up, h_up))
        inv_mask_erosion = cv2.erode(
            inv_mask,
            np.ones(
                (int(2 * self.upscale_factor), int(2 * self.upscale_factor)), np.uint8
            ),
        )
        pasted_face = inv_mask_erosion[:, :, None] * inv_restored
        total_face_area = np.sum(inv_mask_erosion)
        w_edge = int(total_face_area**0.5) // 20
        erosion_radius = w_edge * 2
        inv_mask_center = cv2.erode(
            inv_mask_erosion, np.ones((erosion_radius, erosion_radius), np.uint8)
        )
        blur_size = w_edge * 2
        inv_soft_mask = cv2.GaussianBlur(
            inv_mask_center, (blur_size + 1, blur_size + 1), 0
        )
        inv_soft_mask = inv_soft_mask[:, :, None]
        upsample_img = inv_soft_mask * pasted_face + (1 - inv_soft_mask) * upsample_img
        if np.max(upsample_img) > 256:
            upsample_img = upsample_img.astype(np.uint16)
        else:
            upsample_img = upsample_img.astype(np.uint8)

        return upsample_img


class laplacianSmooth:
    def __init__(self, smoothAlpha=0.3):
        self.smoothAlpha = smoothAlpha
        self.pts_last = None

    def smooth(self, pts_cur):
        if self.pts_last is None:
            self.pts_last = pts_cur.copy()
            return pts_cur.copy()

        x1 = min(pts_cur[:, 0])
        x2 = max(pts_cur[:, 0])
        # y1 = min(pts_cur[:, 1])
        # y2 = max(pts_cur[:, 1])

        width = x2 - x1
        pts_update = []
        for i in range(len(pts_cur)):
            x_new, y_new = pts_cur[i]
            x_old, y_old = self.pts_last[i]
            tmp = (x_new - x_old) ** 2 + (y_new - y_old) ** 2
            w = np.exp(-tmp / (width * self.smoothAlpha))
            x = x_old * w + x_new * (1 - w)
            y = y_old * w + y_new * (1 - w)
            pts_update.append([x, y])

        pts_update = np.array(pts_update)
        self.pts_last = pts_update.copy()

        return pts_update
