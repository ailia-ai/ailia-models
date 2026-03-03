import numpy as np
import cv2
import json

__all__ = [
    'decode_batch',
    'mask_to_bboxes',
    'draw_bbox',
    'save_bboxes_to_json',
]


def decode_batch(
        pixel_cls_scores, pixel_link_scores,
        pixel_conf_threshold=0.8, link_conf_threshold=0.8):
    batch_size = pixel_cls_scores.shape[0]
    batch_mask = []
    for image_idx in range(batch_size):
        image_pos_pixel_scores = pixel_cls_scores[image_idx, :, :]
        image_pos_link_scores = pixel_link_scores[image_idx, :, :, :]
        mask = decode_image(
            image_pos_pixel_scores, image_pos_link_scores,
            pixel_conf_threshold, link_conf_threshold
        )
        batch_mask.append(mask)
    return np.asarray(batch_mask, np.int32)


def get_neighbours(x, y):
    return [
        (x - 1, y - 1), (x, y - 1), (x + 1, y - 1), (x - 1, y), \
        (x + 1, y), (x - 1, y + 1), (x, y + 1), (x + 1, y + 1)
    ]


def is_valid_cord(x, y, w, h):
    return x >= 0 and x < w and y >= 0 and y < h


def decode_image(
        pixel_scores, link_scores,
        pixel_conf_threshold, link_conf_threshold):
    #
    pixel_mask = pixel_scores >= pixel_conf_threshold
    link_mask = link_scores >= link_conf_threshold
    done_mask = np.zeros(pixel_mask.shape, bool)
    result_mask = np.zeros(pixel_mask.shape, np.int32)
    points = list(zip(*np.where(pixel_mask)))
    h, w = np.shape(pixel_mask)
    group_id = 0
    for point in points:
        if done_mask[point]:
            continue
        group_id += 1
        group_q = [point]
        result_mask[point] = group_id
        while len(group_q):
            y, x = group_q[-1]
            group_q.pop()
            if not done_mask[y, x]:
                done_mask[y, x], result_mask[y, x] = True, group_id
                for n_idx, (nx, ny) in enumerate(get_neighbours(x, y)):
                    if is_valid_cord(nx, ny, w, h) and pixel_mask[ny, nx] and (
                            link_mask[y, x, n_idx] or link_mask[ny, nx, 7 - n_idx]):
                        group_q.append((ny, nx))
    return result_mask


def find_contours(mask, method=None):
    if method is None:
        method = cv2.CHAIN_APPROX_SIMPLE
    mask = np.asarray(mask, dtype=np.uint8)
    mask = mask.copy()
    try:
        contours, _ = cv2.findContours(mask, mode=cv2.RETR_CCOMP,
                                       method=method)
    except:
        _, contours, _ = cv2.findContours(mask, mode=cv2.RETR_CCOMP,
                                          method=method)
    return contours


def min_area_rect(cnt):
    """
    Args:
        xs: numpy ndarray with shape=(N,4). N is the number of oriented bboxes. 4 contains [x1, x2, x3, x4]
        ys: numpy ndarray with shape=(N,4), [y1, y2, y3, y4]
            Note that [(x1, y1), (x2, y2), (x3, y3), (x4, y4)] can represent an oriented bbox.
    Return:
        the oriented rects sorrounding the box, in the format:[cx, cy, w, h, theta].
    """
    rect = cv2.minAreaRect(cnt)
    cx, cy = rect[0]
    w, h = rect[1]
    theta = rect[2]
    box = [cx, cy, w, h, theta]
    return box, w * h


def rect_to_xys(rect, image_shape):
    """Convert rect to xys, i.e., eight points
    The `image_shape` is used to to make sure all points return are valid, i.e., within image area
    """
    h, w = image_shape[0:2]

    def get_valid_x(x):
        if x < 0:
            return 0
        if x >= w:
            return w - 1
        return x

    def get_valid_y(y):
        if y < 0:
            return 0
        if y >= h:
            return h - 1
        return y

    rect = ((rect[0], rect[1]), (rect[2], rect[3]), rect[4])
    points = cv2.boxPoints(rect)
    points = np.intp(points)
    for i_xy, (x, y) in enumerate(points):
        x = get_valid_x(x)
        y = get_valid_y(y)
        points[i_xy, :] = [x, y]
    points = np.reshape(points, -1)
    return points


def mask_to_bboxes(
        mask, image_shape, min_area=300,
        min_height=10):
    image_h, image_w = image_shape[0:2]
    bboxes = []
    max_bbox_idx = mask.max()
    mask = cv2.resize(mask, (image_w, image_h),
                      interpolation=cv2.INTER_NEAREST)

    for bbox_idx in range(1, max_bbox_idx + 1):
        bbox_mask = mask == bbox_idx
        cnts = find_contours(bbox_mask)
        if len(cnts) == 0:
            continue
        cnt = cnts[0]
        rect, rect_area = min_area_rect(cnt)

        w, h = rect[2:-1]
        if min(w, h) < min_height:
            continue

        if rect_area < min_area:
            continue

        xys = rect_to_xys(rect, image_shape)
        bboxes.append(xys)

    return bboxes


def points_to_contour(points):
    contours = [[list(p)] for p in points]
    return np.asarray(contours, dtype=np.int32)


def points_to_contours(points):
    return np.asarray([points_to_contour(points)])


def draw_contours(img, contours, idx=-1, color=1, border_width=1):
    cv2.drawContours(img, contours, idx, color, border_width)
    return img


def draw_bbox(img, bboxes):
    COLOR_GREEN = (0, 255, 0)

    for bbox in bboxes:
        points = [int(v) for v in bbox]
        points = np.reshape(points, (4, 2))
        cnts = points_to_contours(points)
        cv2.drawContours(img, cnts, -1, COLOR_GREEN, thickness=3)
    return img

def save_bboxes_to_json(json_path, bboxes):
    b = [arr.tolist() for arr in bboxes]
    with open(json_path, 'w') as f:
        json.dump({"bboxes": b}, f, indent=2)
