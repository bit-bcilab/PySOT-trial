

import numpy as np
from collections import namedtuple
from trial.utils.iou import box_iou

Center = namedtuple('Center', 'x y w h')
Corner = namedtuple('Corner', 'x1 y1 x2 y2')
BBox = Corner


def clip_bbox_center(box_xywh, boundary):
    if len(box_xywh.shape) > 1:
        cx = np.maximum(0., np.minimum(box_xywh[..., 0], boundary[1]))
        cy = np.maximum(0., np.minimum(box_xywh[..., 1], boundary[0]))
        width = np.maximum(10., np.minimum(box_xywh[..., 2], boundary[1]))
        height = np.maximum(10., np.minimum(box_xywh[..., 3], boundary[0]))
        return np.stack([cx, cy, width, height], axis=-1)
    else:
        cx = np.maximum(0., np.minimum(box_xywh[0], boundary[1]))
        cy = np.maximum(0., np.minimum(box_xywh[1], boundary[0]))
        width = np.maximum(10., np.minimum(box_xywh[2], boundary[1]))
        height = np.maximum(10., np.minimum(box_xywh[3], boundary[0]))
        return np.stack([cx, cy, width, height], axis=-1)


def clip_bbox_corner(box, boundary, min_rate=None):
    if len(box.shape) > 1:
        x1 = np.maximum(0., box[..., 0])
        y1 = np.maximum(0., box[..., 1])
        x2 = np.minimum(box[..., 2], boundary[1])
        y2 = np.minimum(box[..., 3], boundary[0])
        return np.stack([x1, y1, x2, y2], axis=-1)
    else:
        x1 = np.maximum(0., box[0])
        y1 = np.maximum(0., box[1])
        x2 = np.minimum(box[2], boundary[1])
        y2 = np.minimum(box[3], boundary[0])
        return np.stack([x1, y1, x2, y2], axis=-1)


def box2roi(bbox, rate_w, rate_h, boundary=None):
    bbox_xywh = corner2center(bbox)
    roi_xywh = bbox_xywh * np.array([1., 1., rate_w, rate_h], np.float32)
    roi = center2corner(roi_xywh)
    if boundary is not None:
        roi = clip_bbox_corner(roi, boundary)
    return roi


def corner2center(corner):
    """
    convert (x1, y1, x2, y2) to (cx, cy, w, h)

    Args:
        corner: Corner / np.array (..., 4) / np.array (4)
    Return:
        Center / np.array (..., 4)  / np.array (4)
    """
    if isinstance(corner, Corner):
        x1, y1, x2, y2 = corner
        return Center((x1 + x2) * 0.5, (y1 + y2) * 0.5, (x2 - x1), (y2 - y1))
    else:
        if isinstance(corner, list):
            x1, y1, x2, y2 = corner[0], corner[1], corner[2], corner[3]
            x = (x1 + x2) * 0.5
            y = (y1 + y2) * 0.5
            w = x2 - x1
            h = y2 - y1
            return np.array([x, y, w, h])
        else:
            if len(corner.shape) > 1:
                x = (corner[..., 2] + corner[..., 0]) * 0.5
                y = (corner[..., 3] + corner[..., 1]) * 0.5
                w = corner[..., 2] - corner[..., 0]
                h = corner[..., 3] - corner[..., 1]
                return np.stack([x, y, w, h], axis=-1)
            else:
                x1, y1, x2, y2 = corner[0], corner[1], corner[2], corner[3]
                x = (x1 + x2) * 0.5
                y = (y1 + y2) * 0.5
                w = x2 - x1
                h = y2 - y1
                return np.array([x, y, w, h])


def center2corner(center):
    """
    convert (cx, cy, w, h) to (x1, y1, x2, y2)

    Args:
        center: Center / np.array (..., 4) / np.array (4)
    Return:
        Corner / np.array (..., 4) / np.array (4)
    """
    if isinstance(center, Center):
        x, y, w, h = center
        return Corner(x - w * 0.5, y - h * 0.5, x + w * 0.5, y + h * 0.5)
    else:
        if isinstance(center, list):
            x, y, w, h = center[0], center[1], center[2], center[3]
            x1 = x - w * 0.5
            y1 = y - h * 0.5
            x2 = x + w * 0.5
            y2 = y + h * 0.5
            return np.array([x1, y1, x2, y2])
        else:
            if len(center.shape) > 1:
                x1 = center[..., 0] - center[..., 2] * 0.5
                y1 = center[..., 1] - center[..., 3] * 0.5
                x2 = center[..., 0] + center[..., 2] * 0.5
                y2 = center[..., 1] + center[..., 3] * 0.5
                return np.stack([x1, y1, x2, y2], axis=-1)
            else:
                x, y, w, h = center[0], center[1], center[2], center[3]
                x1 = x - w * 0.5
                y1 = y - h * 0.5
                x2 = x + w * 0.5
                y2 = y + h * 0.5
                return np.array([x1, y1, x2, y2])


def get_axis_aligned_bbox(region):
    """ convert region to (cx, cy, w, h) that represent by axis aligned box
    """
    nv = region.size
    if nv == 8:
        cx = np.mean(region[0::2])
        cy = np.mean(region[1::2])
        x1 = min(region[0::2])
        x2 = max(region[0::2])
        y1 = min(region[1::2])
        y2 = max(region[1::2])
        A1 = np.linalg.norm(region[0:2] - region[2:4]) * np.linalg.norm(region[2:4] - region[4:6])
        A2 = (x2 - x1) * (y2 - y1)
        s = np.sqrt(A1 / A2)
        w = s * (x2 - x1) + 1
        h = s * (y2 - y1) + 1
    else:
        x = region[0]
        y = region[1]
        w = region[2]
        h = region[3]
        cx = x + w / 2.
        cy = y + h / 2.

    x1 = cx - w / 2.
    y1 = cy - h / 2.
    x2 = cx + w / 2.
    y2 = cy + h / 2.

    bbox = np.array([x1, y1, x2, y2], dtype=np.float32)
    bbox_xywh = np.array([cx, cy, w, h], dtype=np.float32)
    rect = [x1, y1, w, h]
    return rect, bbox, bbox_xywh


def get_axis_aligned_bbox_new(region):
    """ convert region to (cx, cy, w, h) that represent by axis aligned box
    """
    nv = region.size
    if nv == 8:
        x1 = min(region[0::2])
        x2 = max(region[0::2])
        y1 = min(region[1::2])
        y2 = max(region[1::2])

        w = x2 - x1
        h = y2 - y1
        cx = x1 + w / 2.
        cy = y1 + h / 2.
    else:
        x1 = region[0]
        y1 = region[1]
        w = region[2]
        h = region[3]
        cx = x1 + w / 2.
        cy = y1 + h / 2.
        x2 = cx + w / 2.
        y2 = cy + h / 2.

    bbox = np.array([x1, y1, x2, y2], dtype=np.float32)
    bbox_xywh = np.array([cx, cy, w, h], dtype=np.float32)
    rect = [x1, y1, w, h]
    return rect, bbox, bbox_xywh


def non_max_suppression(boxes, scores, conf_thresh=0.64, nms_thresh=0.4):
    pos_index = np.where(scores > conf_thresh)

    # 最高得分低于阈值
    if not pos_index[0].size:
        best_index = np.argmax(scores)
        bbox_xywh = boxes[best_index][None, :]
        bbox = center2corner(bbox_xywh)
        return bbox_xywh, bbox, np.array([scores[best_index]])[None, :]
    else:
        pos_scores = scores[pos_index]

        bbox_xywh = boxes[pos_index]
        bbox = center2corner(bbox_xywh)

        sort_index = np.argsort(pos_scores)[::-1]
        bbox = bbox[sort_index]
        bbox_xywh = bbox_xywh[sort_index]
        pos_scores = pos_scores[sort_index]

        best_bbox = []
        best_bbox_xywh = []
        best_scores = []
        while bbox.shape[0] > 0:
            best_bbox.append(bbox[0])
            best_bbox_xywh.append(bbox_xywh[0])
            best_scores.append(pos_scores[0])
            if bbox.shape[0] == 1:
                break
            overlap = box_iou(bbox[0], bbox[1:])
            mask = overlap < nms_thresh

            bbox = bbox[1:][mask]
            bbox_xywh = bbox_xywh[1:][mask]
            pos_scores = pos_scores[1:][mask]
        return np.array(best_bbox_xywh), np.array(best_bbox), np.array(best_scores)
