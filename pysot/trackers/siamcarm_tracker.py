# Copyright (c) SenseTime. All Rights Reserved.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import numpy as np
import torch.nn.functional as F
import cv2

from pysot.trackers.base_tracker import SiameseTracker, change, sz
from pysot.utils.bbox import bbox_clip, corner2center


class SiamCARMTracker(SiameseTracker):
    def __init__(self, cfg, model):
        super(SiamCARMTracker, self).__init__(cfg, model)
        self.score_size = (cfg.TRACK.INSTANCE_SIZE - cfg.TRACK.EXEMPLAR_SIZE) // cfg.POINT.STRIDE + 1 + cfg.TRACK.BASE_SIZE
        hanning = np.hanning(self.score_size)
        window = np.outer(hanning, hanning)
        self.cls_out_channels = 2
        self.window = window.flatten()
        self.points = self.generate_points(cfg.POINT.STRIDE, self.score_size)

    def generate_points(self, stride, size):
        ori = - (size // 2) * stride
        x, y = np.meshgrid([ori + stride * dx for dx in np.arange(0, size)],
                           [ori + stride * dy for dy in np.arange(0, size)])
        points = np.zeros((size * size, 2), dtype=np.float32)
        points[:, 0], points[:, 1] = x.astype(np.float32).flatten(), y.astype(np.float32).flatten()

        return points

    def _convert_bbox(self, delta, point):
        delta = delta.permute(1, 2, 3, 0).contiguous().view(4, -1)
        delta = delta.detach().cpu().numpy()

        delta[0, :] = point[:, 0] - delta[0, :]
        delta[1, :] = point[:, 1] - delta[1, :]
        delta[2, :] = point[:, 0] + delta[2, :]
        delta[3, :] = point[:, 1] + delta[3, :]
        delta[0, :], delta[1, :], delta[2, :], delta[3, :] = corner2center(delta)
        return delta

    def _convert_score(self, score):
        if self.cls_out_channels == 1:
            score = score.permute(1, 2, 3, 0).contiguous().view(-1)
            score = score.sigmoid().detach().cpu().numpy()
        else:
            score = score.permute(1, 2, 3, 0).contiguous().view(self.cls_out_channels, -1).permute(1, 0)
            score = score.softmax(1).detach()[:, 1].cpu().numpy()
        return score

    def init(self, img, bbox, restart=False):
        """
        args:
            img(np.ndarray): BGR image
            bbox: (x, y, w, h) bbox
        """
        self.center_pos = np.array([bbox[0] + (bbox[2] - 1) / 2, bbox[1] + (bbox[3] - 1) / 2])
        self.size = np.array([bbox[2], bbox[3]])

        # calculate z crop size
        # w_z = self.size[0] + self.cfg.TRACK.CONTEXT_AMOUNT * np.sum(self.size)
        # h_z = self.size[1] + self.cfg.TRACK.CONTEXT_AMOUNT * np.sum(self.size)
        w_z = self.size[0] + 0.5 * np.sum(self.size)
        h_z = self.size[1] + 0.5 * np.sum(self.size)
        s_z = round(np.sqrt(w_z * h_z))

        # calculate channle average
        self.channel_average = np.mean(img, axis=(0, 1))

        # get crop
        z_crop = self.get_subwindow(img, self.center_pos, self.cfg.TRACK.EXEMPLAR_SIZE, s_z, self.channel_average)
        self.model.template(z_crop)

    def track(self, img):
        """
        args:
            img(np.ndarray): BGR image
        return:
            bbox(list):[x, y, width, height]
        """
        w_z = self.size[0] + self.cfg.TRACK.CONTEXT_AMOUNT * np.sum(self.size)
        h_z = self.size[1] + self.cfg.TRACK.CONTEXT_AMOUNT * np.sum(self.size)
        s_z = np.sqrt(w_z * h_z)
        scale_z = self.cfg.TRACK.EXEMPLAR_SIZE / s_z
        s_x = s_z * (self.cfg.TRACK.INSTANCE_SIZE / self.cfg.TRACK.EXEMPLAR_SIZE)
        x_crop = self.get_subwindow(img, self.center_pos, self.cfg.TRACK.INSTANCE_SIZE, round(s_x),
                                    self.channel_average)

        outputs = self.model.track(x_crop)

        score = self._convert_score(outputs['cls'])
        pred_bbox = self._convert_bbox(outputs['loc'], self.points)

        # scale penalty
        s_c = change(sz(pred_bbox[2, :], pred_bbox[3, :], self.cfg.TRACK.CONTEXT_AMOUNT) /
                     (sz(self.size[0] * scale_z, self.size[1] * scale_z, self.cfg.TRACK.CONTEXT_AMOUNT)))
        # aspect ratio penalty
        r_c = change((self.size[0] / self.size[1]) / (pred_bbox[2, :] / pred_bbox[3, :]))
        penalty = np.exp(-(r_c * s_c - 1) * self.cfg.TRACK.PENALTY_K)
        pscore = penalty * score

        # window penalty
        pscore = pscore * (1 - self.cfg.TRACK.WINDOW_INFLUENCE) + self.window * self.cfg.TRACK.WINDOW_INFLUENCE
        best_idx = np.argmax(pscore)
        # best_idx = np.argmax(score)
        bbox = pred_bbox[:, best_idx] / scale_z
        lr = penalty[best_idx] * score[best_idx] * self.cfg.TRACK.LR

        a = score.reshape((25, 25))
        b = pscore.reshape((25, 25))

        cx = bbox[0] + self.center_pos[0]
        cy = bbox[1] + self.center_pos[1]

        # smooth bbox
        width = self.size[0] * (1 - lr) + bbox[2] * lr
        height = self.size[1] * (1 - lr) + bbox[3] * lr

        # clip boundary
        cx, cy, width, height = self._bbox_clip(cx, cy, width, height, img.shape[:2])

        if score[best_idx] > self.cfg.TRACK.CONFIDENCE:
            # udpate state
            self.center_pos = np.array([cx, cy])
            self.size = np.array([width, height])

        bbox = [cx - width / 2,
                cy - height / 2,
                width,
                height]
        best_score = score[best_idx]
        return {
            'bbox': bbox,
            'best_score': best_score
        }

