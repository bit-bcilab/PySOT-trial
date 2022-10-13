# Copyright (c) SenseTime. All Rights Reserved.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import numpy as np
import torch.nn.functional as F


from pysot.utils.anchor import Anchors
from pysot.trackers.base_tracker import SiameseTracker, change, sz


class SiamRPNTracker(SiameseTracker):
    def __init__(self, cfg, model):
        super(SiamRPNTracker, self).__init__(cfg, model)
        self.anchor_num = len(cfg.ANCHOR.RATIOS) * len(cfg.ANCHOR.SCALES)
        self.score_size = (cfg.TRACK.INSTANCE_SIZE - cfg.TRACK.EXEMPLAR_SIZE) // cfg.ANCHOR.STRIDE + 1 + cfg.TRACK.BASE_SIZE
        hanning = np.hanning(self.score_size)
        window = np.outer(hanning, hanning)
        self.window = np.tile(window.flatten(), self.anchor_num)
        self.anchors = self.generate_anchor(self.score_size)

    def generate_anchor(self, score_size):
        anchors = Anchors(self.cfg.ANCHOR.STRIDE, self.cfg.ANCHOR.RATIOS, self.cfg.ANCHOR.SCALES)
        anchor = anchors.anchors
        x1, y1, x2, y2 = anchor[:, 0], anchor[:, 1], anchor[:, 2], anchor[:, 3]
        anchor = np.stack([(x1+x2)*0.5, (y1+y2)*0.5, x2-x1, y2-y1], 1)
        total_stride = anchors.stride
        anchor_num = anchor.shape[0]
        anchor = np.tile(anchor, score_size * score_size).reshape((-1, 4))
        ori = - (score_size // 2) * total_stride
        xx, yy = np.meshgrid([ori + total_stride * dx for dx in range(score_size)],
                             [ori + total_stride * dy for dy in range(score_size)])
        xx, yy = np.tile(xx.flatten(), (anchor_num, 1)).flatten(), np.tile(yy.flatten(), (anchor_num, 1)).flatten()
        anchor[:, 0], anchor[:, 1] = xx.astype(np.float32), yy.astype(np.float32)
        return anchor

    def _convert_bbox(self, delta, anchor):
        delta = delta.permute(1, 2, 3, 0).contiguous().view(4, -1)
        delta = delta.data.cpu().numpy()

        delta[0, :] = delta[0, :] * anchor[:, 2] + anchor[:, 0]
        delta[1, :] = delta[1, :] * anchor[:, 3] + anchor[:, 1]
        delta[2, :] = np.exp(delta[2, :]) * anchor[:, 2]
        delta[3, :] = np.exp(delta[3, :]) * anchor[:, 3]
        return delta

    def _convert_score(self, score):
        score = score.permute(1, 2, 3, 0).contiguous().view(2, -1).permute(1, 0)
        score = F.softmax(score, dim=1).data[:, 1].cpu().numpy()
        return score

    def init(self, img, bbox, restart=False):
        """
        args:
            img(np.ndarray): BGR image
            bbox: (x, y, w, h) bbox
        """
        # restart = False
        self.center_pos = np.array([bbox[0]+(bbox[2]-1)/2,
                                    bbox[1]+(bbox[3]-1)/2])
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
        x_crop = self.get_subwindow(img, self.center_pos,
                                    self.cfg.TRACK.INSTANCE_SIZE,
                                    round(s_x), self.channel_average)

        outputs = self.model.track(x_crop)

        score = self._convert_score(outputs['cls'])
        pred_bbox = self._convert_bbox(outputs['loc'], self.anchors)

        # a = score.reshape((5, 25, 25))

        # scale penalty
        s_c = change(sz(pred_bbox[2, :], pred_bbox[3, :]) / (sz(self.size[0]*scale_z, self.size[1]*scale_z)))
        # aspect ratio penalty
        r_c = change((self.size[0]/self.size[1]) / (pred_bbox[2, :]/pred_bbox[3, :]))
        penalty = np.exp(-(r_c * s_c - 1) * self.cfg.TRACK.PENALTY_K)
        pscore = penalty * score

        # window penalty
        pscore = pscore * (1 - self.cfg.TRACK.WINDOW_INFLUENCE) + self.window * self.cfg.TRACK.WINDOW_INFLUENCE
        best_idx = np.argmax(pscore)
        # best_idx = np.argmax(score)

        bbox = pred_bbox[:, best_idx] / scale_z
        lr = penalty[best_idx] * score[best_idx] * self.cfg.TRACK.LR

        cx = bbox[0] + self.center_pos[0]
        cy = bbox[1] + self.center_pos[1]

        # smooth bbox
        width = self.size[0] * (1 - lr) + bbox[2] * lr
        height = self.size[1] * (1 - lr) + bbox[3] * lr

        # clip boundary
        cx, cy, width, height = self._bbox_clip(cx, cy, width, height, img.shape[:2])

        if score[best_idx] > self.cfg.TRACK.CONFIDENCE:
            self.center_pos = np.array([cx, cy])
            self.size = np.array([width, height])
        # if score.max() > self.cfg.TRACK.CONFIDENCE_LOW:
        #     # udpate state
        #     self.center_pos = np.array([cx, cy])
        #     self.size = np.array([width, height])
        # else:
        #     b = 1

        bbox = [cx - width / 2,
                cy - height / 2,
                width,
                height]
        best_score = score[best_idx]
        return {
                'bbox': bbox,
                'best_score': best_score
               }
