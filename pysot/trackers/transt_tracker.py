

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import math

import cv2
import numpy as np
import torch.nn.functional as F

from pysot.trackers.base_tracker import SiameseTracker, change, sz
from trial.utils.bbox import center2corner, clip_bbox_corner, corner2center


class TransTracker(SiameseTracker):
    def __init__(self, cfg, model):
        super(TransTracker, self).__init__(cfg, model)
        self.score_size = cfg.TRAIN.OUTPUT_SIZE
        hanning = np.hanning(self.score_size)
        window = np.outer(hanning, hanning)
        self.window = window.flatten()

    def _convert_score(self, score):
        score = score.permute(2, 1, 0).contiguous().view(2, -1).permute(1, 0)
        score = F.softmax(score, dim=1).data[:, 0].cpu().numpy()
        return score

    def _convert_bbox(self, delta):
        delta = delta.permute(2, 1, 0).contiguous().view(4, -1)
        delta = delta.data.cpu().numpy()
        return delta

    def init0(self, img, bbox, restart=False):
        """
        args:
            img(np.ndarray): BGR image
            bbox: (x, y, w, h) bbox
        """
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        self.center_pos = np.array([bbox[0] + (bbox[2] - 1) / 2, bbox[1]+(bbox[3] - 1) / 2])
        self.size = np.array([bbox[2], bbox[3]])

        # calculate z crop size
        # w_z = self.size[0] + 0.5 * np.sum(self.size)
        # h_z = self.size[1] + 0.5 * np.sum(self.size)
        # s_z = round(np.sqrt(w_z * h_z))
        w_z = max(self.size[0] + (2 - 1) * ((self.size[0] + self.size[1]) * 0.5), 10.)
        h_z = max(self.size[1] + (2 - 1) * ((self.size[0] + self.size[1]) * 0.5), 10.)
        s_z = math.ceil(math.sqrt(w_z * h_z))

        # calculate channle average
        self.channel_average = np.mean(img, axis=(0, 1))

        # get crop
        z_crop = self.get_subwindow(img, self.center_pos, self.cfg.TRACK.EXEMPLAR_SIZE, s_z, self.channel_average)
        self.model.template(z_crop)

    def track0(self, img):
        """
        args:
            img(np.ndarray): BGR image
        return:
            bbox(list):[x, y, width, height]
        """
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        w_x = max(self.size[0] + (4 - 1) * ((self.size[0] + self.size[1]) * 0.5), 10.)
        h_x = max(self.size[1] + (4 - 1) * ((self.size[0] + self.size[1]) * 0.5), 10.)
        s_x = math.ceil(math.sqrt(w_x * h_x))
        x_crop = self.get_subwindow(img, self.center_pos, self.cfg.TRACK.INSTANCE_SIZE, round(s_x),
                                    self.channel_average)

        outputs = self.model.track(x_crop)
        score = self._convert_score(outputs['pred_logits'])
        pred_bbox = self._convert_bbox(outputs['pred_boxes'])
        a = score.reshape((32, 32))
        # window penalty
        pscore = score * (1 - self.cfg.TRACK.WINDOW_INFLUENCE) + self.window * self.cfg.TRACK.WINDOW_INFLUENCE

        best_idx = np.argmax(pscore)
        bbox = pred_bbox[:, best_idx]
        bbox = bbox * s_x
        cx = self.center_pos[0] + (bbox[0] - s_x / 2)
        cy = self.center_pos[1] + (bbox[1] - s_x / 2)

        # clip boundary
        cx, cy, width, height = self._bbox_clip(cx, cy, bbox[2], bbox[3], img.shape[:2])

        # udpate state
        self.center_pos = np.array([cx, cy])
        self.size = np.array([width, height])

        bbox = [cx - width / 2,
                cy - height / 2,
                width,
                height]

        out = {'bbox': bbox,
               'best_score': pscore[best_idx]}
        return out

    def init(self, img, bbox, restart=False):
        """
        args:
            img(np.ndarray): BGR image
            bbox: (x, y, w, h) bbox
        """
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        self.center_pos = np.array([bbox[0] + (bbox[2] - 1) / 2, bbox[1] + (bbox[3] - 1) / 2])
        self.size = np.array([bbox[2], bbox[3]])

        # calculate z crop size
        w_z = self.size[0] + 0.5 * np.sum(self.size)
        h_z = self.size[1] + 0.5 * np.sum(self.size)
        s_z = round(np.sqrt(w_z * h_z))

        # calculate channle average
        self.channel_average = np.mean(img, axis=(0, 1))

        # get crop
        z_crop = self.get_subwindow(img, self.center_pos, self.cfg.TRACK.EXEMPLAR_SIZE, s_z, self.channel_average)
        self.model.template(z_crop)

    def track_(self, img):
        """
        args:
            img(np.ndarray): BGR image
        return:
            bbox(list):[x, y, width, height]
        """
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        context_amount = self.cfg.TRACK.CONTEXT_AMOUNT
        w_z = self.size[0] + context_amount * np.sum(self.size)
        h_z = self.size[1] + context_amount * np.sum(self.size)
        s_z = np.sqrt(w_z * h_z)
        s_x = s_z * (self.cfg.TRACK.INSTANCE_SIZE / self.cfg.TRACK.EXEMPLAR_SIZE)
        x_crop = self.get_subwindow(img, self.center_pos, self.cfg.TRACK.INSTANCE_SIZE, round(s_x), self.channel_average)

        outputs = self.model.track(x_crop)
        score = self._convert_score(outputs['pred_logits'])
        pred_bbox = self._convert_bbox(outputs['pred_boxes'])
        a = score.reshape((32, 32))
        # window penalty
        pscore = score * (1 - self.cfg.TRACK.WINDOW_INFLUENCE) + self.window * self.cfg.TRACK.WINDOW_INFLUENCE

        best_idx = np.argmax(pscore)
        bbox = pred_bbox[:, best_idx]
        bbox = bbox * s_x
        cx = bbox[0] + self.center_pos[0] - s_x / 2
        cy = bbox[1] + self.center_pos[1] - s_x / 2

        # clip boundary
        cx, cy, width, height = self._bbox_clip(cx, cy, bbox[2], bbox[3], img.shape[:2])

        # udpate state
        self.center_pos = np.array([cx, cy])
        self.size = np.array([width, height])

        bbox = [cx - width / 2,
                cy - height / 2,
                width,
                height]

        out = {'bbox': bbox,
               'best_score': pscore[best_idx]}
        return out

    def track(self, img):
        """
        args:
            img(np.ndarray): BGR image
        return:
            bbox(list):[x, y, width, height]
        """
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        context_amount = self.cfg.TRACK.CONTEXT_AMOUNT
        w_z = self.size[0] + context_amount * np.sum(self.size)
        h_z = self.size[1] + context_amount * np.sum(self.size)
        s_z = np.sqrt(w_z * h_z)
        scale_z = self.cfg.TRACK.EXEMPLAR_SIZE / s_z
        s_x = round(s_z * (self.cfg.TRACK.INSTANCE_SIZE / self.cfg.TRACK.EXEMPLAR_SIZE))
        x_crop = self.get_subwindow(img, self.center_pos, self.cfg.TRACK.INSTANCE_SIZE, s_x, self.channel_average)

        outputs = self.model.track(x_crop)
        score = self._convert_score(outputs['pred_logits'])
        pred_bbox = self._convert_bbox(outputs['pred_boxes'])
        a = score.reshape((32, 32))

        # scale penalty
        s_c = change(sz(pred_bbox[2, :] * s_x, pred_bbox[3, :] * s_x, context_amount) /
                     (sz(self.size[0], self.size[1], context_amount)))
        # aspect ratio penalty
        r_c = change((self.size[0] / self.size[1]) / (pred_bbox[2, :] / pred_bbox[3, :]))
        penalty = np.exp(-(r_c * s_c - 1) * self.cfg.TRACK.PENALTY_K)
        pscore = penalty * score
        # window penalty
        pscore = pscore * (1 - self.cfg.TRACK.WINDOW_INFLUENCE) + self.window * self.cfg.TRACK.WINDOW_INFLUENCE

        best_idx = np.argmax(pscore)
        bbox = pred_bbox[:, best_idx]
        bbox = bbox * s_x
        cx = bbox[0] + self.center_pos[0] - s_x / 2
        cy = bbox[1] + self.center_pos[1] - s_x / 2

        lr = penalty[best_idx] * score[best_idx] * self.cfg.TRACK.LR

        # clip boundary
        cx, cy, width, height = self._bbox_clip(cx, cy, bbox[2], bbox[3], img.shape[:2])

        bbox_xywh = np.array([cx, cy, width, height])
        bbox_xyxy = center2corner(bbox_xywh)
        bbox_xyxy = clip_bbox_corner(bbox_xyxy, img.shape[:2])
        width, height = bbox_xyxy[2] - bbox_xyxy[0], bbox_xyxy[3] - bbox_xyxy[1]

        width = self.size[0] * (1 - lr) + width * lr
        height = self.size[1] * (1 - lr) + height * lr
        if score[best_idx] > self.cfg.TRACK.CONFIDENCE:
            # udpate state
            self.center_pos = np.array([cx, cy])
            self.size = np.array([width, height])

        bbox = [cx - width / 2,
                cy - height / 2,
                width,
                height]

        out = {'bbox': bbox,
               'best_score': pscore[best_idx]}
        return out
