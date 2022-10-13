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


class SiamCARTracker(SiameseTracker):
    def __init__(self, cfg, model):
        super(SiamCARTracker, self).__init__(cfg, model)
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
        cen = outputs['cen'].data.cpu().numpy()
        cen = (cen - cen.min()) / cen.ptp()
        cen = cen.squeeze()
        cen = cen.flatten()

        # scale penalty
        s_c = change(sz(pred_bbox[2, :], pred_bbox[3, :], self.cfg.TRACK.CONTEXT_AMOUNT) /
                     (sz(self.size[0] * scale_z, self.size[1] * scale_z, self.cfg.TRACK.CONTEXT_AMOUNT)))
        # aspect ratio penalty
        r_c = change((self.size[0] / self.size[1]) / (pred_bbox[2, :] / pred_bbox[3, :]))
        penalty = np.exp(-(r_c * s_c - 1) * self.cfg.TRACK.PENALTY_K)
        pscore = penalty * score * cen
        # pscore = penalty * score

        # window penalty
        pscore = pscore * (1 - self.cfg.TRACK.WINDOW_INFLUENCE) + self.window * self.cfg.TRACK.WINDOW_INFLUENCE
        best_idx = np.argmax(pscore)
        # best_idx = np.argmax(score)
        bbox = pred_bbox[:, best_idx] / scale_z
        lr = penalty[best_idx] * score[best_idx] * self.cfg.TRACK.LR

        a = score.reshape((25, 25))
        b = cen.reshape((25, 25))
        c = pscore.reshape((25, 25))

        cx = bbox[0] + self.center_pos[0]
        cy = bbox[1] + self.center_pos[1]

        # smooth bbox
        width = self.size[0] * (1 - lr) + bbox[2] * lr
        height = self.size[1] * (1 - lr) + bbox[3] * lr

        # clip boundary
        cx, cy, width, height = self._bbox_clip(cx, cy, width, height, img.shape[:2])

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


# class SiamCARTracker(SiameseTracker):
#     def __init__(self, cfg, model):
#         super(SiamCARTracker, self).__init__(cfg, model)
#         hanning = np.hanning(cfg.TRACK.SCORE_SIZE)
#         self.window = np.outer(hanning, hanning)
#
#     def _convert_cls(self, cls):
#         cls = F.softmax(cls[:, :, :, :], dim=1).data[:, 1, :, :].cpu().numpy()
#         return cls
#
#     def init(self, img, bbox, restart=False):
#         """
#         args:
#             img(np.ndarray): BGR image
#             bbox: (x, y, w, h) bbox
#         """
#         self.center_pos = np.array([bbox[0]+(bbox[2]-1)/2, bbox[1]+(bbox[3]-1)/2])
#         self.size = np.array([bbox[2], bbox[3]])
#
#         # calculate z crop size
#         w_z = self.size[0] + self.cfg.TRACK.CONTEXT_AMOUNT * np.sum(self.size)
#         h_z = self.size[1] + self.cfg.TRACK.CONTEXT_AMOUNT * np.sum(self.size)
#         s_z = round(np.sqrt(w_z * h_z))
#
#         # calculate channle average
#         self.channel_average = np.mean(img, axis=(0, 1))
#
#         # get crop
#         z_crop = self.get_subwindow(img, self.center_pos, self.cfg.TRACK.EXEMPLAR_SIZE, s_z, self.channel_average)
#         self.model.template(z_crop)
#
#     def cal_penalty(self, lrtbs, penalty_lk):
#         bboxes_w = lrtbs[0, :, :] + lrtbs[2, :, :]
#         bboxes_h = lrtbs[1, :, :] + lrtbs[3, :, :]
#         s_c = change(sz(bboxes_w, bboxes_h) / sz(self.size[0] * self.scale_z, self.size[1] * self.scale_z))
#         r_c = change((self.size[0] / self.size[1]) / (bboxes_w / bboxes_h))
#         penalty = np.exp(-(r_c * s_c - 1) * penalty_lk)
#         return penalty
#
#     def accurate_location(self, max_r_up, max_c_up):
#         dist = int((self.cfg.TRACK.INSTANCE_SIZE - (self.cfg.TRACK.SCORE_SIZE - 1) * 8) / 2)
#         max_r_up += dist
#         max_c_up += dist
#         p_cool_s = np.array([max_r_up, max_c_up])
#         disp = p_cool_s - (np.array([self.cfg.TRACK.INSTANCE_SIZE, self.cfg.TRACK.INSTANCE_SIZE]) - 1.) / 2.
#         return disp
#
#     def coarse_location(self, hp_score_up, p_score_up, scale_score, lrtbs):
#         upsize = (self.cfg.TRACK.SCORE_SIZE - 1) * self.cfg.TRACK.STRIDE + 1
#         max_r_up_hp, max_c_up_hp = np.unravel_index(hp_score_up.argmax(), hp_score_up.shape)
#         max_r = int(round(max_r_up_hp / scale_score))
#         max_c = int(round(max_c_up_hp / scale_score))
#         max_r = bbox_clip(max_r, 0, self.cfg.TRACK.SCORE_SIZE)
#         max_c = bbox_clip(max_c, 0, self.cfg.TRACK.SCORE_SIZE)
#         bbox_region = lrtbs[max_r, max_c, :]
#         min_bbox = int(self.cfg.TRACK.REGION_S * self.cfg.TRACK.EXEMPLAR_SIZE)
#         max_bbox = int(self.cfg.TRACK.REGION_L * self.cfg.TRACK.EXEMPLAR_SIZE)
#         l_region = int(min(max_c_up_hp, bbox_clip(bbox_region[0], min_bbox, max_bbox)) / 2.0)
#         t_region = int(min(max_r_up_hp, bbox_clip(bbox_region[1], min_bbox, max_bbox)) / 2.0)
#
#         r_region = int(min(upsize - max_c_up_hp, bbox_clip(bbox_region[2], min_bbox, max_bbox)) / 2.0)
#         b_region = int(min(upsize - max_r_up_hp, bbox_clip(bbox_region[3], min_bbox, max_bbox)) / 2.0)
#         mask = np.zeros_like(p_score_up)
#         mask[max_r_up_hp - t_region:max_r_up_hp + b_region + 1, max_c_up_hp - l_region:max_c_up_hp + r_region + 1] = 1
#         p_score_up = p_score_up * mask
#         return p_score_up
#
#     def getCenter(self, hp_score_up, p_score_up, scale_score, lrtbs):
#         # corse location
#         score_up = self.coarse_location(hp_score_up, p_score_up, scale_score, lrtbs)
#         # accurate location
#         max_r_up, max_c_up = np.unravel_index(score_up.argmax(), score_up.shape)
#         disp = self.accurate_location(max_r_up, max_c_up)
#         disp_ori = disp / self.scale_z
#         new_cx = disp_ori[1] + self.center_pos[0]
#         new_cy = disp_ori[0] + self.center_pos[1]
#         return max_r_up, max_c_up, new_cx, new_cy
#
#     def track(self, img):
#         """
#         args:
#             img(np.ndarray): BGR image
#         return:
#             bbox(list):[x, y, width, height]
#         """
#         w_z = self.size[0] + self.cfg.TRACK.CONTEXT_AMOUNT * np.sum(self.size)
#         h_z = self.size[1] + self.cfg.TRACK.CONTEXT_AMOUNT * np.sum(self.size)
#         s_z = np.sqrt(w_z * h_z)
#         self.scale_z = self.cfg.TRACK.EXEMPLAR_SIZE / s_z
#         s_x = s_z * (self.cfg.TRACK.INSTANCE_SIZE / self.cfg.TRACK.EXEMPLAR_SIZE)
#         x_crop = self.get_subwindow(img, self.center_pos, self.cfg.TRACK.INSTANCE_SIZE, round(s_x), self.channel_average)
#
#         outputs = self.model.track(x_crop)
#         cls = self._convert_cls(outputs['cls']).squeeze()
#         cen = outputs['cen'].data.cpu().numpy()
#         cen = (cen - cen.min()) / cen.ptp()
#         cen = cen.squeeze()
#         lrtbs = outputs['loc'].data.cpu().numpy().squeeze()
#
#         upsize = (self.cfg.TRACK.SCORE_SIZE-1) * self.cfg.TRACK.STRIDE + 1
#         penalty = self.cal_penalty(lrtbs, self.cfg.TRACK.PENALTY_K)
#         p_score = penalty * cls * cen
#         if self.cfg.TRACK.hanming:
#             hp_score = p_score * (1 - self.cfg.TRACK.WINDOW_INFLUENCE) + self.window * self.cfg.TRACK.WINDOW_INFLUENCE
#         else:
#             hp_score = p_score
#
#         hp_score_up = cv2.resize(hp_score, (upsize, upsize), interpolation=cv2.INTER_CUBIC)
#         p_score_up = cv2.resize(p_score, (upsize, upsize), interpolation=cv2.INTER_CUBIC)
#         cls_up = cv2.resize(cls, (upsize, upsize), interpolation=cv2.INTER_CUBIC)
#         lrtbs = np.transpose(lrtbs, (1, 2, 0))
#         lrtbs_up = cv2.resize(lrtbs, (upsize, upsize), interpolation=cv2.INTER_CUBIC)
#
#         scale_score = upsize / self.cfg.TRACK.SCORE_SIZE
#         # get center
#         max_r_up, max_c_up, new_cx, new_cy = self.getCenter(hp_score_up, p_score_up, scale_score, lrtbs)
#         # get w h
#         ave_w = (lrtbs_up[max_r_up, max_c_up, 0] + lrtbs_up[max_r_up, max_c_up, 2]) / self.scale_z
#         ave_h = (lrtbs_up[max_r_up, max_c_up, 1] + lrtbs_up[max_r_up, max_c_up, 3]) / self.scale_z
#
#         s_c = change(sz(ave_w, ave_h) / sz(self.size[0] * self.scale_z, self.size[1] * self.scale_z))
#         r_c = change((self.size[0] / self.size[1]) / (ave_w / ave_h))
#         penalty = np.exp(-(r_c * s_c - 1) * self.cfg.TRACK.PENALTY_K)
#         lr = penalty * cls_up[max_r_up, max_c_up] * self.cfg.TRACK.LR
#         new_width = lr * ave_w + (1-lr) * self.size[0]
#         new_height = lr * ave_h + (1-lr) * self.size[1]
#
#         # clip boundary
#         cx = bbox_clip(new_cx, 0, img.shape[1])
#         cy = bbox_clip(new_cy, 0, img.shape[0])
#         width = bbox_clip(new_width, 0, img.shape[1])
#         height = bbox_clip(new_height, 0, img.shape[0])
#
#         # udpate state
#         self.center_pos = np.array([cx, cy])
#         self.size = np.array([width, height])
#         bbox = [cx - width / 2,
#                 cy - height / 2,
#                 width,
#                 height]
#
#         return {
#                 'bbox': bbox,
#                }
