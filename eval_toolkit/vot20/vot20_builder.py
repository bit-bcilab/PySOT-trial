from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import cv2
import torch
import eval_toolkit.vot20.vot20 as vot20
import sys
import time
import os

from eval_toolkit.vot20.vot20_utils import *
from pysot.models.model.model_builder import build_model
from pysot.trackers.tracker_builder import build_tracker
from pysot.utils.model_load import load_pretrain
from configs.get_config import get_config, Config

'''stark_vot20 class'''


class vot20_tracker(object):
    def __init__(self, tracker_name, config_file, weight_path, track_cfg=None):
        # create vot tracker

        # load config
        cfg = get_config(tracker_name)
        cfg.merge_from_file(config_file)
        cfg.CUDA = torch.cuda.is_available() and cfg.CUDA

        if track_cfg is not None:
            cfg.TRACK.CONTEXT_AMOUNT = track_cfg.context_amount
            cfg.TRACK.WINDOW_INFLUENCE = track_cfg.window_influence
            cfg.TRACK.PENALTY_K = track_cfg.penalty_k
            cfg.TRACK.LR = track_cfg.size_lr
            cfg.TRACK.CONFIDENCE = track_cfg.confidence

        context_amount = cfg.TRACK.CONTEXT_AMOUNT
        window_influence = cfg.TRACK.WINDOW_INFLUENCE
        penalty_k = cfg.TRACK.PENALTY_K
        size_lr = cfg.TRACK.LR
        confidence = cfg.TRACK.CONFIDENCE

        model_name = tracker_name + '-' + weight_path.split('/')[-1].split('.')[0]
        name = '{:s}_ca-{:.4f}_wi-{:.4f}_pk-{:.4f}_lr-{:.4f}_cf-{:.4f}'.format(model_name, context_amount, window_influence,
                                                                               penalty_k,  size_lr, confidence)
        self.tracker_name = name

        # create model
        model = build_model(tracker_name, cfg)

        # load model
        if torch.cuda.is_available():
            model = load_pretrain(model, weight_path).cuda().eval()
        else:
            model = load_pretrain(model, weight_path, False).eval()

        # build trackers
        tracker = build_tracker(cfg, model)
        self.tracker = tracker

    def initialize(self, img, mask):
        # init on the 1st frame
        self.ih, self.iw = img.shape[:2]
        region = rect_from_mask(mask)

        _ = self.tracker.init(img, region)

    def track(self, img):
        # track
        outputs = self.tracker.track(img)
        pred_bbox = outputs['bbox']
        final_mask = mask_from_rect(pred_bbox, (self.iw, self.ih))
        return pred_bbox, final_mask


def run_vot_exp(tracker_name, config_file, weight_path, vis=False, track_cfg=None):
    torch.set_num_threads(1)

    tracker = vot20_tracker(tracker_name, config_file=config_file, weight_path=weight_path, track_cfg=track_cfg)

    save_root = os.path.join('E://PySOT-Trial/eval_toolkit//vot_test/test_results/', tracker.tracker_name)
    if vis and (not os.path.exists(save_root)):
        os.mkdir(save_root)

    handle = vot20.VOT("mask")
    selection = handle.region()
    imagefile = handle.frame()
    if not imagefile:
        sys.exit(0)
    if vis:
        '''for vis'''
        seq_name = imagefile.split('/')[-3]
        save_v_dir = os.path.join(save_root, seq_name)
        if not os.path.exists(save_v_dir):
            os.mkdir(save_v_dir)
        cur_time = int(time.time() % 10000)
        save_dir = os.path.join(save_v_dir, str(cur_time))
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

    image = cv2.imread(imagefile)
    # mask given by the toolkit ends with the target (zero-padding to the right and down is needed)
    mask = make_full_size(selection, (image.shape[1], image.shape[0]))
    tracker.initialize(image, mask)

    while True:
        imagefile = handle.frame()
        if not imagefile:
            break
        image = cv2.imread(imagefile)
        b1, m = tracker.track(image)
        handle.report(m)
        if vis:
            '''Visualization'''
            # original image
            image_ori = image[:, :, ::-1].copy()  # RGB --> BGR
            image_name = imagefile.split('/')[-1]
            save_path = os.path.join(save_dir, image_name)
            cv2.imwrite(save_path, image_ori)
            # tracker box
            image_b = image_ori.copy()
            cv2.rectangle(image_b, (int(b1[0]), int(b1[1])),
                          (int(b1[0] + b1[2]), int(b1[1] + b1[3])), (0, 0, 255), 2)
            image_b_name = image_name.replace('.jpg', '_bbox.jpg')
            save_path = os.path.join(save_dir, image_b_name)
            cv2.imwrite(save_path, image_b)
            # original image + mask
            image_m = image_ori.copy().astype(np.float32)
            image_m[:, :, 1] += 127.0 * m
            image_m[:, :, 2] += 127.0 * m
            contours, _ = cv2.findContours(m, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            image_m = cv2.drawContours(image_m, contours, -1, (0, 255, 255), 2)
            image_m = image_m.clip(0, 255).astype(np.uint8)
            image_mask_name_m = image_name.replace('.jpg', '_mask.jpg')
            save_path = os.path.join(save_dir, image_mask_name_m)
            cv2.imwrite(save_path, image_m)
