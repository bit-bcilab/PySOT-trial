from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import os
import argparse

import cv2
import torch
import numpy as np
from glob import glob

from configs.get_config import get_config
from configs.DataPath import SYSTEM
from pysot.models.model.model_builder import build_model
from pysot.trackers.tracker_builder import build_tracker
from pysot.utils.model_load import load_pretrain

torch.set_num_threads(1)

parser = argparse.ArgumentParser(description='tracking demo')
parser.add_argument('--tracker', default='SiamAtt', type=str, help='config file')
parser.add_argument('--config', default='experiments/siamatt/config.yaml', type=str, help='config file')
parser.add_argument('--snapshot', default='E://pysot-trial-sub/weights/SiamAtt-provided.pth', type=str, help='model name')
args = parser.parse_args()


def get_frames(video_name):
    if not video_name:
        cap = cv2.VideoCapture(0)
        # warmup
        for i in range(5):
            cap.read()
        while True:
            ret, frame = cap.read()
            if ret:
                yield frame
            else:
                break
    elif video_name.endswith('avi') or video_name.endswith('mp4') or video_name.endswith('mov'):
        cap = cv2.VideoCapture(video_name)
        while True:
            ret, frame = cap.read()
            if ret:
                yield frame
            else:
                break
    else:
        images = glob(os.path.join(video_name, '*.jp*'))
        if SYSTEM == 'Linux':
            images = sorted(images, key=lambda x: int(x.split(os.sep)[-1].split('.')[0]))

        for img in images:
            frame = cv2.imread(img)
            yield frame


def demo(save_path=None):
    tracker_name = args.tracker
    cfg = get_config(tracker_name)
    # load config
    cfg.merge_from_file(args.config)
    cfg.CUDA = torch.cuda.is_available() and cfg.CUDA
    device = torch.device('cuda' if cfg.CUDA else 'cpu')

    # create model
    model = build_model(tracker_name, cfg)
    # load model
    model = load_pretrain(model, args.snapshot).cuda().eval()

    # build trackers
    tracker = build_tracker(cfg, model)

    while True:
        video = input('path: ')
        first_frame = True
        if video != '':
            video_name = video.split(os.sep)[-1].split('.')[0]
            if video_name == 'img':
                video_name = video.split(os.sep)[-2].split('.')[0]
        else:
            video_name = 'webcam'
        cv2.namedWindow(video_name, cv2.WND_PROP_FULLSCREEN)

        count = 0
        for frame in get_frames(video):
            if first_frame:
                try:
                    init_rect = cv2.selectROI(video_name, frame, False, False)
                except:
                    exit()
                # init_rect = [336, 165, 26, 61]

                tracker.init(frame, init_rect)

                first_frame = False
                if save_path is not None:
                    if not os.path.exists(save_path):
                        os.mkdir(save_path)
                    save_path_ = '{:s}/{:s}'.format(save_path, video_name)
                    if not os.path.exists(save_path_):
                        os.mkdir(save_path_)
            else:
                count += 1
                outputs = tracker.track(frame)
                if 'polygon' in outputs:
                    polygon = np.array(outputs['polygon']).astype(np.int32)
                    cv2.polylines(frame, [polygon.reshape((-1, 1, 2))], True, (0, 255, 0), 3)
                    mask = ((outputs['mask'] > cfg.TRACK.MASK_THERSHOLD) * 255)
                    mask = mask.astype(np.uint8)
                    mask = np.stack([mask, mask * 255, mask]).transpose(1, 2, 0)
                    frame = cv2.addWeighted(frame, 0.77, mask, 0.23, -1)
                else:
                    bbox = list(map(int, outputs['bbox']))
                    cv2.rectangle(frame, (bbox[0], bbox[1]),
                                  (bbox[0] + bbox[2], bbox[1] + bbox[3]),
                                  (0, 255, 0), 3)
                cv2.imshow(video_name, frame)
                if save_path is not None:
                    cv2.imwrite('{:s}/{:04d}.jpg'.format(save_path_, count), frame)
                k = cv2.waitKey(40) & 0xff
                if k == 27:
                    break
        cv2.destroyWindow(video_name)


if __name__ == '__main__':
    demo()
    # demo(save_path='results/Visual')
