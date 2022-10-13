# Copyright (c) SenseTime. All Rights Reserved.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import random
import numpy as np
import torch

from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from pysot.utils.log_helper import init_log, print_speed, add_file_handler
from pysot.utils.distributed import dist_init, DistModule, reduce_gradients, average_reduce, get_rank, get_world_size
from pysot.utils.model_load import load_pretrain, restore_from, resume
from pysot.utils.average_meter import AverageMeter
from pysot.utils.misc import describe, commit

from configs.DataPath import SYSTEM
from configs.get_config import get_config
from pysot.models.model.model_builder import build_model
from pysot.utils.anchor import Anchors
from pysot.utils.point import Point
from trial.encoders import get_encoder
from trial.Generator import Generator
from trial.DataReader import DataReader

import argparse
import logging
import os
import time
import math
import json
import yaml
import cv2
from glob import glob
from os.path import join

TORCH_VERSION = int(torch.__version__.split('.')[1])
logger = logging.getLogger('global')
parser = argparse.ArgumentParser(description='validation')
parser.add_argument('--run_mode', default='val', type=str, help='run mode')
parser.add_argument('--checkpoint', default='snapshot/', type=str, help='path of validated checkpoints')
parser.add_argument('--tracker', default='SiamBAN', type=str, help='config file')
parser.add_argument('--config', default='experiments/siamban/trial.yaml', type=str, help='config file')
parser.add_argument('--seed', type=int, default=12345, help='random seed')
parser.add_argument('--local_rank', type=int, default=0, help='compulsory for pytorch launcer')
args = parser.parse_args()


def seed_torch(seed=0):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def build_data_loader(cfg, train_dataset, run_mode='train'):
    """
    torch.utils.data:
    Dataset：存储着数据，数据长度固定，因此每个data都有自己的索引index
    Sampler：分配每个batch所取的数据的indices
    DataLoader：进一步封装Dataset
    """
    num_workers = 0 if run_mode == 'debug' else cfg.TRAIN.NUM_WORKERS
    logger.info('number of workers for DDP: {:d}'.format(num_workers))

    train_sampler = DistributedSampler(train_dataset, shuffle=False) if get_world_size() > 1 else None

    """
    在构建dataset时已进行过随机shuffle。为确保resume的一致性与可复现性，无需再进行shuffle
    The data have been shuffled already when building the Dataset.
    To make sure the model can be reproducing and the consistence when resuming, 
    there is no need to shuffle the Dataloader again.
    """
    train_loader = DataLoader(train_dataset,
                              batch_size=cfg.TRAIN.BATCH_SIZE,
                              num_workers=num_workers,
                              pin_memory=True,
                              sampler=train_sampler)
    return train_loader


def main():
    checkpoint_dir = args.checkpoint
    if os.path.isdir(checkpoint_dir):
        checkpoints = glob(join(checkpoint_dir, '*.pth'))
        checkpoints = sorted(checkpoints, key=lambda x: int(x.split(os.sep)[-1].split('.')[0][12:]))
        num_checkpoints = len(checkpoints)
    else:
        return 0

    if rank == 0:
        init_log('global', logging.INFO)
        add_file_handler('global', os.path.join(checkpoint_dir, 'val_logs.txt'), logging.INFO)
        logger.info("Version Information: \n{}\n".format(commit()))
        logger.info("config \n{}".format(json.dumps(cfg, indent=4)))

        info = 'Validate on {:s}, {:d} checkpoints in total: \n'.format(checkpoint_dir, num_checkpoints)
        for i in range(num_checkpoints):
            info += checkpoints[i].split('\\')[-1]
            info += '\n'
        logger.info(info)

    logger.info('torch version: {:d}, run mode: {:s}'.format(TORCH_VERSION, run_mode))
    logger.info("init done")

    # create model
    model = build_model(tracker_name, cfg).cuda().eval()
    dist_model = DistModule(model)

    logger.info("build train dataset")
    with open(os.path.join(cfg.TRIAL_CFG), 'r', encoding='utf-8') as f:
        cond = f.read()
        trial_settings = yaml.load(cond, Loader=yaml.FullLoader)
        f.close()

    datareader = DataReader(data_settings=trial_settings['DATA_SETTINGS'], num_epoch=cfg.TRAIN.EPOCH, run_mode=run_mode)
    points = Point(cfg.POINT.STRIDE, cfg.TRAIN.OUTPUT_SIZE, cfg.TRAIN.SEARCH_SIZE // 2).points
    model.points = torch.from_numpy(points[None, ...]).cuda()
    model.weights = trial_settings['WEIGHT_SETTINGS']
    model.logger = logger
    if cfg.BASE == 'anchor':
        anchors = Anchors(cfg.ANCHOR.STRIDE, cfg.ANCHOR.RATIOS, cfg.ANCHOR.SCALES)
        anchors.generate_all_anchors(im_c=cfg.TRAIN.SEARCH_SIZE // 2, size=cfg.TRAIN.OUTPUT_SIZE)
        trial_settings['ENCODE_SETTINGS'].update(dict(anchors=anchors.all_anchors))
        trial_settings['ENCODE_SETTINGS_SELF'].update(dict(anchors=anchors.all_anchors))
        model.anchors = torch.from_numpy(anchors.all_anchors[1][None, ...]).cuda()
    elif cfg.BASE == 'point':
        trial_settings['ENCODE_SETTINGS'].update(dict(points=points))
        trial_settings['ENCODE_SETTINGS_SELF'].update(dict(points=points))

    logger.info("build val dataset")
    with open(os.path.join(cfg.VALIDATE_CFG), 'r', encoding='utf-8') as f:
        cond = f.read()
        val_settings = yaml.load(cond, Loader=yaml.FullLoader)
        f.close()

    val_kwargs = dict(datareader=datareader,
                      base=cfg.BASE,
                      mode='validate',
                      encoder=get_encoder(trial_settings['ENCODER']),
                      search_size=[cfg.TRAIN.SEARCH_SIZE, cfg.TRAIN.SEARCH_SIZE],
                      template_size=[cfg.TRAIN.EXEMPLAR_SIZE, cfg.TRAIN.EXEMPLAR_SIZE],
                      output_size=[cfg.TRAIN.OUTPUT_SIZE, cfg.TRAIN.OUTPUT_SIZE],
                      crop_settings=val_settings['CROP_SETTINGS'],
                      aug_settings=val_settings['AUG_SETTINGS'],
                      encode_settings=trial_settings['ENCODE_SETTINGS'],
                      bbox_mask_rate=trial_settings['BBOX_MASK_RATE'],
                      use_all_boxes=trial_settings['MIX_BOXES'])

    val_dataset = Generator(**val_kwargs)
    val_dataset.points = points
    val_loader = build_data_loader(cfg, val_dataset, run_mode=run_mode)

    val_steps_per_epoch = math.ceil(datareader.num_val / (cfg.TRAIN.BATCH_SIZE * world_size))
    average_meter_val = AverageMeter(num=val_steps_per_epoch)
    logger.info("build val done")

    dist_model.module.validate = True
    """
    Start validation
    """
    logger.info('start validation, {:d} pairs in total'.format(datareader.num_val))
    for epoch in range(num_checkpoints):
        logger.info('validating {:s}'.format(checkpoints[epoch].split(os.sep)[-1]))

        with torch.no_grad():
            seed_torch(10000)
            load_pretrain(model, checkpoints[epoch])

            end = time.time()
            for idx_val, val_data in enumerate(val_loader):
                data_time_val = average_reduce(time.time() - end)
                idx_val += 1

                val_outputs = dist_model(val_data)

                batch_time_val = time.time() - end
                batch_info_val = {}
                batch_info_val['batch_time'] = average_reduce(batch_time_val)
                batch_info_val['data_time'] = average_reduce(data_time_val)
                for k, v in sorted(val_outputs.items()):
                    batch_info_val[k] = average_reduce(v.data.item())

                average_meter_val.update(**batch_info_val)

                if rank == 0:
                    if idx_val == val_steps_per_epoch:
                        info = "Epoch: [{:d}], validation info:\n".format(epoch + 1)
                        for cc, (k, v) in enumerate(batch_info_val.items()):
                            if cc % 2 == 0:
                                info += "\t{:s}\t".format(getattr(average_meter_val, k))
                            else:
                                info += "{:s}\n".format(getattr(average_meter_val, k))
                        logger.info(info)

                end = time.time()


if __name__ == '__main__':
    seed_torch(args.seed)
    cv2.setNumThreads(0)
    cv2.ocl.setUseOpenCL(False)

    # load config
    tracker_name = args.tracker
    cfg = get_config(tracker_name)
    cfg.merge_from_file(args.config)

    """pytorch<1.9不支持在windows上分布式训练，强制单节点，worker数设为0"""
    run_mode = args.run_mode
    if SYSTEM == 'Windows' and TORCH_VERSION < 9:
        run_mode = 'debug'
    rank, world_size = dist_init(mode=run_mode)

    main()
