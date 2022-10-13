# Copyright (c) SenseTime. All Rights Reserved.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torch.nn.utils import clip_grad_norm_
from torch.cuda.amp import GradScaler
from torch.utils.data.distributed import DistributedSampler

from pysot.utils.lr_scheduler import build_lr_scheduler
from pysot.utils.log_helper import init_log, print_speed, add_file_handler
from pysot.utils.distributed import dist_init, DistModule, reduce_gradients, average_reduce, get_rank, get_world_size
from pysot.utils.model_load import load_pretrain, restore_from, resume
from pysot.utils.average_meter import AverageMeter
from pysot.utils.misc import describe, commit
from pysot.models.backbone.repvgg import repvgg_model_convert

from configs.DataPath import SYSTEM
from configs.get_config import get_config
from pysot.models.model.model_builder import build_model
from trial.Generator import Generator
from trial.DataReader import DataReader
from pysot.utils.anchor import Anchors
from pysot.utils.point import Point
from trial.encoders import get_encoder

import argparse
import logging
import os
import time
import math
import json
import yaml
import random
import cv2
import numpy as np

TORCH_VERSION = int(torch.__version__.split('.')[1])
logger = logging.getLogger('global')
parser = argparse.ArgumentParser(description='siamese tracking')
parser.add_argument('--run_mode', default='train', type=str, help='run mode')
parser.add_argument('--tracker', default='SiamBAN', type=str, help='config file')
parser.add_argument('--config', default='experiments/siamban/test-trial.yaml', type=str, help='config file')
parser.add_argument('--seed', type=int, default=10000, help='random seed')
parser.add_argument('--local_rank', type=int, default=0, help='compulsory for pytorch launcer')
parser.add_argument('--log_name', default='', type=str, help='name of log file')
args = parser.parse_args()


def is_valid_number(x):
    return not (math.isnan(x) or math.isinf(x) or x > 1e4)


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


def unfreeze_backbone(model, optimizer):
    backbone_params = []
    for layer in cfg.BACKBONE.TRAIN_LAYERS:
        for m in getattr(model.backbone, layer).modules():
            if isinstance(m, nn.BatchNorm2d):
                m.train()
        for param in getattr(model.backbone, layer).parameters():
            param.requires_grad = True
            backbone_params.append(param)
    optimizer.param_groups[0]['params'] = backbone_params
    return optimizer


def build_opt_lr(model):
    for param in model.backbone.parameters():
        param.requires_grad = False
    for m in model.backbone.modules():
        if isinstance(m, nn.BatchNorm2d):
            m.eval()

    trainable_params = []
    trainable_params += [{'params': filter(lambda x: x.requires_grad, model.backbone.parameters()),
                          'lr': cfg.BACKBONE.LAYERS_LR * cfg.TRAIN.BASE_LR}]

    if cfg.ADJUST.ADJUST:
        trainable_params += [{'params': model.neck.parameters(), 'lr': cfg.TRAIN.BASE_LR}]

    head_params = model.get_head_parameters()

    for i in range(len(head_params)):
        trainable_params += [{'params': head_params[i], 'lr': cfg.TRAIN.BASE_LR}]

    # optimizer = torch.optim.SGD(trainable_params, momentum=cfg.TRAIN.MOMENTUM, weight_decay=cfg.TRAIN.WEIGHT_DECAY)
    optimizer = torch.optim.AdamW(trainable_params, lr=cfg.TRAIN.BASE_LR, weight_decay=cfg.TRAIN.WEIGHT_DECAY)

    lr_scheduler = build_lr_scheduler(optimizer, cfg=cfg, epochs=cfg.TRAIN.EPOCH)
    return optimizer, lr_scheduler


def log_grads(model, tb_writer, tb_index):
    grad = {}
    weights = {}
    for name, param in model.named_parameters():
        if param.grad is not None:
            grad[name] = param.grad
            weights[name] = param.data

    feature_norm, head_norm = 0, 0
    for k, g in grad.items():
        _norm = g.data.norm(2)
        weight = weights[k]
        w_norm = weight.norm(2)
        if 'feature' in k:
            feature_norm += _norm ** 2
        else:
            head_norm += _norm ** 2

        tb_writer.add_scalar('grad_all/' + k.replace('.', '/'), _norm, tb_index)
        tb_writer.add_scalar('weight_all/' + k.replace('.', '/'), w_norm, tb_index)
        tb_writer.add_scalar('w-g/' + k.replace('.', '/'), w_norm / (1e-20 + _norm), tb_index)
    tot_norm = feature_norm + head_norm
    tot_norm = tot_norm ** 0.5
    feature_norm = feature_norm ** 0.5
    head_norm = head_norm ** 0.5

    tb_writer.add_scalar('grad/tot', tot_norm, tb_index)
    tb_writer.add_scalar('grad/feature', feature_norm, tb_index)
    tb_writer.add_scalar('grad/head', head_norm, tb_index)


def check_grads(model, max_num=10):
    skip_gd = False
    nan_num = 0

    for name, param in model.named_parameters():
        """avoid gradient explosion"""
        if param.grad is not None:
            if torch.any(torch.isnan(param.grad)):
                nan_num += 1

    if nan_num > 0:
        if nan_num < max_num:
            for name, param in model.named_parameters():
                """avoid gradient explosion"""
                if param.grad is not None:
                    clip_grad = param.grad.data.detach()
                    if torch.any(torch.isnan(clip_grad)):
                        # clip_grad_ = torch.where(torch.isnan(clip_grad) | torch.isinf(clip_grad),
                        #                          1e-9 * torch.ones_like(clip_grad), clip_grad)
                        clip_grad_ = torch.where(torch.isnan(clip_grad), torch.zeros_like(clip_grad), clip_grad)
                        param.grad.data = clip_grad_.detach()
                        logger.info('{:d} NAN Grads, stop gradient descent.'.format(nan_num))
        else:
            skip_gd = True

    return skip_gd, nan_num


def main():
    if get_rank() == 0:
        if not os.path.exists(cfg.TRAIN.SNAPSHOT_DIR):
            os.makedirs(cfg.TRAIN.SNAPSHOT_DIR)
        if not os.path.exists(cfg.TRAIN.LOG_DIR):
            os.makedirs(cfg.TRAIN.LOG_DIR)

        init_log('global', logging.INFO)
        if args.log_name != '':
            log_dir = os.path.join(cfg.TRAIN.LOG_DIR, args.log_name + '-logs.txt')
        else:
            log_dir = os.path.join(cfg.TRAIN.LOG_DIR, 'logs.txt')

        add_file_handler('global', os.path.join(log_dir), logging.INFO)
        logger.info("Version Information: \n{}\n".format(commit()))

        logger.info("config \n{}".format(json.dumps(cfg, indent=4)))
        logger.info('torch version: {:d}, run mode: {:s}'.format(TORCH_VERSION, run_mode))

    logger.info("load hyp setting from {:s}".format(cfg.TRIAL_CFG))
    with open(os.path.join(cfg.TRIAL_CFG), 'r', encoding='utf-8') as f:
        cond = f.read()
        trial_settings = yaml.load(cond, Loader=yaml.FullLoader)
        f.close()
    logger.info("init done")

    logger.info("model prepare")
    # create model
    model = build_model(tracker_name, cfg).cuda().train()

    if args.tracker == 'SiamDCA':
        for param in model.fpn.parameters():
            param.requires_grad = False
        for m in model.fpn.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()
        load_pretrain(model.fpn, 'pretrained_models/siamdca_fpn.pth')

    # only load backbone
    if cfg.BACKBONE.PRETRAINED:
        load_pretrain(model.backbone, cfg.BACKBONE.PRETRAINED)
    # load whole model
    if cfg.TRAIN.PRETRAINED:
        load_pretrain(model, cfg.TRAIN.PRETRAINED)
    if 'RepVGG' in cfg.BACKBONE.TYPE and cfg.BACKBONE.TRAIN_EPOCH >= cfg.TRAIN.EPOCH:
        model.backbone = repvgg_model_convert(model.backbone)
        logger.info("compress RepVGG blocks in backbone.")

    # build optimizer and lr_scheduler
    optimizer, lr_scheduler = build_opt_lr(model)
    logger.info(lr_scheduler)
    logger.info("model prepare done")

    logger.info("build train dataset")
    datareader = DataReader(data_settings=trial_settings['DATA_SETTINGS'], num_epoch=cfg.TRAIN.EPOCH)
    points = Point(cfg.POINT.STRIDE, cfg.TRAIN.OUTPUT_SIZE, cfg.TRAIN.SEARCH_SIZE // 2).points
    model.points = torch.from_numpy(points[None, ...]).cuda()
    model.weights = trial_settings['WEIGHT_SETTINGS']
    model.ti = trial_settings['UPDATE_SETTINGS']['neg_iou_thresh']
    model.ts = trial_settings['UPDATE_SETTINGS']['neg_score_thresh']
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

    train_kwargs = dict(datareader=datareader,
                        base=cfg.BASE,
                        mode='train',
                        encoder=get_encoder(trial_settings['ENCODER']),
                        search_size=[cfg.TRAIN.SEARCH_SIZE, cfg.TRAIN.SEARCH_SIZE],
                        template_size=[cfg.TRAIN.EXEMPLAR_SIZE, cfg.TRAIN.EXEMPLAR_SIZE],
                        output_size=[cfg.TRAIN.OUTPUT_SIZE, cfg.TRAIN.OUTPUT_SIZE],
                        crop_settings=trial_settings['CROP_SETTINGS'],
                        aug_settings=trial_settings['AUG_SETTINGS'],
                        encode_settings=trial_settings['ENCODE_SETTINGS'],
                        bbox_mask_rate=trial_settings['BBOX_MASK_RATE'],
                        use_all_boxes=trial_settings['MIX_BOXES'])
    train_dataset = Generator(**train_kwargs)
    train_dataset.points = points
    # build dataset loader
    train_loader = build_data_loader(cfg, train_dataset, run_mode=run_mode)
    logger.info("build dataset done")

    if cfg.VALIDATE:
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
        logger.info("build val done")
    else:
        val_loader = None

    restart = False
    resume_path = cfg.TRAIN.RESUME
    val_infos = None
    while True:
        stop_epoch, val_infos = train(model, optimizer, lr_scheduler, train_loader, trial_settings,
                                      val_loader=val_loader, resume_path=resume_path, restart=restart,
                                      max_failure=30, failure_steps=6, val_infos=val_infos)
        if stop_epoch < (cfg.TRAIN.EPOCH - 1):
            if stop_epoch > 0:
                resume_path = cfg.TRAIN.SNAPSHOT_DIR + '/checkpoint_e%d.pth' % stop_epoch
            restart = True
            logger.info('Fail in {:d} epoch. Restart.'.format(stop_epoch + 1))
        else:
            break
    logger.info('Train Finish after {:d} epochs'.format(stop_epoch + 1))


def train(model, optimizer, lr_scheduler, train_loader, trial_settings,
          val_loader=None, resume_path=None, restart=False, failure_steps=4, max_failure=20, val_infos=None):
    if get_rank() == 0:
        # create tensorboard writer
        tb_writer = SummaryWriter(cfg.TRAIN.LOG_DIR) if cfg.TRAIN.LOG_DIR else None
    scaler = GradScaler() if AMP else None
    average_meter = AverageMeter()
    steps_per_epoch = math.ceil(train_loader.dataset.datareader.num_per_epoch / (cfg.TRAIN.BATCH_SIZE * world_size))

    if cfg.VALIDATE:
        if val_infos is None:
            val_infos = []
        val_steps_per_epoch = math.ceil(val_loader.dataset.datareader.num_val / (cfg.TRAIN.BATCH_SIZE * world_size))
        average_meter_val = AverageMeter(num=val_steps_per_epoch)

    start_epoch = cfg.TRAIN.START_EPOCH
    # resume training
    if resume_path:
        assert os.path.isfile(resume_path), '{} is not a valid file.'.format(resume_path)
        resume_ckpt = torch.load(resume_path, map_location=lambda storage, loc: storage.cuda(torch.cuda.current_device()))
        start_epoch = resume_ckpt['epoch']
    if start_epoch > cfg.BACKBONE.TRAIN_EPOCH:
        unfreeze_backbone(model, optimizer)
    if resume_path:
        model, optimizer, lr_scheduler = resume(model, resume_ckpt, optimizer, lr_scheduler)
        logger.info('start epoch: {:d}, resume from: {:s}'.format(start_epoch + 1, resume_path))
        del resume_ckpt

    dist_model = DistModule(model)
    self_update = False
    """
    Start training
    """
    for epoch in range(start_epoch, cfg.TRAIN.EPOCH):
        dist_model.train()
        logger.info('epoch: {}'.format(epoch + 1))
        past_steps = steps_per_epoch * epoch
        model.train_epoch = epoch + 1
        nan_steps = 0
        num_failures = 0

        """每个批次的训练数据轮换"""
        train_loader.dataset.datareader.update_index(epoch)
        if epoch == start_epoch and restart:
            seed = args.seed + random.randint(0, 1000)
            seed_torch(seed)
            logger.info("Restart in epoch {:d}, use seed: {:d}".format(epoch + 1, seed))
            train_loader.dataset.datareader.train_index_ = train_loader.dataset.datareader.build_train_index()
        else:
            "每个epoch使用确定的、相同的随机种子seed，确保resume时使用的seed一致"
            seed = args.seed + epoch
            seed_torch(seed)
            logger.info("Train in epoch {:d}, use seed: {:d}".format(epoch + 1, seed))

        if cfg.BACKBONE.TRAIN_EPOCH == epoch:
            logger.info('start training backbone.')
            unfreeze_backbone(model, optimizer)

        """
        Assignment Phase 2:  
        """
        if not self_update and (epoch == cfg.SELF_EPOCH or start_epoch >= cfg.SELF_EPOCH):
            train_loader.dataset.use_all_boxes = False
            train_loader.dataset.encode_settings = trial_settings['ENCODE_SETTINGS_SELF']
            train_loader.dataset.encoder = get_encoder(trial_settings['SELF_ENCODER'])
            model.update_settings = trial_settings['UPDATE_SETTINGS']
            model.weights = trial_settings['WEIGHT_SETTINGS_SELF']
            logger.info("Train Phase 2")
            self_update = True

        if epoch > cfg.TRAIN.LR_WARMUP.EPOCH:
            train_loader.dataset.crop_settings['search']['min_scale_'] = 0.70
            train_loader.dataset.crop_settings['search']['max_scale_'] = 1.40

        for index, pg in enumerate(optimizer.param_groups):
            logger.info('epoch {} lr {}'.format(epoch + 1, pg['lr']))
            if get_rank() == 0:
                tb_writer.add_scalar('lr/group{}'.format(index + 1), pg['lr'], past_steps)

        end = time.time()
        for idx, data in enumerate(train_loader):
            idx += 1
            global_step = idx + past_steps
            data_time = average_reduce(time.time() - end)

            if get_rank() == 0:
                tb_writer.add_scalar('time/data', data_time, global_step)

            outputs = dist_model(data)
            loss = outputs['total_loss']

            if is_valid_number(loss.data.item()):
                optimizer.zero_grad()
                if AMP:
                    scaler.scale(loss).backward()

                    if get_rank() == 0 and cfg.TRAIN.LOG_GRADS:
                        log_grads(dist_model.module, tb_writer, global_step)
                    # clip gradient
                    clip_grad_norm_(dist_model.parameters(), cfg.TRAIN.GRAD_CLIP)

                    skip_gd, nan_num = check_grads(dist_model.module, max_num=0)
                    if not skip_gd:
                        scaler.step(optimizer)
                        scaler.update()
                    else:
                        logger.info('{:d} NAN Grads in epoch {:d}, step {:d}, skip gradient descent.'.format(nan_num, epoch + 1, idx))
                        nan_steps += 1
                        if idx % cfg.TRAIN.PRINT_FREQ == 0:
                            if nan_steps > failure_steps:
                                num_failures += 1
                            nan_steps = 0
                    if num_failures > max_failure:
                        return epoch, val_infos
                else:
                    loss.backward()

                    reduce_gradients(dist_model)

                    if get_rank() == 0 and cfg.TRAIN.LOG_GRADS:
                        log_grads(dist_model.module, tb_writer, global_step)
                    # clip gradient
                    clip_grad_norm_(dist_model.parameters(), cfg.TRAIN.GRAD_CLIP)

                    skip_gd, nan_num = check_grads(dist_model.module, max_num=0)
                    if not skip_gd:
                        optimizer.step()
                    else:
                        logger.info('{:d} NAN Grads in epoch {:d}, step {:d}, skip gradient descent.'.format(nan_num, epoch + 1, idx))
                        nan_steps += 1
                        if idx % cfg.TRAIN.PRINT_FREQ == 0:
                            if nan_steps > failure_steps:
                                num_failures += 1
                            nan_steps = 0
                    if num_failures > max_failure:
                        return epoch, val_infos
            else:
                logger.info('NAN Loss in epoch {:d}, step {:d}'.format(epoch + 1, idx))
                return epoch, val_infos

            batch_time = time.time() - end
            batch_info = {}
            batch_info['batch_time'] = average_reduce(batch_time)
            batch_info['data_time'] = average_reduce(data_time)
            for k, v in sorted(outputs.items()):
                batch_info[k] = average_reduce(v.data.item())

            average_meter.update(**batch_info)

            if get_rank() == 0:
                for k, v in batch_info.items():
                    tb_writer.add_scalar(k, v, global_step)

                if idx % cfg.TRAIN.PRINT_FREQ == 0 or idx == steps_per_epoch:
                    cur_lr = optimizer.state_dict()['param_groups'][-1]['lr']
                    info = "Epoch: [{}][{}/{}] lr: {:.6f}\n".format(epoch + 1, idx, steps_per_epoch, cur_lr)
                    for cc, (k, v) in enumerate(batch_info.items()):
                        if cc % 2 == 0:
                            info += "\t{:s}\t".format(getattr(average_meter, k))
                        else:
                            info += "{:s}\n".format(getattr(average_meter, k))
                    logger.info(info)
                    print_speed(global_step, average_meter.batch_time.avg, cfg.TRAIN.EPOCH * steps_per_epoch)
            end = time.time()

        if cfg.VALIDATE and train_loader.dataset.datareader.num_val > 0:
            with torch.no_grad():
                model.cfg.AMP = False
                dist_model.eval()
                logger.info('start validation, {:d} pairs in total'.format(val_loader.dataset.datareader.num_val))
                seed_torch(10000)
                model.validate = True

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
                            num_val = len(val_infos)
                            if num_val > 0:
                                for i in range(num_val):
                                    logger.info(val_infos[i])
                            info = "Epoch: [{:d}], validation info:\n".format(epoch + 1)
                            for cc, (k, v) in enumerate(batch_info_val.items()):
                                if cc % 2 == 0:
                                    info += "\t{:s}\t".format(getattr(average_meter_val, k))
                                else:
                                    info += "{:s}\n".format(getattr(average_meter_val, k))
                            logger.info(info)
                            val_infos.append(info)

                    end = time.time()

                model.validate = False
                model.cfg.AMP = AMP

        if epoch < cfg.TRAIN.EPOCH - 1:
            average_meter.reset()
            lr_scheduler.step()
        if get_rank() == 0:
            torch.save(
                {'epoch': epoch + 1,
                 'state_dict': dist_model.module.state_dict(),
                 'optimizer': optimizer.state_dict(),
                 'scheduler': lr_scheduler.state_dict()},
                cfg.TRAIN.SNAPSHOT_DIR + '/checkpoint_e%d.pth' % (epoch + 1))

    return epoch, val_infos


if __name__ == '__main__':
    seed_torch(args.seed)
    cv2.setNumThreads(0)
    cv2.ocl.setUseOpenCL(False)

    # load config
    tracker_name = args.tracker
    cfg = get_config(tracker_name)
    cfg.merge_from_file(args.config)

    """pytorch<1.9不支持在windows上分布式训练，强制单节点，worker数设为0"""
    AMP = cfg.AMP
    run_mode = args.run_mode
    if SYSTEM == 'Windows' and TORCH_VERSION < 9:
        run_mode = 'debug'
    rank, world_size = dist_init(mode=run_mode)

    main()
