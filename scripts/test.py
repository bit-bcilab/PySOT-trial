# Copyright (c) SenseTime. All Rights Reserved.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals


import numpy as np
from trial.utils.bbox import get_axis_aligned_bbox, get_axis_aligned_bbox_new
from eval_toolkit.utils.region import vot_overlap, vot_float2str
from eval_toolkit.utils import success_overlap, success_error

import cv2
import os


def convert_bb_to_center(bboxes):
    return np.array([(bboxes[:, 0] + (bboxes[:, 2] - 1) / 2),
                     (bboxes[:, 1] + (bboxes[:, 3] - 1) / 2)]).T


def convert_bb_to_norm_center(bboxes, gt_wh):
    return convert_bb_to_center(bboxes) / (gt_wh+1e-16)


def test(tracker, name, dataset, test_video='', save_path='results', visual=False, test_name=''):
    total_lost = 0
    dataset_name = dataset.name
    base_name = dataset.base_name

    """
    假设从VOT2018中挑选出25个受参数影响较大的序列，组成新的测试集VOT2018-hard
    VOT2018为base dataset，VOT2018-hard为derive dataset
    save manner为 base 时，结果保存在results/VOT2018中（但是只有挑选出的25个序列的结果）；
    save manner为 derive 时，结果保存在results/VOT2018-hard中；
    save manner为 all 时，结果同时保存在results/VOT2018和results/VOT2018-hard中
    
    应用场景：
    1.挑选困难、参数敏感序列进行调参，选择 derive save方式
    2.LaSOT等速度很慢的超大型数据集，可将其分为若干个子集，分配到各个显卡上并行测试
      选择base的save方式，子数据集的结果会保存在同一文件夹中，最后组成完整的结果
    """
    if test_name != '':
        dataset_name_ = test_name + '-' + dataset_name
        base_name_ = test_name + '-' + base_name
    else:
        dataset_name_ = test_name
        base_name_ = test_name

    if dataset.save == 'base':
        save_name = [base_name_]
    elif dataset.save == 'derive':
        save_name = [dataset_name_]
    elif dataset.save == 'all' and dataset_name_ != base_name_:
        save_name = [dataset_name_, base_name_]

    # if dataset.name in ['VOT2016', 'VOT2018', 'VOT2019']:
    if 'VOT20' in base_name and 'VOT2018-LT' != base_name and 'VOT2020' != base_name:
        # restart tracking
        for v_idx, video in enumerate(dataset):
            if test_video != '':
                # test one special video
                if video.name != test_video:
                    continue
            frame_counter = 0
            lost_number = 0
            toc = 0
            pred_bboxes = []
            for idx, (img, gt_bbox) in enumerate(video):
                if len(gt_bbox) == 4:
                    gt_bbox = [gt_bbox[0], gt_bbox[1],
                               gt_bbox[0], gt_bbox[1] + gt_bbox[3] - 1,
                               gt_bbox[0] + gt_bbox[2] - 1, gt_bbox[1] + gt_bbox[3] - 1,
                               gt_bbox[0] + gt_bbox[2] - 1, gt_bbox[1]]
                tic = cv2.getTickCount()
                if idx == frame_counter:
                    if len(pred_bboxes) > 0 and pred_bboxes[-1] == 0:  #
                        cx, cy, w, h = get_axis_aligned_bbox_new(np.array(gt_bbox))[-1]
                        gt_bbox_ = [cx - (w - 1) / 2, cy - (h - 1) / 2, w, h]
                        tracker.init(img, gt_bbox_, restart=True)
                    else:
                        cx, cy, w, h = get_axis_aligned_bbox(np.array(gt_bbox))[-1]
                        if 'VOT2017-TIR' != base_name or 'SiamDCA' in name:
                            cx, cy, w, h = get_axis_aligned_bbox_new(np.array(gt_bbox))[-1]
                        gt_bbox_ = [cx - (w - 1) / 2, cy - (h - 1) / 2, w, h]
                        tracker.init(img, gt_bbox_)

                    pred_bbox = gt_bbox_
                    pred_bboxes.append(1)
                elif idx > frame_counter:
                    outputs = tracker.track(img)
                    pred_bbox = outputs['bbox']
                    overlap = vot_overlap(pred_bbox, gt_bbox, (img.shape[1], img.shape[0]))
                    if overlap > 0:
                        # not lost
                        pred_bboxes.append(pred_bbox)
                    else:
                        # lost object
                        pred_bboxes.append(2)
                        frame_counter = idx + 5  # skip 5 frames
                        lost_number += 1
                else:
                    pred_bboxes.append(0)
                toc += cv2.getTickCount() - tic

                if idx == 0:
                    cv2.destroyAllWindows()
                if visual and idx > frame_counter:
                    cv2.polylines(img, [np.array(gt_bbox, np.int).reshape((-1, 1, 2))],
                                  True, (0, 255, 0), 3)
                    bbox = list(map(int, pred_bbox))
                    cv2.rectangle(img, (bbox[0], bbox[1]),
                                  (bbox[0] + bbox[2], bbox[1] + bbox[3]), (0, 255, 255), 3)
                    cv2.putText(img, str(idx), (40, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
                    cv2.putText(img, str(lost_number), (40, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                    cv2.imshow(video.name, img)
                    k = cv2.waitKey(15) & 0xff
                    if k == 27:
                        cv2.destroyWindow(video.name)
                        break

            toc /= cv2.getTickFrequency()
            print('({:3d}) Video: {:12s} Time: {:4.1f}s Speed: {:3.1f}fps Lost: {:d}'.format(
                v_idx + 1, video.name, toc, idx / toc, lost_number))
            total_lost += lost_number

            # if visual and dataset.name != 'VOT2018-LT' and k != 27:
            #     tracker_traj = np.array(pred_bboxes[1:])
            #     gt_traj = np.array(video.gt_traj)
            #     n_frame = len(gt_traj)
            #     a_o = success_overlap(gt_traj[1:, :], tracker_traj, n_frame)
            #     thresholds = np.arange(0, 51, 1)
            #     gt_center = convert_bb_to_center(gt_traj)
            #     tracker_center = convert_bb_to_center(tracker_traj)
            #     a_p = success_error(gt_center[1:, :], tracker_center, thresholds, n_frame)
            #     print("precision: %.4f, AUC: %.4f" % (a_p[20], np.mean(a_o)))

            if save_path:

                for name_ in save_name:
                    # save results
                    video_path = os.path.join(save_path, name_, name, 'baseline', video.name)
                    if not os.path.isdir(video_path):
                        os.makedirs(video_path)
                    result_path = os.path.join(video_path, '{}_001.txt'.format(video.name))
                    with open(result_path, 'w') as f:
                        for x in pred_bboxes:
                            if isinstance(x, int):
                                f.write("{:d}\n".format(x))
                            else:
                                f.write(','.join([vot_float2str("%.4f", i) for i in x]) + '\n')

        print("{:s} total lost: {:d}".format(name, total_lost))

    else:
        # OPE tracking
        for v_idx, video in enumerate(dataset):
            if test_video != '':
                # test one special video
                if video.name != test_video:
                    continue
            toc = 0
            pred_bboxes = []
            scores = []
            track_times = []
            for idx, (img, gt_bbox) in enumerate(video):
                tic = cv2.getTickCount()
                if idx == 0:
                    # cx, cy, w, h = get_axis_aligned_bbox(np.array(gt_bbox))[0]
                    w, h = gt_bbox[2:]
                    cx = gt_bbox[0] + w / 2
                    cy = gt_bbox[1] + h / 2
                    gt_bbox_ = [cx - (w - 1) / 2, cy - (h - 1) / 2, w, h]
                    tracker.init(img, gt_bbox_)
                    pred_bbox = gt_bbox_
                    scores.append(None)
                    if 'VOT2018-LT' == base_name:
                        pred_bboxes.append([1])
                    else:
                        pred_bboxes.append(pred_bbox)
                else:
                    outputs = tracker.track(img)
                    pred_bbox = outputs['bbox']
                    pred_bboxes.append(pred_bbox)
                    scores.append(outputs['best_score'])
                toc += cv2.getTickCount() - tic
                track_times.append((cv2.getTickCount() - tic) / cv2.getTickFrequency())
                if idx == 0:
                    cv2.destroyAllWindows()
                if visual and idx > 0:
                    if len(gt_bbox) == 4 and not np.isnan(gt_bbox[0]) and gt_bbox[2] != 0. and gt_bbox[3] != 0.:
                        gt_bbox = list(map(int, gt_bbox))
                        cv2.rectangle(img, (gt_bbox[0], gt_bbox[1]),
                                      (gt_bbox[0] + gt_bbox[2], gt_bbox[1] + gt_bbox[3]), (0, 255, 0), 3)
                    else:
                        cv2.putText(img, 'NAN', (-40, -40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
                    pred_bbox = list(map(int, pred_bbox))
                    cv2.rectangle(img, (pred_bbox[0], pred_bbox[1]),
                                  (pred_bbox[0] + pred_bbox[2], pred_bbox[1] + pred_bbox[3]), (0, 255, 255), 3)
                    cv2.putText(img, str(idx), (40, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
                    cv2.imshow(video.name, img)
                    k = cv2.waitKey(15) & 0xff
                    if k == 27:
                        cv2.destroyWindow(video.name)
                        break

            toc /= cv2.getTickFrequency()
            print('({:3d}) Video: {:12s} Time: {:5.1f}s Speed: {:3.1f}fps'.format(
                v_idx + 1, video.name, toc, idx / toc))

            if visual and base_name != 'VOT2018-LT' and k != 27:
                tracker_traj = np.array(pred_bboxes)
                gt_traj = np.array(video.gt_traj)
                n_frame = len(gt_traj)
                a_o = success_overlap(gt_traj, tracker_traj, n_frame)
                thresholds = np.arange(0, 51, 1)
                gt_center = convert_bb_to_center(gt_traj)
                tracker_center = convert_bb_to_center(tracker_traj)
                a_p = success_error(gt_center, tracker_center, thresholds, n_frame)
                print("precision: %.4f, AUC: %.4f" % (a_p[20], np.mean(a_o)))

            if save_path:
                # save results
                for name_ in save_name:
                    if 'VOT2018-LT' == base_name:
                        video_path = os.path.join(save_path, name_, name, 'longterm', video.name)
                        if not os.path.isdir(video_path):
                            os.makedirs(video_path)
                        result_path = os.path.join(video_path, '{}_001.txt'.format(video.name))
                        with open(result_path, 'w') as f:
                            for x in pred_bboxes:
                                f.write(','.join([str(i) for i in x]) + '\n')
                        result_path = os.path.join(video_path,'{}_001_confidence.value'.format(video.name))
                        with open(result_path, 'w') as f:
                            for x in scores:
                                f.write('\n') if x is None else f.write("{:.6f}\n".format(x))
                        result_path = os.path.join(video_path,'{}_time.txt'.format(video.name))
                        with open(result_path, 'w') as f:
                            for x in track_times:
                                f.write("{:.6f}\n".format(x))
                    elif 'GOT-10k' == base_name:
                        video_path = os.path.join(save_path, name_, name, video.name)
                        if not os.path.isdir(video_path):
                            os.makedirs(video_path)
                        result_path = os.path.join(video_path, '{}_001.txt'.format(video.name))
                        with open(result_path, 'w') as f:
                            for x in pred_bboxes:
                                f.write(','.join([str(i) for i in x]) + '\n')
                        result_path = os.path.join(video_path, '{}_time.txt'.format(video.name))
                        with open(result_path, 'w') as f:
                            for x in track_times:
                                f.write("{:.6f}\n".format(x))
                    else:
                        model_path = os.path.join(save_path, name_, name)
                        if not os.path.isdir(model_path):
                            os.makedirs(model_path)
                        result_path = os.path.join(model_path, '{}.txt'.format(video.name))
                        with open(result_path, 'w') as f:
                            for x in pred_bboxes:
                                f.write(','.join([str(i) for i in x]) + '\n')
