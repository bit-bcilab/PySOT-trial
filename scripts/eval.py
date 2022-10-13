

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import numpy as np
from tqdm import tqdm
from multiprocessing import Pool
import os
from os import listdir
from glob import glob
import argparse

from eval_toolkit.datasets import DatasetFactory
from eval_toolkit.evaluation import OPEBenchmark, AccuracyRobustnessBenchmark, EAOBenchmark, F1Benchmark
from eval_toolkit.visualization import draw_f1, draw_eao, draw_success_precision
from configs.DataPath import SYSTEM, get_root

parser = argparse.ArgumentParser(description='tracking evaluation')
# parser.add_argument('--dataset', '-d', default='LaSOT', type=str, help='dataset name')
# parser.add_argument('--dataset', '-d', default='OTB100', type=str, help='dataset name')
# parser.add_argument('--dataset', '-d', default='VOT2018', type=str, help='dataset name')
parser.add_argument('--dataset', '-d', default='GOT-10k', type=str, help='dataset name')
# parser.add_argument('--dataset', '-d', default='NFS30', type=str, help='dataset name')
# parser.add_argument('--dataset', '-d', default='ITB', type=str, help='dataset name')
# parser.add_argument('--dataset', '-d', default='UAV', type=str, help='dataset name')
parser.add_argument('--num', '-n', default=4, type=int, help='number of thread to evaluate')
parser.add_argument('--show_video_level', '-s', dest='show_video_level', action='store_true')
parser.add_argument('--save', default='base', type=str, help='save manner')
parser.add_argument('--save_path', default='results', type=str, help='save path')
parser.set_defaults(show_video_level=False)
args = parser.parse_args()


def main():
    dataset = DatasetFactory.create_dataset(name=args.dataset, dataset_root=get_root(args.dataset), load_img=False)
    dataset.save = args.save

    dataset_name = dataset.name
    base_name = dataset.base_name
    if dataset.save == 'base' or dataset.save == 'all':
        save_name = base_name
    elif dataset.save == 'derive':
        save_name = dataset_name

    tracker_dir = os.path.join(args.save_path, save_name)

    # 在debug模式下运行，方便找出对比算法结果中的异常box文件以及在调参时取出中间结果
    # trackers = ['DaSiamRPN', 'Ocean-off', 'SiamCAR', 'SiamBAN', 'SiamFC++', 'SiamGAT', 'ATOM', 'DiMP-50']
    # trackers = ['TransT-baseline', 'TransT-trial']
    trackers = listdir(os.path.join(args.save_path, save_name))

    assert len(trackers) > 0
    args.num = min(args.num, len(trackers))
    dataset.set_tracker(tracker_dir, trackers)

    if 'VOT20' in base_name and not 'VOT2018-LT' in base_name:
        ar_benchmark = AccuracyRobustnessBenchmark(dataset)
        benchmark = EAOBenchmark(dataset, tags=dataset.tags)

        # a = ar_benchmark.eval(trackers)
        # b = benchmark.eval(trackers)
        # ar_benchmark.show_result(a, b, show_video_level=args.show_video_level)
        # draw_eao(b)

        ar_result = {}
        with Pool(processes=args.num) as pool:
            for ret in tqdm(pool.imap_unordered(ar_benchmark.eval,
                                                trackers), desc='evaluate ar', total=len(trackers), ncols=100):
                ar_result.update(ret)
        # benchmark.show_result(ar_result)

        eao_result = {}
        with Pool(processes=args.num) as pool:
            for ret in tqdm(pool.imap_unordered(benchmark.eval,
                                                trackers), desc='evaluate eao', total=len(trackers), ncols=100):
                eao_result.update(ret)
        # benchmark.show_result(eao_result)

        ar_benchmark.show_result(ar_result, eao_result, show_video_level=args.show_video_level, show_num=50)
        draw_eao(eao_result)

    elif ('OTB' in base_name) or ('UAV' in base_name) or ('NFS' in base_name) or ('GOT-10k' in base_name):
        benchmark = OPEBenchmark(dataset)

        # # 检查问题出现在哪个tracker的哪些序列上
        # ret = benchmark.eval_success(trackers)
        # pret = benchmark.eval_precision(trackers)

        success_ret = {}
        with Pool(processes=args.num) as pool:
            for ret in tqdm(pool.imap_unordered(benchmark.eval_success,
                                                trackers), desc='evaluate success', total=len(trackers), ncols=100):
                success_ret.update(ret)
        precision_ret = {}
        with Pool(processes=args.num) as pool:
            for ret in tqdm(pool.imap_unordered(benchmark.eval_precision,
                                                trackers), desc='evaluate precision', total=len(trackers), ncols=100):
                precision_ret.update(ret)

        # 最大显示数调大，以便观察超参数搜索时的结果
        benchmark.show_result(success_ret, precision_ret, show_video_level=args.show_video_level, max_num=60)

        # 显示每个序列上的平均Overlap
        results = []
        trks = list(precision_ret.keys())
        videos = []
        for tracker in trks:
            result = []
            if not videos:
                videos = list(precision_ret[tracker].keys())
            for video in videos:
                result.append(np.mean(success_ret[tracker][video]))
            result = np.array(result).reshape((-1, 1))
            results.append(result)
        results = np.concatenate(results, axis=-1)

        for attr, videos in dataset.attr.items():
            if attr == 'ALL':
                draw_success_precision(success_ret,
                                       name=dataset_name,
                                       videos=videos,
                                       attr=attr,
                                       precision_ret=precision_ret,
                                       save=True, save_format='png')
            else:
                draw_success_precision(success_ret,
                                       name=dataset_name,
                                       videos=videos,
                                       attr=attr, title_fontsize=13,
                                       precision_ret=precision_ret,
                                       save=True, save_format='png')
            # draw_success_precision(success_ret,
            #                        name=dataset_name,
            #                        videos=videos,
            #                        attr=attr)

    elif 'LaSOT' in base_name:
        benchmark = OPEBenchmark(dataset)
        success_ret = {}
        # success_ret = benchmark.eval_success(trackers)
        with Pool(processes=args.num) as pool:
            for ret in tqdm(pool.imap_unordered(benchmark.eval_success,
                                                trackers), desc='evaluate success', total=len(trackers), ncols=100):
                success_ret.update(ret)
        precision_ret = {}
        with Pool(processes=args.num) as pool:
            for ret in tqdm(pool.imap_unordered(benchmark.eval_precision,
                                                trackers), desc='evaluate precision', total=len(trackers), ncols=100):
                precision_ret.update(ret)
        norm_precision_ret = {}
        with Pool(processes=args.num) as pool:
            for ret in tqdm(pool.imap_unordered(benchmark.eval_norm_precision,
                                                trackers), desc='evaluate norm precision', total=len(trackers), ncols=100):
                norm_precision_ret.update(ret)
        benchmark.show_result(success_ret, precision_ret, norm_precision_ret,
                              show_video_level=args.show_video_level)
        draw_success_precision(success_ret,
                               name=dataset_name,
                               videos=dataset.attr['ALL'],
                               attr='ALL',
                               precision_ret=precision_ret,
                               norm_precision_ret=norm_precision_ret, save=True, title_fontsize=12)

    elif 'ITB' in base_name:
        benchmark = OPEBenchmark(dataset)

        # # 检查问题出现在哪个tracker的哪些序列上
        # ret = benchmark.eval_success(trackers)
        # pret = benchmark.eval_precision(trackers)

        success_ret = {}
        with Pool(processes=args.num) as pool:
            for ret in tqdm(pool.imap_unordered(benchmark.eval_success,
                                                trackers), desc='evaluate success', total=len(trackers), ncols=100):
                success_ret.update(ret)
        precision_ret = {}
        with Pool(processes=args.num) as pool:
            for ret in tqdm(pool.imap_unordered(benchmark.eval_precision,
                                                trackers), desc='evaluate precision', total=len(trackers), ncols=100):
                precision_ret.update(ret)

        benchmark.show_result(success_ret, precision_ret, show_video_level=args.show_video_level, max_num=60)

        mIou_ret, mIou_scen = benchmark.eval_mIoU()
        benchmark.show_result_ITB(mIou_ret, mIou_scen, success_ret, precision_ret)
        a = 0

    elif 'VOT2018-LT' in base_name:
        benchmark = F1Benchmark(dataset)
        f1_result = {}
        with Pool(processes=args.num) as pool:
            for ret in tqdm(pool.imap_unordered(benchmark.eval,
                                                trackers), desc='evaluate f1', total=len(trackers), ncols=100):
                f1_result.update(ret)
        benchmark.show_result(f1_result, show_video_level=args.show_video_level)
        draw_f1(f1_result)


if __name__ == '__main__':
    main()
