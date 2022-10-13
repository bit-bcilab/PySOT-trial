

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import os
from os import listdir
from glob import glob
import argparse

from eval_toolkit.datasets import DatasetFactory
from eval_toolkit.evaluation import OPEBenchmark, EAOBenchmark
from eval_toolkit.visualization.draw_success_precision import draw
from configs.DataPath import get_root, SYSTEM

parser = argparse.ArgumentParser(description='tracking evaluation')
# parser.add_argument('--dataset', '-d', default='LaSOT', type=str, help='dataset name')
# parser.add_argument('--dataset', '-d', default='OTB100', type=str, help='dataset name')
# parser.add_argument('--dataset', '-d', default='VOT2018', type=str, help='dataset name')
parser.add_argument('--dataset', '-d', default='GOT-10k', type=str, help='dataset name')
args = parser.parse_args()


def main():
    dataset = DatasetFactory.create_dataset(name=args.dataset, dataset_root=get_root(args.dataset), load_img=False)
    base_name = dataset.base_name
    tracker_dir = os.path.join('results', base_name)

    trackers = []  # only observe and draw GT
    # trackers = ['SiamDCA', 'SiamFC++', 'ATOM', 'DiMP-50', 'Ocean-off']
    # trackers = ['TransT-baseline', 'TransT-trial']  # 'SiamCARM-baseline', 'SiamCARM-trial''SiamBAN-baseline', 'SiamBAN-trial''SiamRPNpp-baseline', 'SiamRPNpp-trial'
    # trackers = listdir('results/' + base_name)  # draw all the trackers that are in the 'result' dir

    dataset.set_tracker(tracker_dir, trackers)

    if 'VOT20' in args.dataset and 'VOT2020' not in args.dataset:
        benchmark = EAOBenchmark(dataset, tags=dataset.tags)
    else:
        benchmark = OPEBenchmark(dataset)

    videos = list(dataset.videos.keys())
    videos.sort()
    for test_video in videos:
        # 检验、观察GT框，应设置trackers = []
        draw(dataset=benchmark.dataset, dataset_name=dataset.base_name, video_name=test_video,
             eval_trackers=None, draw_gt=True, save=False, wait_key=20)

        # draw(dataset=benchmark.dataset, dataset_name=dataset.base_name, video_name=test_video,
        #      eval_trackers=trackers, draw_gt=False, save=False, wait_key=40)

    while True:
        test_video = input('video name: ')
        if test_video in dataset.videos:
            # draw(dataset=benchmark.dataset, dataset_name=dataset.base_name, video_name=test_video,
            #      eval_trackers=trackers, draw_gt=False, wait_key=40)
            draw(dataset=benchmark.dataset, dataset_name=dataset.base_name, video_name=test_video,
                 eval_trackers=trackers, draw_gt=False, wait_key=1, save=True, width=9, font=5.5)  # , width=4


def response_map(a):
    import cv2
    import matplotlib.pyplot as plt

    b = cv2.resize(a, (64, 64))
    plt.matshow(b)
    plt.axis('off')
    plt.savefig('girl2-108-baseline.png', bbox_inches='tight', pad_inches=0)


if __name__ == '__main__':
    main()
