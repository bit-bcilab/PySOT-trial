

import numpy as np
import torch

from configs.DataPath import get_root
from configs.get_config import get_config, Config
from eval_toolkit.datasets import DatasetFactory
from eval_toolkit.evaluation import OPEBenchmark, EAOBenchmark
from pysot.utils.log_helper import init_log, add_file_handler
from pysot.models.model.model_builder import build_model
from pysot.trackers.tracker_builder import build_tracker
from pysot.utils.model_load import load_pretrain
from pysot.models.backbone.repvgg import repvgg_model_convert
from scripts.test import test

import argparse
import os
import logging

logger = logging.getLogger('global')
parser = argparse.ArgumentParser(description='siamese tracking')

# parser.add_argument('--dataset', default='VOT2020', type=str, help='name of dataset')
# parser.add_argument('--dataset', default='GOT-10k', type=str, help='name of dataset')
parser.add_argument('--dataset', default='VOT2018', type=str, help='name of dataset')
# parser.add_argument('--dataset', default='LaSOT', type=str, help='name of dataset')

parser.add_argument('--tracker', default='TransT', type=str, help='config file')
parser.add_argument('--config', default='experiments/transt/config.yaml', type=str, help='config file')
parser.add_argument('--snapshot', default='E://pysot-trial-sub/weights/TransT-provided.pth', type=str, help='model name')

parser.add_argument('--gpu_id', default=1, type=int, help="gpu id")
parser.add_argument('--result_path', default='results', type=str, help='results path')  # 非tune模式时结果保存的文件夹
parser.add_argument('--save', default='base', type=str, help='save manner')  # 只在数据集中的一部分序列上进行测试时，选择保存结果文件的方式
parser.add_argument('--trk_cfg', default='', type=str, help='track config')  # 输入此次测试时想使用的跟踪参数
parser.add_argument('--test_name', default='', type=str, help='test name')  # 本次测试名，用于消融实验、参数实验等，如para-0.30-0.10

args = parser.parse_args()

torch.cuda.set_device(args.gpu_id)

"""
Train：
CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch --nproc_per_node=2 --master_port=2333 trial/train.py --tracker SiamBAN --config experiments/siamban/trial.yaml
torchrun --nproc_per_node=2 --master_port=10000 trial/train.py --tracker SiamCARM --config experiments/siamcarm/trial.yaml --log_name SiamCARM-trial
torchrun --nproc_per_node=2 --master_port=10000 trial/train.py --tracker TransT --config experiments/transt/trial.yaml --log_name TransT-trial
torchrun --nproc_per_node=2 --master_port=10000 trial/val.py --tracker SiamBAN --config experiments/siamban/trial.yaml --checkpoint snapshot/SiamBAN-test2/                                                         

Test (VOT2018等常规数据集)
python scripts/test_script.py --dataset GOT-10k-all --tracker SiamRPNpp --config experiments/siamrpnpp/config.yaml --snapshot snapshot/SiamRPNpp-trial/checkpoint_e20.pth --save base --test_name SiamRPNpp-trial --gpu_id 0
python scripts/test_script.py --dataset VOT2020 --tracker SiamBAN --config experiments/siamban/config.yaml --snapshot snapshot/para-0.30-0.15/checkpoint_e20.pth --save base --test_name SiamBAN-trial --gpu_id 0
python scripts/test_script.py --dataset VOT2018 --tracker SiamCARM --config experiments/siamcarm/config.yaml --snapshot snapshot/SiamCARM-trial/checkpoint_e20.pth --save base --test_name SiamCARM-trial --gpu_id 0

GOT-10k eval and upload
cd results/GOT-10k/
python scripts/got_report.py arg1 arg2  (分别为output_name, result_dir)
python scripts/got_report.py SiamBAN-test1 SiamBAN
At last, upload .zip GOT-10k evaluation server

VOT2020 eval
CUDA_VISIBLE_DEVICES=0 vot evaluate --workspace /home/-/vot_test/ siamcarm_test1
vot analysis --format json --workspace /home/-/vot_test/ siamcarm_test
"""


def visual_test_single(tracker, track_cfg, dataset, name='test'):
    """
    choose and test videos in a certain dataset:
    type the video name, print auc and precision score, press 'ESC' to break and type next one
    """
    cfg.TRACK.CONTEXT_AMOUNT = track_cfg.context_amount
    cfg.TRACK.WINDOW_INFLUENCE = track_cfg.window_influence
    cfg.TRACK.PENALTY_K = track_cfg.penalty_k
    cfg.TRACK.LR = track_cfg.size_lr
    cfg.TRACK.CONFIDENCE = track_cfg.confidence
    while True:
        test_video = input('video name: ')
        test(tracker, name, dataset, test_video=test_video, save_path='', visual=True)


def test_all(tracker, name, track_cfg, dataset, save_path='results', visual=False, test_name=''):
    cfg.TRACK.CONTEXT_AMOUNT = track_cfg.context_amount
    cfg.TRACK.WINDOW_INFLUENCE = track_cfg.window_influence
    cfg.TRACK.PENALTY_K = track_cfg.penalty_k
    cfg.TRACK.LR = track_cfg.size_lr
    cfg.TRACK.CONFIDENCE = track_cfg.confidence
    test(tracker, name, dataset, test_video='', save_path=save_path, visual=visual, test_name=test_name)
    results = evaluate(dataset, name, save_path, test_name=test_name)
    print('{:s} results: {:.4f}'.format(name, results))


def search(tracker, dataset, configs, save_path='results'):
    test_num = len(configs)
    for n in range(test_num):
        i, j, k, l, m = configs[n]
        track_cfg.context_amount = i
        track_cfg.window_influence = j
        track_cfg.penalty_k = k
        track_cfg.size_lr = l
        track_cfg.confidence = m
        name = '{:s}_ca-{:.4f}_wi-{:.4f}_pk-{:.4f}_lr-{:.4f}_cf-{:.4f}'.format(
            model_name, track_cfg.context_amount, track_cfg.window_influence,
            track_cfg.penalty_k, track_cfg.size_lr, track_cfg.confidence)
        test_all(tracker, name, track_cfg, dataset, save_path=save_path)


def grid_search(tracker, dataset, context_amount, penalty_k, window_influence, size_lr, confidence, save_path='results'):
    for i in context_amount:
        for j in window_influence:
            for k in penalty_k:
                for l in size_lr:
                    for m in confidence:
                        track_cfg = Config()
                        track_cfg.context_amount = i
                        track_cfg.window_influence = j
                        track_cfg.penalty_k = k
                        track_cfg.size_lr = l
                        track_cfg.confidence = m

                        tracker_name = '{:s}_ca-{:.4f}_wi-{:.4f}_pk-{:.4f}_lr-{:.4f}_cf-{:.4f}'.format(
                            model_name, track_cfg.context_amount, track_cfg.window_influence,
                            track_cfg.penalty_k, track_cfg.size_lr, track_cfg.confidence)
                        test_all(tracker, tracker_name, track_cfg, dataset, save_path=save_path)


def evaluate(dataset, tracker_name, result_path='results', test_name=''):
    if test_name == '':
        tracker_dir = os.path.join(result_path, save_name)
    else:
        tracker_dir = os.path.join(result_path, test_name + '-' + save_name)
    trackers = [tracker_name]
    dataset.set_tracker(tracker_dir, trackers)

    if 'VOT20' in args.dataset and 'VOT2020' not in args.dataset:
        benchmark = EAOBenchmark(dataset, tags=dataset.tags)
        results = benchmark.eval(trackers)
        eao = results[tracker_name]['all']
        return eao
    elif 'ITB' in args.dataset:
        benchmark = OPEBenchmark(dataset)
        mIou_ret, mIou_scen = benchmark.eval_mIoU()
        mIoU = np.mean(list(mIou_ret[tracker_name].values()))
        return mIoU
    else:
        benchmark = OPEBenchmark(dataset)
        success_ret = benchmark.eval_success(trackers)
        auc = np.mean(list(success_ret[tracker_name].values()))
        return auc


def obj(trial):
    track_cfg.context_amount = trial.suggest_uniform('context_amount', 0.45, 0.55)
    # track_cfg.context_amount = trial.suggest_uniform('context_amount', 0.45, 0.51)
    # track_cfg.context_amount = 0.5
    track_cfg.window_influence = trial.suggest_uniform('window_influence', 0.25, 0.60)
    # track_cfg.window_influence = trial.suggest_uniform('window_influence', 0.40, 0.60)
    # track_cfg.window_influence = 0.35
    track_cfg.penalty_k = trial.suggest_uniform('penalty_k', 0.02, 0.18)
    # track_cfg.penalty_k = trial.suggest_uniform('penalty_k', 0.08, 0.17)
    # track_cfg.penalty_k = 0.06
    track_cfg.size_lr = trial.suggest_uniform('scale_lr', 0.25, 0.60)
    # track_cfg.size_lr = trial.suggest_uniform('scale_lr', 0.25, 0.40)
    # track_cfg.size_lr = 0.30
    # track_cfg.confidence = trial.suggest_uniform('confidence', 0.0, 0.95)
    # track_cfg.confidence = trial.suggest_uniform('confidence', 0.5, 0.65)
    track_cfg.confidence = 0.

    name = '{:s}_ca-{:.4f}_wi-{:.4f}_pk-{:.4f}_lr-{:.4f}_cf-{:.4f}'.format(
        model_name, track_cfg.context_amount, track_cfg.window_influence,
        track_cfg.penalty_k, track_cfg.size_lr, track_cfg.confidence)

    test_all(tracker, name, track_cfg, dataset, save_path='tune_results', test_name=test_name)
    results = evaluate(dataset, name, result_path='tune_results', test_name=test_name)
    logger.info("{:s} Results: {:.3f}, context_amount: {:.7f}, window_influence: {:.7f}, penalty_k: {:.7f}, "
                "lr: {:.7f}, confidence: {:.7f}".format(model_name, results, track_cfg.context_amount, track_cfg.window_influence,
                                                        track_cfg.penalty_k, track_cfg.size_lr, track_cfg.confidence))
    return results


def tune():
    import optuna

    if not os.path.exists('tune_results/logs/'):
        os.makedirs('tune_results/logs/')
    init_log('global', logging.INFO)

    if test_name == '':
        add_file_handler('global', 'tune_results/logs/{:s}-tune-logs.txt'.format(model_name), logging.INFO)
        if not os.path.exists("{:s}-tune.db".format(model_name)):
            study = optuna.create_study(study_name="{:s}".format(model_name), direction='maximize',
                                        storage="sqlite:///{:s}-tune.db".format(model_name))
        else:
            study = optuna.load_study(study_name="{:s}".format(model_name), storage="sqlite:///{:s}-tune.db".format(model_name))
    else:
        add_file_handler('global', 'tune_results/logs/{:s}-{:s}-tune-logs.txt'.format(test_name, model_name), logging.INFO)
        if not os.path.exists("{:s}-{:s}-tune.db".format(test_name, model_name)):
            study = optuna.create_study(study_name="{:s}-{:s}".format(test_name, model_name), direction='maximize',
                                        storage="sqlite:///{:s}-{:s}-tune.db".format(test_name, model_name))
        else:
            study = optuna.load_study(study_name="{:s}-{:s}".format(test_name, model_name),
                                      storage="sqlite:///{:s}-{:s}-tune.db".format(test_name, model_name))
    study.optimize(obj, n_trials=10000)
    print('Best value: {} (params: {})\n'.format(study.best_value, study.best_params))


def set_cfg(trk_cfg, add_cfg):
    add_cfg = add_cfg.split(', ')
    for i in range(len(add_cfg)):
        new_cfg = add_cfg[i].split(': ')
        if new_cfg[0] == 'lr':
            new_cfg[0] = 'size_lr'
        setattr(trk_cfg, new_cfg[0], float(new_cfg[1]))
    return trk_cfg


if __name__ == '__main__':
    test_name = args.test_name
    tracker_name = args.tracker

    cfg = get_config(tracker_name)
    # load config
    cfg.merge_from_file(args.config)
    cfg.CUDA = torch.cuda.is_available() and cfg.CUDA
    # device = torch.device('cuda' if cfg.CUDA else 'cpu')

    # create model
    model = build_model(tracker_name, cfg)

    # load model
    if torch.cuda.is_available():
        model = load_pretrain(model, args.snapshot).cuda().eval()
    else:
        model = load_pretrain(model, args.snapshot, False).eval()

    model = repvgg_model_convert(model)

    # build trackers
    tracker = build_tracker(cfg, model)

    # dataset_ = 'VOT2018'
    dataset = DatasetFactory.create_dataset(name=args.dataset, dataset_root=get_root(args.dataset), load_img=False)
    dataset.save = args.save
    dataset_name = dataset.name
    base_name = dataset.base_name
    if dataset.save == 'base' or dataset.save == 'all':
        save_name = base_name
    elif dataset.save == 'derive':
        save_name = dataset_name
    model_name = tracker_name + '-' + args.snapshot.split('/')[-1].split('.')[0] + '-' + save_name
    print('test model name: {:s}'.format(model_name))

    track_cfg = Config()
    # track_cfg.context_amount = 0.4600
    # track_cfg.window_influence = 0.5311
    # track_cfg.penalty_k = 0.0969
    # track_cfg.size_lr = 0.5187
    # track_cfg.confidence = 0.0000
    track_cfg.context_amount = cfg.TRACK.CONTEXT_AMOUNT
    track_cfg.window_influence = cfg.TRACK.WINDOW_INFLUENCE
    track_cfg.penalty_k = cfg.TRACK.PENALTY_K
    track_cfg.size_lr = cfg.TRACK.LR
    track_cfg.confidence = cfg.TRACK.CONFIDENCE
    if args.trk_cfg != '':
        track_cfg = set_cfg(track_cfg, args.trk_cfg)

    """
    Test Mode 0: Visualized evaluation
    """
    # visual_test_single(tracker, track_cfg, dataset, name=tracker_name)

    """
    Test Mode 1: Tune
    """
    # tune()

    """
    Test Mode 2: Evaluate the performance of a trackers with corresponding config on the chosen dataset
    """
    # # 便于测试调参跟踪器
    # name = '{:s}_ca-{:.4f}_wi-{:.4f}_pk-{:.4f}_lr-{:.4f}_cf-{:.4f}'.format(
    #     model_name, track_cfg.context_amount, track_cfg.window_influence,
    #     track_cfg.penalty_k, track_cfg.size_lr, track_cfg.confidence)
    # test_all(tracker, name=name, track_cfg=track_cfg, dataset=dataset, save_path=args.result_path, test_name=test_name)

    # 可视化观察
    test_all(tracker, name=tracker_name, track_cfg=track_cfg, dataset=dataset, visual=True, save_path='')

    # # 遍历数据集并保存结果文件
    # test_all(tracker, name=model_name, track_cfg=track_cfg, dataset=dataset, save_path=args.result_path)

    """
    Test Mode 3: Evaluate the performance of a trackers with corresponding config on all datasets
    """
    # datasets = list(TEST_PATH.keys())
    # for dataset_ in datasets:
    #     dataset = DatasetFactory.create_dataset(name=dataset_, dataset_root=get_root(dataset_), load_img=False)
    #     test_all(tracker, name=args.tracker, track_cfg=track_cfg, dataset=dataset, save_path=args.result_path)

    """
    Test Mode 4: Hyper Parameters search
    """
    # configs = [[0.5182268, 0.4869404, 0.1597234, 0.4832103, 0.],
    #            [0.4617189, 0.4691246, 0.1464686, 0.4896833, 0.],
    #            [0.5239012, 0.4551365, 0.1499777, 0.4987685, 0.],
    #            [0.4644649, 0.4530294, 0.1491471, 0.4980676, 0.],
    #            [0.4568417, 0.4596867, 0.0540690, 0.4399954, 0.],
    #            [0.5053916, 0.4749539, 0.1322824, 0.4433954, 0.],
    #            [0.4952466, 0.4805237, 0.1342398, 0.4431467, 0.],
    #            [0.4530331, 0.4754278, 0.0596305, 0.4494217, 0.],
    #            [0.5161703, 0.4873081, 0.1513490, 0.4410120, 0.],
    #            [0.4976381, 0.4895505, 0.1586221, 0.3829806, 0.],
    #            [0.4703795, 0.4437768, 0.1550383, 0.5070977, 0.],
    #            [0.5177144, 0.4703943, 0.1555634, 0.4838249, 0.],
    #            [0.4602901, 0.4557632, 0.1421298, 0.4533330, 0.],
    #            [0.4549012, 0.4638315, 0.0604487, 0.4490459, 0.],
    #            [0.4711217, 0.4879679, 0.1411025, 0.4518217, 0.],
    #            [0.4577403, 0.4543049, 0.0542528, 0.4602919, 0.]]
    # search(tracker, dataset, configs[12:], args.result_path)

    """
    Test Mode 5: Grid search
    """
    # context_amount = [0.49, 0.50, 0.51]
    # window_influence = [0.35, 0.40]
    # penalty_k = [0.04, 0.06, 0.08]
    # size_lr = [0.35, 0.40]
    # confidence = [0.8, 0.9]
    # grid_search(tracker, dataset, context_amount, window_influence, penalty_k, size_lr, confidence, args.result_path)
