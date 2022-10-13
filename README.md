
# PySOT-trial
Official implementation of the *Negative-Driven Training Pipeline For Siamese Visual Tracking*.

This work proposes a tracking-specific and negative-driven training pipeline to improve the performances
of Siamese trackers (and Transformer trackers) without any extra inference cost.

## Acknowledgement and Introduction
This work can be viewed as an extension of the [PySOT](https://github.com/STVIR/pysot), one of the most popular 
open source frameworks for single object tracking. 
PySOT has provided the training and test codes and setting files of the SiamRPNpp and SiamMask.

Furthermore, we collect and integrate four representative and advanced Siamese trackers in our project, including 
SiamAttn, 
SiamCAR, 
SiamBAN and 
SiamGAT, which are all implemented based on PySOT.

In addition, we transplant and reproduce the TransT (a Transformer tracker) in our project.

We would like to greatly appreciate the above outstanding works and their authors for providing the frameworks and toolkits.

We rewrite all the codes for training, including *DataReader*, *Generator*, *Region Crop*, *Loss Function*, *Augmentation* and *Box Decoder & Encoder (with Assignment)*, 
and these codes are mainly put in the '*trial*' directory.

Accordingly, the `forward()` method of the `Tracker Class` in `pysot/models/model/` are modified. 


## Installation
This project is based on the Pytorch deep learning API.
By and large, the codes can run both on the Linux and Windows system.

Please refer to the installation instructions in [INSTALL.md](INSTALL.md).

After finish the environment configuration, modify the paths of training/test datasets in [configs/DataPath.py](configs/DataPath.py)

## Test
Refer to the [scripts/test_script.py](scripts/test_script.py) to set test configuration and running modes.

Same as the PySOT, the '.json' label files need to put in the directories of test datasets, 
which can be downloaded from [here](https://drive.google.com/drive/folders/1eiyfUh9pD6K9eXxrm78uMh93VDIVyc2f?usp=sharing) or [PySOT](https://github.com/STVIR/pysot).

The weights can be found at [here](https://drive.google.com/drive/folders/111enwBcLE2sFd9Gf5OX5f_VW9PSUq_wh?usp=sharing),
where '-trial', '-baseline' and '-provided' represent the baseline weights, improved weights
and the weights provided by the original authors in the corresponding projects.


## Evaluation and Plotting
Evaluation and curve plotting are in [scripts/eval.py](scripts/eval.py).

And can visualize and draw bounding boxes and response maps in [scripts/draw.py](scripts/draw.py).

## Training
Download the '.json' labels of training datasets and pretrained backbones from 
[here](https://drive.google.com/drive/folders/1dchB2QjxucTbq9Yagsh3GOttjXsyCn-o?usp=sharing) 
and [here](https://drive.google.com/drive/folders/1GM-guDQHrOCMsqMXFgjcoPOuHcHZHNcj?usp=sharing), 
and then put them in `json_labels_clean` and `pretrained_models` directory.

Run [trial/train.py](trial/train.py).
DDP Example:
```
torchrun --nproc_per_node=2 --master_port=10000 trial/train.py --tracker SiamCARM --config experiments/siamcarm/trial.yaml --log_name SiamCARM-trial
```

The original training script of PySOT is still kept in the `scripts` directory, and able to run.

## VOT2020 Evaluation

Tutorial: 

https://www.votchallenge.net/howto/tutorial_python.html

https://zhuanlan.zhihu.com/p/314117415

***Important!***

测试中发现使用vot工具包import时存在无法import项目中函数的问题

必须将实验跟踪器的脚本文件放在项目第一级路径下才能正常import项目代码中的函数，如
```
|-- /home/user/
    |-- pysot_trial (project)
        |-- eval_toolkit
            |-- vot20
                ...
        |-- siamban_test2.py (实验跟踪器脚本，需要写清楚参数)
        
    |-- vot_test (work space)
        |-- trackers.ini (跟踪器配置文件，必须将要测试的跟踪器写入)
        |-- config.yaml (测试工具配置文件，主要是stack参数)
        |-- sequences (手动将序列复制到此)
            |-- hand
                ...
        |-- results (保存结果的文件夹)
            |-- siamban_test2
                ...
        |-- logs (测试由于异常停止时，可以在log里查到程序在哪报错了)
```

##### 1. Install:
* 由于github被墙，可能不太稳定，需要多试几次

建议去github项目网页上找到 requirements.txt，提前将其中的库安装好，以加快安装

* 建议在项目虚拟环境下安装vot工具包，但好像也可以单独建立一个环境，然后在trackers.ini中设置使用哪个虚拟环境下的python解释器
```bash
# install git
(pysot)user@node01:~$ conda install git

# install vot toolkit by git
(pysot)user@node01:~$ pip install git+https://github.com/votchallenge/vot-toolkit-python -i https://pypi.tuna.tsinghua.edu.cn/simple/
```

##### 2. Initialize work space:
命令格式：

vot initialize 测试集名称(如vot2019,vot2020) --workspace 测试序列与结果存放文件夹

`需要提前新建一个空白文件夹作为work space，建议至少和项目目录同级`

Example:
```bash
vot initialize vot2020 --workspace /home/user/vot_test
```

##### 3. Copy dataset:
初始化work space和每一次test/evaluate之前，程序会自动检测work space下的sequences文件夹中有没有测试序列，如果没有就下载

同样，由于网络原因，下载很慢，而且每建一个workspace都需要下载的话，比较费时费空间费流量

建议直接使用之前下载好的序列，复制粘贴到sequences子目录即可

uncompress the VOT2020.zip, than cp VOT2020/ /home/user/vot2020_test/sequences

##### 4. Experiment Preparation:

参考Stark的代码 ，将PySOT的tracker类进一步封装为VOT Tracker

原tracker类需要实现以下类方法：

`init(输入为img和第一帧的box, [xc, yc, w, h])`

`track(输入为img，输出为字典，其中必须包含{'bbox': [x1, y1, w, h]})`

相关代码位于 eval_toolkit/vot20 子目录中，主要是在 vot20_builder.py

##### 5. Implement VOT Tracker:

以siamban_test2.py为例进行说明

与测试其他序列的脚本test_script.py相同，测试时需要提供tracker的名称、config文件路径和权重路径

```python
from eval_toolkit.vot20.vot20_builder import run_vot_exp
from configs.get_config import Config
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '1'  # 似乎不起作用
ROOT = '/home/yx/pysot_trial/'
"""
在Log文件夹中的日志里才能看到这些打印结果输出
可以看出，测试程序并不是在项目文件夹下运行
因此会出现调用项目中库出错的问题
所以要将测试跟踪脚本放在项目文件夹第一级
并且在运行测试与验证命令时也要位于项目文件夹目录下
"""
print(str(os.getcwd()))
print(ROOT)
print(str(os.path.join(ROOT, 'experiments/siamban/config.yaml')))
print(os.path.exists(os.path.join(ROOT, 'experiments/siamban/config.yaml')))
# 在这里设置跟踪参数，便于进行调参实验
track_cfg = Config()
track_cfg.context_amount = 0.50
track_cfg.window_influence = 0.40
track_cfg.penalty_k = 0.08
track_cfg.size_lr = 0.45
track_cfg.confidence = 0.
run_vot_exp(tracker_name='SiamBAN',
            config_file=os.path.join(ROOT, 'experiments/siamban/config.yaml'),
            weight_path=os.path.join(ROOT, 'snapshot/para-0.30-0.15/checkpoint_e20.pth'),
            vis=False, track_cfg=track_cfg)

```  

##### 6. Complete Tracker.ini:
在Tracker.ini中，注册跟踪器信息

以上面实现的VOT跟踪器 siamban_test2.py 为例

Example:
```
[transt]  # <tracker-name>
label = siamban_test2
protocol = traxpython
command = siamban_test2 # 跟踪器脚本名，不带.py后缀
paths = /home/user/pysot_trial/  #  跟踪器脚本所在目录

env_PATH = /home/user/pysot_trial/;${PATH}  # 不知道有没有用
# 当 测试跟踪器所在的项目使用的虚拟环境 与 安装vot测试包的虚拟环境 不一致时，可如此设置？未测试过，不知是否可行
# env_PATH = /home/user/.conda/envs/pysot/bin/python3.7;${PATH}  
```  

##### 7. Evaluate and analyze:
！！！确保运行测试前将项目目录export为环境变量

为了避免出现import问题，建议在项目目录一级运行各种命令

例如
```bash
(pysot)user@node01:~/pysot_trial$ CUDA_VISIBLE_DEVICES=1 vot evaluate --workspace /home/user/vot_test/ siamban_test2
(pysot)user@node01:~/pysot_trial$ vot analysis --workspace /home/user/vot_test/ siamban_test2 --format json
```

根据log中错误报告来看，似乎运行测试时，
目录在 `/home/user/.conda/envs/pysot/lib/python3.7/runpy.py`中，
因此会导致无法import项目代码中函数的问题