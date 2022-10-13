from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from yacs.config import CfgNode as CN

__C = CN()

transt_cfg = __C

__C.META_ARC = "transt"

__C.CUDA = True

__C.BASE = 'point'
__C.TRIAL_CFG = 'experiments/siamban/anchor-free-car.yaml'
__C.SELF_EPOCH = 10
__C.VALIDATE = False
__C.VALIDATE_CFG = 'experiments/siamban/ban-val.yaml'
__C.AMP = False
__C.MODE = 'normal'

__C.NORMALIZE_MEAN = [0.485, 0.456, 0.406]
__C.NORMALIZE_STD = [0.229, 0.224, 0.225]


# ------------------------------------------------------------------------ #
# Backbone options
# ------------------------------------------------------------------------ #
__C.BACKBONE = CN()

# Backbone type, current only support resnet18,34,50;alexnet;mobilenet
__C.BACKBONE.TYPE = 'transt_resnet50_pe'

__C.BACKBONE.KWARGS = CN(new_allowed=True)

__C.BACKBONE.KWARGS.output_layers = ['layer3']
__C.BACKBONE.KWARGS.frozen_layers = 'all'
__C.BACKBONE.KWARGS.position_embedding = 'sine'
__C.BACKBONE.KWARGS.hidden_dim = 256

# Pretrained backbone weights
__C.BACKBONE.PRETRAINED = 'pretrained_models/transt_resnet50_pe.pth'

__C.BACKBONE.OUTPUT_LAYERS = ['layer3']

# Train layers
__C.BACKBONE.TRAIN_LAYERS = ['layer2', 'layer3']

# Layer LR
__C.BACKBONE.LAYERS_LR = 0.1

# Switch to train layer
__C.BACKBONE.TRAIN_EPOCH = 10

# ------------------------------------------------------------------------ #
# Transformer options
# ------------------------------------------------------------------------ #
__C.TRANS = CN(new_allowed=True)

__C.TRANS.position_embedding = 'sine'

__C.TRANS.hidden_dim = 256

__C.TRANS.dropout = 0.1

__C.TRANS.nheads = 8

__C.TRANS.dim_feedforward = 2048

__C.TRANS.featurefusion_layers = 4

__C.NUM_CLASSES = 1

# ------------------------------------------------------------------------ #
# Adjust layer options
# ------------------------------------------------------------------------ #
__C.ADJUST = CN()

# Adjust layer
__C.ADJUST.ADJUST = False

__C.POINT = CN()

# Point stride
__C.POINT.STRIDE = 8

# ------------------------------------------------------------------------ #
# Training options
# ------------------------------------------------------------------------ #
__C.TRAIN = CN()

# Anchor Target
# Positive anchor threshold
__C.TRAIN.THR_HIGH = 0.6

# Negative anchor threshold
__C.TRAIN.THR_LOW = 0.3

# Number of negative
__C.TRAIN.NEG_NUM = 16

# Number of positive
__C.TRAIN.POS_NUM = 16

# Number of anchors per images
__C.TRAIN.TOTAL_NUM = 64


__C.TRAIN.EXEMPLAR_SIZE = 128

__C.TRAIN.SEARCH_SIZE = 256

__C.TRAIN.BASE_SIZE = 8

__C.TRAIN.OUTPUT_SIZE = 32

__C.TRAIN.RESUME = ''

__C.TRAIN.PRETRAINED = ''

__C.TRAIN.LOG_DIR = './logs'

__C.TRAIN.SNAPSHOT_DIR = './snapshot'

__C.TRAIN.EPOCH = 20

__C.TRAIN.START_EPOCH = 0

__C.TRAIN.BATCH_SIZE = 38

__C.TRAIN.NUM_WORKERS = 5

__C.TRAIN.MOMENTUM = 0.9

__C.TRAIN.WEIGHT_DECAY = 0.0001

__C.TRAIN.CLS_WEIGHT = 1.0

__C.TRAIN.LOC_WEIGHT = 1.2

__C.TRAIN.MASK_WEIGHT = 0.4

__C.TRAIN.BBOX_WEIGHT = 1.0

__C.TRAIN.PRINT_FREQ = 20

__C.TRAIN.LOG_GRADS = False

__C.TRAIN.GRAD_CLIP = 10.0

__C.TRAIN.BASE_LR = 0.0001

__C.TRAIN.LR = CN()

__C.TRAIN.LR.TYPE = 'log'

__C.TRAIN.LR.KWARGS = CN(new_allowed=True)

__C.TRAIN.LR_WARMUP = CN()

__C.TRAIN.LR_WARMUP.WARMUP = True

__C.TRAIN.LR_WARMUP.TYPE = 'step'

__C.TRAIN.LR_WARMUP.EPOCH = 5

__C.TRAIN.LR_WARMUP.KWARGS = CN(new_allowed=True)

# ------------------------------------------------------------------------ #
# Dataset options
# ------------------------------------------------------------------------ #
__C.DATASET = CN(new_allowed=True)

# Augmentation
# for template
__C.DATASET.TEMPLATE = CN()

# Random shift see [SiamPRN++](https://arxiv.org/pdf/1812.11703)
# for detail discussion
__C.DATASET.TEMPLATE.SHIFT = 4

__C.DATASET.TEMPLATE.SCALE = 0.05

__C.DATASET.TEMPLATE.BLUR = 0.0

__C.DATASET.TEMPLATE.FLIP = 0.0

__C.DATASET.TEMPLATE.COLOR = 0.5

__C.DATASET.TEMPLATE.EXTRA_AUG = 0.0

__C.DATASET.SEARCH = CN()

__C.DATASET.SEARCH.SHIFT = 64

__C.DATASET.SEARCH.SCALE = 0.18

__C.DATASET.SEARCH.BLUR = 0.2

__C.DATASET.SEARCH.FLIP = 0.2

__C.DATASET.SEARCH.COLOR = 0.5

__C.DATASET.SEARCH.EXTRA_AUG = 0.5

# Sample Negative pair see [DaSiamRPN](https://arxiv.org/pdf/1808.06048)
# for detail discussion
__C.DATASET.NEG = 0.2

# improve tracking performance for otb100
__C.DATASET.GRAY = 0.0

#__C.DATASET.NAMES = ('VID', 'COCO', 'DET', 'YOUTUBEBB', 'MOT')
#__C.DATASET.NAMES = ('COCO', 'VID', 'DET', 'YTBVOS', 'YOUTUBEBB')
__C.DATASET.NAMES = ('COCO', 'TKN', 'LASOT', 'YTBVOS', 'GOT10K')

__C.DATASET.VID = CN()
__C.DATASET.VID.ROOT = 'training_dataset/vid/crop511'
__C.DATASET.VID.ANNO = 'training_dataset/vid/train.json'
__C.DATASET.VID.FRAME_RANGE = 100
__C.DATASET.VID.NUM_USE = 100000  # repeat until reach NUM_USE

__C.DATASET.YOUTUBEBB = CN()
__C.DATASET.YOUTUBEBB.ROOT = 'training_dataset/yt_bb/crop511'
__C.DATASET.YOUTUBEBB.ANNO = 'training_dataset/yt_bb/train.json'
__C.DATASET.YOUTUBEBB.FRAME_RANGE = 3
__C.DATASET.YOUTUBEBB.NUM_USE = -1  # use all not repeat

__C.DATASET.COCO = CN()
__C.DATASET.COCO.ROOT = 'training_dataset/coco/crop511'
__C.DATASET.COCO.ANNO = 'training_dataset/coco/train2017.json'
__C.DATASET.COCO.FRAME_RANGE = 1
__C.DATASET.COCO.NUM_USE = -1

__C.DATASET.DET = CN()
__C.DATASET.DET.ROOT = 'training_dataset/det/crop511'
__C.DATASET.DET.ANNO = 'training_dataset/det/train.json'
__C.DATASET.DET.FRAME_RANGE = 1
__C.DATASET.DET.NUM_USE = -1

__C.DATASET.GOT10K = CN()
__C.DATASET.GOT10K.ROOT = 'GOT-10k/crop511'
__C.DATASET.GOT10K.ANNO = 'GOT-10k/train.json'
__C.DATASET.GOT10K.FRAME_RANGE = 30
__C.DATASET.GOT10K.NUM_USE = 150000

__C.DATASET.LASOT = CN()
__C.DATASET.LASOT.ROOT = 'LaSOT/crop511'
__C.DATASET.LASOT.ANNO = 'LaSOT/train.json'
__C.DATASET.LASOT.FRAME_RANGE = 300
__C.DATASET.LASOT.NUM_USE = 200000

__C.DATASET.YTBVOS = CN()
__C.DATASET.YTBVOS.ROOT = 'training_dataset/ytb_vos/crop511'
__C.DATASET.YTBVOS.ANNO = 'training_dataset/ytb_vos/train.json'
__C.DATASET.YTBVOS.FRAME_RANGE = 20
__C.DATASET.YTBVOS.NUM_USE = 150000

__C.DATASET.TKN = CN()
__C.DATASET.TKN.ROOT = 'training_dataset/trackingnet/crop511'
__C.DATASET.TKN.ANNO = 'training_dataset/trackingnet/train.json'
__C.DATASET.TKN.FRAME_RANGE = 100
__C.DATASET.TKN.NUM_USE = 200000

__C.DATASET.VIDEOS_PER_EPOCH = 600000

# ------------------------------------------------------------------------ #
# Tracker options
# ------------------------------------------------------------------------ #
__C.TRACK = CN()

__C.TRACK.TYPE = 'TransT'

# Exemplar size
__C.TRACK.EXEMPLAR_SIZE = 128

# Instance size
__C.TRACK.INSTANCE_SIZE = 256

# Context amount
__C.TRACK.CONTEXT_AMOUNT = 0.5

# Window influence
__C.TRACK.WINDOW_INFLUENCE = 0.44

# Scale penalty
__C.TRACK.PENALTY_K = 0.04

# Interpolation learning rate
__C.TRACK.LR = 0.4

__C.TRACK.CONFIDENCE = 0.0

# Base size
__C.TRACK.BASE_SIZE = 8
