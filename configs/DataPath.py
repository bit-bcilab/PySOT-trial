

import platform

SYSTEM = platform.system()

if SYSTEM == 'Linux':
    ROOT = '/home/Data2/'
    ROOT_PATH = '/home/Data2/training_dataset/'
    GOT_PATH = '/home/Data2/GOT-10k/'
    LSOTB_PATH = '/home/Data/LSOTB-TIR/'
    UAVDT_PATH = '/home/Data2/UAVDT/'
    VisDrone_PATH = '/home/Data2/VisDrone/'
    TEST_PATH = {'OTB50': '/home/Data2/Benchmark/',
                 'OTB100': '/home/Data2/Benchmark/',
                 'LaSOT': '/home/Data2/LaSOT/LaSOTBenchmark/',
                 'GOT-10k': GOT_PATH + 'test/',
                 'TC128': '/home/Data/Temple-color-128/',
                 'VOT2016': '/home/Data2/VOT2016/',
                 'VOT2018': '/home/Data2/VOT2018/',
                 'VOT2019': '/home/Data2/VOT2019/',
                 'VOT2020': '/home/Data2/VOT2020/',
                 'ITB': '/home/Data2/ITB/',
                 'NFS30': '/home/Data/NFS/',
                 'NFS240': '/home/Data/NFS/',
                 'VOT2018-LT': '/home/Data/VOT2018-LT/',
                 'TrackingNet': '/home/Data/TrackingNet/',

                 'UAV123': '/home/Data2/UAV123/data_seq/UAV123/',
                 'UAV20L': '/home/Data2/UAV123/data_seq/UAV123/',
                 'UAV10fps': '/home/Data2/UAV123/UAV123_10fps/data_seq/UAV123_10fps/',
                 'UAVDT': UAVDT_PATH + 'UAV-benchmark-S/',
                 'DTB70': '/home/Data2/DTB70/',

                 'VOT2017-TIR': '/home/Data/VOT2017-TIR/',
                 'LSOTB-TIR': LSOTB_PATH + 'Evaluation Dataset/sequences/',
                 'PTB-TIR': '/home/Data/PTB-TIR/'}
else:
    ROOT = 'I://'
    ROOT_PATH = 'E://DataBase/training_dataset/'
    GOT_PATH = 'F://DataBase/GOT-10k/'
    LSOTB_PATH = 'F://DataBase/LSOTB-TIR/'
    UAVDT_PATH = 'F://DataBase/UAVDT/'
    VisDrone_PATH = 'F://DataBase/VisDrone/'
    TEST_PATH = {'OTB50': 'F://DataBase/Benchmark/',
                 'OTB100': 'F://DataBase/Benchmark/',
                 'LaSOT': 'I://LaSOT/LaSOTBenchmark/',
                 'GOT-10k': GOT_PATH + 'test/',
                 'TC128': 'F://DataBase/Temple-color-128/',
                 'VOT2016': 'F://DataBase/VOT2016/',
                 'VOT2018': 'F://DataBase/VOT2018/',
                 'VOT2019': 'F://DataBase/VOT2019/',
                 'VOT2020': 'F://DataBase/VOT2020/',
                 'ITB': 'F://DataBase/ITB/',
                 'NFS30': 'F://DataBase/NFS/',
                 'NFS240': 'F://DataBase/NFS/',
                 'VOT2018-LT': 'F://DataBase/VOT2018-LT/',
                 'TrackingNet': 'E://BaiduNetdiskDownload/TN/',

                 'UAV123': 'F://DataBase/UAV123/Dataset_UAV123/UAV123/data_seq/UAV123/',
                 'UAV20L': 'F://DataBase/UAV123/Dataset_UAV123/UAV123/data_seq/UAV123/',
                 'UAV10fps': 'F://DataBase/UAV123/UAV123_10fps/data_seq/UAV123_10fps/',
                 'UAVDT': UAVDT_PATH + 'UAV-benchmark-S/',
                 'DTB70': 'F://DataBase/DTB70/',

                 'VOT2017-TIR': 'F://DataBase/VOT2017-TIR/',
                 'LSOTB-TIR': LSOTB_PATH + 'Evaluation Dataset/sequences/',
                 'PTB-TIR': 'F://DataBase/PTB-TIR/'}

COCO_PATH = 'coco/'
DET_PATH = 'det/ILSVRC/Data/DET/train/'
VID_PATH = 'vid/ILSVRC2015/Data/VID/train'
YBB_PATH = 'yt_bb/crop511/'

TRAIN_PATH = dict(COCO=ROOT_PATH + COCO_PATH, COCO_val=ROOT_PATH + COCO_PATH,
                  DET=ROOT_PATH + DET_PATH, DET_val=ROOT_PATH + DET_PATH,
                  VID=ROOT_PATH + VID_PATH, VID_val=ROOT_PATH + VID_PATH,
                  YBB=ROOT_PATH + YBB_PATH, YBB_val=ROOT_PATH + YBB_PATH,
                  GOT=GOT_PATH + '/train', GOT_val=GOT_PATH + '/val', LaSOT=TEST_PATH['LaSOT'],

                  UAVDT_DET=UAVDT_PATH + 'UAV-benchmark-M/', UAVDT_DET_val=UAVDT_PATH + 'UAV-benchmark-M/',
                  VisDrone_DET=VisDrone_PATH + 'VisDrone2019-DET-train/images/',
                  VisDrone_DET_val=VisDrone_PATH + 'VisDrone2019-DET-val/images/',
                  VisDrone_DET_test=VisDrone_PATH + 'VisDrone2019-DET-test-dev/images/',
                  LSOTB=LSOTB_PATH + 'Training_Dataset/LSOTB-TIR_TrainingData/LSOTB-TIR_TrainingData/TrainingData/')

TRAIN_JSON_PATH = 'json_labels_clean/'


def get_root(name):
    return TEST_PATH[get_base_name(name)]


def get_base_name(name):
    if 'OTB50' in name:
        base_name = 'OTB100'
    elif 'OTB100' in name:
        base_name = 'OTB100'
    elif 'GOT-10k' in name:
        base_name = 'GOT-10k'
    elif 'LaSOT' in name:
        base_name = 'LaSOT'
    elif 'ITB' in name:
        base_name = 'ITB'
    elif 'NFS30' in name:
        base_name = 'NFS30'
    elif 'NFS240' in name:
        base_name = 'NFS240'
    elif 'VOT2018-LT' in name:
        base_name = 'VOT2018-LT'
    elif 'TrackingNet' in name:
        base_name = 'TrackingNet'
    elif 'VOT2016' in name:
        base_name = 'VOT2016'
    elif 'VOT2018' in name and '-LT' not in name:
        base_name = 'VOT2018'
    elif 'VOT2019' in name:
        base_name = 'VOT2019'
    elif 'VOT2020' in name:
        base_name = 'VOT2020'

    elif 'UAV123' in name:
        base_name = 'UAV123'
    elif 'UAV20L' in name:
        base_name = 'UAV20L'
    elif 'UAV10fps' in name:
        base_name = 'UAV10fps'
    elif 'UAVDT' in name:
        base_name = 'UAVDT'
    elif 'DTB70' in name:
        base_name = 'DTB70'

    elif 'VOT2017-TIR' in name:
        base_name = 'VOT2017-TIR'
    elif 'LSOTB-TIR' in name:
        base_name = 'LSOTB-TIR'
    elif 'PTB-TIR' in name:
        base_name = 'PTB-TIR'
    return base_name
