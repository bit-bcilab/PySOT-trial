
import os
import json
import math
import numpy as np
from tqdm import tqdm
from glob import glob
import shutil
from multiprocessing import Pool
import xml.etree.ElementTree as ET
import time
import cv2

from trial.utils.bbox import get_axis_aligned_bbox, get_axis_aligned_bbox_new
from trial.utils.bbox import center2corner, Center
from configs.DataPath import TRAIN_PATH, ROOT_PATH, DET_PATH, TRAIN_JSON_PATH, SYSTEM


def _wash_det(video_):
    new_video = {}
    video_name = video_[0]
    video = video_[1]
    frames = list(video.keys())
    num_frame = len(frames)
    for i in range(num_frame):
        new_frame = {}
        frame = video[frames[i]]
        num_boxes = len(frame)
        boxes_name = list(frame.keys())
        for j in range(num_boxes):
            box = frame[boxes_name[j]]
            # box, noisy = get_bbox_ybb([511, 511], box)
            # if not noisy:
            #     w = box[2] - box[0]
            #     h = box[3] - box[1]
            #     if w > 10 and h > 10:
            #         new_video[frames[i]] = video[frames[i]]
            #     else:
            #         a = 0
            # else:
            #     a = 0
            w = box[2] - box[0]
            h = box[3] - box[1]
            if w > 10 and h > 10:
                new_frame[boxes_name[j]] = frame[boxes_name[j]]
            else:
                a = 0
        if new_frame != {}:
            new_video[frames[i]] = new_frame
        else:
            a = 0
    return [new_video, video_name]


def wash_det(label_path):
    new_data = {}
    with open('json_labels/' + label_path) as f:
        data = json.load(f)
        f.close()

    video_names = list(data.keys())
    frames = list(data.values())
    num_video = len(video_names)

    with Pool(processes=16) as pool:
        for new_video in tqdm(pool.imap_unordered(_wash_det, zip(video_names, frames)), desc='evaluate success', total=num_video, ncols=100):
            if new_video[0] != {}:
                new_data[new_video[1]] = new_video[0]
            else:
                a = 0
    print('label: {:s}, length: {:d}'.format(label_path, len(new_data)))

    file = json.dumps(new_data, indent=4)
    fileObject = open(label_path, 'w')
    fileObject.write(file)
    fileObject.close()


def _wash_trc(video_):
    new_video = {}
    video_name = video_[0]
    video = video_[1]
    frames = list(video.keys())
    num_frame = len(frames)

    for i in range(num_frame):
        box = video[frames[i]]
        w = box[2] - box[0]
        h = box[3] - box[1]
        if w > 10 and h > 10:
            new_video[frames[i]] = video[frames[i]]
        else:
            a = 0

    if new_video == {}:
        a = 0
    return [new_video, video_name]


def wash_trc(label_path):
    new_data = {}
    with open('json_labels/' + label_path) as f:
        data = json.load(f)
        f.close()

    video_names = list(data.keys())
    frames = list(data.values())
    num_video = len(video_names)

    with Pool(processes=8) as pool:
        for new_video in tqdm(pool.imap_unordered(_wash_trc, zip(video_names, frames)), desc='wash trc', total=num_video, ncols=100):
            if new_video[0] != {}:
                new_data[new_video[1]] = new_video[0]
            else:
                a = 0

    file = json.dumps(new_data, indent=4)
    fileObject = open(label_path, 'w')
    fileObject.write(file)
    fileObject.close()


def get_bbox_ybb(image_shape, shape, context_amount=0.5, exemplar_size=127):
    imh, imw = image_shape[:2]
    if len(shape) == 4:
        w, h = shape[2] - shape[0], shape[3] - shape[1]
    else:
        w, h = shape
    wc_z = w + context_amount * (w + h)
    hc_z = h + context_amount * (w + h)
    s_z = np.sqrt(wc_z * hc_z)
    scale_z = exemplar_size / s_z
    w = w * scale_z
    h = h * scale_z
    cx, cy = imw // 2, imh // 2

    noisy = max(w, h) / min(w, h) > 2.
    bbox = center2corner(Center(cx, cy, w, h))
    return bbox, noisy


def process_vot2020_imgpath(path):
    videos = os.listdir(path)
    num_videos = len(videos)
    for i in range(num_videos):
        video_dir = os.path.join(path, videos[i])
        if os.path.isdir(video_dir):
            images = glob(os.path.join(video_dir, '*.jp*'))
            if SYSTEM == 'Linux':
                images = sorted(images, key=lambda x: int(x.split(os.sep)[-1].split('.')[0]))
            num_images = len(images)
            new_dir = os.path.join(video_dir, 'color')
            if not os.path.exists(new_dir):
                os.mkdir(new_dir)
            for j in range(num_images):
                old_name_img = images[j]
                img_name = images[j].split(os.sep)[-1]
                new_name_img = new_dir + os.sep + img_name
                shutil.move(old_name_img, new_name_img)
                a = 1
        else:
            a = 0


def process_vot2020_anno(path):
    root = path[:path.index('VOT2020.json')]
    new_data = {}
    with open(path, 'r') as f:
        data = json.load(f)

    num_videos = len(data)
    video_names = list(data.keys())
    for i in range(num_videos):
        video_name = video_names[i]
        video = data[video_name]

        num_frames = len(video['gt_rect'])

        new_data[video_name] = {}
        new_data[video_name]['gt_rect'] = []
        new_data[video_name]['img_names'] = []
        for j in range(num_frames):
            gt_bbox = video['gt_rect'][j]
            old_name = video['img_names'][j]

            rect = get_axis_aligned_bbox_new(np.array(gt_bbox))[0]
            new_data[video_name]['gt_rect'].append(rect)

            new_name = old_name.split('/')[0] + '/color/' + old_name.split('/')[1]
            new_data[video_name]['img_names'].append(new_name)

            # cx, cy, w, h = get_axis_aligned_bbox(np.array(gt_bbox))[-1]
            # cx1, cy1, w1, h1 = get_axis_aligned_bbox_new(np.array(gt_bbox))[-1]
            # img = cv2.imread(os.path.join(root, new_name))
            # box = [cx - (w - 1) / 2, cy - (h - 1) / 2, cx + (w - 1) / 2, cy + (h - 1) / 2]
            # box1 = [cx1 - (w1 - 1) / 2, cy1 - (h1 - 1) / 2, cx1 + (w1 - 1) / 2, cy1 + (h1 - 1) / 2]
            # bbox = list(map(int, box))
            # cv2.rectangle(img, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 0, 255), 2)
            # bbox1 = list(map(int, box1))
            # cv2.rectangle(img, (bbox1[0], bbox1[1]), (bbox1[2], bbox1[3]), (255, 0, 0), 2)
            # cv2.imshow('1', img)
            # k = cv2.waitKey(70) & 0xff
            # if k == 27:
            #     cv2.destroyWindow('1')
            #     break

        new_data[video_name]['attr'] = []
        new_data[video_name]['video_dir'] = video_name
        new_data[video_name]['init_rect'] = new_data[video_name]['gt_rect'][0]

    file = json.dumps(new_data, indent=4)
    fileObject = open('VOT2020.json', 'w')
    fileObject.write(file)
    fileObject.close()


def process_vot2017tir_anno(path):
    videos = os.listdir(path)
    num_videos = len(videos)

    anno = {}

    for i in range(num_videos):
        video_dir = os.path.join(path, videos[i])
        if os.path.isdir(video_dir):
            images = glob(os.path.join(video_dir, '*.png'))
            if SYSTEM == 'Linux':
                images = sorted(images, key=lambda x: int(x.split(os.sep)[-1].split('.')[0]))
            num_frames = len(images)

            occ_file = os.path.join(path, videos[i], 'occlusion.label')
            if os.path.exists(occ_file):
                with open(occ_file, 'r') as f:
                    occ = f.readlines()
                    f.close()
            else:
                occ = ['0'] * num_frames

            cm_file = os.path.join(path, videos[i], 'camera_motion.label')
            if os.path.exists(cm_file):
                with open(cm_file, 'r') as f:
                    cm = f.readlines()
                    f.close()
            else:
                cm = ['0'] * num_frames

            ic_file = os.path.join(path, videos[i], 'dynamics_change.label')
            if os.path.exists(ic_file):
                with open(ic_file, 'r') as f:
                    ic = f.readlines()
                    f.close()
            else:
                ic = ['0'] * num_frames

            mc_file = os.path.join(path, videos[i], 'motion_change.label')
            if os.path.exists(mc_file):
                with open(mc_file, 'r') as f:
                    mc = f.readlines()
                    f.close()
            else:
                mc = ['0'] * num_frames

            sc_file = os.path.join(path, videos[i], 'size_change.label')
            if os.path.exists(sc_file):
                with open(sc_file, 'r') as f:
                    sc = f.readlines()
                    f.close()
            else:
                sc = ['0'] * num_frames

            anno[videos[i]] = {}
            anno[videos[i]]['video_dir'] = videos[i]
            anno[videos[i]]['gt_rect'] = []
            anno[videos[i]]['img_names'] = []
            anno[videos[i]]['camera_motion'] = []
            anno[videos[i]]['illum_change'] = []
            anno[videos[i]]['motion_change'] = []
            anno[videos[i]]['occlusion'] = []
            anno[videos[i]]['size_change'] = []

            gt_file = os.path.join(path, videos[i], 'groundtruth.txt')
            with open(gt_file, 'r') as f:
                gts = f.readlines()
                f.close()

            for j in range(num_frames):
                img_name = videos[i] + '/' + images[j].split(os.sep)[-1]
                gt = gts[j].split()[0].split(',')
                gt = list(map(float, gt))
                anno[videos[i]]['img_names'].append(img_name)
                anno[videos[i]]['gt_rect'].append(gt)
                anno[videos[i]]['camera_motion'].append(int(cm[j].split()[0]))
                anno[videos[i]]['illum_change'].append(int(ic[j].split()[0]))
                anno[videos[i]]['motion_change'].append(int(mc[j].split()[0]))
                anno[videos[i]]['occlusion'].append(int(occ[j].split()[0]))
                anno[videos[i]]['size_change'].append(int(sc[j].split()[0]))
            anno[videos[i]]['init_rect'] = anno[videos[i]]['gt_rect'][0]

    file = json.dumps(anno, indent=4)
    fileObject = open('VOT2017-TIR.json', 'w')
    fileObject.write(file)
    fileObject.close()


def process_lsotb_anno(path):
    videos = os.listdir(path)
    num_videos = len(videos)

    anno = {}

    for i in range(num_videos):
        video_dir = os.path.join(path, videos[i])
        if os.path.isdir(video_dir):
            images = glob(os.path.join(video_dir, 'img', '*.jp*'))
            if SYSTEM == 'Linux':
                images = sorted(images, key=lambda x: int(x.split(os.sep)[-1].split('.')[0]))
            num_frames = len(images)

            anno[videos[i]] = {}
            anno[videos[i]]['video_dir'] = videos[i]
            anno[videos[i]]['gt_rect'] = []
            anno[videos[i]]['img_names'] = []
            anno[videos[i]]['attr'] = []

            gt_file = os.path.join(path, videos[i], 'groundtruth_rect.txt')
            with open(gt_file, 'r') as f:
                gts = f.readlines()
                f.close()

            for j in range(num_frames):
                img_name = videos[i] + '/img/' + images[j].split(os.sep)[-1]
                gt = gts[j].split()[0].split(',')
                gt = list(map(float, gt))
                anno[videos[i]]['img_names'].append(img_name)
                anno[videos[i]]['gt_rect'].append(gt)
            anno[videos[i]]['init_rect'] = anno[videos[i]]['gt_rect'][0]

    file = json.dumps(anno, indent=4)
    fileObject = open('LSOTB.json', 'w')
    fileObject.write(file)
    fileObject.close()


def process_otb_anno(path):
    videos = os.listdir(path)
    num_videos = len(videos)

    anno = {}

    for i in range(num_videos):
        video_dir = os.path.join(path, videos[i])
        if os.path.isdir(video_dir):
            images = glob(os.path.join(video_dir, 'img', '*.jp*'))
            if SYSTEM == 'Linux':
                images = sorted(images, key=lambda x: int(x.split(os.sep)[-1].split('.')[0]))
            num_frames = len(images)

            anno[videos[i]] = {}
            anno[videos[i]]['video_dir'] = videos[i]
            anno[videos[i]]['gt_rect'] = []
            anno[videos[i]]['img_names'] = []
            anno[videos[i]]['attr'] = []

            gt_file = os.path.join(path, videos[i], 'groundtruth_rect.txt')
            with open(gt_file, 'r') as f:
                gts = f.readlines()
                f.close()

            for j in range(num_frames):
                img_name = videos[i] + '/img/' + images[j].split(os.sep)[-1]
                gt = gts[j].split()[0].split(',')
                gt = list(map(float, gt))
                anno[videos[i]]['img_names'].append(img_name)
                if gt[2] <= 0 or gt[3] <= 0:
                    anno[videos[i]]['gt_rect'].append([math.nan, math.nan, math.nan, math.nan])
                else:
                    anno[videos[i]]['gt_rect'].append(gt)
            anno[videos[i]]['init_rect'] = anno[videos[i]]['gt_rect'][0]

    file = json.dumps(anno, indent=4)
    fileObject = open('DTB70.json', 'w')
    fileObject.write(file)
    fileObject.close()


def process_uav20l_anno(path):
    att_name = ['Scale Variation', 'Aspect Ratio Change', 'Low Resolution', 'Fast Motion',
                'Full Occlusion', 'Partial Occlusion', 'Out-of-View', 'Background Clutter',
                'Illumination Variation', 'Viewpoint Change', 'Camera Motion', 'Similar Object']

    videos = os.listdir(os.path.join(path, 'anno', 'UAV20L'))
    num_videos = len(videos)

    anno = {}

    for i in range(num_videos):
        video = videos[i].split('.')[0]
        video_dir = os.path.join(path, 'data_seq', 'UAV123', video)
        if os.path.isdir(video_dir):
            images = glob(os.path.join(video_dir, '*.jp*'))
            if SYSTEM == 'Linux':
                images = sorted(images, key=lambda x: int(x.split(os.sep)[-1].split('.')[0]))
            num_frames = len(images)

            anno[video] = {}

            gt_file = os.path.join(path, 'anno', 'UAV20L', videos[i])
            with open(gt_file, 'r') as f:
                gts = f.readlines()
                f.close()
            att_file = os.path.join(path, 'anno', 'UAV20L', 'att', videos[i])
            with open(att_file, 'r') as f:
                att = f.readlines()
                f.close()
            att = att[0].split()[0].split(',')
            att = list(map(int, att))

            anno[video]['video_dir'] = video
            anno[video]['gt_rect'] = []
            anno[video]['img_names'] = []
            anno[video]['attr'] = []
            for ii in range(len(att)):
                if att[ii] == 1:
                    anno[video]['attr'].append(att_name[ii])

            for j in range(num_frames):
                img_name = video + '/' + images[j].split(os.sep)[-1]
                gt = gts[j].split()[0].split(',')
                gt = list(map(float, gt))
                anno[video]['img_names'].append(img_name)
                anno[video]['gt_rect'].append(gt)
                if gt[0] > 0 and gt[1] > 0 and gt[2] > 0 and gt[3] > 0:
                    a = 1
                else:
                    a = 0
            anno[video]['init_rect'] = anno[video]['gt_rect'][0]

    file = json.dumps(anno, indent=4)
    fileObject = open('UAV20L.json', 'w')
    fileObject.write(file)
    fileObject.close()


def process_uav_anno(path, opt):
    opts = ['UAV123', 'UAV20L', 'UAV123_10fps']
    dataset = opts[opt]
    att_name = ['Scale Variation', 'Aspect Ratio Change', 'Low Resolution', 'Fast Motion',
                'Full Occlusion', 'Partial Occlusion', 'Out-of-View', 'Background Clutter',
                'Illumination Variation', 'Viewpoint Change', 'Camera Motion', 'Similar Object']

    videos = []
    f = open(os.path.join(path, dataset + '.txt'))
    frame_config = f.readlines()
    frame_cfg = {}
    num = len(frame_config)
    for i in range(num):
        video = frame_config[i].split(',')
        name, video_path, fs, fe = video[1][1:-1], video[3].split('\\')[3], int(video[5]), int(video[7])
        frame_cfg[name] = [video_path, fs, fe]
        videos.append(name)
    num_videos = len(videos)

    anno = {}

    for i in range(num_videos):
        video = videos[i].split('.')[0]
        dataset_video = dataset
        if dataset == 'UAV20L':
            dataset_video = 'UAV123'

        video_path, fs, fe = frame_cfg[video]
        video_dir = os.path.join(path, 'data_seq', dataset_video, video_path)
        images = glob(os.path.join(video_dir, '*.jp*'))
        if SYSTEM == 'Linux':
            images = sorted(images, key=lambda x: int(x.split(os.sep)[-1].split('.')[0]))
        images = images[fs-1: fe]
        num_frames = len(images)

        anno[video] = {}

        gt_file = os.path.join(path, 'anno', dataset, video+'.txt')
        with open(gt_file, 'r') as f:
            gts = f.readlines()
            f.close()
        att_file = os.path.join(path, 'anno', dataset, 'att', video+'.txt')
        with open(att_file, 'r') as f:
            att = f.readlines()
            f.close()
        att = att[0].split()[0].split(',')
        att = list(map(int, att))

        anno[video]['video_dir'] = video_path
        anno[video]['gt_rect'] = []
        anno[video]['img_names'] = []
        anno[video]['attr'] = []
        for ii in range(len(att)):
            if att[ii] == 1:
                anno[video]['attr'].append(att_name[ii])

        for j in range(num_frames):
            img_name = video_path + '/' + images[j].split(os.sep)[-1]
            gt = gts[j].split()[0].split(',')
            gt = list(map(float, gt))
            anno[video]['img_names'].append(img_name)
            anno[video]['gt_rect'].append(gt)
            if gt[0] > 0 and gt[1] > 0 and gt[2] > 0 and gt[3] > 0:
                a = 1
            else:
                a = 0
        anno[video]['init_rect'] = anno[video]['gt_rect'][0]

    file = json.dumps(anno, indent=4)
    fileObject = open(dataset + '.json', 'w')
    fileObject.write(file)
    fileObject.close()


def process_uavdt_anno(path):
    att_name = ['background clutter','camera motion','object motion','small object',
                'illumination variations', 'object blur','scale variations','long-term tracking','large occlusion']

    videos = os.listdir(os.path.join(path, 'UAV-benchmark-S'))
    num_videos = len(videos)

    anno = {}

    for i in range(num_videos):
        video = videos[i].split('.')[0]
        video_dir = os.path.join(path, 'UAV-benchmark-S', video)
        if os.path.isdir(video_dir):
            images = glob(os.path.join(video_dir, '*.jp*'))
            if SYSTEM == 'Linux':
                images = sorted(images, key=lambda x: int(x.split(os.sep)[-1].split('.')[0]))
            num_frames = len(images)

            anno[video] = {}

            gt_file = os.path.join(path, 'UAV-benchmark-SOT_v1.0', 'anno', video + '_gt.txt')
            with open(gt_file, 'r') as f:
                gts = f.readlines()
                f.close()
            att_file = os.path.join(path, 'UAV-benchmark-SOT_v1.0', 'anno', 'att', video + '_att.txt')
            with open(att_file, 'r') as f:
                att = f.readlines()
                f.close()
            att = att[0].split()[0].split(',')
            att = list(map(int, att))

            anno[video]['video_dir'] = video
            anno[video]['gt_rect'] = []
            anno[video]['img_names'] = []
            anno[video]['attr'] = []
            for ii in range(len(att)):
                if att[ii] == 1:
                    anno[video]['attr'].append(att_name[ii])

            for j in range(num_frames):
                img_name = video + '/' + images[j].split(os.sep)[-1]
                gt = gts[j].split()[0].split(',')
                gt = list(map(float, gt))
                anno[video]['img_names'].append(img_name)
                anno[video]['gt_rect'].append(gt)
                if gt[0] > 0 and gt[1] > 0 and gt[2] > 0 and gt[3] > 0:
                    a = 1
                else:
                    a = 0
            anno[video]['init_rect'] = anno[video]['gt_rect'][0]

    file = json.dumps(anno, indent=4)
    fileObject = open('UAVDT.json', 'w')
    fileObject.write(file)
    fileObject.close()


if __name__ == '__main__':
    # with open(os.path.join('json_labels_clean', 'lsotb.json'), 'r') as f:
    #     data = json.load(f)
    #     f.close()
    # videos = list(data.keys())
    # num = len(videos)
    # for i in range(num):
    #     video = videos[i]
    #     if len(data[video]) == 0:
    #         data.pop(video)
    # file = json.dumps(data, indent=4)
    # fileObject = open('lsotb.json', 'w')
    # fileObject.write(file)
    # fileObject.close()

    process_uav_anno('F://DataBase/UAV123/UAV123_10fps', opt=2)
    # process_uav_anno('F://DataBase/UAV123/Dataset_UAV123/UAV123/', opt=1)
    # process_uavdt_anno('F://DataBase/UAVDT')
    # process_otb_anno('F://DataBase/DTB70')
    # process_lsotb_anno('F://DataBase/PTB-TIR/')
    # process_lsotb_anno('F://DataBase/LSOTB-TIR/Evaluation Dataset/sequences/')
    # process_vot2017tir_anno('F://DataBase/VOT2017-TIR/')
    # process_vot2017tir_json('F://DataBase/VOT2017-TIR/')
    # process_vot2020_anno('F:\DataBase\VOT2020\VOT2020.json')
    # process_vot2020_imgpath('F://DataBase/VOT2020/')

    # wash_trc('got-val.json')
    # wash_trc('lasot-val.json')
    # wash_det('ybb-val.json')
    # wash_det('coco-val.json')
    # wash_det('det-val.json')
    # wash_det('vid-val.json')

    # wash_det('coco-train.json')
    # wash_det('det-train.json')
    # wash_det('vid-train.json')
    # wash_det('ybb-train.json')

    # wash_trc('lasot-train.json')
    # wash_trc('got-train.json')

    # tic = time.time()
    # wash_det('coco-train.json')
    # t = time.time() - tic
    # print('multi-process time: {:.4f}'.format(t))

    # tic = time.time()
    # wash_trc('coco-train.json')
    # t = time.time() - tic
    # print('single-process time: {:.4f}'.format(t))
