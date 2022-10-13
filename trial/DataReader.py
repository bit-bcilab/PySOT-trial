

import numpy as np

from configs.DataPath import TRAIN_PATH, ROOT_PATH, DET_PATH, TRAIN_JSON_PATH
from trial.utils.rand import random_sys
from trial.utils.bbox import center2corner, Center

from os.path import exists
import jpeg4py as jpeg
import cv2
from PIL import Image
import json
import random
import logging

logger = logging.getLogger("global")


mix_prob = {'GOT': 0.75, 'LaSOT': 0.75, 'GOT_val': 0.75,
            'VID': 0.75, 'VID_val': 0.75,
            'UAVDT_DET': 0.75, 'UAVDT_DET_val': 0.75,
            'LSOTB': 0.7}


def val2train(dataset):
    if '_val' in dataset and dataset not in ['GOT_val', 'VID_val', 'COCO_val', 'DET_val', 'YBB_val',
                                             'UAVDT_DET_val', 'VisDrone_DET_val']:
        index = dataset.index('_val')
        if '_val_val' in dataset:  # LaSOT_val_val, LSOTB_val_val
            dataset = dataset[:(index + 4)]
        else:
            dataset = dataset[:index]
    return dataset


def read_img(img_path, way='opencv', read_bgr=True):
    if way == 'opencv':
        img = cv2.imread(img_path)
        if not read_bgr:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    elif way == 'PIL':
        img = np.array(Image.open(img_path))
        if read_bgr:
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    elif way == 'jpeg':
        if img_path.endswith('.jpg') or img_path.endswith('.JPEG'):
            try:
                img = jpeg.JPEG(img_path).decode()
                if read_bgr:
                    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            except:
                img = read_img(img_path, way='opencv')
        else:
            img = read_img(img_path, way='opencv')
    return img


class DataReader(object):
    def __init__(self, data_settings, num_epoch, run_mode='train'):
        self.num_epoch = num_epoch
        self.read_bgr = data_settings['read_bgr']
        self.used_datasets = data_settings['dataset_used']

        self.sub_datasets = {}
        self.num_per_epoch = 0
        self.num_train = 0
        self.num_val = 0
        self.val_index = []

        for sub_dataset in data_settings['dataset_used']:
            self.sub_datasets[sub_dataset] = data_settings[sub_dataset]

            with open(TRAIN_JSON_PATH + self.sub_datasets[sub_dataset]['label_path']) as f:
                data = json.load(f)
                f.close()
            self.sub_datasets[sub_dataset]['data'] = data

            num_data = self.sub_datasets[sub_dataset]['num_data']
            assert num_data == len(data)

            multiply = self.sub_datasets[sub_dataset]['multiply']
            num_train = self.sub_datasets[sub_dataset]['num_train']
            num_train_objects = self.sub_datasets[sub_dataset]['num_train_objects']
            assert num_train_objects <= (num_train * multiply)

            num_val = self.sub_datasets[sub_dataset]['num_val']
            num_val_objects = self.sub_datasets[sub_dataset]['num_val_objects']
            assert num_val_objects <= (num_val * multiply)

            assert (num_val + num_train) <= num_data

            dataset = [sub_dataset] * num_data
            keys = list(data.keys())
            index = list(zip(dataset, keys))
            random.shuffle(index)

            if num_val > 0:
                train_index = index[:(num_data - num_val)]
                val_index = index[-num_val:]

                val_index = val_index * multiply
                random.shuffle(val_index)
                val_index = val_index[:num_val_objects]
            else:
                train_index = index
                val_index = []

            self.sub_datasets[sub_dataset].update(dict(train_index=train_index, val_index=val_index))

            self.val_index += val_index
            self.num_per_epoch += num_train_objects
            self.num_val += num_val_objects

            logger.info('load {:s} done, train num: {:d}, val num: {:d}'.format(sub_dataset, num_train_objects, num_val_objects))
        random.shuffle(self.val_index)
        logger.info('Training pair number per epoch: {:d}, val pair per epoch: {:d}'.format(self.num_per_epoch, self.num_val))

        self.train_index_ = []
        self.train_index = []
        if run_mode != 'val':
            for i in range(self.num_epoch):
                self.train_index += self.build_train_index()
            self.num_train = self.num_per_epoch * self.num_epoch
        assert len(self.train_index) == self.num_train
        # random.shuffle(self.train_index)
        logger.info('Dataloader done. Total training pair number: %d' % self.num_train)

    def build_train_index(self):
        train_index = []
        for sub_dataset in self.sub_datasets:
            sub_index = self.sub_datasets[sub_dataset]['train_index'].copy()
            if sub_index:
                random.shuffle(sub_index)

                sub_index = sub_index[:self.sub_datasets[sub_dataset]['num_train']]
                sub_index *= self.sub_datasets[sub_dataset]['multiply']
                random.shuffle(sub_index)

                sub_index = sub_index[:self.sub_datasets[sub_dataset]['num_train_objects']]

            train_index += sub_index
        random.shuffle(train_index)
        return train_index

    def update_index(self, epoch):
        self.train_index_ = self.train_index[epoch * self.num_per_epoch: (epoch + 1) * self.num_per_epoch]

    def random_choice(self, rate=0.5, mode='train'):
        if mode == 'train':
            if random_sys() > rate:
                while True:
                    sub_dataset = random.choice(self.used_datasets)
                    if len(self.sub_datasets[sub_dataset]['train_index']) > 0:
                        random_index = random.choice(self.sub_datasets[sub_dataset]['train_index'])
                        return random_index
            else:
                random_index = random.choice(self.train_index)
        else:
            if random_sys() > rate:
                while True:
                    sub_dataset = random.choice(self.used_datasets)
                    if len(self.sub_datasets[sub_dataset]['val_index']) > 0:
                        random_index = random.choice(self.sub_datasets[sub_dataset]['val_index'])
                        return random_index
            else:
                random_index = random.choice(self.val_index)
        return random_index

    def get_random_data(self, video_index, read_all_boxes=False, rate=0.5, mode='train'):
        while True:
            random_index = self.random_choice(rate=rate, mode=mode)
            if random_index[1] != video_index[1]:
                random_mixes, random_image, random_bbox = self.get_data(random_index, read_pair=False, read_all_boxes=read_all_boxes)
                if random_image is not None:
                    return random_index, random_mixes, random_image, random_bbox

    def get_data(self, index, read_pair=True, read_all_boxes=False):
        cv2.setNumThreads(0)
        cv2.ocl.setUseOpenCL(False)

        dataset = index[0]
        index = index[1]
        data = self.sub_datasets[dataset]['data'][index]
        match_range = self.sub_datasets[dataset]['match_range']
        dataset = val2train(dataset)
        path = TRAIN_PATH[dataset] + '/' + index
        all_boxes = []

        if dataset in ['DET', 'DET_val', 'COCO', 'COCO_val',
                       'VisDrone_DET', 'VisDrone_DET_val', 'VisDrone_DET_test']:
            if dataset == 'DET' or dataset == 'DET_val':
                if index[0] == 'a':
                    search_path = ROOT_PATH + DET_PATH + index[:index.index('_')] + '/' + index[2:] + '.JPEG'
                else:
                    search_path = path + '.JPEG'
            else:
                search_path = path + '.jpg'

            samples = list(data.keys())
            num_sample = len(data)
            if num_sample > 1:
                search_index = random.randint(0, num_sample - 1)
            else:
                search_index = 0
            search_box = data[samples[search_index]]['000000']

            if read_pair:
                template_path = search_path

            if read_all_boxes:
                for i in range(num_sample):
                    if i != search_index:
                        all_boxes.append(np.array(data[samples[i]]['000000'], dtype=np.float32))

        elif dataset in ['VID', 'VID_val']:

            num_sample = len(data)

            samples = list(data.keys())

            if num_sample == 1:

                sample_index = 0

            else:

                sample_index = random.randint(0, num_sample - 1)

            sample_data = data[samples[sample_index]]

            frames = list(sample_data.keys())

            num_frame = len(frames)

            search_index = random.randint(0, num_frame - 1)

            search_frame = frames[search_index]

            search_path = path + '/' + search_frame + '.JPEG'

            search_box = sample_data[search_frame]

            if read_pair:

                if match_range == 'all':

                    template_index = random.randint(0, num_frame - 1)

                elif match_range == 'init':

                    template_index = 0

                elif match_range == 'mix':

                    if random_sys() > 0.5:

                        template_index = 0

                    else:

                        template_index = random.randint(0, num_frame - 1)

                else:

                    template_index = random.randint(max(search_index - match_range, 0),

                                                    min(search_index + match_range, num_frame) - 1)

                template_path = path + '/' + frames[template_index] + '.JPEG'

                template_box = sample_data[frames[template_index]]

            if read_all_boxes:

                if num_sample > 1:

                    for i in range(num_sample):

                        if i != sample_index:

                            sample_frames = list(data[samples[i]].keys())

                            if search_frame in sample_frames:
                                all_boxes.append(np.array(data[samples[i]][search_frame], dtype=np.float32))

        elif dataset in ['UAVDT_DET', 'UAVDT_DET_val']:
            num_sample = len(data)
            samples = list(data.keys())
            if num_sample == 1:
                sample_index = 0
            else:
                sample_index = random.randint(0, num_sample - 1)

            sample_data = data[samples[sample_index]]
            frames = list(sample_data.keys())
            num_frame = len(frames)
            search_index = random.randint(0, num_frame - 1)
            search_frame = frames[search_index]
            search_path = path + '/img' + search_frame + '.jpg'
            search_box = sample_data[search_frame]

            if read_pair:
                if match_range == 'all':
                    template_index = random.randint(0, num_frame - 1)
                elif match_range == 'init':
                    template_index = 0
                elif match_range == 'mix':
                    if random_sys() > mix_prob[dataset]:
                        template_index = 0
                    else:
                        template_index = random.randint(0, num_frame - 1)
                else:
                    template_index = random.randint(max(search_index - match_range, 0),
                                                    min(search_index + match_range, num_frame) - 1)

                template_path = path + '/img' + frames[template_index] + '.jpg'
                template_box = sample_data[frames[template_index]]

            if read_all_boxes:
                if num_sample > 1:
                    for i in range(num_sample):
                        if i != sample_index:
                            sample_frames = list(data[samples[i]].keys())
                            if search_frame in sample_frames:
                                all_boxes.append(np.array(data[samples[i]][search_frame], dtype=np.float32))

        elif dataset in ['YBB', 'YBB_val']:
            num_sample = len(data)
            samples = list(data.keys())
            if num_sample == 1:
                sample_index = 0
            else:
                sample_index = random.randint(0, num_sample - 1)

            sample_data = data[samples[sample_index]]
            frames = list(sample_data.keys())
            num_frame = len(frames)
            search_index = random.randint(0, num_frame - 1)
            search_frame = frames[search_index]
            search_path = path + '/' + search_frame + '.' + samples[sample_index] + '.x.jpg'
            search_box = sample_data[search_frame]

            if read_pair:
                if match_range == 'all':
                    template_index = random.randint(0, num_frame - 1)
                elif match_range == 'init':
                    template_index = 0
                elif match_range == 'mix':
                    if random_sys() > 0.5:
                        template_index = 0
                    else:
                        template_index = random.randint(0, num_frame - 1)
                else:
                    template_index = random.randint(max(search_index - match_range, 0),
                                                    min(search_index + match_range, num_frame) - 1)

                template_path = path + '/' + frames[template_index] + '.' + samples[sample_index] + '.x.jpg'
                template_box = sample_data[frames[template_index]]
                if (template_box[2] - template_box[0]) < 3 or (template_box[3] - template_box[1]) < 3:
                    logger.info('noisy {:s}'.format(template_path))
                    template_img = None
                else:
                    template_img = read_img(template_path, way='jpeg', read_bgr=self.read_bgr)
                    if template_img is not None:
                        template_box, noisy = get_bbox_ybb(template_img.shape, template_box)
                        template_box = np.array(template_box, dtype=np.float32)
                        if noisy:
                            template_img = None
                    else:
                        logger.info('lost {:s}'.format(template_path))

            if read_all_boxes:
                if num_sample > 1:
                    for i in range(num_sample):
                        if i != sample_index:
                            sample_frames = list(data[samples[i]].keys())
                            if search_frame in sample_frames:
                                all_boxes.append(np.array(data[samples[i]][search_frame], dtype=np.float32))

            if (search_box[2] - search_box[0]) < 3 or (search_box[3] - search_box[1]) < 3:
                logger.info('noisy {:s}'.format(search_path))
                search_img = None
            else:
                search_img = read_img(search_path, way='jpeg', read_bgr=self.read_bgr)
                if search_img is not None:
                    search_box, noisy = get_bbox_ybb(search_img.shape, search_box)
                    search_box = np.array(search_box, dtype=np.float32)
                    if noisy:
                        search_img = None
                else:
                    logger.info('lost {:s}'.format(search_path))

            if read_pair:
                return all_boxes, search_img, search_box, template_img, template_box

            else:
                return all_boxes, search_img, search_box

        elif dataset in ['GOT', 'LaSOT', 'GOT_val', 'LSOTB']:
            if dataset in ['LaSOT']:
                path = path + '/img/'

            frames = list(data.keys())
            num_frame = len(frames)
            search_index = random.randint(0, num_frame - 1)
            search_frame = frames[search_index]
            search_box = data[search_frame]
            search_path = path + '/' + search_frame + '.jpg'
            if dataset == 'VOT2017_TIR':
                search_path = path + '/' + search_frame + '.png'

            if read_pair:
                if match_range == 'all':
                    template_index = random.randint(0, num_frame - 1)
                elif match_range == 'init':
                    template_index = 0
                elif match_range == 'mix':
                    if random_sys() > mix_prob[dataset]:
                        template_index = 0
                    else:
                        template_index = random.randint(0, num_frame - 1)
                else:
                    template_index = random.randint(max(search_index - match_range, 0),
                                                    min(search_index + match_range, num_frame) - 1)

                template_box = data[frames[template_index]]
                template_path = path + '/' + frames[template_index] + '.jpg'

        if (search_box[2] - search_box[0]) < 3 or (search_box[3] - search_box[1]) < 3:
            logger.info('noisy {:s}'.format(search_path))
            search_img = None
        else:
            search_img = read_img(search_path, way='jpeg', read_bgr=self.read_bgr)
            search_box = np.array(search_box, dtype=np.float32)

        if search_img is None:
            logger.info('lost {:s}'.format(search_path))

        if read_pair:
            if template_path == search_path:
                if search_img is None:
                    template_img = search_img
                else:
                    template_img = search_img.copy()
                template_box = search_box.copy()
            else:
                if (template_box[2] - template_box[0]) < 3 or (template_box[3] - template_box[1]) < 3:
                    logger.info('noisy {:s}'.format(template_path))
                    template_img = None
                else:
                    template_img = read_img(template_path, way='jpeg', read_bgr=self.read_bgr)
                    template_box = np.array(template_box, dtype=np.float32)

            if template_img is None:
                logger.info('lost {:s}'.format(template_path))
            return all_boxes, search_img, search_box, template_img, template_box

        else:
            return all_boxes, search_img, search_box


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
