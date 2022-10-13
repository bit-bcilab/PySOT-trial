

import numpy as np
from torch.utils.data import Dataset

from trial.Augmentation import random_sys, rand, random_crop, image_augmentation
from trial.Augmentation import gray_aug, random_translation, random_occ, random_background, disappear
from trial.utils.bbox import box2roi
from trial.encoders.RectEncoder import rectangle_mask

import cv2
import random
import logging

logger = logging.getLogger("global")


class Generator(Dataset):
    def __init__(self,
                 datareader,
                 base,
                 encoder,
                 search_size,
                 template_size,
                 output_size,
                 crop_settings,
                 aug_settings,
                 encode_settings,
                 use_all_boxes=False,
                 bbox_mask_rate=0.,
                 mode='train'):
        self.encoder = encoder
        self.base = base
        self.datareader = datareader

        self.s_x = search_size
        self.s_z = template_size
        self.s_o = output_size
        self.crop_settings = crop_settings
        self.aug_settings = aug_settings
        self.encode_settings = encode_settings
        self.use_all_boxes = use_all_boxes
        self.bbox_mask_rate = bbox_mask_rate

        self.mode = mode

    def __len__(self):
        if self.mode == 'train':
            return self.datareader.num_per_epoch
        else:
            return self.datareader.num_val

    def __getitem__(self, index):
        cv2.setNumThreads(0)
        cv2.ocl.setUseOpenCL(False)

        neg = random_sys()
        # 最简单负样本对，x和z分别从不同的序列中提取
        pos = 1. if neg >= self.aug_settings['neg_pair']['prob'] else 0.

        if self.mode == 'train':
            video_index = self.datareader.train_index_[index]
        else:
            video_index = self.datareader.val_index[index]

        # 从数据集中读入训练数据：
        # 搜索图像、模板图像、搜索图像上目标的 bbox、模板图像上的 bbox
        while True:
            all_boxes, search_image, search_bbox, template_image, template_bbox = \
                self.datareader.get_data(video_index, read_pair=True, read_all_boxes=self.use_all_boxes)
            if search_image is None or template_image is None:
                video_index = self.datareader.random_choice(mode=self.mode)
            else:
                break

        # 训练时，进行随机放缩与crop
        # crop 图像后，会 keep w/h 的将数据 resize with padding 到 model 要求的输入尺寸下
        template_img, template_box = random_crop(template_image, template_bbox,
                                                 size=self.s_z, settings=self.crop_settings['template'])
        search_img, search_box, all_boxes_ = random_crop(search_image, search_bbox, mix_boxes=all_boxes,
                                                         size=self.s_x, settings=self.crop_settings['search'])
        mix_boxes = None
        if all_boxes_ is not None:
            mix_boxes = all_boxes_

        """
        When validating, only make negative pairs without other augmentations 
        Four negative cases in total:
        (1) Template and search patches are not from the same video. 
            Proposed by DaSiamRPN to prove discrimination power
        (2) Search region randomly shifts so that the target is no longer in search patch. 
            Simulate tracking failure and out-of-view.
        (3) The target is fully covered by the patch randomly cropped from the background region.
            Simulate full occlusion caused by background.
        (4) The target is fully covered by the object randomly cropped from other videos.
            Simulate full occlusion caused by other objects.
        Note that, despite strictly speaking, the 'background' is also consisted of real objects, in this work, the 'object' 
        represents the common targets in single object tracking datasets, such as human, various animals and vehicles.
        
        在验证时仅进行负样本对扩增，不做其他扩增
        四种负样本对情况：
        (1)模板/搜索不来自同一视频————微弱提升辨别力，
        (2)搜索区域中没有目标————模拟出视野/跟踪失败
        (3)目标被随机裁剪自图像中背景区域的patch完全遮挡————模拟背景遮挡
        (4)目标被随机裁剪自其他图像中背景区域的patch完全遮挡————模拟物体遮挡
        负样本对在进行标签分配时，只分配负anchor/points，并且需要在原目标位置处撒上大量负标签，避免网络学习到位置偏见，并增强辨别力
        严格来说，虽然'背景'也是物体组成的，但是本工作中的'物体'特指单目标跟踪数据集中常见的运动物体，如人、各种动物和载具等。
        """
        """
        The data augmentation methods used in positive training pairs:
        正样本对需要经历的数据扩增过程有：
        (1)生成搜索区域时的随机移动：模拟快速运动与跟踪不准的情况
        (2)生成搜索区域时的随机缩放（长宽的变化系数不同）：模拟较剧烈的目标尺度变化
        (3)将目标区域（带一定周围背景区域）随机移动到图像中的其他背景区域：增强网络的辨别能力
        (4)将目标区域（带一定周围背景区域）随机移动到其他视频的图像中的任意区域：增强网络的辨别能力
        (5)MixUp：增强网络的辨别能力
        (6)随机翻转：增强网络的辨别能力
        (7)随机旋转：模拟目标发生的平面内旋转
        (8)随机擦除：模拟局部遮挡，避免网络过度依赖局部特征
        (9)随机改变颜色：模拟光照变化
        (10)滤波处理：模拟模糊、低分辨率等图像低质量情况
        (11)随机分辨率降低：模拟低分辨率等图像低质量情况
        (12)运动模糊：模拟目标高速运动时产生的模糊
        这些数据扩增手段要求在训练时读入的是原始图像与box坐标，而不是提前crop并处理好的图像
        """
        """
        当搜索图像来自目标检测数据集（COCO, DET, VID等)时，充分利用图像上的所有标注。
        无论是正样本对还是负样本对，必须将搜索区域中的其他物体的box都设为负标签，
        并保证该图像在一个epoch中出现多次，第一次被抽中作为目标的物体，在下一次被抽中时可能就充当干扰。
        目的与负样本对扩增相同，充分增强网络的辨别能力，避免网络仅仅只是将训练集中存在的种类的物体从图像中检测出来，而不是通过相似性匹配找到目标。
        如果图像来自目标跟踪视频或者就只有一个标注物体，那么随机的从其他图像中裁剪出物体，随机地放置于目标附近，作为干扰，并同样设为负标签
        """
        if (neg < self.aug_settings['occ_object']['prob']) or (neg < self.aug_settings['neg_pair']['prob']):
            if neg < self.aug_settings['neg_pair']['prob']:
                # 随机抽取一幅图像，作为模板图
                random_index, random_mixes, random_image, random_bbox = \
                    self.datareader.get_random_data(video_index, read_all_boxes=False, mode=self.mode,
                                                    rate=self.aug_settings['neg_pair']['data_rate'])
                template_img, template_box = random_crop(random_image, random_bbox, size=self.s_z,
                                                         settings=self.crop_settings['template'])

            elif neg < self.aug_settings['disappear']['prob']:
                # 随机地在搜索图像中找到一片背景区域作为搜索patch，要求其中不能有目标
                outs = disappear(search_image, search_bbox, size=self.s_x, settings=self.crop_settings['val'])
                if outs[0] is not None:
                    search_img, search_box, mix_boxes = outs
                    pos = 0
                # bg_bbox = random_background(search_image, search_bbox, protect_settings=self.crop_settings['val'],
                #                             min_rate=self.aug_settings['disappear']['crop_rate_min'],
                #                             max_rate=self.aug_settings['disappear']['crop_rate_max'])
                # if bg_bbox is not None:
                #     search_img, search_box, mix_boxes = random_crop(search_image, bg_bbox, mix_boxes=all_boxes,
                #                                                     size=self.s_x, settings=self.crop_settings['val'])
                #     pos = 0

            else:
                # 人为制造full occlusion导致的负样本对
                if neg < self.aug_settings['occ_background']['prob']:
                    # 在当前图像中找到一块背景区域作为遮挡块
                    occ_image = search_image
                    occ_bbox = random_background(occ_image, search_bbox,
                                                 min_rate=self.aug_settings['occ_background']['crop_rate_min'],
                                                 max_rate=self.aug_settings['occ_background']['crop_rate_max'],
                                                 protect_settings=rand(self.aug_settings['occ_background']['protect_rate_min'],
                                                                       self.aug_settings['occ_background']['protect_rate_max']))
                else:
                    # 从其他序列中crop出物体作为遮挡块
                    random_index, random_mixes, random_image, random_bbox = \
                        self.datareader.get_random_data(video_index, read_all_boxes=False, mode=self.mode,
                                                        rate=self.aug_settings['occ_object']['data_rate'])
                    occ_image, occ_bbox = random_image, random_bbox

                if occ_bbox is not None:
                    occ_image_, occ_box = random_crop(occ_image, occ_bbox, self.s_x, settings=self.crop_settings['val'])
                    # 遮挡块需要带一定背景区域
                    occ_box_ = box2roi(occ_box, rate_w=self.roi_rate(), rate_h=self.roi_rate(), boundary=self.s_x)
                    box = list(map(int, occ_box_))
                    # 裁剪出遮挡块
                    occ = occ_image_[box[1]: box[3] + 1, box[0]: box[2] + 1]
                    # 将遮挡块覆盖到目标框位置上，并计算box被遮挡部分的占比
                    search_img_, occed_box, overlap = random_occ(search_img, search_box, obj=occ,
                                                                 center_rate=self.aug_settings['center_rate'],
                                                                 try_num=self.aug_settings['try_num'],
                                                                 overlap_thresh=self.aug_settings['overlap_thresh'])
                    # 当目标大部分被完全覆盖时，才完成有效的遮挡扩增
                    if occed_box is not None and overlap > self.aug_settings['overlap_thresh']:
                        search_img = search_img_
                        pos = 0

        else:
            # 随机将目标移动
            translation = random_sys()
            if translation < self.aug_settings['translation_background']['prob']:
                # 将目标区域（带一定大小的背景）从搜索图像中裁剪出来。记录下box的放大倍数，以便移动后精确计算
                scale_w, scale_h = self.roi_rate(), self.roi_rate()
                search_area = box2roi(search_box, rate_w=scale_w, rate_h=scale_h, boundary=self.s_x)
                box = list(map(int, search_area))
                target = search_img[box[1]: box[3] + 1, box[0]: box[2] + 1]

                # 将目标（带一定背景区域）移动到随机选取的其他图像上
                if translation < self.aug_settings['translation_other']['prob']:
                    random_index, random_mix, random_image, random_bbox = \
                        self.datareader.get_random_data(video_index, read_all_boxes=self.use_all_boxes, mode=self.mode,
                                                        rate=self.aug_settings['translation_other']['data_rate'])

                    random_img, random_box, random_mix_ = random_crop(random_image, random_bbox, mix_boxes=random_mix,
                                                                      size=self.s_x, settings=self.crop_settings['val'])
                    # 将目标区域（带一定大小的背景）移动到随机抽取的图像上，之前的目标仍为正，该图像上的所有物体均为负
                    search_img_, target_box = \
                        random_translation(random_img, random_box, obj=target,
                                           min_rate=self.aug_settings['translation_other']['trans_rate_min'],
                                           max_rate=self.aug_settings['translation_other']['trans_rate_max'])
                    if target_box is not None:
                        random_box = random_box.reshape((-1, 4))
                        search_box = box2roi(target_box, rate_w=1./scale_w, rate_h=1./scale_h, boundary=self.s_x)
                        search_img = search_img_
                        if random_mix_ is not None:
                            mix_boxes = np.concatenate([random_box, random_mix_])
                        else:
                            mix_boxes = random_box
                else:
                    # 将目标（带一定背景区域）移动到图像中其他位置上
                    bg_bbox = random_background(search_image, search_bbox, protect_settings=self.crop_settings['val'],
                                                min_rate=self.aug_settings['translation_background']['crop_rate_min'],
                                                max_rate=self.aug_settings['translation_background']['crop_rate_max'])
                    if bg_bbox is not None:
                        bg_img, bg_box, all_boxes_bg = random_crop(search_image, bg_bbox, mix_boxes=all_boxes,
                                                                   size=self.s_x, settings=self.crop_settings['val'])
                        # 将目标区域（带一定大小的背景）移动到图像的其他位置上，目标仍为正，无mix boxes
                        search_img_, target_box = \
                            random_translation(bg_img, bg_box, obj=target,
                                               min_rate=self.aug_settings['translation_background']['trans_rate_min'],
                                               max_rate=self.aug_settings['translation_background']['trans_rate_max'])
                        if target_box is not None:
                            search_box = box2roi(target_box, rate_w=1./scale_w, rate_h=1./scale_h, boundary=self.s_x)
                            search_img = search_img_
                            mix_boxes = all_boxes_bg

            elif translation < self.aug_settings['mixup']['prob']:
                random_index, random_mixes, random_image, random_bbox = \
                    self.datareader.get_random_data(video_index, read_all_boxes=False, rate=0.5, mode=self.mode)
                random_img, _ = random_crop(random_image, random_bbox, self.s_x, self.crop_settings['search'])
                rate = rand(0.4, 0.5)
                search_img = search_img * (1. - rate) + random_img * rate

        # 当图像中只有目标这一个框时，加入干扰物体
        # 随机加入干扰物体
        # if mix_boxes is None and random_sys() < self.aug_settings['mix']:
        if random_sys() < self.aug_settings['mix']['prob']:
            num_mix = random.randint(self.aug_settings['mix']['min_num'], self.aug_settings['mix']['max_num'])
            mix_boxes_ = []
            counter = 0
            for i in range(num_mix):
                random_index, random_mixes, random_image, random_bbox = \
                    self.datareader.get_random_data(video_index, read_all_boxes=False, mode=self.mode,
                                                    rate=self.aug_settings['mix']['data_rate'])
                random_img, random_box = random_crop(random_image, random_bbox,
                                                     size=self.s_x, settings=self.crop_settings['val'])

                random_box_ = box2roi(random_box, rate_w=self.roi_rate(), rate_h=self.roi_rate(), boundary=self.s_x)
                box = list(map(int, random_box_))
                mix = random_img[box[1]: box[3] + 1, box[0]: box[2] + 1]
                search_img, mixed_box = random_translation(search_img, search_box, obj=mix,
                                                           min_rate=self.aug_settings['mix']['trans_rate_min'],
                                                           max_rate=self.aug_settings['mix']['trans_rate_max'])
                if mixed_box is not None:
                    mix_boxes_.append(mixed_box)
                    counter += 1
            if counter > 0:
                mix_boxes_ = np.array(mix_boxes_).reshape((-1, 4))
                mix_boxes = mix_boxes_

        # 负样本对时，不进行普通扩增
        template_img, template_box = image_augmentation(template_img, template_box, self.aug_settings['template'])
        if pos:
            if mix_boxes is None:
                search_img, search_box = image_augmentation(search_img, search_box, self.aug_settings['search'])
            else:
                search_img, search_box, mix_boxes = image_augmentation(search_img, search_box,
                                                                       self.aug_settings['search'], mix_boxes)

        if self.aug_settings['gray']['prob'] and random_sys() < self.aug_settings['gray']['prob']:
            template_img = gray_aug(template_img)
            search_img = gray_aug(search_img)

        # dataset = video_index[0].replace('/', '_')
        # frame = video_index[1].replace('/', '_')
        # txt = open('ep8-data-1.txt', 'a')
        # txt.write('index: ' + str(index) + ', ' + dataset + ',' + frame + '\r')
        # txt.close()
        # img = search_img.astype(np.uint8)
        # cv2.imwrite('imgs/' + dataset + '-' + frame + '-' + str(index) + '.jpg', img)
        template_img = template_img.transpose((2, 0, 1)).astype(np.float32)
        search_img = search_img.transpose((2, 0, 1)).astype(np.float32)

        cls, delta = self.encoder(search_box, mix_boxes, pos, self.s_x, self.s_o, **self.encode_settings)

        outputs = {'template': template_img,
                   'search': search_img,
                   'bbox': search_box.astype(np.float32),
                   'z_box': template_box.astype(np.float32),
                   'label_cls': cls.astype(np.int64),
                   'label_loc': delta,
                   'pos': pos}
        if self.bbox_mask_rate:
            bbox_mask = rectangle_mask(search_box, points=self.points, rate=self.bbox_mask_rate, positive=pos)
            outputs['bbox_mask'] = bbox_mask.astype(np.float32)
        return outputs

    def roi_rate(self):
        return rand(self.aug_settings['roi_rate_min'], self.aug_settings['roi_rate_max'])

# class LabelUpdater(Dataset):
#     def __init__(self,
#                  batch_size,
#                  base,
#                  settings):
#         self.batch_size = batch_size
#         self.base = base
#         self.settings = settings
#         self.data = {}
#
#     def __len__(self):
#         return self.batch_size
#
#     def prepare(self, data):
#         label = data['label_cls'].cpu().detach().numpy()
#         positive = data['pos'].cpu().detach().numpy()
#         score = data['score'].cpu().detach().numpy()
#         iou = data['iou'].cpu().detach().numpy()
#
#         if self.base == 'anchor':
#             positive_ = positive[:, None, None, None]
#         elif self.base == 'point':
#             positive_ = positive[:, None, None]
#         negative = positive_ == 0.
#         negative = negative.astype(np.float32)
#
#         # 预备正样本
#         pos_ = label == 1.
#         pos_ = pos_.astype(np.float32)
#
#         # pos范围
#         pos_mask = label >= 0.
#         pos_mask = pos_mask.astype(np.float32)
#
#         # neg范围
#         label_ = label * positive_ - 1. * negative * np.ones_like(label)
#         neg_mask = label_ < 0.
#         neg_mask = neg_mask.astype(np.float32)
#
#         # 首先筛选出所有的正样本对内pos范围内点上的iou值
#         pos_iou = iou * pos_mask
#         # 进一步筛选出iou值大于阈值的点作为正样本
#         pos = pos_iou >= self.settings['pos_iou_thresh']
#         pos = pos.astype(np.float32)
#         pos_label = pos * positive_
#
#         # 检查每个batch是否有符合要求的正样本
#         dim = [i for i in range(1, label.ndim)]
#         pos_num_ = pos_label.sum(tuple(dim))
#         if_pos = pos_num_ > 0.
#         if_pos = if_pos.astype(np.float32)
#
#         # 正样本范围外且得分高于阈值的box
#         neg_score = score * neg_mask
#         neg0 = neg_score >= self.settings['neg_score_thresh']
#         neg0 = neg0.astype(np.float32)
#         # 正样本范围外且IOU小于于阈值的box
#         neg_iou = iou * neg_mask
#         neg1 = neg_iou < self.settings['neg_iou_thresh']
#         neg1 = neg1.astype(np.float32)
#         # 两者的交集为异常高得分，将作为难负样本
#         hard_neg = neg0 * neg1
#
#         data = {'pos': positive, 'pos_': pos_, 'if_pos': if_pos, 'pos_iou': pos_iou,
#                 'hard_neg': hard_neg, 'neg_mask': neg_mask, 'pos_label': pos_label, 'pos_num_': pos_num_}
#         self.data = data
#
#     def __getitem__(self, index):
#         print('index: %d' % index)
#         positive = self.data['pos'][index]
#         if_pos = self.data['if_pos'][index]
#         pos_ = self.data['pos_'][index]
#         pos_iou = self.data['pos_iou'][index]
#         hard_neg = self.data['hard_neg'][index]
#         neg_mask = self.data['neg_mask'][index]
#         pos_label = self.data['pos_label'][index]
#         pos_num_ = self.data['pos_num_'][index]
#
#         new_label = np.ones_like(pos_label)
#
#         hard_neg_ = random_choice(hard_neg, random.randint(self.settings['hard_neg_num'] - 4, self.settings['hard_neg_num'] + 4))
#         easy_neg_ = random_choice(neg_mask, random.randint(self.settings['easy_neg_num'] - 2, self.settings['easy_neg_num'] + 2))
#
#         if positive:
#             if if_pos:
#                 # 当满足条件的box数超过总数时，选取满足iou条件的前K名box作为正样本
#                 max_pos_num = random.randint(self.settings['pos_num'] - 2, self.settings['pos_num'] + 2)
#                 if pos_num_ > max_pos_num:
#                     p = np.where(pos_iou > 0.)
#                     pos_index = np.argpartition(pos_iou[p], -max_pos_num, axis=0)[-max_pos_num:]
#                     p_ = tuple(i[pos_index] for i in p)
#                     # 先清空，再填充
#                     pos_label = np.zeros_like(pos_label)
#                     pos_label[p_] = 1.
#             # 有的batch应该是正样本对，但是没有符合IOU阈值要求的box
#             # 无满足条件的box时，使用原分配方式产生的预备正样本标签
#             else:
#                 pos_label = pos_
#
#         # 负样本对时，强制将box位置设置为负样本（基于预备label）
#         else:
#             hard_neg_ = hard_neg_ + pos_
#         neg_label = hard_neg_ + easy_neg_
#         neg_label = neg_label > 0.
#         neg_label = neg_label.astype(np.float32)
#         neg_label = random_choice(neg_label, random.randint(self.settings['neg_num'] - 4, self.settings['neg_num'] + 4))
#
#         # 正Box位置标记为1，负样本位置标记为0，其余位置标记为-1，忽略
#         new_label = -1. * new_label + 1. * neg_label + 2. * pos_label
#         return new_label.astype(np.int64)
