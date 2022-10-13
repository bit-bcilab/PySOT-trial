

import numpy as np

from trial.utils.image import letterbox_image
from trial.utils.bbox import corner2center, Center, center2corner, Corner, clip_bbox_corner, box2roi
from trial.utils.rand import rand, random_sys
from trial.utils.iou import box_iou

import random
import math
import cv2

cv2.setNumThreads(0)
cv2.ocl.setUseOpenCL(False)

PCA_STD = 0.1
RGB_VAR = np.array([[-0.55919361, 0.98062831, - 0.41940627],
                    [1.72091413, 0.19879334, - 1.82968581],
                    [4.64467907, 4.73710203, 4.88324118]], dtype=np.float32)
EIGVAL = np.array([55.46, 4.794, 1.148], dtype=np.float32)
EIGVEC = np.array([[-0.5836, -0.6948, 0.4203],
                   [-0.5808, -0.0045, -0.8140],
                   [-0.5675, 0.7192, 0.4009]], dtype=np.float32)


def random_scale(min_scale, max_scale, min_scale_=None, max_scale_=None, scale_=None):
    if scale_ is not None:
        over_1 = (scale_ > 1.)
    else:
        over_1 = (random_sys() > 0.5)

    if over_1:
        if random_sys() > 0.89 and min_scale_ is not None:
            scale = rand(max_scale, max_scale_)
        else:
            scale = rand(1., max_scale)
    else:
        if random_sys() > 0.89 and max_scale_ is not None:
            scale = rand(min_scale_, min_scale)
        else:
            scale = rand(min_scale, 1.)
    return scale


def random_crop(image, bbox, size, settings, mix_boxes=None):
    """
    Local模式下的 random scale and shift crop augmentation
    refers to the paper of SiamRPN++ and the codes in PySOT
    确定crop区域后，利用opencv的仿射变换从原图上提取出来

    :param image: 原图像
    :param bbox:
    :param size:  网络要求输入尺寸
    :param settings:
    :param mix_boxes:
    :return: 随机放缩与crop扩增的图像与变换后的bbox
    """
    mean_channel = np.mean(image, axis=(0, 1))[None, None, :]
    bbox = Corner(float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3]))
    ox, oy, ow, oh = corner2center(bbox)
    context_amount = settings['context_amount']
    box_protect_rate = settings['box_protect_rate']
    crop_size_rate = settings['crop_size_rate']
    min_scale, max_scale = settings['min_scale'], settings['max_scale']
    min_scale_, max_scale_ = None, None
    if 'min_scale_' in settings and 'max_scale_' in settings:
        min_scale_, max_scale_ = settings['min_scale_'], settings['max_scale_']
    # 保留原长宽比的数据的比例
    scale_w, scale_h = 1., 1.
    if random_sys() > settings['keep_scale_prob']:
        # crop区域的长宽不相等，模拟目标形状变化与畸变
        if random_sys() > 0.82:
            scale_w = random_scale(min_scale, max_scale)
            scale_h = random_scale(min_scale, max_scale, scale_=scale_w)
        else:
            # 计算crop区域时，w,h不等比例地缩放，避免网络学习区域尺寸与目标尺寸之间的依赖关系
            if random_sys() > 0.6:
                scale_w_ = random_scale(min_scale=min_scale, max_scale=max_scale)
                scale_h_ = random_scale(min_scale=min_scale, max_scale=max_scale, scale_=scale_w_)
                ow = ow * scale_w_
                oh = oh * scale_h_
            # 计算crop区域时， w,h等比例缩放，使区域尺寸也随之等比例缩放
            else:
                scale = random_scale(min_scale=min_scale, max_scale=max_scale,
                                     min_scale_=min_scale_, max_scale_=max_scale_)
                ow = ow * scale
                oh = oh * scale

    crop_size = round(np.sqrt((ow + context_amount * (ow + oh)) * (oh + context_amount * (ow + oh))) * crop_size_rate)
    crop_w = int(scale_w * crop_size)
    crop_h = int(scale_h * crop_size)

    crop_box_center = Center(ox, oy, crop_w, crop_h)
    crop_box = center2corner(crop_box_center)

    if random_sys() > settings['keep_center_prob']:
        protect_x = box_protect_rate * crop_w
        protect_y = box_protect_rate * crop_h
        x = np.arange(int(bbox.x2 + protect_x - crop_w) + 1, int(bbox.x1 - protect_x) - 1)
        y = np.arange(int(bbox.y2 + protect_y - crop_h) + 1, int(bbox.y1 - protect_y) - 1)
        if x.shape[0] > 0 and y.shape[0] > 0:
            # random.randint(int(bbox.x2 + protect_x - crop_w), int(bbox.x1 - protect_x + 1))
            x_ = random.choice(x)
            y_ = random.choice(y)
            crop_box = Corner(x_, y_, x_ + crop_w, y_ + crop_h)

    crop_box = np.array(crop_box, np.int64)

    h, w = image.shape[:2]
    left_pad = max(0, -crop_box[0])
    top_pad = max(0, -crop_box[1])
    right_pad = max(0, crop_box[2] - w + 1)
    bottom_pad = max(0, crop_box[3] - h + 1)
    pad_image_size = (h + top_pad + bottom_pad, w + left_pad + right_pad, 3)
    pad_image = np.ones(pad_image_size) * mean_channel
    pad_image = pad_image.astype(np.uint8)
    pad_image[top_pad: top_pad + h, left_pad: left_pad + w, :] = image
    crop_box = Corner(crop_box[0] + left_pad, crop_box[1] + top_pad, crop_box[2] + left_pad, crop_box[3] + top_pad)
    crop_image = pad_image[crop_box[1]: crop_box[3] + 1, crop_box[0]: crop_box[2] + 1, :]

    # box在crop出区域中的相对坐标 = crop patch的左上角在原图像中的绝对坐标 - box在原图像中的绝对坐标
    bbox = Corner(max(bbox.x1 + float(left_pad - crop_box[0]), 0.),
                  max(bbox.y1 + float(top_pad - crop_box[1]), 0.),
                  min(bbox.x2 + float(left_pad - crop_box[0]), float(crop_box[2] - crop_box[0])),
                  min(bbox.y2 + float(top_pad - crop_box[1]), float(crop_box[3] - crop_box[1])))
    bbox = np.array(bbox, dtype=np.float32)

    if mix_boxes is None:
        return letterbox_image(crop_image, size, bbox, padding=mean_channel)
    else:
        # 所有的干扰物体box也要进行坐标系转换
        all_box = mix_boxes.copy()
        num_boxes = len(all_box)
        if num_boxes > 0:
            for i in range(num_boxes):
                # 该过程可并行化
                box = all_box[i]
                box = Corner(max(box[0] + float(left_pad - crop_box[0]), 0.),
                             max(box[1] + float(top_pad - crop_box[1]), 0.),
                             min(box[2] + float(left_pad - crop_box[0]), float(crop_box[2] - crop_box[0])),
                             min(box[3] + float(top_pad - crop_box[1]), float(crop_box[3] - crop_box[1])))
                all_box[i] = box

            all_box_ = np.concatenate([bbox.reshape(-1, 4), np.array(all_box, dtype=np.float32)], axis=0)
            crop_image, all_box_ = letterbox_image(crop_image, size, all_box_, padding=mean_channel)
            bbox = all_box_[0, :]

            target_area = np.prod(bbox[:2] - bbox[2:])
            all_box = all_box_[1:, :].reshape((-1, 4))
            diff = all_box[:, 2:] - all_box[:, :2]
            area = np.prod(diff, axis=1)
            not_ignore = np.where((diff[:, 0] > 0.) & (diff[:, 1] > 0.) & (area > 225.) & (area < 9. * target_area))
            if not_ignore[0].shape[0] > 0:
                all_box = all_box[not_ignore]
            else:
                all_box = None
        else:
            crop_image, bbox = letterbox_image(crop_image, size, bbox, padding=mean_channel)
            all_box = None
        return crop_image, bbox, all_box


def image_augmentation(image, bbox, aug_settings, mix_boxes=None):
    cv2.setNumThreads(0)
    cv2.ocl.setUseOpenCL(False)

    image = image.astype(np.float32)
    random = random_sys(3)

    # flipping and rotating
    if bbox is not None and random[0] < aug_settings['flip']['prob']:
        if mix_boxes is None:
            image, bbox = flip_aug(image, bbox)
        else:
            all_boxes = np.concatenate([bbox.reshape((-1, 4)), mix_boxes], axis=0)
            image, all_boxes = flip_aug(image, all_boxes)
            bbox = all_boxes[0, :]
            mix_boxes = all_boxes[1:, :]

    elif bbox is not None and random[0] < aug_settings['rotate']['prob']:
        angle = rand(aug_settings['rotate']['min_angle'], aug_settings['rotate']['max_angle'])
        if random_sys() >= 0.5:
            angle = angle * -1.
        if mix_boxes is None:
            image, bbox = rotate_aug(image, bbox, angle=angle)
        else:
            all_boxes = np.concatenate([bbox.reshape((-1, 4)), mix_boxes], axis=0)
            image, all_boxes = rotate_aug(image, all_boxes, angle=angle)
            bbox = all_boxes[0, :]
            mix_boxes = all_boxes[1:, :]

    # filtering, motion blurring and random erasing
    if random[1] < aug_settings['blur']['prob']:
        image = blur_aug(image)

    elif random[1] < aug_settings['motion']['prob']:
        degree = int(rand(aug_settings['motion']['min_degree'], aug_settings['motion']['max_degree']))
        angle = int(rand(aug_settings['rotate']['min_angle'], aug_settings['rotate']['max_angle']))
        image = motion_blur(image, degree=degree, angle=angle)

    elif random[1] < aug_settings['erase']['prob']:
        image = erase_aug(image, bbox)

    elif random[1] < aug_settings['resolution']['prob']:
        image = resolution_aug(image, rand(aug_settings['resolution']['low'], aug_settings['resolution']['high']))

    # color and pca variation
    if random[2] < aug_settings['pca']['prob']:
        image = pca_aug(image)
    elif random[2] < aug_settings['color']['prob']:
        image = color_aug(image)

    if mix_boxes is None:
        return image, bbox
    else:
        return image, bbox, mix_boxes


def rand_kernel():
    size = random.randrange(5, 20, 2)
    kernel = np.zeros((size, size))
    c = int(size/2)
    wx = random_sys()
    kernel[:, c] += 1. / size * wx
    kernel[c, :] += 1. / size * (1-wx)
    return kernel


def gray_aug(image):
    image = image.astype(np.uint8)
    grayed = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image = cv2.cvtColor(grayed, cv2.COLOR_GRAY2BGR)
    image = image.astype(np.float32)
    return image


def pca_aug(image):
    rd = []
    for i in range(3):
        rd.append(random.normalvariate(0., PCA_STD))
    alpha = np.array(rd, dtype=np.float32).reshape((-1, ))
    offset = np.dot(EIGVEC * alpha, EIGVAL)
    image = image + offset
    return image


def blur_aug(image):
    kernel = rand_kernel()
    image = cv2.filter2D(image, -1, kernel)
    return image


def random_bright(image, delta=64):
    delta = random.uniform(-delta, delta)
    image += delta
    image = image.clip(min=0, max=255)
    return image


def random_contrast(image, lower=0.5, upper=1.5):
    alpha = random.uniform(lower, upper)
    image *= alpha
    image = image.clip(min=0, max=255)
    return image


def random_saturation(image, lower=0.5, upper=1.5):
    image[:, :, 1] *= random.uniform(lower, upper)
    image = image.clip(min=0, max=255)
    return image


def color_aug(image_):
    image = image_.astype(np.float32)
    aug = [random_bright, random_contrast]
    random.shuffle(aug)
    rd = random_sys(2)
    for i in range(2):
        if rd[i] > 0.5:
            image = aug[i](image)
    return image


def motion_blur(image, degree=10, angle=40):
    cv2.setNumThreads(0)
    cv2.ocl.setUseOpenCL(False)

    # 这里生成任意角度的运动模糊kernel的矩阵， degree越大，模糊程度越高
    m = cv2.getRotationMatrix2D((degree / 2, degree / 2), angle, 1)
    motion_blur_kernel = np.diag(np.ones(degree))
    motion_blur_kernel = cv2.warpAffine(motion_blur_kernel, m, (degree, degree))
    motion_blur_kernel = motion_blur_kernel / degree
    blurred = cv2.filter2D(image, -1, motion_blur_kernel)
    return blurred


def flip_aug(image, bbox):
    image = cv2.flip(image, 1)
    width = image.shape[1]
    if len(bbox.shape) == 1:
        bbox = Corner(width - 1 - bbox[2], bbox[1], width - 1 - bbox[0], bbox[3])
    else:
        for i in range(bbox.shape[0]):
            box = bbox[i, :]
            box = Corner(width - 1 - box[2], box[1], width - 1 - box[0], box[3])
            bbox[i, :] = box
    bbox = np.array(bbox, dtype=np.float32)
    return image, bbox


def rotate_aug(img, bbox, angle=5, scale=1.):
    """
    references：https://blog.csdn.net/saltriver/article/details/79680189
                https://www.ctolib.com/topics-44419.html
    关于仿射变换：https://www.zhihu.com/question/20666664

    :param img: 图像array,(h,w,c)
    :param bbox: [x_min, y_min, x_max, y_max]形式的 bounding box
    :param angle: angle of rotation
    :param scale: 默认 1.
    :return:
        rot_img:旋转后的图像array;
        rot_bboxes:旋转后的boundingbox坐标list
    """
    # ---------------------- 旋转图像 ----------------------
    cv2.setNumThreads(0)
    cv2.ocl.setUseOpenCL(False)

    w = img.shape[1]
    h = img.shape[0]
    # 角度变弧度
    rangle = np.deg2rad(angle)
    # 计算新图像的宽度和高度，分别为最高点和最低点的垂直距离
    nw = (abs(np.sin(rangle) * h) + abs(np.cos(rangle) * w)) * scale
    nh = (abs(np.cos(rangle) * h) + abs(np.sin(rangle) * w)) * scale
    # 获取图像绕着某一点的旋转矩阵
    # getRotationMatrix2D(Point2f center, double angle, double scale)
    # Point2f center：表示旋转的中心点
    # double angle：表示旋转的角度
    # double scale：图像缩放因子
    # 参考：https://cloud.tencent.com/developer/article/1425373
    rot_mat = cv2.getRotationMatrix2D((nw * 0.5, nh * 0.5), angle, scale)  # 返回 2x3 矩阵
    # 新中心点与旧中心点之间的位置
    rot_move = np.dot(rot_mat, np.array([(nw - w) * 0.5, (nh - h) * 0.5, 0]))
    # the move only affects the translation, so update the translation
    # part of the transform
    rot_mat[0, 2] += rot_move[0]
    rot_mat[1, 2] += rot_move[1]
    # 仿射变换
    rotated_img = cv2.warpAffine(img,
                                 rot_mat,
                                 (int(math.ceil(nw)), int(math.ceil(nh))),
                                 flags=cv2.INTER_LANCZOS4,
                                 borderMode=cv2.BORDER_CONSTANT,  # borderMode=cv2.BORDER_REFLECT,
                                 borderValue=tuple(np.mean(img, axis=(0, 1)).astype(np.uint8).tolist()))

    rotated_img = cv2.resize(rotated_img, (w, h), interpolation=cv2.INTER_CUBIC)

    # ---------------------- 矫正boundingbox ----------------------
    # rot_mat是最终的旋转矩阵
    # 获取原始bbox的四个中点，然后将这四个点转换到旋转后的坐标系下
    def rotate_box(box):
        x_min = box[0]
        y_min = box[1]
        x_max = box[2]
        y_max = box[3]
        point1 = np.dot(rot_mat, np.array([(x_min + x_max) / 2, y_min, 1]))
        point2 = np.dot(rot_mat, np.array([x_max, (y_min + y_max) / 2, 1]))
        point3 = np.dot(rot_mat, np.array([(x_min + x_max) / 2, y_max, 1]))
        point4 = np.dot(rot_mat, np.array([x_min, (y_min + y_max) / 2, 1]))

        # 合并np.array
        concat = np.vstack((point1, point2, point3, point4))  # 在竖直方向上堆叠
        # 改变array类型
        concat = concat.astype(np.int32)
        # 得到旋转后的坐标
        rx, ry, rw, rh = cv2.boundingRect(concat)
        rx_min = rx
        ry_min = ry
        rx_max = rx + rw
        ry_max = ry + rh
        # 加入list中
        rotated_box = np.array([rx_min, ry_min, rx_max, ry_max])

        scale_w = float(w) / nw
        scale_h = float(h) / nh
        rotated_box = rotated_box * np.array([scale_w, scale_h, scale_w, scale_h])
        return rotated_box

    if len(bbox.shape) == 1:
        bbox = rotate_box(bbox)
    else:
        for i in range(bbox.shape[0]):
            box = bbox[i, :]
            box = rotate_box(box)
            bbox[i, :] = box

    bbox = np.array(bbox, dtype=np.float32)
    return rotated_img, bbox


def erase_aug(image, bbox):
    """
    随机擦除目标上部分区域

    :param image:
    :param bbox:
    :return:
    """
    ih, iw = image.shape[:2]
    bbox_xywh = corner2center(bbox)
    boundary = bbox_xywh * np.array([1., 1., rand(1., 1.2), rand(1., 1.2)], np.float32)

    w = boundary[2]
    h = boundary[3]

    scale1 = rand(0.2, 0.4)
    scale2 = rand(0.15, rand(0.2, 0.32) / scale1)
    if random_sys() > 0.5:
        ew = int(w * scale1)
        eh = int(h * scale2)
    else:
        ew = int(w * scale2)
        eh = int(h * scale1)

    area = bbox - np.array([0., 0., ew, eh], np.float32)

    mask = np.zeros(shape=(ih, iw), dtype=np.float32)
    box = list(map(int, area))
    mask[box[1]: box[3] + 1, box[0]: box[2] + 1] = 1.

    pos = np.where(mask == 1.)
    num = pos[0].shape[0]
    if num > 0:
        if num > 1:
            choice = random.randint(0, num-1)
        else:
            choice = 0
        y1, x1 = pos[0][choice], pos[1][choice]
        if y1 + eh < bbox[3] and x1 + ew < bbox[2]:
            image[y1: y1 + eh, x1: x1 + ew, :] = 127.5
    return image


def resolution_aug(image, rate):
    """
    先缩小后恢复，造成分辨率损失
    """
    cv2.setNumThreads(0)
    cv2.ocl.setUseOpenCL(False)

    ih, iw = image.shape[:2]
    nw, nh = int(iw * rate), int(ih * rate)
    image = cv2.resize(image, (nw, nh))
    image = cv2.resize(image, (iw, ih))
    return image


def random_translation(image, bbox, obj, min_rate=0.8, max_rate=1.4):
    """
    随机地将object放置在box区域附近

    :param image:
    :param bbox:
    :param obj:
    :param min_rate:
    :param max_rate:
    :return:
    """
    obj_h, obj_w = obj.shape[:2]
    ih, iw = image.shape[:2]
    size = np.array([-obj_w, -obj_h, 0., 0.], np.float32)
    boundary = np.array([0., 0., iw - obj_w - 1, ih - obj_h - 1], np.float32)

    bbox_xywh = corner2center(bbox)
    min_range = bbox_xywh * np.array([1., 1., min_rate, min_rate], np.float32)
    max_range = bbox_xywh * np.array([1., 1., max_rate, max_rate], np.float32)
    max_range_corner_ = center2corner(max_range)
    min_range_corner_ = center2corner(min_range)

    min_range_corner = min_range_corner_ + size
    min_x1y1 = np.maximum(min_range_corner[:2], boundary[:2])
    min_x2y2 = np.minimum(min_range_corner[2:], boundary[2:])
    min_range = np.concatenate([min_x1y1, min_x2y2])

    max_range_corner = max_range_corner_ + size
    max_x1y1 = np.maximum(max_range_corner[:2], boundary[:2])
    max_x2y2 = np.minimum(max_range_corner[2:], boundary[2:])
    max_range = np.concatenate([max_x1y1, max_x2y2])

    min_mask = np.ones(shape=(ih, iw), dtype=np.float32)
    box = list(map(int, min_range))
    min_mask[box[1]: box[3] + 1, box[0]: box[2] + 1] = 0.

    max_mask = np.zeros(shape=(ih, iw), dtype=np.float32)
    box = list(map(int, max_range))
    max_mask[box[1]: box[3] + 1, box[0]: box[2] + 1] = 1.

    mask = min_mask * max_mask

    pos = np.where(mask == 1.)
    num = pos[0].shape[0]
    if num > 0:
        if num > 1:
            choice = random.randint(0, num-1)
        else:
            choice = 0
        y1, x1 = pos[0][choice], pos[1][choice]
        if y1 + obj_h < ih and x1 + obj_w < iw:
            image[y1: y1 + obj_h, x1: x1 + obj_w] = obj
            mixed_box = np.array([x1, y1, x1 + obj_w, y1 + obj_h], dtype=np.float32)
        else:
            mixed_box = None
    else:
        mixed_box = None
    return image, mixed_box


def random_occ(image, bbox, obj, center_rate, try_num=5, overlap_thresh=0.8):
    """
    随机地将object覆盖在box区域上
    """
    overlap = 0.
    obj_h, obj_w = obj.shape[:2]
    ih, iw = image.shape[:2]
    object_area = np.prod(bbox[2:] - bbox[:2])
    x, y, w, h = corner2center(bbox)

    counter = 0
    while counter < try_num and overlap <= overlap_thresh:
        x_rate, y_rate = rand(-center_rate, center_rate), rand(-center_rate, center_rate)
        ox = x + x_rate * w
        oy = y + y_rate * h
        occ_box = np.array([ox, oy, obj_w, obj_h], np.float32)
        occ_box = center2corner(occ_box)
        occ_box = clip_bbox_corner(occ_box, (ih, iw))
        inter_x1y1 = np.maximum(bbox[:2], occ_box[:2])
        inter_x2y2 = np.minimum(bbox[2:], occ_box[2:])
        inter_area = np.prod(inter_x2y2 - inter_x1y1)
        overlap = inter_area / (object_area + 1e-6)
        counter += 1

    if overlap > overlap_thresh:
        occ_box = list(map(int, occ_box))
        x1, y1, x2, y2 = occ_box
        if y1 + obj_h < ih and x1 + obj_w < iw:
            image[y1: y1 + obj_h, x1: x1 + obj_w] = obj
            occ_box = np.array([x1, y1, x1 + obj_w, y1 + obj_h], dtype=np.float32)
        else:
            occ_box = None
    else:
        occ_box = None
    return image, occ_box, overlap


def random_background(image, bbox, protect_settings, min_rate=0.95, max_rate=1.05):
    """
    在搜索图像中，随机crop一块尺寸形状与box相似的背景patch
    三种用途：
    直接把该背景patch覆盖到目标box上，模拟遮挡干扰；
    以该patch为中心，crop出新的搜索区域，模拟目标消失与跟踪失败；
    以该patch为中心，crop出新的搜索区域，之后将目标Box移动到这片背景区域上，以增强网络的辨别能力；
    后两种用途要求以此背景patch为中心得到的搜索区域中不能含有目标box。
    """
    ih, iw = image.shape[:2]

    bbox_xywh = corner2center(bbox)
    w = int(bbox_xywh[2] * rand(min_rate, max_rate))
    h = int(bbox_xywh[3] * rand(min_rate, max_rate))

    # 取背景区域的左上角坐标范围
    boundary = np.array([1., 1., iw - w - 1., ih - h - 1.], np.float32)

    # 保护区，禁止被crop的区域
    # 当在图像中寻找目标的平移区域时，应避免在目标平移的新位置建立的搜索区域中仍然有目标，造成混淆
    # 利用crop_settings，按照建立搜索区域的方式，以目标为中心确定一片禁区，目标必须移动到禁区外
    # 当要从图像中随机crop一片区域作为遮挡物时，只要遮挡区域不与目标有交集即可（为增强辨别力，可允许有一点交集）
    if isinstance(protect_settings, dict):
        t = protect_settings['context_amount'] * (w + h)
        size = int(math.sqrt((w + t) * (h + t)) * protect_settings['crop_size_rate'])
        area = np.array([bbox_xywh[0], bbox_xywh[1], size, size], np.float32)
    else:
        area = bbox_xywh * np.array([1., 1., 1. + protect_settings, 1. + protect_settings], np.float32)

    area = center2corner(area)
    # 保护区的左上角坐标
    area = area + np.array([-w, -h, 0., 0.], np.float32)

    # 不能超过图像区域
    x1y1 = np.maximum(area[:2], np.array([1., 1.], np.float32))
    x2y2 = np.minimum(area[2:], np.array([iw - 1., ih - 1.], np.float32))
    corner = np.concatenate([x1y1, x2y2])

    mask1 = np.zeros(shape=(ih, iw), dtype=np.float32)
    box = list(map(int, boundary))
    mask1[box[1]: box[3], box[0]: box[2]] = 1.

    mask2 = np.ones(shape=(ih, iw), dtype=np.float32)
    box = list(map(int, corner))
    mask2[box[1]: box[3], box[0]: box[2]] = 0.

    mask = mask1 * mask2

    pos = np.where(mask == 1.)
    num = pos[0].shape[0]

    if num > 0:
        if num > 1:
            choice = random.randint(0, num-1)
        else:
            choice = 0
        y1, x1 = pos[0][choice], pos[1][choice]
        if y1 + h < ih and x1 + w < iw:
            mixed_box = np.array([x1, y1, x1 + w, y1 + h], dtype=np.float32)
        else:
            mixed_box = None
    else:
        mixed_box = None
    return mixed_box


def disappear(image, bbox_, size, settings):
    context_amount = settings['context_amount']
    crop_size_rate = settings['crop_size_rate'] * rand(0.8, 1.1)

    ih, iw = image.shape[:2]
    pad_w, pad_h = int(iw * 0.125), int(ih * 0.125)
    nw, nh = iw + 2 * pad_w, ih + 2 * pad_h
    x, y, w, h = corner2center(bbox_)

    crop_size = round(np.sqrt((w + context_amount * (w + h)) * (h + context_amount * (w + h))) * crop_size_rate)
    if crop_size > (nw - w - 5) or crop_size > (nh - h - 5):
        return None, None, None
    crop_box = np.array([[0., 0., crop_size, crop_size]], dtype=np.float32)

    bbox = bbox_ + np.array([pad_w, pad_h, pad_w, pad_h], dtype=np.float32)
    # pad_img_ = pad_img.copy()
    # box = list(map(int, bbox_))
    # cv2.rectangle(pad_img_, (box[0], box[1]), (box[2], box[3]), (0, 255, 0), 2)
    # cv2.imshow('1', pad_img_)
    # cv2.waitKey()

    bbox = box2roi(bbox, rand(0.9, 1.1), rand(0.9, 1.1))

    xs, ys = np.arange(1, nw - crop_size - 1), np.arange(1, nh - crop_size - 1)
    xs, ys = np.meshgrid(xs, ys)
    grid = np.concatenate((xs[..., None], ys[..., None]), axis=-1)
    grid = np.reshape(grid, (-1, 2))
    count = 0
    while count < 5:
        count += 1
        np.random.shuffle(grid)
        pos = grid[:500]
        crop_boxes = np.concatenate([pos, pos], axis=-1) + crop_box
        ious = box_iou(bbox, crop_boxes)
        indexes = np.where(ious <= 0.)[0]
        if len(indexes) > 0:
            index = random.choice(indexes)
            crop_box = crop_boxes[index].astype(np.int)

            mean_channel = np.mean(image, axis=(0, 1)).astype(np.uint8)[None, None, :]
            pad_img = mean_channel * np.ones((nh, nw, 3), dtype=np.uint8)
            pad_img[pad_h: pad_h + ih, pad_w: pad_w + iw] = image

            crop_img = pad_img[crop_box[1]: crop_box[3], crop_box[0]: crop_box[2]]
            box_new_xywh = np.array([crop_size / 2, crop_size / 2, w, h], dtype=np.float32)
            bbox_new = center2corner(box_new_xywh)
            # cv2.imshow('0', pad_img)
            # cv2.imshow('1', crop_img)
            # cv2.waitKey()

            crop_image, bbox_new = letterbox_image(crop_img, size, bbox_new, padding=mean_channel)
            return crop_image, bbox_new, None

    a = 0
    return None, None, None



if __name__ == '__main__':
    from trial.utils.bbox import box2roi

    path = 'F://DataBase/Benchmark/David3/img/0001.jpg'
    image = cv2.imread(path)
    bbox = np.array([83, 200, 118, 331], dtype=np.float32)
    box_ = box2roi(bbox, 2.8, 1.4, [image.shape[1], image.shape[0]])
    box = list(map(int, box_))
    cv2.rectangle(image, (box[0], box[1]), (box[2], box[3]), (0, 255, 0), 2)
    cv2.imshow('1', image)

    bbox = bbox - np.concatenate([box_[:2], box_[:2]])
    patch = image[box[1]: box[3] + 1, box[0]: box[2] + 1, :]
    box = list(map(int, bbox))
    patch_ = patch.copy()
    cv2.rectangle(patch_, (box[0], box[1]), (box[2], box[3]), (0, 255, 0), 2)
    cv2.imshow('2', patch_)

    # import albumentations as A
    # trans = A.Compose([A.MotionBlur(p=0.),
    #                    A.CenterCrop(height=100, width=50, p=0.),
    #                    A.Rotate(limit=60, p=1.)],
    #                   p=1., bbox_params=A.BboxParams(format='pascal_voc', label_fields=[]))
    # while True:
    #     a = trans(image=patch, bboxes=bbox.reshape((-1, 4)))
    #     img = a['image']
    #     box = a['bboxes'][0]
    #     box = list(map(int, box))
    #     cv2.rectangle(img, (box[0], box[1]), (box[2], box[3]), (0, 255, 0), 2)
    #     cv2.imshow('3', img)
    #     cv2.waitKey()

    # while True:
    #     image_color = color_aug(patch.copy())
    # image_b = random_bright(patch.astype(np.float32))
    # cv2.imshow('b', image_b.astype(np.uint8))
    # image_c = random_contrast(patch.astype(np.float32))
    # cv2.imshow('c', image_c.astype(np.uint8))
    # image_s = random_contrast(patch.astype(np.float32))
    # cv2.imshow('s', image_s.astype(np.uint8))

    # image_color = color_aug(patch.copy())
    # cv2.imshow('color', image_color.astype(np.uint8))
    # image_flip, bbox_flip = flip_aug(patch.copy(), bbox)
    # cv2.imshow('filp', image_flip.astype(np.uint8))
    image_rotate, bbox_ratate = rotate_aug(patch.copy(), bbox, angle=-12.)
    box = list(map(int, bbox_ratate))
    rotation = image_rotate.copy()
    cv2.rectangle(rotation, (box[0], box[1]), (box[2], box[3]), (0, 255, 0), 2)
    cv2.imshow('rotate', rotation.astype(np.uint8))
    # image_blur = blur_aug(patch.copy())
    # cv2.imshow('blur', image_blur.astype(np.uint8))
    # image_motion = motion_blur(patch.copy(), degree=6, angle=8)
    # cv2.imshow('motion', image_motion.astype(np.uint8))
    # image_reso = resolution_aug(patch.copy(), 0.5)
    # cv2.imshow('resolution', image_reso.astype(np.uint8))
    # image_erase = erase_aug(patch.copy(), bbox)
    # cv2.imshow('erase', image_erase.astype(np.uint8))
    cv2.waitKey()

    # cv2.imwrite('original.jpg', patch, [int(cv2.IMWRITE_JPEG_QUALITY), 200])
    # cv2.imwrite('color.jpg', image_color, [int(cv2.IMWRITE_JPEG_QUALITY), 200])
    # cv2.imwrite('flip.jpg', image_flip, [int(cv2.IMWRITE_JPEG_QUALITY), 200])
    # cv2.imwrite('rotate.jpg', image_rotate, [int(cv2.IMWRITE_JPEG_QUALITY), 200])
    # cv2.imwrite('blur.jpg', image_blur, [int(cv2.IMWRITE_JPEG_QUALITY), 200])
    # cv2.imwrite('motion.jpg', image_motion, [int(cv2.IMWRITE_JPEG_QUALITY), 200])
    # cv2.imwrite('resolution.jpg', image_reso, [int(cv2.IMWRITE_JPEG_QUALITY), 200])
    # cv2.imwrite('erase.jpg', image_erase, [int(cv2.IMWRITE_JPEG_QUALITY), 200])
    pass
