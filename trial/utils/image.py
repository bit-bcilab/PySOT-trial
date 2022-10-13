

import numpy as np

import cv2
from glob import glob
from os.path import join
import platform

cv2.setNumThreads(0)
cv2.ocl.setUseOpenCL(False)

SYSTEM = platform.system()

RGB_MEAN = np.reshape(np.array([0.485, 0.456, 0.406]), (1, 1, 1, 3))
RGB_STD = np.reshape(np.array([0.229, 0.224, 0.225]), (1, 1, 1, 3))


def get_frames(video_name):
    """
    Read image frame by Opencv

    :param video_name: str, image or video file path
    :return: image matrix
    """
    if not video_name:
        cap = cv2.VideoCapture(0)
        # warmup
        for i in range(5):
            cap.read()
        while True:
            ret, frame = cap.read()
            if ret:
                yield frame
            else:
                break
    elif video_name.endswith('avi') or video_name.endswith('mp4'):
        cap = cv2.VideoCapture(video_name)
        while True:
            ret, frame = cap.read()
            if ret:
                yield frame
            else:
                break
    else:
        images = glob(join(video_name, '*.jp*'))

        if SYSTEM == 'Linux':
            images = sorted(images, key=lambda x: int(x.split('/')[-1].split('.')[0]))

        for img in images:
            frame = cv2.imread(img)
            yield frame


def crop_roi(image, input_size, crop_bbox, padding=(0, 0, 0)):
    """
    crop and resize image like SiamRPN++: 使用仿射变换

    :param image:
    :param input_size:
    :param crop_bbox:
    :param padding:
    :return:
    """
    crop_bbox = [float(x) for x in crop_bbox]
    a = (input_size - 1) / (crop_bbox[2] - crop_bbox[0])
    b = (input_size - 1) / (crop_bbox[3] - crop_bbox[1])
    c = -a * crop_bbox[0]
    d = -b * crop_bbox[1]
    mapping = np.array([[a, 0, c],
                        [0, b, d]]).astype(np.float)
    crop = cv2.warpAffine(image, mapping, (input_size, input_size), borderMode=cv2.BORDER_CONSTANT, borderValue=padding)
    return crop


def letterbox_image(image, input_size, bbox=None, padding=None):
    """
    resize image like YOLO: unchanging aspect ratio by adding padding

    :param image:
    :param input_size:
    :param bbox:
    :param padding:
    :return:
    """
    cv2.setNumThreads(0)
    cv2.ocl.setUseOpenCL(False)

    if bbox is None:
        bbox = []
    if padding is None:
        padding = np.array([128, 128, 128])[None, None, :]
    ih, iw = image.shape[:2]

    if not isinstance(input_size, list) and not isinstance(input_size, list):
        input_size = (input_size, input_size)

    scale = min(input_size[1] / iw, input_size[0] / ih)
    nw = int(iw * scale)
    nh = int(ih * scale)
    dx = (input_size[1] - nw) // 2
    dy = (input_size[0] - nh) // 2

    # image = cv2.resize(image, (nw, nh))
    try:
        image = cv2.resize(image, (nw, nh))
    except:
        print('image shape: ')
        print(image.shape)
        print('iw: {:d}, ih: {:d}, nw: {:d}, nh: {:d}, scale: {:.4f}'.format(iw, ih, nw, nh, scale))
        print(image)

    new_image = padding * np.ones(shape=(input_size[1], input_size[0], 3))
    new_image = new_image.astype(np.uint8)
    new_image[dy:(dy+nh), dx:(dx+nw), :] = image
    offset = np.array([dx, dy, dx, dy], dtype=np.float32)
    if len(bbox):
        bbox = bbox * scale + offset
        return new_image, bbox
    else:
        return new_image


def rgb_normalize(bgr_images, mobilenet=False):
    bgr_images = np.maximum(np.minimum(bgr_images, 255.0), 0.0)
    rgb_images = bgr_images[..., ::-1]
    if not mobilenet:
        rgb_images = rgb_images / 255.
        rgb_images = (rgb_images - RGB_MEAN) / RGB_STD
    else:
        rgb_images = rgb_images / 127.5
        rgb_images = rgb_images - 1.
    return rgb_images.astype(np.float32)


def normalize_batch_cuda(tensor, mean, std, inplace=False):
    """Normalize a tensor image with mean and standard deviation.

    .. note::
        This transform acts out of place by default, i.e., it does not mutates the input tensor.

    See :class:`~torchvision.transforms.Normalize` for more details.

    Args:
        tensor (Tensor): Tensor image of size (C, H, W) to be normalized.
        mean (sequence): Sequence of means for each channel.
        std (sequence): Sequence of standard deviations for each channel.
        inplace(bool,optional): Bool to make this operation inplace.

    Returns:
        Tensor: Normalized Tensor image.
    """
    if not inplace:
        tensor = tensor.clone()
    tensor = tensor.mul(1.0 / 255.0).clamp(0.0, 1.0)
    tensor.sub_(mean).div_(std)
    return tensor
