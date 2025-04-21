import os
import torch
import torchvision
import random
import numpy as np

IMG_EXTENSIONS = ['.jpg', '.JPG', '.jpeg', '.JPEG',
                  '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP']


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)


def get_paths_from_images(path):
    assert os.path.isdir(path), '{:s} is not a valid directory'.format(path)
    images = []
    for dirpath, _, fnames in sorted(os.walk(path)):
        for fname in sorted(fnames):
            if is_image_file(fname):
                img_path = os.path.join(dirpath, fname)
                images.append(img_path)
    assert images, '{:s} has no valid image file'.format(path)
    return sorted(images)


def augment(img_list, hflip=True, rot=True, split='val'):
    # horizontal flip OR rotate
    hflip = hflip and (split == 'train' and random.random() < 0.5)
    vflip = rot and (split == 'train' and random.random() < 0.5)
    rot90 = rot and (split == 'train' and random.random() < 0.5)

    def _augment(img):
        if hflip:
            img = img[:, ::-1, :]
        if vflip:
            img = img[::-1, :, :]
        if rot90:
            img = img.transpose(1, 0, 2)
        return img

    return [_augment(img) for img in img_list]


def transform2numpy(img):
    img = np.array(img)
    img = img.astype(np.float32) / 255.
    if img.ndim == 2:
        img = np.expand_dims(img, axis=2)
    # some images have 4 channels
    if img.shape[2] > 3:
        img = img[:, :, :3]
    return img


def transform2tensor(img, min_max=(0, 1)):
    # HWC to CHW
    img = torch.from_numpy(np.ascontiguousarray(
        np.transpose(img, (2, 0, 1)))).float()
    # to range min_max
    img = img*(min_max[1] - min_max[0]) + min_max[0]
    return img


# implementation by numpy and torch
# def transform_augment(img_list, split='val', min_max=(0, 1)):
#     imgs = [transform2numpy(img) for img in img_list]
#     imgs = augment(imgs, split=split)
#     ret_img = [transform2tensor(img, min_max) for img in imgs]
#     return ret_img


# implementation by torchvision, detail in https://github.com/Janspiry/Image-Super-Resolution-via-Iterative-Refinement/issues/14
totensor = torchvision.transforms.ToTensor()
hflip = torchvision.transforms.RandomHorizontalFlip()
def transform_augment(img_list, split='val', min_max=(0, 1)):    
    imgs = [totensor(img) for img in img_list]
    
    # 检查图像尺寸，分组处理不同分辨率的图像
    lr_imgs = []
    sr_imgs = []
    hr_imgs = []
    
    # 根据图像尺寸分组
    for img in imgs:
        if img.shape[1] == 128 and img.shape[2] == 128:  # LR图像
            lr_imgs.append(img)
        elif img.shape[1] == 256 and img.shape[2] == 256:  # SR或HR图像
            if len(sr_imgs) < 3:  # 前3个256x256图像是SR
                sr_imgs.append(img)
            else:  # 最后一个256x256图像是HR
                hr_imgs.append(img)
    
    # 分别处理不同分辨率的图像组
    if split == 'train':
        if lr_imgs:
            lr_stack = torch.stack(lr_imgs, 0)
            lr_stack = hflip(lr_stack)
            lr_imgs = torch.unbind(lr_stack, dim=0)
        
        if sr_imgs:
            sr_stack = torch.stack(sr_imgs, 0)
            sr_stack = hflip(sr_stack)
            sr_imgs = torch.unbind(sr_stack, dim=0)
            
        if hr_imgs:
            hr_stack = torch.stack(hr_imgs, 0)
            hr_stack = hflip(hr_stack)
            hr_imgs = torch.unbind(hr_stack, dim=0)
    
    # 合并处理后的图像列表
    processed_imgs = list(lr_imgs) + list(sr_imgs) + list(hr_imgs)
    
    # 应用min_max缩放
    ret_img = [img * (min_max[1] - min_max[0]) + min_max[0] for img in processed_imgs]
    return ret_img
