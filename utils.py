import csv
from urllib.request import urlretrieve
import socket
from torch.utils import data
from pathlib import Path
import pickle
from PIL import Image
import numpy as np
import random
from PIL import ImageFilter
import math

import torchvision
from torchvision import transforms
import torch

socket.setdefaulttimeout(3)


def get_web_image(url, path, image_name, count=1):
    if count > 5:
        print("download job failed!")
        return 0

    filename = path / image_name
    try:
        urlretrieve(url, filename)
    except socket.timeout:
        err_info = 'Reloading for %d time' % count if count == 1 else 'Reloading for %d times' % count
        print(err_info)
        get_web_image(url, path, image_name, count + 1)
    except:
        return 0


def read_csv(csv_file):
    data = []
    with open(csv_file, errors='ignore') as csv_f:
        csv_reader = csv.reader(csv_f)
        header = next(csv_reader)
        for row in csv_reader:
            data.append(row[:2])
    return data


class ImageDataset(data.Dataset):
    def __init__(self, dataset_path=Path('/data/wangjiawei/dataset/AIMeetsBeauty'), istrain=True, size=224,
                 transforms=None):
        super(ImageDataset, self).__init__()
        self.istrain = istrain
        self.dataset_path = dataset_path
        self.transforms = transforms
        self.size = size

        # Get a dict that maps image name to label
        # if istrain:
        #     with open('./dataset/img_name_2_label.pkl', "rb") as f:
        #         self.img_name_2_label = pickle.load(f)

        # Get all image paths
        self.img_paths = list(dataset_path.glob('*[.png, .jpg]'))
        self.img_paths = sorted(self.img_paths, key=lambda x: x.name.split('.')[0][1:])

        # Get number of images
        self.num_img = len(self.img_paths)

    def __getitem__(self, idx):
        img_path = self.img_paths[idx]
        img_name = img_path.name

        # Get image
        if self.istrain:
            try:
                img = Image.open(img_path).convert('RGB')
                img = self.padding(self.resize(img, self.size), self.size)
                if self.transforms:
                    img = self.transforms(img)

                # Get label
                # label = self.img_name_2_label[img_name]
            except Exception:
                # print(Exception)
                # print(self.dataset_path / Path(img_name))
                idx = np.random.randint(self.num_img)
                img_name, img = self.__getitem__(idx)

            return img_name, img

        else:
            img = Image.open(img_path).convert('RGB')
            img = self.padding(self.resize(img, self.size), self.size)
            if self.transforms:
                img = self.transforms(img)
            return img_name, img

    def __len__(self):
        return self.num_img

    @staticmethod
    def resize(img, size, interpolation=Image.ANTIALIAS):
        if isinstance(size, int):
            w, h = img.size
            if (w <= h and w == size) or (h <= w and h == size):
                return img
            if w < h:
                oh = size
                ow = int(size * w / h)
                return img.resize((ow, oh), interpolation)
            else:
                ow = size
                oh = int(size * h / w)
                return img.resize((ow, oh), interpolation)
        else:
            return img.resize(size[::-1], interpolation)

    @staticmethod
    def padding(img, size):
        w, h = img.size
        p = Image.new('RGB', (size, size), (255, 255, 255))
        p.paste(img, box=(int((size - w) / 2), int((size - h) / 2)))
        return p



# For reducing the use of share memory and get larger batch size.
class BatchDataLoader(object):
    def __init__(self, dataloader=None, size=224, batch_size=4, patch=4):
        assert isinstance(dataloader,
                          torch.utils.data.dataloader.DataLoader), 'Parameter "dataloader" should be a "torch.utils.data.dataloader.DataLoader"'
        self.dataloader = dataloader
        self.batch_size = batch_size
        self.patch = patch
        self.img_names = []
        self.imgs = torch.empty([self.patch * self.batch_size, 3, size, size])
        self.augmented_img = torch.empty([self.patch * self.batch_size, 3, size, size])
        self.matrix = torch.empty([self.patch * self.batch_size, 7])
        self.labels = []

    def __iter__(self):
        self.dataloader_iter = iter(self.dataloader)
        return self

    def __next__(self):
        self.img_names.clear()
        self.labels.clear()
        for i in range(self.patch):
            start, end = i * self.batch_size, (i + 1) * self.batch_size
            img_names_patch, imgs_patch, augmented_img_patch, matrix, labels_patch = next(self.dataloader_iter)
            self.img_names.extend(img_names_patch)
            self.imgs[start:end] = imgs_patch
            self.augmented_img[start:end] = augmented_img_patch
            self.matrix[start:end] = matrix
            self.labels.extend(labels_patch)

        return self.img_names, self.imgs, self.augmented_img, self.matrix, self.labels


class PairedTransformImageWithMaskDataset(data.Dataset):
    def __init__(self, dataset_path=Path('/data/wangjiawei/dataset/AIMeetsBeauty'), istrain=True, size=224,
                 data_transforms=None):
        super(PairedTransformImageWithMaskDataset, self).__init__()
        self.istrain = istrain
        self.dataset_path = dataset_path
        self.data_transforms = data_transforms
        self.size = size

        self.bg_img_path = list(Path('/data/dataset/ILSVRC2012/val/').iterdir())

        self.aug_data_transforms = transforms.Compose([transforms.ColorJitter(brightness=0.2,
                                                                              contrast=0.2,
                                                                              saturation=0.2,
                                                                              hue=0.05)])
        self.bg_data_transforms = transforms.Compose([transforms.RandomCrop(size, pad_if_needed=True)])
        self.mask_data_transforms = transforms.Compose([transforms.Resize((size, size), interpolation=2),
                                                        transforms.ToTensor()])

        # Get all image paths
        self.img_paths = list(dataset_path.glob('*[.png, .jpg]'))
        self.img_paths = sorted(self.img_paths, key=lambda x: x.name.split('.')[0][1:])

        # Get number of images
        self.num_img = len(self.img_paths)

    def __getitem__(self, idx):
        img_path = self.img_paths[idx]
        img_name = img_path.name

        # Get image
        if self.istrain:
            try:
                img, augmented_img, matrix, img_mask, img_fg_mask = self.get_paired_img(img_path)
                if self.data_transforms:
                    img = self.data_transforms(img)
                    augmented_img = self.data_transforms(self.aug_data_transforms(augmented_img))
                img_mask = self.mask_data_transforms(img_mask)
                img_fg_mask = self.mask_data_transforms(img_fg_mask)
            except Exception:
                idx = np.random.randint(self.num_img)
                img_name, img, augmented_img, matrix, img_mask, img_fg_mask = self.__getitem__(idx)

            return img_name, img, augmented_img, matrix, img_mask, img_fg_mask

    def __len__(self):
        return self.num_img

    def get_paired_img(self, img_path):
        # Get original image.
        img = Image.open(img_path).convert('RGB')
        img = self.padding(self.resize(img, self.size), self.size)
        img_mask = self.get_mask(img)

        # Get foreground image.
        angle, translate, scale, shear = self.get_params(degrees=45, translate=(-0.2, 0.2), scale_ranges=(0.5, 1.2),
                                                         shears=(-0, 0), img_size=(self.size, self.size))

        center = (img.size[0] * 0.5 + 0.5, img.size[1] * 0.5 + 0.5)
        matrix = transforms.functional._get_inverse_affine_matrix(center, angle, translate, scale, shear)
        img_fg = img.transform(img.size, Image.AFFINE, matrix, resample=Image.BILINEAR, fillcolor=(255, 255, 255))
        img_fg_mask = self.get_mask(img_fg)

        # Get background image.
        choice = random.random()
        if choice < 0.33:
            bg_path = random.choice(self.bg_img_path)
            img_bg = Image.open(bg_path).convert('RGB')
            if self.bg_data_transforms:
                img_bg = self.bg_data_transforms(img_bg)
                if random.random() > 0.5:
                    img_bg = img_bg.filter(ImageFilter.BLUR)

            # Get augmented image.
            augmented_img = self._fuse_fg_into_bg(img_fg, img_bg, img_fg_mask)
        elif 0.33 < choice < 0.66:
            img_bg = Image.fromarray(np.ones_like(img_fg) * np.random.randint(0, 255, size=[1, 1, 3], dtype=np.uint8))
            augmented_img = self._fuse_fg_into_bg(img_fg, img_bg, img_fg_mask)
        else:
            augmented_img = img_fg

        # Get aff_para
        aff_para = np.array(angle / 45, dtype=np.int32) + 2
        return img, augmented_img, aff_para, img_mask, img_fg_mask

    @staticmethod
    def _fuse_fg_into_bg(img_fg, img_bg, img_fg_mask):
        # Paste the foreground image to the background image
        img_h, img_w = img_fg.size
        img_bg_h, img_bg_w = img_bg.size
        img_bg.paste(img_fg, box=(int((img_bg_h - img_h) / 2), int((img_bg_w - img_w) / 2)), mask=img_fg_mask)

        return img_bg

    @staticmethod
    def resize(img, size, interpolation=Image.ANTIALIAS):
        if isinstance(size, int):
            w, h = img.size
            if (w <= h and w == size) or (h <= w and h == size):
                return img
            if w < h:
                oh = size
                ow = int(size * w / h)
                return img.resize((ow, oh), interpolation)
            else:
                ow = size
                oh = int(size * h / w)
                return img.resize((ow, oh), interpolation)
        else:
            return img.resize(size[::-1], interpolation)

    @staticmethod
    def padding(img, size, color=(255, 255, 255)):
        w, h = img.size
        p = Image.new('RGB', (size, size), color)
        p.paste(img, box=(int((size - w) / 2), int((size - h) / 2)))
        return p

    @staticmethod
    def get_params(degrees, translate, scale_ranges, shears, img_size):
        """Get parameters for affine transformation

        Returns:
            sequence: params to be passed to the affine transformation
        """
        #         angle = random.uniform(degrees[0], degrees[1])
        angle = np.random.randint(-2, 3) * degrees
        if translate is not None:
            max_dx = translate[0] * img_size[0]
            max_dy = translate[1] * img_size[1]
            translations = (np.round(random.uniform(-max_dx, max_dx)),
                            np.round(random.uniform(-max_dy, max_dy)))
        else:
            translations = (0, 0)

        if scale_ranges is not None:
            scale = random.uniform(scale_ranges[0], scale_ranges[1])
        else:
            scale = 1.0

        if shears is not None:
            shear = random.uniform(shears[0], shears[1])
        else:
            shear = 0.0

        return angle, translations, scale, shear

    @staticmethod
    def get_mask(img, thres=255):
        # Mask
        r, g, b = img.split()
        mask = np.stack([np.array(r), np.array(g), np.array(b)], axis=2).mean(axis=2)
        mask = Image.fromarray(((mask < thres).astype(np.uint8)) * 255)
        return mask

def denormalize(tensor, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)):
    tensor_denorm = tensor * torch.from_numpy(np.array(std, dtype=np.float32)).to(tensor.device).unsqueeze(0).unsqueeze(2).unsqueeze(3) + \
    torch.from_numpy(np.array(mean, dtype=np.float32)).to(tensor.device).unsqueeze(0).unsqueeze(2).unsqueeze(3)

    return tensor_denorm

def norm(x, eps=1e-8):
    return x / (np.linalg.norm(x, axis=1, keepdims=True) + eps)