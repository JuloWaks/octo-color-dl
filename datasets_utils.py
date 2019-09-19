import os
import numpy as np
import torch
from skimage.color import lab2rgb, rgb2lab, rgb2gray
from torchvision import datasets
import torchvision.transforms as transforms
from numba import jit
import time


def create_dirs():
    os.makedirs("images/train/class/", exist_ok=True)  # 40,000 images
    os.makedirs("images/val/class/", exist_ok=True)  #  1,000 images
    for i, file in enumerate(os.listdir("testSet_resize")):
        if i < 1000:  # first 1000 will be val
            os.rename("testSet_resize/" + file, "images/val/class/" + file)
        else:  # others will be val
            os.rename("testSet_resize/" + file, "images/train/class/" + file)


def timing(f):
    def wrap(*args):
        time1 = time.time()
        ret = f(*args)
        time2 = time.time()
        print(
            "{:s} function took {:.3f} ms".format(f.__name__, (time2 - time1) * 1000.0)
        )

        return ret

    return wrap


# @jit
def get_ab(img):
    img_original = np.asarray(img)
    img_lab = rgb2lab(img_original)
    img_lab = (img_lab + 128) / 255
    img_ab = img_lab[:, :, 1:3]
    img_ab = torch.from_numpy(img_ab.transpose((2, 0, 1))).float()
    return img_original, img_ab


# def getL(img):
#     img_original = np.asarray(img)
#     img_original =


# Taken from https://lukemelas.github.io/image-colorization.html
class GrayscaleImageFolder(datasets.ImageFolder):
    """Custom images folder, which converts images to grayscale before loading"""

    def __getitem__(self, index):
        path, target = self.imgs[index]
        img = self.loader(path)
        if self.transform is not None:
            img_original = self.transform(img)
            img_original = np.asarray(img_original)
            img_lab = rgb2lab(img_original)
            img_lab = (img_lab + 128) / 255
            img_ab = img_lab[:, :, 1:3]
            img_ab = torch.from_numpy(img_ab.transpose((2, 0, 1))).float()
            img_original = rgb2gray(img_original)
            img_original = torch.from_numpy(img_original).unsqueeze(0).float()

        if self.target_transform is not None:
            target = self.target_transform(target)
        return img_original, img_ab, target


def get_train_loader(folder, dataloader_kwargs, batch_size=64):
    train_transforms = transforms.Compose(
        [
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            # transforms.ToTensor(),
        ]
    )
    train_imagefolder = GrayscaleImageFolder(folder, train_transforms)
    train_loader = torch.utils.data.DataLoader(
        train_imagefolder,
        shuffle=True,
        batch_size=batch_size,
        num_workers=1,
        **dataloader_kwargs
    )
    return train_loader


def get_val_loader(folder="images/val"):
    val_transforms = transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            # transforms.ToTensor(),
        ]
    )
    val_imagefolder = GrayscaleImageFolder(folder, val_transforms)
    val_loader = torch.utils.data.DataLoader(
        val_imagefolder, batch_size=64, shuffle=False
    )
    return val_loader
