import os
import numpy as np
import torch
from skimage.color import lab2rgb, rgb2lab, rgb2gray
from torchvision import datasets
import torchvision.transforms as transforms


def create_dirs():
    os.makedirs("images/train/class/", exist_ok=True)  # 40,000 images
    os.makedirs("images/val/class/", exist_ok=True)  #  1,000 images
    for i, file in enumerate(os.listdir("testSet_resize")):
        if i < 1000:  # first 1000 will be val
            os.rename("testSet_resize/" + file, "images/val/class/" + file)
        else:  # others will be val
            os.rename("testSet_resize/" + file, "images/train/class/" + file)


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


def get_train_loader(folder="images/train"):
    train_transforms = transforms.Compose(
        [transforms.RandomResizedCrop(224), transforms.RandomHorizontalFlip()]
    )
    train_imagefolder = GrayscaleImageFolder(folder, train_transforms)
    train_loader = torch.utils.data.DataLoader(
        train_imagefolder, batch_size=64, shuffle=True
    )
    return train_loader


def get_val_loader(folder="images/val"):
    val_transforms = transforms.Compose(
        [transforms.Resize(256), transforms.CenterCrop(224)]
    )
    val_imagefolder = GrayscaleImageFolder(folder, val_transforms)
    val_loader = torch.utils.data.DataLoader(
        val_imagefolder, batch_size=64, shuffle=False
    )
    return val_loader
