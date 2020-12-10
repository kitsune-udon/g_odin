import os

import numpy as np
import pytorch_lightning as pl
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision.datasets import CIFAR10
from torchvision.transforms.transforms import (CenterCrop, ColorJitter,
                                               Compose, Normalize,
                                               RandomAffine, RandomGrayscale,
                                               Resize, ToTensor)

from argparse_utils import from_argparse_args


def cifar10_collate_fn(batch):
    images, labels = list(zip(*batch))
    images = torch.stack(images)
    labels = torch.tensor(labels)
    return images, labels


class LSUNResize(Dataset):
    def __init__(self, dataset_root='.', length=10000, transform=None):
        super().__init__()
        self.dataset_root = dataset_root
        self.length = length
        self.transform = transform
        self._set_path()

    def _set_path(self):
        self.imagefile_path = os.path.join(self.dataset_root,
                                           'LSUN_resize')

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        filepath = f"{index}.jpg"
        filepath = os.path.join(self.imagefile_path,
                                filepath)
        img = Image.open(filepath)

        if self.transform is not None:
            img = self.transform(img)

        return img, int(1)


class UniformDataset(Dataset):
    def __init__(self, image_size, length):
        super().__init__()
        self.image_size = image_size
        self.length = length

    def __getitem__(self, index):
        image = np.random.rand(3, self.image_size, self.image_size)
        image = torch.tensor(image, dtype=torch.float)
        label = 1

        return image, label

    def __len__(self):
        return self.length


class ConcatDataset(Dataset):
    def __init__(self, *args):
        super().__init__()
        self._ds = args

    def __getitem__(self, index):
        for d in self._ds:
            if index < len(d):
                return d[index]
            else:
                index -= len(d)

        raise ValueError("invalid index")

    def __len__(self):
        return sum(map(len, self._ds))


class ODINDataModule(pl.LightningDataModule):
    def __init__(self, dataset_root="./dataset",
                 train_batch_size=8, val_batch_size=8, num_workers=0):
        super().__init__()

        self.dataset_root = dataset_root
        self.train_batch_size = train_batch_size
        self.val_batch_size = val_batch_size
        self.num_workers = num_workers

        mean = (125.3/255, 123.0/255, 113.9/255)
        std = (63.0/255, 62.1/255, 66.7/255)

        self.transform = Compose([
            ColorJitter(brightness=0.1,
                        contrast=0.1,
                        saturation=0.1,
                        hue=0
                        ),
            RandomGrayscale(p=0.1),
            RandomAffine(degrees=15,
                         scale=(0.9, 1.1),
                         resample=Image.BILINEAR
                         ),
            ToTensor(),
            Normalize(mean=mean,
                      std=std)
        ])
        self.lsun_transform = Compose([
            ToTensor(),
            Normalize(mean=mean,
                      std=std)
        ])

    def prepare_data(self, *args, **kwargs):
        CIFAR10(self.dataset_root, download=True)

    def train_dataloader(self, *args, **kwargs):
        cifar10_train = CIFAR10(self.dataset_root, train=True,
                                download=False, transform=self.transform)

        return DataLoader(cifar10_train,
                          shuffle=True,
                          num_workers=self.num_workers,
                          batch_size=self.train_batch_size,
                          collate_fn=cifar10_collate_fn)

    def val_dataloader(self, *args, **kwargs):
        cifar10_val = CIFAR10(self.dataset_root, train=False,
                              download=False, transform=self.transform,
                              target_transform=lambda _: 0)
        lsun_resize = LSUNResize(
            self.dataset_root, transform=self.lsun_transform)

        dataset = ConcatDataset(cifar10_val, lsun_resize)

        return DataLoader(dataset,
                          shuffle=False,
                          num_workers=self.num_workers,
                          batch_size=self.val_batch_size)

    @staticmethod
    def add_argparse_args(parser):
        parser.add_argument("--train_batch_size", type=int,
                            default=256, help="batch size for training")
        parser.add_argument("--val_batch_size", type=int,
                            default=256, help="batch size for validation")
        parser.add_argument("--num_workers", type=int, default=4,
                            help="number of processes for dataloader")
        parser.add_argument("--dataset_root", type=str,
                            default="./dataset", help="root path of dataset")

        return parser

    @classmethod
    def from_argparse_args(cls, args, **kwargs):
        return from_argparse_args(cls, args, **kwargs)
