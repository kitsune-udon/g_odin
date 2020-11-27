import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision.datasets import CIFAR10, LSUN
from torchvision.transforms.transforms import (CenterCrop, Compose, Normalize,
                                               Resize, ToTensor)

from argparse_utils import from_argparse_args


def cifar10_collate_fn(batch):
    images, labels = list(zip(*batch))
    images = torch.stack(images)
    labels = torch.tensor(labels)
    return images, labels


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

        self.transform = Compose([
            ToTensor(),
            Normalize(mean=(0.485, 0.456, 0.406),
                      std=(0.229, 0.224, 0.225))
        ])
        self.lsun_transform = Compose([
            Resize(32),
            CenterCrop(32),
            ToTensor(),
            Normalize(mean=(0.485, 0.456, 0.406),
                      std=(0.229, 0.224, 0.225))
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
        lsun_test = LSUN(self.dataset_root, classes='test',
                         transform=self.lsun_transform,
                         target_transform=lambda _: 1)
        dataset = ConcatDataset(cifar10_val, lsun_test)

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
