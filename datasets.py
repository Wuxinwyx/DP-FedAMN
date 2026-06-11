import torch.utils.data as data
from PIL import Image
import numpy as np
import torchvision
import torch
from torchvision.datasets import MNIST, EMNIST, CIFAR10, CIFAR100, SVHN, FashionMNIST, ImageFolder, DatasetFolder, utils

import os
import os.path
import logging

logging.basicConfig()
logger = logging.getLogger()
logger.setLevel(logging.INFO)

IMG_EXTENSIONS = ('.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm', '.tif', '.tiff', '.webp')


def mkdirs(dirpath):
    try:
        os.makedirs(dirpath)
    except Exception as _:
        pass



class CIFAR10_truncated(data.Dataset):

    def __init__(self, root, dataidxs=None, train=True, transform=None, target_transform=None, download=False):

        self.root = root
        self.dataidxs = dataidxs
        self.train = train
        self.transform = transform
        self.target_transform = target_transform
        self.download = download

        self.data, self.target = self.__build_truncated_dataset__()

    def __build_truncated_dataset__(self):

        cifar_dataobj = CIFAR10(self.root, self.train, self.transform, self.target_transform, self.download)

        if torchvision.__version__ == '0.2.1':
            if self.train:
                data, target = cifar_dataobj.train_data, np.array(cifar_dataobj.train_labels)
            else:
                data, target = cifar_dataobj.test_data, np.array(cifar_dataobj.test_labels)
        else:
            data = cifar_dataobj.data
            target = np.array(cifar_dataobj.targets)

        if self.dataidxs is not None:
            data = data[self.dataidxs]
            target = target[self.dataidxs]

        return data, target

    def truncate_channel(self, index):
        for i in range(index.shape[0]):
            gs_index = index[i]
            self.data[gs_index, :, :, 1] = 0.0
            self.data[gs_index, :, :, 2] = 0.0

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img, target = self.data[index], self.target[index]
        # img = Image.fromarray(img)
        # print("cifar10 img:", img)
        # print("cifar10 target:", target)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self):
        return len(self.data)


class CIFAR100_truncated(data.Dataset):

    def __init__(self, root, dataidxs=None, train=True, transform=None, target_transform=None, download=False):

        self.root = root
        self.dataidxs = dataidxs
        self.train = train
        self.transform = transform
        self.target_transform = target_transform
        self.download = download

        self.data, self.target = self.__build_truncated_dataset__()

    def __build_truncated_dataset__(self):

        cifar_dataobj = CIFAR100(self.root, self.train, self.transform, self.target_transform, self.download)

        if torchvision.__version__ == '0.2.1':
            if self.train:
                data, target = cifar_dataobj.train_data, np.array(cifar_dataobj.train_labels)
            else:
                data, target = cifar_dataobj.test_data, np.array(cifar_dataobj.test_labels)
        else:
            data = cifar_dataobj.data
            target = np.array(cifar_dataobj.targets)

        if self.dataidxs is not None:
            data = data[self.dataidxs]
            target = target[self.dataidxs]

        return data, target

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img, target = self.data[index], self.target[index]
        img = Image.fromarray(img)
        # print("cifar10 img:", img)
        # print("cifar10 target:", target)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self):
        return len(self.data)




class ImageFolder_custom(DatasetFolder):
    def __init__(self, root, dataidxs=None, train=True, transform=None, target_transform=None):
        self.root = root
        self.dataidxs = dataidxs
        self.train = train
        self.transform = transform
        self.target_transform = target_transform

        imagefolder_obj = ImageFolder(self.root, self.transform, self.target_transform)
        self.loader = imagefolder_obj.loader
        if self.dataidxs is not None:
            self.samples = np.array(imagefolder_obj.samples)[self.dataidxs]
        else:
            self.samples = np.array(imagefolder_obj.samples)

    def __getitem__(self, index):
        path = self.samples[index][0]
        target = self.samples[index][1]
        target = int(target)
        sample = self.loader(path)
        if self.transform is not None:
            sample = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return sample, target

    def __len__(self):
        if self.dataidxs is None:
            return len(self.samples)
        else:
            return len(self.dataidxs)

class FashionMNIST_truncated(data.Dataset):

    def __init__(self, root, dataidxs=None, train=True, transform=None, target_transform=None, download=False):

        self.root = root
        self.dataidxs = dataidxs
        self.train = train
        self.transform = transform
        self.target_transform = target_transform
        self.download = download

        self.data, self.target = self.__build_truncated_dataset__()

    def __build_truncated_dataset__(self):

        fashion_dataobj = FashionMNIST(self.root, self.train, None, None, self.download)

        if torchvision.__version__ == '0.2.1':
            if self.train:
                data_, target_ = fashion_dataobj.train_data, fashion_dataobj.train_labels
            else:
                data_, target_ = fashion_dataobj.test_data, fashion_dataobj.test_labels
            target_ = np.array(target_)
        else:
            data_ = fashion_dataobj.data
            target_ = fashion_dataobj.targets
            if isinstance(target_, torch.Tensor):
                target_ = target_.cpu().numpy()
            else:
                target_ = np.array(target_)

        if self.dataidxs is not None:
            data_ = data_[self.dataidxs]
            target_ = target_[self.dataidxs]

        return data_, target_

    def truncate_channel(self, index):
        # FashionMNIST 是灰度图，没有多通道，这里保留接口以兼容旧代码
        return

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img, target = self.data[index], self.target[index]

        # FashionMNIST 的 data 通常是 torch.Tensor (H, W)，ToTensor 需要 PIL 或 numpy
        if isinstance(img, torch.Tensor):
            img = img.cpu().numpy()
        img = Image.fromarray(img, mode='L')

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self):
        return len(self.data)

class EMNIST_truncated(data.Dataset):

    def __init__(self, root, dataidxs=None, train=True, transform=None, target_transform=None, download=False):

        self.root = root
        self.dataidxs = dataidxs
        self.train = train
        self.transform = transform
        self.target_transform = target_transform
        self.download = download

        # 只使用 EMNIST 的 balanced split（47类）
        self.split = "balanced"

        self.data, self.target = self.__build_truncated_dataset__()

    def __patch_emnist_url_md5_and_cleanup__(self):
        """
        最小化修改方案：
        1) monkey patch torchvision.datasets.EMNIST 的 url 与 md5，避免旧地址重定向导致 md5 校验失败
        2) 删除可能已下载但损坏的 gzip.zip，避免反复报 File not found or corrupted
        """
        # 注意：这里假设你已经在文件顶部 `from torchvision.datasets import EMNIST`
        EMNIST.url = "https://biometrics.nist.gov/cs_links/EMNIST/gzip.zip"
        EMNIST.md5 = "58c8d27c78d21e728a6bc7b3cc06412e"

        # torchvision EMNIST 默认目录结构：<root>/EMNIST/raw/gzip.zip
        bad_zip = os.path.join(self.root, "EMNIST", "raw", "gzip.zip")
        if os.path.exists(bad_zip):
            try:
                os.remove(bad_zip)
            except Exception:
                # 删除失败就不强行抛错，后续 download 仍会再尝试
                pass

    def __build_truncated_dataset__(self):

        # 关键：在第一次构造 EMNIST 之前 patch
        self.__patch_emnist_url_md5_and_cleanup__()

        emnist_dataobj = EMNIST(self.root, split=self.split, train=self.train,
                               transform=None, target_transform=None, download=self.download)

        data_ = getattr(emnist_dataobj, "data", None)
        target_ = getattr(emnist_dataobj, "targets", None)

        if data_ is None or target_ is None:
            if self.train:
                data_ = getattr(emnist_dataobj, "train_data")
                target_ = getattr(emnist_dataobj, "train_labels")
            else:
                data_ = getattr(emnist_dataobj, "test_data")
                target_ = getattr(emnist_dataobj, "test_labels")

        # targets 转成 numpy，保持与你 CIFAR10_truncated 一致
        if isinstance(target_, torch.Tensor):
            target_ = target_.cpu().numpy()
        else:
            target_ = np.array(target_)

        # 子集索引的稳健处理（避免 numpy 索引 torch.Tensor 报错）
        if self.dataidxs is not None:
            idxs = self.dataidxs
            if isinstance(data_, torch.Tensor):
                if isinstance(idxs, np.ndarray):
                    idxs_t = torch.from_numpy(idxs).long()
                else:
                    idxs_t = torch.tensor(list(idxs), dtype=torch.long)
                data_ = data_[idxs_t]
                target_ = target_[idxs_t.cpu().numpy()]
            else:
                idxs_np = np.array(list(idxs), dtype=np.int64)
                data_ = data_[idxs_np]
                target_ = target_[idxs_np]

        return data_, target_

    def truncate_channel(self, index):
        return

    def __getitem__(self, index):
        img, target = self.data[index], self.target[index]

        if isinstance(img, torch.Tensor):
            img = img.cpu().numpy()
        img = Image.fromarray(img, mode='L')

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self):
        return len(self.data)