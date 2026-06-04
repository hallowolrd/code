import os
import urllib.request
import zipfile

import torchvision
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder


DATASET_CFG = {
    "cifar10": {"num_classes": 10, "in_channels": 3, "img_size": 32},
    "cifar100": {"num_classes": 100, "in_channels": 3, "img_size": 32},
    "tinyimagenet": {"num_classes": 200, "in_channels": 3, "img_size": 64},
    "femnist": {"num_classes": 62, "in_channels": 1, "img_size": 28},
}


def _prepare_tinyimagenet(root):
    data_path = os.path.join(root, "tiny-imagenet-200")
    if not os.path.exists(data_path):
        url = "http://cs231n.stanford.edu/tiny-imagenet-200.zip"
        zip_path = os.path.join(root, "tiny-imagenet-200.zip")
        print("[Dataset] Downloading TinyImageNet...")
        urllib.request.urlretrieve(url, zip_path)
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(root)
        os.remove(zip_path)

    val_dir = os.path.join(data_path, "val")
    anno = os.path.join(val_dir, "val_annotations.txt")
    img_dir = os.path.join(val_dir, "images")
    if os.path.exists(anno):
        with open(anno, "r", encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split("	")
                cls_dir = os.path.join(val_dir, parts[1])
                os.makedirs(cls_dir, exist_ok=True)
                src = os.path.join(img_dir, parts[0])
                if os.path.exists(src):
                    os.rename(src, os.path.join(cls_dir, parts[0]))
        os.remove(anno)
        if os.path.isdir(img_dir) and not os.listdir(img_dir):
            os.rmdir(img_dir)

    return data_path


def get_dataset(name, data_root):
    os.makedirs(data_root, exist_ok=True)

    if name == "cifar10":
        mean, std = (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)
        train_transform = transforms.Compose(
            [
                transforms.RandomCrop(32, 4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean, std),
            ]
        )
        test_transform = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize(mean, std)]
        )
        return (
            torchvision.datasets.CIFAR10(
                data_root, True, download=True, transform=train_transform
            ),
            torchvision.datasets.CIFAR10(
                data_root, False, download=True, transform=test_transform
            ),
        )

    if name == "cifar100":
        mean, std = (0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)
        train_transform = transforms.Compose(
            [
                transforms.RandomCrop(32, 4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean, std),
            ]
        )
        test_transform = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize(mean, std)]
        )
        return (
            torchvision.datasets.CIFAR100(
                data_root, True, download=True, transform=train_transform
            ),
            torchvision.datasets.CIFAR100(
                data_root, False, download=True, transform=test_transform
            ),
        )

    if name == "tinyimagenet":
        data_path = _prepare_tinyimagenet(data_root)
        mean, std = (0.4802, 0.4481, 0.3975), (0.2302, 0.2265, 0.2262)
        train_transform = transforms.Compose(
            [
                transforms.RandomCrop(64, 8),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean, std),
            ]
        )
        test_transform = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize(mean, std)]
        )
        return (
            ImageFolder(os.path.join(data_path, "train"), transform=train_transform),
            ImageFolder(os.path.join(data_path, "val"), transform=test_transform),
        )

    if name == "femnist":
        transform = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
        )
        return (
            torchvision.datasets.EMNIST(
                data_root,
                split="byclass",
                train=True,
                download=True,
                transform=transform,
            ),
            torchvision.datasets.EMNIST(
                data_root,
                split="byclass",
                train=False,
                download=True,
                transform=transform,
            ),
        )

    raise ValueError(f"Unsupported dataset: {name}")


def get_deterministic_train_dataset(name, data_root):
    os.makedirs(data_root, exist_ok=True)

    if name == "cifar10":
        mean, std = (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)
        transform = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize(mean, std)]
        )
        return torchvision.datasets.CIFAR10(
            data_root, True, download=True, transform=transform
        )

    if name == "cifar100":
        mean, std = (0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)
        transform = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize(mean, std)]
        )
        return torchvision.datasets.CIFAR100(
            data_root, True, download=True, transform=transform
        )

    if name == "tinyimagenet":
        data_path = _prepare_tinyimagenet(data_root)
        mean, std = (0.4802, 0.4481, 0.3975), (0.2302, 0.2265, 0.2262)
        transform = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize(mean, std)]
        )
        return ImageFolder(os.path.join(data_path, "train"), transform=transform)

    if name == "femnist":
        transform = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
        )
        return torchvision.datasets.EMNIST(
            data_root,
            split="byclass",
            train=True,
            download=True,
            transform=transform,
        )

    raise ValueError(f"Unsupported dataset: {name}")
